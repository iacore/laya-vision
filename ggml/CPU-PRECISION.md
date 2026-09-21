# Resolved: the CPU backend was inexact on the vision tower

**Status: fixed and verified.** This was the one open accuracy question in the port. The cause
is a ggml-cpu kernel, it was identified by direct measurement, and the vision tower now clears
its gate on both backends.

| backend | before | after |
|---|---:|---:|
| CPU | 1.25e-2 max rel | **3.79e-5** |
| Vulkan (`GGML_VK_DISABLE_F16=1`) | 6.54e-5 | **6.54e-5** |

Against `img_feats.f32`, 36,864 values, mean |ref| = 3.57, relative judged where |ref| > 1.

## The cause

`src/ggml-cpu/vec.h:46` defines `GGML_GELU_FP16` unconditionally. With it set, the **f32** GELU
op does not evaluate the formula -- it rounds its input to f16, looks the result up in a table,
and rounds the result back:

```c
#define GGML_GELU_FP16

#ifdef GGML_GELU_FP16
inline static void ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    ...
    ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
    memcpy(&t, &fp16, sizeof(uint16_t));
    y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[t]);
}
```

Measured directly, `src/gelu_probe.zig`, 4096 values spanning [-12, 12], against the exact
float32 formulas:

| op | CPU max rel | Vulkan max rel |
|---|---:|---:|
| `ggml_gelu` | **3.69e-3** | 2.79e-4 |
| `ggml_gelu_erf` | 0 | 2.03e-4 |

So on CPU `ggml_gelu` is two orders of magnitude worse than it should be, and `ggml_gelu_erf`
is exact. On Vulkan both are f32-rounding-level.

### Why only the vision tower

It is the only graph here that calls `ggml_gelu`:

| graph | activation |
|---|---|
| vision tower (12 layers) | `ggml_gelu` -- `gelu_pytorch_tanh` |
| decision head | `ggml_gelu_erf` -- `nn.GELU()` is the exact erf form |
| encoder (30 layers) | `ggml_silu` -- SwiGLU, no GELU at all |

That is exactly why CPU was the *exact* backend for the other two graphs and the bad one here,
which was the fact that made the original observation confusing.

### The error's shape agreed

Broad and small in relative terms, not a bad element: 96% of outputs differed by more than
1e-4 absolute, worst element 3.9e-4 relative. That is the signature of an f16 quantisation
applied per call and compounding over twelve layers -- which is what it turned out to be.

## Correction to the earlier version of this file

It "ruled out a GELU lookup-table path" because no GELU compile definition appeared in
`build.ninja`. That was the wrong place to look: the define is in the source header, not the
build system. The hypothesis was right and the search was wrong. The lesson is that a negative
grep is only as good as the path it was pointed at.

It also fixed on the LayerNorm variance convention as the leading hypothesis, by order of
magnitude. That was a coincidence: 1/(2*768) = 6.5e-4 is close to the f16 epsilon, and the two
coincide because both are "one small fraction of the same dimension".

## The fix

Spell the same formula out from ops that are exact on every backend, and use it where the lossy
kernel is in play:

```zig
// 0.5*x*(1 + tanh(sqrt(2/pi) * x * (1 + 0.044715*x^2)))
const x2    = mulScalar(sqr(x), 0.044715);
const inner = mulScalar(mul(x, add(x2, repeatTo(ones, x2))), 0.7978845608028654);
return mulScalar(mul(x, add(tanh(inner), repeatTo(ones, inner))), 0.5);
```

This is *the same formula* `ggml_gelu` implements; it is not a different approximation. Vulkan
keeps `ggml_gelu`, because there it already is that formula computed correctly. One formula,
one documented exception, and the exception exists because a kernel is lossy on one backend.

The composition costs about 216 MB of extra activation memory for this graph. That is not free:
it pushes a single Vulkan buffer to 1.07 GB, past this device's per-allocation limit, so the
Vulkan path could no longer allocate at all. Hence the branch -- it is paid only where it buys
accuracy.

## Residual, honestly

- **The branch makes the graph shapes backend-dependent.** Defensible for the reason above, but
  it is a real cost and a reader should know it is there.
- **The structural fix is a gallocr.** `ggml_backend_alloc_ctx_tensors` allocates every tensor
  in the context, so all twelve layers' intermediates stay live at once -- roughly 540 MB of the
  1.07 GB. A `ggml_gallocr` would reuse them and the composition would fit anywhere. That is a
  change to all three modules and was not worth the risk once the accuracy was fixed.
- **This looks worth reporting upstream.** A default-on f16 quantisation of the f32 GELU is a
  surprising choice, and it silently costs ~3.7e-3 relative per call on CPU. `GGML_GELU_FP16`
  existing is not the problem; its being unconditional and undocumented for f32 inputs is.
- The gate is *relative*, not absolute. An earlier 1e-2 absolute bound was wrong: reference
  values reach ~50, so 1e-2 absolute is only 2e-4 relative and rejected a result that a ratio
  judges fine.

## Reproduce

```sh
cd ggml
zig build
LIB=.build/ggml/src:.build/ggml/src/ggml-vulkan

LD_LIBRARY_PATH=$LIB ./zig-out/bin/gelu-probe cpu
LD_LIBRARY_PATH=$LIB GGML_VK_DISABLE_F16=1 ./zig-out/bin/gelu-probe vulkan

LD_LIBRARY_PATH=$LIB ./zig-out/bin/vision ../.weights/hf/smolvlm-mmproj.gguf ../.weights/oracle cpu
LD_LIBRARY_PATH=$LIB GGML_VK_DISABLE_F16=1 \
  ./zig-out/bin/vision ../.weights/hf/smolvlm-mmproj.gguf ../.weights/oracle vulkan
```
