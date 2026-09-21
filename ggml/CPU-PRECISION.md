# Open issue: the CPU backend is inexact on the vision tower

**Status: unresolved.** The SigLIP vision tower does not match the fp32 PyTorch reference on
the CPU backend, and does on Vulkan. This is the reverse of every other graph in this port, so
it is recorded here rather than buried in a commit message.

## The observation

`vision` compared against `img_feats.f32` dumped from the real PyTorch model (36,864 values,
mean |ref| = 3.57):

| backend | max abs diff | max rel, where \|ref\| > 1 | gate |
|---|---:|---:|---|
| Vulkan, `GGML_VK_DISABLE_F16=1` | 1.22e-4 | **6.54e-5** | PASS |
| CPU | 1.93e-2 | **1.25e-2** | FAIL |

For contrast, the other two graphs on the same two backends:

| graph | CPU | Vulkan |
|---|---:|---:|
| decision head (2 layers) | 5.96e-8 | 1.36e-6 |
| encoder text stack (30 layers) | 1.03e-4 | 2.21e-4 |

CPU is the exact backend everywhere else. On the vision tower it is 200x worse than Vulkan.

## Why this is not simply "deep graph, float32"

The error is broad but small in relative terms — it is not one bad element:

```
elements: >1e-2 108, >1e-3 22425, >1e-4 35272 of 36864
worst at idx 33915  (got 5.000351e1 ref 4.998416e1)
```

The worst element is 50.0035 against 49.9842 — 3.9e-4 relative. 96% of elements exceed 1e-4
absolute, but the reference values run to ~50, so that is a spread of *relative* errors from
~1e-5 to ~4e-4 across the whole tensor.

That magnitude is suspiciously close to f16 epsilon (2^-11 = 4.9e-4), which is what made an f16
path the first hypothesis.

## Ruled out

- **f16 weights.** The mmproj GGUF was re-converted with `--outtype f32` and re-verified:
  `Counter({'F32': 198})`. The file also grew 190 MB -> 374 MB, consistent with f16 -> f32. So
  the weights are fp32, and the reference model runs fp32 too (`dtype: "fp32"` in
  `vlm_agent_config.json`, and `_torch_dtype()` returns `torch.float32`).
- **A GELU lookup-table path.** `ggml_table_gelu_f16` is built unconditionally
  (`ggml-cpu/ggml-cpu.c:3883`), but no GELU compile definition appears anywhere in
  `build.ninja` (grep count 0), and the f32 op should not route through the f16 table.
- **The wrong GELU variant.** `ggml_gelu` is the tanh approximation
  (`GELU_COEF_A = 0.044715f`, `ggml-cpu/vec.h:963`), which is what `gelu_pytorch_tanh` asks
  for. `ggml_gelu_erf` would be wrong here, not this.
- **A structurally wrong graph.** Both backends run the identical graph over the identical
  weights. If the graph were wrong both would be wrong; Vulkan reproduces the reference to
  6.5e-5.

## Remaining hypotheses, each with a test

1. **LayerNorm variance convention.** PyTorch `LayerNorm` uses the biased (population)
   variance. If `ggml_norm` used the unbiased form, the relative error would be about
   `1/(2*768) = 6.5e-4` — the right order of magnitude. The vision tower runs 24 LayerNorms
   with `eps = 1e-6`, far more than the head (2) or the encoder (which uses `rms_norm`, not
   `ggml_norm`), so a small per-norm bias compounds here and nowhere else.
   *Test:* a standalone graph of one `ggml_norm` over a fixed 768-vector, against numpy.
   Cheap and decisive. (A first sanity check argued the sign was wrong, but that was
   back-of-envelope on one element, not a measurement.)

2. **A CPU kernel selecting an f16 path.** Read the `GGML_OP_GELU` / `GGML_OP_NORM` dispatch
   in `ggml-cpu.c`/`ops.cpp` and see which vector routine f32 inputs actually take. The table
   is built unconditionally, so the guard is inside the op, not the build.

3. **Attention accumulation over 1024 keys.** `ggml_mul_mat` on a `[1024,1024,12]` softmax is
   the widest reduction in the port. Different tiling between CPU and Vulkan would explain
   CPU != Vulkan, though not obviously why CPU is the worse of the two.

4. **`ggml_soft_max_ext` on 1024-wide rows.**

## The direct route, if the above do not settle it

Bisect by dumping intermediates from the HF model with forward hooks — after patch embedding,
after position embedding, after each of the 12 blocks, after `post_layernorm`, after the
connector — and compare each against the same point in the ggml graph. `oracle/dump_extras.py`
already demonstrates the hook pattern for `image_hidden_states`; the same technique applies at
any module boundary. This turns "the tower is off" into "block 3 is off", which is a much
smaller search.

## Reproduce

```sh
cd ggml
zig build
LIB=.build/ggml/src:.build/ggml/src/ggml-vulkan

LD_LIBRARY_PATH=$LIB ./zig-out/bin/vision ../.weights/hf/smolvlm-mmproj.gguf ../.weights/oracle cpu
LD_LIBRARY_PATH=$LIB GGML_VK_DISABLE_F16=1 \
  ./zig-out/bin/vision ../.weights/hf/smolvlm-mmproj.gguf ../.weights/oracle vulkan
```

Requires `../.weights/oracle/{pixels.f32,img_feats.f32}`, produced by
`oracle/dump_pixels.py` and `oracle/dump_extras.py`.

## What fixed looks like

The same relative bound the other graphs hold — the encoder clears 1e-4 on both backends
without a backend-specific flag. If the cause turns out to be a ggml CPU kernel, the honest
outcome may be an upstream report plus a documented caveat here, not a change this port can
make.

## Note on the gate

The gate is *relative*, not absolute. An earlier 1e-2 absolute bound was wrong: reference
values reach ~50, so 1e-2 absolute is only 2e-4 relative and rejects a result that a ratio
judges fine. The CPU figure fails on either reading, but the criterion is now the correct one.
