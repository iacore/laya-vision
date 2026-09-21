# laya-vision decision head on ggml, in Zig

A from-scratch Zig/C implementation of `thaitea/laya-vision-smolvlm-256m` on ggml: image
preprocessing, tokenizer, SigLIP vision tower and Idefics3 connector, the SmolVLM text stack,
and the 36-tensor decision head, chained into one binary. No Python at runtime, and no
llama.cpp, only ggml.

Every stage is verified against the real PyTorch model rather than against itself -- reference
tensors dumped from a Python forward pass, which each binary has to reproduce.

## Why the head and not the encoder

The published checkpoint is one flat state dict:

| prefix | tensors | who runs it |
|---|---:|---|
| `encoder.*` | 470 | the encoder — **not implemented here yet** |
| `type_emb`, `head.*`, `scorer.*`, `act_head.*`, `temperature` | 36 | **this directory** |

Two properties of *this* checkpoint collapse the work:

- `vlm_agent_config.json` says `option_attention: "causal"`, so the custom bidirectional
  4D option-block mask is not in play — ordinary causal attention is correct.
- There is **no `lm_head`**. The model never generates tokens; it only needs the decoder's
  final hidden states.

The encoder (SigLIP vision tower + Idefics3 connector + a Llama-30L text model) is to be
built on ggml directly, guided by llama.cpp's `clip.cpp` and Llama graph. llama.cpp itself
is not a dependency. Weights are read with ggml's own GGUF reader (`include/gguf.h`), which
needs nothing from llama.cpp.

## Layout of the head

```
h = enc_h + type_emb[qtype]                 [576, L]
h = 2 x TransformerEncoderLayer(576, 9 heads, 2304 ffn, ReLU, norm_first)
m = h[marker_pos]                           [576, k]
p = softmax( scorer(m) / temperature[qtype] )
```

Everything maps onto stock ggml ops — `mul_mat`, `norm`, `gelu_erf`, `soft_max_ext`,
`get_rows`, `cont`, `permute`, `reshape_3d`. **No custom kernels, no custom ops.**

`scorer.0` is a LayerNorm, `scorer.1` a Linear, then **`ggml_gelu_erf`** — PyTorch's default
`nn.GELU()` is the exact erf form, not the tanh approximation. The encoder layers use
**ReLU**, which is `nn.TransformerEncoderLayer`'s default activation.

Bindings come from **translate-c over the real headers** (`b.addTranslateC` in `build.zig`),
not hand-written `extern fn` and not `@cImport` (removed in this Zig). Struct layouts are
therefore the compiler's, not guesses.

## What backbones fit

**laya-vision uses `HuggingFaceTB/SmolVLM-256M-Instruct`** (Idefics3 architecture: SigLIP vision
tower, pixel-shuffle connector, SmolLM2-135M text decoder with 30 layers at d=576). Laya's own
ModernBERT text encoder is what the fork replaced; the 421M figure people quote is upstream Laya,
not this fork.

What fits is governed by two hard requirements in `VLMDecisionModel`:

1. ```python
   d = backbone.config.text_config.hidden_size
   ```
   so the backbone must be a VLM whose config has a nested `text_config` — the Idefics3/SmolVLM
   layout.

2. It calls `encoder.get_image_features(pixel_values, pixel_attention_mask)` and takes
   **`.pooler_output`** from it; the text path separately takes `.last_hidden_state` from the
   encoder forward. The backbone must expose the image-token path that way.

So in practice:

- **Drop-in, same family:** any SmolVLM / SmolVLM2 size — 256M, 500M, 2.2B — and
  **Idefics3-8B-Llama3**. Same code, no changes.
- **Plausible with light work:** Qwen2.5-VL / Qwen3-VL, Gemma 3, Gemma 4 (vision), InternVL — all
  are `image-text-to-text` with a vision tower plus a causal text decoder. They generally have
  `get_image_features`-equivalent hooks, but the tensor naming and processor differ, so
  `build_vlm_model` and `ImagePrep` need attention.
- **Doesn't fit cleanly:** LLaVA-style stacks (separate projector + plain `LlamaForCausalLM`),
  because the image path isn't reached through `AutoModel.get_image_features`.

Two constraints beyond the code shape:

- **`ImagePrep.check` asserts SmolVLM/Idefics3 preprocessing** — `image_mean`/`image_std` = 0.5,
  rescale 1/255, and a matching `image_seq_len` (64 here). A backbone with different
  normalization fails that check loudly, by design.
- **The head is not backbone-agnostic.** It is 36 tensors whose shapes are tied to d=576
  (2304-wide MLP, 9 heads). A bigger backbone means a differently-shaped head and therefore
  retraining — the published `all3-3ep/best` checkpoint is 576-specific. The `image_seq_len` and
  `head_max_len` budgets are baked into the layout too.

For the ggml port specifically, only SmolVLM-256M tensors are converted; the Llama-30L + SigLIP
structure generalizes to the other SmolVLM/Idefics3 sizes, but each would need its own weight
conversion.

## Build

The only dependency is **ggml itself** — no llama.cpp. Defaults point at a standalone ggml
checkout next to this project (`../../ggml`); override with `-Dggml-include` / `-Dggml-lib`.
Vulkan and CPU are both built:

```sh
cmake -S ../../ggml -B .build/ggml \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_VULKAN=ON -DGGML_CPU=ON -DGGML_BLAS=OFF -DGGML_CUDA=OFF \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
cmake --build .build/ggml -j8 --target ggml ggml-base ggml-cpu ggml-vulkan

zig build
```

Standalone ggml puts `ggml`, `ggml-base` and `ggml-cpu` in `<build>/src` and the Vulkan
backend one level down in `<build>/src/ggml-vulkan`; `build.zig` adds both library paths.

## Backends

The program takes a backend argument and defaults to CPU:

```sh
head <head_blobs_dir> <oracle_case_dir> [cpu|vulkan]
```

Tensors are created with `no_alloc = true`, allocated by the backend with
`ggml_backend_alloc_ctx_tensors`, and weights are uploaded with `ggml_backend_tensor_set`.
Writing into `tensor->data` directly would be wrong under Vulkan, where that pointer is not
host memory.

## Reproduce the verification

```sh
# 1. weights -> 36 head tensors (+ raw f32 blobs for the Zig side)
python3 tools/split_weights.py ../.weights/model.safetensors ../.weights/
# blobs/ are emitted by the splitter; see tools/ for the exact form

# 2. reference: run the real model, capture encoder output + option logits
.venv-oracle/bin/python oracle/dump_oracle.py ../.weights/model_dir ../.weights/oracle

# 3. Zig head must reproduce step 2
for c in choice noul score; do
  LD_LIBRARY_PATH=.build/llama/bin ./zig-out/bin/head ../.weights/head_blobs ../.weights/oracle/$c
done
```

## Verified results

Output is per-option probability against the PyTorch model, on a deterministic synthetic
fixture (210x160 image, three questions, `batch_size=1`):

| case | qtype | k | L | temperature | max abs prob diff (CPU) | (Vulkan) |
|---|---:|---:|---:|---:|---:|---:|
| choice | 0 | 4 | 124 | 3.1510 | 5.96e-8 | 1.36e-6 |
| noul | 2 | 2 | 123 | 1.5558 | 0.0 | 2.52e-6 |
| score | 1 | 3 | 125 | 1.0000 | 8.94e-8 | 1.31e-5 |

All six combinations pass. Max absolute logit difference 4.77e-7 on CPU and 1.65e-4 on
Vulkan — float32 accumulation differences, not semantic ones.

## Gotchas found while porting

- **`ggml_permute` arguments are destinations**, not sources: `result->ne[axis[i]] = a->ne[i]`.
  For a transposition the two readings agree, so the bug only shows up on 3-cycles —
  `[HD, NH, L] -> [L, HD, NH]` is `(1,2,0,3)`, not `(2,0,1,3)`. Getting it wrong produced a
  silent `[9,124,64]` and an assert several ops later.
- **`ggml_reshape_*` and `ggml_mul_mat` need contiguous operands.** Slices of a packed QKV
  matmul and the permuted attention tensors must be materialised with `ggml_cont` first.
- An HF `Linear` weight `[out, in]` row-major is already `ne0=in, ne1=out` for ggml;
  no transpose is needed for `y = x W^T`.
- **Backend-agnostic allocation is not optional.** With `no_alloc = false` the tensors live
  in a host arena and only CPU works. Weights must go through `ggml_backend_tensor_set`, not
  a memcpy into `tensor->data`, or Vulkan reads a pointer that is not host memory.
- `struct ggml_backend` is opaque, so `ggml_backend_t` translates to `?*struct_ggml_backend`,
  not a `[*c]` pointer — the CPU/Vulkan init functions do not share the `[*c]` convention of
- `struct ggml_backend` is opaque, so `ggml_backend_t` translates to `?*struct_ggml_backend`,
  not a `[*c]` pointer — the CPU/Vulkan init functions do not share the `[*c]` convention of
  the tensor-returning ops.
- **RoPE convention is `GGML_ROPE_TYPE_NORMAL`, not NEOX.** NEOX is what llama.cpp uses for
  most Llama checkpoints and it is wrong for SmolVLM. The failure is deceptive: at position 0
  rope is identity, so token 0 comes out *exact* and everything after it corrupts — which reads
  as a mask, GQA or position bug, not a rope bug. Print per-position max |diff|; if token 0 is
  exact and token 1 is not, suspect rope first. 41.0 max diff versus 1.0e-4.
- **Vulkan needs `GGML_VK_DISABLE_F16=1` on deep graphs.** The default f16 path cost three
  orders of magnitude over 30 layers (7.1e-2 versus 2.2e-4 against the reference).
  `GGML_VK_DISABLE_COOPMAT=1` alone changes nothing. Shallow graphs are unaffected.
- **Take hyperparameters from GGUF metadata, not the HF config.** This text config has no
  `rope_theta`, so trusting the HF default gives 10000, while `llama.rope.freq_base` says
  100000. Guessing silently degrades every position.

- **`ggml_backend_alloc_ctx_tensors` returns a buffer, and nothing frees it.** Ignore the
  return value and every stage leaks its whole allocation for the life of the process. On CPU
  that is host memory; on Vulkan it is device memory, so a second stage in the same process
  fails to allocate a buffer that would have fit on its own. `ggml_free(ctx)` does not free it.
- **Each `.zig` file in the module has its own globals.** A `var io_g: std.Io` at the top of
  `vision.zig` is a different variable from the one in `laya.zig`, so an imported module's
  helpers read `undefined` and segfault instantly. `io` is passed explicitly into each entry
  point for that reason.
- **PIL resamples 8-bit images a row at a time and rounds in between.** A two-hop resize is
  therefore four passes with three roundings, not two. Leaving out the intra-resize rounding
  put 15,004 pixels more than half a level out instead of 4,091, with outliers of 22 levels
  instead of 1.

## Status

The whole model runs end to end in one binary on both backends:

```sh
laya <model_dir> <text.gguf> <mmproj.gguf> <head_blobs> <image.png> <cpu|vulkan> [case]
```

Each stage against its own reference dump from the real PyTorch model:

| stage | file | CPU | Vulkan |
|---|---|---:|---:|
| preprocessing, PNG -> pixels | `src/preprocess.zig` | within 1 grey level, mean 0.005 | backend-free |
| tokenizer + layout, text -> ids | `src/tokenizer.zig` | 2083/2083 corpus rows exact | backend-free |
| vision tower + connector | `src/vision.zig` | 3.79e-5 max rel | 6.54e-5 max rel |
| encoder text stack, 30 layers | `src/encoder.zig` | 1.03e-4 max abs | 2.21e-4, needs `GGML_VK_DISABLE_F16=1` |
| decision head + act head | `src/head.zig` | 5.96e-8 max prob diff | 1.36e-6 |

End to end from the PNG, all three fixture questions, both backends:

| case | max prob diff, CPU | max prob diff, Vulkan |
|---|---:|---:|
| choice | 1.64e-4 | 1.63e-4 |
| noul | 7.96e-5 | 7.98e-5 |
| score | 5.16e-5 | 5.06e-5 |

With the reference pixels substituted for the PNG -- `LAYA_ORACLE_PIXELS=1` -- the same run
reproduces the reference decisions to **3.0e-7**. That is the number that says the compute is
right: the token ids (0 of 124 mismatched), the marker positions, and every graph agree, and
what is left over is the image path.

### The one honest gap

Preprocessing agrees with the HuggingFace processor to within one grey level everywhere, a mean
of 0.005 levels, with 0.5% of pixels differing by one level. The model is sensitive enough that
this lands as about 1e-4 in the output probabilities.

That is a floor rather than a defect, and it is measurable: perturbing the **reference** pixels
by one level on the same pixels moves the output by the same order -- 7.6e-2 in image features,
against 1.0e-1 here. Closing it would mean reproducing PIL's resample bit for bit instead of
using torch's antialiased resample weights, which is what `laya/preprocess.py` itself uses on
its fast path.

### Performance

ReleaseFast, one process per stage so each includes its own weight load:

| stage | CPU | Vulkan |
|---|---:|---:|
| preprocessing (backend-free) | 24.0 s | - |
| vision tower | 4.05 s | 0.86 s |
| encoder | 0.88 s | 0.73 s |
| head | 0.08 s | 0.15 s |
| **driver, PNG + 3 questions** | **30.7 s** | **26.7 s** |

Preprocessing is the outlier and has not been optimised: it runs ~8.7 G MACs densely, and the
resample kernels are ~95% zeros, so a sparse formulation is worth roughly 50x. The Python's own
processor does the same work in about 13 ms.

### Also worth knowing

- [CPU-PRECISION.md](CPU-PRECISION.md) records a resolved issue: ggml-cpu quantises the f32
  GELU through an f16 table, which cost ~3.7e-3 relative per call and made this tower 200x
  worse on CPU than on Vulkan. Fixed, with the measurement that found it.
- [PLAN.md](PLAN.md) is the plan the remainder was built from. Everything in it is done.

Weights are read from `smolvlm-text.gguf` and `smolvlm-mmproj.gguf` through ggml's own
`gguf.h` -- no llama.cpp at runtime. Those GGUFs were produced by `tools/to_hf_dirs.py` plus
llama.cpp's converter; that conversion step is the last llama.cpp dependency in the toolchain.

## Licence

Code here is a port of a fork whose weights are CC BY-NC-SA 4.0 (non-commercial). See the
repository root.
