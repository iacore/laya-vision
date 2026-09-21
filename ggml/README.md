# laya-vision decision head on ggml, in Zig

A C/Zig implementation of the part of `thaitea/laya-vision-smolvlm-256m` that ggml has no
equivalent for: the 36-tensor **decision head**. The 470-tensor SmolVLM encoder is not
reimplemented here.

The head is verified against the real PyTorch model, not against itself: `head` reads the
encoder output captured from a Python forward pass and must reproduce that model's option
probabilities. All three primitives pass at float32 precision.

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

## Status

Done and verified on **both Vulkan and CPU**, each against a reference dump from the real
PyTorch model:

- the **decision head** (`src/head.zig`) — `enc_h` -> option probabilities, max |dprob| 5.96e-8
  on CPU and 1.36e-6 on Vulkan, all three primitives
- the **encoder text stack** (`src/encoder.zig`) — token ids + image features -> `enc_h`,
  max |diff| 1.03e-4 on CPU and 2.21e-4 on Vulkan with `GGML_VK_DISABLE_F16=1`, all three
  primitives
- the **vision tower and connector** (`src/vision.zig`) — pixels -> 64 image vectors. Verified
  on Vulkan at max rel 6.5e-5. **The CPU backend is inexact here and that is unresolved** — see
  [CPU-PRECISION.md](CPU-PRECISION.md)
- the **chain of the two**: feeding the encoder's own `enc_h` (not the oracle's — the files
  differ) into the head reproduces the reference decisions to max |dprob| 3.6e-7

So the whole compute path from token ids + image features to option probabilities is verified.
What is *not* verified is everything upstream of that.

**Open issue:** [CPU-PRECISION.md](CPU-PRECISION.md) — the vision tower is 200x less accurate on
CPU than on Vulkan, the reverse of every other graph here. Unresolved.

**Not done — the two ends are still Python.** You cannot yet hand this an image or a question:

- the **image preprocessing** (`ImagePrep`'s resize and normalise), so `vision` consumes a
  `pixels.f32` dumped from the oracle rather than a PNG. There is no `png -> pixels` path in
  the port, though `pixels -> img_feats` now exists.
- the **tokenizer** and the `build_vlm_inputs` layout, so `input_ids` come from the oracle.
  There is no `text -> token ids` path in the port.
- **one program.** `encoder` and `head` are separate binaries chained through a file by hand;
  nothing in the code drives both.

Also outstanding: `act_head` (its 4 hand features need `ggml_top_k`; auxiliary, not on the
decision path). Everything here is built at `-Odebug`, so no timing in this README should be
taken as a performance measurement.

Weights are read from `smolvlm-text.gguf` and `smolvlm-mmproj.gguf` through ggml's own
`gguf.h` — no llama.cpp at runtime. Those GGUFs were produced by `tools/to_hf_dirs.py` plus
llama.cpp's converter; that conversion step is the last remaining llama.cpp dependency and
would need replacing to drop it from the toolchain entirely.

## Licence

Code here is a port of a fork whose weights are CC BY-NC-SA 4.0 (non-commercial). See the
repository root.
