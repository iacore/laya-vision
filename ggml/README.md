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
  the tensor-returning ops.

## Status

Done and verified: weight split, reference oracle, and the decision head in Zig on ggml
running on **both Vulkan and CPU**.

**Not done:** the encoder. `enc_h` currently comes from the Python oracle dump. The encoder
is SigLIP (12 layers, d=768) + an Idefics3 connector (pixel shuffle + one matmul) + a
Llama-30L text model (d=576, 9 q / 3 kv heads, head_dim 64, SwiGLU, RMSNorm, RoPE), to be
built on ggml directly. Also outstanding: the `build_vlm_inputs` token layout, the
tokenizer, and `act_head` (its 4 hand features need `ggml_top_k`; auxiliary, not on the
decision path).

The two GGUFs (`smolvlm-text.gguf`, `smolvlm-mmproj.gguf`) produced by `tools/to_hf_dirs.py`
+ llama.cpp's converter are reusable as plain weight containers — ggml's `gguf.h` reads
them without llama.cpp — but the converter step itself would need replacing if llama.cpp is
removed from the toolchain entirely.

## Licence

Code here is a port of a fork whose weights are CC BY-NC-SA 4.0 (non-commercial). See the
repository root.
