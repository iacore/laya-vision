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
| `encoder.*` | 470 | llama.cpp / ggml, already implemented in C |
| `type_emb`, `head.*`, `scorer.*`, `act_head.*`, `temperature` | 36 | **this directory** |

Two properties of *this* checkpoint collapse the work:

- `vlm_agent_config.json` says `option_attention: "causal"`, so the custom bidirectional
  4D option-block mask is not in play — llama.cpp's ordinary causal attention is correct.
- There is **no `lm_head`**. The model never generates tokens; it only needs the decoder's
  final hidden states, which is what llama.cpp's embeddings path exposes.

So the port is: weight conversion, the head, the token layout, and the driver.

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

Needs a ggml build and its headers. Defaults point at the llama.cpp tree used in development:

```sh
cmake -S ~/computing/extension/llama.cpp -B .build/llama \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build .build/llama -j8 --target ggml ggml-base ggml-cpu

zig build                 # -Dggml-include=... -Dggml-lib=... to override
```

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

| case | qtype | k | L | temperature | max abs prob diff |
|---|---:|---:|---:|---:|---:|
| choice | 0 | 4 | 124 | 3.1510 | 5.96e-8 |
| noul | 2 | 2 | 123 | 1.5558 | 0.0 |
| score | 1 | 3 | 125 | 1.0000 | 1.04e-7 |

Max absolute logit difference 4.77e-7. These are float32 accumulation differences, not
semantic ones.

## Gotchas found while porting

- **`ggml_permute` arguments are destinations**, not sources: `result->ne[axis[i]] = a->ne[i]`.
  For a transposition the two readings agree, so the bug only shows up on 3-cycles —
  `[HD, NH, L] -> [L, HD, NH]` is `(1,2,0,3)`, not `(2,0,1,3)`. Getting it wrong produced a
  silent `[9,124,64]` and an assert several ops later.
- **`ggml_reshape_*` and `ggml_mul_mat` need contiguous operands.** Slices of a packed QKV
  matmul and the permuted attention tensors must be materialised with `ggml_cont` first.
- An HF `Linear` weight `[out, in]` row-major is already `ne0=in, ne1=out` for ggml;
  no transpose is needed for `y = x W^T`.

## Status

Stages 1-3 of the port are done and verified: weight split, reference oracle, head in Zig.
**Not done:** running the encoder through llama.cpp to produce `enc_h` (stage 4), the
`build_vlm_inputs` token layout, the tokenizer, and `act_head` (the 4 hand features need
`ggml_top_k`; it is auxiliary and not part of the decision path).

## Licence

Code here is a port of a fork whose weights are CC BY-NC-SA 4.0 (non-commercial). See the
repository root.
