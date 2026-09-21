# Plan: what is not done

Every graph in the model now exists on ggml and is verified against PyTorch dumps — vision tower,
encoder, head. What is missing is the two ends: nothing yet turns an image or a question into
the tensors those graphs consume, and nothing drives all three in one process.

Spec = the code that currently does this in Python, which is the reference to port against, not
to guess at.

## 1. Image preprocessing (`png -> pixels.f32`)

**Today:** `vision` reads `pixels.f32`, dumped by `oracle/dump_pixels.py`.

**Spec:** `ImagePrep` in `laya/preprocess.py`. The checkpoint records no `image_size` or
`preprocess` key, so `from_config(..., default_backend="processor")` selects `image_size=512`,
`backend="processor"`, `interpolation="processor"` — confirmed at runtime, not inferred.

The "processor" path is two LANCZOS hops, then normalisation:

    210x160 --(longest_edge 2048)--> 2048x1560 --(max_image_size 512)--> 512x512
    then (v - 127.5) / 127.5          (image_mean = image_std = 0.5, rescale 1/255)

Output range is [-1, 1], square, all-ones pixel mask — a square resize never pads.

**Work:** port `stage1_size` and `_axis_weights` (torch's antialiased LANCZOS resample weights,
constructed in float64) and compose the two hops into one matrix pair, as the Python does. Add
a decoder for the input file; `stb_image` (single public-domain header) is the cheap option, or
accept raw RGB and skip the decoder entirely for a first cut.

**Verify:** byte-level agreement with `pixels.f32` for the fixture.

**Risk:** the resample weight construction is fiddly — kernel support, downsampling stretch,
border clipping, renormalisation. Low conceptual risk, high typo risk, and the failure mode is a
slightly wrong image rather than an error. Mitigation: the Python documents each step and the
upstream repo asserts agreement with torchvision in `tests/test_vlm.py`.

**Out of scope:** the `"gpu"` preprocessing backend (resize on device). The checkpoint uses
`"processor"`, so nothing needs it.

## 2. Tokenizer and layout (`text -> input_ids`)

**Today:** `input_ids.i32` comes from the oracle. Traced anchor for the fixture:

    [1, 11126, 42, 49189, 49152, 49190 x64, ...]

**Two separable parts, and the first is smaller than it looks.**

**(a) The layout.** `build_vlm_inputs` assembles a fixed string — no Jinja engine is needed,
because the chat template is already bypassed by hand:

    "<|im_start|>User:" + <image token run> + <state text>
      + "\n<qtype> question: <instructions><end_of_utterance>\nAssistant: Options:\n"
      + "- <option>\n" per option

Plus the budget arithmetic, which is where the details live and all of it is already read off
the checkpoint: `head_max_len=256`, `max_len=1024`, option text truncated to 48 tokens,
`opt_budget = head_max_len - sum(len(opt)+1)`, and when that falls below 16 every option is cut
to `max(4, (head_max_len-16)//n - 1)`. Option rendering must match `render_options` exactly:
choice -> `key: desc` (or bare `key` when the description is null/empty), score -> `level i: c`,
noul -> `false: ...` / `true: ...`.

**(b) The tokenizer.** Byte-level BPE, 49,280 entries. The GGUF already carries
`tokenizer.ggml.tokens` and `tokenizer.ggml.merges`, so the vocabulary and merge table are local
and need no Python. Special tokens (`<|im_start|>`, `<end_of_utterance>`,
`<fake_token_around_image>`, `<image>`) must be matched before BPE runs.

**Verify:** exact integer equality against the oracle's `input_ids.i32` for all three cases. No
tolerance — these are ids.

**Risk:** the pre-tokenizer. HF's `tokenizer.json` carries a specific pre-tokenization regex, and
that is the one place a port diverges silently, producing plausible-but-wrong ids. Mitigation:
dump the *intermediate* pre-tokenization split for the fixture and compare stage by stage rather
than only at the end.

**Fallback if (b) proves disproportionate:** declare the tokenizer out of scope for the C/Zig
port, keep it as a one-time Python step, and document the boundary explicitly. The compute is
the part that cannot be reused; a tokenizer is well-served by existing implementations.

## 3. One program

**Today:** three binaries chained by hand through files.

**Work:** one process, one backend, three graphs. Chain in memory — `vision` output becomes the
encoder's image splice, `enc_h` becomes the head's input — with no file round-trips. The
host-side steps stay host-side: im2col, the embedding gather and image splice, marker positions,
the causal mask.

**Open design question:** one ggml context holding all weights (~1 GB: 32 MB head + 622 MB text
+ 374 MB vision) versus three contexts sharing one backend. The latter keeps per-graph allocation
simple and is probably right; it needs measuring, not guessing.

**Verify:** the same decisions as the current chained runs (max |dprob| 3.6e-7), plus a single
invocation that takes an image and a question.

## 4. `act_head`

**Today:** not implemented. On the decision path? No — it produces the auxiliary
`action.act_probability`.

**Spec:** `[top1, top1-top2, normalised entropy, k/255]` concatenated with the pooled hidden
state -> 580 -> Linear(580,256) -> GELU -> Linear(256,2) -> softmax. `ggml_top_k` exists in
ggml.h and covers the top-2.

**Note:** the pooled vector is the *last real token* of the head's hidden states, not a marker,
so the head graph has to expose it. `VLMDecisionModel.forward` computes it as
`h[attention_mask.sum(-1) - 1]`.

**Verify:** against `act_pre.f32`, `act_logits.f32`, `act_probs.f32`.

## 5. Release build and timings

Everything so far is `-Odebug`, so **no number in this repository is a performance measurement.**
Build with `-Doptimize=ReleaseFast` and time the three stages on both backends. The only external
reference point is the model card: ~71 ms per image-question on an L4, bf16.

## Order

1 and 2 are independent and unlock real inputs. 3 needs the code but not necessarily 1 and 2 —
it can be built against the file-based fixtures first and rewired after. 4 is small and
independent. 5 comes last, once correctness is stable.

## Before calling any of this verified

The vision tower is 200x less accurate on CPU than Vulkan and that is unexplained —
see [CPU-PRECISION.md](CPU-PRECISION.md). Until it is resolved or explicitly caveated, the model
is not verified on CPU, which is one of the two required backends.

## Acceptance

A single binary takes a PNG and a question, runs on CPU and Vulkan, and returns the same decision
probabilities the Python model returns. The compute already meets that bar; the inputs and the
driver do not yet exist.
