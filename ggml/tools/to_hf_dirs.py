#!/usr/bin/env python3
"""Repack the laya-vision encoder into two HuggingFace-layout directories that
llama.cpp's converters accept.

The checkpoint's encoder is stored under `encoder.` with a `text_model.` / `vision_model.`
split, which is `AutoModel`'s state-dict shape, not the HF `Idefics3ForConditionalGeneration`
layout. llama.cpp splits SmolVLM conversion in two:

  * the vision tower + connector become an `mmproj` GGUF (conversion/smolvlm.py is a
    MmprojModel; its filter_tensors keeps only vision_model / model.connector tensors)
  * the text model is converted on its own, as an ordinary Llama

so this produces two directories:

  <out>/text/    config.json = the text_config promoted to top level, architectures
                 ["LlamaForCausalLM"]; weights renamed model.text_model.X -> model.X
  <out>/mmproj/  config.json = the Idefics3 config; weights encoder.X -> model.X

The checkpoint has no lm_head (it is AutoModel, not AutoModelForCausalLM -- the model never
generates). The text GGUF is only ever used for its hidden states, but both the converter and
the loader want the tensor present, so lm_head.weight is materialised as a copy of
embed_tokens.weight and tie_word_embeddings is set true to match.

Usage:
    to_hf_dirs.py <model.safetensors> <backbone_config.json> <processor_dir> <out_dir>
"""

import json
import os
import shutil
import struct
import sys

DTYPE_SIZE = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "U8": 1, "I8": 1}


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def nbytes(e):
    n = 1
    for d in e["shape"]:
        n *= d
    return n * DTYPE_SIZE[e["dtype"]]


def write_safetensors(path, shapes, blobs):
    hdr, off = {}, 0
    for name in shapes:
        b = blobs[name]
        hdr[name] = {"dtype": "F32", "shape": shapes[name], "data_offsets": [off, off + len(b)]}
        off += len(b)
    blob = json.dumps(hdr, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name in shapes:
            f.write(blobs[name])
    return off


def main(argv):
    if len(argv) != 5:
        print(__doc__)
        return 2
    src, cfg_path, proc_dir, out_dir = argv[1:]

    hdr, base = read_header(src)
    cfg = json.load(open(cfg_path))

    enc = {k: v for k, v in hdr.items() if k.startswith("encoder.")}

    text_shapes, text_blobs = {}, {}
    mm_shapes, mm_blobs = {}, {}
    embed_key = None

    with open(src, "rb") as f:
        for name, e in enc.items():
            s, en = e["data_offsets"]
            f.seek(base + s)
            raw = f.read(en - s)
            if len(raw) != nbytes(e):
                raise SystemExit("short read: %s" % name)
            short = name[len("encoder."):]
            if short.startswith("text_model."):
                new = "model." + short[len("text_model."):]
                text_shapes[new] = e["shape"]
                text_blobs[new] = raw
                if new == "model.embed_tokens.weight":
                    embed_key = new
            else:
                new = "model." + short
                mm_shapes[new] = e["shape"]
                mm_blobs[new] = raw

    if embed_key is None:
        raise SystemExit("embed_tokens not found; cannot synthesise lm_head")
    # never used for generation, but the converter wants it; tie it
    text_shapes["lm_head.weight"] = text_shapes[embed_key]
    text_blobs["lm_head.weight"] = text_blobs[embed_key]

    # ---- text dir ----
    tdir = os.path.join(out_dir, "text")
    os.makedirs(tdir, exist_ok=True)
    tcfg = dict(cfg["text_config"])
    tcfg["architectures"] = ["LlamaForCausalLM"]
    tcfg["model_type"] = "llama"
    tcfg["tie_word_embeddings"] = True
    tcfg["vocab_size"] = text_shapes[embed_key][0]
    for junk in ("_name_or_path", "perceiver_config", "_attn_implementation_autoset"):
        tcfg.pop(junk, None)
    with open(os.path.join(tdir, "config.json"), "w") as f:
        json.dump(tcfg, f, indent=2)
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        p = os.path.join(proc_dir, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(tdir, fn))
    tb = write_safetensors(os.path.join(tdir, "model.safetensors"), text_shapes, text_blobs)

    # ---- mmproj dir ----
    mdir = os.path.join(out_dir, "mmproj")
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(mdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    # llama.cpp reads image_mean/image_std/size off the top level, but the Idefics3
    # processor nests them under "image_processor". Flatten, keeping image_seq_len.
    pcfg = os.path.join(proc_dir, "processor_config.json")
    if os.path.exists(pcfg):
        raw = json.load(open(pcfg))
        flat = dict(raw.get("image_processor", raw))
        for k, v in raw.items():
            if k != "image_processor":
                flat.setdefault(k, v)
        # llama.cpp takes clip's resize target from size.longest_edge, which the Idefics3
        # processor uses for its 2048px stage-1 hop. The model was trained and evaluated at
        # 512 with do_image_splitting=False, which is what yields image_seq_len=64, so point
        # both at 512 to match the Python path.
        flat["size"] = {"longest_edge": 512}
        with open(os.path.join(mdir, "preprocessor_config.json"), "w") as f:
            json.dump(flat, f, indent=2)
    mb = write_safetensors(os.path.join(mdir, "model.safetensors"), mm_shapes, mm_blobs)

    print("text   : %3d tensors  %.1f MiB  -> %s" % (len(text_shapes), tb / 1048576, tdir))
    print("mmproj : %3d tensors  %.1f MiB  -> %s" % (len(mm_shapes), mb / 1048576, mdir))
    print("vocab  : %d" % text_shapes[embed_key][0])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
