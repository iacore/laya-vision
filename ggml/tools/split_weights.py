#!/usr/bin/env python3
"""Split the laya-vision state dict into the decision head and the encoder backbone.

The published thaitea/laya-vision-smolvlm-256m checkpoint is one flat state dict of
VLMDecisionModel:

    encoder.*     470 tensors   SmolVLM/Idefics3 backbone (SigLIP + connector + Llama)
    type_emb.*      1 tensor    [3, 576]
    head.*         24 tensors   2 x TransformerEncoderLayer(576, 9, 2304)
    scorer.*        6 tensors   LayerNorm -> Linear -> GELU -> Linear
    act_head.*      4 tensors   Linear(580,256) -> GELU -> Linear(256,2)
    temperature     1 tensor    [3]

Only those last 36 tensors are ours to implement; the encoder is handled by llama.cpp.
This tool carves them into a small standalone safetensors file so the Zig/ggml head can
be developed and numerically verified without touching a 946 MB blob.

Stdlib only; no torch, no safetensors package.

Usage:
    split_weights.py <model.safetensors> <out_dir>

Writes <out_dir>/head.safetensors and <out_dir>/head.json
"""

import json
import os
import struct
import sys

HEAD_PREFIXES = ("head.", "scorer.", "act_head.", "type_emb.", "temperature")
DTYPE_SIZE = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "U8": 1, "I8": 1}


def read_header(path):
    """Return (header_dict, data_start_offset)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("%s: too short to be safetensors" % path)
        (n,) = struct.unpack("<Q", raw)
        blob = f.read(n)
        if len(blob) != n:
            raise ValueError("%s: truncated header" % path)
    hdr = json.loads(blob)
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def nbytes(entry):
    n = 1
    for d in entry["shape"]:
        n *= d
    return n * DTYPE_SIZE[entry["dtype"]]


def is_head(name):
    return name.startswith(HEAD_PREFIXES)


def write_safetensors(path, header, blobs):
    """Write a safetensors file. blobs maps tensor name -> raw bytes."""
    out_hdr = {}
    offset = 0
    for name in header:
        b = blobs[name]
        out_hdr[name] = {
            "dtype": header[name]["dtype"],
            "shape": header[name]["shape"],
            "data_offsets": [offset, offset + len(b)],
        }
        offset += len(b)

    blob = json.dumps(out_hdr, separators=(",", ":")).encode("utf-8")
    pad = (-len(blob)) % 8
    blob += b" " * pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name in header:
            f.write(blobs[name])
    return offset


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2

    src, out_dir = argv[1], argv[2]
    os.makedirs(out_dir, exist_ok=True)

    hdr, data_start = read_header(src)
    total = len(hdr)
    head_names = [k for k in hdr if is_head(k)]
    enc_names = [k for k in hdr if not is_head(k)]

    if not head_names:
        raise SystemExit("no head tensors found - is this a laya-vision checkpoint?")

    blobs = {}
    with open(src, "rb") as f:
        for name in head_names:
            e = hdr[name]
            start, end = e["data_offsets"]
            if end - start != nbytes(e):
                raise SystemExit("%s: offset/size mismatch in header" % name)
            f.seek(data_start + start)
            b = f.read(end - start)
            if len(b) != end - start:
                raise SystemExit("%s: short read" % name)
            blobs[name] = b

    ordered = {k: hdr[k] for k in head_names}
    head_path = os.path.join(out_dir, "head.safetensors")
    written = write_safetensors(head_path, ordered, blobs)

    manifest = {
        "source": os.path.basename(src),
        "head_tensors": len(head_names),
        "encoder_tensors": len(enc_names),
        "total_tensors": total,
        "bytes": written,
        "tensors": [
            {"name": k, "dtype": hdr[k]["dtype"], "shape": hdr[k]["shape"]} for k in head_names
        ],
    }
    with open(os.path.join(out_dir, "head.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    vhdr, vstart = read_header(head_path)
    if set(vhdr) != set(head_names):
        raise SystemExit("verify: tensor set differs after write")
    got = 0
    with open(head_path, "rb") as f:
        for name in head_names:
            s, e = vhdr[name]["data_offsets"]
            f.seek(vstart + s)
            b = f.read(e - s)
            if b != blobs[name]:
                raise SystemExit("verify: %s bytes differ after round-trip" % name)
            if vhdr[name]["shape"] != hdr[name]["shape"]:
                raise SystemExit("verify: %s shape differs" % name)
            got += len(b)

    print("source tensors        : %d" % total)
    print("encoder tensors kept  : %d (left for llama.cpp)" % len(enc_names))
    print("head tensors written  : %d" % len(head_names))
    print("head payload          : %.2f MiB" % (got / 1048576))
    print("wrote                 : %s" % head_path)
    print("wrote                 : %s" % os.path.join(out_dir, "head.json"))
    print("verify                : round-trip byte-identical, %d bytes" % got)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
