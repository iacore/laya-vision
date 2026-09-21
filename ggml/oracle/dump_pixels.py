#!/usr/bin/env python3
"""Dump the exact preprocessed pixel tensor `ImagePrep` produces for the fixture.

The vision tower port has to start from the same numbers the Python path fed it, so the
resize/normalise is verified separately from the graph rather than tangled with it.

Usage: dump_pixels.py <model_dir> <out_dir>
"""

import json
import os
import sys

import numpy as np

import laya
from laya.vlm import split_state, vlm_prefix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_oracle import fixture_image


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    model_dir, out_dir = argv[1], argv[2]

    agent = laya.load_vlm(model_dir, device="cpu")
    images, _ = split_state({"image": fixture_image(), "note": "x"})
    prefix = vlm_prefix(agent.processor, images, agent.prep)

    pv = prefix["pixel_values"]
    pam = prefix["pixel_attention_mask"]
    arr = pv[0].numpy().astype(np.float32) if pv.ndim == 4 else pv.numpy().astype(np.float32)
    arr.tofile(os.path.join(out_dir, "pixels.f32"))
    np.asarray(pam.numpy(), dtype=np.uint8).tofile(os.path.join(out_dir, "pixel_mask.u8"))

    print("image_size      ", agent.prep.image_size)
    print("image_seq_len   ", agent.prep.image_seq_len)
    print("backend/prep    ", agent.prep.backend, agent.prep.interpolation)
    print("pixel_values    ", arr.shape, "->", os.path.join(out_dir, "pixels.f32"))
    print("pixel range     ", float(arr.min()), float(arr.max()), "mean", float(arr.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
