#!/usr/bin/env python3
"""Dump the remaining reference artefacts the encoder port needs.

  img_feats.f32   the 64x576 image embeddings the text model consumes, captured from the
                  real run, so the Llama stack can be verified without depending on image
                  preprocessing matching yet
  encoder.json    the shape/rope/norm constants the graphs are built from
  gguf tensors    names + shapes of every tensor in the two GGUFs, as the loader's contract

Usage: dump_extras.py <model_dir> <oracle_out_dir> <text.gguf> <mmproj.gguf>
"""

import json
import os
import sys

import numpy as np
import torch

import laya


def main(argv):
    if len(argv) != 5:
        print(__doc__)
        return 2
    model_dir, out_dir, text_gguf, mmproj_gguf = argv[1:]

    agent = laya.load_vlm(model_dir, device="cpu")
    model = agent.model
    cfg = agent.cfg

    from laya.vlm import build_vlm_inputs, collate_vlm, split_state, vlm_prefix
    from laya.common import QTYPES, render_options

    # rebuild the choice fixture exactly as dump_oracle.py does
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dump_oracle import fixture_image, CASES

    state = {"image": fixture_image(), "note": "synthetic fixture, deterministic"}

    cap = {}
    h = model.register_forward_pre_hook(
        lambda m, a, kw: cap.update(img_feats=kw.get("image_hidden_states")), with_kwargs=True
    )
    try:
        agent.predict(state, {"q": CASES["choice"]["answer"]}, batch_size=1)
    finally:
        h.remove()

    feats = cap["img_feats"][0].float().cpu().numpy()
    feats.tofile(os.path.join(out_dir, "img_feats.f32"))
    print("img_feats", feats.shape, "->", os.path.join(out_dir, "img_feats.f32"))

    tc = cfg_backbone = json.load(open(os.path.join(model_dir, "backbone", "config.json")))
    text = tc["text_config"]
    enc = {
        "image_token_id": tc.get("image_token_id"),
        "image_seq_len": feats.shape[0],
        "text_hidden": text["hidden_size"],
        "text_layers": text["num_hidden_layers"],
        "text_heads": text["num_attention_heads"],
        "text_kv_heads": text["num_key_value_heads"],
        "text_head_dim": text["head_dim"],
        "text_intermediate": text["intermediate_size"],
        "text_vocab": text["vocab_size"],
        "rope_theta": text.get("rope_theta"),
        "rope_scaling": text.get("rope_scaling"),
        "rms_norm_eps": text.get("rms_norm_eps"),
        "max_position_embeddings": text.get("max_position_embeddings"),
        "vision_hidden": tc["vision_config"]["hidden_size"],
        "vision_layers": tc["vision_config"]["num_hidden_layers"],
        "vision_heads": tc["vision_config"]["num_attention_heads"],
        "vision_intermediate": tc["vision_config"]["intermediate_size"],
        "vision_patch": tc["vision_config"]["patch_size"],
        "vision_image_size": tc["vision_config"]["image_size"],
        "scale_factor": tc.get("scale_factor"),
        "head_max_len": cfg.get("head_max_len"),
        "max_len": cfg.get("max_len"),
    }
    with open(os.path.join(out_dir, "encoder.json"), "w") as f:
        json.dump(enc, f, indent=2)
    print(json.dumps(enc, indent=2))

    sys.path.insert(0, os.path.expanduser("~/computing/learned/llama.cpp/gguf-py"))
    import gguf

    for path in (text_gguf, mmproj_gguf):
        r = gguf.GGUFReader(path)
        print()
        print("=== %s : %d tensors ===" % (os.path.basename(path), len(r.tensors)))
        for t in r.tensors:
            print("  %-42s %-8s %s" % (t.name, t.tensor_type.name, list(t.shape)[::-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
