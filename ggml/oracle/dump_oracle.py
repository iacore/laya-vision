#!/usr/bin/env python3
"""Dump reference tensors for the ggml/Zig decision head.

Runs the real laya-vision model on a deterministic synthetic fixture and captures, per
question, everything the head consumes and produces:

    enc_h.f32        [L, 576]   encoder last_hidden_state, BEFORE type_emb is added
    marker_pos.i32   [k]        index of each option terminator in the sequence
    marker_mask.u8   [k]
    qtype.i32        [1]        0=choice 1=score 2=noul
    m.f32            [k, 576]   the gathered marker rows fed to scorer
    logits.f32       [k]        scorer output, pre-softmax
    probs.f32        [k]        final probabilities (mask fill, /temperature, softmax)
    act_pre.f32      [580]      act_head input (576 pooled + 4 hand features)
    act_logits.f32   [2]
    act_probs.f32    [2]

The Zig head starts at enc_h and must reproduce logits/probs. Nothing here is a
reimplementation of the model: every tensor comes from a forward hook on the real
module, and `probs` is recomputed from the captured logits with the checkpoint's own
temperature so the arithmetic being checked is exactly the runtime's.

Usage:
    dump_oracle.py <model_dir> <out_dir>
"""

import json
import os
import sys

import numpy as np
import torch

import laya
from laya.common import QTYPES, confidence_from_probs, render_options, temp_bucket


def fixture_image():
    """Deterministic 210x160 RGB fixture; no file, no RNG."""
    h, w = 210, 160
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (xx * 255 // max(1, w - 1)).astype(np.uint8)
    img[..., 1] = (yy * 255 // max(1, h - 1)).astype(np.uint8)
    img[..., 2] = (((xx // 16) + (yy // 16)) % 2 * 255).astype(np.uint8)
    img[h // 4 : h // 2, w // 4 : w // 2] = 255
    return img


CASES = {
    "choice": {
        "answer": {
            "type": "choice",
            "instructions": "What setting does the image show?",
            "criteria": {
                "indoor": "Indoors",
                "outdoor": "Outdoors",
                "unknown": "Unclear",
                "abstract": "An abstract pattern",
            },
        }
    },
    "noul": {
        "answer": {
            "type": "noul",
            "instructions": "Is there a bright rectangular block in the image?",
        }
    },
    "score": {
        "answer": {
            "type": "score",
            "instructions": "How much readable text is in the image?",
            "criteria": ["None", "A few words", "Many words"],
        }
    },
}


def wbin(path, arr, dtype):
    np.ascontiguousarray(arr, dtype=dtype).tofile(path)


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    model_dir, out_dir = argv[1], argv[2]
    os.makedirs(out_dir, exist_ok=True)

    agent = laya.load_vlm(model_dir, device="cpu")
    model = agent.model
    temperature = agent.cfg["temperature"]
    by_options = agent.cfg.get("temperature_by_options", {})
    print("temperature           :", temperature)
    print("temperature_by_options:", by_options)
    print("option_attention      :", model.option_attention)
    print("dtype                 :", agent.cfg.get("dtype"))

    img = fixture_image()
    state = {"image": img, "note": "synthetic fixture, deterministic"}

    manifest = {"model_dir": os.path.abspath(model_dir), "cases": {}}

    for case, spec in CASES.items():
        qid = "q"
        cap = {}
        handles = [
            model.register_forward_pre_hook(
                lambda m, a, kw: cap.update(
                    input_ids=a[0], attention_mask=a[1], marker_pos=a[2],
                    marker_mask=a[3], qtype=a[4],
                ),
                with_kwargs=True,
            ),
            model.encoder.register_forward_hook(
                lambda m, i, o: cap.update(enc_h=getattr(o, "last_hidden_state", o[0]))
            ),
            model.scorer.register_forward_hook(
                lambda m, i, o: cap.update(m_in=i[0], logits=o)
            ),
            model.act_head.register_forward_hook(
                lambda m, i, o: cap.update(act_pre=i[0], act_logits=o)
            ),
        ]
        try:
            out = agent.predict(state, {qid: spec["answer"]}, batch_size=1)
        finally:
            for h in handles:
                h.remove()

        internal = agent._to_internal(spec["answer"])
        k = len(render_options(internal))
        qt = QTYPES[internal["t"]]
        t_scale = float(by_options.get(temp_bucket(qt, k), temperature[qt]))

        enc_h = cap["enc_h"][0].float().cpu().numpy()
        mpos = cap["marker_pos"][0, :k].cpu().numpy().astype(np.int64)
        mmask = cap["marker_mask"][0, :k].cpu().numpy().astype(np.uint8)
        qtype = np.array([cap["qtype"][0].item()], np.int64)
        m_in = cap["m_in"][0, :k].float().cpu().numpy()
        logits = cap["logits"][0, :k].float().cpu().numpy().reshape(-1)
        act_pre = cap["act_pre"][0].float().cpu().numpy().reshape(-1)
        act_logits = cap["act_logits"][0].float().cpu().numpy().reshape(-1)

        # exactly the runtime's arithmetic, reproduced from the captured logits
        z = logits / max(1e-3, t_scale)
        e = np.exp(z - z.max())
        probs = e / e.sum()
        act_probs = np.exp(act_logits - act_logits.max())
        act_probs = act_probs / act_probs.sum()

        d = os.path.join(out_dir, case)
        os.makedirs(d, exist_ok=True)
        wbin(os.path.join(d, "enc_h.f32"), enc_h, np.float32)
        wbin(os.path.join(d, "marker_pos.i32"), mpos, np.int32)
        wbin(os.path.join(d, "marker_mask.u8"), mmask, np.uint8)
        wbin(os.path.join(d, "qtype.i32"), qtype, np.int32)
        wbin(os.path.join(d, "m.f32"), m_in, np.float32)
        wbin(os.path.join(d, "logits.f32"), logits, np.float32)
        wbin(os.path.join(d, "probs.f32"), probs, np.float32)
        wbin(os.path.join(d, "act_pre.f32"), act_pre, np.float32)
        wbin(os.path.join(d, "act_logits.f32"), act_logits, np.float32)
        wbin(os.path.join(d, "act_probs.f32"), act_probs, np.float32)

        got = out["answers"][qid]
        manifest["cases"][case] = {
            "qtype": int(qt),
            "k": int(k),
            "L": int(enc_h.shape[0]),
            "temperature": t_scale,
            "marker_pos": [int(x) for x in mpos],
            "logits": [float(x) for x in logits],
            "probs": [float(x) for x in probs],
            "confidence": confidence_from_probs(probs, k),
            "model_answer": got,
            "tensors": {
                "enc_h": list(enc_h.shape),
                "m": list(m_in.shape),
                "act_pre": list(act_pre.shape),
            },
        }
        probs_s = json.dumps([round(float(x), 5) for x in probs])
        got_s = json.dumps(got)
        print(f"  {case:<7s} k={k} L={enc_h.shape[0]} temp={t_scale:.4f}  probs={probs_s}  answer={got_s}")

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote", os.path.join(out_dir, "manifest.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
