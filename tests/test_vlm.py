"""Tests for the experimental SmolVLM-backed decision model. Downloads HuggingFaceTB/SmolVLM-256M-Instruct (~0.5 GB)."""
import math
import random
import time

import pytest
import torch
from PIL import Image

from laya.common import render_options
from laya.vlm import OPTION_BULLET, OPTION_END, VLMAgent, build_vlm_inputs, split_state
from laya.vlm_train import synthetic_examples, train

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

QUESTIONS = {
    "color": {
        "type": "choice",
        "instructions": "What color is the square?",
        "criteria": {"red": "the square is red", "blue": "the square is blue", "green": "the square is green"},
    },
    "size": {"type": "score", "instructions": "How much of the image does the square fill?", "criteria": ["tiny", "about half", "almost all"]},
    "is_red": {"type": "noul", "instructions": "Is the square red?"},
}


def square(color):
    img = Image.new("RGB", (96, 96), (255, 255, 255))
    img.paste(Image.new("RGB", (48, 48), color), (24, 24))
    return img


@pytest.fixture(scope="module")
def agent():
    torch.manual_seed(0)
    return VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device=DEVICE)


def check_schema(res, questions):
    assert res["model"] == "laya-vlm"
    assert set(res["answers"]) == set(questions)
    for qid, qdef in questions.items():
        a = res["answers"][qid]
        assert a["type"] == qdef["type"]
        assert 0.0 <= a["confidence"] <= 1.0
        assert 0.0 <= a["action"]["act_probability"] <= 1.0
        if a["type"] == "choice":
            assert list(a["probabilities"]) == list(qdef["criteria"])
            assert a["choice"] in qdef["criteria"]
            assert math.isclose(sum(a["probabilities"].values()), 1.0, abs_tol=1e-3)
        elif a["type"] == "score":
            assert list(a["probabilities"]) == [str(i) for i in range(len(qdef["criteria"]))]
            assert math.isclose(sum(a["probabilities"].values()), 1.0, abs_tol=1e-3)
            assert 0.0 <= a["score"] <= len(qdef["criteria"]) - 1
        else:
            assert 0.0 <= a["noul"] <= 1.0
    assert res["usage"]["input_tokens"] > 0


def test_predict_image_and_text(agent):
    for state in ({"image": square((220, 20, 20)), "caption": "a test card"}, {"images": [square((220, 20, 20)), square((20, 40, 220))]}):
        res = agent.predict(state, QUESTIONS)
        check_schema(res, QUESTIONS)
        assert res["usage"]["images"] == len(split_state(state)[0])
    res = agent.predict("Customer: I was billed twice, please refund.", QUESTIONS)
    check_schema(res, QUESTIONS)
    assert res["usage"]["images"] == 0
    res = agent.predict({"image": square((20, 40, 220))}, QUESTIONS, n_permutations=3)
    check_schema(res, QUESTIONS)


def test_bidirectional_option_attention(agent):
    agent.model.option_attention = "bidirectional"
    try:
        check_schema(agent.predict({"image": square((20, 40, 220))}, QUESTIONS), QUESTIONS)
    finally:
        agent.model.option_attention = "causal"


def test_marker_positions(agent):
    tok = agent.processor.tokenizer
    (end_id,) = tok(OPTION_END, add_special_tokens=False)["input_ids"]
    for state in ({"image": square((0, 0, 255)), "note": "x"}, "plain text state"):
        for qdef in QUESTIONS.values():
            q = VLMAgent._to_internal(qdef)
            opts = render_options(q)
            order = list(range(len(opts)))
            random.Random(1).shuffle(order)
            it = build_vlm_inputs(agent.processor, state, q, option_order=order)
            ids, markers = it["ids"], it["markers"]
            assert len(markers) == len(opts)
            start = it["option_span"][0]
            for j, m in enumerate(markers):
                assert ids[m] == end_id
                assert tok.decode(ids[start : m + 1]) == OPTION_BULLET + opts[order[j]] + OPTION_END
                start = m + 1
            assert markers[-1] == len(ids) - 1 == it["option_span"][1] - 1


def test_train_head_only(agent):
    model = agent.model
    enc_names = ["vision_model.embeddings.patch_embedding.weight", "connector.modality_projection.proj.weight",
                 "text_model.embed_tokens.weight", "text_model.layers.29.mlp.down_proj.weight", "text_model.norm.weight"]
    enc_params = dict(model.encoder.named_parameters())
    enc_before = {n: enc_params[n].detach().clone() for n in enc_names}
    head_before = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("encoder.")}

    losses = train(model, agent.processor, synthetic_examples(4), steps=3, batch_size=2, freeze="head", device=DEVICE)

    assert len(losses) == 3 and all(math.isfinite(x) for x in losses)
    assert all(p.grad is None for p in model.encoder.parameters())
    for n in enc_names:
        assert torch.equal(enc_params[n], enc_before[n]), n
    head_now = dict(model.named_parameters())
    changed = [n for n, v in head_before.items() if not torch.equal(head_now[n], v)]
    assert any(n.startswith("scorer.") for n in changed)
    assert any(n.startswith("head.") for n in changed)
    assert not model.training


@pytest.mark.parametrize("include_backbone", [True, False])
def test_save_load_roundtrip(agent, tmp_path, include_backbone):
    agent.temperature = [1.3, 0.8, 1.1]
    state = {"image": square((220, 20, 20)), "caption": "a test card"}
    before = [agent.predict(state, QUESTIONS), agent.predict("plain text", QUESTIONS)]
    agent.save(str(tmp_path), include_backbone=include_backbone)
    loaded = VLMAgent(str(tmp_path), device=DEVICE)
    after = [loaded.predict(state, QUESTIONS), loaded.predict("plain text", QUESTIONS)]
    assert after == before
    assert loaded.cfg["temperature"] == [1.3, 0.8, 1.1]
    del loaded


def test_latency(agent):
    q = {"is_red": QUESTIONS["is_red"]}
    for name, state in (("image", {"image": square((220, 20, 20))}), ("text", "Customer: I was billed twice, please refund.")):
        agent.predict(state, q)  # warm-up
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            agent.predict(state, q)
            if DEVICE == "mps":
                torch.mps.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        print("\nlatency %s state, 1 noul question, %s: median %.1f ms (min %.1f, max %.1f)" % (name, DEVICE, ts[2], ts[0], ts[-1]))
