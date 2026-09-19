"""Image inputs via SigLIP2 + projector spliced into ModernBERT.

Uses the real convaiinnovations/laya checkpoint and google/siglip2-base-patch16-224, on CPU (fp32) by default or on
LAYA_TEST_DEVICE (e.g. cuda, where Agent.predict runs under bf16/fp16 autocast).
The projector is untrained here, so image answers are only checked for shape/validity, not correctness.
"""
import gc
import math
import os

import pytest
import torch
from PIL import Image as PILImage

import laya
from laya.common import build_sequence, collate_items
from laya.vision import DEFAULT_VISION_ENCODER, param_groups, split_image_state
from laya.vision_train import ItemBuilder, set_stage1_mode, stage1_step, synthetic_records

MODEL = "convaiinnovations/laya"
DEVICE = os.environ.get("LAYA_TEST_DEVICE", "cpu")
N_IMG = 64

TEXT_STATE = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
}
QUESTIONS = {
    "dept": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {"billing": "invoices, refunds", "technical": "bugs, outages", "sales": "pricing", "other": "else"},
    },
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "critical"]},
    "churn": {"type": "noul", "instructions": "Does the user threaten to cancel?"},
}
IMAGE_QUESTIONS = {
    "colour": {"type": "choice", "instructions": "What colour is the square?", "criteria": ["red", "green", "blue"]},
    "is_red": {"type": "noul", "instructions": "Is the square red?"},
    "brightness": {"type": "score", "instructions": "How bright is the image?", "criteria": ["dark", "medium", "bright"]},
}


TEXT_STATES = (TEXT_STATE, "plain string state", {"image": "a caption string stays text"})
to_internal = laya.Agent._to_internal


@pytest.fixture(scope="module")
def text_ref():
    """Outputs of the plain text model, computed once and the model released so only one ModernBERT-large is
    resident at a time (keeps peak RSS low)."""
    agent = laya.load(MODEL, device=DEVICE)
    assert not hasattr(agent.model, "vision") and agent.model.n_image_tokens == 0
    ref = {
        "tok": agent.tok,
        "outputs": [(agent.predict(s, QUESTIONS), raw_logits(agent, s, QUESTIONS)) for s in TEXT_STATES],
    }
    ref["image_error"] = None
    try:
        agent.predict({"image": square((255, 0, 0))}, IMAGE_QUESTIONS)
    except ValueError as e:  # not pytest.raises: its traceback would keep `agent` alive
        ref["image_error"] = str(e)
    del agent
    gc.collect()
    return ref


@pytest.fixture(scope="module")
def vision_agent(text_ref):
    torch.manual_seed(0)
    return laya.load(MODEL, device=DEVICE, vision_encoder=DEFAULT_VISION_ENCODER, n_image_tokens=N_IMG)


def square(rgb, size=64):
    return PILImage.new("RGB", (size, size), rgb)


def raw_logits(agent, state, questions):
    items = []
    for q in questions.values():
        seq, markers = build_sequence(agent.tok, state, agent._to_internal(q))[:2]
        items.append({"ids": seq, "markers": markers, "qtype": laya.QTYPES[q["type"]]})
    b = collate_items([items], agent.tok.pad_token_id)
    b = {k: v.to(agent.device) if torch.is_tensor(v) else v for k, v in b.items()}
    with torch.no_grad():
        logits, act = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    return logits.cpu(), act.cpu()


# ---- 1. text-only behaviour is unchanged ----------------------------------------------------------------------


def test_text_only_identical_with_vision(text_ref, vision_agent):
    assert vision_agent.model.n_image_tokens == N_IMG
    assert any(k.startswith("proj.") for k in vision_agent.model.state_dict())
    for state, (ref_pred, (ref_logits, ref_act)) in zip(TEXT_STATES, text_ref["outputs"]):
        assert vision_agent.predict(state, QUESTIONS) == ref_pred
        logits, act = raw_logits(vision_agent, state, QUESTIONS)
        assert torch.equal(logits, ref_logits) and torch.equal(act, ref_act)


def test_text_sequence_unchanged_by_image_args(text_ref):
    tok, q = text_ref["tok"], to_internal(QUESTIONS["dept"])
    ids, markers = build_sequence(tok, TEXT_STATE, q)
    ids2, markers2, image_pos = build_sequence(tok, TEXT_STATE, q, n_image_tokens=N_IMG, image_token_id=7)
    assert (ids, markers) == (ids2, markers2) and image_pos == []


def test_image_sequence_layout(text_ref):
    tok, q = text_ref["tok"], to_internal(QUESTIONS["dept"])
    img = square((255, 0, 0))
    assert split_image_state({"image": img, "note": "x"}) == (img, {"note": "x"})

    ids, markers, pos = build_sequence(tok, {"image": img, "note": "shelf 3"}, q, n_image_tokens=N_IMG, image_token_id=7)
    assert len(pos) == N_IMG and pos == list(range(pos[0], pos[0] + N_IMG))
    assert ids[pos[0] - 1] == tok.sep_token_id and ids[pos[-1] + 1] == tok.sep_token_id
    assert all(ids[p] == 7 for p in pos) and ids[-1] == tok.sep_token_id
    assert ids[pos[-1] + 2 : -1] == tok('{"note": "shelf 3"}', add_special_tokens=False)["input_ids"]
    assert all(ids[m] == tok.mask_token_id for m in markers)

    ids_only, _, pos_only = build_sequence(tok, laya.Image(img), q, n_image_tokens=N_IMG, image_token_id=7)
    assert pos_only == pos and ids_only == ids[: pos[-1] + 2]

    with pytest.raises(ValueError):
        build_sequence(tok, {"image": img}, q)


# ---- 2. image path ------------------------------------------------------------------------------------------


def _check_schema(ans):
    c = ans["colour"]
    assert c["type"] == "choice" and c["choice"] in ("red", "green", "blue")
    assert set(c["probabilities"]) == {"red", "green", "blue"}
    assert math.isclose(sum(c["probabilities"].values()), 1.0, abs_tol=1e-3)
    assert 0.0 <= c["confidence"] <= 1.0

    s = ans["brightness"]
    assert s["type"] == "score" and 0.0 <= s["score"] <= 2.0
    assert set(s["probabilities"]) == {"0", "1", "2"}
    assert math.isclose(sum(s["probabilities"].values()), 1.0, abs_tol=1e-3)

    n = ans["is_red"]
    assert n["type"] == "noul" and 0.0 <= n["noul"] <= 1.0 and 0.5 <= n["confidence"] <= 1.0
    for a in ans.values():
        assert 0.0 <= a["action"]["act_probability"] <= 1.0


def test_predict_on_images(vision_agent):
    gate = vision_agent.model.proj.gate
    assert gate.item() == 0.0  # fresh projector is a no-op: every image looks the same
    red0 = vision_agent.predict({"image": square((255, 0, 0))}, IMAGE_QUESTIONS)
    blue0 = vision_agent.predict(laya.Image(square((0, 0, 255))), IMAGE_QUESTIONS)
    assert red0["answers"] == blue0["answers"]
    try:
        with torch.no_grad():
            gate.fill_(1.0)
        red = vision_agent.predict({"image": square((255, 0, 0))}, IMAGE_QUESTIONS)
        blue = vision_agent.predict(laya.Image(square((0, 0, 255))), IMAGE_QUESTIONS)
        with_text = vision_agent.predict({"image": square((255, 0, 0)), "caption": "product photo"}, IMAGE_QUESTIONS)
    finally:
        with torch.no_grad():
            gate.zero_()
    for r in (red0, red, blue, with_text):
        assert set(r["answers"]) == set(IMAGE_QUESTIONS)
        _check_schema(r["answers"])
    # With the gate open, pixels reach the encoder: different images give different distributions.
    assert red["answers"]["colour"]["probabilities"] != blue["answers"]["colour"]["probabilities"]
    print("\nred :", red["answers"]["colour"]["probabilities"], "P(is_red)=", red["answers"]["is_red"]["noul"])
    print("blue:", blue["answers"]["colour"]["probabilities"], "P(is_red)=", blue["answers"]["is_red"]["noul"])


def test_mixed_batch_matches_separate(vision_agent):
    """A text item batched with an image item goes through inputs_embeds but must match the input_ids path."""
    img_item = ItemBuilder(vision_agent)(synthetic_records(1)[0])
    q = vision_agent._to_internal(QUESTIONS["churn"])
    seq, markers, _ = build_sequence(vision_agent.tok, TEXT_STATE, q, n_image_tokens=N_IMG)
    txt_item = {"ids": seq, "markers": markers, "qtype": laya.QTYPES["noul"]}
    m = vision_agent.model
    dev = vision_agent.device
    with torch.no_grad():
        b = collate_items([[txt_item, img_item]], vision_agent.tok.pad_token_id)
        assert b["image_index"].tolist() == [-1, 0]
        b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
        mixed, _ = m(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
                     pixel_values=b["pixel_values"], image_pos=b["image_pos"], image_index=b["image_index"])
        t = collate_items([[txt_item]], vision_agent.tok.pad_token_id)
        t = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in t.items()}
        alone, _ = m(t["input_ids"], t["attention_mask"], t["marker_pos"], t["marker_mask"], t["qtype"])
    torch.testing.assert_close(mixed[0, :2], alone[0, :2], atol=1e-4, rtol=1e-4)


def test_image_without_vision_raises(text_ref):
    assert text_ref["image_error"] and "vision_encoder" in text_ref["image_error"]


# ---- 3. stage-1 training smoke ------------------------------------------------------------------------------


def test_stage1_training_smoke(vision_agent):
    model = vision_agent.model
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    try:
        trainable = laya.freeze_for_alignment(model, train_head=True)
        groups = param_groups(model)
        assert {id(p) for p in trainable} == {id(p) for p in groups["proj"] + groups["head"]}
        opt = torch.optim.AdamW(
            [{"params": groups["proj"], "lr": 1e-3}, {"params": groups["head"], "lr": 2e-5}], weight_decay=0.01
        )
        set_stage1_mode(model)
        data = synthetic_records(6)
        build = ItemBuilder(vision_agent)
        torch.manual_seed(0)
        for step in range(3):
            items = [build(rec) for rec in data[2 * step : 2 * step + 2]]
            stats = stage1_step(model, collate_items([items], vision_agent.tok.pad_token_id), opt, vision_agent.device)
            print("step %d: %s" % (step + 1, stats))
            assert math.isfinite(stats["loss"]) and math.isfinite(stats["reward"])

        after = model.state_dict()
        changed = {k for k in before if not torch.equal(before[k], after[k])}
        assert changed, "no parameter changed"
        assert all(k.startswith(("proj.", "head.", "type_emb.", "scorer.", "act_head.")) for k in changed), sorted(changed)[:5]
        assert any(k.startswith("proj.") for k in changed)
        assert not any(k.startswith(("encoder.", "vision.")) for k in changed)
    finally:
        model.load_state_dict(before)
        for p in model.parameters():
            p.requires_grad = True
        model.eval()


# ---- data format + evaluation -------------------------------------------------------------------------------------


def test_vision_data_roundtrip(tmp_path, monkeypatch):
    import io
    import json

    from laya import vision_data

    def fake_source():
        for i in range(6):
            buf = io.BytesIO()
            square((10 * i, 0, 0), size=900).save(buf, "PNG")
            q = {"type": "noul", "instructions": "Is it red?", "criteria": None}
            # records 0/1 share image "img0", like VQAv2 questions on one COCO image
            yield {"id": str(i), "state_text": None, "question": q, "label": i % 2}, "img%d" % (i // 2 * 2 if i < 2 else i), buf.getvalue(), "val" if i == 5 else "train"

    monkeypatch.setitem(vision_data.SOURCES, "fake", ("fake/hub", fake_source))
    meta = vision_data.build_dataset("fake", str(tmp_path), {"train": 3, "val": 5}, workers=2, log=lambda m: None)
    vision_data.mark_ready(str(tmp_path), "fake", meta)
    assert meta["counts"] == {"train": 3, "val": 1} and meta["images"] == 3
    assert sorted(os.listdir(tmp_path)) == ["fake"]  # temp dir renamed away
    assert os.path.exists(tmp_path / "fake" / "_READY")
    lines = [json.loads(l) for l in open(tmp_path / "fake" / "train.jsonl")]
    assert [r["image"] for r in lines] == ["images/img0.jpg", "images/img0.jpg", "images/img2.jpg"]
    assert set(lines[0]) == {"id", "image", "state_text", "question", "label"}
    assert max(PILImage.open(tmp_path / "fake" / "images" / "img0.jpg").size) == 512
    recs = vision_data.read_records(str(tmp_path), "fake", "val")
    assert recs[0]["dataset"] == "fake" and os.path.isabs(recs[0]["image"])


def test_evaluate_reports_per_dataset(vision_agent):
    from laya.vision_train import evaluate, make_loader

    recs = synthetic_records(6)
    loader = make_loader(recs, ItemBuilder(vision_agent), 4, vision_agent.tok.pad_token_id, shuffle=False)
    res = evaluate(vision_agent.model, loader, vision_agent.device, vision_agent.temperature, vision_agent.temperature_by_options)
    m = res["synthetic"]
    assert m["n"] == 6 and 0 <= m["acc"] <= 1 and 0 <= m["ece"] <= 1 and 0 <= m["ece_temp"] <= 1 and m["nll"] > 0
