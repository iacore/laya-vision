"""Tests for the text-only decision runtime. Builds a checkpoint from a tiny random BERT (~1 MB): the
answers are noise, the schema is not."""
import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from laya.agent import Agent
from laya.common import build_model

ENCODER = "hf-internal-testing/tiny-random-BertModel"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

QUESTIONS = {
    "dept": {
        "type": "choice",
        "instructions": "Which team should handle the message?",
        "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs and outages"},
    },
    "urgency": {"type": "score", "instructions": "How urgent is the message?", "criteria": ["low", "medium", "high"]},
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
}
STATE = "Customer: I was billed twice, please refund."


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    """A checkpoint in the layout ``Agent`` loads: config, encoder config, tokenizer and weights."""
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("laya-text")
    cfg = {"encoder": ENCODER, "head_layers": 1}
    model = build_model(cfg)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(d / "model.safetensors"))
    model.encoder.config.save_pretrained(str(d / "encoder"))
    AutoTokenizer.from_pretrained(ENCODER).save_pretrained(str(d / "tokenizer"))
    (d / "rl_agent_config.json").write_text(json.dumps(cfg))
    return Agent(str(d), device=DEVICE)


def test_action_field_is_opt_in(agent):
    """The act head gets no gradient in training (the fine-tune folds it in as 0.0 * act.sum()), so its
    probability is untrained noise and must not ride along on every answer by default."""
    default, opted_in = agent.predict(STATE, QUESTIONS), agent.predict(STATE, QUESTIONS, include_action=True)
    assert set(default["answers"]) == set(QUESTIONS)
    for qid, qdef in QUESTIONS.items():
        a, b = default["answers"][qid], opted_in["answers"][qid]
        assert a["type"] == qdef["type"] and 0.0 <= a["confidence"] <= 1.0
        assert "action" not in a
        assert 0.0 <= b["action"]["act_probability"] <= 1.0
        assert {k: v for k, v in b.items() if k != "action"} == a
