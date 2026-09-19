"""Training sketch for the SmolVLM-backed decision model (``laya.vlm``).

Fine-tunes on multiple-choice VQA with the same objective as the text model's notebooks: a soft
cross-entropy term plus a proper-scoring-rule policy-gradient term over noisy logits (``proper_reward``).
Options are shuffled per example so the causal backbone cannot learn a position prior.

Data sources (adapters take one HF ``datasets`` row each):
  * A-OKVQA (``HuggingFaceM4/A-OKVQA``)      -> ``choice``
  * ScienceQA (``derek-thomas/ScienceQA``)   -> ``choice`` (image optional, hint as context)
  * VQAv2 yes/no (``HuggingFaceM4/VQAv2``)   -> ``noul`` with soft target = fraction of "yes" votes

Smoke run on a tiny synthetic batch (no downloads beyond the backbone):
    python -m laya.vlm_train --synthetic --steps 3 --freeze head
"""
import argparse
import math
import random
from typing import Dict, Iterable, List, Optional

import torch

from .common import QTYPES, proper_reward, render_options
from .vlm import VLMAgent, VLMDecisionModel, build_vlm_inputs, collate_vlm, set_trainable

# ---------------------------------------------------------------------------------------------------------
# Examples: {"state": ..., "q": {"t", "ins", "crit"}, "target": [prob per option in label order]}
# ---------------------------------------------------------------------------------------------------------


def _one_hot(i: int, k: int) -> List[float]:
    return [1.0 if j == i else 0.0 for j in range(k)]


def _choice_example(state, question: str, choices: List[str], answer: int) -> Optional[Dict]:
    choices = [str(c) for c in choices]
    if len(set(choices)) != len(choices) or not 0 <= answer < len(choices):
        return None
    return {"state": state, "q": {"t": "choice", "ins": question, "crit": {c: None for c in choices}}, "target": _one_hot(answer, len(choices))}


def aokvqa_example(row: Dict) -> Optional[Dict]:
    return _choice_example({"image": row["image"]}, row["question"], row["choices"], int(row["correct_choice_idx"]))


def scienceqa_example(row: Dict) -> Optional[Dict]:
    state = {"context": row["hint"]} if row.get("hint") else {}
    if row.get("image") is not None:
        state["image"] = row["image"]
    return _choice_example(state or "", row["question"], row["choices"], int(row["answer"]))


def vqav2_yesno_example(row: Dict) -> Optional[Dict]:
    if row.get("answer_type") != "yes/no":
        return None
    votes = [a["answer"] if isinstance(a, dict) else a for a in row.get("answers") or []]
    votes = [v for v in votes if v in ("yes", "no")]
    if votes:
        p_yes = sum(v == "yes" for v in votes) / len(votes)
    elif row.get("multiple_choice_answer") in ("yes", "no"):
        p_yes = float(row["multiple_choice_answer"] == "yes")
    else:
        return None
    return {"state": {"image": row["image"]}, "q": {"t": "noul", "ins": row["question"], "crit": None}, "target": [1.0 - p_yes, p_yes]}


ADAPTERS = {
    "aokvqa": ("HuggingFaceM4/A-OKVQA", aokvqa_example),
    "scienceqa": ("derek-thomas/ScienceQA", scienceqa_example),
    "vqav2_yesno": ("HuggingFaceM4/VQAv2", vqav2_yesno_example),
}


def load_hf_examples(name: str, split: str = "train", limit: int = 1000) -> List[Dict]:
    """Stream ``limit`` usable examples from one of ``ADAPTERS`` (requires the ``datasets`` package)."""
    from datasets import load_dataset

    repo, fn = ADAPTERS[name]
    out = []
    for row in load_dataset(repo, split=split, streaming=True):
        ex = fn(row)
        if ex is not None:
            out.append(ex)
        if len(out) >= limit:
            break
    return out


def synthetic_examples(n: int = 8, seed: int = 0) -> List[Dict]:
    """Coloured squares on white: colour (choice), is-red (noul), size (score)."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    colors = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 170, 60)}
    sizes = ["small", "medium", "large"]
    out = []
    for _ in range(n):
        c, s = rng.choice(list(colors)), rng.randrange(3)
        img = Image.new("RGB", (96, 96), (255, 255, 255))
        half = (8, 20, 40)[s]
        ImageDraw.Draw(img).rectangle([48 - half, 48 - half, 48 + half, 48 + half], fill=colors[c])
        state = {"image": img}
        out.append({"state": state, "q": {"t": "choice", "ins": "What color is the square?", "crit": {k: None for k in colors}}, "target": _one_hot(list(colors).index(c), 3)})
        out.append({"state": state, "q": {"t": "noul", "ins": "Is the square red?", "crit": None}, "target": _one_hot(int(c == "red"), 2)})
        out.append({"state": state, "q": {"t": "score", "ins": "How large is the square?", "crit": sizes}, "target": _one_hot(s, 3)})
    return out


def make_item(processor, ex: Dict, rng: random.Random, shuffle: bool = True, max_len: int = 1024, head_max_len: int = 256) -> Dict:
    """Tokenize one example with a random option order; the target is permuted to marker order."""
    k = len(render_options(ex["q"]))
    order = list(range(k))
    if shuffle:
        rng.shuffle(order)
    it = build_vlm_inputs(processor, ex["state"], ex["q"], max_len, head_max_len, option_order=order)
    it["target"] = [ex["target"][i] for i in order]
    it["label"] = max(range(k), key=lambda j: it["target"][j])
    it["qtype"] = QTYPES[ex["q"]["t"]]
    return it


# ---------------------------------------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------------------------------------


def vlm_loss(logits, target, qtype, mask, sigma: float = 0.3, group_size: int = 4, w_ce: float = 1.0):
    """Proper-scoring-rule policy gradient over noisy logits + soft cross-entropy (as in the text notebooks)."""
    logits = logits.float()
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((group_size,) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    return loss_rl + w_ce * loss_ce, r.mean()


def _to(b: Dict, device, dtype) -> Dict:
    out = {}
    for k, v in b.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if k == "pixel_values":
                v = v.to(dtype)
        out[k] = v
    return out


def train(
    model: VLMDecisionModel,
    processor,
    examples: List[Dict],
    steps: int = 100,
    batch_size: int = 2,
    freeze: str = "head",
    n_last: int = 4,
    lr_head: float = 1e-4,
    lr_backbone: float = 2e-5,
    sigma: float = 0.3,
    device: Optional[str] = None,
    seed: int = 0,
    log_every: int = 1,
) -> List[float]:
    """Minimal single-device loop. Returns per-step losses."""
    device = torch.device(device or next(model.parameters()).device)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    n_train = set_trainable(model, freeze, n_last=n_last)
    enc = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    groups = [{"params": head, "lr": lr_head}] + ([{"params": enc, "lr": lr_backbone}] if enc else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    model.to(device).train()
    dtype = model.encoder.dtype
    print("training %d params (freeze=%s) on %s" % (n_train, freeze, device))
    losses = []
    for step in range(steps):
        batch = [make_item(processor, rng.choice(examples), rng) for _ in range(batch_size)]
        b = _to(collate_vlm(batch, processor.tokenizer.pad_token_id), device, dtype)
        logits, act = model(
            b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
            pixel_values=b["pixel_values"], pixel_attention_mask=b["pixel_attention_mask"], option_span=b["option_span"],
        )
        loss, reward = vlm_loss(logits, b["target"], b["qtype"], b["marker_mask"], sigma=sigma)
        loss = loss + 0.0 * act.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        losses.append(loss.item())
        if log_every and step % log_every == 0:
            print("step %d | loss %.4f | reward %.3f" % (step, losses[-1], reward.item()))
    model.eval()
    return losses


@torch.no_grad()
def fit_temperatures(model: VLMDecisionModel, processor, examples: List[Dict], device=None) -> List[float]:
    """Per-type temperature scaling by LBFGS on held-out examples (same as the text training notebook)."""
    device = torch.device(device or next(model.parameters()).device)
    model.eval()
    per_type = {0: [], 1: [], 2: []}
    rng = random.Random(0)
    for ex in examples:
        it = make_item(processor, ex, rng)
        b = _to(collate_vlm([it], processor.tokenizer.pad_token_id), device, model.encoder.dtype)
        logits, _ = model(
            b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
            pixel_values=b["pixel_values"], pixel_attention_mask=b["pixel_attention_mask"], option_span=b["option_span"],
        )
        k = len(it["markers"])
        per_type[it["qtype"]].append((logits[0, :k].float().cpu(), torch.tensor(it["target"])))
    temps = []
    for t in range(3):
        sel = per_type[t]
        if len(sel) < 10:
            temps.append(1.0)
            continue
        kmax = max(len(z) for z, _ in sel)
        Z = torch.full((len(sel), kmax), -1e4)
        T = torch.zeros((len(sel), kmax))
        for i, (z, y) in enumerate(sel):
            Z[i, : len(z)], T[i, : len(y)] = z, y
        with torch.enable_grad():
            log_t = torch.zeros(1, requires_grad=True)
            lbfgs = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

            def closure():
                lbfgs.zero_grad()
                loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
                loss.backward()
                return loss

            lbfgs.step(closure)
        temps.append(float(torch.clamp(log_t.exp(), 0.1, 10.0)))
    return temps


def main(argv: Optional[Iterable[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument("--init", default=None, help="saved VLM agent dir to continue from")
    ap.add_argument("--synthetic", action="store_true", help="train on coloured-square toy data")
    ap.add_argument("--dataset", action="append", choices=sorted(ADAPTERS), default=[])
    ap.add_argument("--limit", type=int, default=1000, help="examples per dataset")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--freeze", choices=["head", "last_n", "full"], default="head")
    ap.add_argument("--n-last", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    agent = VLMAgent(args.init, backbone=None if args.init else args.backbone, device=args.device)
    examples = synthetic_examples() if args.synthetic else []
    for name in args.dataset:
        examples += load_hf_examples(name, limit=args.limit)
    if not examples:
        ap.error("no training data: pass --synthetic and/or --dataset")
    losses = train(
        agent.model, agent.processor, examples, steps=args.steps, batch_size=args.batch_size,
        freeze=args.freeze, n_last=args.n_last, device=str(agent.device),
    )
    print("final loss %.4f (finite=%s)" % (losses[-1], math.isfinite(losses[-1])))
    if args.out:
        agent.temperature = fit_temperatures(agent.model, agent.processor, examples)
        agent.save(args.out, include_backbone=args.freeze != "head")
        print("saved to", args.out)


if __name__ == "__main__":
    main()
