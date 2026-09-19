"""Stage-1 image alignment sketch (experimental).

Attach a SigLIP tower + fresh projector to a text checkpoint, freeze ModernBERT and the vision tower, and train the
projector (plus the decision head) with the same RLCD objective as the fine-tuning notebooks: Gaussian-perturbed
logits scored by `proper_reward`, GRPO group-mean baseline.

VQA datasets map onto the existing question types:
  - multiple choice (A-OKVQA, ScienceQA)     -> choice, criteria = answer options
  - yes/no (VQAv2 answer_type == "yes/no")   -> noul

Smoke run on synthetic coloured squares (no dataset download):
    python -m laya.vision_train --dataset synthetic --steps 3 --batch-size 2 --device cpu
Real run (needs `pip install datasets`):
    python -m laya.vision_train --dataset aokvqa --steps 2000 --batch-size 16 --device cuda --out laya-vision
"""
import argparse
import json
import os
import random
from typing import Dict, Iterator, List, Optional

import torch

from .agent import Agent
from .common import QTYPES, build_sequence, collate_items, proper_reward, render_options
from .vision import DEFAULT_VISION_ENCODER, Image, freeze_for_alignment


def vqa_to_question(ex: Dict) -> Dict:
    """{"question", "choices", "answer": int} -> choice; {"question", "answer": bool} -> noul."""
    if "choices" in ex:
        return {"t": "choice", "ins": ex["question"], "crit": {c: None for c in ex["choices"]}}
    return {"t": "noul", "ins": ex["question"], "crit": None}


def make_item(agent: Agent, ex: Dict, pixels: Optional[torch.Tensor] = None) -> Optional[Dict]:
    """Tokenize one VQA example into a training item with a one-hot target. ex["image"] is a PIL image or path;
    pass pixels to reuse an already preprocessed tensor."""
    q = vqa_to_question(ex)
    k = len(render_options(q))
    label = int(ex["answer"])
    if pixels is None:
        pixels = agent.preprocess_image(ex["image"])
    img = ex["image"]
    state = {"image": Image(img) if isinstance(img, (str, os.PathLike)) else img}
    if ex.get("context"):
        state["context"] = ex["context"]
    seq, markers, image_pos = build_sequence(
        agent.tok,
        state,
        q,
        agent.cfg.get("max_len", 512),
        agent.cfg.get("head_max_len", 192),
        n_image_tokens=agent.model.n_image_tokens,
        image_token_id=agent.image_token_id,
    )
    if len(markers) != k:
        return None
    return {
        "ids": seq,
        "markers": markers,
        "image_pos": image_pos,
        "pixel_values": pixels,
        "qtype": QTYPES[q["t"]],
        "target": [1.0 if i == label else 0.0 for i in range(k)],
        "label": label,
    }


def rlcd_loss(logits, target, qtype, mask, sigma: float = 0.3, group_size: int = 4):
    """Notebook RLCD objective: sample G perturbed logit vectors, reward them with proper_reward, and push the
    Gaussian policy mean (the model logits) towards above-baseline samples. Returns (loss, mean reward)."""
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((group_size,) + logits.shape, device=logits.device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
    return -(adv * logp).mean(), r.mean()


def stage1_step(model, batch: Dict, optimizer, device, sigma: float = 0.3, group_size: int = 4) -> Dict[str, float]:
    logits, _ = model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["marker_pos"].to(device),
        batch["marker_mask"].to(device),
        batch["qtype"].to(device),
        pixel_values=batch["pixel_values"].to(device),
        image_pos=batch["image_pos"].to(device),
        image_index=batch["image_index"].to(device),
    )
    mask = batch["marker_mask"].to(device)
    loss, reward = rlcd_loss(logits.float(), batch["target"].to(device), batch["qtype"].to(device), mask, sigma, group_size)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
    optimizer.step()
    acc = (logits.detach().argmax(-1).cpu() == batch["label"]).float().mean()
    return {"loss": loss.item(), "reward": reward.item(), "acc": acc.item()}


def set_stage1_mode(model):
    """Train mode for the trainable parts, eval mode for the frozen encoder and vision tower."""
    model.train()
    model.encoder.eval()
    model.vision.eval()


# ---- data -----------------------------------------------------------------------------------------------------


def synthetic_examples(n: int, size: int = 64, seed: int = 0) -> List[Dict]:
    """Solid-colour squares with a colour choice question and a yes/no question."""
    from PIL import Image as PILImage

    colors = {"red": (220, 30, 30), "green": (30, 180, 30), "blue": (30, 30, 220)}
    names = list(colors)
    rng = random.Random(seed)
    out = []
    for i in range(n):
        c = rng.choice(names)
        img = PILImage.new("RGB", (size, size), colors[c])
        if i % 2 == 0:
            out.append({"image": img, "question": "What colour is the square?", "choices": names, "answer": names.index(c)})
        else:
            probe = rng.choice(names)
            out.append({"image": img, "question": "Is the square %s?" % probe, "answer": probe == c})
    return out


def aokvqa_examples(split: str = "train") -> Iterator[Dict]:
    from datasets import load_dataset

    for ex in load_dataset("HuggingFaceM4/A-OKVQA", split=split, streaming=True):
        yield {"image": ex["image"], "question": ex["question"], "choices": ex["choices"], "answer": ex["correct_choice_idx"]}


def scienceqa_examples(split: str = "train") -> Iterator[Dict]:
    from datasets import load_dataset

    for ex in load_dataset("derek-thomas/ScienceQA", split=split, streaming=True):
        if ex["image"] is None:
            continue
        yield {"image": ex["image"], "question": ex["question"], "choices": ex["choices"], "answer": ex["answer"], "context": ex.get("hint") or None}


def vqav2_yesno_examples(split: str = "validation") -> Iterator[Dict]:
    from datasets import load_dataset

    for ex in load_dataset("lmms-lab/VQAv2", split=split, streaming=True):
        if ex["answer_type"] == "yes/no" and ex["multiple_choice_answer"] in ("yes", "no"):
            yield {"image": ex["image"], "question": ex["question"], "answer": ex["multiple_choice_answer"] == "yes"}


DATASETS = {"aokvqa": aokvqa_examples, "scienceqa": scienceqa_examples, "vqav2_yesno": vqav2_yesno_examples}


def save_checkpoint(agent: Agent, out_dir: str):
    """Write an Agent-loadable directory: model.safetensors (incl. vision./proj.), rl_agent_config.json with the
    vision keys, tokenizer/ and encoder/ config."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.detach().contiguous().cpu() for k, v in agent.model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(agent.cfg, f, indent=2)
    agent.tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    agent.model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--vision-encoder", default=DEFAULT_VISION_ENCODER)
    ap.add_argument("--n-image-tokens", type=int, default=64)
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic"] + list(DATASETS))
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr-proj", type=float, default=1e-3)
    ap.add_argument("--lr-head", type=float, default=2e-5)
    ap.add_argument("--freeze-head", action="store_true", help="train the projector only")
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    agent = Agent(args.model, device=args.device, vision_encoder=args.vision_encoder, n_image_tokens=args.n_image_tokens)
    model, device = agent.model, agent.device
    freeze_for_alignment(model, train_head=not args.freeze_head)
    groups = [{"params": [p for p in model.proj.parameters()], "lr": args.lr_proj}]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("proj.")]
    if head:
        groups.append({"params": head, "lr": args.lr_head})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    set_stage1_mode(model)

    if args.dataset == "synthetic":
        data = iter(synthetic_examples(args.steps * args.batch_size))
    else:
        data = DATASETS[args.dataset]()
    for step in range(args.steps):
        items = []
        while len(items) < args.batch_size:
            it = make_item(agent, next(data))
            if it is not None:
                items.append(it)
        stats = stage1_step(model, collate_items([items], agent.tok.pad_token_id), opt, device, args.sigma)
        print("step %d | loss %.4f | reward %.3f | acc %.2f" % (step + 1, stats["loss"], stats["reward"], stats["acc"]))

    if args.out:
        save_checkpoint(agent, args.out)
        print("saved", args.out)


if __name__ == "__main__":
    main()
