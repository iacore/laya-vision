"""Stage-1 image alignment (experimental).

Attach a SigLIP tower + fresh projector to a text checkpoint, freeze ModernBERT and the vision tower, and train the
projector (plus, optionally, the decision head) on VQA records in the shared format of laya.vision_data:
multiple choice (A-OKVQA, ScienceQA) -> choice, VQAv2 yes/no -> noul.

Losses (both built on proper_reward):
  direct  maximise proper_reward(softmax(logits), target) (log + spherical [+ RPS]); low variance, the default here
  rlcd    the notebooks' objective: Gaussian-perturbed logits scored by proper_reward, GRPO group-mean baseline

Smoke run on synthetic coloured squares (no dataset download):
    python -m laya.vision_train --dataset synthetic --steps 3 --batch-size 2 --device cpu
On prepared data (see modal_app.py / laya.vision_data):
    python -m laya.vision_train --data-root /data/vqa --datasets aokvqa,scienceqa,vqav2_yesno --minutes 20 --out ckpt
"""
import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from .agent import Agent
from .common import QTYPES, build_sequence, collate_items, ece_score, proper_reward, render_options, temp_bucket
from .vision import DEFAULT_VISION_ENCODER, SIGLIP_MEAN, SIGLIP_STD, Image, freeze_for_alignment, preprocess_image
from .vision_data import read_records


class ItemBuilder:
    """Record (laya.vision_data format; "image" is a path or PIL image) -> training item. Holds no model, so it can be
    shipped to DataLoader workers."""

    def __init__(self, agent: Agent):
        self.tok = agent.tok
        self.max_len = agent.cfg.get("max_len", 512)
        self.head_max_len = agent.cfg.get("head_max_len", 192)
        self.n_img = agent.model.n_image_tokens
        self.image_token_id = agent.image_token_id
        self.size = agent.cfg.get("image_size", agent.model.image_size)
        self.mean = agent.cfg.get("image_mean", SIGLIP_MEAN)
        self.std = agent.cfg.get("image_std", SIGLIP_STD)

    def __call__(self, rec: Dict) -> Optional[Dict]:
        q = Agent._to_internal(rec["question"])
        k = len(render_options(q))
        img = rec["image"]
        state = {"image": Image(img) if isinstance(img, (str, os.PathLike)) else img}
        if rec.get("state_text"):
            state["context"] = rec["state_text"]
        try:
            seq, markers, image_pos = build_sequence(
                self.tok, state, q, self.max_len, self.head_max_len, n_image_tokens=self.n_img, image_token_id=self.image_token_id
            )
        except ValueError:
            return None
        if len(markers) != k or not 0 <= rec["label"] < k:
            return None
        return {
            "ids": seq,
            "markers": markers,
            "image_pos": image_pos,
            "pixel_values": preprocess_image(state["image"], self.size, self.mean, self.std),
            "qtype": QTYPES[q["t"]],
            "target": [1.0 if i == rec["label"] else 0.0 for i in range(k)],
            "label": rec["label"],
            "dataset": rec.get("dataset", "?"),
        }


class RecordDataset(torch.utils.data.Dataset):
    """Builds items on demand. `threads` > 1 fetches a batch's items concurrently (per DataLoader worker), which
    hides per-file read latency on network volumes."""

    def __init__(self, records: List[Dict], builder: ItemBuilder, threads: int = 1):
        self.records, self.builder, self.threads = records, builder, threads
        self._pool = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.builder(self.records[i])

    def __getitems__(self, indices):
        if self.threads <= 1:
            return [self[i] for i in indices]
        if self._pool is None:  # created lazily inside each worker process
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(self.threads)
        return list(self._pool.map(self.__getitem__, indices))

    def __getstate__(self):
        return {**self.__dict__, "_pool": None}


def make_loader(records, builder, batch_size, pad_id, shuffle, workers=0, seed=0, threads=1):
    def collate(items):
        return collate_items([[it for it in items if it is not None]], pad_id)

    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        RecordDataset(records, builder, threads),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate,
        generator=g,
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


# ---- losses -------------------------------------------------------------------------------------------------------


def direct_loss(logits, target, qtype, mask):
    q = torch.softmax(logits.masked_fill(~mask, -1e4), -1)
    r = proper_reward(q, target, qtype, mask)
    return -r.mean(), r.mean().detach()


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


def _forward(model, batch, device, amp_dtype=None):
    with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
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
    return logits.float()


def stage1_step(model, batch: Dict, optimizer, device, loss: str = "rlcd", sigma: float = 0.3, amp_dtype=None, scheduler=None) -> Dict[str, float]:
    logits = _forward(model, batch, device, amp_dtype)
    mask, target, qtype = batch["marker_mask"].to(device), batch["target"].to(device), batch["qtype"].to(device)
    if loss == "direct":
        l, reward = direct_loss(logits, target, qtype, mask)
    else:
        l, reward = rlcd_loss(logits, target, qtype, mask, sigma)
    optimizer.zero_grad(set_to_none=True)
    l.backward()
    gnorm = torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    acc = (logits.detach().argmax(-1).cpu() == batch["label"]).float().mean()
    return {"loss": l.item(), "reward": reward.item(), "acc": acc.item(), "grad_norm": float(gnorm)}


def set_stage1_mode(model):
    """Train mode for the trainable parts, eval mode for the frozen encoder and vision tower."""
    model.train()
    model.encoder.eval()
    model.vision.eval()


# ---- evaluation ---------------------------------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, device, temperature, temperature_by_options, amp_dtype=None, shuffle_control=True) -> Dict[str, Dict]:
    """Per-dataset accuracy, NLL and ECE (raw softmax and with the checkpoint's per-bucket temperatures, i.e. what
    Agent.predict reports). With shuffle_control, also accuracy when each row gets another row's image
    (acc_shuffled): acc - acc_shuffled is what the image contributes beyond question/option priors."""
    was_training = model.training
    model.eval()
    acc = {}
    for b in loader:
        if b is None:
            continue
        logits = _forward(model, b, device, amp_dtype).cpu().numpy()
        shuffled = None
        if shuffle_control and len(b["pixel_values"]) > 1:
            sb = dict(b, image_index=b["image_index"].roll(1))
            shuffled = _forward(model, sb, device, amp_dtype).cpu().numpy()
        for r, meta in enumerate(b["meta"]):
            k, qt, y = int(b["marker_mask"][r].sum()), int(b["qtype"][r]), int(b["label"][r])
            t = temperature_by_options.get(temp_bucket(qt, k), temperature[qt])
            row = acc.setdefault(meta["dataset"], {"conf": [], "conf_t": [], "correct": [], "nll": [], "shuffled": []})
            if shuffled is not None:
                row["shuffled"].append(float(shuffled[r, :k].argmax() == y))
            for key, z in (("conf", logits[r, :k]), ("conf_t", logits[r, :k] / max(1e-3, float(t)))):
                p = np.exp(z - z.max())
                p /= p.sum()
                row[key].append(float(p.max()))
                if key == "conf":
                    row["correct"].append(float(p.argmax() == y))
                    row["nll"].append(float(-np.log(max(p[y], 1e-12))))
    out = {}
    for name, row in sorted(acc.items()):
        conf, conf_t, correct = np.array(row["conf"]), np.array(row["conf_t"]), np.array(row["correct"])
        out[name] = {
            "n": len(correct),
            "acc": round(float(correct.mean()), 4),
            "nll": round(float(np.mean(row["nll"])), 4),
            "ece": round(ece_score(conf, correct), 4),
            "ece_temp": round(ece_score(conf_t, correct), 4),
            "mean_conf": round(float(conf.mean()), 4),
        }
        if row["shuffled"]:
            out[name]["acc_shuffled"] = round(float(np.mean(row["shuffled"])), 4)
    if was_training:
        set_stage1_mode(model)
    return out


# ---- data ---------------------------------------------------------------------------------------------------------


def synthetic_records(n: int, size: int = 64, seed: int = 0) -> List[Dict]:
    """Solid-colour squares with a colour choice question and a yes/no question, in the vision_data record format."""
    from PIL import Image as PILImage

    colors = {"red": (220, 30, 30), "green": (30, 180, 30), "blue": (30, 30, 220)}
    names = list(colors)
    rng = random.Random(seed)
    out = []
    for i in range(n):
        c = rng.choice(names)
        rec = {"id": "syn-%d" % i, "image": PILImage.new("RGB", (size, size), colors[c]), "state_text": None, "dataset": "synthetic"}
        if i % 2 == 0:
            rec["question"] = {"type": "choice", "instructions": "What colour is the square?", "criteria": names}
            rec["label"] = names.index(c)
        else:
            probe = rng.choice(names)
            rec["question"] = {"type": "noul", "instructions": "Is the square %s?" % probe, "criteria": None}
            rec["label"] = int(probe == c)
        out.append(rec)
    return out


def save_checkpoint(agent: Agent, out_dir: str, dtype: Optional[torch.dtype] = torch.float16):
    """Write an Agent-loadable directory: model.safetensors (incl. vision./proj.; fp16 by default, as the released
    text checkpoint), rl_agent_config.json with the vision keys, tokenizer/ and encoder/ config."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    sd = {}
    for k, v in agent.model.state_dict().items():
        v = v.detach().contiguous().cpu()
        sd[k] = v.to(dtype) if dtype is not None and v.is_floating_point() and k != "temperature" else v
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(agent.cfg, f, indent=2)
    agent.tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    agent.model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))


# ---- training loop ------------------------------------------------------------------------------------------------


def train(
    agent: Agent,
    train_records: List[Dict],
    val_records: List[Dict],
    steps: int = 1000,
    minutes: Optional[float] = None,
    batch_size: int = 32,
    lr_proj: float = 1e-3,
    lr_head: float = 1e-5,
    train_head: bool = True,
    loss: str = "direct",
    sigma: float = 0.3,
    eval_every: int = 500,
    workers: int = 0,
    threads: int = 1,
    amp_dtype: Optional[torch.dtype] = None,
    out: Optional[str] = None,
    log=print,
) -> List[Dict]:
    """Stage-1 loop: frozen encoder + vision, AdamW on proj (+ head), linear warmup then cosine decay. Evaluates on
    val_records at step 0, every eval_every steps and at the end; stops at `steps` or after `minutes`."""
    model, device = agent.model, agent.device
    freeze_for_alignment(model, train_head=train_head)
    groups = [{"params": list(model.proj.parameters()), "lr": lr_proj}]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("proj.")]
    if head:
        groups.append({"params": head, "lr": lr_head})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    warm = max(1, min(200, steps // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps))))
    )
    builder = ItemBuilder(agent)
    pad = agent.tok.pad_token_id
    train_loader = make_loader(train_records, builder, batch_size, pad, shuffle=True, workers=workers, threads=threads)
    val_loader = make_loader(val_records, builder, 2 * batch_size, pad, shuffle=False, workers=workers, threads=threads)
    temps = (agent.temperature, agent.temperature_by_options)
    history: List[Dict] = []

    def record(entry):
        history.append(entry)
        log(json.dumps(entry))
        if out:
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(entry) + "\n")

    t0 = time.time()
    record({"step": 0, "eval": evaluate(model, val_loader, device, *temps, amp_dtype=amp_dtype), "elapsed_s": 0})
    set_stage1_mode(model)
    step, window = 0, []
    done = False
    while not done:
        t_data = time.time()
        for batch in train_loader:
            if batch is None:
                continue
            step += 1
            t_step = time.time()
            stats = stage1_step(model, batch, opt, device, loss, sigma, amp_dtype, sched)
            if device.type == "cuda":
                torch.cuda.synchronize()
            stats.update(data_s=t_step - t_data, step_s=time.time() - t_step)
            window.append(stats)
            if step % 50 == 0:
                avg = {k: round(float(np.mean([w[k] for w in window])), 4) for k in window[0]}
                record({"step": step, "train": avg, "lr_proj": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0)})
                window = []
            done = step >= steps or (minutes is not None and time.time() - t0 > 60 * minutes)
            if step % eval_every == 0 or done:
                record({"step": step, "eval": evaluate(model, val_loader, device, *temps, amp_dtype=amp_dtype), "elapsed_s": round(time.time() - t0)})
            if done:
                break
            t_data = time.time()
    model.eval()
    if out:
        save_checkpoint(agent, out)
        log("saved %s" % out)
    return history


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--vision-encoder", default=DEFAULT_VISION_ENCODER)
    ap.add_argument("--n-image-tokens", type=int, default=64)
    ap.add_argument("--dataset", default=None, choices=["synthetic"], help="synthetic smoke data instead of --data-root")
    ap.add_argument("--data-root", default="/data/vqa")
    ap.add_argument("--datasets", default="aokvqa,scienceqa,vqav2_yesno")
    ap.add_argument("--val-limit", type=int, default=2000, help="val records per dataset")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr-proj", type=float, default=1e-3)
    ap.add_argument("--lr-head", type=float, default=1e-5)
    ap.add_argument("--freeze-head", action="store_true", help="train the projector only")
    ap.add_argument("--loss", default="direct", choices=["direct", "rlcd"])
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    agent = Agent(args.model, device=args.device, vision_encoder=args.vision_encoder, n_image_tokens=args.n_image_tokens)
    if args.dataset == "synthetic":
        train_recs = synthetic_records(max(8, args.steps * args.batch_size))
        val_recs = synthetic_records(8, seed=1)
    else:
        names = args.datasets.split(",")
        train_recs = [r for n in names for r in read_records(args.data_root, n, "train")]
        val_recs = [r for n in names for r in read_records(args.data_root, n, "val", args.val_limit)]
    amp = torch.bfloat16 if agent.device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    train(
        agent, train_recs, val_recs, steps=args.steps, minutes=args.minutes, batch_size=args.batch_size,
        lr_proj=args.lr_proj, lr_head=args.lr_head, train_head=not args.freeze_head, loss=args.loss,
        eval_every=args.eval_every, workers=args.workers, amp_dtype=amp, out=args.out,
    )


if __name__ == "__main__":
    main()
