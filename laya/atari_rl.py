"""PPO fine-tuning of the Atari decision model (``laya.vlm``) on one game's score.

The imitation-trained checkpoint already answers ``laya.games.atari_question`` with a distribution over the
game's actions. That distribution, divided by the checkpoint's calibrated ``choice`` temperature, is exactly
the policy the play loop uses, so it is the policy this trains: ``pi(a|s) = softmax(logits(s) / T)`` with ``T``
frozen at the checkpoint's value. Gradients flow through ``1/T`` into the raw logits; nothing else changes
about how the model is called, so a checkpoint trained here plays through ``laya.atari_train.play`` unchanged.

What makes it fast enough for a VLM policy:

* **One prompt, many frames.** For a fixed game the token sequence (question, options, option order) is the
  same on every frame; only the pixels change. ``PromptTemplate`` builds it once and the rollout re-uses it,
  so no tokenisation happens per step.
* **Image features computed once per frame.** The vision tower and the connector are frozen, so a frame's
  image tokens never change. The rollout keeps them (bf16, on the host) and the PPO update replays them
  through the language model only. The vision tower is most of the FLOPs, so this roughly halves the cost of
  a training sample.
* **The reference policy shares them too.** The KL anchor is a second copy of the starting checkpoint; since
  it has the same frozen vision tower, its rollout pass is language-model-only.

Trainable by default: the decision head (option scorer, head transformer, type embedding), the last
``n_last`` language-model layers and the final norm, plus a fresh value head. The vision tower, the
connector and the lower language-model layers stay frozen (``laya.vlm.set_trainable(mode="last_n")``).

Local smoke test (CPU, a few PPO steps on real ALE frames, no checkpoint needed):
    python -m laya.atari_rl --smoke
"""
import argparse
import json
import math
import os
import random
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .common import QTYPES, render_options, temp_bucket
from .games import atari_question
from .vlm import VLMAgent, build_vlm_inputs, set_trainable, vlm_prefix

# ---------------------------------------------------------------------------------------------------------
# Prompt template and frame preprocessing
# ---------------------------------------------------------------------------------------------------------


class PromptTemplate:
    """The fixed part of one game's question: token ids, option readout positions and the policy temperature.

    Built from a real frame, because the number of image tokens depends on the frame's shape. Every rollout
    step re-uses ``ids`` / ``markers`` verbatim, which is what makes the sequence free to build.
    """

    def __init__(self, agent: VLMAgent, game: str, actions: Sequence[str], frame: np.ndarray):
        from PIL import Image  # noqa: F401  (frames arrive as arrays; Image is used below)

        if agent.model.option_attention != "causal":
            raise ValueError("atari_rl assumes option_attention='causal' (got %r)" % agent.model.option_attention)
        self.game, self.actions = game, list(actions)
        self.question = atari_question(game, actions)["action"]
        q = VLMAgent._to_internal(self.question)
        self.k = len(render_options(q))
        if self.k != len(actions):
            raise ValueError("option count %d != %d actions" % (self.k, len(actions)))
        it = build_vlm_inputs(agent.processor, {"image": Image.fromarray(frame)}, q,
                              agent.cfg.get("max_len", 1024), agent.cfg.get("head_max_len", 256))
        if it["n_images"] != 1 or len(it["markers"]) != self.k:
            raise ValueError("unexpected template: %d images, %d markers" % (it["n_images"], len(it["markers"])))
        self.ids = torch.tensor(it["ids"], dtype=torch.long)
        self.markers = torch.tensor(it["markers"], dtype=torch.long)
        self.length = len(it["ids"])
        self.frame_shape = tuple(frame.shape)
        self.temperature = max(1e-3, float(agent.temperature_by_options.get(
            temp_bucket(QTYPES["choice"], self.k), agent.temperature[QTYPES["choice"]])))


class FramePreprocessor:
    """RGB frames -> the pixel tensors ``laya.vlm.vlm_prefix`` would produce, a batch at a time.

    A 210x160 ALE frame becomes a 512x512 tensor, and that costs about 14 ms of CPU per frame -- more than the
    whole model forward. It does not reduce much: the processor resizes with LANCZOS, which torchvision only
    implements on CPU, and a hand-written chain of the same steps does not reproduce its pixels. So the cost
    is paid on a thread pool and hidden behind the GPU instead (see ``collect``: the lanes overlap this with
    the forward passes).

    Whichever path is used, its output is checked against ``vlm_prefix`` at construction, because the pixels
    the policy sees during the rollout have to be the pixels imitation training and
    ``laya.atari_train.play`` use.
    """

    TOL = 1e-5

    def __init__(self, processor, frame: np.ndarray, threads: int = 8, device=None, backbone: Optional[str] = None):
        from PIL import Image

        self.processor, self.pool = processor, ThreadPoolExecutor(max(1, threads))
        self.device = torch.device(device) if device is not None else None
        ref = vlm_prefix(processor, [Image.fromarray(frame)])
        self.ref_pixels, self.ref_mask = ref["pixel_values"], ref["pixel_attention_mask"]
        self.fast = None
        if backbone:
            try:
                from transformers import AutoImageProcessor

                self.fast = AutoImageProcessor.from_pretrained(backbone, backend="torchvision")
            except Exception as e:  # noqa: BLE001
                print("no torchvision image processor (%s)" % e)
        candidates = [("numpy_threads", self._threaded)]
        if self.fast is not None:
            candidates.append(("torchvision_cpu", lambda f: self._fast(f, torch.device("cpu"))))
        candidates.append(("vlm_prefix_threads", self._threaded_prefix))
        self.path, self.max_abs_diff = None, None
        for name, fn in candidates:
            try:
                pv, pam = fn([frame, frame])
                diff = float((pv[:1].float().cpu() - self.ref_pixels.float()).abs().max())
                ok = pv.shape[1:] == self.ref_pixels.shape[1:] and diff <= self.TOL \
                    and torch.equal(pam[:1].cpu(), self.ref_mask)
            except Exception as e:  # noqa: BLE001  (a processor signature change just moves to the next path)
                print("preprocessing path %s unusable: %r" % (name, e))
                continue
            print("preprocessing path %s: max |dpixel| vs vlm_prefix %.2e -> %s" % (name, diff, "ok" if ok else "reject"))
            if ok:
                self.path, self._fn, self.max_abs_diff = name, fn, diff
                break
        if self.path is None:
            raise ValueError("no preprocessing path reproduces vlm_prefix")
        self.pixel_shape = tuple(self.ref_pixels.shape[1:])

    def _fast(self, frames, device) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.fast(images=[[f] for f in frames], do_image_splitting=False, return_tensors="pt", device=device)
        return out["pixel_values"][:, 0], out["pixel_attention_mask"][:, 0].bool()

    def _numpy_one(self, frame: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image

        out = self.processor.image_processor(images=[[Image.fromarray(frame)]], do_image_splitting=False,
                                            return_tensors="pt")
        return out["pixel_values"][0], out["pixel_attention_mask"][0].bool()

    def _prefix_one(self, frame: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        from PIL import Image

        p = vlm_prefix(self.processor, [Image.fromarray(frame)])
        return p["pixel_values"], p["pixel_attention_mask"]

    def _gather(self, one, frames):
        outs = list(self.pool.map(one, frames))
        return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])

    def _threaded(self, frames):
        return self._gather(self._numpy_one, frames)

    def _threaded_prefix(self, frames):
        return self._gather(self._prefix_one, frames)

    def __call__(self, frames: Sequence[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._fn(frames)


# ---------------------------------------------------------------------------------------------------------
# Model pieces
# ---------------------------------------------------------------------------------------------------------


class ValueHead(nn.Module):
    """State value from the pooled hidden state (the last real token, as ``VLMDecisionModel``'s act head uses).

    The last layer starts at zero so the run begins with V(s) = 0 everywhere instead of random values.
    """

    def __init__(self, d: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled.float()).squeeze(-1)


@torch.no_grad()
def encode_frames(model, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor,
                  amp: bool = True) -> torch.Tensor:
    """Frozen vision tower + connector: [B, 3, H, W] -> image token features [B, S, d]."""
    dev = next(model.parameters()).device
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
        return model.encode_images(pixel_values.to(dev, model.encoder.dtype), pixel_attention_mask.to(dev))


def head_forward(model, tmpl: PromptTemplate, feats: torch.Tensor, value_head: Optional[ValueHead] = None,
                 detach_value: bool = True, amp: bool = True):
    """``VLMDecisionModel.forward`` for one fixed prompt and cached image features.

    Equivalent to ``model(input_ids, attention_mask, marker_pos, marker_mask, qtype, image_hidden_states=feats,
    ...)`` when every row has the same unpadded sequence and the same option order, which is the case here;
    ``modal_atari_rl.py::bench`` asserts the option logits match ``laya.atari_train.action_probs``. Returns
    the raw option logits (before the policy temperature) and, with a value head, V(s).
    """
    dev = feats.device
    feats = feats.to(model.encoder.dtype)  # inputs_merger scatters these into fp32 token embeddings
    b = feats.shape[0]
    ids = tmpl.ids.to(dev)[None].expand(b, -1)
    att = torch.ones((b, tmpl.length), dtype=torch.long, device=dev)
    qtype = torch.zeros(b, dtype=torch.long, device=dev)
    pad = torch.zeros((b, tmpl.length), dtype=torch.bool, device=dev)  # nothing is padded; matches the general path
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
        h = model.encoder(input_ids=ids, attention_mask=att, image_hidden_states=feats, use_cache=False).last_hidden_state
        h = h.float() + model.type_emb(qtype)[:, None, :]
        if model.head is not None:
            for layer in model.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        logits = model.scorer(h[:, tmpl.markers.to(dev)]).squeeze(-1).float()
    value = None
    if value_head is not None:
        pooled = h[:, -1]
        value = value_head(pooled.detach() if detach_value else pooled)
    return logits, value


# ---------------------------------------------------------------------------------------------------------
# Vectorised ALE
# ---------------------------------------------------------------------------------------------------------


class VecAtari:
    """``n`` independent ``ALE/<game>-v5`` envs stepped in lockstep, matching ``laya.atari_train.play``.

    FIRE is pressed on reset and after each lost life and is not an agent decision; its reward is folded into
    the decision that preceded it. A finished episode is replaced immediately (new seed) so the rollout never
    stalls. ``done`` is what the learner sees: with ``episodic_life`` a lost life ends the credit horizon
    without ending the game, which is the standard Atari trick and matters a lot for Breakout. Full-game raw
    scores are kept for logging and checkpoint selection.
    """

    def __init__(self, game: str, n: int, seed: int = 0, max_steps: int = 4500, episodic_life: bool = True,
                 auto_fire: bool = True, history: int = 200):
        import ale_py
        import gymnasium as gym

        gym.register_envs(ale_py)
        self.envs = [gym.make("ALE/%s-v5" % game) for _ in range(n)]
        self.action_names = self.envs[0].unwrapped.get_action_meanings()
        self.fire = self.action_names.index("FIRE") if auto_fire and "FIRE" in self.action_names else None
        self.n, self.max_steps, self.episodic_life = n, max_steps, episodic_life
        self.next_seed = seed
        self.obs: List[np.ndarray] = [None] * n
        self.lives, self.score, self.steps = [0] * n, [0.0] * n, [0] * n
        self.episode_scores: deque = deque(maxlen=history)
        self.episode_steps: deque = deque(maxlen=history)
        self.episodes = 0
        self.counts: Counter = Counter()
        for i in range(n):
            self._reset(i)

    def _reset(self, i: int):
        o, info = self.envs[i].reset(seed=self.next_seed)
        self.next_seed += 1
        r0 = 0.0
        if self.fire is not None:
            o, r, _, _, info = self.envs[i].step(self.fire)
            r0 += r
        self.obs[i], self.lives[i], self.score[i], self.steps[i] = o, info.get("lives", 0), r0, 0

    def step(self, actions: Sequence[int]) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        rew = np.zeros(self.n, dtype=np.float32)
        done = np.zeros(self.n, dtype=bool)
        for i, a in enumerate(actions):
            a = int(a)
            self.counts[self.action_names[a]] += 1
            o, r, term, trunc, info = self.envs[i].step(a)
            self.steps[i] += 1
            lost = False
            if not (term or trunc) and info.get("lives", self.lives[i]) < self.lives[i]:
                lost = True
                if self.fire is not None:
                    o, r2, term, trunc, info = self.envs[i].step(self.fire)
                    r += r2
            self.lives[i] = info.get("lives", self.lives[i])
            self.score[i] += r
            rew[i] = r
            if term or trunc or self.steps[i] >= self.max_steps:
                self.episode_scores.append(float(self.score[i]))
                self.episode_steps.append(self.steps[i])
                self.episodes += 1
                done[i] = True
                self._reset(i)
            else:
                self.obs[i] = o
                done[i] = lost and self.episodic_life
        return self.obs, rew, done

    def close(self):
        for e in self.envs:
            e.close()


class Lanes:
    """Several ``VecAtari`` groups so preprocessing one group's frames overlaps the GPU work of another.

    Preprocessing a batch of frames costs more CPU time than the forward passes cost GPU time, and neither can
    start before the other finishes for the *same* group. Splitting the envs into independent lanes fixes
    that: while lane g is on the GPU, lane g+1's frames are already being prepared on the thread pool. Lanes
    are just more envs, so the rollout is one batch over all of them.
    """

    def __init__(self, game: str, n_envs: int, n_lanes: int = 2, seed: int = 0, **kw):
        if n_envs % n_lanes:
            raise ValueError("n_envs %d must divide into %d lanes" % (n_envs, n_lanes))
        per = n_envs // n_lanes
        self.lanes = [VecAtari(game, per, seed=seed + g * 1000000, **kw) for g in range(n_lanes)]
        self.n, self.n_lanes = n_envs, n_lanes
        self.action_names = self.lanes[0].action_names

    @property
    def episodes(self) -> int:
        return sum(l.episodes for l in self.lanes)

    @property
    def counts(self) -> Counter:
        total: Counter = Counter()
        for l in self.lanes:
            total.update(l.counts)
        return total

    def recent(self, k: int = 30) -> Optional[float]:
        """Mean of the most recent completed episodes, taking the last ``k // n_lanes`` from each lane."""
        per = max(1, k // self.n_lanes)
        s = [x for l in self.lanes for x in list(l.episode_scores)[-per:]]
        return float(np.mean(s)) if s else None

    def recent_steps(self, k: int = 30) -> Optional[float]:
        per = max(1, k // self.n_lanes)
        s = [x for l in self.lanes for x in list(l.episode_steps)[-per:]]
        return float(np.mean(s)) if s else None

    def close(self):
        for l in self.lanes:
            l.close()


# ---------------------------------------------------------------------------------------------------------
# Rollout and PPO
# ---------------------------------------------------------------------------------------------------------


class Rollout:
    """One on-policy batch: cached image features on the host plus the per-step tensors PPO needs."""

    def __init__(self):
        self.feats: List[torch.Tensor] = []
        self.action: List[torch.Tensor] = []
        self.logp: List[torch.Tensor] = []
        self.value: List[torch.Tensor] = []
        self.ref_logp: List[torch.Tensor] = []
        self.reward: List[np.ndarray] = []
        self.done: List[np.ndarray] = []
        self.last_value: Optional[torch.Tensor] = None


@torch.no_grad()
def collect(runner, lanes: Lanes, steps: int, clip_reward: bool = True) -> Tuple[Rollout, Dict]:
    """Run ``steps`` decisions per env with the current policy, storing what the update needs.

    Lanes are processed in a fixed rotation, and each lane's next batch of frames is queued for preprocessing
    as soon as its envs have stepped, so that work happens while the following lane is on the GPU. Everything
    stored is concatenated in lane order, so the rollout looks like one batch of ``lanes.n`` envs.

    The reference policy's log-probs are computed here (language model only, re-using the policy's image
    features) so the update needs no extra forward.
    """
    r = Rollout()
    t0, t_gpu = time.time(), 0.0
    n_lanes = lanes.n_lanes
    pending = deque((g, runner.pipeline.submit(runner.prep, list(lanes.lanes[g].obs))) for g in range(n_lanes))

    def act(pv, pam, keep=True):
        nonlocal t_gpu
        tg = time.time()
        feats = encode_frames(runner.model, pv, pam, runner.amp)
        logits, value = head_forward(runner.model, runner.tmpl, feats, runner.value_head, amp=runner.amp)
        logp_all = torch.log_softmax(logits / runner.tmpl.temperature, -1)
        if not keep:
            t_gpu += time.time() - tg
            return None, value.cpu()
        ref_logits, _ = head_forward(runner.ref, runner.tmpl, feats, None, amp=runner.amp)
        ref_logp = torch.log_softmax(ref_logits / runner.tmpl.temperature, -1)
        action = torch.multinomial(logp_all.exp(), 1).squeeze(-1)
        t_gpu += time.time() - tg
        return {"feats": feats.to("cpu", torch.bfloat16), "action": action.cpu(),
                "logp": logp_all.gather(-1, action[:, None]).squeeze(-1).cpu(), "value": value.cpu(),
                "ref_logp": ref_logp.cpu(), "acts": action.cpu().numpy()}, None

    for _ in range(steps):
        per_lane = []
        for _ in range(n_lanes):
            g, fut = pending.popleft()
            pv, pam = fut.result()
            out, _ = act(pv, pam)
            _, rew, done = lanes.lanes[g].step(out["acts"])
            pending.append((g, runner.pipeline.submit(runner.prep, list(lanes.lanes[g].obs))))
            out["reward"] = np.clip(rew, -1.0, 1.0) if clip_reward else rew
            out["done"] = done
            per_lane.append(out)
        for key, store in (("feats", r.feats), ("action", r.action), ("logp", r.logp), ("value", r.value),
                           ("ref_logp", r.ref_logp)):
            store.append(torch.cat([o[key] for o in per_lane]))
        r.reward.append(np.concatenate([o["reward"] for o in per_lane]))
        r.done.append(np.concatenate([o["done"] for o in per_lane]))
    tail = {}
    for _ in range(n_lanes):
        g, fut = pending.popleft()
        pv, pam = fut.result()
        _, tail[g] = act(pv, pam, keep=False)
    r.last_value = torch.cat([tail[g] for g in range(n_lanes)])
    dt = time.time() - t0
    return r, {"seconds": dt, "gpu_seconds": t_gpu, "steps_per_s": steps * lanes.n / max(1e-9, dt)}


def gae(r: Rollout, gamma: float = 0.99, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimate over the rollout; ``done`` cuts the horizon (no bootstrap)."""
    t, n = len(r.reward), r.reward[0].shape[0]
    values = torch.stack(r.value)
    rew = torch.tensor(np.stack(r.reward), dtype=torch.float32)
    nonterm = 1.0 - torch.tensor(np.stack(r.done), dtype=torch.float32)
    adv = torch.zeros((t, n), dtype=torch.float32)
    running = torch.zeros(n, dtype=torch.float32)
    for i in reversed(range(t)):
        next_value = r.last_value if i == t - 1 else values[i + 1]
        delta = rew[i] + gamma * next_value * nonterm[i] - values[i]
        running = delta + gamma * lam * nonterm[i] * running
        adv[i] = running
    return adv, adv + values


def ppo_update(runner, r: Rollout, adv: torch.Tensor, ret: torch.Tensor, epochs: int = 2, minibatch: int = 64,
               clip: float = 0.1, vf_coef: float = 0.5, ent_coef: float = 0.01, kl_coef: float = 0.05,
               max_grad_norm: float = 0.5, target_kl_old: float = 0.03, pg_coef: float = 1.0) -> Dict:
    """Clipped PPO on the cached image features, with a KL penalty against the starting policy.

    The penalty is the exact forward KL ``KL(pi || pi_ref)`` over the game's actions (cheap at 4 actions, and
    much lower variance than a sampled estimate). ``target_kl_old`` early-stops the epochs when the policy has
    moved too far from the rollout policy. ``pg_coef = 0`` trains only the value head (used for the first few
    iterations, so the first policy step is not taken against a value function that is still all zeros).

    The model stays in ``eval`` mode: the decision head has dropout, and with it active the recomputed
    log-probs would not be comparable to the rollout's, which is exactly what the PPO ratio assumes.
    """
    dev = runner.device
    feats = torch.cat(r.feats)
    action = torch.cat(r.action)
    logp_old = torch.cat(r.logp)
    ref_logp = torch.cat(r.ref_logp)
    adv_f, ret_f = adv.reshape(-1), ret.reshape(-1)
    value_old = torch.stack(r.value).reshape(-1)
    n = feats.shape[0]
    stats = {k: 0.0 for k in ("pg_loss", "v_loss", "entropy", "kl_ref", "kl_old", "clipfrac")}
    nb, stop = 0, False
    for _ in range(epochs):
        order = torch.randperm(n)
        for s in range(0, n - minibatch + 1, minibatch):
            idx = order[s: s + minibatch]
            f = feats[idx].to(dev, non_blocking=True).to(runner.model.encoder.dtype)
            logits, value = head_forward(runner.model, runner.tmpl, f, runner.value_head,
                                         detach_value=runner.detach_value, amp=runner.amp)
            logp_all = torch.log_softmax(logits / runner.tmpl.temperature, -1)
            a = action[idx].to(dev)
            logp = logp_all.gather(-1, a[:, None]).squeeze(-1)
            lo = logp_old[idx].to(dev)
            ratio = (logp - lo).exp()
            a_mb = adv_f[idx].to(dev)
            a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
            pg = torch.max(-a_mb * ratio, -a_mb * ratio.clamp(1 - clip, 1 + clip)).mean()
            v_loss = 0.5 * ((value - ret_f[idx].to(dev)) ** 2).mean()
            p = logp_all.exp()
            entropy = -(p * logp_all).sum(-1).mean()
            kl_ref = (p * (logp_all - ref_logp[idx].to(dev))).sum(-1).mean()
            # during the value warmup the policy terms are dropped entirely, so the backward pass stays inside
            # the value head (its input is detached) instead of traversing the whole trunk for zero gradients
            loss = vf_coef * v_loss
            if pg_coef:
                loss = loss + pg_coef * (pg - ent_coef * entropy + kl_coef * kl_ref)
            runner.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(runner.trainable_params, max_grad_norm)
            runner.opt.step()
            with torch.no_grad():
                kl_old = (lo - logp).mean()
            stats["pg_loss"] += float(pg)
            stats["v_loss"] += float(v_loss)
            stats["entropy"] += float(entropy)
            stats["kl_ref"] += float(kl_ref)
            stats["kl_old"] += float(kl_old)
            stats["clipfrac"] += float(((ratio - 1).abs() > clip).float().mean())
            nb += 1
            if pg_coef and target_kl_old and float(kl_old) > target_kl_old * 4:
                stop = True
                break
        if stop:
            break
    out = {k: v / max(1, nb) for k, v in stats.items()}
    out.update(minibatches=nb, early_stop=stop,
               explained_var=float(1 - (ret_f - value_old).var() / max(1e-8, float(ret_f.var()))))
    return out


# ---------------------------------------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------------------------------------

HEAD_PREFIXES = ("type_emb.", "head.", "scorer.")


class Trainer:
    """Holds the policy, the frozen reference, the value head and the optimizer for one game."""

    def __init__(self, agent: VLMAgent, ref_agent: VLMAgent, game: str, actions: Sequence[str], frame: np.ndarray,
                 n_last: int = 8, lr_head: float = 3e-5, lr_backbone: float = 1e-5, lr_value: float = 1e-3,
                 detach_value: bool = True, prep_threads: int = 8, amp: Optional[bool] = None):
        self.agent, self.model = agent, agent.model
        self.ref = ref_agent.model
        self.device = agent.device
        self.amp = self.device.type == "cuda" if amp is None else amp
        self.detach_value = detach_value
        self.tmpl = PromptTemplate(agent, game, actions, frame)
        self.prep = FramePreprocessor(agent.processor, frame, prep_threads, device=self.device,
                                      backbone=agent.cfg.get("backbone"))
        self.pipeline = ThreadPoolExecutor(1)  # one lane's preprocessing runs while the next is on the GPU
        set_trainable(self.model, "last_n", n_last=n_last)
        self.model.act_head.requires_grad_(False)  # unused here; keep it out of the graph and the optimizer
        self.ref.requires_grad_(False)
        self.ref.eval()
        d = self.model.encoder.config.text_config.hidden_size
        self.value_head = ValueHead(d).to(self.device)
        enc = [p for n, p in self.model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
        head = [p for n, p in self.model.named_parameters()
                if p.requires_grad and n.startswith(HEAD_PREFIXES)]
        self.opt = torch.optim.AdamW([{"params": head, "lr": lr_head},
                                      {"params": enc, "lr": lr_backbone},
                                      {"params": list(self.value_head.parameters()), "lr": lr_value}],
                                     weight_decay=0.0, eps=1e-5)
        self.base_lrs = [g["lr"] for g in self.opt.param_groups]
        self.trainable_params = head + enc + list(self.value_head.parameters())
        self.n_trainable = sum(p.numel() for p in self.trainable_params)
        self.model.eval()

    def trainable_state(self) -> Dict[str, torch.Tensor]:
        """The parameters this run changes (everything else equals the starting checkpoint)."""
        keep = {n for n, p in self.model.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items() if k in keep}

    def set_lr_scale(self, f: float):
        for g, lr in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = lr * f


def policy_entropy_bits(counts: Counter) -> float:
    total = max(1, sum(counts.values()))
    p = np.array([c / total for c in counts.values()])
    return float(-(p * np.log2(np.clip(p, 1e-12, 1))).sum())


def train_ppo(
    agent: VLMAgent,
    ref_agent: VLMAgent,
    game: str = "Breakout",
    n_envs: int = 128,
    n_lanes: int = 2,
    rollout_steps: int = 32,
    iterations: int = 100000,
    max_minutes: float = 100.0,
    epochs: int = 2,
    minibatch: int = 64,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip: float = 0.1,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    kl_coef: float = 0.05,
    kl_target: float = 0.3,
    value_warmup_iters: int = 4,
    n_last: int = 8,
    lr_head: float = 3e-5,
    lr_backbone: float = 1e-5,
    lr_value: float = 1e-3,
    lr_final_frac: float = 0.2,
    clip_reward: bool = True,
    episodic_life: bool = True,
    max_episode_steps: int = 4500,
    seed: int = 0,
    prep_threads: int = 8,
    select_episodes: int = 30,
    ckpt_seconds: float = 600.0,
    state: Optional[Dict] = None,
    on_checkpoint=None,
    on_best=None,
    log: Optional[Dict] = None,
) -> Dict:
    """PPO on ``game``'s score, resuming from ``state`` if given.

    ``on_checkpoint(payload, it)`` is called every ``ckpt_seconds`` with a resumable state dict, and
    ``on_best(score, it)`` whenever the rolling mean of the last ``select_episodes`` full-game raw scores is
    the best so far (checkpoint selection is by game score, not by loss). ``kl_coef`` adapts towards
    ``kl_target`` nats of KL against the starting policy: loose enough for the action distribution to really
    change, tight enough that collapsing onto one action (which costs ``-log pi_ref(a)``, about 0.9 to 2 nats
    on Breakout) is penalised hard.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    it0, frames, best, best_state, elapsed0 = 0, 0, -math.inf, None, 0.0
    if state:
        it0, frames, best, kl_coef = state["iteration"], state["frames"], state["best"], state["kl_coef"]
        best_state, elapsed0 = state.get("best_state"), state.get("elapsed_minutes", 0.0)
    # fresh episodes on a resume, but not the same seeds the lost work already played
    venv = Lanes(game, n_envs, n_lanes, seed=seed * 100000000 + 1 + it0 * 1000, max_steps=max_episode_steps,
                 episodic_life=episodic_life)
    runner = Trainer(agent, ref_agent, game, venv.action_names, venv.lanes[0].obs[0], n_last=n_last,
                     lr_head=lr_head, lr_backbone=lr_backbone, lr_value=lr_value, prep_threads=prep_threads)
    log = log if log is not None else {}
    log.setdefault("iters", [])
    if state:
        runner.model.load_state_dict(state["model"], strict=False)
        runner.value_head.load_state_dict(state["value_head"])
        runner.opt.load_state_dict(state["opt"])
        torch.set_rng_state(state["torch_rng"])
        print("resumed at iteration %d (%d decisions, %.1f min spent, best mean score %.2f)"
              % (it0, frames, elapsed0, best), flush=True)
    print("trainable %d params (last %d LM layers + head + value head), policy temperature %.3f, "
          "%d envs in %d lanes x %d steps = %d decisions per iteration, actions %s"
          % (runner.n_trainable, n_last, runner.tmpl.temperature, n_envs, n_lanes, rollout_steps,
             n_envs * rollout_steps, venv.action_names), flush=True)

    t_start = time.time()
    t_ckpt = time.time()
    total_steps = 0
    for it in range(it0, iterations):
        elapsed = elapsed0 + (time.time() - t_start) / 60  # across resumes, so the budget and LR schedule hold
        if elapsed >= max_minutes:
            print("time budget reached after %.1f min" % elapsed, flush=True)
            break
        progress = min(1.0, elapsed / max_minutes)
        runner.set_lr_scale(1.0 - (1.0 - lr_final_frac) * progress)
        before = dict(venv.counts)
        r, rs = collect(runner, venv, rollout_steps, clip_reward=clip_reward)
        adv, ret = gae(r, gamma, lam)
        pg_coef = 0.0 if it < value_warmup_iters else 1.0
        us = ppo_update(runner, r, adv, ret, epochs=epochs, minibatch=minibatch, clip=clip, vf_coef=vf_coef,
                        ent_coef=ent_coef, kl_coef=kl_coef, pg_coef=pg_coef)
        frames += rollout_steps * n_envs
        total_steps += rollout_steps * n_envs
        if not pg_coef:
            pass
        elif us["kl_ref"] > 2 * kl_target:
            kl_coef = min(10.0, kl_coef * 1.5)
        elif us["kl_ref"] < 0.5 * kl_target:
            kl_coef = max(1e-3, kl_coef / 1.5)
        mix = Counter({a: venv.counts[a] - before.get(a, 0) for a in venv.counts})
        share = {a: round(c / max(1, sum(mix.values())), 3) for a, c in mix.most_common()}
        score = venv.recent(select_episodes)
        row = {"iteration": it, "minutes": round(elapsed, 2), "decisions": frames,
               "steps_per_s": round(rs["steps_per_s"], 1), "rollout_s": round(rs["seconds"], 1),
               "episodes": venv.episodes, "mean_score": score,
               "mean_steps": venv.recent_steps(select_episodes),
               "action_share": share, "action_entropy_bits": round(policy_entropy_bits(mix), 3),
               "kl_coef": round(kl_coef, 4), "pg_coef": pg_coef, "reward_rate": float(np.stack(r.reward).mean()),
               **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in us.items()}}
        log["iters"].append(row)
        print("[it %d | %.1f min | %d dec | %.0f dec/s] score %s (%d eps) | entropy %.3f | kl_ref %.4f (c %.3f) "
              "| pg %.4f v %.4f | ev %.2f | %s"
              % (it, elapsed, frames, rs["steps_per_s"], "n/a" if score is None else "%.2f" % score, venv.episodes,
                 us["entropy"], us["kl_ref"], kl_coef, us["pg_loss"], us["v_loss"], us["explained_var"],
                 " ".join("%s %.2f" % (a, s) for a, s in share.items())), flush=True)
        if score is not None and venv.episodes >= select_episodes and score > best:
            best = score
            best_state = {"model": runner.trainable_state(),
                          "value_head": {k: v.detach().cpu().clone() for k, v in runner.value_head.state_dict().items()},
                          "iteration": it, "mean_score": score, "decisions": frames}
            log["best"] = {"iteration": it, "mean_score": score, "decisions": frames, "episodes": venv.episodes}
            if on_best is not None:
                on_best(score, it)
        if on_checkpoint is not None and time.time() - t_ckpt > ckpt_seconds:
            on_checkpoint({"model": runner.trainable_state(), "value_head": runner.value_head.state_dict(),
                           "opt": runner.opt.state_dict(), "iteration": it + 1, "frames": frames, "best": best,
                           "best_state": best_state, "kl_coef": kl_coef, "torch_rng": torch.get_rng_state(),
                           "elapsed_minutes": elapsed}, it)
            t_ckpt = time.time()
    final_score = venv.recent(select_episodes)
    venv.close()
    session = (time.time() - t_start) / 60
    return {"iterations": len(log["iters"]), "decisions": frames, "minutes": elapsed0 + session,
            "session_minutes": session, "steps_per_s": total_steps / max(1e-9, session * 60),
            "best": log.get("best"), "final_score": final_score, "kl_coef": kl_coef, "best_state": best_state,
            "trainer": runner}


# ---------------------------------------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="a few PPO steps on a fresh (untrained) head")
    ap.add_argument("--game", default="Breakout")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--envs", type=int, default=2)
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=2)
    args = ap.parse_args(argv)
    if not args.smoke:
        ap.error("pass --smoke")
    agent = VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device=args.device)
    ref = VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device=args.device)
    ref.model.load_state_dict(agent.model.state_dict())
    out = train_ppo(agent, ref, args.game, n_envs=args.envs, n_lanes=args.lanes, rollout_steps=args.steps,
                    iterations=args.iterations, max_minutes=60, minibatch=args.envs, epochs=1, n_last=2,
                    prep_threads=2, select_episodes=2, value_warmup_iters=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("best_state", "trainer")}, indent=1, default=str))


if __name__ == "__main__":
    main()
