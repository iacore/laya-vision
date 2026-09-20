"""Modal jobs for RL fine-tuning the Atari decision model on one game's score (PPO, ``laya.atari_rl``).

    modal run modal_atari_rl.py::bench                                    # shapes, throughput, correctness
    modal run modal_atari_rl.py::baseline --model atari-8g-1f/best        # the imitation start's score
    modal run --detach modal_atari_rl.py::train --run-name atari-rl-bo1
    modal run modal_atari_rl.py::evaluate --model atari-rl-bo1/best

Volumes (created out of band; never ``modal deploy`` this app):
    laya-datasets     -> /data       (read-only; /data/atari/expert/<Game>/meta.json holds the baselines)
    laya-hf-cache     -> /cache/hf   (HF_HOME)
    laya-checkpoints  -> /ckpt       (this app writes only under /ckpt/smolvlm/atari-rl-*)

Play-evaluation uses ``laya.atari_train.play`` and ``model_policy`` unchanged, so the numbers are directly
comparable to ``modal_atari_train.py::atari_eval``: ALE v5 defaults (sticky actions 0.25), auto-FIRE on reset
and life loss, episodes capped at 4,500 decisions, seeds ``seed + i``.
"""
import json
import os
import time

import modal

app = modal.App("laya-atari-rl")

hf_vol = modal.Volume.from_name("laya-hf-cache")
data_vol = modal.Volume.from_name("laya-datasets")
ckpt_vol = modal.Volume.from_name("laya-checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.0",
        "torchvision==0.29.0",
        "transformers==5.17.0",
        "safetensors",
        "huggingface_hub",
        "numpy",
        "pillow",
        "num2words",
        "ale-py",
        "gymnasium",
    )
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("laya")
)

CKPT_ROOT = "/ckpt/smolvlm"
ATARI_ROOT = "/data/atari"
TRAIN_VOLUMES = {"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol}


def _baselines(game: str) -> dict:
    """Expert and random scores under the 4,500-decision cap, from the expert dataset's meta.json."""
    try:
        with open(os.path.join(ATARI_ROOT, "expert", game, "meta.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return {"expert": None, "random": None, "capped": False}
    if meta.get("expert_score_cap4500") is not None:
        return {"expert": meta["expert_score_cap4500"], "random": meta["random_score_cap4500"], "capped": True}
    return {"expert": meta.get("expert_score"), "random": meta.get("random_score"), "capped": False}


def _normalized(score, base):
    if base["expert"] is None or base["random"] is None or base["expert"] == base["random"]:
        return None
    return (score - base["random"]) / (base["expert"] - base["random"])


# ---------------------------------------------------------------------------------------------------------
# Bench: does the fast path agree with the slow one, and how many env-steps per second does it run at
# ---------------------------------------------------------------------------------------------------------


@app.function(image=image, gpu="A100", cpu=16, memory=32768, timeout=30 * 60, volumes=TRAIN_VOLUMES)
def bench(model: str = "atari-8g-1f/best", game: str = "Breakout", n_envs: int = 128, n_lanes: int = 2,
          steps: int = 8, prep_threads: int = 12, minibatch: int = 128):
    """Shapes, the correctness check against ``laya.atari_train.action_probs``, and rollout throughput.

    The correctness check is the important one: the rollout builds the token sequence once and feeds cached
    image features, so it must produce the same option logits as the ordinary per-frame path that training and
    ``play`` use. Anything else means the policy being trained is not the policy being evaluated.
    """
    import numpy as np
    import torch

    from laya.atari_rl import Lanes, PromptTemplate, Trainer, collect, encode_frames, gae, head_forward, ppo_update
    from laya.atari_train import action_probs
    from laya.vlm import VLMAgent, set_trainable

    ckpt_vol.reload()
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    path = os.path.join(CKPT_ROOT, model)
    agent = VLMAgent(path, device="cuda")
    out = {"model": model, "game": game, "dtype": agent.cfg.get("dtype"),
           "temperature": list(agent.temperature), "temperature_by_options": dict(agent.temperature_by_options)}

    venv = Lanes(game, n_envs, n_lanes, seed=7)
    obs0 = venv.lanes[0].obs
    out["actions"] = venv.action_names
    tmpl = PromptTemplate(agent, game, venv.action_names, obs0[0])
    out.update(seq_len=tmpl.length, markers=tmpl.markers.tolist(), policy_temperature=tmpl.temperature)

    # --- the last-N freezing stage must exist on this backbone, and must leave the vision tower frozen
    n_train = set_trainable(agent.model, "last_n", n_last=8)
    enc_trainable = sorted({n.split(".layers.")[0] for n, p in agent.model.named_parameters()
                            if p.requires_grad and n.startswith("encoder.")})
    out.update(n_trainable=n_train, trainable_encoder_modules=enc_trainable,
               vision_frozen=not any(p.requires_grad for p in agent.model.encoder.vision_model.parameters()))
    print("template: %d tokens, markers %s, policy T %.3f | trainable %d params in %s | vision frozen %s"
          % (tmpl.length, tmpl.markers.tolist(), tmpl.temperature, n_train, enc_trainable, out["vision_frozen"]),
          flush=True)

    # --- fast path vs laya.atari_train.action_probs on the same frames.
    # action_probs runs in the model's own dtype with no autocast, so fp32-vs-fp32 is the real equivalence
    # test; the bf16 row says whether autocast (what the rollout uses) changes the greedy action.
    from laya.atari_rl import FramePreprocessor

    frames = [obs0[i] for i in range(16)]
    slow = action_probs(agent, frames, tmpl.question)
    prep = FramePreprocessor(agent.processor, obs0[0], threads=prep_threads, device="cuda",
                             backbone=agent.cfg.get("backbone"))
    out["preprocess_path"] = prep.path
    out["preprocess_max_abs_diff"] = prep.max_abs_diff
    out["pixel_shape"] = list(prep.pixel_shape)
    pv, pam = prep(frames)
    for tag, amp in (("fp32", False), ("bf16", True)):
        feats = encode_frames(agent.model, pv, pam, amp=amp)
        out["image_feature_shape"] = list(feats.shape[1:])
        with torch.no_grad():
            logits, _ = head_forward(agent.model, tmpl, feats, None, amp=amp)
        fast = torch.softmax(logits / tmpl.temperature, -1).float().cpu().numpy()
        out["max_abs_prob_diff_" + tag] = float(np.abs(fast - slow).max())
        out["same_argmax_" + tag] = bool((fast.argmax(-1) == slow.argmax(-1)).all())
        print("fast(%s) vs action_probs(fp32): max |dp| %.6f, same argmax %s"
              % (tag, out["max_abs_prob_diff_" + tag], out["same_argmax_" + tag]))
    print("action_probs[0]", np.round(slow[0], 4), "fast[0]", np.round(fast[0], 4))

    # --- component timings
    t = time.time()
    for _ in range(3):
        pv, pam = prep([obs0[i % len(obs0)] for i in range(n_envs // n_lanes)])
        torch.cuda.synchronize()
    out["preprocess_ms_per_batch"] = round(1000 * (time.time() - t) / 3, 1)
    t = time.time()
    for _ in range(3):
        f = encode_frames(agent.model, pv, pam, amp=True)
        torch.cuda.synchronize()
    out["vision_ms_per_batch"] = round(1000 * (time.time() - t) / 3, 1)
    t = time.time()
    for _ in range(3):
        head_forward(agent.model, tmpl, f, None, amp=True)
        torch.cuda.synchronize()
    out["lm_ms_per_batch"] = round(1000 * (time.time() - t) / 3, 1)
    acts = np.zeros(n_envs // n_lanes, dtype=int)
    t = time.time()
    for _ in range(3):
        venv.lanes[0].step(acts)
    out["env_step_ms_per_batch"] = round(1000 * (time.time() - t) / 3, 1)

    # --- a real rollout plus a PPO update
    ref = VLMAgent(path, device="cuda")
    runner = Trainer(agent, ref, game, venv.action_names, obs0[0], n_last=8, prep_threads=prep_threads)
    r, rs = collect(runner, venv, steps)
    out.update(rollout_steps_per_s=round(rs["steps_per_s"], 1), rollout_gpu_frac=round(rs["gpu_seconds"] / rs["seconds"], 2),
               feats_mb=round(sum(x.numel() * x.element_size() for x in r.feats) / 2**20, 1))
    adv, ret = gae(r)
    t = time.time()
    us = ppo_update(runner, r, adv, ret, epochs=1, minibatch=min(minibatch, n_envs))
    out["update_s"] = round(time.time() - t, 2)
    out["update"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in us.items()}
    out["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    # a full training iteration at rollout_steps=64, epochs=2: rollout at the measured rate plus the update
    update_per_sample = out["update_s"] / (steps * n_envs)
    out["projected_decisions_per_s"] = round(1 / (1 / out["rollout_steps_per_s"] + 2 * update_per_sample), 1)
    out["batch_note"] = "per-batch timings are for one lane (%d envs)" % (n_envs // n_lanes)
    out["projected_decisions_per_100min"] = int(out["projected_decisions_per_s"] * 6000)
    venv.close()
    print(json.dumps(out, indent=1))
    return out


@app.function(image=image, gpu="L4", cpu=16, memory=16384, timeout=20 * 60, volumes={"/cache/hf": hf_vol})
def prep_bench(batch: int = 64, repeats: int = 3, threads: int = 8):
    """Which preprocessing path reproduces ``vlm_prefix``, and how fast each one is.

    Preprocessing dominated the first rollout measurement (a 210x160 frame becomes a padded 512x512 tensor),
    so this is worth pinning down on a cheap GPU rather than on the A100.
    """
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    from laya.atari_rl import FramePreprocessor
    from laya.vlm import DEFAULT_BACKBONE, vlm_prefix

    proc = AutoProcessor.from_pretrained(DEFAULT_BACKBONE)
    frames = [np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8) for _ in range(batch)]
    ref = vlm_prefix(proc, [Image.fromarray(frames[0])])["pixel_values"]
    out = {"batch": batch, "cpus": os.cpu_count(), "ref_shape": list(ref.shape),
           "gpu": torch.cuda.get_device_name(0)}
    prep = FramePreprocessor(proc, frames[0], threads=threads, device="cuda", backbone=DEFAULT_BACKBONE)
    out["chosen_path"] = prep.path

    out["resize_steps"] = getattr(prep, "steps", None)

    def timed(name, fn):
        try:
            t = time.time()
            for _ in range(repeats):
                r = fn()
                torch.cuda.synchronize()
            out[name + "_ms"] = round(1000 * (time.time() - t) / repeats, 1)
            return r
        except Exception as e:  # noqa: BLE001
            out[name + "_error"] = repr(e)[:400]

    for name, fn in (("pil_resize_gpu_normalize", lambda: prep._split(frames, torch.device("cuda"))),
                     ("torchvision_cpu", lambda: prep._fast(frames, torch.device("cpu"))),
                     ("numpy_threads", lambda: prep._threaded(frames))):
        got = timed(name, fn)
        if got is not None:
            out[name + "_max_abs_diff"] = float((got[0][:1].float().cpu() - ref.float()).abs().max())

    ip = proc.image_processor
    out["processor"] = {k: str(getattr(ip, k, None)) for k in
                        ("size", "max_image_size", "resample", "do_resize", "do_rescale", "rescale_factor",
                         "do_normalize", "image_mean", "image_std", "do_pad", "do_image_splitting")}
    probe = np.asarray(Image.fromarray(frames[0]))
    try:
        r1 = ip.resize(probe, ip.size, ip.resample)
        out["processor_resize_size_shape"] = list(np.asarray(r1).shape)
        r2 = ip.resize(np.asarray(r1), ip.max_image_size, ip.resample)
        out["processor_resize_max_shape"] = list(np.asarray(r2).shape)
    except Exception as e:  # noqa: BLE001
        out["processor_resize_error"] = repr(e)[:400]

    # where the per-frame cost actually goes
    steps = prep.steps
    timed("resize_step1_only", lambda: list(prep.pool.map(
        lambda f: np.asarray(Image.fromarray(f).resize((steps[0][1], steps[0][0]), prep.resample)), frames)))
    timed("resize_all_steps", lambda: list(prep.pool.map(prep._resize_one, frames)))
    resized = np.stack(list(prep.pool.map(prep._resize_one, frames)))
    timed("upload_normalize_only", lambda: ((torch.from_numpy(resized).to("cuda").permute(0, 3, 1, 2).float()
                                             * prep.rescale) - prep.mean) / prep.std)
    print(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------------------------------------
# Play-evaluation (L4, so it does not eat the A100 budget)
# ---------------------------------------------------------------------------------------------------------


@app.function(image=image, gpu="L4", cpu=4, timeout=120 * 60,
              volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol.read_only()})
def play_eval(model: str, game: str = "Breakout", episodes: int = 10, max_steps: int = 4500, sample: bool = False,
              seed: int = 100_000, random_episodes: int = 10, dtype: str = "bf16"):
    """``episodes`` episodes of ``ALE/<game>-v5`` with a checkpoint, top-action or sampled, plus random play.

    Straight ``laya.atari_train.play`` / ``model_policy`` / ``random_policy``: the same protocol as
    ``modal_atari_train.py::play_atari``, only with a configurable episode count.
    """
    from laya.atari_train import game_actions, model_policy, play, random_policy
    from laya.vlm import VLMAgent

    t0 = time.time()
    ckpt_vol.reload()
    data_vol.reload()
    actions = game_actions(game)
    path = os.path.join(CKPT_ROOT, model)
    agent = VLMAgent(path if os.path.exists(path) else model, device="cuda", dtype=dtype)
    res = play(game, model_policy(agent, game, actions, sample, seed), episodes, max_steps, seed)
    rnd = play(game, random_policy(len(actions), seed), random_episodes, max_steps, seed)
    base = _baselines(game)
    total = max(1, sum(res["actions"].values()))
    out = {"model": model, "game": game, "sample": sample, "episodes": episodes, "max_steps": max_steps,
           "mean_score": res["mean_score"], "scores": res["scores"], "steps": res["steps"], "capped": res["capped"],
           "action_share": {a: round(c / total, 4) for a, c in sorted(res["actions"].items(), key=lambda kv: -kv[1])},
           "random_score": rnd["mean_score"], "baseline_random": base["random"], "baseline_expert": base["expert"],
           "baseline_capped": base["capped"], "normalized": _normalized(res["mean_score"], base),
           "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out, indent=1))
    return out


def _print_rows(rows):
    print("%-26s %-7s %8s %8s %8s %8s %7s  %s"
          % ("model", "policy", "score", "random", "expert", "norm", "steps", "action share"))
    for r in rows:
        f = lambda v: "-" if v is None else "%.2f" % v  # noqa: E731
        print("%-26s %-7s %8.1f %8.1f %8s %8s %7.0f  %s"
              % (r["model"], "sampled" if r["sample"] else "top", r["mean_score"], r["random_score"],
                 f(r["baseline_expert"]), f(r["normalized"]), sum(r["steps"]) / len(r["steps"]),
                 " ".join("%s %.2f" % (a, s) for a, s in r["action_share"].items())))


@app.local_entrypoint()
def baseline(model: str = "atari-8g-1f/best", game: str = "Breakout", episodes: int = 10, out: str = ""):
    """The imitation start's score under the evaluation protocol, top-action and sampled."""
    rows = list(play_eval.starmap([(model, game, episodes, 4500, s) for s in (False, True)]))
    _print_rows(rows)
    if out:
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)
        print("wrote", out)


@app.local_entrypoint()
def evaluate(model: str, game: str = "Breakout", episodes: int = 10, compare: str = "atari-8g-1f/best", out: str = ""):
    """Score an RL run against its imitation start, both policies, in parallel."""
    jobs = [(m, game, episodes, 4500, s) for m in ([compare, model] if compare else [model]) for s in (False, True)]
    rows = list(play_eval.starmap(jobs))
    _print_rows(rows)
    if out:
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)
        print("wrote", out)


# ---------------------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="A100",
    cpu=16,
    memory=65536,
    timeout=180 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    volumes=TRAIN_VOLUMES,
)
def train_rl(
    run_name: str = "atari-rl-breakout",
    game: str = "Breakout",
    init: str = "atari-8g-1f/best",
    n_envs: int = 128,
    n_lanes: int = 2,
    rollout_steps: int = 32,
    max_minutes: float = 100.0,
    epochs: int = 2,
    minibatch: int = 128,
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
    clip_reward: bool = True,
    episodic_life: bool = True,
    seed: int = 0,
    prep_threads: int = 12,
    select_episodes: int = 30,
    ckpt_seconds: float = 600.0,
    quick_eval_episodes: int = 5,
):
    """PPO fine-tune from ``init`` on ``game``, writing /ckpt/smolvlm/<run_name>/{state.pt,best,final,metrics.json}.

    Resumes from ``state.pt`` if it is there, and writes one about every 10 minutes, so a preempted container
    (``retries``) loses at most that. ``best/`` is the checkpoint with the highest rolling mean of the last
    ``select_episodes`` full-game raw scores from the rollout itself -- selection by game score, not by loss.
    """
    import torch

    from laya.atari_rl import train_ppo
    from laya.vlm import VLMAgent

    if not run_name.startswith("atari-rl-"):
        raise SystemExit("run_name must start with 'atari-rl-' (this app writes only /ckpt/smolvlm/atari-rl-*)")
    t_start = time.time()
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)
    ckpt_vol.reload()
    data_vol.reload()
    out_dir = os.path.join(CKPT_ROOT, run_name)
    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, "state.pt")

    init_path = os.path.join(CKPT_ROOT, init)
    agent = VLMAgent(init_path, device="cuda")
    ref = VLMAgent(init_path, device="cuda")
    hf_vol.commit()

    state, log = None, {"iters": []}
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        log = state.get("log") or log
        print("found %s: resuming from iteration %d" % (state_path, state["iteration"]), flush=True)
    args = dict(run_name=run_name, game=game, init=init, n_envs=n_envs, n_lanes=n_lanes, rollout_steps=rollout_steps,
                max_minutes=max_minutes, epochs=epochs, minibatch=minibatch, gamma=gamma, lam=lam, clip=clip,
                vf_coef=vf_coef, ent_coef=ent_coef, kl_coef=kl_coef, kl_target=kl_target, n_last=n_last,
                lr_head=lr_head, lr_backbone=lr_backbone, lr_value=lr_value, clip_reward=clip_reward,
                episodic_life=episodic_life, value_warmup_iters=value_warmup_iters, seed=seed,
                select_episodes=select_episodes)
    log["args"] = args
    print("args:", json.dumps(args), flush=True)

    def write_log():
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(log, f, indent=2)

    def on_checkpoint(payload, it):
        t = time.time()
        payload["log"] = log
        tmp = state_path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, state_path)
        write_log()
        ckpt_vol.commit()
        print("  checkpointed iteration %d (%.1f MB, %.1f s)"
              % (it, os.path.getsize(state_path) / 2**20, time.time() - t), flush=True)

    def on_best(score, it):
        print("  new best rolling score %.2f at iteration %d" % (score, it), flush=True)

    res = train_ppo(agent, ref, game, n_envs=n_envs, n_lanes=n_lanes, rollout_steps=rollout_steps, max_minutes=max_minutes,
                    epochs=epochs, minibatch=minibatch, gamma=gamma, lam=lam, clip=clip, vf_coef=vf_coef,
                    ent_coef=ent_coef, kl_coef=kl_coef, kl_target=kl_target, value_warmup_iters=value_warmup_iters,
                    n_last=n_last, lr_head=lr_head, lr_backbone=lr_backbone, lr_value=lr_value,
                    clip_reward=clip_reward, episodic_life=episodic_life, seed=seed, prep_threads=prep_threads,
                    select_episodes=select_episodes, ckpt_seconds=ckpt_seconds, state=state,
                    on_checkpoint=on_checkpoint, on_best=on_best, log=log)
    runner, best_state = res.pop("trainer"), res.pop("best_state")
    log["result"] = {k: v for k, v in res.items()}
    print("rollout: %d decisions in %.1f min (%.0f decisions/s); final rolling score %s, best %s"
          % (res["decisions"], res["minutes"], res["steps_per_s"], res["final_score"], res["best"]), flush=True)

    agent.cfg["atari_rl"] = {"run": run_name, "game": game, "init": init, "decisions": res["decisions"],
                            "algorithm": "ppo", "n_last": n_last, "best": res["best"]}
    agent.save(os.path.join(out_dir, "final"))
    if best_state is not None:
        agent.model.load_state_dict(best_state["model"], strict=False)
        agent.save(os.path.join(out_dir, "best"))
        torch.save(best_state["value_head"], os.path.join(out_dir, "best", "value_head.pt"))
        print("saved best/ from iteration %d (rolling score %.2f)"
              % (best_state["iteration"], best_state["mean_score"]), flush=True)
    write_log()
    ckpt_vol.commit()

    if quick_eval_episodes:
        from laya.atari_train import game_actions, model_policy, play

        actions = game_actions(game)
        agent.model.eval()
        quick = {}
        for name, sample in (("top", False), ("sampled", True)):
            p = play(game, model_policy(agent, game, actions, sample, 100_000), quick_eval_episodes, 4500, 100_000)
            quick[name] = {"mean_score": p["mean_score"], "scores": p["scores"], "actions": p["actions"]}
            print("quick eval (%s, %d episodes): mean %.1f %s" % (name, quick_eval_episodes, p["mean_score"],
                                                                  p["scores"]), flush=True)
        log["quick_eval"] = quick
        write_log()
        ckpt_vol.commit()
    print("done in %.1f min" % ((time.time() - t_start) / 60), flush=True)
    return {"run": run_name, "result": log["result"], "quick_eval": log.get("quick_eval")}


@app.local_entrypoint()
def train(run_name: str = "atari-rl-breakout", game: str = "Breakout", init: str = "atari-8g-1f/best",
          max_minutes: float = 100.0, n_envs: int = 128, rollout_steps: int = 32, kl_coef: float = 0.05,
          ent_coef: float = 0.01, lr_head: float = 3e-5, lr_backbone: float = 1e-5, n_last: int = 8,
          epochs: int = 2, seed: int = 0):
    r = train_rl.remote(run_name=run_name, game=game, init=init, max_minutes=max_minutes, n_envs=n_envs,
                        rollout_steps=rollout_steps, kl_coef=kl_coef, ent_coef=ent_coef, lr_head=lr_head,
                        lr_backbone=lr_backbone, n_last=n_last, epochs=epochs, seed=seed)
    print(json.dumps(r, indent=1))


@app.local_entrypoint()
def smoke(run_name: str = "atari-rl-smoke"):
    """End to end in a few minutes: train, checkpoint, resume from that checkpoint, then play-evaluate.

    The second ``train_rl`` call is the point: it must find ``state.pt`` and continue, which is what a Modal
    preemption retry does.
    """
    kw = dict(run_name=run_name, n_envs=8, n_lanes=2, rollout_steps=8, minibatch=8, value_warmup_iters=1,
              select_episodes=2, prep_threads=8, ckpt_seconds=20.0)
    print(json.dumps(train_rl.remote(max_minutes=2.0, quick_eval_episodes=0, **kw), indent=1))
    print(json.dumps(train_rl.remote(max_minutes=4.0, quick_eval_episodes=2, **kw), indent=1))
    rows = list(play_eval.starmap([("%s/best" % run_name, "Breakout", 2, 600, s) for s in (False, True)]))
    _print_rows(rows)
