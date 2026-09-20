"""Modal jobs for the game-only Atari model: the plain SmolVLM backbone with a fresh head, trained only on Atari
frames (no photo-VQA data), then scored by playing ALE games.

    modal run modal_atari_train.py::datasets                       # which (source, game) pairs are ready
    modal run modal_atari_train.py::smoke                          # end to end on a tiny synthetic dataset
    modal run --detach modal_atari_train.py::train_atari --run-name atari-v1 [--sources expert,atari_head,jat]
    modal run --detach modal_atari_train.py::train_atari --run-name atari8-2f --sources expert2f --frames 2 \
        --games Breakout,Pong --init-from atari-expert-v1/best
    modal run modal_atari_train.py::atari_eval --model atari-v1/best [--games Breakout,Pong] [--episodes 3] [--sample]
    modal run modal_atari_train.py::renormalize --results play.json [--out play_renorm.json]

Volumes (created out of band; never ``modal deploy`` this app):
    laya-datasets     -> /data       (read-only; /data/atari/<source>/<Game>/, see docs/atari-data-format.md)
    laya-hf-cache     -> /cache/hf   (HF_HOME)
    laya-checkpoints  -> /ckpt       (this app writes only under /ckpt/smolvlm/atari-*)
"""
import json
import os
import statistics
import time

import modal

app = modal.App("laya-atari")

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

BACKBONE = "HuggingFaceTB/SmolVLM-256M-Instruct"
CKPT_ROOT = "/ckpt/smolvlm"
ATARI_ROOT = "/data/atari"
SYNTH_ROOT = "/tmp/atari_synth"


def _split(s: str):
    return [x for x in s.split(",") if x]


@app.function(image=image, timeout=10 * 60, volumes={"/data": data_vol.read_only()})
def list_ready(root: str = ATARI_ROOT):
    """Ready (source, game) pairs with their frame format and record counts from meta.json."""
    from laya.atari_train import ready_datasets

    data_vol.reload()
    return [{"name": d["name"], "frame_format": d["frame_format"], "train": (d["meta"].get("train") or {}).get("records"),
             "val": (d["meta"].get("val") or {}).get("records"), "expert_score": d["meta"].get("expert_score"),
             "random_score": d["meta"].get("random_score")} for d in ready_datasets(root)]


@app.local_entrypoint()
def datasets():
    rows = list_ready.remote()
    for r in rows:
        print("%-30s %-12s train %6s val %5s  expert %s random %s" % (r["name"], r["frame_format"], r["train"], r["val"],
                                                                     r["expert_score"], r["random_score"]))
    print("%d ready datasets" % len(rows))


@app.function(
    image=image,
    gpu="A100",
    cpu=24,
    memory=65536,
    timeout=150 * 60,
    volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol},
)
def train_atari(
    run_name: str = "atari-v1",
    sources: str = "expert,atari_head,jat",
    games: str = "",
    passes: float = 1.5,
    max_minutes: float = 85.0,
    batch_size: int = 32,
    lr_head: float = 1e-4,
    lr_backbone: float = 2e-5,
    lr_ref_batch: int = 16,
    warmup_frac: float = 0.03,
    n_evals: int = 6,
    val_per_ds: int = 300,
    final_val_per_ds: int = 1000,
    train_eval_per_game: int = 100,
    n_calib: int = 60,
    max_passes: float = 4.0,
    max_train_per_ds: int = 0,
    num_workers: int = 22,
    synthetic: bool = False,
    frames: int = 1,
    init_from: str = "",
):
    """Train SmolVLM (fresh head, vision tower frozen) on Atari frames only; save /ckpt/smolvlm/<run_name>/best.

    * Data: every ready ``/data/atari/<source>/<Game>`` for ``sources`` (and ``games`` if given). Games are sampled
      equally, the sources of a game are pooled. ``passes`` counts samples over the pooled train set;
      ``max_passes`` caps the passes over any one game (a capped game leaves the mix). ``max_minutes`` caps the
      training wall-clock (the LR schedule follows whichever of step or time progress is further).
    * ``n_calib`` frames per (source, game), from held-out train episodes, are kept out of training for fitting
      temperatures.
    * ``n_evals`` periodic evals on up to ``val_per_ds`` val frames per (source, game): per-game accuracy, ECE and
      NLL against the labels, plus ``train_eval_per_game`` seen train frames per game for the overfitting gap.
      ``best/`` is saved whenever the mean per-game val NLL improves (calibration matters more than accuracy).
    * The final model is the best one, with per-type and per-option-count temperatures fitted on the holdout,
      then scored on up to ``final_val_per_ds`` val frames per (source, game), raw and calibrated.
    * ``frames=2`` gives the model ``{"images": [prev_image, image]}`` from the ``expert2f`` layout instead of the
      single frame; the value is saved in the checkpoint config, so ``play_atari`` matches it by default.
    * ``init_from`` (a run under /ckpt/smolvlm, e.g. ``atari-expert-v1/best``) continues from a trained checkpoint
      instead of a fresh head; question types absent from the calibration holdout keep its temperature.
    """
    import math

    import torch

    from laya.atari_train import (even_subsample, fit_option_temperatures, load_atari, per_game_metrics,
                                  scale_records, write_synthetic)
    from laya.vlm import VLMAgent
    from laya.vlm_train import collect_logits, fit_temperatures_from, train

    if not run_name.startswith("atari-"):
        raise SystemExit("run_name must start with 'atari-' (this app writes only /ckpt/smolvlm/atari-*)")
    t_start = time.time()
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    out_dir = os.path.join(CKPT_ROOT, run_name)
    root = ATARI_ROOT
    if synthetic:
        root = SYNTH_ROOT
        print("synthetic data:", write_synthetic(root, n_train=96, n_val=24))
    else:
        data_vol.reload()
    data = load_atari(root, _split(sources), _split(games), n_calib=n_calib, val_limit=final_val_per_ds,
                      train_limit=max_train_per_ds or None, frames=frames)
    if not data["train"]:
        raise SystemExit("no ready Atari data for sources=%s games=%s" % (sources, games))
    names, train_ex = data["games"], data["train"]
    # balance by game: ItemStream samples each "dataset" group equally, so group the pooled sources under the game
    train_by_game = [dict(ex, dataset=ex["game"]) for ex in train_ex]
    val_small = []
    for ds in data["datasets"]:
        val_small += even_subsample([ex for ex in data["val"] if ex["dataset"] == ds["name"]], val_per_ds)
    train_eval = []
    for g in names:
        train_eval += [ex for ex in train_by_game if ex["game"] == g][:train_eval_per_game]

    scale = math.sqrt(batch_size / lr_ref_batch)
    lr_h, lr_b = lr_head * scale, lr_backbone * scale
    steps = int(math.ceil(passes * len(train_ex) / batch_size))
    eval_every = max(1, steps // max(1, n_evals))
    warmup = max(1, int(warmup_frac * steps))
    sizes = {g: sum(ex["game"] == g for ex in train_ex) for g in names}
    per_game = passes * len(train_ex) / len(names)
    print("plan: %d steps x batch %d (%.2f passes of %d frames), warmup %d, eval every %d on %d val frames, "
          "lr head %.2e backbone %.2e, max %.0f min" % (steps, batch_size, passes, len(train_ex), warmup, eval_every,
                                                        len(val_small), lr_h, lr_b, max_minutes))
    print("expected passes per game with equal sampling (before max_passes=%s): %s"
          % (max_passes or None, {g: round(per_game / sizes[g], 2) for g in names}))

    if init_from:
        agent = VLMAgent(os.path.join(CKPT_ROOT, init_from), device="cuda")
        print("initialised from %s (temperatures %s)" % (init_from, [round(t, 3) for t in agent.temperature]))
    else:
        agent = VLMAgent(backbone=BACKBONE, device="cuda")
    init_temps = list(agent.temperature)
    agent.cfg["atari_frames"] = frames
    hf_vol.commit()
    model, proc = agent.model, agent.processor
    ev_kw = dict(batch_size=64, num_workers=num_workers)
    datasets_log = [{k: d[k] for k in ("name", "frame_format", "n_train", "n_calib", "n_val", "n_soft")} for d in data["datasets"]]
    log = {"run": run_name, "games": names, "datasets": datasets_log, "frames": frames,
           "args": dict(sources=sources, games=games, frames=frames, init_from=init_from, passes=passes, max_minutes=max_minutes, batch_size=batch_size,
                        lr_head=lr_h, lr_backbone=lr_b, warmup=warmup, steps=steps, eval_every=eval_every,
                        max_passes=max_passes, n_calib=n_calib, val_per_ds=val_per_ds, synthetic=synthetic),
           "evals": []}
    best = {"nll": math.inf, "step": None, "state": None}

    def write_log():
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(log, f, indent=2)

    def eval_fn(step, keep=True):
        te = time.time()
        val_m = per_game_metrics(collect_logits(model, proc, val_small, **ev_kw))
        tr_m = per_game_metrics(collect_logits(model, proc, train_eval, **ev_kw))
        row = {"step": step, "passes": round(step * batch_size / len(train_ex), 3), "val": val_m, "train": tr_m}
        log["evals"].append(row)
        print("[eval step %d, %.2f passes] val mean per-game acc %.4f ece %.4f nll %.4f | train acc %.4f nll %.4f"
              % (step, row["passes"], val_m["mean"]["acc"], val_m["mean"]["ece"], val_m["mean"]["nll"],
                 tr_m["mean"]["acc"], tr_m["mean"]["nll"]), flush=True)
        print("  " + " | ".join("%s %.3f/%.2f" % (g, m["acc"], m["nll"]) for g, m in val_m["games"].items()), flush=True)
        if keep and val_m["mean"]["nll"] < best["nll"]:
            best.update(nll=val_m["mean"]["nll"], step=step,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            agent.save(os.path.join(out_dir, "best"))
            print("  new best (step %d, mean val nll %.4f); saved %s/best" % (step, best["nll"], out_dir), flush=True)
        log["best_step"], log["best_mean_val_nll"] = best["step"], best["nll"]
        write_log()
        ckpt_vol.commit()
        print("  eval + save took %.1f min" % ((time.time() - te) / 60), flush=True)

    eval_fn(0, keep=False)  # untrained head: the reference point for NLL
    t_train = time.time()

    def maybe_eval(step):
        # evals at 1/n_evals, 2/n_evals, ... of training progress (steps or wall-clock, whichever is further)
        progress = max(step / steps, (time.time() - t_train) / (max_minutes * 60))
        if progress >= len(log["evals"]) / n_evals:
            eval_fn(step)

    stats = {}
    losses = train(
        model, proc, train_by_game, steps=steps, batch_size=batch_size, freeze="full", lr_head=lr_h, lr_backbone=lr_b,
        device="cuda", log_every=100, max_minutes=max_minutes, num_workers=num_workers, warmup=warmup,
        eval_fn=maybe_eval, eval_every=min(25, eval_every), max_passes=max_passes or None, stats=stats,
    )
    if log["evals"][-1]["step"] != stats["steps"]:
        eval_fn(stats["steps"])
    chunk = max(1, len(losses) // 10)
    log["train_stats"] = dict(stats, passes={g: round(stats["samples_per_dataset"].get(g, 0) / sizes[g], 2) for g in names})
    log["loss_curve"] = [{"steps": "%d-%d" % (i, min(i + chunk, len(losses)) - 1), "mean_loss": sum(losses[i:i + chunk]) / len(losses[i:i + chunk])}
                         for i in range(0, len(losses), chunk)]
    print("train stats:", json.dumps(log["train_stats"]))
    print("loss curve (10 chunks):", ", ".join("%.3f" % c["mean_loss"] for c in log["loss_curve"]))

    model.load_state_dict(best["state"])
    model.eval()
    print("final model: best checkpoint from step %d (mean per-game val nll %.4f)" % (best["step"], best["nll"]))
    calib_records = collect_logits(model, proc, data["calib"], **ev_kw)
    temps = fit_temperatures_from(calib_records)
    n_by_type = [sum(r["qtype"] == t for r in calib_records) for t in range(3)]
    temps = [t if n >= 10 else init_temps[i] for i, (t, n) in enumerate(zip(temps, n_by_type))]
    by_options = fit_option_temperatures(calib_records)
    val_records = collect_logits(model, proc, data["val"], **ev_kw)
    log["temperature"], log["temperature_by_options"] = temps, by_options
    log["final"] = {"step": best["step"], "n_val": len(val_records), "val_raw": per_game_metrics(val_records),
                    "val_calibrated": per_game_metrics(scale_records(val_records, temps, by_options))}
    print("temperatures (choice, score, noul):", [round(t, 3) for t in temps], "| by option count:",
          {k: round(v, 3) for k, v in by_options.items()})
    for key in ("val_raw", "val_calibrated"):
        m = log["final"][key]
        print("[final %s] mean per-game acc %.4f ece %.4f nll %.4f | pooled acc %.4f ece %.4f nll %.4f" % (
            key, m["mean"]["acc"], m["mean"]["ece"], m["mean"]["nll"], m["all"]["acc"], m["all"]["ece"], m["all"]["nll"]))
    cal = log["final"]["val_calibrated"]
    print("%-28s %6s %7s %7s %7s" % ("per source/game (calibrated)", "n", "acc", "ece", "nll"))
    for n_, m in cal["datasets"].items():
        print("%-28s %6d %7.3f %7.3f %7.3f" % (n_, m["n"], m["acc"], m["ece"], m["nll"]))

    agent.temperature, agent.temperature_by_options = temps, by_options
    agent.save(os.path.join(out_dir, "best"))
    write_log()
    ckpt_vol.commit()
    print("saved %s/best with temperatures (%.1f min total)" % (out_dir, (time.time() - t_start) / 60))
    return {"run": run_name, "games": names, "best_step": best["step"], "best_mean_val_nll": best["nll"],
            "temperature": temps, "temperature_by_options": by_options,
            "final_mean": {k: log["final"][k]["mean"] for k in ("val_raw", "val_calibrated")}, "train_stats": log["train_stats"]}


@app.function(image=image, timeout=5 * 60, volumes={"/ckpt": ckpt_vol.read_only()})
def run_games(model: str):
    """Games a run was trained on, from its metrics.json (``model`` like ``atari-v1/best``)."""
    ckpt_vol.reload()
    with open(os.path.join(CKPT_ROOT, model.split("/")[0], "metrics.json")) as f:
        return json.load(f)["games"]


# Games whose expert baseline makes the normalised score meaningless (excluded) or unstable (flagged).
BASELINE_EXCLUDE = {"Solaris": "expert scores below random"}
BASELINE_FLAGS = {
    "Skiing": "expert is NOOP on every frame; copying it is trivial",
    "PrivateEye": "expert barely above random",
    "Pitfall": "expert stands still, barely above random",
    "MontezumaRevenge": "expert barely above random",
    "Tutankham": "greedy expert gets stuck",
    "DoubleDunk": "stalling until the step cap scores well",
    "Tennis": "stalling until the step cap scores well",
}


def expert_baseline(game: str, root: str = ATARI_ROOT) -> dict:
    """Expert and random scores from /data/atari/expert/<game>/meta.json, read fresh on every call.

    Prefers ``expert_score_cap4500`` / ``random_score_cap4500`` (measured under play-eval's 4,500-step cap and
    settings); falls back to the uncapped scores with ``capped`` False.
    """
    try:
        with open(os.path.join(root, "expert", game, "meta.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return {"expert": None, "random": None, "capped": False}
    if meta.get("expert_score_cap4500") is not None and meta.get("random_score_cap4500") is not None:
        return {"expert": meta["expert_score_cap4500"], "random": meta["random_score_cap4500"], "capped": True}
    return {"expert": meta.get("expert_score"), "random": meta.get("random_score"), "capped": False}


def normalize(r: dict, base: dict) -> dict:
    """Add (model - random) / (expert - random) from ``base`` to a play result, with exclusion / flag notes."""
    r = dict(r, expert_score=base["expert"], baseline_random=base["random"], baseline_capped=base["capped"],
             normalized=None, flag=None)
    if base["expert"] is not None and base["random"] is not None and base["expert"] != base["random"]:
        r["normalized"] = (r["model_score"] - base["random"]) / (base["expert"] - base["random"])
    g = r["game"]
    if g in BASELINE_EXCLUDE:
        r["flag"] = "excluded: " + BASELINE_EXCLUDE[g]
    elif g in BASELINE_FLAGS:
        r["flag"] = BASELINE_FLAGS[g]
    elif r["normalized"] is not None and not -0.5 <= r["normalized"] <= 1.5:
        r["flag"] = "normalised score outside [-0.5, 1.5]"
    elif r["normalized"] is not None and not base["capped"]:
        r["flag"] = "uncapped baseline"
    return r


@app.function(image=image, timeout=5 * 60, volumes={"/data": data_vol.read_only()})
def expert_baselines(games: list):
    data_vol.reload()
    return {g: expert_baseline(g) for g in games}


@app.function(image=image, gpu="L4", cpu=4, timeout=60 * 60,
              volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol.read_only()})
def play_atari(game: str, model: str, episodes: int = 3, max_steps: int = 4500, seed: int = 100_000,
               random_episodes: int = 10, sample: bool = False, frames: int = 0):
    """Play ``ALE/<game>-v5`` with a checkpoint (bf16) and with random actions, same settings.

    The model's action is the most likely one, or with ``sample`` drawn from its calibrated probabilities.
    ``frames`` is 1, 2 (the previous observation goes in too, by the ``expert2f`` rule), or 0 to take the
    checkpoint's own ``atari_frames``.
    The model sees the raw RGB observation and the ``laya.games.atari_question`` question; FIRE is pressed on
    reset and after each lost life (not counted toward ``max_steps``). Episode i uses seed ``seed + i``.
    ``random_score`` is measured here under the same settings; the normalised score uses the expert meta.json
    baselines (``expert_baseline``), read at eval time.
    """
    from laya.atari_train import game_actions, model_policy, play, random_policy
    from laya.vlm import VLMAgent

    t0 = time.time()
    actions = game_actions(game)
    rnd = play(game, random_policy(len(actions), seed), random_episodes, max_steps, seed)
    path = os.path.join(CKPT_ROOT, model)
    agent = VLMAgent(path if os.path.exists(path) else model, device="cuda", dtype="bf16")
    n_frames = frames or int(agent.cfg.get("atari_frames", 1))
    res = play(game, model_policy(agent, game, actions, sample, seed, n_frames), episodes, max_steps, seed)
    data_vol.reload()
    out = normalize({"game": game, "model": model, "sample": sample, "frames": n_frames, "model_score": res["mean_score"],
                     "model_scores": res["scores"], "model_steps": res["steps"], "model_capped": res["capped"],
                     "actions": res["actions"], "random_score": rnd["mean_score"], "random_steps": rnd["steps"],
                     "seconds": round(time.time() - t0, 1)}, expert_baseline(game))
    print(json.dumps(out))
    return out


def _summary(results):
    lines = ["%-18s %10s %10s %10s %10s %8s %6s  %s" % ("game", "model", "random", "base rnd", "expert", "norm", "steps",
                                                         "top actions / flag")]
    norms, clean, beat = [], [], 0
    for r in sorted(results, key=lambda r: r["game"]):
        top = ", ".join("%s %d%%" % (a, 100 * c / max(1, sum(r["actions"].values())))
                        for a, c in sorted(r["actions"].items(), key=lambda kv: -kv[1])[:3])
        f = lambda v: "-" if v is None else "%.1f" % v  # noqa: E731
        lines.append("%-18s %10.1f %10.1f %10s %10s %8s %6d  %s" % (
            r["game"], r["model_score"], r["random_score"], f(r.get("baseline_random")), f(r["expert_score"]),
            "-" if r["normalized"] is None else "%.2f" % r["normalized"], sum(r["model_steps"]) / len(r["model_steps"]),
            top + ("  [%s]" % r["flag"] if r.get("flag") else "")))
        beat += r["model_score"] > r["random_score"]
        if r["normalized"] is not None and r["game"] not in BASELINE_EXCLUDE:
            norms.append(r["normalized"])
            if not r.get("flag"):
                clean.append(r["normalized"])
    med = statistics.median(norms) if norms else None
    med_clean = statistics.median(clean) if clean else None
    fmt = lambda v: "-" if v is None else "%.3f" % v  # noqa: E731
    lines.append("median normalised score %s over %d games (Solaris excluded); %s over %d unflagged games; "
                 "beats random (own 10-episode random) in %d of %d games"
                 % (fmt(med), len(norms), fmt(med_clean), len(clean), beat, len(results)))
    return "\n".join(lines), {"median_normalized": med, "n_normalized": len(norms), "median_normalized_unflagged": med_clean,
                              "n_unflagged": len(clean), "beat_random": beat, "n_games": len(results)}


@app.local_entrypoint()
def atari_eval(model: str, games: str = "", episodes: int = 3, max_steps: int = 4500, sample: bool = False,
               frames: int = 0, out: str = ""):
    """modal run modal_atari_train.py::atari_eval --model atari-v1/best  -- every trained game in parallel on L4s."""
    game_list = _split(games) or run_games.remote(model)
    print("playing %d games x %d episodes with %s (%s, frames=%s): %s" % (len(game_list), episodes, model,
          "sampled" if sample else "greedy", frames or "from checkpoint", ", ".join(game_list)))
    results = []
    for r in play_atari.starmap([(g, model, episodes, max_steps, 100_000, 10, sample, frames) for g in game_list],
                                return_exceptions=True):
        if isinstance(r, Exception):
            print("failed:", repr(r))
        else:
            results.append(r)
    text, summary = _summary(results)
    print(text)
    if out:
        with open(out, "w") as f:
            json.dump({"model": model, "sample": sample, "frames": frames, "results": results, "summary": summary}, f, indent=2)
        print("wrote", out)


@app.local_entrypoint()
def renormalize(results: str, out: str = ""):
    """modal run modal_atari_train.py::renormalize --results play.json  -- recompute normalised scores of saved
    atari_eval results from the current expert meta.json baselines (no replay)."""
    with open(results) as f:
        data = json.load(f)
    base = expert_baselines.remote(sorted({r["game"] for r in data["results"]}))
    data["results"] = [normalize(r, base[r["game"]]) for r in data["results"]]
    text, data["summary"] = _summary(data["results"])
    print("%s (%s)" % (data["model"], "sampled" if data.get("sample") else "greedy"))
    print(text)
    if out:
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print("wrote", out)


@app.local_entrypoint()
def smoke(frames: int = 1, init_from: str = ""):
    """End to end on synthetic data: a few training steps on the A100 job, then play-eval of the saved checkpoint."""
    r = train_atari.remote(run_name="atari-smoke", passes=4.0, max_minutes=3.0, n_evals=2, val_per_ds=24, n_calib=8,
                           num_workers=8, synthetic=True, frames=frames, init_from=init_from)
    print(json.dumps(r, indent=1))
    results = list(play_atari.starmap([(g, "atari-smoke/best", 2, 200) for g in r["games"]]))
    print(_summary(results)[0])
