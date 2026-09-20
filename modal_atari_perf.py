"""Modal jobs measuring what image preprocessing costs the Atari decision model, and what the cheap path saves.

    modal run modal_atari_perf.py::deltas                          # pixel and probability deltas from the filter
    modal run modal_atari_perf.py::bench                           # the benchmark table (one A100, ~10 min)
    modal run modal_atari_perf.py::bench --gpu L4 --game Pong

Three configurations are compared throughout, single-frame and two-frame:

    512 processor   what `atari-8g-2f` was trained and played with: the Hugging Face processor, on the CPU,
                    two LANCZOS hops (210x160 -> 2048x1560 -> 512x512), 64 image tokens
    512 gpu         `laya.preprocess.ImagePrep`, the same filter as one matrix pair on the GPU, 64 image tokens
    256 gpu         the same, straight to 256x256, 16 image tokens

Volumes (created out of band; never ``modal deploy`` this app):
    laya-datasets     -> /data       (read-only)
    laya-hf-cache     -> /cache/hf   (HF_HOME)
    laya-checkpoints  -> /ckpt       (read-only here)
"""
import json
import os
import time

import modal

app = modal.App("laya-atari-perf")

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
DEFAULT_MODEL = "atari-8g-2f/best"

#: (label, image_size, preprocess backend) -- the interpolation is always the processor's own filter
CONFIGS = (("512 processor", 512, "processor"), ("512 gpu", 512, "gpu"), ("256 gpu", 256, "gpu"))


def _frames(game: str, n: int, root: str = ATARI_ROOT, source: str = "expert2f"):
    """``n`` consecutive (previous, current) frame pairs from the real training data, plus the PNG decode cost."""
    import numpy as np
    from PIL import Image

    d = os.path.join(root, source, game)
    with open(os.path.join(d, "train.jsonl")) as f:
        recs = [json.loads(line) for line in f][: n + 1]
    paths = [(os.path.join(d, r.get("prev_image") or r["image"]), os.path.join(d, r["image"])) for r in recs[:n]]
    t0 = time.perf_counter()
    pairs = [(np.asarray(Image.open(p).convert("RGB")), np.asarray(Image.open(c).convert("RGB"))) for p, c in paths]
    decode_ms = (time.perf_counter() - t0) / max(1, 2 * len(pairs)) * 1000
    return [p for p, _ in pairs], [c for _, c in pairs], decode_ms


@app.function(image=image, gpu="L4", cpu=4, timeout=40 * 60,
              volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol.read_only()})
def filter_deltas(model: str = DEFAULT_MODEL, game: str = "Breakout", n: int = 50):
    """How far the cheap path's pixels and the model's probabilities move when the filter changes.

    The pixel deltas are against the Hugging Face processor's own output, in 0-255 units. The probability deltas
    use one checkpoint's weights under both paths, so they isolate the preprocessing: a model *trained* on the
    new path never sees this difference.
    """
    import numpy as np
    import torch
    from transformers import AutoProcessor

    from laya.atari_train import action_probs, game_actions
    from laya.games import atari_question
    from laya.preprocess import ImagePrep
    from laya.vlm import PREFIX_TEXT, VLMAgent

    data_vol.reload()
    prev, cur, decode_ms = _frames(game, n)
    print("%d frame pairs from %s/expert2f, PNG decode %.2f ms/frame" % (len(cur), game, decode_ms))
    out = {"game": game, "n": len(cur), "png_decode_ms_per_frame": round(decode_ms, 3), "pixels": {}, "probs": {}}

    # --- pixels, against the processor at each size -------------------------------------------------------
    for size in (512, 256):
        proc = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
        ImagePrep(image_size=size, backend="processor").apply(proc)
        ref = torch.cat([proc(text=[PREFIX_TEXT + proc.image_token], images=[[f]], do_image_splitting=False,
                              return_tensors="pt")["pixel_values"][0] for f in cur])
        for interp in ("processor", "lanczos", "bicubic", "bilinear"):
            pv, _ = ImagePrep(image_size=size, interpolation=interp).pixel_values(cur, device="cuda")
            d = (pv.cpu() - ref.reshape(pv.shape).cpu()).abs() * 127.5
            key = "%d %s" % (size, interp)
            out["pixels"][key] = {"max": round(float(d.max()), 4), "mean": round(float(d.mean()), 5),
                                  "p999": round(float(d.flatten().quantile(0.999)), 4)}
            print("  pixels %-18s max %8.4f  mean %8.5f  p99.9 %8.4f" % (key, d.max(), d.mean(), d.flatten().quantile(0.999)))

    # --- probabilities, one set of weights under both paths -----------------------------------------------
    path = os.path.join(CKPT_ROOT, model)
    q = atari_question(game, game_actions(game))["action"]
    base = VLMAgent(path, device="cuda", dtype="bf16")
    frames_cfg = int(base.cfg.get("atari_frames", 1))
    ref_p = _probs(base, cur, prev if frames_cfg == 2 else None, q)
    del base
    torch.cuda.empty_cache()
    for interp in ("processor", "lanczos", "bicubic"):
        agent = VLMAgent(path, device="cuda", dtype="bf16", preprocess="gpu", image_interpolation=interp)
        p = _probs(agent, cur, prev if frames_cfg == 2 else None, q)
        d = np.abs(p - ref_p)
        row = {"max_abs": round(float(d.max()), 5), "mean_abs": round(float(d.mean()), 5),
               "mean_total_variation": round(float(0.5 * d.sum(-1).mean()), 5),
               "top_action_agrees": int((p.argmax(-1) == ref_p.argmax(-1)).sum()), "n": len(cur)}
        out["probs"]["512 gpu %s" % interp] = row
        print("  probs  512 gpu %-10s max %.5f  mean %.5f  TV %.5f  top action agrees %d/%d"
              % (interp, row["max_abs"], row["mean_abs"], row["mean_total_variation"], row["top_action_agrees"], len(cur)))
        del agent
        torch.cuda.empty_cache()
    print(json.dumps(out))
    return out


def _probs(agent, cur, prev, q, batch: int = 10):
    import numpy as np

    from laya.atari_train import action_probs

    return np.concatenate([action_probs(agent, cur[i:i + batch], q, None if prev is None else prev[i:i + batch])
                           for i in range(0, len(cur), batch)])


@app.function(image=image, gpu="A100", cpu=24, memory=32768, timeout=60 * 60,
              retries=modal.Retries(max_retries=3),
              volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol.read_only()})
def benchmark(model: str = DEFAULT_MODEL, game: str = "Breakout", n_prep: int = 64, play_steps: int = 250,
              train_steps: int = 40, batch_size: int = 32, num_workers: int = 22):
    """Preprocessing ms/frame, play decisions/s and training steps/s for the three configurations.

    Preprocessing is split into the part that blocks the caller on the CPU (building one decision's inputs) and
    the part that runs on the GPU. Play uses ``laya.atari_train.play`` on a real game, four episodes in lockstep,
    so decisions/s is the number the rollout loop actually gets. Training steps/s runs the real loop on real
    frames with the same loader settings as ``train_atari``.
    """
    import numpy as np
    import torch

    from laya.atari_train import encode_frames, game_actions, load_atari, model_policy, play
    from laya.games import atari_question
    from laya.preprocess import FrameFeatureCache, ImagePrep
    from laya.vlm import VLMAgent, vlm_prefix
    from laya.vlm_train import train

    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| cpus", os.cpu_count())
    data_vol.reload()
    ckpt_vol.reload()
    prev, cur, decode_ms = _frames(game, n_prep)
    actions = game_actions(game)
    q = atari_question(game, actions)["action"]
    rows, path = [], os.path.join(CKPT_ROOT, model)
    print("PNG decode from the dataset: %.2f ms/frame (%d frames)" % (decode_ms, 2 * len(cur)))

    def timed(fn, n=None, warmup=2):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = n or 20
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1000

    for label, size, backend in CONFIGS:
        agent = VLMAgent(path, device="cuda", dtype="bf16", image_size=size, preprocess=backend)
        for n_frames in (1, 2):
            imgs = [cur[0]] if n_frames == 1 else [prev[0], cur[0]]
            row = {"config": label, "image_size": size, "preprocess": backend, "frames": n_frames,
                   "image_tokens": agent.prep.image_seq_len * n_frames}
            # CPU-side cost of one decision's image inputs, and the device-side resize
            row["prep_cpu_ms_per_decision"] = round(timed(lambda: vlm_prefix(agent.processor, imgs, agent.prep)), 4)
            row["prep_cpu_ms_per_frame"] = round(row["prep_cpu_ms_per_decision"] / n_frames, 4)
            if backend == "gpu":
                row["prep_gpu_ms_per_frame"] = round(
                    timed(lambda: agent.prep.pixel_values(imgs * 16, device="cuda")) / (16 * n_frames), 4)
            for cached in ((False, True) if n_frames == 2 else (False,)):
                pol = model_policy(agent, game, actions, frames=n_frames, cache_features=cached)
                t0 = time.perf_counter()
                res = play(game, pol, episodes=4, max_steps=play_steps, seed=99)
                dt = time.perf_counter() - t0
                key = "decisions_per_s_cached" if cached else "decisions_per_s"
                row[key] = round(sum(res["steps"]) / dt, 2)
                print("  %-14s frames=%d %-8s %6.1f decisions/s (%d decisions in %.1fs)"
                      % (label, n_frames, "cached" if cached else "", row[key], sum(res["steps"]), dt))
            rows.append(row)
        del agent
        torch.cuda.empty_cache()

    # --- training steps/s on real data --------------------------------------------------------------------
    for label, size, backend in CONFIGS:
        for n_frames in (1, 2):
            data = load_atari(ATARI_ROOT, ["expert2f"], [game], n_calib=0, val_limit=1, train_limit=2000,
                              frames=n_frames)
            agent = VLMAgent(path, device="cuda", image_size=size, preprocess=backend)
            stats = {}
            train(agent.model, agent.processor, [dict(e, dataset=e["game"]) for e in data["train"]],
                  steps=train_steps, batch_size=batch_size, freeze="full", device="cuda", log_every=0,
                  num_workers=num_workers, stats=stats)
            row = next(r for r in rows if r["config"] == label and r["frames"] == n_frames)
            row["train_steps_per_s"] = round(stats["steps_per_s"], 3)
            row["train_data_wait_frac"] = round(stats["data_wait_frac"], 4)
            print("  %-14s frames=%d train %.3f steps/s (batch %d, loader wait %.0f%%)"
                  % (label, n_frames, stats["steps_per_s"], batch_size, 100 * stats["data_wait_frac"]))
            del agent
            torch.cuda.empty_cache()

    print("\n%-14s %6s %7s %9s %9s %9s %9s %9s" % ("config", "frames", "img tok", "prep cpu", "prep gpu",
                                                   "dec/s", "dec/s cch", "steps/s"))
    for r in rows:
        print("%-14s %6d %7d %9.3f %9s %9.1f %9s %9.3f" % (
            r["config"], r["frames"], r["image_tokens"], r["prep_cpu_ms_per_frame"],
            "%.4f" % r["prep_gpu_ms_per_frame"] if "prep_gpu_ms_per_frame" in r else "-",
            r["decisions_per_s"], "%.1f" % r["decisions_per_s_cached"] if "decisions_per_s_cached" in r else "-",
            r.get("train_steps_per_s", float("nan"))))
    out = {"model": model, "game": game, "gpu": torch.cuda.get_device_name(0), "batch_size": batch_size,
           "png_decode_ms_per_frame": round(decode_ms, 3), "rows": rows}
    print(json.dumps(out))
    return out


@app.local_entrypoint()
def deltas(model: str = DEFAULT_MODEL, game: str = "Breakout", n: int = 50, out: str = ""):
    r = filter_deltas.remote(model, game, n)
    if out:
        with open(out, "w") as f:
            json.dump(r, f, indent=2)
        print("wrote", out)


@app.local_entrypoint()
def bench(model: str = DEFAULT_MODEL, game: str = "Breakout", play_steps: int = 250, train_steps: int = 40,
          batch_size: int = 32, out: str = ""):
    r = benchmark.remote(model, game, 64, play_steps, train_steps, batch_size)
    if out:
        with open(out, "w") as f:
            json.dump(r, f, indent=2)
        print("wrote", out)
