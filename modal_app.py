"""Modal jobs for the SmolVLM-backed decision model (``laya.vlm``).

    modal run modal_app.py::test                     # pytest on a GPU + latency
    modal run modal_app.py::finetune --minutes 18    # short fine-tune + held-out acc / ECE
    modal run modal_app.py::evaluate --run-name <run> # re-evaluate a saved checkpoint

Volumes (created out of band; never ``modal deploy`` this app):
    laya-hf-cache     -> /cache/hf   (HF_HOME, shared model weights)
    laya-datasets     -> /data       (read-only; /data/vqa/<name>/{<split>.jsonl, images/, _READY})
    laya-checkpoints  -> /ckpt       (this app writes only under /ckpt/smolvlm/)
"""
import json
import os
import subprocess
import sys
import time

import modal

app = modal.App("laya-smolvlm")

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
        "datasets",
        "pytest",
        "num2words",
    )
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("tests", "/root/tests")
    .add_local_python_source("laya")
)

BACKBONE = "HuggingFaceTB/SmolVLM-256M-Instruct"
DATASETS = ("aokvqa", "scienceqa", "vqav2_yesno")
CKPT_ROOT = "/ckpt/smolvlm"


@app.function(image=image, gpu="L4", timeout=30 * 60, volumes={"/cache/hf": hf_vol})
def test():
    """Run tests/test_vlm.py on the GPU, then time predict() in fp32 and bf16."""
    import torch

    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    rc = subprocess.run(
        [sys.executable, "-m", "pytest", "/root/tests/test_vlm.py", "-v", "-s", "-p", "no:cacheprovider", "-W", "ignore"],
        cwd="/root",
    ).returncode
    hf_vol.commit()

    from PIL import Image

    from laya.vlm import VLMAgent

    img = Image.new("RGB", (96, 96), (255, 255, 255))
    img.paste(Image.new("RGB", (48, 48), (220, 20, 20)), (24, 24))
    one = {"is_red": {"type": "noul", "instructions": "Is the square red?"}}
    three = dict(one, color={"type": "choice", "instructions": "What color?", "criteria": ["red", "blue", "green"]},
                 size={"type": "score", "instructions": "How big?", "criteria": ["small", "medium", "large"]})
    for dtype in ("fp32", "bf16"):
        agent = VLMAgent(backbone=BACKBONE, device="cuda", dtype=dtype)
        for label, state, qs in (("image, 1 q", {"image": img}, one), ("image, 3 q", {"image": img}, three),
                                 ("text, 1 q", "Customer: I was billed twice.", one), ("text, 3 q", "Customer: I was billed twice.", three)):
            for _ in range(3):
                agent.predict(state, qs)
            ts = []
            for _ in range(20):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                agent.predict(state, qs)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            ts.sort()
            print("latency %-5s %-11s median %6.1f ms  p90 %6.1f ms" % (dtype, label, ts[10], ts[18]))
        del agent
    if rc != 0:
        raise SystemExit("pytest failed with exit code %d" % rc)


def _load_split(name: str, split: str, limit):
    from laya.vlm_train import load_jsonl_examples

    return load_jsonl_examples("/data/vqa", name, split, limit=limit)


@app.function(
    image=image,
    gpu="A10G",
    cpu=16,
    memory=32768,
    timeout=40 * 60,
    volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol},
)
def finetune(
    datasets: str = ",".join(DATASETS),
    minutes: float = 18.0,
    freeze: str = "full",
    n_last: int = 8,
    batch_size: int = 16,
    lr_head: float = 1e-4,
    lr_backbone: float = 2e-5,
    train_split: str = "train",
    val_split: str = "val",
    max_train: int = 0,
    max_val: int = 0,
    n_calib: int = 300,
    eval_every: int = 400,
    val_caps: str = "",
    num_workers: int = 14,
    synthetic: bool = False,
    run_name: str = "",
):
    """Short fine-tune on the prepared VQA sets; logs loss and held-out accuracy / ECE, saves to /ckpt/smolvlm/<run>.

    ``max_train`` / ``max_val`` = 0 means the whole split; otherwise the first N records in file order.
    ``val_caps`` overrides the val cap per dataset, e.g. ``"vqav2_yesno=1000"``.
    """
    import random

    import torch

    from laya.vlm import VLMAgent
    from laya.vlm_train import collect_logits, fit_temperatures_from, format_metrics, metrics_from, synthetic_examples, train

    t_start = time.time()
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    run_name = run_name or time.strftime("run-%Y%m%d-%H%M%S")
    out_dir = os.path.join(CKPT_ROOT, run_name)

    caps = {k: int(v) for k, v in (kv.split("=") for kv in val_caps.split(",") if kv)}
    train_ex, calib_ex, val_ex = [], [], []
    if synthetic:
        for ex in synthetic_examples(64, seed=0):
            train_ex.append(dict(ex, dataset="synthetic"))
        calib_ex = [dict(ex, dataset="synthetic") for ex in synthetic_examples(16, seed=2)]
        val_ex = [dict(ex, dataset="synthetic") for ex in synthetic_examples(32, seed=1)]
    else:
        data_vol.reload()
        for name in [d for d in datasets.split(",") if d]:
            if not os.path.exists("/data/vqa/%s/_READY" % name):
                print("dataset %s not ready (no _READY); skipping" % name)
                continue
            tr = _load_split(name, train_split, max_train + n_calib if max_train else 0)
            random.Random(0).shuffle(tr)
            calib_ex += tr[:n_calib]
            train_ex += tr[n_calib:]
            va = _load_split(name, val_split, caps.get(name, max_val) or 0)
            val_ex += va
            print("dataset %s: %d train, %d calib, %d val" % (name, len(tr) - n_calib, min(n_calib, len(tr)), len(va)))
    if not train_ex:
        raise SystemExit("no training data (datasets not ready and --synthetic not set)")

    agent = VLMAgent(backbone=BACKBONE, device="cuda")
    hf_vol.commit()
    model, proc = agent.model, agent.processor
    ev_kw = dict(batch_size=32, num_workers=num_workers)
    small_val = []
    for name in sorted({ex["dataset"] for ex in val_ex}):
        small_val += [ex for ex in val_ex if ex["dataset"] == name][:200]

    log = {"run": run_name, "args": dict(datasets=datasets, minutes=minutes, freeze=freeze, n_last=n_last, batch_size=batch_size,
                                         lr_head=lr_head, lr_backbone=lr_backbone, max_train=max_train, max_val=max_val, val_caps=caps),
           "evals": []}
    base = metrics_from(collect_logits(model, proc, small_val, **ev_kw))
    print("[eval step 0, untrained head] " + format_metrics(base), flush=True)
    log["evals"].append({"step": 0, **base})

    def eval_fn(step):
        m = metrics_from(collect_logits(model, proc, small_val, **ev_kw))
        print("[eval step %d] %s" % (step, format_metrics(m)), flush=True)
        log["evals"].append({"step": step, **m})

    losses = train(
        model, proc, train_ex, steps=10**9, batch_size=batch_size, freeze=freeze, n_last=n_last,
        lr_head=lr_head, lr_backbone=lr_backbone, device="cuda", log_every=50, max_minutes=minutes,
        num_workers=num_workers, warmup=100, eval_fn=eval_fn, eval_every=eval_every,
    )
    log["steps"], log["examples_seen"] = len(losses), len(losses) * batch_size
    log["loss_first50"], log["loss_last50"] = sum(losses[:50]) / min(50, len(losses)), sum(losses[-50:]) / min(50, len(losses))
    print("trained %d steps (%d examples); loss first50 %.4f -> last50 %.4f"
          % (log["steps"], log["examples_seen"], log["loss_first50"], log["loss_last50"]), flush=True)

    t_eval = time.time()
    temps = fit_temperatures_from(collect_logits(model, proc, calib_ex, **ev_kw))
    val_records = collect_logits(model, proc, val_ex, **ev_kw)
    print("calib + val eval: %d examples in %.1f min" % (len(calib_ex) + len(val_ex), (time.time() - t_eval) / 60))
    log["temperature"] = temps
    log["val_raw"] = metrics_from(val_records)
    log["val_calibrated"] = metrics_from(val_records, temps)
    print("temperatures (choice, score, noul):", [round(t, 3) for t in temps])
    print("[final val, T=1]        " + format_metrics(log["val_raw"]))
    print("[final val, calibrated] " + format_metrics(log["val_calibrated"]))

    agent.temperature = temps
    agent.save(out_dir, include_backbone=freeze != "head")
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(log, f, indent=2)
    ckpt_vol.commit()
    print("saved %s (%.1f min total)" % (out_dir, (time.time() - t_start) / 60))
    return {k: log[k] for k in ("run", "steps", "loss_first50", "loss_last50", "temperature", "val_raw", "val_calibrated")}


@app.function(
    image=image,
    gpu="A10G",
    cpu=16,
    memory=32768,
    timeout=30 * 60,
    volumes={"/cache/hf": hf_vol, "/data": data_vol.read_only(), "/ckpt": ckpt_vol.read_only()},
)
def evaluate(run_name: str, datasets: str = ",".join(DATASETS), val_split: str = "val", max_val: int = 0):
    """Evaluate a saved checkpoint (/ckpt/smolvlm/<run_name>) on the val splits, raw and with its temperatures."""
    import torch

    from laya.vlm import VLMAgent
    from laya.vlm_train import collect_logits, format_metrics, metrics_from

    print("GPU:", torch.cuda.get_device_name(0))
    data_vol.reload()
    val_ex = []
    for name in [d for d in datasets.split(",") if d]:
        if not os.path.exists("/data/vqa/%s/_READY" % name):
            print("dataset %s not ready (no _READY); skipping" % name)
            continue
        va = _load_split(name, val_split, max_val or 0)
        print("dataset %s: %d val" % (name, len(va)))
        val_ex += va
    agent = VLMAgent(os.path.join(CKPT_ROOT, run_name), device="cuda")
    records = collect_logits(agent.model, agent.processor, val_ex, batch_size=32, num_workers=14)
    raw, cal = metrics_from(records), metrics_from(records, agent.temperature)
    print("temperatures (choice, score, noul):", [round(t, 3) for t in agent.temperature])
    print("[val, T=1]        " + format_metrics(raw))
    print("[val, calibrated] " + format_metrics(cal))
    return {"val_raw": raw, "val_calibrated": cal}
