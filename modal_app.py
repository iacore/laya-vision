"""Modal jobs for the SigLIP projector branch: shared VQA data prep, GPU tests and stage-1 alignment.

    modal run --detach modal_app.py::prepare                      # CPU: /data/vqa/{aokvqa,scienceqa,vqav2_yesno}
    modal run modal_app.py::test                                  # pytest on an L4 (CUDA + bf16 autocast)
    modal run --detach modal_app.py::train --minutes 20           # stage-1 on one A100 -> /ckpt/siglip-projector/<run>
                                                                  # (CE + proper_reward, proj + head + top-4 encoder layers)

Volumes (pre-created): laya-hf-cache at /cache/hf (HF_HOME), laya-datasets at /data, laya-checkpoints at /ckpt
(this branch writes only under /ckpt/siglip-projector/).
"""
import os
import subprocess
import time
from pathlib import Path

import modal

ROOT = Path(__file__).parent
DATA_ROOT = "/data/vqa"
CKPT_ROOT = "/ckpt/siglip-projector"
DATASETS = ("aokvqa", "scienceqa", "vqav2_yesno")

app = modal.App("laya-siglip-projector")
hf_vol = modal.Volume.from_name("laya-hf-cache")
data_vol = modal.Volume.from_name("laya-datasets")
ckpt_vol = modal.Volume.from_name("laya-checkpoints")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.0",
        "transformers==5.17.0",
        "safetensors",
        "huggingface_hub",
        "numpy",
        "pillow",
        "datasets",
        "pytest",
    )
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/root"})
    .workdir("/root")
    .add_local_dir(ROOT / "laya", "/root/laya")
    .add_local_dir(ROOT / "tests", "/root/tests")
)


# ---- data ---------------------------------------------------------------------------------------------------------


@app.function(image=image, cpu=8, memory=16384, timeout=3 * 3600, volumes={"/data": data_vol})
def prepare_data(dataset: str, train_cap: int = 50_000, val_cap: int = 5_000) -> dict:
    from laya.vision_data import README, build_dataset, mark_ready

    os.makedirs(DATA_ROOT, exist_ok=True)
    with open(os.path.join(DATA_ROOT, "README.md"), "w") as f:
        f.write(README)
    meta = build_dataset(dataset, DATA_ROOT, {"train": train_cap, "val": val_cap}, log=lambda m: print(m, flush=True))
    data_vol.commit()  # data first, then _READY, so a reader that sees _READY also sees the files
    mark_ready(DATA_ROOT, dataset, meta)
    data_vol.commit()
    print("READY", dataset, flush=True)
    return meta


@app.function(image=image, cpu=2, timeout=900)
def probe_source(dataset: str, n: int = 3) -> list:
    """First n converted records of a source (no writes), to check field mappings."""
    from laya.vision_data import SOURCES, reencode_jpeg

    out = []
    for rec, image_id, data, split in SOURCES[dataset][1]():
        out.append({**rec, "image_id": image_id, "split": split, "jpeg_bytes": len(reencode_jpeg(data))})
        if len(out) >= n:
            break
    return out


@app.local_entrypoint()
def probe(datasets: str = ",".join(DATASETS)):
    names = datasets.split(",")
    for name, recs in zip(names, probe_source.map(names, return_exceptions=True)):
        print("==", name)
        for r in recs if isinstance(recs, list) else [recs]:
            print(r)


@app.local_entrypoint()
def prepare(datasets: str = ",".join(DATASETS), train_cap: int = 50_000, val_cap: int = 5_000):
    names = [d for d in datasets.split(",") if d]
    for name, meta in zip(names, prepare_data.map(names, kwargs={"train_cap": train_cap, "val_cap": val_cap}, return_exceptions=True)):
        print(name, meta)


# ---- tests --------------------------------------------------------------------------------------------------------


@app.function(image=image, gpu="L4", cpu=4, memory=32768, timeout=3600, volumes={"/cache/hf": hf_vol})
def run_tests(pytest_args: str = "") -> tuple:
    import torch

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
    cmd = ["python", "-m", "pytest", "tests/test_vision_projector.py", "-v", "-p", "no:cacheprovider", "-rA"] + pytest_args.split()
    proc = subprocess.run(cmd, env={**os.environ, "LAYA_TEST_DEVICE": "cuda"}, capture_output=True, text=True)
    hf_vol.commit()
    return proc.returncode, proc.stdout[-20000:], proc.stderr[-5000:]


@app.local_entrypoint()
def test(pytest_args: str = ""):
    code, out, err = run_tests.remote(pytest_args)
    print(out)
    if code:
        print(err)
        raise SystemExit(code)


# ---- stage-1 training ---------------------------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="A100",
    cpu=16,
    memory=32768,
    timeout=50 * 60,
    volumes={"/cache/hf": hf_vol, "/data": data_vol, "/ckpt": ckpt_vol},
)
def train_stage1(
    run_name: str,
    minutes: float = 20,
    datasets: str = ",".join(DATASETS),
    batch_size: int = 32,
    max_steps: int = 4000,
    lr_proj: float = 5e-4,
    lr_head: float = 1e-4,
    lr_top: float = 2e-5,
    top_layers: int = 4,
    freeze_head: bool = False,
    loss: str = "direct",
    ce_weight: float = 1.0,
    warmup: int = 200,
    calib_per_dataset: int = 300,
    val_limits: str = "vqav2_yesno=1000",
    eval_every: int = 500,
    early_stop_step: int = 1000,
) -> list:
    """Stage-1 run. Calibration: the last `calib_per_dataset` train records of each dataset (file order) are held
    out of training and used to fit per-dataset temperatures at every eval. Periodic evals use full val except
    `val_limits` (first N in file order); the final eval uses full val everywhere."""
    import torch

    import laya
    from laya.vision_data import read_records
    from laya.vision_train import train

    data_vol.reload()
    names = datasets.split(",")
    limits = {k: int(v) for k, v in (kv.split("=") for kv in val_limits.split(",") if kv)}
    for n in names:
        if not os.path.exists(os.path.join(DATA_ROOT, n, "_READY")):
            raise RuntimeError("dataset %s is not ready in %s" % (n, DATA_ROOT))
    train_recs, calib_recs, val_recs, final_val = [], [], [], []
    for n in names:
        recs = read_records(DATA_ROOT, n, "train")
        train_recs += recs[:-calib_per_dataset]
        calib_recs += recs[-calib_per_dataset:]
        full = read_records(DATA_ROOT, n, "val")
        final_val += full
        val_recs += full[: limits.get(n, len(full))]
    print("train", len(train_recs), "calib", len(calib_recs), "val", len(val_recs), "final_val", len(final_val), flush=True)

    torch.manual_seed(0)
    agent = laya.load("convaiinnovations/laya", device="cuda", vision_encoder=laya.vision.DEFAULT_VISION_ENCODER)
    hf_vol.commit()
    out = os.path.join(CKPT_ROOT, run_name)
    history = train(
        agent,
        train_recs,
        val_recs,
        steps=max_steps,
        minutes=minutes,
        batch_size=batch_size,
        lr_proj=lr_proj,
        lr_head=lr_head,
        lr_top=lr_top,
        train_head=not freeze_head,
        top_layers=top_layers,
        loss=loss,
        ce_weight=ce_weight,
        warmup=warmup,
        eval_every=eval_every,
        calib_records=calib_recs,
        final_val_records=final_val,
        early_stop_step=early_stop_step,
        workers=14,
        threads=8,
        amp_dtype=torch.bfloat16,
        out=out,
        log=lambda m: print(m, flush=True),
    )
    ckpt_vol.commit()
    return history


@app.local_entrypoint()
def train(
    minutes: float = 20,
    run_name: str = "",
    datasets: str = ",".join(DATASETS),
    batch_size: int = 32,
    max_steps: int = 4000,
    lr_proj: float = 5e-4,
    lr_head: float = 1e-4,
    lr_top: float = 2e-5,
    top_layers: int = 4,
    freeze_head: bool = False,
    loss: str = "direct",
    ce_weight: float = 1.0,
    eval_every: int = 500,
    early_stop_step: int = 1000,
):
    run_name = run_name or time.strftime("stage1-%Y%m%d-%H%M%S")
    history = train_stage1.remote(
        run_name, minutes=minutes, datasets=datasets, batch_size=batch_size, max_steps=max_steps, lr_proj=lr_proj,
        lr_head=lr_head, lr_top=lr_top, top_layers=top_layers, freeze_head=freeze_head, loss=loss,
        ce_weight=ce_weight, eval_every=eval_every, early_stop_step=early_stop_step,
    )
    for h in history:
        if "eval" in h or "final_eval" in h:
            print(h["step"], h.get("eval") or h.get("final_eval"))
