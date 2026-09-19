"""VQA datasets in the shared laya record format (experimental).

Layout, one directory per dataset under a root such as /data/vqa:

    <root>/<dataset>/images/<image_id>.jpg    RGB JPEG, long side <= 512 px (may be shared by several records)
    <root>/<dataset>/<split>.jsonl            one record per line (splits: train, val)
    <root>/<dataset>/meta.json                source, counts, caps
    <root>/<dataset>/_READY                   written last; the dataset is complete once it exists

Record:
    {"id": str, "image": "images/<image_id>.jpg", "state_text": str | null,
     "question": {"type": "choice" | "score" | "noul", "instructions": str, "criteria": dict | list | null},
     "label": int}

`question` is the public laya question schema (as passed to Agent.predict). `label` indexes the rendered options:
the criteria order for choice/score, and 0 = false, 1 = true for noul.
"""
import hashlib
import io
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional, Tuple

MAX_SIDE = 512

README = __doc__ + """
Datasets
--------
aokvqa       HuggingFaceM4/A-OKVQA, multiple-choice options -> choice. train = official train, val = official
             validation (the test split has no labels).
scienceqa    derek-thomas/ScienceQA, questions with an image only -> choice. state_text = the hint/context when
             non-empty. train = official train, val = official validation.
vqav2_yesno  lmms-lab-encoder/VQAv2 (formerly lmms-lab/VQAv2), answer_type == "yes/no" -> noul (yes = 1). Only
             the official *validation* split has answers, so it is re-split by image_id (md5 bucket, ~10% val);
             train and val share no images. Do not evaluate on official VQAv2 val with a model trained on this.

Reading
-------
    import json
    recs = [json.loads(l) for l in open("/data/vqa/aokvqa/train.jsonl")]
    img = PIL.Image.open("/data/vqa/aokvqa/" + recs[0]["image"])
    agent.predict({"image": img, "context": recs[0]["state_text"]} if recs[0]["state_text"] else img,
                  {"q": recs[0]["question"]})

On Modal, reload the volume (`vol.reload()`) after `_READY` appears to see the files.
"""


# ---- sources: yield (record_without_image_path, image_id, image_bytes, split) -----------------------------------


def _stream(name: str, split: str):
    import datasets

    ds = datasets.load_dataset(name, split=split, streaming=True)
    return ds.cast_column("image", datasets.Image(decode=False))


def _choice(instructions: str, choices: List[str]) -> Optional[Dict]:
    choices = [str(c) for c in choices]
    if len(choices) < 2 or len(set(choices)) != len(choices):
        return None  # choice criteria become dict keys; duplicates would shift label indices
    return {"type": "choice", "instructions": instructions, "criteria": choices}


def aokvqa_source() -> Iterator[Tuple[Dict, str, bytes, str]]:
    for split, out in (("train", "train"), ("validation", "val")):
        for ex in _stream("HuggingFaceM4/A-OKVQA", split):
            q = _choice(ex["question"], ex["choices"])
            if q is None or ex["correct_choice_idx"] is None:
                continue
            rec = {"id": ex["question_id"], "state_text": None, "question": q, "label": int(ex["correct_choice_idx"])}
            yield rec, ex["question_id"], ex["image"]["bytes"], out


def scienceqa_source() -> Iterator[Tuple[Dict, str, bytes, str]]:
    for split, out in (("train", "train"), ("validation", "val")):
        for i, ex in enumerate(_stream("derek-thomas/ScienceQA", split)):
            if ex["image"] is None or not ex["image"].get("bytes"):
                continue
            q = _choice(ex["question"], ex["choices"])
            if q is None:
                continue
            rid = "%s-%d" % (out, i)
            rec = {"id": rid, "state_text": (ex.get("hint") or "").strip() or None, "question": q, "label": int(ex["answer"])}
            yield rec, rid, ex["image"]["bytes"], out


def _vqav2_split(image_id: int, val_pct: int = 10) -> str:
    return "val" if int(hashlib.md5(str(image_id).encode()).hexdigest(), 16) % 100 < val_pct else "train"


def vqav2_yesno_source() -> Iterator[Tuple[Dict, str, bytes, str]]:
    for ex in _stream("lmms-lab-encoder/VQAv2", "validation"):
        if ex["answer_type"] != "yes/no" or ex["multiple_choice_answer"] not in ("yes", "no"):
            continue
        q = {"type": "noul", "instructions": ex["question"], "criteria": None}
        rec = {"id": str(ex["question_id"]), "state_text": None, "question": q, "label": int(ex["multiple_choice_answer"] == "yes")}
        yield rec, str(ex["image_id"]), ex["image"]["bytes"], _vqav2_split(ex["image_id"])


SOURCES = {
    "aokvqa": ("HuggingFaceM4/A-OKVQA", aokvqa_source),
    "scienceqa": ("derek-thomas/ScienceQA", scienceqa_source),
    "vqav2_yesno": ("lmms-lab-encoder/VQAv2", vqav2_yesno_source),
}


# ---- writing ------------------------------------------------------------------------------------------------------


def reencode_jpeg(data: bytes, max_side: int = MAX_SIDE) -> bytes:
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((max_side, max_side), Image.BICUBIC)
    out = io.BytesIO()
    img.save(out, "JPEG", quality=90)
    return out.getvalue()


def build_dataset(name: str, root: str, caps: Dict[str, int], workers: int = 8, log=print) -> Dict:
    """Stream `name` from the Hub into <root>/.tmp-<name>, then rename it to <root>/<name>. Does not write _READY."""
    hub_id, source = SOURCES[name]
    tmp = os.path.join(root, ".tmp-" + name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(os.path.join(tmp, "images"))
    files = {s: open(os.path.join(tmp, "%s.jsonl" % s), "w") for s in caps}
    counts = {s: 0 for s in caps}
    labels = {s: {} for s in caps}
    written, skipped_bad = set(), 0
    t0 = time.time()

    def save(image_id, data):
        path = os.path.join(tmp, "images", image_id + ".jpg")
        try:
            with open(path, "wb") as f:
                f.write(reencode_jpeg(data))
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(workers) as pool:
        pending = []
        for rec, image_id, data, split in source():
            if split not in caps or counts[split] >= caps[split]:
                if all(counts[s] >= caps[s] for s in caps):
                    break
                continue
            if image_id not in written:
                written.add(image_id)
                pending.append((rec, split, pool.submit(save, image_id, data)))
            else:
                pending.append((rec, split, None))
            rec["image"] = "images/%s.jpg" % image_id
            counts[split] += 1
            if len(pending) >= 256:
                skipped_bad += _flush(pending, files, labels, counts)
            if sum(counts.values()) % 5000 == 0:
                log("%s: %s (%.0fs)" % (name, counts, time.time() - t0))
        skipped_bad += _flush(pending, files, labels, counts)
    for f in files.values():
        f.close()

    meta = {
        "dataset": name,
        "source": hub_id,
        "counts": counts,
        "label_counts": labels,
        "images": len(written),
        "caps": caps,
        "max_side": MAX_SIDE,
        "skipped_undecodable": skipped_bad,
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    final = os.path.join(root, name)
    if os.path.exists(final):
        old = os.path.join(root, ".old-%s-%d" % (name, int(time.time())))
        os.rename(final, old)
        shutil.rmtree(old, ignore_errors=True)
    os.rename(tmp, final)
    log("%s: done %s" % (name, meta))
    return meta


def _flush(pending, files, labels, counts) -> int:
    """Write records whose image saved successfully; drop the rest (their image failed to decode)."""
    bad_images = set()
    bad = 0
    for rec, split, fut in pending:
        if fut is not None and not fut.result():
            bad_images.add(rec["image"])
        if rec["image"] in bad_images:
            counts[split] -= 1
            bad += 1
            continue
        files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
        labels[split][rec["label"]] = labels[split].get(rec["label"], 0) + 1
    pending.clear()
    return bad


def mark_ready(root: str, name: str, meta: Dict):
    with open(os.path.join(root, name, "_READY"), "w") as f:
        json.dump({"ready_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **meta}, f, indent=2)


# ---- reading ------------------------------------------------------------------------------------------------------


def read_records(root: str, name: str, split: str, limit: Optional[int] = None) -> List[Dict]:
    """Records of <root>/<name>/<split>.jsonl with "image" made absolute and "dataset" added."""
    base = os.path.join(root, name)
    out = []
    with open(os.path.join(base, "%s.jsonl" % split)) as f:
        for line in f:
            rec = json.loads(line)
            rec["image"] = os.path.join(base, rec["image"])
            rec["dataset"] = name
            out.append(rec)
            if limit and len(out) >= limit:
                break
    return out
