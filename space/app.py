"""Gradio demo for Laya Vision: calibrated typed decisions (choice / yes-no) about an image."""
import json
import os
import time
import urllib.request

import gradio as gr
import torch

import laya

MODEL_ID = os.environ.get("LAYA_MODEL", "thaitea/laya-vision-smolvlm-256m")
torch.set_num_threads(max(1, os.cpu_count() or 1))
agent = laya.load_vlm(MODEL_ID, device="cpu")

EXAMPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg"
EXAMPLE_PATH = "example_cat.jpg"
try:
    if not os.path.exists(EXAMPLE_PATH):
        req = urllib.request.Request(EXAMPLE_URL, headers={"User-Agent": "laya-vision-demo/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r, open(EXAMPLE_PATH, "wb") as f:
            f.write(r.read())
except Exception:
    EXAMPLE_PATH = None

DEFAULT_JSON = json.dumps(
    {
        "animal": {"type": "choice", "instructions": "What animal is in the photo?",
                   "criteria": ["cat", "dog", "bird", "horse", "none"]},
        "outdoors": {"type": "noul", "instructions": "Was this photo taken outdoors?"},
        "person": {"type": "noul", "instructions": "Is there a person in the image?"},
    },
    indent=2,
)


def quick_questions(yes_no: str, mc_question: str, mc_options: str) -> dict:
    qs = {}
    for i, line in enumerate(l.strip() for l in (yes_no or "").splitlines()):
        if line:
            qs["yes/no %d" % (i + 1)] = {"type": "noul", "instructions": line}
    opts = [o.strip() for o in (mc_options or "").split(",") if o.strip()]
    if (mc_question or "").strip():
        if len(opts) < 2:
            raise gr.Error("A multiple-choice question needs at least two comma-separated options.")
        qs["multiple choice"] = {"type": "choice", "instructions": mc_question.strip(), "criteria": opts}
    return qs


def render(questions: dict, out: dict, ms: float) -> str:
    rows = ["| Question | Answer | Confidence |", "|---|---|---|"]
    for qid, a in out["answers"].items():
        ins = questions[qid]["instructions"]
        if a["type"] == "choice":
            probs = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])
            detail = ", ".join("%s %.0f%%" % (k, 100 * v) for k, v in probs)
            rows.append("| %s | **%s** (%s) | %.2f |" % (ins, a["choice"], detail, a["confidence"]))
        elif a["type"] == "noul":
            p = a["noul"]
            rows.append("| %s | **%s**, P(yes) = %.1f%% | %.2f |" % (ins, "yes" if p >= 0.5 else "no", 100 * p, a["confidence"]))
        else:
            rows.append("| %s | score %.2f (untrained type, ignore) | %.2f |" % (ins, a["score"], a["confidence"]))
    rows.append("\n_%.0f ms on CPU_" % ms)
    return "\n".join(rows)


def run(image, context, mode, yes_no, mc_question, mc_options, questions_json):
    if image is None:
        raise gr.Error("Upload an image first.")
    if mode == "JSON":
        try:
            questions = json.loads(questions_json)
        except json.JSONDecodeError as e:
            raise gr.Error("Invalid JSON: %s" % e)
    else:
        questions = quick_questions(yes_no, mc_question, mc_options)
    if not questions:
        raise gr.Error("Add at least one question.")
    state = {"image": image.convert("RGB")}
    if (context or "").strip():
        state["context"] = context.strip()
    t0 = time.time()
    try:
        out = agent.predict(state, questions)
    except (KeyError, TypeError, ValueError) as e:
        raise gr.Error("Bad question definition: %s" % e)
    return render(questions, out, (time.time() - t0) * 1000), out


with gr.Blocks(title="Laya Vision") as demo:
    gr.Markdown(
        "# Laya Vision\n"
        "Calibrated yes/no and multiple-choice decisions about an image, in one forward pass with no text generation. "
        "Model: [thaitea/laya-vision-smolvlm-256m](https://huggingface.co/thaitea/laya-vision-smolvlm-256m) · "
        "Code: [r33drichards/laya-vision](https://github.com/r33drichards/laya-vision)\n\n"
        "Experimental. The model was trained on everyday photos (COCO) and science diagrams. "
        "`score` questions are not trained yet. On this free CPU Space, expect 1–3 s per image."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Image")
            context = gr.Textbox(label="Optional text context", placeholder="e.g. customer says it arrived broken")
            mode = gr.Radio(["Quick", "JSON"], value="Quick", label="Question format")
            with gr.Group(visible=True) as quick_box:
                yes_no = gr.Textbox(label="Yes/no questions (one per line)", lines=3,
                                    value="Was this photo taken outdoors?\nIs there a person in the image?")
                mc_question = gr.Textbox(label="Multiple-choice question", value="What animal is in the photo?")
                mc_options = gr.Textbox(label="Options (comma-separated)", value="cat, dog, bird, horse, none")
            with gr.Group(visible=False) as json_box:
                questions_json = gr.Code(value=DEFAULT_JSON, language="json",
                                         label="Questions (laya predict schema: choice / noul)")
            btn = gr.Button("Ask", variant="primary")
        with gr.Column():
            table = gr.Markdown()
            raw = gr.JSON(label="Raw output")

    mode.change(lambda m: (gr.update(visible=m == "Quick"), gr.update(visible=m == "JSON")), mode, [quick_box, json_box])
    inputs = [image, context, mode, yes_no, mc_question, mc_options, questions_json]
    btn.click(run, inputs, [table, raw])
    if EXAMPLE_PATH:
        gr.Examples([[EXAMPLE_PATH]], inputs=[image], label="Example (photo: Alvesgaspar, CC BY-SA 3.0, Wikimedia Commons)")

demo.queue(max_size=16).launch(show_error=True)
