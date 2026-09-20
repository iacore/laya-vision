---
license: cc-by-nc-sa-4.0
base_model: HuggingFaceTB/SmolVLM-256M-Instruct
library_name: laya
pipeline_tag: visual-question-answering
datasets:
- HuggingFaceM4/A-OKVQA
- derek-thomas/ScienceQA
- lmms-lab-encoder/VQAv2
language:
- en
tags:
- laya
- calibration
- decision-model
- vision
- smolvlm
---

# Laya Vision (SmolVLM-256M)

This model makes calibrated, typed decisions about an **image plus optional text**. It answers `choice`, `score` and `noul` (yes/no probability) questions in one forward pass, with no text generation.

It adds image input to [Laya](https://github.com/NandhaKishorM/laya) by replacing Laya's ModernBERT encoder with [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct). Laya's `predict(state, questions)` API, proper-scoring-rule training and temperature calibration are unchanged.

- **Code:** [github.com/r33drichards/laya-vision](https://github.com/r33drichards/laya-vision)
- **Demo:** [thaitea/laya-vision-demo](https://huggingface.co/spaces/thaitea/laya-vision-demo)
- **Status:** experimental. This is an independent research fork, not affiliated with Convai Innovations, the authors of Laya.

## Usage

```bash
git clone https://github.com/r33drichards/laya-vision && pip install -e ./laya-vision torchvision
```

```python
import laya
from PIL import Image

agent = laya.load_vlm("thaitea/laya-vision-smolvlm-256m")
result = agent.predict(
    {"image": Image.open("photo.jpg"), "note": "customer says it arrived broken"},
    {
        "damaged":  {"type": "noul",   "instructions": "Does the item in the photo look damaged?"},
        "category": {"type": "choice", "instructions": "What kind of item is this?",
                     "criteria": ["electronics", "clothing", "furniture", "food", "other"]},
    },
)
result["answers"]["damaged"]["noul"]        # calibrated P(true)
result["answers"]["category"]["choice"]     # top option; see ["probabilities"], ["confidence"]
```

## Results

Scores are on the full validation splits. "Calibrated" uses the per-type temperatures stored in `vlm_agent_config.json`, which are applied automatically.

| Dataset | Question type | Chance | Accuracy | ECE raw | ECE calibrated |
|---|---|---|---|---|---|
| A-OKVQA (n=1,138) | 4-way `choice` | 25% | 63.1% | 0.266 | 0.094 |
| ScienceQA, image subset (n=2,097) | 2–5-way `choice` | ~36% | 89.0% | 0.080 | 0.034 |
| VQAv2 yes/no (n=5,000)\* | `noul` | 50% | 73.2% | 0.085 | 0.042 |
| **All (n=8,235)** | | | **75.9%** | 0.108 | **0.035** |

\* The VQAv2 train/val split is a re-split of the official VQAv2 *validation* set by image (the only official split with answers), so these numbers are not comparable to published VQAv2 results.

- **Latency:** about 71 ms for one image question on an NVIDIA L4 (bf16). The image is encoded once per `predict` call and shared by every question.
- **Option order:** across 4 rotations of the A-OKVQA option order, accuracy varies by 1.2 points (63.1 / 61.9 / 62.1 / 61.9). Averaging over orders (`n_permutations=4`) doesn't help.

## Training

- **Backbone:** SmolVLM-256M-Instruct. The vision tower is frozen, and the language model and decision head are trained, about 150M parameters.
- **How options are read:** the question and state come first and the options are listed last. Each option is scored from the hidden state at the end of its own line.
- **Loss:** RLCD \u2014 a policy gradient on strictly proper scoring rules (log + 0.5 \u00d7 spherical, minus the ranked probability score on ordinal questions), with no cross-entropy term. Eight noisy copies of the logits are scored per step and exploration noise decays from 1.0 to 0.3. The option order is shuffled at random.
- **Data:**
  - A-OKVQA: 17k questions, `choice`
  - ScienceQA image subset: 6k questions, `choice`
  - VQAv2 yes/no: 50k questions, `noul`
  - Datasets were sampled equally — about 72k samples drawn from each — so the smaller sets repeat more: ScienceQA made 12.2 passes, A-OKVQA 4.3 and VQAv2 1.5. No per-dataset cap was applied.
- **Schedule:** 6,783 steps at batch 32 on one A100 (27.8 minutes training plus 6.8 evaluating). Head LR 1.4e-4, backbone LR 2.8e-5, 3% warmup, then cosine decay. The checkpoint with the best mean validation accuracy was kept — step 5,650, epoch 2.5; the third epoch made validation accuracy worse.
- **Calibration:** per-type temperatures fitted on the last 300 training records of each dataset, which were held out of training: choice 3.15, noul 1.56.
- `training_metrics.json` has every evaluation from the run.

## Limitations

- **`score` questions are untrained.** There was no ordinal image data, so their outputs are meaningless.
- **A-OKVQA overfits.** Training accuracy reaches 94.7% at the kept checkpoint against 63.1% on validation, and raw confidence is too high there, so rely on the calibrated outputs.
- **ScienceQA repeats most.** At 12.2 passes its training accuracy reaches 99.3%, so treat its validation number as the most optimistic of the three.
- **Narrow domain.** The model was trained on everyday photos (COCO) and science diagrams. Expect to fine-tune on your own data for other domains.
- **One image per state.** Each image is resized to a single 512 px tile.
- **Text-only states work but are unevaluated.** This model is not a drop-in replacement for Laya's text checkpoint.
- **The `action` field is not trained.** Every answer carries `result["answers"][qid]["action"]["act_probability"]` from the act/escalate head. That head enters the loss only as `0.0 * act.sum()`, so it receives exactly zero gradient and reports its random initialisation. The value still varies with the input — the confidence features feeding it are real, the weights reading them are not — so it looks like a calibrated gate and is not one. Ignore it; do not gate on it.

## License

The weights are released under **CC BY-NC-SA 4.0**, because they were trained partly on ScienceQA, which uses that license. The base model, SmolVLM, is Apache 2.0. A-OKVQA is Apache 2.0. VQAv2 annotations are CC BY 4.0, and its COCO images carry Flickr terms. The code is Apache 2.0.
