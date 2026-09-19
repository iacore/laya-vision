"""Image inputs (experimental): SigLIP vision tower, image state handling and stage-1 freezing helpers.

An image is a kind of state. Pass either an image directly or a dict with one image-valued key:

    agent.predict(laya.Image("photo.jpg"), questions)
    agent.predict({"image": pil_image, "caption": "shelf 3"}, questions)

Only `laya.Image` and `PIL.Image.Image` values count as images; plain strings are always text, so existing
text states (e.g. {"image": "a photo of a cat"}) keep their meaning.
"""
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

DEFAULT_VISION_ENCODER = "google/siglip2-base-patch16-224"
DEFAULT_N_IMAGE_TOKENS = 64
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


class Image:
    """Marks a value as an image. `src` is a file path, PIL image, uint8 HxWx3 array, or raw bytes."""

    def __init__(self, src: Any):
        self.src = src

    def load(self):
        return load_image(self.src)

    def __repr__(self):
        return "laya.Image(%r)" % (self.src if isinstance(self.src, (str, os.PathLike)) else type(self.src).__name__)


def _is_image(v: Any) -> bool:
    if isinstance(v, Image):
        return True
    pil = sys.modules.get("PIL.Image")
    return pil is not None and isinstance(v, pil.Image)


def split_image_state(state: Any) -> Tuple[Optional[Any], Any]:
    """Split state into (image, text_state). Text-only states are returned unchanged as (None, state).

    Supported: an image itself (text_state is None) or a dict with exactly one image-valued key (the remaining
    keys are the text state, or None when nothing remains).
    """
    if _is_image(state):
        return state, None
    if isinstance(state, dict):
        keys = [k for k, v in state.items() if _is_image(v)]
        if not keys:
            return None, state
        if len(keys) > 1:
            raise ValueError("only one image per state is supported, got image keys %r" % keys)
        rest = {k: v for k, v in state.items() if k != keys[0]}
        return state[keys[0]], (rest or None)
    return None, state


def load_image(src: Any):
    """Load an image-like value into an RGB PIL image."""
    from PIL import Image as PILImage

    if isinstance(src, Image):
        src = src.src
    if isinstance(src, PILImage.Image):
        img = src
    elif isinstance(src, (str, os.PathLike)):
        img = PILImage.open(src)
    elif isinstance(src, (bytes, bytearray)):
        import io

        img = PILImage.open(io.BytesIO(src))
    elif isinstance(src, np.ndarray):
        img = PILImage.fromarray(src)
    else:
        raise TypeError("unsupported image type %s" % type(src).__name__)
    return img.convert("RGB")


def preprocess_image(src: Any, size: int = 224, mean=SIGLIP_MEAN, std=SIGLIP_STD) -> torch.Tensor:
    """SigLIP preprocessing: bilinear resize to size x size, scale to [0, 1], normalize. Returns [3, size, size]."""
    from PIL import Image as PILImage

    img = load_image(src).resize((size, size), PILImage.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (x - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]


def image_token_id(tok) -> int:
    """Placeholder id for image slots. Its embedding is always overwritten; an unused vocab entry keeps it distinct
    from [MASK]/[SEP]/[PAD] when inspecting sequences. Falls back to the pad id."""
    for t in ("[unused0]", "<|image|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i != tok.unk_token_id and i not in (tok.mask_token_id, tok.sep_token_id, tok.cls_token_id):
            return i
    return tok.pad_token_id


def build_vision_tower(name: str, pretrained: bool = True) -> nn.Module:
    """SigLIP / SigLIP2 (fixed resolution, model_type 'siglip') vision tower without the MAP pooling head."""
    from transformers import AutoConfig, SiglipVisionConfig, SiglipVisionModel

    mt = AutoConfig.from_pretrained(name).model_type
    if mt not in ("siglip", "siglip_vision_model"):
        raise ValueError(
            "vision_encoder %r has model_type %r; only fixed-resolution SigLIP/SigLIP2 checkpoints "
            "(model_type 'siglip', e.g. %r) are supported" % (name, mt, DEFAULT_VISION_ENCODER)
        )
    if pretrained:
        # Full SigLIP checkpoints also hold the text tower; silence the report listing those skipped keys.
        from transformers.utils import logging as hf_logging

        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            vt = SiglipVisionModel.from_pretrained(name, attn_implementation="sdpa")
        finally:
            hf_logging.set_verbosity(verbosity)
    else:
        vt = SiglipVisionModel(SiglipVisionConfig.from_pretrained(name))
    # The MAP head is never used (we read patch tokens); drop it so it neither loads nor sits idle in DDP.
    m = getattr(vt, "vision_model", vt)
    if getattr(m, "use_head", False):
        m.use_head = False
        del m.head
    return vt


def param_groups(model: nn.Module) -> Dict[str, List[nn.Parameter]]:
    """Split DecisionModel params into encoder / vision / proj / head (everything else: head, type_emb, scorer,
    act_head)."""
    groups: Dict[str, List[nn.Parameter]] = {"encoder": [], "vision": [], "proj": [], "head": []}
    for n, p in model.named_parameters():
        top = n.split(".", 1)[0]
        groups[top if top in groups else "head"].append(p)
    return groups


def encoder_top_params(model: nn.Module, n_layers: int) -> List[nn.Parameter]:
    """Params of the top n_layers ModernBERT layers plus its final norm."""
    if n_layers <= 0:
        return []
    enc = model.encoder
    params = [p for layer in enc.layers[-n_layers:] for p in layer.parameters()]
    return params + list(enc.final_norm.parameters())


def freeze_for_alignment(model: nn.Module, train_head: bool = True, train_top_layers: int = 0) -> List[nn.Parameter]:
    """Stage-1 alignment: freeze ModernBERT + vision tower, train the projector (and the decision head when
    train_head, and the top train_top_layers encoder layers + final norm). Returns the trainable parameters."""
    groups = param_groups(model)
    trainable = set(id(p) for p in groups["proj"] + (groups["head"] if train_head else []))
    trainable |= set(id(p) for p in encoder_top_params(model, train_top_layers))
    for p in model.parameters():
        p.requires_grad = id(p) in trainable
    return [p for p in model.parameters() if p.requires_grad]
