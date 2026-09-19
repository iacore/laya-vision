"""Core model architecture, token sequence construction, and confidence estimation for laya."""
import json
import math
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .vision import DEFAULT_N_IMAGE_TOKENS, build_vision_tower, split_image_state

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


def serialize_state(state: Union[str, dict, list]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_options(q: Dict) -> List[str]:
    """Render option texts in label-index order. Noul is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return [
        "false: " + (crit.get("false") or "no, the statement does not hold"),
        "true: " + (crit.get("true") or "yes, the statement holds"),
    ]


def build_sequence(
    tok,
    state: Union[str, dict, list],
    q: Dict,
    max_len: int = 512,
    head_max_len: int = 192,
    option_order: Optional[List[int]] = None,
    truncate_left: bool = False,
    n_image_tokens: int = 0,
    image_token_id: Optional[int] = None,
):
    """Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP].

    Vision models pass n_image_tokens > 0 and get (ids, markers, image_pos). If state holds an image (see
    laya.vision.split_image_state) the image segment goes before the text state:
    ... [SEP] <img> x n_image_tokens [SEP] state [SEP], where the text part is dropped for image-only states.
    image_pos lists the placeholder positions ([] for text-only states, whose ids are unchanged). The pixels
    themselves are encoded separately by the caller.
    """
    image, state = split_image_state(state)
    if image is not None and n_image_tokens <= 0:
        raise ValueError("state contains an image but n_image_tokens=0; load the model with a vision_encoder")
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_ids = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in order:
        opt_ids.append(
            [tok.mask_token_id]
            + tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48]
        )
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    image_pos = []
    if image is not None:
        if len(ids) + n_image_tokens + 1 > max_len:
            raise ValueError("%d image tokens do not fit in max_len=%d after the question" % (n_image_tokens, max_len))
        image_pos = list(range(len(ids), len(ids) + n_image_tokens))
        ids.extend([tok.pad_token_id if image_token_id is None else image_token_id] * n_image_tokens)
        ids.append(tok.sep_token_id)
    if image is None or state is not None:
        room = max(0, max_len - len(ids) - 1)
        st = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
        st = st[-room:] if truncate_left else st[:room]
        ids = ids + st + [tok.sep_token_id]
    if n_image_tokens > 0:
        return ids[:max_len], [m for m in markers if m < max_len], image_pos
    return ids[:max_len], [m for m in markers if m < max_len]


def adaptive_pool_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """[n_out, n_in] averaging weights with adaptive_avg_pool semantics. Used as a separable 2D pool because MPS
    lacks adaptive pooling for non-divisible sizes (e.g. 14x14 SigLIP patches -> 8x8)."""
    m = torch.zeros(n_out, n_in)
    for i in range(n_out):
        a, b = (i * n_in) // n_out, -(-(i + 1) * n_in // n_out)
        m[i, a:b] = 1.0 / (b - a)
    return m


class DecisionModel(nn.Module):
    """Bidirectional transformer encoder backbone + typed decision head.

    With a vision tower, image patches are avg-pooled to n_image_tokens, projected to the encoder width by `proj`
    and spliced into the encoder's input embeddings at the image placeholder positions.
    """

    def __init__(
        self,
        encoder: nn.Module,
        head_layers: int = 2,
        n_act: int = 2,
        dropout: float = 0.1,
        vision: Optional[nn.Module] = None,
        n_image_tokens: int = DEFAULT_N_IMAGE_TOKENS,
    ):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        self.n_image_tokens = 0
        if vision is not None:
            side = math.isqrt(n_image_tokens)
            if side * side != n_image_tokens:
                raise ValueError("n_image_tokens must be a perfect square, got %d" % n_image_tokens)
            vd = vision.config.hidden_size
            self.vision = vision
            self.proj = nn.Sequential(nn.LayerNorm(vd), nn.Linear(vd, d), nn.GELU(), nn.Linear(d, d))
            self.n_image_tokens = n_image_tokens
            self.image_size = vision.config.image_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))
        self.head_checkpointing = False

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[U, 3, H, W] -> [U, n_image_tokens, d] projected patch embeddings."""
        interp = pixel_values.shape[-1] != self.vision.config.image_size
        x = self.vision(pixel_values=pixel_values, interpolate_pos_encoding=interp).last_hidden_state
        u, n, vd = x.shape
        if n != self.n_image_tokens:
            g, side = math.isqrt(n), math.isqrt(self.n_image_tokens)
            pool = adaptive_pool_matrix(g, side).to(x)
            x = torch.einsum("ia,jb,uabd->uijd", pool, pool, x.reshape(u, g, g, vd)).reshape(u, side * side, vd)
        return self.proj(x)

    def embed_with_images(self, input_ids, pixel_values, image_pos, image_index=None) -> torch.Tensor:
        """Token embeddings with rows' image placeholders replaced by projected patches.

        image_pos: [B, n_image_tokens] placeholder positions; image_index: [B] row into pixel_values (-1: no image,
        default: row b uses image b).
        """
        emb = self.encoder.get_input_embeddings()(input_ids)
        if image_index is None:
            image_index = torch.arange(input_ids.size(0), device=input_ids.device)
        rows = (image_index >= 0).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            return emb
        img = self.encode_images(pixel_values).to(emb.dtype)
        pos = image_pos[rows]
        return emb.index_put((rows[:, None].expand_as(pos), pos), img[image_index[rows]])

    def forward(
        self,
        input_ids,
        attention_mask,
        marker_pos,
        marker_mask,
        qtype,
        detach_encoder: bool = False,
        pixel_values=None,
        image_pos=None,
        image_index=None,
    ):
        if pixel_values is None:
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            emb = self.embed_with_images(input_ids, pixel_values, image_pos, image_index)
            h = self.encoder(inputs_embeds=emb, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def build_model(cfg: Dict, encoder_dir: Optional[str] = None, vision_pretrained: bool = True) -> DecisionModel:
    """Build the decision model. With cfg["vision_encoder"] set, also a vision tower (pretrained weights unless
    vision_pretrained=False, e.g. when the checkpoint already holds vision.* weights) and a fresh projector."""
    from transformers import AutoConfig, AutoModel

    if encoder_dir and os.path.exists(encoder_dir):
        ecfg = AutoConfig.from_pretrained(encoder_dir)
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    else:
        enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")
    vision = build_vision_tower(cfg["vision_encoder"], vision_pretrained) if cfg.get("vision_encoder") else None
    return DecisionModel(
        enc,
        cfg.get("head_layers", 2),
        len(cfg.get("act_costs", {})) + 1,
        vision=vision,
        n_image_tokens=cfg.get("n_image_tokens", DEFAULT_N_IMAGE_TOKENS),
    )


def proper_reward(
    q: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """Strictly proper scoring rule reward: log score + spherical score + ranked probability score.

    q: [..., N, K] reported distributions
    target: [N, K] (one-hot or soft target distributions)
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def td_lambda_targets(p_true: torch.Tensor, batch: Dict, lam: float = 1.0) -> torch.Tensor:
    """TD(lambda) targets for multi-turn conversation trajectories."""
    target = batch["target"].clone()
    groups = batch.get("ep_group")
    if groups is None:
        return target
    for g in torch.unique(groups[groups >= 0]).tolist():
        idx = (groups == g).nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(batch["ep_step"][idx])]
        y = batch["target"][idx[-1], 1]
        G = y
        for j in range(len(idx) - 1, -1, -1):
            if j < len(idx) - 1:
                G = (1 - lam) * p_true[idx[j + 1]] + lam * G
            target[idx[j], 0], target[idx[j], 1] = 1 - G, G
    return target


def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error across confidence bins."""
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k)."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def amp_dtype(name: Optional[str]) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def collate_items(batch, pad_id: int):
    """Pad items into a batch. Items with "pixel_values" ([3, H, W]) and "image_pos" add pixel_values [U, 3, H, W]
    (items sharing one tensor object share a row), image_index [n] (-1: no image), has_image [n] and image_pos
    [n, n_image_tokens]; text-only batches get none of these keys."""
    items = [it for group in batch for it in group]
    if not items:
        return None
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    has_target = any("target" in it for it in items)
    target = torch.zeros((n, kmax), dtype=torch.float32) if has_target else None

    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        if has_target and "target" in it:
            target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)

    res = {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it.get("label", -1) for it in items]),
        "meta": [
            {k: it[k] for k in it if k not in ("ids", "markers", "target", "pixel_values", "image_pos")}
            for it in items
        ],
    }
    if target is not None:
        res["target"] = target

    if any(it.get("pixel_values") is not None for it in items):
        uniq, image_index = [], []
        for it in items:
            px = it.get("pixel_values")
            j = next((u for u, seen in enumerate(uniq) if seen is px), None) if px is not None else -1
            if j is None:
                uniq.append(px)
                j = len(uniq) - 1
            image_index.append(j)
        nimg = max(len(it.get("image_pos") or []) for it in items)
        ipos = torch.zeros((n, nimg), dtype=torch.long)
        for i, it in enumerate(items):
            if image_index[i] >= 0:
                ipos[i] = torch.tensor(it["image_pos"])
        res["pixel_values"] = torch.stack(uniq)
        res["image_index"] = torch.tensor(image_index)
        res["has_image"] = res["image_index"] >= 0
        res["image_pos"] = ipos
    return res
