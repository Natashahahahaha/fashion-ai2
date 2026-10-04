"""Style estimates (CLIP zero-shot) and style/occasion matching tables.

The style scores are a *soft zero-shot estimate*: the cosine similarity of
the garment's CLIP image embedding to text prompts such as "a photo of
streetwear clothing", softmaxed over the style list. They are useful as a
ranking signal but are not verified labels. The UI shows them as estimates
and lets the user override an item's style; a user override is stored with
``style_source = "user"``.

The occasion table is hand-written domain heuristics, documented here so it
can be tuned. It is not learned from data.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping

import numpy as np

from src.embeddings.clip_encoder import ImageEncoder, l2_normalize

STYLES: tuple[str, ...] = ("casual", "streetwear", "formal", "smart casual", "sporty", "minimal", "vintage")
OCCASIONS: tuple[str, ...] = ("college", "everyday", "party", "formal event", "date", "interview")

_STYLE_PROMPTS: dict[str, tuple[str, ...]] = {
    "casual": ("casual everyday clothing", "a relaxed casual piece of clothing"),
    "streetwear": ("streetwear clothing", "urban streetwear fashion"),
    "formal": ("formal clothing", "elegant formal wear"),
    "smart casual": ("smart casual clothing", "business casual clothing"),
    "sporty": ("sportswear", "athletic sporty clothing"),
    "minimal": ("minimalist clothing", "simple plain minimalist fashion"),
    "vintage": ("vintage clothing", "retro vintage fashion"),
}
_TEMPLATE = "a photo of {}"
CLIP_LOGIT_SCALE = 100.0  # CLIP's learned temperature for ViT-B/32 is ~100

# How well each style suits each occasion, 0..1 (hand-written heuristic).
OCCASION_STYLE_AFFINITY: dict[str, dict[str, float]] = {
    "college": {
        "casual": 1.0,
        "streetwear": 0.9,
        "minimal": 0.8,
        "sporty": 0.7,
        "vintage": 0.7,
        "smart casual": 0.6,
        "formal": 0.1,
    },
    "everyday": {
        "casual": 1.0,
        "minimal": 0.9,
        "streetwear": 0.8,
        "smart casual": 0.7,
        "sporty": 0.7,
        "vintage": 0.7,
        "formal": 0.2,
    },
    "party": {
        "streetwear": 0.8,
        "formal": 0.7,
        "vintage": 0.7,
        "smart casual": 0.6,
        "minimal": 0.5,
        "casual": 0.4,
        "sporty": 0.1,
    },
    "formal event": {
        "formal": 1.0,
        "smart casual": 0.6,
        "minimal": 0.5,
        "vintage": 0.3,
        "casual": 0.1,
        "streetwear": 0.05,
        "sporty": 0.0,
    },
    "date": {"smart casual": 1.0, "minimal": 0.8, "formal": 0.7, "vintage": 0.7, "casual": 0.6, "streetwear": 0.5, "sporty": 0.1},
    "interview": {
        "formal": 1.0,
        "smart casual": 0.9,
        "minimal": 0.7,
        "casual": 0.2,
        "vintage": 0.2,
        "streetwear": 0.05,
        "sporty": 0.0,
    },
}


class StyleClassifier:
    """Zero-shot style scorer on top of a shared CLIP encoder."""

    def __init__(self, encoder: ImageEncoder):
        self.encoder = encoder
        self._text: np.ndarray | None = None
        self._lock = threading.Lock()

    def _text_matrix(self) -> np.ndarray:
        if self._text is None:
            with self._lock:
                if self._text is None:
                    rows = []
                    for style in STYLES:
                        prompts = [_TEMPLATE.format(p) for p in _STYLE_PROMPTS[style]]
                        emb = self.encoder.encode_text(prompts)
                        rows.append(l2_normalize(emb.mean(axis=0)))
                    self._text = np.stack(rows)
        return self._text

    def scores(self, image_embedding: np.ndarray) -> dict[str, float]:
        """Softmax style distribution for one normalised image embedding."""
        logits = CLIP_LOGIT_SCALE * (self._text_matrix() @ np.asarray(image_embedding, dtype=np.float32))
        logits = logits - float(np.max(logits))
        probs = np.exp(logits)
        probs /= probs.sum()
        return {s: round(float(p), 4) for s, p in zip(STYLES, probs, strict=True)}


def top_style(scores: Mapping[str, float] | None) -> str | None:
    if not scores:
        return None
    return max(scores.items(), key=lambda kv: kv[1])[0]


def style_similarity(a: Mapping[str, float] | None, b: Mapping[str, float] | None) -> float | None:
    """Bhattacharyya coefficient of two style distributions (1 = identical)."""
    if not a or not b:
        return None
    return float(sum(math.sqrt(max(a.get(s, 0.0), 0.0) * max(b.get(s, 0.0), 0.0)) for s in STYLES))


def style_match(scores: Mapping[str, float] | None, style: str) -> float | None:
    """How strongly an item leans towards ``style`` relative to its top style (0..1)."""
    if not scores:
        return None
    peak = max(scores.values()) or 1.0
    return float(min(1.0, scores.get(style, 0.0) / peak))


def occasion_match(scores: Mapping[str, float] | None, occasion: str) -> float | None:
    """Expected style/occasion affinity under the item's style distribution (0..1)."""
    if not scores or occasion not in OCCASION_STYLE_AFFINITY:
        return None
    table = OCCASION_STYLE_AFFINITY[occasion]
    total = sum(scores.values()) or 1.0
    return float(sum(p * table.get(s, 0.0) for s, p in scores.items()) / total)


def user_style_scores(style: str) -> dict[str, float]:
    """One-hot distribution for a style the user set by hand."""
    if style not in STYLES:
        raise ValueError(f"Unknown style {style!r}; expected one of {STYLES}")
    return {s: (1.0 if s == style else 0.0) for s in STYLES}
