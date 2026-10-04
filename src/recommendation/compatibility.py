"""Pairwise compatibility: visual + category + colour + style (+ context).

Each signal is a number in [0, 1]. The pair score is the weighted mean of the
signals that are *available* for that pair, with weights from
``Settings.weights``. A missing signal (e.g. no style estimate) is dropped and
the remaining weights re-normalised, so the score falls back towards visual
compatibility instead of inventing a value.

Visual signal:
  * baseline: CLIP cosine similarity, linearly rescaled from the range
    observed for garment crops ([VISUAL_COS_LOW, VISUAL_COS_HIGH]) to [0, 1].
    This measures visual coherence, which is a proxy, not "compatibility".
  * learned: if a trained checkpoint is loaded, its sigmoid probability
    replaces the cosine proxy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from src.attributes.style import occasion_match, style_match
from src.config import ScoringWeights
from src.detection.categories import ACCESSORY, BOTTOM, ONE_PIECE, OUTERWEAR, SHOES, TOP
from src.wardrobe.schemas import WardrobeItem

# Observed CLIP ViT-B/32 image-image cosine range for garment crops
# (roughly 0.55-0.90 on real crops). Values outside are clipped.
VISUAL_COS_LOW = 0.45
VISUAL_COS_HIGH = 0.90

# How naturally two slots combine in one outfit. 0 = cannot be worn together
# (two bottoms, a dress plus trousers). Hand-written, symmetric.
_PAIR_TABLE: dict[frozenset[str], float] = {
    frozenset({TOP, BOTTOM}): 1.0,
    frozenset({TOP, SHOES}): 0.8,
    frozenset({BOTTOM, SHOES}): 0.9,
    frozenset({TOP, OUTERWEAR}): 0.9,
    frozenset({BOTTOM, OUTERWEAR}): 0.8,
    frozenset({OUTERWEAR, SHOES}): 0.8,
    frozenset({ONE_PIECE, SHOES}): 1.0,
    frozenset({ONE_PIECE, OUTERWEAR}): 0.9,
    frozenset({ONE_PIECE, ACCESSORY}): 0.8,
    frozenset({TOP, ACCESSORY}): 0.7,
    frozenset({BOTTOM, ACCESSORY}): 0.7,
    frozenset({SHOES, ACCESSORY}): 0.7,
    frozenset({OUTERWEAR, ACCESSORY}): 0.7,
    frozenset({TOP, ONE_PIECE}): 0.0,
    frozenset({BOTTOM, ONE_PIECE}): 0.0,
}


def category_pair_score(slot_a: str, slot_b: str) -> float:
    if slot_a == slot_b:
        return 0.0  # two tops, two pairs of shoes, two accessories... handled by templates
    return _PAIR_TABLE.get(frozenset({slot_a, slot_b}), 0.5)


def rescale_cosine(cos: float | np.ndarray) -> float | np.ndarray:
    scaled = (np.asarray(cos, dtype=np.float32) - VISUAL_COS_LOW) / (VISUAL_COS_HIGH - VISUAL_COS_LOW)
    out = np.clip(scaled, 0.0, 1.0)
    return float(out) if np.ndim(out) == 0 else out


def weighted_mean(components: Mapping[str, float | None], weights: ScoringWeights) -> float:
    w = weights.as_dict()
    num = den = 0.0
    for name, value in components.items():
        if value is None or w.get(name, 0.0) <= 0:
            continue
        num += w[name] * float(value)
        den += w[name]
    return num / den if den > 0 else 0.0


@dataclass
class PairScore:
    score: float
    components: dict[str, float | None]
    reasons: dict[str, str] = field(default_factory=dict)


def item_context(item: WardrobeItem, style: str | None, occasion: str | None) -> float | None:
    """How well one item fits the requested style/occasion (None if not requested or unknown)."""
    scores = item.metadata.get("style_scores")
    values = []
    if style:
        m = style_match(scores, style)
        if m is not None:
            values.append(m)
    if occasion:
        m = occasion_match(scores, occasion)
        if m is not None:
            values.append(m)
    return sum(values) / len(values) if values else None
