"""Pairwise plausibility: one fitted model over several compatibility signals.

Signals per item pair (all computed, none invented):

    learned_logit  logit of the learned compatibility model (only if one is in use)
    clip_cos       CLIP image-image cosine similarity
    colour         colour-relation score (src/attributes/colors.py)
    style_sim      agreement of the items' style distributions
    category       slot-pair plausibility table (src/recommendation/compatibility.py)

They are combined by a logistic regression whose weights were **fitted on the
validation split of the Maryland Polyvore-derived dataset** (observed outfit
pairs vs constructed negatives) by ``scripts/fit_ranker.py`` and stored in
``ranker_model.json`` next to this file. Two variants exist: with and without
the learned feature, so a fresh clone (no checkpoint) still uses fitted
weights. The output is a plausibility *score* for ranking; it is calibrated
only on that dataset's validation pairs, not on users' photos.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.attributes.colors import RELATION_SCORE, colour_relation
from src.attributes.style import STYLES
from src.recommendation.compatibility import category_pair_score

MODEL_PATH = Path(__file__).with_name("ranker_model.json")
FEATURES_WITH_LEARNED = ("learned_logit", "clip_cos", "colour", "style_sim", "category")
FEATURES_BASELINE = ("clip_cos", "colour", "style_sim", "category")


@dataclass
class ItemTable:
    """Per-item inputs needed to compute pair features (row i = item i)."""

    ids: list[str]
    embeddings: np.ndarray  # (n, d), L2-normalised
    slots: list[str | None]
    colours: list[list[dict[str, Any]] | None]
    style: np.ndarray  # (n, len(STYLES)); rows of NaN = unknown

    @classmethod
    def from_wardrobe(cls, items: list[Any], embeddings: dict[str, np.ndarray]) -> ItemTable:
        style = np.full((len(items), len(STYLES)), np.nan, dtype=np.float32)
        for i, it in enumerate(items):
            sc = it.metadata.get("style_scores")
            if sc:
                style[i] = [float(sc.get(s, 0.0)) for s in STYLES]
        return cls(
            ids=[it.id for it in items],
            embeddings=np.stack([embeddings[it.id] for it in items]).astype(np.float32) if items else np.zeros((0, 1)),
            slots=[it.category for it in items],
            colours=[it.metadata.get("dominant_colors") for it in items],
            style=style,
        )


def pair_features(table: ItemTable, ia: np.ndarray, ib: np.ndarray, learned: Any | None = None) -> dict[str, np.ndarray]:
    """Feature arrays for the pairs (ia[k], ib[k]). Unknown values are NaN."""
    E = table.embeddings
    feats: dict[str, np.ndarray] = {"clip_cos": np.sum(E[ia] * E[ib], axis=1).astype(np.float64)}
    if learned is not None:
        p = np.clip(learned.predict(E[ia], E[ib]).astype(np.float64), 1e-4, 1 - 1e-4)
        feats["learned_logit"] = np.log(p / (1 - p))
    sa, sb = table.style[ia], table.style[ib]
    feats["style_sim"] = np.sum(np.sqrt(np.clip(sa, 0, None) * np.clip(sb, 0, None)), axis=1).astype(np.float64)
    colour = np.full(len(ia), np.nan)
    category = np.full(len(ia), np.nan)
    for k, (i, j) in enumerate(zip(ia.tolist(), ib.tolist(), strict=True)):
        rel = colour_relation(table.colours[i], table.colours[j])
        if rel is not None:
            colour[k] = RELATION_SCORE[rel]
        si, sj = table.slots[i], table.slots[j]
        if si and sj:
            category[k] = category_pair_score(si, sj)
    feats["colour"] = colour
    feats["category"] = category
    return feats


@dataclass
class PairModel:
    """Logistic-regression combiner loaded from ranker_model.json."""

    variant: str
    features: tuple[str, ...]
    coef: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray
    impute: np.ndarray  # value used for a missing feature (training mean)
    outfit_threshold: float | None
    info: dict[str, Any] = field(default_factory=dict)
    fitted: bool = True

    def score(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        X = np.stack([feats[f] for f in self.features], axis=1)
        X = np.where(np.isnan(X), self.impute[None, :], X)
        z = ((X - self.mean) / self.scale) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def contributions(self, feats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Per-feature contribution to the logit (for explanations): coef * standardised value."""
        out = {}
        for k, f in enumerate(self.features):
            x = np.where(np.isnan(feats[f]), self.impute[k], feats[f])
            out[f] = self.coef[k] * (x - self.mean[k]) / self.scale[k]
        return out


def _fallback(variant: str) -> PairModel:
    """Unfitted equal-weight combiner, used only if ranker_model.json is missing."""
    feats = FEATURES_WITH_LEARNED if variant == "with_learned" else FEATURES_BASELINE
    n = len(feats)
    return PairModel(
        variant=variant,
        features=feats,
        coef=np.ones(n),
        intercept=0.0,
        mean=np.array([0.0 if f == "learned_logit" else 0.6 for f in feats]),
        scale=np.array([1.0 if f == "learned_logit" else 0.15 for f in feats]),
        impute=np.array([0.0 if f == "learned_logit" else 0.6 for f in feats]),
        outfit_threshold=None,
        info={"note": "ranker_model.json not found: unfitted equal-weight fallback, no quality threshold"},
        fitted=False,
    )


def load_pair_model(with_learned: bool, path: Path = MODEL_PATH) -> PairModel:
    variant = "with_learned" if with_learned else "baseline"
    if not path.is_file():
        return _fallback(variant)
    data = json.loads(path.read_text(encoding="utf-8"))
    v = data["variants"][variant]
    return PairModel(
        variant=variant,
        features=tuple(v["features"]),
        coef=np.asarray(v["coef"], dtype=np.float64),
        intercept=float(v["intercept"]),
        mean=np.asarray(v["mean"], dtype=np.float64),
        scale=np.asarray(v["scale"], dtype=np.float64),
        impute=np.asarray(v["impute"], dtype=np.float64),
        outfit_threshold=v.get("outfit_threshold"),
        info={k: data[k] for k in ("dataset", "fitted_on", "fitted_at") if k in data}
        | {"metrics": v.get("metrics", {}), "dropped_features": v.get("dropped_features", {})},
    )


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


ACCESSORY_PAIR_WEIGHT = 0.5  # accessories matter less to whether an outfit works


def outfit_plausibility(pair_scores: list[tuple[str | None, str | None, float]]) -> float:
    """Weighted mean of pair scores for one outfit; pairs involving an accessory count half."""
    num = den = 0.0
    for sa, sb, p in pair_scores:
        w = ACCESSORY_PAIR_WEIGHT if "accessory" in (sa, sb) else 1.0
        num += w * p
        den += w
    return num / den if den else 0.0
