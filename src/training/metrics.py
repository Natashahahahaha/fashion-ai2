"""Evaluation metrics for pairwise compatibility scores (no model code here)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def _has_both(y: np.ndarray) -> bool:
    return len(np.unique(y)) == 2


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """ECE with equal-width probability bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(ece)


def classification_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5, probabilistic: bool = True
) -> dict[str, Any]:
    """Threshold-free (ROC-AUC, PR-AUC) and thresholded metrics; calibration if scores are probabilities."""
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    preds = (s >= threshold).astype(int)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
        "roc_auc": float(roc_auc_score(y, s)) if _has_both(y) else float("nan"),
        "pr_auc": float(average_precision_score(y, s)) if _has_both(y) else float("nan"),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
    }
    if probabilistic:
        out["brier"] = float(np.mean((s - y) ** 2))
        out["ece_10bin"] = expected_calibration_error(y, s)
    return out


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1 (to be chosen on VALIDATION data only)."""
    from sklearn.metrics import precision_recall_curve

    y = np.asarray(labels).astype(int)
    if not _has_both(y):
        return 0.5
    prec, rec, thr = precision_recall_curve(y, scores)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, s = np.asarray(labels), np.asarray(scores)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if _has_both(y[idx]):
            vals.append(metric(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_auc_difference(
    labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, n: int = 500, seed: int = 0
) -> dict[str, float]:
    """ROC-AUC(a) - ROC-AUC(b) on the same resampled pairs, with a 95% CI."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    y = np.asarray(labels).astype(int)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if _has_both(y[idx]):
            diffs.append(roc_auc_score(y[idx], scores_a[idx]) - roc_auc_score(y[idx], scores_b[idx]))
    point = float(roc_auc_score(y, scores_a) - roc_auc_score(y, scores_b)) if _has_both(y) else float("nan")
    lo, hi = (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))) if diffs else (float("nan"), float("nan"))
    return {"difference": point, "ci95_low": lo, "ci95_high": hi}


def auc_by_group(labels: np.ndarray, scores: np.ndarray, groups: Sequence[str], min_each: int = 30) -> dict[str, dict[str, Any]]:
    """ROC-AUC per group label (e.g. category pair), only where both classes have >= min_each examples."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(labels).astype(int)
    s = np.asarray(scores)
    g = np.asarray(groups)
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(set(groups)):
        m = g == name
        pos, neg = int(y[m].sum()), int((1 - y[m]).sum())
        if pos >= min_each and neg >= min_each:
            out[name] = {"roc_auc": float(roc_auc_score(y[m], s[m])), "positives": pos, "negatives": neg}
    return out


def positives_vs_kind_auc(
    labels: np.ndarray, scores: np.ndarray, kinds: Sequence[str], negative_kind: str
) -> dict[str, Any] | None:
    """ROC-AUC of all positives against one kind of negative (e.g. hard negatives only)."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(labels).astype(int)
    k = np.asarray(kinds)
    m = (y == 1) | (k == negative_kind)
    if (k == negative_kind).sum() == 0 or (y[m] == 1).sum() == 0:
        return None
    return {"roc_auc": float(roc_auc_score(y[m], np.asarray(scores)[m])), "negatives": int((k == negative_kind).sum())}
