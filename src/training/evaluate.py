"""Evaluation of a saved compatibility checkpoint: pair metrics and Polyvore FITB."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from src.models.learned import LearnedCompatibility
from src.training.data import EmbeddingDirectory, Pair
from src.training.metrics import classification_metrics


def evaluate_pairs(model: LearnedCompatibility, store: EmbeddingDirectory, pairs: list[Pair]) -> dict[str, Any]:
    a, b, y, missing = store.tensors(pairs)
    if len(y) == 0:
        return {"n": 0, "missing_embeddings": missing}
    probs = model.predict(a, b)
    m = classification_metrics(y, probs)
    m["missing_embeddings"] = missing
    return m


def evaluate_fitb(
    model: LearnedCompatibility,
    store: EmbeddingDirectory,
    outfits: list[list[str]],
    num_distractors: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Fill-in-the-blank: hide one item, rank it against random distractors.

    Uses random distractors from the same split (the official Polyvore FITB
    question files use category-matched distractors, which is harder; numbers
    from this function are therefore not directly comparable to papers).
    """
    rng = random.Random(seed)
    pool = sorted({i for o in outfits for i in o if store.get(i) is not None})
    correct = total = 0
    for outfit in outfits:
        usable = [i for i in outfit if store.get(i) is not None]
        if len(usable) < 2 or len(pool) <= num_distractors + len(usable):
            continue
        blank = rng.randrange(len(usable))
        answer, rest = usable[blank], usable[:blank] + usable[blank + 1 :]
        distractors: list[str] = []
        while len(distractors) < num_distractors:
            c = rng.choice(pool)
            if c not in outfit and c not in distractors:
                distractors.append(c)
        candidates = [answer] + distractors
        rest_embs = np.stack([store.get(i) for i in rest])  # type: ignore[misc]
        scores = []
        for c in candidates:
            ce = np.repeat(store.get(c)[None], len(rest), axis=0)  # type: ignore[index]
            scores.append(float(model.predict(ce, rest_embs).mean()))
        correct += int(int(np.argmax(scores)) == 0)
        total += 1
    return {
        "fitb_accuracy": correct / total if total else float("nan"),
        "questions": total,
        "chance": 1.0 / (1 + num_distractors),
    }


def load_for_eval(checkpoint: Path, embeddings_dir: Path, device: str = "cpu") -> tuple[LearnedCompatibility, EmbeddingDirectory]:
    store = EmbeddingDirectory(embeddings_dir)
    model = LearnedCompatibility.load(checkpoint, expected_encoder=store.encoder_config, device=device)
    return model, store
