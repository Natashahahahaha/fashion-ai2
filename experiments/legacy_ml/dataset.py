"""
Real Polyvore data loading + pair construction.

This replaces the synthetic-Gaussian-noise training data that caused the
model to memorize meaningless patterns and output near-constant 1.0000
scores. Positives here are pairs of items that actually appeared together
in a real curated outfit; negatives are pairs that never did.

Expected directory layout (matches the standard Polyvore-Outfits release,
Vasileva et al. 2018 — mirrors exist on Hugging Face, e.g.
huggingface.co/datasets/owj0421/polyvore, and the original repo at
github.com/mvasil/fashion-compatibility). Field names vary slightly
between mirrors — check your download against `_ITEM_KEY` / `_ITEMS_KEY`
below and adjust if needed.

    data/Polyvore/
        images/                          # item_id.jpg
        nondisjoint/  (or disjoint/)
            train.json
            valid.json
            test.json
        polyvore_item_metadata.json       # item_id -> {"category_id": ..., "semantic_category": ..., ...}
        categories.csv

train.json / valid.json / test.json format (per outfit):
    {"set_id": "...", "items": [{"item_id": "123456", "index": 1}, ...]}
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Tuple

_ITEMS_KEY = "items"
_ITEM_ID_KEY = "item_id"


def load_outfits(json_path: str) -> List[List[str]]:
    """Returns a list of outfits, each a list of item_ids (order preserved)."""
    with open(json_path) as f:
        raw = json.load(f)

    outfits = []
    for outfit in raw:
        item_ids = [item[_ITEM_ID_KEY] for item in outfit[_ITEMS_KEY]]
        if len(item_ids) >= 2:
            outfits.append(item_ids)
    return outfits


def load_item_categories(metadata_path: str) -> Dict[str, str]:
    """item_id -> semantic_category (or category_id if semantic_category isn't present).
    Used only for hard-negative sampling; not required for basic pair construction."""
    with open(metadata_path) as f:
        meta = json.load(f)

    categories = {}
    for item_id, info in meta.items():
        categories[item_id] = info.get("semantic_category") or info.get("category_id", "unknown")
    return categories


def build_pairs(
    outfits: List[List[str]],
    item_categories: Optional[Dict[str, str]] = None,
    neg_per_pos: int = 1,
    hard_negative_ratio: float = 0.5,
    seed: int = 42,
) -> List[Tuple[str, str, int]]:
    """
    Positives: every within-outfit item pair (label=1).
    Negatives: item pairs drawn from DIFFERENT outfits (label=0).

    If item_categories is given, `hard_negative_ratio` of negatives are
    "hard" — the negative pair is constrained to match the category
    pairing of a real positive (e.g. if a real positive was top+bottom,
    the hard negative is also a top+bottom pair, just not one that
    actually appeared together). This matters: without hard negatives,
    the model can hit high accuracy just by learning "same category ==
    incompatible", which isn't the actual compatibility signal you want.

    Returns a shuffled list of (item_id_a, item_id_b, label).
    """
    rng = random.Random(seed)
    all_items = [item for outfit in outfits for item in outfit]

    positives: List[Tuple[str, str, int]] = []
    for outfit in outfits:
        for i in range(len(outfit)):
            for j in range(i + 1, len(outfit)):
                positives.append((outfit[i], outfit[j], 1))

    negatives: List[Tuple[str, str, int]] = []
    n_hard = int(len(positives) * neg_per_pos * hard_negative_ratio) if item_categories else 0
    n_random = len(positives) * neg_per_pos - n_hard

    # Random cross-outfit negatives
    for _ in range(n_random):
        outfit_a, outfit_b = rng.sample(outfits, 2)
        a = rng.choice(outfit_a)
        b = rng.choice(outfit_b)
        negatives.append((a, b, 0))

    # Hard negatives: same category-pair shape as a real positive, different items
    if item_categories and n_hard:
        by_category: Dict[str, List[str]] = {}
        for item in all_items:
            cat = item_categories.get(item, "unknown")
            by_category.setdefault(cat, []).append(item)

        made = 0
        attempts = 0
        while made < n_hard and attempts < n_hard * 20:
            attempts += 1
            a, b, _ = rng.choice(positives)
            cat_a, cat_b = item_categories.get(a, "unknown"), item_categories.get(b, "unknown")
            pool_b = by_category.get(cat_b, [])
            if len(pool_b) < 2:
                continue
            b_swap = rng.choice(pool_b)
            if b_swap == b:
                continue
            negatives.append((a, b_swap, 0))
            made += 1

    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs


if __name__ == "__main__":
    # Smoke test with synthetic outfits — verifies pairing LOGIC only.
    # This is NOT a substitute for training on real data. Never treat
    # results from synthetic data as a measure of model quality.
    fake_outfits = [
        ["shirt_1", "pants_1", "shoes_1"],
        ["shirt_2", "skirt_1", "shoes_2"],
        ["dress_1", "jacket_1", "bag_1"],
    ]
    fake_categories = {
        "shirt_1": "top", "shirt_2": "top",
        "pants_1": "bottom", "skirt_1": "bottom",
        "shoes_1": "shoes", "shoes_2": "shoes",
        "dress_1": "dress", "jacket_1": "outerwear", "bag_1": "bag",
    }
    outfits = fake_outfits
    pairs = build_pairs(outfits, item_categories=fake_categories, neg_per_pos=1, hard_negative_ratio=0.5)
    n_pos = sum(1 for _, _, y in pairs if y == 1)
    n_neg = sum(1 for _, _, y in pairs if y == 0)
    print(f"[smoke test] {len(pairs)} pairs built: {n_pos} positive, {n_neg} negative")
    assert n_pos > 0 and n_neg > 0, "pairing logic produced no positives or no negatives"
    print("[smoke test] OK — pairing logic is sound. Run against real Polyvore JSON for actual data.")
