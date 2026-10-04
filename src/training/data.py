"""Compatibility datasets: loading, pair construction, leakage-aware splits.

Two input formats are supported (see README "Dataset format"):

1. **Polyvore Outfits** (Vasileva et al., 2018): official
   ``{train,valid,test}.json`` outfit files. Positives are item pairs that
   appear in the same curated outfit; negatives pair items from different
   outfits (half of them category-matched "hard" negatives).
2. **Generic pairs CSV**: ``item_a,item_b,label[,split]`` with label 1/0.
   Without a ``split`` column, items (not pairs) are assigned to splits so
   that no item appears in more than one split.

Embeddings must be pre-computed with ``scripts/embed_dataset.py``, which
uses the same CLIP encoder as the app and writes ``encoder.json``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.errors import DatasetMissingError

Pair = tuple[str, str, int]
# (item_a, item_b, label, kind). kind says where the label came from:
#   observed_positive   items that really appeared together in one outfit
#   random_negative     CONSTRUCTED: items from two different outfits
#   hard_negative       CONSTRUCTED: like a real positive's category pairing, other items
#   labelled_positive / labelled_negative   labels supplied in a pairs CSV
TypedPair = tuple[str, str, int, str]
SPLITS = ("train", "valid", "test")

NEGATIVE_STRATEGY = (
    "Positives are OBSERVED: every pair of items that appear together in one curated outfit. "
    "Negatives are CONSTRUCTED (not human judgements of incompatibility): "
    "(a) random negatives pair items drawn from two different outfits; "
    "(b) hard negatives (only when item categories are known) take a real positive (a, b) and replace b "
    "with another item of b's category from elsewhere, so the category pairing matches real outfits. "
    "Constructed negatives never coincide with an observed positive pair. "
    "A constructed negative may still be a combination a person would wear; it is 'not observed together', "
    "not 'known incompatible'."
)


# ------------------------------------------------------------ Polyvore
def load_polyvore_outfits(json_path: Path) -> list[list[str]]:
    """Outfits as lists of item ids, from a Polyvore ``{split}.json`` file."""
    path = Path(json_path)
    if not path.is_file():
        raise DatasetMissingError(f"Polyvore split file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        outfits = [[str(it["item_id"]) for it in o["items"]] for o in raw]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DatasetMissingError(
            f"{path} is not in Polyvore format (expected [{{'set_id':…, 'items':[{{'item_id':…}}]}}]): {exc}"
        ) from exc
    return [o for o in outfits if len(o) >= 2]


def load_item_categories(metadata_path: Path | None) -> dict[str, str] | None:
    if metadata_path is None:
        return None
    path = Path(metadata_path)
    if not path.is_file():
        raise DatasetMissingError(f"Item metadata file not found: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v.get("semantic_category") or v.get("category_id") or "unknown") for k, v in meta.items()}


def build_outfit_pairs(
    outfits: list[list[str]],
    item_categories: dict[str, str] | None = None,
    neg_per_pos: int = 1,
    hard_negative_ratio: float = 0.5,
    seed: int = 42,
) -> list[Pair]:
    """Untyped (a, b, label) version of :func:`build_outfit_pairs_typed`."""
    return [
        (a, b, y) for a, b, y, _ in build_outfit_pairs_typed(outfits, item_categories, neg_per_pos, hard_negative_ratio, seed)
    ]


def build_outfit_pairs_typed(
    outfits: list[list[str]],
    item_categories: dict[str, str] | None = None,
    neg_per_pos: int = 1,
    hard_negative_ratio: float = 0.5,
    seed: int = 42,
) -> list[TypedPair]:
    """Positive within-outfit pairs + an equal number of cross-outfit negatives.

    Hard negatives keep the category pairing of a real positive (e.g. top +
    bottom) but swap in an item from a different outfit, so the model cannot
    score well just by learning "different categories go together".
    """
    if len(outfits) < 2:
        raise DatasetMissingError("Need at least two outfits to build negative pairs.")
    rng = random.Random(seed)
    outfit_sets = [set(o) for o in outfits]
    positives: list[TypedPair] = [
        (o[i], o[j], 1, "observed_positive") for o in outfits for i in range(len(o)) for j in range(i + 1, len(o))
    ]
    pos_keys = {frozenset((a, b)) for a, b, _, _ in positives}

    n_neg = len(positives) * neg_per_pos
    n_hard = int(n_neg * hard_negative_ratio) if item_categories else 0
    negatives: list[TypedPair] = []

    def ok(a: str, b: str) -> bool:
        return a != b and frozenset((a, b)) not in pos_keys

    attempts = 0
    while len(negatives) < n_neg - n_hard and attempts < n_neg * 20:
        attempts += 1
        ia, ib = rng.sample(range(len(outfits)), 2)
        a, b = rng.choice(outfits[ia]), rng.choice(outfits[ib])
        if ok(a, b) and b not in outfit_sets[ia]:
            negatives.append((a, b, 0, "random_negative"))

    if item_categories and n_hard:
        by_cat: dict[str, list[str]] = {}
        for o in outfits:
            for it in o:
                by_cat.setdefault(item_categories.get(it, "unknown"), []).append(it)
        made = attempts = 0
        while made < n_hard and attempts < n_hard * 20:
            attempts += 1
            a, b, _, _ = rng.choice(positives)
            pool = by_cat.get(item_categories.get(b, "unknown"), [])
            if len(pool) < 2:
                continue
            swap = rng.choice(pool)
            if ok(a, swap):
                negatives.append((a, swap, 0, "hard_negative"))
                made += 1

    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs


def typed_from_csv_rows(rows: Iterable[tuple[str, str, int, str | None]]) -> list[tuple[str, str, int, str, str | None]]:
    """(a, b, label, kind, split) for user-labelled CSV pairs."""
    return [(a, b, y, "labelled_positive" if y == 1 else "labelled_negative", split) for a, b, y, split in rows]


# ---------------------------------------------------------- pairs CSV
def load_pairs_csv(path: Path) -> list[tuple[str, str, int, str | None]]:
    path = Path(path)
    if not path.is_file():
        raise DatasetMissingError(f"Pairs CSV not found: {path}")
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"item_a", "item_b", "label"} - set(reader.fieldnames or [])
        if missing:
            raise DatasetMissingError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
        for n, row in enumerate(reader, start=2):
            try:
                label = int(float(row["label"]))
            except ValueError as exc:
                raise DatasetMissingError(f"{path}:{n}: label must be 0 or 1, got {row['label']!r}") from exc
            if label not in (0, 1):
                raise DatasetMissingError(f"{path}:{n}: label must be 0 or 1, got {label}")
            split = (row.get("split") or "").strip().lower() or None
            if split is not None and split not in SPLITS:
                raise DatasetMissingError(f"{path}:{n}: split must be one of {SPLITS}, got {split!r}")
            rows.append((row["item_a"].strip(), row["item_b"].strip(), label, split))
    if not rows:
        raise DatasetMissingError(f"{path} contains no pairs.")
    return rows


def _item_split(item_id: str, valid_frac: float, test_frac: float, seed: int) -> str:
    h = int(hashlib.sha1(f"{seed}:{item_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + valid_frac:
        return "valid"
    return "train"


def split_pairs(
    rows: Iterable[tuple[str, str, int, str | None]],
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[dict[str, list[Pair]], int]:
    """Use explicit splits if given; otherwise an item-disjoint split.

    Returns (splits, n_dropped). Pairs whose two items land in different
    splits are dropped, so no item is shared between train and evaluation.
    """
    out: dict[str, list[Pair]] = {s: [] for s in SPLITS}
    dropped = 0
    for a, b, y, split in rows:
        if split is None:
            sa = _item_split(a, valid_frac, test_frac, seed)
            sb = _item_split(b, valid_frac, test_frac, seed)
            if sa != sb:
                dropped += 1
                continue
            split = sa
        out[split].append((a, b, y))
    return out, dropped


# ----------------------------------------------------------- embeddings
class EmbeddingDirectory:
    """``<item_id>.npy`` files plus ``encoder.json`` written by embed_dataset.py."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise DatasetMissingError(f"Embeddings directory not found: {self.directory}. Run scripts/embed_dataset.py first.")
        cfg = self.directory / "encoder.json"
        if not cfg.is_file():
            raise DatasetMissingError(
                f"{cfg} is missing, so the encoder used for these embeddings is unknown. "
                "Re-create them with scripts/embed_dataset.py."
            )
        self.encoder_config = json.loads(cfg.read_text(encoding="utf-8"))
        self._cache: dict[str, np.ndarray | None] = {}

    def get(self, item_id: str) -> np.ndarray | None:
        if item_id not in self._cache:
            path = self.directory / f"{item_id}.npy"
            try:
                emb = np.load(path, allow_pickle=False).astype(np.float32).reshape(-1)
                ok = emb.shape[0] == int(self.encoder_config["dim"]) and np.isfinite(emb).all()
                self._cache[item_id] = emb if ok else None
            except (OSError, ValueError):
                self._cache[item_id] = None
        return self._cache[item_id]

    def tensors(self, pairs: Sequence[tuple[Any, ...]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Stack embeddings for pairs -> (A, B, labels, n_missing). Pairs may be typed."""
        a_rows, b_rows, labels, _ = self.tensors_with_index(pairs)
        return a_rows, b_rows, labels, len(pairs) - len(labels)

    def tensors_with_index(self, pairs: Sequence[tuple[Any, ...]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        """Like tensors() but returns the indices of the pairs that had both embeddings."""
        a_rows, b_rows, labels, kept = [], [], [], []
        for idx, p in enumerate(pairs):
            a, b, y = p[0], p[1], p[2]
            ea, eb = self.get(a), self.get(b)
            if ea is None or eb is None:
                continue
            a_rows.append(ea)
            b_rows.append(eb)
            labels.append(y)
            kept.append(idx)
        dim = int(self.encoder_config["dim"])
        if not labels:
            return np.zeros((0, dim), np.float32), np.zeros((0, dim), np.float32), np.zeros(0, np.float32), kept
        return np.stack(a_rows), np.stack(b_rows), np.asarray(labels, dtype=np.float32), kept
