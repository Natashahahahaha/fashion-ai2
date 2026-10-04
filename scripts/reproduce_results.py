"""Re-derive the reported learned-model and baseline numbers from the current code + saved checkpoint.

    python scripts/reproduce_results.py

Rebuilds the audited, item-disjoint splits exactly as training does (same code,
same seed), checks that no item is shared between train/valid/test, scores the
TEST pairs with the random, CLIP-cosine, heuristic and learned scorers, and
prints the numbers next to the ones stored in the checkpoint. Needs the
downloaded + embedded dataset (download_dataset.py, embed_dataset.py) and a
trained checkpoint (train_compatibility.py). Image checks re-read every image,
so on a slow disk this takes several minutes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap
import numpy as np

from src.config import get_settings, resolve_device
from src.models.learned import LearnedCompatibility
from src.training.audit import CATEGORY_TO_SLOT, DatasetSource, audit_and_prepare
from src.training.data import EmbeddingDirectory, load_item_categories
from src.training.metrics import positives_vs_kind_auc
from src.training.train import _heuristic_scores


def main() -> int:
    from sklearn.metrics import average_precision_score, roc_auc_score

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=_bootstrap.ROOT / "data" / "datasets" / "maryland-polyvore")
    parser.add_argument("--checkpoint", type=Path, default=get_settings().compatibility_checkpoint)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    D = args.dataset_dir
    prepared = D / "prepared"
    if not (prepared / "metadata.json").is_file() or not (D / "embeddings").is_dir():
        print("error: dataset not prepared/embedded. Run download_dataset.py and embed_dataset.py first.", file=sys.stderr)
        return 2
    if not args.checkpoint.is_file():
        print(f"error: no checkpoint at {args.checkpoint}. Run train_compatibility.py first.", file=sys.stderr)
        return 2

    source = DatasetSource(polyvore_dir=prepared, metadata=prepared / "metadata.json", images_dir=prepared / "images")
    report, _, disjoint = audit_and_prepare(source, check_images=True, seed=args.seed)
    print("dataset gate:", report["gate"]["verdict"])
    items = {s: {x for p in disjoint[s] for x in (p[0], p[1])} for s in disjoint}
    overlap = {
        "train/valid": len(items["train"] & items["valid"]),
        "train/test": len(items["train"] & items["test"]),
        "valid/test": len(items["valid"] & items["test"]),
    }
    print("item overlap between splits:", overlap)

    store = EmbeddingDirectory(D / "embeddings")
    a, b, y, kept = store.tensors_with_index(disjoint["test"])
    pairs = [disjoint["test"][i] for i in kept]
    kinds = [p[3] for p in pairs]
    model = LearnedCompatibility.load(args.checkpoint, store.encoder_config, device=resolve_device("auto"))
    cats = load_item_categories(prepared / "metadata.json") or {}

    def slot(i: str) -> str | None:
        return CATEGORY_TO_SLOT.get(cats.get(i, "").lower())

    rng = np.random.default_rng(args.seed)
    scores = {
        "random": rng.random(len(y)),
        "clip_cosine": np.sum(a * b, axis=1),
        "heuristic_visual_category": _heuristic_scores(a, b, [slot(p[0]) for p in pairs], [slot(p[1]) for p in pairs]),
        "learned": model.predict(a, b),
    }
    print(f"test pairs: {len(y)} (positives {int(y.sum())})")
    for name, s in scores.items():
        hard = positives_vs_kind_auc(y, s, kinds, "hard_negative") or {"roc_auc": float("nan")}
        print(
            f"  {name:<28} ROC-AUC {roc_auc_score(y, s):.4f}  PR-AUC {average_precision_score(y, s):.4f}  "
            f"positives vs hard negatives {hard['roc_auc']:.4f}"
        )
    saved = model.metrics or {}
    print(
        "stored in checkpoint: learned",
        round(saved["test"]["roc_auc"], 4),
        "95% CI",
        [round(x, 4) for x in saved["test"]["roc_auc_ci95"]],
        "| clip",
        round(saved["baselines"]["clip_cosine"]["roc_auc"], 4),
        "| heuristic",
        round(saved["baselines"]["heuristic_visual_category"]["roc_auc"], 4),
        "| random",
        round(saved["baselines"]["random"]["roc_auc"], 4),
        "| n",
        saved["test"]["n"],
    )
    return 0 if all(v == 0 for v in overlap.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
