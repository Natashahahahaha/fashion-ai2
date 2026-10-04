"""Train the learned compatibility model on REAL compatibility data.

Polyvore Outfits:
    python scripts/train_compatibility.py \
        --polyvore-dir data/raw/polyvore_outfits/nondisjoint \
        --metadata data/raw/polyvore_outfits/polyvore_item_metadata.json \
        --embeddings-dir data/processed/polyvore/embeddings

Generic pairs CSV (item_a,item_b,label[,split]):
    python scripts/train_compatibility.py --pairs-csv pairs.csv \
        --embeddings-dir data/processed/mydata/embeddings

The checkpoint is written to CHECKPOINT_DIR/compatibility_model.pt, the same
file the app loads. The metrics printed at the end are computed on the
held-out test split of the dataset you supplied.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from src.config import get_settings, resolve_device
from src.errors import FashionAIError
from src.training.data import EmbeddingDirectory
from src.training.train import TrainConfig, train_compatibility


def main() -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--polyvore-dir", type=Path, help="directory with train.json / valid.json / test.json")
    source.add_argument("--pairs-csv", type=Path, help="CSV with item_a,item_b,label[,split]")
    parser.add_argument("--metadata", type=Path, help="polyvore_item_metadata.json (enables hard negatives)")
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, help="dataset images (enables corrupt/duplicate checks in the audit)")
    parser.add_argument("--manifest", type=Path, help="dataset manifest written by scripts/download_dataset.py")
    parser.add_argument("--audit-dir", type=Path, default=None, help="where dataset_audit.json/.md go (default: artifacts/)")
    parser.add_argument("--max-audit-images", type=int, default=None, help="audit a deterministic sample of N images")
    parser.add_argument("--out", type=Path, default=s.compatibility_checkpoint)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-item-overlap",
        action="store_true",
        help="keep valid/test pairs whose items also appear in training (NOT recommended; inflates metrics)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.polyvore_dir is None and args.pairs_csv is None:
        print(
            "error: no training dataset given. Pass --polyvore-dir (Polyvore Outfits split JSONs) or --pairs-csv "
            "(item_a,item_b,label[,split]). See README 'Training the compatibility model'. Training on "
            "synthetic/random data is intentionally not supported.",
            file=sys.stderr,
        )
        return 2
    try:
        store = EmbeddingDirectory(args.embeddings_dir)
        cfg = TrainConfig(
            embeddings_dir=args.embeddings_dir,
            out_path=args.out,
            polyvore_dir=args.polyvore_dir,
            polyvore_metadata=args.metadata,
            pairs_csv=args.pairs_csv,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            seed=args.seed,
            allow_item_overlap=args.allow_item_overlap,
            images_dir=args.images_dir,
            manifest_path=args.manifest,
            audit_dir=args.audit_dir or (_bootstrap.ROOT / "artifacts"),
            max_audit_images=args.max_audit_images,
            device=resolve_device(s.device),
            # Embeddings must come from the CLIP model the app uses.
            expected_encoder={"model_name": s.clip_model, "dim": store.encoder_config.get("dim")},
        )
        result = train_compatibility(cfg)
    except FashionAIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    test = result["metrics"]["test"]
    print(f"Dataset gate: {result['audit']['gate']['verdict']}")
    print(
        "\nMetrics on the held-out TEST split of the dataset you supplied (model from the best epoch "
        "by validation ROC-AUC). They describe this dataset only, not fashion compatibility in general:"
    )
    overlap = result["training"].get("overlap_pairs_dropped")
    if overlap:
        print(f"(item-disjoint evaluation: dropped {overlap['valid']} valid / {overlap['test']} test pairs sharing items)")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in test.items()}, indent=2))
    print("\nBaselines on the same test pairs (ROC-AUC):")
    for name, b in result["metrics"]["baselines"].items():
        print(f"  {name:<28} {b['roc_auc']:.4f}  CI95 {b['roc_auc_ci95']}")
    cmp_ = result["metrics"]["comparison"]
    print(f"  learned model                {test['roc_auc']:.4f}  CI95 {test['roc_auc_ci95']}")
    print(f"Learned - {cmp_['strongest_non_random_baseline']}: {cmp_['learned_minus_baseline_roc_auc']}")
    if cmp_["beats_best_baseline"]:
        print("The learned model beat the strongest baseline (95% CI of the difference excludes 0).")
    else:
        print("The learned model did NOT beat the strongest baseline; the app keeps using the baseline (USE_LEARNED_MODEL=auto).")
    print(f"checkpoint: {result['checkpoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
