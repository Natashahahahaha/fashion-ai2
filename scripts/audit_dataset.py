"""Audit a compatibility dataset and decide whether it can support a credible model.

    python scripts/audit_dataset.py --polyvore-dir data/raw/polyvore_outfits/disjoint \
        --metadata data/raw/polyvore_outfits/polyvore_item_metadata.json \
        --images-dir data/raw/polyvore_outfits/images

Writes artifacts/dataset_audit.json (machine-readable) and artifacts/dataset_audit.md.
Exit code: 0 = SUFFICIENT / SUFFICIENT_WITH_LIMITATIONS, 3 = INSUFFICIENT, 2 = error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap

from src.errors import FashionAIError
from src.training.audit import INSUFFICIENT, DatasetSource, audit_dataset, write_reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--polyvore-dir", type=Path)
    src.add_argument("--pairs-csv", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--manifest", type=Path, help="dataset manifest (name, source, license...)")
    parser.add_argument("--max-images", type=int, default=None, help="audit a deterministic sample of N images")
    parser.add_argument("--out-dir", type=Path, default=_bootstrap.ROOT / "artifacts")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest and args.manifest.is_file() else {}
    source = DatasetSource(
        polyvore_dir=args.polyvore_dir,
        pairs_csv=args.pairs_csv,
        metadata=args.metadata,
        images_dir=args.images_dir,
        name=manifest.get("name"),
        extra={k: manifest[k] for k in ("source", "url", "license", "version") if k in manifest},
    )
    try:
        report = audit_dataset(source, check_images=args.images_dir is not None, max_images=args.max_images)
    except FashionAIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_reports(report, args.out_dir / "dataset_audit.json", args.out_dir / "dataset_audit.md")
    gate = report["gate"]
    print(f"Verdict: {gate['verdict']}")
    for f in gate["failures"]:
        print(f"  BLOCKING: {f}")
    for lim in gate["limitations"]:
        print(f"  limitation: {lim}")
    eff = report["effective"]["after_item_disjoint_split"]
    for split in ("train", "valid", "test"):
        e = eff[split]
        print(f"  {split:<5} items={e['items']:>7} positives={e['positive_pairs']:>8} negatives={e['negative_pairs']:>8}")
    print(f"Reports: {args.out_dir / 'dataset_audit.json'} and dataset_audit.md")
    return 3 if gate["verdict"] == INSUFFICIENT else 0


if __name__ == "__main__":
    sys.exit(main())
