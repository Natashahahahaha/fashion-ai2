"""Command-line ingestion: photo -> detections -> crops -> CLIP -> wardrobe.

python scripts/ingest.py path/to/photo.jpg [more.jpg ...] [--min-confidence 0.3] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys

import _bootstrap  # noqa: F401

from src.errors import FashionAIError
from src.service import FashionAIService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="image file(s) to ingest")
    parser.add_argument("--min-confidence", type=float, default=None, help="only add detections at/above this confidence")
    parser.add_argument("--dry-run", action="store_true", help="detect only, do not add to the wardrobe")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    svc = FashionAIService()
    try:
        for path in args.images:
            dets = svc.detector.detect(path)
            print(f"{path}: {len(dets)} detection(s)")
            for d in dets:
                print(f"  {d.label:<10} {d.category:<10} conf={d.confidence:.2f} bbox={list(d.bbox)}")
            keep = [d for d in dets if args.min_confidence is None or d.confidence >= args.min_confidence]
            if args.dry_run or not keep:
                continue
            for item in svc.manager.add_detections(keep):
                print(f"  + added {item.id}: {item.display_name} ({item.category})")
    except FashionAIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wardrobe now has {svc.manager.count()} item(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
