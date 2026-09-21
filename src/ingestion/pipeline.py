"""
End-to-end Day 1 pipeline: one closet-dump photo -> wardrobe_items.json

    python pipeline.py path/to/photo.jpg
    python pipeline.py path/to/photo.jpg --sam-checkpoint sam_vit_b_01ec64.pth
    python review_cli.py ./wardrobe_run/wardrobe_items.json   # correct flagged items
"""

from __future__ import annotations

import argparse
import json
import os

from segment import ClothingSegmenter
from extract_attributes import AttributeExtractor


def run_pipeline(image_path: str, output_dir: str, sam_checkpoint: str | None = None):
    crops_dir = os.path.join(output_dir, "crops")
    segmenter = ClothingSegmenter(checkpoint_path=sam_checkpoint)
    extractor = AttributeExtractor()

    crop_paths = segmenter.segment(image_path, crops_dir)
    print(f"[pipeline] {len(crop_paths)} candidate region(s) detected from {image_path}")

    items = []
    for crop_path in crop_paths:
        item = extractor.extract(crop_path)
        if item is not None:
            item.source_image = image_path
            items.append(item)

    auto_accepted = [i for i in items if not i.needs_review]
    review_queue = [i for i in items if i.needs_review]
    print(f"[pipeline] {len(auto_accepted)} auto-accepted, {len(review_queue)} flagged for review")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "wardrobe_items.json")
    with open(out_path, "w") as f:
        json.dump([i.to_dict() for i in items], f, indent=2, default=str)

    print(f"[pipeline] wrote {out_path}")
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import a closet-dump photo into structured wardrobe items.")
    parser.add_argument("image", help="Path to a closet-dump / flat-lay photo")
    parser.add_argument("--out", default="./wardrobe_run", help="Output directory")
    parser.add_argument(
        "--sam-checkpoint",
        default=None,
        help="Path to a SAM checkpoint (e.g. sam_vit_b_01ec64.pth). Omit to use the mock grid segmenter.",
    )
    args = parser.parse_args()

    run_pipeline(args.image, args.out, args.sam_checkpoint)
