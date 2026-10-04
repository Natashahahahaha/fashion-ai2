r"""
Run ONCE, locally, to build a small demo wardrobe for the deployed app —
keeps the Space small instead of bundling your full 105MB metadata file
and 205K embeddings.

Run from your project root:
    python final_deploy\select_demo_wardrobe.py --metadata data\Polyvore\polyvore_item_metadata.json --embeddings-dir data\Polyvore\embeddings --images-dir data\Polyvore\images --out final_deploy\demo_wardrobe --per-category 6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out", default="demo_wardrobe")
    parser.add_argument("--per-category", type=int, default=6)
    parser.add_argument("--max-categories", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.metadata) as f:
        meta = json.load(f)

    by_category = defaultdict(list)
    for item_id, info in meta.items():
        emb_path = os.path.join(args.embeddings_dir, f"{item_id}.pt")
        img_path = os.path.join(args.images_dir, f"{item_id}.jpg")
        if not (os.path.exists(emb_path) and os.path.exists(img_path)):
            continue
        category = info.get("semantic_category") or info.get("category_id") or "unknown"
        by_category[category].append(item_id)

    print(f"[select] found {len(by_category)} categories with usable items")
    for cat, items in sorted(by_category.items(), key=lambda kv: -len(kv[1]))[:15]:
        print(f"  {cat}: {len(items)} items")

    rng = random.Random(args.seed)
    top_categories = sorted(by_category.items(), key=lambda kv: -len(kv[1]))[: args.max_categories]

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "embeddings"), exist_ok=True)

    selected_meta = {}
    for category, items in top_categories:
        chosen = rng.sample(items, min(args.per_category, len(items)))
        for item_id in chosen:
            shutil.copy(os.path.join(args.images_dir, f"{item_id}.jpg"), os.path.join(args.out, "images", f"{item_id}.jpg"))
            shutil.copy(os.path.join(args.embeddings_dir, f"{item_id}.pt"), os.path.join(args.out, "embeddings", f"{item_id}.pt"))
            info = meta[item_id]
            selected_meta[item_id] = {
                "category": category,
                "title": info.get("title") or info.get("url_name") or item_id,
            }

    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(selected_meta, f, indent=2)

    print(f"[select] wrote {len(selected_meta)} items across {len(top_categories)} categories to {args.out}/")
    counts = defaultdict(int)
    for v in selected_meta.values():
        counts[v["category"]] += 1
    for cat, n in counts.items():
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
