"""Build a test wardrobe from real product photos (Polyvore TEST split) for manual/browser checks.

    python scripts/demo_wardrobe.py --data-dir data/demo29            # 8/6/5/4/3/3 = 29 items
    python scripts/demo_wardrobe.py --data-dir data/demo50 --scale 2   # twice as many

Items are drawn deterministically (``--seed``) from outfits in the test split,
so none of them were used to fit the ranker (validation split) or train the
learned model (train split). Images go through the normal ``add_item`` path
(CLIP embedding, colours, style estimate). The target directory must be empty
or not exist yet; an existing wardrobe is never modified.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import _bootstrap
import numpy as np

from src.config import get_settings
from src.service import FashionAIService
from src.training.data import load_polyvore_outfits

COUNTS = {"top": 8, "bottom": 6, "shoes": 5, "outerwear": 4, "one_piece": 3, "accessory": 3}
# accessories that read as one item in a photo (no jewellery sets / sunglasses close-ups)
ACCESSORY_FINE = {"Shoulder Bags", "Tote Bags", "Handbags", "Clutches", "Backpacks", "Hats", "Scarves", "Belts"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=_bootstrap.ROOT / "data" / "datasets" / "maryland-polyvore")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.data_dir.exists() and any(args.data_dir.iterdir()):
        print(f"error: {args.data_dir} is not empty; choose a new directory.", file=sys.stderr)
        return 2
    prepared = args.dataset_dir / "prepared"
    meta = json.loads((prepared / "metadata.json").read_text(encoding="utf-8"))
    test_items = sorted({i for o in load_polyvore_outfits(prepared / "test.json") for i in o})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(test_items)

    picked: dict[str, list[str]] = {c: [] for c in COUNTS}
    for iid in test_items:
        m = meta.get(iid) or {}
        cat = m.get("semantic_category")
        if cat not in picked or len(picked[cat]) >= COUNTS[cat] * args.scale:
            continue
        if cat == "accessory" and m.get("fine_category") not in ACCESSORY_FINE:
            continue
        if not (prepared / "images" / f"{iid}.jpg").is_file():
            continue
        picked[cat].append(iid)
    short = {c: COUNTS[c] * args.scale - len(v) for c, v in picked.items() if len(v) < COUNTS[c] * args.scale}
    if short:
        print(f"error: not enough test items for {short}", file=sys.stderr)
        return 1

    settings = dataclasses.replace(get_settings(), data_dir=args.data_dir)
    settings.ensure_dirs()
    svc = FashionAIService(settings)
    n = 0
    for cat, ids in picked.items():
        for iid in ids:
            m = meta[iid]
            label = str(m.get("fine_category") or cat).lower()
            svc.manager.add_item(
                prepared / "images" / f"{iid}.jpg",
                category=cat,
                label=label,
                extra_metadata={"polyvore_id": iid, "title": m.get("title", "")},
            )
            n += 1
            print(f"  {n:>3}  {cat:<10} {label:<20} {m.get('title', '')[:60]}")
    print(f"\n{n} items in {args.data_dir} ({', '.join(f'{len(v)} {k}' for k, v in picked.items())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
