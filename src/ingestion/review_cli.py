"""
Minimal human-in-the-loop correction loop for the Day 1 ingestion foundation.

This is deliberately a terminal script, not a UI — the point on Day 1 is
proving the correction *mechanism* (low-confidence items get surfaced,
corrections get persisted as structured data) works. Swap in a real
frontend later; the underlying wardrobe_items.json contract doesn't change.

    python review_cli.py ./wardrobe_run/wardrobe_items.json
"""

from __future__ import annotations

import json
import sys


def review(items_path: str) -> None:
    with open(items_path) as f:
        items = json.load(f)

    flagged = [i for i in items if i.get("needs_review")]
    if not flagged:
        print("Nothing flagged for review.")
        return

    print(f"{len(flagged)} item(s) flagged for review.\n")

    changed = False
    for item in items:
        if not item.get("needs_review"):
            continue

        print("=" * 50)
        print(f"Crop: {item['crop_path']}")
        print(f"Guess: {item['category']} / {item['color']}  (confidence {item['confidence']:.2f})")
        ans = input(
            "Enter = accept guess | 's' = skip | or correct as "
            "'category=cropped cardigan, color=dark brown': "
        ).strip()

        if ans == "" or ans.lower() == "s":
            item["needs_review"] = False
            continue

        for pair in ans.split(","):
            if "=" not in pair:
                continue
            key, value = (p.strip() for p in pair.split("=", 1))
            if key in item:
                item[key] = value

        item["needs_review"] = False
        item["human_corrected"] = True
        changed = True
        print(f"  -> saved: {json.dumps({k: item[k] for k in ('category', 'color') if k in item})}")

    if changed:
        with open(items_path, "w") as f:
            json.dump(items, f, indent=2)
        print(f"\nCorrections saved to {items_path}")
    else:
        print("\nNo corrections made.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "./wardrobe_run/wardrobe_items.json"
    review(path)
