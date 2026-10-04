"""
See what an outfit actually looks like instead of squinting at item IDs.
Takes any number of item IDs and saves one side-by-side image with each
item's category/title underneath it.

    python visualize_outfit.py 211990161 183179503 --out outfit_check.png

Works for a pair from recommend.py, or a full combo from recommend_topk.py
— just pass however many IDs you have.
"""

from __future__ import annotations

import argparse
import json
import os

from PIL import Image, ImageDraw


def load_metadata(path):
    with open(path) as f:
        return json.load(f)


def get_label(meta, item_id):
    info = meta.get(item_id, {})
    category = info.get("semantic_category") or info.get("category_id") or "?"
    title = info.get("title") or info.get("url_name") or ""
    label = f"{item_id}\n{category}"
    if title:
        label += f"\n{title[:30]}"
    return label


def build_collage(item_ids, images_dir, meta, thumb_height=300, label_height=70):
    thumbs, labels = [], []
    for item_id in item_ids:
        img_path = os.path.join(images_dir, f"{item_id}.jpg")
        if not os.path.exists(img_path):
            print(f"[visualize] WARNING: no image for {item_id} at {img_path} — skipping")
            continue
        img = Image.open(img_path).convert("RGB")
        ratio = thumb_height / img.height
        img = img.resize((int(img.width * ratio), thumb_height))
        thumbs.append(img)
        labels.append(get_label(meta, item_id))

    if not thumbs:
        raise SystemExit("No valid images found for any given item ID — check the IDs and --images-dir path.")

    total_width = sum(t.width for t in thumbs) + 10 * (len(thumbs) - 1)
    canvas = Image.new("RGB", (total_width, thumb_height + label_height), "white")
    draw = ImageDraw.Draw(canvas)

    x = 0
    for img, label in zip(thumbs, labels):
        canvas.paste(img, (x, 0))
        draw.multiline_text((x + 5, thumb_height + 5), label, fill="black")
        x += img.width + 10

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("item_ids", nargs="+", help="Item IDs to display side by side")
    parser.add_argument("--images-dir", default="data/Polyvore/images")
    parser.add_argument("--metadata", default="data/Polyvore/polyvore_item_metadata.json")
    parser.add_argument("--out", default="outfit_check.png")
    args = parser.parse_args()

    meta = load_metadata(args.metadata)
    collage = build_collage(args.item_ids, args.images_dir, meta)
    collage.save(args.out)
    print(f"[visualize] saved {args.out} — open it in File Explorer to see the outfit")


if __name__ == "__main__":
    main()