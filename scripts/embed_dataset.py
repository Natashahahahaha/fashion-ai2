"""Pre-compute CLIP embeddings for a dataset's item images (for training).

Uses the same encoder (CLIP_MODEL) as the app and writes one
``<item_id>.npy`` per image plus ``encoder.json`` describing the embedding
space. Item id = image file stem. Existing embeddings are skipped, so the
script can be resumed.

    python scripts/embed_dataset.py --images-dir data/raw/polyvore_outfits/images \
        --out data/processed/polyvore/embeddings
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from src.embeddings.clip_encoder import get_encoder
from src.errors import FashionAIError, InvalidImageError
from src.imaging import ALLOWED_EXTENSIONS, load_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="embed at most N images (quick tests)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not args.images_dir.is_dir():
        print(f"error: images directory not found: {args.images_dir}", file=sys.stderr)
        return 1
    files = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in ALLOWED_EXTENSIONS)
    if args.limit:
        files = files[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        enc = get_encoder()
        cfg_path = args.out / "encoder.json"
        if cfg_path.exists():
            existing = json.loads(cfg_path.read_text(encoding="utf-8"))
            if existing.get("model_name") != enc.model_name or existing.get("dim") != enc.dim:
                print(f"error: {args.out} holds embeddings from {existing}; refusing to mix encoders.", file=sys.stderr)
                return 1
        cfg_path.write_text(json.dumps(enc.config(), indent=2), encoding="utf-8")
        todo = [p for p in files if not (args.out / f"{p.stem}.npy").exists()]
        print(f"{len(files)} images, {len(todo)} to embed with {enc.model_name} on {enc.device}")
        failed = 0
        for start in range(0, len(todo), args.batch_size):
            batch, images = [], []
            for p in todo[start : start + args.batch_size]:
                try:
                    images.append(load_image(p))
                    batch.append(p)
                except InvalidImageError as exc:
                    failed += 1
                    logging.warning("skip %s: %s", p.name, exc)
            if not batch:
                continue
            vecs = enc.encode_images(images)
            for p, v in zip(batch, vecs, strict=True):
                np.save(args.out / f"{p.stem}.npy", v.astype(np.float32), allow_pickle=False)
            print(f"  {min(start + args.batch_size, len(todo))}/{len(todo)}")
    except FashionAIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"done ({failed} unreadable image(s) skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
