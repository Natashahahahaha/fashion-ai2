"""End-to-end smoke test with the REAL models (YOLO-World + CLIP).

    python scripts/smoke_test.py                 # downloads two public sample photos
    python scripts/smoke_test.py my1.jpg my2.jpg  # or use your own photos

Runs: image -> detector -> crops -> CLIP embeddings -> wardrobe insertion ->
item recommendations -> outfit generation, in a temporary data directory
(your real wardrobe is not touched). Exits non-zero on any failure.
First run downloads model weights (~25 MB YOLO-World, ~340 MB YOLO-World's
CLIP text encoder, ~600 MB Hugging Face CLIP).
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from src.config import get_settings
from src.service import FashionAIService

SAMPLES = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
}


def fetch_samples(dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for name, url in SAMPLES.items():
        p = dest / name
        if not p.exists():
            print(f"downloading sample {url}")
            urllib.request.urlretrieve(url, p)
        out.append(p)
    return out


def check(cond: bool, msg: str) -> None:
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    base = get_settings()
    images = args.images or fetch_samples(base.data_dir / "samples")
    with tempfile.TemporaryDirectory(prefix="fashion_smoke_") as tmp:
        # Same models/checkpoints as the app, throwaway wardrobe.
        settings = dataclasses.replace(base, data_dir=Path(tmp))
        svc = FashionAIService(settings)
        print(f"device={svc.device}  clip={settings.clip_model}  detector={settings.detector_model}")

        t = time.time()
        all_items = []
        for img in images:
            print(f"\n[1] detect {img}")
            dets = svc.detector.detect(img)
            for d in dets:
                alts = ", ".join(f"{a.label} {a.confidence:.2f}" for a in d.alternatives[:3])
                flags = ",".join(f.code for f in d.flags)
                print(
                    f"     {d.label:<10} {d.category:<10} {d.confidence:.2f} {list(d.bbox)}"
                    + (f"  alt[{alts}]" if alts else "")
                    + (f"  REVIEW[{flags}] -> {d.suggested_category}" if flags else "")
                )
            check(
                all(set(d.to_dict()) >= {"id", "label", "confidence", "bbox", "crop_path"} for d in dets),
                "canonical detection schema",
            )
            check(all(d.bbox[0] < d.bbox[2] and d.bbox[1] < d.bbox[3] for d in dets), "valid bounding boxes")
            check(all(d.crop_path and Path(d.crop_path).is_file() for d in dets), "crops written to disk")
            print("[2] embed + store")
            items = svc.manager.add_detections(dets)
            all_items += items
            for it in items:
                emb = svc.manager.load_embedding(it)
                assert emb is not None
                check(
                    emb.shape == (svc.encoder.dim,) and abs(float(np.linalg.norm(emb)) - 1) < 1e-3,
                    f"{it.label}: {emb.shape} unit-norm embedding",
                )
        check(len(all_items) > 0, f"{len(all_items)} items in wardrobe")
        print(f"   detection+embedding took {time.time() - t:.1f}s")

        gen = svc.generator()
        print(f"\n[3] scoring mode: {svc.learned_status().state} ({svc.learned_status().message})")
        from src.recommendation.validity import is_valid_outfit
        from src.wardrobe.manager import needs_review

        flagged = [it for it in all_items if needs_review(it)]
        print(f"     {len(flagged)} item(s) need review and are excluded: " + ", ".join(it.label for it in flagged))
        anchor = next((it for it in all_items if not needs_review(it)), all_items[0])
        recs, msg = gen.recommend_for_item(anchor.id, k=5)
        print(f"[4] recommendations for {anchor.display_name}: {msg or ''}")
        for r in recs:
            print(f"     {r.pair.score:.3f}  {r.item.display_name} ({r.item.category})")
        scores = [r.pair.score for r in recs]
        check(scores == sorted(scores, reverse=True), "recommendations ranked")

        print("[5] outfits")
        res = gen.generate_outfits(k=5)
        if res.message:
            print(f"     note: {res.message}")
        for o in res.outfits:
            print(f"     {o.score:.3f}  " + " + ".join(f"{i.label}" for i in o.items))
            for reason in o.result.reasons:
                print(f"            - {reason}")
        diag = res.diagnostics
        print(
            f"     funnel: {diag.get('wardrobe_items')} items -> {diag.get('eligible_items')} eligible -> "
            f"{diag.get('structural_combinations', 0)} structural -> {diag.get('candidates_scored', 0)} valid scored -> "
            f"{diag.get('above_threshold', 0)} above threshold -> {diag.get('returned', 0)} returned"
        )
        check(all(is_valid_outfit(o.items).valid for o in res.outfits), f"{len(res.outfits)} outfit(s), all pass validity")
        check(not any(needs_review(i) for o in res.outfits for i in o.items), "no item awaiting review is used in an outfit")
        check(len(res.outfits) > 0 or bool(res.message), "outfits generated, or an explanation why not")

        print("[6] delete")
        check(svc.manager.delete_item(anchor.id) and svc.manager.get_item(anchor.id) is None, "item deleted")
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
