"""Opt-in tests against the REAL YOLO-World and CLIP weights.

    RUN_MODEL_TESTS=1 pytest tests/test_real_models.py

Skipped by default because they download ~1 GB of weights on first run.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

from src.config import get_settings

pytestmark = pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1 to run")

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "samples" / "bus.jpg"


@pytest.fixture(scope="module")
def sample() -> Path:
    if not SAMPLE.exists():
        import urllib.request

        SAMPLE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", SAMPLE)
    return SAMPLE


def test_real_clip_dimension_and_norm(sample):
    from src.embeddings.clip_encoder import ClipEncoder

    enc = ClipEncoder(settings=get_settings())
    v = enc.encode_image(sample)
    assert enc.dim == 512 and v.shape == (512,)
    assert abs(float(np.linalg.norm(v)) - 1) < 1e-4


def test_real_detector_finds_clothes(sample, tmp_path):
    from src.detection.detector import FashionDetector

    s = dataclasses.replace(get_settings(), data_dir=tmp_path)
    dets = FashionDetector(s).detect(sample)
    assert dets, "expected at least one garment in bus.jpg"
    assert all(Path(d.crop_path).is_file() for d in dets)


def test_real_bus_jumpsuit_is_flagged_and_excluded(sample, tmp_path):
    """Real-image sanity check for the reported failure: in bus.jpg YOLO-World labels a
    person in coat + jeans 'jumpsuit'. It must be flagged, offer the coat alternative,
    and never reach outfit generation unreviewed."""
    from src.service import FashionAIService
    from src.wardrobe.manager import needs_review

    s = dataclasses.replace(get_settings(), data_dir=tmp_path)
    svc = FashionAIService(s)
    dets = svc.detector.detect(sample)
    one_pieces = [d for d in dets if d.category == "one_piece"]
    assert one_pieces, [(d.label, d.confidence) for d in dets]
    assert all(d.needs_review for d in one_pieces)
    assert any(a.category == "outerwear" for d in one_pieces for a in d.alternatives)
    items = svc.manager.add_detections(dets)
    res = svc.generator().generate_outfits(k=5)
    used = {i.id for o in res.outfits for i in o.items}
    assert not any(needs_review(it) and it.id in used for it in items)
    assert not any(i.category == "one_piece" for o in res.outfits for i in o.items)
