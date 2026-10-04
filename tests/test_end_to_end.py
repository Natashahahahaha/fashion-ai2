"""Image -> detector -> crops -> embeddings -> wardrobe -> recommendations -> outfits.

The detector backend and CLIP are fakes; everything else is production code.
"""

from __future__ import annotations

from pathlib import Path

from conftest import FakeBackend
from src.detection.detector import FashionDetector, RawDetection
from src.service import FashionAIService


def test_full_pipeline(settings, encoder, scene):
    path, boxes = scene
    raw = [
        RawDetection("shirt", 0.91, boxes["shirt"]),
        RawDetection("jeans", 0.88, boxes["jeans"]),
        RawDetection("sneakers", 0.76, boxes["sneakers"]),
        RawDetection("jacket", 0.55, boxes["jacket"]),
    ]
    detector = FashionDetector(settings, backend=FakeBackend(raw))
    svc = FashionAIService(settings, encoder=encoder, detector=detector)

    detections, items = svc.manager.ingest_image(path, svc.detector)
    assert len(detections) == 4 and len(items) == 4
    assert {i.category for i in items} == {"top", "bottom", "shoes", "outerwear"}
    for det, item in zip(detections, items, strict=True):
        assert item.metadata["bbox"] == list(det.bbox)
        assert item.metadata["label_source"] == "detector"
        assert Path(item.metadata["source_image"]).is_file()
        assert svc.manager.load_embedding(item).shape == (encoder.dim,)
    by_cat = {i.category: i for i in items}
    assert by_cat["bottom"].metadata["color"] == "navy"
    assert by_cat["shoes"].metadata["color"] == "black"

    # learned model absent -> transparent baseline
    assert svc.learned_status().state == "missing"
    gen = svc.generator()

    recs, msg = gen.recommend_for_item(by_cat["top"].id)
    assert msg is None and {r.item.category for r in recs} == {"bottom", "shoes", "outerwear"}

    res = gen.generate_outfits(k=3, style="casual", occasion="everyday")
    # synthetic colour blocks may fall below the fitted quality threshold: that must be
    # reported, never padded with a weak look
    assert res.outfits or "quality threshold" in res.message, res.message
    from src.recommendation.outfit_generator import OutfitGenerator

    res = OutfitGenerator(svc.manager, quality_threshold=0.0).generate_outfits(k=3, style="casual", occasion="everyday")
    assert res.outfits, res.message
    best = res.outfits[0]
    assert {"top", "bottom", "shoes"} <= {i.category for i in best.items}
    d = best.to_dict()
    assert {"score", "items", "components", "reasons", "caveats", "structure", "colour", "scoring"} <= set(d)
    assert d["structure"].startswith("Valid ")
    assert d["items"][0]["image_path"]

    diag = svc.diagnostics()
    assert diag["wardrobe_items"] == 4 and diag["embeddings_ok"] == 4 and diag["device"] == "cpu"

    assert svc.manager.delete_item(items[0].id)
    assert svc.manager.count() == 3


def test_user_category_override(settings, encoder, scene):
    path, boxes = scene
    detector = FashionDetector(settings, backend=FakeBackend([RawDetection("dress", 0.6, boxes["jeans"])]))
    svc = FashionAIService(settings, encoder=encoder, detector=detector)
    det = detector.detect(path)[0]
    item = svc.manager.add_detections([det], overrides={det.id: "bottom"})[0]
    assert item.category == "bottom"
    assert item.metadata["label_source"] == "user" and item.metadata["detector_label"] == "dress"
