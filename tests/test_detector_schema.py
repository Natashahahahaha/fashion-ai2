from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import FakeBackend, make_detector
from src.detection.categories import DETECTOR_PROMPTS, LABEL_TO_SLOT, SLOTS
from src.detection.detector import FashionDetector, RawDetection
from src.detection.schemas import DetectionResult, clip_bbox
from src.errors import DetectorUnavailableError, InvalidImageError

REQUIRED_FIELDS = {"id", "label", "confidence", "bbox", "crop_path"}


def test_detect_returns_canonical_schema(settings, scene):
    path, boxes = scene
    raw = [RawDetection("shirt", 0.9, boxes["shirt"]), RawDetection("jeans", 0.8, boxes["jeans"])]
    dets = make_detector(settings, raw).detect(path)

    assert len(dets) == 2
    for d in dets:
        assert isinstance(d, DetectionResult)
        assert REQUIRED_FIELDS <= set(d.to_dict())
        assert d.category in SLOTS
        assert 0.0 <= d.confidence <= 1.0
        assert all(isinstance(v, int) for v in d.bbox)
        assert d.source_image == str(path)
    assert [d.label for d in dets] == ["shirt", "jeans"]
    assert dets[1].category == "bottom"
    assert DetectionResult.from_dict(dets[0].to_dict()) == dets[0]


def test_crops_written_and_match_boxes(settings, scene):
    path, boxes = scene
    det = make_detector(settings, [RawDetection("jeans", 0.7, boxes["jeans"])], crop_padding=0.0).detect(path)[0]
    assert det.crop_path and Path(det.crop_path).is_file()
    with Image.open(det.crop_path) as crop:
        x1, y1, x2, y2 = det.bbox
        assert crop.size == (x2 - x1, y2 - y1)
        # crop really is the jeans region (dark blue), not some other area
        r, g, b = np.asarray(crop.convert("RGB")).reshape(-1, 3).mean(0)
        assert b > r + 30


def test_boxes_are_clipped_and_invalid_boxes_dropped(settings, scene):
    path, _ = scene  # image is 400x600
    raw = [
        RawDetection("coat", 0.9, (-50, -20, 150, 300)),  # partially outside -> clipped
        RawDetection("skirt", 0.9, (390, 10, 500, 100)),  # 10px wide after clipping -> dropped
        RawDetection("dress", 0.9, (200, 200, 200, 400)),  # zero width -> dropped
        RawDetection("shoes", 0.9, (float("nan"), 0, 50, 50)),  # malformed -> dropped
        RawDetection("pants", 0.9, (300, 500, 100, 300)),  # inverted corners -> normalised
    ]
    dets = make_detector(settings, raw).detect(path)
    by_label = {d.label: d for d in dets}
    assert set(by_label) == {"coat", "pants"}
    assert by_label["coat"].bbox == (0, 0, 150, 300)
    assert by_label["pants"].bbox == (100, 300, 300, 500)
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        assert 0 <= x1 < x2 <= 400 and 0 <= y1 < y2 <= 600


def test_threshold_unknown_labels_and_sorting(settings, scene):
    path, boxes = scene
    raw = [
        RawDetection("shirt", 0.40, boxes["shirt"]),
        RawDetection("person", 0.99, boxes["jeans"]),  # not in our vocabulary -> never emitted
        RawDetection("jacket", 0.10, boxes["jacket"]),  # below threshold
        RawDetection("Sneakers", 0.95, boxes["sneakers"]),  # case-normalised
    ]
    dets = make_detector(settings, raw, confidence_threshold=0.25).detect(path)
    assert [d.label for d in dets] == ["sneakers", "shirt"]
    assert dets[0].confidence >= dets[1].confidence


def test_empty_detection(settings, scene):
    path, _ = scene
    assert make_detector(settings, []).detect(path) == []


def test_accepts_pil_and_rgb_array(settings, scene):
    path, boxes = scene
    img = Image.open(path).convert("RGB")
    seen = []

    class Spy(FakeBackend):
        def __call__(self, image):
            seen.append(image)
            return super().__call__(image)

    det = FashionDetector(settings, backend=Spy([RawDetection("jeans", 0.9, boxes["jeans"])]))
    assert len(det.detect(img)) == 1
    assert len(det.detect(np.asarray(img))) == 1
    # backend always receives an RGB PIL image (Ultralytics would treat numpy as BGR)
    assert all(isinstance(s, Image.Image) and s.mode == "RGB" for s in seen)
    # jeans pixel stays blue after the array path, i.e. no RGB/BGR swap
    assert seen[1].getpixel((200, 350))[2] > seen[1].getpixel((200, 350))[0]


def test_invalid_inputs(settings, tmp_path):
    det = make_detector(settings, [])
    with pytest.raises(InvalidImageError):
        det.detect(tmp_path / "missing.jpg")
    bad = tmp_path / "not_an_image.jpg"
    bad.write_bytes(b"definitely not a jpeg")
    with pytest.raises(InvalidImageError):
        det.detect(bad)
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    with pytest.raises(InvalidImageError, match="Unsupported file type"):
        det.detect(txt)


def test_ids_are_stable_and_redetect_replaces_crops(settings, scene):
    path, boxes = scene
    det = make_detector(settings, [RawDetection("jeans", 0.9, boxes["jeans"]), RawDetection("shirt", 0.8, boxes["shirt"])])
    first = det.detect(path)
    stray = Path(first[0].crop_path).parent / "stale.jpg"
    stray.write_bytes(b"x")
    second = det.detect(path)
    assert [d.id for d in first] == [d.id for d in second]  # deterministic ids
    assert len({d.id for d in second}) == 2
    assert all(Path(d.crop_path).is_file() for d in second)
    assert not stray.exists()  # the crop folder is rebuilt, no leftovers


def test_malformed_backend_output_is_skipped_not_raised(settings, scene):
    path, boxes = scene

    class Weird:
        label = "shirt"
        confidence = "very high"
        xyxy = (0, 0, 10, 10)

    raw = [
        RawDetection("jeans", 0.9, boxes["jeans"]),
        Weird(),  # non-numeric confidence
        RawDetection("shirt", float("nan"), boxes["shirt"]),
        RawDetection("shirt", 1.7, boxes["shirt"]),  # confidence outside [0, 1]
        RawDetection("shirt", 0.8, (1, 2, 3)),  # wrong number of coordinates
        RawDetection("jeans", 0.9, boxes["jeans"]),  # exact duplicate
        None,
    ]
    det = make_detector(settings, raw)
    out = det.detect(path)
    assert [d.label for d in out] == ["jeans"]
    assert det.last_dropped == 6


def test_backend_crash_becomes_readable_error(settings, scene):
    from src.errors import DetectionError

    def boom(image):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(DetectionError, match="detector failed"):
        FashionDetector(settings, backend=boom).detect(scene[0])


def test_large_image_boxes_mapped_back_to_original(settings, tmp_path):
    big = Image.new("RGB", (6000, 4000), (255, 255, 255))
    big.paste(Image.new("RGB", (1500, 2000), (30, 50, 110)), (3000, 1000))
    seen = []

    def backend(image):
        seen.append(image.size)
        s = image.width / 6000
        return [RawDetection("jeans", 0.9, (3000 * s, 1000 * s, 4500 * s, 3000 * s))]

    out = FashionDetector(settings, backend=backend, crop_padding=0.0).detect(big)
    assert max(seen[0]) <= 2048  # inference on a downscaled copy
    x1, y1, x2, y2 = out[0].bbox
    assert abs(x1 - 3000) <= 3 and abs(y1 - 1000) <= 3 and abs(x2 - 4500) <= 3 and abs(y2 - 3000) <= 3
    with Image.open(out[0].crop_path) as crop:
        assert crop.size[0] > 1400  # crop taken from the full-resolution original


def test_tiny_image_rejected(settings):
    with pytest.raises(InvalidImageError, match="too small"):
        make_detector(settings, []).detect(Image.new("RGB", (4, 4)))


def test_missing_backend_package_reports_unavailable(settings, monkeypatch, scene):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "ultralytics":
            raise ImportError("no ultralytics")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    det = FashionDetector(settings)
    with pytest.raises(DetectorUnavailableError, match="ultralytics"):
        det.detect(scene[0])
    assert det.load_error


def test_vocabulary_is_consistent():
    assert set(DETECTOR_PROMPTS) == set(LABEL_TO_SLOT)
    assert set(LABEL_TO_SLOT.values()) <= set(SLOTS)


@pytest.mark.parametrize(
    "bbox,expected",
    [((10, 10, 50, 50), (10, 10, 50, 50)), ((-5, -5, 30, 30), (0, 0, 30, 30)), ((0, 0, 5, 50), None), (("a", 0, 1, 1), None)],
)
def test_clip_bbox(bbox, expected):
    assert clip_bbox(bbox, 100, 100) == expected


def test_decoder_failures_of_any_type_become_invalid_image(settings, tmp_path, monkeypatch):
    """Ultralytics patches Image.open; undecodable files can then raise ImportError (pi_heif)."""
    import PIL.Image

    bad = tmp_path / "x.jpg"
    bad.write_bytes(b"garbage")

    def patched_open(*a, **k):
        raise ModuleNotFoundError("No module named 'pi_heif'")

    monkeypatch.setattr(PIL.Image, "open", patched_open)
    with pytest.raises(InvalidImageError, match="not a recognised image"):
        make_detector(settings, []).detect(bad)


def test_truncated_jpeg_rejected(settings, scene, tmp_path):
    data = scene[0].read_bytes()
    cut = tmp_path / "cut.jpg"
    cut.write_bytes(data[: len(data) // 3])
    with pytest.raises(InvalidImageError):
        make_detector(settings, []).detect(cut)


# ------------------------------------------------- regions, alternatives, review
def test_overlapping_labels_become_one_region_with_alternatives(settings, scene):
    path, boxes = scene
    x1, y1, x2, y2 = boxes["jacket"]
    raw = [
        RawDetection("jacket", 0.55, boxes["jacket"]),
        RawDetection("coat", 0.30, (x1 + 2, y1 + 2, x2, y2)),
        RawDetection("blazer", 0.12, boxes["jacket"]),  # below display floor alone, kept as evidence
    ]
    dets = make_detector(settings, raw).detect(path)
    assert len(dets) == 1
    d = dets[0]
    assert d.label == "jacket" and [a.label for a in d.alternatives] == ["coat", "blazer"]
    assert not d.needs_review  # all alternatives are the same slot: no category ambiguity
    assert DetectionResult.from_dict(d.to_dict()) == d


def test_one_piece_containing_separates_is_flagged_with_suggestion(settings, scene):
    path, boxes = scene
    person = (100, 40, 300, 480)
    raw = [RawDetection("jumpsuit", 0.72, person), RawDetection("coat", 0.20, person), RawDetection("jeans", 0.6, boxes["jeans"])]
    dets = {d.label: d for d in make_detector(settings, raw).detect(path)}
    js = dets["jumpsuit"]
    assert js.needs_review and "contains_separates" in {f.code for f in js.flags}
    assert js.suggested_category == "outerwear"
    assert not dets["jeans"].needs_review


def test_ambiguous_and_low_confidence_regions_are_flagged_not_dropped(settings, scene):
    path, boxes = scene
    raw = [
        RawDetection("shirt", 0.40, boxes["shirt"]),
        RawDetection("dress", 0.35, boxes["shirt"]),  # different slot, 0.35 >= 0.6 * 0.40
        RawDetection("sneakers", 0.18, boxes["sneakers"]),  # above floor 0.15, below accept 0.25
    ]
    dets = {d.label: d for d in make_detector(settings, raw, confidence_threshold=0.25).detect(path)}
    assert {f.code for f in dets["shirt"].flags} == {"ambiguous_category"}
    assert dets["shirt"].alternatives[0].category == "one_piece"
    assert {f.code for f in dets["sneakers"].flags} == {"low_confidence"}
    assert all(d.needs_review for d in dets.values())


def test_display_floor_is_the_only_drop_threshold(settings, scene):
    path, boxes = scene
    raw = [RawDetection("jeans", 0.14, boxes["jeans"]), RawDetection("shirt", 0.16, boxes["shirt"])]
    dets = make_detector(settings, raw, display_floor=0.15).detect(path)
    assert [d.label for d in dets] == ["shirt"]


def test_review_message_lists_each_alternative_once(settings, scene):
    path, _ = scene
    person = (100, 40, 300, 480)
    raw = [
        RawDetection("jumpsuit", 0.72, person),
        RawDetection("coat", 0.20, person),
        RawDetection("jacket", 0.13, (110, 50, 290, 380)),  # an alternative AND a contained box
    ]
    js = make_detector(settings, raw).detect(path)[0]
    msg = next(f.message for f in js.flags if f.code == "contains_separates")
    assert msg.count("coat") == 1 and msg.count("jacket") == 1, msg
