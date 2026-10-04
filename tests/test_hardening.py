"""Robustness tests added in the QA pass: duplicates, integrity, restarts, odd wardrobes."""

from __future__ import annotations

import sqlite3
from collections import Counter

import numpy as np
import pytest

from conftest import FakeEncoder, populate, solid
from src.errors import FashionAIError
from src.recommendation.outfit_generator import OutfitGenerator
from src.training.train import remove_item_overlap
from src.wardrobe.manager import WardrobeManager


def gen(manager, **kw) -> OutfitGenerator:
    kw.setdefault("quality_threshold", 0.0)
    return OutfitGenerator(manager, **kw)


# ------------------------------------------------------------------ wardrobe
def test_duplicate_is_flagged_not_blocked(manager):
    first = manager.add_item(solid((200, 30, 40)), category="top")
    second = manager.add_item(solid((200, 30, 40)), category="top")
    other_cat = manager.add_item(solid((200, 30, 40)), category="shoes")
    assert "possible_duplicate_of" not in first.metadata
    assert second.metadata["possible_duplicate_of"] == [first.id]
    assert "possible_duplicate_of" not in other_cat.metadata  # duplicates are per category
    assert manager.count() == 3  # user data is never dropped silently


def test_wardrobe_survives_restart_with_valid_paths(settings):
    m1 = WardrobeManager(settings, encoder=FakeEncoder())
    items = populate(m1)
    del m1
    m2 = WardrobeManager(settings, encoder=FakeEncoder())  # fresh process-equivalent
    assert {i.id for i in m2.list_items()} == {i.id for i in items.values()}
    for it in m2.list_items():
        assert m2.resolve(it.image_path).is_file()
        emb = m2.load_embedding(it)
        assert emb is not None and abs(float(np.linalg.norm(emb)) - 1) < 1e-4
    assert gen(m2).generate_outfits(k=3).outfits


def test_no_temporary_files_left_after_add(manager):
    populate(manager)
    leftovers = [
        p for d in (manager.settings.images_dir, manager.settings.embeddings_dir) for p in d.iterdir() if ".tmp" in p.name
    ]
    assert leftovers == []


def test_integrity_report_and_orphan_cleanup(manager):
    items = populate(manager)
    manager.resolve(items["red top"].image_path).unlink()
    orphan = manager.settings.images_dir / "leftover.jpg"
    solid((1, 2, 3)).save(orphan)
    with sqlite3.connect(manager.settings.db_path) as conn:
        conn.execute("UPDATE items SET category = 'hat-ish' WHERE id = ?", (items["navy bag"].id,))
    report = manager.integrity_report()
    assert report.missing_images == [items["red top"].id]
    assert report.invalid_category == [items["navy bag"].id]
    assert report.orphan_files == [orphan]
    assert manager.remove_orphan_files() == 1 and not orphan.exists()
    assert manager.count() == len(items)  # cleanup never touches referenced data


def test_invalid_category_row_is_skipped_by_generator(manager):
    items = populate(manager)
    with sqlite3.connect(manager.settings.db_path) as conn:
        conn.execute("UPDATE items SET category = 'spaceship' WHERE id = ?", (items["white tee"].id,))
    res = gen(manager).generate_outfits(k=5)
    assert res.outfits and items["white tee"].id in res.skipped_items
    assert all(items["white tee"].id not in o.item_ids for o in res.outfits)


def test_stale_when_embedding_dim_changes(settings):
    m = WardrobeManager(settings, encoder=FakeEncoder(dim=32))
    m.add_item(solid((10, 200, 10)), category="top")
    m2 = WardrobeManager(settings, encoder=FakeEncoder(dim=16))  # same model name, different size
    assert len(m2.embedding_status().stale) == 1
    assert m2.load_embeddings(m2.list_items()) == {}


def test_partial_batch_failure_reports_progress(manager, scene):
    from conftest import make_detector
    from src.detection.detector import RawDetection

    path, boxes = scene
    dets = make_detector(
        manager.settings, [RawDetection("shirt", 0.9, boxes["shirt"]), RawDetection("jeans", 0.8, boxes["jeans"])]
    ).detect(path)
    from pathlib import Path

    Path(dets[1].crop_path).unlink()  # second crop vanished
    with pytest.raises(FashionAIError, match="Added 1 item"):
        manager.add_detections(dets)
    assert manager.count() == 1


# ----------------------------------------------------------------- outfits
def test_single_category_wardrobe_explains_what_is_missing(manager):
    for c in [(10, 10, 10), (200, 200, 200), (90, 30, 30)]:
        manager.add_item(solid(c), category="shoes")
    res = gen(manager).generate_outfits(k=3)
    assert res.outfits == []
    assert "3 shoes" in res.message and "top and one bottom" in res.message


def test_accessory_only_cannot_form_outfit_and_must_include_accessory_needs_core(manager):
    bag = manager.add_item(solid((20, 30, 70)), category="accessory")
    res = gen(manager).generate_outfits(k=3, required_item_id=bag.id)
    assert res.outfits == [] and "needs shoes" in res.message and "1 accessory" in res.message
    for c in [(15, 15, 15)]:
        manager.add_item(solid(c), category="shoes")
    res = gen(manager).generate_outfits(k=3, required_item_id=bag.id)
    assert res.outfits == [] and "Not enough items" in res.message


def test_multiple_shoes_and_tops_never_two_per_slot(manager):
    populate(manager)
    for c in [(120, 120, 120), (180, 40, 40)]:
        manager.add_item(solid(c), category="shoes")
        manager.add_item(solid(c), category="top")
    res = gen(manager).generate_outfits(k=10)
    for o in res.outfits:
        counts = Counter(i.category for i in o.items)
        assert max(counts.values()) == 1
        assert not ({"one_piece"} <= set(counts) and ({"top"} & set(counts) or {"bottom"} & set(counts)))
        assert counts["shoes"] == 1


def test_near_duplicates_never_make_two_looks_that_differ_only_by_copy(manager):
    populate(manager)
    tee = manager.list_items("top")[0]
    for _ in range(3):  # the same garment added three more times
        manager.add_item(manager.resolve(tee.image_path), category="top")
    dup_ids = {i.id for i in manager.list_items("top") if i.id == tee.id or i.metadata.get("possible_duplicate_of")}
    assert len(dup_ids) == 4
    res = gen(manager).generate_outfits(k=10, include_outerwear=False, include_accessory=False)
    assert res.outfits
    cores = [frozenset("tee" if i in dup_ids else i for i in o.item_ids) for o in res.outfits]
    assert len(set(cores)) == len(cores)


def test_generation_is_deterministic(manager):
    populate(manager)
    a = gen(manager).generate_outfits(k=5, style="casual", occasion="date")
    b = gen(manager).generate_outfits(k=5, style="casual", occasion="date")
    assert [(o.item_ids, round(o.score, 10)) for o in a.outfits] == [(o.item_ids, round(o.score, 10)) for o in b.outfits]


# ---------------------------------------------------------------- scoring
def test_explanations_never_mention_uncomputed_attributes(manager):
    populate(manager)
    res = gen(manager).generate_outfits(k=5, style="formal", occasion="interview")
    # ("fit" alone is allowed: "fit for 'interview'" is the computed occasion score)
    banned = ("material", "silhouette", "fabric", "pattern", "texture", "slim", "loose")
    for o in res.outfits:
        text = " ".join(o.result.reasons).lower()
        assert not any(b in text for b in banned), text


# ---------------------------------------------------------------- training
def test_item_overlap_filter_makes_evaluation_item_disjoint():
    splits = {
        "train": [("a", "b", 1), ("c", "d", 0)],
        "valid": [("a", "x", 1), ("y", "z", 0)],
        "test": [("y", "q", 1), ("m", "n", 0), ("d", "n", 1)],
    }
    dropped = remove_item_overlap(splits)
    assert dropped == {"valid": 1, "test": 2}
    assert splits["valid"] == [("y", "z", 0)] and splits["test"] == [("m", "n", 0)]


# --------------------------------------------------------- release fixes
def test_requested_count_reported_when_fewer_looks_possible(manager):
    manager.add_item(solid((240, 240, 240)), category="top")
    manager.add_item(solid((40, 60, 120)), category="bottom")
    manager.add_item(solid((15, 15, 15)), category="shoes")
    res = gen(manager).generate_outfits(k=5)
    assert len(res.outfits) == 1 and res.requested == 5
    assert "1 strong outfit(s) found" in res.message


def test_detection_labels_stay_inside_image_and_crop_thumbs_are_square(tmp_path):
    from PIL import Image

    from src.detection.schemas import DetectionResult
    from src.ui.components import THUMB, crop_thumbnail, draw_detections

    img = Image.new("RGB", (300, 200), (255, 255, 255))
    det = DetectionResult(id="x", label="jacket", category="outerwear", confidence=0.41, bbox=(270, 50, 300, 150))
    out = draw_detections(img, [det])
    assert out.size == img.size
    # the label box was shifted left, so pixels right of x=270 near the top got coloured
    assert any(out.getpixel((x, 40)) != (255, 255, 255) for x in range(200, 270))
    tall = tmp_path / "tall.jpg"
    Image.new("RGB", (50, 400), (10, 10, 10)).save(tall)
    assert crop_thumbnail(str(tall)).size == (THUMB, THUMB)
    assert crop_thumbnail(str(tmp_path / "missing.jpg")).size == (THUMB, THUMB)


def test_items_listed_in_insertion_order(manager):
    ids = [manager.add_item(solid((i * 20, 50, 200 - i * 20)), category="top").id for i in range(6)]
    assert [i.id for i in manager.list_items()] == ids


def test_learned_mode_explanations_state_uncalibrated_score(manager, settings):
    import torch

    from src.models.compatibility_net import OutfitCompatibilityNet
    from src.models.learned import load_learned_model, save_checkpoint

    populate(manager)
    torch.manual_seed(0)
    enc = {"model_name": "fake-clip", "dim": 32}
    save_checkpoint(
        settings.compatibility_checkpoint,
        OutfitCompatibilityNet(embed_dim=32, hidden_dims=(16,)),
        enc,
        {"comparison": {"beats_best_baseline": True}},
        {},
    )
    learned = load_learned_model(settings, enc).model
    res = OutfitGenerator(manager, learned=learned, quality_threshold=0.0).generate_outfits(k=2)
    o = res.outfits[0]
    assert "learned compatibility model" in o.result.explanation["scoring"]
    text = " ".join(o.result.reasons + o.result.caveats)
    assert "probability" not in text or "not a probability" in text
    assert 0.0 < o.result.components["learned"] < 1.0
