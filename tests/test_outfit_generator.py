"""Outfit generation: validity gate, full-wardrobe coverage, diversity, no padding.

Generation mechanics are tested with ``quality_threshold=0.0`` (every valid
candidate counts as strong) unless a test is about the threshold itself, so
results do not depend on how the fitted ranker scores synthetic colour blocks.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from PIL import Image

from conftest import make_detector, populate, solid
from src.detection.detector import RawDetection
from src.recommendation.outfit_generator import OutfitGenerator
from src.recommendation.validity import is_valid_outfit
from src.wardrobe.manager import REVIEW_CONFIRMED, REVIEW_NEEDED, needs_review


def gen(manager, **kw) -> OutfitGenerator:
    kw.setdefault("quality_threshold", 0.0)
    return OutfitGenerator(manager, **kw)


def two_tone(rng: np.random.Generator, i: int) -> Image.Image:
    """A distinct synthetic garment: two random colour bands (not near-duplicates of each other)."""
    img = Image.new("RGB", (120, 160), tuple(int(x) for x in rng.integers(0, 256, 3)))
    img.paste(Image.new("RGB", (120, 80), tuple(int(x) for x in rng.integers(0, 256, 3))), (0, 80 if i % 2 else 0))
    return img


def assert_valid(outfits) -> None:
    for o in outfits:
        v = is_valid_outfit(o.items)
        assert v.valid, (o.item_ids, v.reasons)
    assert len({frozenset(o.item_ids) for o in outfits}) == len(outfits)  # never the same look twice


# ------------------------------------------------------------ basic flow
def test_empty_wardrobe(manager):
    res = gen(manager).generate_outfits(k=5)
    assert res.outfits == [] and "empty" in res.message.lower()


def test_generates_ranked_valid_outfits(manager):
    populate(manager)
    res = gen(manager).generate_outfits(k=5)
    assert 1 <= len(res.outfits) <= 5
    assert_valid(res.outfits)
    for o in res.outfits:
        assert 0 < o.score <= 1
        assert o.result.reasons[0].startswith("Valid ") and o.result.reasons[0].endswith("structure")
        assert {"plausibility", "colour", "style", "visual_similarity", "category"} <= set(o.result.components)


def test_required_item_always_included(manager):
    items = populate(manager)
    for name in ("red top", "green dress", "brown jacket", "navy bag", "white sneakers"):
        req = items[name]
        res = gen(manager).generate_outfits(k=3, required_item_id=req.id)
        assert res.outfits, (name, res.message)
        assert all(req.id in o.item_ids for o in res.outfits), name
        assert_valid(res.outfits)


def test_style_and_occasion_add_context_signal(manager):
    populate(manager)
    res = gen(manager).generate_outfits(k=3, style="formal", occasion="interview")
    assert res.outfits and all(o.result.components["context"] is not None for o in res.outfits)
    plain = gen(manager).generate_outfits(k=3)
    assert all(o.result.components["context"] is None for o in plain.outfits)


def test_optional_layers_can_be_disabled(manager):
    populate(manager)
    res = gen(manager).generate_outfits(k=5, include_outerwear=False, include_accessory=False)
    assert res.outfits
    assert all({"outerwear", "accessory"}.isdisjoint(i.category for i in o.items) for o in res.outfits)


def test_no_shoes_means_no_outfit_and_a_clear_message(manager):
    manager.add_item(solid((240, 240, 240)), category="top")
    manager.add_item(solid((40, 60, 120)), category="bottom")
    res = gen(manager).generate_outfits(k=3)
    assert res.outfits == [] and "shoes" in res.message.lower()


def test_not_enough_items_message(manager):
    manager.add_item(solid((240, 240, 240)), category="top")
    manager.add_item(solid((15, 15, 15)), category="shoes")
    res = gen(manager).generate_outfits(k=3)
    assert res.outfits == [] and "top and one bottom" in res.message.lower()


def test_item_recommendations_cover_all_compatible_items(manager):
    items = populate(manager)
    recs, msg = gen(manager).recommend_for_item(items["white tee"].id)
    assert msg is None
    cats = Counter(r.item.category for r in recs)
    assert "top" not in cats and "one_piece" not in cats  # cannot be worn together with a top
    # every compatible item is returned (no top-k cut), except near-identical images of the tee
    # itself (the fake encoder makes white solids near-duplicates; real CLIP does not)
    embs = manager.load_embeddings(manager.list_items())
    tee = embs[items["white tee"].id]
    expected = {
        it.id
        for it in manager.list_items()
        if it.category in ("bottom", "shoes", "outerwear", "accessory") and float(embs[it.id] @ tee) < 0.97
    }
    assert {r.item.id for r in recs} == expected and len(expected) >= 4
    scores = [r.pair.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_corrupt_embedding_items_skipped_and_reported(manager):
    items = populate(manager)
    manager.resolve(items["black shoes"].embedding_path).write_bytes(b"x")
    res = gen(manager).generate_outfits(k=5)
    assert items["black shoes"].id in res.skipped_items
    assert res.diagnostics["excluded"] == {"missing or invalid embedding": 1}
    assert all(items["black shoes"].id not in o.item_ids for o in res.outfits)


# ------------------------------------------------- adversarial validity suite
def _items(manager):
    rng = np.random.default_rng(3)
    spec = ["top", "top", "bottom", "shoes", "shoes", "one_piece", "outerwear", "accessory"]
    return [manager.add_item(two_tone(rng, i), category=c, label=c) for i, c in enumerate(spec)]


@pytest.mark.parametrize(
    "picks,reason",
    [
        ([5, 0, 3], "dress/one-piece cannot be worn with a separate top"),  # one-piece + top
        ([5, 2, 3], "dress/one-piece cannot be worn with a separate top or bottom"),  # one-piece + bottom
        ([0, 1, 2, 3], "2 items of the same type"),  # two tops
        ([0, 2, 3, 4], "2 items of the same type"),  # two pairs of shoes
        ([0, 2], "no shoes"),  # top + bottom only
        ([0, 3], "a top needs a bottom"),  # top + shoes
        ([2, 3], "a bottom needs a top"),  # bottom + shoes
        ([6, 2, 3], "a bottom needs a top"),  # outerwear standing in for a top
        ([7, 3], "no main garment"),  # accessory + shoes
        ([6, 7, 3], "no main garment"),  # layers without a body
    ],
)
def test_adversarial_combinations_are_rejected(manager, picks, reason):
    items = _items(manager)
    v = is_valid_outfit([items[i] for i in picks])
    assert not v.valid and any(reason in r for r in v.reasons), v.reasons


@pytest.mark.parametrize("picks", [[0, 2, 3], [5, 3], [0, 2, 3, 6], [5, 3, 6, 7], [1, 2, 4, 6, 7]])
def test_valid_structures_pass(manager, picks):
    items = _items(manager)
    v = is_valid_outfit([items[i] for i in picks])
    assert v.valid, v.reasons


def test_same_photo_region_twice_is_one_garment(manager, scene):
    """One garment detected twice with different labels -> both items can never meet in an outfit."""
    path, boxes = scene
    x1, y1, x2, y2 = boxes["shirt"]
    dets = make_detector(
        manager.settings,
        [RawDetection("shirt", 0.9, boxes["shirt"]), RawDetection("jacket", 0.9, (x1 + 4, y1 + 4, x2 + 4, y2 + 4))],
    ).detect(path)
    # region grouping merges them into ONE region with an alternative ...
    assert len(dets) == 1 and dets[0].alternatives
    # ... and if the user forces both in anyway (two uploads), validity still separates them
    a = manager.add_detections(dets)[0]
    b = manager.add_detections(dets, overrides={dets[0].id: "outerwear"})[0]
    jeans = manager.add_item(solid((40, 60, 120)), category="bottom")
    shoes = manager.add_item(solid((15, 15, 15)), category="shoes")
    v = is_valid_outfit([a, jeans, shoes, b])
    assert not v.valid and any("same region" in r for r in v.reasons)
    res = gen(manager).generate_outfits(k=5)
    assert all(not {a.id, b.id} <= set(o.item_ids) for o in res.outfits)


# ------------------------------------------ bad detection -> review -> fixed
def _bus_like_photo(manager, scene):
    """A person region the detector calls 'jumpsuit' (0.72), with coat/jacket alternatives and
    separately detected jeans + shoes inside it: the real bus.jpg failure mode."""
    path, boxes = scene
    person = (100, 40, 300, 480)
    raw = [
        RawDetection("jumpsuit", 0.72, person),
        RawDetection("coat", 0.20, person),
        RawDetection("jacket", 0.13, (102, 42, 298, 478)),
        RawDetection("jeans", 0.60, boxes["jeans"]),
        RawDetection("sneakers", 0.70, boxes["sneakers"]),
    ]
    return make_detector(manager.settings, raw).detect(path)


def test_bus_jumpsuit_is_flagged_with_alternatives_not_hidden(manager, scene):
    dets = _bus_like_photo(manager, scene)
    js = next(d for d in dets if d.label == "jumpsuit")
    assert js.needs_review
    assert {f.code for f in js.flags} >= {"contains_separates"}
    assert [a.label for a in js.alternatives][:2] == ["coat", "jacket"]
    assert js.suggested_category == "outerwear"
    assert {d.label for d in dets} == {"jumpsuit", "jeans", "sneakers"}  # nothing hidden


def test_bus_jumpsuit_never_becomes_an_outfit_until_reviewed(manager, scene):
    dets = _bus_like_photo(manager, scene)
    stored = manager.add_detections(dets)  # added without review
    js = next(i for i in stored if i.label == "jumpsuit")
    assert needs_review(js) and js.metadata["review_status"] == REVIEW_NEEDED
    res = gen(manager).generate_outfits(k=5)
    assert res.outfits == []  # jumpsuit + sneakers would have been the only "outfit"
    assert res.diagnostics["excluded"] == {"needs review (uncertain detection)": 1}
    assert js.id in res.skipped_items

    # user corrects it to the suggested category: a coat, not a body garment
    manager.update_item(js.id, category="outerwear")
    fixed = manager.get_item(js.id)
    assert fixed.category == "outerwear" and fixed.metadata["review_status"] == REVIEW_CONFIRMED
    manager.add_item(solid((240, 240, 240)), category="top", label="white tee")
    res = gen(manager).generate_outfits(k=5)
    assert res.outfits
    assert_valid(res.outfits)
    for o in res.outfits:
        assert "one_piece" not in {i.category for i in o.items}


def test_confirming_a_flagged_item_makes_it_eligible(manager, scene):
    dets = _bus_like_photo(manager, scene)
    js = next(i for i in manager.add_detections(dets) if i.label == "jumpsuit")
    manager.confirm_item(js.id)
    assert not needs_review(manager.get_item(js.id))
    res = gen(manager).generate_outfits(k=5, required_item_id=js.id)
    assert res.outfits and all(js.id in o.item_ids for o in res.outfits)


def test_reviewed_flagged_detection_is_confirmed_on_add(manager, scene):
    dets = _bus_like_photo(manager, scene)
    js = next(i for i in manager.add_detections(dets, reviewed=True) if i.label == "jumpsuit")
    assert js.metadata["review_status"] == REVIEW_CONFIRMED and not needs_review(js)


# ------------------------------------------------- full wardrobe coverage
SPEC_29 = {"top": 8, "bottom": 6, "shoes": 5, "outerwear": 4, "one_piece": 3, "accessory": 3}


@pytest.fixture
def wardrobe29(manager):
    rng = np.random.default_rng(11)
    n = 0
    for cat, count in SPEC_29.items():
        for _ in range(count):
            manager.add_item(two_tone(rng, n), category=cat, label=f"{cat} {n}")
            n += 1
    assert manager.count() == 29
    return manager


def test_full_wardrobe_reaches_candidate_generation(wardrobe29):
    res = gen(wardrobe29).generate_outfits(k=5)
    d = res.diagnostics
    assert d["wardrobe_items"] == 29 and d["eligible_items"] == 29 and d["excluded"] == {}
    assert d["eligible_by_category"] == SPEC_29
    core = 8 * 6 * 5 + 3 * 5  # every top x bottom x shoes, every one-piece x shoes
    assert d["candidates_scored"] + d["rejected_by_validity"] == core
    assert d["candidates_scored"] >= core - 5
    assert d["pruned_for_compute"] == {}
    assert d["above_threshold"] == d["candidates_scored"] == res.strong_candidates
    # every core garment is present in the scored candidate set (no slicing / top-k pools)
    exposure = {e["id"]: e for e in d["exposure"]}
    assert len(exposure) == 29
    core_ids = [i.id for i in wardrobe29.list_items() if i.category in ("top", "bottom", "shoes", "one_piece")]
    assert all(exposure[i]["in_strong_candidates"] > 0 for i in core_ids)


def test_larger_k_explores_more_of_the_wardrobe(wardrobe29):
    r3 = gen(wardrobe29).generate_outfits(k=3)
    r5 = gen(wardrobe29).generate_outfits(k=5)
    r10 = gen(wardrobe29).generate_outfits(k=10)
    assert [len(r.outfits) for r in (r3, r5, r10)] == [3, 5, 10]
    for r in (r3, r5, r10):
        assert_valid(r.outfits)
        assert r.message is None
    u = [r.diagnostics["unique_items_in_results"] for r in (r3, r5, r10)]
    assert u[0] < u[1] < u[2]
    assert u[2] >= 15  # 10 diverse looks reach at least half of the 29 items
    assert r10.outfits[:5] == r5.outfits[:5] or [o.item_ids for o in r10.outfits[:5]] == [o.item_ids for o in r5.outfits]
    cats = r10.diagnostics["categories_in_results"]
    assert {"top", "bottom", "shoes"} <= set(cats)


def test_diversity_caps_shared_core_garments(wardrobe29):
    res = gen(wardrobe29).generate_outfits(k=10)
    groups = []
    for o in res.outfits:
        groups.append({i.id for i in o.items if i.category in ("top", "bottom", "shoes", "one_piece")})
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            shared = len(groups[i] & groups[j])
            assert shared <= 0.5 * max(len(groups[i]), len(groups[j])), (groups[i], groups[j])


def test_compute_guard_pruning_is_reported(manager):
    rng = np.random.default_rng(7)
    for i in range(12):
        for category in ("top", "bottom", "shoes"):
            manager.add_item(two_tone(rng, i), category=category)
    res = gen(manager, max_candidates=200).generate_outfits(k=5, include_outerwear=False, include_accessory=False)
    assert len(res.outfits) == 5
    pruned = res.diagnostics["pruned_for_compute"]["top + bottom + shoes"]
    assert pruned["before"] == 12**3 and pruned["kept_per_slot"] == 5
    assert res.combinations_scored <= 125


# -------------------------------------------------------- never padded
def test_fewer_looks_than_requested_are_not_padded(manager):
    manager.add_item(solid((240, 240, 240)), category="top")
    manager.add_item(solid((40, 60, 120)), category="bottom")
    manager.add_item(solid((15, 15, 15)), category="shoes")
    res = gen(manager).generate_outfits(k=5, include_outerwear=False, include_accessory=False)
    assert len(res.outfits) == 1 and res.requested == 5
    assert res.message.startswith("1 strong outfit(s) found") and "you asked for 5" in res.message


def test_similar_strong_candidates_are_explained_not_shown(manager):
    populate(manager)
    res = gen(manager).generate_outfits(k=20)
    assert len(res.outfits) < 20
    assert "strong outfit(s) found" in res.message
    if res.strong_candidates > len(res.outfits):
        assert "too similar" in res.message


def test_quality_threshold_returns_nothing_rather_than_weak_looks(manager):
    populate(manager)
    res = gen(manager, quality_threshold=0.999).generate_outfits(k=5)
    assert res.outfits == []
    assert "none reached the quality threshold" in res.message
    assert res.diagnostics["candidates_scored"] > 0 and res.diagnostics["above_threshold"] == 0


def test_generation_is_deterministic(wardrobe29):
    a = gen(wardrobe29).generate_outfits(k=5, style="casual", occasion="date")
    b = gen(wardrobe29).generate_outfits(k=5, style="casual", occasion="date")
    assert [(o.item_ids, round(o.score, 10)) for o in a.outfits] == [(o.item_ids, round(o.score, 10)) for o in b.outfits]
