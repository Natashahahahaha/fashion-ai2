from __future__ import annotations

import numpy as np
import pytest

from conftest import populate, solid
from src.attributes.colors import dominant_colors, outfit_color_summary, pair_color_harmony, rgb_to_name
from src.attributes.style import STYLES, occasion_match, style_match, style_similarity, user_style_scores
from src.config import ScoringWeights
from src.errors import CheckpointError
from src.models.compatibility_net import OutfitCompatibilityNet
from src.models.learned import LearnedCompatibility, load_learned_model, save_checkpoint
from src.recommendation.compatibility import category_pair_score, rescale_cosine, weighted_mean
from src.recommendation.outfit_generator import OutfitGenerator
from src.recommendation.pair_model import (
    FEATURES_BASELINE,
    FEATURES_WITH_LEARNED,
    ItemTable,
    load_pair_model,
    outfit_plausibility,
    pair_features,
)


@pytest.mark.parametrize(
    "rgb,name",
    [
        ((10, 10, 10), "black"),
        ((250, 250, 250), "white"),
        ((128, 128, 128), "gray"),
        ((30, 40, 100), "navy"),
        ((200, 30, 40), "red"),
        ((40, 140, 60), "green"),
        ((215, 195, 160), "beige"),
        ((110, 70, 35), "brown"),
    ],
)
def test_color_naming(rgb, name):
    assert rgb_to_name(rgb) == name


def test_dominant_colors_ignore_uniform_background():
    colors = dominant_colors(solid((200, 30, 40)))
    assert colors[0]["name"] == "red"
    assert abs(sum(c["fraction"] for c in colors) - 1) < 0.01


HEX = {
    "black": "#141414",
    "white": "#f5f5f5",
    "navy": "#1e2850",
    "brown": "#6e4623",
    "red": "#c81e28",
    "green": "#288c3c",
    "purple": "#783c96",
    "yellow": "#f0d228",
}


def colours(*names):
    return [{"name": n, "hex": HEX[n], "fraction": 1.0 / len(names)} for n in names]


def test_color_harmony_rules():
    c = colours
    neutral, _ = pair_color_harmony(c("black"), c("white"))
    anchored, _ = pair_color_harmony(c("red"), c("navy"))
    clash, reason = pair_color_harmony(c("red"), c("green"))
    assert neutral > anchored > clash
    assert pair_color_harmony(None, c("red")) is None
    score, why = outfit_color_summary([c("red"), c("green"), c("purple"), c("yellow")])
    assert "Busy palette" in why and score < 0.6


def test_style_helpers():
    a = user_style_scores("formal")
    assert style_similarity(a, a) == pytest.approx(1.0)
    assert style_similarity(a, user_style_scores("sporty")) == pytest.approx(0.0)
    assert style_match(a, "formal") == 1.0 and style_match(a, "casual") == 0.0
    assert occasion_match(a, "interview") > occasion_match(a, "college")
    assert style_similarity(None, a) is None
    with pytest.raises(ValueError):
        user_style_scores("cyberpunk")
    assert set(a) == set(STYLES)


def test_category_pairs_and_rescale():
    assert category_pair_score("top", "bottom") == 1.0
    assert category_pair_score("top", "top") == 0.0
    assert category_pair_score("one_piece", "bottom") == 0.0
    assert rescale_cosine(0.0) == 0.0 and rescale_cosine(1.0) == 1.0
    assert 0.0 < rescale_cosine(0.7) < 1.0


def test_weighted_mean_renormalises_missing_signals():
    w = ScoringWeights()
    assert weighted_mean({"visual": 0.8, "color": None, "style": None}, w) == pytest.approx(0.8)
    full = weighted_mean({"visual": 1.0, "color": 0.0, "style": 0.0, "category": 0.0, "context": 0.0}, w)
    assert full == pytest.approx(w.visual / sum(w.as_dict().values()))


def test_colour_relations_are_defensible():
    c = colours
    contrast, _ = pair_color_harmony(c("black"), c("white"))
    dark_pair, _ = pair_color_harmony(c("black"), c("navy"))
    brown_navy, _ = pair_color_harmony(c("brown"), c("navy"))
    assert contrast > dark_pair  # black + navy has little contrast; not "perfect"
    assert brown_navy < 0.9
    score, why = outfit_color_summary([c("black"), c("brown"), c("navy")])
    assert score < 0.9, (score, why)


def _table(manager):
    items = manager.list_items()
    return items, ItemTable.from_wardrobe(items, manager.load_embeddings(items))


def test_pair_features_match_definitions(manager):
    populate(manager)
    items, table = _table(manager)
    ia, ib = np.array([0, 0]), np.array([3, 1])  # white tee x blue jeans, white tee x black shirt
    f = pair_features(table, ia, ib)
    assert set(f) == set(FEATURES_BASELINE)
    assert f["clip_cos"][0] == pytest.approx(float(table.embeddings[0] @ table.embeddings[3]), abs=1e-6)
    assert f["category"][0] == 1.0 and f["category"][1] == 0.0  # top+bottom vs two tops
    assert 0.0 <= np.nanmin(f["colour"]) and np.nanmax(f["colour"]) <= 1.0


def test_fitted_pair_model_is_loaded_and_used(manager):
    model = load_pair_model(with_learned=False)
    assert set(model.features) <= set(FEATURES_BASELINE) and model.features
    assert (model.coef >= 0).all()  # fitted with non-negative weights; contradicted signals are dropped
    assert set(model.features) | set(model.info.get("dropped_features", {})) == set(FEATURES_BASELINE)
    populate(manager)
    items, table = _table(manager)
    ia, ib = np.triu_indices(len(items), 1)
    f = pair_features(table, ia, ib)
    p = model.score(f)
    assert p.shape == ia.shape and ((p > 0) & (p < 1)).all()
    # explanation contributions are exactly the terms of the logit
    z = sum(model.contributions(f).values()) + model.intercept
    np.testing.assert_allclose(1 / (1 + np.exp(-z)), p, rtol=1e-9)
    with_learned = load_pair_model(with_learned=True)
    assert "learned_logit" in with_learned.features and set(with_learned.features) <= set(FEATURES_WITH_LEARNED)


def test_missing_ranker_file_falls_back_without_threshold(tmp_path):
    m = load_pair_model(with_learned=False, path=tmp_path / "nope.json")
    assert not m.fitted and m.outfit_threshold is None and "not found" in m.info["note"]


def test_outfit_plausibility_weights_accessory_pairs_half():
    assert outfit_plausibility([("top", "bottom", 0.8), ("top", "shoes", 0.6)]) == pytest.approx(0.7)
    with_acc = outfit_plausibility([("top", "bottom", 0.8), ("top", "accessory", 0.2)])
    assert with_acc == pytest.approx((0.8 + 0.5 * 0.2) / 1.5)
    assert outfit_plausibility([]) == 0.0


# ------------------------------------------------------------- learned model
def test_model_outputs_logits_and_symmetric_proba():
    import torch

    torch.manual_seed(0)
    m = OutfitCompatibilityNet(embed_dim=8, hidden_dims=(16,)).eval()
    a, b = torch.randn(5, 8), torch.randn(5, 8)
    logits = m(a, b)
    assert logits.shape == (5,)
    p = m.predict_proba(a, b)
    assert torch.allclose(p, m.predict_proba(b, a)) and ((p > 0) & (p < 1)).all()
    with pytest.raises(ValueError):
        m(torch.randn(5, 7), torch.randn(5, 7))


def test_checkpoint_roundtrip_and_encoder_check(settings):
    import torch

    torch.manual_seed(0)
    model = OutfitCompatibilityNet(embed_dim=32, hidden_dims=(16,))
    enc = {"model_name": "fake-clip", "dim": 32, "normalized": True}
    save_checkpoint(settings.compatibility_checkpoint, model, enc, {"test": {"roc_auc": 0.5}}, {"source": "unit-test"})
    loaded = LearnedCompatibility.load(settings.compatibility_checkpoint, enc)
    x = np.random.default_rng(0).normal(size=(4, 32)).astype(np.float32)
    probs = loaded.predict(x, x[::-1])
    assert probs.shape == (4,) and ((probs > 0) & (probs < 1)).all()
    with pytest.raises(CheckpointError, match="trained on embeddings"):
        LearnedCompatibility.load(settings.compatibility_checkpoint, {"model_name": "other", "dim": 512})


def test_legacy_or_missing_checkpoints_fall_back_to_baseline(settings):
    import torch

    status = load_learned_model(settings, {"model_name": "fake-clip", "dim": 32})
    assert status.state == "missing" and status.model is None
    # A bare state_dict like the old repository produced is rejected, not silently used.
    torch.save(OutfitCompatibilityNet(embed_dim=32).state_dict(), settings.compatibility_checkpoint)
    status = load_learned_model(settings, {"model_name": "fake-clip", "dim": 32})
    assert status.state == "error" and status.model is None and "format" in status.message


def test_learned_model_is_one_signal_in_the_outfit_score(manager, settings):
    import torch

    populate(manager)
    items = manager.list_items()
    embs = manager.load_embeddings(items)
    torch.manual_seed(1)
    model = OutfitCompatibilityNet(embed_dim=32, hidden_dims=(16,))
    beat = {"comparison": {"beats_best_baseline": True}}
    save_checkpoint(settings.compatibility_checkpoint, model, {"model_name": "fake-clip", "dim": 32}, beat, {})
    learned = load_learned_model(settings, {"model_name": "fake-clip", "dim": 32}).model
    assert learned is not None
    g = OutfitGenerator(manager, learned=learned, quality_threshold=0.0)
    assert "learned_logit" in g.model.features and "learned compatibility model" in g.scoring_mode
    res = g.generate_outfits(k=1, include_outerwear=False, include_accessory=False)
    o = res.outfits[0]
    ids = o.item_ids
    probs = [float(learned.predict(embs[a][None], embs[b][None])[0]) for i, a in enumerate(ids) for b in ids[i + 1 :]]
    assert o.result.components["learned"] == pytest.approx(float(np.mean(probs)), abs=1e-4)
    assert o.result.components["plausibility"] == pytest.approx(o.score)
    # the learned model is one input to the outfit score, not the score itself
    assert o.result.components["learned"] != pytest.approx(o.score, abs=1e-6)


def test_checkpoint_that_did_not_beat_baseline_is_not_used_automatically(settings):
    import dataclasses

    import torch

    torch.manual_seed(0)
    model = OutfitCompatibilityNet(embed_dim=32, hidden_dims=(16,))
    enc = {"model_name": "fake-clip", "dim": 32}
    metrics = {"comparison": {"beats_best_baseline": False, "strongest_non_random_baseline": "clip_cosine"}}
    save_checkpoint(settings.compatibility_checkpoint, model, enc, metrics, {})
    status = load_learned_model(settings, enc)
    assert status.state == "not_better" and status.model is None and "did not beat" in status.message
    forced = load_learned_model(dataclasses.replace(settings, use_learned_model="true"), enc)
    assert forced.state == "loaded" and forced.model is not None and "did not beat" in forced.message
