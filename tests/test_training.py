"""Training pipeline mechanics.

These tests use tiny generated fixtures to check that the code runs, uses the
right loss, splits without leakage and writes a loadable checkpoint. The
fixture data is NOT fashion data and no metric from it means anything.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.errors import DatasetMissingError
from src.models.learned import LearnedCompatibility
from src.training.audit import QualityThresholds
from src.training.data import EmbeddingDirectory, build_outfit_pairs, load_pairs_csv, split_pairs
from src.training.train import TrainConfig, train_compatibility


def write_embeddings(directory: Path, ids: list[str], dim: int = 16, seed: int = 0) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in ids:
        v = rng.normal(size=dim).astype(np.float32)
        np.save(directory / f"{i}.npy", v / np.linalg.norm(v))
    (directory / "encoder.json").write_text(json.dumps({"model_name": "fixture", "dim": dim}))


def write_polyvore(directory: Path, n_outfits: int, prefix: str) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    outfits, ids = [], []
    for o in range(n_outfits):
        items = [f"{prefix}{o}_{k}" for k in range(3)]
        ids += items
        outfits.append({"set_id": f"{prefix}{o}", "items": [{"item_id": i, "index": n} for n, i in enumerate(items)]})
    return outfits, ids  # type: ignore[return-value]


def test_missing_dataset_fails_with_instructions(tmp_path):
    write_embeddings(tmp_path / "emb", ["a"])
    cfg = TrainConfig(embeddings_dir=tmp_path / "emb", out_path=tmp_path / "ckpt.pt")
    with pytest.raises(DatasetMissingError, match="No training dataset given"):
        train_compatibility(cfg)
    with pytest.raises(DatasetMissingError, match="embed_dataset"):
        EmbeddingDirectory(tmp_path / "nope")


def test_pairs_csv_validation(tmp_path):
    p = tmp_path / "pairs.csv"
    p.write_text("item_a,item_b,label\na,b,2\n")
    with pytest.raises(DatasetMissingError, match="label must be 0 or 1"):
        load_pairs_csv(p)
    p.write_text("a,b\n1,2\n")
    with pytest.raises(DatasetMissingError, match="missing column"):
        load_pairs_csv(p)


def test_item_disjoint_split_has_no_leakage():
    rows = [(f"i{a}", f"i{b}", (a + b) % 2, None) for a in range(40) for b in range(a + 1, 40)]
    splits, dropped = split_pairs(rows, valid_frac=0.2, test_frac=0.2)
    items = {s: {x for a, b, _ in splits[s] for x in (a, b)} for s in splits}
    assert items["train"].isdisjoint(items["test"]) and items["train"].isdisjoint(items["valid"])
    assert dropped > 0 and all(splits.values())


def test_outfit_pairs_negatives_never_real_positives():
    outfits = [[f"o{o}_{k}" for k in range(3)] for o in range(20)]
    pairs = build_outfit_pairs(outfits, {i: i.split("_")[1] for o in outfits for i in o})
    pos = {frozenset((a, b)) for a, b, y in pairs if y == 1}
    neg = {frozenset((a, b)) for a, b, y in pairs if y == 0}
    assert pos and neg and pos.isdisjoint(neg)


# Fixture-scale thresholds, ONLY so the training loop can be exercised in a unit
# test. Real runs use QualityThresholds() defaults, which these fixtures fail.
TEST_ONLY_THRESHOLDS = QualityThresholds(
    min_train_outfits=10,
    min_unique_items=50,
    min_train_positive_pairs=50,
    min_valid_positive_pairs=10,
    min_test_positive_pairs=10,
    max_top_category_share=0.9,
)


def test_training_runs_and_saves_loadable_checkpoint(tmp_path, monkeypatch):
    import torch.nn as nn

    used = []
    real = nn.BCEWithLogitsLoss

    class Spy(real):  # type: ignore[misc,valid-type]
        def forward(self, logits, target):
            used.append((logits.min().item(), logits.max().item()))
            return super().forward(logits, target)

    monkeypatch.setattr(nn, "BCEWithLogitsLoss", Spy)

    root = tmp_path / "poly"
    all_ids = []
    for split, n in (("train", 60), ("valid", 15), ("test", 15)):
        outfits, ids = write_polyvore(root, n, split)
        (root / f"{split}.json").write_text(json.dumps(outfits))
        all_ids += ids
    write_embeddings(tmp_path / "emb", all_ids)
    meta = {i: {"semantic_category": ["tops", "bottoms", "shoes"][int(i.rsplit("_", 1)[1])]} for i in all_ids}
    (root / "meta.json").write_text(json.dumps(meta))

    out = tmp_path / "ckpt" / "compatibility_model.pt"
    result = train_compatibility(
        TrainConfig(
            embeddings_dir=tmp_path / "emb",
            out_path=out,
            polyvore_dir=root,
            polyvore_metadata=root / "meta.json",
            epochs=2,
            batch_size=32,
            hidden_dims=(16,),
            thresholds=TEST_ONLY_THRESHOLDS,
            bootstrap=50,
        )
    )
    assert used, "BCEWithLogitsLoss was not used"
    m = result["metrics"]
    assert {"roc_auc", "pr_auc", "precision", "recall", "f1", "brier", "ece_10bin", "roc_auc_ci95"} <= set(m["test"])
    assert m["threshold_selected_on"].startswith("validation")
    assert set(m["baselines"]) == {"random", "clip_cosine", "heuristic_visual_category"}
    assert "beats_best_baseline" in m["comparison"]
    assert m["test"]["test_items_seen_in_training"] == 0  # item-disjoint
    assert m["test"]["vs_hard_negatives"] is not None  # metadata -> hard negatives exist
    assert (tmp_path / "ckpt" / "dataset_audit.json").is_file() and (tmp_path / "ckpt" / "dataset_audit.md").is_file()
    model = LearnedCompatibility.load(out, {"model_name": "fixture", "dim": 16})
    assert model.training["dataset_gate"] != "INSUFFICIENT"
    assert model.metrics["comparison"] == m["comparison"]


def test_gate_blocks_training_with_default_thresholds(tmp_path):
    root = tmp_path / "poly"
    all_ids = []
    for split, n in (("train", 60), ("valid", 15), ("test", 15)):
        outfits, ids = write_polyvore(root, n, split)
        (root / f"{split}.json").write_text(json.dumps(outfits))
        all_ids += ids
    write_embeddings(tmp_path / "emb", all_ids)
    out = tmp_path / "ckpt" / "model.pt"
    with pytest.raises(DatasetMissingError, match="INSUFFICIENT"):
        train_compatibility(TrainConfig(embeddings_dir=tmp_path / "emb", out_path=out, polyvore_dir=root))
    assert not out.exists()  # nothing was trained
    report = json.loads((tmp_path / "ckpt" / "dataset_audit.json").read_text())
    assert report["gate"]["verdict"] == "INSUFFICIENT" and report["gate"]["failures"]


def test_too_small_dataset_rejected(tmp_path):
    p = tmp_path / "pairs.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_a", "item_b", "label", "split"])
        for s in ("train", "valid", "test"):
            w.writerow(["a", "b", 1, s])
            w.writerow(["a", "c", 0, s])
    write_embeddings(tmp_path / "emb", ["a", "b", "c"])
    with pytest.raises(DatasetMissingError, match="INSUFFICIENT"):
        train_compatibility(TrainConfig(embeddings_dir=tmp_path / "emb", out_path=tmp_path / "x.pt", pairs_csv=p))


def test_encoder_mismatch_rejected(tmp_path):
    write_embeddings(tmp_path / "emb", ["a"])
    cfg = TrainConfig(
        embeddings_dir=tmp_path / "emb",
        out_path=tmp_path / "x.pt",
        pairs_csv=tmp_path / "p.csv",
        expected_encoder={"model_name": "openai/clip-vit-base-patch32", "dim": 512},
    )
    with pytest.raises(DatasetMissingError, match="Re-run scripts/embed_dataset.py"):
        train_compatibility(cfg)
