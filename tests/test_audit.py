"""Dataset audit / quality gate and evaluation-metric tests (fixture data; verdicts only, no model claims)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.training.audit import (
    INSUFFICIENT,
    SUFFICIENT_WITH_LIMITATIONS,
    DatasetSource,
    QualityThresholds,
    audit_dataset,
    render_markdown,
)
from src.training.data import build_outfit_pairs_typed
from src.training.metrics import (
    best_f1_threshold,
    classification_metrics,
    expected_calibration_error,
    paired_bootstrap_auc_difference,
)

ROOT = Path(__file__).resolve().parent.parent
CATS = ["tops", "bottoms", "shoes", "outerwear", "all-body", "bags"]
LOOSE = QualityThresholds(
    min_train_outfits=10, min_unique_items=40, min_train_positive_pairs=20, min_valid_positive_pairs=5, min_test_positive_pairs=5,
    max_top_category_share=0.9,
)  # fmt: skip


def make_polyvore(root: Path, sizes=(40, 10, 10), shared_items: int = 0, images: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    meta = {}
    first_train_item = None
    rng = np.random.default_rng(0)
    for split, n in zip(("train", "valid", "test"), sizes, strict=True):
        outfits = []
        for o in range(n):
            items = [f"{split}{o}_{k}" for k in range(3)]
            if split != "train" and o < shared_items and first_train_item:
                items[0] = first_train_item  # leak a training item
            for k, it in enumerate(items):
                meta.setdefault(it, {"semantic_category": CATS[(o + k) % len(CATS)], "title": f"item {it}"})
            outfits.append({"set_id": f"{split}{o}", "items": [{"item_id": i, "index": j} for j, i in enumerate(items)]})
            first_train_item = first_train_item or items[0]
        (root / f"{split}.json").write_text(json.dumps(outfits))
    (root / "meta.json").write_text(json.dumps(meta))
    if images:
        (root / "images").mkdir(exist_ok=True)
        for it in meta:
            Image.fromarray(rng.integers(0, 255, (40, 30, 3), dtype=np.uint8)).save(root / "images" / f"{it}.jpg")
    return root


def test_typed_pairs_separate_observed_from_constructed():
    outfits = [[f"o{o}_{k}" for k in range(3)] for o in range(20)]
    pairs = build_outfit_pairs_typed(outfits, {i: CATS[int(i[-1])] for o in outfits for i in o})
    kinds = {k for *_, k in pairs}
    assert kinds == {"observed_positive", "random_negative", "hard_negative"}
    assert all((y == 1) == (k == "observed_positive") for _, _, y, k in pairs)


def test_small_dataset_is_insufficient_with_reasons(tmp_path):
    root = make_polyvore(tmp_path / "p")
    report = audit_dataset(DatasetSource(polyvore_dir=root, metadata=root / "meta.json"), check_images=False)
    gate = report["gate"]
    assert gate["verdict"] == INSUFFICIENT
    assert any("training outfits" in f for f in gate["failures"]) and gate["what_is_missing"]
    assert report["raw"]["outfits"] == {"train": 40, "valid": 10, "test": 10}
    assert report["raw"]["outfit_size_distribution"] == {"3": 60}
    assert report["metadata"]["has_text"] is True
    md = render_markdown(report)
    assert "INSUFFICIENT" in md and "Blocking problems" in md
    json.dumps(report)  # machine-readable


def test_leakage_is_measured_and_filtered(tmp_path):
    root = make_polyvore(tmp_path / "p", shared_items=4)
    report = audit_dataset(DatasetSource(polyvore_dir=root, metadata=root / "meta.json"), LOOSE, check_images=False)
    assert report["leakage"]["items_shared"]["train-valid"] == 1
    assert report["leakage"]["items_shared"]["train-test"] == 1
    assert sum(report["effective"]["item_disjoint_pairs_dropped"].values()) > 0
    eff = report["effective"]["after_item_disjoint_split"]
    assert eff["valid"]["positive_pairs"] < report["effective"]["after_cleaning"]["valid"]["positive_pairs"]
    assert any("share items" in x for x in report["gate"]["limitations"])


def test_image_checks_find_missing_corrupt_and_duplicates(tmp_path):
    root = make_polyvore(tmp_path / "p", images=True)
    imgs = root / "images"
    (imgs / "train0_0.jpg").unlink()  # missing
    (imgs / "train1_0.jpg").write_bytes(b"broken")  # corrupt
    dup = (imgs / "train2_0.jpg").read_bytes()
    (imgs / "test3_1.jpg").write_bytes(dup)  # exact duplicate across splits
    report = audit_dataset(DatasetSource(polyvore_dir=root, metadata=root / "meta.json", images_dir=imgs), LOOSE)
    img = report["images"]
    assert img["missing"] == 1 and img["corrupt"] == 1
    assert img["exact_duplicate_groups"] == 1 and img["perceptual_duplicate_groups"] >= 1
    assert report["leakage"]["perceptual_duplicate_groups_across_splits"] == 1
    assert img["width"]["min"] == 30 and img["height"]["max"] == 40


def test_loose_thresholds_pass_with_limitations(tmp_path):
    root = make_polyvore(tmp_path / "p")
    report = audit_dataset(DatasetSource(polyvore_dir=root, metadata=root / "meta.json"), LOOSE, check_images=False)
    assert report["gate"]["verdict"] == SUFFICIENT_WITH_LIMITATIONS, report["gate"]
    assert any("user" in x for x in report["gate"]["limitations"])


def test_missing_core_category_is_blocking(tmp_path):
    root = make_polyvore(tmp_path / "p")
    meta = json.loads((root / "meta.json").read_text())
    for v in meta.values():
        if v["semantic_category"] == "shoes":
            v["semantic_category"] = "bags"
    (root / "meta.json").write_text(json.dumps(meta))
    report = audit_dataset(DatasetSource(polyvore_dir=root, metadata=root / "meta.json"), LOOSE, check_images=False)
    assert report["gate"]["verdict"] == INSUFFICIENT
    assert any("core categories: shoes" in f for f in report["gate"]["failures"])


def test_audit_script_exit_codes(tmp_path):
    root = make_polyvore(tmp_path / "p")
    out = tmp_path / "artifacts"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_dataset.py"),
            "--polyvore-dir",
            str(root),
            "--metadata",
            str(root / "meta.json"),
            "--out-dir",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 3, proc.stderr
    assert "INSUFFICIENT" in proc.stdout
    assert (out / "dataset_audit.json").is_file() and (out / "dataset_audit.md").is_file()


def test_metrics_helpers():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = classification_metrics(y, p)
    assert m["roc_auc"] == pytest.approx(0.75) and 0 <= m["pr_auc"] <= 1 and m["brier"] > 0
    assert expected_calibration_error(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0
    assert 0.1 <= best_f1_threshold(y, p) <= 0.8
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 2, 400)
    good = yy + rng.normal(0, 0.5, 400)
    noise = rng.normal(0, 1, 400)
    d = paired_bootstrap_auc_difference(yy, good, noise, n=200)
    assert d["difference"] > 0 and d["ci95_low"] > 0
