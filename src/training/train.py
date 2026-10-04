"""Training / evaluation of the learned compatibility model on real data.

Flow (nothing is trained unless every step passes):

1. audit the dataset (``src/training/audit.py``) and stop if it is INSUFFICIENT;
2. use the audit's cleaned, item-disjoint typed pairs
   (observed positives vs constructed negatives);
3. train with ``BCEWithLogitsLoss``; select the epoch by validation ROC-AUC
   and the decision threshold by validation F1 (the test split is not used for
   any choice);
4. evaluate once on test, against random, CLIP-similarity and
   heuristic baselines, with bootstrap confidence intervals;
5. save the checkpoint with the audit summary, dataset manifest and the
   baseline comparison. The app only uses the model automatically if it beat
   the best baseline (see ``src/models/learned.py``).
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.errors import DatasetMissingError
from src.training.audit import (
    CATEGORY_TO_SLOT,
    INSUFFICIENT,
    DatasetSource,
    QualityThresholds,
    audit_and_prepare,
    write_reports,
)
from src.training.data import EmbeddingDirectory, Pair, load_item_categories
from src.training.metrics import (
    auc_by_group,
    best_f1_threshold,
    bootstrap_ci,
    classification_metrics,
    paired_bootstrap_auc_difference,
    positives_vs_kind_auc,
)

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    embeddings_dir: Path
    out_path: Path
    polyvore_dir: Path | None = None  # contains train.json / valid.json / test.json
    polyvore_metadata: Path | None = None
    pairs_csv: Path | None = None
    images_dir: Path | None = None  # enables image checks in the audit
    manifest_path: Path | None = None  # dataset manifest from scripts/download_dataset.py
    audit_dir: Path | None = None  # where dataset_audit.json/.md are written (default: next to checkpoint)
    check_images: bool = True
    max_audit_images: int | None = None
    epochs: int = 15
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dims: tuple[int, ...] = (1024, 256, 64)
    dropout: float = 0.3
    patience: int = 3
    seed: int = 42
    device: str = "cpu"
    expected_encoder: dict[str, Any] | None = None
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    allow_item_overlap: bool = False  # diagnostic only: keep valid/test pairs that contain training items
    bootstrap: int = 500
    extra: dict[str, Any] = field(default_factory=dict)


def remove_item_overlap(splits: dict[str, list[Pair]]) -> dict[str, int]:
    """Drop valid/test pairs containing any item seen in an earlier split (item-disjoint evaluation)."""
    train_items = {x for p in splits["train"] for x in (p[0], p[1])}
    valid_before = len(splits["valid"])
    splits["valid"] = [p for p in splits["valid"] if p[0] not in train_items and p[1] not in train_items]
    seen = train_items | {x for p in splits["valid"] for x in (p[0], p[1])}
    test_before = len(splits["test"])
    splits["test"] = [p for p in splits["test"] if p[0] not in seen and p[1] not in seen]
    return {"valid": valid_before - len(splits["valid"]), "test": test_before - len(splits["test"])}


def _source(cfg: TrainConfig) -> DatasetSource:
    if cfg.polyvore_dir is None and cfg.pairs_csv is None:
        raise DatasetMissingError(
            "No training dataset given. Provide --polyvore-dir (Polyvore Outfits split JSONs) or --pairs-csv "
            "(item_a,item_b,label). See README section 'Optional learned compatibility model'. "
            "Training on synthetic/random vectors is intentionally not supported."
        )
    manifest = {}
    if cfg.manifest_path is not None and Path(cfg.manifest_path).is_file():
        manifest = json.loads(Path(cfg.manifest_path).read_text(encoding="utf-8"))
    return DatasetSource(
        polyvore_dir=cfg.polyvore_dir,
        pairs_csv=cfg.pairs_csv,
        metadata=cfg.polyvore_metadata,
        images_dir=cfg.images_dir,
        name=manifest.get("name"),
        extra={k: manifest[k] for k in ("source", "url", "license", "version") if k in manifest},
    )


def _heuristic_scores(a: np.ndarray, b: np.ndarray, slots_a: list[str | None], slots_b: list[str | None]) -> np.ndarray:
    """The app's baseline restricted to signals available for dataset items: visual + category."""
    from src.config import ScoringWeights
    from src.recommendation.compatibility import category_pair_score, rescale_cosine, weighted_mean

    w = ScoringWeights()
    vis = np.asarray(rescale_cosine(np.sum(a * b, axis=1)))
    out = np.empty(len(a))
    for i in range(len(a)):
        sa, sb = slots_a[i], slots_b[i]
        cat = category_pair_score(sa, sb) if sa and sb else None
        out[i] = weighted_mean({"visual": float(vis[i]), "category": cat}, w)
    return out


def train_compatibility(cfg: TrainConfig) -> dict[str, Any]:
    """Audit, train, evaluate against baselines, save checkpoint. Raises DatasetMissingError if the gate fails."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.compatibility_net import OutfitCompatibilityNet
    from src.models.learned import encoder_configs_match, save_checkpoint

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ---- 0. embeddings must come from the app's encoder (cheap check first)
    store = EmbeddingDirectory(cfg.embeddings_dir)
    if cfg.expected_encoder is not None and not encoder_configs_match(store.encoder_config, cfg.expected_encoder):
        raise DatasetMissingError(
            f"Embeddings in {cfg.embeddings_dir} were made with {store.encoder_config}, but the app is configured "
            f"for {cfg.expected_encoder}. Re-run scripts/embed_dataset.py with the current CLIP_MODEL."
        )

    # ---- 1. data quality gate
    source = _source(cfg)
    report, cleaned, disjoint = audit_and_prepare(
        source,
        cfg.thresholds,
        check_images=cfg.check_images and cfg.images_dir is not None,
        max_images=cfg.max_audit_images,
        seed=cfg.seed,
    )
    audit_dir = Path(cfg.audit_dir) if cfg.audit_dir else Path(cfg.out_path).parent
    write_reports(report, audit_dir / "dataset_audit.json", audit_dir / "dataset_audit.md")
    gate = report["gate"]
    if gate["verdict"] == INSUFFICIENT:
        raise DatasetMissingError(
            "Dataset quality gate: INSUFFICIENT, so no model was trained.\n  - "
            + "\n  - ".join(gate["failures"])
            + f"\nFull report: {audit_dir / 'dataset_audit.md'}"
        )
    log.info("dataset gate: %s", gate["verdict"])

    # ---- 2. pairs -> tensors (kinds kept aligned for evaluation)
    splits = cleaned if cfg.allow_item_overlap else disjoint
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    kept_pairs: dict[str, list[tuple[Any, ...]]] = {}
    info: dict[str, Any] = {
        "item_overlap_filter": "disabled (diagnostic run; evaluation may share items with training)"
        if cfg.allow_item_overlap
        else "enabled (item-disjoint)",
    }
    for split, pairs in splits.items():
        a, b, y, kept = store.tensors_with_index(pairs)
        arrays[split] = (a, b, y)
        kept_pairs[split] = [pairs[i] for i in kept]
        info[f"{split}_pairs"] = len(kept)
        info[f"{split}_pairs_missing_embedding"] = len(pairs) - len(kept)
        log.info("%s: %d pairs (%d without embeddings)", split, len(kept), len(pairs) - len(kept))
        if len(kept) == 0 or len(np.unique(y)) < 2:
            raise DatasetMissingError(f"The {split} split has no usable pairs with both labels after embedding lookup.")
    missing_rate = sum(info[f"{s}_pairs_missing_embedding"] for s in splits) / max(sum(len(p) for p in splits.values()), 1)
    if missing_rate > cfg.thresholds.max_missing_or_corrupt_rate:
        raise DatasetMissingError(
            f"{missing_rate:.1%} of pairs have no embedding. Run scripts/embed_dataset.py on all images first."
        )

    # ---- 3. train, select on validation
    dim = int(store.encoder_config["dim"])
    model = OutfitCompatibilityNet(embed_dim=dim, hidden_dims=cfg.hidden_dims, dropout=cfg.dropout).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()  # the model returns raw logits

    def loader(split: str, shuffle: bool) -> DataLoader:
        a, b, y = arrays[split]
        ds = TensorDataset(torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(y))
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=shuffle and len(ds) > cfg.batch_size)

    def predict(split: str) -> tuple[np.ndarray, float]:
        model.eval()
        probs, loss_sum = [], 0.0
        with torch.no_grad():
            for ea, eb, y in loader(split, False):
                ea, eb, y = ea.to(cfg.device), eb.to(cfg.device), y.to(cfg.device)
                logits = model(ea, eb)
                loss_sum += criterion(logits, y).item() * len(y)
                probs.append(0.5 * (torch.sigmoid(logits) + torch.sigmoid(model(eb, ea))).cpu().numpy())
        return np.concatenate(probs), loss_sum / len(arrays[split][2])

    history = []
    best_auc, best_state, best_epoch, bad_epochs = -math.inf, None, 0, 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for ea, eb, y in loader("train", True):
            ea, eb, y = ea.to(cfg.device), eb.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            loss = criterion(model(ea, eb), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
        vp, vloss = predict("valid")
        val = classification_metrics(arrays["valid"][2], vp)
        row = {
            "epoch": epoch,
            "train_loss": total / max(n, 1),
            "val_loss": vloss,
            "val_roc_auc": val["roc_auc"],
            "val_pr_auc": val["pr_auc"],
        }
        history.append(row)
        log.info("epoch %d train_loss=%.4f val_loss=%.4f val_auc=%.4f", epoch, row["train_loss"], vloss, val["roc_auc"])
        score = val["roc_auc"] if not math.isnan(val["roc_auc"]) else -vloss
        if score > best_auc:
            best_auc, best_epoch, bad_epochs = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                log.info("early stopping after epoch %d", epoch)
                break
    assert best_state is not None
    model.load_state_dict(best_state)

    vp, _ = predict("valid")
    threshold = best_f1_threshold(arrays["valid"][2], vp)  # chosen on validation only
    valid_metrics = classification_metrics(arrays["valid"][2], vp, threshold)

    # ---- 4. single evaluation on test
    tp, _ = predict("test")
    ta, tb, ty = arrays["test"]
    test_pairs = kept_pairs["test"]
    kinds = [p[3] for p in test_pairs]
    categories = load_item_categories(cfg.polyvore_metadata) if cfg.polyvore_metadata else None

    def slot(item: str) -> str | None:
        return CATEGORY_TO_SLOT.get(categories.get(item, "").lower()) if categories else None

    slots_a = [slot(p[0]) for p in test_pairs]
    slots_b = [slot(p[1]) for p in test_pairs]

    from sklearn.metrics import average_precision_score, roc_auc_score

    test = classification_metrics(ty, tp, threshold)
    test["at_threshold_0.5"] = classification_metrics(ty, tp, 0.5)
    test["roc_auc_ci95"] = bootstrap_ci(ty, tp, roc_auc_score, n=cfg.bootstrap, seed=cfg.seed)
    test["pr_auc_ci95"] = bootstrap_ci(ty, tp, average_precision_score, n=cfg.bootstrap, seed=cfg.seed)
    test["vs_random_negatives"] = positives_vs_kind_auc(ty, tp, kinds, "random_negative")
    test["vs_hard_negatives"] = positives_vs_kind_auc(ty, tp, kinds, "hard_negative")
    if categories:
        groups = ["+".join(sorted([sa or "unknown", sb or "unknown"])) for sa, sb in zip(slots_a, slots_b, strict=True)]
        test["by_category_pair"] = auc_by_group(ty, tp, groups)
    train_items = {x for p in kept_pairs["train"] for x in (p[0], p[1])}
    test["test_items_seen_in_training"] = len({x for p in test_pairs for x in (p[0], p[1])} & train_items)

    # ---- baselines on exactly the same test pairs
    rng = np.random.default_rng(cfg.seed)
    baseline_scores = {
        "random": rng.random(len(ty)),
        "clip_cosine": np.sum(ta * tb, axis=1),
        "heuristic_visual_category": _heuristic_scores(ta, tb, slots_a, slots_b),
    }
    baselines = {}
    for name, s in baseline_scores.items():
        m = classification_metrics(ty, s, threshold=float(np.median(s)), probabilistic=False)
        m["roc_auc_ci95"] = bootstrap_ci(ty, s, roc_auc_score, n=cfg.bootstrap, seed=cfg.seed)
        m["vs_hard_negatives"] = positives_vs_kind_auc(ty, s, kinds, "hard_negative")
        baselines[name] = {k: m[k] for k in ("roc_auc", "pr_auc", "roc_auc_ci95", "vs_hard_negatives")}
    strongest = max(("clip_cosine", "heuristic_visual_category"), key=lambda k: baselines[k]["roc_auc"])
    diff = paired_bootstrap_auc_difference(ty, tp, baseline_scores[strongest], n=cfg.bootstrap, seed=cfg.seed)
    comparison = {
        "strongest_non_random_baseline": strongest,
        "learned_minus_baseline_roc_auc": diff,
        "beats_best_baseline": bool(diff["ci95_low"] > 0),
        "note": "beats_best_baseline requires the 95% CI of the paired ROC-AUC difference to exclude 0.",
    }

    metrics = {
        "best_epoch": best_epoch,
        "threshold_selected_on": "validation (max F1)",
        "valid": valid_metrics,
        "test": test,
        "baselines": baselines,
        "comparison": comparison,
    }
    training_info = {
        **info,
        "dataset_gate": gate["verdict"],
        "dataset_limitations": gate["limitations"],
        "dataset": report["dataset"],
        "dataset_effective_size": report["effective"]["after_item_disjoint_split"],
        "epochs_run": len(history),
        "seconds": round(time.time() - t0, 1),
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(cfg).items()
            if k not in ("expected_encoder", "thresholds", "extra")
        },
        "history": history,
    }
    save_checkpoint(cfg.out_path, model, store.encoder_config, metrics, training_info)
    return {"metrics": metrics, "training": training_info, "checkpoint": str(cfg.out_path), "audit": report}
