"""Dataset quality audit and training gate for the learned compatibility model.

The audit answers one question with evidence: *is this dataset sufficient to
train a credible pairwise outfit-compatibility model?* It reports raw
counts, category coverage, image health, exact/perceptual duplicates,
cross-split leakage and the effective size after cleaning and item-disjoint
splitting, then classifies the dataset as

    SUFFICIENT | SUFFICIENT_WITH_LIMITATIONS | INSUFFICIENT

The thresholds below are conservative engineering judgements for this task
(documented, not derived from a study). ``train_compatibility`` refuses to
train on an INSUFFICIENT dataset.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.errors import DatasetMissingError
from src.training.data import (
    NEGATIVE_STRATEGY,
    SPLITS,
    build_outfit_pairs_typed,
    load_pairs_csv,
    load_polyvore_outfits,
    split_pairs,
)

AUDIT_SCHEMA_VERSION = 1
SUFFICIENT = "SUFFICIENT"
SUFFICIENT_WITH_LIMITATIONS = "SUFFICIENT_WITH_LIMITATIONS"
INSUFFICIENT = "INSUFFICIENT"

# Polyvore semantic categories (and our own slot names) -> outfit slots.
CATEGORY_TO_SLOT = {
    "tops": "top", "top": "top",
    "bottoms": "bottom", "bottom": "bottom",
    "all-body": "one_piece", "one_piece": "one_piece", "dresses": "one_piece",
    "outerwear": "outerwear",
    "shoes": "shoes",
    "bags": "accessory", "jewellery": "accessory", "jewelry": "accessory", "accessories": "accessory",
    "sunglasses": "accessory", "hats": "accessory", "scarves": "accessory", "accessory": "accessory",
}  # fmt: skip
CORE_SLOTS = ("top", "bottom", "shoes")
EXTENDED_SLOTS = ("one_piece", "outerwear", "accessory")


@dataclass(frozen=True)
class QualityThresholds:
    """Minimums for a *credible* model on this task (engineering judgement)."""

    min_train_outfits: int = 2000  # Polyvore-format datasets only
    min_unique_items: int = 5000
    min_train_positive_pairs: int = 10000
    min_valid_positive_pairs: int = 500
    min_test_positive_pairs: int = 1000
    max_top_category_share: float = 0.5
    max_missing_or_corrupt_rate: float = 0.05
    max_duplicate_image_rate: float = 0.05
    max_cross_split_duplicate_rate: float = 0.01


@dataclass
class DatasetSource:
    polyvore_dir: Path | None = None  # contains train.json / valid.json / test.json
    pairs_csv: Path | None = None
    metadata: Path | None = None  # item_id -> {"semantic_category": ..., "title": ...}
    images_dir: Path | None = None  # <item_id>.jpg
    name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _display_path(path: Path | None) -> str:
    """Project-relative path when possible, so reports carry no user-specific absolute paths."""
    from src.config import PROJECT_ROOT

    if path is None:
        return ""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


# ----------------------------------------------------------------- images
def _dhash(path: Path) -> str | None:
    """64-bit difference hash (robust to resizing/recompression)."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((9, 8))
            px = list(g.getdata())
    except Exception:
        return None
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return f"{bits:016x}"


def audit_images(images_dir: Path, item_ids: list[str], max_images: int | None = None) -> dict[str, Any]:
    """Missing/corrupt files, resolution distribution, exact and perceptual duplicates."""
    from PIL import Image

    ids = sorted(item_ids)
    sampled = max_images is not None and len(ids) > max_images
    if sampled:
        step = len(ids) / max_images  # type: ignore[operator]
        ids = [ids[int(i * step)] for i in range(max_images)]  # type: ignore[arg-type]
    missing, corrupt = [], []
    widths, heights = [], []
    by_sha: dict[str, list[str]] = {}
    by_dhash: dict[str, list[str]] = {}
    for item_id in ids:
        path = images_dir / f"{item_id}.jpg"
        if not path.is_file():
            missing.append(item_id)
            continue
        try:
            data = path.read_bytes()
            with Image.open(path) as im:
                im.load()
                widths.append(im.width)
                heights.append(im.height)
        except Exception:
            corrupt.append(item_id)
            continue
        by_sha.setdefault(hashlib.sha1(data).hexdigest(), []).append(item_id)
        dh = _dhash(path)
        if dh is not None:
            by_dhash.setdefault(dh, []).append(item_id)
    exact_groups = [g for g in by_sha.values() if len(g) > 1]
    percept_groups = [g for g in by_dhash.values() if len(g) > 1]
    checked = len(ids)
    ok = checked - len(missing) - len(corrupt)

    def dist(v: list[int]) -> dict[str, float]:
        return {"min": min(v), "median": statistics.median(v), "max": max(v)} if v else {}

    return {
        "checked": True,
        "sampled": sampled,
        "items_checked": checked,
        "found_ok": ok,
        "missing": len(missing),
        "corrupt": len(corrupt),
        "missing_or_corrupt_rate": (len(missing) + len(corrupt)) / checked if checked else 0.0,
        "width": dist(widths),
        "height": dist(heights),
        "small_images_lt_64px": sum(1 for w, h in zip(widths, heights, strict=True) if min(w, h) < 64),
        "exact_duplicate_groups": len(exact_groups),
        "items_in_exact_duplicates": sum(len(g) for g in exact_groups),
        "perceptual_duplicate_groups": len(percept_groups),
        "items_in_perceptual_duplicates": sum(len(g) for g in percept_groups),
        "duplicate_image_rate": (sum(len(g) - 1 for g in percept_groups) / ok) if ok else 0.0,
        "_missing_ids": missing,
        "_corrupt_ids": corrupt,
        "_dup_groups": percept_groups,
    }


# ------------------------------------------------------------------ audit
def _load_categories(metadata: Path | None) -> tuple[dict[str, str] | None, dict[str, Any]]:
    if metadata is None:
        return None, {"available": False}
    if not metadata.is_file():
        raise DatasetMissingError(f"Metadata file not found: {metadata}")
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    fields: Counter[str] = Counter()
    text_items = 0
    cats: dict[str, str] = {}
    for item_id, info in meta.items():
        if not isinstance(info, dict):
            continue
        fields.update(info.keys())
        if any(str(info.get(k) or "").strip() for k in ("title", "description", "url_name")):
            text_items += 1
        cats[str(item_id)] = str(info.get("semantic_category") or info.get("category") or info.get("category_id") or "unknown")
    return cats, {
        "available": True,
        "items_with_metadata": len(cats),
        "fields": sorted(fields),
        "items_with_text": text_items,
        "has_text": text_items > 0,
    }


def _split_stats(pairs: list[tuple[Any, ...]]) -> dict[str, int]:
    kinds = Counter(p[3] for p in pairs)
    return {
        "items": len({x for p in pairs for x in (p[0], p[1])}),
        "positive_pairs": sum(1 for p in pairs if p[2] == 1),
        "negative_pairs": sum(1 for p in pairs if p[2] == 0),
        **{f"kind_{k}": v for k, v in sorted(kinds.items())},
    }


def _remove_overlap(splits: dict[str, list[tuple[Any, ...]]]) -> dict[str, int]:
    train_items = {x for p in splits["train"] for x in (p[0], p[1])}
    before_v = len(splits["valid"])
    splits["valid"] = [p for p in splits["valid"] if p[0] not in train_items and p[1] not in train_items]
    seen = train_items | {x for p in splits["valid"] for x in (p[0], p[1])}
    before_t = len(splits["test"])
    splits["test"] = [p for p in splits["test"] if p[0] not in seen and p[1] not in seen]
    return {"valid": before_v - len(splits["valid"]), "test": before_t - len(splits["test"])}


def audit_and_prepare(
    source: DatasetSource,
    thresholds: QualityThresholds | None = None,
    check_images: bool = True,
    max_images: int | None = None,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, list[tuple[Any, ...]]], dict[str, list[tuple[Any, ...]]]]:
    """Audit a dataset; also return the cleaned and the cleaned+item-disjoint typed pair splits."""
    th = thresholds or QualityThresholds()
    if source.polyvore_dir is None and source.pairs_csv is None:
        raise DatasetMissingError("Nothing to audit: give a Polyvore split directory or a pairs CSV.")
    categories, meta_info = _load_categories(source.metadata)
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": source.name or ("polyvore-format" if source.polyvore_dir else "pairs-csv"),
            "format": "polyvore_outfits" if source.polyvore_dir else "pairs_csv",
            "path": _display_path(source.polyvore_dir or source.pairs_csv),
            "users_or_accounts": "not available in this format",
            **source.extra,
        },
        "metadata": meta_info,
        "thresholds": asdict(th),
        "negative_strategy": NEGATIVE_STRATEGY if source.polyvore_dir else "Labels supplied by the CSV (not constructed).",
    }

    # ---- raw structure and pairs
    raw_splits: dict[str, list[tuple[Any, ...]]] = {}
    outfits_by_split: dict[str, list[list[str]]] = {}
    if source.polyvore_dir is not None:
        for offset, split in enumerate(SPLITS):
            outfits = load_polyvore_outfits(source.polyvore_dir / f"{split}.json")
            outfits_by_split[split] = outfits
            raw_splits[split] = list(build_outfit_pairs_typed(outfits, categories, seed=seed + offset))
        all_items = sorted({i for o in sum(outfits_by_split.values(), []) for i in o})
        sizes = Counter(len(o) for o in sum(outfits_by_split.values(), []))
        outfit_keys = {s: {frozenset(o) for o in outfits_by_split[s]} for s in SPLITS}
        report["raw"] = {
            "outfits": {s: len(outfits_by_split[s]) for s in SPLITS},
            "unique_outfits_total": len(set().union(*outfit_keys.values())),
            "items": len(all_items),
            "outfit_size_distribution": {str(k): v for k, v in sorted(sizes.items())},
            "duplicate_outfits_within_splits": {s: len(outfits_by_split[s]) - len(outfit_keys[s]) for s in SPLITS},
        }
        item_split_sets = {s: {i for o in outfits_by_split[s] for i in o} for s in SPLITS}
        report["leakage"] = {
            "items_shared": {
                "train-valid": len(item_split_sets["train"] & item_split_sets["valid"]),
                "train-test": len(item_split_sets["train"] & item_split_sets["test"]),
                "valid-test": len(item_split_sets["valid"] & item_split_sets["test"]),
            },
            "identical_outfits_across_splits": len(outfit_keys["train"] & (outfit_keys["valid"] | outfit_keys["test"])),
            "items_in_multiple_outfits": sum(
                1 for c in Counter(i for o in sum(outfits_by_split.values(), []) for i in set(o)).values() if c > 1
            ),
        }
    else:
        assert source.pairs_csv is not None
        rows = load_pairs_csv(source.pairs_csv)
        csv_splits, dropped = split_pairs(rows, seed=seed)
        raw_splits = {
            s: [(a, b, y, "labelled_positive" if y == 1 else "labelled_negative") for a, b, y in csv_splits[s]] for s in SPLITS
        }
        all_items = sorted({x for a, b, _, _ in rows for x in (a, b)})
        report["raw"] = {
            "pairs": len(rows),
            "positive_pairs": sum(1 for r in rows if r[2] == 1),
            "negative_pairs": sum(1 for r in rows if r[2] == 0),
            "items": len(all_items),
            "explicit_split_column": any(r[3] for r in rows),
            "cross_split_pairs_dropped_by_hash_split": dropped,
        }
        sets = {s: {x for p in raw_splits[s] for x in (p[0], p[1])} for s in SPLITS}
        report["leakage"] = {
            "items_shared": {
                "train-valid": len(sets["train"] & sets["valid"]),
                "train-test": len(sets["train"] & sets["test"]),
                "valid-test": len(sets["valid"] & sets["test"]),
            }
        }
    report["ids"] = {"unique_item_ids": len(all_items), "all_string_ids": all(isinstance(i, str) for i in all_items)}
    if categories is not None:
        report["ids"]["items_missing_metadata"] = sum(1 for i in all_items if i not in categories)

    # ---- categories
    if categories is not None:
        cat_counts = Counter(categories.get(i, "unknown") for i in all_items)
        slot_counts = Counter(CATEGORY_TO_SLOT.get(c.lower(), "other") for c in cat_counts.elements())
        total = sum(cat_counts.values()) or 1
        report["categories"] = {
            "n_categories": len(cat_counts),
            "distribution": dict(cat_counts.most_common()),
            "slot_coverage": dict(slot_counts.most_common()),
            "top_category_share": cat_counts.most_common(1)[0][1] / total,
            "imbalance_ratio_max_min": cat_counts.most_common()[0][1] / max(cat_counts.most_common()[-1][1], 1),
        }
    else:
        report["categories"] = {"available": False}

    # ---- images
    dup_groups: list[list[str]] = []
    bad_items: set[str] = set()
    if source.images_dir is not None and check_images:
        img = audit_images(source.images_dir, all_items, max_images)
        bad_items = set(img.pop("_missing_ids")) | set(img.pop("_corrupt_ids"))
        dup_groups = img.pop("_dup_groups")
        report["images"] = img
        if source.polyvore_dir is not None:
            split_of = {i: s for s in SPLITS for o in outfits_by_split[s] for i in o}
            cross = sum(1 for g in dup_groups if len({split_of.get(i) for i in g}) > 1)
            report["leakage"]["perceptual_duplicate_groups_across_splits"] = cross
    else:
        report["images"] = {"checked": False, "reason": "no images directory given" if source.images_dir is None else "skipped"}

    # ---- cleaning: drop pairs with missing/corrupt images, collapse perceptual duplicates
    canonical = {i: g[0] for g in dup_groups for i in g}

    def clean(pairs: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
        out = []
        for p in pairs:
            a, b = canonical.get(p[0], p[0]), canonical.get(p[1], p[1])
            if a in bad_items or b in bad_items or a == b:
                continue
            out.append((a, b, *p[2:]))
        return out

    cleaned = {s: clean(raw_splits[s]) for s in SPLITS}
    report["effective"] = {
        "raw_pairs": {s: _split_stats(raw_splits[s]) for s in SPLITS},
        "after_cleaning": {s: _split_stats(cleaned[s]) for s in SPLITS},
    }
    disjoint = {s: list(v) for s, v in cleaned.items()}
    dropped_overlap = _remove_overlap(disjoint)
    report["effective"]["item_disjoint_pairs_dropped"] = dropped_overlap
    report["effective"]["after_item_disjoint_split"] = {s: _split_stats(disjoint[s]) for s in SPLITS}
    if source.polyvore_dir is not None:
        train_items = {i for o in outfits_by_split["train"] for i in o}
        report["effective"]["fully_item_disjoint_outfits"] = {
            "valid": sum(1 for o in outfits_by_split["valid"] if not set(o) & train_items),
            "test": sum(1 for o in outfits_by_split["test"] if not set(o) & train_items),
        }

    report["gate"] = _decide(report, th, source)
    return report, cleaned, disjoint


def audit_dataset(
    source: DatasetSource,
    thresholds: QualityThresholds | None = None,
    check_images: bool = True,
    max_images: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Audit a dataset and return a JSON-serialisable report (see module docstring)."""
    return audit_and_prepare(source, thresholds, check_images, max_images, seed)[0]


def _decide(report: dict[str, Any], th: QualityThresholds, source: DatasetSource) -> dict[str, Any]:
    failures: list[str] = []
    limitations: list[str] = []
    missing: list[str] = []
    eff = report["effective"]["after_item_disjoint_split"]

    if source.polyvore_dir is not None:
        n_train = report["raw"]["outfits"]["train"]
        if n_train < th.min_train_outfits:
            failures.append(f"only {n_train} training outfits (need >= {th.min_train_outfits})")
            missing.append(f"at least {th.min_train_outfits - n_train} more real training outfits")
    elif report["raw"]["positive_pairs"] == 0:
        failures.append("no positive (compatible) pairs in the CSV")

    n_items = report["ids"]["unique_item_ids"]
    if n_items < th.min_unique_items:
        failures.append(f"only {n_items} unique items (need >= {th.min_unique_items})")
        missing.append(f"at least {th.min_unique_items - n_items} more distinct items")
    for split, need in (
        ("train", th.min_train_positive_pairs),
        ("valid", th.min_valid_positive_pairs),
        ("test", th.min_test_positive_pairs),
    ):
        got = eff[split]["positive_pairs"]
        if got < need:
            failures.append(f"{split}: {got} positive pairs after cleaning + item-disjoint split (need >= {need})")
    for split in SPLITS:
        if eff[split]["negative_pairs"] == 0:
            failures.append(f"{split}: no negative pairs after filtering")

    cats = report.get("categories", {})
    if cats.get("available", True) and "slot_coverage" in cats:
        cov = cats["slot_coverage"]
        absent_core = [s for s in CORE_SLOTS if cov.get(s, 0) == 0]
        if absent_core:
            failures.append(f"no items in core categories: {', '.join(absent_core)}")
        absent_ext = [s for s in EXTENDED_SLOTS if cov.get(s, 0) == 0]
        if absent_ext:
            limitations.append(f"no items for: {', '.join(absent_ext)}")
        if cats["top_category_share"] > th.max_top_category_share:
            failures.append(f"severe category imbalance: largest category is {cats['top_category_share']:.0%} of items")
    else:
        limitations.append("no category metadata: hard negatives and per-category evaluation are impossible")

    img = report["images"]
    if img.get("checked"):
        if img["missing_or_corrupt_rate"] > th.max_missing_or_corrupt_rate:
            failures.append(
                f"{img['missing_or_corrupt_rate']:.1%} of images missing/corrupt (max {th.max_missing_or_corrupt_rate:.0%})"
            )
        if img["duplicate_image_rate"] > th.max_duplicate_image_rate:
            failures.append(f"duplicate-image rate {img['duplicate_image_rate']:.1%} (max {th.max_duplicate_image_rate:.0%})")
        if img["sampled"]:
            limitations.append(f"images audited on a sample of {img['items_checked']} items")
        cross = report["leakage"].get("perceptual_duplicate_groups_across_splits")
        if cross:
            rate = cross / max(n_items, 1)
            (failures if rate > th.max_cross_split_duplicate_rate else limitations).append(
                f"{cross} duplicate-image groups span different splits (collapsed during cleaning)"
            )
    else:
        limitations.append("images not checked (corrupt/duplicate rates unknown)")

    shared = report["leakage"]["items_shared"]
    if any(shared.values()):
        limitations.append(
            f"raw splits share items {shared}; evaluation uses item-disjoint filtering "
            f"(dropped {report['effective']['item_disjoint_pairs_dropped']})"
        )
    limitations.append("no user/account identifiers, so user-level leakage cannot be checked")
    if source.polyvore_dir is not None and report["leakage"].get("items_in_multiple_outfits", 1) == 0:
        limitations.append(
            "item ids are unique per outfit, so reuse of the same garment across outfits/splits is only detectable "
            "through image hashing (exact/dHash); different photos of the same product are not detected"
        )
    if report["dataset"].get("license") in (None, "", "unknown"):
        limitations.append("license not recorded; verify terms before redistribution")

    verdict = INSUFFICIENT if failures else (SUFFICIENT_WITH_LIMITATIONS if limitations else SUFFICIENT)
    return {"verdict": verdict, "failures": failures, "limitations": limitations, "what_is_missing": missing}


# --------------------------------------------------------------- reports
def write_reports(report: dict[str, Any], json_path: Path, md_path: Path | None = None) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if md_path is not None:
        md_path.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(r: dict[str, Any]) -> str:
    g = r["gate"]
    lines = [
        f"# Dataset audit: {r['dataset']['name']}",
        "",
        f"Generated {r['generated_at']} (schema v{r['schema_version']}).",
        "",
        f"## Verdict: **{g['verdict']}**",
        "",
        "Question: *Is this dataset sufficient to train a credible fashion compatibility model?*",
        "",
    ]
    if g["failures"]:
        lines += ["### Blocking problems", *[f"- {x}" for x in g["failures"]], ""]
    if g["what_is_missing"]:
        lines += ["### What would be needed", *[f"- {x}" for x in g["what_is_missing"]], ""]
    if g["limitations"]:
        lines += ["### Limitations", *[f"- {x}" for x in g["limitations"]], ""]
    lines += ["## Dataset", "```json", json.dumps(r["dataset"], indent=2, default=str), "```", ""]
    lines += ["## Raw structure", "```json", json.dumps(r["raw"], indent=2), "```", ""]
    lines += ["## Categories", "```json", json.dumps(r.get("categories"), indent=2), "```", ""]
    lines += ["## Images", "```json", json.dumps(r.get("images"), indent=2), "```", ""]
    lines += ["## Leakage", "```json", json.dumps(r.get("leakage"), indent=2), "```", ""]
    lines += ["## Negative construction", r["negative_strategy"], ""]
    lines += ["## Effective size", "```json", json.dumps(r["effective"], indent=2), "```", ""]
    return "\n".join(lines) + "\n"
