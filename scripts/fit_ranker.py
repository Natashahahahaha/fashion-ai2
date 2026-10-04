"""Fit the pair ranker and the outfit quality threshold on real outfit data.

    python scripts/fit_ranker.py            # uses data/datasets/maryland-polyvore (download + embed first)

What it does (no test data is used for any choice):
  1. rebuilds the audited, item-disjoint pair splits (same code/seeds as training);
  2. computes pair features (learned model, CLIP cosine, colour relation,
     style agreement, category pair) exactly as the app does;
  3. fits two logistic-regression combiners on the VALIDATION pairs
     (with and without the learned feature). Weights are constrained to be
     non-negative: every signal is defined so that higher = more compatible,
     so a feature that receives a negative weight is contradicted by the data;
     it is dropped (and the reason recorded) and the model is refitted;
  4. builds outfit-level sets: real validation outfits vs category-preserving
     fake outfits (every item swapped for a random item of the same slot), and
     picks the quality threshold maximising TPR - FPR on VALIDATION;
  5. reports everything once on TEST and writes src/recommendation/ranker_model.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap
import numpy as np
from PIL import Image

from src.attributes.colors import dominant_colors
from src.attributes.style import StyleClassifier
from src.config import get_settings, resolve_device
from src.embeddings.clip_encoder import get_encoder
from src.models.learned import LearnedCompatibility
from src.recommendation.pair_model import (
    FEATURES_BASELINE,
    FEATURES_WITH_LEARNED,
    MODEL_PATH,
    ItemTable,
    outfit_plausibility,
    pair_features,
)
from src.training.audit import DatasetSource, audit_and_prepare
from src.training.data import EmbeddingDirectory, load_polyvore_outfits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=_bootstrap.ROOT / "data" / "datasets" / "maryland-polyvore")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    D = args.dataset_dir
    prepared, emb_dir = D / "prepared", D / "embeddings"
    if not (prepared / "valid.json").is_file() or not emb_dir.is_dir():
        print("error: dataset not prepared/embedded. Run download_dataset.py and embed_dataset.py first.", file=sys.stderr)
        return 2
    settings = get_settings()
    meta = json.loads((prepared / "metadata.json").read_text(encoding="utf-8"))

    print("rebuilding audited item-disjoint splits...")
    report, _, splits = audit_and_prepare(
        DatasetSource(polyvore_dir=prepared, metadata=prepared / "metadata.json", images_dir=prepared / "images"),
        check_images=True,
        seed=args.seed,
    )
    store = EmbeddingDirectory(emb_dir)
    learned = LearnedCompatibility.load(settings.compatibility_checkpoint, store.encoder_config, device=resolve_device("auto"))

    valid_outfits = load_polyvore_outfits(prepared / "valid.json")
    test_outfits = load_polyvore_outfits(prepared / "test.json")
    items = sorted(
        {x for s in ("valid", "test") for p in splits[s] for x in (p[0], p[1])}
        | {i for o in valid_outfits + test_outfits for i in o}
    )
    items = [i for i in items if store.get(i) is not None]
    index = {iid: k for k, iid in enumerate(items)}
    print(f"computing colours + style for {len(items)} items...")
    E = np.stack([store.get(i) for i in items]).astype(np.float32)  # type: ignore[misc]
    cache_path = D / "colour_cache.json"  # pixel statistics only; safe to delete
    cache: dict[str, list[dict] | None] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    colours: list[list[dict] | None] = []
    for k, iid in enumerate(items):
        if iid not in cache:
            with Image.open(prepared / "images" / f"{iid}.jpg") as im:
                cache[iid] = dominant_colors(im.convert("RGB"))
        colours.append(cache[iid])
        if (k + 1) % 5000 == 0:
            print(f"  {k + 1}/{len(items)}", flush=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    clf = StyleClassifier(get_encoder(settings))
    T = clf._text_matrix()
    logits = 100.0 * (E @ T.T)
    logits -= logits.max(axis=1, keepdims=True)
    style = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    table = ItemTable(ids=items, embeddings=E, slots=[meta[i]["semantic_category"] for i in items], colours=colours, style=style)

    def split_arrays(split: str):
        pairs = [p for p in splits[split] if p[0] in index and p[1] in index]
        ia = np.array([index[p[0]] for p in pairs])
        ib = np.array([index[p[1]] for p in pairs])
        y = np.array([p[2] for p in pairs])
        kinds = np.array([p[3] for p in pairs])
        return pair_features(table, ia, ib, learned), y, kinds

    fv, yv, kv = split_arrays("valid")
    ft, yt, kt = split_arrays("test")
    print(f"pairs: valid {len(yv)}, test {len(yt)}")

    rng = np.random.default_rng(args.seed)
    pools: dict[str, list[int]] = {}
    for i, s in enumerate(table.slots):
        pools.setdefault(str(s), []).append(i)

    def outfit_sets(outfits: list[list[str]]):
        real, fake = [], []
        for o in outfits:
            idx = [index[i] for i in o if i in index]
            if len(idx) < 2:
                continue
            real.append(idx)
            fake.append([int(rng.choice(pools[str(table.slots[i])])) for i in idx])
        return real, fake

    out: dict = {
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fitted_on": "validation split (pairs and outfits); metrics below are on the untouched test split",
        "dataset": report["dataset"],
        "negative_strategy": "pairs: observed vs constructed (random + hard); outfits: real vs category-preserving random swaps",
        "variants": {},
    }
    for variant, all_feats in (("with_learned", FEATURES_WITH_LEARNED), ("baseline", FEATURES_BASELINE)):
        feats = tuple(all_feats)
        dropped: dict[str, str] = {}
        while True:
            Xv = np.stack([fv[f] for f in feats], axis=1)
            impute = np.nanmean(Xv, axis=0)
            Xv = np.where(np.isnan(Xv), impute, Xv)
            mean, scale = Xv.mean(axis=0), Xv.std(axis=0) + 1e-9
            lr = LogisticRegression(C=1.0, max_iter=1000).fit((Xv - mean) / scale, yv)
            worst = int(np.argmin(lr.coef_[0]))
            if lr.coef_[0][worst] >= 0:
                break
            name = feats[worst]
            dropped[name] = (
                f"negative fitted weight ({lr.coef_[0][worst]:+.3f}) on validation pairs: "
                "the signal is contradicted by real outfits, so it is not used for ranking"
            )
            print(f"[{variant}] dropping {name}: {dropped[name]}")
            feats = tuple(f for f in feats if f != name)

        def score(f: dict, feats=feats, lr=lr, impute=impute, mean=mean, scale=scale) -> np.ndarray:
            X = np.stack([f[n] for n in feats], axis=1)
            X = np.where(np.isnan(X), impute, X)
            return lr.predict_proba((X - mean) / scale)[:, 1]

        pv, pt = score(fv), score(ft)

        def outfit_scores(real_fake, feats_fn=score, feats=feats):
            res = []
            for idx in real_fake:
                pairs = [(a, b) for k, a in enumerate(idx) for b in idx[k + 1 :]]
                ia, ib = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])
                p = feats_fn(pair_features(table, ia, ib, learned if "learned_logit" in feats else None))
                res.append(
                    outfit_plausibility([(table.slots[a], table.slots[b], float(q)) for (a, b), q in zip(pairs, p, strict=True)])
                )
            return np.array(res)

        rv, fk = outfit_sets(valid_outfits)
        sv_real, sv_fake = outfit_scores(rv), outfit_scores(fk)
        cand = np.unique(np.concatenate([sv_real, sv_fake]))
        tpr = np.array([(sv_real >= c).mean() for c in cand])
        fpr = np.array([(sv_fake >= c).mean() for c in cand])
        thr = float(cand[int(np.argmax(tpr - fpr))])
        rt, ft_ = outfit_sets(test_outfits)
        st_real, st_fake = outfit_scores(rt), outfit_scores(ft_)
        hard = (yt == 1) | (kt == "hard_negative")
        metrics = {
            "pair_test_roc_auc": float(roc_auc_score(yt, pt)),
            "pair_test_roc_auc_hard_negatives": float(roc_auc_score(yt[hard], pt[hard])),
            "pair_valid_roc_auc": float(roc_auc_score(yv, pv)),
            "outfit_threshold_chosen_on": "validation, max(TPR - FPR) real vs category-preserving fake outfits",
            "outfit_valid_tpr": float((sv_real >= thr).mean()),
            "outfit_valid_fpr": float((sv_fake >= thr).mean()),
            "outfit_test_tpr": float((st_real >= thr).mean()),
            "outfit_test_fpr": float((st_fake >= thr).mean()),
            "outfit_test_roc_auc": float(
                roc_auc_score(np.r_[np.ones(len(st_real)), np.zeros(len(st_fake))], np.r_[st_real, st_fake])
            ),
            "n_valid_outfits": len(rv),
            "n_test_outfits": len(rt),
        }
        out["variants"][variant] = {
            "features": list(feats),
            "dropped_features": dropped,
            "coef": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0]),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "impute": impute.tolist(),
            "outfit_threshold": thr,
            "metrics": metrics,
        }
        print(
            f"\n[{variant}] coefficients (standardised): "
            + ", ".join(f"{f}={c:+.3f}" for f, c in zip(feats, lr.coef_[0], strict=True))
        )
        print(json.dumps(metrics, indent=2))

    single = {
        "learned_alone_test_roc_auc": float(roc_auc_score(yt, ft["learned_logit"])),
        "clip_alone_test_roc_auc": float(roc_auc_score(yt, ft["clip_cos"])),
    }
    out["reference"] = single
    print("\nreference single signals on test:", json.dumps(single))
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
