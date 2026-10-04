"""Outfit generation over the FULL eligible wardrobe.

Pipeline (each stage is counted in ``OutfitResult.diagnostics``):

    wardrobe -> eligibility (known category, not awaiting review, valid embedding)
             -> pair scores for ALL eligible item pairs (one batched pass; learned
                model + colour + style + CLIP + category via the fitted pair model)
             -> structural templates (top+bottom+shoes, one-piece+shoes) enumerated
                with numpy over the full slot pools
             -> hard validity gate (conflicting items masked out BEFORE ranking)
             -> optional layers (best outerwear / accessory per candidate, if it helps)
             -> quality threshold (fitted on validation outfits; below = not shown)
             -> diversity-aware selection (MMR + max shared core garments)
             -> at most k outfits; fewer if fewer are good. Never padded.

The only size limit is a compute guard (``max_candidates`` base combinations
per template, default 200,000). If a template exceeds it, slot pools are
pruned by each item's best pair score and the pruning is reported.
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.attributes.colors import outfit_color_summary
from src.config import ScoringWeights
from src.detection.categories import ACCESSORY, BOTTOM, OUTERWEAR, SHOES, SLOT_DISPLAY, TOP
from src.recommendation.compatibility import PairScore, category_pair_score, item_context
from src.recommendation.pair_model import ACCESSORY_PAIR_WEIGHT, ItemTable, PairModel, load_pair_model, pair_features
from src.recommendation.validity import CORE_TEMPLATES, conflict_matrix, structure_text
from src.wardrobe.manager import DUPLICATE_COSINE, WardrobeManager, exclusion_reason
from src.wardrobe.schemas import WardrobeItem

log = logging.getLogger(__name__)

DIVERSITY_LAMBDA = 0.6  # MMR trade-off: score - lambda * item overlap with already selected looks
MAX_SHARED_CORE = 0.5  # a new look may share at most half of its core garments with any selected look
MODERATE_CONFIDENCE = 0.40  # detector score below this is mentioned as a caveat on the card
STRONG_CONTRIBUTION = 0.15  # |standardised logit contribution| above this is called out in explanations
MMR_POOL = 5000  # top-ranked strong candidates given to the (quadratic) diversity step


@dataclass
class OutfitScore:
    score: float  # outfit plausibility (weighted mean pair score)
    components: dict[str, float | None]
    reasons: list[str] = field(default_factory=list)  # concise "why"
    caveats: list[str] = field(default_factory=list)  # confidence/quality notes
    explanation: dict[str, str] = field(default_factory=dict)
    rank_score: float = 0.0
    complete: bool = True


@dataclass
class Outfit:
    items: list[WardrobeItem]
    result: OutfitScore

    @property
    def score(self) -> float:
        return self.result.score

    @property
    def item_ids(self) -> list[str]:
        return [it.id for it in self.items]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "items": [
                {"id": it.id, "category": it.category, "label": it.label, "image_path": it.image_path} for it in self.items
            ],
            "components": {k: (None if v is None else round(v, 4)) for k, v in self.result.components.items()},
            "reasons": list(self.result.reasons),
            "caveats": list(self.result.caveats),
            **self.result.explanation,
        }


@dataclass
class OutfitResult:
    outfits: list[Outfit] = field(default_factory=list)
    message: str | None = None
    requested: int = 0
    strong_candidates: int = 0  # candidates above the quality threshold (before diversity)
    skipped_items: list[str] = field(default_factory=list)  # excluded item ids
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def combinations_scored(self) -> int:
        return int(self.diagnostics.get("candidates_scored", 0))


@dataclass
class ItemRecommendation:
    item: WardrobeItem
    pair: PairScore


class _Scored:
    """Pair-level matrices for the eligible items (computed once per request, batched)."""

    def __init__(self, items: list[WardrobeItem], embeddings: dict[str, np.ndarray], model: PairModel, learned: Any | None):
        self.items = items
        self.index = {it.id: i for i, it in enumerate(items)}
        n = len(items)
        self.table = ItemTable.from_wardrobe(items, embeddings)
        self.features: dict[str, np.ndarray] = {}
        self.contrib: dict[str, np.ndarray] = {}
        self.P = np.zeros((n, n))
        if n >= 2:
            ia, ib = np.triu_indices(n, 1)
            feats = pair_features(self.table, ia, ib, learned if "learned_logit" in model.features else None)
            p = model.score(feats)
            self.P[ia, ib] = p
            self.P[ib, ia] = p
            if "learned_logit" in feats:
                feats = {**feats, "learned_prob": 1 / (1 + np.exp(-feats["learned_logit"]))}
            for name, vals in feats.items():
                m = np.full((n, n), np.nan)
                m[ia, ib] = vals
                m[ib, ia] = vals
                self.features[name] = m
            for name, vals in model.contributions(feats).items():
                m = np.zeros((n, n))
                m[ia, ib] = vals
                m[ib, ia] = vals
                self.contrib[name] = m
        self.conflicts = conflict_matrix(items, self.table.embeddings if n else None)
        self.groups = self._duplicate_groups()

    def _duplicate_groups(self) -> list[int]:
        """Near-duplicate group per item (same category + near-identical image)."""
        n = len(self.items)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        if n:
            cos = self.table.embeddings @ self.table.embeddings.T
            for i, j in zip(*np.where(np.triu(cos >= DUPLICATE_COSINE, 1)), strict=True):
                if self.items[int(i)].category == self.items[int(j)].category:
                    parent[find(int(i))] = find(int(j))
        return [find(i) for i in range(n)]


class OutfitGenerator:
    def __init__(
        self,
        manager: WardrobeManager,
        learned: Any | None = None,
        pair_model: PairModel | None = None,
        weights: ScoringWeights | None = None,
        max_candidates: int = 200_000,
        quality_threshold: float | None = None,
        diversity_lambda: float = DIVERSITY_LAMBDA,
        max_shared_core: float = MAX_SHARED_CORE,
    ):
        self.manager = manager
        self.learned = learned
        self.model = pair_model or load_pair_model(with_learned=learned is not None)
        self.weights = weights or ScoringWeights()
        self.max_candidates = max_candidates
        self.threshold = quality_threshold if quality_threshold is not None else self.model.outfit_threshold
        self.diversity_lambda = diversity_lambda
        self.max_shared_core = max_shared_core

    @property
    def scoring_mode(self) -> str:
        names = {
            "learned_logit": "learned compatibility model",
            "clip_cos": "CLIP similarity",
            "colour": "colour",
            "style_sim": "style",
            "category": "garment-type pairing",
        }
        used = " + ".join(names.get(f, f) for f in self.model.features)
        how = "fitted combination" if self.model.fitted else "UNFITTED equal weights"
        if "learned_logit" not in self.model.features:
            how += ", no learned model"
        return f"{used} ({how})"

    # --------------------------------------------------------------- inputs
    def _eligible(self) -> tuple[list[WardrobeItem], dict[str, np.ndarray], dict[str, Any]]:
        items = self.manager.list_items()
        embeddings = self.manager.load_embeddings(items)
        eligible, excluded = [], []
        for it in items:
            why = exclusion_reason(it) or (None if it.id in embeddings else "missing or invalid embedding")
            if why:
                excluded.append({"id": it.id, "label": it.label, "category": it.category, "reason": why})
            else:
                eligible.append(it)
        info: dict[str, Any] = {
            "wardrobe_items": len(items),
            "eligible_items": len(eligible),
            "excluded": dict(Counter(e["reason"] for e in excluded)),
            "excluded_items": excluded,
            "eligible_by_category": dict(Counter(it.category for it in eligible)),
        }
        return eligible, embeddings, info

    # ------------------------------------------------------------- outfits
    def generate_outfits(
        self,
        k: int = 5,
        style: str | None = None,
        occasion: str | None = None,
        required_item_id: str | None = None,
        include_outerwear: bool = True,
        include_accessory: bool = True,
    ) -> OutfitResult:
        t0 = time.perf_counter()
        items, embeddings, diag = self._eligible()
        result = OutfitResult(requested=k, skipped_items=[e["id"] for e in diag["excluded_items"]], diagnostics=diag)
        diag.update(
            scoring_mode=self.scoring_mode,
            quality_threshold=self.threshold,
            pair_model_fitted=self.model.fitted,
            ranking_features=list(self.model.features),
            unused_features=dict(self.model.info.get("dropped_features", {})),
        )
        if not items:
            result.message = (
                "Your wardrobe is empty. Add some clothes on the Wardrobe page first."
                if not diag["wardrobe_items"]
                else "No wardrobe item is currently usable for outfits (see the excluded items)."
            )
            return result

        S = _Scored(items, embeddings, self.model, self.learned)
        pools = {slot: [i for i, it in enumerate(items) if it.category == slot] for slot in SLOT_DISPLAY}
        required_idx = S.index.get(required_item_id) if required_item_id else None
        if required_item_id and required_idx is None:
            ex = next((e for e in diag["excluded_items"] if e["id"] == required_item_id), None)
            result.message = f"The selected item cannot be used: {ex['reason']}." if ex else "The selected item no longer exists."
            return result
        forced_optional = None
        if required_idx is not None:
            slot = items[required_idx].category
            if slot in (OUTERWEAR, ACCESSORY):
                forced_optional = slot
            pools[slot] = [required_idx]

        ctx = np.array([np.nan if (cv := item_context(it, style, occasion)) is None else cv for it in items])
        funnel: dict[str, Any] = {
            "structural_combinations": 0,
            "rejected_by_validity": 0,
            "candidates_scored": 0,
            "pruned_for_compute": {},
        }
        templates = [
            tpl
            for tpl in CORE_TEMPLATES
            if all(pools[s] for s in tpl)
            and (required_idx is None or items[required_idx].category in tpl + (OUTERWEAR, ACCESSORY))
        ]
        diag["templates_used"] = [" + ".join(t) for t in templates]
        if not templates:
            diag.update(funnel)
            result.message = self._missing_message(pools, items[required_idx] if required_idx is not None else None)
            return result
        all_cands: list[dict[str, Any]] = []
        for tpl in templates:
            all_cands += self._template_candidates(
                tpl, pools, S, ctx, funnel, include_outerwear, include_accessory, forced_optional
            )
        diag.update(funnel)

        thr = self.threshold
        strong = [c for c in all_cands if thr is None or c["score"] >= thr]
        diag["above_threshold"] = len(strong)
        result.strong_candidates = len(strong)
        strong.sort(key=lambda c: (-c["rank"], c["key"]))
        chosen = self._diverse(strong[:MMR_POOL], k, S)
        for c in chosen:
            result.outfits.append(Outfit([items[i] for i in c["idx"]], self._explain(c, S, style, occasion)))

        avail = Counter(i for c in strong for i in c["idx"])
        sel = Counter(i for c in chosen for i in c["idx"])
        diag["exposure"] = [
            {
                "id": it.id,
                "label": it.label,
                "category": it.category,
                "in_strong_candidates": int(avail[i]),
                "selected": int(sel[i]),
            }
            for i, it in enumerate(items)
        ]
        diag["returned"] = len(chosen)
        diag["unique_items_in_results"] = len(sel)
        diag["categories_in_results"] = dict(Counter(items[i].category for i in sel))
        diag["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        if not chosen:
            if not all_cands:
                result.message = (
                    "Valid outfit structures exist, but every combination was rejected by the validity rules "
                    "(the same garment twice or conflicting items)."
                )
            else:
                result.message = (
                    f"{len(all_cands)} valid combinations were scored, but none reached the quality threshold "
                    f"({thr:.2f}). Add more items or review flagged ones."
                )
        elif len(chosen) < k:
            extra = ""
            if len(strong) > len(chosen):
                extra = f" {len(strong) - len(chosen)} other strong combination(s) were too similar to these to add variety."
            result.message = f"{len(chosen)} strong outfit(s) found from your current wardrobe (you asked for {k}).{extra}"
        return result

    def _template_candidates(
        self,
        tpl: tuple[str, ...],
        pools: dict[str, list[int]],
        S: _Scored,
        ctx: np.ndarray,
        funnel: dict[str, Any],
        include_outerwear: bool,
        include_accessory: bool,
        forced_optional: str | None,
    ) -> list[dict[str, Any]]:
        P, C = S.P, S.conflicts
        slot_pools = [np.array(pools[s]) for s in tpl]
        total = math.prod(len(p) for p in slot_pools)
        if total > self.max_candidates:
            keep = max(2, int(self.max_candidates ** (1.0 / len(tpl))))
            pruned = []
            for i, pool in enumerate(slot_pools):
                others = np.concatenate([p for j, p in enumerate(slot_pools) if j != i])
                strength = P[np.ix_(pool, others)].max(axis=1)
                pruned.append(pool[np.argsort(-strength)[:keep]])
            funnel["pruned_for_compute"][" + ".join(tpl)] = {"before": total, "kept_per_slot": keep}
            slot_pools = pruned
        cols = [g.ravel() for g in np.meshgrid(*slot_pools, indexing="ij")]
        opts = [
            s
            for s in (OUTERWEAR, ACCESSORY)
            if pools[s]
            and (s == forced_optional or (s == OUTERWEAR and include_outerwear) or (s == ACCESSORY and include_accessory))
        ]
        funnel["structural_combinations"] += len(cols[0]) * math.prod(
            len(pools[s]) + (0 if s == forced_optional else 1) for s in opts
        )

        # core pairs: validity mask + score sums (all candidates at once)
        valid = np.ones(len(cols[0]), dtype=bool)
        total_sum = np.zeros(len(cols[0]))
        for a, b in itertools.combinations(range(len(cols)), 2):
            valid &= ~C[cols[a], cols[b]]
            total_sum += P[cols[a], cols[b]]
        weight = np.full(len(cols[0]), float(len(cols) * (len(cols) - 1) // 2))
        funnel["rejected_by_validity"] += int((~valid).sum())
        cols = [c[valid] for c in cols]
        total_sum, weight = total_sum[valid], weight[valid]
        members = list(cols)  # arrays of item index per member (-1 = absent)
        chosen_layers: dict[str, np.ndarray] = {}
        for slot in opts:
            pool = np.array(pools[slot])
            n = len(total_sum)
            pw = ACCESSORY_PAIR_WEIGHT if slot == ACCESSORY else 1.0
            add = np.zeros((n, len(pool)))
            bad = np.zeros((n, len(pool)), dtype=bool)
            count = np.zeros(n)
            for col in members:
                has = col >= 0
                safe = np.where(has, col, 0)
                add += np.where(has[:, None], P[np.ix_(safe, pool)], 0.0)
                bad |= has[:, None] & C[np.ix_(safe, pool)]
                count += has
            new_mean = (total_sum[:, None] + pw * add) / (weight[:, None] + pw * count[:, None])
            new_mean[bad] = -np.inf
            best = np.argmax(new_mean, axis=1)
            best_val = new_mean[np.arange(n), best]
            current = total_sum / weight
            use = np.isfinite(best_val) & ((best_val >= current) | (slot == forced_optional))
            if slot == forced_optional:  # the required layer must fit; otherwise drop the candidate
                keep_rows = np.isfinite(best_val)
                funnel["rejected_by_validity"] += int((~keep_rows).sum())
            else:
                keep_rows = np.ones(n, dtype=bool)
            picked = np.where(use, pool[best], -1)
            total_sum = np.where(use, total_sum + pw * add[np.arange(n), best], total_sum)
            weight = np.where(use, weight + pw * count, weight)
            chosen_layers[slot] = picked
            members.append(picked)
            if not keep_rows.all():
                cols = [c[keep_rows] for c in cols]
                members = [m[keep_rows] for m in members]
                chosen_layers = {s2: v[keep_rows] for s2, v in chosen_layers.items()}
                total_sum, weight = total_sum[keep_rows], weight[keep_rows]
        funnel["candidates_scored"] += len(total_sum)
        scores = total_sum / weight if len(total_sum) else np.zeros(0)
        if len(scores):
            core_ctx = np.stack([ctx[c] for c in cols], axis=1)
            known = ~np.isnan(core_ctx)
            n_known = known.sum(axis=1)
            ctx_mean = np.where(n_known > 0, np.where(known, core_ctx, 0.0).sum(axis=1) / np.maximum(n_known, 1), np.nan)
        else:
            ctx_mean = np.zeros(0)
        rank = scores + np.where(np.isnan(ctx_mean), 0.0, self.weights.context * (ctx_mean - 0.5))
        out = []
        for r in range(len(scores)):
            core = [int(c[r]) for c in cols]
            idx = core + [int(chosen_layers[s][r]) for s in opts if chosen_layers[s][r] >= 0]
            out.append(
                {
                    "idx": idx,
                    "core": core,
                    "score": float(scores[r]),
                    "rank": float(rank[r]),
                    "ctx": None if np.isnan(ctx_mean[r]) else float(ctx_mean[r]),
                    "key": tuple(idx),
                }
            )
        return out

    def _diverse(self, cands: list[dict[str, Any]], k: int, S: _Scored) -> list[dict[str, Any]]:
        """Greedy MMR: rank minus overlap with chosen looks; hard cap on shared core garments."""
        g = S.groups

        def overlap(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, float]:
            ca, cb = {g[i] for i in a["core"]}, {g[i] for i in b["core"]}
            ia, ib = {g[i] for i in a["idx"]}, {g[i] for i in b["idx"]}
            return len(ca & cb) / max(len(ca), len(cb)), len(ia & ib) / max(len(ia), len(ib))

        chosen: list[dict[str, Any]] = []
        remaining = list(cands)
        while remaining and len(chosen) < k:
            best, best_val = None, -np.inf
            for c in remaining:
                val = c["rank"]
                if chosen:
                    ov = [overlap(c, s) for s in chosen]
                    if max(o[0] for o in ov) > self.max_shared_core + 1e-9:
                        continue
                    val -= self.diversity_lambda * max(o[1] for o in ov)
                if val > best_val:
                    best, best_val = c, val
            if best is None:
                break
            chosen.append(best)
            remaining.remove(best)
        return chosen

    # ---------------------------------------------------------- explanation
    def _explain(self, cand: dict[str, Any], S: _Scored, style: str | None, occasion: str | None) -> OutfitScore:
        idx = cand["idx"]
        items = [S.items[i] for i in idx]
        pairs = list(itertools.combinations(idx, 2))

        def mean_of(name: str) -> float | None:
            if name not in S.features:
                return None
            vals = [v for v in (S.features[name][a, b] for a, b in pairs) if not np.isnan(v)]
            return float(np.mean(vals)) if vals else None

        contrib = {name: float(np.mean([m[a, b] for a, b in pairs])) for name, m in S.contrib.items()}
        components: dict[str, float | None] = {
            "plausibility": cand["score"],
            "learned": mean_of("learned_prob"),
            "colour": mean_of("colour"),
            "style": mean_of("style_sim"),
            "visual_similarity": mean_of("clip_cos"),
            "category": mean_of("category"),
            "context": cand["ctx"],
        }
        colour = outfit_color_summary([it.metadata.get("dominant_colors") for it in items])
        tops = [str(it.metadata["style"]) for it in items if it.metadata.get("style")]
        strong_text = {
            "learned_logit": "Strong learned compatibility (model trained on real outfits)",
            "colour": f"Good colour coordination: {colour[1]}" if colour else "Good colour coordination",
            "style_sim": f"Consistent {tops[0]} style (CLIP estimate)"
            if tops and len(set(tops)) == 1
            else "Styles agree (CLIP estimate)",
            "clip_cos": "Pieces look visually coherent together",
            "category": "Classic pairing of these garment types",
        }
        weak_text = {
            "learned_logit": "learned compatibility signal is weak",
            "colour": f"colours: {colour[1]}" if colour else "colours do not coordinate well",
            "style_sim": f"mixed styles ({', '.join(sorted(set(tops)))})" if tops else "mixed styles",
            "clip_cos": "pieces look very different from each other",
            "category": "unusual combination of garment types",
        }
        reasons = [structure_text([it.category for it in items])]
        reasons += [
            strong_text[n]
            for n, v in sorted(contrib.items(), key=lambda kv: -kv[1])
            if v >= STRONG_CONTRIBUTION and n in strong_text
        ]
        caveats = []
        weak = [
            weak_text[n] for n, v in sorted(contrib.items(), key=lambda kv: kv[1]) if v <= -STRONG_CONTRIBUTION and n in weak_text
        ]
        if weak:
            caveats.append("Weaker: " + "; ".join(weak))
        if cand["ctx"] is not None:
            wanted = " / ".join(x for x in (style, occasion) if x)
            fit = "good" if cand["ctx"] >= 0.6 else "partial" if cand["ctx"] >= 0.35 else "weak"
            (reasons if fit == "good" else caveats).append(f"{fit.capitalize()} fit for '{wanted}' ({cand['ctx']:.2f})")
        for it in items:
            if (
                it.confidence is not None
                and it.confidence < MODERATE_CONFIDENCE
                and it.metadata.get("review_status") != "confirmed"
            ):
                caveats.append(f"{it.label} was detected with moderate confidence ({it.confidence:.2f})")
        explanation = {
            "structure": reasons[0],
            "colour": colour[1] if colour else "no colour data",
            "style": f"top styles: {', '.join(tops)}" if tops else "no style data",
            "scoring": self.scoring_mode,
        }
        return OutfitScore(
            score=cand["score"],
            components=components,
            reasons=reasons,
            caveats=caveats,
            explanation=explanation,
            rank_score=cand["rank"],
        )

    @staticmethod
    def _missing_message(pools: dict[str, list[int]], required: WardrobeItem | None) -> str:
        have = ", ".join(f"{len(v)} {SLOT_DISPLAY[k].lower()}" for k, v in sorted(pools.items()) if v) or "nothing usable"
        if required is not None and required.category in (TOP, BOTTOM):
            other = BOTTOM if required.category == TOP else TOP
            if not pools[other]:
                return f"To build an outfit around this {required.label} you need at least one {SLOT_DISPLAY[other].lower()}. Usable items: {have}."
        if not pools[SHOES]:
            return f"Every outfit needs shoes, and there are no usable shoes in your wardrobe. Usable items: {have}."
        return (
            "Not enough items for an outfit: add at least one top and one bottom, or a dress/one-piece, "
            f"plus shoes. Usable items: {have}."
        )

    # ---------------------------------------------------- item-to-item recs
    def recommend_for_item(
        self, item_id: str, k: int | None = None, style: str | None = None, occasion: str | None = None
    ) -> tuple[list[ItemRecommendation], str | None]:
        items, embeddings, diag = self._eligible()
        a = next((i for i, it in enumerate(items) if it.id == item_id), None)
        if a is None:
            ex = next((e for e in diag["excluded_items"] if e["id"] == item_id), None)
            return [], (f"This item cannot be used yet: {ex['reason']}." if ex else "That item no longer exists.")
        anchor = items[a]
        S = _Scored(items, embeddings, self.model, self.learned)
        labels = {
            "learned_logit": "learned compatibility",
            "colour": "colour",
            "style_sim": "style",
            "clip_cos": "visual coherence",
            "category": "garment-type pairing",
        }
        recs = []
        for j, other in enumerate(items):
            if j == a or S.conflicts[a, j] or category_pair_score(anchor.category, other.category) <= 0:
                continue

            def f(name: str, j: int = j) -> float | None:
                if name not in S.features or np.isnan(S.features[name][a, j]):
                    return None
                return float(S.features[name][a, j])

            comps: dict[str, float | None] = {
                "plausibility": float(S.P[a, j]),
                "learned": f("learned_prob"),
                "colour": f("colour"),
                "style": f("style_sim"),
                "visual_similarity": f("clip_cos"),
                "category": f("category"),
            }
            ctx = [c for c in (item_context(anchor, style, occasion), item_context(other, style, occasion)) if c is not None]
            score = float(S.P[a, j])
            if ctx:
                ctx_val = sum(ctx) / len(ctx)
                comps["context"] = ctx_val
                score += self.weights.context * (ctx_val - 0.5)
            strongest = [
                labels[n]
                for n, v in sorted(((n, float(m[a, j])) for n, m in S.contrib.items()), key=lambda kv: -kv[1])
                if v >= STRONG_CONTRIBUTION
            ]
            reasons = {"strongest_signals": ", ".join(strongest) or "no single strong signal"}
            recs.append(ItemRecommendation(other, PairScore(score=score, components=comps, reasons=reasons)))
        if not recs:
            return [], "No other usable item in your wardrobe can be worn with this one yet."
        recs.sort(key=lambda r: -r.pair.score)
        return (recs if k is None else recs[:k]), None
