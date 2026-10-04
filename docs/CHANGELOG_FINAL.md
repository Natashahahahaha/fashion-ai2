# Technical changelog

How the project got from a non-starting prototype to the current pipeline.
Ordered by phase; each entry says what was wrong, what changed and how it was
verified. Details and numbers are in `docs/PROJECT_REPORT.md`.

## Phase 1: Make it run

* **Broken entry point.** The Flask app (`final_deploy/app.py`) crashed at
  import unless a gitignored checkpoint, demo images and embeddings were
  present. It was replaced by a single Streamlit entry point, `app.py`, which
  starts on an empty machine and downloads weights on first use. The Flask app
  is archived in `experiments/legacy_flask_app/`.
* **Detector API mismatch.** Two detectors with incompatible APIs
  (`detect` vs `detect_and_crop`, `path` vs `box` fields) were replaced by one
  `FashionDetector.detect(image) -> list[DetectionResult]` with a canonical
  schema, validated boxes and deterministic ids.
* **Broken imports.** `src.ml.model` and similar imports that did not resolve
  were removed with the code that needed them (archived in `experiments/legacy_ml/`).
* **Legacy cleanup.** Empty stub files, backup copies of the Flask app, a
  `.bak` trainer, one-off Polyvore scripts and a debug image were removed from
  the working tree or archived under `experiments/` (still in git history).

## Phase 2: Remove fabricated results and attributes

* **Fake training removed.** `train_compatibility.py` trained on random
  Gaussian vectors, applied `BCELoss` to logits and reported the result as a
  benchmark. It was deleted, not archived.
* **Fake crops and attributes removed.** The SAM "mock" fallback (3×3 grid
  crops) and LLaVA-guessed material/pattern attributes are no longer used. The
  app never states silhouette, fit, material or pattern; a test checks this.

## Phase 3: Core application

* **CLIP consolidation.** Three CLIP loaders became one cached, lazy loader
  (`src/embeddings/clip_encoder.py`) shared by the app, dataset embedding and
  training, so wardrobe and training embeddings live in one space.
* **Wardrobe database.** A shared JSON file became SQLite + image and `.npy`
  files with atomic writes, relative paths, embedding-model tracking (stale
  detection), integrity reports, orphan cleanup and duplicate flagging.
* **Recommendation system v1.** Slot templates (top + bottom + shoes,
  one-piece + shoes), optional layers, CLIP + colour + style + category +
  occasion signals with explanations built from computed values.
* **UI.** Four Streamlit pages: Wardrobe, Outfit Generator, Item
  Recommendations, Diagnostics.

## Phase 4: CUDA

* `DEVICE=auto` with explicit placement of YOLO-World and CLIP on the GPU at
  load time (previously the detector reported CPU until its first prediction).
* Device, GPU, CUDA build, VRAM and per-model device shown in the sidebar and
  on Diagnostics, with a warning on CPU fallback.
* **Installation trap documented**: a PyPI torchvision can replace a CUDA
  torch with a CPU wheel; torch and torchvision must be installed together from
  the CUDA index first.
* Verified on an RTX 3050 6 GB (CUDA 12.8): models on `cuda:0`, 0 MB
  allocated-memory growth over 30 repeated cycles.

## Phase 5: Real data and a learned model

* **Dataset acquisition.** `scripts/download_dataset.py` fetches Maryland
  Polyvore (Han et al. 2017) outfits from GitHub and images from the
  `Marqo/polyvore` mirror, pinned to exact revisions with size and SHA-256
  verification, HTTP-range resume and safe extraction.
* **Dataset quality gate.** `src/training/audit.py` audits counts, categories,
  missing/corrupt images, exact and perceptual duplicates and cross-split
  leakage, and classifies the data. Training refuses INSUFFICIENT data. Verdict
  for Polyvore: **SUFFICIENT WITH LIMITATIONS**.
* **Learned compatibility model.** A pairwise MLP over CLIP embeddings,
  trained with `BCEWithLogitsLoss` on item-disjoint splits; the epoch and
  threshold were chosen on validation, and the test split was used once. Test
  ROC-AUC 0.725 (95% CI 0.720–0.730) vs 0.580 for CLIP similarity and the old
  heuristic. It is used automatically only if it beats the strongest baseline
  (paired bootstrap CI) and its encoder matches.

## Phase 6: Detector uncertainty (the bus.jpg fix)

* **Problem.** Agnostic NMS + top-label-only storage turned a man in a coat and
  jeans (`bus.jpg`) into "jumpsuit 0.72". The coat/jacket evidence was
  discarded, the one-piece was accepted, and "jumpsuit + shoes" scored ~0.80.
* **Change.** Class-wise NMS with a 0.10 evidence floor; overlapping boxes are
  grouped into regions with **alternatives**; review flags (`low_confidence`,
  `ambiguous_category`, `contains_separates`, `small_crop`) and a suggested
  category; flagged regions are unticked in the review form; unreviewed
  flagged items are stored as **needs_review** and excluded from outfits until
  confirmed or corrected.
* **Verified** with the real model: the jumpsuit is flagged with coat 0.20 /
  jacket 0.13 and suggested *outerwear*; no outfit is produced from it.

## Phase 7: Hard validity and full-wardrobe generation

* **Hard validity gate** before scoring: one body (top + bottom or one
  one-piece), one item per slot, shoes required, layers never replace the body,
  eligible items only, same-photo-region / near-duplicate items never combined.
  Curated adversarial tests.
* **The 5–7 item limitation.** Root causes: cross-label merging, the 0.25 score
  cutoff, the < 0.3 review untick, a per-slot candidate cap, layers tried only
  on the top 20–30 base outfits, a 12-candidate layer cap and repeat padding.
  All were removed or replaced.
* **Full generation.** Exhaustive vectorised enumeration of every valid
  combination over all eligible items; a compute guard of 200,000 combinations
  per template that is reported if it is ever hit; a user-selectable 1–20 looks;
  fewer looks plus a message instead of padding.
* **Diversity system.** MMR selection (λ = 0.6) plus a cap of half the core
  garments shared with any chosen look; near-duplicates count as one garment.
* Verified on a 29-item real-photo wardrobe: 3/5/10/20 looks requested →
  3/5/10/20 returned covering 11/19/25/28 unique items.

## Phase 8: Validation-fitted ranking and the colour finding

* **Fitted ranker.** `scripts/fit_ranker.py` fits a non-negative logistic
  combiner over learned logit, CLIP cosine, style agreement and garment-type
  pairing on validation pairs, with a variant for clones without a checkpoint.
  It also chooses the outfit quality threshold on validation outfits. Test pair
  ROC-AUC: 0.727 (with learned model), 0.604 (without).
* **Colour signal removed from ranking.** The hand-written colour-relation
  score (rewritten to use lightness contrast, so black + brown + navy is no
  longer "near-perfect") received a **negative** fitted weight in both
  variants (−0.100 / −0.219). It is still computed and displayed, labelled
  "not used in ranking".

## Phase 9: Diagnostics, UI and QA

* **Diagnostics.** A pipeline funnel computed by the real generator (items →
  eligible → combinations → valid → above threshold → returned → unique items),
  per-item exposure, the fitted-ranker summary and the unused signals.
* **UI.** The gallery shows every item, with counts, filter (incl. *Needs
  review*), search, sort and display-only pagination; outfit cards show ✓
  reasons, ⚠ caveats and score details; "Showing N of M strong outfit
  candidates".
* **Browser QA** (visible Chrome, CUDA): review flow, adding detections, 5/10
  looks, fewer-than-requested message, all pages, no console errors. It found
  one bug (duplicated alternatives in flag messages), which was fixed and tested.
* **Release tooling.** `scripts/reproduce_results.py` re-derives the reported
  model and baseline numbers; `scripts/demo_wardrobe.py` rebuilds the test
  wardrobe from the Polyvore test split.
