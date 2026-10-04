# Fashion AI: wardrobe digitisation and outfit recommendation

A local Streamlit application. You photograph your clothes; it detects each
garment with a zero-shot detector, shows uncertain detections for your review,
stores the garments with CLIP embeddings in a local wardrobe, and builds
ranked, explained outfits from everything you own.

> **Scope, stated plainly.**
> * Garment detection is **zero-shot** (YOLO-World prompted with clothing
>   words). It makes mistakes; uncertain detections are flagged for review.
> * Outfit ranking combines a **learned compatibility signal** with CLIP
>   similarity, style agreement and garment-type pairing, using weights
>   **fitted on validation data** from a public outfit dataset. It is a
>   ranking score, not a measure of taste or a calibrated probability.
> * The learned model's test ROC-AUC of 0.725 is on a **Polyvore-derived,
>   item-disjoint evaluation**. It is **not** "72.5% accuracy", and quality on
>   your own photos was not measured.
> * Verified on Windows 11 with one NVIDIA GPU (RTX 3050 6 GB, CUDA 12.8) and
>   with CPU-only PyTorch. Nothing else was tested.

Full technical report: [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) ·
exact commands: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) ·
history: [`docs/CHANGELOG_FINAL.md`](docs/CHANGELOG_FINAL.md)

---

## 1. Features

* **Detection with uncertainty.** Several garments per photo. Each region has a
  label, score, box, crop, the **alternative labels** the detector also saw
  there, and **review flags**: ambiguous category, a dress/jumpsuit box that
  contains separates, low score, tiny crop.
* **Human review.** Flagged regions are unticked, with alternatives and a
  suggested category shown. Flagged items stored without review are marked
  *needs review* and kept out of outfits until you confirm or correct them.
* **Wardrobe** of every stored item: per-category counts, filter (incl.
  *Needs review*), search, sort, display-only pagination, edit, delete.
* **Outfits from the whole wardrobe.** Every valid combination is generated;
  impossible ones (dress + trousers, two tops, no shoes …) are rejected before
  scoring; a validation-chosen quality threshold and diversity-aware selection
  pick 1–20 looks. If fewer good looks exist you get fewer, with a message.
  The list is never padded.
* **Explanations** from computed values only (✓ reasons, ⚠ caveats, score
  details), plus a pipeline funnel and per-item exposure.
* **Item recommendations**: every usable item that can be worn with a chosen piece, ranked.
* **Diagnostics**: device and model state, learned-model and ranker metrics,
  embedding/file health, the last outfit run's funnel.
* **Training pipeline** for the learned model, on real, audited data only.

## 2. Architecture

```
photo ─► YOLO-World (zero-shot) ─► regions + alternatives + flags ─► human review / correction
                                                                            │
           SQLite wardrobe (+ images, CLIP embeddings, colours, style, review status)
                                                                            │
 eligible items ─► structural templates ─► HARD VALIDITY FILTER ─► every valid combination
        ─► pair signals (learned model + CLIP + style + garment-type pairing)
        ─► validation-fitted ranking ─► quality threshold ─► diversity-aware selection ─► k looks
```

**Why.** An earlier version turned a man in a coat and jeans (`bus.jpg`)
into "jumpsuit 0.72", accepted "jumpsuit + shoes" as an outfit and scored it
~0.80. Each stage above breaks one link of that chain: uncertainty is kept and
shown, unreviewed uncertain items are excluded, impossible structures are
rejected before any score is computed, and the learned model is one signal in
a fitted ranking. See report sections 4–6.

| Concern | Module |
|---|---|
| Configuration | `src/config.py` (env vars / `.env`) |
| Detection | `src/detection/` |
| CLIP embeddings | `src/embeddings/clip_encoder.py` (the only CLIP loader) |
| Colours, style | `src/attributes/` |
| Storage | `src/wardrobe/` |
| Pair signals + fitted ranker | `src/recommendation/pair_model.py`, `ranker_model.json` |
| Validity rules | `src/recommendation/validity.py` |
| Generation, diversity, explanations | `src/recommendation/outfit_generator.py` |
| Learned model, training, audit | `src/models/`, `src/training/` |
| UI | `app.py`, `src/ui/` |

## 3. Installation

Python ≥ 3.10 (verified 3.13), `git` on PATH (one dependency installs from
GitHub), ~1.5 GB of disk for model weights.

```bash
git clone https://github.com/Natashahahahaha/fashion-ai2.git
cd fashion-ai2
python -m venv .venv
.venv\Scripts\activate                   # macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip

# 1) PyTorch FIRST: torch AND torchvision together from the same index
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA (recommended)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU only

# 2) everything else
pip install -r requirements-dev.txt      # app + pytest/ruff/mypy (or requirements.txt for the app only)

# 3) check: True on a working CUDA setup
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**CUDA trap.** Ultralytics depends on torchvision. If pip takes torchvision
from PyPI, it can pull a CPU-only torch and silently replace your CUDA build
(this happened during verification). Installing both from the CUDA index first
avoids it. The sidebar and Diagnostics show the actual device.

**CPU fallback.** `DEVICE=auto` (default) uses CUDA when available, otherwise
CPU. CPU works but is slower (detection ~0.2 s vs ~30 ms per photo on the
test laptop). Settings are listed in `.env.example` (copy it to `.env`).

## 4. Running

```bash
streamlit run app.py                     # http://localhost:8501
```

Weights download on first use (YOLO-World ~25 MB plus a one-time ~340 MB text
encoder to build the vocabulary cache; CLIP ~600 MB). The wardrobe lives in
`DATA_DIR` (default `data/`, gitignored), so a fresh clone starts empty.

Pages: **Wardrobe** (upload → detect → review → add; gallery), **Outfit
Generator**, **Item Recommendations**, **System / Diagnostics**.
Command-line ingestion: `python scripts/ingest.py photo.jpg [--dry-run]`
(flagged detections are stored as *needs review*).

To try the 29-item real-photo test wardrobe (requires the dataset, section 5):

```bash
python scripts/demo_wardrobe.py --data-dir data/demo29
DATA_DIR=data/demo29 streamlit run app.py     # PowerShell: $env:DATA_DIR="data/demo29"; streamlit run app.py
```

## 5. Dataset, training and evaluation

The app works without any of this: a fresh clone ranks with the fitted
combiner's no-checkpoint variant.

```bash
python scripts/download_dataset.py           # Maryland Polyvore; pinned, SHA-256 verified, resumable; audits automatically
python scripts/embed_dataset.py --images-dir data/datasets/maryland-polyvore/prepared/images \
    --out data/datasets/maryland-polyvore/embeddings
python scripts/train_compatibility.py --polyvore-dir data/datasets/maryland-polyvore/prepared \
    --metadata data/datasets/maryland-polyvore/prepared/metadata.json \
    --images-dir data/datasets/maryland-polyvore/prepared/images \
    --manifest data/datasets/maryland-polyvore/manifest.json \
    --embeddings-dir data/datasets/maryland-polyvore/embeddings
python scripts/fit_ranker.py                 # ranking weights + outfit threshold, fitted on validation data
python scripts/reproduce_results.py          # re-derives the test numbers below
```

* **Dataset:** Maryland Polyvore (Han et al., ACM MM 2017): outfits from
  `xthan/polyvore-dataset`, images from the `Marqo/polyvore` mirror (both
  Apache-2.0, pinned revisions). Audit verdict: **SUFFICIENT WITH LIMITATIONS**
  (`artifacts/dataset_audit.md`).
* **Splits are item-disjoint** (overlap train/valid/test = 0 items, verified);
  model selection and ranker fitting use validation; test is used only for
  reporting.
* The checkpoint (`checkpoints/compatibility_model.pt`) is generated locally
  and **not committed**. It is used only if it beat the strongest baseline on
  its test split and matches the app's CLIP model.

## 6. Results

Item-disjoint test split, 41,745 pairs (observed Polyvore pairs vs constructed
negatives), reproduced with `scripts/reproduce_results.py` and `scripts/fit_ranker.py`:

| Scorer | ROC-AUC (95% CI) | vs hard negatives only |
|---|---|---|
| Random | 0.503 (0.498–0.509) | 0.503 |
| CLIP cosine similarity | 0.580 (0.575–0.586) | 0.584 |
| Hand-written app heuristic | 0.580 (0.575–0.586) | 0.564 |
| **Learned compatibility model** | **0.725 (0.720–0.730)** | **0.716** |
| Fitted ranker, with learned model | 0.727 | 0.716 |
| Fitted ranker, no checkpoint | 0.604 | 0.594 |

* ROC-AUC is the chance that an observed pair outranks a constructed negative.
  **0.725 is not "72.5% accuracy"**, and it measures agreement with curated
  Polyvore co-occurrence, not human preference.
* **Colour finding:** the hand-written colour rules received a **negative
  weight** when the ranker was fitted (Polyvore outfits are often tonal; the
  rules reward contrast). Colour is still computed and shown, but **not used
  in ranking**.
* **Full-wardrobe check** (29 real product photos from the test split: 8 tops,
  6 bottoms, 5 shoes, 4 outerwear, 3 dresses, 3 accessories): 255 core
  combinations (5,100 with optional layers), 148 above the threshold;
  3 / 5 / 10 / 20 looks requested → 3 / 5 / 10 / 20 returned using
  11 / 19 / 25 / 28 unique items.

## 7. Testing

```bash
pytest                                   # offline, ~30 s; fakes replace model inference only
RUN_MODEL_TESTS=1 pytest tests/test_real_models.py tests/test_real_ui_workflow.py   # real YOLO-World + CLIP
python scripts/smoke_test.py             # end-to-end with real models, temporary data dir
ruff check . && ruff format --check . && mypy src scripts app.py
```

All of the above passed in both the CPU and the CUDA environment (counts in
`docs/PROJECT_REPORT.md`, section 27).

## 8. Limitations

* Detection is zero-shot and imperfect (person regions as "jumpsuit",
  pants/jeans confusion, missed small accessories). Flags catch the patterns
  they encode, not every error.
* The ranker and threshold were fitted on clean Polyvore product images; crops
  from your photos include background and skin. Scores are not recalibrated
  for them, and performance on your photos was not measured.
* Polyvore is curated, noisy-labelled co-occurrence data; random negatives are
  easy; the learned model is not a universal fashion judge.
* Style is a coarse CLIP zero-shot estimate. Material, pattern, fit and
  silhouette are not inferred and never claimed.
* Single user, local storage, no authentication. Above 200,000 combinations
  per template, slot pools are pruned (and the UI says so).
* Only Windows 11 + RTX 3050 6 GB / CUDA 12.8 and CPU-only PyTorch were tested.

## 9. Project structure

```
app.py                     Streamlit entry point
src/
  config.py  errors.py  imaging.py  service.py
  detection/       categories.py  schemas.py  detector.py
  embeddings/      clip_encoder.py
  attributes/      colors.py  style.py
  wardrobe/        schemas.py  store.py  manager.py
  recommendation/  compatibility.py  pair_model.py  ranker_model.json  validity.py  outfit_generator.py
  models/          compatibility_net.py  learned.py
  training/        data.py  datasets.py  audit.py  metrics.py  train.py  evaluate.py
  ui/              state.py  components.py  pages.py
scripts/           download_dataset.py  audit_dataset.py  embed_dataset.py  train_compatibility.py
                   fit_ranker.py  reproduce_results.py  evaluate_compatibility.py
                   demo_wardrobe.py  ingest.py  smoke_test.py
artifacts/         dataset manifest (pinned sources, SHA-256) and dataset audit
docs/              project report, reproducibility guide, changelog
tests/             unit, integration, UI and opt-in real-model tests
experiments/       archived earlier prototypes, not used by the app
```

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Sidebar says "Running on CPU" on a GPU machine | A CPU torch was installed (often via torchvision). Reinstall `torch torchvision` together from the CUDA index. |
| `pip install` fails on `clip @ git+https://…` | Install git / check GitHub access. Needed once, to build the detector's vocabulary cache. |
| "The detector is unavailable…" | The first-run download failed or `clip` is missing; the message names the cause. You can still add whole photos as single items. |
| An item is not used in outfits | It needs review (Wardrobe → filter *Needs review* → confirm or correct), or its embedding is missing/stale. The outfit page lists excluded items with reasons. |
| Fewer looks than requested | Only that many strong, distinct looks exist; the funnel shows how many combinations were scored and passed. |
| "No clothing was detected" | Try a clearer photo, lower `DETECTION_FLOOR`, or add the whole photo as one item. |
| "The wardrobe database … is unreadable" | Move `data/wardrobe/wardrobe.db` aside to start fresh. |
| Training stops with "INSUFFICIENT" | Read `artifacts/dataset_audit.md`; nothing was trained. |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |

## 11. Licence and data provenance

* No licence file is included yet, so default copyright applies to the code.
  Add a licence before redistributing.
* Model weights are downloaded from Ultralytics and Hugging Face under their
  own licences and are not committed.
* Dataset: Maryland Polyvore (Han, Wu, Jiang, Davis, *Learning Fashion
  Compatibility with Bidirectional LSTMs*, ACM MM 2017), Apache-2.0
  repositories. The product images originate from Polyvore.com and may be
  third-party copyright: research use only, not redistributed here.
  `artifacts/dataset_manifest.json` records the exact sources and checksums.
