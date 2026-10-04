# experiments/ — archived, not part of the application

Code from earlier iterations of the project, kept for reference. Nothing here
is imported by `app.py`, `src/`, `scripts/` or `tests/`, and it is excluded from
linting, type checking and tests. Much of it does **not** run as-is.

| Folder | What it was | Why it is here |
|---|---|---|
| `legacy_flask_app/` | The previous Flask web app (`app.py`), its YOLO-World + SAM + LLaVA `vision_pipeline.py`, demo-wardrobe tooling and LLaVA-extracted attributes for 30 Polyvore items. | It crashed at import without a gitignored checkpoint, demo images and embeddings; it loaded CLIP separately from training; its upload path wrote to a shared JSON file. Replaced by the Streamlit app in `src/ui/`. |
| `ollama_attributes/` | Prompted attribute extraction with a local Ollama vision model (LLaVA), a SAM segmenter with a 3x3 grid "mock" fallback, and a terminal review loop. | Needs a running Ollama server. The grid fallback produced fake crops. Possible future add-on for richer attributes (material, pattern), which the main app deliberately does not guess. |
| `legacy_ml/` | The original detector (`ModernFashionDetector.detect_and_crop`), wardrobe manager, Polyvore pair builder, training loop, FITB evaluation and recommender scripts. | Mutually incompatible APIs (`detect` vs `detect_and_crop`, `path` vs `box`), broken `src.ml.model` imports, a third CLIP loader, and checkpoint paths that disagreed. The useful parts (YOLO-World prompting, hard-negative pair construction, BCEWithLogits training, FITB) were rewritten into `src/detection`, `src/training`. |
| `polyvore_scripts/` | One-off Polyvore inspection/visualisation scripts and the `unpack_data.py` downloader. | Hard-coded paths. `unpack_data.py` may help when fetching Polyvore but has not been re-verified. |

The synthetic-vector trainer (`train_compatibility.py`, which reported
"benchmark" numbers from random Gaussian vectors using `BCELoss` on logits) was
deleted rather than archived: its results were meaningless by construction.
It remains in git history.

Also removed from the working tree rather than archived (all still in git
history): empty stub modules from the old `final_deploy/backend/` and
`final_deploy/tests/` folders, two backup copies of the Flask app
(`app_v4_selfcontained.py`, `app_working_backup.py`), `train_OLD_broken.py.bak`,
a root-level `ingest.py` superseded by `scripts/ingest.py`, an empty `saved`
file and a debug image (`outfit_check.png`).
