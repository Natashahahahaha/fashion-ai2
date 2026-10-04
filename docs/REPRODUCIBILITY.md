# Reproducibility guide

Every command below exists in this repository and was run on the verification
machine (Windows 11, Python 3.13; CPU-only PyTorch 2.14.1 and PyTorch
2.11.0+cu128 on an RTX 3050 6 GB). Commands are shown for a POSIX shell (Git
Bash works on Windows); PowerShell equivalents are given where they differ.

Use one virtual environment per runtime. The verification used `.venv`
(CPU) and `.venv-cuda` (CUDA); both names are gitignored.

## 1. CPU setup

```bash
python -m venv .venv
source .venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate ; PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt    # app + pytest, ruff, mypy
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # ... False
```

## 2. CUDA setup

```bash
python -m venv .venv-cuda
source .venv-cuda/Scripts/activate
python -m pip install --upgrade pip
# 1) torch AND torchvision together, from the CUDA index, BEFORE anything else
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 2) the rest
pip install -r requirements-dev.txt
# 3) check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# expected on the verification machine: 2.11.0+cu128 12.8 True
```

**The CUDA installation trap.** Ultralytics depends on torchvision. If
torchvision is resolved from PyPI (for example because `requirements.txt` was
installed first, or torchvision was upgraded on its own), pip can pull a newer
CPU-only torch and silently replace the CUDA build. This happened during
verification. Always install `torch` and `torchvision` together from the CUDA
index first. If it happens anyway, reinstall both with the same
`--index-url ... cu128` command. The app's sidebar and Diagnostics page show
whether it is actually running on CUDA.

Pick the CUDA wheel that matches your driver (see pytorch.org). Only CUDA 12.8
on an RTX 3050 6 GB was tested.

## 3. Dataset download

```bash
python scripts/download_dataset.py --list    # available datasets
python scripts/download_dataset.py           # Maryland Polyvore; asks before the 2.5 GB download
python scripts/download_dataset.py --yes     # no confirmation prompt
```

* Writes `data/datasets/maryland-polyvore/{prepared,manifest.json}` (gitignored).
* Both sources are pinned (GitHub `ba8aa71b6589`, Hugging Face `8c782ee447fa`);
  every file's size and SHA-256 are checked against
  `artifacts/dataset_manifest.json`.
* Interrupted downloads resume when the command is run again.
* About 6 GB on disk after preparation.
* Licence: Apache-2.0 for both repositories; the product images come from
  Polyvore.com and are for research use only. Do not redistribute them.

## 4. Dataset audit

`download_dataset.py` runs the audit automatically and writes
`artifacts/dataset_audit.{json,md}`. To re-run it on its own:

```bash
python scripts/audit_dataset.py \
    --polyvore-dir data/datasets/maryland-polyvore/prepared \
    --metadata data/datasets/maryland-polyvore/prepared/metadata.json \
    --images-dir data/datasets/maryland-polyvore/prepared/images
```

Expected verdict: **SUFFICIENT_WITH_LIMITATIONS** (exit code 0). Image checks
read all ~72k images; on a slow disk this takes several minutes.

## 5. Model training

```bash
# embed with the same CLIP model as the app (CUDA strongly recommended)
python scripts/embed_dataset.py \
    --images-dir data/datasets/maryland-polyvore/prepared/images \
    --out data/datasets/maryland-polyvore/embeddings

# train the learned compatibility model (re-runs the quality gate first)
python scripts/train_compatibility.py \
    --polyvore-dir data/datasets/maryland-polyvore/prepared \
    --metadata data/datasets/maryland-polyvore/prepared/metadata.json \
    --images-dir data/datasets/maryland-polyvore/prepared/images \
    --manifest data/datasets/maryland-polyvore/manifest.json \
    --embeddings-dir data/datasets/maryland-polyvore/embeddings
# -> checkpoints/compatibility_model.pt (gitignored); ~1 min on the RTX 3050

# fit the ranking combiner + outfit threshold on validation data
python scripts/fit_ranker.py
# -> src/recommendation/ranker_model.json (committed); first run caches colours
#    in data/datasets/maryland-polyvore/colour_cache.json

# re-derive the reported test numbers from the checkpoint and regenerated splits
python scripts/reproduce_results.py
```

Expected from `reproduce_results.py` (verified 2026-10-04):

```
dataset gate: SUFFICIENT_WITH_LIMITATIONS
item overlap between splits: {'train/valid': 0, 'train/test': 0, 'valid/test': 0}
test pairs: 41745 (positives 20868)
  random                       ROC-AUC 0.5028 ...
  clip_cosine                  ROC-AUC 0.5802 ...
  heuristic_visual_category    ROC-AUC 0.5801 ...
  learned                      ROC-AUC 0.7247 ...
```

A retrained model will not reproduce 0.7247 to four decimals unless the
run is identical (same seed, library versions and hardware); expect a value
close to it. `scripts/evaluate_compatibility.py` evaluates a checkpoint on raw
split pairs plus fill-in-the-blank. It skips the training-time audit, so its
numbers differ from the stored ones.

## 6. Testing

```bash
pytest                                   # offline; fakes replace model inference only (~30 s)
ruff check . && ruff format --check .
mypy src scripts app.py

# opt-in tests with the real YOLO-World + CLIP weights (downloads ~1 GB on first run)
RUN_MODEL_TESTS=1 pytest tests/test_real_models.py tests/test_real_ui_workflow.py
# PowerShell: $env:RUN_MODEL_TESTS="1"; pytest tests/test_real_models.py tests/test_real_ui_workflow.py
```

Force the CPU path in the CUDA environment with `DEVICE=cpu`
(PowerShell: `$env:DEVICE="cpu"`).

## 7. Smoke test

```bash
python scripts/smoke_test.py
```

Uses a temporary data directory (your wardrobe is untouched), downloads two
Ultralytics sample photos into `data/samples/`, and checks detection → crops
→ embeddings → storage → recommendations → outfits → deletion. The sample
photos contain no detectable top, so the expected outfit result is an
explanation of what is missing, not an outfit.

## 8. Launching Streamlit

```bash
streamlit run app.py                     # http://localhost:8501, wardrobe in data/
```

A fresh clone starts with an empty wardrobe. To reproduce the 29-item
real-photo wardrobe used in the report (needs the downloaded dataset):

```bash
python scripts/demo_wardrobe.py --data-dir data/demo29
DATA_DIR=data/demo29 streamlit run app.py
# PowerShell: $env:DATA_DIR="data/demo29"; streamlit run app.py
```

`demo_wardrobe.py` refuses to write into a non-empty directory, so it never
touches an existing wardrobe.
