# Compatibility Model — Fix for the 1.0000-Everything Bug

## What was actually wrong

Gemini's diagnosis is right: the Siamese MLP was trained on synthetic
Gaussian noise instead of real fashion data, so it memorized meaningless
patterns in that noise and outputs a near-constant ~1.0 on anything real.
The architecture itself (concat + |diff| + product features -> dense/BN/
ReLU/Dropout -> sigmoid, BCE + AdamW) is a legitimate, standard pairwise
compatibility design — nothing there needs to change.

Two things were fixed here on top of "use real data":

1. **Embeddings are L2-normalized** (`embed.py`). Unnormalized CLIP
   embeddings let a linear layer win by exploiting vector *magnitude*
   instead of direction — a second, independent way to end up saturated
   regardless of what's actually being compared.
2. **BCEWithLogitsLoss on raw logits** instead of Sigmoid-inside-the-model
   + plain BCELoss (`model.py`). Numerically more stable, standard
   practice — the model still outputs a 0-1 probability via
   `predict_proba()`, just not through the same path used for the loss.

Nothing else about the architecture changed.

## What's here

- `dataset.py` — loads real Polyvore outfits, builds **actual** positive
  pairs (items that really appeared together) and negatives (cross-outfit
  pairs, half of them category-matched "hard" negatives so the model can't
  just learn "different category = incompatible"). Verified with a pure-
  Python smoke test (no torch needed) — run `python dataset.py`.
- `model.py` — `OutfitCompatibilityNet`, same name/architecture family as
  before, so it should slot in wherever your code already references it.
- `embed.py` — CLIP embedding extraction with L2 normalization.
- `train.py` — real training loop, reports **AUC every epoch**, not just
  loss. Warns you if val AUC lands near 0.5 (still not learning anything).
- `evaluate_fitb.py` — the standard Polyvore Fill-in-the-Blank benchmark,
  so your number is comparable to published results instead of invented.

I could not install torch in the sandbox that wrote this (disk-space
constrained), so I verified the feature-construction math (concat/abs-
diff/elementwise-product, embedding normalization, FITB broadcast shapes)
with equivalent numpy operations instead — confirmed correct, but you
should still smoke-test the actual torch model on your machine before a
long training run.

## Getting real data

Polyvore-outfits (Vasileva et al. 2018) mirrors:
- `huggingface.co/datasets/owj0421/polyvore` (refactored, HF Datasets format)
- Original: `github.com/mvasil/fashion-compatibility` / `github.com/xthan/polyvore-dataset`

Expected layout (adjust `dataset.py`'s `_ITEM_ID_KEY`/`_ITEMS_KEY` if your
mirror's field names differ slightly — they vary between releases):

```
data/Polyvore/
    images/
    nondisjoint/{train,valid,test}.json
    polyvore_item_metadata.json
    categories.csv
```

## Run it

```bash
pip install torch transformers scikit-learn pillow

# 1. Extract embeddings once (reads images/, writes one .pt per item)
python -c "
from embed import embed_wardrobe
import json, os

with open('data/Polyvore/polyvore_item_metadata.json') as f:
    meta = json.load(f)
paths = {iid: f'data/Polyvore/images/{iid}.jpg' for iid in meta}
embed_wardrobe(paths, 'data/Polyvore/embeddings')
"

# 2. Train on real pairs, watch val_auc climb above 0.5
python train.py \
    --train-json data/Polyvore/nondisjoint/train.json \
    --valid-json data/Polyvore/nondisjoint/valid.json \
    --metadata data/Polyvore/polyvore_item_metadata.json \
    --embeddings-dir data/Polyvore/embeddings \
    --out checkpoints/compatibility_net.pt

# 3. Evaluate on the standard FITB task
python evaluate_fitb.py \
    --test-json data/Polyvore/nondisjoint/test.json \
    --embeddings-dir data/Polyvore/embeddings \
    --checkpoint checkpoints/compatibility_net.pt
```

## Rule going forward

Never call a number real if it came from synthetic placeholder data. If
you need to smoke-test that code *runs* before real data is ready, that's
fine — but label it exactly that in the output ("shape/smoke test,
not a real result"), the way `dataset.py`'s `__main__` block does. That
distinction is the whole reason Day 3 exists.

## Separate issue: the detection pipeline

Not part of this fix, but worth doing in parallel — YOLOS fine-tuned on
Fashionpedia is failing on flatlay/closet-dump photos because Fashionpedia
is mostly worn-on-body imagery; it was never trained to find garments with
no human body to anchor to. That's a genuine domain mismatch, not a bug to
patch. For the closet-dump import path specifically, swap to the SAM +
prompted-VLM pipeline from Day 1 (`wardrobe_import/segment.py` +
`extract_attributes.py`) — SAM is class-agnostic and doesn't care whether
a body is present. Keep YOLOS/Fashionpedia as the detector for a later
mirror-selfie / worn-outfit import mode, where body-anchored detection is
actually the right tool.
