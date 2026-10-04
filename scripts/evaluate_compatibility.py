"""Evaluate the saved compatibility checkpoint on a Polyvore split (pairs + fill-in-the-blank).

    python scripts/evaluate_compatibility.py \
        --polyvore-split data/datasets/maryland-polyvore/prepared/test.json \
        --embeddings-dir data/datasets/maryland-polyvore/embeddings

Pairs are rebuilt from the split's outfits WITHOUT the training-time audit and
item-disjoint filter, so the numbers differ from the checkpoint's stored test
metrics. To reproduce those exactly, use scripts/reproduce_results.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from src.config import get_settings, resolve_device
from src.errors import FashionAIError
from src.training.data import build_outfit_pairs, load_polyvore_outfits
from src.training.evaluate import evaluate_fitb, evaluate_pairs, load_for_eval


def main() -> int:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--polyvore-split", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=s.compatibility_checkpoint)
    parser.add_argument("--num-distractors", type=int, default=3)
    args = parser.parse_args()
    try:
        model, store = load_for_eval(args.checkpoint, args.embeddings_dir, resolve_device(s.device))
        outfits = load_polyvore_outfits(args.polyvore_split)
        pairs = evaluate_pairs(model, store, build_outfit_pairs(outfits, seed=123))
        fitb = evaluate_fitb(model, store, outfits, args.num_distractors)
    except FashionAIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"pair_metrics": pairs, "fitb": fitb}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
