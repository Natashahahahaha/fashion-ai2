"""
Fill-In-The-Blank (FITB) evaluation — the standard Polyvore compatibility
benchmark task, so your numbers are comparable to published results
instead of a self-invented "accuracy."

For each test outfit: remove one item, present the model with the correct
item plus a few random distractors, and check whether it correctly scores
the true item highest against the rest of the outfit.

    python evaluate_fitb.py \
        --test-json data/Polyvore/nondisjoint/test.json \
        --embeddings-dir data/Polyvore/embeddings \
        --checkpoint checkpoints/compatibility_net.pt \
        --num-distractors 3
"""

from __future__ import annotations

import argparse
import random

import torch

from dataset import load_outfits
from embed import load_embedding
from model import OutfitCompatibilityNet


def score_candidate(model, candidate_emb, remaining_embs, device="cpu"):
    """Average predicted compatibility between the candidate and every
    other item already in the outfit."""
    candidate_batch = candidate_emb.unsqueeze(0).repeat(len(remaining_embs), 1).to(device)
    remaining_batch = torch.stack(remaining_embs).to(device)
    with torch.no_grad():
        probs = model.predict_proba(candidate_batch, remaining_batch)
    return probs.mean().item()


def evaluate_fitb(model, outfits, embeddings_dir, num_distractors=3, seed=42, device="cpu"):
    rng = random.Random(seed)
    all_items = [item for outfit in outfits for item in outfit]

    correct = 0
    total = 0

    for outfit in outfits:
        if len(outfit) < 2:
            continue

        blank_idx = rng.randrange(len(outfit))
        true_item = outfit[blank_idx]
        remaining_items = outfit[:blank_idx] + outfit[blank_idx + 1:]

        try:
            remaining_embs = [load_embedding(f"{embeddings_dir}/{i}.pt") for i in remaining_items]
            true_emb = load_embedding(f"{embeddings_dir}/{true_item}.pt")
        except FileNotFoundError:
            continue

        distractors = []
        while len(distractors) < num_distractors:
            candidate = rng.choice(all_items)
            if candidate not in outfit and candidate not in distractors:
                distractors.append(candidate)

        try:
            distractor_embs = [load_embedding(f"{embeddings_dir}/{d}.pt") for d in distractors]
        except FileNotFoundError:
            continue

        candidates = [(true_item, true_emb)] + list(zip(distractors, distractor_embs))
        scores = [score_candidate(model, emb, remaining_embs, device) for _, emb in candidates]

        predicted_idx = scores.index(max(scores))
        if candidates[predicted_idx][0] == true_item:
            correct += 1
        total += 1

    accuracy = correct / total if total else float("nan")
    return accuracy, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-distractors", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = OutfitCompatibilityNet().to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True))
    model.eval()

    test_outfits = load_outfits(args.test_json)
    accuracy, n = evaluate_fitb(model, test_outfits, args.embeddings_dir, args.num_distractors, device=args.device)

    # Random-chance baseline for a 1-correct-of-(1+num_distractors) task
    chance = 1.0 / (1 + args.num_distractors)
    print(f"FITB accuracy: {accuracy:.4f} on {n} outfits (random chance: {chance:.4f})")
    if accuracy <= chance + 0.05:
        print("WARNING: barely above random chance — treat this checkpoint as not working yet.")


if __name__ == "__main__":
    main()
