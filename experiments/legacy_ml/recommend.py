from __future__ import annotations

import argparse
import os
import torch

from embed import load_embedding
from model import OutfitCompatibilityNet


def score_pair(model, embeddings_dir, item_a_id, item_b_id, device="cpu"):
    model.eval()
    emb_a = load_embedding(os.path.join(embeddings_dir, f"{item_a_id}.pt")).unsqueeze(0).to(device)
    emb_b = load_embedding(os.path.join(embeddings_dir, f"{item_b_id}.pt")).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(emb_a, emb_b)
        prob = torch.sigmoid(logits).item()
    return prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/compatibility_net.pt")
    parser.add_argument("--embeddings-dir", default="data/Polyvore/embeddings")
    parser.add_argument("--item-a", required=True, help="Anchor item ID")
    parser.add_argument("--item-b", required=True, help="Candidate item ID to score against anchor")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = OutfitCompatibilityNet().to(args.device)
    state_dict = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model.load_state_dict(state_dict)

    score = score_path = score_pair(model, args.embeddings_dir, args.item_a, args.item_b, args.device)
    print(f"[recommend] Compatibility score between {args.item_a} and {args.item_b}: {score:.4f}")


if __name__ == "__main__":
    main()