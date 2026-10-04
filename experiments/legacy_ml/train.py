"""
Train OutfitCompatibilityNet on real Polyvore pairs and report a real
metric (AUC) every epoch — not a vibe check.

    python train.py \
        --train-json data/Polyvore/nondisjoint/train.json \
        --valid-json data/Polyvore/nondisjoint/valid.json \
        --metadata data/Polyvore/polyvore_item_metadata.json \
        --embeddings-dir data/Polyvore/embeddings \
        --out checkpoints/compatibility_net.pt

If AUC after training is ~0.5, the model still isn't learning anything —
that's the same failure as before, just measured honestly instead of
hidden behind confident-looking 1.0000 scores. If that happens: check
that embeddings are actually normalized (embed.py), and that hard
negatives are being generated (a model that reaches high AUC on
random-only negatives but collapses when hard negatives are added is
learning "different category" rather than "compatible" — a real, common
failure mode worth knowing about even after this fix).
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataset import load_outfits, load_item_categories, build_pairs
from embed import load_embedding
from model import OutfitCompatibilityNet

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None
    print("[train] scikit-learn not installed — pip install scikit-learn for AUC reporting.")


class PairDataset(Dataset):
    def __init__(self, pairs, embeddings_dir):
        # Preload every embedding these pairs need ONCE, into memory.
        # Reading from disk per-pair, per-epoch (the original version)
        # means ~2.7M individual small file reads for a dataset this size
        # — that's what was hanging. This does one pass over the files
        # up front, then every __getitem__ call after is an in-memory
        # dict lookup instead of a disk read.
        unique_items = set()
        for a, b, _ in pairs:
            unique_items.add(a)
            unique_items.add(b)

        print(f"[PairDataset] preloading {len(unique_items)} unique embeddings into memory...")
        self.embeddings = {}
        for i, item_id in enumerate(unique_items):
            path = os.path.join(embeddings_dir, f"{item_id}.pt")
            try:
                self.embeddings[item_id] = load_embedding(path)
            except FileNotFoundError:
                continue
            if (i + 1) % 20000 == 0:
                print(f"[PairDataset] {i + 1}/{len(unique_items)} loaded")

        self.pairs = [(a, b, y) for a, b, y in pairs if a in self.embeddings and b in self.embeddings]
        print(f"[PairDataset] {len(self.pairs)}/{len(pairs)} pairs usable "
              f"({len(pairs) - len(self.pairs)} dropped — missing embedding file)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item_a, item_b, label = self.pairs[idx]
        return self.embeddings[item_a], self.embeddings[item_b], torch.tensor(label, dtype=torch.float32)


def run_epoch(model, loader, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_labels, all_scores = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for emb_a, emb_b, labels in loader:
            emb_a, emb_b, labels = emb_a.to(device), emb_b.to(device), labels.to(device)

            logits = model(emb_a, emb_b)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_labels.extend(labels.detach().cpu().tolist())
            all_scores.extend(torch.sigmoid(logits).detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_scores) if roc_auc_score and len(set(all_labels)) > 1 else float("nan")
    return avg_loss, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--valid-json", required=True)
    parser.add_argument("--metadata", default=None, help="polyvore_item_metadata.json, for hard negatives")
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--out", default="checkpoints/compatibility_net.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    item_categories = load_item_categories(args.metadata) if args.metadata else None

    train_outfits = load_outfits(args.train_json)
    valid_outfits = load_outfits(args.valid_json)
    print(f"[train] {len(train_outfits)} train outfits, {len(valid_outfits)} valid outfits")

    train_pairs = build_pairs(train_outfits, item_categories=item_categories)
    valid_pairs = build_pairs(valid_outfits, item_categories=item_categories)
    print(f"[train] {len(train_pairs)} train pairs, {len(valid_pairs)} valid pairs")

    train_loader = DataLoader(PairDataset(train_pairs, args.embeddings_dir), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(PairDataset(valid_pairs, args.embeddings_dir), batch_size=args.batch_size)

    model = OutfitCompatibilityNet().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auc = -1.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_auc = run_epoch(model, train_loader, optimizer, args.device)
        val_loss, val_auc = run_epoch(model, valid_loader, None, args.device)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} train_auc={train_auc:.4f} "
              f"val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            # state_dict only — plain tensors, safe to load with weights_only=True later
            torch.save(model.state_dict(), args.out)
            print(f"[epoch {epoch}] new best val_auc={val_auc:.4f} -> saved {args.out}")

    print(f"[train] done. best val_auc={best_auc:.4f}")
    if best_auc < 0.6:
        print("[train] WARNING: val_auc below 0.6 — the model is close to random. "
              "Do not use this checkpoint for recommendations yet.")


if __name__ == "__main__":
    main()