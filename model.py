"""
Siamese compatibility MLP — matches the architecture already described for
your OutfitCompatibilityNet (concat + abs-diff + elementwise-product
features, dense + BatchNorm + ReLU + Dropout stack, sigmoid output).

The architecture was never the problem. Keep it. What has to change is
what you train it on (dataset.py, real Polyvore pairs) and how embeddings
are prepared (embed.py — L2-normalized, so the model can't win by
exploiting raw embedding scale instead of actual semantic similarity).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OutfitCompatibilityNet(nn.Module):
    def __init__(self, embed_dim: int = 512, hidden_dims=(1024, 256, 64), dropout: float = 0.3):
        super().__init__()

        # [emb1, emb2, |emb1 - emb2|, emb1 * emb2] concatenated
        input_dim = embed_dim * 4

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """emb1, emb2: (batch, embed_dim), expected to already be L2-normalized.
        Returns raw logits, shape (batch,) — apply sigmoid outside (BCEWithLogitsLoss
        expects logits, not probabilities; keeps training numerically stable)."""
        diff = torch.abs(emb1 - emb2)
        prod = emb1 * emb2
        features = torch.cat([emb1, emb2, diff, prod], dim=-1)
        logits = self.net(features).squeeze(-1)
        return logits

    def predict_proba(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(emb1, emb2))
