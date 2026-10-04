"""Pairwise outfit-compatibility network (optional learned component).

Architecture kept from the original project: a Siamese-style MLP over
``[a, b, |a - b|, a * b]`` of two L2-normalised CLIP embeddings. It outputs
**raw logits**; training uses ``BCEWithLogitsLoss`` and inference applies
``sigmoid``. The model only means something after training on real
compatibility data (see ``scripts/train_compatibility.py``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class OutfitCompatibilityNet(nn.Module):
    def __init__(self, embed_dim: int, hidden_dims: tuple[int, ...] = (1024, 256, 64), dropout: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout

        layers: list[nn.Module] = []
        prev = embed_dim * 4
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def config(self) -> dict[str, Any]:
        return {"embed_dim": self.embed_dim, "hidden_dims": list(self.hidden_dims), "dropout": self.dropout}

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """(batch, embed_dim) x2 -> raw logits of shape (batch,). No sigmoid here."""
        if emb_a.shape != emb_b.shape or emb_a.dim() != 2 or emb_a.shape[1] != self.embed_dim:
            raise ValueError(f"Expected two (batch, {self.embed_dim}) tensors, got {tuple(emb_a.shape)} and {tuple(emb_b.shape)}")
        features = torch.cat([emb_a, emb_b, torch.abs(emb_a - emb_b), emb_a * emb_b], dim=-1)
        return self.net(features).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """Symmetrised probability: mean of sigmoid(f(a,b)) and sigmoid(f(b,a))."""
        return 0.5 * (torch.sigmoid(self.forward(emb_a, emb_b)) + torch.sigmoid(self.forward(emb_b, emb_a)))
