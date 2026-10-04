"""
Extract CLIP embeddings for Polyvore items, L2-normalized.

Normalization matters: raw CLIP embeddings aren't unit-length, and feeding
unnormalized vectors into a linear layer lets the model latch onto
embedding *magnitude* rather than the direction that actually encodes
visual/semantic similarity. Combined with training on real pairs (see
dataset.py), this closes off the easiest way for the net to cheat its way
to a degenerate "always compatible" output.

Requires (on your machine — this needs a CLIP model download, which the
sandbox that wrote this file doesn't have network access for):
    pip install transformers torch pillow
"""

from __future__ import annotations

import os
from typing import Dict, Iterable

import torch
import torch.nn.functional as F
from PIL import Image


class ClipEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def embed_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)  # (1, 512)
        return F.normalize(features, p=2, dim=-1).squeeze(0).cpu()  # unit-norm, (512,)


def embed_wardrobe(
    item_image_paths: Dict[str, str],
    output_dir: str,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
) -> Dict[str, str]:
    """item_image_paths: {item_id: path_to_image}. Saves one .pt file per
    item to output_dir, returns {item_id: saved_path}."""
    os.makedirs(output_dir, exist_ok=True)
    embedder = ClipEmbedder(model_name=model_name, device=device)

    saved_paths = {}
    for i, (item_id, image_path) in enumerate(item_image_paths.items()):
        try:
            emb = embedder.embed_image(image_path)
        except Exception as e:
            print(f"[embed] skipping {item_id} ({image_path}): {e}")
            continue

        out_path = os.path.join(output_dir, f"{item_id}.pt")
        torch.save(emb, out_path)
        saved_paths[item_id] = out_path

        if (i + 1) % 200 == 0:
            print(f"[embed] {i + 1}/{len(item_image_paths)} done")

    return saved_paths


def load_embedding(path: str) -> torch.Tensor:
    """Safe load — plain tensors only, no arbitrary pickle execution."""
    return torch.load(path, weights_only=True)
