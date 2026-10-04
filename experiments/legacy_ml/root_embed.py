from __future__ import annotations
import os
from typing import Dict
import torch
import torch.nn.functional as F
from PIL import Image

class ClipEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def embed_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        return F.normalize(features, p=2, dim=-1).squeeze(0).cpu()

def embed_wardrobe(
    item_image_paths: Dict[str, str],
    output_dir: str,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str = None,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing CLIP embedder on: {device.upper()}")
    
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

        if (i + 1) % 500 == 0:
            print(f"[embed] {i + 1}/{len(item_image_paths)} done")

    return saved_paths

def load_embedding(path: str) -> torch.Tensor:
    return torch.load(path, weights_only=True)