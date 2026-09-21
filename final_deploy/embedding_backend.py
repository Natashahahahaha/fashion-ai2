from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "openai/clip-vit-base-patch32"

_model = None
_processor = None


def load_clip():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    from transformers import CLIPProcessor, CLIPVisionModelWithProjection

    print(f"[embed] loading {MODEL_NAME} on {DEVICE}")
    _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    _model = CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME)
    _model.to(DEVICE)
    _model.eval()
    print(f"[embed] CLIP device={DEVICE}; output_dim={_model.config.projection_dim}")
    return _model, _processor


def encode_image(image_path: str | Path) -> torch.Tensor:
    model, processor = load_clip()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        features = model(**inputs).image_embeds
        features = torch.nn.functional.normalize(features, dim=-1)

    # Keep the stored representation compatible with the existing 512-d
    # OutfitCompatibilityNet interface.
    return features[0].detach().cpu()
