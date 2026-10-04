"""Canonical wardrobe item schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WardrobeItem:
    """One garment in the user's wardrobe.

    ``image_path`` / ``embedding_path`` are stored relative to DATA_DIR so the
    data directory can be moved. ``metadata`` holds only values the system
    actually computed or the user entered. Common keys:

        label            fine detector label ("jeans") or user label
        label_source     "detector" | "user"
        dominant_colors  [{"name", "hex", "fraction"}] (pixel statistics)
        color            primary colour name
        style_scores     {style: probability} (CLIP zero-shot or user one-hot)
        style            top style
        style_source     "clip_zero_shot" | "user"
        source_image     original upload path
        bbox             [x1, y1, x2, y2] in the source image
        embedding_model  CLIP model id the embedding was produced with
        embedding_dim    embedding length
    """

    id: str
    category: str
    image_path: str
    embedding_path: str
    confidence: float | None = None
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return str(self.metadata.get("label") or self.category)

    @property
    def display_name(self) -> str:
        color = self.metadata.get("color")
        return f"{color} {self.label}" if color else self.label
