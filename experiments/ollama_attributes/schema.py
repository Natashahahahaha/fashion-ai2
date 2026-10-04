"""
Data model for a single detected wardrobe item.

Deliberately allows uncertainty (material="uncertain", confidence score,
needs_review flag) instead of forcing the model to guess — matches the
"don't hallucinate that something is cotton" requirement from the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class Formality(str, Enum):
    CASUAL = "casual"
    SMART_CASUAL = "smart_casual"
    FORMAL = "formal"
    ATHLETIC = "athletic"
    UNKNOWN = "unknown"


@dataclass
class WardrobeItem:
    item_id: str
    category: str
    subcategory: Optional[str] = None
    color: Optional[str] = None
    pattern: Optional[str] = None
    silhouette: Optional[str] = None
    sleeve: Optional[str] = None
    neckline: Optional[str] = None
    material: str = "uncertain"
    formality: Formality = Formality.UNKNOWN
    season: Optional[str] = None
    style_tags: list[str] = field(default_factory=list)

    confidence: float = 0.0
    needs_review: bool = False
    human_corrected: bool = False

    crop_path: Optional[str] = None
    source_image: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["formality"] = self.formality.value
        return d
