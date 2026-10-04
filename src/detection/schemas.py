"""Canonical detection schema shared by the detector, the wardrobe and the UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Alternative:
    """Another label the detector gave (with its own score) to the same image region."""

    label: str
    category: str
    confidence: float


@dataclass(frozen=True)
class ReviewFlag:
    """Why a detection should be reviewed before it is used for outfits."""

    code: str  # low_confidence | ambiguous_category | contains_separates | small_crop
    message: str


@dataclass(frozen=True)
class DetectionResult:
    """One detected garment region.

    Attributes:
        id: stable id (hash of image, label and box; also the crop file stem).
        label: top detector label for the region, e.g. "jeans" (from DETECTOR_PROMPTS).
        category: coarse outfit slot of ``label`` (see categories.SLOTS).
        confidence: detector score of ``label`` in [0, 1]. YOLO-World scores are
            per-prompt sigmoid scores, not a probability distribution over labels.
        bbox: ``(x1, y1, x2, y2)`` integer pixels, clipped to the image.
        crop_path: saved crop, or None if crops were not saved.
        source_image: path of the source image, if known.
        alternatives: other labels the detector assigned to (almost) the same box,
            highest score first. Empty when the region was unambiguous.
        flags: reasons this region needs review (empty = auto-accepted).
        suggested_category: a better slot suggested by the evidence (e.g. a
            "jumpsuit" region that also matched "coat"), or None.
    """

    id: str
    label: str
    category: str
    confidence: float
    bbox: BBox
    crop_path: str | None = None
    source_image: str | None = None
    alternatives: tuple[Alternative, ...] = ()
    flags: tuple[ReviewFlag, ...] = ()
    suggested_category: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        return bool(self.flags)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        d["needs_review"] = self.needs_review
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DetectionResult:
        return cls(
            id=str(d["id"]),
            label=str(d["label"]),
            category=str(d["category"]),
            confidence=float(d["confidence"]),
            bbox=tuple(int(v) for v in d["bbox"]),  # type: ignore[arg-type]
            crop_path=d.get("crop_path"),
            source_image=d.get("source_image"),
            alternatives=tuple(Alternative(**a) for a in d.get("alternatives") or ()),
            flags=tuple(ReviewFlag(**f) for f in d.get("flags") or ()),
            suggested_category=d.get("suggested_category"),
            extra=dict(d.get("extra") or {}),
        )


def clip_bbox(
    bbox: tuple[float, float, float, float] | list[float],
    width: int,
    height: int,
    min_size: int = 8,
) -> BBox | None:
    """Clip ``bbox`` to the image and round to ints.

    Returns None for boxes that are malformed (non-finite, wrong length),
    inverted, or smaller than ``min_size`` pixels on either side after
    clipping.
    """
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if any(v != v or v in (float("inf"), float("-inf")) for v in (x1, y1, x2, y2)):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    cx1 = int(round(max(0.0, min(x1, width))))
    cy1 = int(round(max(0.0, min(y1, height))))
    cx2 = int(round(max(0.0, min(x2, width))))
    cy2 = int(round(max(0.0, min(y2, height))))
    if cx2 - cx1 < min_size or cy2 - cy1 < min_size:
        return None
    return (cx1, cy1, cx2, cy2)


def box_area(b: BBox) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def box_iou(a: BBox, b: BBox) -> float:
    inter = box_area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(inner: BBox, outer: BBox) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    inter = box_area((max(inner[0], outer[0]), max(inner[1], outer[1]), min(inner[2], outer[2]), min(inner[3], outer[3])))
    a = box_area(inner)
    return inter / a if a > 0 else 0.0
