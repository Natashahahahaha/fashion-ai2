"""Hard outfit validity: structural rules every candidate must pass BEFORE scoring.

A rejected combination is never scored, shown or "rescued" by a high model
score. Rules:

  * body: exactly one of {top + bottom} or {one-piece (dress/jumpsuit)}
  * a one-piece is never combined with a separate top or bottom
  * at most one item per slot (no two tops, two bottoms, two pairs of shoes ...)
  * shoes are required
  * outerwear / accessory are optional layers, never a substitute for the body
  * every item must be eligible (known category, not awaiting review)
  * two items that are the same region of the same photo (boxes overlapping with
    IoU >= 0.5: one garment detected twice under different labels) or
    near-identical images cannot be combined
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from src.detection.categories import ACCESSORY, BOTTOM, ONE_PIECE, OUTERWEAR, SHOES, SLOT_DISPLAY, SLOTS, TOP
from src.detection.schemas import box_iou
from src.wardrobe.manager import DUPLICATE_COSINE, exclusion_reason
from src.wardrobe.schemas import WardrobeItem

# Allowed outfit structures (core) and optional layers.
CORE_TEMPLATES: tuple[tuple[str, ...], ...] = ((TOP, BOTTOM, SHOES), (ONE_PIECE, SHOES))
OPTIONAL_SLOTS: tuple[str, ...] = (OUTERWEAR, ACCESSORY)
SAME_REGION_IOU = 0.5


@dataclass
class ValidityResult:
    valid: bool
    reasons: list[str] = field(default_factory=list)


def same_region(a: WardrobeItem, b: WardrobeItem) -> bool:
    """Two items cut from (nearly) the same box of the same photo = one physical garment."""
    sa, sb = a.metadata.get("source_image"), b.metadata.get("source_image")
    ba, bb = a.metadata.get("bbox"), b.metadata.get("bbox")
    if not sa or sa != sb or not ba or not bb:
        return False
    ta, tb = tuple(int(v) for v in ba), tuple(int(v) for v in bb)
    # Containment alone is NOT "same garment": a coat box legitimately encloses the shirt
    # under it. Nested one-piece/separates boxes are already rejected by the slot rules.
    return bool(box_iou(ta, tb) >= SAME_REGION_IOU)  # type: ignore[arg-type]


def conflict_matrix(items: Sequence[WardrobeItem], embeddings: np.ndarray | None) -> np.ndarray:
    """Boolean (n, n): True where two items may not appear in the same outfit."""
    n = len(items)
    conflicts = np.zeros((n, n), dtype=bool)
    if embeddings is not None and n:
        cos = embeddings @ embeddings.T
        conflicts |= cos >= DUPLICATE_COSINE
    for i in range(n):
        for j in range(i + 1, n):
            if same_region(items[i], items[j]):
                conflicts[i, j] = conflicts[j, i] = True
    np.fill_diagonal(conflicts, False)
    return conflicts


def is_valid_outfit(items: Sequence[WardrobeItem], conflicts: set[frozenset[str]] | None = None) -> ValidityResult:
    """Check one candidate against every hard rule (reasons explain each failure)."""
    reasons: list[str] = []
    slots = Counter(it.category for it in items)
    for it in items:
        why = exclusion_reason(it)
        if why:
            reasons.append(f"{it.label} ({it.category}) is not eligible: {why}")
    for slot, n in slots.items():
        if slot in SLOTS and n > 1:
            reasons.append(f"contains {n} items of the same type ({SLOT_DISPLAY[slot].lower()})")
    if slots[ONE_PIECE] and (slots[TOP] or slots[BOTTOM]):
        reasons.append("a dress/one-piece cannot be worn with a separate top or bottom")
    has_body = slots[ONE_PIECE] >= 1 or (slots[TOP] >= 1 and slots[BOTTOM] >= 1)
    if not has_body:
        if slots[TOP] and not slots[BOTTOM]:
            reasons.append("a top needs a bottom (or use a dress/one-piece)")
        elif slots[BOTTOM] and not slots[TOP]:
            reasons.append("a bottom needs a top (or use a dress/one-piece)")
        else:
            reasons.append("no main garment: needs a top + bottom or a dress/one-piece")
    if not slots[SHOES]:
        reasons.append("no shoes")
    ids = [it.id for it in items]
    if conflicts:
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if frozenset((a, b)) in conflicts:
                    reasons.append("two items are the same garment (same photo region or identical image)")
    for i, item_a in enumerate(items):
        for item_b in items[i + 1 :]:
            if same_region(item_a, item_b):
                reasons.append(f"{item_a.label} and {item_b.label} are the same region of the same photo")
    return ValidityResult(valid=not reasons, reasons=list(dict.fromkeys(reasons)))


def structure_text(slots: Sequence[str]) -> str:
    order = [TOP, BOTTOM, ONE_PIECE, SHOES, OUTERWEAR, ACCESSORY]
    parts = [SLOT_DISPLAY[s].lower() for s in sorted(slots, key=order.index)]
    return "Valid " + " + ".join(parts) + " structure"
