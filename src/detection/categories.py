"""Clothing label vocabulary and the coarse outfit slots it maps to.

YOLO-World is an *open-vocabulary* detector: it detects whatever text prompts
it is given, zero-shot. ``DETECTOR_PROMPTS`` is therefore the exact list of
labels the detector can ever emit. Accuracy differs a lot between prompts
(e.g. "jeans" vs "pants" is often confused); the label is a best guess the
user can correct before adding an item.
"""

from __future__ import annotations

# Coarse slots used to build outfits.
TOP = "top"
BOTTOM = "bottom"
ONE_PIECE = "one_piece"
OUTERWEAR = "outerwear"
SHOES = "shoes"
ACCESSORY = "accessory"

SLOTS: tuple[str, ...] = (TOP, BOTTOM, ONE_PIECE, OUTERWEAR, SHOES, ACCESSORY)

SLOT_DISPLAY = {
    TOP: "Top",
    BOTTOM: "Bottom",
    ONE_PIECE: "Dress / one-piece",
    OUTERWEAR: "Outerwear",
    SHOES: "Shoes",
    ACCESSORY: "Accessory",
}

# fine label -> slot. Keys are the detector's text prompts.
LABEL_TO_SLOT: dict[str, str] = {
    "shirt": TOP,
    "t-shirt": TOP,
    "blouse": TOP,
    "sweater": TOP,
    "hoodie": TOP,
    "tank top": TOP,
    "jacket": OUTERWEAR,
    "coat": OUTERWEAR,
    "blazer": OUTERWEAR,
    "pants": BOTTOM,
    "jeans": BOTTOM,
    "shorts": BOTTOM,
    "skirt": BOTTOM,
    "dress": ONE_PIECE,
    "jumpsuit": ONE_PIECE,
    "shoes": SHOES,
    "sneakers": SHOES,
    "boots": SHOES,
    "sandals": SHOES,
    "heels": SHOES,
    "handbag": ACCESSORY,
    "backpack": ACCESSORY,
    "hat": ACCESSORY,
    "sunglasses": ACCESSORY,
    "scarf": ACCESSORY,
    "belt": ACCESSORY,
}

DETECTOR_PROMPTS: tuple[str, ...] = tuple(LABEL_TO_SLOT)


def slot_for_label(label: str) -> str | None:
    """Slot for a detector label, or None for labels outside the vocabulary."""
    return LABEL_TO_SLOT.get(label.strip().lower())


def labels_for_slot(slot: str) -> list[str]:
    return [label for label, s in LABEL_TO_SLOT.items() if s == slot]
