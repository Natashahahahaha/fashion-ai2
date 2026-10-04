"""Dataset-specific helpers (category mapping for the Polyvore family of datasets).

Polyvore sets mix garments with home decor, beauty products, furniture etc.
Only items whose fine category maps unambiguously to an outfit slot are kept.
Generic labels ("Clothing", "Accessories", "Men's Fashion") are dropped because
their slot cannot be determined, and so are swimwear/underwear/costumes,
which are not part of the outfits this app builds.
"""

from __future__ import annotations

_SLOTS: dict[str, tuple[str, ...]] = {
    "one_piece": (
        "dresses", "day dresses", "cocktail dresses", "gowns", "rompers", "jumpsuits", "wedding dresses",
    ),
    "outerwear": (
        "jackets", "coats", "blazers", "outerwear", "vests", "men's jackets", "men's coats",
        "men's sportcoats & blazers", "men's vests", "activewear jackets",
    ),
    "top": (
        "tops", "sweaters", "blouses", "t-shirts", "tank tops", "cardigans", "sweatshirts", "hoodies", "tunics",
        "camisoles", "men's t-shirts", "men's casual shirts", "men's dress shirts", "men's shirts", "men's polos",
        "men's sweaters", "men's tank tops", "men's hoodies", "activewear tops", "activewear tank tops",
    ),
    "bottom": (
        "knee length skirts", "mini skirts", "long skirts", "skirts", "shorts", "pants", "skinny jeans", "jeans",
        "boyfriend jeans", "straight leg jeans", "flared jeans", "bootcut jeans", "wide leg jeans",
        "capri & cropped pants", "leggings", "men's jeans", "men's casual pants", "men's dress pants", "men's shorts",
        "activewear pants", "activewear shorts", "men's activewear pants",
    ),
    "shoes": (
        "sandals", "pumps", "ankle booties", "sneakers", "shoes", "flats", "boots", "knee high boots",
        "over-the-knee boots", "mid calf boots", "loafers & moccasins", "oxfords", "flip flops", "clogs",
        "athletic shoes", "men's sneakers", "men's shoes", "men's boots", "men's dress shoes",
        "men's loafers & moccasins", "men's oxfords", "men's work boots", "men's athletic shoes",
    ),
    "accessory": (
        "earrings", "shoulder bags", "necklaces", "bracelets & bangles", "clutches", "rings", "sunglasses",
        "handbags", "tote bags", "hats", "backpacks", "watches", "scarves", "hair accessories", "wallets", "belts",
        "jewelry", "brooches", "eyeglasses", "eyewear", "gloves", "messenger bags", "charms & pendants", "bags",
        "men's watches", "men's hats", "men's sunglasses", "men's wallets", "men's backpacks", "men's belts",
        "men's bracelets", "ties", "bow ties", "men's scarves", "men's bags", "men's messenger bags", "briefcases",
        "men's briefcases", "men's gloves", "men's rings", "men's necklaces", "men's eyeglasses", "cuff links",
        "suspenders",
    ),
}  # fmt: skip

CATEGORY_SLOT: dict[str, str] = {name: slot for slot, names in _SLOTS.items() for name in names}


def fashion_slot(fine_category: str) -> str | None:
    """Outfit slot for a Polyvore fine category, or None if not an unambiguous outfit item."""
    return CATEGORY_SLOT.get(fine_category.strip().lower())
