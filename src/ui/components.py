"""Reusable Streamlit widgets."""

from __future__ import annotations

import html
from collections.abc import Sequence
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.detection.categories import SLOT_DISPLAY
from src.detection.schemas import DetectionResult
from src.wardrobe.manager import WardrobeManager
from src.wardrobe.schemas import WardrobeItem

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46a0a0", "#f032e6", "#808000"]


def placeholder(text: str = "image missing", size: int = 256) -> Image.Image:
    img = Image.new("RGB", (size, size), (235, 235, 235))
    d = ImageDraw.Draw(img)
    d.text((12, size // 2 - 6), text, fill=(120, 120, 120))
    return img


THUMB = 320


@st.cache_data(max_entries=1000, show_spinner=False)
def _thumbnail(path: str, mtime: float) -> Image.Image:
    """Cached per (path, modification time), so reruns do not re-read every image."""
    with Image.open(path) as im:
        return ImageOps.pad(im.convert("RGB"), (THUMB, THUMB), color=(236, 236, 236))


def item_image(manager: WardrobeManager, item: WardrobeItem) -> Image.Image:
    """Square, letterboxed thumbnail so grids line up regardless of crop shape."""
    path = manager.resolve(item.image_path)
    try:
        return _thumbnail(str(path), path.stat().st_mtime)
    except (OSError, ValueError):
        return placeholder(size=THUMB)


def crop_thumbnail(path: str) -> Image.Image:
    """Square thumbnail of a detector crop (same letterboxing as the gallery)."""
    p = Path(path)
    try:
        return _thumbnail(str(p), p.stat().st_mtime)
    except (OSError, ValueError):
        return placeholder(size=THUMB)


def short_id(item: WardrobeItem) -> str:
    return f"#{item.id[:4]}"


def draw_detections(img: Image.Image, detections: Sequence[DetectionResult]) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    width = max(2, out.width // 300)
    try:
        font = ImageFont.load_default(size=max(12, out.width // 50))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for i, det in enumerate(detections):
        color = PALETTE[i % len(PALETTE)]
        d.rectangle(det.bbox, outline=color, width=width)
        tag = f"{i + 1}. {det.label} {det.confidence:.2f}"
        font_size = int(getattr(font, "size", 11))
        x, y = det.bbox[0], max(0, det.bbox[1] - font_size - 4)
        text_w = d.textbbox((0, 0), tag, font=font)[2]
        x = int(max(0, min(x, out.width - text_w - 2)))  # keep the label inside the image
        tb = d.textbbox((x, y), tag, font=font)
        d.rectangle(tb, fill=color)
        d.text((x, y), tag, fill="white", font=font)
    return out


def item_caption(item: WardrobeItem) -> str:
    """Short, fixed-shape caption so cards in a row line up."""
    md = item.metadata
    bits = [f"**{item.label}** `{short_id(item)}`", SLOT_DISPLAY.get(item.category, item.category)]
    if md.get("color"):
        bits.append(f"colour: {md['color']}")
    if md.get("style"):
        bits.append(f"style: {md['style']}" + ("" if md.get("style_source") == "user" else " (est.)"))
    if item.confidence is not None:
        bits.append(f"conf: {item.confidence:.2f}")
    return "  \n".join(bits)


def color_swatches(item: WardrobeItem) -> str:
    colors = item.metadata.get("dominant_colors") or []
    return " ".join(
        f"<span title='{html.escape(str(c.get('name', '')))} {float(c.get('fraction', 0)):.0%}' "
        "style='display:inline-block;width:16px;height:16px;border-radius:3px;border:1px solid #9995;"
        f"background:{html.escape(str(c.get('hex', '#999999')))}'></span>"
        for c in colors
    )


def score_bar(label: str, value: float | None) -> None:
    if value is None:
        st.caption(f"{label}: n/a (not available for these items)")
    else:
        st.progress(min(max(value, 0.0), 1.0), text=f"{label}: {value:.2f}")


def file_exists(path: str | Path) -> bool:
    try:
        return Path(path).is_file()
    except OSError:
        return False
