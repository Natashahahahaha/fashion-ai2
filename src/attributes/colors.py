"""Dominant-colour extraction and colour-harmony heuristics.

Everything here is deterministic pixel statistics (k-means in RGB plus a
fixed HSV naming table). No model is involved, and the names are coarse
on purpose: "navy" vs "blue" is about as fine as photo colours stay
reliable under arbitrary lighting.
"""

from __future__ import annotations

import colorsys
import itertools
from typing import Any

import numpy as np
from PIL import Image

NEUTRALS = frozenset({"black", "white", "gray", "beige", "brown", "navy", "olive"})

# Representative hue (degrees) of each chromatic colour name, for harmony rules.
HUE_OF = {
    "red": 0.0,
    "burgundy": 350.0,
    "pink": 330.0,
    "orange": 28.0,
    "yellow": 55.0,
    "green": 120.0,
    "teal": 180.0,
    "light blue": 205.0,
    "blue": 220.0,
    "purple": 275.0,
}


def rgb_to_name(rgb: tuple[float, float, float] | np.ndarray) -> str:
    """Map an RGB colour (0-255) to a coarse colour name."""
    r, g, b = (float(c) / 255.0 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360.0
    if v < 0.20:
        # Dark but strongly saturated pixels keep their hue (dark denim is
        # ~h235 s0.8 v0.13 in real photos, not black).
        if v >= 0.08 and s >= 0.55:
            if 195 <= h < 255:
                return "navy"
            if h < 12 or h >= 345:
                return "burgundy"
        return "black"
    if s < 0.12 or (s < 0.20 and v < 0.40):
        return "white" if v > 0.85 else "gray"
    # Light warm low-saturation: beige/cream, including its shadowed side (v ~0.5).
    if 18 <= h < 60 and s < 0.40 and v > 0.50:
        return "beige"
    if 8 <= h < 45 and v < 0.62:
        return "brown"
    if 45 <= h < 80 and v < 0.55:
        return "olive"
    if h < 12 or h >= 345:
        if v < 0.45:
            return "burgundy"
        return "pink" if (s < 0.45 and v > 0.75) else "red"
    if h < 40:
        return "orange"
    if h < 68:
        return "yellow"
    if h < 165:
        return "green"
    if h < 195:
        return "teal"
    if h < 255:
        if v < 0.45:
            return "navy"
        return "light blue" if (s < 0.45 and v > 0.70) else "blue"
    if h < 290:
        return "purple"
    return "pink"


def _kmeans(pixels: np.ndarray, k: int, iters: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Tiny deterministic k-means (k-means++ init). Returns (centers, counts)."""
    rng = np.random.default_rng(seed)
    n = len(pixels)
    k = max(1, min(k, n))
    centers = [pixels[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((pixels[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        if total <= 0:
            break
        centers.append(pixels[rng.choice(n, p=d2 / total)])
    c = np.asarray(centers, dtype=np.float32)
    assign = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        assign = ((pixels[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        new_c = np.array([pixels[assign == j].mean(0) if np.any(assign == j) else c[j] for j in range(len(c))])
        if np.allclose(new_c, c, atol=1e-4):
            c = new_c
            break
        c = new_c
    counts = np.bincount(assign, minlength=len(c))
    return c, counts


def dominant_colors(img: Image.Image, k: int = 4, max_side: int = 96, min_fraction: float = 0.06) -> list[dict[str, Any]]:
    """Dominant colours of a garment crop, largest first.

    Background handling: if the image border is near-uniform (typical of
    product shots), pixels close to the border colour are ignored. Otherwise
    (crops from photos) only the central 70% of the crop is used, which is
    where the garment is.

    Returns ``[{"name", "hex", "fraction"}]`` with fractions summing to ~1.
    """
    small = img.convert("RGB").copy()
    small.thumbnail((max_side, max_side))
    arr = np.asarray(small, dtype=np.float32)
    h, w = arr.shape[:2]
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    bg = np.median(border, axis=0)
    spread = np.abs(border - bg).mean() / 255.0

    pixels = arr.reshape(-1, 3)
    if spread < 0.08:
        fg = pixels[np.linalg.norm(pixels - bg, axis=1) / 255.0 > 0.12]
        if len(fg) >= 0.08 * len(pixels):
            pixels = fg
    else:
        y0, y1 = int(h * 0.15), max(int(h * 0.85), int(h * 0.15) + 1)
        x0, x1 = int(w * 0.15), max(int(w * 0.85), int(w * 0.15) + 1)
        pixels = arr[y0:y1, x0:x1].reshape(-1, 3)

    if len(pixels) == 0:
        return []
    centers, counts = _kmeans(pixels, k)

    merged: dict[str, dict[str, Any]] = {}
    total = float(counts.sum())
    for center, count in zip(centers, counts, strict=True):
        if count == 0:
            continue
        name = rgb_to_name(center)
        entry = merged.setdefault(name, {"name": name, "rgb_sum": np.zeros(3), "count": 0})
        entry["rgb_sum"] += center * count
        entry["count"] += int(count)

    result = []
    for entry in merged.values():
        frac = entry["count"] / total
        if frac < min_fraction:
            continue
        rgb = (entry["rgb_sum"] / entry["count"]).clip(0, 255).astype(int)
        result.append({"name": entry["name"], "hex": "#{:02x}{:02x}{:02x}".format(*rgb), "fraction": round(frac, 3)})
    result.sort(key=lambda d: -d["fraction"])
    norm = sum(d["fraction"] for d in result) or 1.0
    for d in result:
        d["fraction"] = round(d["fraction"] / norm, 3)
    return result


def primary_color(colors: list[dict[str, Any]] | None) -> str | None:
    return colors[0]["name"] if colors else None


def _hue_distance(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


# Relation -> score. Ordinal rules of thumb (not professional colour theory).
# Evaluated against real outfits by scripts/fit_ranker.py: the signal received a
# negative weight, so it is shown to the user but not used for ranking.
RELATION_SCORE = {
    "neutral_contrast": 0.80,  # e.g. white + navy, beige + black (clear lightness contrast)
    "accent_with_neutral": 0.75,  # one colour grounded by a neutral
    "monochrome": 0.70,  # same colour family with clear lightness difference
    "analogous": 0.70,  # neighbouring hues
    "complementary": 0.60,  # opposite hues: bold, high contrast
    "low_contrast_neutrals": 0.55,  # e.g. black + navy, black + dark brown: hard to tell apart
    "tone_on_tone": 0.50,  # same colour, almost the same lightness
    "conflicting": 0.35,  # unrelated saturated hues
}
RELATION_TEXT = {
    "neutral_contrast": "neutrals with clear light/dark contrast",
    "accent_with_neutral": "colour grounded by a neutral",
    "monochrome": "monochrome (same colour, different shades)",
    "analogous": "analogous colours",
    "complementary": "complementary high-contrast colours",
    "low_contrast_neutrals": "dark/low-contrast neutrals that blur together",
    "tone_on_tone": "same colour and shade (little contrast)",
    "conflicting": "competing strong colours",
}


def _lightness(colors: list[dict[str, Any]]) -> float | None:
    hexv = colors[0].get("hex") if colors else None
    if not hexv or len(str(hexv)) != 7:
        return None
    r, g, b = (int(str(hexv)[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b  # relative luminance (sRGB weights)


def colour_relation(colors_a: list[dict[str, Any]] | None, colors_b: list[dict[str, Any]] | None) -> str | None:
    """Classify how two items' primary colours relate (None if unknown)."""
    a, b = primary_color(colors_a), primary_color(colors_b)
    if a is None or b is None:
        return None
    la, lb = _lightness(colors_a or []), _lightness(colors_b or [])
    contrast = abs(la - lb) if la is not None and lb is not None else None
    na, nb = a in NEUTRALS, b in NEUTRALS
    if na and nb:
        if contrast is not None and contrast >= 0.30:
            return "neutral_contrast"
        return "tone_on_tone" if a == b else "low_contrast_neutrals"
    if na or nb:
        return "accent_with_neutral"
    if a == b:
        return "monochrome" if contrast is not None and contrast >= 0.20 else "tone_on_tone"
    d = _hue_distance(HUE_OF[a], HUE_OF[b])
    if d <= 45:
        return "analogous"
    if 150 <= d <= 210:
        return "complementary"
    return "conflicting"


def pair_color_harmony(colors_a: list[dict[str, Any]] | None, colors_b: list[dict[str, Any]] | None) -> tuple[float, str] | None:
    """(score in [0,1], reason) for two items' primary colours, or None if unknown."""
    rel = colour_relation(colors_a, colors_b)
    if rel is None:
        return None
    a, b = primary_color(colors_a), primary_color(colors_b)
    return RELATION_SCORE[rel], f"{RELATION_TEXT[rel]} ({a} + {b})"


def outfit_color_summary(color_lists: list[list[dict[str, Any]] | None]) -> tuple[float, str] | None:
    """Outfit-level colour score and a one-line reason, or None without colour data."""
    pairs = [pair_color_harmony(a, b) for a, b in itertools.combinations(color_lists, 2)]
    known = [p for p in pairs if p is not None]
    if not known:
        return None
    score = sum(s for s, _ in known) / len(known)
    primaries = [primary_color(c) for c in color_lists if c]
    accents = sorted({p for p in primaries if p and p not in NEUTRALS})
    neutrals = sorted({p for p in primaries if p and p in NEUTRALS})
    if len(accents) > 2:
        score -= 0.10 * (len(accents) - 2)
        reason = f"Busy palette: {len(accents)} accent colours ({', '.join(accents)})"
    elif not accents:
        reason = f"Neutral palette ({', '.join(neutrals)})"
        if score < 0.6:
            reason += " with little light/dark contrast"
    elif neutrals:
        reason = f"{' and '.join(accents).capitalize()} balanced by neutrals ({', '.join(neutrals)})"
    else:
        reason = f"Colour story: {', '.join(accents)}"
    worst = min(known, key=lambda p: p[0])
    if worst[0] < 0.5:
        reason += f"; note {worst[1]}"
    return max(0.0, min(1.0, score)), reason
