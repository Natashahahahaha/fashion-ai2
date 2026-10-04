"""Shared fixtures. Heavy models are replaced by deterministic fakes here.

The fakes exercise the real pipeline code (detector post-processing, storage,
scoring, generation); only YOLO-World and CLIP inference are substituted.
Real-model coverage lives in tests/test_real_models.py (opt-in) and
scripts/smoke_test.py.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.detection.detector import FashionDetector, RawDetection  # noqa: E402
from src.embeddings.clip_encoder import l2_normalize  # noqa: E402
from src.imaging import load_image  # noqa: E402
from src.wardrobe.manager import WardrobeManager  # noqa: E402


class FakeEncoder:
    """Deterministic stand-in for CLIP: colour-layout features -> fixed projection."""

    model_name = "fake-clip"

    def __init__(self, dim: int = 32):
        self._dim = dim
        rng = np.random.default_rng(0)
        self._proj = rng.normal(size=(48, dim)).astype(np.float32)
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    def encode_image(self, image: Any) -> np.ndarray:
        return self.encode_images([image])[0]

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        self.calls += len(images)
        rows = []
        for im in images:
            small = np.asarray(load_image(im).resize((4, 4)), dtype=np.float32).reshape(-1) / 255.0
            rows.append((small - 0.5) @ self._proj)
        return l2_normalize(np.stack(rows)) if rows else np.zeros((0, self._dim), np.float32)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for t in texts:
            seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).normal(size=self._dim))
        return l2_normalize(np.stack(rows).astype(np.float32))

    def config(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "dim": self._dim, "normalized": True}


class FakeBackend:
    """Detector backend returning preset raw detections."""

    def __init__(self, raw: list[RawDetection]):
        self.raw = raw
        self.calls = 0

    def __call__(self, image: Image.Image) -> list[RawDetection]:
        self.calls += 1
        return list(self.raw)


def solid(
    color: tuple[int, int, int], size: tuple[int, int] = (120, 160), border: tuple[int, int, int] | None = (255, 255, 255)
) -> Image.Image:
    """A garment-like test image: coloured block on a white background."""
    img = Image.new("RGB", size, border or color)
    inner = Image.new("RGB", (size[0] - 30, size[1] - 30), color)
    img.paste(inner, (15, 15))
    return img


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(device="cpu", data_dir=tmp_path / "data", checkpoint_dir=tmp_path / "checkpoints")
    s.ensure_dirs()
    return s


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def manager(settings: Settings, encoder: FakeEncoder) -> WardrobeManager:
    return WardrobeManager(settings, encoder=encoder)


def make_detector(settings: Settings, raw: list[RawDetection], **kw: Any) -> FashionDetector:
    return FashionDetector(settings, backend=FakeBackend(raw), **kw)


@pytest.fixture
def scene(tmp_path: Path) -> tuple[Path, dict[str, tuple[int, int, int, int]]]:
    """A synthetic 'outfit photo': four coloured garments on white, with their boxes."""
    img = Image.new("RGB", (400, 600), (255, 255, 255))
    boxes = {
        "shirt": (100, 40, 300, 220),
        "jeans": (110, 230, 290, 470),
        "sneakers": (120, 500, 280, 580),
        "jacket": (10, 30, 90, 260),
    }
    colors = {"shirt": (245, 245, 245), "jeans": (30, 50, 110), "sneakers": (20, 20, 20), "jacket": (120, 80, 40)}
    for name, (x1, y1, x2, y2) in boxes.items():
        img.paste(Image.new("RGB", (x2 - x1, y2 - y1), colors[name]), (x1, y1))
    # Give the white shirt an outline so it is distinguishable from the background.
    img.paste(Image.new("RGB", (200, 4), (180, 180, 180)), (100, 40))
    path = tmp_path / "scene.jpg"
    img.save(path, quality=95)
    return path, boxes


def populate(manager: WardrobeManager) -> dict[str, Any]:
    """Small wardrobe: 3 tops, 2 bottoms, 2 shoes, 1 dress, 1 jacket, 1 bag."""
    spec = [
        ("white tee", "top", "t-shirt", (240, 240, 240)),
        ("black shirt", "top", "shirt", (25, 25, 25)),
        ("red top", "top", "blouse", (200, 30, 40)),
        ("blue jeans", "bottom", "jeans", (40, 60, 120)),
        ("beige pants", "bottom", "pants", (215, 195, 160)),
        ("black shoes", "shoes", "shoes", (15, 15, 15)),
        ("white sneakers", "shoes", "sneakers", (235, 235, 235)),
        ("green dress", "one_piece", "dress", (40, 140, 60)),
        ("brown jacket", "outerwear", "jacket", (110, 70, 35)),
        ("navy bag", "accessory", "handbag", (20, 30, 70)),
    ]
    out = {}
    for name, cat, label, color in spec:
        out[name] = manager.add_item(solid(color), category=cat, label=label)
    return out
