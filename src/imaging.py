"""Image loading shared by every pipeline stage (always returns RGB PIL images)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.errors import InvalidImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ImageInput = str | Path | bytes | Image.Image | np.ndarray

NOT_AN_IMAGE = (
    "This file is not a recognised image. It may be corrupt, or another file type renamed to an image "
    "extension. Please upload a JPG, PNG, WEBP or BMP photo."
)

# Guard against decompression bombs while still allowing large phone photos.
Image.MAX_IMAGE_PIXELS = 80_000_000


def load_image(source: ImageInput) -> Image.Image:
    """Load ``source`` as an RGB ``PIL.Image``.

    Accepts a file path, raw bytes, a PIL image or an ``HxWx3`` uint8 numpy
    array **in RGB order** (not OpenCV's BGR). EXIF orientation is applied so
    phone photos are upright before detection and cropping.
    """
    try:
        if isinstance(source, Image.Image):
            img = source
        elif isinstance(source, np.ndarray):
            if source.ndim == 2:
                source = np.stack([source] * 3, axis=-1)
            if source.ndim != 3 or source.shape[2] not in (3, 4):
                raise InvalidImageError(f"Expected an HxWx3 RGB array, got shape {source.shape}")
            img = Image.fromarray(source.astype(np.uint8))
        elif isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
            img.load()
        else:
            path = Path(source)
            if not path.is_file():
                raise InvalidImageError(f"Image file not found: {path}")
            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                raise InvalidImageError(
                    f"Unsupported file type '{path.suffix}'. Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                )
            with Image.open(path) as opened:
                opened.load()
                img = opened.copy()
                img.info = dict(opened.info)
        # Apply EXIF orientation so phone photos are upright (returns a copy).
        img = ImageOps.exif_transpose(img) or img
    except InvalidImageError:
        raise
    except UnidentifiedImageError as exc:
        raise InvalidImageError(NOT_AN_IMAGE) from exc
    except Image.DecompressionBombError as exc:
        raise InvalidImageError(
            "This image is too large to process safely (over 160 million pixels). Please resize it and try again."
        ) from exc
    except (OSError, ValueError) as exc:
        raise InvalidImageError(
            f"The image could not be decoded ({exc}). It may be corrupt or only partially downloaded; "
            "try re-saving or re-exporting it."
        ) from exc
    except Exception as exc:
        # Ultralytics monkeypatches Image.open and, for undecodable files, tries an
        # optional HEIF plugin, which can raise ImportError/other errors instead of
        # a PIL error. Any decoder failure means "not a usable image" to the user.
        raise InvalidImageError(NOT_AN_IMAGE) from exc

    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA", "P"):
            # Composite transparency onto white so cut-outs do not turn black.
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            img = background
        else:
            img = img.convert("RGB")

    if img.width < 8 or img.height < 8:
        raise InvalidImageError(f"Image is too small ({img.width}x{img.height}).")
    return img
