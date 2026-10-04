"""The one CLIP encoder used by the whole application.

Every stored embedding is produced here, so wardrobe vectors, recommendation
scoring and the learned compatibility model all share one embedding space.
``config()`` describes that space (model id, dimension, normalisation) and is
saved with every embedding and checkpoint so mismatches are detectable.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.config import Settings, get_settings, resolve_device
from src.errors import EmbeddingError
from src.imaging import ImageInput, load_image

log = logging.getLogger(__name__)


@runtime_checkable
class ImageEncoder(Protocol):
    """What the rest of the code needs from an encoder (lets tests inject fakes)."""

    model_name: str

    @property
    def dim(self) -> int: ...

    def encode_image(self, image: ImageInput) -> np.ndarray: ...

    def encode_images(self, images: Sequence[ImageInput]) -> np.ndarray: ...

    def encode_text(self, texts: Sequence[str]) -> np.ndarray: ...

    def config(self) -> dict[str, Any]: ...


def _from_pretrained(cls: Any, name: str) -> Any:
    """Load from the local Hugging Face cache first (no network round-trips);
    download only if the files are not cached yet."""
    try:
        return cls.from_pretrained(name, local_files_only=True)
    except Exception:
        return cls.from_pretrained(name)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


class ClipEncoder:
    """Lazy-loading CLIP wrapper producing L2-normalised float32 vectors.

    ``model`` / ``processor`` / ``tokenizer`` can be injected (used by tests
    with a tiny randomly initialised CLIP so no download is needed).
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        settings: Settings | None = None,
        model: Any = None,
        processor: Any = None,
        tokenizer: Any = None,
        batch_size: int = 16,
    ):
        settings = settings or get_settings()
        self.model_name = model_name or settings.clip_model
        self.device = resolve_device(device or settings.device)
        self.batch_size = batch_size
        self._model = model
        self._processor = processor
        self._tokenizer = tokenizer
        self._dim: int | None = None
        self._lock = threading.Lock()
        if model is not None:
            self._model = model.to(self.device).eval()

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        with self._lock:
            if self._model is not None and self._processor is not None:
                return
            try:
                import torch  # noqa: F401
                from transformers import CLIPImageProcessor, CLIPModel
            except ImportError as exc:
                raise EmbeddingError(
                    "CLIP needs 'torch' and 'transformers'. Install with `pip install -r requirements.txt`."
                ) from exc
            try:
                log.info("Loading CLIP %s on %s", self.model_name, self.device)
                if self._model is None:
                    model = _from_pretrained(CLIPModel, self.model_name)
                    self._model = model.to(self.device).eval()
                if self._processor is None:
                    self._processor = _from_pretrained(CLIPImageProcessor, self.model_name)
            except Exception as exc:  # OSError when offline / unknown id, etc.
                raise EmbeddingError(
                    f"Could not load CLIP model '{self.model_name}': {type(exc).__name__}: {exc}. "
                    "The first run downloads ~600 MB from Hugging Face; check your connection "
                    "or set CLIP_MODEL to a locally cached model."
                ) from exc

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    @property
    def dim(self) -> int:
        """Embedding dimension, read from the model config (verified on first encode)."""
        if self._dim is None:
            self.load()
            self._dim = int(self._model.config.projection_dim)
        return self._dim

    def config(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "dim": self.dim, "normalized": True}

    # ---------------------------------------------------------------- encode
    def encode_image(self, image: ImageInput) -> np.ndarray:
        """Encode one image -> ``(dim,)`` float32, unit L2 norm."""
        return self.encode_images([image])[0]

    def encode_images(self, images: Sequence[ImageInput]) -> np.ndarray:
        """Encode images -> ``(n, dim)`` float32, each row unit L2 norm."""
        self.load()
        import torch

        if len(images) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        pil = [load_image(im) for im in images]
        chunks: list[np.ndarray] = []
        for start in range(0, len(pil), self.batch_size):
            batch = pil[start : start + self.batch_size]
            inputs = self._processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.inference_mode():
                # Equivalent to CLIPModel.get_image_features, written out because
                # its return type changed between transformers 4.x and 5.x.
                pooled = self._model.vision_model(pixel_values=pixel_values).pooler_output
                feats = self._model.visual_projection(pooled)
            chunks.append(feats.float().cpu().numpy())
        return self._finalize(np.concatenate(chunks, axis=0))

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Encode text prompts -> ``(n, dim)`` float32, unit L2 norm."""
        self.load()
        import torch

        if self._tokenizer is None:
            try:
                from transformers import CLIPTokenizer

                self._tokenizer = _from_pretrained(CLIPTokenizer, self.model_name)
            except Exception as exc:
                raise EmbeddingError(f"Could not load the CLIP tokenizer for '{self.model_name}': {exc}") from exc
        tokens = self._tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        tokens = {k: v.to(self.device) for k, v in tokens.items() if k in ("input_ids", "attention_mask")}
        with torch.inference_mode():
            pooled = self._model.text_model(**tokens).pooler_output
            feats = self._model.text_projection(pooled)
        return self._finalize(feats.float().cpu().numpy())

    def _finalize(self, feats: np.ndarray) -> np.ndarray:
        if feats.ndim != 2 or feats.shape[1] != self.dim:
            raise EmbeddingError(f"CLIP returned shape {feats.shape}, expected (n, {self.dim}).")
        if not np.isfinite(feats).all():
            raise EmbeddingError("CLIP produced non-finite values (NaN/inf).")
        return l2_normalize(feats)


_ENCODERS: dict[tuple[str, str], ClipEncoder] = {}
_ENCODERS_LOCK = threading.Lock()


def get_encoder(settings: Settings | None = None) -> ClipEncoder:
    """Process-wide cached encoder for the configured model/device."""
    settings = settings or get_settings()
    key = (settings.clip_model, resolve_device(settings.device))
    with _ENCODERS_LOCK:
        if key not in _ENCODERS:
            _ENCODERS[key] = ClipEncoder(settings=settings)
        return _ENCODERS[key]
