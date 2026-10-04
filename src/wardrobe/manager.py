"""High-level wardrobe operations: image -> stored item with embedding + attributes."""

from __future__ import annotations

import logging
import os
import shutil
import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.attributes.colors import dominant_colors, primary_color
from src.attributes.style import STYLES, StyleClassifier, top_style, user_style_scores
from src.config import Settings, get_settings
from src.detection.categories import SLOTS
from src.detection.schemas import DetectionResult
from src.embeddings.clip_encoder import ImageEncoder
from src.errors import EmbeddingError, FashionAIError, WardrobeStoreError
from src.imaging import ImageInput, load_image
from src.wardrobe.schemas import WardrobeItem
from src.wardrobe.store import WardrobeStore

log = logging.getLogger(__name__)

MAX_STORED_SIDE = 768

# Review status of an item's category (stored in metadata["review_status"]).
REVIEW_AUTO = "auto"  # detector result with no review flags
REVIEW_NEEDED = "needs_review"  # detector flagged it; excluded from outfits until reviewed
REVIEW_CONFIRMED = "confirmed"  # a person accepted or corrected the category
REVIEW_USER = "user"  # category chosen by the user when adding the item


def needs_review(item: WardrobeItem) -> bool:
    return item.metadata.get("review_status") == REVIEW_NEEDED


def exclusion_reason(item: WardrobeItem) -> str | None:
    """Why an item cannot be used for outfit generation (None = eligible)."""
    if item.category not in SLOTS:
        return "unknown category"
    if needs_review(item):
        return "needs review (uncertain detection)"
    return None


# CLIP cosine above which a new item is flagged as a likely duplicate of an
# existing item of the same category. Measured with ViT-B/32 on real crops:
# the same crop re-encoded >= 0.95-0.99, mirrored >= 0.98, while different
# garments from one photo were <= 0.885. 0.97 therefore catches re-adding the
# same photo/crop; it does NOT reliably catch the same garment photographed
# again (that gap is too narrow to threshold safely).
DUPLICATE_COSINE = 0.97


def _atomic_write_image(img: Any, path: Path) -> None:
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    img.save(tmp, format="JPEG", quality=92)
    os.replace(tmp, path)


def _atomic_write_npy(arr: np.ndarray, path: Path) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, arr.astype(np.float32), allow_pickle=False)
    os.replace(tmp, path)


@dataclass
class IntegrityReport:
    missing_images: list[str] = field(default_factory=list)
    invalid_category: list[str] = field(default_factory=list)
    orphan_files: list[Path] = field(default_factory=list)


@dataclass
class EmbeddingStatus:
    ok: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    corrupt: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)  # produced by a different CLIP model/dim

    @property
    def unusable(self) -> list[str]:
        return self.missing + self.corrupt + self.stale


class WardrobeManager:
    """Owns the wardrobe store, the files on disk and the embedding cache.

    ``encoder`` may be passed directly or as a zero-arg factory, so the UI
    can list and delete items without loading CLIP at all.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        encoder: ImageEncoder | Callable[[], ImageEncoder] | None = None,
        store: WardrobeStore | None = None,
        compute_style: bool = True,
    ):
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.store = store or WardrobeStore(self.settings.db_path)
        self._encoder_or_factory = encoder
        self._encoder: ImageEncoder | None = None
        self._style: StyleClassifier | None = None
        self.compute_style = compute_style
        self._cache: dict[str, tuple[float, np.ndarray]] = {}
        self._cache_lock = threading.Lock()

    # --------------------------------------------------------------- helpers
    @property
    def encoder(self) -> ImageEncoder:
        if self._encoder is None:
            src = self._encoder_or_factory
            if src is None:
                from src.embeddings.clip_encoder import get_encoder

                self._encoder = get_encoder(self.settings)
            elif callable(src) and not hasattr(src, "encode_image"):
                self._encoder = src()
            else:
                self._encoder = src  # type: ignore[assignment]
        assert self._encoder is not None
        return self._encoder

    def _style_classifier(self) -> StyleClassifier:
        if self._style is None:
            self._style = StyleClassifier(self.encoder)
        return self._style

    def resolve(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else self.settings.data_dir / p

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.settings.data_dir).as_posix()
        except ValueError:
            return str(path)

    # ------------------------------------------------------------------- add
    def add_item(
        self,
        image: ImageInput,
        category: str,
        label: str | None = None,
        confidence: float | None = None,
        label_source: str = "user",
        source_image: str | None = None,
        bbox: Iterable[int] | None = None,
        style: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> WardrobeItem:
        """Store one garment image: save image, embed with CLIP, compute attributes, insert row."""
        if category not in SLOTS:
            raise FashionAIError(f"Unknown category {category!r}. Choose one of: {', '.join(SLOTS)}")
        if style is not None and style not in STYLES:
            raise FashionAIError(f"Unknown style {style!r}. Choose one of: {', '.join(STYLES)}")

        img = load_image(image)
        img.thumbnail((MAX_STORED_SIDE, MAX_STORED_SIDE))

        # Embed first: if CLIP is unavailable nothing is written to disk.
        embedding = self.encoder.encode_image(img)
        self._validate_embedding(embedding)

        item_id = uuid.uuid4().hex[:12]
        image_path = self.settings.images_dir / f"{item_id}.jpg"
        emb_path = self.settings.embeddings_dir / f"{item_id}.npy"

        metadata: dict[str, Any] = {
            "label": (label or category).strip().lower(),
            "label_source": label_source,
            "review_status": REVIEW_USER if label_source == "user" else REVIEW_AUTO,
            "embedding_model": self.encoder.model_name,
            "embedding_dim": int(embedding.shape[0]),
        }
        colors = dominant_colors(img)
        if colors:
            metadata["dominant_colors"] = colors
            metadata["color"] = primary_color(colors)
        if style is not None:
            metadata.update(style_scores=user_style_scores(style), style=style, style_source="user")
        elif self.compute_style:
            try:
                scores = self._style_classifier().scores(embedding)
                metadata.update(style_scores=scores, style=top_style(scores), style_source="clip_zero_shot")
            except Exception as exc:  # text tower unavailable: degrade, don't fail the upload
                log.warning("style estimate skipped: %s", exc)
        if source_image:
            metadata["source_image"] = str(source_image)
        if bbox is not None:
            metadata["bbox"] = [int(v) for v in bbox]
        if extra_metadata:
            metadata.update(extra_metadata)
        duplicates = self.find_duplicates(embedding, category)
        if duplicates:
            metadata["possible_duplicate_of"] = duplicates

        try:
            _atomic_write_image(img, image_path)
            _atomic_write_npy(embedding, emb_path)
            item = WardrobeItem(
                id=item_id,
                category=category,
                image_path=self._relative(image_path),
                embedding_path=self._relative(emb_path),
                confidence=None if confidence is None else float(confidence),
                metadata=metadata,
            )
            self.store.add_item(item)
        except Exception:
            image_path.unlink(missing_ok=True)
            emb_path.unlink(missing_ok=True)
            raise
        return item

    def add_detections(
        self, detections: Iterable[DetectionResult], overrides: dict[str, str] | None = None, reviewed: bool = False
    ) -> list[WardrobeItem]:
        """Add detector crops. ``overrides`` maps detection id -> user-corrected category."""
        overrides = overrides or {}
        added: list[WardrobeItem] = []
        for det in detections:
            try:
                added.append(self._add_detection(det, overrides, reviewed))
            except FashionAIError as exc:
                if not added:
                    raise
                raise FashionAIError(f"Added {len(added)} item(s), then failed on '{det.label}': {exc}") from exc
        return added

    def _add_detection(self, det: DetectionResult, overrides: dict[str, str], reviewed: bool = False) -> WardrobeItem:
        """Store one detection. ``reviewed`` = the user saw it in the review step and chose to add it."""
        if not det.crop_path:
            raise FashionAIError(f"Detection {det.id} has no saved crop; run detect(save_crops=True).")
        category = overrides.get(det.id, det.category)
        corrected = category != det.category
        if corrected or (reviewed and det.needs_review):
            status = REVIEW_CONFIRMED  # a person looked at the flags and decided
        elif det.needs_review:
            status = REVIEW_NEEDED
        else:
            status = REVIEW_AUTO
        return self.add_item(
            det.crop_path,
            category=category,
            label=category if corrected else det.label,
            confidence=det.confidence,
            label_source="user" if corrected else "detector",
            source_image=det.source_image,
            bbox=det.bbox,
            extra_metadata={
                "detector_label": det.label,
                "detection_id": det.id,
                "alternatives": [
                    {"label": a.label, "category": a.category, "confidence": a.confidence} for a in det.alternatives
                ],
                "review_flags": [{"code": f.code, "message": f.message} for f in det.flags],
                "suggested_category": det.suggested_category,
                "review_status": status,
            },
        )

    def ingest_image(self, image_path: str | Path, detector: Any) -> tuple[list[DetectionResult], list[WardrobeItem]]:
        """Full pipeline for one photo: detect -> crop -> embed -> store every detection."""
        stored = self._store_upload(Path(image_path))
        detections = detector.detect(stored)
        return detections, self.add_detections(detections)

    def _store_upload(self, path: Path) -> Path:
        load_image(path)  # validate before copying
        dest = self.settings.uploads_dir / f"{uuid.uuid4().hex[:12]}{path.suffix.lower()}"
        shutil.copyfile(path, dest)
        return dest

    def find_duplicates(
        self, embedding: np.ndarray, category: str | None = None, threshold: float = DUPLICATE_COSINE
    ) -> list[str]:
        """Ids of stored items (of ``category``, if given) whose embedding is nearly identical."""
        items = self.list_items(category)
        embs = self.load_embeddings(items)
        out = []
        for item_id, emb in embs.items():
            if emb.shape == embedding.shape and float(emb @ embedding) >= threshold:
                out.append(item_id)
        return out

    # ---------------------------------------------------------------- update
    def update_item(
        self, item_id: str, category: str | None = None, style: str | None = None, label: str | None = None
    ) -> WardrobeItem:
        item = self.get_item(item_id)
        if item is None:
            raise FashionAIError(f"Item {item_id} not found")
        if category is not None and category not in SLOTS:
            raise FashionAIError(f"Unknown category {category!r}")
        md = dict(item.metadata)
        if category is not None:
            md["review_status"] = REVIEW_CONFIRMED  # choosing a category is a review decision
        if style is not None:
            md.update(style_scores=user_style_scores(style), style=style, style_source="user")
        if label is not None:
            md.update(label=label.strip().lower(), label_source="user")
        if category is not None and category != item.category and label is None:
            md.update(label=category, label_source="user")
        self.store.update_item(item_id, category=category, metadata=md)
        updated = self.get_item(item_id)
        assert updated is not None
        return updated

    def confirm_item(self, item_id: str) -> WardrobeItem:
        """Accept a flagged item's current category after review."""
        item = self.get_item(item_id)
        if item is None:
            raise FashionAIError(f"Item {item_id} not found")
        md = dict(item.metadata)
        md["review_status"] = REVIEW_CONFIRMED
        self.store.update_item(item_id, metadata=md)
        updated = self.get_item(item_id)
        assert updated is not None
        return updated

    # ------------------------------------------------------------------ read
    def get_item(self, item_id: str) -> WardrobeItem | None:
        return self.store.get_item(item_id)

    def list_items(self, category: str | None = None) -> list[WardrobeItem]:
        return self.store.list_items(category)

    def search_items(self, **filters: Any) -> list[WardrobeItem]:
        return self.store.search_items(**filters)

    def count(self) -> int:
        return self.store.count()

    # ---------------------------------------------------------------- delete
    def delete_item(self, item_id: str) -> bool:
        item = self.store.get_item(item_id)
        if item is None:
            return False
        self.store.delete_item(item_id)
        for rel in (item.image_path, item.embedding_path):
            try:
                self.resolve(rel).unlink(missing_ok=True)
            except OSError as exc:
                log.warning("could not delete %s: %s", rel, exc)
        with self._cache_lock:
            self._cache.pop(item_id, None)
        return True

    def clear_wardrobe(self) -> int:
        n = self.store.clear()
        for d in (self.settings.images_dir, self.settings.embeddings_dir):
            for f in d.glob("*"):
                if f.is_file():
                    f.unlink(missing_ok=True)
        with self._cache_lock:
            self._cache.clear()
        return n

    # ------------------------------------------------------------ embeddings
    def _validate_embedding(self, emb: np.ndarray, expected_dim: int | None = None) -> None:
        if emb.ndim != 1 or emb.size == 0:
            raise EmbeddingError(f"Embedding has shape {emb.shape}, expected a 1-D vector")
        if expected_dim is not None and emb.shape[0] != expected_dim:
            raise EmbeddingError(f"Embedding has dimension {emb.shape[0]}, expected {expected_dim}")
        if not np.isfinite(emb).all():
            raise EmbeddingError("Embedding contains NaN/inf values")
        norm = float(np.linalg.norm(emb))
        if not 0.98 < norm < 1.02:
            raise EmbeddingError(f"Embedding is not unit-normalised (norm={norm:.3f})")

    def load_embedding(self, item: WardrobeItem) -> np.ndarray | None:
        """Load and validate an item's embedding (cached). None if missing/corrupt."""
        path = self.resolve(item.embedding_path)
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        with self._cache_lock:
            cached = self._cache.get(item.id)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        try:
            emb = np.load(path, allow_pickle=False).astype(np.float32).reshape(-1)
            self._validate_embedding(emb, item.metadata.get("embedding_dim"))
        except (OSError, ValueError, EmbeddingError) as exc:
            log.warning("corrupt embedding for %s (%s): %s", item.id, path, exc)
            return None
        with self._cache_lock:
            self._cache[item.id] = (mtime, emb)
        return emb

    def load_embeddings(self, items: Iterable[WardrobeItem]) -> dict[str, np.ndarray]:
        """Embeddings for every usable item (missing/corrupt/stale ones are skipped)."""
        model, dim = self._configured_encoder()
        out = {}
        for item in items:
            if self._is_stale(item, model, dim):
                continue
            emb = self.load_embedding(item)
            if emb is not None:
                out[item.id] = emb
        return out

    def _configured_encoder(self) -> tuple[str | None, int | None]:
        """(model name, dim) of the current encoder, without loading CLIP if avoidable."""
        if self._encoder is not None:
            dim = None
            if getattr(self._encoder, "is_loaded", True):
                dim = int(self._encoder.dim)
            return self._encoder.model_name, dim
        src = self._encoder_or_factory
        if src is not None and hasattr(src, "model_name"):
            # An encoder instance: its dim is known without work once it is loaded.
            dim = int(getattr(src, "dim")) if getattr(src, "is_loaded", True) else None  # noqa: B009
            return str(src.model_name), dim
        return (self.settings.clip_model if src is None else None), None

    def _configured_model_name(self) -> str | None:
        return self._configured_encoder()[0]

    def _is_stale(self, item: WardrobeItem, model: str | None, dim: int | None) -> bool:
        md = item.metadata
        if model and md.get("embedding_model") not in (None, model):
            return True
        return dim is not None and md.get("embedding_dim") not in (None, dim)

    def embedding_status(self) -> EmbeddingStatus:
        status = EmbeddingStatus()
        model, dim = self._configured_encoder()
        for item in self.list_items():
            path = self.resolve(item.embedding_path)
            if not path.is_file():
                status.missing.append(item.id)
            elif self._is_stale(item, model, dim):
                status.stale.append(item.id)
            elif self.load_embedding(item) is None:
                status.corrupt.append(item.id)
            else:
                status.ok.append(item.id)
        return status

    def integrity_report(self) -> IntegrityReport:
        """Missing image files, rows with unknown categories, and unreferenced files."""
        report = IntegrityReport()
        referenced: set[Path] = set()
        for item in self.list_items():
            img = self.resolve(item.image_path)
            referenced.add(img.resolve())
            referenced.add(self.resolve(item.embedding_path).resolve())
            if not img.is_file():
                report.missing_images.append(item.id)
            if item.category not in SLOTS:
                report.invalid_category.append(item.id)
        for d in (self.settings.images_dir, self.settings.embeddings_dir):
            for f in sorted(d.glob("*")):
                if f.is_file() and f.resolve() not in referenced:
                    report.orphan_files.append(f)
        return report

    def remove_orphan_files(self) -> int:
        """Delete image/embedding files no wardrobe row refers to (e.g. after a crash)."""
        files = self.integrity_report().orphan_files
        for f in files:
            f.unlink(missing_ok=True)
        return len(files)

    def reembed(self, item_ids: Iterable[str] | None = None) -> int:
        """Recompute embeddings (and style estimates) from stored images."""
        ids = list(item_ids) if item_ids is not None else [i.id for i in self.list_items()]
        done = 0
        for item_id in ids:
            item = self.get_item(item_id)
            if item is None:
                continue
            img_path = self.resolve(item.image_path)
            if not img_path.is_file():
                log.warning("cannot re-embed %s: image missing", item_id)
                continue
            emb = self.encoder.encode_image(img_path)
            self._validate_embedding(emb)
            emb_path = self.resolve(item.embedding_path)
            _atomic_write_npy(emb, emb_path)
            md = dict(item.metadata)
            md.update(embedding_model=self.encoder.model_name, embedding_dim=int(emb.shape[0]))
            if md.get("style_source") != "user" and self.compute_style:
                try:
                    scores = self._style_classifier().scores(emb)
                    md.update(style_scores=scores, style=top_style(scores), style_source="clip_zero_shot")
                except Exception as exc:
                    log.warning("style estimate skipped: %s", exc)
            if not self.store.update_item(item_id, metadata=md):
                raise WardrobeStoreError(f"Item {item_id} disappeared during re-embedding")
            with self._cache_lock:
                self._cache.pop(item_id, None)
            done += 1
        return done
