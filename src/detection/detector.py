"""The one clothing detector used by the whole application.

Usage::

    detector = FashionDetector()
    detections = detector.detect("photo.jpg")   # -> list[DetectionResult]

The default backend is Ultralytics YOLO-World, prompted with the clothing
vocabulary in ``categories.DETECTOR_PROMPTS``. The backend is pluggable (any
callable ``PIL.Image -> list[RawDetection]``) so tests can run without
downloading weights, and so a different detector can be dropped in without
touching the rest of the pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.config import Settings, get_settings, resolve_device
from src.detection.categories import DETECTOR_PROMPTS, slot_for_label
from src.detection.schemas import (
    Alternative,
    DetectionResult,
    ReviewFlag,
    box_area,
    box_iou,
    clip_bbox,
    containment,
)
from src.errors import DetectionError, DetectorUnavailableError
from src.imaging import ImageInput, load_image

log = logging.getLogger(__name__)

# Inference runs on a copy whose longest side is at most this many pixels
# (YOLO resizes to 640 anyway); boxes are mapped back to the original image.
MAX_INFERENCE_SIDE = 2048


@dataclass(frozen=True)
class RawDetection:
    """Backend output before validation: label text, confidence, xyxy pixels."""

    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]


DetectorBackend = Callable[[Image.Image], list[RawDetection]]


def vocabulary_key(weights: Path, classes: Sequence[str]) -> str:
    """Short hash identifying a (base weights, prompt list) combination."""
    return hashlib.sha1((weights.name + "\n" + "\n".join(classes)).encode("utf-8")).hexdigest()[:8]


class UltralyticsWorldBackend:
    """YOLO-World (open-vocabulary YOLO) via the ``ultralytics`` package.

    Encoding the text prompts needs YOLO-World's CLIP text encoder (and the
    ``clip`` package installed from GitHub). That is done once; the model with
    the encoded vocabulary is then saved next to the base weights as
    ``<stem>.fashion-<hash>.pt`` and later starts load that file directly,
    without the text encoder. Changing the prompt list changes the hash.
    """

    def __init__(
        self,
        weights: Path,
        classes: Sequence[str],
        device: str,
        confidence_threshold: float,
        iou_threshold: float,
        max_detections: int = 100,  # compute cap only; real photos yield far fewer
    ):
        # Never let Ultralytics pip-install packages at runtime; report instead.
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        try:
            from ultralytics import YOLO, YOLOWorld
        except ImportError as exc:
            raise DetectorUnavailableError(
                "The detector needs the 'ultralytics' package, which is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        self.classes = list(classes)
        self.device = device
        self.conf = confidence_threshold
        self.iou = iou_threshold
        self.max_det = max_detections
        self.cached_path = weights.with_name(f"{weights.stem}.fashion-{vocabulary_key(weights, self.classes)}.pt")
        weights.parent.mkdir(parents=True, exist_ok=True)

        if self.cached_path.is_file():
            try:
                model: Any = YOLO(str(self.cached_path))
                if list(model.names.values()) == self.classes:
                    self.model = model
                    self._to_device()
                    return
                log.warning("cached detector %s has a different vocabulary; rebuilding", self.cached_path)
            except Exception as exc:
                log.warning("cached detector %s unreadable (%s); rebuilding", self.cached_path, exc)

        try:
            # Downloads the official weights to this exact path if missing.
            model = YOLOWorld(str(weights))
        except Exception as exc:
            raise DetectorUnavailableError(
                f"Could not load the YOLO-World weights at {weights} ({type(exc).__name__}: {exc}). "
                "The first run downloads them (~25 MB): check your internet connection, or download "
                f"'{weights.name}' from the Ultralytics releases page and place it there."
            ) from exc
        try:
            self._encode_vocabulary(model, weights.parent)
        except ImportError as exc:
            raise DetectorUnavailableError(
                "YOLO-World needs the 'clip' package to encode its clothing vocabulary the first time. "
                "Install it with `pip install -r requirements.txt` (this needs git, because the package "
                f"is installed from GitHub). Details: {exc}"
            ) from exc
        except Exception as exc:
            raise DetectorUnavailableError(
                f"Could not prepare YOLO-World's clothing vocabulary ({type(exc).__name__}: {exc}). "
                "The first run downloads a ~340 MB text encoder: check your internet connection."
            ) from exc
        self.model = model
        try:
            model.save(str(self.cached_path))
        except Exception as exc:  # not fatal: we just pay the encoding cost again next time
            log.warning("could not cache the detector at %s: %s", self.cached_path, exc)
        self._to_device()

    def _to_device(self) -> None:
        """Move weights to the target device now (Ultralytics would otherwise do it on the first predict)."""
        try:
            self.model.to(self.device)
        except Exception as exc:
            raise DetectorUnavailableError(
                f"Could not move YOLO-World to device '{self.device}' ({type(exc).__name__}: {exc}). "
                "Check the NVIDIA driver / PyTorch CUDA build, or set DEVICE=cpu."
            ) from exc

    def _encode_vocabulary(self, model: Any, weights_dir: Path) -> None:
        import ultralytics.nn.text_model as text_model

        # The text encoder is downloaded to Ultralytics' global WEIGHTS_DIR
        # (by default the directory Ultralytics was first run from). Point it
        # at CHECKPOINT_DIR for this call only; global settings are untouched.
        previous = text_model.WEIGHTS_DIR
        text_model.WEIGHTS_DIR = weights_dir
        try:
            model.set_classes(self.classes)
        finally:
            text_model.WEIGHTS_DIR = previous
        inner = getattr(model, "model", None)
        if inner is not None and getattr(inner, "clip_model", None) is not None:
            inner.clip_model = None  # do not keep (or save) the 340 MB text encoder

    def __call__(self, image: Image.Image) -> list[RawDetection]:
        # Passing a PIL image (not a numpy array) matters: Ultralytics treats
        # numpy input as BGR, but converts PIL input correctly itself.
        results = self.model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            # Class-wise NMS: overlapping garments of different types (coat over shirt,
            # jeans under a coat) must survive; FashionDetector merges same-region boxes.
            agnostic_nms=False,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        result: Any = list(results)[0]
        names = result.names
        boxes = result.boxes
        out: list[RawDetection] = []
        if boxes is None:
            return out
        for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True):
            label = names[int(cls)] if isinstance(names, dict) else self.classes[int(cls)]
            out.append(RawDetection(label=str(label), confidence=float(conf), xyxy=tuple(xyxy)))  # type: ignore[arg-type]
        return out


# Region / review heuristics. These are documented engineering choices based on
# inspecting real YOLO-World outputs (e.g. the bus.jpg "jumpsuit" that was really
# a coat over jeans), not values fitted on labelled data.
EVIDENCE_FLOOR = 0.10  # raw boxes at/above this are kept as evidence (alternatives, containment)
REGION_IOU = 0.55  # boxes overlapping this much describe the same physical region
AMBIGUITY_RATIO = 0.6  # a different-slot label scoring >= 60% of the top label = ambiguous
CONTAINED = 0.8  # a box is "inside" a region if >= 80% of its area lies within it
PART_OF_REGION = 0.75  # ...and covers at most 75% of the region (a part, not the same box)
MIN_CROP_SIDE = 32  # px; smaller crops are too low-resolution to trust


@dataclass
class _Region:
    conf: float
    bbox: tuple[int, int, int, int]
    label: str
    slot: str
    alternatives: list[tuple[float, str, str]]  # (conf, label, slot)


class FashionDetector:
    """Detect garments, group overlapping predictions into regions, flag uncertain ones.

    Pipeline per image:
      1. YOLO-World with *class-wise* NMS and a low evidence floor, so overlapping
         garments of different types (a coat over a shirt, jeans under a coat)
         are not deleted by cross-class suppression.
      2. Boxes describing the same region are merged; the top label becomes the
         region's label and the others are kept as ``alternatives``.
      3. Each region is checked and flagged for review when it is low-confidence,
         ambiguous between slots, a one-piece that also matches/contains
         separates, or a tiny crop. Flagged regions are returned, not hidden.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        backend: DetectorBackend | None = None,
        confidence_threshold: float | None = None,
        display_floor: float | None = None,
        crop_padding: float = 0.03,
        min_box_size: int = 16,
        min_area_fraction: float = 0.002,
    ):
        self.settings = settings or get_settings()
        self.device = resolve_device(self.settings.device)
        # Regions at/above this score (and with no other flag) are auto-accepted.
        self.confidence_threshold = self.settings.confidence_threshold if confidence_threshold is None else confidence_threshold
        # Regions between display_floor and confidence_threshold are shown as low-confidence.
        self.display_floor = min(
            self.settings.detection_floor if display_floor is None else display_floor, self.confidence_threshold
        )
        self.crop_padding = crop_padding
        self.min_box_size = min_box_size
        self.min_area_fraction = min_area_fraction
        self._backend: DetectorBackend | None = backend
        self._load_error: str | None = None
        self.last_dropped = 0  # malformed/duplicate raw predictions skipped in the last call
        self.last_raw_count = 0  # raw predictions returned by the backend in the last call

    # ------------------------------------------------------------------ load
    @property
    def weights_path(self) -> Path:
        return self.settings.detector_weights_path

    @property
    def cached_weights_path(self) -> Path:
        return self.weights_path.with_name(
            f"{self.weights_path.stem}.fashion-{vocabulary_key(self.weights_path, DETECTOR_PROMPTS)}.pt"
        )

    def load(self) -> None:
        """Load the backend now (otherwise it is loaded on first ``detect``)."""
        if self._backend is not None:
            return
        try:
            self._backend = UltralyticsWorldBackend(
                weights=self.weights_path,
                classes=DETECTOR_PROMPTS,
                device=self.device,
                confidence_threshold=min(EVIDENCE_FLOOR, self.display_floor),
                iou_threshold=self.settings.iou_threshold,
            )
            self._load_error = None
            log.info("YOLO-World ready (%s) on %s", self.cached_weights_path.name, self.device)
        except DetectorUnavailableError as exc:
            self._load_error = str(exc)
            raise

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    # ---------------------------------------------------------------- detect
    def detect(
        self,
        image: ImageInput,
        save_crops: bool = True,
        crop_dir: Path | None = None,
    ) -> list[DetectionResult]:
        """Detect clothing regions in ``image`` (highest score first).

        Every region at/above ``display_floor`` is returned; uncertain ones carry
        review ``flags``. Malformed backend predictions are skipped (counted in
        ``last_dropped``), never raised. Raises ``InvalidImageError`` for
        unreadable input, ``DetectorUnavailableError`` if the model cannot be
        loaded and ``DetectionError`` if inference itself fails.
        """
        source_image = str(image) if isinstance(image, (str, Path)) else None
        img = load_image(image)
        self.load()
        assert self._backend is not None

        width, height = img.size
        scale = 1.0
        work = img
        if max(width, height) > MAX_INFERENCE_SIDE:
            scale = max(width, height) / MAX_INFERENCE_SIDE
            work = img.resize((max(1, round(width / scale)), max(1, round(height / scale))), Image.Resampling.LANCZOS)

        try:
            raw = list(self._backend(work))
        except Exception as exc:
            raise DetectionError(
                f"The detector failed on this image ({type(exc).__name__}: {exc}). "
                "Try another photo; if it keeps happening, check the Diagnostics page."
            ) from exc
        self.last_raw_count = len(raw)

        image_key = _image_key(img)
        min_area = self.min_area_fraction * width * height
        seen: set[tuple[str, tuple[int, int, int, int]]] = set()
        evidence: list[tuple[float, tuple[int, int, int, int], str, str]] = []
        dropped = 0
        for r in raw:
            parsed = self._validate(r, scale, width, height, min_area)
            if parsed is None:
                dropped += 1
                continue
            conf, bbox, label, slot = parsed
            if (label, bbox) in seen:  # exact duplicate prediction
                dropped += 1
                continue
            seen.add((label, bbox))
            if conf >= min(EVIDENCE_FLOOR, self.display_floor):
                evidence.append(parsed)
        self.last_dropped = dropped
        evidence.sort(key=lambda v: (-v[0], v[1]))

        regions = [r for r in self._group(evidence) if r.conf >= self.display_floor]

        out_dir: Path | None = None
        if save_crops and regions:
            out_dir = crop_dir or (self.settings.crops_dir / image_key)
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            out_dir.mkdir(parents=True, exist_ok=True)

        detections: list[DetectionResult] = []
        for reg in regions:
            flags, suggested = self._review(reg, evidence)
            det_id = hashlib.sha1(f"{image_key}|{reg.label}|{reg.bbox}".encode()).hexdigest()[:12]
            crop_path: str | None = None
            if out_dir is not None:
                crop = img.crop(self._padded(reg.bbox, width, height))
                path = out_dir / f"{det_id}.jpg"
                crop.save(path, format="JPEG", quality=92)
                crop_path = str(path)
            detections.append(
                DetectionResult(
                    id=det_id,
                    label=reg.label,
                    category=reg.slot,
                    confidence=round(reg.conf, 4),
                    bbox=reg.bbox,
                    crop_path=crop_path,
                    source_image=source_image,
                    alternatives=tuple(Alternative(lab, sl, round(c, 4)) for c, lab, sl in reg.alternatives),
                    flags=tuple(flags),
                    suggested_category=suggested,
                )
            )
        return detections

    @staticmethod
    def _group(evidence: list[tuple[float, tuple[int, int, int, int], str, str]]) -> list[_Region]:
        """Greedy grouping (highest score first) of boxes that describe the same region."""
        regions: list[_Region] = []
        for conf, bbox, label, slot in evidence:
            home = next((r for r in regions if box_iou(r.bbox, bbox) >= REGION_IOU), None)
            if home is None:
                regions.append(_Region(conf, bbox, label, slot, []))
            elif label != home.label and all(label != lab for _, lab, _ in home.alternatives):
                home.alternatives.append((conf, label, slot))
        return regions

    def _review(
        self, reg: _Region, evidence: list[tuple[float, tuple[int, int, int, int], str, str]]
    ) -> tuple[list[ReviewFlag], str | None]:
        flags: list[ReviewFlag] = []
        suggested: str | None = None
        if reg.conf < self.confidence_threshold:
            flags.append(ReviewFlag("low_confidence", f"Low detector score ({reg.conf:.2f} < {self.confidence_threshold:.2f})."))
        other_slot = [(c, lab, sl) for c, lab, sl in reg.alternatives if sl != reg.slot]
        close = [a for a in other_slot if a[0] >= AMBIGUITY_RATIO * reg.conf]
        if close:
            c, lab, sl = close[0]
            flags.append(
                ReviewFlag("ambiguous_category", f"Also detected as {lab} ({c:.2f}), close to {reg.label} ({reg.conf:.2f}).")
            )
            suggested = suggested or sl
        if reg.slot == "one_piece":
            separates_alt = [a for a in other_slot if a[2] in ("top", "bottom", "outerwear")]
            region_area = box_area(reg.bbox)
            parts = [
                (c, lab, sl)
                for c, b, lab, sl in evidence
                if sl in ("top", "bottom", "outerwear")
                and containment(b, reg.bbox) >= CONTAINED
                and box_area(b) <= PART_OF_REGION * region_area
            ]
            if separates_alt or parts:
                best: dict[str, float] = {}  # the same evidence can be both an alternative and a contained box
                for c, lab, _ in separates_alt + parts:
                    best[lab] = max(c, best.get(lab, 0.0))
                bits = [f"{lab} {c:.2f}" for lab, c in sorted(best.items(), key=lambda kv: -kv[1])[:4]]
                flags.append(
                    ReviewFlag(
                        "contains_separates",
                        "This one-piece region also matches separate garments ("
                        + ", ".join(bits)
                        + "). It may be a person wearing separates, not a dress/jumpsuit.",
                    )
                )
                if separates_alt:
                    suggested = separates_alt[0][2]
        w, h = reg.bbox[2] - reg.bbox[0], reg.bbox[3] - reg.bbox[1]
        if min(w, h) < MIN_CROP_SIDE:
            flags.append(ReviewFlag("small_crop", f"Crop is only {w}x{h} px; too small to judge reliably."))
        return flags, suggested

    def _validate(
        self, r: Any, scale: float, width: int, height: int, min_area: float
    ) -> tuple[float, tuple[int, int, int, int], str, str] | None:
        """Normalise one raw prediction, or None if it is malformed/unusable."""
        try:
            label = str(r.label).strip().lower()
            conf = float(r.confidence)
            coords = [float(v) * scale for v in r.xyxy]
        except (AttributeError, TypeError, ValueError):
            return None
        slot = slot_for_label(label)
        if slot is None or not math.isfinite(conf) or not (0.0 <= conf <= 1.0):
            return None
        if len(coords) != 4:
            return None
        bbox = clip_bbox(coords, width, height, min_size=self.min_box_size)
        if bbox is None or (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < min_area:
            return None
        return conf, bbox, label, slot

    def _padded(self, bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        px = int((x2 - x1) * self.crop_padding)
        py = int((y2 - y1) * self.crop_padding)
        return (max(0, x1 - px), max(0, y1 - py), min(width, x2 + px), min(height, y2 + py))


def _image_key(img: Image.Image) -> str:
    """Stable short id for an image's pixels (re-detecting replaces old crops)."""
    small = img.copy()
    small.thumbnail((128, 128))
    return hashlib.sha1(small.tobytes() + str(img.size).encode()).hexdigest()[:12]
