"""Single source of configuration for the whole application.

Every path, model name and tunable lives here. Values come from (in order of
precedence): real environment variables, a ``.env`` file in the project root,
then the defaults below. Nothing else in the code base should hard-code a model
name, a checkpoint location or a data directory.

Supported variables (all optional):

    DEVICE                  auto | cpu | cuda | cuda:0 | mps
    DETECTOR_MODEL          YOLO-World weights file name or path
    CONFIDENCE_THRESHOLD    detector score at/above which a region is auto-accepted (0-1)
    DETECTION_FLOOR         lowest score shown (as 'low confidence, review') (0-1)
    IOU_THRESHOLD           NMS IoU threshold (0-1)
    CLIP_MODEL              Hugging Face id of the CLIP model
    DATA_DIR                where the wardrobe database / images live
    CHECKPOINT_DIR          where model weights live
    USE_LEARNED_MODEL       auto | true | false
    MAX_UPLOAD_MB           upload size limit for the UI
    DETECTION_FLOOR         regions scoring below this are not shown at all (0-1)
    WEIGHT_CONTEXT          how strongly a requested style/occasion re-ranks outfits
    WEIGHT_VISUAL, WEIGHT_COLOR, WEIGHT_STYLE, WEIGHT_CATEGORY
                            weights of the hand-written heuristic baseline used only for
                            comparison in training/evaluation (the app ranks with the
                            fitted combiner in src/recommendation/ranker_model.json)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMPATIBILITY_CHECKPOINT_NAME = "compatibility_model.pt"


def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env reader (KEY=VALUE lines, # comments). No extra dependency."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class _Env:
    def __init__(self, dotenv: dict[str, str]):
        self._dotenv = dotenv

    def get(self, key: str, default: str) -> str:
        value = os.environ.get(key)
        if value is None or value == "":
            value = self._dotenv.get(key, default)
        return value

    def get_float(self, key: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
        raw = self.get(key, str(default))
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"Configuration error: {key}={raw!r} is not a number") from exc
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            raise ValueError(f"Configuration error: {key}={value} must be within [{lo}, {hi}]")
        return value

    def get_path(self, key: str, default: Path) -> Path:
        value = Path(self.get(key, str(default))).expanduser()
        return value if value.is_absolute() else (PROJECT_ROOT / value)


@dataclass(frozen=True)
class ScoringWeights:
    """Relative weights of the recommendation signals.

    These are hand-chosen starting values, not fitted parameters. They are
    re-normalised at scoring time over whichever signals are actually
    available for a given pair/outfit (e.g. style is dropped if an item has
    no style estimate), so they do not need to sum to 1.
    """

    visual: float = 0.35
    color: float = 0.20
    style: float = 0.20
    category: float = 0.15
    context: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return {
            "visual": self.visual,
            "color": self.color,
            "style": self.style,
            "category": self.category,
            "context": self.context,
        }


@dataclass(frozen=True)
class Settings:
    device: str = "auto"
    detector_model: str = "yolov8s-worldv2.pt"
    confidence_threshold: float = 0.25
    detection_floor: float = 0.15
    iou_threshold: float = 0.45
    clip_model: str = "openai/clip-vit-base-patch32"
    data_dir: Path = PROJECT_ROOT / "data"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    use_learned_model: str = "auto"
    max_upload_mb: float = 15.0
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    # ---- derived locations -------------------------------------------------
    @property
    def wardrobe_dir(self) -> Path:
        return self.data_dir / "wardrobe"

    @property
    def images_dir(self) -> Path:
        return self.wardrobe_dir / "images"

    @property
    def embeddings_dir(self) -> Path:
        return self.wardrobe_dir / "embeddings"

    @property
    def db_path(self) -> Path:
        return self.wardrobe_dir / "wardrobe.db"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def crops_dir(self) -> Path:
        return self.data_dir / "processed" / "crops"

    @property
    def compatibility_checkpoint(self) -> Path:
        """The ONE place the learned compatibility model is saved and loaded."""
        return self.checkpoint_dir / COMPATIBILITY_CHECKPOINT_NAME

    @property
    def detector_weights_path(self) -> Path:
        """Detector weights path. A bare file name is resolved inside CHECKPOINT_DIR."""
        p = Path(self.detector_model).expanduser()
        if p.is_absolute() or len(p.parts) > 1:
            return p if p.is_absolute() else PROJECT_ROOT / p
        return self.checkpoint_dir / p

    def ensure_dirs(self) -> None:
        for d in (self.images_dir, self.embeddings_dir, self.uploads_dir, self.crops_dir, self.checkpoint_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Build Settings from the environment. ``env`` overrides os.environ (used by tests)."""
    dotenv = _load_dotenv(PROJECT_ROOT / ".env")
    if env is not None:
        dotenv = {**dotenv, **env}
    e = _Env(dotenv)

    use_learned = e.get("USE_LEARNED_MODEL", "auto").lower()
    if use_learned not in {"auto", "true", "false"}:
        raise ValueError(f"Configuration error: USE_LEARNED_MODEL must be auto|true|false, got {use_learned!r}")

    weights = ScoringWeights(
        visual=e.get_float("WEIGHT_VISUAL", ScoringWeights.visual, 0.0),
        color=e.get_float("WEIGHT_COLOR", ScoringWeights.color, 0.0),
        style=e.get_float("WEIGHT_STYLE", ScoringWeights.style, 0.0),
        category=e.get_float("WEIGHT_CATEGORY", ScoringWeights.category, 0.0),
        context=e.get_float("WEIGHT_CONTEXT", ScoringWeights.context, 0.0),
    )
    if sum(weights.as_dict().values()) <= 0:
        raise ValueError("Configuration error: at least one scoring weight must be positive")

    return Settings(
        device=e.get("DEVICE", "auto").lower(),
        detector_model=e.get("DETECTOR_MODEL", Settings.detector_model),
        confidence_threshold=e.get_float("CONFIDENCE_THRESHOLD", Settings.confidence_threshold, 0.0, 1.0),
        detection_floor=e.get_float("DETECTION_FLOOR", Settings.detection_floor, 0.0, 1.0),
        iou_threshold=e.get_float("IOU_THRESHOLD", Settings.iou_threshold, 0.0, 1.0),
        clip_model=e.get("CLIP_MODEL", Settings.clip_model),
        data_dir=e.get_path("DATA_DIR", PROJECT_ROOT / "data"),
        checkpoint_dir=e.get_path("CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"),
        use_learned_model=use_learned,
        max_upload_mb=e.get_float("MAX_UPLOAD_MB", Settings.max_upload_mb, 0.1),
        weights=weights,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def resolve_device(preference: str = "auto") -> str:
    """Turn a DEVICE preference into a concrete torch device string.

    Falls back to CPU (with no error) when CUDA/MPS is requested but missing,
    so the app always starts.
    """
    pref = (preference or "auto").lower()
    try:
        import torch
    except ImportError:
        return "cpu"

    if pref == "cpu":
        return "cpu"
    if pref.startswith("cuda"):
        return pref if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        mps = getattr(torch.backends, "mps", None)
        return "mps" if mps is not None and mps.is_available() else "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
