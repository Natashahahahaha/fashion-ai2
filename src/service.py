"""Wires detector, encoder, wardrobe and recommender together.

The Streamlit UI, the CLI scripts and the smoke test all use this facade, so
there is one place where components are constructed. Heavy models load
lazily: listing or deleting wardrobe items never loads CLIP or YOLO.
"""

from __future__ import annotations

import threading
from typing import Any

from src.config import Settings, get_settings, resolve_device
from src.detection.detector import FashionDetector
from src.embeddings.clip_encoder import ImageEncoder, get_encoder
from src.models.learned import LearnedModelStatus, load_learned_model
from src.recommendation.outfit_generator import OutfitGenerator
from src.wardrobe.manager import WardrobeManager


class FashionAIService:
    def __init__(
        self,
        settings: Settings | None = None,
        encoder: ImageEncoder | None = None,
        detector: FashionDetector | None = None,
    ):
        self.settings = settings or get_settings()
        self._encoder = encoder
        self.detector = detector or FashionDetector(self.settings)
        self.manager = WardrobeManager(self.settings, encoder=self._get_encoder)
        self._learned: LearnedModelStatus | None = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return resolve_device(self.settings.device)

    def _get_encoder(self) -> ImageEncoder:
        if self._encoder is None:
            self._encoder = get_encoder(self.settings)
        return self._encoder

    @property
    def encoder(self) -> ImageEncoder:
        return self._get_encoder()

    @property
    def encoder_loaded(self) -> bool:
        enc = self._encoder
        return enc is not None and bool(getattr(enc, "is_loaded", True))

    def expected_encoder_config(self) -> dict[str, Any]:
        """Identity of the app's embedding space without loading CLIP weights.

        Uses the already-loaded encoder if there is one, otherwise only the
        model's small config file (projection_dim) from the local cache.
        """
        if self.encoder_loaded:
            return self.encoder.config()
        try:
            from transformers import CLIPConfig

            try:
                cfg = CLIPConfig.from_pretrained(self.settings.clip_model, local_files_only=True)
            except Exception:
                cfg = CLIPConfig.from_pretrained(self.settings.clip_model)
            if cfg.projection_dim is None:
                raise ValueError("CLIP config has no projection_dim")
            return {"model_name": self.settings.clip_model, "dim": int(cfg.projection_dim), "normalized": True}
        except Exception:
            return self.encoder.config()  # last resort: load the encoder

    def learned_status(self) -> LearnedModelStatus:
        """Load the optional learned model once (never loads CLIP weights just to check compatibility)."""
        with self._lock:
            if self._learned is None:
                needs_encoder = self.settings.use_learned_model != "false" and self.settings.compatibility_checkpoint.is_file()
                enc_cfg = self.expected_encoder_config() if needs_encoder else None
                self._learned = load_learned_model(self.settings, enc_cfg)
            return self._learned

    def generator(self) -> OutfitGenerator:
        return OutfitGenerator(self.manager, learned=self.learned_status().model, weights=self.settings.weights)

    def device_info(self) -> dict[str, Any]:
        """Active device, CUDA build/availability, GPU name/VRAM and a warning for unintended CPU mode."""
        import torch

        requested = (self.settings.device or "auto").lower()
        active = self.device
        info: dict[str, Any] = {
            "requested": requested,
            "active": active,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,  # None for CPU-only PyTorch wheels
            "cuda_available": False,
            "gpu": None,
            "vram_total_mb": None,
            "vram_free_mb": None,
            "vram_allocated_mb": None,
            "warning": None,
        }
        try:
            info["cuda_available"] = bool(torch.cuda.is_available())
            if info["cuda_available"]:
                idx = torch.device(active).index or 0 if active.startswith("cuda") else 0
                info["gpu"] = torch.cuda.get_device_name(idx)
                free, total = torch.cuda.mem_get_info(idx)
                info["vram_total_mb"] = round(total / 2**20)
                info["vram_free_mb"] = round(free / 2**20)
                info["vram_allocated_mb"] = round(torch.cuda.memory_allocated(idx) / 2**20)
        except Exception as exc:  # broken driver / incompatible CUDA runtime
            info["warning"] = (
                f"CUDA initialisation failed ({type(exc).__name__}: {exc}). Running on CPU. "
                "Check the NVIDIA driver and that the installed PyTorch CUDA build matches it; nothing was changed automatically."
            )
            return info
        if active == "cpu" and info["warning"] is None:
            if requested.startswith("cuda") and not info["cuda_available"]:
                info["warning"] = "DEVICE requests CUDA, but CUDA was not detected, so the app fell back to CPU (much slower)."
            elif info["torch_cuda_build"] is None:
                info["warning"] = (
                    "Running on CPU: the installed PyTorch is a CPU-only build. CUDA is the recommended runtime; "
                    "see README 'Installation' for installing the CUDA build. CPU works but is much slower."
                )
            elif not info["cuda_available"]:
                info["warning"] = "Running on CPU: no usable NVIDIA GPU/driver was detected. CPU works but is much slower."
        return info

    def model_devices(self) -> dict[str, str | None]:
        """Where the loaded models actually live (None = not loaded yet)."""
        clip_dev = None
        enc = self._encoder
        model = getattr(enc, "_model", None) if enc is not None else None
        if model is not None:
            try:
                clip_dev = str(next(model.parameters()).device)
            except Exception:
                clip_dev = None
        det_dev = None
        backend = getattr(self.detector, "_backend", None)
        if backend is not None and hasattr(backend, "model"):
            try:
                det_dev = str(next(backend.model.model.parameters()).device)
            except Exception:
                det_dev = getattr(backend, "device", None)
        return {"clip": clip_dev, "detector": det_dev}

    last_outfit_diagnostics: dict[str, Any] | None = None  # set by the UI after each generation

    def diagnostics(self) -> dict[str, Any]:
        import torch

        s = self.settings
        status = self.manager.embedding_status()
        integrity = self.manager.integrity_report()
        return {
            "device": self.device,
            "device_setting": s.device,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_info": self.device_info(),
            "model_devices": self.model_devices(),
            "detector_model": s.detector_model,
            "detector_weights": str(self.detector.weights_path),
            "detector_weights_present": self.detector.weights_path.is_file(),
            "detector_cache": str(self.detector.cached_weights_path),
            "detector_cache_present": self.detector.cached_weights_path.is_file(),
            "detector_loaded": self.detector.is_loaded,
            "detector_error": self.detector.load_error,
            "confidence_threshold": s.confidence_threshold,
            "clip_model": s.clip_model,
            "clip_loaded": self.encoder_loaded,
            "wardrobe_items": self.manager.count(),
            "items_by_category": self.manager.store.count_by_category(),
            "embeddings_ok": len(status.ok),
            "embeddings_missing": status.missing,
            "embeddings_corrupt": status.corrupt,
            "embeddings_stale": status.stale,
            "missing_images": integrity.missing_images,
            "invalid_category": integrity.invalid_category,
            "orphan_files": len(integrity.orphan_files),
            "checkpoint_path": str(s.compatibility_checkpoint),
            "checkpoint_present": s.compatibility_checkpoint.is_file(),
            "use_learned_model": s.use_learned_model,
            "weights": s.weights.as_dict(),
            "data_dir": str(s.data_dir),
            "db_path": str(s.db_path),
        }
