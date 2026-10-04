"""Saving/loading the learned compatibility checkpoint.

Training and inference both go through this module and through
``Settings.compatibility_checkpoint``, so there is exactly one checkpoint
location and one format. The checkpoint embeds the model config and the
CLIP encoder config it was trained with; loading refuses a checkpoint whose
encoder does not match the app's encoder, because its predictions would be
meaningless in a different embedding space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Settings, resolve_device
from src.errors import CheckpointError

log = logging.getLogger(__name__)

CHECKPOINT_FORMAT = "fashion-ai/compatibility/v1"


def save_checkpoint(
    path: Path,
    model: Any,
    encoder_config: dict[str, Any],
    metrics: dict[str, Any],
    training_info: dict[str, Any],
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": model.config(),
        "encoder_config": dict(encoder_config),
        "metrics": metrics,
        "training": training_info,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def encoder_configs_match(trained: dict[str, Any], current: dict[str, Any]) -> bool:
    return trained.get("model_name") == current.get("model_name") and int(trained.get("dim", -1)) == int(current.get("dim", -2))


@dataclass
class LearnedCompatibility:
    model: Any
    encoder_config: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    path: Path | None = None

    @classmethod
    def load(cls, path: Path, expected_encoder: dict[str, Any] | None = None, device: str = "cpu") -> LearnedCompatibility:
        import torch

        from src.models.compatibility_net import OutfitCompatibilityNet

        path = Path(path)
        if not path.is_file():
            raise CheckpointError(f"No compatibility checkpoint at {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise CheckpointError(f"Checkpoint {path} could not be read: {type(exc).__name__}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            raise CheckpointError(
                f"Checkpoint {path} is not in the '{CHECKPOINT_FORMAT}' format. Checkpoints from the previous "
                "code base (bare state_dicts trained on synthetic vectors) are not supported; retrain with "
                "scripts/train_compatibility.py."
            )
        enc = payload.get("encoder_config") or {}
        if expected_encoder is not None and not encoder_configs_match(enc, expected_encoder):
            raise CheckpointError(
                f"Checkpoint was trained on embeddings from {enc.get('model_name')} (dim {enc.get('dim')}), "
                f"but the app uses {expected_encoder.get('model_name')} (dim {expected_encoder.get('dim')})."
            )
        try:
            cfg = payload["model_config"]
            model = OutfitCompatibilityNet(
                embed_dim=int(cfg["embed_dim"]), hidden_dims=tuple(cfg["hidden_dims"]), dropout=float(cfg["dropout"])
            )
            model.load_state_dict(payload["state_dict"])
        except Exception as exc:
            raise CheckpointError(f"Checkpoint {path} does not match the model architecture: {exc}") from exc
        model.to(device).eval()
        return cls(
            model=model,
            encoder_config=enc,
            metrics=dict(payload.get("metrics") or {}),
            training=dict(payload.get("training") or {}),
            device=device,
            path=path,
        )

    def predict(self, emb_a: np.ndarray, emb_b: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Compatibility probabilities for row-aligned embedding pairs, shape (n,)."""
        import torch

        a = np.ascontiguousarray(np.atleast_2d(np.asarray(emb_a, dtype=np.float32)))
        b = np.ascontiguousarray(np.atleast_2d(np.asarray(emb_b, dtype=np.float32)))
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
        out = []
        for i in range(0, len(a), batch_size):
            ta = torch.from_numpy(a[i : i + batch_size]).to(self.device)
            tb = torch.from_numpy(b[i : i + batch_size]).to(self.device)
            out.append(self.model.predict_proba(ta, tb).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


@dataclass
class LearnedModelStatus:
    state: str  # "disabled" | "missing" | "loaded" | "not_better" | "error"
    message: str
    model: LearnedCompatibility | None = None


def load_learned_model(settings: Settings, encoder_config: dict[str, Any] | None) -> LearnedModelStatus:
    """Load the learned model according to USE_LEARNED_MODEL. Never raises.

    ``auto``: use the checkpoint if present and compatible, else the baseline.
    ``true``: same, but a missing/incompatible checkpoint is reported as an error.
    ``false``: always the baseline.
    """
    path = settings.compatibility_checkpoint
    if settings.use_learned_model == "false":
        return LearnedModelStatus("disabled", "Learned model disabled (USE_LEARNED_MODEL=false); using the baseline.")
    if not path.is_file():
        state = "error" if settings.use_learned_model == "true" else "missing"
        return LearnedModelStatus(state, f"No trained checkpoint at {path}; using the CLIP + heuristics baseline.")
    try:
        model = LearnedCompatibility.load(path, encoder_config, device=resolve_device(settings.device))
    except CheckpointError as exc:
        return LearnedModelStatus("error", f"{exc} Falling back to the baseline.")
    comparison = model.metrics.get("comparison") or {}
    if not comparison.get("beats_best_baseline", False):
        detail = comparison.get("learned_minus_baseline_roc_auc")
        why = f" (test ROC-AUC difference vs {comparison.get('strongest_non_random_baseline')}: {detail})" if detail else ""
        if settings.use_learned_model == "auto":
            return LearnedModelStatus(
                "not_better",
                f"A trained checkpoint exists at {path}, but it did not beat the strongest baseline on its test split"
                f"{why}, so the CLIP + heuristics baseline is used. Set USE_LEARNED_MODEL=true to force it.",
            )
        return LearnedModelStatus(
            "loaded",
            f"Learned model loaded from {path} because USE_LEARNED_MODEL=true, although it did not beat the baseline{why}.",
            model,
        )
    return LearnedModelStatus(
        "loaded", f"Learned compatibility model loaded from {path} (beat the baseline on its test split).", model
    )
