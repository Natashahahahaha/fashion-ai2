from __future__ import annotations

from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, load_settings, resolve_device


def test_defaults():
    s = load_settings({})
    assert s.compatibility_checkpoint == s.checkpoint_dir / "compatibility_model.pt"
    assert s.detector_weights_path.parent == s.checkpoint_dir
    assert s.db_path == s.data_dir / "wardrobe" / "wardrobe.db"


def test_env_overrides(tmp_path):
    s = load_settings(
        {"DATA_DIR": str(tmp_path), "CONFIDENCE_THRESHOLD": "0.4", "CLIP_MODEL": "x/y", "WEIGHT_STYLE": "0", "DEVICE": "CPU"}
    )
    assert s.data_dir == tmp_path and s.confidence_threshold == 0.4 and s.clip_model == "x/y"
    assert s.weights.style == 0.0 and s.device == "cpu"
    rel = load_settings({"CHECKPOINT_DIR": "my_ckpts"})
    assert rel.checkpoint_dir == PROJECT_ROOT / "my_ckpts"
    path_model = load_settings({"DETECTOR_MODEL": "weights/custom.pt"})
    assert path_model.detector_weights_path == PROJECT_ROOT / Path("weights/custom.pt")


@pytest.mark.parametrize(
    "env",
    [
        {"CONFIDENCE_THRESHOLD": "1.5"},
        {"CONFIDENCE_THRESHOLD": "abc"},
        {"USE_LEARNED_MODEL": "maybe"},
        {k: "0" for k in ("WEIGHT_VISUAL", "WEIGHT_COLOR", "WEIGHT_STYLE", "WEIGHT_CATEGORY", "WEIGHT_CONTEXT")},
    ],
)
def test_invalid_config_rejected(env):
    with pytest.raises(ValueError, match="Configuration error"):
        load_settings(env)


def test_device_resolution_falls_back_to_cpu():
    import torch

    assert resolve_device("cpu") == "cpu"
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto") == expected
    assert resolve_device("cuda") == expected
