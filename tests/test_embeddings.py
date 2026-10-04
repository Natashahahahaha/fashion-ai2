"""ClipEncoder tests with a tiny randomly initialised CLIP (real code path, no download)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.embeddings.clip_encoder import ClipEncoder, l2_normalize
from src.errors import EmbeddingError


@pytest.fixture(scope="module")
def tiny_encoder():
    torch = pytest.importorskip("torch")
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel

    torch.manual_seed(0)
    cfg = CLIPConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "vocab_size": 100,
            "max_position_embeddings": 16,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 32,
            "patch_size": 8,
        },
        projection_dim=24,
    )
    processor = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})

    class Tok:
        def __call__(self, texts, **kw):
            ids = torch.tensor([[min(ord(c), 99) for c in t[:8].ljust(8)] for t in texts])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    return ClipEncoder(model_name="tiny-test-clip", device="cpu", model=CLIPModel(cfg), processor=processor, tokenizer=Tok())


def _img(color):
    return Image.new("RGB", (64, 48), color)


def test_dimension_comes_from_model_config(tiny_encoder):
    assert tiny_encoder.dim == 24
    assert tiny_encoder.config() == {"model_name": "tiny-test-clip", "dim": 24, "normalized": True}


def test_image_embeddings_are_unit_norm_float32(tiny_encoder):
    v = tiny_encoder.encode_image(_img((200, 10, 10)))
    assert v.shape == (24,) and v.dtype == np.float32
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_batch_matches_single_and_is_deterministic(tiny_encoder):
    imgs = [_img((200, 10, 10)), _img((10, 200, 10)), _img((10, 10, 200))]
    batch = tiny_encoder.encode_images(imgs)
    assert batch.shape == (3, 24)
    np.testing.assert_allclose(batch[1], tiny_encoder.encode_image(imgs[1]), atol=1e-5)
    np.testing.assert_allclose(batch, tiny_encoder.encode_images(imgs), atol=1e-6)
    assert tiny_encoder.encode_images([]).shape == (0, 24)


def test_text_embeddings(tiny_encoder):
    t = tiny_encoder.encode_text(["a photo of jeans", "a photo of a dress"])
    assert t.shape == (2, 24)
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-5)


def test_cosine_similarity_is_dot_product(tiny_encoder):
    a, b = tiny_encoder.encode_images([_img((200, 10, 10)), _img((10, 10, 200))])
    assert -1.0 <= float(a @ b) <= 1.0
    assert float(a @ a) == pytest.approx(1.0, abs=1e-5)


def test_unloadable_model_raises_embedding_error():
    enc = ClipEncoder(model_name="this-org/does-not-exist-xyz", device="cpu")
    with pytest.raises(EmbeddingError, match="Could not load CLIP"):
        enc.encode_image(_img((1, 2, 3)))


def test_l2_normalize_handles_zero_vector():
    out = l2_normalize(np.zeros((2, 3)))
    assert np.isfinite(out).all()
