from __future__ import annotations

import numpy as np
import pytest

from conftest import populate, solid
from src.errors import FashionAIError, WardrobeStoreError
from src.wardrobe.manager import WardrobeManager
from src.wardrobe.store import WardrobeStore


def test_add_get_list(manager, settings):
    item = manager.add_item(solid((30, 50, 110)), category="bottom", label="jeans", confidence=0.8)
    assert manager.get_item(item.id) == item
    assert [i.id for i in manager.list_items()] == [item.id]
    assert manager.resolve(item.image_path).is_file()
    assert manager.resolve(item.embedding_path).is_file()
    # stored relative to DATA_DIR so the folder is relocatable
    assert not item.image_path.startswith(str(settings.data_dir))
    md = item.metadata
    assert md["label"] == "jeans" and md["embedding_model"] == "fake-clip" and md["embedding_dim"] == 32
    assert md["color"] == "navy"
    assert md["style_source"] == "clip_zero_shot" and abs(sum(md["style_scores"].values()) - 1) < 1e-3


def test_persists_across_instances(settings, encoder):
    m1 = WardrobeManager(settings, encoder=encoder)
    item = m1.add_item(solid((200, 30, 40)), category="top")
    m2 = WardrobeManager(settings, encoder=encoder)
    assert m2.get_item(item.id).metadata == item.metadata


def test_delete_removes_row_and_files(manager):
    item = manager.add_item(solid((10, 10, 10)), category="shoes")
    img, emb = manager.resolve(item.image_path), manager.resolve(item.embedding_path)
    assert manager.delete_item(item.id) is True
    assert manager.get_item(item.id) is None
    assert not img.exists() and not emb.exists()
    assert manager.delete_item(item.id) is False


def test_clear_and_search(manager):
    items = populate(manager)
    assert len(manager.search_items(category="top")) == 3
    assert [i.id for i in manager.search_items(label="jeans")] == [items["blue jeans"].id]
    assert items["black shoes"].id in {i.id for i in manager.search_items(color="black")}
    assert manager.search_items(text="dress")[0].id == items["green dress"].id
    assert manager.clear_wardrobe() == len(items)
    assert manager.count() == 0
    assert not any(manager.settings.images_dir.iterdir())


def test_update_item_user_overrides(manager):
    item = manager.add_item(solid((30, 50, 110)), category="bottom", label="jeans")
    updated = manager.update_item(item.id, category="top", style="formal")
    assert updated.category == "top"
    assert updated.metadata["style"] == "formal" and updated.metadata["style_source"] == "user"
    assert updated.metadata["label_source"] == "user"


def test_rejects_unknown_category(manager):
    with pytest.raises(FashionAIError):
        manager.add_item(solid((1, 2, 3)), category="spaceship")


def test_failed_embedding_writes_nothing(settings):
    class Broken:
        model_name = "broken"
        dim = 4

        def encode_image(self, img):
            return np.array([np.nan, 0, 0, 0], dtype=np.float32)

    m = WardrobeManager(settings, encoder=Broken(), compute_style=False)
    with pytest.raises(FashionAIError):
        m.add_item(solid((1, 2, 3)), category="top")
    assert m.count() == 0
    assert not any(settings.images_dir.iterdir())


def test_corrupt_and_missing_embeddings_are_reported(manager):
    items = populate(manager)
    bad = items["red top"]
    manager.resolve(bad.embedding_path).write_bytes(b"garbage")
    gone = items["beige pants"]
    manager.resolve(gone.embedding_path).unlink()
    status = manager.embedding_status()
    assert bad.id in status.corrupt and gone.id in status.missing
    loaded = manager.load_embeddings(manager.list_items())
    assert bad.id not in loaded and gone.id not in loaded and len(loaded) == len(items) - 2
    # re-embedding repairs both
    assert manager.reembed(status.unusable) == 2
    assert not manager.embedding_status().unusable


def test_stale_embeddings_detected_when_model_changes(settings, encoder):
    m = WardrobeManager(settings, encoder=encoder)
    m.add_item(solid((1, 200, 3)), category="top")

    class Other(type(encoder)):
        model_name = "other-clip"

    m2 = WardrobeManager(settings, encoder=Other())
    assert len(m2.embedding_status().stale) == 1
    assert m2.load_embeddings(m2.list_items()) == {}


def test_malformed_database_gives_readable_error(tmp_path):
    db = tmp_path / "wardrobe.db"
    db.write_bytes(b"this is not sqlite" * 100)
    with pytest.raises(WardrobeStoreError, match="unreadable"):
        WardrobeStore(db)


def test_bad_metadata_json_does_not_crash(manager):
    item = manager.add_item(solid((5, 5, 5)), category="shoes")
    import sqlite3

    with sqlite3.connect(manager.settings.db_path) as conn:
        conn.execute("UPDATE items SET metadata_json = '{broken' WHERE id = ?", (item.id,))
    loaded = manager.get_item(item.id)
    assert "_metadata_error" in loaded.metadata
