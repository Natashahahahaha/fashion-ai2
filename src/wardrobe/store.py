"""SQLite persistence for wardrobe items (images/embeddings live on disk).

A short-lived connection is opened per operation, which keeps the store safe
to use from Streamlit's worker threads without sharing a connection.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.errors import WardrobeStoreError
from src.wardrobe.schemas import WardrobeItem

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id             TEXT PRIMARY KEY,
    category       TEXT NOT NULL,
    image_path     TEXT NOT NULL,
    embedding_path TEXT NOT NULL,
    confidence     REAL,
    created_at     TEXT NOT NULL,
    metadata_json  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_items_category ON items(category);
"""

_COLUMNS = "id, category, image_path, embedding_path, confidence, created_at, metadata_json"


class WardrobeStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
        except sqlite3.Error as exc:
            raise WardrobeStoreError(f"Cannot open wardrobe database {self.db_path}: {exc}") from exc
        try:
            yield conn
            conn.commit()
        except sqlite3.DatabaseError as exc:
            conn.rollback()
            raise WardrobeStoreError(
                f"The wardrobe database at {self.db_path} is unreadable ({exc}). "
                "Move or delete that file to start a fresh wardrobe."
            ) from exc
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise WardrobeStoreError(f"Wardrobe database schema v{version} is newer than this app (v{SCHEMA_VERSION}).")
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @staticmethod
    def _row_to_item(row: tuple[Any, ...]) -> WardrobeItem:
        try:
            metadata = json.loads(row[6] or "{}")
            if not isinstance(metadata, dict):
                metadata = {}
        except json.JSONDecodeError:
            metadata = {"_metadata_error": "metadata_json was not valid JSON"}
        return WardrobeItem(
            id=row[0],
            category=row[1],
            image_path=row[2],
            embedding_path=row[3],
            confidence=row[4],
            created_at=row[5],
            metadata=metadata,
        )

    # ------------------------------------------------------------------ CRUD
    def add_item(self, item: WardrobeItem) -> WardrobeItem:
        if not item.created_at:
            item.created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with self._connect() as conn:
            try:
                conn.execute(
                    f"INSERT INTO items ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        item.id,
                        item.category,
                        item.image_path,
                        item.embedding_path,
                        item.confidence,
                        item.created_at,
                        json.dumps(item.metadata, ensure_ascii=False),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise WardrobeStoreError(f"Item {item.id} already exists") from exc
        return item

    def get_item(self, item_id: str) -> WardrobeItem | None:
        with self._connect() as conn:
            row = conn.execute(f"SELECT {_COLUMNS} FROM items WHERE id = ?", (item_id,)).fetchone()
        return self._row_to_item(row) if row else None

    def list_items(self, category: str | None = None) -> list[WardrobeItem]:
        with self._connect() as conn:
            if category:
                rows = conn.execute(
                    f"SELECT {_COLUMNS} FROM items WHERE category = ? ORDER BY created_at, id", (category,)
                ).fetchall()
            else:
                rows = conn.execute(f"SELECT {_COLUMNS} FROM items ORDER BY created_at, id").fetchall()
        return [self._row_to_item(r) for r in rows]

    def search_items(
        self,
        category: str | None = None,
        label: str | None = None,
        color: str | None = None,
        style: str | None = None,
        text: str | None = None,
    ) -> list[WardrobeItem]:
        """Filter items by exact category and case-insensitive label/colour/style/free text."""
        items = self.list_items(category)

        def keep(it: WardrobeItem) -> bool:
            md = it.metadata
            if label and label.lower() != str(md.get("label", "")).lower():
                return False
            if color:
                names = {c.get("name", "").lower() for c in md.get("dominant_colors") or []}
                if color.lower() not in names and color.lower() != str(md.get("color", "")).lower():
                    return False
            if style and style.lower() != str(md.get("style", "")).lower():
                return False
            if text:
                hay = " ".join(str(v) for v in (it.category, md.get("label"), md.get("color"), md.get("style"))).lower()
                if text.lower() not in hay:
                    return False
            return True

        return [it for it in items if keep(it)]

    def update_item(self, item_id: str, category: str | None = None, metadata: dict[str, Any] | None = None) -> bool:
        sets, params = [], []
        if category is not None:
            sets.append("category = ?")
            params.append(category)
        if metadata is not None:
            sets.append("metadata_json = ?")
            params.append(json.dumps(metadata, ensure_ascii=False))
        if not sets:
            return False
        with self._connect() as conn:
            cur = conn.execute(f"UPDATE items SET {', '.join(sets)} WHERE id = ?", (*params, item_id))
            return cur.rowcount > 0

    def delete_item(self, item_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
            return cur.rowcount > 0

    def clear(self) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM items")
            return cur.rowcount

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM items").fetchone()[0])

    def count_by_category(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT category, COUNT(*) FROM items GROUP BY category").fetchall()
        return {r[0]: int(r[1]) for r in rows}
