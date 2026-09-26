"""SQLite manifest: artifact identity -> content hash + provenance/lineage.

One ``manifest.db`` (itself a replicable artifact) records, for every artifact
the pipeline produces, its content hash, codec, the ``code_version`` that
produced it, and the content hashes of its ``upstream`` inputs. This gives
journal-grade traceability: any score blob can be walked back to the exact
feature + ground-truth blobs (and thus the dataset/seed/method/protocol) behind it.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Optional

from balds.schema.artifact import ManifestEntry

_SCHEMA = """
CREATE TABLE IF NOT EXISTS manifest (
    item_key     TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    spec_digest  TEXT NOT NULL,
    relpath      TEXT NOT NULL,
    blob_hash    TEXT NOT NULL,
    size         INTEGER NOT NULL,
    codec        TEXT NOT NULL,
    code_version TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    upstream     TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_blob ON manifest(blob_hash);
CREATE INDEX IF NOT EXISTS idx_relpath ON manifest(relpath);
"""


class ManifestDB:
    """Thin typed wrapper over a single SQLite file."""

    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def put(self, item_key: str, relpath: str, entry: ManifestEntry) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO manifest "
            "(item_key, kind, spec_digest, relpath, blob_hash, size, codec, "
            " code_version, created_at, upstream) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (item_key, entry.kind, entry.spec_digest, relpath, entry.blob_hash,
             entry.size, entry.codec, entry.code_version, entry.created_at,
             json.dumps(list(entry.upstream))),
        )
        self._conn.commit()

    def get(self, item_key: str) -> Optional[ManifestEntry]:
        row = self._conn.execute(
            "SELECT * FROM manifest WHERE item_key = ?", (item_key,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def by_relpath(self, relpath: str) -> Optional[ManifestEntry]:
        row = self._conn.execute(
            "SELECT * FROM manifest WHERE relpath = ? ORDER BY created_at DESC LIMIT 1",
            (relpath,),
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def all_blob_hashes(self) -> set[str]:
        rows = self._conn.execute("SELECT DISTINCT blob_hash FROM manifest").fetchall()
        return {r["blob_hash"] for r in rows}

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM manifest").fetchone()["n"]

    def iter_records(self):
        """Yield ``(relpath, blob_hash)`` for every manifest row."""
        for row in self._conn.execute("SELECT relpath, blob_hash FROM manifest"):
            yield row["relpath"], row["blob_hash"]

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> ManifestEntry:
        return ManifestEntry(
            kind=row["kind"], spec_digest=row["spec_digest"], blob_hash=row["blob_hash"],
            size=row["size"], codec=row["codec"], code_version=row["code_version"],
            created_at=row["created_at"], upstream=tuple(json.loads(row["upstream"])),
        )

    def close(self) -> None:
        self._conn.close()
