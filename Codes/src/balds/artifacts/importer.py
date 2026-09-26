"""Backfill the manifest with provenance for pre-existing (legacy) artifacts.

The store can already *fetch* legacy files by their human-mirror path without any
manifest entry. This importer additionally records a content hash for files that
are already present locally, registered under ``code_version="legacy"`` so their
provenance is unknown-but-tracked and integrity checks apply going forward.
"""
from __future__ import annotations

import os

from balds.schema.artifact import ManifestEntry
from .content_store import _now, sha256_file
from .manifest import ManifestDB


def import_local_tree(local_root: str, manifest: ManifestDB) -> int:
    """Register every file under ``local_root`` (except the CAS + the DB) as legacy.

    Returns the number of files registered.
    """
    n = 0
    for dirpath, _dirs, files in os.walk(local_root):
        if os.sep + "cas" in os.sep + os.path.relpath(dirpath, local_root):
            continue
        for fname in files:
            if fname == "manifest.db":
                continue
            abspath = os.path.join(dirpath, fname)
            relpath = os.path.relpath(abspath, local_root)
            entry = ManifestEntry(
                kind="legacy", spec_digest="legacy", blob_hash=sha256_file(abspath),
                size=os.path.getsize(abspath), codec="raw", code_version="legacy",
                created_at=_now(), upstream=(),
            )
            manifest.put(f"legacy|{relpath}", relpath, entry)
            n += 1
    return n
