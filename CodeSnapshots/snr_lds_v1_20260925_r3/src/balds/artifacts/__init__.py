"""Data-Access layer: content-addressed, multi-backend artifact store.

Importing this package registers the built-in codecs and backends (side-effect
registration into :data:`balds.schema.registry`). It stays importable without a
GPU stack — torch is only touched lazily inside the ``pt`` codec.
"""
from __future__ import annotations

from . import addressing  # noqa: F401  (path conventions)
from . import codecs  # noqa: F401  (registers npy/pt/json/pickle/npz codecs)
from .backends.local import LocalBackend
from .content_store import ArtifactStore, sha256_file
from .manifest import ManifestDB
from .source import load_structured

__all__ = [
    "addressing",
    "ArtifactStore",
    "ManifestDB",
    "LocalBackend",
    "sha256_file",
    "load_structured",
]
