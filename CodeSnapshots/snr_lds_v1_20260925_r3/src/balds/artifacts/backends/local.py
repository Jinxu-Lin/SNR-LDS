"""Local-filesystem backend: the on-disk cache / human-mirror tree root."""
from __future__ import annotations

import hashlib
import os
import shutil

from balds.schema.registry import BACKENDS
from .base import Backend


@BACKENDS.register("local")
class LocalBackend(Backend):
    """Files under a local ``root`` directory.

    Doubles as the materialisation target: ``abspath`` is the real path other
    code (and legacy tools) can read directly.
    """

    name = "local"

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)

    def abspath(self, relpath: str) -> str:
        return os.path.join(self.root, relpath)

    def has(self, relpath: str) -> bool:
        return os.path.isfile(self.abspath(relpath))

    def fetch(self, relpath: str, dest_abspath: str) -> None:
        src = self.abspath(relpath)
        os.makedirs(os.path.dirname(dest_abspath), exist_ok=True)
        if os.path.abspath(src) != os.path.abspath(dest_abspath):
            shutil.copyfile(src, dest_abspath)

    def send(self, src_abspath: str, relpath: str) -> None:
        dst = self.abspath(relpath)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.abspath(src_abspath) != os.path.abspath(dst):
            shutil.copyfile(src_abspath, dst)

    def sha256(self, relpath: str) -> str | None:
        path = self.abspath(relpath)
        if not os.path.isfile(path):
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def remove(self, relpath: str) -> None:
        """Delete this backend's copy (used only by verified pruning)."""
        path = self.abspath(relpath)
        if os.path.isfile(path):
            os.remove(path)
