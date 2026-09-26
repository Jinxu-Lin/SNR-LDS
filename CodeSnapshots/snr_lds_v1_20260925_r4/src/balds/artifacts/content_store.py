"""``ArtifactStore`` — the concrete Data-Access implementation.

Combines stable relative addressing, codecs, local files and a SQLite manifest.
Writes carry content hashes and provenance; the artifact layout remains
compatible with the paper experiment library.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from balds.schema.artifact import ArtifactKind, ArtifactStore, ManifestEntry
from balds.schema.runspec import RunSpec
from . import addressing
from .backends.base import Backend
from .backends.local import LocalBackend
from .codecs import codec_for, codec_name_for
from .manifest import ManifestDB


def sha256_file(path: str, *, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (handles multi-hundred-MB tensors)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ArtifactStore(ArtifactStore):
    """Content-tracked local artifact store."""

    def __init__(
        self,
        manifest: ManifestDB,
        backends: list[Backend],
        *,
        code_version: str = "dev",
    ) -> None:
        if len(backends) != 1 or not isinstance(backends[0], LocalBackend):
            raise ValueError("exactly one LocalBackend is required")
        self.manifest = manifest
        self.local: LocalBackend = backends[0]
        self.code_version = code_version

    # --- identity -------------------------------------------------------
    @staticmethod
    def _item_key(kind: ArtifactKind, spec: RunSpec, key: dict) -> str:
        ks = ",".join(f"{k}={key[k]}" for k in sorted(key))
        return f"{kind.value}|{spec.digest()}|{ks}"

    # --- ArtifactStore --------------------------------------------------
    def exists(self, kind: ArtifactKind, spec: RunSpec, **key: Any) -> bool:
        """A manifest row alone does not establish that artifact bytes exist."""
        return self.local.has(addressing.relpath(kind, spec, **key))

    def describe(self, kind: ArtifactKind, spec: RunSpec, **key: Any) -> dict:
        """Provenance handle for an artifact: its mirror relpath, its content
        hash (computed now, from the local bytes) and the manifest row if any.
        Consumers embed this in derived artifacts so a result can be traced to
        the exact input matrix it came from (AN-2 removal sets)."""
        rel = addressing.relpath(kind, spec, **key)
        abspath = self.ensure_local(kind, spec, **key)
        entry = self.manifest.get(self._item_key(kind, spec, key))
        current_hash = sha256_file(abspath)
        current_size = os.path.getsize(abspath)
        return {
            "relpath": rel, "sha256": current_hash, "size": current_size,
            "manifest_status": (
                "missing" if entry is None else
                "match" if entry.blob_hash == current_hash and entry.size == current_size
                else "mismatch"
            ),
            "manifest_sha256": entry.blob_hash if entry else None,
            "manifest_size": entry.size if entry else None,
            "manifest_code_version": entry.code_version if entry else None,
            "manifest_created_at": entry.created_at if entry else None,
            "upstream": list(entry.upstream) if entry else [],
        }

    def has_local(self, kind: ArtifactKind, spec: RunSpec, **key: Any) -> bool:
        """Whether the artifact bytes exist locally."""
        return self.local.has(addressing.relpath(kind, spec, **key))

    def ensure_local(self, kind: ArtifactKind, spec: RunSpec, **key: Any) -> str:
        rel = addressing.relpath(kind, spec, **key)
        abspath = self.local.abspath(rel)
        if self.local.has(rel):
            return abspath
        raise FileNotFoundError(f"{rel} not found under {self.local.root}")

    def local_temporary_files(self, kind: ArtifactKind, spec: RunSpec,
                              **key: Any) -> list[str]:
        """Return atomic-write leftovers without exposing paths to app code."""
        rel = addressing.relpath(kind, spec, **key)
        root = os.path.dirname(self.local.abspath(rel))
        if not os.path.isdir(root):
            return []
        found: list[str] = []
        for parent, _dirs, files in os.walk(root):
            for name in files:
                if name.startswith(".tmp-") or ".tmp-" in name or name.endswith(".tmp"):
                    found.append(os.path.relpath(os.path.join(parent, name), root))
        return sorted(found)

    def describe_external(self, relpath: str) -> dict:
        """Describe non-artifact evidence, constrained below the data root."""
        if os.path.isabs(relpath) or ".." in relpath.split("/"):
            raise ValueError("external evidence path must be relative and may not traverse")
        abspath = self.local.abspath(relpath)
        if not os.path.isfile(abspath):
            raise FileNotFoundError(relpath)
        return {"relpath": relpath, "sha256": sha256_file(abspath),
                "size": os.path.getsize(abspath), "manifest_status": "not_applicable"}

    def load_external(self, relpath: str) -> Any:
        if os.path.isabs(relpath) or ".." in relpath.split("/"):
            raise ValueError("external evidence path must be relative and may not traverse")
        abspath = self.local.abspath(relpath)
        if relpath.endswith(".json"):
            with open(abspath, encoding="utf-8") as handle:
                return json.load(handle)
        if relpath.endswith(".npz"):
            with np.load(abspath, allow_pickle=False) as archive:
                return {name: archive[name] for name in archive.files}
        if relpath.endswith(".csv"):
            import csv
            with open(abspath, encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        raise ValueError("external evidence decoder accepts only .json, .npz or .csv")

    def save_external_image(self, relpath: str, image, *, replace: bool = False) -> dict:
        """Atomically materialise a fetched source image below ``data_root``.

        Raw third-party rows are external inputs, not CFA artifacts, so they do
        not receive a manifest identity.  They still use the store's constrained
        path resolution, atomic publication and immediate content description.
        """
        if os.path.isabs(relpath) or ".." in relpath.split("/"):
            raise ValueError("external image path must be relative and may not traverse")
        if not relpath.lower().endswith(".png"):
            raise ValueError("external fetched images are materialised as .png")
        abspath = self.local.abspath(relpath)
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        if os.path.exists(abspath) and not replace:
            return self.describe_external(relpath)
        tmp = f"{abspath}.tmp-{os.getpid()}-{uuid.uuid4().hex}.png"
        try:
            image.convert("RGB").save(tmp, format="PNG")
            with open(tmp, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp, abspath)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return self.describe_external(relpath)

    def list_external_files(self, rel_dir: str, *, suffix: str = "") -> list[str]:
        if os.path.isabs(rel_dir) or ".." in rel_dir.split("/"):
            raise ValueError("external directory must be relative and may not traverse")
        root = self.local.abspath(rel_dir)
        if not os.path.isdir(root):
            return []
        return sorted(name for name in os.listdir(root)
                      if os.path.isfile(os.path.join(root, name)) and name.endswith(suffix))

    @staticmethod
    def _validate_external_dir(rel_dir: str) -> None:
        if os.path.isabs(rel_dir) or ".." in rel_dir.split("/"):
            raise ValueError("external directory must be relative and may not traverse")

    def external_directory(self, rel_dir: str) -> str:
        """Return a materialised local directory below ``data_root``."""
        self._validate_external_dir(rel_dir)
        root = self.local.abspath(rel_dir)
        os.makedirs(root, exist_ok=True)
        return root

    def list_external_tree_files(self, rel_dir: str, *, suffix: str = "") -> list[str]:
        """List local external files recursively in path order."""
        self._validate_external_dir(rel_dir)
        root = self.local.abspath(rel_dir)
        if not os.path.isdir(root):
            return []
        return sorted(
            os.path.join(parent, name)
            for parent, _dirs, files in os.walk(root)
            for name in files
            if name.endswith(suffix)
        )

    def remove_external_scratch(self, rel_dir: str) -> bool:
        """Remove one explicitly named scratch tree, never raw data or shared caches."""
        self._validate_external_dir(rel_dir)
        if not rel_dir.startswith("_scratch/"):
            raise ValueError("external scratch removal is restricted to _scratch/")
        root = self.local.abspath(rel_dir)
        if not os.path.isdir(root):
            return False
        shutil.rmtree(root)
        return True

    def load(self, kind: ArtifactKind, spec: RunSpec, **key: Any) -> Any:
        abspath = self.ensure_local(kind, spec, **key)
        return codec_for(kind).load(abspath)

    def save(
        self,
        kind: ArtifactKind,
        spec: RunSpec,
        obj: Any,
        *,
        upstream: tuple[str, ...] = (),
        **key: Any,
    ) -> ManifestEntry:
        rel = addressing.relpath(kind, spec, **key)
        abspath = self.local.abspath(rel)
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        codec_for(kind).dump(obj, abspath)
        entry = ManifestEntry(
            kind=kind.value, spec_digest=spec.digest(), blob_hash=sha256_file(abspath),
            size=os.path.getsize(abspath), codec=codec_name_for(kind),
            code_version=self.code_version, created_at=_now(), upstream=tuple(upstream),
        )
        self.manifest.put(self._item_key(kind, spec, key), rel, entry)
        return entry

    def save_atomic(
        self,
        kind: ArtifactKind,
        spec: RunSpec,
        obj: Any,
        *,
        upstream: tuple[str, ...] = (),
        validator: Callable[[Any], None] | None = None,
        replace: bool = False,
        **key: Any,
    ) -> ManifestEntry:
        """Publish only fully encoded, decodable bytes, then commit lineage.

        The temporary has the codec suffix at its end because ``numpy.save`` /
        ``savez`` otherwise append a second suffix and leave the intended path
        empty.  A byte-identical final without a manifest row is the expected
        recovery case after a crash between ``os.replace`` and ``manifest.put``.
        """
        rel = addressing.relpath(kind, spec, **key)
        abspath = self.local.abspath(rel)
        parent = os.path.dirname(abspath)
        os.makedirs(parent, exist_ok=True)
        codec = codec_for(kind)
        item_key = self._item_key(kind, spec, key)
        prior_entry = self.manifest.get(item_key)
        if prior_entry is not None and os.path.isfile(abspath):
            current_hash = sha256_file(abspath)
            current_size = os.path.getsize(abspath)
            if (prior_entry.blob_hash != current_hash or prior_entry.size != current_size):
                raise IOError(
                    f"refusing atomic publication for {rel}: existing final bytes conflict "
                    "with their manifest row"
                )
        tmp = f"{abspath}.tmp-{os.getpid()}-{uuid.uuid4().hex}{codec.suffix}"
        try:
            codec.dump(obj, tmp)
            decoded = codec.load(tmp)
            if validator is not None:
                validator(decoded)
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
            blob_hash = sha256_file(tmp)
            size = os.path.getsize(tmp)

            if (prior_entry is not None and not os.path.isfile(abspath) and
                    prior_entry.blob_hash != blob_hash and not replace):
                raise FileExistsError(
                    f"refusing to reuse manifested identity {rel}: prior hash "
                    f"{prior_entry.blob_hash} != candidate {blob_hash}"
                )

            if os.path.isfile(abspath):
                current_hash = sha256_file(abspath)
                if current_hash != blob_hash and not replace:
                    raise FileExistsError(
                        f"refusing to replace {rel}: existing hash {current_hash} "
                        f"!= candidate {blob_hash}"
                    )
                if current_hash == blob_hash:
                    current = codec.load(abspath)
                    if validator is not None:
                        validator(current)
                    os.remove(tmp)
                    tmp = ""
                else:
                    os.replace(tmp, abspath)
                    tmp = ""
            else:
                os.replace(tmp, abspath)
                tmp = ""

            # Persist the directory entry before advertising it in SQLite.
            try:
                fd = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError:
                # Some remote/virtual filesystems do not support directory
                # fsync.  Atomic rename still holds; validation already passed.
                pass

            entry = ManifestEntry(
                kind=kind.value, spec_digest=spec.digest(), blob_hash=blob_hash,
                size=size, codec=codec_name_for(kind), code_version=self.code_version,
                created_at=_now(), upstream=tuple(upstream),
            )
            self.manifest.put(item_key, rel, entry)
            return entry
        finally:
            if tmp and os.path.isfile(tmp):
                os.remove(tmp)

    def discard(self, kind: ArtifactKind, spec: RunSpec, **key: Any) -> bool:
        """Delete this machine's copy of an intermediate; ``True`` if it was there.

        This is only correct for artifacts whose content is fully captured by an
        already-saved downstream artifact — today that is exactly the
        per-query-block score partials, dropped after the whole matrix is filed.
        The manifest row is left in place: it is a lineage record, and
        :meth:`exists` never accepts a row as proof of bytes.
        """
        rel = addressing.relpath(kind, spec, **key)
        if not self.local.has(rel):
            return False
        self.local.remove(rel)
        return True

    def local_blocks(self, kind: ArtifactKind, spec: RunSpec, **key: Any) -> list[dict]:
        """``[{"q0", "q1"[, "r0", "r1"]}, ...]`` for every block partial of
        ``kind`` present on THIS machine for this cell, ordered by query range
        then train-row range (a whole-rows partial sorts before the row-sharded
        partials of the same query block).

        Each dict is a ready-to-use store key. Assemblers discover row partials
        without parsing storage paths themselves.
        """
        d = self.local.abspath(addressing.block_dir(kind, spec, **key))
        if not os.path.isdir(d):
            return []
        found = [k for k in (addressing.parse_block_key(kind, n) for n in os.listdir(d))
                 if k is not None]
        return sorted(found, key=lambda k: (k["q0"], k["q1"],
                                            k.get("r0", -1), k.get("r1", -1)))

    def verify_local(self) -> list[str]:
        """Re-hash every locally-present manifested artifact; return mismatched relpaths."""
        bad: list[str] = []
        for relpath, blob_hash in self.manifest.iter_records():
            abspath = self.local.abspath(relpath)
            if os.path.isfile(abspath) and sha256_file(abspath) != blob_hash:
                bad.append(relpath)
        return bad
