"""Isolated E3c files; no writes to legacy artifacts or the shared manifest."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time
import uuid

from .addressing import relpath
from .codecs import JsonCodec, NpyCodec, PtCodec, TextCodec, codec_for


def read_json(path):
    return JsonCodec().load(str(path))


def atomic(path, value, codec=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = codec or JsonCodec()
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}{codec.suffix}")
    try:
        codec.dump(value, str(tmp))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_text(path, text):
    atomic(path, text, TextCodec())


class ReadOnlyInputs:
    def __init__(self, root):
        self.root = Path(root)

    def load(self, kind, spec, **key):
        return codec_for(kind).load(str(self.root / relpath(kind, spec, **key)))


def input_inventory(root, paths):
    return [{"path": str(p), "bytes": (Path(root) / p).stat().st_size} for p in paths]


@contextmanager
def exclusive(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "worker.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another worker owns {root}") from exc
        yield


class RepeatFiles:
    def __init__(self, root, identity):
        self.root, self.identity = Path(root), identity
        self.io_seconds = 0.0
        path = self.root / "identity.json"
        if path.exists():
            if read_json(path) != identity:
                raise ValueError(f"effective configuration / repeat identity conflict: {path}")
        else:
            if self.root.exists() and any(p.name != "worker.lock" for p in self.root.iterdir()):
                raise ValueError("outputs without identity cannot be resumed")
            atomic(path, identity)

    def get(self, name, ids):
        path = self.root / (name + ".pt")
        if not path.exists():
            return None
        started = time.monotonic()
        item = PtCodec().load(str(path))
        self.io_seconds += time.monotonic() - started
        if item["identity"] != self.identity or item["ids"] != ids:
            raise ValueError(f"mixed repeat / row / query identity in {path}")
        return item["value"]

    def put(self, name, value, ids):
        started = time.monotonic()
        atomic(self.root / (name + ".pt"),
               {"identity": self.identity, "ids": ids, "value": value}, PtCodec())
        self.io_seconds += time.monotonic() - started

    def score(self, method, track, value, ids):
        started = time.monotonic()
        path = self.root / method / track / "scores.npy"
        atomic(path, value, NpyCodec())
        atomic(path.with_suffix(".json"),
               {"identity": self.identity, "ids": ids, "shape": list(value.shape),
                "dtype": str(value.dtype)})
        self.io_seconds += time.monotonic() - started


def load_score(root, method, track, identity, ids, shape):
    import numpy as np
    path = Path(root) / method / track / "scores.npy"
    meta_path = path.with_suffix(".json")
    if not path.exists() or not meta_path.exists():
        return None
    meta = read_json(meta_path)
    if meta["identity"] != identity or meta["ids"] != ids:
        raise ValueError(f"mixed repeat/config/query identity: {path}")
    value = NpyCodec().load(str(path))
    if value.shape != tuple(shape) or not np.isfinite(value).all() or value.dtype != np.float32:
        raise ValueError(f"invalid score shape/dtype/finite values: {path}")
    return value


def load_tensor(path):
    return PtCodec().load(str(path))


def save_tensor(path, value):
    atomic(path, value, PtCodec())
