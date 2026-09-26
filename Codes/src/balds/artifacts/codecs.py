"""Path-based serialization codecs (the only place that touches torch/np IO).

Each codec knows how to ``load(path) -> obj`` and ``dump(obj, path)``. They
operate on filesystem paths (not in-memory bytes) so multi-hundred-MB tensors
never round-trip through a ``BytesIO``. ``torch`` is imported lazily, so the
store layer remains importable without a GPU stack for the path/manifest logic.

Codecs are registered in :data:`balds.schema.registry.CODECS` and selected per
:class:`~balds.schema.artifact.ArtifactKind` by :func:`codec_for`.
"""
from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from typing import Any

from balds.schema.artifact import ArtifactKind
from balds.schema.registry import CODECS


class Codec(ABC):
    """Bidirectional path serializer for one on-disk format."""

    suffix: str

    @abstractmethod
    def load(self, path: str) -> Any: ...

    @abstractmethod
    def dump(self, obj: Any, path: str) -> None: ...


class PtCodec(Codec):
    """torch ``.pt`` tensors / state dicts (legacy-compatible)."""

    suffix = ".pt"

    def load(self, path: str) -> Any:
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)

    def dump(self, obj: Any, path: str) -> None:
        import torch
        torch.save(obj, path)


class NpyCodec(Codec):
    """NumPy ``.npy`` arrays."""

    suffix = ".npy"

    def load(self, path: str) -> Any:
        import numpy as np
        return np.load(path, allow_pickle=False)

    def dump(self, obj: Any, path: str) -> None:
        import numpy as np
        np.save(path, np.asarray(obj))


class NpzCodec(Codec):
    """NumPy ``.npz`` bundles (a dict of named arrays)."""

    suffix = ".npz"

    def load(self, path: str) -> dict:
        import numpy as np
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def dump(self, obj: dict, path: str) -> None:
        import numpy as np
        arrays = {str(k): np.asarray(v) for k, v in obj.items()}
        bad = [k for k, v in arrays.items() if v.dtype.hasobject]
        if bad:
            raise ValueError(f"NPZ object arrays are forbidden (keys {bad})")
        np.savez(path, **arrays)


class JsonCodec(Codec):
    """UTF-8 JSON (metadata, loss history, LDS result tables)."""

    suffix = ".json"

    def load(self, path: str) -> Any:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def dump(self, obj: Any, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, allow_nan=False)


class PickleCodec(Codec):
    """Pickle ``.pkl`` (legacy subset masks)."""

    suffix = ".pkl"

    def load(self, path: str) -> Any:
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def dump(self, obj: Any, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


class BytesCodec(Codec):
    """Opaque bytes (rendered figures). The app renders to memory; only the
    store touches the disk."""

    suffix = ".png"

    def load(self, path: str) -> bytes:
        with open(path, "rb") as fh:
            return fh.read()

    def dump(self, obj: bytes, path: str) -> None:
        with open(path, "wb") as fh:
            fh.write(bytes(obj))


class TextCodec(Codec):
    """UTF-8 plain text (human-editable CSV evidence)."""

    suffix = ".csv"

    def load(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            return fh.read()

    def dump(self, obj: str, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(str(obj))


# Register stateless codec *instances* (consumers call .load/.dump directly).
CODECS.add("png", BytesCodec())
CODECS.add("pt", PtCodec())
CODECS.add("npy", NpyCodec())
CODECS.add("npz", NpzCodec())
CODECS.add("json", JsonCodec())
CODECS.add("pickle", PickleCodec())
CODECS.add("csv", TextCodec())


# ArtifactKind -> codec name. Kept legacy-format for parity; safetensors adoption
# for features/scores is a later opt-in change behind a re-snapshot.
_KIND_CODEC: dict[ArtifactKind, str] = {
    ArtifactKind.CHECKPOINT: "pt",
    ArtifactKind.SUBSET_CHECKPOINT: "pt",
    ArtifactKind.GENERATION: "pt",
    ArtifactKind.GEN_TRAJECTORY: "pt",
    ArtifactKind.GEN_POOL: "pt",
    ArtifactKind.TRAIN_FEATURES: "pt",
    ArtifactKind.REPEAT_FEATURES: "pt",
    ArtifactKind.QUERY_FEATURES: "pt",
    ArtifactKind.LATENTS: "pt",
    ArtifactKind.PROMPT_EMBEDS: "pt",
    ArtifactKind.INJECT_META: "json",
    ArtifactKind.INJECT_DETECTOR: "pt",
    ArtifactKind.INJECT_CANDIDATES: "json",
    ArtifactKind.INJECT_REVIEW_SHEET: "png",
    ArtifactKind.INJECT_REVIEW_TABLE: "csv",
    ArtifactKind.INJECT_QUERIES: "pt",
    ArtifactKind.INJECT_MINED: "pt",
    ArtifactKind.INJECT_RESULT: "json",
    ArtifactKind.INJECT_CF_MASK: "npy",
    ArtifactKind.INJECT_CF_META: "json",
    ArtifactKind.INJECT_CF_CHECKPOINT: "pt",
    ArtifactKind.EKFAC_FACTORS: "pt",
    ArtifactKind.ERROR_WEIGHT: "npy",
    ArtifactKind.GT_MATRIX: "npy",
    ArtifactKind.GT_LOSSES: "npy",
    ArtifactKind.GT_LOSSROW: "npy",
    ArtifactKind.SCORES: "npy",
    ArtifactKind.SCORES_LAMBDA: "npy",
    ArtifactKind.REPEAT_SCORES: "npz",
    ArtifactKind.SCORES_BLOCK: "npz",
    ArtifactKind.REPEAT_SCORES_BLOCK: "npz",
    ArtifactKind.FEATURE_META: "json",
    ArtifactKind.LOSS_HISTORY: "json",
    ArtifactKind.LDS_RESULT: "json",
    ArtifactKind.SCORES_META: "json",
    ArtifactKind.CF_MASK: "npy",
    ArtifactKind.CF_META: "json",
    ArtifactKind.CF_CHECKPOINT: "pt",
    ArtifactKind.CF_GENERATION: "pt",
    ArtifactKind.CF_ANALYSIS: "json",
    ArtifactKind.CF_FIGURE: "png",
    ArtifactKind.SUBSET_MASKS: "pickle",
}


def codec_for(kind: ArtifactKind) -> Codec:
    """Return the codec instance for an artifact kind."""
    return CODECS.get(_KIND_CODEC[kind])


def codec_name_for(kind: ArtifactKind) -> str:
    """Return the registered codec name for an artifact kind (for manifest records)."""
    return _KIND_CODEC[kind]
