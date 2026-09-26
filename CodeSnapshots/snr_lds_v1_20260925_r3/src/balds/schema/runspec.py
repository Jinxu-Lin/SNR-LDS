"""``RunSpec`` — the typed identity of everything the pipeline produces.

A frozen, hashable description of *what* produced an artifact. It is the single
source of truth for artifact identity and replaces every hand-rolled
``os.path.join(..., f"seed_{seed}", ...)`` string and the dead ``Paths`` class
in the legacy code.

Two uses:

* :meth:`RunSpec.digest` gives a deterministic content-independent identity used
  as the manifest key ``(ArtifactKind, digest)`` -> ``blob_hash``. Because the
  digest includes ``code_version``, re-running with changed code forks a new
  artifact instead of silently overwriting — essential for journal-grade
  reproducibility.
* :func:`balds.artifacts.addressing.relpath` turns a ``RunSpec`` into the *human*
  mirror path (identical to the legacy ``_Data/`` layout), so existing artifacts
  are addressable with zero migration.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Identity of a pipeline artifact.

    Attributes
    ----------
    dataset:
        Registered dataset name, e.g. ``"cifar2"``.
    process:
        Generative process, ``"cfm"`` (flow matching) or ``"ddpm"`` (diffusion);
        the two are one parameter of the same pipeline.
    conditional:
        Whether the model is class-conditional (legacy checkpoints use the
        ``cfm_cond`` directory segment).
    seed:
        Training seed of the full model (the subset masks use their own fixed
        seed, independent of this).
    method:
        Registered score-method name (``None`` for non-score artifacts such as
        checkpoints or features).
    query_type:
        ``"gen"`` (model-generated queries), ``"val"`` (held-out test images),
        or ``"inject"`` (human-reviewed contamination queries; no LDS ground truth).
    code_version:
        Short git SHA stamped by the composition root; the reproducibility
        anchor folded into :meth:`digest`.
    """

    dataset: str
    process: str = "cfm"
    conditional: bool = True
    seed: int = 42
    method: str | None = None
    query_type: str = "gen"
    code_version: str = "dev"

    def __post_init__(self) -> None:
        if self.query_type not in {"gen", "val", "inject"}:
            raise ValueError(f"query_type must be gen, val or inject, got {self.query_type!r}")

    def digest(self) -> str:
        """Stable 16-hex-char content-independent identity (blake2b)."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()

    def with_(self, **changes: object) -> RunSpec:
        """Return a copy with the given fields replaced (frozen-friendly)."""
        return replace(self, **changes)

    @property
    def is_val(self) -> bool:
        return self.query_type == "val"
