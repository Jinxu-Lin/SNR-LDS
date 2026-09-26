"""Horizontal-axis (direct-output attribution) abstractions.

These seams collapse the legacy ~46-name ``if/elif`` zoo into composable units:
a :class:`ScoreMethod` is ``weight ∘ kernel ∘ transform``. The critical fix over
a naive refactor is that :class:`ScoreTransform` receives ``(raw, e_n, params)``
— so the ``das1smart`` adaptive-tau family (which needs the error term *and*
tunable parameters inside the transform) is expressible without editing the seam.

``numpy``/``torch`` appear only in annotations (deferred), keeping contracts light.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch


@dataclass
class FeatureSet:
    """Output of featurization for one (dataset, seed, T, protocol).

    ``error`` (the per-sample weight ``e_n``) is produced *eagerly* here, not
    computed lazily during scoring as in the legacy code.
    """

    grads: "torch.Tensor"                       # (N, proj_dim) projected gradients g_n
    error: "np.ndarray | None"                  # (N,) e_n; None for raw-kernel methods (das/dtrak/trak/l1norm/l2norm)
    feat_method: str                            # producing featurizer, e.g. "das", "dtrak", "trak"
    meta: dict[str, Any] = field(default_factory=dict)
    # (R, M, proj_dim): the same M training samples re-featurized R times with
    # fresh noise under an identical protocol. Only the shrinkage layer needs it
    # (σ̂ = the spread of a score across repeats); None for every other method.
    repeats: "torch.Tensor | None" = None
    # Per-training-checkpoint features, oldest first, for the TracIn family
    # (TracInCP/GAS average a similarity over checkpoints instead of using the
    # final model alone). Both sides are carried because the average is over
    # matched pairs — train and query features from the SAME checkpoint. None
    # for every other method.
    ckpt_grads: "list[torch.Tensor] | None" = None
    ckpt_query: "list[torch.Tensor] | None" = None
    ckpt_steps: "tuple[int | None, ...]" = ()


class FeatureExtractor(ABC):
    """Computes per-sample projected gradient features for a model + data."""

    @abstractmethod
    def extract(
        self, model: Any, subset: Any, process: Any, *, T: int, proj_seed: int
    ) -> FeatureSet:
        ...


class ScoreTransform(ABC):
    """Maps a raw kernel score matrix to a final score matrix.

    Parameters of ``__call__``
    --------------------------
    raw:
        ``(N, Q)`` weighted kernel scores (the per-sample weight is already
        applied by the :class:`ScoreMethod` before the transform).
    e_n:
        ``(N,)`` per-sample error term, or ``None``; available to transforms
        that need it (the ``smart`` family).
    params:
        Transform-specific keyword parameters (e.g. ``tau_min``/``tau_max``).
    """

    @abstractmethod
    def __call__(self, raw: "np.ndarray", e_n: "np.ndarray | None", *, params: dict) -> "np.ndarray":
        ...


class ScoreMethod(ABC):
    """A named attribution method: ``weight ∘ kernel ∘ transform``.

    Attributes
    ----------
    name:
        Registered method name (e.g. ``"das1squ"``).
    feat_method:
        Which :class:`FeatureSet` (featurizer/T) this method consumes; lets many
        methods share one ``train_features.pt`` (e.g. all ``das1*`` reuse
        ``dtrak`` features).
    needs_error_weight:
        Declarative flag; the composition root fetches ``e_n`` only when ``True``.
    """

    name: str
    feat_method: str
    needs_error_weight: bool = False

    @abstractmethod
    def score(self, feats: FeatureSet, query_grads: "torch.Tensor", lam: float) -> "np.ndarray":
        """Return the ``(N, Q)`` score matrix for regularisation strength ``lam``."""
