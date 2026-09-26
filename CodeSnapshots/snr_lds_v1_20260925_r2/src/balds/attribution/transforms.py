"""Score transforms applied to a raw ``(N, Q)`` kernel/score matrix.

Unified behind :class:`~balds.schema.estimator.ScoreTransform`
(``__call__(raw, e_n, *, params) -> (N, Q)``):

* **shape transforms** — ``identity`` (every production method), plus the
  terminal transforms the squaring-mystery experiments apply at the end of the
  pipeline (``square``, ``signsqu``, ``abs``);
* the shrinkage-curve primitives ``shrinkage_curve`` / ``sigma_from_repeats``
  used by :mod:`balds.attribution.shrinkage`.
"""
from __future__ import annotations

import numpy as np

from balds.schema.estimator import ScoreTransform
from balds.schema.registry import TRANSFORMS


def _shape(scores: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return scores
    if name == "square":
        return scores ** 2
    if name == "abs":
        return np.abs(scores)
    if name == "signsqu":
        return scores * np.abs(scores)
    raise ValueError(f"unknown shape transform '{name}'")


class _ShapeTransform(ScoreTransform):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, raw, e_n, *, params):
        return _shape(raw, self.name)


for _n in ("identity", "square", "abs", "signsqu"):
    TRANSFORMS.add(_n, _ShapeTransform(_n))


# --- shrinkage-curve primitives (verbatim; the main-table artifacts were produced by them) ---

def shrinkage_curve(tau: np.ndarray, sigma: np.ndarray, gamma: float) -> np.ndarray:
    r"""The paper's weight curve :math:`f_c(\tau)=\tau^3/(\tau^2+\gamma\hat\sigma^2)`.

    Reads as a soft gate on |τ| relative to the noise scale: far above the knee
    it passes τ through (``f_c → τ``), far below it suppresses cubically
    (``f_c ≈ τ³/(γσ̂²)``), so scores indistinguishable from measurement noise are
    driven to zero while the head keeps its magnitude ordering. ``γ`` sets where
    the knee sits in units of σ̂²; ``γ → ∞`` degenerates to the cube transform
    and ``γ → 0`` to the identity, which is why γ must be frozen rather than
    tuned per seed (a single-seed scan runs off to the cube end — observed in
    ``_Data/reports/tweedie_probe_2026-08-07``).

    ``sigma`` broadcasts against ``tau``: a per-query row vector ``(1, Q)`` gives
    each query its own knee, which is the calibrated form — score scale varies
    query to query, so one global σ̂ would over-shrink some and under-shrink others.
    Equivariant to a positive rescaling of (τ, σ̂) jointly, hence LDS-safe.
    """
    denom = tau ** 2 + gamma * (sigma ** 2)
    return np.where(denom > 0, tau ** 3 / np.maximum(denom, 1e-30), 0.0)


def sigma_from_repeats(repeat_scores: list[np.ndarray]) -> np.ndarray:
    """Per-query noise scale σ̂ from R repeat-featurized score matrices.

    Each element is the ``(M, Q)`` score matrix of the same M training samples
    re-featurized with fresh noise through the same kernel. The per-entry spread
    across repeats is the measurement noise of a single score; pooling it over
    the M samples gives one σ̂ per query, stable at R=4 where a per-entry
    estimate would not be.

    Pools in quadrature (root-mean-square of the per-sample standard deviations)
    rather than averaging them, since variances are what add.

    The estimate MUST come from repeats under an identical protocol. Feeding it
    spreads that also contain protocol differences reads real signal as noise and
    over-shrinks — measured at 0.62 → 0.28 in the pilot.
    """
    if len(repeat_scores) < 2:
        raise ValueError(f"σ̂ needs at least 2 repeats, got {len(repeat_scores)}")
    stack = np.stack(repeat_scores, axis=0)                  # (R, M, Q)
    per_sample = stack.std(axis=0, ddof=1)                   # (M, Q)
    return np.sqrt((per_sample ** 2).mean(axis=0, keepdims=True))   # (1, Q)
