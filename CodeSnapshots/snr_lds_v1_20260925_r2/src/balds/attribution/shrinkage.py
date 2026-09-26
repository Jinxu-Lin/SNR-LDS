"""The shrinkage layer as a first-class object (Tweedie-FMAS's Algorithm 2).

Before this module the curve was welded into ``ShrinkageKernelMethod`` and
therefore only reachable through the ``trak_kernel`` path. AB-5 ("baseline x
denoising-layer gain") needs the SAME layer applied to any ``(N, Q)`` score
matrix with a sigma-hat from ANY measurement path — projected-kernel repeats
(``repeat_features``) or the EK-FAC eigenbasis repeats (``repeat_scores``) —
under exactly the same arithmetic, or the (A)/(B) verdict in the task brief is
polluted by implementation drift rather than by the methods.

The two numeric primitives stay where they were (``transforms.shrinkage_curve``
and ``transforms.sigma_from_repeats``; re-exported here) so the call path of the
paper's method is unchanged to the byte — the main table is already filled from
it. Nothing here touches eval/store: selection is injected as a callable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .transforms import shrinkage_curve, sigma_from_repeats  # noqa: F401  (re-export)

__all__ = ["ShrinkageLayer", "shrinkage_curve", "sigma_from_repeats",
           "snr_stats", "select_after_shrinkage"]


@dataclass(frozen=True)
class ShrinkageLayer:
    """``f_c(τ) = τ³ / (τ² + γ·σ̂²)`` with the knee γ carried per instance.

    γ is NOT read from any registry or global: callers resolve it per call
    (``app.config.resolve_gamma``) and build the layer with it. Registered
    method singletons therefore never hold a γ that a later call could inherit
    by accident (task brief §5.1).
    """

    gamma: float

    def apply(self, raw: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Shape one ``(N, Q)`` matrix with its ``(1, Q)`` per-query σ̂.

        Same expression, same dtype flow as the original in-method code:
        float32 in, float32 out, so a matrix produced through the layer is
        ``np.array_equal`` to what ``ShrinkageKernelMethod`` used to emit.
        """
        return shrinkage_curve(raw, sigma, float(self.gamma)).astype(np.float32)

    def sweep(self, raw_by_key: Mapping, sigma_by_key: Mapping,
              gammas: Sequence[float]) -> dict[float, dict]:
        """``{gamma: {key: shaped}}`` over a grid of knees.

        ``key`` is whatever indexes the raw matrices — the kernel λ for the
        projected methods, the EK-FAC damping for the curvature path. σ̂ is
        keyed the same way because it depends on λ/damping (task brief §5.2):
        one global σ̂ would compare knees at the wrong noise scale.
        """
        missing = [k for k in raw_by_key if k not in sigma_by_key]
        if missing:
            raise KeyError(f"no sigma-hat for keys {missing}")
        return {float(g): {k: ShrinkageLayer(float(g)).apply(raw, sigma_by_key[k])
                           for k, raw in raw_by_key.items()}
                for g in gammas}


def snr_stats(raw: np.ndarray, sigma: np.ndarray) -> dict:
    """Reportable noise figures for one score matrix (task brief C2).

    ``SNR_q = mean_i |τ_iq| / σ̂_q`` averaged over queries; ``sigma_mean`` is
    the query-average of σ̂. A method whose gradients are already averaged over
    many MC draws (EK-FAC, mc=250) arrives with a small σ̂ and a large SNR — the
    number that separates "the layer does nothing for it" from "it has already
    averaged itself".
    """
    sig = np.asarray(sigma, dtype=np.float64).reshape(-1)
    absmean = np.abs(np.asarray(raw, dtype=np.float64)).mean(axis=0)   # (Q,)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_q = np.where(sig > 0, absmean / sig, np.nan)
    return {"sigma_mean": float(np.mean(sig)),
            "snr": float(np.nanmean(snr_q)) if np.isfinite(snr_q).any() else float("nan")}


def select_after_shrinkage(layer: ShrinkageLayer, raw_by_key: Mapping,
                           sigma_by_key: Mapping, keys: Sequence,
                           select: Callable):
    """Shape every candidate, THEN let the selector pick the key.

    ``select(shaped_by_key, keys) -> choice`` is the pipeline's λ/damping
    selector with the ground truth already bound. Selecting on the raw
    matrices and shaping afterwards would hand a baseline a damping tuned for
    the wrong objective — the P0 failure mode named in the task brief (B3).
    Returns ``(shaped_by_key, choice)``.
    """
    shaped = layer.sweep(raw_by_key, sigma_by_key, [layer.gamma])[float(layer.gamma)]
    return shaped, select(shaped, list(keys))
