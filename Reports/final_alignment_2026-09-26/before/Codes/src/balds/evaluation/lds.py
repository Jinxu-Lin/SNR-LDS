"""Linear Datamodeling Score (LDS) — ground-truth-free attribution evaluation.

Ported from the legacy ``fmas/evaluation.py`` for parity:

* ``predicted_influence(scores, masks)`` — ``ĝ_m(q) = Σ_i (1 - z_m^{(i)}) · τ^{(i)}(q)``;
* ``compute_lds(gt, pred)`` — mean over queries of Spearman(gt[:,q], pred[:,q]).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def spearman(a, b) -> float:
    """Spearman rank correlation (scipy), as a float (0.0 if degenerate/NaN)."""
    corr, _ = stats.spearmanr(np.asarray(a).ravel(), np.asarray(b).ravel())
    return 0.0 if np.isnan(corr) else float(corr)


def predicted_influence(scores: np.ndarray, masks: list, subset_indices) -> np.ndarray:
    """``(M, Q)`` predicted subset losses from per-sample scores and inclusion masks.

    ``scores`` is ``(N, Q)``; ``masks[s]`` is a boolean ``(N,)`` inclusion vector
    (``z_m^{(i)}``); excluded samples ``(1 - z)`` contribute to the held-out loss.
    """
    M = len(subset_indices)
    _, Q = scores.shape
    pred = np.zeros((M, Q))
    for m_idx, s_idx in enumerate(subset_indices):
        excluded = 1.0 - masks[s_idx].astype(np.float64)
        pred[m_idx] = excluded @ scores
    return pred


def support_head_scores(scores: np.ndarray, kappa: float) -> np.ndarray:
    """Keep the largest ``ceil(kappa*N)`` native scores per query, zero elsewhere.

    Values are not squared, centred or renormalised. Ties at the selection
    boundary are resolved by ascending training index; downstream Spearman
    continues to use scipy's average ranks for value ties.
    """
    values = np.asarray(scores)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite (N,Q) matrix")
    if not 0.0 < float(kappa) <= 1.0:
        raise ValueError("kappa must lie in (0,1]")
    n, q = values.shape
    k = int(np.ceil(float(kappa) * n))
    if k == n:
        return values.copy()
    result = np.zeros_like(values)
    indices = np.arange(n)
    for column in range(q):
        chosen = np.lexsort((indices, -values[:, column]))[:k]
        result[chosen, column] = values[chosen, column]
    return result


def compute_lds(gt_matrix: np.ndarray, pred_matrix: np.ndarray):
    """Return ``(per_query_lds (Q,), mean_lds)`` — per-query Spearman, then averaged."""
    _, Q = gt_matrix.shape
    per_query = np.zeros(Q)
    for q in range(Q):
        gt_col, pred_col = gt_matrix[:, q], pred_matrix[:, q]
        if np.std(gt_col) < 1e-12 or np.std(pred_col) < 1e-12:
            per_query[q] = 0.0
        else:
            per_query[q] = spearman(gt_col, pred_col)
    return per_query, float(np.mean(per_query))


def lds_of_scores(scores: np.ndarray, gt_matrix: np.ndarray, masks: list,
                  subset_indices=None) -> float:
    """Convenience: mean LDS of a ``(N, Q)`` score matrix against a GT matrix."""
    if subset_indices is None:
        subset_indices = list(range(min(gt_matrix.shape[0], len(masks))))
    pred = predicted_influence(scores, masks, subset_indices)
    _, mean_lds = compute_lds(gt_matrix[:len(subset_indices)], pred)
    return mean_lds
