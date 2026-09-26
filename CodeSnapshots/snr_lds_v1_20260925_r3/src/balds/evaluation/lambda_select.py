"""λ-regularisation selection policies.

The legacy pipeline picks λ by LDS against the *same* ground truth used for the
final report — an oracle/in-sample choice with leakage. Both policies are
registered here; ``oracle`` reproduces legacy behaviour (parity), ``holdout``
selects λ on a disjoint query split and is the honest default switched on in
Phase 5. New policies are added by ``@SELECTORS.register`` (OCP).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from balds.schema.registry import SELECTORS
from .lds import compute_lds, predicted_influence, spearman


@dataclass(frozen=True)
class LambdaChoice:
    best_lam: float
    best_lds: float
    per_lambda_lds: dict


class LambdaSelector(ABC):
    """Chooses the best λ from per-λ score matrices given GT + masks."""

    @abstractmethod
    def select(self, scores_by_lambda: dict, lambda_order: list, gt_matrix,
               masks: list, subset_indices) -> LambdaChoice:
        ...


class OracleSelector(LambdaSelector):
    """Pick λ maximising mean LDS on the full GT (in-sample; legacy parity)."""

    def select(self, scores_by_lambda, lambda_order, gt_matrix, masks, subset_indices):
        per_lambda: dict = {}
        best_lam, best_lds = lambda_order[0], -float("inf")
        gt = gt_matrix[:len(subset_indices)]
        for lam in lambda_order:
            pred = predicted_influence(scores_by_lambda[lam], masks, subset_indices)
            _, mean_lds = compute_lds(gt, pred)
            per_lambda[lam] = mean_lds
            if mean_lds > best_lds:
                best_lds, best_lam = mean_lds, lam
        return LambdaChoice(best_lam, best_lds, per_lambda)


class HoldoutSelector(LambdaSelector):
    """Pick λ on a disjoint query split; report LDS on the held-out split."""

    def __init__(self, frac: float = 0.5, seed: int = 0) -> None:
        self.frac = frac
        self.seed = seed

    def _split(self, q: int):
        rng = np.random.RandomState(self.seed)
        perm = rng.permutation(q)
        cut = int(q * self.frac)
        return perm[:cut], perm[cut:]

    def _mean_lds_cols(self, gt, pred, cols) -> float:
        vals = []
        for c in cols:
            if np.std(gt[:, c]) >= 1e-12 and np.std(pred[:, c]) >= 1e-12:
                vals.append(spearman(gt[:, c], pred[:, c]))
            else:
                vals.append(0.0)
        return float(np.mean(vals)) if len(vals) else 0.0

    def select(self, scores_by_lambda, lambda_order, gt_matrix, masks, subset_indices):
        gt = gt_matrix[:len(subset_indices)]
        q = gt.shape[1]
        sel_cols, eval_cols = self._split(q)
        per_lambda: dict = {}
        best_lam, best_sel = lambda_order[0], -float("inf")
        preds = {}
        for lam in lambda_order:
            preds[lam] = predicted_influence(scores_by_lambda[lam], masks, subset_indices)
            sel_lds = self._mean_lds_cols(gt, preds[lam], sel_cols)
            per_lambda[lam] = sel_lds
            if sel_lds > best_sel:
                best_sel, best_lam = sel_lds, lam
        eval_lds = self._mean_lds_cols(gt, preds[best_lam], eval_cols)
        return LambdaChoice(best_lam, eval_lds, per_lambda)


# register stateless selector instances (holdout uses default frac/seed)
SELECTORS.add("oracle", OracleSelector())
SELECTORS.add("holdout", HoldoutSelector())
