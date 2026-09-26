"""Statistical procedures for the evaluation layer.

Ported from ``fmas/utils.py`` and **wired in** (the legacy ``bootstrap_ci`` and
Wilcoxon helpers were defined but never called). Used by the report layer to
render LDS confidence intervals and paired significance tests.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(scores, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42):
    """Bootstrap ``(mean, lo, hi)`` confidence interval for the mean of ``scores``."""
    scores = np.asarray(scores).ravel()
    n = len(scores)
    rng = np.random.RandomState(seed)
    means = np.array([rng.choice(scores, size=n, replace=True).mean()
                      for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    return float(np.mean(scores)), float(np.percentile(means, 100 * alpha)), \
        float(np.percentile(means, 100 * (1 - alpha)))


def paired_wilcoxon(a, b):
    """Paired Wilcoxon signed-rank test on ``a - b``; returns ``(stat, pvalue)``."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    stat, pval = stats.wilcoxon(a - b)
    return float(stat), float(pval)


def holm_bonferroni(p_values, alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down rejections for a family of p-values."""
    p_values = list(p_values)
    order = np.argsort(p_values)
    decisions = [False] * len(p_values)
    for rank, index in enumerate(order):
        if p_values[index] <= alpha / (len(p_values) - rank):
            decisions[index] = True
        else:
            break
    return decisions
