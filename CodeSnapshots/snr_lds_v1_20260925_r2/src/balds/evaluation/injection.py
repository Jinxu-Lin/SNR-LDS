"""Metrics and validation-only regularisation selection for INJECT."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


METRICS = ("auroc_pool", "precision_at_k_pool", "ap_pool",
           "recall_at_k_global", "ap_global")


def pool_rows(train_labels, host: int) -> np.ndarray:
    return np.asarray(train_labels, dtype=np.int64) == int(host)


def _pair_row(meta: dict, host: int) -> dict:
    rows = [row for row in meta["pairs"].values() if int(row["host_label"]) == int(host)]
    if len(rows) != 1:
        raise ValueError(f"metadata must contain exactly one pair for host {host}")
    return rows[0]


def gt_rows(meta: dict, host: int, n: int | None = None) -> np.ndarray:
    row = _pair_row(meta, host)
    key = ("replaced_cifar10_indices" if "replaced_cifar10_indices" in row
           else "foreign_rows")
    if key not in row:
        raise ValueError("pair metadata has no injected training-row indices")
    indices = np.asarray(row[key], dtype=np.int64)
    if n is None:
        n = 50_000 if key == "replaced_cifar10_indices" else 5_000
    size = int(n)
    mask = np.zeros(size, dtype=bool)
    if np.any(indices < 0) or np.any(indices >= size):
        raise ValueError("injected row lies outside the training score axis")
    mask[indices] = True
    return mask


def _average_precision(scores: np.ndarray, positive: np.ndarray) -> float:
    n_pos = int(positive.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = positive[order].astype(np.float64)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / n_pos)


def _auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    u = float(ranks[positive].sum()) - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def query_metrics(score_col, train_labels, meta: dict, host: int, *, k: int) -> dict:
    """Five INJECT metrics; higher scores always mean more influential.

    AP and top-k use a stable descending sort, so score ties are broken by the
    original training-row index. AUROC gives mathematical half-credit to ties.
    """
    scores = np.asarray(score_col, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.int64)
    if scores.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("one score column must align with train_labels")
    if not np.isfinite(scores).all():
        raise ValueError("injection scores must be finite")
    k = int(k)
    if k <= 0 or k > len(scores):
        raise ValueError("k must lie inside the score axis")
    pool = pool_rows(labels, host)
    gt = gt_rows(meta, host, len(scores))
    if np.any(gt & ~pool):
        raise ValueError("injected GT rows must belong to their host pool")
    pool_scores, pool_gt = scores[pool], gt[pool]
    k_pool = min(k, len(pool_scores))
    top_pool = np.argsort(-pool_scores, kind="stable")[:k_pool]
    top_global = np.argsort(-scores, kind="stable")[:k]
    n_gt = int(gt.sum())
    result = {
        "auroc_pool": _auroc(pool_scores, pool_gt),
        "precision_at_k_pool": float(pool_gt[top_pool].sum()) / k_pool,
        "ap_pool": _average_precision(pool_scores, pool_gt),
        "recall_at_k_global": float(gt[top_global].sum()) / n_gt,
        "ap_global": _average_precision(scores, gt),
        "_chance": {
            "auroc_pool": 0.5,
            "precision_at_k_pool": n_gt / len(pool_scores),
            "ap_pool": n_gt / len(pool_scores),
            "recall_at_k_global": k / len(scores),
            "ap_global": n_gt / len(scores),
        },
    }
    return result


def _summary(rows: list[dict], n_boot: int, rng: np.random.RandomState) -> dict:
    out = {"n_queries": len(rows)}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        mean = float(np.mean(values))
        if len(values) <= 1 or n_boot <= 0:
            lo = hi = mean
        else:
            draws = rng.randint(0, len(values), size=(int(n_boot), len(values)))
            means = values[draws].mean(axis=1)
            lo, hi = (float(v) for v in np.percentile(means, [2.5, 97.5]))
        out[metric] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
    return out


def aggregate(rows, *, groups=("concept", "tier", "overall"),
              n_boot: int, seed: int) -> dict:
    """Aggregate per-query rows with within-group query bootstrap intervals."""
    rows = list(rows)
    if not rows:
        raise ValueError("cannot aggregate an empty query set")
    rng = np.random.RandomState(int(seed))
    requested = set(groups)
    result = {}
    if "concept" in requested:
        result["per_concept"] = {
            str(name): _summary([row for row in rows if str(row["concept"]) == str(name)],
                                n_boot, rng)
            for name in sorted({str(row["concept"]) for row in rows})
        }
    if "tier" in requested:
        result["per_tier"] = {
            str(name): _summary([row for row in rows if str(row["tier"]) == str(name)],
                                n_boot, rng)
            for name in sorted({str(row["tier"]) for row in rows})
        }
    if "overall" in requested:
        result["overall"] = _summary(rows, n_boot, rng)
    result["chance"] = {
        metric: float(np.mean([row["_chance"][metric] for row in rows]))
        for metric in METRICS
    }
    return result


@dataclass(frozen=True)
class LambdaChoice:
    best_lam: float
    best_ap_val: float
    per_lambda_ap_val: dict[float, float]
    per_lambda_ap_test: dict[float, float]


def select_lambda_inject(scores_by_lambda, lambda_order, split, host_labels,
                         train_labels, meta, *, k: int) -> LambdaChoice:
    """Select solely by mean within-pool AP over split==0 (validation)."""
    split = np.asarray(split, dtype=np.int8)
    hosts = np.asarray(host_labels, dtype=np.int64)
    if split.shape != hosts.shape or not np.any(split == 0):
        raise ValueError("inject_val selection needs aligned queries and a non-empty val split")
    if not np.any(split == 1):
        raise ValueError("inject_val selection needs a non-empty test split for reporting")

    def mean_ap(matrix, which: int) -> float:
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[1] != len(hosts):
            raise ValueError("score matrices must have shape (N,Q) for INJECT queries")
        values = [query_metrics(matrix[:, q], train_labels, meta, int(hosts[q]), k=k)["ap_pool"]
                  for q in np.flatnonzero(split == which)]
        return float(np.mean(values))

    val, test = {}, {}
    best_lam, best_ap = float(lambda_order[0]), -float("inf")
    for lam in lambda_order:
        lam = float(lam)
        val[lam] = mean_ap(scores_by_lambda[lam], 0)
        test[lam] = mean_ap(scores_by_lambda[lam], 1)
        if val[lam] > best_ap:
            best_lam, best_ap = lam, val[lam]
    return LambdaChoice(best_lam, best_ap, val, test)
