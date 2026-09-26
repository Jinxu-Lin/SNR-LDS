from __future__ import annotations

import numpy as np

from balds.evaluation.injection import aggregate, gt_rows, pool_rows, query_metrics, select_lambda_inject


def _meta(n_per_host=10, k=2):
    return {"pairs": {
        "zero": {"host_label": 0, "concept_name": "c0", "tier": "coarse",
                 "replaced_cifar10_indices": list(range(k))},
        "one": {"host_label": 1, "concept_name": "c1", "tier": "fine",
                "replaced_cifar10_indices": list(range(n_per_host, n_per_host + k))},
    }}


def _perfect(host, n_per_host=10, k=2):
    scores = np.arange(2 * n_per_host, dtype=float) * -1e-4
    start = host * n_per_host
    scores[start:start + k] = [2.0, 1.0]
    return scores


def test_perfect_pool_and_global_metrics_and_masks():
    labels = np.repeat([0, 1], 10)
    meta = _meta()
    metrics = query_metrics(_perfect(0), labels, meta, 0, k=2)
    for name in ("auroc_pool", "precision_at_k_pool", "ap_pool",
                 "recall_at_k_global", "ap_global"):
        assert metrics[name] == 1.0
    assert pool_rows(labels, 1).sum() == 10
    assert gt_rows(meta, 1, len(labels)).nonzero()[0].tolist() == [10, 11]


def test_artbench_foreign_rows_are_the_injection_ground_truth():
    meta = {"pairs": {"host": {"host_label": 3, "foreign_rows": [8, 9]}}}
    mask = gt_rows(meta, 3, n=12)
    assert np.flatnonzero(mask).tolist() == [8, 9]


def test_random_metrics_are_near_chance():
    n, k = 10_000, 200
    labels = np.zeros(n, dtype=int)
    meta = {"pairs": {"zero": {
        "host_label": 0, "concept_name": "c0", "tier": "coarse",
        "replaced_cifar10_indices": list(range(k)),
    }}}
    scores = np.random.RandomState(7).randn(n)
    metrics = query_metrics(scores, labels, meta, 0, k=k)
    assert abs(metrics["auroc_pool"] - 0.5) < 0.05
    assert abs(metrics["precision_at_k_pool"] - k / n) < 0.02
    assert abs(metrics["ap_pool"] - k / n) < 0.02


def test_selector_uses_val_only_and_reports_both_curves():
    labels = np.repeat([0, 1], 10)
    meta = _meta()
    # λ=.1 owns validation; λ=1 owns test. A leaked selector would pick by a
    # different objective, while inject_val must deterministically choose .1.
    bad0 = -_perfect(0)
    bad1 = -_perfect(1)
    scores = {
        0.1: np.stack([_perfect(0), bad0, _perfect(1), bad1], axis=1),
        1.0: np.stack([bad0, _perfect(0), bad1, _perfect(1)], axis=1),
    }
    choice = select_lambda_inject(
        scores, [0.1, 1.0], split=np.array([0, 1, 0, 1]),
        host_labels=np.array([0, 0, 1, 1]), train_labels=labels, meta=meta, k=2)
    assert choice.best_lam == 0.1 and choice.best_ap_val == 1.0
    assert choice.per_lambda_ap_val[0.1] > choice.per_lambda_ap_val[1.0]
    assert choice.per_lambda_ap_test[0.1] < choice.per_lambda_ap_test[1.0]


def test_bootstrap_aggregate_shapes_and_chance():
    labels = np.repeat([0, 1], 10)
    meta = _meta()
    rows = []
    for host in (0, 1):
        for _ in range(3):
            row = query_metrics(_perfect(host), labels, meta, host, k=2)
            row.update(concept=f"c{host}", tier="fine" if host else "coarse")
            rows.append(row)
    out = aggregate(rows, groups=("concept", "tier", "overall"), n_boot=100, seed=42)
    assert set(out) == {"per_concept", "per_tier", "overall", "chance"}
    assert out["overall"]["n_queries"] == 6
    for summary in [out["overall"], *out["per_concept"].values()]:
        assert summary["ap_pool"]["ci_lo"] <= summary["ap_pool"]["mean"] <= \
            summary["ap_pool"]["ci_hi"]
    assert out["chance"]["auroc_pool"] == 0.5
