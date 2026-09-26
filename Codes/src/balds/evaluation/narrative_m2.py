"""Fixed-pilot M2 statistics. No fitting, resampling experiment or IO."""
import numpy as np
from .lds import compute_lds, predicted_influence

BANDS = (0., .01, .05, .20, .50, 1.)


def variance_ratio(mean, variance):
    """Ratio of sums, with no centering or per-coordinate ratio averaging."""
    numerator = float(np.sum(variance, dtype=np.float64))
    denominator = float(np.sum(np.square(mean), dtype=np.float64))
    return numerator / denominator if denominator > 0 else None


def variation_deciles(scores, query_ids):
    """Figure 1: per-query equal bins, increasing absolute repeat mean."""
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError("at least two finite repeats required")
    mean, variance = x.mean(axis=0), x.var(axis=0, ddof=1)
    if len(query_ids) != mean.shape[1] or len(mean) % 10:
        raise ValueError("query IDs must align and train rows must divide into ten equal bins")
    order = np.argsort(np.abs(mean), axis=0, kind="stable")
    rows = []
    for band in range(10):
        ids = order[len(mean) * band // 10:len(mean) * (band + 1) // 10]
        values = [variance_ratio(mean[ids[:, j], j], variance[ids[:, j], j])
                  for j in range(mean.shape[1])]
        defined = [v for v in values if v is not None]
        rows.append(dict(rank_start=band * 10, rank_end=(band + 1) * 10,
                         samples_per_query=len(ids), query_ids=list(map(int, query_ids)),
                         varratio_per_query=values, valid_queries=len(defined),
                         varratio_mean=float(np.mean(defined)) if defined else None,
                         varratio_median=float(np.median(defined)) if defined else None,
                         varratio_min=float(np.min(defined)) if defined else None,
                         varratio_max=float(np.max(defined)) if defined else None))
    return rows


def pilot_bands(pilot):
    x = np.asarray(pilot)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("pilot must be finite N by Q")
    order = np.argsort(-np.abs(x), axis=0, kind="stable")
    bands = np.empty(x.shape, dtype=np.int8)
    edges = [int(p * len(x)) for p in BANDS]
    for b, (lo, hi) in enumerate(zip(edges, edges[1:])):
        for q in range(x.shape[1]):
            bands[order[lo:hi, q], q] = b
    return bands


def ordering_counts(a, b):
    """Strict-pair denominator; ties excluded, reported separately (no jitter)."""
    a, b = np.asarray(a), np.asarray(b)
    both = (a != 0) & (b != 0)
    n = int(both.sum())
    correct = int(np.sum((a == b) & both))
    return {"pairs": len(a), "strict_pairs": n, "agree": correct,
            "disagree": n - correct, "agreement": correct / n if n else None,
            "error_rate": (n - correct) / n if n else None,
            "left_ties": int((a == 0).sum()), "right_ties": int((b == 0).sum()),
            "both_ties": int(((a == 0) & (b == 0)).sum())}


def m2a_statistics(scores, pilot, masks, response, query_ids):
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError("at least two finite repeats required")
    r, n, q = x.shape
    masks, response = np.asarray(masks), np.asarray(response)
    if (np.shape(pilot) != (n, q) or response.shape != (len(masks), q)
            or masks.shape[1] != n or len(query_ids) != q
            or not np.isin(masks, [0, 1]).all() or not np.isfinite(response).all()):
        raise ValueError("score/pilot/mask/response IDs or shapes disagree")
    bands = pilot_bands(pilot)
    mean, sd = x.mean(0), x.std(0, ddof=1)
    per_band, per_repeat, ordering = [], [], []
    pairs = np.triu_indices(len(masks), 1)
    predictions = []
    for repeat in range(r):
        full = predicted_influence(x[repeat], masks, range(len(masks)))
        predictions.append(full)
        for band in [-1, *range(5)]:
            pred = full if band == -1 else predicted_influence(
                np.where(bands == band, x[repeat], 0), masks, range(len(masks)))
            lds = compute_lds(response, pred)[0]
            for j, query in enumerate(query_ids):
                per_repeat.append({"repeat": repeat, "query_id": query,
                                   "band": band, "lds": float(lds[j]),
                                   "constant_prediction": bool(np.ptp(pred[:, j]) == 0),
                                   "constant_response": bool(np.ptp(response[:, j]) == 0)})
    for j, query in enumerate(query_ids):
        for band in range(5):
            support = bands[:, j] == band
            lds = [a["lds"] for a in per_repeat if a["query_id"] == query and a["band"] == band]
            per_band.append({"query_id": query, "band": band, "n": int(support.sum()),
                "varratio": variance_ratio(mean[support, j], np.square(sd[support, j])),
                "lds_mean": float(np.mean(lds)), "lds_repeat_sd": float(np.std(lds, ddof=1))})
        truth = np.sign(response[pairs[0], j] - response[pairs[1], j])
        signs = [np.sign(p[pairs[0], j] - p[pairs[1], j]) for p in predictions]
        for a in range(r):
            ordering.append({"query_id": query, "kind": "measured_response", "left": a,
                             "right": -1, **ordering_counts(signs[a], truth)})
            for b in range(a + 1, r):
                ordering.append({"query_id": query, "kind": "between_repeats", "left": a,
                                 "right": b, **ordering_counts(signs[a], signs[b])})
    query_summary = []
    for query in query_ids:
        row = {"query_id": query}
        for kind in ("measured_response", "between_repeats"):
            values = [a["agreement"] for a in ordering if a["query_id"] == query
                      and a["kind"] == kind and a["agreement"] is not None]
            row[kind + "_agreement"] = float(np.mean(values)) if values else None
        full = [a["lds"] for a in per_repeat if a["query_id"] == query and a["band"] == -1]
        row.update(full_lds_mean=float(np.mean(full)), full_lds_sd=float(np.std(full, ddof=1)))
        query_summary.append(row)
    return {"coordinates": {"mean": mean, "sample_sd": sd, "band": bands},
            "bands": per_band, "mean_deciles": variation_deciles(x, query_ids), "repeat_lds": per_repeat, "ordering": ordering,
            "queries": query_summary,
            "summary": {key: float(np.mean([v[key] for v in query_summary if v[key] is not None]))
                        if any(v[key] is not None for v in query_summary) else None
                        for key in ("measured_response_agreement", "between_repeats_agreement", "full_lds_mean")}}
