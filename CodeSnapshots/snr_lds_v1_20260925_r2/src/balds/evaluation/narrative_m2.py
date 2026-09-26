"""Fixed-pilot M2 statistics. No fitting, resampling experiment or IO."""
import numpy as np
from .lds import compute_lds, predicted_influence

BANDS = (0., .01, .05, .20, .50, 1.)


def rep_quantiles(values):
    """Linear quartiles/median in extended nonnegative reals; +inf is not discarded."""
    ordered=np.sort(values)
    result=[]
    for p in (.25,.5,.75):
        position=p*(len(ordered)-1)
        lo,hi=int(np.floor(position)),int(np.ceil(position))
        if lo==hi or np.isinf(ordered[lo]):result.append(float(ordered[lo]))
        elif np.isinf(ordered[hi]):result.append(float('inf'))
        else:result.append(float(ordered[lo]+(position-lo)*(ordered[hi]-ordered[lo])))
    return result


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
    numerator = np.abs(mean - np.median(mean, axis=0))
    rep = np.full_like(mean, np.nan)
    np.divide(numerator, sd, out=rep, where=sd > 0)
    rep[(sd == 0) & (numerator != 0)] = np.inf
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
            values = rep[bands[:, j] == band, j]
            defined = values[~np.isnan(values)]
            quartiles = rep_quantiles(defined) if len(defined) else [None] * 3
            lds = [a["lds"] for a in per_repeat if a["query_id"] == query and a["band"] == band]
            per_band.append({"query_id": query, "band": band, "n": len(values),
                "undefined_0_over_0": int(np.isnan(values).sum()),
                "infinite_nonzero_over_0": int(np.isinf(values).sum()),
                "rep_q25": "+inf" if len(defined) and np.isinf(quartiles[0]) else quartiles[0],
                "rep_median": "+inf" if len(defined) and np.isinf(quartiles[1]) else quartiles[1],
                "rep_q75": "+inf" if len(defined) and np.isinf(quartiles[2]) else quartiles[2],
                "rep_iqr": (float(quartiles[2] - quartiles[0])
                            if len(defined) and np.isfinite(quartiles).all() else None),
                "rep_lt1": float(np.mean(defined < 1)) if len(defined) else None,
                "rep_denominator": len(defined), "lds_mean": float(np.mean(lds)),
                "lds_repeat_sd": float(np.std(lds, ddof=1))})
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
    return {"coordinates": {"mean": mean, "sample_sd": sd, "rep": rep, "band": bands},
            "bands": per_band, "repeat_lds": per_repeat, "ordering": ordering,
            "queries": query_summary,
            "summary": {key: float(np.mean([v[key] for v in query_summary if v[key] is not None]))
                        if any(v[key] is not None for v in query_summary) else None
                        for key in ("measured_response_agreement", "between_repeats_agreement", "full_lds_mean")}}
