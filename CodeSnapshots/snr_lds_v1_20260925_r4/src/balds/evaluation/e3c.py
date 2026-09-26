"""Pure CPU paired E3c statistics, preserving query clusters across every axis."""
from itertools import combinations
import numpy as np

from .lds import compute_lds, predicted_influence, support_head_scores


def preference_agreement(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.where(a == b, 1.0, np.where((a == 0) | (b == 0), 0.5, 0.0))


def head_ids(scores, kappa=.05):
    n = scores.shape[0]
    return [set(np.lexsort((np.arange(n), -scores[:, q]))[:int(np.ceil(kappa * n))].tolist())
            for q in range(scores.shape[1])]


def analyze_track(scores, gt, masks, query_ids, kappas, *, bootstrap_count=2000, bootstrap_seed=20260918):
    """scores[r][method] has (N,Q); only fully validated repeat pairs enter."""
    methods = ("fmas_raw", "dtrak_T100")
    repeats = sorted(scores)
    rows, repeat_rows, pairs, preferences, preference_pairs = [], [], [], [], []
    if not repeats:
        return {"repeats": [], "per_query": [], "per_repeat": [], "head_pairs": [],
                "preferences": [], "preference_pairs": [], "summary": []}
    qn = len(query_ids)
    gt, masks = np.asarray(gt), np.asarray(masks)
    if gt.shape != (len(masks), qn) or not np.isfinite(gt).all():
        raise ValueError("GT/mask/query alignment mismatch")
    values = np.empty((len(repeats), 2, len(kappas), qn))
    heads = {}
    for ri, r in enumerate(repeats):
        if set(scores[r]) != set(methods):
            raise ValueError("incomplete paired repeat")
        for mi, method in enumerate(methods):
            s = np.asarray(scores[r][method])
            if s.shape != (masks.shape[1], qn) or not np.isfinite(s).all():
                raise ValueError("missing rows/nonfinite scores")
            heads[r, method] = head_ids(s)
            for ki, k in enumerate(kappas):
                pred = predicted_influence(support_head_scores(s, k), masks, list(range(len(masks))))
                perq, mean = compute_lds(gt, pred)
                values[ri, mi, ki] = perq
                repeat_rows.append({"repeat_id": r, "method": method, "kappa": k, "lds": mean})
                for qi, q in enumerate(query_ids):
                    rows.append({"repeat_id": r, "method": method, "kappa": k, "query_id": q,
                                 "lds": float(perq[qi]), "constant_prediction": bool(np.std(pred[:, qi]) < 1e-12),
                                 "constant_response": bool(np.std(gt[:, qi]) < 1e-12)})
    signs = np.sign(values[:, 0] - values[:, 1])
    for ri, r in enumerate(repeats):
        for ki, k in enumerate(kappas):
            for qi, q in enumerate(query_ids):
                preferences.append({"repeat_id": r, "query_id": q, "kappa": k,
                                    "fmas_minus_dtrak": float(values[ri, 0, ki, qi] - values[ri, 1, ki, qi]),
                                    "preference": int(signs[ri, ki, qi]), "tie": bool(signs[ri, ki, qi] == 0)})
    agreement = []
    for a, b in combinations(range(len(repeats)), 2):
        agree = preference_agreement(signs[a], signs[b])
        agreement.append(agree)
        for ki, k in enumerate(kappas):
            for qi, q in enumerate(query_ids):
                preference_pairs.append({"repeat_a": repeats[a], "repeat_b": repeats[b],
                    "query_id": q, "kappa": k, "agreement": float(agree[ki, qi]),
                    "ties_in_pair": int(signs[a, ki, qi] == 0) + int(signs[b, ki, qi] == 0)})
        for method in methods:
            for qi, q in enumerate(query_ids):
                ha, hb = heads[repeats[a], method][qi], heads[repeats[b], method][qi]
                pairs.append({"repeat_a": repeats[a], "repeat_b": repeats[b], "method": method,
                              "query_id": q, "kappa": .05, "jaccard": len(ha & hb) / len(ha | hb)})
    agreement = np.asarray(agreement)
    rng = np.random.default_rng(bootstrap_seed)
    # ONE query index vector per draw is shared by all repeats, methods and kappas.
    draws = rng.integers(0, qn, size=(bootstrap_count, qn))
    boot_means = values[..., draws].mean(axis=-1)  # R,M,K,B
    def interval(x):
        return np.quantile(x, [.025, .975], axis=-1).tolist()
    summaries = []
    for ki, k in enumerate(kappas):
        diff = values[:, 0, ki] - values[:, 1, ki]
        item = {"kappa": k, "repeat_count": len(repeats), "query_count": qn,
                "fmas_minus_dtrak": float(diff.mean()),
                "difference_ci95": interval(diff[:, draws].mean(axis=(0, 2))),
                "ties": int((signs[:, ki] == 0).sum()),
                "fmas_wins": int((signs[:, ki] > 0).sum()), "dtrak_wins": int((signs[:, ki] < 0).sum()),
                "preference_agreement": float(agreement[:, ki].mean()) if len(agreement) else None,
                "agreement_ci95": interval(agreement[:, ki, draws].mean(axis=(0, 2))) if len(agreement) else None}
        for mi, method in enumerate(methods):
            means = values[:, mi, ki].mean(axis=-1)
            item[method] = {"mean_lds": float(means.mean()),
                            "repeat_sd_ddof1": float(means.std(ddof=1)) if len(repeats) > 1 else None,
                            "mean_ci95": interval(boot_means[:, mi, ki].mean(axis=0)),
                            "sd_ci95": interval(boot_means[:, mi, ki].std(axis=0, ddof=1)) if len(repeats) > 1 else None}
        summaries.append(item)
    return {"repeats": repeats, "per_query": rows, "per_repeat": repeat_rows,
            "head_pairs": pairs, "preferences": preferences, "preference_pairs": preference_pairs, "summary": summaries,
            "bootstrap": {"count": bootstrap_count, "seed": bootstrap_seed, "unit": "query_cluster"},
            "constant_prediction_count": sum(x["constant_prediction"] for x in rows),
            "constant_response_count": sum(x["constant_response"] for x in rows)}
