"""SNR-LDS: zero-mean noise fitting, strict amplitude selection and native aggregation.

One float64 central-grid KDE fit per query; no subset responses enter fitting.
DAS fits signed pre-square t and aggregates its native square exactly once.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

SNR_RULE = "snr_zero_mean_gaussian_v1"
RULE = SNR_RULE
SNR_ZETAS = (1.0, 2.0, 3.0, 4.0)




def _zetas(values):
    result = tuple(float(v) for v in values)
    if (not result or len(set(result)) != len(result)
            or not np.isfinite(result).all() or any(v <= 0 for v in result)):
        raise ValueError("zetas must be distinct finite positive values")
    return result


def zero_mean_background(grid, log_density):
    """Weighted ``A + C*x**2`` fit used by SNR-LDS.

    Multiplying both design and target by ``sqrt(f)`` implements the specified
    ``sum f * residual**2`` objective.  The common ``max(log f)`` rescaling
    avoids overflow and has no effect on the minimizer.
    """
    grid = np.asarray(grid, dtype=np.float64)
    log_density = np.asarray(log_density, dtype=np.float64)
    if (grid.ndim != 1 or log_density.shape != grid.shape or len(grid) < 2
            or not np.isfinite(grid).all() or not np.isfinite(log_density).all()):
        return {}, "nonfinite_fit_input"
    design = np.column_stack((np.ones(len(grid)), grid * grid))
    weight = np.exp(.5 * (log_density - np.max(log_density)))
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * weight[:, None], log_density * weight, rcond=None)
    A, C = coefficients
    params = {"A": float(A), "C": float(C)}
    if rank != 2 or not np.isfinite(coefficients).all():
        return params, "invalid_zero_mean_quadratic"
    if C >= 0:
        return params, "nonnegative_C"
    sigma_x = float(np.sqrt(-1.0 / (2.0 * C)))
    params["sigma_x"] = sigma_x
    if not np.isfinite(sigma_x) or sigma_x <= 0:
        return params, "invalid_sigma"
    return params, None


def fit_snr_background(scores, *, zetas=SNR_ZETAS, device="cpu", chunk=256):
    """Fit the frozen zero-mean Gaussian SNR rule to one signed vector.

    The KDE is evaluated only on the fixed 101-point central grid.  No density
    is evaluated at the N training coordinates.  All requested thresholds are
    derived from this single fit.
    """
    values = np.asarray(scores, dtype=np.float64)
    zs = _zetas(zetas)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("one finite score vector with at least two values is required")
    empty = np.zeros(len(values), dtype=bool)
    if np.all(values == 0):
        return dict(rule=SNR_RULE, status="all_zero_empty", reason=None, rms=0.0,
                    x=np.zeros_like(values), grid=None, log_density=None,
                    params={"bandwidth": None, "mad_width": 0.0,
                            "sigma_x": None, "sigma": 0.0},
                    selected={z: empty.copy() for z in zs})

    maximum = float(np.max(np.abs(values)))
    rms = float(maximum * np.sqrt(np.mean((values / maximum) ** 2)))
    x = values / rms
    sd = float(np.std(x, ddof=1))
    q25, q75 = np.percentile(x, [25, 75], method="linear")
    iqr = float(q75 - q25)
    bandwidth = float(.9 * min(sd, iqr / 1.34) * len(x) ** (-.2))
    median = float(np.median(x))
    width = float(1.4826 * np.median(np.abs(x - median)))
    params = dict(bandwidth=bandwidth, mad_width=width, median=median,
                  sd=sd, iqr=iqr)
    result = dict(rule=SNR_RULE, status="fit_failed", reason=None, rms=rms,
                  x=x, grid=None, log_density=None, params=params,
                  selected={z: empty.copy() for z in zs})
    if not np.isfinite([rms, bandwidth, width]).all() or rms <= 0 or bandwidth <= 0:
        result["reason"] = "zero_or_nonfinite_bandwidth"
        return result
    if width <= 0:
        result["reason"] = "zero_central_width"
        return result

    # The central window is deliberately centered at zero, not at the sample
    # median used to determine its robust half-width.
    grid = np.linspace(-width, width, 101, dtype=np.float64)
    log_density = kde_log_density(grid, x, bandwidth, chunk, device=device)
    fitted, error = zero_mean_background(grid, log_density)
    params.update(fitted)
    result.update(grid=grid, log_density=log_density)
    if error is not None:
        result["reason"] = error
        return result
    sigma_x = params["sigma_x"]
    sigma = float(rms * sigma_x)
    params["sigma"] = sigma
    if not np.isfinite(sigma) or sigma <= 0:
        result["reason"] = "invalid_sigma"
        return result
    ratio = np.abs(x) / sigma_x
    result["selected"] = {z: ratio > z for z in zs}  # strict by definition
    result["status"] = "ok"
    return result


def evaluate_snr(scores, keep_masks, responses, *, fit_scores=None,
                 zetas=SNR_ZETAS, primary_zeta=3.0, device="cpu", chunk=256):
    """Evaluate Full and zero-mean SNR-LDS for aligned ``(N,Q)`` arrays."""
    from .lds import compute_lds, predicted_influence

    zs = _zetas(zetas)
    primary = float(primary_zeta)
    if primary not in zs:
        raise ValueError("primary_zeta must be present in zetas")
    values = np.asarray(scores, dtype=np.float64)
    fitting = values if fit_scores is None else np.asarray(fit_scores, dtype=np.float64)
    masks = np.asarray(keep_masks)
    response = np.asarray(responses, dtype=np.float64)
    if values.ndim != 2 or fitting.shape != values.shape or min(values.shape) < 1:
        raise ValueError("scores and fit_scores must be aligned (N,Q) matrices")
    n, q = values.shape
    if masks.ndim != 2 or masks.shape[1] != n or masks.shape[0] < 2:
        raise ValueError("keep_masks must have shape (M,N), with M >= 2")
    if response.shape != (len(masks), q):
        raise ValueError("responses must have shape (M,Q)")
    if not np.isin(masks, [0, 1]).all():
        raise ValueError("keep_masks must contain retention indicators 0/1")
    if not all(np.isfinite(a).all() for a in (values, fitting, response)):
        raise ValueError("scores, fit_scores and responses must be finite")

    full_prediction = predicted_influence(values, masks, range(len(masks)))
    full_lds = compute_lds(response, full_prediction)[0]
    selected_all = np.zeros((len(zs), n, q), dtype=bool)
    snr_predictions = np.full((len(zs), len(masks), q), np.nan, dtype=np.float64)
    rows, fits = [], []
    for column in range(q):
        fit = fit_snr_background(fitting[:, column], zetas=zs, device=device, chunk=chunk)
        fits.append(fit)
        valid = fit["status"] != "fit_failed"
        by_zeta = {}
        for zi, zeta in enumerate(zs):
            chosen = fit["selected"][zeta]
            selected_all[zi, :, column] = chosen
            if valid:
                prediction = predicted_influence(
                    np.where(chosen, values[:, column], 0.0)[:, None],
                    masks, range(len(masks)))[:, 0]
                snr_predictions[zi, :, column] = prediction
                lds = float(compute_lds(response[:, column:column + 1],
                                        prediction[:, None])[0][0])
                by_zeta[str(zeta)] = dict(
                    snr_lds=lds, n_selected=int(chosen.sum()),
                    retained_fraction=float(chosen.mean()),
                    constant_prediction=bool(np.ptp(prediction) == 0))
            else:
                by_zeta[str(zeta)] = dict(snr_lds=None, n_selected=None,
                                          retained_fraction=None,
                                          constant_prediction=None)
        main = by_zeta[str(primary)]
        status = fit["status"]
        if valid and main["n_selected"] == 0 and status == "ok":
            status = "empty"
        rows.append(dict(
            query_id=column, status=status, fit_status=fit["status"], reason=fit["reason"],
            full_lds=float(full_lds[column]), snr_lds=main["snr_lds"],
            n_selected=main["n_selected"], retained_fraction=main["retained_fraction"],
            negative_fit_selected=(int(np.sum(selected_all[zs.index(primary), :, column]
                                                  & (fitting[:, column] < 0)))
                                   if valid else None),
            constant_prediction=main["constant_prediction"],
            constant_response=bool(np.ptp(response[:, column]) == 0),
            rms=fit["rms"], params=fit["params"], by_zeta=by_zeta))

    valid_rows = [row for row in rows if row["snr_lds"] is not None]
    summary = dict(
        n_total=q, n_valid=len(valid_rows), n_failed=q - len(valid_rows),
        n_fit_failed=sum(row["fit_status"] == "fit_failed" for row in rows),
        n_all_zero=sum(row["fit_status"] == "all_zero_empty" for row in rows),
        n_empty=sum(row["n_selected"] == 0 for row in valid_rows),
        n_constant_prediction=sum(row["constant_prediction"] for row in valid_rows),
        n_constant_response=sum(row["constant_response"] for row in valid_rows),
        full_lds=(float(np.mean([row["full_lds"] for row in valid_rows]))
                  if valid_rows else None),
        snr_lds=(float(np.mean([row["snr_lds"] for row in valid_rows]))
                 if valid_rows else None),
        primary_zeta=primary, zetas=list(zs))
    return dict(rule=SNR_RULE, per_query=rows, summary=summary,
                selected=selected_all[zs.index(primary)],
                selected_by_zeta=selected_all, zetas=np.asarray(zs),
                full_prediction=full_prediction,
                snr_prediction=snr_predictions[zs.index(primary)],
                snr_prediction_by_zeta=snr_predictions, fits=fits)


def evaluate_snr_das(presquare, keep_masks, responses, **kwargs):
    """SNR paper DAS: fit signed ``t`` and aggregate ``t**2`` exactly once."""
    signed = np.asarray(presquare, dtype=np.float64)
    return evaluate_snr(signed * signed, keep_masks, responses,
                        fit_scores=signed, **kwargs)

def kde_log_density(points, samples, bandwidth, chunk=256, *, device='cpu'):
    """Exact float64 Gaussian KDE; CUDA changes execution, not bandwidth or samples.

    Only ``chunk x N`` distances reside on device, never the full N x N matrix.
    The CPU path remains the SciPy reference. No density floor or approximation.
    """
    points, samples = np.asarray(points, float), np.asarray(samples, float)
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError("bandwidth must be finite and positive")
    if chunk < 1:
        raise ValueError('KDE chunk must be positive')
    result = np.empty(points.size)
    normalizer = np.log(len(samples)) + np.log(bandwidth) + .5*np.log(2*np.pi)
    if device != 'cpu':
        import torch
        target = torch.device(device)
        if target.type != 'cuda':
            raise ValueError('KDE device must be cpu or cuda[:index]')
        with torch.inference_mode():
            sample_tensor = torch.as_tensor(samples, dtype=torch.float64, device=target)
            for start in range(0, len(points), chunk):
                query = torch.as_tensor(points[start:start+chunk], dtype=torch.float64, device=target)
                distance = query[:, None] - sample_tensor[None, :]
                distance.div_(bandwidth).square_().mul_(-.5)
                result[start:start+chunk] = (torch.logsumexp(distance, dim=1)-normalizer).cpu().numpy()
        return result
    for start in range(0, len(points), chunk):
        distance = (points[start:start+chunk, None] - samples[None, :]) / bandwidth
        result[start:start+chunk] = logsumexp(-.5*distance**2, axis=1) - normalizer
    return result




def benchmark_method_selection(rows, utility_rows, *, bootstrap_count=2000,
                               seed=20260920):
    """Choose one method from three C2-generation seeds, then score 50 utilities.

    Ranking never reads measured utility.  Within each model seed Full and SNR
    use the same intersection of valid query IDs across all four candidates;
    candidate means are then averaged with equal model-seed weight.  Exact
    ranking ties remain ties and their measured utilities are averaged.
    """
    methods = ("fmas_raw", "dtrak_T100", "das_native_sq", "ekfac_if")
    expected_seeds = (42, 123, 456)
    records = {}
    for row in rows:
        key = (row.get("method"), int(row.get("seed", -1)), int(row["query_id"]))
        if key in records:
            raise ValueError("duplicate E-DEL benchmark row")
        records[key] = row
    actual_seeds = tuple(sorted({key[1] for key in records if key[0] in methods}))
    if actual_seeds != expected_seeds:
        raise ValueError("E-DEL benchmark requires model seeds 42,123,456")

    per_seed, candidate_values = [], {"full": {m: [] for m in methods},
                                      "snr": {m: [] for m in methods}}
    for model_seed in expected_seeds:
        query_sets = []
        for method in methods:
            query_sets.append({q for (m, s, q), row in records.items()
                               if m == method and s == model_seed
                               and row.get("snr_lds") is not None})
        common = sorted(set.intersection(*query_sets)) if query_sets else []
        if not common:
            return dict(rule=SNR_RULE, status="unavailable", methods=list(methods),
                        reason=f"empty common valid query set for seed {model_seed}",
                        per_seed=per_seed, per_query=[], summary=[])
        means = {metric: {} for metric in ("full", "snr")}
        for method in methods:
            for metric, field in (("full", "full_lds"), ("snr", "snr_lds")):
                value = float(np.mean([records[method, model_seed, q][field] for q in common]))
                means[metric][method] = value
                candidate_values[metric][method].append(value)
        per_seed.append(dict(seed=model_seed, valid_queries=common,
                             n_valid=len(common), means=means))

    benchmark = {metric: {method: float(np.mean(values))
                          for method, values in candidates.items()}
                 for metric, candidates in candidate_values.items()}
    choices = {}
    for metric in ("full", "snr"):
        vector = np.asarray([benchmark[metric][method] for method in methods])
        best = np.max(vector)
        choices[metric] = [method for method, value in zip(methods, vector) if value == best]

    utilities = {}
    for row in utility_rows:
        if row["transform"] != "native":
            continue
        key = (row["method"], int(row["query"]), int(row["k"]))
        if key in utilities:
            raise ValueError("duplicate E-DEL native utility row")
        utilities[key] = float(row["utility"])
    required = {(method, query, k) for method in methods for query in range(50)
                for k in (300, 1000)}
    if set(utilities) != required or not np.isfinite(list(utilities.values())).all():
        raise ValueError("E-DEL requires exactly 400 finite native utility rows")
    per_query = []
    for query in range(50):
        for k in (300, 1000):
            full_u = float(np.mean([utilities[m, query, k] for m in choices["full"]]))
            snr_u = float(np.mean([utilities[m, query, k] for m in choices["snr"]]))
            per_query.append(dict(query_id=query, k=k,
                                  full_choice=choices["full"], snr_choice=choices["snr"],
                                  full_utility=full_u, snr_utility=snr_u,
                                  delta=snr_u - full_u,
                                  agreement=choices["full"] == choices["snr"]))
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, 50, (bootstrap_count, 50))
    summary = []
    for k in (300, 1000):
        selected = [row for row in per_query if row["k"] == k]
        delta = np.asarray([row["delta"] for row in selected])
        summary.append(dict(
            k=k, n_queries=50,
            full_utility=float(np.mean([row["full_utility"] for row in selected])),
            snr_utility=float(np.mean([row["snr_utility"] for row in selected])),
            delta_mean=float(delta.mean()),
            paired_query_ci95=np.quantile(delta[draw].mean(axis=1), [.025, .975]).tolist(),
            agreement=choices["full"] == choices["snr"]))
    return dict(rule=SNR_RULE, status="ok", methods=list(methods), per_seed=per_seed,
                benchmark_mean=benchmark, choices=choices, per_query=per_query,
                summary=summary, bootstrap_count=bootstrap_count, bootstrap_seed=seed)
