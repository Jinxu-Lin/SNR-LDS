"""Portable R16 stage and independent-pilot analysis for the paper."""
from pathlib import Path
import numpy as np
from . import e3c
from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.artifacts.e3c import ReadOnlyInputs, atomic, load_score, read_json
from balds.artifacts.codecs import NpyCodec, NpzCodec
from balds.evaluation.narrative_m2 import m2a_statistics
from balds.evaluation.background import SNR_RULE, fit_snr_background
from balds.evaluation.lds import compute_lds, predicted_influence


def _cross_mask(values, threshold=3.0):
    mean = values.mean(axis=0)
    sd = values.std(axis=0, ddof=1)
    ratio = np.zeros_like(mean)
    np.divide(np.abs(mean), sd, out=ratio, where=sd > 0)
    ratio[(sd == 0) & (mean != 0)] = np.inf
    # zero mean / zero SD remains zero and is not selected.
    return ratio > threshold, ratio, int(np.sum((sd == 0) & (mean != 0))), int(np.sum((sd == 0) & (mean == 0)))


def snr_repeatability_statistics(scores, pilot, masks, response, query_ids,
                                 *, zeta=3.0):
    """Frozen-pilot SNR and fixed 8/8 cross-reference appendix diagnostics."""
    values = np.asarray(scores, dtype=np.float64)
    pilot = np.asarray(pilot, dtype=np.float64)
    masks = np.asarray(masks)
    response = np.asarray(response, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 16 or not np.isfinite(values).all():
        raise ValueError("SNR repeatability requires exactly 16 finite repeats")
    repeats, n, q = values.shape
    if (pilot.shape != (n, q) or masks.ndim != 2 or masks.shape[1] != n
            or response.shape != (len(masks), q) or len(query_ids) != q
            or not np.isin(masks, [0, 1]).all() or not np.isfinite(pilot).all()
            or not np.isfinite(response).all()):
        raise ValueError("score/pilot/mask/response identities or shapes differ")
    deletion = 1.0 - masks.astype(np.float64)
    pilot_selected = np.zeros((n, q), dtype=bool)
    pilot_sigma = np.full(q, np.nan)
    pilot_snr = np.full((n, q), np.nan)
    pilot_rows = []
    for column, query_id in enumerate(query_ids):
        fit = fit_snr_background(pilot[:, column], zetas=(zeta,))
        if fit['status'] != 'fit_failed':
            pilot_selected[:, column] = fit['selected'][float(zeta)]
            sigma = fit['params'].get('sigma', 0.0)
            pilot_sigma[column] = sigma
            if sigma > 0:
                pilot_snr[:, column] = np.abs(pilot[:, column]) / sigma
        pilot_rows.append(dict(query_id=int(query_id), status=fit['status'],
                               reason=fit['reason'], sigma=(None if not np.isfinite(pilot_sigma[column])
                                                            else float(pilot_sigma[column])),
                               retained=int(pilot_selected[:, column].sum())))

    repeat_sd = values.std(axis=0, ddof=1)
    sd_over_sigma = np.divide(repeat_sd, pilot_sigma[None, :],
                              out=np.full_like(repeat_sd, np.nan),
                              where=np.isfinite(pilot_sigma[None, :]) & (pilot_sigma[None, :] > 0))
    residual = values - values.mean(axis=0, keepdims=True)
    group_summary = []
    for label, support in (("retained", pilot_selected), ("excluded", ~pilot_selected)):
        finite_ratio = sd_over_sigma[support & np.isfinite(sd_over_sigma)]
        residual_values = residual[:, support].ravel()
        group_summary.append(dict(
            group=label, n_coordinates=int(support.sum()),
            repeat_sd_over_sigma=(np.quantile(finite_ratio, [.25, .5, .75]).tolist()
                                  if len(finite_ratio) else None),
            centered_residual=(np.quantile(residual_values, [.025, .25, .5, .75, .975]).tolist()
                               if len(residual_values) else None)))

    per_repeat = []
    subset_predictions = {"full": [], "retained": [], "excluded": []}
    for repeat in range(repeats):
        supports = {"full": np.ones((n, q), bool), "retained": pilot_selected,
                    "excluded": ~pilot_selected}
        for group, support in supports.items():
            prediction = deletion @ np.where(support, values[repeat], 0.0)
            subset_predictions[group].append(prediction)
            lds = compute_lds(response, prediction)[0]
            for column, query_id in enumerate(query_ids):
                per_repeat.append(dict(repeat=repeat, query_id=int(query_id), group=group,
                                       lds=float(lds[column])))
    # Variance is computed after forming every complete subset prediction; no
    # independence approximation over training coordinates is used.
    prediction_variance = []
    for group, predictions in subset_predictions.items():
        variance = np.var(np.asarray(predictions), axis=0, ddof=1)
        for column, query_id in enumerate(query_ids):
            prediction_variance.append(dict(group=group, query_id=int(query_id),
                                            mean_variance=float(variance[:, column].mean())))

    cross = []
    for learn, evaluate_ids in ((range(0, 8), range(8, 16)),
                                (range(8, 16), range(0, 8))):
        support, ratio, infinite, zero_zero = _cross_mask(values[list(learn)], zeta)
        direction = f"{learn.start}-{learn.stop - 1}_to_{evaluate_ids.start}-{evaluate_ids.stop - 1}"
        for repeat in evaluate_ids:
            prediction = deletion @ np.where(support, values[repeat], 0.0)
            lds = compute_lds(response, prediction)[0]
            for column, query_id in enumerate(query_ids):
                cross.append(dict(direction=direction, repeat=repeat, query_id=int(query_id),
                                  lds=float(lds[column]), selected=int(support[:, column].sum()),
                                  infinite_ratio=infinite, zero_over_zero=zero_zero))
    return dict(rule=SNR_RULE, zeta=float(zeta), query_ids=list(map(int, query_ids)),
                pilot=pilot_rows, pilot_snr=pilot_snr, selected=pilot_selected,
                repeat_sd=repeat_sd, repeat_sd_over_sigma=sd_over_sigma,
                group_summary=group_summary, repeat_lds=per_repeat,
                prediction_variance=prediction_variance, cross_reference=cross)


def summarize(config, data_root, output_root, pilot_manifest):
    """Analyze completed repeats using explicitly supplied independent pilots.

    Manifest: ``{"fmas_raw": {"gen": "path.npy", "val": "path.npy"},
    "dtrak_T100": {...}}``. Paths are relative to the manifest directory.
    Each pilot must have N rows and the 16 configured query columns in order.
    Missing repeats are reported, never filled or replaced.
    """
    cfg, root = config, Path(output_root)
    manifest = Path(pilot_manifest).resolve()
    pilots = read_json(manifest)
    package = read_json(root / 'package.json')
    e3c.verify_package(cfg, data_root, package)
    states = e3c.status(cfg, root)
    complete = [r['repeat_id'] for r in states if r['complete']]
    report = {'complete_repeats': complete, 'expected_repeats': cfg['repeats'],
              'complete': len(complete) == cfg['repeats'], 'methods': {}}
    if len(complete) < 2:
        atomic(root / 'analysis/summary.json', report)
        return report
    inputs = ReadOnlyInputs(data_root)
    spec = RunSpec(dataset=cfg['dataset'], seed=cfg['model_seed'],
                   process=cfg['process'], conditional=cfg['conditional'])
    masks = np.asarray(inputs.load(K.SUBSET_MASKS, spec))[:64]
    for method in e3c.METHODS:
        report['methods'][method] = {}
        for track in e3c.TRACKS:
            ids = cfg['selection'][track]
            scores = np.stack([load_score(root / f'repeat_{r}', method, track,
                       e3c.repeat_identity(cfg, package, r), e3c.score_ids(cfg, track),
                       (cfg['n_train'], len(ids))) for r in complete])
            pilot_path = Path(pilots[method][track])
            if not pilot_path.is_absolute():
                pilot_path = manifest.parent / pilot_path
            pilot = NpyCodec().load(str(pilot_path))
            response = np.asarray(inputs.load(K.GT_MATRIX, spec.with_(query_type=track)))[:64, ids]
            stats = m2a_statistics(scores, pilot, masks, response, ids)
            coordinates = stats.pop('coordinates')
            snr = snr_repeatability_statistics(scores, pilot, masks, response, ids)
            snr_coordinates = {key: snr.pop(key) for key in
                               ('pilot_snr', 'selected', 'repeat_sd', 'repeat_sd_over_sigma')}
            out = root / 'analysis' / method / track
            atomic(out / 'coordinates.npz', coordinates, NpzCodec())
            atomic(out / 'statistics.json', stats)
            atomic(out / 'snr_coordinates.npz', snr_coordinates, NpzCodec())
            atomic(out / 'snr_statistics.json', snr)
            report['methods'][method][track] = dict(legacy=stats['summary'],
                                                    snr_status='produced',
                                                    snr_rule=SNR_RULE)
    atomic(root / 'analysis/summary.json', report)
    return report
