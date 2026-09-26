import numpy as np
import pytest
import json

import balds.evaluation.background as background
from balds.evaluation.background import (
    SNR_RULE, benchmark_method_selection, evaluate_snr, evaluate_snr_das,
    fit_snr_background, zero_mean_background,
)
from balds.workflows.controls import fixed_snr_controls
from balds.workflows.repeatability import _cross_mask, snr_repeatability_statistics
from balds.workflows.benchmark import run_manifest, verify_manifest


def arrays():
    score = np.r_[np.linspace(-.4, .4, 96), -5., -3., 3., 5.][:, None]
    keep = np.ones((8, len(score)), dtype=bool)
    for row in range(8):
        keep[row, row::8] = False
    response = (1.0 - keep) @ score
    return score, keep, response


def test_known_zero_mean_quadratic_and_no_linear_term():
    grid = np.linspace(-.8, .8, 101)
    A, C = 1.7, -2.0
    params, error = zero_mean_background(grid, A + C * grid**2)
    assert error is None
    assert set(params) == {'A', 'C', 'sigma_x'}
    np.testing.assert_allclose([params['A'], params['C'], params['sigma_x']],
                               [A, C, .5], rtol=1e-13, atol=1e-13)


def test_asymmetric_fit_keeps_grid_at_zero_and_never_evaluates_kde_at_n(monkeypatch):
    values = np.r_[np.linspace(-1, .2, 170), np.linspace(.2, 4, 37)]
    calls = []
    original = background.kde_log_density

    def checked(points, samples, bandwidth, *args, **kwargs):
        calls.append((len(points), len(samples)))
        return original(points, samples, bandwidth, *args, **kwargs)

    monkeypatch.setattr(background, 'kde_log_density', checked)
    fit = fit_snr_background(values)
    assert fit['status'] == 'ok'
    assert len(values) != 101 and calls == [(101, len(values))]
    assert fit['grid'][0] == pytest.approx(-fit['grid'][-1])
    assert fit['grid'][50] == pytest.approx(0)
    assert 'B' not in fit['params'] and 'mu0' not in fit['params'] and 'pi0' not in fit['params']


def test_strict_boundary_two_sided_and_positive_scale_invariance(monkeypatch):
    score, keep, response = arrays()
    base = evaluate_snr(score, keep, response)
    scaled = evaluate_snr(19.0 * score, keep, 19.0 * response)
    np.testing.assert_array_equal(base['selected_by_zeta'], scaled['selected_by_zeta'])
    np.testing.assert_allclose(base['snr_prediction_by_zeta'] * 19.0,
                               scaled['snr_prediction_by_zeta'])
    assert base['summary']['snr_lds'] == pytest.approx(scaled['summary']['snr_lds'])

    original = background.fit_snr_background
    def boundary(values, **kwargs):
        result = original(values, **kwargs)
        result['status'] = 'ok'
        result['reason'] = None
        result['x'] = np.array([-3., 3., -3.01, 3.01] + [0.] * (len(values) - 4))
        result['params'].update(sigma_x=1., sigma=result['rms'])
        result['selected'] = {z: np.abs(result['x']) > z for z in kwargs['zetas']}
        return result
    monkeypatch.setattr(background, 'fit_snr_background', boundary)
    chosen = background.evaluate_snr(np.ones((100, 1)), keep, response, zetas=(3.,))['selected'][:, 0]
    np.testing.assert_array_equal(chosen[:4], [False, False, True, True])


def test_fit_once_for_four_thresholds_and_das_square_once(monkeypatch):
    score, keep, _ = arrays()
    response = (1.0 - keep) @ (score * score)
    calls = 0
    original = background.fit_snr_background
    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(background, 'fit_snr_background', counted)
    result = evaluate_snr_das(score, keep, response)
    assert calls == 1
    deletion = 1.0 - keep.astype(float)
    np.testing.assert_allclose(result['full_prediction'], deletion @ (score * score))
    assert not np.allclose(result['full_prediction'], deletion @ score)
    assert not np.allclose(result['full_prediction'], deletion @ (score ** 4))
    selected = result['selected'][:, 0]
    np.testing.assert_allclose(result['snr_prediction'][:, 0],
                               (deletion @ np.where(selected[:, None], score * score, 0.0))[:, 0])


def test_zero_failure_and_constant_statuses_are_distinct():
    score, keep, response = arrays()
    result = evaluate_snr(np.column_stack([np.zeros(len(score)), np.ones(len(score))]),
                          keep, np.column_stack([response, response]))
    assert result['per_query'][0]['fit_status'] == 'all_zero_empty'
    assert result['per_query'][0]['snr_lds'] == 0
    assert result['per_query'][1]['fit_status'] == 'fit_failed'
    assert result['per_query'][1]['snr_lds'] is None
    assert result['summary']['n_all_zero'] == 1
    assert result['summary']['n_fit_failed'] == 1


def test_fixed_control_mask_is_fitted_only_on_raw(monkeypatch):
    score, keep, response = arrays()
    calls = 0
    original = background.fit_snr_background
    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr('balds.workflows.controls.fit_snr_background', counted)
    result = fixed_snr_controls(score, {'native': score, 'signed_square': np.sign(score)*score**2},
                                1.0-keep, response, head_percentages=(50,))
    assert calls == score.shape[1]
    assert {row['rule'] for row in result['rows']} == {'full', 'snr', 'head_50', 'snr_head_50'}


def test_benchmark_level_selection_differs_from_per_query_oracle():
    methods = ['fmas_raw', 'dtrak_T100', 'das_native_sq', 'ekfac_if']
    rows = []
    for seed in (42, 123, 456):
        for query in range(3):
            for index, method in enumerate(methods):
                # fmas wins the global mean; another method can win individual queries.
                value = 10. if method == 'fmas_raw' else (20. if index == query + 1 else 0.)
                rows.append(dict(method=method, seed=seed, query_id=query,
                                 full_lds=value, snr_lds=value))
    utilities = [dict(method=method, query=query, k=k, transform='native', utility=index + 1.)
                 for query in range(50) for index, method in enumerate(methods)
                 for k in (300, 1000)]
    result = benchmark_method_selection(rows, utilities, bootstrap_count=31)
    assert result['status'] == 'ok'
    assert result['choices']['full'] == ['fmas_raw']
    assert len(result['per_query']) == 100
    assert all(row['full_utility'] == 1. for row in result['per_query'])


def test_cross_reference_zero_sd_rules_and_full_prediction_variance():
    sample = np.zeros((8, 3, 1))
    sample[:, 0, 0] = 2
    selected, ratio, infinite, zero_zero = _cross_mask(sample)
    assert selected[0, 0] and np.isinf(ratio[0, 0]) and infinite == 1 and zero_zero == 2

    rng = np.random.default_rng(7)
    scores = rng.normal(size=(16, 20, 2))
    pilot = rng.normal(size=(20, 2))
    keep = rng.integers(0, 2, size=(8, 20)).astype(bool)
    response = rng.normal(size=(8, 2))
    result = snr_repeatability_statistics(scores, pilot, keep, response, [4, 9])
    assert result['rule'] == SNR_RULE
    assert len(result['repeat_lds']) == 16 * 2 * 3
    assert len(result['cross_reference']) == 16 * 2
    assert len(result['prediction_variance']) == 3 * 2


def test_atomic_query_resume_noncontinuous_ids_and_verify(tmp_path, monkeypatch):
    score, keep, response = arrays()
    np.save(tmp_path / 'scores.npy', np.column_stack([score, -score]))
    np.save(tmp_path / 'masks.npy', keep)
    np.save(tmp_path / 'response.npy', np.column_stack([response, response]))
    train_ids = list(range(len(score)))
    manifest = [dict(
        panel_id='p', masks='masks.npy', response='response.npy', train_ids=train_ids,
        mask_train_ids=train_ids, query_ids=[96, 100], response_query_ids=[96, 100],
        n_subsets=len(keep), methods=[dict(
            method='fmas_raw', scores='scores.npy', train_ids=train_ids,
            query_ids=[96, 100], score_space='native',
            identity={'seed': 42, 'dataset': 'tiny', 'query_type': 'val'})])]
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    output = tmp_path / 'out'
    run_manifest(path, tmp_path, output, save_fits=True)
    checkpoint = output / 'p/fmas_raw/queries/query_96.npz'
    before = checkpoint.read_bytes()

    monkeypatch.setattr('balds.workflows.benchmark.evaluate_snr',
                        lambda *a, **k: pytest.fail('completed resume refitted'))
    run_manifest(path, tmp_path, output, save_fits=True)
    assert checkpoint.read_bytes() == before
    verified = verify_manifest(path, tmp_path, output)
    assert verified['verdict'] == 'pass' and verified['verified_queries'] == 2


@pytest.mark.parametrize('scale', [1.0, 100.0])
def test_float32_das_verify_promotes_before_square_without_relaxing_tolerance(tmp_path, scale):
    score, keep, response = arrays()
    score = np.asarray(score * scale, dtype=np.float32)
    np.save(tmp_path / 'scores.npy', score)
    np.save(tmp_path / 'masks.npy', keep)
    np.save(tmp_path / 'response.npy', response)
    ids = list(range(len(score)))
    manifest = [dict(panel_id='p', masks='masks.npy', response='response.npy',
        train_ids=ids, mask_train_ids=ids, query_ids=[0], response_query_ids=[0],
        n_subsets=len(keep), methods=[dict(method='das_native_sq', scores='scores.npy',
        train_ids=ids, query_ids=[0], score_space='das-presquare', identity={})])]
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    out = tmp_path / 'out'
    run_manifest(path, tmp_path, out)
    assert verify_manifest(path, tmp_path, out)['verified_queries'] == 1
    checkpoint = out / 'p/das_native_sq/queries/query_0.npz'
    with np.load(checkpoint) as saved:
        values = {k: saved[k] for k in saved.files}
    values['full_prediction'][0] += 1e-6
    np.savez_compressed(checkpoint, **values)
    with pytest.raises(ValueError, match='SNR verification failed'):
        verify_manifest(path, tmp_path, out)
