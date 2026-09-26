import json

import numpy as np
import pytest
from scipy.stats import gaussian_kde

from balds.evaluation.background import (central_background, evaluate, evaluate_das,
                                         fit_background, kde_log_density, four_method_selection)
from balds.workflows.benchmark import evaluate_files, run_manifest


def fixture():
    t = np.r_[np.linspace(-.2, .2, 200), -4., -2., 1., 3.][:, None]
    keep = np.ones((8, len(t)), dtype=bool)
    for i in range(8):
        keep[i, np.arange(i, len(t), 8)] = False
    response = (1-keep.astype(float)) @ (t*t)
    return t, keep, response


def test_exact_kde_agrees_with_scipy():
    samples = np.array([-2., -.7, -.1, .2, .3, 1., 3.])
    points = np.linspace(-4, 4, 19)
    h = .43
    reference = gaussian_kde(samples, bw_method=h / samples.std(ddof=1))
    np.testing.assert_allclose(kde_log_density(points, samples, h, chunk=3),
                               reference.logpdf(points), rtol=1e-13, atol=1e-13)


def test_weighted_fit_recovers_known_background_above_one():
    x = np.linspace(-.5, .9, 101)
    mu, sigma, pi = .17, .31, 1.07
    log_density = np.log(pi) - np.log(sigma*np.sqrt(2*np.pi)) - .5*((x-mu)/sigma)**2
    params, error = central_background(x, log_density, allow_pi0_above_one=True)
    assert error is None
    np.testing.assert_allclose([params['mu0'], params['sigma0'], params['pi0']],
                              [mu, sigma, pi], rtol=1e-12)
    assert central_background(x, log_density)[1] == 'pi0_out_of_range'


def test_das_fits_signed_t_but_aggregates_native_square():
    t, keep, response = fixture()
    result = evaluate_das(t, keep, response)
    fit = fit_background(t[:, 0])
    assert fit['status'] == 'ok'
    assert np.any(fit['selected'] & (t[:, 0] < 0))
    np.testing.assert_array_equal(result['selected'][:, 0], fit['selected'])
    deleted = 1-keep.astype(float)
    np.testing.assert_allclose(result['full_prediction'], deleted @ (t*t))
    np.testing.assert_allclose(result['ba_prediction'], deleted @ np.where(result['selected'], t*t, 0))
    assert not np.allclose(result['full_prediction'], deleted @ t)
    assert not np.allclose(result['full_prediction'], deleted @ (t**4))
    # A fitted suppression value only selects; it never rescales retained scores.
    assert not np.allclose(result['ba_prediction'], deleted @ (t*t*fit['gamma'][:, None]))


def test_failure_is_missing_and_zero_selection_is_valid():
    t, keep, response = fixture()
    result = evaluate(np.column_stack([np.ones(len(t)), np.zeros(len(t))]), keep,
                      np.repeat(response, 2, axis=1))
    assert result['per_query'][0]['ba_lds'] is None
    assert result['per_query'][1]['ba_lds'] == 0
    assert result['summary']['n_valid'] == 1
    assert result['summary']['n_empty'] == 1
    assert result['summary']['full_lds'] == result['per_query'][1]['full_lds']


def test_explicit_fit_input_equals_das_api_and_files(tmp_path):
    t, keep, response = fixture()
    a = evaluate(t*t, keep, response, fit_scores=t)
    b = evaluate_das(t, keep, response)
    np.testing.assert_array_equal(a['ba_prediction'], b['ba_prediction'])
    for name, value in [('scores', t), ('masks', keep), ('responses', response)]:
        np.save(tmp_path / f'{name}.npy', value)
    evaluate_files(tmp_path/'scores.npy', tmp_path/'masks.npy', tmp_path/'responses.npy',
                   tmp_path/'out', score_space='das-presquare')
    metadata = json.loads((tmp_path/'out/summary.json').read_text())
    assert metadata['provenance']['aggregation_space'] == 'native_squared'
    assert metadata['provenance']['fit_space'] == 'signed_presquare'


def test_batch_query_identity_and_missing_method(tmp_path):
    t, keep, response = fixture()
    np.save(tmp_path/'scores.npy', np.column_stack([t, -t]))
    np.save(tmp_path/'masks.npy', keep)
    np.save(tmp_path/'responses.npy', np.column_stack([response, response]))
    ids = list(range(len(t)))
    item = dict(method='das_native_sq', scores='scores.npy', train_ids=ids,
                query_ids=[8, 4], score_space='das-presquare')
    panel = dict(panel_id='example', masks='masks.npy', response='responses.npy',
                 train_ids=ids, mask_train_ids=ids, query_ids=[4], response_query_ids=[8, 4],
                 methods=[item, dict(item, method='pending', scores='missing.npy')])
    manifest = tmp_path/'panels.json'
    manifest.write_text(json.dumps([panel]))
    run_manifest(manifest, tmp_path, tmp_path/'result')
    result = json.loads((tmp_path/'result/example/panel.json').read_text())
    assert result['complete'] is False
    assert result['rows'][0]['query_id'] == 4
    assert result['rows'][1]['status'] == 'missing_input'
    item['score_space'] = 'native'
    manifest.write_text(json.dumps([panel]))
    with pytest.raises(ValueError, match='paper DAS'):
        run_manifest(manifest, tmp_path, tmp_path/'wrong')


def test_deletion_ties_average_utility_and_preserve_common_queries():
    methods = ['fmas_raw', 'dtrak_T100', 'das_native_sq', 'ekfac_if']
    rows = [dict(method=m, query_id=q, full_lds=0., ba_lds=(None if q == 7 and i == 0 else float(i)))
            for q in range(50) for i, m in enumerate(methods)]
    utilities = [dict(method=m, query=q, k=k, transform='native', utility=i+1.)
                 for q in range(50) for i, m in enumerate(methods) for k in (300, 1000)]
    result = four_method_selection(rows, utilities, bootstrap_count=31)
    assert len(result['valid_queries']) == 49 and 7 not in result['valid_queries']
    for row in result['per_query']:
        assert row['full_utility'] == 2.5
        assert row['ba_utility'] == 4.
        assert row['full_choice'] == methods
    assert result['summary'][0]['delta_mean'] == 1.5


def test_partial_method_keeps_requested_coverage_denominator(tmp_path):
    t, keep, response = fixture()
    for name, array in [('scores', t), ('masks', keep),
                         ('responses', np.column_stack([response, response]))]:
        np.save(tmp_path/f'{name}.npy', array)
    ids = list(range(len(t)))
    panel = dict(panel_id='partial', masks='masks.npy', response='responses.npy',
                 train_ids=ids, query_ids=[7, 9], response_query_ids=[7, 9],
                 methods=[dict(method='example', scores='scores.npy', train_ids=ids,
                               query_ids=[7], score_space='native')])
    (tmp_path/'manifest.json').write_text(json.dumps([panel]))
    run_manifest(tmp_path/'manifest.json', tmp_path, tmp_path/'output')
    result = json.loads((tmp_path/'output/partial/panel.json').read_text())
    summary = result['summary'][0]
    assert summary['n_total'] == 2
    assert summary['n_evaluated'] == 1
    assert summary['n_missing_query'] == 1
    assert summary['n_failed'] == 0
