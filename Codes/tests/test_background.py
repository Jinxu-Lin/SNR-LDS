import json
import numpy as np
import pytest
from scipy.stats import gaussian_kde
from balds.evaluation.background import kde_log_density
from balds.workflows.benchmark import run_manifest

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
