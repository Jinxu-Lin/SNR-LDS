import importlib.util
import json
from pathlib import Path

import pytest


def test_query_intervals_preserve_seed_sets_and_track_pairing():
    spec = importlib.util.spec_from_file_location(
        'summarize', Path(__file__).parents[1] / 'tools/summarize_benchmarks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cells = [dict(rows=[dict(query_id=0, x=0.), dict(query_id=1, x=1.)]),
             dict(rows=[dict(query_id=0, x=1.), dict(query_id=1, x=0.)])]
    assert module._query_bootstrap(cells, 'x', shared_queries=True) == [.5, .5]
    lo, hi = module._query_bootstrap(cells, 'x', shared_queries=False)
    assert lo < .5 < hi
    # Each model has a valid estimate even when their valid query IDs differ.
    disjoint = [dict(rows=[dict(query_id=0, x=.2)]), dict(rows=[dict(query_id=1, x=.6)])]
    assert module._query_bootstrap(disjoint, 'x') == pytest.approx([.4, .4])
    assert module._query_bootstrap(disjoint, 'x', shared_queries=True) == pytest.approx([.4, .4])


def test_snr_all_failed_group_keeps_schema_and_paired_full_ci(tmp_path):
    spec = importlib.util.spec_from_file_location(
        'summarize', Path(__file__).parents[1] / 'tools/summarize_benchmarks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pid = 'example_s42_gen'
    methods = []
    for method, good in [('a_failed', False), ('z_valid', True)]:
        methods.append(dict(method=method, identity=dict(
            dataset='example', query_type='gen', seed=42)))
        directory = tmp_path / pid / method
        directory.mkdir(parents=True)
        summary = dict(rule='snr_zero_mean_gaussian_v1', n_total=2,
                       n_valid=int(good), n_failed=2-int(good),
                       full_lds=.2 if good else None, snr_lds=.4 if good else None)
        rows = [dict(query_id=0, status='ok' if good else 'fit_failed',
                     full_lds=.2, snr_lds=.4 if good else None),
                dict(query_id=1, status='fit_failed', full_lds=.99, snr_lds=None)]
        (directory / 'summary.json').write_text(json.dumps(summary))
        (directory / 'per_query.json').write_text(json.dumps(rows))
    (tmp_path / pid / 'panel.json').write_text(json.dumps(
        {'summary': [dict(method=m['method']) for m in methods]}))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([dict(panel_id=pid, methods=methods)]))
    result = module.summarize(manifest, tmp_path, tmp_path / 'out')
    failed, valid = result['records']
    assert result['rule'] == 'snr_zero_mean_gaussian_v1'
    assert failed['snr_lds'] is None and failed['n_seeds'] == 0
    assert 'ba_lds' not in failed
    assert valid['full_lds_query_ci95'] == pytest.approx([.2, .2])
    assert len((tmp_path / 'out/summary.csv').read_text().splitlines()) == 3
    assert not (tmp_path / 'out/summary.csv.tmp').exists()


def test_seed_summary_shared_validation_and_stale_output(tmp_path):
    spec = importlib.util.spec_from_file_location('summarize', Path(__file__).parents[1]/'tools/summarize_benchmarks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    panels = []
    for seed in [42, 123, 456]:
        pid = f'example_s{seed}_val'
        methods = []
        for method in ['fmas_raw', 'pixel_dot']:
            methods.append(dict(method=method, identity=dict(dataset='example', query_type='val', seed=seed)))
            directory = tmp_path / pid / method
            directory.mkdir(parents=True)
            metric = .2 if method == 'pixel_dot' else .1 + (seed != 42)*.2
            summary = dict(n_total=2, n_valid=1, n_failed=1, n_empty=0, full_lds=metric, snr_lds=metric+.1)
            (directory/'summary.json').write_text(json.dumps(summary))
            (directory/'per_query.json').write_text(json.dumps([dict(query_id=3, status='ok', full_lds=metric, snr_lds=metric+.1)]))
        panels.append(dict(panel_id=pid, methods=methods))
        # The final model's stale outputs must not be used.
        status = 'missing_input' if seed == 456 else 'ok'
        (tmp_path/pid/'panel.json').write_text(json.dumps({'summary':[dict(method=m['method'], status=status, **(dict(n_total=5, n_evaluated=2, n_missing_query=3) if m['method']=='fmas_raw' else {})) for m in methods]}))
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(panels))
    result=module.summarize(manifest,tmp_path,tmp_path/'out')
    fmas,pixel=result['records']
    assert fmas['n_seeds']==2 and fmas['n_valid']==2 and fmas['n_total']==10
    assert fmas['n_missing_query']==6
    assert fmas['full_lds']==pytest.approx(.2)
    assert fmas['full_lds_std']==pytest.approx(.2/(2**.5))
    assert pixel['n_seeds']==1 and pixel['n_total']==2
    assert len(result['missing'])==2
    path=tmp_path/'example_s123_val/pixel_dot/per_query.json'
    rows=json.loads(path.read_text()); rows[0]['snr_lds']=.9
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='shared validation'):
        module.summarize(manifest,tmp_path,tmp_path/'bad')
