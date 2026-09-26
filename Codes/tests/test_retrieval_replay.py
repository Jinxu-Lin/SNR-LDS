import numpy as np
import pytest
from balds.workflows import retrieval_replay as r
from balds.workflows.source_assembly import report


def test_validation_first_tie_and_source_identity_are_enforced():
    metadata = dict(best_lam=1., lambda_sweep=[1., .1], per_lambda_ap_val={'0.1': .5, '1': .5})
    assert r._selection(metadata)[0] == 1.
    with pytest.raises(ValueError, match='first maximum'):
        r._selection({**metadata, 'best_lam': .1})
    r._assert_source({'identity': {'query_source': 'reviewed-v2'}}, 'reviewed-v2')
    with pytest.raises(ValueError, match='query source'):
        r._assert_source({'queries_sha256': 'old-v1'}, 'reviewed-v2')


def test_retrieval_reports_equal_concept_weight_and_original_query_labels():
    labels = np.repeat([0,1],100)
    meta = {'pairs': {'a': dict(host_label=0,concept_name='a',tier='fine',foreign_rows=[0]),
                      'b': dict(host_label=1,concept_name='b',tier='coarse',foreign_rows=[100])}}
    matrix = np.repeat(np.arange(200,dtype=float)[:,None],3,axis=1)
    matrix[0,:2] = 1000.
    matrix[100,2] = -1.
    queries = {'host_labels': np.array([1,0,1,0,1])}
    result = report(matrix, [1,3,4], queries, labels, meta)
    assert [x['query_id'] for x in result['per_query']] == [1,3,4]
    assert result['per_concept']['a']['ap_global'] == 1.
    assert result['per_concept']['b']['ap_global'] == 1./200.
    assert result['means']['ap_global'] == (1.+1./200.)/2
    assert result['means']['ap_global'] != np.mean([x['ap_global'] for x in result['per_query']])
    assert result['per_tier']['fine']['ap_global'] == 1.


def test_accepted_manifest_has_all13_methods_both_platforms_and_only_relative_paths():
    panels = r.archive_manifest()
    assert [p['process'] for p in panels] == ['cfm','ddpm']
    for panel in panels:
        assert len(panel['methods']) == 13
        assert set(x['method'] for x in panel['methods']) == set(r.METHOD_LABELS)
        for item in panel['methods']:
            for key in ('scores','meta','validation','val_score_root'):
                if key in item:
                    assert not item[key].startswith('/') and '..' not in item[key].split('/')
