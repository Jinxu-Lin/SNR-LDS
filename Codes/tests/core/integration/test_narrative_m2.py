from copy import deepcopy
import numpy as np
import pytest

from balds.workflows import e3c
from balds.evaluation.narrative_m2 import pilot_bands, m2a_statistics, ordering_counts
from balds.artifacts.e3c import RepeatFiles, atomic


def test_stream_legacy_equality_and_new_distinct():
    cfg = e3c.resolve_config()
    old = deepcopy(cfg)
    old['repeats'] = 4
    streams = []
    for r in range(16):
        for method in e3c.METHODS:
            for role, track in [('train', 'shared'), ('query', 'gen'), ('query', 'val')]:
                s = e3c.mc_stream(cfg, r, method, role, track)
                if r < 4:
                    assert s == e3c.mc_stream(old, r, method, role, track)
                streams.append((s['seed'], s['phase']))
    assert len(set(streams)) == 96


def test_fixed_pilot_ties_variance_ratio_and_pair_denominators():
    pilot = np.ones((100, 1))
    bands = pilot_bands(pilot)
    assert [int((bands == k).sum()) for k in range(5)] == [1, 4, 15, 30, 50]
    assert bands[0, 0] == 0 and bands[-1, 0] == 4
    x = np.repeat(np.arange(100, dtype=float)[None, :, None], 16, axis=0)
    x[:, 50, 0] = 49
    masks = np.zeros((64, 100), bool)
    masks[np.arange(64), np.arange(64)] = True
    result = m2a_statistics(x, pilot, masks, np.arange(64)[:, None], [95])
    assert result['bands'][0]['varratio'] is None  # zero mean and zero variance
    assert all(row['varratio'] == 0 for row in result['bands'][1:])
    assert len(result['mean_deciles']) == 10
    assert len(result['ordering']) == 16 + 120
    assert all(r['pairs'] == 2016 for r in result['ordering'])
    assert result['summary']['between_repeats_agreement'] == 1
    import json
    json.dumps({k:v for k,v in result.items() if k != 'coordinates'}, allow_nan=False)
    ties = ordering_counts([0, 1, -1, 0], [1, 1, 1, 0])
    assert ties['strict_pairs'] == 2 and ties['agreement'] == .5
    assert ties['both_ties'] == 1 and ties['left_ties'] == 2



def test_variation_ratio_matches_final_figure_ratio_of_sums_and_rank_direction():
    from balds.evaluation.narrative_m2 import variance_ratio, variation_deciles
    assert variance_ratio(np.array([1., 10.]), np.array([4., 9.])) == 13. / 101.
    means = np.arange(1., 101.)[:, None]
    offsets = np.array([-1., 0., 1.])[:, None, None]
    scores = means[None] + offsets
    actual = variation_deciles(scores, [17])
    mean, variance = scores.mean(0), scores.var(0, ddof=1)
    order = np.argsort(np.abs(mean), axis=0, kind="stable")
    for band, row in enumerate(actual):
        ids = order[10 * band:10 * (band + 1), 0]
        expected = variance[ids, 0].sum() / np.square(mean[ids, 0]).sum()
        assert row['varratio_mean'] == pytest.approx(expected)
        assert row['rank_start'] == band * 10
    assert actual[0]['varratio_mean'] > actual[-1]['varratio_mean']
