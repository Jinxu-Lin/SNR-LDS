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


def test_fixed_pilot_ties_rep_zero_variance_and_pair_denominators():
    pilot = np.ones((100, 1))
    bands = pilot_bands(pilot)
    assert [int((bands == k).sum()) for k in range(5)] == [1, 4, 15, 30, 50]
    assert bands[0, 0] == 0 and bands[-1, 0] == 4
    x = np.repeat(np.arange(100, dtype=float)[None, :, None], 16, axis=0)
    x[:, 50, 0] = 49
    masks = np.zeros((64, 100), bool)
    masks[np.arange(64), np.arange(64)] = True
    result = m2a_statistics(x, pilot, masks, np.arange(64)[:, None], [95])
    assert np.isnan(result['coordinates']['rep'][49:51]).all()
    assert np.isinf(result['coordinates']['rep'][0]).all()
    assert len(result['ordering']) == 16 + 120
    assert all(r['pairs'] == 2016 for r in result['ordering'])
    assert result['summary']['between_repeats_agreement'] == 1
    import json
    json.dumps({k:v for k,v in result.items() if k != 'coordinates'}, allow_nan=False)
    ties = ordering_counts([0, 1, -1, 0], [1, 1, 1, 0])
    assert ties['strict_pairs'] == 2 and ties['agreement'] == .5
    assert ties['both_ties'] == 1 and ties['left_ties'] == 2

