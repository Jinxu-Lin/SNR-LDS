"""Panel construction must preserve archive column IDs across score families."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_ddpm_column_identity_and_ab2_subset_count(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / 'tools' / 'prepare_benchmarks.py'
    spec = importlib.util.spec_from_file_location('prepare_benchmarks', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import balds.workflows.common as common
    import balds.data.base as base
    monkeypatch.setattr(common, '_load_ds', lambda *_: (None, SimpleNamespace(labels=[])))
    selected = list(range(100, 200))
    monkeypatch.setattr(base, 'balanced_query_indices', lambda *_: np.array(selected))
    panels = module.prepare(tmp_path, ['cifar2_das', 'artbench2_256'])
    ddpm = next(p for p in panels if p['panel_id'] == 'cifar2_das_s42_val')
    assert ddpm['query_ids'] == selected
    assert ddpm['response_query_ids'] == list(range(1000))
    assert ddpm['n_subsets'] == 64
    assert all(m['query_ids'] == selected for m in ddpm['methods'])
    das = next(m for m in ddpm['methods'] if m['method'] == 'das_native_sq')
    assert das['score_space'] == 'das-presquare'
    for panel in panels[2:]:
        assert panel['n_subsets'] == 32
        das = next(m for m in panel['methods'] if m['method'] == 'das_native_sq')
        assert das['scores'].endswith('/das_presquare_lambda1.npy')
        assert not Path(das['scores']).is_absolute()
    assert not list(tmp_path.rglob('*.npy'))  # Preparation never invents missing inputs.
