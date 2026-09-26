"""Scientific adapter tests; no model loading or real feature computation."""
import numpy as np
import pytest
import torch
from balds.artifacts.e3c import atomic
from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.workflows import ddpm_baselines as d


def test_four_checkpoint_mean_matches_reference_dot_and_cosine():
    gen = torch.Generator().manual_seed(3)
    train = [torch.randn(7, 5, generator=gen) * (i + 1) for i in range(4)]
    query = [torch.randn(3, 5, generator=gen) * (i + 2) for i in range(4)]
    for method in d.CHECKPOINT_METHODS:
        products = []
        for g, q in zip(train, query):
            if method == 'gas_T100':
                g = torch.nn.functional.normalize(g, dim=1)
                q = torch.nn.functional.normalize(q, dim=1)
            products.append((g @ q.T).numpy())
        expected = sum(products) / 4
        np.testing.assert_allclose(d.checkpoint_scores(train, query, method), expected, rtol=1e-6, atol=1e-6)
        assert not np.allclose(expected, products[-1])
    with pytest.raises(ValueError, match='all four'):
        d.checkpoint_scores(train[:-1], query[:-1], 'gas_T100')
    with pytest.raises(ValueError, match='all four'):
        d.checkpoint_scores(train, query, 'gas_T100', steps=tuple(reversed(d.STEPS)))


def test_manifest_retains_noncontinuous_labels_and_gt_never_overwrites(tmp_path):
    ids = list(range(99)) + [100]
    panels = [dict(panel_id=f'cifar2_das_s42_{t}', query_ids=ids,
                   response_query_ids=list(range(1000)), train_ids=list(range(5000)), n_subsets=64)
              for t in ('gen','val')]
    manifest = tmp_path / 'manifest.json'
    atomic(manifest, panels)
    selected, columns = d.panels_from_manifest(manifest)
    assert columns['val'][-1] == selected['val']['query_ids'][-1] == 100
    values = np.arange(2000).reshape(2,1000)
    class Store:
        def load(self, *args, **kwargs): return values
        def save(self, *args, **kwargs): self.saved = args
    store = Store()
    view = d.QueryGTView(store, columns, selected)
    spec = RunSpec(dataset='cifar2_das', process='ddpm', query_type='val')
    np.testing.assert_array_equal(view.load(K.GT_MATRIX, spec), values[:,ids])
    assert values.shape == (2,1000)
    view.save(K.SCORES_META, spec, {'best_lam': 0.05})
    assert store.saved[2]['query_ids'] == ids and store.saved[2]['query_indices'] == ids
    assert 'logical_query_ids' not in store.saved[2]
    selected['val']['response_query_ids'] = list(range(99))
    with pytest.raises(ValueError, match='response width'):
        view.load(K.GT_MATRIX, spec)


def test_root_relative_artifact_paths_reject_foreign_roots(tmp_path):
    assert d.relative_path(tmp_path, 'results/checkpoints') == 'results/checkpoints'
    assert d.relative_path(tmp_path, tmp_path / 'results/step.pt') == 'results/step.pt'
    with pytest.raises(ValueError, match='inside'):
        d.relative_path(tmp_path, '../foreign.pt')
