import numpy as np
import torch

from balds.attribution.kernel import trak_kernel
from balds.attribution.parameter_weighting import (
    fit_weights,
    fixed_kernel,
    group_contributions,
    snr_topk_objective,
    weighted_scores,
)
from balds.attribution.featurize import GradFeaturizer, GroupedGradFeaturizer
from balds.models.flow import FlowMatching
from balds.data.base import ImageDataset


def test_group_contributions_recover_dtrak_and_group_readouts():
    torch.manual_seed(4)
    train = torch.randn(7, 5)
    grouped = torch.randn(3, 2, 5)
    query = grouped.sum(dim=1)
    ridge = 0.7

    contributions = group_contributions(train, grouped, ridge)
    ordinary = trak_kernel(train, query, ridge, device="cpu")
    assert np.allclose(contributions.sum(dim=2).numpy(), ordinary, rtol=2e-6, atol=2e-6)

    uniform = torch.full((2,), 0.5)
    assert torch.allclose(weighted_scores(contributions, uniform),
                          contributions.sum(dim=2) / 2, rtol=1e-6, atol=1e-6)
    assert torch.equal(weighted_scores(contributions, torch.tensor([1.0, 0.0])),
                       contributions[:, :, 0])

    direct_query = torch.einsum("qgp,g->qp", grouped, torch.tensor([0.25, 0.75]))
    inverse = fixed_kernel(train, ridge)
    direct = (direct_query @ (train @ inverse).T).T
    assert torch.allclose(weighted_scores(contributions, torch.tensor([0.25, 0.75])),
                          direct, rtol=2e-6, atol=2e-6)


def test_weight_objective_is_manual_topk_and_fitting_is_finite():
    contributions = torch.tensor([
        [[2.0, 0.0], [0.0, 1.0]],
        [[1.0, 1.0], [1.0, 0.0]],
        [[0.0, 2.0], [0.5, 0.5]],
    ])
    raw = torch.zeros(2, requires_grad=True)
    scores = contributions.mean(dim=2)
    expected = -(scores / scores.norm(dim=0, keepdim=True)).topk(2, dim=0).values.mean()
    actual = snr_topk_objective(contributions, raw, 2)
    assert torch.equal(actual, expected)
    actual.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()

    fit = fit_weights(contributions, epochs=4, lr=0.03, top_k=2,
                      weight_decay=0.1, scheduler="cosine", seed=11)
    assert np.isclose(fit.weights.sum(), 1.0)
    assert np.all(fit.weights >= 0)
    assert len(fit.losses) == 4 and np.isfinite(fit.losses).all()


class _TinyProcessModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(1, 2, 1)
        self.second = torch.nn.Conv2d(2, 1, 1)

    def forward(self, x, t, labels):
        return self.second(torch.tanh(self.first(x)))


def test_grouped_real_projector_path_sums_to_full_projection():
    torch.manual_seed(9)
    model = _TinyProcessModel()
    dataset = ImageDataset(torch.randn(3, 1, 2, 2), torch.zeros(3, dtype=torch.long))
    kwargs = dict(proj_dim=7, proj_seed=3, T=2, loss_type="msl2", batch_size=2,
                  device="cpu", projection="torch_chunked")
    full = GradFeaturizer(model, FlowMatching(), **kwargs).featurize(
        dataset, seed=13, log_every=0)
    grouped = GroupedGradFeaturizer(model, FlowMatching(), **kwargs).featurize_grouped(
        dataset, seed=13, log_every=0)
    assert torch.allclose(grouped.sum(dim=1), full, rtol=2e-6, atol=2e-6)
    streamed = torch.cat(list(GroupedGradFeaturizer(model, FlowMatching(), **kwargs).iter_grouped(
        dataset, seed=13, log_every=0)))
    assert torch.equal(streamed, grouped)


def test_streamed_fit_matches_full_query_objective_and_weights():
    values = torch.randn(13, 7, 5, generator=torch.Generator().manual_seed(8))
    kwargs = dict(epochs=5, lr=0.01, top_k=3, weight_decay=0.12)
    dense = fit_weights(values, **kwargs)
    streamed = fit_weights(lambda: (values[:, q:q+2] for q in range(0, 7, 2)),
                           shape=values.shape, **kwargs)
    np.testing.assert_allclose(streamed.weights, dense.weights, atol=1e-7, rtol=2e-6)
    np.testing.assert_allclose(streamed.losses, dense.losses, atol=1e-7, rtol=2e-6)
