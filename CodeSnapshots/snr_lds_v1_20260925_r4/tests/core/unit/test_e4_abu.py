import torch

from balds.attribution.abu import (
    DiagonalNaturalGradient,
    EigenKroneckerNaturalGradient,
    apply_natural_update,
    finite_loss_difference,
    restored_model,
)


def test_natural_gradient_and_finite_quadratic_difference():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -2.0]]))
    original = model.weight.detach().clone()
    gradient = {"weight": torch.tensor([[2.0, 6.0]])}
    preconditioner = DiagonalNaturalGradient(
        {"weight": torch.tensor([[2.0, 3.0]])}, damping=0.0)

    with restored_model(model):
        apply_natural_update(model, gradient, preconditioner, step_size=0.1)
        assert torch.allclose(model.weight, original + torch.tensor([[0.1, 0.2]]))
    assert torch.equal(model.weight, original)

    # L(theta)=.5 theta^T H theta; finite difference and its small-step limit.
    theta = torch.tensor([1.0, -2.0], dtype=torch.float64)
    hessian = torch.diag(torch.tensor([2.0, 3.0], dtype=torch.float64))
    grad = hessian @ theta
    direction = torch.linalg.solve(hessian, grad)
    for gamma in (1e-1, 1e-4):
        before = 0.5 * theta @ hessian @ theta
        after = 0.5 * (theta + gamma * direction) @ hessian @ (theta + gamma * direction)
        exact = gamma * grad.dot(direction) + 0.5 * gamma**2 * direction @ hessian @ direction
        assert torch.allclose(after - before, exact)
    gamma = 1e-5
    delta = (0.5 * (theta + gamma * direction) @ hessian @ (theta + gamma * direction)
             - 0.5 * theta @ hessian @ theta)
    assert torch.allclose(delta / gamma, grad.dot(direction), rtol=2e-5, atol=2e-5)


def test_ekfac_basis_and_flip_difference_order():
    left = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    right = torch.eye(2)
    eigenvalues = torch.tensor([[2.0, 4.0], [5.0, 10.0]])
    gradient = torch.tensor([[2.0, 8.0], [5.0, 20.0]])
    preconditioner = EigenKroneckerNaturalGradient(
        {"w": (left, right, eigenvalues)}, damping=0.0)
    coordinates = left.T @ gradient @ right
    expected = left @ (coordinates / eigenvalues) @ right.T
    assert torch.allclose(preconditioner.apply({"w": gradient})["w"], expected)

    before = torch.tensor([[1.0, 10.0], [5.0, 2.0]])
    after = torch.tensor([[4.0, 11.0], [4.0, 7.0]])
    # max(after-before), not max(after)-max(before)
    assert torch.equal(finite_loss_difference(before, after, flip_mode="max"),
                       torch.tensor([3.0, 5.0]))


def test_two_query_order_and_exception_restore_are_invariant():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.5, -0.25]]))
    original = model.weight.detach().clone()
    preconditioner = DiagonalNaturalGradient({"weight": torch.ones_like(model.weight)},
                                             damping=0.5)
    gradients = [torch.tensor([[1.0, 2.0]]), torch.tensor([[-3.0, 1.0]])]

    def evaluate(order):
        out = {}
        for index in order:
            with restored_model(model, ["weight"]):
                apply_natural_update(model, {"weight": gradients[index]}, preconditioner,
                                     step_size=0.2)
                out[index] = model.weight.detach().clone()
            assert torch.equal(model.weight, original)
        return out

    assert all(torch.equal(evaluate([0, 1])[i], evaluate([1, 0])[i]) for i in (0, 1))
    try:
        with restored_model(model, ["weight"]):
            model.weight.data.add_(10)
            raise RuntimeError("query failed")
    except RuntimeError:
        pass
    assert torch.equal(model.weight, original)


def test_project_ekfac_relative_layer_inverse_and_normalization():
    import copy
    import pytest
    from balds.attribution.ekfac import EkfacFactors
    dtype = torch.float64
    left = torch.linalg.qr(torch.tensor([[1., 2.], [-2., 1.]], dtype=dtype)).Q
    right = torch.linalg.qr(torch.tensor([[3., 2.], [2., -3.]], dtype=dtype)).Q
    lam = torch.tensor([[1., 2.], [4., 8.]], dtype=dtype)
    bias_lam = torch.tensor([3., 6.], dtype=dtype)
    names = ["conv.weight", "conv.bias"]
    meta = dict(curvature_kind="empirical_fisher", parameter_names=names,
                normalization="mean_per_datum_gradient_outer_product")
    factors = EkfacFactors({"conv": dict(Q_A=right, Q_B=left, lam_w=lam, lam_b=bias_lam)}, meta)
    pre = EigenKroneckerNaturalGradient.from_factors(factors, names, damping=0.1)
    delta = 0.1 * torch.cat([lam.flatten(), bias_lam]).mean()
    grad = torch.tensor([1., 2., -3., 4.], dtype=dtype)
    gb = torch.tensor([2., -1.], dtype=dtype)
    basis = torch.kron(left.contiguous(), right.contiguous())
    H = basis @ torch.diag(lam.flatten()) @ basis.T + delta*torch.eye(4, dtype=dtype)
    Hb = left @ torch.diag(bias_lam) @ left.T + delta*torch.eye(2, dtype=dtype)
    actual = pre.apply({names[0]: grad.reshape(2, 1, 1, 2), names[1]: gb})
    assert torch.allclose(actual[names[0]].flatten(), torch.linalg.solve(H, grad), atol=1e-12)
    assert torch.allclose(actual[names[1]], torch.linalg.solve(Hb, gb), atol=1e-12)
    assert pre.layer_damping[names[0]] == pre.layer_damping[names[1]]
    assert pre.layer_damping[names[0]] == pytest.approx(float(delta), abs=1e-15)
    summed = copy.deepcopy(factors)
    summed.meta.update(normalization="sum_per_datum_gradient_outer_product", n_eigenvalue_draws=17)
    summed.layers["conv"]["lam_w"] *= 17
    summed.layers["conv"]["lam_b"] *= 17
    other = EigenKroneckerNaturalGradient.from_factors(summed, names, damping=0.1)
    assert torch.allclose(other.apply({names[0]: grad.reshape(2, 2), names[1]: gb})[names[1]], actual[names[1]])
    with pytest.raises(ValueError, match="parameter domain"):
        EigenKroneckerNaturalGradient.from_factors(factors, names[:1], damping=0.1)
    factors.meta["curvature_kind"] = "ggn_model"
    with pytest.raises(ValueError, match="not legacy GGN"):
        EigenKroneckerNaturalGradient.from_factors(factors, names, damping=0.1)


def test_empirical_eigenvalue_pass_matches_per_datum_autograd(monkeypatch):
    import balds.attribution.ekfac as ekfac
    from balds.models.flow import FlowMatching
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1, bias=True))
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = model
        def forward(self, x, t, labels):
            return self.net(x)
    wrapped = Adapter()
    images = torch.tensor([[[[1., 2.]]], [[[3., -1.]]], [[[0.5, 2.5]]]])
    def draws(images, process, gen, **kwargs):
        return images, torch.zeros(len(images)), torch.zeros(len(images), 2, 1, 2)
    monkeypatch.setattr(ekfac, "_draw_batch", draws)
    fitter = ekfac.EkfacFitter(wrapped, FlowMatching(), device="cpu", curvature_kind="empirical_fisher")
    bases = {"net.0": {"Q_A": torch.eye(1), "Q_B": torch.eye(2)}}
    kwargs = dict(epochs=1, seed=4, hflip=False)
    one = fitter.fit_eigenvalue_pass(images, None, bases, batch_size=1, **kwargs)
    many = fitter.fit_eigenvalue_pass(images, None, bases, batch_size=3, **kwargs)
    gw, gb = [], []
    for x in images:
        loss = model(x[None]).square().mean()
        w, b = torch.autograd.grad(loss, tuple(model.parameters()))
        gw.append(w.reshape(2, 1).square())
        gb.append(b.square())
    for values in (one, many):
        assert torch.allclose(values["net.0"]["lam_w"], torch.stack(gw).mean(0), rtol=2e-6)
        assert torch.allclose(values["net.0"]["lam_b"], torch.stack(gb).mean(0), rtol=2e-6)


def test_paired_losses_use_encoded_flip_and_restore_buffers():
    from balds.attribution.abu import measure_losses
    from balds.models.flow import FlowMatching
    from balds.data.base import ImageDataset
    class ZeroModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.))
            self.register_buffer("counter", torch.tensor(0.))
        def forward(self, x, t, labels):
            self.counter.add_(1)
            return x * self.weight
    model = ZeroModel()
    images = torch.tensor([[[[1., 2.]]], [[[3., 4.]]]])
    flipped = images + 7  # deliberately NOT a spatial flip of a latent
    labels = torch.zeros(2, dtype=torch.long)
    kwargs = dict(mc=2, seed=3, device="cpu")
    with restored_model(model):
        before = measure_losses(model, FlowMatching(), ImageDataset(images, labels),
                                 flip=True, flipped_images=flipped, **kwargs)
        after = measure_losses(model, FlowMatching(), ImageDataset(images, labels),
                                flip=True, flipped_images=flipped, **kwargs)
        alternate = measure_losses(model, FlowMatching(), ImageDataset(flipped, labels), **kwargs)
        assert torch.equal(before, after)
        assert torch.equal(before[:, 1], alternate)
    assert model.counter == 0
    assert model.training


def test_dead_ekfac_block_is_finite_zero_update():
    pre = EigenKroneckerNaturalGradient({"dead": (torch.eye(2), torch.eye(1), torch.zeros(2, 1))},
                                        damping=0.1, damping_mode="relative_layer")
    assert torch.equal(pre.apply({"dead": torch.zeros(2, 1)})["dead"], torch.zeros(2, 1))
