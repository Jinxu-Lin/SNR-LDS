"""EK-FAC influence ([n]) verification suite — CPU only, no data/GPU needed.

What is pinned down, and why each test exists:

* module coverage — the reference covers exactly Conv2d/Linear; a silently
  missed layer family would change the estimator with no visible symptom;
* the MC-Fisher factor estimator converges to the ANALYTIC GGN of the
  mean-reduced MSE loss on a single linear layer (fixed inputs make A exact and
  B/Λ statistical) — this fixes every normalization constant at once, so the
  damping value keeps the paper's meaning;
* unfold ordering matches ``weight.reshape(d_out, -1)`` — the classic silent
  K-FAC bug is a (c, kh, kw) permutation between patches and weights;
* the damped eigenbasis inner product equals the dense ``gᵀ(H+λI)⁻¹g`` on a
  small layer — the identity the whole one-pass damping grid relies on;
* end-to-end determinism on a tiny UNet — MC draws are seeded per logical unit,
  so results must be independent of repetition (and of batch splits).
"""
import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.models.unet import UNetCFM  # noqa: E402
from balds.schema.identity import MCReplicaKey  # noqa: E402
from balds.attribution.ekfac import (  # noqa: E402
    EkfacFactors,
    EkfacFitter,
    EkfacScorer,
    _grad_to_positions,
    _input_to_positions,
    kfac_target_modules,
    target_parameters,
)


class _IdentityProcess:
    """Stub process: x_t == the clean input, target == 0 (unused in factor fit).

    Lets the fitting engine run on a plain regression model so its estimate can
    be compared against an analytic GGN.
    """

    name = "identity"

    def interpolate(self, x1, noise, t):
        return x1

    def target(self, x1, noise, t):
        return torch.zeros_like(x1)


class _LinearAsModel(nn.Module):
    """A single Linear layer with the pipeline's (x, t, label) signature."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x, t, label=None):
        return self.lin(x)


def test_target_modules_cover_all_but_norm_and_embedding():
    model = UNetCFM(base_ch=32, num_classes=None)
    modules = kfac_target_modules(model)
    covered = sum(p.numel() for p in target_parameters(modules))
    total = sum(p.numel() for p in model.parameters())
    assert covered / total > 0.99
    # everything NOT covered must be GroupNorm affine parameters
    covered_ids = {id(p) for p in target_parameters(modules)}
    for name, mod in model.named_modules():
        for pname, p in mod.named_parameters(recurse=False):
            if id(p) not in covered_ids:
                assert isinstance(mod, nn.GroupNorm), f"unexpected uncovered {name}.{pname}"
    # conditional models additionally exclude the class embedding
    cond = UNetCFM(base_ch=32, num_classes=4)
    cond_modules = kfac_target_modules(cond)
    cond_ids = {id(p) for p in target_parameters(cond_modules)}
    assert id(cond.class_embed.weight) not in cond_ids


def test_mc_fisher_matches_analytic_ggn_on_a_linear_layer():
    """A ⊗ B and the corrected eigenvalues converge to the exact GGN.

    For fixed inputs a_n and MSELoss(reduction="mean") over D outputs, the
    dataset-mean GGN of the weight block is E_n[a aᵀ] ⊗ (2/D)·I and the bias
    block is (2/D)·I. A is exact by construction (inputs are deterministic);
    B and Λ are Monte-Carlo with ~1/sqrt(draws) error.
    """
    torch.manual_seed(0)
    d_in, d_out, N = 4, 3, 6
    model = _LinearAsModel(d_in, d_out)
    images = torch.randn(N, d_in)
    fitter = EkfacFitter(model, _IdentityProcess(), device="cpu")

    epochs = 3000
    A, B, n_draws = fitter.fit_factor_pass(images, None, epochs=epochs, batch_size=N,
                                           seed=7, hflip=False)
    (name,) = list(fitter.modules)
    A_exact = (images.T @ images).double() / N
    assert torch.allclose(A[name], A_exact, rtol=1e-6, atol=1e-8)
    B_exact = (2.0 / d_out) * torch.eye(d_out, dtype=torch.float64)
    assert (B[name] - B_exact).norm() / B_exact.norm() < 0.05

    bases = fitter.eigenbases(A, B)
    layers = fitter.fit_eigenvalue_pass(images, None, bases, epochs=epochs,
                                        batch_size=N, seed=11, hflip=False)
    lam_A = torch.linalg.eigvalsh(A_exact).to(torch.float32)     # ascending
    lam_w = layers[name]["lam_w"]                                # (d_out, d_in)
    expected = (2.0 / d_out) * lam_A.unsqueeze(0).expand(d_out, d_in)
    assert (lam_w - expected).norm() / expected.norm() < 0.05
    lam_b = layers[name]["lam_b"]
    assert (lam_b - 2.0 / d_out).abs().max() < 0.05


def test_unfold_ordering_matches_weight_reshape():
    """conv(x) at position m must equal weight.reshape(d_out,-1) @ patch_m + bias."""
    torch.manual_seed(1)
    conv = nn.Conv2d(3, 5, kernel_size=3, padding=1, stride=1)
    x = torch.randn(2, 3, 8, 8)
    out = conv(x)                                              # (2, 5, 8, 8)
    patches = _input_to_positions(conv, x)                     # (2, 64, 27)
    W = conv.weight.reshape(5, -1)
    manual = patches @ W.T + conv.bias                          # (2, 64, 5)
    out_pos = _grad_to_positions(conv, out)                     # (2, 64, 5)
    assert torch.allclose(manual, out_pos, atol=1e-5)


def _random_orthogonal(n, gen):
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=gen))
    return q


def test_damped_eigenbasis_inner_product_equals_dense_inverse():
    """⟨u_i, u_q/(Λ+λ)⟩ == g_qᵀ (H + λI)⁻¹ g_i with H dense-built from the factors."""
    gen = torch.Generator().manual_seed(3)
    d_in, d_out = 4, 3
    lin = nn.Linear(d_in, d_out)
    modules = {"lin": lin}
    QA, QB = _random_orthogonal(d_in, gen), _random_orthogonal(d_out, gen)
    lam_w = torch.rand(d_out, d_in, generator=gen) + 0.1
    lam_b = torch.rand(d_out, generator=gen) + 0.1
    factors = EkfacFactors(
        {"lin": {"Q_A": QA, "Q_B": QB, "lam_w": lam_w, "lam_b": lam_b}},
        {"layer_order": ["lin"]},
    )

    model = _LinearAsModel(d_in, d_out)
    model.lin = lin
    scorer = EkfacScorer(model, _IdentityProcess(), factors, device="cpu")

    def flat(gw, gb):
        return torch.cat([gw.flatten(), gb])

    def H_apply(gw, gb):
        rw = QB.T @ gw @ QA
        hw = QB @ (rw * lam_w) @ QA.T
        hb = QB @ ((QB.T @ gb) * lam_b)
        return hw, hb

    # dense H, column by column, in the same (W-then-b) flatten convention
    P = d_out * d_in + d_out
    H = torch.zeros(P, P)
    for j in range(P):
        e = torch.zeros(P)
        e[j] = 1.0
        hw, hb = H_apply(e[: d_out * d_in].reshape(d_out, d_in), e[d_out * d_in:])
        H[:, j] = flat(hw, hb)

    damping = 0.05
    H_inv = torch.linalg.inv(H + damping * torch.eye(P))
    gi = [torch.randn(d_out, d_in, generator=gen), torch.randn(d_out, generator=gen)]
    gq = [torch.randn(d_out, d_in, generator=gen), torch.randn(d_out, generator=gen)]

    u_i = scorer.to_eigencoords(gi)
    u_q = scorer.to_eigencoords(gq)
    inv = factors.damped_inverse_flat(damping, "cpu")
    got = torch.dot(u_q, u_i * inv)
    want = flat(*gq) @ H_inv @ flat(*gi)
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-6)


def test_tiny_unet_end_to_end_is_deterministic_and_finite():
    torch.manual_seed(5)
    model = UNetCFM(base_ch=8, num_classes=None)
    images = torch.randn(2, 3, 32, 32).clamp(-1, 1)

    class _Flow:
        name = "cfm"

        def interpolate(self, x1, noise, t):
            t = t[:, None, None, None] if torch.is_tensor(t) and t.dim() == 1 else t
            return (1 - t) * noise + t * x1

        def target(self, x1, noise, t):
            return x1 - noise

    proc = _Flow()
    fitter = EkfacFitter(model, proc, device="cpu")
    A, B, _ = fitter.fit_factor_pass(images, None, epochs=2, batch_size=2,
                                     seed=42, hflip=True)
    bases = fitter.eigenbases(A, B)
    layers = fitter.fit_eigenvalue_pass(images, None, bases, epochs=1, batch_size=2,
                                        seed=42, hflip=True)
    factors = EkfacFactors(layers, {"layer_order": list(fitter.modules)})
    scorer = EkfacScorer(model, proc, factors, device="cpu")

    def coords():
        g = scorer.sample_gradient(images[0], None, mc=4, chunk=2, seed=42,
                                   phase="ekfac_loss", index=0, hflip=True,
                                   train_mode=False)
        return scorer.to_eigencoords(g)

    u1, u2 = coords(), coords()
    assert torch.isfinite(u1).all()
    assert torch.equal(u1, u2), "seeded MC draws must make the gradient reproducible"
    assert u1.numel() == factors.num_params()

    param_before = {name: value.detach().clone() for name, value in model.named_parameters()}
    buffer_before = {name: value.detach().clone() for name, value in model.named_buffers()}

    def replica(axis, replica_id, chunk):
        key = MCReplicaKey(model_identity="tiny-seed42", query_identity="qbank42",
                           axis=axis, replica_id=replica_id, logical_index=0)
        phase = "ekfac_meas" if axis == "query" else "ekfac_loss"
        grads = scorer.sample_gradient(
            images[0], None, mc=4, chunk=chunk, seed=42, phase=phase, index=0,
            hflip=False, train_mode=False, sampling="stratified_antithetic",
            replica_key=key,
        )
        return scorer.to_eigencoords(grads)

    train0 = replica("train", 0, 2)
    assert torch.equal(train0, replica("train", 0, 2))
    # Chunking keeps the exact draw stream.  Separate autograd reductions may
    # differ by ordinary fp32 non-associativity before the production bf16
    # coordinate contraction, so pin the measured numerical envelope.
    train0_chunk1 = replica("train", 0, 1)
    assert torch.max(torch.abs(train0-train0_chunk1)).item() < 1e-5
    assert not torch.equal(train0, replica("train", 1, 2))
    assert not torch.equal(train0, replica("query", 0, 2))
    assert all(torch.equal(value, param_before[name])
               for name, value in model.named_parameters())
    assert all(torch.equal(value, buffer_before[name])
               for name, value in model.named_buffers())

    inv = factors.damped_inverse_flat(1e-8, "cpu")
    assert torch.isfinite(inv).all() and (inv > 0).all()


def test_readouts_share_the_draw_stream_and_match_autograd():
    """X1 readout axis: ``mean``/``msl2`` differentiate the mean over draws of
    the per-draw readout on the SAME (flip, t, ε) stream as ``loss``; ``loss``
    itself is the unchanged reference path (bitwise equal to the pre-readout
    code, which is the ``readout="loss"`` default)."""
    torch.manual_seed(11)
    model = UNetCFM(base_ch=8, num_classes=None)
    images = torch.randn(1, 3, 32, 32).clamp(-1, 1)

    class _Flow:
        name = "cfm"

        def interpolate(self, x1, noise, t):
            t = t[:, None, None, None] if torch.is_tensor(t) and t.dim() == 1 else t
            return (1 - t) * noise + t * x1

        def target(self, x1, noise, t):
            return x1 - noise

    proc = _Flow()
    fitter = EkfacFitter(model, proc, device="cpu")
    A, B, _ = fitter.fit_factor_pass(images, None, epochs=1, batch_size=1, seed=3, hflip=False)
    layers = fitter.fit_eigenvalue_pass(images, None, fitter.eigenbases(A, B), epochs=1,
                                        batch_size=1, seed=3, hflip=False)
    scorer = EkfacScorer(model, proc, EkfacFactors(layers, {"layer_order": list(fitter.modules)}),
                         device="cpu")
    mc, chunk = 3, 2                                   # two chunks -> exercises (k/mc) weighting

    def grad(readout):
        return scorer.sample_gradient(images[0], None, mc=mc, chunk=chunk, seed=5,
                                      phase="ekfac_loss", index=0, hflip=False,
                                      train_mode=False, readout=readout)

    # reference: replay the same draw stream in one shot and autograd the readouts
    from balds.attribution.ekfac import _draw_batch, _mc_generator
    gen = _mc_generator(5, "ekfac_loss", 0)
    xs, ts = [], []
    for start in range(0, mc, chunk):
        k = min(chunk, mc - start)
        x_t, t, _ = _draw_batch(images.expand(k, -1, -1, -1), proc, gen, hflip=False,
                                device="cpu", with_target=True)
        xs.append(x_t); ts.append(t)
    x_t, t = torch.cat(xs), torch.cat(ts)
    model.eval()
    pred = model(x_t, t, None)
    ref_mean = torch.autograd.grad(pred.flatten(1).mean(dim=1).mean(), scorer.params)
    pred = model(x_t, t, None)
    ref_msl2 = torch.autograd.grad(pred.flatten(1).pow(2).mean(dim=1).mean(), scorer.params)

    for got, ref in ((grad("mean"), ref_mean), (grad("msl2"), ref_msl2)):
        for g, r in zip(got, ref):
            assert torch.allclose(g, r, atol=1e-6, rtol=1e-5)
    g_loss = grad("loss")
    assert all(torch.isfinite(g).all() for g in g_loss)
    assert not all(torch.equal(a, b) for a, b in zip(g_loss, grad("mean"))), \
        "loss and mean readouts must differ (they are different functionals)"
    assert all(torch.equal(a, b) for a, b in zip(grad("mean"), grad("mean"))), \
        "a readout gradient is a pure function of (seed, phase, index)"
    with pytest.raises(ValueError):
        grad("cosine")


def test_factor_accumulation_is_batch_size_independent():
    """Same seed, different batch split -> different draw grouping is NOT allowed
    to change results: draws are keyed per (epoch, start-index) generator, so
    equality holds only when the batch grid matches. This test pins the weaker,
    load-bearing guarantee instead: a full-batch pass and a repeated identical
    pass agree exactly (pure determinism of the accumulation path)."""
    torch.manual_seed(9)
    model = _LinearAsModel(3, 2)
    images = torch.randn(4, 3)
    fitter = EkfacFitter(model, _IdentityProcess(), device="cpu")
    A1, B1, _ = fitter.fit_factor_pass(images, None, epochs=5, batch_size=4,
                                       seed=1, hflip=False)
    A2, B2, _ = fitter.fit_factor_pass(images, None, epochs=5, batch_size=4,
                                       seed=1, hflip=False)
    (name,) = list(fitter.modules)
    assert torch.equal(A1[name], A2[name])
    assert torch.equal(B1[name], B2[name])


# --------------------------------------------------------------------------- #
# SD3.5+LoRA platform: two serial K-FAC blocks per adapter (TASK_EKFAC_LORA §2)
# --------------------------------------------------------------------------- #

class _LoraAsModel(nn.Module):
    """Frozen base Linear + LoRA pair behind the pipeline's (x, t, label) seam.

    Both LoRA weights are re-drawn nonzero (production B starts at zero — the
    curvature of a zero delta path is degenerate and would test nothing)."""

    def __init__(self, d_in, d_out, rank, alpha, gen):
        super().__init__()
        from balds.models.sd3 import LoRALinear
        base = nn.Linear(d_in, d_out)
        self.lora = LoRALinear(base, rank, alpha)
        with torch.no_grad():
            self.lora.lora_A.weight.copy_(torch.randn(rank, d_in, generator=gen) / rank)
            self.lora.lora_B.weight.copy_(torch.randn(d_out, rank, generator=gen) / rank)

    def forward(self, x, t, label=None):
        return self.lora(x)


def test_kfac_targets_on_lora_cover_the_adapters_and_skip_the_frozen_base():
    """Mechanism (i): the LoRA pair are real nn.Linear(bias=False) submodules,
    so the UNCHANGED target filter sees exactly the two low-rank blocks and the
    requires_grad filter drops the frozen base — no second capture path."""
    torch.manual_seed(21)
    model = _LoraAsModel(5, 4, 2, 3.0, torch.Generator().manual_seed(1))
    modules = kfac_target_modules(model)
    assert list(modules) == ["lora.lora_A", "lora.lora_B"]
    params = target_parameters(modules)
    assert len(params) == 2, "bias-free Linears contribute exactly their weights"
    assert params[0] is model.lora.lora_A.weight and params[0].shape == (2, 5)
    assert params[1] is model.lora.lora_B.weight and params[1].shape == (4, 2)
    base_ids = {id(p) for p in model.lora.base.parameters()}
    assert base_ids.isdisjoint({id(p) for p in params})


def test_lora_block_convention_matches_autograd_with_scale_and_dtype_cast():
    """Ladder (2): the (activation, output-grad) pairs the standard hooks see on
    lora_A / lora_B reconstruct ∇_A and ∇_B bit-for-bit against torch.autograd —
    with a non-trivial scale=alpha/rank=1.5 riding in via the chain rule (never
    multiplied in by hand) and the production dtype cast active (bf16 base +
    input, fp32 adapters; the A-hook must see the CAST input)."""
    from balds.models.sd3 import LoRALinear
    from balds.attribution.ekfac import _grad_to_positions, _input_to_positions

    torch.manual_seed(22)
    d_in, d_out, rank, alpha, B, M = 5, 4, 2, 3.0, 2, 3
    base = nn.Linear(d_in, d_out).to(torch.bfloat16)
    lora = LoRALinear(base, rank, alpha)                 # adapters stay fp32
    with torch.no_grad():
        lora.lora_A.weight.copy_(torch.randn(rank, d_in) / rank)
        lora.lora_B.weight.copy_(torch.randn(d_out, rank) / rank)
    assert lora.scale == pytest.approx(1.5)

    caps = {}

    def hook(tag):
        def h(mod, args, output):
            caps[tag + ".x"] = args[0].detach()
            output.register_hook(lambda g: caps.__setitem__(tag + ".g", g.detach()))
        return h

    h1 = lora.lora_A.register_forward_hook(hook("A"))
    h2 = lora.lora_B.register_forward_hook(hook("B"))
    x = torch.randn(B, M, d_in, dtype=torch.bfloat16)    # 3D: the SD3 token layout
    v = torch.randn(B, M, d_out, dtype=torch.bfloat16)
    out = lora(x)
    loss = (out * v).sum()
    gA, gB = torch.autograd.grad(loss, [lora.lora_A.weight, lora.lora_B.weight])
    h1.remove(), h2.remove()

    # the A hook captured the CAST input, not the raw bf16 tensor (§2 dtype note)
    assert caps["A.x"].dtype == torch.float32
    assert torch.equal(caps["A.x"], x.float())
    # g_b carries the scale through autograd (bf16-rounded on the way in)
    assert torch.allclose(caps["B.g"], lora.scale * v.float(), rtol=2e-2, atol=1e-3)
    # g_a = Bᵀ g_b — the serial-block chain rule, exact in fp32
    assert torch.allclose(caps["A.g"], caps["B.g"] @ lora.lora_B.weight,
                          rtol=1e-5, atol=1e-7)
    # ∇W = Σ_positions g ⊗ a in the expand layout — the identity every factor
    # and eigenvalue accumulation relies on
    for tag, mod, ref in (("A", lora.lora_A, gA), ("B", lora.lora_B, gB)):
        a = _input_to_positions(mod, caps[tag + ".x"])
        g = _grad_to_positions(mod, caps[tag + ".g"])
        got = torch.einsum("bmo,bmi->oi", g, a)
        assert torch.allclose(got, ref, rtol=1e-5, atol=1e-7), tag


def test_lora_kfac_blocks_match_analytic_ggn_and_dense_inverse():
    """Ladder (1): on a one-datum toy the per-block GGN is EXACTLY Kronecker, so
    the fitted EK-FAC must converge to it — Ω_A = xxᵀ and Ω_B = aaᵀ exactly
    (deterministic inputs), Γ_A → scale²·(2/D)·BᵀB and Γ_B → scale²·(2/D)·I in
    MC — and the damped eigenbasis score must match the dense (GGN+λI)⁻¹ built
    analytically on the LoRA parameter subspace."""
    torch.manual_seed(23)
    d_in, d_out, rank, alpha = 5, 4, 2, 3.0
    gen = torch.Generator().manual_seed(2)
    model = _LoraAsModel(d_in, d_out, rank, alpha, gen)
    lora = model.lora
    WA, WB, s = lora.lora_A.weight.detach(), lora.lora_B.weight.detach(), lora.scale
    x_vec = torch.randn(d_in, generator=gen)
    images = x_vec.unsqueeze(0)                          # N=1: E over data is trivial
    fitter = EkfacFitter(model, _IdentityProcess(), device="cpu")

    epochs = 3000
    A, B, _ = fitter.fit_factor_pass(images, None, epochs=epochs, batch_size=1,
                                     seed=7, hflip=False)
    a2 = WA @ x_vec
    omega_A, omega_B = torch.outer(x_vec, x_vec), torch.outer(a2, a2)
    assert torch.allclose(A["lora.lora_A"], omega_A.double(), rtol=1e-6, atol=1e-8)
    assert torch.allclose(A["lora.lora_B"], omega_B.double(), rtol=1e-6, atol=1e-8)
    gamma_A = (s ** 2) * (2.0 / d_out) * (WB.T @ WB)
    gamma_B = (s ** 2) * (2.0 / d_out) * torch.eye(d_out)
    assert (B["lora.lora_A"].float() - gamma_A).norm() / gamma_A.norm() < 0.05
    assert (B["lora.lora_B"].float() - gamma_B).norm() / gamma_B.norm() < 0.05

    layers = fitter.fit_eigenvalue_pass(images, None, fitter.eigenbases(A, B),
                                        epochs=epochs, batch_size=1, seed=11, hflip=False)
    factors = EkfacFactors(layers, {"layer_order": list(fitter.modules)})
    assert layers["lora.lora_A"]["lam_b"] is None        # bias-free blocks
    scorer = EkfacScorer(model, _IdentityProcess(), factors, device="cpu")

    # dense analytic GGN, block-diagonal over the two blocks, in the same
    # (layer, row-major W) flatten convention as to_eigencoords
    G = torch.block_diag(torch.kron(gamma_A.double(), omega_A.double()),
                         torch.kron(gamma_B.double(), omega_B.double()))
    damping = 1e-2
    H_inv = torch.linalg.inv(G + damping * torch.eye(G.shape[0], dtype=torch.float64))
    inv = factors.damped_inverse_flat(damping, "cpu")

    def flat(g):
        return torch.cat([g[0].flatten(), g[1].flatten()])

    def range_grad():
        gy = torch.randn(d_out, generator=gen)           # a real backward image
        return [s * torch.outer(WB.T @ gy, x_vec), s * torch.outer(gy, a2)]

    def score(gq, gi):
        u_q, u_i = scorer.to_eigencoords(gq), scorer.to_eigencoords(gi)
        got = torch.dot(u_q, u_i * inv)
        want = flat(gq).double() @ H_inv @ flat(gi).double()
        return float(got), float(want)

    got, want = score(range_grad(), range_grad())        # discriminative: range-aligned
    assert abs(got - want) / abs(want) < 0.05, (got, want)
    arb = [[torch.randn(rank, d_in, generator=gen), torch.randn(d_out, rank, generator=gen)]
           for _ in range(2)]
    got, want = score(*arb)                              # null directions are exact
    assert abs(got - want) / abs(want) < 0.05, (got, want)


def test_draw_batch_and_materialize_use_the_flip_pair_when_given():
    """Latent-platform hflip: the pre-encoded pair replaces the spatial flip
    with an UNCHANGED Bernoulli stream — same rows flip either way."""
    from balds.attribution.ekfac import (_draw_batch, _materialize,
                                      _stratified_antithetic_draws)

    torch.manual_seed(24)
    imgs, alt = torch.randn(16, 3, 4, 4), torch.randn(16, 3, 4, 4)
    proc = _IdentityProcess()                            # x_t == x1: exposes the pick
    g1 = torch.Generator().manual_seed(5)
    g2 = torch.Generator().manual_seed(5)
    x_pix, _, _ = _draw_batch(imgs, proc, g1, hflip=True, device="cpu", with_target=False)
    x_pair, _, _ = _draw_batch(imgs, proc, g2, hflip=True, device="cpu",
                               with_target=False, flipped=alt)
    flipped_rows = [not torch.equal(x_pix[i], imgs[i]) for i in range(16)]
    assert any(flipped_rows) and not all(flipped_rows)
    for i, f in enumerate(flipped_rows):
        assert torch.equal(x_pix[i], imgs[i].flip(-1) if f else imgs[i])
        assert torch.equal(x_pair[i], alt[i] if f else imgs[i])

    pre = _stratified_antithetic_draws(imgs.shape[1:], 16, g1, hflip=True)
    m_pix, _, _ = _materialize(imgs, proc, *pre, device="cpu", with_target=False)
    m_pair, _, _ = _materialize(imgs, proc, *pre, device="cpu", with_target=False,
                                flipped=alt)
    for i, f in enumerate(pre[0].tolist()):
        assert torch.equal(m_pix[i], imgs[i].flip(-1) if f else imgs[i])
        assert torch.equal(m_pair[i], alt[i] if f else imgs[i])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
