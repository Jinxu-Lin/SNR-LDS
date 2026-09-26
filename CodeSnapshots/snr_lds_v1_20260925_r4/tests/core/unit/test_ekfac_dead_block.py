"""blockshrink damping on a dead block (Λ_l ≡ 0): finite, zero contribution,
and bit-identical elsewhere (ArtBench-2 all-NaN incident, 2026-09-07)."""
import torch
from balds.attribution.ekfac import EkfacFactors


def _factors(dead: bool):
    g = torch.Generator().manual_seed(0)
    layers = {
        "a": {"lam_w": torch.rand(3, 4, generator=g) + 0.1, "lam_b": torch.rand(3, generator=g) + 0.1},
        "b": {"lam_w": torch.rand(2, 5, generator=g) + 0.1, "lam_b": torch.rand(2, generator=g) + 0.1},
        "c": {"lam_w": torch.rand(2, 2, generator=g) + 0.1, "lam_b": None},
    }
    if dead:   # same draw stream either way, so the live blocks are identical
        layers["b"] = {"lam_w": torch.zeros(2, 5), "lam_b": torch.zeros(2)}
    return EkfacFactors(layers, meta={"layer_order": ["a", "b", "c"]})


N_A = 3 * 4 + 3
N_B = 2 * 5 + 2


def test_dead_block_gives_finite_zero_inverse_under_blockshrink():
    inv = _factors(dead=True).damped_inverse_flat(1e-2, "cpu", mode="blockshrink")
    assert torch.isfinite(inv).all()
    assert (inv[N_A:N_A + N_B] == 0).all()
    assert (inv[:N_A] > 0).all() and (inv[N_A + N_B:] > 0).all()


def test_live_blocks_bit_identical_with_and_without_dead_neighbor():
    live = _factors(dead=False).damped_inverse_flat(1e-2, "cpu", mode="blockshrink")
    with_dead = _factors(dead=True).damped_inverse_flat(1e-2, "cpu", mode="blockshrink")
    assert torch.equal(live[:N_A], with_dead[:N_A])
    assert torch.equal(live[N_A + N_B:], with_dead[N_A + N_B:])


def test_global_mode_unchanged_on_dead_block():
    inv = _factors(dead=True).damped_inverse_flat(1e-2, "cpu", mode="global")
    assert torch.isfinite(inv).all()
    assert torch.allclose(inv[N_A:N_A + N_B], torch.full((N_B,), 100.0))
