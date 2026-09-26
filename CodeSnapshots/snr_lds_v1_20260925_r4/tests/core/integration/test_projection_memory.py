"""PROJECTOR SLAB BUDGET (HP-p blocker found 2026-08-14).

``TorchChunkedProjector`` materialises a ``(chunk, proj_dim)`` Gaussian slab per
step. ``chunk`` used to be a fixed constant, so slab memory grew linearly with
the projection dimension: 2 GiB at p=1024, 8 GiB at p=4096, 64 GiB at p=32768 —
the last two simply OOM a 24 GB card, which is where the HP-p sweep died.

The invariant is the slab's element count, not the chunk: ``chunk`` is derived
so every p costs the same memory. It is a constant rather than a tunable because
the per-chunk seeding folds the chunk size into the random basis — two machines
on different budgets would produce silently incomparable features.

Run (CPU only, tiny grad_dim):  python Codes/tests/integration/test_projection_memory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.attribution.projection import SLAB_ELEMENTS, TorchChunkedProjector

GRAD_DIM = 35_750_000          # the real UNetCFM parameter count


def test_slab_memory_is_flat_across_projection_dims():
    sizes = {}
    for p in (1024, 4096, 16384, 32768):
        proj = TorchChunkedProjector(grad_dim=GRAD_DIM, proj_dim=p, seed=0)
        sizes[p] = proj.chunk * p * 4 / 2**30            # GiB per slab
        assert sizes[p] <= 2.01, f"p={p}: slab {sizes[p]:.2f} GiB exceeds the budget"
        assert proj.chunk >= 1
    assert max(sizes.values()) - min(sizes.values()) < 0.01, sizes
    # the old fixed chunk would have asked for 64 GiB at p=32768
    assert 524288 * 32768 * 4 / 2**30 > 60
    print(f"  slab flat at ~{max(sizes.values()):.2f} GiB for p in {sorted(sizes)} "
          f"(fixed chunk would have needed 64 GiB at p=32768)")


def test_small_p_keeps_the_historical_chunk():
    """p=1024 lands exactly on the previous constant, so nothing changes there."""
    assert TorchChunkedProjector(grad_dim=GRAD_DIM, proj_dim=1024, seed=0).chunk == 524288
    assert SLAB_ELEMENTS == 524288 * 1024
    print("  p=1024 keeps chunk=524288 (the pre-fix constant)")


def test_projection_is_correct_and_deterministic():
    """Chunked accumulation must equal the dense projection it stands in for."""
    g, p = 4096, 64
    grads = torch.randn(3, g)
    proj = TorchChunkedProjector(grad_dim=g, proj_dim=p, seed=7, chunk=1000)
    a = proj.project(grads)
    b = proj.project(grads)
    assert torch.equal(a, b), "same projector must be deterministic"
    # rebuild the dense matrix the same way the chunks do, and compare
    import math
    dense = torch.zeros(g, p)
    gen = torch.Generator(device=grads.device)
    for ci, start in enumerate(range(0, g, 1000)):
        end = min(start + 1000, g)
        gen.manual_seed(7 * 1_000_003 + ci)
        dense[start:end] = torch.randn(end - start, p, generator=gen)
    expected = (grads @ dense) * (1.0 / math.sqrt(p))
    assert torch.allclose(a, expected, atol=1e-4), (a - expected).abs().max()
    print("  chunked projection == the dense projection it stands in for")


def test_explicit_chunk_still_honoured():
    proj = TorchChunkedProjector(grad_dim=GRAD_DIM, proj_dim=4096, seed=0, chunk=8192)
    assert proj.chunk == 8192, "an explicit chunk must win (basis-compat escape hatch)"
    print("  explicit chunk overrides the derived one")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} projection-memory tests passed")
