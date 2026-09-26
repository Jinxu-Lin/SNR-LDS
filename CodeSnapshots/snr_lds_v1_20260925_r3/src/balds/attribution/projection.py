"""Random projection backends (the protocol Π axis).

``cuda_jl`` is the fast ``fast_jl`` CudaProjector used to produce the legacy
features (required for bit-exact feature parity); ``basic`` is a pure-torch
fallback that runs anywhere (no CUDA-compiler dependency) and is fine for
self-consistent runs where train and query features use the same projector.

New projection schemes register via ``@PROJECTORS.register`` (OCP) and become
selectable as a protocol axis value.
"""
from __future__ import annotations

from balds.schema.registry import PROJECTORS


@PROJECTORS.register("cuda_jl")
def _cuda_jl(*, grad_dim, proj_dim, seed, batch_size, device):
    from trak.projectors import CudaProjector, ProjectionType
    # fast_jl kernel capacities are 8/16/32, not the actual gradient batch.
    # Keep the input rows untouched; CudaProjector handles smaller batches.
    kernel_batch_size = 8 if batch_size <= 8 else 16 if batch_size <= 16 else 32
    return CudaProjector(grad_dim=grad_dim, proj_dim=proj_dim, seed=seed,
                         proj_type=ProjectionType.normal, max_batch_size=kernel_batch_size,
                         device=device)


@PROJECTORS.register("basic")
def _basic(*, grad_dim, proj_dim, seed, batch_size, device):
    from trak.projectors import BasicProjector, ProjectionType
    return BasicProjector(grad_dim=grad_dim, proj_dim=proj_dim, seed=seed,
                          proj_type=ProjectionType.normal, device=device,
                          max_batch_size=batch_size)


#: Elements per Gaussian slab — the INVARIANT this projector is budgeted on.
#: The slab is ``chunk x proj_dim`` floats, so holding ``chunk`` fixed made the
#: memory grow linearly with p: 2 GiB at p=1024, 8 GiB at p=4096, and 64 GiB at
#: p=32768, which simply OOMs (observed on a 24 GB card during the HP-p sweep).
#: Fixing the element count instead and deriving ``chunk = SLAB_ELEMENTS //
#: proj_dim`` bounds the slab at ~2 GiB for every p. It is a constant rather
#: than a tunable because the per-chunk seeding makes the chunk size part of the
#: random basis: two machines using different budgets would produce features
#: that are silently incomparable.
SLAB_ELEMENTS = 524_288 * 1024


class TorchChunkedProjector:
    """Matrix-free Gaussian JL projection in pure torch (no ``fast_jl`` needed).

    Projects the gradient in chunks of ``chunk`` parameters, generating each
    ``(chunk, proj_dim)`` Gaussian slab deterministically (seeded per chunk) and
    accumulating ``grads[:, slab] @ P`` — so the full ``(grad_dim, proj_dim)``
    matrix (585 GB at 36M×4096) is never materialised. Slower than ``fast_jl``
    but runs anywhere; self-consistent (train+query share the seed) so it yields
    valid attribution, though a *different* random basis than the legacy
    ``CudaProjector`` (won't match those cached features).

    ``chunk`` defaults to ``SLAB_ELEMENTS // proj_dim`` so slab memory is flat
    across projection dimensions; the basis is then a pure function of
    ``(seed, proj_dim, grad_dim)`` and reproduces anywhere.
    """

    def __init__(self, *, grad_dim, proj_dim, seed, chunk=None, device=None):
        self.grad_dim, self.proj_dim, self.seed = grad_dim, proj_dim, seed
        self.chunk = int(chunk) if chunk else max(1, SLAB_ELEMENTS // int(proj_dim))
        self.device = device

    def project(self, grads, model_id: int = 0):
        import math
        import torch
        B = grads.shape[0]
        out = torch.zeros(B, self.proj_dim, device=grads.device, dtype=torch.float32)
        gen = torch.Generator(device=grads.device)
        for ci, start in enumerate(range(0, self.grad_dim, self.chunk)):
            end = min(start + self.chunk, self.grad_dim)
            gen.manual_seed(self.seed * 1_000_003 + ci)
            P = torch.randn(end - start, self.proj_dim, generator=gen,
                            device=grads.device, dtype=torch.float32)
            out += grads[:, start:end].float() @ P
            del P
        return out * (1.0 / math.sqrt(self.proj_dim))


@PROJECTORS.register("torch_chunked")
def _torch_chunked(*, grad_dim, proj_dim, seed, batch_size, device, chunk=None):
    return TorchChunkedProjector(grad_dim=grad_dim, proj_dim=proj_dim, seed=seed,
                                 chunk=chunk, device=device)


def build_projector(kind: str, *, grad_dim: int, proj_dim: int, seed: int,
                    batch_size: int, device: str):
    """Build a projector. ``kind="auto"`` prefers ``cuda_jl``, falls back to ``basic``.

    Returns ``(projector, resolved_kind)``.
    """
    # auto: prefer the fast fast_jl projector; fall back to the matrix-free
    # pure-torch one (NOT "basic", which materialises a huge matrix and OOMs).
    order = ("cuda_jl", "torch_chunked") if kind == "auto" else (kind,)
    last_err: Exception | None = None
    for k in order:
        try:
            proj = PROJECTORS.get(k)(grad_dim=grad_dim, proj_dim=proj_dim, seed=seed,
                                     batch_size=batch_size, device=device)
            return proj, k
        except Exception as exc:  # fast_jl missing / build absent -> try next
            last_err = exc
    raise RuntimeError(f"no projector available for kind={kind!r}: {last_err}")
