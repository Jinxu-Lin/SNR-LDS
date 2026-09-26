"""The generative-process seam (CFM vs DDPM).

A :class:`GenerativeProcess` owns the forward corruption, the regression target,
the per-sample loss, and — crucially — the timestep grids. Centralising the
grids here fixes the legacy bug where featurization, the error term, and the
ground-truth loss each used a *different*, hard-coded grid.

``torch`` is only referenced in type annotations, kept out of runtime imports so
the contracts package stays importable without a GPU stack.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class GenerativeProcess(Protocol):
    """A diffusion/flow process used for both training and attribution.

    Concrete processes (``cfm``, ``ddpm``) register themselves in
    :data:`balds.schema.registry.PROCESSES`.
    """

    name: str

    def interpolate(self, x1: "torch.Tensor", noise: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
        """Corrupted state ``x_t`` (CFM: ``(1-t)·noise + t·x1``; DDPM: ``q_sample``)."""
        ...

    def target(self, x1: "torch.Tensor", noise: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
        """Regression target the network predicts (CFM velocity ``x1-noise``; DDPM ``noise``)."""
        ...

    def per_sample_loss(
        self, pred: "torch.Tensor", x1: "torch.Tensor", noise: "torch.Tensor", t: "torch.Tensor"
    ) -> "torch.Tensor":
        """Per-sample MSE between prediction and target, shape ``(B,)``."""
        ...

    def time_grid(self, n: int, kind: str) -> "torch.Tensor":
        """Timestep grid of size ``n`` for a given consumer.

        ``kind`` is one of ``"featurize"``, ``"error"``, ``"gt"``. Initially each
        returns the *legacy* grid verbatim (parity); unifying them is a later,
        opt-in, re-snapshotted change rather than a silent one.
        """
        ...
