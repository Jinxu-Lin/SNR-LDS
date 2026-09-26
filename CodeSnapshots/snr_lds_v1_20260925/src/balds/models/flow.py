"""Flow Matching (rectified-flow / independent-coupling CFM) process.

Predicts the velocity ``v_θ``. Independent Gaussian coupling ``x0 ~ N(0, I)``
(the legacy "OT-CFM" is actually independent-coupling — the OT variant is a
declared ablation, not the default). Owns the three legacy timestep grids so
featurization / error / ground-truth never drift apart.
"""
from __future__ import annotations

import torch

from balds.schema.registry import PROCESSES


class FlowMatching:
    """``x_t = (1-t)·noise + t·x1``; target velocity ``u_t = x1 - noise``."""

    name = "cfm"

    def interpolate(self, x1, noise, t):
        # Float ``t`` keeps the exact legacy scalar arithmetic (bit-parity for
        # the featurize/error paths); tensor ``t`` broadcasts per-sample.
        if torch.is_tensor(t):
            if t.dim() == 1:
                t = t[:, None, None, None]
        return (1 - t) * noise + t * x1

    def target(self, x1, noise, t):
        return x1 - noise

    def per_sample_loss(self, pred, x1, noise, t):
        u_t = self.target(x1, noise, t)
        return (pred - u_t).pow(2).mean(dim=(1, 2, 3))

    def time_grid(self, n: int, kind: str) -> torch.Tensor:
        """Legacy grids, kept verbatim by ``kind`` (parity; unify later, opt-in)."""
        if kind == "featurize":
            return torch.arange(n, dtype=torch.float32) / n          # i/T_grad  -> [0, .9]
        if kind == "error":
            return (torch.arange(n, dtype=torch.float32) + 1) / n     # (i+1)/T_error -> (0, 1]
        if kind == "gt":
            return torch.linspace(0.05, 0.95, n)                      # GT loss grid
        raise KeyError(f"unknown time_grid kind '{kind}'")


PROCESSES.add("cfm", FlowMatching())
