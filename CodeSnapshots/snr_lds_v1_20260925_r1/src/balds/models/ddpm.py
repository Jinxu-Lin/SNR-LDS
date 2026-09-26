"""DDPM process + linear schedule (the matched-architecture diffusion baseline).

Registered as the ``ddpm`` process so the DDPM path is no longer a config gap
(the legacy ``main.yaml`` lacked a ``ddpm_train`` block). Time enters the model
normalized to ``[0, 1]`` (``t_idx / T``), matching the legacy convention.
"""
from __future__ import annotations

import torch

from balds.schema.registry import PROCESSES


class DDPMSchedule:
    """DDPM schedule with cached ``alpha``-bar quantities.

    With neither ``betas`` nor ``alphas_cumprod`` this is the historical pixel
    schedule (linear 1e-4..0.02).  Latent diffusion platforms pass the exact
    public scheduler table instead; no approximation by the pixel schedule is
    allowed.
    """

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: str = "cpu", *, betas=None, alphas_cumprod=None) -> None:
        if betas is not None and alphas_cumprod is not None:
            raise ValueError("pass either betas or alphas_cumprod, not both")
        bars = None
        if alphas_cumprod is not None:
            bars = torch.as_tensor(alphas_cumprod, dtype=torch.float32, device=device)
            if bars.ndim != 1 or bars.numel() == 0:
                raise ValueError("alphas_cumprod must be a non-empty 1-D table")
            prev = torch.cat([torch.ones(1, device=bars.device, dtype=bars.dtype), bars[:-1]])
            betas = 1.0 - bars / prev
        if betas is None:
            table = torch.linspace(beta_start, beta_end, int(T), device=device)
        else:
            table = torch.as_tensor(betas, dtype=torch.float32, device=device).clone()
            if table.ndim != 1 or table.numel() == 0:
                raise ValueError("betas must be a non-empty 1-D table")
        if not bool(torch.isfinite(table).all()) or not bool(((table > 0) & (table < 1)).all()):
            raise ValueError("every beta must be finite and strictly between 0 and 1")
        self.T = int(table.numel())
        self.betas = table
        self.alphas = 1.0 - self.betas
        # Preserve an explicitly supplied cumulative table exactly. Rebuilding
        # it from beta ratios introduces an avoidable last-bit drift from the
        # diffusers scheduler that owns the platform.
        self.alpha_bars = bars.clone() if bars is not None else torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_alphas = torch.sqrt(self.alphas)

    def to(self, device):
        for a in ("betas", "alphas", "alpha_bars", "sqrt_alpha_bars",
                  "sqrt_one_minus_alpha_bars", "sqrt_alphas"):
            setattr(self, a, getattr(self, a).to(device))
        return self

    def q_sample(self, x0, t_idx, noise):
        sqrt_ab = self.sqrt_alpha_bars[t_idx][:, None, None, None]
        sqrt_1_ab = self.sqrt_one_minus_alpha_bars[t_idx][:, None, None, None]
        return sqrt_ab * x0 + sqrt_1_ab * noise


class DDPM:
    """Diffusion process. ``t`` is a continuous level in ``[0, 1]`` -> index ``round(t·T)``.

    ``interpolate`` returns ``x_t = q_sample``; ``target`` is the added noise ``ε``.
    """

    name = "ddpm"

    def __init__(self, T: int = 1000, *, schedule_betas=None,
                 alphas_cumprod=None) -> None:
        if schedule_betas is not None and alphas_cumprod is not None:
            raise ValueError("pass either schedule_betas or alphas_cumprod, not both")
        table = schedule_betas if schedule_betas is not None else None
        self._schedule_betas = (None if table is None
                                else torch.as_tensor(table, dtype=torch.float32).cpu().clone())
        self._alphas_cumprod = (None if alphas_cumprod is None else
                                torch.as_tensor(alphas_cumprod, dtype=torch.float32).cpu().clone())
        if self._schedule_betas is not None:
            T = int(self._schedule_betas.numel())
        elif self._alphas_cumprod is not None:
            T = int(self._alphas_cumprod.numel())
        self.T = int(T)
        self._sched: DDPMSchedule | None = None

    def _schedule(self, device) -> DDPMSchedule:
        if self._sched is None or self._sched.betas.device != torch.device(device):
            self._sched = DDPMSchedule(
                self.T, device=device, betas=self._schedule_betas,
                alphas_cumprod=self._alphas_cumprod)
        return self._sched

    @property
    def schedule_betas(self) -> torch.Tensor:
        """A CPU copy of the active table (useful for artifact provenance)."""
        return self._schedule("cpu").betas.detach().cpu().clone()

    def _t_index(self, t: torch.Tensor) -> torch.Tensor:
        # Exact inverse of the codebase's normalized-time convention ``idx / T``
        # (time_grid, ground_truth): round(t·T) recovers ``idx`` bit-exactly;
        # the old ``round(t·(T-1))`` drifted one step mid-schedule.
        return torch.clamp((t * self.T).round().long(), 0, self.T - 1)

    def interpolate(self, x1, noise, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x1.device, dtype=x1.dtype)
        if t.dim() == 0:
            t = t.expand(x1.shape[0])
        return self._schedule(x1.device).q_sample(x1, self._t_index(t), noise)

    def target(self, x1, noise, t):
        return noise

    def per_sample_loss(self, pred, x1, noise, t):
        return (pred - noise).pow(2).mean(dim=(1, 2, 3))

    def time_grid(self, n: int, kind: str) -> torch.Tensor:
        # Diffusion uses normalized-index grids; gt mirrors the legacy linspace(0,T-1)/T.
        if kind in ("featurize", "error"):
            return torch.arange(n, dtype=torch.float32) / n
        if kind == "gt":
            return torch.linspace(0, self.T - 1, n).round() / self.T
        raise KeyError(f"unknown time_grid kind '{kind}'")


PROCESSES.add("ddpm", DDPM())
