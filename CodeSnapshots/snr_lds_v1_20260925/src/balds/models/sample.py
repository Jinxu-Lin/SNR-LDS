"""Inference samplers: Euler ODE for flow, DDIM and ancestral DDPM reverse."""
from __future__ import annotations

import torch


@torch.no_grad()
def euler_solve(model, x0, *, steps: int = 100, device: str = "cuda", class_label=None,
                capture_at=None):
    """Integrate ``dx/dt = v_θ(x, t)`` from ``t=0`` to ``1`` with left-endpoint Euler.

    ``capture_at`` (a set of step indices) additionally returns the intermediate
    states at those steps — the generation *journey* that Journey-TRAK attributes
    over. It is captured rather than reconstructed because the trajectory is the
    one thing a re-noised final image cannot reproduce: x_t on the path is not
    distributed like an independently noised x_0, which is the entire premise of
    the method. Returns ``x`` unchanged when ``capture_at`` is None, so the
    production sampling path is untouched.
    """
    x = x0.to(device)
    dt = 1.0 / steps
    if class_label is not None:
        class_label = class_label.to(device)
    want = set(capture_at) if capture_at is not None else None
    traj: dict[int, torch.Tensor] = {}
    for i in range(steps):
        if want is not None and i in want:
            traj[i] = x.detach().clone()
        t = torch.full((x.shape[0],), i * dt, device=device, dtype=x.dtype)
        v = model(x, t, class_label) if class_label is not None else model(x, t)
        x = x + v * dt
    if want is None:
        return x
    if steps in want:
        traj[steps] = x.detach().clone()
    return x, traj


@torch.no_grad()
def ddpm_sample(model, schedule, shape, *, class_label=None, device: str = "cuda",
                guidance_scale: float = 0.0, num_classes: int | None = None,
                x_T: torch.Tensor | None = None):
    """Full ancestral DDPM reverse sampling (time fed to the model as ``t_idx / T``)."""
    x = torch.randn(shape, device=device) if x_T is None else x_T.to(device)
    if tuple(x.shape) != tuple(shape):
        raise ValueError(f"x_T shape {tuple(x.shape)} does not match requested {tuple(shape)}")
    T = schedule.T
    for i in reversed(range(T)):
        t_idx = torch.full((shape[0],), i, device=device, dtype=torch.long)
        t_norm = t_idx.float() / T
        if guidance_scale > 0 and class_label is not None and num_classes is not None:
            eps_c = model(x, t_norm, class_label)
            eps_u = model(x, t_norm, torch.full_like(class_label, num_classes))
            eps = eps_u + guidance_scale * (eps_c - eps_u)
        elif class_label is not None:
            eps = model(x, t_norm, class_label)
        else:
            eps = model(x, t_norm)
        alpha, alpha_bar = schedule.alphas[i], schedule.alpha_bars[i]
        mean = (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * eps) / schedule.sqrt_alphas[i]
        x = mean + torch.sqrt(schedule.betas[i]) * torch.randn_like(x) if i > 0 else mean
    return x


@torch.no_grad()
def ddim_sample(model, schedule, x_T, *, steps: int, eta: float = 0.0,
                class_label=None, device: str = "cuda"):
    """DDIM sampling on an explicit :class:`DDPMSchedule`.

    The visited indices are ``linspace(0,T-1,steps).round()`` in descending
    order and the model receives ``idx/T``, exactly round-tripping
    :meth:`DDPM._t_index`.  ``eta=0`` is deterministic given ``x_T``.
    """
    steps = int(steps)
    if steps < 1 or steps > int(schedule.T):
        raise ValueError(f"steps must be in [1, {schedule.T}], got {steps}")
    if eta < 0:
        raise ValueError("eta must be non-negative")
    x = x_T.to(device)
    labels = None if class_label is None else class_label.to(device)
    indices = torch.linspace(0, schedule.T - 1, steps).round().long().tolist()
    for pos in reversed(range(steps)):
        idx = int(indices[pos])
        t_idx = torch.full((x.shape[0],), idx, device=device, dtype=torch.long)
        t_norm = t_idx.float() / schedule.T
        eps = model(x, t_norm, labels) if labels is not None else model(x, t_norm)
        alpha_bar = schedule.alpha_bars[idx]
        alpha_bar_prev = (schedule.alpha_bars[int(indices[pos - 1])]
                          if pos > 0 else torch.ones_like(alpha_bar))
        pred_x0 = (x - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        variance = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar) *
                    (1.0 - alpha_bar / alpha_bar_prev))
        sigma = float(eta) * torch.sqrt(torch.clamp(variance, min=0.0))
        direction = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma.square(),
                                           min=0.0)) * eps
        x = torch.sqrt(alpha_bar_prev) * pred_x0 + direction
        if pos > 0 and eta > 0:
            x = x + sigma * torch.randn_like(x)
    return x
