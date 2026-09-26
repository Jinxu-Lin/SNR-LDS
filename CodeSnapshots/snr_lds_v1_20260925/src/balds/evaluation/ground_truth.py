"""Ground-truth loss computation for LDS (ported from ``fmas/evaluation.py``).

For each (subset model, query) the GT influence is the model's average direct-output
loss over ``T_avg`` timesteps — the DAS-standard LDS ground truth. CFM and DDPM use
their respective forward processes and per-timestep deterministic seeding; ported
verbatim so the produced ``gt_losses`` match the legacy values bit-for-bit.

Timesteps are processed in chunks of ``target_batch // Q`` steps per forward call
instead of one at a time (2026-07-09): at Q<=1000 a single-timestep forward pass is
kernel-launch-bound, not compute-bound (~10-20% GPU util observed), so 1000
sequential Python-loop forward calls dominate wall-clock. Batching several
timesteps' worth of samples into one forward call is safe here because (a) the
model uses GroupNorm, which normalises per-sample and is unaffected by what else
shares the batch, and (b) every step's noise is still drawn via its own
``torch.manual_seed`` call, in the same step order as before, entirely on the CPU
RNG before any GPU work — only the number of forward passes changes, not the
noise realisation for any step.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def compute_query_losses(model, queries, labels, model_type: str, device: str, *,
                         process=None, schedule=None, T_avg: int = 1000, seed: int = 0,
                         target_batch: int = 2000) -> np.ndarray:
    """Per-query average diffusion/flow loss for one model; shape ``(Q,)``.

    ``model_type`` is ``"cfm"`` (velocity, ts=linspace(0.05,0.95)) or ``"ddpm"``
    (noise, integer timestep grid; ``schedule`` required). ``seed`` is the GT
    noise seed (the ζ realisation / e_seed). ``target_batch`` caps how many
    query x timestep samples go into one forward pass (``chunk = target_batch //
    Q`` timesteps at a time); raise it if GPU memory allows, lower it if it OOMs
    (memory scales ~linearly with chunk size).
    """
    if process is not None and getattr(process, "name", model_type) != model_type:
        raise ValueError(f"process instance does not match model_type={model_type!r}")
    if model_type == "ddpm" and schedule is None and process is not None:
        schedule = process._schedule(device)
    model.eval().to(device)
    Q = queries.shape[0]
    chunk = max(1, min(T_avg, target_batch // Q))
    per_sample = torch.zeros(Q)
    q_dev = queries.to(device)
    lb_dev = labels.to(device)

    if model_type == "cfm":
        ts = torch.linspace(0.05, 0.95, T_avg)
        for start in range(0, T_avg, chunk):
            end = min(start + chunk, T_avg)
            noises = []
            for step_idx in range(start, end):
                torch.manual_seed(seed * 1000 + step_idx)
                noises.append(torch.randn(Q, *queries.shape[1:]))
            K = end - start
            x0 = torch.cat(noises, dim=0).to(device, non_blocking=True)
            t_rep = ts[start:end].to(device).repeat_interleave(Q)
            q_rep = q_dev.repeat(K, 1, 1, 1)
            lb_rep = lb_dev.repeat(K)
            x_t = (1 - t_rep).view(-1, 1, 1, 1) * x0 + t_rep.view(-1, 1, 1, 1) * q_rep
            u_t = q_rep - x0
            v_pred = model(x_t, t_rep, lb_rep)
            sq = (v_pred - u_t).pow(2).mean(dim=(1, 2, 3)).cpu().view(K, Q)
            per_sample += sq.sum(dim=0)
    else:
        assert schedule is not None, "DDPM ground truth needs a DDPMSchedule"
        T = schedule.T
        ts = torch.linspace(0, T - 1, T_avg).long()
        for start in range(0, T_avg, chunk):
            end = min(start + chunk, T_avg)
            noises = []
            for step_idx in range(start, end):
                torch.manual_seed(seed * 1000 + ts[step_idx].item())
                noises.append(torch.randn(Q, *queries.shape[1:]))
            K = end - start
            noise = torch.cat(noises, dim=0).to(device, non_blocking=True)
            t_idx_rep = ts[start:end].to(device).repeat_interleave(Q)
            q_rep = q_dev.repeat(K, 1, 1, 1)
            lb_rep = lb_dev.repeat(K)
            x_t = schedule.q_sample(q_rep, t_idx_rep, noise=noise)
            eps_pred = model(x_t, t_idx_rep.float() / T, lb_rep)
            sq = (eps_pred - noise).pow(2).mean(dim=(1, 2, 3)).cpu().view(K, Q)
            per_sample += sq.sum(dim=0)

    return (per_sample / T_avg).numpy()
