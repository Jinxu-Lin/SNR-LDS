"""The per-sample error weight ``e_n`` (DAS error term), computed EAGERLY.

``e_n`` is the per-sample direct-output prediction RMSE, averaged over timesteps
(``sqrt`` -> L2-normalize across timesteps -> mean). In the legacy code this was
computed lazily inside scoring; here it is a first-class featurization output.
Ports ``DTrakFM.compute_error_train`` (note its distinct per-timestep seeding
``seed + t_step``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_error_weight(model, process, dataset, *, seed: int = 42, T_error: int = 1000,
                         batch_size: int = 16, device: str = "cuda",
                         max_samples: Optional[int] = None, log_every: int = 500) -> np.ndarray:
    """Return ``(N,)`` float32 per-sample error weights ``e_n``."""
    N = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    timesteps = [(i + 1) / T_error for i in range(T_error)]
    errors = np.zeros((N, T_error), dtype=np.float32)
    model.eval()

    for bstart in range(0, N, batch_size):
        bend = min(bstart + batch_size, N)
        bs = bend - bstart
        x1 = torch.stack([dataset.images[i] for i in range(bstart, bend)]).to(device).float()
        labels = [dataset.labels[i] for i in range(bstart, bend)]
        labels = [l.item() if isinstance(l, torch.Tensor) else l for l in labels]
        class_vec = torch.tensor(labels, device=device, dtype=torch.long)

        for t_step, t_val in enumerate(timesteps):
            torch.manual_seed(seed + t_step)
            x0 = torch.randn_like(x1)
            # process-owned interpolation/target (float t keeps the CFM path
            # bit-identical to the legacy inline expression; DDPM q-samples)
            x_t = process.interpolate(x1, x0, t_val)
            v_target = process.target(x1, x0, t_val)
            t_vec = torch.full((bs,), t_val, device=device)
            v_pred = model(x_t, t_vec, class_vec).float()
            mse = F.mse_loss(v_pred, v_target, reduction="none").mean(dim=(1, 2, 3))
            errors[bstart:bend, t_step] = mse.cpu().numpy()
        if log_every and (bend % log_every < batch_size or bend == N):
            print(f"  error_weight: {bend}/{N}")

    errors = np.sqrt(errors)
    norms = np.linalg.norm(errors, axis=1, keepdims=True)
    return (errors / (norms + 1e-8)).mean(axis=1).astype(np.float32)
