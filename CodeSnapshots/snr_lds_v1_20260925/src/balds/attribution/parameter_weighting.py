"""Parameter-group weighting for projected D-TRAK.

The weights act on the query side only.  ``group_contributions`` deliberately
builds one training kernel and uses it for every group; fitting a kernel per
group is a different method.  This module contains no model or artifact I/O so
the matrix implementation can be checked independently of the orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def parameter_slices(named_parameters) -> tuple[list[str], list[slice]]:
    """One complete, non-overlapping group per trainable parameter tensor."""
    names, slices = [], []
    offset = 0
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        stop = offset + parameter.numel()
        names.append(str(name))
        slices.append(slice(offset, stop))
        offset = stop
    if not names:
        raise ValueError("parameter weighting needs at least one trainable parameter")
    return names, slices


def project_parameter_groups(flat_gradient: torch.Tensor, slices: list[slice], projector) -> torch.Tensor:
    """Project disjoint raw-gradient groups through one shared global projector.

    ``flat_gradient`` is already normalised using the norm of the *complete*
    gradient.  Each call masks all but one group, so even projection backends
    that do not expose their random matrix use exactly the same basis as the
    ordinary full-gradient path.
    """
    if flat_gradient.ndim != 2:
        raise ValueError("flat_gradient must have shape (batch, parameters)")
    covered = torch.zeros(flat_gradient.shape[1], dtype=torch.bool)
    outputs = []
    for group, slc in enumerate(slices):
        start, stop = int(slc.start or 0), int(slc.stop or flat_gradient.shape[1])
        if start < 0 or stop > flat_gradient.shape[1] or start >= stop:
            raise ValueError(f"invalid parameter slice {slc}")
        if bool(covered[start:stop].any()):
            raise ValueError("parameter groups overlap")
        covered[start:stop] = True
        masked = torch.zeros_like(flat_gradient)
        masked[:, start:stop] = flat_gradient[:, start:stop]
        outputs.append(projector.project(masked, model_id=0))
    if not bool(covered.all()):
        raise ValueError("parameter groups do not cover the complete gradient")
    return torch.stack(outputs, dim=1)  # (B, groups, projection)


def fixed_kernel(train_features: torch.Tensor, ridge: float, *,
                 normalize: str = "mean_abs", device: str = "cpu") -> torch.Tensor:
    """The single D-TRAK inverse kernel, including its historical scaling."""
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    comp = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    train = torch.as_tensor(train_features, dtype=torch.float32, device=comp)
    gram = train.T @ train
    inverse = torch.linalg.inv(gram + float(ridge) * torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device))
    if normalize == "mean_abs":
        scale = inverse.abs().mean()
        if not bool(torch.isfinite(scale)) or float(scale) == 0.0:
            raise ValueError("kernel mean-absolute scale is not positive and finite")
        inverse = inverse / scale
    elif normalize != "none":
        raise ValueError(f"unknown kernel normalization {normalize!r}")
    return inverse


def group_contributions(train_features: torch.Tensor, grouped_queries: torch.Tensor,
                        ridge: float, *, normalize: str = "mean_abs",
                        device: str = "cpu") -> torch.Tensor:
    """Return ``C`` with shape ``(N, Q, groups)`` from one fixed kernel."""
    comp = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    train = torch.as_tensor(train_features, dtype=torch.float32, device=comp)
    queries = torch.as_tensor(grouped_queries, dtype=torch.float32, device=comp)
    if train.ndim != 2 or queries.ndim != 3 or train.shape[1] != queries.shape[2]:
        raise ValueError("expected train (N,p) and grouped queries (Q,G,p)")
    inverse = fixed_kernel(train, ridge, normalize=normalize, device=comp)
    train_kernel = train @ inverse
    return torch.einsum("np,qgp->nqg", train_kernel, queries).cpu()


def weighted_scores(contributions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    contributions = torch.as_tensor(contributions, dtype=torch.float32)
    weights = torch.as_tensor(weights, dtype=torch.float32)
    if contributions.ndim != 3 or weights.shape != (contributions.shape[2],):
        raise ValueError("expected contributions (N,Q,G) and weights (G,)")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("weights must be finite and non-negative")
    return torch.einsum("nqg,g->nq", contributions, weights)


def snr_topk_objective(contributions: torch.Tensor, raw_weights: torch.Tensor,
                       top_k: int, *, eps: float = 1e-12) -> torch.Tensor:
    """Algorithm-1 objective: negative top-k mean after per-query L2 scaling."""
    if top_k <= 0 or top_k > contributions.shape[0]:
        raise ValueError(f"top_k must be in [1,{contributions.shape[0]}]")
    weights = torch.softmax(raw_weights, dim=0)
    scores = torch.einsum("nqg,g->nq", contributions, weights)
    normalised = scores / scores.norm(dim=0, keepdim=True).clamp_min(eps)
    return -normalised.topk(top_k, dim=0).values.mean()


@dataclass(frozen=True)
class WeightFitResult:
    weights: np.ndarray
    raw_weights: np.ndarray
    losses: tuple[float, ...]
    stopped_epoch: int


def fit_weights(contributions, *, epochs: int = 10, lr: float = 0.01,
                top_k: int = 10, weight_decay: float = 0.0,
                scheduler: str = "cosine", seed: int = 0,
                device: str = "cpu", shape=None) -> WeightFitResult:
    """Fit raw logits with AdamW; softmax is the only non-negativity map.

    ``weight_decay`` is AdamW decay on the raw logits.  No second L2 penalty is
    added, matching Algorithm 1 rather than silently combining two regularisers.
    """
    if epochs <= 0 or lr <= 0 or weight_decay < 0:
        raise ValueError("epochs/lr must be positive and weight_decay non-negative")
    comp = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    if callable(contributions):
        if shape is None or len(shape) != 3 or min(shape) <= 0:
            raise ValueError("streamed contributions need a positive (N,Q,G) shape")
        batches = contributions
    else:
        values = torch.as_tensor(contributions, dtype=torch.float32)
        shape = values.shape
        if values.ndim != 3:
            raise ValueError("contributions must be a finite (N,Q,G) tensor")
        batches = lambda: iter((values,))
    torch.manual_seed(int(seed))
    raw = torch.nn.Parameter(torch.zeros(shape[2], device=comp))
    optimiser = torch.optim.AdamW([raw], lr=float(lr), weight_decay=float(weight_decay))
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=int(epochs))
    elif scheduler == "none":
        sched = None
    else:
        raise ValueError("scheduler must be 'cosine' or 'none'")
    losses = []
    for _ in range(int(epochs)):
        optimiser.zero_grad(set_to_none=True)
        total_loss, seen = 0.0, 0
        for batch in batches():
            values = torch.as_tensor(batch, dtype=torch.float32, device=comp)
            if (values.ndim != 3 or values.shape[0] != shape[0]
                    or values.shape[2] != shape[2] or not bool(torch.isfinite(values).all())):
                raise ValueError("contributions must be finite matching (N,Q_chunk,G) tensors")
            # All N rows stay together: query-wise norm/top-k are unchanged.
            loss = snr_topk_objective(values, raw, int(top_k)) * (values.shape[1] / shape[1])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("parameter-weight objective became non-finite")
            loss.backward()
            total_loss += float(loss.detach().cpu())
            seen += values.shape[1]
        if seen != shape[1]:
            raise ValueError("contribution stream does not cover all queries")
        optimiser.step()
        if sched is not None:
            sched.step()
        losses.append(total_loss)
    weights = torch.softmax(raw.detach(), dim=0)
    return WeightFitResult(weights.cpu().numpy().astype(np.float32),
                           raw.detach().cpu().numpy().astype(np.float32),
                           tuple(losses), int(epochs))
