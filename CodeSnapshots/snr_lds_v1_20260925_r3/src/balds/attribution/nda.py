"""Nonparametric Data Attribution patch matching with global normalization."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def extract_patches(images: torch.Tensor, patch_size: int, *, mask_value: float = 1e3,
                    dilation: int = 1) -> torch.Tensor:
    """Same-padded patches, with official large-value boundary masking."""
    if images.ndim != 4 or patch_size <= 0:
        raise ValueError("images must be (B,C,H,W) and patch_size positive")
    pad = ((int(patch_size) - 1) * int(dilation)) // 2
    padded = F.pad(images, (pad, pad, pad, pad), value=0.0)
    patches = F.unfold(padded, kernel_size=patch_size, dilation=dilation).transpose(1, 2)
    mask = torch.ones_like(images)
    mask = F.pad(mask, (pad, pad, pad, pad), value=0.0)
    valid = F.unfold(mask, kernel_size=patch_size, dilation=dilation).transpose(1, 2) > 0
    return patches + float(mask_value) * (~valid).to(patches.dtype)


def patch_projection(patch_size: int, projected_size: int, *, device=None,
                     dtype=torch.float32) -> torch.Tensor:
    """Official patch-internal block projection (not whole-image resize)."""
    k, s = int(patch_size), int(projected_size)
    if k < s or k % s:
        raise ValueError("projected_size must divide patch_size")
    ratio = k // s
    matrix = torch.zeros(s * s, k * k, device=device, dtype=dtype)
    for row in range(s):
        for col in range(s):
            indices = [(row * ratio + di) * k + (col * ratio + dj)
                       for di in range(ratio) for dj in range(ratio)]
            # The official implementation uses 1/2 for its ratio-2 setting.
            matrix[row * s + col, indices] = 1.0 / ratio
    return matrix


def _project(patches: torch.Tensor, channels: int, patch_size: int,
             projected_size: int | None) -> torch.Tensor:
    if projected_size is None:
        return patches
    matrix = patch_projection(patch_size, projected_size, device=patches.device,
                              dtype=patches.dtype)
    shaped = patches.view(*patches.shape[:-1], channels, patch_size * patch_size)
    return torch.matmul(shaped, matrix.T).flatten(-2)


def _logits(query_patches, train_patches, variance: float):
    if variance <= 0:
        raise ValueError("noise variance must be positive")
    qnorm = query_patches.square().sum(dim=1, keepdim=True)
    tnorm = train_patches.square().sum(dim=2).unsqueeze(0)
    cross = torch.einsum("pd,nld->pnl", query_patches, train_patches)
    distance = (qnorm.unsqueeze(-1) + tnorm - 2.0 * cross).clamp_min(0.0)
    return -distance / (2.0 * float(variance))


def patch_match_scores(query: torch.Tensor, train: torch.Tensor, *, signal_scale: float,
                       noise_std: float, patch_size: int, mask_value: float = 1e3,
                       spatial_topk: int | None = None, train_chunk: int | None = None,
                       query_patch_chunk: int | None = None,
                       projected_size: int | None = None) -> torch.Tensor:
    """Score all train images for one query using a global two-pass log-sum-exp.

    Chunking changes only evaluation order.  The denominator for every query
    patch still spans every train image and every candidate train location.
    """
    query = torch.as_tensor(query)
    train = torch.as_tensor(train, device=query.device, dtype=query.dtype)
    if query.ndim == 3:
        query = query.unsqueeze(0)
    if query.shape[0] != 1 or train.ndim != 4 or query.shape[1:] != train.shape[1:]:
        raise ValueError("query must be (C,H,W), train (N,C,H,W), with matching geometry")
    channels = int(train.shape[1])
    qpatch = extract_patches(query, patch_size, mask_value=mask_value)[0]
    qpatch = _project(qpatch, channels, patch_size, projected_size)
    n = int(train.shape[0])
    tc = int(train_chunk or n)
    pc = int(query_patch_chunk or qpatch.shape[0])
    scores = torch.zeros(n, device=train.device, dtype=torch.float64)
    for p0 in range(0, qpatch.shape[0], pc):
        qp = qpatch[p0:p0 + pc]
        global_lse = torch.full((len(qp),), -torch.inf, device=train.device,
                                dtype=torch.float64)
        for n0 in range(0, n, tc):
            part = extract_patches(train[n0:n0 + tc] * float(signal_scale), patch_size,
                                   mask_value=mask_value)
            part = _project(part, channels, patch_size, projected_size)
            logits = _logits(qp, part, float(noise_std) ** 2).double()
            global_lse = torch.logaddexp(global_lse, torch.logsumexp(logits.flatten(1), dim=1))
        # Recompute one train chunk at a time for the normalized pass. Keeping
        # pass-1 logits would make "chunking" retain the full N*locations
        # tensor and defeat the memory contract.
        for n0 in range(0, n, tc):
            part = extract_patches(train[n0:n0 + tc] * float(signal_scale), patch_size,
                                   mask_value=mask_value)
            part = _project(part, channels, patch_size, projected_size)
            logits = _logits(qp, part, float(noise_std) ** 2).double()
            weights = torch.exp(logits - global_lse[:, None, None])
            if spatial_topk is not None:
                k = min(int(spatial_topk), weights.shape[2])
                if k <= 0:
                    raise ValueError("spatial_topk must be positive")
                per_image = weights.topk(k, dim=2).values.sum(dim=2)
            else:
                per_image = weights.sum(dim=2)
            scores[n0:n0 + per_image.shape[1]] += per_image.sum(dim=0)
    result = scores.float()
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("NDA scores became non-finite")
    return result


def resolve_recipes(count: int, *, patch_size: int, projected_size=None,
                    second_patch_size=None, second_projected_size=None,
                    two_scale_alpha=0.5, variant="single", recipes=None) -> list[dict]:
    """Expand the legacy scalar recipe; per-time entries override only named fields."""
    base = dict(patch_size=patch_size, projected_size=projected_size,
                second_patch_size=second_patch_size, second_projected_size=second_projected_size,
                two_scale_alpha=two_scale_alpha)
    entries = [{} for _ in range(count)] if not recipes else recipes
    if len(entries) != count:
        raise ValueError("timestep_recipes must have one entry per configured timestep")
    if variant not in {"single", "downscaled", "two_scale"}:
        raise ValueError("variant must be single, downscaled or two_scale")
    result = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - set(base):
            raise ValueError("unknown timestep recipe fields")
        recipe = {**base, **entry}
        for key in ("patch_size", "projected_size", "second_patch_size", "second_projected_size"):
            value = recipe[key]
            if value is not None and (isinstance(value, bool) or int(value) != value or value <= 0):
                raise ValueError(f"{key} must be a positive integer")
            recipe[key] = None if value is None else int(value)
        if recipe["patch_size"] is None:
            raise ValueError("patch_size is required")
        if variant == "downscaled":
            if recipe["projected_size"] is None:
                raise ValueError("downscaled NDA needs projected_size")
            patch_projection(recipe["patch_size"], recipe["projected_size"])
        if variant == "two_scale":
            if recipe["second_patch_size"] is None or recipe["second_projected_size"] is None:
                raise ValueError("two_scale NDA needs its second patch/projection sizes")
            patch_projection(recipe["second_patch_size"], recipe["second_projected_size"])
        recipe["two_scale_alpha"] = float(recipe["two_scale_alpha"])
        if not 0 <= recipe["two_scale_alpha"] <= 1:
            raise ValueError("two_scale_alpha must lie in [0,1]")
        result.append(recipe)
    return result


def nda_scores(train: torch.Tensor, queries: torch.Tensor, *, alpha_bars,
               patch_size: int, seed: int, variant: str = "single",
               projected_size: int | None = None, second_patch_size: int | None = None,
               second_projected_size: int | None = None, two_scale_alpha: float = 0.5,
               spatial_topk: int | None = None, train_chunk: int | None = None,
               query_patch_chunk: int | None = None, mask_value: float = 1e3,
               timestep_recipes=None, query_ids=None, timestep_indices=None) -> torch.Tensor:
    """Return finite ``(N,Q)`` native NDA scores, averaged over configured times."""
    train, queries = torch.as_tensor(train), torch.as_tensor(queries)
    bars = [float(value) for value in alpha_bars]
    if not bars or any(not 0.0 < value < 1.0 for value in bars):
        raise ValueError("alpha_bars must contain values strictly inside (0,1)")
    recipes = resolve_recipes(len(bars), patch_size=patch_size, projected_size=projected_size,
                              second_patch_size=second_patch_size,
                              second_projected_size=second_projected_size,
                              two_scale_alpha=two_scale_alpha, variant=variant,
                              recipes=timestep_recipes)
    query_ids = list(range(len(queries))) if query_ids is None else list(query_ids)
    timestep_indices = list(range(len(bars))) if timestep_indices is None else list(timestep_indices)
    for ids, count in ((query_ids, len(queries)), (timestep_indices, len(bars))):
        if len(ids) != count or len(set(ids)) != count or any(int(v) != v or v < 0 for v in ids):
            raise ValueError("query/timestep indices must be unique non-negative original positions")
    result = torch.zeros(train.shape[0], queries.shape[0], dtype=torch.float32,
                         device=train.device)
    for qi, clean in enumerate(queries):
        column = torch.zeros(train.shape[0], dtype=torch.float32, device=train.device)
        for ti, alpha_bar in enumerate(bars):
            recipe = recipes[ti]
            generator = torch.Generator(device="cpu").manual_seed(
                ((int(seed) * 1_000_003 + int(query_ids[qi])) * 1_000_033
                 + int(timestep_indices[ti])) % (2**63 - 1))
            noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(clean)
            noisy = alpha_bar ** 0.5 * clean + (1.0 - alpha_bar) ** 0.5 * noise
            common = dict(signal_scale=alpha_bar ** 0.5, noise_std=(1.0-alpha_bar) ** 0.5,
                          mask_value=mask_value, spatial_topk=spatial_topk,
                          train_chunk=train_chunk, query_patch_chunk=query_patch_chunk)
            first_projection = recipe["projected_size"] if variant == "downscaled" else None
            score = patch_match_scores(noisy, train, patch_size=recipe["patch_size"],
                                       projected_size=first_projection, **common)
            if variant == "two_scale":
                second = patch_match_scores(noisy, train, patch_size=recipe["second_patch_size"],
                                            projected_size=recipe["second_projected_size"], **common)
                alpha = recipe["two_scale_alpha"]
                score = alpha * score + (1.0-alpha) * second
            column += score
        result[:, qi] = column / len(bars)
    return result
