"""Numerical primitives for finite-update AbU+ attribution."""
from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Mapping

import torch


class DiagonalNaturalGradient:
    """Apply ``(diag(F) + damping)^-1`` to named gradients."""

    def __init__(self, diagonal: Mapping[str, torch.Tensor], *, damping: float,
                 gain: float = 1.0) -> None:
        if damping < 0 or gain <= 0:
            raise ValueError("damping must be non-negative and gain positive")
        self.diagonal = {str(k): torch.as_tensor(v).detach() for k, v in diagonal.items()}
        self.damping, self.gain = float(damping), float(gain)

    def apply(self, gradients: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if set(gradients) != set(self.diagonal):
            missing = sorted(set(gradients) ^ set(self.diagonal))
            raise ValueError(f"gradient/Fisher parameter mismatch: {missing[:5]}")
        out = {}
        for name, grad in gradients.items():
            diag = self.diagonal[name].to(device=grad.device, dtype=grad.dtype)
            if diag.shape != grad.shape:
                raise ValueError(f"{name}: Fisher {tuple(diag.shape)} != gradient {tuple(grad.shape)}")
            denom = self.gain * diag + self.damping
            if bool((denom <= 0).any()) or not bool(torch.isfinite(denom).all()):
                raise ValueError(f"{name}: non-positive/non-finite preconditioner denominator")
            out[name] = grad / denom
        return out


class EigenKroneckerNaturalGradient:
    """EK-FAC inverse for Linear/flattened Conv weights and separate bias blocks."""

    def __init__(self, blocks: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                 *, damping: float = 0.0, gain: float = 1.0,
                 damping_mode: str = "absolute", layer_names=None) -> None:
        if not math.isfinite(damping) or damping < 0 or not math.isfinite(gain) or gain <= 0:
            raise ValueError("damping must be finite non-negative and gain finite positive")
        if damping_mode not in {"absolute", "relative_layer"}:
            raise ValueError("damping_mode must be absolute or relative_layer")
        self.blocks = dict(blocks)
        self.damping, self.gain = float(damping), float(gain)
        self.damping_mode = damping_mode
        layers = layer_names or {name: name for name in blocks}
        if set(layers) != set(blocks):
            raise ValueError("layer names must cover exactly the curvature blocks")
        sums, counts = {}, {}
        for name, (_, _, spectrum) in blocks.items():
            if not bool(torch.isfinite(spectrum).all()) or bool((spectrum < 0).any()):
                raise ValueError("EK-FAC spectra must be finite non-negative")
            layer = layers[name]
            sums[layer] = sums.get(layer, 0.0) + float(spectrum.double().sum()) * gain
            counts[layer] = counts.get(layer, 0) + spectrum.numel()
        self.layer_damping = {name: (damping * sums[layers[name]] / counts[layers[name]]
                                    if damping_mode == "relative_layer" else damping)
                              for name in blocks}

    @classmethod
    def from_factors(cls, factors, parameter_names, *, damping: float):
        """Adapt project factors; legacy GGN/model-Fisher is deliberately rejected."""
        meta = factors.meta
        if meta.get("curvature_kind") != "empirical_fisher":
            raise ValueError("AbU EK-FAC requires empirical_fisher, not legacy GGN/model-Fisher")
        normalization = meta.get("normalization")
        if normalization == "mean_per_datum_gradient_outer_product":
            gain = 1.0
        elif normalization == "sum_per_datum_gradient_outer_product":
            draws = int(meta.get("n_eigenvalue_draws", 0))
            if draws <= 0:
                raise ValueError("summed curvature requires n_eigenvalue_draws")
            gain = 1.0 / draws
        else:
            raise ValueError("unknown empirical curvature normalization")
        if set(meta.get("parameter_names", [])) != set(parameter_names):
            raise ValueError("curvature metadata parameter domain does not match AbU parameters")
        blocks, layers = {}, {}
        for name in parameter_names:
            layer, _, field = name.rpartition(".")
            if layer not in factors.layers or field not in {"weight", "bias"}:
                raise ValueError(f"unsupported/missing EK-FAC parameter {name}")
            ly = factors.layers[layer]
            if field == "weight":
                blocks[name] = (ly["Q_B"], ly["Q_A"], ly["lam_w"])
            else:
                if ly["lam_b"] is None:
                    raise ValueError(f"missing bias curvature for {name}")
                blocks[name] = (ly["Q_B"], torch.ones(1, 1), ly["lam_b"][:, None])
            layers[name] = layer
        return cls(blocks, damping=damping, gain=gain, damping_mode="relative_layer", layer_names=layers)

    def apply(self, gradients: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if set(gradients) != set(self.blocks):
            raise ValueError("gradient and curvature block names must match")
        out = {}
        for name, grad in gradients.items():
            shape = grad.shape
            grad = grad.reshape(grad.shape[0], -1)
            left, right, eigenvalues = (v.to(grad) for v in self.blocks[name])
            if grad.ndim != 2 or left.shape != (grad.shape[0], grad.shape[0]) or \
                    right.shape != (grad.shape[1], grad.shape[1]) or eigenvalues.shape != grad.shape:
                raise ValueError(f"{name}: incompatible EK-FAC shapes")
            coordinates = left.T @ grad @ right
            denom = self.gain * eigenvalues + self.layer_damping[name]
            if not bool(torch.isfinite(denom).all()) or bool((denom < 0).any()):
                raise ValueError(f"{name}: invalid EK-FAC denominator")
            if bool((denom == 0).all()) and self.damping_mode == "relative_layer":
                # Entirely dead LoRA layers have no curvature information.
                out[name] = torch.zeros_like(grad).reshape(shape)
                continue
            if bool((denom <= 0).any()):
                raise ValueError(f"{name}: singular EK-FAC block; positive damping required")
            out[name] = (left @ (coordinates / denom) @ right.T).reshape(shape)
        return out


def finite_loss_difference(before, after, *, flip_mode: str = "none") -> torch.Tensor:
    """Compute the finite update effect, with official max-over-flips ordering."""
    before, after = torch.as_tensor(before), torch.as_tensor(after)
    if before.shape != after.shape or not bool(torch.isfinite(before).all()) or \
            not bool(torch.isfinite(after).all()):
        raise ValueError("before/after losses must be congruent and finite")
    delta = after - before
    if flip_mode == "none":
        if delta.ndim != 1:
            raise ValueError("flip_mode='none' expects shape (N,)")
        return delta
    if flip_mode == "max":
        if delta.ndim != 2 or delta.shape[1] != 2:
            raise ValueError("flip_mode='max' expects (N,2) [no-flip, flip]")
        return delta.max(dim=1).values
    raise ValueError("flip_mode must be 'none' or 'max'")


def apply_natural_update(model, gradients: Mapping[str, torch.Tensor], preconditioner,
                         *, step_size: float) -> None:
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    updates = preconditioner.apply(gradients)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, update in updates.items():
            if name not in params or params[name].shape != update.shape:
                raise ValueError(f"update for unknown/incompatible parameter {name!r}")
            params[name].add_(update.to(params[name]), alpha=float(step_size))


@contextmanager
def restored_model(model, parameter_names=None):
    """Restore parameters, buffers, gradients and train/eval mode on every exit."""
    selected = (set(parameter_names) if parameter_names is not None else
                {name for name, value in model.named_parameters() if value.requires_grad})
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()
                  if name in selected}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    mode = bool(model.training)
    try:
        yield model
    finally:
        with torch.no_grad():
            for name, value in model.named_parameters():
                if name in parameters:
                    value.copy_(parameters[name])
                value.grad = None
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        model.train(mode)


def deterministic_noise(shape, *, seed: int, sample_id: int, draw: int,
                        device, dtype) -> torch.Tensor:
    """Noise bound to a logical sample/draw, independent of loop ordering."""
    logical_seed = ((int(seed) * 1_000_003 + int(sample_id)) * 1_000_033 + int(draw)) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(logical_seed)
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)


def query_loss_gradient(model, process, image: torch.Tensor, label: int, *,
                        parameter_names: list[str], mc: int, seed: int,
                        query_id: int, allow_unused: bool = False) -> dict[str, torch.Tensor]:
    """MC gradient of the task loss for one query."""
    if mc <= 0:
        raise ValueError("mc must be positive")
    params = dict(model.named_parameters())
    unknown = [name for name in parameter_names if name not in params or not params[name].requires_grad]
    if unknown:
        raise ValueError(f"unavailable update parameters: {unknown[:5]}")
    model.zero_grad(set_to_none=True)
    image = image.unsqueeze(0) if image.ndim == 3 else image
    label_t = torch.tensor([int(label)], device=image.device, dtype=torch.long)
    for draw in range(int(mc)):
        noise = deterministic_noise(image.shape, seed=seed, sample_id=query_id,
                                    draw=draw, device=image.device, dtype=image.dtype)
        tgen = torch.Generator(device="cpu").manual_seed(
            ((int(seed) + 17) * 1_000_003 + int(query_id) * 1_009 + draw) % (2**63 - 1))
        t = torch.rand(1, generator=tgen).to(device=image.device, dtype=image.dtype)
        state = process.interpolate(image, noise, t)
        prediction = model(state, t, label_t)
        (process.per_sample_loss(prediction, image, noise, t).mean() / int(mc)).backward()
    missing = [name for name in parameter_names if params[name].grad is None]
    if missing and not allow_unused:
        raise ValueError(f"selected update parameters have no query gradient: {missing[:5]}")
    return {name: (torch.zeros_like(params[name]) if params[name].grad is None
                   else params[name].grad.detach().clone()) for name in parameter_names}


@torch.no_grad()
def measure_losses(model, process, dataset, *, mc: int, seed: int,
                   device: str, flip: bool = False, flipped_images=None) -> torch.Tensor:
    """Per-sample task losses with paired, identity-keyed random draws."""
    if mc <= 0:
        raise ValueError("mc must be positive")
    if flipped_images is not None and len(flipped_images) != len(dataset):
        raise ValueError("flipped images must align with the complete training set")
    model.eval()
    values = torch.empty(len(dataset), 2 if flip else 1, dtype=torch.float32)
    for index in range(len(dataset)):
        image, label = dataset[index]
        image = torch.as_tensor(image, device=device).unsqueeze(0)
        alt = (image.flip(-1) if flipped_images is None else
               torch.as_tensor(flipped_images[index], device=device).unsqueeze(0))
        orientations = (image, alt) if flip else (image,)
        for orientation, current in enumerate(orientations):
            total = torch.zeros((), device=device, dtype=torch.float32)
            for draw in range(int(mc)):
                noise = deterministic_noise(current.shape, seed=seed, sample_id=index,
                                            draw=draw, device=device, dtype=current.dtype)
                t = process.time_grid(int(mc), "gt")[draw].to(device=device, dtype=current.dtype).view(1)
                state = process.interpolate(current, noise, t)
                label_t = torch.tensor([int(label)], device=device, dtype=torch.long)
                pred = model(state, t, label_t)
                total += process.per_sample_loss(pred, current, noise, t).mean().float()
            values[index, orientation] = (total / int(mc)).cpu()
    return values[:, 0] if not flip else values


def empirical_fisher_diagonal(model, process, dataset, *, parameter_names: list[str],
                              samples: int, mc: int, seed: int, device: str) -> dict[str, torch.Tensor]:
    """Mean squared per-example score gradient for the selected parameter domain."""
    count = min(int(samples), len(dataset))
    if count <= 0:
        raise ValueError("Fisher fitting needs at least one sample")
    accum = {name: torch.zeros_like(dict(model.named_parameters())[name], device=device)
             for name in parameter_names}
    for index in range(count):
        image, label = dataset[index]
        grads = query_loss_gradient(model, process, torch.as_tensor(image, device=device), int(label),
                                    parameter_names=parameter_names, mc=mc, seed=seed,
                                    query_id=index)
        for name, grad in grads.items():
            accum[name].add_(grad.square())
    return {name: (value / count).cpu() for name, value in accum.items()}
