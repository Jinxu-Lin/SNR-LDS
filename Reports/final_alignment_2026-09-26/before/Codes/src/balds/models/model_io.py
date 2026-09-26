"""Model (de)serialization helpers that touch *no* disk.

The store loads/saves the raw checkpoint dict (via the ``pt`` codec); these
functions only reconstruct an :class:`UNetCFM` from a checkpoint dict and build
the dict to save — so the Business layer never calls ``torch.load``/``torch.save``.
"""
from __future__ import annotations

from typing import Any

import torch

from .das_ddpm import PLATFORM as DAS_DDPM_PLATFORM, build_das_ddpm
from .unet import UNetCFM


def build_model(ckpt: dict, device: str = "cuda") -> torch.nn.Module:
    """Reconstruct a trained model from a checkpoint dict.

    A ``platform: das_ddpm`` checkpoint (the DAS archive's diffusers UNet, filed
    by ``balds import-das --model``) comes back as a ``DasDdpmAdapter``. Every
    other checkpoint is a ``UNetCFM``: the architecture is recovered from the
    stored ``base_ch``/``num_classes``; ``torch.compile`` ``_orig_mod.`` prefixes
    are stripped automatically.
    """
    if ckpt.get("platform") == DAS_DDPM_PLATFORM:
        return build_das_ddpm(ckpt, device=device)
    base_ch = ckpt.get("base_ch", 128)
    num_classes = ckpt.get("num_classes", None)
    model = UNetCFM(base_ch=base_ch, num_classes=num_classes).to(device)
    state = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def checkpoint_state(model: torch.nn.Module, *, base_ch: int, num_classes: int,
                     step: int, extra: dict[str, Any] | None = None) -> dict:
    """Build a checkpoint dict for the store to persist (matches legacy schema)."""
    state = model.state_dict()
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    ckpt = {"step": step, "model": state, "base_ch": base_ch, "num_classes": num_classes}
    if extra:
        ckpt.update(extra)
    return ckpt
