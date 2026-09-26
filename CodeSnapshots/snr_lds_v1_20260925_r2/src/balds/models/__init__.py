"""Generative processes + the shared U-Net backbone (Business layer).

Importing this package registers the ``cfm`` and ``ddpm`` processes. Model
(de)serialization is split so the Business layer never performs disk IO:
:func:`build_model` reconstructs from a checkpoint dict the store loaded.
"""
from __future__ import annotations

from . import flow  # noqa: F401  (registers "cfm")
from . import ddpm  # noqa: F401  (registers "ddpm")
from .ddpm import DDPMSchedule
from .model_io import build_model, checkpoint_state
from .sample import ddim_sample, ddpm_sample, euler_solve
from .train import train_model
from .unet import UNetCFM

__all__ = [
    "UNetCFM", "build_model", "checkpoint_state", "DDPMSchedule",
    "euler_solve", "ddpm_sample", "ddim_sample", "train_model",
]
