"""The DAS release archive's CIFAR-2 DDPM behind the common model seam.

The archive ships its trained main model as a diffusers ``DDPMPipeline``
directory (``CIFAR2/saved/5000-0.5/ddpm/ddpm_<seed>``): an unconditional
``UNet2DModel`` plus a ``DDPMScheduler``. ``balds import-das --model`` files it as
a ``platform: das_ddpm`` checkpoint, and :func:`balds.models.model_io.build_model`
dispatches such checkpoints here, so the curvature path (``balds ekfac fit/score``)
runs on DAS's own model with no other change.

Three archive facts this module carries, each read off the shipped code/weights:

* **Corruption.** The scheduler is linear beta 1e-4..0.02, T=1000, epsilon
  prediction — exactly the registered ``ddpm`` process. :func:`verify_scheduler_config`
  refuses anything else: a different schedule would feed the model inputs it
  was never trained on, and every gradient would still look plausible.
* **Dropout.** ``_code_DAS/CIFAR2/01_train.py::create_model`` sets ``p=0.1`` on
  every ``nn.Dropout`` after ``from_config``; ``config.json`` does not record it
  (current diffusers would default it to 0). Under the diffusers that trained the
  model (0.16.1, per the config) the only Dropout modules were the resnets' —
  its legacy ``AttentionBlock`` had none, and the checkpoint's
  ``query/key/value/proj_attn`` keys are that block. Current diffusers adds an
  ``Attention.to_out.1`` Dropout; it stays at 0 here.
* **Attention keys.** The legacy names are renamed to ``to_q/to_k/to_v/to_out.0``
  (the mapping diffusers' own ``_convert_deprecated_attention_blocks`` applies)
  and the load is strict.
* **Mid-block attention norm.** The 0.16 ``UNetMidBlock2D`` gave its attention a
  GroupNorm(``norm_num_groups``) unconditionally (the checkpoint carries
  ``mid_block.attentions.0.group_norm``); current diffusers drops it under
  ``resnet_time_scale_shift: scale_shift`` unless ``attn_norm_num_groups`` is
  set. A config without that key is from before it existed, so it is set to
  ``norm_num_groups`` here (the only block it affects is the mid block).

diffusers is imported lazily so the rest of the package does not need it.
"""
from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import nn

#: ``ckpt["platform"]`` value that routes a checkpoint to :func:`build_das_ddpm`.
PLATFORM = "das_ddpm"

#: What the registered ``ddpm`` process is (``balds.models.ddpm``).
EXPECTED_SCHEDULER = {
    "beta_schedule": "linear",
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
}

#: ``01_train.py`` training dropout, applied to the resnet Dropout modules.
RESNET_DROPOUT = 0.1

_LEGACY_ATTENTION = {"query": "to_q", "key": "to_k", "value": "to_v", "proj_attn": "to_out.0"}


def verify_scheduler_config(scheduler_config: Mapping) -> dict:
    """Return the corruption-defining scheduler facts, or raise on any mismatch."""
    for key, want in EXPECTED_SCHEDULER.items():
        got = scheduler_config.get(key)
        if isinstance(want, float):
            ok = isinstance(got, (int, float)) and abs(float(got) - want) <= 1e-12
        else:
            ok = got == want
        if not ok:
            raise ValueError(
                f"DAS scheduler {key}={got!r}, expected {want!r}: the registered `ddpm` "
                f"process (linear 1e-4..0.02, T=1000, epsilon) would not be this "
                f"model's corruption")
    if scheduler_config.get("trained_betas") is not None:
        raise ValueError("DAS scheduler carries trained_betas; the registered `ddpm` "
                         "process is the plain linear table")
    return {key: scheduler_config[key] for key in EXPECTED_SCHEDULER}


def convert_legacy_attention_keys(state: Mapping[str, torch.Tensor]) -> dict:
    """Rename diffusers<0.17 ``AttentionBlock`` parameters to ``Attention`` names."""
    out = {}
    for key, value in state.items():
        head, _, leaf = key.rpartition(".")
        path, _, name = head.rpartition(".")
        if name in _LEGACY_ATTENTION and ".attentions." in f".{path}.":
            key = f"{path}.{_LEGACY_ATTENTION[name]}.{leaf}"
        out[key] = value
    return out


class DasDdpmAdapter(nn.Module):
    """Unconditional diffusers UNet behind the ``(x_t, t, class_label)`` seam (ε out)."""

    def __init__(self, unet: nn.Module, *, num_train_timesteps: int = 1000) -> None:
        super().__init__()
        self.unet = unet
        self.num_train_timesteps = int(num_train_timesteps)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                class_label: Optional[torch.Tensor] = None) -> torch.Tensor:
        # class_label is ignored: the model has no classes, and the platform's
        # artifacts live under the default conditional identity (das_import.py),
        # so callers do pass labels.
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x_t.device, dtype=torch.float32)
        if t.dim() == 0:
            t = t.expand(x_t.shape[0])
        # the expression of DDPM._t_index, so the model sees the index that corrupted x_t
        timestep = torch.clamp((t * self.num_train_timesteps).round().long(),
                               0, self.num_train_timesteps - 1)
        return self.unet(x_t, timestep, return_dict=True).sample.float()


def build_das_ddpm(ckpt: dict, device: str = "cuda") -> DasDdpmAdapter:
    """Reconstruct the archive model from a ``platform: das_ddpm`` checkpoint dict."""
    from diffusers import UNet2DModel
    from diffusers.models.resnet import ResnetBlock2D

    verify_scheduler_config(ckpt["scheduler_config"])
    config = dict(ckpt["unet_config"])
    config.setdefault("attn_norm_num_groups", config.get("norm_num_groups", 32))
    unet = UNet2DModel.from_config(config)
    unet.load_state_dict(convert_legacy_attention_keys(ckpt["unet_state"]), strict=True)
    for block in unet.modules():
        if isinstance(block, ResnetBlock2D):
            block.dropout.p = RESNET_DROPOUT
    unet.requires_grad_(True)          # EK-FAC covers the whole model, as on CIFAR
    adapter = DasDdpmAdapter(
        unet, num_train_timesteps=int(ckpt["scheduler_config"]["num_train_timesteps"]))
    return adapter.to(device).eval()
