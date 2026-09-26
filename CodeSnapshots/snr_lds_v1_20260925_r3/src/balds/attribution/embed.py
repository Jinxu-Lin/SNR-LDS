"""Non-gradient image embeddings — the ``pixel`` / ``clip`` baselines ([k]).

D-TRAK's table reports two attribution baselines that never touch the model's
gradients: raw-pixel similarity and CLIP-embedding similarity. They exist to show
that a gradient method has to beat "these two images look alike", which for a
*generative* model is a real risk — a diffusion model's training samples that
resemble a query often do influence it.

Architecturally they are the same shape as projected gradients, ``(N, d)``, so
they ride the existing feature artifacts and are consumed by the existing
``grad_dot`` / ``grad_cos`` similarity method. What they must NOT inherit is the
gradient pipeline's other machinery: there is no readout, no timestep count, no
error weight and no sigma-hat, because there is no model and no noise.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from balds.schema.logging import get_logger
from balds.schema.registry import Registry

log = get_logger(__name__)

#: name -> callable(images: (N,3,H,W) float tensor, *, device, batch_size, resize)
#: -> (N, d). ``resize`` is an optional spatial size applied before embedding;
#: an embedder that has no meaningful use for it must REFUSE it rather than
#: ignore it, or a caller's resize silently stops being applied.
EMBEDDERS: Registry = Registry("embedders")


def _to_unit_interval(images: torch.Tensor) -> torch.Tensor:
    """Datasets are stored in [-1, 1] (the generative convention); encoders and
    pixel distances both want [0, 1]. Done once here so the two embedders cannot
    disagree about the input range."""
    return (images.float() / 2.0 + 0.5).clamp(0.0, 1.0)


class PixelEmbedder:
    """Flattened raw pixels. The literal "do the images look alike" baseline.

    ``resize`` (area-average downsample, applied in [0,1] space before
    flattening) exists because the baseline's dimension is the image's: 3072 at
    32x32, but 196,608 at 256x256, where one 5000-row train feature is 3.9 GB.
    The 256px platforms pass 64 (see ``featurize.pixel_resize_by_dataset``);
    ``None`` keeps the literal pixels, so the CIFAR artifacts are unchanged.
    Area averaging is an exact box filter for integer factors (256/64 = 4) and
    is deterministic, so the feature is reproducible bit for bit.
    """

    name = "pixel"

    def __call__(self, images: torch.Tensor, *, device: str = "cuda",
                 batch_size: int = 256, resize: Optional[int] = None) -> torch.Tensor:
        x = _to_unit_interval(images)
        if resize is not None and tuple(x.shape[-2:]) != (int(resize), int(resize)):
            x = torch.nn.functional.interpolate(x, size=(int(resize), int(resize)),
                                                mode="area")
        return x.reshape(x.shape[0], -1).contiguous()


class ClipEmbedder:
    """CLIP image-tower embeddings (``openai/clip-vit-base-patch32``).

    Uses the ordinary Hugging Face cache and honors HF_HUB_OFFLINE.
    Missing public weights download on first use when network access is enabled.
    """

    name = "clip"
    model_id = "openai/clip-vit-base-patch32"

    def _load(self, device: str):
        try:
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        except ImportError as e:                                   # pragma: no cover
            raise RuntimeError("the clip embedder needs `transformers`") from e

        # Honor the caller's Hugging Face cache/offline policy. Fresh users can
        # download the public model without a server-specific preparation step.
        proc = CLIPImageProcessor.from_pretrained(self.model_id)
        model = CLIPVisionModelWithProjection.from_pretrained(self.model_id).to(device).eval()

        return proc, model

    def _run_loaded(self, proc, model, images: torch.Tensor, *, device: str,
                    batch_size: int) -> torch.Tensor:
        x = _to_unit_interval(images)
        out = []
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                batch = x[i:i + batch_size]
                # the processor handles resize to 224 + CLIP's own normalisation;
                # do_rescale=False because the batch is already in [0, 1]
                px = proc(images=list(batch), return_tensors="pt",
                          do_rescale=False)["pixel_values"].to(device)
                out.append(model(pixel_values=px).image_embeds.float().cpu())
                if i == 0:
                    log.info("clip: %s -> embed dim %d", self.model_id, out[0].shape[1])
        return torch.cat(out, dim=0)

    def __call__(self, images: torch.Tensor, *, device: str = "cuda",
                 batch_size: int = 256, resize: Optional[int] = None) -> torch.Tensor:
        if resize is not None:
            raise ValueError("the clip embedder defines its own input resize (224 via "
                             "CLIPImageProcessor); pass resize=None")
        proc, model = self._load(device)
        return self._run_loaded(proc, model, images, device=device, batch_size=batch_size)

    def embed_dataset(self, dataset, *, device: str = "cuda",
                      batch_size: int = 256) -> torch.Tensor:
        """Embed a path-backed dataset without ever materialising all pixels."""
        proc, model = self._load(device)
        out = []
        for start in range(0, len(dataset), int(batch_size)):
            stop = min(start + int(batch_size), len(dataset))
            images = torch.stack([torch.as_tensor(dataset[index][0])
                                  for index in range(start, stop)])
            out.append(self._run_loaded(proc, model, images, device=device,
                                        batch_size=int(batch_size)))
            log.info("clip detector embeddings: %d/%d", stop, len(dataset))
        return torch.cat(out, dim=0)


EMBEDDERS.add("pixel", PixelEmbedder())
EMBEDDERS.add("clip", ClipEmbedder())


def embed(feat: str, images: Any, *, device: str = "cuda",
          batch_size: int = 256, resize: Optional[int] = None) -> torch.Tensor:
    """Embed ``images`` with the registered embedder named ``feat``."""
    return EMBEDDERS.get(feat)(torch.as_tensor(images), device=device,
                               batch_size=batch_size, resize=resize)
