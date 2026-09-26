"""Ground-truth-production use-cases (ports stages 02 train / 03 generate / 04 subsets).

These let ``balds`` build a full LDS pipeline from scratch — train the full model,
generate queries, train the M subset models, and compute the ground-truth loss
matrix — so the package no longer depends on the legacy ``Codes/`` scripts.
Heavy stages (training, 64-subset GT) are GPU/time-bound but logically identical
to the legacy pipeline; the GT loss computation is parity-verified.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import torch

from balds.schema.artifact import ArtifactKind as K
from balds.schema.logging import get_logger, set_context
from balds.schema.registry import PROCESSES
from balds.schema.runspec import RunSpec
from balds.data import generate_subsets, get_dataset
from balds.data.base import (ImageDataset, SubsetView, balanced_query_indices,
                               select_val_queries)
from balds.models import (DDPMSchedule, build_model, checkpoint_state, ddim_sample,
                            ddpm_sample, euler_solve, train_model)
from balds.evaluation.ground_truth import compute_query_losses
from balds.attribution import FEAT_T_GRID, GradFeaturizer, compute_error_weight
from balds.attribution.embed import EMBEDDERS, embed
from balds.artifacts.progress import ProgressCSV, progress_path

from .config import resolve_pixel_resize, resolve_Q

log = get_logger("balds.pipeline")

# feat name -> (readout, timesteps). Readouts are D-TRAK's output-function
# options (see featurize.py): das=mean(ε), dtrak=mean(ε²)=L_Square (the real
# D-TRAK), trak=L_Simple(mse), l1norm=Σ|ε|, l2norm=sqrt(Σε²). Each has a T=100
# sibling. (2026-07-09: the mean readout was renamed dtrak->das.)
#: readout -> the loss_type the featurizer implements. The unsuffixed name is the
#: T=10 default; every other T in _FEAT_T_GRID gets a "_T{n}" sibling. Adding a
#: timestep count here is the only edit needed to make it available end to end —
#: HP-TN sweeps T, and hand-listing recipes had already left T=20/50 missing.
_READOUTS = {"das": "mean", "dtrak": "msl2", "trak": "mse", "l1norm": "l1", "l2norm": "l2"}

_FEAT_RECIPES = {name: (loss, 10) for name, loss in _READOUTS.items()}
_FEAT_RECIPES.update({f"{name}_T{t}": (loss, t)
                      for name, loss in _READOUTS.items() for t in FEAT_T_GRID})
# e_n (per-sample residual RMSE) is readout-independent; stored once under this
# feat name and loaded from it by every das1-family method (ScoreUseCase).
_EN_FEAT = "das"

#: Journey-TRAK's query features ([e]): gradients at the generation trajectory's
#: own latents. Query-side and gen-track only by construction — a val query is a
#: real held-out image that was never generated, so it has no journey.
_JOURNEY_FEAT = "journey"
#: L_Simple is unavailable on a trajectory latent (its target needs the (x_0,
#: noise) pair that produced x_t, which a point on the ODE path does not have),
#: so the journey uses the target-free mean readout — the same one the paper's
#: own method uses, which also keeps the train side a plain `das_T*`.
_JOURNEY_READOUT = "mean"

#: Everything `--feat` accepts: gradient recipes plus the non-gradient embedders.
#: Exposed here because the CLI may not import `horizontal` (layer rule) and
#: assembling the list in two places is how one of them goes stale.
FEAT_CHOICES: tuple[str, ...] = tuple(sorted([*_FEAT_RECIPES, *EMBEDDERS, _JOURNEY_FEAT]))
EMBED_FEATS: tuple[str, ...] = tuple(sorted(EMBEDDERS))


def _raw_dir(cfg, dataset: str) -> str:
    raw = cfg["datasets"]["raw_dirs"]
    return raw["artbench"] if dataset.startswith("artbench") else raw["cifar"]


def _load_ds(cfg, dataset: str):
    """``(train, test)`` for a dataset, with EVERY raw location it needs.

    One helper rather than ``get_dataset(ds, _raw_dir(cfg, ds))`` repeated at
    nine call sites, because an IMPORTED platform needs a second raw tree:
    ``cifar2_das`` takes its pixels from the CIFAR cache but its split from the
    DAS archive's index files. Passing ``raw_dirs`` once here means a new call
    site cannot be the one that forgets; single-tree loaders ignore it.
    """
    inject_cfg = cfg.get("inject", {}) or {}
    inject_spec = inject_cfg.get(dataset)
    return get_dataset(dataset, _raw_dir(cfg, dataset),
                       raw_dirs=cfg["datasets"]["raw_dirs"],
                       inject_spec=inject_spec)


def _steps(cfg, n_samples: int, batch_size: int) -> int:
    """Optimizer steps from a fixed EPOCH budget over the actual dataset size
    (DAS-aligned recipe, [g] 2026-08-09): steps = round(epochs * N / batch).
    Changing batch_size keeps total sample exposure invariant."""
    return max(1, round(cfg["train"]["epochs"] * n_samples / batch_size))


def _subset_steps(cfg, n_subset: int, batch_size: int) -> int:
    """Subset retrains use the SAME recipe: full ``train.epochs`` over the
    subset's actual size (D4: DAS 05_ldstrain.sh trains 200 epochs on the
    subset), so steps vary per Bernoulli mask (~3.9k @ 2.5k samples, bs128)."""
    return max(1, round(cfg["train"]["epochs"] * n_subset / batch_size))


def _recipe_kwargs(t: dict) -> dict:
    """DAS-recipe knobs forwarded to train_model; absent keys fall back to the
    legacy behaviour so minimal test configs keep working."""
    return dict(weight_decay=float(t.get("weight_decay", 0.0)),
                warmup_frac=float(t.get("warmup_frac", 0.0)),
                dropout=float(t.get("dropout", 0.0)),
                hflip=bool(t.get("hflip", False)))


def _replay_diagnostics(replayed: torch.Tensor, stored: torch.Tensor) -> dict:
    """How far a replay landed from the generation it is supposed to reproduce.

    ``max |Δ|`` alone cannot tell "the same image, rounded differently" from "a
    different image": a 100-step Euler chain on a bf16 2.24B model amplifies
    last-bit differences, and the kernel choice moves with the machine, the torch
    build and even the batch size. The per-sample cosine separates the two —
    rounding leaves it at ~1, a genuinely different sample does not — so the gate
    reports both and whoever reads the failure has the evidence in hand.
    """
    d = (replayed - stored).abs()
    a = replayed.reshape(replayed.shape[0], -1).float()
    b = stored.reshape(stored.shape[0], -1).float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    return {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
            "n_cos_below_999": int((cos < 0.999).sum()), "Q": int(cos.numel())}


def _balanced_labels(classes: list[int], Q: int) -> torch.Tensor:
    q_per = Q // len(classes)
    assert q_per * len(classes) == Q, f"Q={Q} not divisible by {len(classes)} classes"
    return torch.tensor([c for c in classes for _ in range(q_per)], dtype=torch.long)


def _pool_noise(indices, *, gen_seed: int, shape: tuple[int, ...]) -> torch.Tensor:
    """CPU noise keyed by absolute pool row, independent of processing batches."""
    rows = []
    modulus = 2**63 - 1
    for index in indices:
        generator = torch.Generator(device="cpu")
        generator.manual_seed((int(gen_seed) * 1_000_003 + int(index)) % modulus)
        rows.append(torch.randn(shape, generator=generator))
    return torch.stack(rows)


#: ``inject mine`` noise keys: row ``r`` of class ``c`` is keyed ``(c + 1) * 10**9 + r``
#: under the mining gen_seed (``inject.mine.gen_seed``). Pool rows are keyed by their
#: index (< 10**9) under the pool gen_seed, so no mining draw reuses a filed pool's noise.
MINE_ROW_KEY = 10**9


def _mine_noise(cls: int, rows, *, gen_seed: int, shape: tuple[int, ...]) -> torch.Tensor:
    """Mining noise for ``rows`` of class ``cls``, independent of chunking and batches."""
    return _pool_noise([(int(cls) + 1) * MINE_ROW_KEY + int(r) for r in rows],
                       gen_seed=gen_seed, shape=shape)


def _sample_pool_rows(model, schedule, x_T, *, sampler: str, steps: int, eta: float,
                      class_label, device: str) -> torch.Tensor:
    """One batch of the screening-pool recipe as uint8 CPU images (pools and mining)."""
    if sampler == "euler":
        samples = euler_solve(model, x_T, steps=steps, device=device,
                              class_label=class_label)
    elif sampler == "ddim":
        samples = ddim_sample(model, schedule, x_T, steps=steps, eta=float(eta),
                              class_label=class_label, device=device)
    else:
        samples = ddpm_sample(
            model, schedule, tuple(x_T.shape), x_T=x_T,
            class_label=class_label, device=device)
    return ((samples.cpu() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)


