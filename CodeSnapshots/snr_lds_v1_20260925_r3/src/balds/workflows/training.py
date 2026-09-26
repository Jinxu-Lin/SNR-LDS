"""TrainUseCase paper reproduction stage."""
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
from .common import (_raw_dir, _load_ds, _steps, _subset_steps, _recipe_kwargs, _replay_diagnostics, _balanced_labels, _pool_noise, _mine_noise, _sample_pool_rows, _FEAT_RECIPES, _EN_FEAT, _JOURNEY_FEAT, _JOURNEY_READOUT, FEAT_CHOICES, EMBED_FEATS, MINE_ROW_KEY)
log = get_logger(__name__)

class TrainUseCase:
    """Stage 02: train the full class-conditional model and persist the checkpoint."""

    def __init__(self, store, cfg, *, device: str = "cuda:0") -> None:
        self.store, self.cfg, self.device = store, cfg, device

    def run(self, dataset: str, seed: int, *, process: str = "cfm",
            steps: Optional[int] = None, conditional: bool = True,
            force: bool = False) -> dict:
        from .latent import is_latent_platform, run_latent_train
        if is_latent_platform(self.cfg, dataset):
            # latent platform ([o]): LoRA finetune with its own recipe knobs
            return run_latent_train(self.store, self.cfg, dataset, seed,
                                    process=process, device=self.device, force=force)
        spec = RunSpec(dataset=dataset, process=process, seed=seed, conditional=conditional)
        set_context(run_id=spec.digest()[:6], trace_id=f"train:{dataset}:{process}")
        if self.store.exists(K.CHECKPOINT, spec) and not force:
            log.info("checkpoint exists, skipping"); return {"skipped": True}
        train_ds, _ = _load_ds(self.cfg, dataset)
        t = self.cfg["train"]
        steps = steps or _steps(self.cfg, len(train_ds), t["batch_size"])
        num_classes = self.cfg["model"]["num_classes"] if conditional else None
        log.info("training %s on %s: steps=%d batch=%d N=%d conditional=%s", process,
                 dataset, steps, t["batch_size"], len(train_ds), conditional)
        prog = ProgressCSV(progress_path(self.cfg["storage"]["data_root"],
                                         f"train_{dataset}_{process}_seed{seed}"))
        # Mid-training checkpoints at checkpoint_fracs of total steps ([g]; the
        # hard prerequisite for TracInCP/GAS — cannot be backfilled post-hoc).
        fracs = t.get("checkpoint_fracs", []) or []
        ckpt_steps = sorted({int(round(f * steps)) for f in fracs if 0 < f < 1})

        from .config import resolve_p_uncond
        p_unc = resolve_p_uncond(self.cfg, dataset) if conditional else None

        def _save_mid(step, mid_model):
            mid = checkpoint_state(mid_model, base_ch=self.cfg["model"]["base_ch"],
                                   num_classes=num_classes, step=step,
                                   extra={"model_type": process, "conditional": conditional,
                                          "p_uncond": p_unc})
            self.store.save(K.CHECKPOINT, spec, mid, step=step)
            log.info("mid checkpoint saved at step %d/%d", step, steps)

        model, history = train_model(
            process, train_ds, steps=steps, batch_size=t["batch_size"], lr=t["lr"],
            seed=seed, num_classes=num_classes, p_uncond=resolve_p_uncond(self.cfg, dataset),
            device=self.device, base_ch=self.cfg["model"]["base_ch"],
            checkpoint_steps=tuple(ckpt_steps), on_checkpoint=_save_mid,
            on_progress=prog.log, **_recipe_kwargs(t))
        ckpt = checkpoint_state(model, base_ch=self.cfg["model"]["base_ch"],
                                num_classes=num_classes, step=steps,
                                extra={"model_type": process, "conditional": conditional,
                                       "p_uncond": p_unc})
        self.store.save(K.CHECKPOINT, spec, ckpt)
        self.store.save(K.LOSS_HISTORY, spec, {"model_type": process, "steps": steps,
                                               "conditional": conditional, "p_uncond": p_unc,
                                               "history": history})
        return {"steps": steps, "mid_checkpoints": ckpt_steps,
                "final_loss": history[-1]["loss"] if history else None}
