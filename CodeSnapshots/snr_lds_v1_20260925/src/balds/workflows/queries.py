"""GenerateUseCase paper reproduction stage."""
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

class GenerateUseCase:
    """Stage 03: generate Q class-balanced query images from the full model."""

    def __init__(self, store, cfg, *, device: str = "cuda:0") -> None:
        self.store, self.cfg, self.device = store, cfg, device

    def run(self, dataset: str, seed: int, *, process: str = "cfm", Q: Optional[int] = None,
            ode_steps: Optional[int] = None, gen_seed: Optional[int] = None,
            conditional: bool = True, force: bool = False, pool: Optional[int] = None,
            tag: Optional[str] = None, sampler: Optional[str] = None,
            steps: Optional[int] = None, eta: float = 0.0,
            batch: Optional[int] = None, store_latents: bool = False,
            cf: Optional[str] = None, artifact_name: Optional[str] = None) -> dict:
        from .latent import is_latent_platform, run_latent_generate, run_latent_pool
        if is_latent_platform(self.cfg, dataset):
            if not conditional:
                raise ValueError("latent LoRA platforms are prompt-conditional; drop --uncond")
            if cf is not None:
                raise ValueError("--cf counterfactual pools are available only on pixel platforms")
            if pool is not None:
                if artifact_name is not None:
                    raise ValueError("--artifact-name is for query generations, not pools")
                return run_latent_pool(
                    self.store, self.cfg, dataset, seed, process=process,
                    device=self.device, n=int(pool), tag=tag, sampler=sampler,
                    steps=steps, eta=eta, gen_seed=gen_seed, batch=batch,
                    force=force, store_latents=store_latents)
            if batch is not None:
                raise ValueError("--batch for ordinary generation is supported only on pixel CFM; latent pools support batching")
            return run_latent_generate(self.store, self.cfg, dataset, seed,
                                       process=process, device=self.device, Q=Q,
                                       ode_steps=ode_steps, gen_seed=gen_seed,
                                       force=force, artifact_name=artifact_name)
        ode_steps = int(ode_steps or 100)       # historical pixel default
        if store_latents:
            raise ValueError("--store-latents is available only for latent-platform pools")
        spec = RunSpec(dataset=dataset, process=process, seed=seed, conditional=conditional)
        set_context(run_id=spec.digest()[:6], trace_id=f"generate:{dataset}:{process}")
        if pool is not None:
            if artifact_name is not None:
                raise ValueError("--artifact-name is for query generations, not pools")
            return self._pool(spec, int(pool), gen_seed=gen_seed, tag=tag,
                              sampler=sampler, steps=steps, eta=eta, batch=batch,
                              cf=cf, force=force)
        if cf is not None:
            raise ValueError("--cf is valid only together with --pool")
        classes = self.cfg["datasets"][dataset]["classes"]
        Q = Q or resolve_Q(self.cfg, dataset)
        generation_batch_size = Q if batch is None else int(batch)
        if generation_batch_size <= 0:
            raise ValueError("--batch must be positive")
        if batch is not None and process != "cfm":
            raise ValueError("--batch for ordinary generation is supported only on pixel CFM; DDPM pools support batching")
        generation_key = {"name": artifact_name} if artifact_name is not None else {}
        if self.store.exists(K.GENERATION, spec, **generation_key) and not force:
            # The artifact path does not encode Q, so an existing generation at a
            # different Q would be silently reused as this run's query set —
            # exactly the kind of mismatch that only shows up as a wrong number
            # much later. Refuse loudly instead of skipping.
            have = int(self.store.load(K.GENERATION, spec, **generation_key)["samples"].shape[0])
            if have != Q:
                raise ValueError(
                    f"existing generation for this identity has Q={have} but Q={Q} was requested; "
                    f"the path does not encode Q. Pass --force to regenerate (it overwrites), or "
                    f"use the existing Q.")
            log.info("generations exist (Q=%d), skipping", Q); return {"skipped": True}
        # Sampling noise is fixed per seed, not shared across seeds: with one
        # hardcoded gen_seed every seed's queries started from identical noise,
        # so the three replicates probed nearly the same corner of the model's
        # output space. Derived (base + seed) so it stays reproducible.
        gen_seed = gen_seed if gen_seed is not None else int(self.cfg["lds"].get("gen_seed_base", 123)) + seed
        # Unconditional models ignore class_label entirely, so a "balanced" label
        # vector would be fiction — the class mix of the samples is whatever the
        # noise produces. Store zeros there so no downstream reader mistakes it
        # for class information. (Only the val track is required to be balanced.)
        labels = _balanced_labels(classes, Q) if conditional else torch.zeros(Q, dtype=torch.long)
        generation_started = time.perf_counter()
        ckpt = self.store.load(K.CHECKPOINT, spec)
        model = build_model(ckpt, device=self.device)
        torch.manual_seed(gen_seed)
        x0 = torch.randn(Q, 3, 32, 32)
        class_label = labels.to(self.device) if conditional else None
        with torch.no_grad():
            if process == "cfm":
                if batch is None:
                    # Preserve the historical full-batch numerical path by default.
                    samples = euler_solve(model, x0, steps=ode_steps, device=self.device,
                                          class_label=class_label).cpu()
                else:
                    # Draw the complete original noise tensor BEFORE batching, as
                    # in the accepted independent weight-learning query recipe.
                    parts = []
                    for start in range(0, Q, generation_batch_size):
                        stop = min(start + generation_batch_size, Q)
                        labels_part = None if class_label is None else class_label[start:stop]
                        parts.append(euler_solve(model, x0[start:stop], steps=ode_steps,
                                                device=self.device,
                                                class_label=labels_part).cpu())
                    samples = torch.cat(parts, dim=0)
            else:
                sched = DDPMSchedule(T=self.cfg.get("ddpm", {}).get("T", 1000)).to(self.device)
                # Legacy conditional checkpoints were trained with classifier-free
                # dropout and keep their exact two-forward guidance path.  New
                # p_uncond=0 identities have no learned null class and therefore
                # use one plain conditional forward per step.
                p_uncond = ckpt.get("p_uncond")
                guidance = (1.0 if conditional and
                            (p_uncond is None or float(p_uncond) > 0.0) else 0.0)
                samples = ddpm_sample(model, sched, (Q, 3, 32, 32), class_label=class_label,
                                      device=self.device, guidance_scale=guidance,
                                      num_classes=self.cfg["model"]["num_classes"]).cpu()
        generation_seconds = time.perf_counter() - generation_started
        self.store.save(K.GENERATION, spec, {"samples": samples, "labels": labels,
                                             "Q": Q, "gen_seed": gen_seed,
                                             "ode_steps": ode_steps, "conditional": conditional,
                                             "artifact_name": artifact_name or "samples",
                                             "generation_batch_size": generation_batch_size,
                                             "timing_seconds": {
                                                 "model_load_and_generation": generation_seconds,
                                             }},
                        **generation_key)
        log.info("generated %d samples (%d/class, classes=%s, gen_seed=%d, ode_steps=%d)",
                 Q, Q // len(classes), classes, gen_seed, ode_steps)
        return {"Q": Q, "gen_seed": gen_seed, "ode_steps": ode_steps,
                "generation_batch_size": generation_batch_size, "shape": list(samples.shape)}

    def _pool(self, spec: RunSpec, n: int, *, gen_seed: Optional[int],
              tag: Optional[str], sampler: Optional[str], steps: Optional[int],
              eta: float, batch: Optional[int], cf: Optional[str], force: bool) -> dict:
        """Generate a class-balanced screening pool without touching LDS queries."""
        if n <= 0:
            raise ValueError("--pool must be positive")
        process = spec.process
        sampler = sampler or ("ddim" if process == "ddpm" else "euler")
        allowed = {"ddpm": {"ddim", "ancestral"}, "cfm": {"euler"}}
        if process not in allowed or sampler not in allowed[process]:
            raise ValueError(f"sampler {sampler!r} is incompatible with process {process!r}")
        schedule = None
        if process == "ddpm":
            schedule = DDPMSchedule(T=self.cfg.get("ddpm", {}).get("T", 1000)).to(self.device)
        if steps is None:
            steps = (schedule.T if sampler == "ancestral" else
                     50 if sampler == "ddim" else 100)
        steps = int(steps)
        if sampler == "ancestral" and steps != schedule.T:
            raise ValueError(f"ancestral sampling always uses the full {schedule.T} DDPM steps")
        if steps <= 0:
            raise ValueError("sampler steps must be positive")
        if sampler != "ddim" and float(eta) != 0.0:
            raise ValueError("eta is defined only for the DDIM sampler")
        batch = int(batch or self.cfg.get("inject", {}).get("pool", {}).get("batch_size", 512))
        if batch <= 0:
            raise ValueError("--batch must be positive")
        gen_seed = int(gen_seed if gen_seed is not None else
                       self.cfg.get("inject", {}).get("pool", {}).get("gen_seed", 20260914))
        tag = tag or f"{sampler}{steps}_n{n}"
        # Address validation happens before any model or artifact is touched.
        from balds.artifacts.addressing import relpath
        pool_key = {"tag": tag, **({"cf": cf} if cf else {})}
        relpath(K.GEN_POOL, spec, **pool_key)
        if self.store.exists(K.GEN_POOL, spec, **pool_key) and not force:
            raise FileExistsError(f"generation pool tag {tag!r} already exists; use --force")

        classes = self.cfg["datasets"][spec.dataset]["classes"]
        labels = (_balanced_labels(classes, n) if spec.conditional
                  else torch.zeros(n, dtype=torch.long))
        ckpt_kind = K.INJECT_CF_CHECKPOINT if cf else K.CHECKPOINT
        ckpt_key = {"drop_tag": cf} if cf else {}
        ckpt = self.store.load(ckpt_kind, spec, **ckpt_key)
        checkpoint_sha = self.store.describe(ckpt_kind, spec, **ckpt_key)["sha256"]
        model = build_model(ckpt, device=self.device)
        pieces = []
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            x_T = _pool_noise(range(start, stop), gen_seed=gen_seed, shape=(3, 32, 32))
            class_label = labels[start:stop].to(self.device) if spec.conditional else None
            pieces.append(_sample_pool_rows(model, schedule, x_T, sampler=sampler, steps=steps,
                                            eta=eta, class_label=class_label,
                                            device=self.device))
        images_u8 = torch.cat(pieces)
        payload = {
            "images_u8": images_u8, "labels": labels, "n": n,
            "gen_seed": gen_seed, "sampler": sampler, "steps": steps,
            "eta": float(eta), "guidance": 0.0,
            "checkpoint_sha256": checkpoint_sha,
            "code_version": getattr(self.store, "code_version", "dev"),
            "conditional": bool(spec.conditional), "p_uncond": ckpt.get("p_uncond"),
            "cf": cf,
        }
        self.store.save(K.GEN_POOL, spec, payload, **pool_key,
                        upstream=(checkpoint_sha,))
        return {"tag": tag, "n": n, "shape": list(images_u8.shape),
                "sampler": sampler, "steps": steps, "gen_seed": gen_seed, "cf": cf}

    def trajectory(self, dataset: str, seed: int, *, process: str = "cfm",
                   n_points: Optional[int] = None, conditional: bool = True,
                   force: bool = False) -> dict:
        """Journey-TRAK ([e]) prerequisite: the intermediate states of the ODE
        that produced this identity's existing query samples.

        Replays the sampler rather than regenerating: the same gen_seed, Q,
        ode_steps and labels are read back off the stored generation, so the
        trajectory necessarily belongs to the samples already being attributed.
        Generating afresh would give a *different* journey to the same file name
        and nothing downstream could tell.
        """
        from .latent import build_latent_model, is_latent_platform
        latent = is_latent_platform(self.cfg, dataset)
        if latent and not conditional:
            raise ValueError("the SD3.5 platform is class-conditional by construction; "
                             "--uncond has no meaning here")
        spec = RunSpec(dataset=dataset, process=process, seed=seed, conditional=conditional)
        set_context(run_id=spec.digest()[:6], trace_id=f"trajectory:{dataset}:{process}")
        if process != "cfm":
            raise ValueError("trajectory capture is implemented for the cfm (ODE) sampler only; "
                             "the ddpm path is ancestral and would need its own capture")
        if self.store.exists(K.GEN_TRAJECTORY, spec) and not force:
            log.info("trajectory exists, skipping (use --force)")
            return {"skipped": True}
        gen = self.store.load(K.GENERATION, spec)
        Q, gen_seed = int(gen["Q"]), int(gen["gen_seed"])
        ode_steps, labels = int(gen["ode_steps"]), gen["labels"]
        n_points = int(n_points or self.cfg["lds"].get("journey_points", 10))
        # evenly spaced over the journey, endpoints included
        idx = sorted({int(round(i * ode_steps / (n_points - 1))) for i in range(n_points)}) \
            if n_points > 1 else [ode_steps]
        ckpt = self.store.load(K.CHECKPOINT, spec)
        model = (build_latent_model(self.store, self.cfg, dataset, ckpt, device=self.device)
                 if latent else build_model(ckpt, device=self.device))
        # x0's shape comes from what was GENERATED, not from a per-platform
        # constant: (Q, 3, 32, 32) pixels on the UNet platform, (Q, 16, 32, 32)
        # latents on SD3.5. Same draw either way, so the CIFAR path is unchanged.
        shape = tuple(int(v) for v in gen["samples"].shape[1:])
        torch.manual_seed(gen_seed)                       # same noise as the stored samples
        x0 = torch.randn(Q, *shape)
        with torch.no_grad():
            final, traj = euler_solve(model, x0, steps=ode_steps, device=self.device,
                                      class_label=labels.to(self.device), capture_at=idx)
        # The replay must land on the samples we are attributing, or the journey
        # belongs to some other image. Compared in the space the sampler works in
        # -- latents on SD3.5, where `samples` IS the latent tensor; comparing the
        # decoded previews instead would fold the VAE's own error into the check.
        diag = _replay_diagnostics(final.cpu(), gen["samples"])
        drift = diag["max_abs_diff"]
        tol = float(self.cfg["lds"].get("journey_drift_tol", 1e-3))
        if drift > tol:
            raise ValueError(
                f"replayed sampler diverged from the stored generation "
                f"(max |Δ| = {drift:.3g} > lds.journey_drift_tol = {tol:g}); the trajectory "
                f"would not belong to these samples. Diagnostics: mean |Δ| = "
                f"{diag['mean_abs_diff']:.3g}, per-sample cosine min "
                f"{diag['cos_min']:.6f} / mean {diag['cos_mean']:.6f}, "
                f"{diag['n_cos_below_999']}/{diag['Q']} samples below cos 0.999. "
                f"A cosine that is still ~1 means the images are the same and only the "
                f"tolerance is wrong for this platform (a 100-step bf16 ODE amplifies "
                f"last-bit rounding, and the kernel choice moves with the machine, the "
                f"torch build and even the batch size — the ArtBench-2 generation's own "
                f"producing box measured 8.5e-2 in latent space at a different batch, "
                f"RESULTS ab2_2026-09-03 §1). A cosine that has actually dropped means a "
                f"different generation: check gen_seed/ode_steps/checkpoint. Either way "
                f"the fix is a decision, not a flag: re-run on the machine that produced "
                f"`samples.pt`, or have the caliber (lds.journey_drift_tol) adjudicated.")
        steps = sorted(traj)
        states = torch.stack([traj[s].cpu() for s in steps], dim=1)       # (Q, P, C, H, W)
        self.store.save(K.GEN_TRAJECTORY, spec,
                        {"states": states, "steps": torch.tensor(steps, dtype=torch.long),
                         "ode_steps": ode_steps, "Q": Q, "gen_seed": gen_seed,
                         "labels": labels})
        log.info("trajectory: %d points %s of %d ODE steps, replay drift %.2e (tol %g), "
                 "per-sample cosine min %.6f", len(steps), steps, ode_steps, drift, tol,
                 diag["cos_min"])
        return {"points": len(steps), "steps": steps, "shape": list(states.shape),
                "replay_drift": drift, "drift_tol": tol, **diag}
