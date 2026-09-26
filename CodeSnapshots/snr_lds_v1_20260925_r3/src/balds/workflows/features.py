"""FeaturizeUseCase paper reproduction stage."""
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

class FeaturizeUseCase:
    """Stage 06: project per-sample gradients into feature tensors.

    Produces the train / query gradient features (and, for the train split, the
    per-sample error weight ``e_n``) that ``ScoreUseCase`` later loads. The
    ``feat`` name selects readout + timesteps (``_FEAT_RECIPES``): das=mean(ε),
    dtrak=mean(ε²)=L_Square, trak=L_Simple(mse), l1norm=Σ|ε|, l2norm=sqrt(Σε²),
    each with a ``_T100`` sibling. All idempotent: an existing artifact is
    skipped unless ``force``. Writes progress to ``_Data/progress/featurize_*.csv``.
    """

    def __init__(self, store, cfg, *, device: str = "cuda:0") -> None:
        self.store, self.cfg, self.device = store, cfg, device

    # -- platform seams ------------------------------------------------------
    def _is_latent(self, dataset: str) -> bool:
        from .latent import is_latent_platform
        return is_latent_platform(self.cfg, dataset)

    def _load_model(self, base: RunSpec, ckey: dict):
        ckpt = self.store.load(K.CHECKPOINT, base, **ckey)
        if self._is_latent(base.dataset):
            from .latent import build_latent_model
            return build_latent_model(self.store, self.cfg, base.dataset, ckpt,
                                      device=self.device)
        return build_model(ckpt, device=self.device)

    def _train_test(self, dataset: str):
        if self._is_latent(dataset):
            from .latent import latent_test, latent_train_plain
            return latent_train_plain(self.store, dataset), latent_test(self.store, dataset)
        return _load_ds(self.cfg, dataset)

    def _feat_batch(self, dataset: str) -> int:
        if self._is_latent(dataset):
            from .latent import platform_cfg
            return int(platform_cfg(self.cfg, dataset)["featurize_batch_size"])
        return int(self.cfg["featurize"]["batch_size"])

    def run(self, dataset: str, seed: int, *, process: str = "cfm", feat: str = "das",
            split: str = "train", query_type: str = "gen", max_samples: Optional[int] = None,
            conditional: bool = True, proj_dim: Optional[int] = None,
            proj_seed: Optional[int] = None, force: bool = False,
            ckpt_step: Optional[int] = None, no_error_weight: bool = False) -> dict:
        if no_error_weight and split not in ("train", "both"):
            raise ValueError("--no-error-weight is valid only with --split train|both")
        if EMBEDDERS.has(feat):
            # `_embed` owns the platform seam: the embedders eat IMAGES on both
            # platforms, only the image source differs (dataset pixels vs the raw
            # 256px folder + the generation's decoded previews).
            return self._embed(dataset, seed, process=process, feat=feat, split=split,
                               query_type=query_type, max_samples=max_samples,
                               conditional=conditional, force=force)
        if feat == _JOURNEY_FEAT:
            return self._journey(dataset, seed, process=process, split=split,
                                 query_type=query_type, conditional=conditional,
                                 proj_dim=proj_dim, proj_seed=proj_seed,
                                 max_samples=max_samples, force=force)
        if feat not in _FEAT_RECIPES:
            raise ValueError(f"unknown feat '{feat}'; choose from "
                             f"{list(_FEAT_RECIPES) + list(EMBEDDERS)}")
        loss_type, T = _FEAT_RECIPES[feat]
        base = RunSpec(dataset=dataset, seed=seed, process=process, conditional=conditional)
        set_context(run_id=base.digest()[:6], trace_id=f"featurize:{dataset}:{process}:{feat}")
        if self._is_latent(dataset):
            from .latent import platform_process
            proc, _ = platform_process(self.cfg, dataset, device=self.device)
        else:
            proc = PROCESSES.get(process)
        # TracInCP/GAS read the same recipe at several training checkpoints, so the
        # checkpoint is part of the feature's identity — both the model loaded here
        # and the path written below. Absent, everything is the final model as before.
        ckey = {"step": int(ckpt_step)} if ckpt_step is not None else {}
        model = self._load_model(base, ckey)
        train_ds, test_ds = self._train_test(dataset)
        fcfg = self.cfg["featurize"]
        # An explicit projection makes it part of the artifact identity (HP-p
        # sweeps four p in one tree); leaving it unset uses the configured
        # default and keeps the production path free of a projection segment.
        pdim = int(proj_dim) if proj_dim is not None else int(fcfg["proj_dim"])
        pseed = int(proj_seed) if proj_seed is not None else int(fcfg["proj_seed"])
        pkey = ({"proj": f"p{pdim}s{pseed}"}
                if (proj_dim is not None or proj_seed is not None) else {})
        pkey.update(ckey)
        # NOTE: e_n deliberately does NOT take pkey. It is a no-grad residual
        # norm — no gradient, no projection — so it is bit-identical across every
        # rung of an HP-p sweep. Storing it per-projection would recompute the
        # same 5000x1000 forward passes once per rung for nothing.
        fz = GradFeaturizer(model, proc, proj_dim=pdim,
                            proj_seed=pseed, T=T, loss_type=loss_type,
                            batch_size=self._feat_batch(dataset), device=self.device,
                            projection=str(self.cfg["featurize"].get("projection", "auto")))
        # progress tag carries the mid-checkpoint step (the artifact path does too):
        # without it a `--ckpt-step` run appends to the final-checkpoint run's CSV and
        step_tag = f"_step{int(ckpt_step)}" if ckpt_step is not None else ""
        prog = ProgressCSV(progress_path(self.cfg["storage"]["data_root"],
                                         f"featurize_{dataset}_{process}_{feat}_seed{seed}{step_tag}"))
        log.info("featurize %s/%s feat=%s split=%s proj=%s", dataset, process, feat, split,
                 fz.projector_kind)
        out: dict = {"feat": feat, "process": process, "projector": fz.projector_kind}

        if split == "repeat":
            # R re-featurizations of the SAME M training samples with fresh
            # featurization noise and an identical projection basis. Their score
            # spread is sigma-hat for the shrinkage layer; only the noise seed
            # moves, because a spread contaminated by protocol differences reads
            # signal as noise and over-shrinks (0.62 -> 0.28 in the pilot).
            if ckpt_step is not None:
                raise ValueError(
                    "sigma-hat is the noise of the FINAL model's featurization; "
                    "repeats at a mid-training checkpoint would be measuring a "
                    "different estimator. Drop --ckpt-step for --split repeat.")
            tw = self.cfg.get("tweedie", {})
            R = int(tw.get("repeats", 4))
            M = int(tw.get("sigma_samples", 256))
            if self.store.exists(K.REPEAT_FEATURES, base, feat=feat, **pkey) and not force:
                log.info("repeat features exist, skipping (use --force)")
                return {**out, "repeats": "skipped"}
            sub_ds = ImageDataset(train_ds.images[:M], train_ds.labels[:M])
            reps = []
            for r in range(R):
                # seed offset keeps every repeat reproducible and distinct from
                # the production featurization (which uses `seed` itself)
                rseed = seed + 1_000_000 * (r + 1)
                Fr = fz.featurize(sub_ds, seed=rseed, log_every=500,
                                  on_progress=lambda **kw: prog.log(split=f"repeat{r}", **kw))
                reps.append(Fr.cpu() if hasattr(Fr, "cpu") else Fr)
                log.info("repeat %d/%d done (seed=%d, M=%d)", r + 1, R, rseed, M)
            import torch as _t
            stacked = _t.stack([_t.as_tensor(x) for x in reps])      # (R, M, p)
            self.store.save(K.REPEAT_FEATURES, base, stacked, feat=feat, **pkey)
            out["repeats"] = list(stacked.shape)
            return out

        if split in ("train", "both"):
            if self.store.exists(K.TRAIN_FEATURES, base, feat=feat, **pkey) and not force:
                log.info("train features exist, skipping (use --force)")
                out["train_features"] = "skipped"
            else:
                F = fz.featurize(train_ds, seed=seed, max_samples=max_samples, log_every=500,
                                 on_progress=lambda **kw: prog.log(split="train", **kw))
                self.store.save(K.TRAIN_FEATURES, base, F, feat=feat, **pkey)
                out["train_features"] = list(F.shape)
            # e_n: readout-independent, stored once under _EN_FEAT (ScoreUseCase
            # loads it from there); process segment already separates cfm vs ddpm.
            # It deliberately carries no step segment, so a mid-checkpoint run must
            # NOT write it — it would compute the mid model's residuals and store
            # them at the final model's address, silently corrupting every das1
            # score for that identity.
            if no_error_weight:
                out["error_weight"] = "skipped (--no-error-weight)"
            elif ckpt_step is not None:
                out["error_weight"] = "skipped (mid-checkpoint run)"
            elif self.store.exists(K.ERROR_WEIGHT, base, feat=_EN_FEAT) and not force:
                out["error_weight"] = "skipped"
            else:
                from .config import resolve_t_avg
                e_n = compute_error_weight(model, proc, train_ds, seed=seed,
                                           T_error=resolve_t_avg(self.cfg, dataset),
                                           batch_size=self._feat_batch(dataset),
                                           max_samples=max_samples, device=self.device)
                self.store.save(K.ERROR_WEIGHT, base, e_n, feat=_EN_FEAT)
                out["error_weight"] = list(e_n.shape)

        if split in ("query", "both"):
            qspec = RunSpec(dataset=dataset, seed=seed, process=process,
                            query_type=query_type, conditional=conditional)
            if self.store.exists(K.QUERY_FEATURES, qspec, feat=feat, **pkey) and not force:
                log.info("query features exist, skipping (use --force)")
                out[f"query_{query_type}"] = "skipped"
            else:
                query_ds = self._query_dataset(base, test_ds, query_type)
                Fq = fz.featurize(query_ds, seed=seed, max_samples=max_samples, log_every=500,
                                  on_progress=lambda **kw: prog.log(split=f"query_{query_type}", **kw))
                self.store.save(K.QUERY_FEATURES, qspec, Fq, feat=feat, **pkey)
                out[f"query_{query_type}"] = list(Fq.shape)
        return out

    def _query_dataset(self, base: RunSpec, test_ds, query_type: str):
        """The query images for a track. Shared so the gradient and embedding
        paths cannot drift into attributing over different query sets."""
        if query_type == "val":
            # `dataset` (a NameError left by the resolve_Q refactor) -> base.dataset;
            # same value for every dataset that has ever run this path, since
            # resolve_Q falls back to lds.Q when the table has no row.
            return select_val_queries(test_ds, resolve_Q(self.cfg, base.dataset))
        if query_type == "inject":
            queries = self.store.load(K.INJECT_QUERIES, base)
            if self._is_latent(base.dataset):
                from .latent import encode_query_images
                return encode_query_images(
                    self.cfg, base.dataset, queries["images_u8"],
                    queries["host_labels"], device=self.device)
            images = torch.as_tensor(queries["images_u8"]).float().div(127.5).sub(1.0)
            return ImageDataset(images, torch.as_tensor(queries["host_labels"]).long())
        gen = self.store.load(K.GENERATION, base)
        return ImageDataset(gen["samples"], gen["labels"])

    def _journey(self, dataset: str, seed: int, *, process: str, split: str,
                 query_type: str, conditional: bool, proj_dim, proj_seed,
                 max_samples=None, force: bool = False) -> dict:
        """Journey-TRAK query features ([e]): gradients along the generation path.

        Only the QUERY side and only the ``gen`` track. A val query is a real
        held-out image the model never generated, so it has no journey — pairing
        one with trajectory features would silently compare two different things.
        The train side stays the ordinary gradient features, which is what the
        method's ``feat_method`` names.
        """
        if split not in ("query", "both"):
            raise ValueError("'journey' is a query-side feature only; the train side of "
                             "journey_trak uses the ordinary gradient features "
                             "(--feat das_T100 etc.)")
        if query_type != "gen":
            raise ValueError("'journey' exists only for the gen track: val/inject queries "
                             "have no generation trajectory")
        base = RunSpec(dataset=dataset, seed=seed, process=process, conditional=conditional)
        qspec = RunSpec(dataset=dataset, seed=seed, process=process,
                        query_type=query_type, conditional=conditional)
        set_context(run_id=base.digest()[:6], trace_id=f"journey:{dataset}:{process}")
        fcfg = self.cfg["featurize"]
        pdim = int(proj_dim) if proj_dim is not None else int(fcfg["proj_dim"])
        pseed = int(proj_seed) if proj_seed is not None else int(fcfg["proj_seed"])
        pkey = ({"proj": f"p{pdim}s{pseed}"}
                if (proj_dim is not None or proj_seed is not None) else {})
        if self.store.exists(K.QUERY_FEATURES, qspec, feat=_JOURNEY_FEAT, **pkey) and not force:
            log.info("journey features exist, skipping (use --force)")
            return {"feat": _JOURNEY_FEAT, f"query_{query_type}": "skipped"}
        if not self.store.exists(K.GEN_TRAJECTORY, base):
            raise FileNotFoundError(
                "no generation trajectory for this identity — run "
                "`balds generate --trajectory ...` first (it replays the stored "
                "generation's ODE and captures the intermediate states)")
        traj = self.store.load(K.GEN_TRAJECTORY, base)
        states, steps = traj["states"], traj["steps"]
        ode_steps = int(traj["ode_steps"])
        # euler_solve holds state x at t = i/steps before taking step i, so the
        # captured index maps to that same time — the model must see the time it
        # actually ran at, not a re-derived grid.
        t_values = [float(int(s)) / ode_steps for s in steps]
        # platform seam: the SD3.5 model is rebuilt from its LoRA-only checkpoint
        # and featurized at the platform's own batch size. The readout, the time
        # grid and the projection are the SAME for both platforms -- the journey
        # is the same estimator over a different model.
        model = self._load_model(base, {})
        if self._is_latent(dataset):
            from .latent import platform_process
            proc, _ = platform_process(self.cfg, dataset, device=self.device)
        else:
            proc = PROCESSES.get(process)
        fz = GradFeaturizer(model, proc, proj_dim=pdim, proj_seed=pseed,
                            T=len(t_values), loss_type=_JOURNEY_READOUT,
                            batch_size=self._feat_batch(dataset), device=self.device,
                            projection=str(self.cfg["featurize"].get("projection", "auto")))
        prog = ProgressCSV(progress_path(self.cfg["storage"]["data_root"],
                                         f"featurize_{dataset}_{process}_journey_seed{seed}"))
        if max_samples is not None:              # smoke/debug only, same as --split query
            n = min(int(max_samples), int(states.shape[0]))
            states, labels = states[:n], traj["labels"][:n]
        else:
            labels = traj["labels"]
        Fq = fz.featurize_states(states, t_values, labels,
                                 on_progress=lambda **kw: prog.log(split="journey", **kw))
        self.store.save(K.QUERY_FEATURES, qspec, Fq, feat=_JOURNEY_FEAT, **pkey)
        log.info("journey features: %s over %d trajectory points %s",
                 list(Fq.shape), len(t_values), [round(t, 3) for t in t_values])
        return {"feat": _JOURNEY_FEAT, "points": len(t_values),
                f"query_{query_type}": list(Fq.shape)}

    #: raw 256px images held resident while streaming an embedding split. The
    #: features are ~0.2 GB but the pixels behind them are 3.9 GB (5000 x 3 x
    #: 256 x 256 fp32), so the latent platform's image source is consumed in
    #: chunks. Rounded to a multiple of the embed batch size at the call site so
    #: the chunking cannot move a batch boundary - CLIP's own loop is what makes
    #: the numbers, and it must see the same batches either way.
    _EMBED_CHUNK = 64

    def _train_test_images(self, dataset: str):
        """Raw-IMAGE (train, test) datasets - the pixels, not the latents."""
        return _load_ds(self.cfg, dataset)

    def _latent_image_split(self, dataset: str, split: str):
        """The raw 256px images behind a cached latent split, order pinned.

        The embedders take IMAGES; every other stage of the latent platform
        takes the cached VAE latents. The two must enumerate the SAME samples in
        the SAME order, or the pixel/CLIP baseline attributes over a permutation
        of the training set - a silently wrong answer, not a crash. Both sides
        come from the one deterministic loader (``_load_artbench2_256`` fixes its
        2500/style subsample on RandomState(42) and sorts by filename), and the
        cached split's labels are compared element-wise here so a future
        reordering of either side fails loudly instead of quietly.
        """
        train_ds, test_ds = self._train_test_images(dataset)
        ds = train_ds if split == "train" else test_ds
        cached = self.store.load(K.LATENTS, RunSpec(dataset=dataset), split=split)["labels"]
        cached = torch.as_tensor(cached).long()
        have = torch.as_tensor(ds.labels).long()
        if have.shape != cached.shape or not bool(torch.equal(have, cached)):
            raise ValueError(
                f"the raw {split} images of {dataset} do not line up with the cached "
                f"latents/{dataset}/{split}.pt ({tuple(have.shape)} vs {tuple(cached.shape)} "
                f"labels): the image-space baselines would attribute over a different "
                f"sample order than every latent-space stage")
        return ds

    def _decode_generation(self, dataset: str, latents) -> torch.Tensor:
        """VAE-decode stored generation latents to images in [-1, 1].

        Deliberately NOT persisted: ``generations/.../samples.pt`` is a
        production artifact and this decode is derivable from it. (The AB2
        generation already carries ``images_u8``; this is the fallback for a
        generation produced before that field existed.)
        """
        from balds.models.sd3 import decode_latents
        from .latent import platform_cfg, platform_spec
        acfg = platform_cfg(self.cfg, dataset)
        from balds.models.sd3 import load_vae
        vae = load_vae(acfg["base_model"], device=self.device)
        imgs = decode_latents(vae, latents, device=self.device).float().cpu()
        del vae
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return imgs

    def _latent_query_images(self, base: RunSpec, query_type: str):
        """``(image-source, indices)`` for a query track on the latent platform."""
        if query_type == "inject":
            queries = self.store.load(K.INJECT_QUERIES, base)
            images = torch.as_tensor(queries["images_u8"]).float().div(127.5).sub(1.0)
            return ImageDataset(images, torch.as_tensor(queries["host_labels"]).long()), \
                list(range(len(images)))
        if query_type == "val":
            ds = self._latent_image_split(base.dataset, "test")
            idx = balanced_query_indices(ds.labels, resolve_Q(self.cfg, base.dataset))
            return ds, idx
        gen = self.store.load(K.GENERATION, base)
        imgs = (gen["images_u8"].float() / 127.5 - 1.0 if "images_u8" in gen
                else self._decode_generation(base.dataset, gen["samples"]))
        return ImageDataset(imgs, gen["labels"]), list(range(int(imgs.shape[0])))

    def _embed_stream(self, feat: str, ds, indices, *, bs: int, resize, prog=None,
                      tag: str = "") -> torch.Tensor:
        """Embed a lazily-decoded image source in chunks; concatenated result."""
        indices = list(indices)
        chunk = bs * max(1, self._EMBED_CHUNK // max(1, bs))
        parts, total = [], len(indices)
        for start in range(0, total, chunk):
            sl = indices[start:start + chunk]
            imgs = torch.stack([torch.as_tensor(ds[int(i)][0]) for i in sl])
            parts.append(embed(feat, imgs, device=self.device, batch_size=bs, resize=resize))
            done = min(start + chunk, total)
            log.info("embed %s %s: %d/%d", feat, tag, done, total)
            if prog is not None:
                prog.log(done=done, total=total, phase=tag)
        return torch.cat(parts, dim=0)

    def _embed(self, dataset: str, seed: int, *, process: str, feat: str, split: str,
               query_type: str, max_samples, conditional: bool, force: bool) -> dict:
        """The non-gradient baselines ([k]): raw pixels and CLIP embeddings.

        Deliberately bypasses the whole gradient pipeline. There is no model, no
        readout, no timestep count, no projection, no error weight and no
        sigma-hat here - an embedding of an image is none of those things, and
        letting it inherit that machinery is how a baseline quietly stops being
        the baseline it is named after.

        On the latent platform ([o]) the only thing that changes is where the
        IMAGES come from: the raw 256px folder for train/val (order pinned to the
        latent cache) and the generation's decoded previews for the gen track.
        The embedder itself is platform-independent - it always eats an
        ``(N, 3, H, W)`` tensor in [-1, 1].
        """
        base = RunSpec(dataset=dataset, seed=seed, process=process, conditional=conditional)
        set_context(run_id=base.digest()[:6], trace_id=f"embed:{dataset}:{feat}")
        if split == "repeat":
            raise ValueError(f"'{feat}' is deterministic - there is no featurization "
                             f"noise to measure, so --split repeat is meaningless")
        latent = self._is_latent(dataset)
        # pixel's dimension IS the image's; the 256px platforms flatten an
        # area-averaged 64x64 rather than a 196,608-vector (resolve_pixel_resize).
        resize = resolve_pixel_resize(self.cfg, dataset) if feat == "pixel" else None
        bs = int(self.cfg["featurize"]["batch_size"])
        train_ds = test_ds = None
        if not latent:
            train_ds, test_ds = self._train_test_images(dataset)
        prog = ProgressCSV(progress_path(self.cfg["storage"]["data_root"],
                                         f"featurize_{dataset}_{process}_{feat}_seed{seed}"))
        out: dict = {"feat": feat, "process": process, "embedder": feat,
                     "pixel_resize": resize}
        dim = None
        if latent:
            from .latent import platform_of
        platform = platform_of(self.cfg, dataset) if latent else "unet"
        log.info("embed %s feat=%s split=%s platform=%s resize=%s", dataset, feat, split,
                 platform, resize)

        if split in ("train", "both"):
            if self.store.exists(K.TRAIN_FEATURES, base, feat=feat) and not force:
                out["train_features"] = "skipped"
            else:
                if latent:
                    src = self._latent_image_split(dataset, "train")
                    n = len(src) if max_samples is None else min(int(max_samples), len(src))
                    F = self._embed_stream(feat, src, range(n), bs=bs, resize=resize,
                                           prog=prog, tag="train")
                else:
                    imgs = (train_ds.images if max_samples is None
                            else train_ds.images[:max_samples])
                    F = embed(feat, imgs, device=self.device, batch_size=bs, resize=resize)
                self.store.save(K.TRAIN_FEATURES, base, F, feat=feat)
                out["train_features"] = list(F.shape)
                dim = int(F.shape[1])

        if split in ("query", "both"):
            qspec = RunSpec(dataset=dataset, seed=seed, process=process,
                            query_type=query_type, conditional=conditional)
            if self.store.exists(K.QUERY_FEATURES, qspec, feat=feat) and not force:
                out[f"query_{query_type}"] = "skipped"
            else:
                if latent:
                    src, idx = self._latent_query_images(base, query_type)
                    Fq = self._embed_stream(feat, src, idx, bs=bs, resize=resize,
                                            prog=prog, tag=f"query_{query_type}")
                else:
                    qds = self._query_dataset(base, test_ds, query_type)
                    Fq = embed(feat, qds.images, device=self.device, batch_size=bs,
                               resize=resize)
                self.store.save(K.QUERY_FEATURES, qspec, Fq, feat=feat)
                out[f"query_{query_type}"] = list(Fq.shape)
                dim = int(Fq.shape[1])

        # Recipe sidecar. `pixel_resize` is the one knob the path does NOT encode
        # (a 64x64 pixel feature and a 256x256 one are both "featurize/pixel/..."),
        # so the table note that has to say which one it is reads it from here.
        if dim is not None and (force or not self.store.exists(K.FEATURE_META, base, feat=feat)):
            self.store.save(K.FEATURE_META, base, {
                "feat": feat, "kind": "embedding", "embedder": feat, "dim": dim,
                "pixel_resize": resize, "platform": platform,
                "image_source": ("raw 256px folder (train/val) + generation images_u8 (gen)"
                                 if latent else "dataset images"),
                "gradient": False, "projection": None, "timesteps": None,
            }, feat=feat)
        return out
