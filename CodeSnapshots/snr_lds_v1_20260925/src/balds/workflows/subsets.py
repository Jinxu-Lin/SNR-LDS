"""SubsetsUseCase paper reproduction stage."""
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

class SubsetsUseCase:
    """Stage 04: masks -> train M subset models -> GT loss matrix (M, Q)."""

    def __init__(self, store, cfg, *, device: str = "cuda:0") -> None:
        self.store, self.cfg, self.device = store, cfg, device

    def _is_latent(self, dataset: str) -> bool:
        from .latent import is_latent_platform
        return is_latent_platform(self.cfg, dataset)

    def generate_masks(self, dataset: str, seed: int = 42, *, M: Optional[int] = None,
                       alpha: Optional[float] = None, conditional: bool = True) -> dict:
        spec = RunSpec(dataset=dataset, seed=seed, conditional=conditional)
        if self.store.exists(K.SUBSET_MASKS, spec):
            log.info("masks exist, skipping"); return {"skipped": True}
        from .config import resolve_M
        M = M or resolve_M(self.cfg, dataset)
        alpha = alpha if alpha is not None else float(self.cfg["lds"]["alpha"])
        if self._is_latent(dataset):
            # N from the cached latents — the raw 256px folder need not be here
            from .latent import latent_train_plain
            N = len(latent_train_plain(self.store, dataset))
        else:
            train_ds, _ = _load_ds(self.cfg, dataset)
            N = len(train_ds)
        masks = generate_subsets(N, alpha, M, seed=42)
        self.store.save(K.SUBSET_MASKS, spec, [m.astype(np.bool_) for m in masks])
        log.info("generated %d masks N=%d alpha=%.2f", M, N, alpha)
        return {"M": M, "N": N}

    def train_subsets(self, dataset: str, seed: int, *, process: str = "cfm",
                      rank: int = 0, world: int = 1, replica: int = 0,
                      conditional: bool = True) -> dict:
        if self._is_latent(dataset):
            from .latent import run_latent_train_subsets
            return run_latent_train_subsets(self.store, self.cfg, dataset, seed,
                                            process=process, device=self.device,
                                            rank=rank, world=world, replica=replica)
        spec = RunSpec(dataset=dataset, process=process, seed=seed, conditional=conditional)
        rkey = {"replica": replica} if replica else {}
        masks = self.store.load(K.SUBSET_MASKS, spec)
        train_ds, _ = _load_ds(self.cfg, dataset)
        t = self.cfg["train"]
        num_classes = self.cfg["model"]["num_classes"] if conditional else None
        tasks = [m for m in range(len(masks)) if m % world == rank]
        log.info("training %d/%d subset models (rank %d/%d, replica %d), batch=%d",
                 len(tasks), len(masks), rank, world, replica, t["batch_size"])
        rep_tag = f"_r{replica}" if replica else ""
        prog = ProgressCSV(progress_path(
            self.cfg["storage"]["data_root"],
            f"subsets_{dataset}_{process}_seed{seed}{rep_tag}_rank{rank}of{world}"))
        for i, m in enumerate(tasks):
            if self.store.exists(K.SUBSET_CHECKPOINT, spec, m=m, **rkey):
                prog.log(done=i + 1, total=len(tasks), subset_m=m, final_loss="",
                         skipped=1)
                continue
            sub = SubsetView(ImageDataset(train_ds.images, train_ds.labels), masks[m])
            # Same recipe as the full model, full epoch budget on the subset's
            # ACTUAL size ([g]/D4). Replica runs ([m], noise-floor) reuse the
            # same masks but shift the retrain seed by a fixed large offset.
            from .config import resolve_p_uncond
            steps = _subset_steps(self.cfg, len(sub), t["batch_size"])
            retrain_seed = seed + m + 1 + replica * 100003
            model, hist = train_model(process, sub, steps=steps,
                                      batch_size=t["batch_size"], lr=t["lr"], seed=retrain_seed,
                                      num_classes=num_classes, p_uncond=resolve_p_uncond(self.cfg, dataset),
                                      device=self.device, base_ch=self.cfg["model"]["base_ch"],
                                      **_recipe_kwargs(t))
            ckpt = checkpoint_state(model, base_ch=self.cfg["model"]["base_ch"],
                                    num_classes=num_classes, step=steps,
                                    extra={"model_type": process, "conditional": conditional,
                                           "p_uncond": resolve_p_uncond(self.cfg, dataset) if conditional else None})
            self.store.save(K.SUBSET_CHECKPOINT, spec, ckpt, m=m, **rkey)
            prog.log(done=i + 1, total=len(tasks), subset_m=m,
                     final_loss=hist[-1]["loss"] if hist else "", skipped=0)
        return {"trained": len(tasks), "replica": replica}

    def _loss_row(self, spec, m: int, queries, labels, process, sched, T_avg,
                  target_batch, e_seed, rkey):
        """GT loss of subset model ``m`` on every query; shape ``(Q,)``.

        Row ``m`` depends only on (subset ``m``, ``e_seed``) — never on iteration
        order or which other rows share the run (``compute_query_losses`` seeds its
        noise per e_seed/timestep). That independence is what makes the sharded
        path below reassemble byte-identically to the single-process one.
        """
        ckpt = self.store.load(K.SUBSET_CHECKPOINT, spec, m=m, **rkey)
        if self._is_latent(spec.dataset):
            from .latent import build_latent_model
            model = build_latent_model(self.store, self.cfg, spec.dataset, ckpt,
                                       device=self.device)
        else:
            model = build_model(ckpt, device=self.device)
        proc_obj = getattr(self, "_loss_process", PROCESSES.get(process))
        row = compute_query_losses(model, queries, labels, process, self.device,
                                   process=proc_obj, schedule=sched,
                                   T_avg=T_avg, seed=e_seed,
                                   target_batch=target_batch)
        del model
        torch.cuda.empty_cache()
        return row

    def compute_losses(self, dataset: str, seed: int, *, process: str = "cfm",
                       query_type: str = "gen", e_seed: int = 0, replica: int = 0,
                       rank: int = 0, world: int = 1, conditional: bool = True,
                       force: bool = False, chain: Optional[int] = None) -> dict:
        """GT losses of one subset chain on one identity's queries.

        ``chain`` (§6.2-14 decoupling, 2026-08-17): the subset-chain seed, when
        it differs from the query seed — "chain R's 64 subset models scored on
        identity S's queries", the cross combinations that gt_avg averages.
        Omitted or equal to ``seed`` = the legacy diagonal (byte-identical
        artifacts). The chain's checkpoints are read from its own tree; the
        queries, masks and ζ noise (``e_seed``) are the identity's, so a cross
        matrix is comparable row-for-row with the diagonal one.
        """
        spec = RunSpec(dataset=dataset, process=process, seed=seed, query_type=query_type,
                       conditional=conditional)
        if chain is not None and int(chain) == int(seed):
            chain = None                    # diagonal: keep the legacy artifact name
        if chain is not None and query_type == "val":
            raise ValueError(
                "val queries are seed-independent, so 'chain R on identity S' is the "
                f"same computation as chain R's own val losses — run `subsets losses "
                f"--seed {int(chain)} --query-type val` instead (gt_avg reads those "
                f"files directly; §6.2-14 val track costs zero new compute)")
        rkey = {"replica": replica} if replica else {}
        ckey = {**rkey, **({"chain": int(chain)} if chain is not None else {})}
        if self.store.exists(K.GT_LOSSES, spec, eseed=e_seed, **ckey) and not force:
            log.info("gt_losses exist (query_type=%s eseed=%d chain=%s), skipping",
                     query_type, e_seed, chain)
            return {"skipped": True}
        masks = self.store.load(K.SUBSET_MASKS, spec)
        M = len(masks)
        queries, labels = self._queries(spec, dataset, process)
        if self._is_latent(dataset):
            from .latent import platform_process
            proc_obj, _ = platform_process(self.cfg, dataset, device=self.device)
        else:
            proc_obj = PROCESSES.get(process)
        sched = (proc_obj._schedule(self.device) if process == "ddpm" else None)
        self._loss_process = proc_obj
        from .config import resolve_t_avg
        T_avg = resolve_t_avg(self.cfg, dataset)
        target_batch = int(self.cfg["lds"].get("chunk_target_batch", 2000))
        # subset checkpoints come from the CHAIN's tree; everything else (queries,
        # e_seed noise, masks) stays the identity's
        ckpt_spec = spec if chain is None else spec.with_(seed=int(chain))

        def _one(m):  # bind the per-run arguments once
            return self._loss_row(ckpt_spec, m, queries, labels, process, sched,
                                  T_avg, target_batch, e_seed, rkey)

        rep_tag = f"_r{replica}" if replica else ""
        chain_tag = f"_chain{chain}" if chain is not None else ""
        if world == 1:
            # Single-process path: build the (M, Q) float64 matrix in memory and
            # save it directly — byte-identical to the pre-sharding implementation.
            prog = ProgressCSV(progress_path(
                self.cfg["storage"]["data_root"],
                f"gtlosses_{dataset}_{process}_seed{seed}{rep_tag}{chain_tag}_{query_type}_e{e_seed}"))
            loss_matrix = np.zeros((M, queries.shape[0]))
            for m in range(M):
                loss_matrix[m] = _one(m)
                log.info("subset %d/%d mean_loss=%.4f", m, M, loss_matrix[m].mean())
                prog.log(done=m + 1, total=M, mean_loss=round(float(loss_matrix[m].mean()), 6))
            self.store.save(K.GT_LOSSES, spec, loss_matrix, eseed=e_seed, **ckey)
            return {"shape": list(loss_matrix.shape), "e_seed": e_seed,
                    "replica": replica, "chain": chain}

        # Sharded path (world > 1): this rank computes only rows m % world == rank,
        # persisting each as its own GT_LOSSROW so a killed rank resumes per-subset
        # (not per-eseed) and N GPUs can split one eseed. The rank that finds all M
        # rows present assembles the full float64 matrix (identical to world==1).
        mine = [m for m in range(M) if m % world == rank]
        prog = ProgressCSV(progress_path(
            self.cfg["storage"]["data_root"],
            f"gtlosses_{dataset}_{process}_seed{seed}{rep_tag}{chain_tag}_{query_type}_e{e_seed}_rank{rank}of{world}"))
        done = 0
        for m in mine:
            if self.store.exists(K.GT_LOSSROW, spec, eseed=e_seed, m=m, **ckey) and not force:
                done += 1
                prog.log(done=done, total=len(mine), skipped_subset=m)
                continue
            row = _one(m)
            self.store.save(K.GT_LOSSROW, spec, row, eseed=e_seed, m=m, **ckey)
            done += 1
            log.info("subset %d (rank %d/%d) mean_loss=%.4f", m, rank, world, float(row.mean()))
            prog.log(done=done, total=len(mine), mean_loss=round(float(row.mean()), 6))
        return self._finalize_gt_losses(spec, e_seed, M, rank=rank, world=world, rkey=ckey)

    def _finalize_gt_losses(self, spec, e_seed: int, M: int, *, rank: int, world: int, rkey=None) -> dict:
        """Assemble the per-subset GT_LOSSROW shards into the full (M, Q) gt_losses
        matrix, once every row is present. A no-op until then, so whichever rank
        finishes last performs the assembly; if that rank dies first, a cheap re-run
        (every row skips) finalizes. Row loads are guarded because a peer rank may be
        mid-write — that just defers finalize to the next completing rank."""
        rkey = rkey or {}
        present = [m for m in range(M) if self.store.exists(K.GT_LOSSROW, spec, eseed=e_seed, m=m, **rkey)]
        if len(present) < M:
            return {"finalized": False, "rows": len(present), "need": M,
                    "rank": rank, "world": world, "e_seed": e_seed}
        try:
            rows = [self.store.load(K.GT_LOSSROW, spec, eseed=e_seed, m=m, **rkey) for m in range(M)]
        except Exception as exc:  # a peer rank may be mid-write; let it finalize
            log.info("gt_losses finalize deferred (row read failed: %s)", exc)
            return {"finalized": False, "rows": len(present), "need": M, "deferred": True}
        loss_matrix = np.zeros((M, rows[0].shape[0]))  # float64, matching world==1
        for m in range(M):
            loss_matrix[m] = rows[m]
        self.store.save(K.GT_LOSSES, spec, loss_matrix, eseed=e_seed, **rkey)
        log.info("gt_losses finalized from %d subset rows (eseed=%d)", M, e_seed)
        return {"finalized": True, "shape": list(loss_matrix.shape), "e_seed": e_seed}

    def compute_gt(self, dataset: str, seed: int, *, process: str = "cfm",
                   query_type: str = "gen", e_seeds: Optional[list[int]] = None,
                   replica: int = 0, conditional: bool = True,
                   chains: Optional[list[int]] = None) -> dict:
        """Average GT losses into the consumer-facing ``gt_matrix``.

        Without ``chains``: the legacy single-chain average over ``e_seeds``
        (byte-identical behaviour). With ``chains`` (§6.2-14 gt_avg): average
        over subset-retrain chains × ζ replicas and write the result to the
        SAME ``gt_matrix`` path — this IS the adjudicated switch of the main
        table's GT, so score/evaluate/gamma-scan pick it up transparently.
        Per chain R: the diagonal (R == seed) reads the identity's own losses;
        gen-track crosses read the ``chain``-keyed files from ``subsets losses
        --chain R``; val-track crosses read chain R's own val losses (val
        queries are seed-independent — zero new compute). Strict by design:
        every (chain, e_seed) cell must exist, because a silently partial
        average would produce a plausible matrix with wrong provenance.
        """
        spec = RunSpec(dataset=dataset, process=process, seed=seed, query_type=query_type,
                       conditional=conditional)
        rkey = {"replica": replica} if replica else {}
        e_seeds = e_seeds if e_seeds is not None else [0]

        if chains:
            if replica:
                raise ValueError("gt_avg over chains and the noise-floor replica are "
                                 "different instruments — a replica GT is single-chain")
            chains = [int(c) for c in chains]
            mats, missing = [], []
            for ch in chains:
                for e in e_seeds:
                    if ch == int(seed):
                        k_spec, key = spec, {"eseed": e}
                    elif spec.is_val:
                        # chain R on the shared val queries == chain R's own val
                        # losses; read the existing diagonal file of identity R
                        k_spec, key = spec.with_(seed=ch), {"eseed": e}
                    else:
                        k_spec, key = spec, {"eseed": e, "chain": ch}
                    if self.store.exists(K.GT_LOSSES, k_spec, **key):
                        mats.append(self.store.load(K.GT_LOSSES, k_spec, **key))
                    else:
                        missing.append((ch, e))
            if missing:
                raise FileNotFoundError(
                    f"gt_avg needs every (chain, e_seed) cell; missing {missing} for "
                    f"{dataset}/{query_type}/seed_{seed} — run `subsets losses` "
                    f"(gen: with --chain R; val: as identity R) for those first")
            Mmin = min(mat.shape[0] for mat in mats)
            gt = np.mean([mat[:Mmin] for mat in mats], axis=0)
            self.store.save(K.GT_MATRIX, spec, gt, **rkey)
            log.info("gt_avg matrix %s = mean over %d chains %s x %d e_seeds (§6.2-14)",
                     gt.shape, len(chains), chains, len(e_seeds))
            return {"shape": list(gt.shape), "n_eseeds": len(e_seeds),
                    "chains": chains, "gt_avg": True}

        mats = []
        for e in e_seeds:
            if self.store.exists(K.GT_LOSSES, spec, eseed=e, **rkey):
                mats.append(self.store.load(K.GT_LOSSES, spec, eseed=e, **rkey))
        if not mats:
            raise FileNotFoundError("no gt_losses found; run compute_losses first")
        Mmin = min(mat.shape[0] for mat in mats)
        gt = np.mean([mat[:Mmin] for mat in mats], axis=0)
        self.store.save(K.GT_MATRIX, spec, gt, **rkey)
        log.info("GT matrix %s averaged over %d e_seeds", gt.shape, len(mats))
        return {"shape": list(gt.shape), "n_eseeds": len(mats)}

    def _queries(self, spec, dataset, process):
        if spec.is_val:
            if self._is_latent(dataset):
                from .latent import latent_test
                test = latent_test(self.store, dataset)
            else:
                _, test = _load_ds(self.cfg, dataset)
            vq = select_val_queries(test, resolve_Q(self.cfg, dataset))
            return vq.images, vq.labels
        gen = self.store.load(K.GENERATION, spec.with_(query_type="gen"))
        return gen["samples"], gen["labels"]
