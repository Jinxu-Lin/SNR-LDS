"""EK-FAC influence use-cases (fit the curvature package; score with it).

Two stages, mirroring the featurize -> score split of the projection-kernel
methods — but nothing here consumes projected features, e_n or sigma-hat: the
whole point of the K-FAC Influence baseline is a curvature-preconditioned inner
product of *unprojected* per-layer gradients.

* ``EkfacFitUseCase``   — two MC passes over the training set produce the
  per-layer eigenbases + corrected eigenvalues (``EKFAC_FACTORS`` artifact,
  query-independent, one per (dataset, seed, process)).
* ``EkfacScoreUseCase`` — streams per-sample gradients once and evaluates the
  whole damping grid in that single pass (scores decompose per damping in the
  eigenbasis), then reuses the pipeline's λ-selector machinery to pick the
  reported damping. Saves a standard ``SCORES`` artifact under method name
  ``ekfac_if`` so ``balds evaluate --method ekfac_if`` works unchanged.
"""
from __future__ import annotations

import json
import socket
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from balds.schema.artifact import ArtifactKind as K
from balds.schema.logging import get_logger, set_context
from balds.schema.registry import PROCESSES, SELECTORS
from balds.schema.runspec import RunSpec
from balds.data.base import balanced_query_indices, select_val_queries
from balds.models import build_model
from balds.attribution.ekfac import EkfacFactors, EkfacFitter, EkfacScorer
from balds.attribution.shrinkage import (ShrinkageLayer, select_after_shrinkage,
                                      sigma_from_repeats, snr_stats)
from .config import lds_rows, resolve_Q, resolve_gamma
from balds.artifacts.progress import ProgressCSV, progress_path
from .common import _load_ds

log = get_logger("balds.ekfac")

#: Method naming (2026-09-03): the registry table in ``balds.attribution.methods``
#: is the single source of truth — raw curvature methods bind their solver
#: caliber (readout / sampling / damping_mode) to their NAME, layered names
#: ride on a raw method, and ``fmas`` resolves via the paper alias.
from balds.attribution.methods import CURVATURE_LAYERED, CURVATURE_RAW, VARIANT_TAGS
from balds.schema.registry import METHODS
from .config import resolve_paper_method

#: The reference EK-FAC IF baseline (Mlodozeniec et al.) — its progress files
#: keep their historical (suffix-free) names.
METHOD_NAME = "ekfac_if"
#: Solver-variant separator for ad-hoc audits: ``<raw>__<tag>`` with a registered tag.
VARIANT_SEP = "__"
#: Default caliber for names that bind nothing (the reference stream).
_REFERENCE = {"sampling": "iid", "damping_mode": "global"}


def split_method(method: str) -> tuple[str, dict, str | None]:
    """``(filing_name, solver, layer)`` for any `balds ekfac` method name.

    ``filing_name`` is the raw name the artifacts live under; ``solver`` is the
    bound caliber ``{readout, sampling, damping_mode}`` the name commits to;
    ``layer`` is ``None`` (bare) or ``"shrink"`` (f_c).
    ``fmas`` is resolved by the caller (``resolve_paper_method``) before this.
    """
    if method in CURVATURE_RAW:
        return method, dict(CURVATURE_RAW[method]), None
    if method in CURVATURE_LAYERED:
        raw, layer = CURVATURE_LAYERED[method]
        return raw, dict(CURVATURE_RAW[raw]), layer
    if VARIANT_SEP in method:
        layer = None
        variant = method
        if method.endswith("_tweedie"):
            variant, layer = method[:-len("_tweedie")], "shrink"
        base, tag = variant.split(VARIANT_SEP, 1)
        if base in CURVATURE_RAW and tag in VARIANT_TAGS:
            solver = {**CURVATURE_RAW[base], **_REFERENCE, **VARIANT_TAGS[tag]}
            solver["readout"] = CURVATURE_RAW[base]["readout"]
            for name, bound in CURVATURE_RAW.items():
                if bound == solver:
                    raise ValueError(
                        f"variant {variant!r} resolves to the same caliber as the named "
                        f"method {name!r}; use --method {name} (one caliber, one home)")
            return variant, solver, layer
        raise ValueError(
            f"unknown solver variant {method!r}: tags must be one of {sorted(VARIANT_TAGS)} "
            f"on a raw method {sorted(CURVATURE_RAW)} (optionally + _tweedie)")
    raise ValueError(f"balds ekfac score handles {sorted(CURVATURE_RAW)}, "
                     f"{sorted(CURVATURE_LAYERED)}, `fmas`, and registered "
                     f"'<raw>__<tag>' variants, not {method!r}")


def _bind_solver(e: dict, solver: dict, method: str) -> dict:
    """Return the ekfac config with the name-bound caliber applied.

    The method name IS the caliber: a config value equal to the reference
    default is overridden silently, any other value that disagrees with the
    binding is refused — scores filed under a name must be what that name
    says (the 2026-09-03 audit's silent-mismatch trap).
    """
    out = dict(e)
    for key in ("sampling", "damping_mode"):
        have, want = str(out.get(key, _REFERENCE[key])), str(solver[key])
        if have != want and have != _REFERENCE[key]:
            raise ValueError(
                f"--set ekfac.{key}={have!r} conflicts with method {method!r}, whose "
                f"caliber is bound to ekfac.{key}={want!r}. Drop the override (the name "
                f"already implies it) or score under a variant name that binds {have!r}.")
        out[key] = want
    return out


def _grid(e: dict) -> list[float]:
    """The selection grid: dampings λ (global) or shrinkage intensities ϱ (blockshrink).
    Either way the values key the per-grid-point score cache (``lam=``)."""
    key = "blockshrink_grid" if str(e.get("damping_mode", "global")) == "blockshrink" else "damping_grid"
    return [float(d) for d in e[key]]


def _method_suffix(raw_name: str) -> str:
    """Progress-file suffix: empty for the original ekfac_if (names unchanged)."""
    return "" if raw_name == METHOD_NAME else f"_{raw_name}"


def _ekfac_cfg(cfg) -> dict:
    e = dict(cfg.get("ekfac", {}) or {})
    e.setdefault("fit_epochs", 125)
    e.setdefault("eig_epochs", 125)
    e.setdefault("fit_batch_size", 64)
    e.setdefault("eig_batch_size", 16)
    e.setdefault("mc_loss", 250)
    e.setdefault("mc_measurement", 250)
    e.setdefault("grad_chunk", 125)
    e.setdefault("damping_grid",
                 [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])
    e.setdefault("query_coords_dtype", "bfloat16")
    e.setdefault("query_chunk", 0)              # 0 = all queries resident (CIFAR default); >0 = stream
    e.setdefault("row_chunk", 0)                # 0 = every train row in one partial; >0 = row tiles
                                                # (EKFAC-ROWSHARD: the axis --row-rank/--row-world split)
    e.setdefault("train_mode_for_loss_grads", True)
    e.setdefault("max_train_samples", None)     # debug/smoke only (fit pass cap)
    e.setdefault("sampling", "iid")             # iid (reference) | stratified_antithetic
    e.setdefault("damping_mode", "global")      # global (reference) | blockshrink (per-layer ϱ·mean Λ_l)
    e.setdefault("fit_seed_offset", 0)          # diagnostics only: re-fit factors on a different MC stream
    e.setdefault("blockshrink_grid", [1e-4, 1e-3, 1e-2, 1e-1, 1.0])   # ϱ grid when blockshrink
    return e


def _is_latent(cfg, dataset: str) -> bool:
    from .latent import is_latent_platform
    return is_latent_platform(cfg, dataset)


def _require_conditional(conditional: bool) -> None:
    if not conditional:
        raise ValueError("latent LoRA platforms are prompt-conditional; every stage "
                         "passes style indices as class_label — drop --uncond")


class EkfacFitUseCase:
    """Fit the EK-FAC curvature package for one (dataset, seed, process).

    On the SD3.5+LoRA platform (TASK_EKFAC_LORA, 2026-09-02) the covered layers
    are the ``lora_A``/``lora_B`` Linear submodules — two serial K-FAC blocks
    per adapter (§2 of the task book: Ω_A=E[xxᵀ], Γ_A=E[g_a g_aᵀ] rank², and
    Ω_B=E[aaᵀ] rank², Γ_B=E[g_b g_bᵀ]; scale rides in through autograd). The
    frozen base never enters (requires_grad filter). Data source = the latents
    cache; hflip augmentation = the pre-encoded flipped-latents cache (a latent
    cannot be flipped spatially), matching the training loop's coin flip.
    MC-Fisher normalisation η·√(2/D) picks D up from the model output shape,
    which IS the latent shape here — nothing to configure.
    """

    def __init__(self, store, cfg, *, device: str = "cuda:0") -> None:
        self.store, self.cfg, self.device = store, cfg, device

    def run(self, dataset: str, seed: int, *, process: str = "cfm",
            conditional: bool = True, force: bool = False) -> dict:
        latent = _is_latent(self.cfg, dataset)
        if latent:
            _require_conditional(conditional)
        base = RunSpec(dataset=dataset, seed=seed, process=process, conditional=conditional)
        set_context(run_id=base.digest()[:6], trace_id=f"ekfac-fit:{dataset}:{process}")
        if self.store.exists(K.EKFAC_FACTORS, base) and not force:
            log.info("ekfac factors exist, skipping (use --force)")
            return {"skipped": True}
        e = _ekfac_cfg(self.cfg)
        if latent:
            from .latent import platform_process
            proc, _ = platform_process(self.cfg, dataset, device=self.device)
        else:
            proc = PROCESSES.get(process)
        if latent:
            from .latent import build_latent_model, latent_train_pair
            model = build_latent_model(self.store, self.cfg, dataset,
                                       self.store.load(K.CHECKPOINT, base),
                                       device=self.device)
            pair = latent_train_pair(self.store, dataset)
            images, flipped, labels = pair.images, pair.images_flipped, pair.labels
            # train-time augmentation on this platform is ALWAYS the coin flip
            # between the two pre-encoded orientations (LatentPairDataset), so
            # the factor fit draws the same augmentation regardless of train.hflip
            hflip = True
        else:
            hflip = bool(self.cfg.get("train", {}).get("hflip", False))
            model = build_model(self.store.load(K.CHECKPOINT, base), device=self.device)
            train_ds, _ = _load_ds(self.cfg, dataset)
            images, labels = train_ds.images, (train_ds.labels if conditional else None)
            flipped = None
        mts = e.get("max_train_samples")
        if mts:
            # debug/smoke cap — factors fit on a slice are NOT production quality;
            # recorded in meta so a later consumer can tell.
            images = images[: int(mts)]
            labels = None if labels is None else labels[: int(mts)]
            flipped = None if flipped is None else flipped[: int(mts)]
            log.warning("ekfac fit capped to %d train samples (debug only)", int(mts))

        fitter = EkfacFitter(model, proc, device=self.device)
        prog = ProgressCSV(progress_path(self.cfg["storage"]["data_root"],
                                         f"ekfac_fit_{dataset}_{process}_seed{seed}"))
        log.info("ekfac fit: N=%d layers=%d fit_epochs=%d eig_epochs=%d hflip=%s",
                 images.shape[0], len(fitter.modules), e["fit_epochs"], e["eig_epochs"], hflip)

        A, B, n_draws = fitter.fit_factor_pass(
            images, labels, epochs=int(e["fit_epochs"]),
            batch_size=int(e["fit_batch_size"]), seed=seed + int(e.get("fit_seed_offset", 0)), hflip=hflip,
            flipped=flipped, on_progress=prog.log)
        bases = fitter.eigenbases(A, B)
        del A, B
        layers = fitter.fit_eigenvalue_pass(
            images, labels, bases, epochs=int(e["eig_epochs"]),
            batch_size=int(e["eig_batch_size"]), seed=seed + int(e.get("fit_seed_offset", 0)), hflip=hflip,
            flipped=flipped, on_progress=prog.log)
        meta = {
            "layer_order": list(fitter.modules),
            "fit_epochs": int(e["fit_epochs"]), "eig_epochs": int(e["eig_epochs"]),
            "n_factor_draws": n_draws, "hflip": hflip, "seed": seed,
            "process": process, "conditional": conditional, "dataset": dataset,
            "max_train_samples": mts,
        }
        if latent:
            # latent-only meta keys so the CIFAR factor artifact stays byte-stable.
            # train_mode_for_loss_grads (§3-f): SD3.5-M carries no dropout and no
            # batch-norm, so train() vs eval() is numerically a no-op on this
            # platform — recorded here rather than silently assumed.
            from .latent import platform_of
            meta.update({"platform": platform_of(self.cfg, dataset),
                         "hflip_source": "latent_pair_cache",
                         "train_mode_noop": True})
        factors = EkfacFactors(layers, meta=meta)
        self.store.save(K.EKFAC_FACTORS, base, factors.state_dict())
        log.info("ekfac factors saved: %d layers, %d preconditioned params",
                 len(layers), factors.num_params())
        return {"layers": len(layers), "n_params": factors.num_params(),
                "factor_draws": n_draws}


# --------------------------------------------------------------------------- #
# artifact readers shared with gamma-scan
# --------------------------------------------------------------------------- #

def load_repeat_scores(store, raw_spec: RunSpec) -> dict:
    """The ``(D, R, M, Q)`` repeat-scored subset for ``raw_spec`` (a raw ekfac method).

    Refuses with the remedy when absent — the shrinkage layer must never fall
    back to a guessed σ̂ (an σ̂ from the wrong source over-shrinks and the output
    gives nothing away; methods.py applies the same rule to the kernel path).
    """
    if not store.exists(K.REPEAT_SCORES, raw_spec):
        raise ValueError(
            f"'{raw_spec.method}_tweedie' needs the repeat-scored subset to measure sigma-hat; "
            f"run `balds ekfac repeats --method {raw_spec.method} --dataset {raw_spec.dataset} "
            f"--seed {raw_spec.seed} --query-type {raw_spec.query_type}` "
            f"(same mc/damping config) first")
    return store.load(K.REPEAT_SCORES, raw_spec)


def _check_repeat_caliber(reps: dict, e: dict, raw_spec: RunSpec) -> None:
    """Refuse a σ̂ measured under a different solver caliber than the one in play.

    ``sampling``/``damping_mode`` change what the repeat spread *is*, so consuming
    a σ̂ from the wrong scheme mis-shrinks silently — the output looks fine either
    way (the same failure mode that made σ̂'s source a hard error in the first
    place). Artifacts written before 2026-09-02 carry neither field; those warn
    rather than raise, since the pre-existing C2/C10 repeats are still valid for
    the runs that produced them.
    """
    for key, default in (("sampling", "iid"), ("damping_mode", "global")):
        if key not in reps:
            log.warning(
                "repeat_scores for %s predates caliber stamping (no %r recorded) — "
                "cannot verify it was measured under %s=%r; rerun `balds ekfac repeats "
                "--force` to stamp it", raw_spec.method, key, key, str(e.get(key, default)))
            continue
        got, want = str(reps[key]), str(e.get(key, default))
        if got != want:
            raise ValueError(
                f"repeat_scores were measured with {key}={got!r} but this run uses "
                f"{key}={want!r}; σ̂ from the wrong solver caliber over/under-shrinks "
                f"invisibly. Rerun `balds ekfac repeats --method {raw_spec.method} "
                f"--dataset {raw_spec.dataset} --seed {raw_spec.seed} "
                f"--query-type {raw_spec.query_type} --force` under the current config")


def load_raw_by_damping(store, raw_spec: RunSpec, dampings) -> dict:
    """``{damping: (N, Q) raw ekfac_if scores}`` from the per-damping cache."""
    missing = [d for d in dampings if not store.exists(K.SCORES_LAMBDA, raw_spec, lam=d)]
    if missing:
        raise ValueError(
            f"raw {raw_spec.method} scores are not cached for dampings {missing}; run "
            f"`balds ekfac score --method {raw_spec.method} --dataset {raw_spec.dataset} "
            f"--seed {raw_spec.seed} --query-type {raw_spec.query_type}` first "
            f"(it streams the train gradients once and caches every damping)")
    return {float(d): store.load(K.SCORES_LAMBDA, raw_spec, lam=d) for d in dampings}


# --------------------------------------------------------------------------- #
# shared preparation: model, queries, damping inverses, query eigen-coords
# --------------------------------------------------------------------------- #

def _prepare(store, cfg, device, dataset, seed, *, query_type, process, conditional,
             readout: str = "loss", e: dict | None = None, prepared_datasets=None):
    """Everything the train-side streams need, computed once: factors, model,
    train set, the query eigen-coordinates (production seed/phase — the σ̂
    measurement must NOT re-draw the query side, task brief B2.3) and the
    damped inverses for the configured grid. ``prepared_datasets`` optionally
    supplies a portable native-order pixel snapshot; legacy callers still load
    the registered dataset exactly as before."""
    e = e if e is not None else _ekfac_cfg(cfg)
    base = RunSpec(dataset=dataset, seed=seed, process=process, conditional=conditional)
    factors = EkfacFactors.from_state_dict(store.load(K.EKFAC_FACTORS, base))
    latent = _is_latent(cfg, dataset)
    if latent:
        from .latent import platform_process
        proc, _ = platform_process(cfg, dataset, device=device)
    else:
        proc = PROCESSES.get(process)
    if latent:
        _require_conditional(conditional)
        from .latent import build_latent_model, latent_test, latent_train_pair
        model = build_latent_model(store, cfg, dataset,
                                   store.load(K.CHECKPOINT, base), device=device)
        train_ds = latent_train_pair(store, dataset)
        test_ds = latent_test(store, dataset) if query_type == "val" else None
        hflip = True                       # coin flip between pre-encoded orientations
    else:
        model = build_model(store.load(K.CHECKPOINT, base), device=device)
        train_ds, test_ds = (prepared_datasets if prepared_datasets is not None
                             else _load_ds(cfg, dataset))
        hflip = bool(cfg.get("train", {}).get("hflip", False))
    mts = e.get("max_train_samples")
    if mts:
        # debug/smoke cap (same knob as the fit stage): a stream over a slice
        # yields a (mts, Q) matrix that is NOT a production score matrix
        m = int(mts)
        if latent:
            from balds.data.base import LatentPairDataset
            train_ds = LatentPairDataset(train_ds.images[:m],
                                         train_ds.images_flipped[:m], train_ds.labels[:m])
        else:
            from balds.data.base import ImageDataset
            train_ds = ImageDataset(train_ds.images[:m], train_ds.labels[:m])
        log.warning("ekfac train side capped to %d samples (debug only)", m)
    if query_type == "inject":
        if latent:
            from .latent import encode_query_images
            queries = store.load(K.INJECT_QUERIES, base)
            query_ds = encode_query_images(
                cfg, dataset, queries["images_u8"], queries["host_labels"],
                device=device)
            q_images, q_labels = query_ds.images, query_ds.labels
        else:
            queries = store.load(K.INJECT_QUERIES, base)
            q_images = torch.as_tensor(queries["images_u8"]).float().div(127.5).sub(1.0)
            q_labels = torch.as_tensor(queries["host_labels"]).long()
    elif query_type == "val":
        q_ds = select_val_queries(test_ds, resolve_Q(cfg, dataset))
        q_images, q_labels = q_ds.images, q_ds.labels
    else:
        gen = store.load(K.GENERATION, base)
        q_images, q_labels = gen["samples"], gen["labels"]
        q_want = resolve_Q(cfg, dataset)
        if q_images.shape[0] > q_want:
            # an imported generation can hold more queries than this run scores
            # (cifar2_das archives 1000): the first Q, the importer's own rule
            q_images, q_labels = q_images[:q_want], q_labels[:q_want]
    Q = q_images.shape[0]
    dampings = _grid(e)
    qdtype = getattr(torch, str(e["query_coords_dtype"]))
    ctx = SimpleNamespace(
        e=e, base=base, factors=factors, model=model, proc=proc,
        train_ds=train_ds, N=len(train_ds), q_images=q_images, q_labels=q_labels, Q=Q,
        dampings=dampings, qdtype=qdtype,
        mc_meas=int(e["mc_measurement"]), mc_loss=int(e["mc_loss"]),
        chunk=int(e["grad_chunk"]),
        hflip=hflip,
        train_flipped=getattr(train_ds, "images_flipped", None),
        train_mode=bool(e["train_mode_for_loss_grads"]),
        conditional=conditional, seed=seed, device=device, readout=readout,
        sampling=str(e["sampling"]), damping_mode=str(e["damping_mode"]),
    )
    ctx.scorer = EkfacScorer(model, proc, factors, device=device)
    ctx.P = factors.num_params()
    return ctx


def _gt_for_queries(cfg, dataset: str, query_type: str, gt: np.ndarray) -> np.ndarray:
    """The GT columns of the queries ``_prepare`` scores.

    The identity whenever the stored matrix is Q wide (every generated platform).
    An imported platform's GT can be wider (cifar2_das: 1000 columns in archive
    order); at a smaller Q the columns are the rows ``_prepare`` takes — the
    first Q for gen, ``balanced_query_indices`` over the test split for val — so
    score column q and GT column q are the same query.
    """
    Q = resolve_Q(cfg, dataset)
    if gt.shape[1] <= Q:
        return gt
    if query_type == "val":
        _, test_ds = _load_ds(cfg, dataset)
        return gt[:, balanced_query_indices(test_ds.labels, Q)]
    return gt[:, :Q]


def _query_chunk(ctx) -> int:
    """Queries per resident block. 0/unset = everything resident (the CIFAR
    default, unchanged); ArtBench production must set ``ekfac.query_chunk``
    (LoRA P≈95.6M ⇒ a full (Q=100, P) bf16 block ≈ 19 GB — task book §3-e)."""
    return int(ctx.e.get("query_chunk") or 0) or ctx.Q


def _query_pass(ctx, prog, q0: int = 0, q1: Optional[int] = None) -> torch.Tensor:
    """Eigen-coordinates for queries ``[q0, q1)`` — a ``(q1-q0, P)`` block.

    Each query's MC draw is keyed by its GLOBAL index (seed, "ekfac_meas", q),
    so a chunked pass reproduces bit-for-bit the block the resident pass would
    hold — chunking changes residency, never the estimator."""
    q1 = ctx.Q if q1 is None else q1
    qcoords = torch.empty(q1 - q0, ctx.P, dtype=ctx.qdtype, device=ctx.device)
    for j, q in enumerate(range(q0, q1)):
        lb = int(ctx.q_labels[q]) if ctx.conditional else None
        g = ctx.scorer.sample_gradient(ctx.q_images[q], lb, mc=ctx.mc_meas, chunk=ctx.chunk,
                                       seed=ctx.seed, phase="ekfac_meas", index=q,
                                       hflip=False, train_mode=False, readout=ctx.readout,
                                       sampling=ctx.sampling)
        qcoords[j] = ctx.scorer.to_eigencoords(g).to(ctx.qdtype)
        if (q + 1) % 10 == 0 or q + 1 == q1:
            prog.log(done=q + 1, total=ctx.Q, phase="query_grads")
    return qcoords


def _train_flip(ctx, index: int):
    return None if ctx.train_flipped is None else ctx.train_flipped[index]


def _check_finite_inverses(inv: dict, damping_mode: str) -> None:
    """Fail in seconds, not after a 40 h stream: a non-finite damped inverse
    (e.g. 1/(0+0) on a zero-curvature block under blockshrink) poisons every
    score through the sum over blocks (ArtBench-2 gen, 2026-09-07: rc=0 with
    500000/500000 NaN)."""
    for d, v in inv.items():
        bad = int((~torch.isfinite(v)).sum())
        if bad:
            raise ValueError(f"damped inverse for damping={d:g} ({damping_mode}) has {bad} "
                             f"non-finite entries; refusing to stream scores")


def _check_finite_row(row: dict, *, what: str, index: int) -> None:
    """Guard on the first streamed row(s): any NaN/Inf here means the whole
    matrix would be garbage, so stop before burning the card."""
    for d, v in row.items():
        if not np.isfinite(v).all():
            raise ValueError(f"{what}: non-finite scores at train index {index}, "
                             f"damping={d:g} — aborting before the full stream")


def _train_row(ctx, qcoords, inv, image, label, *, seed: int, index: int,
               flipped=None) -> dict:
    """``{damping: (Q_block,) scores}`` for one training sample under ``seed``."""
    u = _train_coords(ctx, image, label, seed=seed, index=index, flipped=flipped)
    return _score_coords(ctx, qcoords, inv, u)


def _train_coords(ctx, image, label, *, seed: int, index: int, flipped=None):
    """Native FP32 eigen-coordinates; independent of query track/block."""
    g = ctx.scorer.sample_gradient(image, label, mc=ctx.mc_loss, chunk=ctx.chunk,
                                   seed=seed, phase="ekfac_loss", index=index,
                                   hflip=ctx.hflip, train_mode=ctx.train_mode,
                                   readout=ctx.readout, sampling=ctx.sampling,
                                   flipped=flipped)
    return ctx.scorer.to_eigencoords(g)                       # (P,) float32


def _score_coords(ctx, qcoords, inv, u):
    """Keep the native per-row cast, matvec shape and damping order."""
    return {d: (qcoords @ (u * inv[d]).to(ctx.qdtype)).float().cpu().numpy()
            for d in ctx.dampings}


# --------------------------------------------------------------------------- #
# block partials: interrupt survival + two-level sharding
# (TASK EKFAC-SHARD 2026-09-09; TASK EKFAC-ROWSHARD 2026-09-09)
#
# A query block is a self-contained unit of work: the train side is streamed
# once per block and every MC draw on both sides is keyed by (seed, phase,
# GLOBAL index) — see ``_query_pass`` — so block [q0, q1) depends on nothing
# outside itself. Persisting each block as it completes therefore turns a 40 h
# stream into a resumable one, and handing different blocks to different
# processes/machines is a pure residency change, exactly like ``query_chunk``
# itself. Nothing here may touch a whole-matrix address: SCORES_LAMBDA / SCORES
# / REPEAT_SCORES are written only once EVERY block is on disk.
#
# The SAME argument goes one level down. ``_train_row``'s draw is keyed by
# (seed, "ekfac_loss", GLOBAL row index), so train row i's contribution to a
# block is independent of which rows were streamed before it, in what process,
# on what machine. A block partial is therefore addressed by TWO half-open
# ranges — the query range and the train-row range [r0, r1) — and a shard
# streams only its own rows. The two axes cost very differently:
#
#   * splitting the QUERY axis re-streams the whole train side per block, so
#     `world` shards do `world`x the train-gradient work in total;
#   * splitting the ROW axis re-runs only the block's query pass (Q_block MC
#     gradients against N rows' worth — under a percent for the production
#     shapes), so it is the axis that buys cards without buying work.
#
# Row tiles come from ``ekfac.row_chunk`` exactly as query blocks come from
# ``ekfac.query_chunk``: the config fixes the tiling, ``--row-rank/--row-world``
# only decide who computes which tile. That keeps the tiling — and hence the
# assembly — independent of how many cards happened to be available, so the
# master can assemble with no shard flags at all.
# --------------------------------------------------------------------------- #

#: Block-partial fields that must agree with the run consuming them — every one
#: of them changes what the numbers ARE. A mismatch is a hard error, never a
#: silent recompute: a resume that quietly mixes two calibers files a plausible
#: matrix whose columns disagree, and nothing downstream can tell.
_BLOCK_IDENTITY = ("dampings", "method", "dataset", "query_type", "process", "readout",
                   "sampling", "damping_mode", "seed", "conditional", "N", "Q", "P",
                   "mc_loss", "mc_measurement", "hflip", "train_mode",
                   "fit_epochs", "eig_epochs")
#: repeats adds the sigma-hat measurement's own axes.
_REPEAT_BLOCK_IDENTITY = _BLOCK_IDENTITY + ("R", "M", "repeat_seed_stride")


def _query_blocks(Q: int, qchunk: int) -> list[tuple[int, int]]:
    """The ``[q0, q1)`` tiling of the query axis at this residency setting."""
    return [(q0, min(q0 + qchunk, Q)) for q0 in range(0, Q, qchunk)]


def _row_tiles(rows: int, rchunk: int) -> list[tuple[int, int]]:
    """The ``[r0, r1)`` tiling of the train-row axis (``ekfac.row_chunk``).

    Contiguous, not modular: the range IS the address, so assembly is a
    coverage check on ``[0, rows)`` and a slice assignment, and a shard's
    partial says on its face which rows it holds. (Modular classes would tile
    the axis just as well but could not be named by a range, so neither the
    filename nor the coverage check could be written down.)
    """
    return [(r0, min(r0 + rchunk, rows)) for r0 in range(0, rows, rchunk)]


def _row_chunk(e: dict, rows: int) -> int:
    """Train rows per tile. 0/unset = every row in one tile (the default, and
    the only shape that existed before EKFAC-ROWSHARD)."""
    return int(e.get("row_chunk") or 0) or rows


def _check_shard(rank: int, world: int, *, what: str = "--rank/--world") -> None:
    """Refuse a bad shard spec up front, before a minute of model/factor loading."""
    if world < 1 or not (0 <= rank < world):
        raise ValueError(f"{what} must satisfy 0 <= rank < world; got rank={rank} world={world}")


def _bkey(q0: int, q1: int, r0: int, r1: int, rows: int) -> dict:
    """Store key of one block partial.

    The whole-rows tile keeps the ``(q0, q1)``-only key EKFAC-SHARD used, so its
    path is byte-identical and every partial written before the row axis existed
    is still addressed by exactly the key that finds it.
    """
    k = {"q0": int(q0), "q1": int(q1)}
    if (int(r0), int(r1)) != (0, int(rows)):
        k.update(r0=int(r0), r1=int(r1))
    return k


def _rng(r0: int, r1: int) -> str:
    return f"{r0}:{r1}"


def _shard_of(blocks, tiles, *, rank: int, world: int, row_rank: int, row_world: int):
    """``[(bi, (q0, q1), ti, (r0, r1))]`` this shard owns.

    Two independent modular splits (the `subsets train/losses` idiom on both
    axes): query blocks by ``rank % world``, train-row tiles by
    ``row_rank % row_world``. Block-major order, so a process that owns several
    row tiles of the same block computes that block's query pass once.
    """
    _check_shard(rank, world)
    _check_shard(row_rank, row_world, what="--row-rank/--row-world")
    return [(bi, b, ti, t)
            for bi, b in enumerate(blocks) if bi % world == rank
            for ti, t in enumerate(tiles) if ti % row_world == row_rank]


def _shard_suffix(rank: int, world: int, row_rank: int, row_world: int) -> str:
    """Progress-file suffix identifying this shard.

    Both axes appear, so two processes on one node that differ only in their row
    tile do not write the same progress CSV (an unsharded run's filename is
    unchanged).
    """
    return ((f"_rank{rank}of{world}" if world > 1 else "")
            + (f"_row{row_rank}of{row_world}" if row_world > 1 else ""))


def _warn_shard_shape(blocks, tiles, qchunk: int, rchunk: int, Q: int, rows: int,
                      *, world: int, row_world: int) -> None:
    """Refuse to let a shard silently have nothing to do.

    ``world``/``row_world`` select from a tiling that the CONFIG fixes; asking
    for more shards than there are tiles leaves the high ranks idle while the
    operator believes the cell is split N ways.
    """
    if world > 1 and len(blocks) == 1:
        log.warning("--world %d but this cell has ONE query block (ekfac.query_chunk=%d "
                    ">= Q=%d): only rank 0 has work. --rank/--world splits QUERY BLOCKS, so "
                    "set ekfac.query_chunk to at most Q/world — or split the train rows "
                    "instead with --row-rank/--row-world + ekfac.row_chunk, which does not "
                    "re-stream the train side.", world, qchunk, Q)
    if row_world > 1 and len(tiles) == 1:
        log.warning("--row-world %d but this cell has ONE train-row tile "
                    "(ekfac.row_chunk=%d >= N=%d): only row-rank 0 has work. Set "
                    "ekfac.row_chunk to at most N/row_world.", row_world, rchunk, rows)


def _block_meta(ctx, *, raw_name: str, query_type: str, code_version: str) -> dict:
    """Identity + provenance stamped into every block partial (npz: flat arrays)."""
    fmeta = getattr(ctx.factors, "meta", {}) or {}
    return {
        "dampings": np.asarray(ctx.dampings, dtype=np.float64),
        "method": np.array(raw_name), "dataset": np.array(ctx.base.dataset),
        "query_type": np.array(query_type), "process": np.array(ctx.base.process),
        "readout": np.array(ctx.readout), "sampling": np.array(ctx.sampling),
        "damping_mode": np.array(ctx.damping_mode),
        "seed": np.int64(ctx.seed), "conditional": np.int64(bool(ctx.conditional)),
        "N": np.int64(ctx.N), "Q": np.int64(ctx.Q), "P": np.int64(ctx.P),
        "mc_loss": np.int64(ctx.mc_loss), "mc_measurement": np.int64(ctx.mc_meas),
        "hflip": np.int64(bool(ctx.hflip)), "train_mode": np.int64(bool(ctx.train_mode)),
        "fit_epochs": np.int64(fmeta.get("fit_epochs", -1)),
        "eig_epochs": np.int64(fmeta.get("eig_epochs", -1)),
        # Informational only: residency knobs and the code that wrote the block
        # do not move the numbers (README 1.5), so they warn instead of raising.
        # They ARE recorded per partial in the assembled artifact's meta, because
        # README pit 16 makes grad_chunk a caliber field for iid sampling (the
        # iid branch draws inside the chunk loop) while leaving it free under
        # stratified_antithetic — a distinction only a reader of the provenance
        # can make after the fact. Host + torch version travel for the same
        # reason: "same numbers" across machines is a claim about a stack.
        "grad_chunk": np.int64(ctx.chunk), "code_version": np.array(str(code_version)),
        "hostname": np.array(socket.gethostname()),
        "torch_version": np.array(str(torch.__version__)),
    }


def _partial_provenance(z: dict, q0: int, q1: int, r0: int, r1: int) -> dict:
    """The per-partial provenance row recorded in the assembled artifact's meta."""
    return {"q": _rng(q0, q1), "r": _rng(r0, r1),
            "host": _as_py(z.get("hostname", "?")),
            "torch": _as_py(z.get("torch_version", "?")),
            "grad_chunk": int(_as_py(z.get("grad_chunk", -1))),
            "code_version": _as_py(z.get("code_version", "?"))}


def _as_py(v):
    """A block-meta value as a plain Python object (npz gives 0-d/1-d arrays)."""
    a = np.asarray(v)
    if a.dtype.kind in "US":
        return str(a) if a.ndim == 0 else [str(x) for x in a.ravel()]
    return a.item() if a.ndim == 0 else [float(x) for x in a.ravel()]


def _check_block(z: dict, want: dict, keys, *, what: str, q0: int, q1: int,
                 r0: int, r1: int, rows: int) -> None:
    """Refuse a block partial that was not produced under THIS run's caliber."""
    bad = []
    if int(np.asarray(z.get("q0", -1))) != q0 or int(np.asarray(z.get("q1", -1))) != q1:
        bad.append(f"query range: partial=[{_as_py(z.get('q0', -1))}:"
                   f"{_as_py(z.get('q1', -1))}] run=[{q0}:{q1}]")
    # A partial written before the row axis existed (EKFAC-SHARD) stamps no
    # r0/r1 and holds every row — read it as [0, rows), which is what it is.
    got_r0 = int(np.asarray(z["r0"])) if "r0" in z else 0
    got_r1 = int(np.asarray(z["r1"])) if "r1" in z else rows
    if (got_r0, got_r1) != (r0, r1):
        bad.append(f"train-row range: partial=[{got_r0}:{got_r1}] run=[{r0}:{r1}]")
    for k in keys:
        if k not in z:
            bad.append(f"{k}: absent from the partial (written before it was stamped)")
        elif _as_py(z[k]) != _as_py(want[k]):
            bad.append(f"{k}: partial={_as_py(z[k])!r} run={_as_py(want[k])!r}")
    if bad:
        raise ValueError(
            f"{what}: the stored partial for queries [{q0}:{q1}] rows [{r0}:{r1}] "
            f"disagrees with this run:\n  "
            + "\n  ".join(bad)
            + f"\nResuming across calibers would file a matrix whose columns disagree. Delete "
              f"this cell's blocks/ directory, or re-run under the caliber the partials were "
              f"measured with — never mix them.")
    got_cv, want_cv = _as_py(z.get("code_version", "?")), _as_py(want["code_version"])
    if got_cv != want_cv:
        log.warning("%s: block [%d:%d] rows [%d:%d] was written by code_version=%s, this run "
                    "is %s (the caliber matches, so resuming; recorded for provenance)",
                    what, q0, q1, r0, r1, got_cv, want_cv)


def _run_shard(store, kind, spec, blocks, tiles, rows, meta, ident_keys, *,
               rank: int, world: int, row_rank: int, row_world: int,
               ignore_partials: bool, compute, what: str) -> int:
    """Stream this shard's (query block, row tile) units, persisting each as it lands.

    Returns how many units this pass actually computed. An existing partial is
    identity-checked and skipped (that is the resume); so is a whole-rows
    partial of the same query block, because it already contains this tile's
    rows — that is what lets one block stay resident while another is
    row-sharded. ``ignore_partials`` (``--rescore`` / ``--force``) recomputes
    and overwrites this shard's units.
    """
    computed = 0
    for bi, (q0, q1), ti, (r0, r1) in _shard_of(blocks, tiles, rank=rank, world=world,
                                                row_rank=row_rank, row_world=row_world):
        key = _bkey(q0, q1, r0, r1, rows)
        if not ignore_partials:
            whole = {"q0": q0, "q1": q1}
            if "r0" in key and store.exists(kind, spec, **whole):
                _check_block(store.load(kind, spec, **whole), meta, ident_keys,
                             what=what, q0=q0, q1=q1, r0=0, r1=rows, rows=rows)
                log.info("%s: block %d [%d:%d] is on disk with ALL rows — rows [%d:%d] "
                         "need no separate partial", what, bi, q0, q1, r0, r1)
                continue
            if store.exists(kind, spec, **key):
                _check_block(store.load(kind, spec, **key), meta, ident_keys,
                             what=what, q0=q0, q1=q1, r0=r0, r1=r1, rows=rows)
                log.info("%s: block %d [%d:%d] rows [%d:%d] already on disk — resuming past it",
                         what, bi, q0, q1, r0, r1)
                continue
        arr = compute(q0, q1, r0, r1)
        if not np.isfinite(arr).all():
            raise ValueError(f"{what}: block [{q0}:{q1}] rows [{r0}:{r1}] has "
                             f"{int((~np.isfinite(arr)).sum())} non-finite entries; not saving")
        store.save(kind, spec,
                   {**meta, "q0": np.int64(q0), "q1": np.int64(q1),
                    "r0": np.int64(r0), "r1": np.int64(r1), "block": arr}, **key)
        computed += 1
        log.info("%s: block %d [%d:%d] rows [%d:%d] saved %s",
                 what, bi, q0, q1, r0, r1, list(arr.shape))
    return computed


def _row_sources(store, kind, spec, q0: int, q1: int, tiles, rows: int, local, *, what: str):
    """``(sources, missing)`` for one query block's train-row axis.

    ``sources`` is ``[(r0, r1, key)]`` covering ``[0, rows)`` exactly, in
    ascending row order — or ``None`` while the coverage still has holes, with
    ``missing`` naming them. A whole-rows partial short-circuits everything: it
    already holds every row, and by the per-row keying it is bit-identical to
    the concatenation of any row tiling of the same block.

    Candidates come from BOTH the expected tiling (``store.exists``, which may
    live on a remote tier) and whatever row partials are on this disk (a peer
    may have sharded the rows differently). Overlapping ranges are a hard error
    rather than a silent last-writer-wins: two partials claiming row i are two
    answers for it, and the assembler must not pick one.
    """
    whole = {"q0": q0, "q1": q1}
    if store.exists(kind, spec, **whole):
        return [(0, rows, whole)], []
    found: dict[tuple[int, int], dict] = {}
    for r0, r1 in tiles:
        key = _bkey(q0, q1, r0, r1, rows)
        if "r0" in key and store.exists(kind, spec, **key):
            found[(r0, r1)] = key
    for k in local:                      # row partials of a foreign tiling
        if k["q0"] == q0 and k["q1"] == q1 and "r0" in k:
            found.setdefault((k["r0"], k["r1"]), k)
    cur, missing, bad = 0, [], []
    for r0, r1 in sorted(found):
        if r1 > rows or r0 >= r1:
            bad.append(f"[{r0}:{r1}] is not a sub-range of [0:{rows}]")
        elif r0 < cur:
            bad.append(f"[{r0}:{r1}] overlaps the rows already covered by [0:{cur}]")
        elif r0 > cur:
            missing.append(_rng(cur, r0))
        cur = max(cur, r1)
    if bad:
        raise ValueError(
            f"{what}: the train-row partials of query block [{q0}:{q1}] do not tile "
            f"[0:{rows}]:\n  " + "\n  ".join(bad)
            + f"\nPresent ranges: {[_rng(a, b) for a, b in sorted(found)]}. Overlapping "
              f"partials come from mixing two ekfac.row_chunk settings on one cell; delete "
              f"the stale ones from this cell's blocks/ directory and re-run.")
    if cur < rows:
        missing.append(_rng(cur, rows))
    if missing:
        return None, missing
    return [(r0, r1, found[(r0, r1)]) for r0, r1 in sorted(found)], []


def _assemble_blocks(store, kind, spec, blocks, tiles, rows, meta, ident_keys, *,
                     prefix: tuple, Q: int, rank: int, world: int,
                     row_rank: int, row_world: int, what: str):
    """Stitch the block partials into the full ``prefix + (rows, Q)`` float32 array.

    Two levels, one deterministic order: for each query block in ascending q0,
    its row partials in ascending r0. ``(None, info)`` while any row of any
    block is missing — a peer shard still owns it, or this process was
    interrupted; a gap is the NORMAL mid-flight state of a sharded cell, so it
    files nothing and reports, exactly as a missing block does. Whichever run
    finds full coverage performs the assembly (``_finalize_gt_losses``'s
    last-finisher rule); if that run dies first, a cheap re-run finalizes
    because every present partial skips.
    """
    local = store.local_blocks(kind, spec)
    plan, missing = {}, []
    for q0, q1 in blocks:
        src, miss = _row_sources(store, kind, spec, q0, q1, tiles, rows, local, what=what)
        if src is None:
            # a block with nothing at all reads as one missing block (the
            # EKFAC-SHARD census); a partially covered one names its row gaps
            missing.extend([_rng(q0, q1)] if miss == [_rng(0, rows)]
                           else [f"{_rng(q0, q1)} rows {m}" for m in miss])
        else:
            plan[(q0, q1)] = src
    info = {"blocks_done": len(plan), "blocks_total": len(blocks),
            "rank": rank, "world": world, "row_rank": row_rank, "row_world": row_world,
            "missing_blocks": missing}
    if missing:
        log.info("%s: %d/%d query blocks fully covered (missing %s) — filing nothing yet",
                 what, len(plan), len(blocks), missing)
        return None, {"finalized": False, **info}
    full = np.zeros(prefix + (rows, Q), dtype=np.float32)
    prov = []
    for q0, q1 in blocks:
        for r0, r1, key in plan[(q0, q1)]:
            try:
                z = store.load(kind, spec, **key)
            except Exception as exc:        # a peer shard may be mid-write
                log.info("%s: assembly deferred (block [%d:%d] rows [%d:%d] unreadable: %s)",
                         what, q0, q1, r0, r1, exc)
                return None, {"finalized": False, "deferred": True, **info}
            _check_block(z, meta, ident_keys, what=what, q0=q0, q1=q1,
                         r0=r0, r1=r1, rows=rows)
            blk = z["block"]
            if tuple(blk.shape) != prefix + (r1 - r0, q1 - q0):
                raise ValueError(f"{what}: block [{q0}:{q1}] rows [{r0}:{r1}] has shape "
                                 f"{list(blk.shape)}, expected "
                                 f"{list(prefix + (r1 - r0, q1 - q0))}")
            full[..., r0:r1, q0:q1] = blk
            prov.append(_partial_provenance(z, q0, q1, r0, r1))
    if not np.isfinite(full).all():
        raise ValueError(f"{what}: the assembled matrix has "
                         f"{int((~np.isfinite(full)).sum())} non-finite entries; not saving")
    hosts = sorted({p["host"] for p in prov})
    if len(hosts) > 1 or len({p["grad_chunk"] for p in prov}) > 1:
        log.info("%s: assembled from %d partials across hosts=%s grad_chunk=%s "
                 "(recorded per partial in the artifact's meta; see README pit 16 — "
                 "a mixed grad_chunk is only estimator-neutral under "
                 "stratified_antithetic)", what, len(prov), hosts,
                 sorted({p["grad_chunk"] for p in prov}))
    return full, {"finalized": True, "partials": prov, **info}


def _drop_blocks(store, kind, spec) -> int:
    """Delete this machine's block partials once the whole matrix is safely filed.

    Enumerated from the disk rather than from the current tiling, so partials
    left by another ``query_chunk``/``row_chunk`` setting go too — the whole
    point of ``--clean-partials`` is that the cell's blocks/ directory is empty
    afterwards.
    """
    return sum(int(store.discard(kind, spec, **k)) for k in store.local_blocks(kind, spec))


class EkfacScoreUseCase:
    """Score with the fitted factors.

    ``method="ekfac_if"``: stream the train gradients once, evaluate the whole
    damping grid, cache EVERY damping's raw matrix (``SCORES_LAMBDA``), select
    the damping on the raw matrices, save ``scores.npy``.

    ``method="ekfac_if_tweedie"``: the same raw matrices (from the cache when
    present, else streamed) behind the shrinkage layer with σ̂ from the
    repeat-scored subset — recomputed PER damping — and the damping selected on
    the SHAPED matrices (task brief B3, P0). γ is the per-method calibrated
    knee; an uncalibrated cell refuses.

    The stream itself is block-resumable and shardable on two axes (EKFAC-SHARD
    + EKFAC-ROWSHARD, 2026-09-09): each ``(query block, train-row tile)``
    partial is filed under ``SCORES_BLOCK`` as it completes, a restart skips the
    partials already on disk, ``rank``/``world`` split the query blocks and
    ``row_rank``/``row_world`` split each block's train rows. Only the run that
    finds every block's rows fully covered assembles and files
    ``scores_lambda``/``scores`` — both counts are residency axes, so the
    assembled matrix is bit-identical however it was split.
    """

    def __init__(self, store, cfg, *, device: str = "cuda:0") -> None:
        self.store, self.cfg, self.device = store, cfg, device

    def run(self, dataset: str, seed: int, *, query_type: str = "gen",
            process: str = "cfm", conditional: bool = True,
            rescore: bool = False, method: str = METHOD_NAME,
            rank: int = 0, world: int = 1, row_rank: int = 0, row_world: int = 1,
            clean_partials: bool = False) -> dict:
        _check_shard(rank, world)
        _check_shard(row_rank, row_world, what="--row-rank/--row-world")
        method = resolve_paper_method(self.cfg, method)
        raw_name, solver, layer = split_method(method)
        readout, is_tweedie = solver["readout"], layer == "shrink"
        if query_type == "inject" and is_tweedie:
            raise ValueError("repeat/shrinkage methods have no calibrated gamma on the inject track")
        spec = RunSpec(dataset=dataset, seed=seed, method=method,
                       query_type=query_type, process=process, conditional=conditional)
        set_context(run_id=spec.digest()[:6], trace_id=f"ekfac-score:{dataset}:{query_type}")
        raw_spec = spec.with_(method=raw_name)
        e = _bind_solver(_ekfac_cfg(self.cfg), solver, method)
        dampings = _grid(e)
        cache_ok = all(self.store.exists(K.SCORES_LAMBDA, raw_spec, lam=d) for d in dampings)
        keep_filed = False
        if self.store.exists(K.SCORES, spec) and not rescore:
            if is_tweedie or cache_ok:
                log.info("%s scores exist, skipping (use --rescore)", method)
                return {"skipped": True}
            # ekfac_if scores filed before the per-damping cache existed (the
            # 08-21 production runs): build the cache with a fresh stream but do
            # NOT overwrite the filed scores.npy — a re-stream is a new MC draw
            # and would silently move an already-reported main-table number.
            keep_filed = True
            log.warning("%s scores exist but the per-damping cache does not: streaming "
                        "to build the cache; the filed scores.npy is left untouched "
                        "(use --rescore to re-file from the same stream)", method)
        injecting = query_type == "inject"
        if injecting:
            from balds.evaluation.injection import select_lambda_inject
            selector_name = ((self.cfg["score"].get("lambda_selector_by_track") or {})
                             .get("inject"))
            if selector_name != "inject_val":
                raise ValueError("inject scoring requires score.lambda_selector_by_track.inject=inject_val")
            queries = self.store.load(K.INJECT_QUERIES, spec)
            inject_meta = self.store.load(K.INJECT_META, RunSpec(dataset=dataset))
            train_ds, _ = _load_ds(self.cfg, dataset)
            train_labels = train_ds.labels
            default_k = inject_meta.get("per_class", inject_meta.get("foreign_per_style"))
            k_inject = int(self.cfg["inject"].get("k", default_k))
            gt = masks = subset_idx = selector = None
        else:
            masks, gt = lds_rows(self.cfg, dataset, self.store.load(K.SUBSET_MASKS, spec),
                                 _gt_for_queries(self.cfg, dataset, query_type,
                                                 self.store.load(K.GT_MATRIX, spec)))
            mts = e.get("max_train_samples")
            if mts:
                # debug cap: keep the selector shape-consistent with the capped stream
                masks = [m[: int(mts)] for m in masks]
            subset_idx = list(range(min(gt.shape[0], len(masks))))
            selector = SELECTORS.get(self.cfg["score"].get("lambda_selector", "oracle"))

        # For the shrinkage method, resolve γ and the repeats BEFORE any expensive
        # streaming so a mis-configured run fails in seconds, not hours.
        if is_tweedie:
            gamma = resolve_gamma(self.cfg, dataset, method)          # strict
            reps = load_repeat_scores(self.store, raw_spec)
            _check_repeat_caliber(reps, e, raw_spec)
            rep_d = [float(d) for d in reps["dampings"]]
            if rep_d != dampings:
                raise ValueError(
                    f"repeat_scores were measured on dampings {rep_d} but the config "
                    f"grid is {dampings}; rerun `balds ekfac repeats --force` on the "
                    f"current grid so σ̂ and the raw matrices share every damping")

        cached = cache_ok and not (rescore and not is_tweedie)
        # block census of this run, reported alongside the result so an operator
        # (and a shard script) can see what is on disk without reading the log
        binfo: dict = {"finalized": True, "cached": True}
        if cached:
            raw_by_d = load_raw_by_damping(self.store, raw_spec, dampings)
            log.info("raw %s matrices loaded from the per-damping cache", raw_name)
        else:
            restream = rescore and not is_tweedie
            if restream and (world > 1 or row_world > 1):
                raise ValueError(
                    "--rescore with --world > 1 or --row-world > 1 is refused: the first shard "
                    "to finish would assemble a mix of its own fresh partials and the peers' "
                    "stale ones. Delete this cell's blocks/ directory "
                    "(scores/<method>/<dataset>[_val]/seed_<n>/blocks/) on every shard, then "
                    "launch the shards WITHOUT --rescore.")
            raw_by_d, binfo = self._stream(
                dataset, seed, query_type=query_type, process=process,
                conditional=conditional, readout=readout, raw_name=raw_name, e=e,
                rank=rank, world=world, row_rank=row_rank, row_world=row_world,
                ignore_partials=restream)
            if raw_by_d is None:
                # this shard did its share; the run that finds every block files the
                # whole-matrix artifacts (nothing is written to a final address here)
                log.info("%s: %d/%d query blocks fully covered — scores_lambda/scores not "
                         "filed yet", method, binfo["blocks_done"], binfo["blocks_total"])
                return {"method": method, **binfo}
            for d, mat in raw_by_d.items():
                self.store.save(K.SCORES_LAMBDA, raw_spec, mat, lam=d)
            if clean_partials:
                n = _drop_blocks(self.store, K.SCORES_BLOCK, raw_spec)
                log.info("dropped %d block partials (--clean-partials); the per-damping "
                         "cache is the durable artifact", n)
            if is_tweedie and not self.store.exists(K.SCORES, raw_spec):
                # the raw selection is free once streamed; file it so the raw
                # baseline exists without a second stream
                ch = selector.select(raw_by_d, dampings, gt, masks, subset_idx)
                self.store.save(K.SCORES, raw_spec, raw_by_d[ch.best_lam])
                log.info("raw %s also saved (damping=%.3g lds=%.4f)", raw_name,
                         ch.best_lam, ch.best_lds)

        if not is_tweedie:
            if injecting:
                choice = select_lambda_inject(
                    raw_by_d, dampings, queries["split"], queries["host_labels"],
                    train_labels, inject_meta, k=k_inject)
            else:
                choice = selector.select(raw_by_d, dampings, gt, masks, subset_idx)
            best = raw_by_d[choice.best_lam]
            extra: dict = {}
            if keep_filed and not injecting:
                return {"method": method, "cache_built": True, "scores_kept": True,
                        "best_damping_of_stream": choice.best_lam,
                        "best_lds_of_stream": choice.best_lds, **binfo}
            if not injecting and self.store.exists(K.REPEAT_SCORES, raw_spec):  # opportunistic C2
                reps = self.store.load(K.REPEAT_SCORES, raw_spec)
                i = [float(d) for d in reps["dampings"]].index(choice.best_lam)
                sig = sigma_from_repeats([reps["scores"][i, r] for r in range(reps["scores"].shape[1])])
                extra = snr_stats(best, sig)
        else:
            sig_by_d = {d: sigma_from_repeats([reps["scores"][i, r]
                                               for r in range(reps["scores"].shape[1])])
                        for i, d in enumerate(dampings)}
            shaped, choice = select_after_shrinkage(
                ShrinkageLayer(gamma), raw_by_d, sig_by_d, dampings,
                lambda sh, keys: selector.select(sh, keys, gt, masks, subset_idx))
            best = shaped[choice.best_lam]
            extra = {"gamma": gamma,
                     **snr_stats(raw_by_d[choice.best_lam], sig_by_d[choice.best_lam])}
        self.store.save(K.SCORES, spec, best)
        if injecting:
            oracle_test_lam = max(
                dampings, key=lambda lam: choice.per_lambda_ap_test[float(lam)])
            self.store.save(K.SCORES_META, spec, {
                "method": method, "raw_name": raw_name, "readout": readout,
                "layer": layer, "Q": int(best.shape[1]), "best_lam": float(choice.best_lam),
                "selector": "inject_val",
                "sampling": e.get("sampling"), "damping_mode": e.get("damping_mode"),
                "lambda_sweep": [float(d) for d in dampings],
                "per_lambda_ap_val": {f"{d:g}": float(v)
                                      for d, v in choice.per_lambda_ap_val.items()},
                "per_lambda_ap_test": {f"{d:g}": float(v)
                                       for d, v in choice.per_lambda_ap_test.items()},
                "oracle_test_lam": float(oracle_test_lam),
                "oracle_test_ap": float(choice.per_lambda_ap_test[float(oracle_test_lam)]),
                "queries_sha256": queries["queries_sha256"], "k": k_inject,
                "partials": binfo.get("partials"),
            })
            log.info("%s damping=%.3g val AP=%.4f (selector=inject_val)",
                     method, choice.best_lam, choice.best_ap_val)
            return {"method": method, "best_damping": choice.best_lam,
                    "best_ap_val": choice.best_ap_val,
                    "per_damping_val": {f"{d:g}": float(v)
                                        for d, v in choice.per_lambda_ap_val.items()},
                    "shape": list(best.shape), "selector": "inject_val", **binfo}
        # provenance sidecar, same shape as the projected path's: the per-damping
        # curve is what the boundary-argmax rule is checked against, and it was
        # previously only ever returned to the caller (i.e. lived in a log line).
        self.store.save(K.SCORES_META, spec, {
            "method": method, "raw_name": raw_name, "readout": readout, "layer": layer,
            "Q": int(best.shape[1]),
            "best_lam": float(choice.best_lam), "best_lds": float(choice.best_lds),
            "gamma": (float(gamma) if is_tweedie else None),
            "sampling": e.get("sampling"), "damping_mode": e.get("damping_mode"),
            "lambda_sweep": [float(d) for d in dampings],
            "selector": selector.__class__.__name__,
            "per_lambda_lds": {f"{d:g}": float(v)
                               for d, v in choice.per_lambda_lds.items()},
            # who computed which (query range, row range) of the stream this
            # matrix came from: host, torch version and grad_chunk per partial
            # (README pit 16). ``None`` when the matrices came from the cache.
            "partials": binfo.get("partials")})
        log.info("%s damping=%.3g lds=%.4f (selector=%s)", method,
                 choice.best_lam, choice.best_lds, selector.__class__.__name__)
        return {"method": method, "best_damping": choice.best_lam,
                "best_lds": choice.best_lds,
                "per_damping": {f"{d:g}": float(v)
                                for d, v in choice.per_lambda_lds.items()},
                "shape": list(best.shape),
                "selector": selector.__class__.__name__, **binfo, **extra}

    def _stream(self, dataset, seed, *, query_type, process, conditional,
                readout: str = "loss", raw_name: str = METHOD_NAME, e: dict | None = None,
                rank: int = 0, world: int = 1, row_rank: int = 0, row_world: int = 1,
                ignore_partials: bool = False):
        """Stream this shard's (query block, row tile) units -> ``({d: (N, Q)}, info)``.

        The scores are ``None`` while any block's rows are not fully covered (a
        peer shard owns them, or this process was interrupted before finishing
        its own): the caller then files nothing and reports the census.
        """
        ctx = _prepare(self.store, self.cfg, self.device, dataset, seed,
                       query_type=query_type, process=process, conditional=conditional,
                       readout=readout, e=e)
        block_spec = RunSpec(dataset=dataset, seed=seed, method=raw_name,
                             query_type=query_type, process=process, conditional=conditional)
        prog = ProgressCSV(progress_path(
            self.cfg["storage"]["data_root"],
            f"ekfac_score_{dataset}_{process}_seed{seed}_{query_type}"
            + _method_suffix(raw_name)
            + _shard_suffix(rank, world, row_rank, row_world)))
        qchunk = _query_chunk(ctx)
        blocks = _query_blocks(ctx.Q, qchunk)
        rchunk = _row_chunk(ctx.e, ctx.N)
        tiles = _row_tiles(ctx.N, rchunk)
        qc_gib = ctx.Q * ctx.P * ctx.qdtype.itemsize / 2 ** 30
        log.info("ekfac score: method=%s readout=%s sampling=%s damping_mode=%s N=%d Q=%d P=%d "
                 "mc=%d/%d grid=%d dtype=%s qcoords=%.1fGiB query_chunk=%d blocks=%d "
                 "row_chunk=%d row_tiles=%d",
                 raw_name, readout, ctx.sampling, ctx.damping_mode, ctx.N, ctx.Q, ctx.P,
                 ctx.mc_loss, ctx.mc_meas, len(ctx.dampings), ctx.qdtype, qc_gib, qchunk,
                 len(blocks), rchunk, len(tiles))
        if qchunk < ctx.Q:
            # residency knob, not an estimator knob: every query block re-streams
            # the train side with the SAME per-sample seeds, so the assembled
            # matrix is identical to the resident run — at ceil(Q/chunk)x the
            # train-gradient cost. The dispatcher trades chunk size vs. VRAM.
            log.warning("query_chunk=%d < Q=%d: %d train streams (identical scores, "
                        "%dx train-gradient cost; VRAM per block %.1fGiB)", qchunk, ctx.Q,
                        len(blocks), len(blocks), qc_gib * qchunk / ctx.Q)
        _warn_shard_shape(blocks, tiles, qchunk, rchunk, ctx.Q, ctx.N,
                          world=world, row_world=row_world)
        meta = _block_meta(ctx, raw_name=raw_name, query_type=query_type,
                           code_version=getattr(self.store, "code_version", "?"))
        inv_box: list = []          # built on first need: an assemble-only pass needs none
        qbox: list = []             # the resident query block, reused across its row tiles

        def _inverses():
            if not inv_box:
                iv = {d: ctx.factors.damped_inverse_flat(d, self.device, mode=ctx.damping_mode)
                      for d in ctx.dampings}
                _check_finite_inverses(iv, ctx.damping_mode)
                inv_box.append(iv)
            return inv_box[0]

        def _qcoords(q0, q1):
            """Eigen-coords of block [q0, q1), computed once per block.

            ``_shard_of`` is block-major, so a process holding several row tiles
            of one block pays its query pass once — the only work the row axis
            would otherwise duplicate. The previous block is dropped first, so
            residency is one block either way.
            """
            if not qbox or qbox[0][0] != (q0, q1):
                qbox.clear()
                qbox.append(((q0, q1), _query_pass(ctx, prog, q0, q1)))
            return qbox[0][1]

        def _compute(q0, q1, r0, r1):
            inv = _inverses()
            qcoords = _qcoords(q0, q1)
            phase = "train_grads" if len(blocks) == 1 and len(tiles) == 1 \
                else f"train_grads[{q0}:{q1}]r[{r0}:{r1}]"
            blk = np.zeros((len(ctx.dampings), r1 - r0, q1 - q0), dtype=np.float32)
            for i in range(r0, r1):
                lb = int(ctx.train_ds.labels[i]) if conditional else None
                row = _train_row(ctx, qcoords, inv, ctx.train_ds.images[i], lb,
                                 seed=seed, index=i, flipped=_train_flip(ctx, i))
                if i - r0 < 2:      # guard the first rows THIS unit streams
                    _check_finite_row(row, what="ekfac score", index=i)
                for di, d in enumerate(ctx.dampings):
                    blk[di, i - r0] = row[d]
                if (i - r0 + 1) % 25 == 0 or i + 1 == r1:
                    prog.log(done=i - r0 + 1, total=r1 - r0, phase=phase)
            return blk

        _run_shard(self.store, K.SCORES_BLOCK, block_spec, blocks, tiles, ctx.N,
                   meta, _BLOCK_IDENTITY, rank=rank, world=world,
                   row_rank=row_rank, row_world=row_world,
                   ignore_partials=ignore_partials, compute=_compute, what="ekfac score")
        qbox.clear()
        full, info = _assemble_blocks(
            self.store, K.SCORES_BLOCK, block_spec, blocks, tiles, ctx.N, meta,
            _BLOCK_IDENTITY, prefix=(len(ctx.dampings),), Q=ctx.Q, rank=rank, world=world,
            row_rank=row_rank, row_world=row_world, what="ekfac score")
        if full is None:
            return None, info
        return {d: full[di] for di, d in enumerate(ctx.dampings)}, info


class EkfacRepeatsUseCase:
    """σ̂ measurement for the curvature path (task brief B2): the first M
    training samples re-scored R times with fresh train-side MC draws, against
    the production query eigen-coordinates and the same damped inverses.

    Mirrors ``featurize --split repeat`` knob for knob so EK-FAC's and FMAS's σ̂
    rest on the same samples (``train_ds.images[:M]``) and the same repeat-seed
    rule (``seed + 1_000_000·(r+1)``); only the train side re-draws. Output
    ``(D, R, M, Q)`` — one σ̂ per damping downstream, never one global scalar.

    Block-resumable and shardable on the same two axes as ``score``
    (EKFAC-SHARD + EKFAC-ROWSHARD, 2026-09-09): each ``(D, R, r1-r0, q1-q0)``
    partial is filed as it completes and ``REPEAT_SCORES`` only once every
    block's rows are fully covered. The row axis here is the σ̂ subset
    ``[0, M)`` — those ARE the first M train rows, keyed identically — so
    ``ekfac.row_chunk`` tiles M rather than N.
    """

    def __init__(self, store, cfg, *, device: str = "cuda:0") -> None:
        self.store, self.cfg, self.device = store, cfg, device

    def run(self, dataset: str, seed: int, *, query_type: str = "gen",
            process: str = "cfm", conditional: bool = True, force: bool = False,
            method: str = METHOD_NAME, rank: int = 0, world: int = 1,
            row_rank: int = 0, row_world: int = 1,
            clean_partials: bool = False) -> dict:
        _check_shard(rank, world)
        _check_shard(row_rank, row_world, what="--row-rank/--row-world")
        method = resolve_paper_method(self.cfg, method)
        raw_name, solver, layer = split_method(method)
        readout = solver["readout"]
        if layer is not None:
            raise ValueError(f"`balds ekfac repeats` takes the RAW method ({sorted(CURVATURE_RAW)} "
                             f"or a variant); {method!r} reads those repeats, it does not produce them")
        raw_spec = RunSpec(dataset=dataset, seed=seed, method=raw_name,
                           query_type=query_type, process=process, conditional=conditional)
        set_context(run_id=raw_spec.digest()[:6], trace_id=f"ekfac-repeats:{dataset}:{query_type}")
        e = _bind_solver(_ekfac_cfg(self.cfg), solver, method)
        if self.store.exists(K.REPEAT_SCORES, raw_spec) and not force:
            log.info("repeat scores exist, skipping (use --force)")
            return {"skipped": True}
        if force and (world > 1 or row_world > 1):
            raise ValueError(
                "--force with --world > 1 or --row-world > 1 is refused: the first shard to "
                "finish would assemble a mix of its own fresh partials and the peers' stale "
                "ones. Delete this cell's blocks/ directory "
                "(scores/<method>/<dataset>[_val]/seed_<n>/blocks/) on every shard, then "
                "launch the shards WITHOUT --force.")
        tw = self.cfg.get("tweedie", {}) or {}
        R = int(tw.get("repeats", 4))
        M = int(tw.get("sigma_samples", 256))
        ctx = _prepare(self.store, self.cfg, self.device, dataset, seed,
                       query_type=query_type, process=process, conditional=conditional,
                       readout=readout, e=e)
        M = min(M, ctx.N)
        prog = ProgressCSV(progress_path(
            self.cfg["storage"]["data_root"],
            f"ekfac_repeats_{dataset}_{process}_seed{seed}_{query_type}"
            + _method_suffix(raw_name)
            + _shard_suffix(rank, world, row_rank, row_world)))
        qchunk = _query_chunk(ctx)
        blocks = _query_blocks(ctx.Q, qchunk)
        # the σ̂ rows ARE the first M train rows (same keying), so the row axis
        # tiles [0, M) here, not [0, N)
        rchunk = _row_chunk(ctx.e, M)
        tiles = _row_tiles(M, rchunk)
        log.info("ekfac repeats: method=%s readout=%s sampling=%s damping_mode=%s R=%d M=%d "
                 "Q=%d dampings=%d mc=%d query_chunk=%d blocks=%d row_chunk=%d row_tiles=%d",
                 raw_name, readout, ctx.sampling, ctx.damping_mode, R, M, ctx.Q,
                 len(ctx.dampings), ctx.mc_loss, qchunk, len(blocks), rchunk, len(tiles))
        _warn_shard_shape(blocks, tiles, qchunk, rchunk, ctx.Q, M,
                          world=world, row_world=row_world)
        D = len(ctx.dampings)
        meta = {**_block_meta(ctx, raw_name=raw_name, query_type=query_type,
                              code_version=getattr(self.store, "code_version", "?")),
                "R": np.int64(R), "M": np.int64(M), "repeat_seed_stride": np.int64(1_000_000)}
        inv_box: list = []
        qbox: list = []

        def _inverses():
            if not inv_box:
                iv = {d: ctx.factors.damped_inverse_flat(d, self.device, mode=ctx.damping_mode)
                      for d in ctx.dampings}
                _check_finite_inverses(iv, ctx.damping_mode)
                inv_box.append(iv)
            return inv_box[0]

        def _qcoords(q0, q1):
            """Production draw, NOT re-drawn; computed once per block (see _stream)."""
            if not qbox or qbox[0][0] != (q0, q1):
                qbox.clear()
                qbox.append(((q0, q1), _query_pass(ctx, prog, q0, q1)))
            return qbox[0][1]

        def _compute(q0, q1, r0, r1):
            inv = _inverses()
            qcoords = _qcoords(q0, q1)
            blk = np.zeros((D, R, r1 - r0, q1 - q0), dtype=np.float32)
            for r in range(R):
                rseed = seed + 1_000_000 * (r + 1)       # pipeline.py's repeat-seed rule
                phase = f"repeat{r}" if len(blocks) == 1 and len(tiles) == 1 \
                    else f"repeat{r}[{q0}:{q1}]r[{r0}:{r1}]"
                for i in range(r0, r1):
                    lb = int(ctx.train_ds.labels[i]) if conditional else None
                    row = _train_row(ctx, qcoords, inv, ctx.train_ds.images[i], lb,
                                     seed=rseed, index=i, flipped=_train_flip(ctx, i))
                    if r == 0 and i - r0 < 2:
                        _check_finite_row(row, what="ekfac repeats", index=i)
                    for di, d in enumerate(ctx.dampings):
                        blk[di, r, i - r0] = row[d]
                    if (i - r0 + 1) % 25 == 0 or i + 1 == r1:
                        prog.log(done=r * (r1 - r0) + i - r0 + 1, total=R * (r1 - r0),
                                 phase=phase)
            return blk

        _run_shard(self.store, K.REPEAT_SCORES_BLOCK, raw_spec, blocks, tiles, M, meta,
                   _REPEAT_BLOCK_IDENTITY, rank=rank, world=world,
                   row_rank=row_rank, row_world=row_world, ignore_partials=force,
                   compute=_compute, what="ekfac repeats")
        qbox.clear()
        scores, binfo = _assemble_blocks(
            self.store, K.REPEAT_SCORES_BLOCK, raw_spec, blocks, tiles, M, meta,
            _REPEAT_BLOCK_IDENTITY, prefix=(D, R), Q=ctx.Q, rank=rank, world=world,
            row_rank=row_rank, row_world=row_world, what="ekfac repeats")
        if scores is None:
            log.info("ekfac repeats: %d/%d query blocks fully covered — repeat_scores not "
                     "filed yet", binfo["blocks_done"], binfo["blocks_total"])
            return {"method": raw_name, **binfo}
        self.store.save(K.REPEAT_SCORES, raw_spec, {
            "dampings": np.asarray(ctx.dampings, dtype=np.float64),
            "scores": scores,
            "indices": np.arange(M, dtype=np.int64),
            # meta as flat arrays (npz cannot hold dicts without pickling)
            "R": np.int64(R), "M": np.int64(M), "mc_loss": np.int64(ctx.mc_loss),
            "mc_measurement": np.int64(ctx.mc_meas), "hflip": np.int64(ctx.hflip),
            "train_mode": np.int64(ctx.train_mode), "seed": np.int64(seed),
            "repeat_seed_stride": np.int64(1_000_000),
            # solver caliber: σ̂ measured under one sampling/damping scheme and
            # consumed under another is a silent mis-shrink, so the two knobs
            # travel with the artifact and are re-checked on load (see
            # _check_repeat_caliber). Stored as 0-d unicode arrays.
            "sampling": np.array(ctx.sampling), "damping_mode": np.array(ctx.damping_mode),
            # per-partial provenance of the two-level shard that produced this
            # σ̂: (q-range, r-range, host, torch, grad_chunk). A JSON string
            # because npz holds arrays, not records — and allow_pickle=False.
            "partials": np.array(json.dumps(binfo.get("partials") or [])),
        })
        if clean_partials:
            n = _drop_blocks(self.store, K.REPEAT_SCORES_BLOCK, raw_spec)
            log.info("dropped %d repeat block partials (--clean-partials)", n)
        return {"shape": list(scores.shape), "R": R, "M": M,
                "dampings": ctx.dampings, **binfo}
