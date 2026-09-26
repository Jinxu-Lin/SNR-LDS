"""Counterfactual chain: removal sets -> retrains -> same-seed regeneration ->
query-loss change / pixel L2 / CLIP analysis with an AUC against random removal.

For each generated query, each method selects its native top-k training
scores. Retraining removes those rows and regeneration replays the original
query noise. Current paper arms are raw FMAS, D-TRAK, native-square DAS and
EK-FAC IF, with independent random removal sets as a reference. Every artifact
is filed under ``<method>_k<k>`` so different removal sizes remain separate.

Two platforms behind one seam:

* SD3.5+LoRA (ArtBench): retrain = ``run_latent_train_subsets`` with the mask
  source parameterised (same recipe, seeded LoRA init, sharding, idempotency);
  regeneration decodes latents through the VAE;
* UNet (CIFAR): retrain = the LDS subset recipe (``train_model`` on a masked
  ``SubsetView``, full epoch budget on the subset's size); samples ARE pixels.

Regeneration REPLAYS the original generation exactly (same ``gen_seed``, the
same full-batch ``x0`` draw, same sampler and step count, same labels), so the
only difference between an original and its counterfactual is the k removed
training images; ``verify`` re-runs that replay with the undeleted model and
demands bit-identity with the stored ``samples.pt``. Every removal set is
traceable to the exact score matrix (relpath + sha256) and the configuration that
produced it (``SCORES_META`` sidecar).

Analysis reports, per method: the query's Flow-Matching loss change under the
retrained model (the estimand of Definition 3.1, on the very query whose set
was removed), pixel L2 and CLIP cosine between the original and regenerated
images, and for each of the three the AUC of the method's records against the
random-removal records (MUCS-style statistic: 0.5 = no better than random).
"""
from __future__ import annotations

import hashlib
import io
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from balds.schema.artifact import ArtifactKind as K
from balds.schema.logging import get_logger, set_context
from balds.schema.registry import PROCESSES
from balds.schema.runspec import RunSpec
from balds.data.base import ImageDataset, SubsetView, balanced_query_indices
from balds.evaluation.counterfactual import (
    auc_vs_reference,
    cosine,
    pixel_l2,
    random_removal_mask,
    topk_removal_mask,
)
from balds.evaluation.ground_truth import compute_query_losses
from balds.models import DDPMSchedule, build_model, checkpoint_state, euler_solve, train_model
from balds.artifacts.progress import ProgressCSV, progress_path
from .config import resolve_p_uncond, resolve_paper_method, resolve_t_avg
from .common import _load_ds, _recipe_kwargs, _subset_steps

log = get_logger("balds.counterfactual")

#: Methods whose top-k sets are compared (the paper's method + the strongest baseline).
METHOD_ARMS: tuple[str, ...] = ("fmas", "dtrak_T100")
RANDOM_ARM = "random"
ALL_ARMS: tuple[str, ...] = (*METHOD_ARMS, RANDOM_ARM)
#: the UNDELETED model replayed on the same machine (torch build) as the arms;
#: k-independent. When it exists, ``analyze`` measures L2/CLIP against it, so a
#: cross-machine replay floor (§6.2-34 (b)) cannot leak into the distances.
REFERENCE_ARM = "reference"

#: Distinct init-seed stream for counterfactual retrains (so "q3" never shares
#: a seed with LDS "subset_3" of the same identity).
CF_SEED_OFFSET = 7_000_000

#: How many queries an arm covers when neither --query-indices nor --balanced
#: is given. Historic default; the six already-filed arms were built with it.
DEFAULT_N_QUERIES = 50

#: The fixed-rho build seam is deliberately narrower than the general curvature
#: scorer: it is an auditable source override for the adopted FMAS method only.
FMAS_METHOD = "fmas_raw"
FMAS_RAW_METHOD = "fmas_raw"
FIXED_RHO_SELECTION = "fixed_rho"

CALIBER_SD35 = (
    "platform: SD3.5-Medium (2.24B, flow matching) + LoRA r128 (95.6M params) at 256x256, "
    "euler-100, NO classifier-free guidance; N=5000. NOT comparable in absolute value to "
    "DAS (SD1.x 512->256, LoRA r128 = 25.5M, DDIM-50 + CFG 7.5; their L2 203.76 / CLIP 0.83): "
    "methods are only comparable to each other and to random removal. "
    "L2 = ||a-b||_2 on [0,1]-scaled pixels (uint8/255, flattened); CLIP = cosine of "
    "openai/clip-vit-base-patch32 image embeddings; loss = query FM loss averaged over "
    "T_avg timepoints x 3 noise seeds (the LDS ground-truth quantity)."
)
CALIBER_UNET = (
    "platform: 36M U-Net flow matching at 32x32 (the CIFAR LDS platform), euler-100, "
    "full-model retrain on the same recipe as the LDS subsets (200 epochs on the kept "
    "samples). L2 = ||a-b||_2 on [0,1]-scaled pixels; CLIP = cosine of "
    "openai/clip-vit-base-patch32 image embeddings (32px upsampled by the processor); "
    "loss = query FM loss averaged over T_avg timepoints x 3 noise seeds (the LDS "
    "ground-truth quantity). Methods comparable to each other and to random removal."
)


def arm_id(arm: str, k: int) -> str:
    """Artifact key of one (method, k) arm: ``fmas_k300``, ``random_k1000`` ..."""
    return f"{arm}_k{int(k)}"


def _base(dataset: str, seed: int, process: str, conditional: bool) -> RunSpec:
    return RunSpec(dataset=dataset, process=process, seed=seed, conditional=conditional)


def format_rho(rho: float) -> str:
    """Canonical token shared with ``SCORES_LAMBDA`` addressing and CF meta."""
    value = float(rho)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"FMAS rho must be a finite positive number, got {rho!r}")
    return f"{value:.2e}"


def _array_content_sha256(value: np.ndarray) -> str:
    """Stable hash of an in-memory ndarray's identity and C-order contents.

    The hash names the exact matrix used to select a deletion set without
    introducing a second score artifact or claiming an oracle selected it.
    """
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    # Preserve the historical hash domain so archived identities remain valid.
    h.update(b"cfa.ndarray.v1\0")
    h.update(arr.dtype.str.encode("ascii"))
    h.update(b"\0")
    h.update(",".join(str(int(v)) for v in arr.shape).encode("ascii"))
    h.update(b"\0")
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def fixed_rho_fmas_scores(store, cfg: dict, spec: RunSpec, rho: float) -> tuple[np.ndarray, dict]:
    """Load the paper's bilinear FMAS scores at an explicit curvature damping."""
    method = resolve_paper_method(cfg, spec.method or "fmas")
    if method != FMAS_RAW_METHOD:
        raise ValueError("fixed rho is only supported for FMAS")
    raw_spec = spec.with_(method=FMAS_RAW_METHOD)
    token = format_rho(rho)
    value = float(token)
    if not store.exists(K.SCORES_LAMBDA, raw_spec, lam=value):
        raise ValueError(f"fixed-rho FMAS source is missing at rho={token}")
    scores = np.asarray(store.load(K.SCORES_LAMBDA, raw_spec, lam=value))
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("fixed-rho scores must be a finite N by Q matrix")
    raw = {"method": FMAS_RAW_METHOD, **store.describe(K.SCORES_LAMBDA, raw_spec, lam=value)}
    return scores, {"method": method, "selection": FIXED_RHO_SELECTION,
                    "rho": token, "rho_value": value, "transform": "identity", "raw_scores": raw}


class CounterfactualUseCase:
    """The four stages + the parity self-check, with injection seams for the
    heavy parts (model materialisation, decode, CLIP) so the whole chain is
    testable on a stub."""

    def __init__(self, store, cfg, *, device: str = "cuda:0",
                 model_factory: Optional[Callable] = None,
                 decode_fn: Optional[Callable] = None,
                 embed_fn: Optional[Callable] = None) -> None:
        self.store, self.cfg, self.device = store, cfg, device
        self._model_factory = model_factory
        self._decode_fn = decode_fn
        self._embed_fn = embed_fn

    # ------------------------------------------------------------------ seams
    def _is_latent(self, dataset: str) -> bool:
        from .latent import is_latent_platform
        return is_latent_platform(self.cfg, dataset)

    def _model(self, dataset: str, ckpt: Optional[dict]):
        if self._model_factory is not None:
            return self._model_factory(ckpt)
        if self._is_latent(dataset):
            from .latent import build_latent_model
            return build_latent_model(self.store, self.cfg, dataset, ckpt, device=self.device)
        if ckpt is None:
            raise ValueError("the UNet platform materialises models from checkpoints only")
        return build_model(ckpt, device=self.device)

    def _load_into(self, model, dataset: str, ckpt: dict):
        """Put ``ckpt``'s weights into ``model`` (LoRA state on the latent
        platform); on the UNet platform a fresh model is built instead."""
        if self._is_latent(dataset):
            from balds.models.sd3 import load_lora_state
            load_lora_state(model, ckpt["lora"])
            return model.eval()
        return self._model(dataset, ckpt)

    def _decoder(self, dataset: str):
        """``samples -> uint8 images``; the real latent one loads the VAE once."""
        if self._decode_fn is not None:
            return self._decode_fn
        if not self._is_latent(dataset):
            return lambda x: ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        from balds.models.sd3 import decode_latents
        from .latent import platform_cfg, platform_spec
        from balds.models.sd3 import load_vae
        vae = load_vae(platform_cfg(self.cfg, dataset)["base_model"], device=self.device)

        def dec(latents: torch.Tensor) -> torch.Tensor:
            imgs = decode_latents(vae, latents, device=self.device)
            return ((imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        return dec

    def _embedder(self):
        """``uint8 images (n,3,H,W) -> (n, d) embeddings`` (CLIP image tower)."""
        if self._embed_fn is not None:
            return self._embed_fn
        from balds.attribution.embed import embed

        def emb(images_u8: torch.Tensor) -> torch.Tensor:
            x = images_u8.float() / 127.5 - 1.0                # [-1, 1], what embed() expects
            return embed("clip", x, device=self.device)
        return emb

    def _train_set(self, dataset: str):
        if self._is_latent(dataset):
            from .latent import latent_train_plain
            return latent_train_plain(self.store, dataset)
        train_ds, _ = _load_ds(self.cfg, dataset)
        return ImageDataset(train_ds.images, train_ds.labels)

    def _caliber(self, dataset: str, k: int) -> str:
        return (CALIBER_SD35 if self._is_latent(dataset) else CALIBER_UNET) + f" k={int(k)} removed."

    # --------------------------------------------------------- 1. removal sets
    def build(self, dataset: str, seed: int, *, process: str = "cfm", conditional: bool = True,
              arms: Optional[Sequence[str]] = None, k: int = 1000,
              n_queries: Optional[int] = None,
              query_indices: Optional[Sequence[int]] = None,
              balanced: Optional[int] = None,
              n_random: int = 5, query_type: str = "gen",
              fmas_rho: Optional[float] = None, force: bool = False) -> dict:
        """Removal sets for each arm over a chosen SET of queries.

        Which queries an arm covers is one decision expressed three ways, so the
        three are mutually exclusive:

        * ``n_queries`` (default 50) — the first n columns, the historic behaviour;
        * ``query_indices`` — an explicit list;
        * ``balanced`` K — K/num_classes per class, taken from the generation's
          own labels.

        ``balanced`` exists because a prefix is not a sample of the query set:
        ``_balanced_labels`` lays the generated queries out class by class, so on
        ArtBench-2's Q=100 the first 50 are ALL style 4 and ``--n-queries 10``
        buys ten queries of one class. It is resolved to explicit indices HERE
        and stored in the meta — the artifact records indices, never a rule, so
        reproducing an arm never re-derives anything from a labels vector that a
        later regeneration could change.
        """
        base = _base(dataset, seed, process, conditional)
        set_context(run_id=base.digest()[:6], trace_id=f"cf-build:{dataset}")
        arms = list(arms or ALL_ARMS)
        rho_token = format_rho(fmas_rho) if fmas_rho is not None else None
        if fmas_rho is not None and not any(
                resolve_paper_method(self.cfg, arm) == FMAS_METHOD for arm in arms):
            raise ValueError("--fmas-rho is only valid when the build includes the FMAS arm")
        qidx = self._resolve_queries(base, n_queries=n_queries,
                                     query_indices=query_indices, balanced=balanced)
        out: dict = {}
        for arm in arms:
            aid = arm_id(arm, k)
            fixed_for_arm = (fmas_rho is not None and arm != RANDOM_ARM and
                             resolve_paper_method(self.cfg, arm) == FMAS_METHOD)
            if self.store.exists(K.CF_META, base, arm=aid) and not force:
                meta = self.store.load(K.CF_META, base, arm=aid)
                if fixed_for_arm:
                    src = meta.get("source") or {}
                    if (src.get("selection") != FIXED_RHO_SELECTION or
                            src.get("rho") != rho_token):
                        raise ValueError(
                            f"arm {aid} already exists with a different score source; "
                            f"requested fixed rho={rho_token}. Inspect its CF meta and use "
                            "--force only if replacing those removal masks is intended.")
                if all(self.store.exists(K.CF_MASK, base, arm=aid, qi=i)
                       for i in range(int(meta["n_sets"]))):
                    log.info("arm %s: %d removal sets exist, skipping", aid, meta["n_sets"])
                    out[aid] = {"skipped": True, "n_sets": meta["n_sets"]}
                    continue
            if arm == RANDOM_ARM:
                N = len(self._train_set(dataset))
                seeds = [seed * 100003 + 7919 * (r + 1) for r in range(n_random)]
                masks = [random_removal_mask(N, k, s) for s in seeds]
                meta = {"arm": aid, "method": arm, "k": int(k), "N": int(N), "n_sets": len(masks),
                        "n_queries": len(qidx), "query_indices": list(qidx),
                        "query_type": query_type,
                        "random_seeds": seeds, "source": None,
                        "semantics": "qi indexes an independent random set; each set "
                                     "regenerates every query in query_indices"}
                upstream: tuple[str, ...] = ()
            else:
                method = "das_T100" if arm == "das_native_sq" else resolve_paper_method(self.cfg, arm)
                sspec = RunSpec(dataset=dataset, seed=seed, method=method, query_type=query_type,
                                process=process, conditional=conditional)
                if fixed_for_arm:
                    scores, source = fixed_rho_fmas_scores(
                        self.store, self.cfg, sspec, float(fmas_rho))
                    upstream = (source["raw_scores"]["sha256"],)
                else:
                    scores = np.asarray(self.store.load(K.SCORES, sspec))
                    if arm == "das_native_sq":
                        scores = scores.astype(float) ** 2
                    source = {"method": method, **self.store.describe(K.SCORES, sspec)}
                    if self.store.exists(K.SCORES_META, sspec):
                        source.update({f"scores_{kk}": v for kk, v in
                                       self.store.load(K.SCORES_META, sspec).items()})
                    else:
                        source["scores_best_lam"] = None        # pre-sidecar artifact
                        source["scores_gamma"] = None
                    upstream = ()
                N, Q = scores.shape
                if qidx and max(qidx) >= Q:
                    raise ValueError(f"{arm}: scores have Q={Q} but query index "
                                     f"{max(qidx)} was requested")
                masks = [topk_removal_mask(scores[:, qi], k) for qi in qidx]
                meta = {"arm": aid, "method": arm, "k": int(k), "N": int(N), "n_sets": len(masks),
                        "n_queries": len(qidx), "query_indices": list(qidx),
                        "query_type": query_type,
                        "source": source,
                        "semantics": "qi = position in query_indices; the set is that "
                                     "query's top-k most positive-influence training samples"}
            for qi, mk in enumerate(masks):
                assert int((~mk).sum()) == k
                self.store.save(K.CF_MASK, base, mk.astype(np.bool_), upstream=upstream,
                                arm=aid, qi=qi)
            self.store.save(K.CF_META, base, meta, upstream=upstream, arm=aid)
            log.info("arm %s: %d removal sets of k=%d built over queries %s",
                     aid, len(masks), k, qidx)
            out[aid] = {"n_sets": len(masks), "k": k, "query_indices": list(qidx)}
        return out

    def _resolve_queries(self, base: RunSpec, *, n_queries, query_indices,
                         balanced) -> list[int]:
        """The explicit query index list an arm is built over."""
        given = [name for name, v in (("n_queries", n_queries),
                                      ("query_indices", query_indices),
                                      ("balanced", balanced)) if v is not None]
        if len(given) > 1:
            raise ValueError(f"--n-queries / --query-indices / --balanced are three ways "
                             f"to say the same thing and are mutually exclusive; got {given}")
        if query_indices is not None:
            qidx = [int(v) for v in query_indices]
            if len(set(qidx)) != len(qidx):
                raise ValueError(f"--query-indices has duplicates: {qidx}")
            if any(v < 0 for v in qidx):
                raise ValueError(f"--query-indices must be non-negative: {qidx}")
            return qidx
        if balanced is not None:
            # class-balanced over the GENERATION's labels, resolved to indices now
            labels = self.store.load(K.GENERATION, base)["labels"]
            return balanced_query_indices(labels, int(balanced))
        return list(range(int(n_queries if n_queries is not None else DEFAULT_N_QUERIES)))

    @staticmethod
    def _query_indices(meta: dict) -> list[int]:
        """The query indices an arm covers, from its meta.

        Backward compatible on purpose: a meta written before this key existed
        (the six arms already on disk) means the first ``n_queries`` columns,
        which is exactly what it meant then."""
        qi = meta.get("query_indices")
        if qi is None:
            return list(range(int(meta["n_queries"])))
        return [int(v) for v in qi]

    def _sets(self, base: RunSpec, aid: str):
        meta = self.store.load(K.CF_META, base, arm=aid)
        masks = [np.asarray(self.store.load(K.CF_MASK, base, arm=aid, qi=i), dtype=bool)
                 for i in range(int(meta["n_sets"]))]
        return masks, meta

    # ------------------------------------------------------------ 2. retrain
    def retrain(self, dataset: str, seed: int, *, arm: str, k: int = 1000,
                process: str = "cfm", conditional: bool = True,
                rank: int = 0, world: int = 1) -> dict:
        aid = arm_id(arm, k)
        base = _base(dataset, seed, process, conditional)
        set_context(run_id=base.digest()[:6], trace_id=f"cf-retrain:{dataset}:{aid}")
        masks, meta = self._sets(base, aid)
        if self._is_latent(dataset):
            from .latent import run_latent_train_subsets
            return run_latent_train_subsets(
                self.store, self.cfg, dataset, seed, process=process, device=self.device,
                rank=rank, world=world, masks=masks, ckpt_kind=K.CF_CHECKPOINT,
                key_fn=lambda m: {"arm": aid, "qi": m},
                progress_tag=f"cf_{aid}_{dataset}_{process}_seed{seed}",
                seed_offset=CF_SEED_OFFSET,
                model_factory=(lambda: self._model(dataset, None)))
        return self._retrain_unet(base, aid, masks, rank=rank, world=world)

    def _retrain_unet(self, base: RunSpec, aid: str, masks, *, rank: int, world: int) -> dict:
        """The LDS subset recipe, mask source = the removal sets (mirrors
        ``SubsetsUseCase.train_subsets`` knob for knob: full epoch budget on the
        kept samples, same optimizer/schedule/augmentation, seeded init)."""
        train_ds = self._train_set(base.dataset)
        t = self.cfg["train"]
        num_classes = self.cfg["model"]["num_classes"] if base.conditional else None
        p_unc = resolve_p_uncond(self.cfg, base.dataset)
        tasks = [m for m in range(len(masks)) if m % world == rank]
        prog = ProgressCSV(progress_path(
            self.cfg["storage"]["data_root"],
            f"cf_{aid}_{base.dataset}_{base.process}_seed{base.seed}_rank{rank}of{world}"))
        log.info("counterfactual retrain (UNet): %d/%d sets (rank %d/%d), batch=%d",
                 len(tasks), len(masks), rank, world, t["batch_size"])
        trained = 0
        for i, m in enumerate(tasks):
            if self.store.exists(K.CF_CHECKPOINT, base, arm=aid, qi=m):
                prog.log(done=i + 1, total=len(tasks), subset_m=m, final_loss="", skipped=1)
                continue
            sub = SubsetView(train_ds, masks[m])
            steps = _subset_steps(self.cfg, len(sub), t["batch_size"])
            retrain_seed = base.seed + m + 1 + CF_SEED_OFFSET
            model, hist = train_model(base.process, sub, steps=steps,
                                      batch_size=t["batch_size"], lr=t["lr"], seed=retrain_seed,
                                      num_classes=num_classes, p_uncond=p_unc,
                                      device=self.device, base_ch=self.cfg["model"]["base_ch"],
                                      compile_model=bool(t.get("compile", True)),
                                      **_recipe_kwargs(t))
            ckpt = checkpoint_state(model, base_ch=self.cfg["model"]["base_ch"],
                                    num_classes=num_classes, step=steps,
                                    extra={"model_type": base.process, "conditional": base.conditional,
                                           "p_uncond": p_unc if base.conditional else None,
                                           "counterfactual_arm": aid, "qi": m})
            self.store.save(K.CF_CHECKPOINT, base, ckpt, arm=aid, qi=m)
            trained += 1
            prog.log(done=i + 1, total=len(tasks), subset_m=m,
                     final_loss=hist[-1]["loss"] if hist else "", skipped=0)
        return {"trained": trained, "shard": len(tasks), "n_sets": len(masks)}

    # --------------------------------------------------------- 3. regenerate
    @staticmethod
    def replay(model, gen: dict, *, device: str) -> torch.Tensor:
        """Re-run the stored generation's exact path: same seed, the same
        FULL-batch ``x0`` draw (batch structure is part of the numerics), same
        sampler/steps/labels. Returns samples ``(Q, ...)`` on the CPU."""
        Q = int(gen["Q"])
        shape = tuple(int(v) for v in gen["samples"].shape[1:])
        torch.manual_seed(int(gen["gen_seed"]))
        x0 = torch.randn(Q, *shape)
        with torch.no_grad():
            return euler_solve(model, x0, steps=int(gen["ode_steps"]), device=device,
                               class_label=gen["labels"].to(device)).cpu()

    def regenerate(self, dataset: str, seed: int, *, arm: str, k: int = 1000,
                   process: str = "cfm", conditional: bool = True,
                   rank: int = 0, world: int = 1, force: bool = False) -> dict:
        base = _base(dataset, seed, process, conditional)
        if process != "cfm":
            raise ValueError("same-seed regeneration replays the Euler ODE; the ddpm sampler "
                             "is ancestral and has no deterministic replay here")
        if arm == REFERENCE_ARM:
            return self._regenerate_reference(dataset, base, force=force)
        aid = arm_id(arm, k)
        set_context(run_id=base.digest()[:6], trace_id=f"cf-regen:{dataset}:{aid}")
        gen = self.store.load(K.GENERATION, base)
        _, meta = self._sets(base, aid)
        n_sets = int(meta["n_sets"])
        tasks = [i for i in range(n_sets) if i % world == rank]
        prog = ProgressCSV(progress_path(
            self.cfg["storage"]["data_root"],
            f"cf_regen_{aid}_{dataset}_{process}_seed{seed}_rank{rank}of{world}"))
        model, decode = None, None
        done = 0
        for i, qi in enumerate(tasks):
            if self.store.exists(K.CF_GENERATION, base, arm=aid, qi=qi) and not force:
                prog.log(done=i + 1, total=len(tasks), qi=qi, skipped=1)
                continue
            if not self.store.exists(K.CF_CHECKPOINT, base, arm=aid, qi=qi):
                raise FileNotFoundError(f"no counterfactual model for arm={aid} qi={qi}; "
                                        f"run `balds counterfactual retrain --arm {arm} --k {k}` first")
            if decode is None:
                decode = self._decoder(dataset)
            ckpt = self.store.load(K.CF_CHECKPOINT, base, arm=aid, qi=qi)
            if self._is_latent(dataset):
                if model is None:
                    model = self._model(dataset, None)       # base once per shard
                model = self._load_into(model, dataset, ckpt)
            else:
                model = self._model(dataset, ckpt)
            model.eval()
            samples = self.replay(model, gen, device=self.device)
            # qi is the arm's SET index; the query it belongs to is meta's index
            # list at that position (identity when the arm covers 0..n-1).
            qsel = self._query_indices(meta)
            idx = ([qsel[qi]] if arm != RANDOM_ARM else list(qsel))
            sel = torch.tensor(idx, dtype=torch.long)
            images_u8 = decode(samples[sel])
            self.store.save(K.CF_GENERATION, base, {
                "arm": aid, "qi": qi, "indices": sel, "samples": samples[sel],
                "images_u8": images_u8, "gen_seed": int(gen["gen_seed"]),
                "ode_steps": int(gen["ode_steps"]), "batch_Q": int(gen["Q"]),
            }, arm=aid, qi=qi)
            done += 1
            prog.log(done=i + 1, total=len(tasks), qi=qi, skipped=0)
        return {"arm": aid, "regenerated": done, "shard": len(tasks), "n_sets": n_sets}

    def _regenerate_reference(self, dataset: str, base: RunSpec, *, force: bool = False) -> dict:
        """Replay the stored generation with the UNDELETED model here and file it
        as the ``reference`` arm (qi=0, every query). Same replay path as the
        arms, so it shares whatever floating-point build they were made on."""
        set_context(run_id=base.digest()[:6], trace_id=f"cf-regen:{dataset}:{REFERENCE_ARM}")
        if self.store.exists(K.CF_GENERATION, base, arm=REFERENCE_ARM, qi=0) and not force:
            log.info("reference replay exists, skipping (use --force)")
            return {"arm": REFERENCE_ARM, "regenerated": 0, "shard": 1, "n_sets": 1}
        gen = self.store.load(K.GENERATION, base)
        model = self._model(dataset, self.store.load(K.CHECKPOINT, base))
        model.eval()
        samples = self.replay(model, gen, device=self.device)
        sel = torch.arange(int(gen["Q"]), dtype=torch.long)
        self.store.save(K.CF_GENERATION, base, {
            "arm": REFERENCE_ARM, "qi": 0, "indices": sel, "samples": samples,
            "images_u8": self._decoder(dataset)(samples), "gen_seed": int(gen["gen_seed"]),
            "ode_steps": int(gen["ode_steps"]), "batch_Q": int(gen["Q"]),
            "torch": torch.__version__,
        }, arm=REFERENCE_ARM, qi=0)
        return {"arm": REFERENCE_ARM, "regenerated": 1, "shard": 1, "n_sets": 1}

    def verify(self, dataset: str, seed: int, *, process: str = "cfm", conditional: bool = True,
               strict: bool = True, l2_tol: Optional[float] = None) -> dict:
        """Replay the stored generation with the UNDELETED model, twice.

        Two-level verdict (§6.2-34 (b)):

        * ``strict``   — replay is bit-identical to ``samples.pt`` (and its decoded
          previews): the historical requirement, met on the machine that generated.
        * ``tolerant`` — the two replays here are bit-identical to EACH OTHER (so
          arm-vs-arm differences carry no nondeterminism), the decoded previews
          differ from the stored ones by at most one uint8 level, and the per-query
          pixel-L2 replay floor is at most ``l2_tol`` (config
          ``counterfactual.replay_l2_tol``, default 0.05 — 2× the 2026-09-04 torch
          2.6-vs-2.12 measurement of 0.0225, 1/380 of the between-query scale).
        * ``fail``     — anything else; ``strict=True`` raises.

        The check is filed as ``analysis/<seed>/replay_check.json`` so ``analyze``
        can report the floor next to the effect it must stay small against."""
        base = _base(dataset, seed, process, conditional)
        set_context(run_id=base.digest()[:6], trace_id=f"cf-verify:{dataset}")
        gen = self.store.load(K.GENERATION, base)
        model = self._model(dataset, self.store.load(K.CHECKPOINT, base))
        model.eval()
        samples = self.replay(model, gen, device=self.device)
        again = self.replay(model, gen, device=self.device)
        self_ok = bool(torch.equal(samples, again))
        same = bool(torch.equal(samples, gen["samples"]))
        max_diff = float((samples - gen["samples"]).abs().max())
        tol = float(l2_tol if l2_tol is not None else
                    (self.cfg.get("counterfactual", {}) or {}).get("replay_l2_tol", 0.05))
        decode = self._decoder(dataset)
        imgs = decode(samples)
        stored = gen["images_u8"] if "images_u8" in gen else decode(gen["samples"])
        floors = [pixel_l2(stored[i].numpy(), imgs[i].numpy()) for i in range(imgs.shape[0])]
        out = {"samples_identical": same, "self_consistent": self_ok, "max_abs_diff": max_diff,
               "images_identical": bool(torch.equal(imgs, stored)),
               "u8_max_delta": int((imgs.to(torch.int16) - stored.to(torch.int16)).abs().max()),
               "l2_floor_mean": float(np.mean(floors)), "l2_floor_max": float(np.max(floors)),
               "l2_tol": tol, "torch": torch.__version__,
               "Q": int(gen["Q"]), "gen_seed": int(gen["gen_seed"]),
               "ode_steps": int(gen["ode_steps"])}
        if same and out["images_identical"]:
            out["verdict"] = "strict"
        elif self_ok and out["u8_max_delta"] <= 1 and out["l2_floor_max"] <= tol:
            out["verdict"] = "tolerant"
        else:
            out["verdict"] = "fail"
        self.store.save(K.CF_ANALYSIS, base, out, name="replay_check")
        if strict and out["verdict"] == "fail":
            raise ValueError(f"replay verification FAILED: {out} — counterfactual distances "
                             f"would measure replay drift, not data (see §6.2-34 (b))")
        log.info("replay parity (%s): %s", out["verdict"], out)
        return out

    # ------------------------------------------------- 4. query-loss + analyse
    def _query_loss(self, dataset: str, model, queries: torch.Tensor, labels: torch.Tensor,
                    process: str, e_seeds=(0, 1, 2)) -> np.ndarray:
        """The LDS ground-truth quantity for the given queries under ``model``:
        FM loss averaged over T_avg timepoints and the e-seed noise draws."""
        T_avg = resolve_t_avg(self.cfg, dataset)
        if self._is_latent(dataset):
            from .latent import platform_process
            proc, _ = platform_process(self.cfg, dataset, device=self.device)
        else:
            proc = PROCESSES.get(process)
        sched = proc._schedule(self.device) if process == "ddpm" else None
        rows = [compute_query_losses(model, queries, labels, process, self.device,
                                     process=proc, schedule=sched, T_avg=T_avg, seed=int(e),
                                     target_batch=int(self.cfg["lds"]["chunk_target_batch"]))
                for e in e_seeds]
        return np.mean(np.stack(rows, 0), 0)

    def analyze(self, dataset: str, seed: int, *, k: int = 1000, process: str = "cfm",
                conditional: bool = True, arms: Optional[Sequence[str]] = None,
                n_qual: int = 8, with_loss: bool = True) -> dict:
        base = _base(dataset, seed, process, conditional)
        set_context(run_id=base.digest()[:6], trace_id=f"cf-analyze:{dataset}:k{k}")
        arms = list(arms or ALL_ARMS)
        gen = self.store.load(K.GENERATION, base)
        decode = self._decoder(dataset)
        if self.store.exists(K.CF_GENERATION, base, arm=REFERENCE_ARM, qi=0):
            ref = self.store.load(K.CF_GENERATION, base, arm=REFERENCE_ARM, qi=0)
            orig, reference = ref["images_u8"], "replay"
            log.info("L2/CLIP measured against the same-machine reference replay "
                     "(torch %s)", ref.get("torch", "?"))
        else:
            orig = gen["images_u8"] if "images_u8" in gen else decode(gen["samples"])
            reference = "stored"
        replay_check = (self.store.load(K.CF_ANALYSIS, base, name="replay_check")
                        if self.store.exists(K.CF_ANALYSIS, base, name="replay_check") else None)
        embed = self._embedder()
        # the undeleted model's loss on every query (the reference for Δ loss)
        base_loss = None
        if with_loss:
            model0 = self._model(dataset, self.store.load(K.CHECKPOINT, base))
            model0.eval()
            base_loss = self._query_loss(dataset, model0, gen["samples"], gen["labels"], process)
        records: dict[str, list[dict]] = {}
        qual: dict[str, dict[int, torch.Tensor]] = {}
        base_model = None
        for arm in arms:
            aid = arm_id(arm, k)
            _, meta = self._sets(base, aid)
            recs, cf_imgs, cf_idx, cf_qi = [], [], [], []
            cf_loss: dict[tuple[int, int], float] = {}
            for qi in range(int(meta["n_sets"])):
                if not self.store.exists(K.CF_GENERATION, base, arm=aid, qi=qi):
                    continue
                g = self.store.load(K.CF_GENERATION, base, arm=aid, qi=qi)
                idxs = g["indices"].tolist()
                for j, idx in enumerate(idxs):
                    cf_imgs.append(g["images_u8"][j]); cf_idx.append(idx); cf_qi.append(qi)
                if with_loss and self.store.exists(K.CF_CHECKPOINT, base, arm=aid, qi=qi):
                    ckpt = self.store.load(K.CF_CHECKPOINT, base, arm=aid, qi=qi)
                    if self._is_latent(dataset):
                        if base_model is None:
                            base_model = self._model(dataset, None)
                        m = self._load_into(base_model, dataset, ckpt)
                    else:
                        m = self._model(dataset, ckpt)
                    m.eval()
                    sel = torch.tensor(idxs, dtype=torch.long)
                    ql = self._query_loss(dataset, m, gen["samples"][sel], gen["labels"][sel], process)
                    for j, idx in enumerate(idxs):
                        cf_loss[(qi, idx)] = float(ql[j])
            if not cf_imgs:
                log.warning("arm %s: no regenerated images yet", aid)
                records[aid] = []
                continue
            cf_stack = torch.stack(cf_imgs)
            e_cf = embed(cf_stack).float().cpu().numpy()
            e_or = embed(orig[torch.tensor(cf_idx)]).float().cpu().numpy()
            for j in range(len(cf_imgs)):
                rec = {"qi": int(cf_qi[j]), "query": int(cf_idx[j]),
                       "l2": pixel_l2(orig[cf_idx[j]].numpy(), cf_imgs[j].numpy()),
                       "clip": cosine(e_or[j], e_cf[j])}
                key = (int(cf_qi[j]), int(cf_idx[j]))
                if base_loss is not None and key in cf_loss:
                    rec["loss_cf"] = cf_loss[key]
                    rec["loss_base"] = float(base_loss[cf_idx[j]])
                    rec["dloss"] = cf_loss[key] - float(base_loss[cf_idx[j]])
                recs.append(rec)
                if cf_idx[j] not in qual.setdefault(aid, {}):
                    qual[aid][cf_idx[j]] = cf_imgs[j]
            records[aid] = recs
        summary = {aid: _summ(recs) for aid, recs in records.items()}
        rand_id = arm_id(RANDOM_ARM, k)
        auc = {}
        if records.get(rand_id):
            for aid, recs in records.items():
                if aid == rand_id or not recs:
                    continue
                auc[aid] = _auc_block(recs, records[rand_id])
        result = {"caliber": self._caliber(dataset, k), "dataset": dataset, "seed": seed,
                  "k": int(k), "arms": [arm_id(a, k) for a in arms], "reference": reference,
                  "summary": summary, "auc_vs_random": auc, "records": records}
        if replay_check is not None and reference == "stored":
            # the cross-build replay floor must stay negligible against the effect
            floor = float(replay_check["l2_floor_max"])
            effects = [s["l2_median"] for a, s in summary.items()
                       if a != rand_id and s.get("l2_median")]
            ratio = floor / min(effects) if effects else float("nan")
            max_ratio = float((self.cfg.get("counterfactual", {}) or {})
                              .get("replay_floor_max_ratio", 0.05))
            result["replay_floor"] = {"l2_floor_max": floor, "verdict": replay_check["verdict"],
                                      "floor_over_min_effect": ratio, "max_ratio": max_ratio}
            if ratio > max_ratio:
                raise ValueError(f"replay floor {floor:.4f} is {ratio:.1%} of the smallest "
                                 f"arm effect (limit {max_ratio:.0%}); file a same-machine "
                                 f"`regenerate --arm reference` before analysing")
        self.store.save(K.CF_ANALYSIS, base, result, name=f"summary_k{k}")
        self.store.save(K.CF_FIGURE, base, _boxplot_png(records, k), name=f"boxplot_k{k}")
        self.store.save(K.CF_FIGURE, base,
                        _qualitative_png(orig, qual, [arm_id(a, k) for a in arms], n_qual),
                        name=f"qualitative_k{k}")
        log.info("counterfactual analysis k=%d: %s auc=%s", k,
                 {a: s.get("n") for a, s in summary.items()}, auc)
        return {"caliber": result["caliber"], "summary": summary, "auc_vs_random": auc,
                "figures": [f"boxplot_k{k}", f"qualitative_k{k}"]}


def _summ(recs: list[dict]) -> dict:
    if not recs:
        return {"n": 0}
    out = {"n": len(recs)}
    for key in ("l2", "clip", "dloss"):
        vals = np.array([r[key] for r in recs if key in r], dtype=np.float64)
        if vals.size:
            out.update({f"{key}_mean": float(vals.mean()), f"{key}_median": float(np.median(vals)),
                        f"{key}_std": float(vals.std())})
    return out


def _auc_block(recs: list[dict], ref: list[dict]) -> dict:
    """AUC of a method's records against the random-removal records, per metric.
    Larger change = more influential set: L2 up, loss up, CLIP cosine DOWN (so
    the CLIP AUC is taken on 1 - cosine)."""
    out = {}
    for key, flip in (("l2", False), ("clip", True), ("dloss", False)):
        a = [(-r[key] if flip else r[key]) for r in recs if key in r]
        b = [(-r[key] if flip else r[key]) for r in ref if key in r]
        if a and b:
            out[key] = auc_vs_reference(np.asarray(a), np.asarray(b))
    return out


def _fig_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    return buf.getvalue()


def _boxplot_png(records: dict[str, list[dict]], k: int) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in records if records[a]]
    panels = [("l2", "pixel L2 (orig vs counterfactual)"),
              ("clip", "CLIP cosine (orig vs counterfactual)")]
    if any("dloss" in r for a in arms for r in records[a]):
        panels.append(("dloss", "query FM-loss change (retrained - original)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (key, title) in zip(axes, panels):
        data = [[r[key] for r in records[a] if key in r] for a in arms]
        if any(len(d) for d in data):
            ax.boxplot(data, tick_labels=arms, showfliers=True)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"counterfactual removal of top-{k} per method (values comparable across "
                 f"methods and random only)", fontsize=9)
    fig.tight_layout()
    out = _fig_bytes(fig)
    plt.close(fig)
    return out


def _qualitative_png(orig: torch.Tensor, qual: dict[str, dict[int, torch.Tensor]],
                     arms: list[str], n_qual: int) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    common = [q for q in range(orig.shape[0])
              if all(q in qual.get(a, {}) for a in arms)][:n_qual]
    cols = ["original", *arms]
    rows = max(1, len(common))
    fig, axes = plt.subplots(rows, len(cols), figsize=(2.2 * len(cols), 2.2 * rows))
    axes = np.array(axes).reshape(rows, len(cols))
    for r, q in enumerate(common):
        for c, name in enumerate(cols):
            img = orig[q] if name == "original" else qual[name][q]
            axes[r, c].imshow(img.permute(1, 2, 0).numpy())
            axes[r, c].set_axis_off()
            if r == 0:
                axes[r, c].set_title(name, fontsize=8)
    if not common:
        for ax in axes.ravel():
            ax.set_axis_off()
        axes[0, 0].set_title("no query regenerated in every arm yet", fontsize=8)
    fig.tight_layout()
    out = _fig_bytes(fig)
    plt.close(fig)
    return out
