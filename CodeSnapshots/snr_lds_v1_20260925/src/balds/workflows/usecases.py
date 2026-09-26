"""Application use-cases — the orchestration of the Business layer.

Each use-case receives an :class:`~balds.schema.artifact.ArtifactStore` and a
config, and coordinates featurization / scoring / evaluation / aggregation. It is
the *only* place that both reads artifacts (via the store) and runs business
logic; the CLI calls these and renders, never the other way round.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from balds.schema.artifact import ArtifactKind as K
from balds.schema.estimator import FeatureSet
from balds.schema.logging import get_logger, set_context
from balds.schema.registry import METHODS, SELECTORS
from balds.schema.runspec import RunSpec
from balds.workflows.checkpoints import attribution_checkpoints
from balds.workflows.config import resolve_gamma
from balds.evaluation.lds import compute_lds, lds_of_scores, predicted_influence, support_head_scores
from balds.evaluation.stats import bootstrap_ci
from balds.attribution.shrinkage import (ShrinkageLayer, select_after_shrinkage,
                                      sigma_from_repeats, snr_stats)

log = get_logger("balds.workflows")


def progress_overview(cfg) -> list[dict]:
    """Summaries of ``<data_root>/progress/*.csv`` (CLI-facing; app owns the
    store import so the presentation layer never touches ``balds.artifacts``)."""
    import os
    from balds.artifacts.progress import summarize
    return summarize(os.path.join(cfg["storage"]["data_root"], "progress"))


class ScoreUseCase:
    """Compute attribution scores for a method via the λ-sweep + a selector."""

    def __init__(self, store, cfg, *, device: str = "cuda") -> None:
        self.store, self.cfg, self.device = store, cfg, device

    def run(self, method: str, dataset: str, seed: int, *, query_type: str = "gen",
            process: str = "cfm", conditional: bool = True, proj_dim: Optional[int] = None,
            proj_seed: Optional[int] = None, rescore: bool = False) -> dict:
        from .config import resolve_paper_method
        method = resolve_paper_method(self.cfg, method)
        spec = RunSpec(dataset=dataset, seed=seed, method=method, query_type=query_type,
                       process=process, conditional=conditional)
        set_context(run_id=spec.digest()[:6], trace_id=f"score:{dataset}:{method}")
        # Scores inherit the identity of the features they came from: an HP-p rung
        # must read its own p's features and write scores that cannot be confused
        # with another rung's (all four rungs shared one path before).
        _fc = self.cfg["featurize"]
        pkey = ({"proj": f"p{proj_dim if proj_dim is not None else _fc['proj_dim']}"
                         f"s{proj_seed if proj_seed is not None else _fc['proj_seed']}"}
                if (proj_dim is not None or proj_seed is not None) else {})
        if self.store.exists(K.SCORES, spec, **pkey) and not rescore:
            log.info("scores exist, skipping (use rescore=True to recompute)")
            return {"skipped": True}

        m = METHODS.get(method)
        if query_type == "inject" and getattr(m, "needs_repeats", False):
            raise ValueError("repeat/shrinkage methods have no calibrated gamma on the inject track")
        feat = m.feat_method
        if feat is None:
            raise ValueError(
                f"method '{method}' has no projected-feature recipe; it is scored by "
                f"its own entry point (`balds ekfac score --method {method}`), not `balds score`")
        # Journey-TRAK is the one method whose two sides come from different
        # recipes: training samples get ordinary gradient features, queries get
        # gradients along their own generation trajectory. Everything else reads
        # one recipe on both sides.
        qfeat_name = getattr(m, "query_feat_method", None) or feat
        import torch
        train_feat = self.store.load(K.TRAIN_FEATURES, spec, feat=feat, **pkey)
        qfeat = self.store.load(K.QUERY_FEATURES, spec, feat=qfeat_name, **pkey)
        if not isinstance(qfeat, torch.Tensor):
            qfeat = torch.as_tensor(qfeat)
        e_n = self.store.load(K.ERROR_WEIGHT, spec, feat="das") if m.needs_error_weight else None   # projection-independent: shared by every HP-p rung
        repeats = None
        gamma = None
        if getattr(m, "needs_repeats", False):
            # sigma-hat is projection-dependent, so the repeats must come from
            # the SAME rung as the features being calibrated (hence **pkey).
            repeats = self.store.load(K.REPEAT_FEATURES, spec, feat=feat, **pkey)
            # γ is resolved per (dataset, method) for THIS call and passed down;
            # it is never written back onto the registry singleton (an AB-5
            # driver loops methods x datasets in one process — inherited state
            # would carry the previous dataset's knee, invisibly).
            gamma = resolve_gamma(self.cfg, dataset, method)
            log.info("shrinkage layer: R=%d repeats, gamma=%.4g (dataset %s, method %s)",
                     len(repeats), gamma, dataset, method)
        ck_g = ck_q = None
        ck_steps: tuple = ()
        if getattr(m, "needs_checkpoints", False):
            base = RunSpec(dataset=dataset, seed=seed, process=process,
                           conditional=conditional)
            ck_steps = tuple(attribution_checkpoints(self.store, self.cfg, base))
            if len(ck_steps) < 2:
                raise ValueError(
                    f"{method} averages over training checkpoints but only the final "
                    f"model exists for this identity. Mid-training checkpoints cannot "
                    f"be backfilled — the model must be retrained with "
                    f"train.checkpoint_fracs set.")
            ck_g, ck_q = [], []
            for st in ck_steps:
                skey = {**pkey, **({"step": st} if st is not None else {})}
                ck_g.append(self.store.load(K.TRAIN_FEATURES, spec, feat=feat, **skey))
                qf = self.store.load(K.QUERY_FEATURES, spec, feat=feat, **skey)
                ck_q.append(qf if isinstance(qf, torch.Tensor) else torch.as_tensor(qf))
            log.info("tracin family: averaging over %d checkpoints %s",
                     len(ck_steps), [s if s is not None else "final" for s in ck_steps])
        if query_type == "inject":
            from balds.evaluation.injection import select_lambda_inject
            from .common import _load_ds

            selector_name = ((self.cfg["score"].get("lambda_selector_by_track") or {})
                             .get("inject"))
            if selector_name != "inject_val":
                raise ValueError("inject scoring requires score.lambda_selector_by_track.inject=inject_val")
            lam_sweep = [float(x) for x in self.cfg["score"]["lambda_sweep"]]
            feats = FeatureSet(grads=train_feat, error=e_n, feat_method=feat,
                               ckpt_grads=ck_g, ckpt_query=ck_q, ckpt_steps=ck_steps)
            scores_by_lambda = {
                lam: m.score(feats, qfeat, lam, device=self.device) for lam in lam_sweep
            }
            queries = self.store.load(K.INJECT_QUERIES, spec)
            inject_meta = self.store.load(K.INJECT_META, RunSpec(dataset=dataset))
            train_ds, _ = _load_ds(self.cfg, dataset)
            default_k = inject_meta.get("per_class", inject_meta.get("foreign_per_style"))
            k = int(self.cfg["inject"].get("k", default_k))
            choice = select_lambda_inject(
                scores_by_lambda, lam_sweep, queries["split"], queries["host_labels"],
                train_ds.labels, inject_meta, k=k)
            best = scores_by_lambda[choice.best_lam]
            oracle_test_lam = max(
                lam_sweep, key=lambda lam: choice.per_lambda_ap_test[float(lam)])
            self.store.save(K.SCORES, spec, best, **pkey)
            self.store.save(K.SCORES_META, spec, {
                "method": method, "best_lam": float(choice.best_lam),
                "selector": "inject_val", "feat": feat, "proj": pkey.get("proj"),
                "per_lambda_ap_val": {f"{lam:g}": float(value)
                                      for lam, value in choice.per_lambda_ap_val.items()},
                "per_lambda_ap_test": {f"{lam:g}": float(value)
                                       for lam, value in choice.per_lambda_ap_test.items()},
                "oracle_test_lam": float(oracle_test_lam),
                "oracle_test_ap": float(choice.per_lambda_ap_test[float(oracle_test_lam)]),
                "queries_sha256": queries["queries_sha256"], "k": k,
                "lambda_sweep": lam_sweep,
            }, **pkey)
            log.info("method=%s lambda=%.3g val AP=%.4f (inject_val)",
                     method, choice.best_lam, choice.best_ap_val)
            return {"best_lam": choice.best_lam, "best_ap_val": choice.best_ap_val,
                    "shape": list(best.shape), "selector": "inject_val"}
        from .config import lds_rows
        masks, gt = lds_rows(self.cfg, dataset, self.store.load(K.SUBSET_MASKS, spec),
                             self.store.load(K.GT_MATRIX, spec))
        subset_idx = list(range(min(gt.shape[0], len(masks))))

        lam_sweep = [float(x) for x in self.cfg["score"]["lambda_sweep"]]
        feats = FeatureSet(grads=train_feat, error=e_n, feat_method=feat, repeats=repeats,
                           ckpt_grads=ck_g, ckpt_query=ck_q, ckpt_steps=ck_steps)
        selector = SELECTORS.get(self.cfg["score"].get("lambda_selector", "oracle"))
        extra: dict = {}
        if gamma is not None:
            # Shrinkage methods: raw kernel + σ̂ per λ, shaped by the layer, and
            # λ selected on the SHAPED matrices (never on the raw ones) — the
            # same ordering the curvature path follows, so AB-5 compares like
            # with like. Same arithmetic as the old in-method code, byte for byte.
            raw_by_lam, sig_by_lam = {}, {}
            for lam in lam_sweep:
                raw_by_lam[lam], sig_by_lam[lam] = m.raw_and_sigma(
                    feats, qfeat, lam, device=self.device)
            scores_by_lambda, choice = select_after_shrinkage(
                ShrinkageLayer(gamma), raw_by_lam, sig_by_lam, lam_sweep,
                lambda shaped, keys: selector.select(shaped, keys, gt, masks, subset_idx))
            extra = {"gamma": gamma,
                     **snr_stats(raw_by_lam[choice.best_lam], sig_by_lam[choice.best_lam])}
        else:
            scores_by_lambda = {
                lam: m.score(feats, qfeat, lam, device=self.device) for lam in lam_sweep
            }
            choice = selector.select(scores_by_lambda, lam_sweep, gt, masks, subset_idx)
        best = scores_by_lambda[choice.best_lam]
        self.store.save(K.SCORES, spec, best, **pkey)
        # provenance sidecar: the λ (and γ) that produced scores.npy were never
        # persisted before, so a derived artifact (AN-2 removal sets) could not
        # be traced back to them
        self.store.save(K.SCORES_META, spec, {
            "method": method, "best_lam": float(choice.best_lam),
            "best_lds": float(choice.best_lds), "selector": selector.__class__.__name__,
            "gamma": (float(gamma) if gamma is not None else None),
            "lambda_sweep": lam_sweep, "feat": feat, "proj": pkey.get("proj"),
            # the whole λ–LDS curve, not just its argmax: HP-λ (附录 G) and the
            # boundary-argmax rule ("边界 argmax 不是知道了最优，是还没扫到") both
            # need every rung, and recovering them meant re-scoring every cell.
            "per_lambda_lds": {f"{k:g}": float(v)
                               for k, v in choice.per_lambda_lds.items()}}, **pkey)
        log.info("method=%s lambda=%.3g lds=%.4f", method, choice.best_lam, choice.best_lds)
        return {"best_lam": choice.best_lam, "best_lds": choice.best_lds,
                "shape": list(best.shape), "selector": selector.__class__.__name__,
                **extra}


class EvaluateUseCase:
    """Evaluate saved scores: mean LDS + bootstrap CI over per-query correlations."""

    def __init__(self, store, cfg) -> None:
        self.store, self.cfg = store, cfg

    def run(self, method: str, dataset: str, seed: int, *, query_type: str = "gen",
            process: str = "cfm", conditional: bool = True,
            proj_dim: Optional[int] = None, proj_seed: Optional[int] = None,
            config_tag: Optional[str] = None, kappas=None) -> dict:
        if query_type == "inject":
            raise ValueError("inject has no LDS ground truth; use `balds inject evaluate`")
        from .config import resolve_paper_method
        method = resolve_paper_method(self.cfg, method)
        spec = RunSpec(dataset=dataset, seed=seed, method=method, query_type=query_type,
                       process=process, conditional=conditional)
        set_context(run_id=spec.digest()[:6], trace_id=f"eval:{dataset}:{method}")
        # Scores inherit the identity of the features they came from: an HP-p rung
        # must read its own p's features and write scores that cannot be confused
        # with another rung's (all four rungs shared one path before).
        _fc = self.cfg["featurize"]
        pkey = ({"proj": f"p{proj_dim if proj_dim is not None else _fc['proj_dim']}"
                         f"s{proj_seed if proj_seed is not None else _fc['proj_seed']}"}
                if (proj_dim is not None or proj_seed is not None) else {})
        if config_tag is not None:
            pkey["config"] = config_tag
        scores = self.store.load(K.SCORES, spec, **pkey)
        from .config import lds_rows
        masks, gt = lds_rows(self.cfg, dataset, self.store.load(K.SUBSET_MASKS, spec),
                             self.store.load(K.GT_MATRIX, spec))
        subset_idx = list(range(min(gt.shape[0], len(masks))))
        pred = predicted_influence(scores, masks, subset_idx)
        per_query, mean_lds = compute_lds(gt[:len(subset_idx)], pred)
        mean, lo, hi = bootstrap_ci(per_query)
        log.info("method=%s LDS=%.4f CI=[%.4f, %.4f]", method, mean_lds, lo, hi)
        result = {"mean_lds": mean_lds, "ci_lo": lo, "ci_hi": hi,
                  "n_queries": len(per_query)}
        if kappas is not None:
            curve = {}
            for kappa in kappas:
                head = support_head_scores(scores, float(kappa))
                head_pred = predicted_influence(head, masks, subset_idx)
                head_per, head_mean = compute_lds(gt[:len(subset_idx)], head_pred)
                _, head_lo, head_hi = bootstrap_ci(head_per)
                curve[f"{float(kappa):g}"] = {
                    "mean_lds": head_mean, "ci_lo": head_lo, "ci_hi": head_hi,
                    "head_size": int(np.ceil(float(kappa) * scores.shape[0])),
                }
            result["lds_by_kappa"] = curve
        return result


