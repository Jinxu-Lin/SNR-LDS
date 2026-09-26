"""Application entry points for the three E4 direct baselines."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from balds.schema.artifact import ArtifactKind as K
from balds.schema.logging import get_logger, set_context
from balds.schema.registry import PROCESSES
from balds.schema.runspec import RunSpec
from balds.data.base import ImageDataset, balanced_query_indices
from balds.attribution.abu import (
    DiagonalNaturalGradient,
    EigenKroneckerNaturalGradient,
    apply_natural_update,
    empirical_fisher_diagonal,
    finite_loss_difference,
    measure_losses,
    query_loss_gradient,
    restored_model,
)
from balds.attribution.nda import nda_scores, resolve_recipes
from balds.attribution.parameter_weighting import fit_weights, fixed_kernel
from balds.attribution.featurize import GroupedGradFeaturizer
from balds.attribution.ekfac import EkfacFactors, EkfacFitter, kfac_target_modules

from .config import resolve_Q
from .features import FeaturizeUseCase
from .common import _load_ds

log = get_logger("balds.baselines")
IMPLEMENTATION_VERSION = "code_baselines_closeout/v2"


def _cfg(cfg: dict, name: str) -> dict:
    value = dict(cfg.get(name, {}) or {})
    if not value:
        raise ValueError(f"missing configuration block {name!r}")
    return value


def _method_spec(method: str, dataset: str, seed: int, process: str,
                 conditional: bool, query_type: str = "gen") -> RunSpec:
    return RunSpec(dataset=dataset, seed=seed, process=process, conditional=conditional,
                   query_type=query_type, method=method)


def _provenance(store, spec: RunSpec) -> dict:
    return {
        "implementation": IMPLEMENTATION_VERSION,
        "code_version": getattr(store, "code_version", "unknown"),
        "identity": {
            "dataset": spec.dataset, "seed": spec.seed, "process": spec.process,
            "conditional": spec.conditional, "query_type": spec.query_type,
        },
    }


def _platform(fz: FeaturizeUseCase, dataset: str, seed: int, process: str,
              conditional: bool):
    base = RunSpec(dataset=dataset, seed=seed, process=process, conditional=conditional)
    model = fz._load_model(base, {})
    train, test = fz._train_test(dataset)
    if fz._is_latent(dataset):
        from .latent import platform_process
        generative_process, _ = platform_process(fz.cfg, dataset, device=fz.device)
    else:
        generative_process = PROCESSES.get(process)
    return base, model, train, test, generative_process


def _selected_parameters(model, config: dict) -> list[str]:
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    domain = config.get("parameter_domain", "trainable")
    if domain == "ekfac":
        supported = {f"{layer}.{field}".lstrip(".")
                     for layer, module in kfac_target_modules(model).items()
                     for field, p in module.named_parameters(recurse=False) if p.requires_grad}
        names = [name for name in names if name in supported]
    elif domain != "trainable":
        raise ValueError("parameter_domain must be trainable or ekfac")
    patterns = [str(value) for value in config.get("parameter_patterns", [])]
    if patterns:
        names = [name for name in names if any(pattern in name for pattern in patterns)]
    if not names:
        raise ValueError("the configured parameter adapter selected no trainable parameters")
    return names


def _validate_scores(scores, n: int, q: int) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float32)
    if array.shape != (n, q):
        raise ValueError(f"score shape {array.shape} != expected {(n, q)}")
    if not np.isfinite(array).all():
        raise FloatingPointError("scores contain NaN or infinity")
    return array


def _query_ids(store, cfg: dict, base: RunSpec, test, query_type: str, count: int) -> list[int]:
    if query_type == "val":
        values = balanced_query_indices(test.labels, resolve_Q(cfg, base.dataset))
    elif query_type == "gen":
        generation = store.load(K.GENERATION, base)
        values = generation.get("query_ids", list(range(count)))
    else:
        values = list(range(count))
    values = [int(value) for value in values]
    if len(values) != count or len(set(values)) != count:
        raise ValueError("query identity list must be unique and match the score columns")
    return values


def _require_same_config(blob: dict, config: dict, *, artifact: str) -> None:
    if blob.get("config") != config:
        raise ValueError(f"{artifact} already exists under this config tag with different "
                         "scientific settings; choose a new --config-tag")


class ParameterWeightingUseCase:
    METHOD = "dtrak_param_weighted_T100"

    def __init__(self, store, cfg, *, device: str = "cuda:0") -> None:
        self.store, self.cfg, self.device = store, cfg, device

    def _featurizer(self, model, process, dataset: str, config: dict):
        fcfg = self.cfg["featurize"]
        return GroupedGradFeaturizer(
            model, process, proj_dim=int(config.get("proj_dim", fcfg["proj_dim"])),
            proj_seed=int(config.get("proj_seed", fcfg["proj_seed"])), T=100,
            loss_type="msl2", batch_size=int(config.get("batch_size", 1)),
            device=self.device, projection=str(config.get("projection", "auto")),
        )

    def _train_features(self, base: RunSpec, config: dict) -> torch.Tensor:
        """Load only the unsegmented production D-TRAK feature identity.

        The legacy artifact has no gradient-recipe sidecar, so dimensions and
        projection seed are pinned to the active production featurizer config;
        an HP-p rung must not be silently read from the default path.
        """
        feat = str(config.get("feature", "dtrak_T100"))
        if feat != "dtrak_T100":
            raise ValueError("parameter-weighted D-TRAK requires feature='dtrak_T100'")
        fcfg = self.cfg["featurize"]
        dim = int(config.get("proj_dim", fcfg["proj_dim"]))
        pseed = int(config.get("proj_seed", fcfg["proj_seed"]))
        if dim != int(fcfg["proj_dim"]) or pseed != int(fcfg["proj_seed"]):
            raise ValueError("the reused unsegmented dtrak_T100 training features require "
                             "parameter_weighting proj_dim/proj_seed to match featurize defaults")
        values = torch.as_tensor(self.store.load(K.TRAIN_FEATURES, base, feat=feat))
        if values.ndim != 2 or values.shape[1] != dim:
            raise ValueError(f"dtrak_T100 training feature shape {tuple(values.shape)} is "
                             f"incompatible with projection dimension {dim}")
        return values

    def _kernel(self, base, config):
        features = self._train_features(base, config).to(self.device, dtype=torch.float32)
        inverse = fixed_kernel(features, float(config["ridge"]),
                               normalize=str(config.get("kernel_normalize", "mean_abs")),
                               device=self.device)
        return features @ inverse

    @staticmethod
    def _grouped_chunks(featurizer, queries, config, seed):
        chunk = int(config.get("contribution_query_chunk", 16))
        if chunk <= 0:
            raise ValueError("contribution_query_chunk must be positive")
        pending, count, q0 = [], 0, 0
        for batch in featurizer.iter_grouped(queries, seed=int(config.get("query_seed", seed))):
            pending.append(batch)
            count += len(batch)
            if count >= chunk:
                yield q0, torch.cat(pending)
                q0 += count
                pending, count = [], 0
        if pending:
            yield q0, torch.cat(pending)

    def prepare(self, dataset: str, seed: int, *, config_tag: str,
                process: str = "cfm", conditional: bool = True,
                force: bool = False) -> dict:
        config = _cfg(self.cfg, "parameter_weighting")
        spec = _method_spec(self.METHOD, dataset, seed, process, conditional)
        key = {"config": config_tag}
        if self.store.exists(K.PARAM_CONTRIBUTIONS, spec, **key) and not force:
            _require_same_config(self.store.load(K.PARAM_CONTRIBUTIONS, spec, **key),
                                 config, artifact="parameter contributions")
            return {"skipped": True, "config_tag": config_tag}
        set_context(run_id=spec.digest()[:6], trace_id=f"parameter-weighting:prepare:{dataset}")
        helper = FeaturizeUseCase(self.store, self.cfg, device=self.device)
        base, model, _, _, proc = _platform(helper, dataset, seed, process, conditional)
        feat = str(config.get("feature", "dtrak_T100"))
        train_kernel = self._kernel(base, config)
        generation_name = str(config.get("learning_generation", "weight_learning_samples"))
        if generation_name == "samples":
            raise ValueError("weight-learning queries must use an independent named generation, "
                             "not the main-table samples artifact")
        learning = self.store.load(K.GENERATION, base, name=generation_name)
        if self.store.exists(K.GENERATION, base):
            evaluation = self.store.load(K.GENERATION, base)
            if learning.get("gen_seed") is not None and \
                    learning.get("gen_seed") == evaluation.get("gen_seed"):
                raise ValueError("weight-learning and main-table generations share gen_seed; "
                                 "generate an independent learning query set")
        count = int(config.get("learning_query_count", 500))
        ids = [int(v) for v in config.get("learning_query_ids", [])]
        if not ids:
            ids = list(range(count))
        if count <= 0 or len(ids) != count or len(set(ids)) != len(ids) or min(ids, default=0) < 0:
            raise ValueError("learning_query_ids must be unique, non-negative and match learning_query_count")
        samples = torch.as_tensor(learning["samples"])
        labels = torch.as_tensor(learning["labels"])
        if max(ids, default=-1) >= len(samples):
            raise ValueError(f"learning generation has {len(samples)} rows but IDs require {max(ids)}")
        queries = ImageDataset(samples[ids], labels[ids])
        timer = time.perf_counter()
        featurizer = self._featurizer(model, proc, dataset, config)
        blocks, contribution_seconds = [], 0.0
        for q0, grouped in self._grouped_chunks(featurizer, queries, config, seed):
            started = time.perf_counter()
            contributions = torch.einsum("np,qgp->nqg", train_kernel,
                                         grouped.to(self.device)).cpu()
            q1 = q0 + len(grouped)
            self.store.save(K.PARAM_CONTRIBUTIONS, spec,
                            {"contributions": contributions, "config": config,
                             "learning_query_ids": ids[q0:q1]}, q0=q0, q1=q1, **key)
            blocks.append([q0, q1])
            contribution_seconds += time.perf_counter() - started
        gradient_seconds = time.perf_counter() - timer - contribution_seconds
        group_names = featurizer.group_names
        payload = {
            "format": "query_blocks_v1", "blocks": blocks,
            "shape": [len(train_kernel), count, len(group_names)], "group_names": group_names,
            "config": config,
            "learning_query_ids": ids, "learning_generation": generation_name,
            "learning_generation_meta": {
                key: learning[key] for key in ("Q", "gen_seed", "ode_steps", "steps",
                                                "timing_seconds") if key in learning
            },
            "ridge": float(config["ridge"]), "feature": feat,
            "projection": {"dim": featurizer.proj_dim,
                           "seed": int(config.get("proj_seed", self.cfg["featurize"]["proj_seed"])),
                           "backend": featurizer.projector_kind},
            "timing_seconds": {"query_gradient_projection": gradient_seconds,
                               "contribution_multiply": contribution_seconds},
        }
        self.store.save(K.PARAM_CONTRIBUTIONS, spec, payload, **key)
        return {"shape": payload["shape"], "groups": len(group_names),
                "config_tag": config_tag, "timing_seconds": payload["timing_seconds"]}

    def fit(self, dataset: str, seed: int, *, config_tag: str,
            process: str = "cfm", conditional: bool = True, force: bool = False) -> dict:
        config = _cfg(self.cfg, "parameter_weighting")
        spec = _method_spec(self.METHOD, dataset, seed, process, conditional)
        key = {"config": config_tag}
        if self.store.exists(K.PARAM_WEIGHTS, spec, **key) and not force:
            _require_same_config(self.store.load(K.PARAM_WEIGHTS, spec, **key),
                                 config, artifact="parameter weights")
            return {"skipped": True, "config_tag": config_tag}
        source = self.store.load(K.PARAM_CONTRIBUTIONS, spec, **key)
        _require_same_config(source, config, artifact="parameter contributions")
        def batches():
            cursor = 0
            for q0, q1 in source["blocks"]:
                if q0 != cursor or q1 <= q0:
                    raise ValueError("contribution blocks have gaps/overlap")
                block = self.store.load(K.PARAM_CONTRIBUTIONS, spec, q0=q0, q1=q1, **key)
                _require_same_config(block, config, artifact="contribution block")
                if (block["learning_query_ids"] != source["learning_query_ids"][q0:q1]
                        or block["contributions"].shape[1] != q1 - q0):
                    raise ValueError("contribution block query identity mismatch")
                yield block["contributions"]
                cursor = q1
        contributions = batches if "blocks" in source else source["contributions"]
        timer = time.perf_counter()
        result = fit_weights(
            contributions, shape=source.get("shape"), epochs=int(config.get("epochs", 10)),
            lr=float(config.get("lr", 0.01)), top_k=int(config["top_k"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
            scheduler=str(config.get("scheduler", "cosine")),
            seed=int(config.get("initialization_seed", 0)), device=self.device)
        seconds = time.perf_counter() - timer
        payload = {"weights": torch.from_numpy(result.weights),
                   "raw_weights": torch.from_numpy(result.raw_weights),
                   "losses": list(result.losses), "stopped_epoch": result.stopped_epoch,
                   "group_names": source["group_names"], "timing_seconds": seconds,
                   "config": config,
                   "regularization": "AdamW raw-logit weight_decay",
                   "learning_query_ids": source["learning_query_ids"]}
        self.store.save(K.PARAM_WEIGHTS, spec, payload, **key)
        return {"groups": len(result.weights), "stopped_epoch": result.stopped_epoch,
                "timing_seconds": seconds, "config_tag": config_tag}

    def score(self, dataset: str, seed: int, *, config_tag: str, query_type: str = "gen",
              process: str = "cfm", conditional: bool = True,
              rescore: bool = False) -> dict:
        config = _cfg(self.cfg, "parameter_weighting")
        spec = _method_spec(self.METHOD, dataset, seed, process, conditional, query_type)
        key = {"config": config_tag}
        if self.store.exists(K.SCORES, spec, **key) and not rescore:
            _require_same_config(self.store.load(K.SCORES_META, spec, **key),
                                 config, artifact="parameter-weighted scores")
            return {"skipped": True, "config_tag": config_tag}
        helper = FeaturizeUseCase(self.store, self.cfg, device=self.device)
        base, model, train, test, proc = _platform(helper, dataset, seed, process, conditional)
        query_ds = helper._query_dataset(base, test, query_type)
        query_ids = _query_ids(self.store, self.cfg, base, test, query_type, len(query_ds))
        weights_blob = self.store.load(K.PARAM_WEIGHTS, spec.with_(query_type="gen"), **key)
        _require_same_config(weights_blob, config, artifact="parameter weights")
        featurizer = self._featurizer(model, proc, dataset, config)
        if featurizer.group_names != list(weights_blob["group_names"]):
            raise ValueError("current model parameter groups do not match the fitted weights")
        timer = time.perf_counter()
        train_kernel = self._kernel(base, config)
        weights = weights_blob["weights"].to(self.device)
        scores = np.empty((len(train), len(query_ds)), dtype=np.float32)
        score_seconds = 0.0
        for q0, grouped in self._grouped_chunks(featurizer, query_ds, config, seed):
            started = time.perf_counter()
            # Weight query groups first; scoring never allocates N x Q x G.
            query = torch.einsum("qgp,g->qp", grouped.to(self.device), weights)
            scores[:, q0:q0 + len(grouped)] = (train_kernel @ query.T).cpu().numpy()
            score_seconds += time.perf_counter() - started
        gradient_seconds = time.perf_counter() - timer - score_seconds
        scores = _validate_scores(scores, len(train), len(query_ds))
        self.store.save(K.SCORES, spec, scores, **key)
        self.store.save(K.SCORES_META, spec, {
            "method": self.METHOD, "config_tag": config_tag, "config": config,
            **_provenance(self.store, spec),
            "group_names": featurizer.group_names,
            "learning_query_ids": weights_blob["learning_query_ids"],
            "train_ids": list(range(len(train))), "query_ids": query_ids,
            "shape": list(scores.shape), "support_direction": "larger_is_more_supportive",
            "timing_seconds": {"evaluation_query_gradient_projection": gradient_seconds,
                               "evaluation_contribution_and_weighting": score_seconds},
            "source": "arXiv:2506.05647v4; project global-projection adaptation",
        }, **key)
        return {"shape": list(scores.shape),
                "timing_seconds": {"evaluation_query_gradient_projection": gradient_seconds,
                                   "evaluation_contribution_and_weighting": score_seconds},
                "config_tag": config_tag}


class AbuUseCase:
    METHOD = "abu_plus"

    def __init__(self, store, cfg, *, device: str = "cuda:0") -> None:
        self.store, self.cfg, self.device = store, cfg, device

    @staticmethod
    def _kind(config):
        kind = str(config.get("preconditioner", "diagonal_empirical_fisher"))
        if kind not in {"diagonal_empirical_fisher", "ekfac"}:
            raise ValueError("AbU preconditioner must be diagonal_empirical_fisher or ekfac")
        if config.get("flip_mode", "none") not in {"none", "max"}:
            raise ValueError("flip_mode must be none or max")
        if kind == "ekfac" and float(config.get("fisher_gain", 1.0)) != 1.0:
            raise ValueError("EK-FAC gain is derived from curvature metadata; fisher_gain must be 1")
        return kind

    def _flipped(self, helper, dataset, train, config):
        if config.get("flip_mode", "none") != "max" or not helper._is_latent(dataset):
            return None
        from .latent import latent_train_pair
        pair = latent_train_pair(self.store, dataset)
        if pair.images_flipped.shape != train.images.shape or not torch.equal(pair.labels, train.labels):
            raise ValueError("latent pair cache does not align with the training set")
        return pair.images_flipped

    def _ekfac(self, base, model, train, proc, parameters, config, flipped):
        source = str(config.get("curvature_source", "fit"))
        if source == "ekfac_factors":
            factors = EkfacFactors.from_state_dict(self.store.load(K.EKFAC_FACTORS, base))
            for field, expected in {"dataset": base.dataset, "seed": base.seed,
                                    "process": base.process, "conditional": base.conditional,
                                    "train_samples": len(train)}.items():
                if factors.meta.get(field) != expected:
                    raise ValueError(f"empirical curvature identity mismatch: {field}")
        elif source == "fit":
            fitter = EkfacFitter(model, proc, device=self.device, curvature_kind="empirical_fisher")
            supported = {f"{layer}.{field}".lstrip(".")
                         for layer, module in fitter.modules.items()
                         for field, p in module.named_parameters(recurse=False) if p.requires_grad}
            if set(parameters) - supported:
                raise ValueError("EK-FAC parameter domain includes unsupported parameters; "
                                 "choose parameter_domain=ekfac or explicit supported patterns")
            selected_layers = {name.rpartition(".")[0] for name in parameters}
            fitter.modules = {name: module for name, module in fitter.modules.items() if name in selected_layers}
            fit_epochs, eig_epochs = int(config["ekfac_fit_epochs"]), int(config["ekfac_eig_epochs"])
            batch_size = int(config["ekfac_batch_size"])
            if min(fit_epochs, eig_epochs, batch_size) <= 0:
                raise ValueError("EK-FAC epochs and batch size must be positive")
            kwargs = dict(batch_size=batch_size, seed=int(config.get("fisher_seed", base.seed)),
                          hflip=config.get("flip_mode", "none") == "max", flipped=flipped)
            labels = train.labels if base.conditional else None
            A, B, draws = fitter.fit_factor_pass(train.images, labels, epochs=fit_epochs, **kwargs)
            bases = fitter.eigenbases(A, B)
            del A, B
            layers = fitter.fit_eigenvalue_pass(train.images, labels, bases, epochs=eig_epochs, **kwargs)
            factors = EkfacFactors(layers, meta={
                "layer_order": list(layers), "curvature_kind": "empirical_fisher",
                "normalization": "mean_per_datum_gradient_outer_product",
                "parameter_names": parameters, "train_samples": len(train),
                "n_factor_draws": draws, "n_eigenvalue_draws": eig_epochs * len(train),
                "fit_epochs": fit_epochs, "eig_epochs": eig_epochs, "batch_size": batch_size,
                "fisher_seed": kwargs["seed"], "hflip": kwargs["hflip"],
                "hflip_source": "latent_pair_cache" if flipped is not None else "pixel_or_none",
                "bias_convention": "separate_weight_bias_blocks",
                "dataset": base.dataset, "seed": base.seed, "process": base.process,
                "conditional": base.conditional, "output_loss": "per_datum_mean_mse",
            })
        else:
            raise ValueError("curvature_source must be fit or ekfac_factors")
        self._ekfac_preconditioner(model, factors, parameters, config)
        return factors

    @staticmethod
    def _ekfac_preconditioner(model, factors, parameters, config):
        preconditioner = EigenKroneckerNaturalGradient.from_factors(
            factors, parameters, damping=float(config["damping"]))
        named = dict(model.named_parameters())
        for name, (left, right, spectrum) in preconditioner.blocks.items():
            if name not in named or not named[name].requires_grad:
                raise ValueError(f"curvature parameter absent/frozen: {name}")
            shape = (named[name].shape[0], named[name].numel() // named[name].shape[0])
            if spectrum.shape != shape or left.shape != (shape[0], shape[0]) or right.shape != (shape[1], shape[1]):
                raise ValueError(f"curvature shape mismatch: {name}")
        return preconditioner

    def prepare(self, dataset: str, seed: int, *, config_tag: str,
                process: str = "cfm", conditional: bool = True,
                force: bool = False) -> dict:
        config = _cfg(self.cfg, "abu")
        kind = self._kind(config)
        spec = _method_spec(self.METHOD, dataset, seed, process, conditional)
        key = {"config": config_tag}
        if self.store.exists(K.ABU_PREP, spec, **key) and not force:
            _require_same_config(self.store.load(K.ABU_PREP, spec, **key),
                                 config, artifact="AbU+ prepare")
            return {"skipped": True, "config_tag": config_tag}
        helper = FeaturizeUseCase(self.store, self.cfg, device=self.device)
        base, model, train, _, proc = _platform(helper, dataset, seed, process, conditional)
        parameters = _selected_parameters(model, config)
        flip = str(config.get("flip_mode", "none")) == "max"
        flipped = self._flipped(helper, dataset, train, config)
        timer = time.perf_counter()
        if kind == "ekfac":
            factors = self._ekfac(base, model, train, proc, parameters, config, flipped)
            curvature = {"fisher_ekfac": factors.state_dict()}
        else:
            model.eval()
            curvature = {"fisher_diagonal": empirical_fisher_diagonal(
                model, proc, train, parameter_names=parameters,
                samples=int(config["fisher_samples"]), mc=int(config["fisher_mc"]),
                seed=int(config.get("fisher_seed", seed)), device=self.device)}
        fisher_seconds = time.perf_counter() - timer
        model.eval()
        timer = time.perf_counter()
        baseline = measure_losses(model, proc, train, mc=int(config["measurement_mc"]),
                                  seed=int(config.get("measurement_seed", seed)),
                                  device=self.device, flip=flip, flipped_images=flipped)
        baseline_seconds = time.perf_counter() - timer
        payload = {"baseline_losses": baseline, **curvature, "preconditioner": kind,
                   "parameter_names": parameters, "config": config,
                   "timing_seconds": {"baseline_loss": baseline_seconds,
                                      kind: fisher_seconds}}
        self.store.save(K.ABU_PREP, spec, payload, **key)
        return {"train_samples": len(train), "parameters": len(parameters),
                "timing_seconds": payload["timing_seconds"], "config_tag": config_tag}

    def score(self, dataset: str, seed: int, *, config_tag: str, query_type: str = "gen",
              process: str = "cfm", conditional: bool = True,
              rescore: bool = False) -> dict:
        config = _cfg(self.cfg, "abu")
        kind = self._kind(config)
        spec = _method_spec(self.METHOD, dataset, seed, process, conditional, query_type)
        key = {"config": config_tag}
        if self.store.exists(K.SCORES, spec, **key) and not rescore:
            _require_same_config(self.store.load(K.SCORES_META, spec, **key),
                                 config, artifact="AbU+ scores")
            return {"skipped": True, "config_tag": config_tag}
        helper = FeaturizeUseCase(self.store, self.cfg, device=self.device)
        base, model, train, test, proc = _platform(helper, dataset, seed, process, conditional)
        queries = helper._query_dataset(base, test, query_type)
        query_ids = _query_ids(self.store, self.cfg, base, test, query_type, len(queries))
        prep = self.store.load(K.ABU_PREP, spec.with_(query_type="gen"), **key)
        if prep["config"] != config:
            raise ValueError("AbU+ prepare artifact config differs from the active config tag")
        if prep["parameter_names"] != _selected_parameters(model, config):
            raise ValueError("AbU prepare/current parameter domains differ")
        if kind == "ekfac":
            factors = EkfacFactors.from_state_dict(prep["fisher_ekfac"])
            preconditioner = self._ekfac_preconditioner(model, factors, prep["parameter_names"], config)
            curvature_meta = {**factors.meta, "effective_gain": preconditioner.gain,
                              "damping_mode": preconditioner.damping_mode,
                              "layer_damping": preconditioner.layer_damping}
        else:
            preconditioner = DiagonalNaturalGradient(
                prep["fisher_diagonal"], damping=float(config["damping"]),
                gain=float(config.get("fisher_gain", 1.0)))
            curvature_meta = {"effective_gain": preconditioner.gain, "damping_mode": "absolute"}
        flipped = self._flipped(helper, dataset, train, config)
        columns, query_timings, loss_timings = [], [], []
        flip = str(config.get("flip_mode", "none"))
        model.eval()
        for query_id in range(len(queries)):
            image, label = queries[query_id]
            with restored_model(model, prep["parameter_names"]):
                timer = time.perf_counter()
                gradients = query_loss_gradient(
                    model, proc, torch.as_tensor(image, device=self.device), int(label),
                    parameter_names=list(prep["parameter_names"]),
                    mc=int(config["query_mc"]), seed=int(config.get("query_seed", seed)),
                    query_id=query_ids[query_id], allow_unused=kind == "ekfac")
                apply_natural_update(model, gradients, preconditioner,
                                     step_size=float(config["step_size"]))
                query_timings.append(time.perf_counter() - timer)
                timer = time.perf_counter()
                updated = measure_losses(
                    model, proc, train, mc=int(config["measurement_mc"]),
                    seed=int(config.get("measurement_seed", seed)), device=self.device,
                    flip=flip == "max", flipped_images=flipped)
                loss_timings.append(time.perf_counter() - timer)
                columns.append(finite_loss_difference(
                    prep["baseline_losses"], updated, flip_mode=flip).cpu())
        scores = _validate_scores(torch.stack(columns, dim=1).numpy(), len(train), len(queries))
        self.store.save(K.SCORES, spec, scores, **key)
        self.store.save(K.SCORES_META, spec, {
            "method": self.METHOD, "config_tag": config_tag, "config": config,
            **_provenance(self.store, spec),
            "preconditioner": kind, "curvature": curvature_meta,
            "parameter_names": prep["parameter_names"],
            "train_ids": list(range(len(train))), "query_ids": query_ids,
            "shape": list(scores.shape), "support_direction": "larger_is_more_supportive",
            "timing_seconds": {"query_gradient_and_update": sum(query_timings),
                               "updated_training_loss": sum(loss_timings)},
            "timing_seconds_per_query": {
                "query_gradient_and_update": query_timings,
                "updated_training_loss": loss_timings,
            },
            "source": "FastGDA AbU+ commit a85a6a3538f50d4f5227d318f351e12ae12caa11",
        }, **key)
        return {"shape": list(scores.shape), "config_tag": config_tag,
                "timing_seconds": {"query_gradient_and_update": sum(query_timings),
                                   "updated_training_loss": sum(loss_timings)}}


def _stack_images(dataset, indices=None) -> torch.Tensor:
    indices = list(range(len(dataset))) if indices is None else [int(v) for v in indices]
    images = []
    for index in indices:
        item = dataset[index]
        image = item[0] if isinstance(item, tuple) else dataset.images[index]
        images.append(torch.as_tensor(image))
    return torch.stack(images)


class NdaUseCase:
    METHOD = "nda"

    def __init__(self, store, cfg, *, device: str = "cuda:0") -> None:
        self.store, self.cfg, self.device = store, cfg, device

    def score(self, dataset: str, seed: int, *, config_tag: str, query_type: str = "gen",
              process: str = "cfm", conditional: bool = True,
              rescore: bool = False) -> dict:
        config = _cfg(self.cfg, "nda")
        spec = _method_spec(self.METHOD, dataset, seed, process, conditional, query_type)
        key = {"config": config_tag}
        if self.store.exists(K.SCORES, spec, **key) and not rescore:
            _require_same_config(self.store.load(K.SCORES_META, spec, **key),
                                 config, artifact="NDA scores")
            return {"skipped": True, "config_tag": config_tag}
        base = RunSpec(dataset=dataset, seed=seed, process=process, conditional=conditional)
        timer = time.perf_counter()
        train_ds, test_ds = _load_ds(self.cfg, dataset)
        train = _stack_images(train_ds).to(self.device)
        if query_type == "val":
            indices = balanced_query_indices(test_ds.labels, resolve_Q(self.cfg, dataset))
            queries = _stack_images(test_ds, indices).to(self.device)
            query_ids = indices
        elif query_type == "gen":
            generated = self.store.load(K.GENERATION, base)
            if "images_u8" in generated:
                queries = torch.as_tensor(generated["images_u8"]).float().div(127.5).sub(1.0)
            else:
                queries = torch.as_tensor(generated["samples"])
            queries = queries.to(self.device)
            query_ids = _query_ids(self.store, self.cfg, base, test_ds, "gen", len(queries))
        else:
            raise ValueError("NDA E4 supports gen/val query tracks, not inject")
        preparation_seconds = time.perf_counter() - timer
        from balds.models.ddpm import DDPMSchedule
        schedule = DDPMSchedule(T=int(config.get("schedule_steps", 1000)), device="cpu")
        timesteps = [int(v) for v in config["timesteps"]]
        if any(value < 0 or value >= schedule.T for value in timesteps):
            raise ValueError("NDA timestep outside the diffusion schedule")
        bars = [float(schedule.alpha_bars[value]) for value in timesteps]
        recipes = resolve_recipes(
            len(timesteps), **{k: config[k] for k in (
                "patch_size", "projected_size", "second_patch_size", "second_projected_size",
                "two_scale_alpha", "variant") if k in config},
            recipes=config.get("timestep_recipes"))
        timestep_indices = config.get("timestep_indices") or list(range(len(timesteps)))
        cuda_measure = str(self.device).startswith("cuda") and torch.cuda.is_available()
        if cuda_measure:
            torch.cuda.reset_peak_memory_stats(self.device)
        timer = time.perf_counter()
        scores_t = nda_scores(
            train, queries, alpha_bars=bars, patch_size=int(config["patch_size"]),
            seed=int(config.get("noise_seed", seed)), variant=str(config.get("variant", "single")),
            projected_size=config.get("projected_size"),
            second_patch_size=config.get("second_patch_size"),
            second_projected_size=config.get("second_projected_size"),
            two_scale_alpha=float(config.get("two_scale_alpha", 0.5)),
            timestep_recipes=recipes, query_ids=query_ids, timestep_indices=timestep_indices,
            spatial_topk=config.get("spatial_topk"), train_chunk=config.get("train_chunk"),
            query_patch_chunk=config.get("query_patch_chunk"),
            mask_value=float(config.get("mask_value", 1e3)))
        seconds = time.perf_counter() - timer
        peak_memory = int(torch.cuda.max_memory_allocated(self.device)) if cuda_measure else None
        scores = _validate_scores(scores_t.cpu().numpy(), len(train_ds), len(queries))
        self.store.save(K.SCORES, spec, scores, **key)
        self.store.save(K.SCORES_META, spec, {
            "method": self.METHOD, "config_tag": config_tag, "config": config,
            **_provenance(self.store, spec),
            "train_ids": list(range(len(train_ds))), "query_ids": query_ids,
            "shape": list(scores.shape), "support_direction": "larger_is_more_supportive",
            "alpha_bars": bars,
            "timestep_recipes": recipes, "timesteps": timesteps,
            "timestep_indices": timestep_indices,
            "aggregation": "arithmetic mean over configured timesteps; alpha*first+(1-alpha)*second",
            "image_geometry": list(train.shape[1:]),
            "patch_counts": {
                "query_per_scale": [int(queries.shape[-2] * queries.shape[-1])]
                + ([int(queries.shape[-2] * queries.shape[-1])]
                   if str(config.get("variant", "single")) == "two_scale" else []),
                "train_images": len(train_ds),
            },
            "timing_seconds": {"image_preparation": preparation_seconds,
                               "patch_matching": seconds},
            "peak_cuda_memory_bytes": peak_memory,
            "source": "sail-sg/NDA commit a1d786a6ce3fbe4f64ad9e4e9aba5ef5f6c40059",
            "adaptation": "explicit DDPM noise kernel retained for CFM outputs",
        }, **key)
        return {"shape": list(scores.shape),
                "timing_seconds": {"image_preparation": preparation_seconds,
                                   "patch_matching": seconds},
                "peak_cuda_memory_bytes": peak_memory,
                "config_tag": config_tag}
