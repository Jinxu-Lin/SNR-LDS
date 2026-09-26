"""E3c: independent full train AND query MC repeats, fixed scientific inputs."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import platform
import time

import numpy as np
import torch

from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.data.base import ImageDataset, balanced_query_indices, select_val_queries
from balds.attribution.featurize import GradFeaturizer
from balds.attribution.kernel import trak_inverse
from balds.artifacts.addressing import relpath
from balds.artifacts.e3c import (ReadOnlyInputs, RepeatFiles, atomic, exclusive, input_inventory,
                           load_score, load_tensor, read_json, save_tensor, write_text)
from .config import REPO_ROOT, load_config
from .curvature import _prepare, _train_row
from .common import _FEAT_RECIPES, _load_ds

METHODS = ("fmas_raw", "dtrak_T100")
TRACKS = ("gen", "val")


def resolve_config(path=None):
    """Load the portable R16 recipe, including fixed original query columns."""
    import importlib.resources
    import json
    cfg = (read_json(path) if path is not None else
           json.loads(importlib.resources.files("balds").joinpath("conf/repeatability.json").read_text()))
    for track in TRACKS:
        ids = cfg["selection"][track]
        if len(ids) != 16 or sorted(set(ids)) != ids or not 0 <= min(ids) <= max(ids) < 100:
            raise ValueError("invalid original query positions")
    return cfg


def mc_stream(cfg, repeat, method, role, track="shared"):
    if repeat not in range(cfg["repeats"]) or method not in METHODS or role not in ("train", "query"):
        raise ValueError("invalid MC stream")
    if (role == "train" and track != "shared") or (role == "query" and track not in TRACKS):
        raise ValueError("train is shared; query requires original track")
    rng = cfg["rng"]
    base = rng["base_seed"] + rng["stride"] * (repeat + rng[f"{role}_repeat_offset"])
    return {"base_seed": base,
            "seed": base + rng["method_offsets"][method] + rng["track_offsets"][track],
            "phase": rng["fmas_phases"][role] if method == "fmas_raw" else "native_grid",
            "method": method, "role": role, "track": track}


def score_ids(cfg, track):
    return {"train": list(range(cfg["n_train"])), "query": cfg["selection"][track], "track": track}


def prepare(cfg, input_root, output_root):
    """CPU-only: freeze portable pixels in native order, inventory exact inputs."""
    root, out = Path(input_root), Path(output_root)
    with exclusive(out / "inputs"):
        package_path = out / "package.json"
        if package_path.exists():
            package = read_json(package_path)
            verify_package(cfg, root, package)
            return package
        native = load_config({"storage.data_root": str(root)})
        inputs = ReadOnlyInputs(root)
        spec = RunSpec(dataset=cfg["dataset"], seed=42, process="cfm", conditional=False)
        paths = [relpath(k, spec) for k in (K.CHECKPOINT, K.EKFAC_FACTORS, K.GENERATION, K.SUBSET_MASKS)]
        for track in TRACKS:
            qspec = RunSpec(dataset=cfg["dataset"], seed=42, process="cfm", conditional=False, query_type=track)
            paths.append(relpath(K.GT_MATRIX, qspec))
            gt = inputs.load(K.GT_MATRIX, qspec)
            if gt.shape != (64, 100) or not np.isfinite(gt).all():
                raise ValueError(f"expected existing (64,100) response mean: {track}")
        masks = np.asarray(inputs.load(K.SUBSET_MASKS, spec))
        if masks.ndim != 2 or masks.shape[0] < 64 or masks.shape[1] != 5000 or not np.isin(masks, [0, 1]).all():
            raise ValueError("invalid original masks")
        input_inventory(root, paths)  # fail missing sources BEFORE materializing pixels
        train, test = _load_ds(native, cfg["dataset"])
        if len(train) != 5000 or len(select_val_queries(test, 100)) != 100:
            raise ValueError("dataset identity mismatch")
        gen = inputs.load(K.GENERATION, spec)
        if len(gen["samples"]) < 100:
            raise ValueError("generation has fewer than original 100 queries")
        snapshot = out / "inputs/dataset.pt"
        save_tensor(snapshot, {"train_images": train.images, "train_labels": train.labels,
                               "test_images": test.images, "test_labels": test.labels,
                               "val_positions_in_filtered_test": balanced_query_indices(test.labels, 100),
                               "dataset": cfg["dataset"], "loader": "balds.data.cifar._load_cifar2_5k"})
        atomic(out / "inputs/resolved_config.json", cfg)
        atomic(out / "inputs/query_selection.json", cfg["selection"])
        write_text(out / "inputs/lambda_source.yaml", f"dtrak_lambda: {cfg['dtrak']['lambda']}\n")
        paths += [str(p.relative_to(root)) for p in
                  (snapshot, out / "inputs/resolved_config.json", out / "inputs/query_selection.json",
                   out / "inputs/lambda_source.yaml")]
        package = {"config": cfg, "inputs": input_inventory(root, paths),
                   "dataset_snapshot": str(snapshot.relative_to(root)),
                   "projection_evidence": "fixed torch_chunked basis; features recomputed for each independent repeat"}
        atomic(package_path, package)
        write_text(out / "inputs.files-from.txt", "\n".join(paths + [str(package_path.relative_to(root))]) + "\n")
        return package


def verify_package(cfg, root, package):
    if package["config"] != cfg:
        raise ValueError("prepared configuration conflict")
    if input_inventory(root, [x["path"] for x in package["inputs"]]) != package["inputs"]:
        raise ValueError("input file size mismatch; restore exact prepared sources")


class Telemetry:
    def __init__(self, files, device):
        self.files, self.device = files, device
        self.stage, self.started, self.done, self.total = "initializing", time.monotonic(), 0, 0
        self.first = None
        self.computed_units = 0
        self.previous_done = 0
        self.seconds = {}
        old = files.root / "progress.json"
        if old.exists():
            self.seconds = read_json(old).get("stage_seconds", {})

    def update(self, done=0, total=0, **extra):
        self.done, self.total = done, total
        elapsed = time.monotonic() - self.started
        # Based ONLY on actually computed units this invocation, never a probe.
        if extra.get("computed", False):
            # Native feature callbacks advance a batch; FMAS callbacks a row.
            self.computed_units += max(0, done - self.previous_done)
            if self.first is None:
                self.first = (done, elapsed)
        self.previous_done = done
        eta = None
        if self.first is not None and done >= self.first[0]:
            eta = max(0, total - done) * elapsed / max(1, self.computed_units)
        atomic(self.files.root / "progress.json", {
            "stage": self.stage, "done": done, "total": total, "updated_unix": time.time(),
            "stage_elapsed_seconds": elapsed, "stage_eta_seconds": eta,
            "stage_seconds": self.seconds, "io_seconds_this_attempt": self.files.io_seconds,
            "computed_units_this_stage_attempt": self.computed_units,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(self.device) if str(self.device).startswith("cuda") else 0,
            "pid": os.getpid(), **extra})

    @contextmanager
    def stage_run(self, stage):
        self.stage, self.started, self.first = stage, time.monotonic(), None
        self.computed_units, self.previous_done = 0, 0
        self.update()
        try:
            yield
        finally:
            self.seconds[stage] = self.seconds.get(stage, 0.0) + time.monotonic() - self.started
            self.update(self.done, self.total)


def finite(value, shape):
    good = torch.isfinite(value).all() if torch.is_tensor(value) else np.isfinite(value).all()
    if tuple(value.shape) != tuple(shape) or not good:
        raise ValueError(f"invalid shape/finite values, expected {shape}")
    return value


def fmas_pass(ctx, queries, cfg, repeat, files, progress):
    """32 shared query gradients, one train stream and one fixed inverse."""
    qids = [(t, q) for t in TRACKS for q in cfg["selection"][t]]
    ids = [[t, q] for t, q in qids]
    with progress.stage_run("fmas_query"):
        coords = files.get("fmas_raw/query_coords", ids)
        if coords is None:
            rows = []
            for j, (track, q) in enumerate(qids):
                stream = mc_stream(cfg, repeat, "fmas_raw", "query", track)
                g = ctx.scorer.sample_gradient(queries[track].images[q], None,
                        mc=ctx.mc_meas, chunk=ctx.chunk, seed=stream["seed"], phase=stream["phase"],
                        index=q, hflip=False, train_mode=False, readout=ctx.readout, sampling=ctx.sampling)
                rows.append(ctx.scorer.to_eigencoords(g).to(ctx.qdtype))
                progress.update(j + 1, len(qids), computed=True)
            coords = torch.stack(rows)
            files.put("fmas_raw/query_coords", coords.cpu(), ids)
        finite(coords, (len(qids), ctx.P))
        coords = coords.to(ctx.device)
    inverse = {cfg["fmas"]["rho"]: ctx.factors.damped_inverse_flat(
        cfg["fmas"]["rho"], ctx.device, mode="blockshrink")}
    for value in inverse.values():
        finite(value, (ctx.P,))
    stream = mc_stream(cfg, repeat, "fmas_raw", "train")
    blocks = []
    with progress.stage_run("fmas_train"):
        for start in range(0, ctx.N, cfg["fmas"]["row_block"]):
            end = min(ctx.N, start + cfg["fmas"]["row_block"])
            rows = list(range(start, end))
            name = f"fmas_raw/rows/{start}_{end}"
            block = files.get(name, {"train": rows, "query": ids})
            if block is None:
                values = []
                for i in rows:
                    values.append(_train_row(ctx, coords, inverse, ctx.train_ds.images[i], None,
                                              seed=stream["seed"], index=i)[cfg["fmas"]["rho"]])
                    progress.update(i + 1, ctx.N, computed=True)
                block = np.stack(values)
                finite(block, (end - start, len(qids)))
                files.put(name, block, {"train": rows, "query": ids})
            blocks.append(finite(block, (end - start, len(qids))))
            progress.update(end, ctx.N)
        result = np.concatenate(blocks)
        offset = 0
        for track in TRACKS:
            count = len(cfg["selection"][track])
            ids = score_ids(cfg, track)
            if load_score(files.root, "fmas_raw", track, files.identity, ids, (ctx.N, count)) is None:
                files.score("fmas_raw", track, result[:, offset:offset + count], ids)
            offset += count


def feature_pass(fz, ds, seed, block_size, files, name, progress, *, offset=0, total=None):
    """Native grid batches, including original last partial; resume only at batch boundaries."""
    if block_size % fz.batch_size:
        raise ValueError("feature blocks must preserve native batch boundaries")
    total = len(ds) if total is None else total
    blocks = []
    for start in range(0, len(ds), block_size):
        end = min(len(ds), start + block_size)
        ids = list(range(start, end))
        key = f"dtrak_T100/{name}/{start}_{end}"
        value = files.get(key, ids)
        if value is None:
            subset = ImageDataset(ds.images[start:end], ds.labels[start:end])
            value = fz.featurize(subset, seed=seed, log_every=fz.batch_size,
                  on_progress=lambda **kw: progress.update(offset + start + kw["done"], total, computed=True))
            finite(value, (end - start, fz.proj_dim))
            files.put(key, value, ids)
        blocks.append(finite(value, (end - start, fz.proj_dim)))
        progress.update(offset + end, total)
    return torch.cat(blocks)


def dtrak_pass(model, proc, train, queries, cfg, repeat, files, progress, device):
    d = cfg["dtrak"]
    loss_type, steps = _FEAT_RECIPES[d["method"]]
    if (loss_type, steps) != (d["loss_type"], d["T"]):
        raise ValueError("native D-TRAK recipe drift")
    fz = GradFeaturizer(model, proc, device=device, **{k: d[k] for k in
         ("proj_dim", "proj_seed", "T", "loss_type", "batch_size", "projection", "normalize",
          "t_sampling", "time_weight", "output_mask")})
    if fz.projector_kind != d["projection"]:
        raise ValueError("projection backend drift")
    atomic(files.root / "dtrak_T100/backend.json", {"resolved_backend": fz.projector_kind,
            "proj_dim": d["proj_dim"], "proj_seed": d["proj_seed"]})
    with progress.stage_run("dtrak_train"):
        G = feature_pass(fz, train, mc_stream(cfg, repeat, "dtrak_T100", "train")["seed"],
                         d["feature_block"], files, "train", progress)
    qfeatures = []
    with progress.stage_run("dtrak_query"):
        for track in TRACKS:
            full = feature_pass(fz, queries[track], mc_stream(cfg, repeat, "dtrak_T100", "query", track)["seed"],
                                d["feature_block"], files, "query_" + track, progress,
                                offset=TRACKS.index(track) * len(queries[track]),
                                total=sum(len(q) for q in queries.values()))
            qfeatures.append(full[cfg["selection"][track]])
    G = G.float().to(device)
    with progress.stage_run("dtrak_kernel"):
        inv = files.get("dtrak_T100/kernel_inverse", list(range(len(train))))
        if inv is None:
            inv = trak_inverse(G, d["lambda"], normalize=d["kernel_normalize"])
            finite(inv, (d["proj_dim"], d["proj_dim"]))
            files.put("dtrak_T100/kernel_inverse", inv.cpu(), list(range(len(train))))
        inv = finite(inv, (d["proj_dim"], d["proj_dim"])).to(device)
    with progress.stage_run("dtrak_score"):
        # Same float32 multiplication order as trak_kernel; shared inverse/solve.
        features_k = G @ inv
        for track, Q in zip(TRACKS, qfeatures):
            ids = score_ids(cfg, track)
            if load_score(files.root, "dtrak_T100", track, files.identity, ids,
                          (len(train), len(cfg["selection"][track]))) is None:
                scores = (Q.float().to(device) @ features_k.T).T.cpu().numpy().astype(np.float32)
                finite(scores, (len(train), len(cfg["selection"][track])))
                files.score("dtrak_T100", track, scores, ids)


def repeat_identity(cfg, package, repeat):
    if repeat not in range(cfg["repeats"]):
        raise ValueError("repeat-id outside configured range")
    return {"repeat_id": repeat, "config": cfg, "inputs": package["inputs"],
            "streams": [mc_stream(cfg, repeat, method, role, track)
                        for method in METHODS for role, track in
                        (("train", "shared"), ("query", "gen"), ("query", "val"))]}


def run_repeat(cfg, repeat, device, input_root, output_root):
    package = read_json(Path(output_root) / "package.json")
    verify_package(cfg, input_root, package)
    root = Path(output_root) / f"repeat_{repeat}"
    with exclusive(root):
        files = RepeatFiles(root, repeat_identity(cfg, package, repeat))
        torch.set_num_threads(cfg["cpu_threads"])
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        progress = Telemetry(files, device)
        atomic(root / "runtime.json", {"host": platform.node(), "pid": os.getpid(),
               "device": device, "torch": str(torch.__version__), "python": platform.python_version(),
               "threads": torch.get_num_threads(), "started_unix": time.time()})
        try:
            needed = {m: any(load_score(root, m, t, files.identity, score_ids(cfg, t),
                        (cfg["n_train"], len(cfg["selection"][t]))) is None for t in TRACKS) for m in METHODS}
            if any(needed.values()):
                pack = load_tensor(Path(input_root) / package["dataset_snapshot"])
                train = ImageDataset(pack["train_images"], pack["train_labels"])
                test = ImageDataset(pack["test_images"], pack["test_labels"])
                e = {**cfg["fmas"], "blockshrink_grid": [cfg["fmas"]["rho"]]}
                native = load_config({"storage.data_root": str(input_root), "train.hflip": cfg["fmas"]["hflip"]})
                ctx = _prepare(ReadOnlyInputs(input_root), native, device, cfg["dataset"], cfg["model_seed"],
                               query_type="gen", process="cfm", conditional=False, e=e, prepared_datasets=(train, test))
                queries = {"gen": ImageDataset(ctx.q_images[:100], ctx.q_labels[:100]),
                           "val": select_val_queries(test, 100)}
                if len(train) != cfg["n_train"] or any(len(q) != cfg["n_queries"] for q in queries.values()):
                    raise ValueError("prepared train/query row count disagrees with fixed configuration")
                if needed["fmas_raw"]:
                    fmas_pass(ctx, queries, cfg, repeat, files, progress)
                if needed["dtrak_T100"]:
                    # Release eigenbasis before large projected-gradient batches.
                    model, proc = ctx.model, ctx.proc
                    del ctx
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    dtrak_pass(model, proc, train, queries, cfg, repeat, files, progress, device)
            for method in METHODS:
                for track in TRACKS:
                    if load_score(root, method, track, files.identity, score_ids(cfg, track),
                                  (cfg["n_train"], len(cfg["selection"][track]))) is None:
                        raise ValueError("repeat is missing a final score; cannot mark complete")
            progress.stage = "complete"
            progress.update(cfg["n_train"], cfg["n_train"])
        except BaseException as exc:
            progress.update(progress.done, progress.total, failed=True, error=repr(exc))
            raise


def status(cfg, output_root):
    root = Path(output_root)
    package = read_json(root / "package.json")
    if package["config"] != cfg:
        raise ValueError("package configuration conflict")
    result = []
    for r in range(cfg["repeats"]):
        folder = root / f"repeat_{r}"
        identity = repeat_identity(cfg, package, r)
        if not (folder / "identity.json").exists() and any(folder.glob("*/*/scores.npy")):
            raise ValueError("scores without repeat identity cannot be accepted")
        if (folder / "identity.json").exists() and read_json(folder / "identity.json") != identity:
            raise ValueError("mixed repeat identity")
        complete = [f"{m}/{t}" for m in METHODS for t in TRACKS if load_score(
            folder, m, t, identity, score_ids(cfg, t), (cfg["n_train"], len(cfg["selection"][t]))) is not None]
        result.append({"repeat_id": r, "complete_scores": complete,
                       "complete": len(complete) == 4,
                       "saved_blocks": [str(p.relative_to(folder)) for p in sorted(folder.glob("*/*/*.pt"))],
                       "progress": read_json(folder / "progress.json") if (folder / "progress.json").exists() else None})
    return result


