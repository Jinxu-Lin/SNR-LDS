"""Produce the CFM common-200 retrieval blocks consumed by source_assembly.

The accepted experiment uses 250 MC draws on each side, 100 resident query
columns and 1,000 training rows per tile. The 300 columns retain their original
positions in the 500-query source, including when selecting the MC random seed.
Only score blocks and operational records are written; validation-only lambda
selection and metric publication are a separate CPU assembly step.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch

from balds import __version__
from balds.artifacts.addressing import block_dir, parse_block_key, relpath
from balds.artifacts.codecs import codec_for
from balds.artifacts.e3c import atomic, exclusive, read_json, RepeatFiles, write_text
from balds.artifacts.progress import ProgressCSV, progress_path
from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from . import curvature as c
from .config import load_config
from .source_assembly import query_selection


METHODS = ("fmas_raw", "ekfac_if")
PROTOCOL = "E5-COMMON200-20260920-v1"
ROWS = 50000
# Pinned to the accepted method identity, rather than mutable training defaults.
EKFAC = {
    "fit_epochs": 125, "eig_epochs": 125, "fit_batch_size": 64,
    "eig_batch_size": 16, "mc_loss": 250, "mc_measurement": 250,
    "grad_chunk": 125,
    "damping_grid": [1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
    "query_coords_dtype": "bfloat16", "query_chunk": 100, "row_chunk": 1000,
    "train_mode_for_loss_grads": True, "sampling": "iid", "damping_mode": "global",
    "blockshrink_grid": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "fit_seed_offset": 0, "max_train_samples": None,
}
IDENTITY_KEYS = c._BLOCK_IDENTITY + ("original_query_ids", "grad_chunk")


def _spec(method):
    if method not in METHODS:
        raise ValueError(f"retrieval scoring requires one of {METHODS}")
    return RunSpec(dataset="cifar10_inj4", seed=42, process="cfm", conditional=True,
                   query_type="inject", method=method)


def _selection_path(root, selection):
    relative = Path(selection)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("selection must be a data-root-relative path")
    path = Path(root) / relative
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("selection must remain inside the input data root")
    return path


class ScoreBlocks:
    """Load master inputs; atomically write score blocks to a distinct root."""
    def __init__(self, inputs, outputs, code_version=None):
        self.inputs, self.outputs = Path(inputs), Path(outputs)
        self.code_version = code_version or f"ba-lds-{__version__}"
        if self.inputs.resolve().is_relative_to(self.outputs.resolve()):
            raise ValueError("score output must not be the input root or its ancestor")

    def path(self, kind, spec, **key):
        root = self.outputs if kind == K.SCORES_BLOCK else self.inputs
        return root / relpath(kind, spec, **key)

    def load(self, kind, spec, **key):
        return codec_for(kind).load(str(self.path(kind, spec, **key)))

    def exists(self, kind, spec, **key):
        return self.path(kind, spec, **key).is_file()

    def save(self, kind, spec, value, **key):
        if kind != K.SCORES_BLOCK:
            raise ValueError("retrieval producer may write only isolated score blocks")
        atomic(self.path(kind, spec, **key), value, codec_for(kind))

    def local_blocks(self, kind, spec):
        if kind != K.SCORES_BLOCK:
            raise ValueError("only score blocks are owned by this producer")
        return [key for path in sorted((self.outputs / block_dir(kind, spec)).glob("*.npz"))
                if (key := parse_block_key(kind, path.name)) is not None]


def scientific_identity(queries, selection, method, query_chunk=100):
    _spec(method)
    if query_chunk < 1:
        raise ValueError("query_chunk must be positive")
    val_ids, test_ids = query_selection(queries, selection)
    if len(queries["split"]) != 500:
        raise ValueError("common-200 requires the original 500-query source")
    _, solver, _ = c.split_method(method)
    e = c._bind_solver(deepcopy(EKFAC), solver, method)
    e["query_chunk"] = query_chunk
    return {
        "protocol": PROTOCOL, "method": method, "dataset": "cifar10_inj4",
        "seed": 42, "process": "cfm", "conditional": True,
        "query_ids": val_ids + test_ids, "val_ids": val_ids, "test_ids": test_ids,
        "query_source": queries.get("queries_sha256"), "ekfac": e,
        "hflip": True, "selector": "inject_val", "k": 200,
    }


def _input_plan(store, method, selection, query_chunk):
    spec = _spec(method)
    if query_chunk < 1:
        raise ValueError("query_chunk must be positive")
    selected = _selection_path(store.inputs, selection)
    paths = [relpath(k, spec) for k in (K.CHECKPOINT, K.EKFAC_FACTORS, K.INJECT_QUERIES)]
    paths += [relpath(K.INJECT_META, RunSpec(dataset=spec.dataset)), str(Path(selection))]
    missing = [p for p in paths if not (store.inputs / p).is_file()]
    raw = []
    missing_raw = []
    for dataset in ("cifar10", "cifar100"):
        found = [p for p in (store.inputs / "hf_cache" / dataset).rglob("*")
                 if p.is_file() and (p.suffix == ".arrow" or p.name == "dataset_info.json")]
        raw.extend(str(p.relative_to(store.inputs)) for p in found)
        if not any(p.suffix == ".arrow" for p in found):
            missing_raw.append(f"hf_cache/{dataset}/**/*.arrow")
    identity = None
    if store.exists(K.INJECT_QUERIES, spec) and selected.is_file():
        identity = scientific_identity(store.load(K.INJECT_QUERIES, spec),
                                       read_json(selected), method, query_chunk)
    return {
        "code_version": store.code_version, "method": method,
        "query_axis": "original500positions", "identity": identity,
        "inputs": paths, "raw_data": sorted(raw), "missing_inputs": missing,
        "missing_raw_data": missing_raw, "ready": not missing and not missing_raw,
        "fit": False, "assembly_required": True,
    }


def prepare(input_data_root, output_root, *, method, selection, query_chunk=100,
            code_version=None):
    """Write a local input inventory and immutable identity, without models."""
    store = ScoreBlocks(input_data_root, output_root, code_version)
    plan = _input_plan(store, method, selection, query_chunk)
    with exclusive(store.outputs / method / "identity"):
        if (not (store.outputs / method / "identity" / "identity.json").is_file()
                and store.local_blocks(K.SCORES_BLOCK, _spec(method))):
            raise ValueError("score blocks have no method identity; cannot establish provenance")
        if plan["identity"] is not None:
            RepeatFiles(store.outputs / method / "identity", plan["identity"])
        atomic(store.outputs / method / "package.json", plan)
        write_text(store.outputs / method / "inputs.files-from.txt",
                   "\n".join(plan["inputs"] + plan["raw_data"]) + "\n")
    return plan


def _expected_meta(identity, first):
    """Expected scientific fields without loading a checkpoint or factors."""
    e = identity["ekfac"]
    return {
        "dampings": np.asarray(c._grid(e), dtype=np.float64),
        "method": identity["method"], "dataset": identity["dataset"],
        "query_type": "inject", "process": "cfm", "readout": "loss",
        "sampling": e["sampling"], "damping_mode": e["damping_mode"],
        "seed": 42, "conditional": 1, "N": ROWS, "Q": 300, "P": first["P"],
        "mc_loss": e["mc_loss"], "mc_measurement": e["mc_measurement"],
        "hflip": 1, "train_mode": 1, "fit_epochs": 125, "eig_epochs": 125,
        "original_query_ids": identity["query_ids"], "grad_chunk": e["grad_chunk"],
        "code_version": first.get("code_version", "?"),
    }


def _coverage(store, spec, identity):
    """Metadata/coverage census. Final assembly checks block values and finiteness."""
    e = identity["ekfac"]
    blocks, tiles = c._query_blocks(300, e["query_chunk"]), c._row_tiles(ROWS, e["row_chunk"])
    local = store.local_blocks(K.SCORES_BLOCK, spec)
    expected = None
    for key in local:
        if (key["q0"], key["q1"]) not in blocks:
            raise ValueError(f"score block has a foreign query tiling: {key}")
        with np.load(store.path(K.SCORES_BLOCK, spec, **key), allow_pickle=False) as archive:
            meta = {name: archive[name] for name in archive.files if name != "block"}
        if expected is None:
            expected = _expected_meta(identity, meta)
        c._check_block(meta, expected, IDENTITY_KEYS, what="retrieval status",
                       q0=key["q0"], q1=key["q1"], r0=key.get("r0", 0),
                       r1=key.get("r1", ROWS), rows=ROWS)
        if "query_source" in meta and str(meta["query_source"]) != identity["query_source"]:
            raise ValueError("score block source query identity differs")
    missing, done = [], 0
    for q0, q1 in blocks:
        sources, gaps = c._row_sources(store, K.SCORES_BLOCK, spec, q0, q1,
                                      tiles, ROWS, local, what="retrieval status")
        if sources is None:
            missing += [{"q0": q0, "q1": q1, "rows": gap} for gap in gaps]
        else:
            done += 1
    return {"blocks_complete": not missing, "blocks_done": done,
            "blocks_total": len(blocks), "partial_files": len(local),
            "missing_blocks": missing, "assembly_required": True,
            "validation": "metadata and coverage; values checked by assemble_retrieval"}


def status(input_data_root, output_root, *, method, selection, query_chunk=100,
           code_version=None):
    """Read identities and block headers only; never prepare a model or write."""
    store = ScoreBlocks(input_data_root, output_root, code_version)
    plan = _input_plan(store, method, selection, query_chunk)
    saved = store.outputs / method / "identity" / "identity.json"
    if plan["identity"] is None:
        return {**plan, "blocks_complete": False, "missing_identity_inputs": True}
    if saved.is_file() and read_json(saved) != plan["identity"]:
        raise ValueError("existing retrieval identity differs from the requested inputs")
    if not saved.is_file() and store.local_blocks(K.SCORES_BLOCK, _spec(method)):
        raise ValueError("score blocks have no method identity; cannot establish provenance")
    return {**plan, **_coverage(store, _spec(method), plan["identity"])}


def _query_pass(ctx, prog, q0, q1):
    """Original E5 query pass: RNG index is the ORIGINAL query position."""
    coords = torch.empty(q1 - q0, ctx.P, dtype=ctx.qdtype, device=ctx.device)
    for j, q in enumerate(range(q0, q1)):
        label = int(ctx.q_labels[q]) if ctx.conditional else None
        gradient = ctx.scorer.sample_gradient(
            ctx.q_images[q], label, mc=ctx.mc_meas, chunk=ctx.chunk,
            seed=ctx.seed, phase="ekfac_meas", index=int(ctx.original_query_ids[q]),
            hflip=False, train_mode=False, readout=ctx.readout, sampling=ctx.sampling)
        coords[j] = ctx.scorer.to_eigencoords(gradient).to(ctx.qdtype)
        if (q + 1) % 10 == 0 or q + 1 == q1:
            prog.log(done=q + 1, total=ctx.Q, phase="query_grads")
    return coords


def _score_blocks(store, cfg, ctx, spec, *, rank, world, row_rank, row_world, timing):
    """Reuse the native row scorer and atomic resume/tiling protocol."""
    blocks = c._query_blocks(ctx.Q, c._query_chunk(ctx))
    tiles = c._row_tiles(ctx.N, c._row_chunk(ctx.e, ctx.N))
    prog = ProgressCSV(progress_path(cfg["storage"]["data_root"],
        f"retrieval_{spec.method}" + c._shard_suffix(rank, world, row_rank, row_world)))
    meta = c._block_meta(ctx, raw_name=spec.method, query_type="inject",
                         code_version=store.code_version)
    meta["original_query_ids"] = ctx.original_query_ids
    meta["query_source"] = np.array(ctx.query_source)
    inv_box, qbox = [], []

    def compute(q0, q1, r0, r1):
        started = time.monotonic()
        if not inv_box:
            inverse = {d: ctx.factors.damped_inverse_flat(d, ctx.device, mode=ctx.damping_mode)
                       for d in ctx.dampings}
            c._check_finite_inverses(inverse, ctx.damping_mode)
            inv_box.append(inverse)
        if not qbox or qbox[0][0] != (q0, q1):
            qbox.clear()
            qbox.append(((q0, q1), _query_pass(ctx, prog, q0, q1)))
        result = np.zeros((len(ctx.dampings), r1 - r0, q1 - q0), dtype=np.float32)
        for i in range(r0, r1):
            label = int(ctx.train_ds.labels[i]) if ctx.conditional else None
            row = c._train_row(ctx, qbox[0][1], inv_box[0], ctx.train_ds.images[i], label,
                               seed=ctx.seed, index=i, flipped=c._train_flip(ctx, i))
            if i - r0 < 2:
                c._check_finite_row(row, what="retrieval", index=i)
            for di, damping in enumerate(ctx.dampings):
                result[di, i - r0] = row[damping]
            if (i - r0 + 1) % 25 == 0 or i + 1 == r1:
                prog.log(done=i - r0 + 1, total=r1 - r0,
                         phase=f"train_grads[{q0}:{q1}]r[{r0}:{r1}]")
        timing("score", time.monotonic() - started, dict(q0=q0, q1=q1, r0=r0, r1=r1))
        return result

    return c._run_shard(store, K.SCORES_BLOCK, spec, blocks, tiles, ctx.N, meta,
                        IDENTITY_KEYS, rank=rank, world=world, row_rank=row_rank,
                        row_world=row_world, ignore_partials=False, compute=compute,
                        what="retrieval")


def run(input_data_root, output_root, *, method, selection, device="cuda:0",
        query_chunk=100, rank=0, world=1, row_rank=0, row_world=1,
        cpu_threads=2, code_version=None):
    """Compute this worker's blocks; resume validated blocks without overwriting."""
    c._check_shard(rank, world)
    c._check_shard(row_rank, row_world, what="--row-rank/--row-world")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    spec = _spec(method)
    store = ScoreBlocks(input_data_root, output_root, code_version)
    with exclusive(store.outputs / method / f"worker_{rank}_{row_rank}"):
        plan = prepare(input_data_root, output_root, method=method, selection=selection,
                       query_chunk=query_chunk, code_version=code_version)
        if not plan["ready"]:
            raise FileNotFoundError(f"missing retrieval inputs: {plan['missing_inputs'] + plan['missing_raw_data']}")
        identity = plan["identity"]
        census = _coverage(store, spec, identity)
        if census["blocks_complete"]:
            return {**census, "cached": True, "computed_tiles": 0}
        torch.set_num_threads(cpu_threads)
        cfg = load_config({"storage.data_root": str(Path(input_data_root).resolve())})
        cfg["storage"]["data_root"] = str(store.outputs)
        cfg["train"]["hflip"] = identity["hflip"]
        e = identity["ekfac"]

        def timing(stage, seconds, block):
            track = ("precompute" if stage == "precompute" else "val" if block["q1"] <= 100
                     else "test" if block["q0"] >= 100 else "mixed")
            atomic(store.outputs / method / "timing" / f"{time.time_ns()}_{rank}_{row_rank}.json",
                   {"phase": track, "seconds": seconds, "train_rows": block.get("r1", 0) - block.get("r0", 0),
                    "queries": block.get("q1", 0) - block.get("q0", 0), **block})

        started = time.monotonic()
        ctx = c._prepare(store, cfg, device, spec.dataset, spec.seed,
                         query_type="inject", process="cfm", conditional=True, readout="loss", e=e)
        if ctx.N != ROWS or ctx.Q != 500:
            raise ValueError(f"expected 50000 training rows and 500 original queries; got {ctx.N}, {ctx.Q}")
        ids = np.asarray(identity["query_ids"], dtype=np.int64)
        ctx.q_images, ctx.q_labels = ctx.q_images[ids], ctx.q_labels[ids]
        ctx.Q, ctx.original_query_ids, ctx.query_source = len(ids), ids, identity["query_source"]
        for key in ("fit_epochs", "eig_epochs"):
            if int((getattr(ctx.factors, "meta", {}) or {}).get(key, -1)) != e[key]:
                raise ValueError(f"curvature factors have a different {key} identity")
        timing("precompute", time.monotonic() - started, {})
        computed = _score_blocks(store, cfg, ctx, spec, rank=rank, world=world,
                                 row_rank=row_rank, row_world=row_world, timing=timing)
        result = {**_coverage(store, spec, identity), "computed_tiles": computed,
                  "rank": rank, "world": world, "row_rank": row_rank, "row_world": row_world,
                  "code_version": store.code_version}
        atomic(store.outputs / method / f"worker_{rank}_{row_rank}" / "receipt.json", result)
        return result
