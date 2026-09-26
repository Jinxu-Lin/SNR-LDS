"""Row-major EK-FAC IF: shared CPU query coordinates, one train pass for gen/val.

Uses the native sampler, per-row matvec and SCORES_BLOCK format. No model/MC
changes, full train-gradient cache, GPU experiment, or new storage framework.
"""
from __future__ import annotations

import gc
import numpy as np
import torch

from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.artifacts.progress import ProgressCSV, progress_path
from .config import resolve_Q
from .curvature import (
    EkfacScoreUseCase, _prepare, _ekfac_cfg, _bind_solver, split_method,
    _block_meta, _BLOCK_IDENTITY, _as_py, _check_block, _check_finite_inverses,
    _query_blocks, _query_chunk, _query_pass, _train_coords, _score_coords,
    _train_flip, _row_tiles, _bkey, _assemble_blocks, _grid,
)


class JointEkfacScoreUseCase:
    """Only native ekfac_if, two tracks with equal Q; disjoint contiguous units."""

    FEAT = "ekfac_if_eigencoords"
    TRACKS = ("gen", "val")
    CHECKS = (*_BLOCK_IDENTITY, "grad_chunk", "query_chunk", "query_coords_dtype")

    def __init__(self, store, cfg, *, device="cuda:0"):
        self.store, self.cfg, self.device = store, cfg, device

    def _spec(self, dataset, seed, process, conditional, track):
        return RunSpec(dataset=dataset, seed=seed, process=process,
                       conditional=conditional, method="ekfac_if", query_type=track)

    def _context(self, dataset, seed, process, conditional, track):
        _, solver, _ = split_method("ekfac_if")
        e = _bind_solver(_ekfac_cfg(self.cfg), solver, "ekfac_if")
        return _prepare(self.store, self.cfg, self.device, dataset, seed,
                        process=process, conditional=conditional, query_type=track,
                        readout="loss", e=e)

    def _meta(self, ctx, track):
        return {**_block_meta(ctx, raw_name="ekfac_if", query_type=track,
                             code_version=getattr(self.store, "code_version", "?")),
                "query_chunk": np.int64(_query_chunk(ctx)),
                "query_coords_dtype": np.array(str(ctx.qdtype)),
                "execution_layout": np.array("joint_row_major_v1")}

    @staticmethod
    def _check_cache(payload, want):
        for field in JointEkfacScoreUseCase.CHECKS:
            if field not in payload or _as_py(payload[field]) != _as_py(want[field]):
                raise ValueError(f"query coordinate cache mismatch: {field}")
        coords = payload["coords"]
        if tuple(coords.shape) != (int(want["Q"]), int(want["P"])) or \
                str(coords.dtype) != _as_py(want["query_coords_dtype"]):
            raise ValueError("query coordinate shape/dtype mismatch")
        # Bound temporary memory when checking a production 18 GiB tensor.
        if not all(bool(torch.isfinite(row).all()) for row in coords):
            raise ValueError("non-finite query coordinates")

    def prepare(self, dataset, seed, *, process="cfm", conditional=True):
        """Compute only query gradients. Run once, then copy both native artifacts."""
        out = {}
        for track in self.TRACKS:
            ctx = self._context(dataset, seed, process, conditional, track)
            spec = self._spec(dataset, seed, process, conditional, track)
            meta = self._meta(ctx, track)
            if self.store.has_local(K.QUERY_FEATURES, spec, feat=self.FEAT):
                payload = self.store.load(K.QUERY_FEATURES, spec, feat=self.FEAT)
                self._check_cache(payload, meta)
                out[track] = "reused"
                del payload
            else:
                prog = ProgressCSV(progress_path(self.cfg["storage"]["data_root"],
                                    f"ekfac_joint_queries_{dataset}_{process}_{seed}_{track}"))
                cpu = torch.empty((ctx.Q, ctx.P), dtype=ctx.qdtype, device="cpu")
                for q0, q1 in _query_blocks(ctx.Q, _query_chunk(ctx)):
                    q = _query_pass(ctx, prog, q0, q1)
                    cpu[q0:q1].copy_(q.cpu())
                    del q
                payload = {**meta, "coords": cpu}
                self._check_cache(payload, meta)
                self.store.save_atomic(K.QUERY_FEATURES, spec, payload, feat=self.FEAT)
                out[track] = "produced"
                del payload, cpu
            del ctx
            gc.collect()
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
        return out

    @staticmethod
    def unit_tiles(rows, unit, units, row_chunk):
        if units <= 0 or not 0 <= unit < units or row_chunk <= 0:
            raise ValueError("invalid unit/units/row_chunk")
        # Each unit owns a contiguous range of complete tiles; no overlap.
        tiles = _row_tiles(rows, row_chunk)
        lo, hi = len(tiles) * unit // units, len(tiles) * (unit + 1) // units
        return tiles[lo:hi]

    def run(self, dataset, seed, *, unit, units=8, row_chunk=25,
            process="cfm", conditional=True):
        """No query extraction here. Persist native blocks after each small row tile."""
        ctx = self._context(dataset, seed, process, conditional, "gen")
        tiles = self.unit_tiles(ctx.N, unit, units, row_chunk)
        caches, metas, specs = {}, {}, {}
        for track in self.TRACKS:
            spec = self._spec(dataset, seed, process, conditional, track)
            payload = self.store.load(K.QUERY_FEATURES, spec, feat=self.FEAT)
            want = self._meta(ctx, track)
            self._check_cache(payload, want)
            caches[track], metas[track], specs[track] = payload, want, spec
        blocks = _query_blocks(ctx.Q, _query_chunk(ctx))
        prog = ProgressCSV(progress_path(self.cfg["storage"]["data_root"],
                            f"ekfac_joint_{dataset}_{process}_{seed}_unit{unit}"))
        inverses = None
        computed_rows = computed_blocks = 0
        for r0, r1 in tiles:
            missing = []
            for track in self.TRACKS:
                for q0, q1 in blocks:
                    key = _bkey(q0, q1, r0, r1, ctx.N)
                    if self.store.has_local(K.SCORES_BLOCK, specs[track], **key):
                        z = self.store.load(K.SCORES_BLOCK, specs[track], **key)
                        _check_block(z, metas[track], self.CHECKS, what="joint row resume",
                                     q0=q0, q1=q1, r0=r0, r1=r1, rows=ctx.N)
                        if z["block"].shape != (len(ctx.dampings), r1-r0, q1-q0) or \
                                not np.isfinite(z["block"]).all():
                            raise ValueError("invalid existing joint scores block")
                    else:
                        missing.append((track, q0, q1, key))
            if not missing:
                continue
            if inverses is None:
                inverses = {d: ctx.factors.damped_inverse_flat(d, self.device,
                             mode=ctx.damping_mode) for d in ctx.dampings}
                _check_finite_inverses(inverses, ctx.damping_mode)
            # FP32 CPU row buffer ~8.9 GiB for 25 AB2 rows; never N x P.
            train = torch.empty((r1-r0, ctx.P), dtype=torch.float32, device="cpu")
            for i in range(r0, r1):
                label = int(ctx.train_ds.labels[i]) if conditional else None
                u = _train_coords(ctx, ctx.train_ds.images[i], label, seed=seed,
                                  index=i, flipped=_train_flip(ctx, i))
                if not bool(torch.isfinite(u).all()):
                    raise ValueError(f"non-finite train coordinates at row {i}")
                train[i-r0].copy_(u.cpu())
                del u
                prog.log(done=i-r0+1, total=r1-r0, phase=f"train_coords[{r0}:{r1}]")
            computed_rows += r1-r0
            for track, q0, q1, key in missing:
                q = caches[track]["coords"][q0:q1].to(self.device)
                arr = np.empty((len(ctx.dampings), r1-r0, q1-q0), dtype=np.float32)
                for j in range(r1-r0):
                    row = _score_coords(ctx, q, inverses, train[j].to(self.device))
                    for di, d in enumerate(ctx.dampings):
                        arr[di, j] = row[d]
                del q
                if not np.isfinite(arr).all():
                    raise ValueError("non-finite joint scores")
                payload = {**metas[track], "block": arr, "q0": q0, "q1": q1,
                           "r0": r0, "r1": r1,
                           "query_cache_host": caches[track]["hostname"],
                           "query_cache_code_version": caches[track]["code_version"]}
                self.store.save_atomic(K.SCORES_BLOCK, specs[track], payload, **key)
                computed_blocks += 1
            del train
            prog.log(done=r1, total=ctx.N, phase="saved_all_query_blocks")
        return {"unit": unit, "units": units, "row_tiles": tiles,
                "computed_train_rows": computed_rows, "computed_blocks": computed_blocks}

    def assemble(self, dataset, seed, *, row_chunk=25, process="cfm", conditional=True):
        """CPU only, fail on missing coverage. Never falls back to gradient work."""
        by_track = {}
        for track in self.TRACKS:
            spec = self._spec(dataset, seed, process, conditional, track)
            payload = self.store.load(K.QUERY_FEATURES, spec, feat=self.FEAT)
            n, q = int(payload["N"]), int(payload["Q"])
            blocks = _query_blocks(q, int(payload["query_chunk"]))
            meta = {k: v for k, v in payload.items() if k != "coords"}
            del payload
            full, info = _assemble_blocks(self.store, K.SCORES_BLOCK, spec, blocks,
                _row_tiles(n, row_chunk), n, meta, self.CHECKS,
                prefix=(len(meta["dampings"]),), Q=q, rank=0, world=1,
                row_rank=0, row_world=1, what="joint CPU assembly")
            if full is None:
                raise ValueError(f"{track} incomplete: {info['missing_blocks']}")
            by_track[track] = spec, meta, full, info
        _, solver, _ = split_method("ekfac_if")
        e = _bind_solver(_ekfac_cfg(self.cfg), solver, "ekfac_if")
        for spec, meta, full, info in by_track.values():
            for field, value in {"dampings": _grid(e), "mc_loss": int(e["mc_loss"]),
                                 "mc_measurement": int(e["mc_measurement"]),
                                 "grad_chunk": int(e["grad_chunk"]),
                                 "Q": resolve_Q(self.cfg, dataset)}.items():
                if _as_py(meta[field]) != _as_py(value):
                    raise ValueError(f"assembly config mismatch: {field}")
        for track, (spec, meta, full, info) in by_track.items():
            # Cross-host rsync brings bytes, not sqlite rows. Enrol the verified
            # native blocks without copying a worker's entire manifest database.
            for key in self.store.local_blocks(K.SCORES_BLOCK, spec):
                self.store.save_atomic(K.SCORES_BLOCK, spec,
                                      self.store.load(K.SCORES_BLOCK, spec, **key), **key)
            for di, damping in enumerate(meta["dampings"]):
                self.store.save_atomic(K.SCORES_LAMBDA, spec, full[di], lam=float(damping))
        # Existing selectors, metadata and final artifact paths; cached lambdas
        # guarantee the native usecase cannot enter its gradient stream.
        results = {}
        for track, (spec, meta, full, info) in by_track.items():
            results[track] = EkfacScoreUseCase(self.store, self.cfg, device="cpu").run(
                dataset, seed, process=process, conditional=conditional, query_type=track,
                method="ekfac_if")
            filed = self.store.load(K.SCORES_META, spec)
            filed.update(partials=info["partials"], execution_layout="joint_row_major_v1")
            self.store.save_atomic(K.SCORES_META, spec, filed, replace=True)
        return results
