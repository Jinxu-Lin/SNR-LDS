"""GT LOSSES SHARDING PARITY: `subsets losses --rank/--world` splits the 64
subset rows across GPUs and reassembles a gt_losses matrix byte-identical to the
single-process (`world=1`) run.

Background: before this change `--rank/--world` were accepted by the subsets
parser but only wired into `train` — `compute_losses` ignored them and always ran
all M subsets in one process, so val-side GT (Q=1000, ~20x gen's compute) could
not be split across a multi-GPU box and a killed run restarted a whole e_seed
from subset 0. The sharded path now writes one GT_LOSSROW per subset (resumable
per subset) and the last rank assembles the full matrix.

The row math (`compute_query_losses`) is unchanged and parity-tested separately
(test_ground_truth_chunking); here we stub `_loss_row` with a deterministic
per-(subset, e_seed) value so the test needs no model/data/GPU and directly
exercises the new orchestration: row partition, per-subset persistence, resume
skip, finalize timing, and assembly order.

Run (no GPU/data needed):  conda run -n da python tests/integration/test_gt_losses_sharding.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows.subsets import SubsetsUseCase
from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB

M = 7   # not a multiple of the world sizes below -> exercises uneven shards
Q = 5


class StubUseCase(SubsetsUseCase):
    """Deterministic stand-in: row m encodes m (to catch assembly-order bugs) and
    e_seed, at float32 like the real return; no model/query/GPU touched."""

    def _loss_row(self, spec, m, queries, labels, process, sched, T_avg, target_batch, e_seed, rkey=None):
        base = np.arange(queries.shape[0], dtype=np.float32)
        return (base + 100.0 * m + 0.5 * e_seed).astype(np.float32)

    def _queries(self, spec, dataset, process):
        return torch.zeros(Q, 3, 32, 32), torch.zeros(Q, dtype=torch.long)


def _fresh(tmp):
    root = tempfile.mkdtemp(dir=tmp)
    store = ArtifactStore(ManifestDB(root + "/m.db"), [LocalBackend(root + "/data")],
                             code_version="t")
    cfg = {"storage": {"data_root": root + "/data"},
           "lds": {"T_avg": 10, "chunk_target_batch": 2000}}
    uc = StubUseCase(store, cfg, device="cpu")
    spec = RunSpec(dataset="cifar2", seed=42, process="cfm", query_type="val")
    store.save(K.SUBSET_MASKS, spec, [0] * M)   # only len(masks) = M is used
    return store, uc, spec


def _run(uc, *, world, ranks, e_seed=0):
    return [uc.compute_losses("cifar2", 42, process="cfm", query_type="val",
                              e_seed=e_seed, rank=r, world=world) for r in ranks]


def test_sharded_matches_single_process():
    tmp = tempfile.mkdtemp()
    s1, u1, spec = _fresh(tmp); _run(u1, world=1, ranks=[0])
    A = s1.load(K.GT_LOSSES, spec, eseed=0)

    for world in (2, 3, M):     # incl. world == M (one subset per shard)
        s, u, sp = _fresh(tmp); _run(u, world=world, ranks=list(range(world)))
        B = s.load(K.GT_LOSSES, spec, eseed=0)
        assert np.array_equal(A, B), f"world={world} not byte-identical to world=1"

    assert A.dtype == np.float64, A.dtype
    for m in range(M):          # assembly order is correct (row m really is row m)
        assert np.array_equal(A[m], (np.arange(Q) + 100.0 * m).astype(np.float32))
    print(f"  [parity] world=1 vs world in (2,3,{M}) all byte-identical; assembly order correct")


def test_finalize_only_when_all_rows_present():
    tmp = tempfile.mkdtemp()
    s, u, spec = _fresh(tmp)
    r0 = _run(u, world=2, ranks=[0])[0]                 # only even subsets done
    assert r0["finalized"] is False and r0["rows"] < M, r0
    assert not s.exists(K.GT_LOSSES, spec, eseed=0), "matrix finalized with rows missing"
    r1 = _run(u, world=2, ranks=[1])[0]                 # odd subsets -> all present
    assert r1["finalized"] is True, r1
    assert s.exists(K.GT_LOSSES, spec, eseed=0)
    print("  [finalize] full matrix appears only after the last shard completes")


def test_rerun_shard_skips_and_still_finalizes():
    tmp = tempfile.mkdtemp()
    s, u, spec = _fresh(tmp)
    _run(u, world=2, ranks=[0])
    r0b = _run(u, world=2, ranks=[0])[0]                # re-run same shard: rows already exist
    assert r0b["finalized"] is False, r0b               # still waiting on rank 1
    _run(u, world=2, ranks=[1])
    assert s.exists(K.GT_LOSSES, spec, eseed=0)
    # matches world=1
    s1, u1, _ = _fresh(tmp); _run(u1, world=1, ranks=[0])
    assert np.array_equal(s.load(K.GT_LOSSES, spec, eseed=0),
                          s1.load(K.GT_LOSSES, spec, eseed=0))
    print("  [resume] re-running a finished shard is a no-op; result still correct")


def test_world1_writes_no_rows_and_saves_matrix():
    tmp = tempfile.mkdtemp()
    s, u, spec = _fresh(tmp)
    out = _run(u, world=1, ranks=[0])[0]
    assert "shape" in out and out["shape"] == [M, Q], out
    assert not s.exists(K.GT_LOSSROW, spec, eseed=0, m=0), "world=1 should not emit per-subset rows"
    print("  [world=1] byte-path unchanged: matrix saved, no GT_LOSSROW shards emitted")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} gt-losses sharding tests passed")
