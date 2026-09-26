"""Train-row sharding of the curvature streams (EKFAC-ROWSHARD, 2026-09-09).

EKFAC-SHARD split a cell by QUERY block, which buys cards at the price of work:
every block re-streams the whole train side, so `world` shards do `world`x the
train-gradient cost. This suite covers the axis one level below it — the same
query block's N train rows split across processes — which re-runs only the
block's (cheap) query pass and therefore scales without multiplying work.

The load-bearing claim is again that residency does not touch a bit.
``_train_row``'s MC draw is keyed by (seed, "ekfac_loss", GLOBAL row index), so
row i is the same number whether it was streamed first, last, or alone. The
tests assert array-level identity between

    resident  vs  2 blocks x 2 row shards (4 processes)
              vs  mixed (one block resident, one block row-sharded)
              vs  a row shard interrupted and resumed

for the selected ``scores`` AND every per-damping ``scores_lambda``, plus the
same for the σ̂ ``repeat_scores`` (whose row axis is the M-sample subset).

The guard rails are the coverage arithmetic: overlapping row partials are a hard
error (two answers for one row, and the assembler must not pick one), an
identity mismatch is a hard error, and a coverage GAP files nothing while
naming the missing rows — a gap is the normal mid-flight state of a sharded
cell, so it must report, not raise.

CPU only, a base_ch=8 UNet, MC budgets of 1-2 draws: seconds.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows.config import load_config  # noqa: E402
from balds.workflows.curvature import (EkfacFitUseCase, EkfacRepeatsUseCase,  # noqa: E402
                               EkfacScoreUseCase)
from balds.schema.artifact import ArtifactKind as K  # noqa: E402
from balds.schema.registry import DATASETS  # noqa: E402
from balds.schema.runspec import RunSpec  # noqa: E402
from balds.data.base import ImageDataset  # noqa: E402
from balds.models import checkpoint_state  # noqa: E402
from balds.models.unet import UNetCFM  # noqa: E402
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB  # noqa: E402

_DS = "_ekfac_rowblock_toy"
_N, _M, _Q = 6, 4, 4          # 6 train rows, 4 subsets, 4 queries
_SEED = 42
_GRID = [1e-8, 1e-4]


def _toy_loader(data_dir, **_):
    g = torch.Generator().manual_seed(0)
    train = ImageDataset(torch.randn(_N, 3, 32, 32, generator=g).clamp(-1, 1),
                         torch.zeros(_N, dtype=torch.long))
    test = ImageDataset(torch.randn(4, 3, 32, 32, generator=g).clamp(-1, 1),
                        torch.zeros(4, dtype=torch.long))
    return train, test


if not DATASETS.has(_DS):
    DATASETS.add(_DS, _toy_loader)


def _cfg(root: str, **over):
    cfg = load_config({
        "storage.data_root": f"{root}/data",
        "ekfac.fit_epochs": 1, "ekfac.eig_epochs": 1,
        "ekfac.fit_batch_size": 3, "ekfac.eig_batch_size": 3,
        "ekfac.mc_loss": 2, "ekfac.mc_measurement": 2, "ekfac.grad_chunk": 2,
        "ekfac.query_coords_dtype": "float32",
        "tweedie.repeats": 2, "tweedie.sigma_samples": 4,
        **over})
    cfg["ekfac"]["damping_grid"] = list(_GRID)
    return cfg


def _store(root: str) -> ArtifactStore:
    return ArtifactStore(ManifestDB(f"{root}/m.db"), [LocalBackend(f"{root}/data")],
                            code_version="test")


def _spec(method: str = "ekfac_if") -> RunSpec:
    return RunSpec(dataset=_DS, seed=_SEED, method=method, query_type="gen",
                   process="cfm", conditional=False)


@pytest.fixture(scope="module")
def seeded():
    """One fitted cell, built once; every variant runs on a byte-copy of it."""
    root = tempfile.mkdtemp()
    store, cfg = _store(root), _cfg(root)
    base = RunSpec(dataset=_DS, seed=_SEED, process="cfm", conditional=False)
    g = torch.Generator().manual_seed(1)
    torch.manual_seed(1)
    model = UNetCFM(base_ch=8, num_classes=None)
    store.save(K.CHECKPOINT, base,
               checkpoint_state(model, base_ch=8, num_classes=None, step=1,
                                extra={"model_type": "cfm", "conditional": False}))
    store.save(K.GENERATION, base,
               {"samples": torch.randn(_Q, 3, 32, 32, generator=g).clamp(-1, 1),
                "labels": torch.zeros(_Q, dtype=torch.long),
                "Q": _Q, "gen_seed": 0, "ode_steps": 2, "conditional": False})
    store.save(K.SUBSET_MASKS, base,
               [np.random.RandomState(m).rand(_N) < 0.5 for m in range(_M)])
    store.save(K.GT_MATRIX, base.with_(query_type="gen"),
               np.random.RandomState(7).rand(_M, _Q))
    EkfacFitUseCase(store, cfg, device="cpu").run(_DS, _SEED, process="cfm", conditional=False)
    return root


def _clone(seeded_root: str, **over):
    dst = tempfile.mkdtemp()
    shutil.copytree(f"{seeded_root}/data", f"{dst}/data")
    return _store(dst), _cfg(dst, **over), dst


def _score(store, cfg, **kw):
    return EkfacScoreUseCase(store, cfg, device="cpu").run(
        _DS, _SEED, query_type="gen", process="cfm", conditional=False, **kw)


def _repeats(store, cfg, **kw):
    return EkfacRepeatsUseCase(store, cfg, device="cpu").run(
        _DS, _SEED, query_type="gen", process="cfm", conditional=False, **kw)


def _matrices(store, method: str = "ekfac_if") -> dict:
    spec = _spec(method)
    return {"scores": store.load(K.SCORES, spec),
            **{f"lam{d:g}": store.load(K.SCORES_LAMBDA, spec, lam=d) for d in _GRID}}


def _assert_same(a: dict, b: dict, what: str):
    assert set(a) == set(b)
    for k in a:
        assert np.array_equal(a[k], b[k]), f"{what}: {k} moved"


def _blocks(store, kind=K.SCORES_BLOCK) -> list:
    return store.local_blocks(kind, _spec())


def _unfile(store) -> None:
    """Delete the whole-matrix artifacts, keeping the block partials.

    Forces the next run down the assembly path instead of the per-damping cache
    (which would make a "finalized" assertion vacuous).
    """
    for d in _GRID:
        Path(store.local.abspath(
            f"scores/ekfac_if/{_DS}/seed_42/scores_lambda_{d:.2e}.npy")).unlink()
    Path(store.local.abspath(f"scores/ekfac_if/{_DS}/seed_42/scores.npy")).unlink()


# --------------------------------------------------------------------------- #
# the load-bearing claim: the second residency axis does not touch a bit either
# --------------------------------------------------------------------------- #

def test_resident_row_sharded_mixed_and_resumed_file_identical_matrices(seeded):
    # (a) resident: one query block, one row tile — the pre-ROWSHARD shape
    s_res, cfg_res, _ = _clone(seeded)
    out = _score(s_res, cfg_res)
    assert out["shape"] == [_N, _Q] and out["blocks_total"] == 1
    assert out["row_world"] == 1
    ref = _matrices(s_res)
    # the resident partial keeps the EKFAC-SHARD filename (no row suffix)
    assert _blocks(s_res) == [{"q0": 0, "q1": _Q}]

    # (b) 2 query blocks x 2 row shards = the four processes an 8-card layout
    #     would run per pair of cards. NONE of them may file a matrix alone.
    s_sh, cfg_sh, _ = _clone(seeded, **{"ekfac.query_chunk": 2, "ekfac.row_chunk": 3})
    done = []
    for rank in (0, 1):
        for row_rank in (0, 1):
            o = _score(s_sh, cfg_sh, rank=rank, world=2, row_rank=row_rank, row_world=2)
            done.append(o)
            if o is not done[-1]:                       # pragma: no cover
                raise AssertionError
    assert [o["finalized"] for o in done] == [False, False, False, True]
    assert done[0]["missing_blocks"] == ["0:2 rows 3:6", "2:4"]
    assert done[-1]["shape"] == [_N, _Q]
    _assert_same(ref, _matrices(s_sh), "world=2 x row_world=2")
    # four partials, each naming the rows it holds
    assert _blocks(s_sh) == [{"q0": 0, "q1": 2, "r0": 0, "r1": 3},
                            {"q0": 0, "q1": 2, "r0": 3, "r1": 6},
                            {"q0": 2, "q1": 4, "r0": 0, "r1": 3},
                            {"q0": 2, "q1": 4, "r0": 3, "r1": 6}]

    # (c) mixed: block [0:2) streamed whole-rows first (row_chunk unset), then
    #     block [2:4) row-sharded. The whole-rows partial covers its block, so
    #     no row partial of it is ever computed.
    s_mx, cfg_mx, root_mx = _clone(seeded, **{"ekfac.query_chunk": 2})
    assert _score(s_mx, cfg_mx, rank=0, world=2)["finalized"] is False
    assert _blocks(s_mx) == [{"q0": 0, "q1": 2}]
    cfg_rows = _cfg(root_mx, **{"ekfac.query_chunk": 2, "ekfac.row_chunk": 3})
    for row_rank in (0, 1):
        last = _score(s_mx, cfg_rows, rank=1, world=2, row_rank=row_rank, row_world=2)
    assert last["finalized"] is True
    _assert_same(ref, _matrices(s_mx), "one block resident + one block row-sharded")
    assert _blocks(s_mx) == [{"q0": 0, "q1": 2},
                            {"q0": 2, "q1": 4, "r0": 0, "r1": 3},
                            {"q0": 2, "q1": 4, "r0": 3, "r1": 6}]

    # (d) a row shard interrupted after its first tile, then resumed by the
    #     ordinary single-process command
    s_rs, cfg_rs, _ = _clone(seeded, **{"ekfac.row_chunk": 2})       # 3 row tiles
    part = _score(s_rs, cfg_rs, row_rank=0, row_world=3)
    assert part["finalized"] is False
    assert part["missing_blocks"] == ["0:4 rows 2:6"]
    assert _blocks(s_rs) == [{"q0": 0, "q1": _Q, "r0": 0, "r1": 2}]
    assert not s_rs.exists(K.SCORES, _spec())
    fin = _score(s_rs, cfg_rs)                    # row_world=1: fills 2:4, 4:6, files
    assert fin["finalized"] is True
    _assert_same(ref, _matrices(s_rs), "row shard interrupted and resumed")


def test_repeats_row_shard_to_an_identical_sigma_hat(seeded):
    """The σ̂ rows ARE the first M train rows, so the row axis tiles [0, M)."""
    s_res, cfg_res, _ = _clone(seeded)
    out = _repeats(s_res, cfg_res)
    assert out["shape"] == [len(_GRID), 2, 4, _Q]
    ref = s_res.load(K.REPEAT_SCORES, _spec())["scores"]

    s_sh, cfg_sh, _ = _clone(seeded, **{"ekfac.query_chunk": 2, "ekfac.row_chunk": 2})
    outs = [_repeats(s_sh, cfg_sh, rank=r, world=2, row_rank=rr, row_world=2)
            for r in (0, 1) for rr in (0, 1)]
    assert [o["finalized"] for o in outs] == [False, False, False, True]
    assert np.array_equal(ref, s_sh.load(K.REPEAT_SCORES, _spec())["scores"])
    assert _blocks(s_sh, K.REPEAT_SCORES_BLOCK) == [
        {"q0": 0, "q1": 2, "r0": 0, "r1": 2}, {"q0": 0, "q1": 2, "r0": 2, "r1": 4},
        {"q0": 2, "q1": 4, "r0": 0, "r1": 2}, {"q0": 2, "q1": 4, "r0": 2, "r1": 4}]


def test_row_partials_of_a_foreign_tiling_still_assemble(seeded):
    """A peer that chose its own row_chunk is discovered, not ignored.

    Two machines given different row_chunks still produce a legal cover as long
    as the ranges tile [0, N); the assembler reads what is on the disk rather
    than only what its own config predicts.
    """
    store, cfg, root = _clone(seeded, **{"ekfac.row_chunk": 2})
    _score(store, cfg, row_rank=0, row_world=3)                  # rows [0:2)
    other = _cfg(root, **{"ekfac.row_chunk": 4})                 # tiles [0:4), [4:6)
    out = _score(store, other, row_rank=1, row_world=2)          # rows [4:6)
    assert out["finalized"] is False and out["missing_blocks"] == ["0:4 rows 2:4"]
    third = _cfg(root, **{"ekfac.row_chunk": 2})
    assert _score(store, third, row_rank=1, row_world=3)["finalized"] is True   # rows [2:4)
    ref_store, ref_cfg, _ = _clone(seeded)
    _score(ref_store, ref_cfg)
    _assert_same(_matrices(ref_store), _matrices(store), "3 foreign row tilings")


# --------------------------------------------------------------------------- #
# guard rails
# --------------------------------------------------------------------------- #

def test_overlapping_row_partials_are_a_hard_error(seeded):
    """Two partials claiming row i are two answers for it — refuse to pick."""
    store, cfg, root = _clone(seeded, **{"ekfac.row_chunk": 2})
    _score(store, cfg, row_rank=0, row_world=3)                  # rows [0:2)
    wide = _cfg(root, **{"ekfac.row_chunk": 3})                  # tiles [0:3), [3:6)
    # rows [0:3) now double-claim rows 0-1; the assembly step refuses, naming
    # the ranges (the compute step cannot: a peer's partial may land at any time)
    with pytest.raises(ValueError, match="overlaps"):
        _score(store, wide, row_rank=0, row_world=2)
    with pytest.raises(ValueError, match="overlaps"):
        _score(store, wide, row_rank=1, row_world=2)
    assert not store.exists(K.SCORES, _spec()), "the refused run must file nothing"


def test_a_row_partial_of_a_different_caliber_refuses_to_resume(seeded):
    """Identity mismatch is a hard error on the row axis too."""
    store, cfg, root = _clone(seeded, **{"ekfac.row_chunk": 3})
    _score(store, cfg, row_rank=0, row_world=2)
    other = _cfg(root, **{"ekfac.row_chunk": 3, "ekfac.mc_loss": 3})
    with pytest.raises(ValueError, match="mc_loss"):
        _score(store, other)
    assert not store.exists(K.SCORES, _spec())
    assert _score(store, cfg)["finalized"] is True         # the right caliber resumes


def test_a_coverage_gap_files_nothing_and_names_the_missing_rows(seeded):
    """A gap is a peer still running, not a corrupt cell: report, do not raise."""
    store, cfg, _ = _clone(seeded, **{"ekfac.row_chunk": 2})
    out = _score(store, cfg, row_rank=1, row_world=3)           # only rows [2:4)
    assert out["finalized"] is False
    assert out["missing_blocks"] == ["0:4 rows 0:2", "0:4 rows 4:6"]
    assert not store.exists(K.SCORES, _spec())
    assert not any(store.exists(K.SCORES_LAMBDA, _spec(), lam=d) for d in _GRID)


def test_rescore_and_force_are_refused_while_row_sharding(seeded):
    store, cfg, _ = _clone(seeded, **{"ekfac.row_chunk": 3})
    _score(store, cfg)
    with pytest.raises(ValueError, match="row-world"):
        _score(store, cfg, rescore=True, row_world=2, row_rank=0)
    with pytest.raises(ValueError, match="row-world"):
        _repeats(store, cfg, force=True, row_world=2, row_rank=0)


def test_bad_row_rank_world_is_refused(seeded):
    store, cfg, _ = _clone(seeded, **{"ekfac.row_chunk": 3})
    for row_rank, row_world in ((2, 2), (-1, 2), (0, 0)):
        with pytest.raises(ValueError, match="row-rank"):
            _score(store, cfg, row_rank=row_rank, row_world=row_world)


def test_the_assembled_artifacts_record_per_partial_provenance(seeded):
    """README pit 16: a merged matrix must say which host/torch/grad_chunk
    produced which (query range, row range) — under iid sampling a mixed
    grad_chunk moves the numbers, and only the provenance can reveal it."""
    store, cfg, _ = _clone(seeded, **{"ekfac.query_chunk": 2, "ekfac.row_chunk": 3})
    for row_rank in (0, 1):
        out = _score(store, cfg, row_rank=row_rank, row_world=2)
    assert out["finalized"] is True
    prov = store.load(K.SCORES_META, _spec())["partials"]
    assert [(p["q"], p["r"]) for p in prov] == [
        ("0:2", "0:3"), ("0:2", "3:6"), ("2:4", "0:3"), ("2:4", "3:6")]
    assert all(p["grad_chunk"] == 2 and p["host"] and p["torch"] for p in prov)

    rstore, rcfg, _ = _clone(seeded, **{"ekfac.row_chunk": 2})
    for row_rank in (0, 1):
        _repeats(rstore, rcfg, row_rank=row_rank, row_world=2)
    rprov = json.loads(str(rstore.load(K.REPEAT_SCORES, _spec())["partials"]))
    assert [(p["q"], p["r"]) for p in rprov] == [("0:4", "0:2"), ("0:4", "2:4")]


def test_partials_written_before_the_row_axis_existed_still_assemble(seeded):
    """A val shard may already be mid-flight on the EKFAC-SHARD layout when this
    lands: its (q, 25-block) partials carry no r0/r1 and no host/torch stamp.
    They must keep addressing, keep passing the identity check, and assemble to
    the same bits — nothing on disk needs a rerun.
    """
    store, cfg, root = _clone(seeded, **{"ekfac.query_chunk": 2})
    _score(store, cfg)
    ref = _matrices(store)
    # rewrite each partial with exactly the key set EKFAC-SHARD wrote
    blocks_dir = Path(root) / "data" / "scores" / "ekfac_if" / _DS / "seed_42" / "blocks"
    files = sorted(blocks_dir.glob("scores_q*.npz"))
    assert [f.name for f in files] == ["scores_q0_2.npz", "scores_q2_4.npz"]
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            old = {k: z[k] for k in z.files
                   if k not in ("r0", "r1", "hostname", "torch_version")}
        np.savez(f, **old)
    _unfile(store)          # so the run has to re-assemble from the partials

    out = _score(store, cfg)
    assert out["finalized"] is True and out["blocks_total"] == 2
    _assert_same(ref, _matrices(store), "EKFAC-SHARD partials re-assembled")
    # they read as whole-rows partials, and their provenance degrades to "?"
    prov = store.load(K.SCORES_META, _spec())["partials"]
    assert [(p["q"], p["r"]) for p in prov] == [("0:2", "0:6"), ("2:4", "0:6")]
    assert [p["host"] for p in prov] == ["?", "?"]
    # and a row-sharded command over such a cell recomputes nothing: the
    # whole-rows partial already covers its block's rows
    _unfile(store)
    rows_cfg = _cfg(root, **{"ekfac.query_chunk": 2, "ekfac.row_chunk": 3})
    assert _score(store, rows_cfg, row_rank=0, row_world=2)["finalized"] is True
    assert _blocks(store) == [{"q0": 0, "q1": 2}, {"q0": 2, "q1": 4}]
    _assert_same(ref, _matrices(store), "row-shard command over EKFAC-SHARD partials")


def test_clean_partials_drops_row_partials_too(seeded):
    store, cfg, _ = _clone(seeded, **{"ekfac.row_chunk": 3})
    out = _score(store, cfg, clean_partials=True)
    assert out["finalized"] is True
    assert _blocks(store) == []
    assert store.exists(K.SCORES, _spec())
    assert all(store.exists(K.SCORES_LAMBDA, _spec(), lam=d) for d in _GRID)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
