"""Per-query-block partials for the curvature streams (EKFAC-SHARD, 2026-09-09).

A 40 h `balds ekfac score` used to hold the whole (N, Q) matrix in RAM and write
nothing until the last training row: an interruption lost everything, and one
cell could not be split across cards. Each query block is now filed as it
completes, a restart resumes past the blocks already on disk, and
``--rank/--world`` hands different blocks to different processes/machines.

The claim that has to be nailed down is that none of this is an estimator
change — the block count is a residency axis (every MC draw is keyed by the
GLOBAL query/train index), so this suite asserts array-level identity between

    resident  vs  sharded (two processes' worth, world=2)  vs  interrupted+resumed

for the selected ``scores`` AND every ``scores_lambda``, plus the same for the
σ̂ ``repeat_scores``. The guard rails get their own tests: a partial whose
caliber disagrees refuses to be resumed (rather than silently mixing calibers
into one matrix), and a run that has NOT finished every block files nothing at
the whole-matrix addresses.

CPU only, a base_ch=8 UNet, MC budgets of 1-2 draws: seconds.
"""
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

_DS = "_ekfac_block_toy"
_N, _M, _Q = 6, 4, 4
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
        "tweedie.repeats": 2, "tweedie.sigma_samples": 3,
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
    """One fitted cell (checkpoint + generation + masks + GT + factors), built once.

    Every variant below runs on a byte-copy of this tree, so the ONLY difference
    between the resident and the sharded run is how the query axis was tiled —
    no re-fitting, no second draw of the toy model.
    """
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
    """A fresh store over a byte-copy of the fitted cell."""
    dst = tempfile.mkdtemp()
    shutil.copytree(f"{seeded_root}/data", f"{dst}/data")
    return _store(dst), _cfg(dst, **over), dst


def _score(store, cfg, **kw):
    return EkfacScoreUseCase(store, cfg, device="cpu").run(
        _DS, _SEED, query_type="gen", process="cfm", conditional=False, **kw)


def _matrices(store, method: str = "ekfac_if") -> dict:
    spec = _spec(method)
    return {"scores": store.load(K.SCORES, spec),
            **{f"lam{d:g}": store.load(K.SCORES_LAMBDA, spec, lam=d) for d in _GRID}}


def _assert_same(a: dict, b: dict, what: str):
    assert set(a) == set(b)
    for k in a:
        assert np.array_equal(a[k], b[k]), f"{what}: {k} moved"


# --------------------------------------------------------------------------- #
# the load-bearing claim: residency does not touch a single bit
# --------------------------------------------------------------------------- #

def test_resident_sharded_and_resumed_file_identical_matrices(seeded):
    # (a) resident: one block, the pre-EKFAC-SHARD shape of the run
    s_res, cfg_res, _ = _clone(seeded)
    out = _score(s_res, cfg_res)
    assert out["shape"] == [_N, _Q] and out["blocks_total"] == 1
    ref = _matrices(s_res)

    # (b) sharded across two processes: query_chunk=1 -> 4 blocks, rank r takes
    #     blocks r, r+2. Neither shard may file anything on its own.
    s_sh, cfg_sh, _ = _clone(seeded, **{"ekfac.query_chunk": 1})
    first = _score(s_sh, cfg_sh, rank=0, world=2)
    assert first["finalized"] is False and first["blocks_done"] == 2
    assert first["missing_blocks"] == ["1:2", "3:4"]
    assert not s_sh.exists(K.SCORES, _spec()), "a shard filed a partial matrix"
    assert not any(s_sh.exists(K.SCORES_LAMBDA, _spec(), lam=d) for d in _GRID)
    second = _score(s_sh, cfg_sh, rank=1, world=2)          # last finisher assembles
    assert second["finalized"] is True and second["shape"] == [_N, _Q]
    _assert_same(ref, _matrices(s_sh), "world=2 sharding")

    # (c) interrupted, then resumed: rank 0 of 4 leaves exactly one block on
    #     disk; the ordinary single-process command picks up where it stopped.
    s_rs, cfg_rs, _ = _clone(seeded, **{"ekfac.query_chunk": 1})
    part = _score(s_rs, cfg_rs, rank=0, world=4)
    assert part["finalized"] is False and part["blocks_done"] == 1
    assert s_rs.exists(K.SCORES_BLOCK, _spec(), q0=0, q1=1)
    assert not s_rs.exists(K.SCORES, _spec())
    done = _score(s_rs, cfg_rs)                            # world=1: resume + finalize
    assert done["finalized"] is True
    _assert_same(ref, _matrices(s_rs), "resume after interruption")

    # every block partial is a real store artifact under the cell's blocks/ dir
    assert [b["q0"] for b in s_rs.local_blocks(K.SCORES_BLOCK, _spec())] == [0, 1, 2, 3]


def test_repeats_shard_and_resume_to_an_identical_sigma_hat(seeded):
    """`ekfac repeats` carries the same (D, R, M, Q) structure and the same rule."""
    s_res, cfg_res, _ = _clone(seeded)
    out = EkfacRepeatsUseCase(s_res, cfg_res, device="cpu").run(
        _DS, _SEED, query_type="gen", process="cfm", conditional=False)
    assert out["shape"] == [len(_GRID), 2, 3, _Q]
    ref = s_res.load(K.REPEAT_SCORES, _spec())["scores"]

    s_sh, cfg_sh, _ = _clone(seeded, **{"ekfac.query_chunk": 2})    # 2 blocks
    uc = EkfacRepeatsUseCase(s_sh, cfg_sh, device="cpu")
    first = uc.run(_DS, _SEED, query_type="gen", process="cfm", conditional=False,
                   rank=0, world=2)
    assert first["finalized"] is False
    assert not s_sh.exists(K.REPEAT_SCORES, _spec()), "a shard filed a partial sigma-hat"
    second = uc.run(_DS, _SEED, query_type="gen", process="cfm", conditional=False,
                    rank=1, world=2)
    assert second["finalized"] is True
    assert np.array_equal(ref, s_sh.load(K.REPEAT_SCORES, _spec())["scores"])


# --------------------------------------------------------------------------- #
# guard rails
# --------------------------------------------------------------------------- #

def test_a_partial_of_a_different_caliber_refuses_to_resume(seeded):
    """Identity mismatch is a hard error, never a silent recompute or a mix."""
    store, cfg, root = _clone(seeded, **{"ekfac.query_chunk": 1})
    _score(store, cfg, rank=0, world=4)                    # one block on disk
    other = _cfg(root, **{"ekfac.query_chunk": 1, "ekfac.mc_loss": 3})
    with pytest.raises(ValueError, match="mc_loss"):
        _score(store, other)
    assert not store.exists(K.SCORES, _spec()), "the refused run must file nothing"
    # the caliber it WAS measured under still resumes fine
    assert _score(store, cfg)["finalized"] is True


def test_clean_partials_drops_the_blocks_and_keeps_the_matrices(seeded):
    store, cfg, _ = _clone(seeded, **{"ekfac.query_chunk": 2})
    out = _score(store, cfg, clean_partials=True)
    assert out["finalized"] is True
    assert store.local_blocks(K.SCORES_BLOCK, _spec()) == []
    assert store.exists(K.SCORES, _spec())
    assert all(store.exists(K.SCORES_LAMBDA, _spec(), lam=d) for d in _GRID)


def test_rescore_restreams_and_is_refused_while_sharding(seeded):
    store, cfg, _ = _clone(seeded, **{"ekfac.query_chunk": 2})
    _score(store, cfg)
    before = _matrices(store)
    with pytest.raises(ValueError, match="rescore"):
        _score(store, cfg, rescore=True, world=2, rank=0)
    # a single-process rescore re-streams (same seeds -> same numbers) rather
    # than assembling the stale partials
    assert _score(store, cfg, rescore=True)["finalized"] is True
    _assert_same(before, _matrices(store), "single-process rescore")


def test_bad_rank_world_is_refused(seeded):
    store, cfg, _ = _clone(seeded, **{"ekfac.query_chunk": 1})
    for rank, world in ((2, 2), (-1, 2), (0, 0)):
        with pytest.raises(ValueError, match="rank"):
            _score(store, cfg, rank=rank, world=world)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
