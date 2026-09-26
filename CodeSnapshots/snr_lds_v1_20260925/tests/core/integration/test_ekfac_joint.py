"""Native vs joint gen/val scoring, real tiny UNet, CPU only."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from balds.workflows.curvature import EkfacScoreUseCase
from balds.workflows.curvature_joint import JointEkfacScoreUseCase
import balds.workflows.curvature_joint as joint
from balds.schema.artifact import ArtifactKind as K
from test_ekfac_row_sharding import seeded, _clone, _spec, _DS, _SEED, _GRID, _N, _Q


def make(seeded_root):
    store, cfg, root = _clone(seeded_root, **{
        "ekfac.query_chunk": 2, "ekfac.row_chunk": 2,
        f"lds.Q_by_dataset.{_DS}": _Q})
    store.save(K.GT_MATRIX, _spec().with_(query_type="val"),
               np.random.RandomState(8).rand(4, _Q))
    return store, cfg


def matrices(store, track):
    spec = _spec().with_(query_type=track)
    return {"selected": store.load(K.SCORES, spec), **{
        str(d): store.load(K.SCORES_LAMBDA, spec, lam=d) for d in _GRID}}


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_joint_matches_native_and_computes_each_train_row_once(seeded, monkeypatch, dtype):
    reference, cfg_ref = make(seeded)
    cfg_ref["ekfac"]["query_coords_dtype"] = dtype
    for track in ("gen", "val"):
        EkfacScoreUseCase(reference, cfg_ref, device="cpu").run(
            _DS, _SEED, conditional=False, query_type=track)
    store, cfg = make(seeded)
    cfg["ekfac"]["query_coords_dtype"] = dtype
    uc = JointEkfacScoreUseCase(store, cfg, device="cpu")
    assert uc.prepare(_DS, _SEED, conditional=False) == {"gen": "produced", "val": "produced"}
    assert uc.prepare(_DS, _SEED, conditional=False) == {"gen": "reused", "val": "reused"}
    real = joint._train_coords
    seen = []
    def counted(*args, **kw):
        seen.append(kw["index"])
        return real(*args, **kw)
    monkeypatch.setattr(joint, "_train_coords", counted)
    monkeypatch.setattr(joint, "_query_pass", lambda *a, **kw:
                        pytest.fail("row worker must not extract queries"))
    # Reversed unit order demonstrates global row IDs, not process-local RNG.
    uc.run(_DS, _SEED, unit=1, units=2, row_chunk=2, conditional=False)
    with pytest.raises(ValueError, match="incomplete"):
        uc.assemble(_DS, _SEED, row_chunk=2, conditional=False)
    assert not store.exists(K.SCORES, _spec())
    uc.run(_DS, _SEED, unit=0, units=2, row_chunk=2, conditional=False)
    assert sorted(seen) == list(range(_N))
    seen.clear()
    assert uc.run(_DS, _SEED, unit=1, units=2, row_chunk=2,
                  conditional=False)["computed_train_rows"] == 0
    assert seen == []
    uc.assemble(_DS, _SEED, row_chunk=2, conditional=False)
    for track in ("gen", "val"):
        for key, value in matrices(reference, track).items():
            assert np.array_equal(value, matrices(store, track)[key]), (dtype, track, key)
        meta = store.load(K.SCORES_META, _spec().with_(query_type=track))
        assert len(meta["partials"]) == 6


def test_interrupted_tile_keeps_published_blocks_and_resumes(seeded, monkeypatch):
    store, cfg = make(seeded)
    uc = JointEkfacScoreUseCase(store, cfg, device="cpu")
    uc.prepare(_DS, _SEED, conditional=False)
    save = store.save_atomic
    published = []
    def interrupt(kind, spec, payload, **kw):
        if kind == K.SCORES_BLOCK:
            if published:
                raise RuntimeError("simulated interruption")
            published.append((spec, kw, payload["block"].copy()))
        return save(kind, spec, payload, **kw)
    monkeypatch.setattr(store, "save_atomic", interrupt)
    with pytest.raises(RuntimeError, match="simulated"):
        uc.run(_DS, _SEED, unit=0, units=1, row_chunk=2, conditional=False)
    monkeypatch.setattr(store, "save_atomic", save)
    uc.run(_DS, _SEED, unit=0, units=1, row_chunk=2, conditional=False)
    spec, key, expected = published[0]
    assert np.array_equal(store.load(K.SCORES_BLOCK, spec, **key)["block"], expected)
    uc.assemble(_DS, _SEED, row_chunk=2, conditional=False)
    cfg["ekfac"]["grad_chunk"] = 1
    with pytest.raises(ValueError, match="grad_chunk"):
        uc.run(_DS, _SEED, unit=0, units=1, row_chunk=2, conditional=False)




