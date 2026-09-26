"""§6.2-14 GT seed decoupling — chain/gt_avg integration, CPU only.

The adjudication: the subset-chain seed and the query seed become two
independent knobs. What must hold, and is pinned here on a tiny real store:

* ``--chain R`` scores chain R's subset checkpoints on identity S's queries,
  under a chain-keyed artifact; chain == seed normalises to the LEGACY name
  (diagonal single source, no duplicate files);
* the val track refuses ``--chain`` (chain R's own val losses ARE that
  computation — the zero-new-compute half of the adjudication);
* ``gt_avg`` (``compute_gt(chains=...)``) averages chains x ζ and writes the
  standard ``gt_matrix`` — gen reads diagonal + cross files, val reads each
  chain's own val losses; it is strict about missing cells;
* the sharded (--rank/--world) path with a chain reassembles to exactly the
  single-process matrix.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows.config import load_config  # noqa: E402
from balds.workflows.subsets import SubsetsUseCase  # noqa: E402
from balds.schema.artifact import ArtifactKind as K  # noqa: E402
from balds.schema.registry import DATASETS  # noqa: E402
from balds.schema.runspec import RunSpec  # noqa: E402
from balds.data.base import ImageDataset  # noqa: E402
from balds.models import checkpoint_state  # noqa: E402
from balds.models.unet import UNetCFM  # noqa: E402
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB  # noqa: E402

_DS = "_gtchain_toy"
_M, _Q = 2, 2
_CHAINS = (42, 123)


def _toy_loader(data_dir, **_):
    g = torch.Generator().manual_seed(0)
    train = ImageDataset(torch.randn(6, 3, 32, 32, generator=g).clamp(-1, 1),
                         torch.tensor([0, 1, 0, 1, 0, 1]))
    test = ImageDataset(torch.randn(4, 3, 32, 32, generator=g).clamp(-1, 1),
                        torch.tensor([0, 0, 1, 1]))
    return train, test


if not DATASETS.has(_DS):
    DATASETS.add(_DS, _toy_loader)


@pytest.fixture(scope="module")
def env():
    root = tempfile.mkdtemp()
    store = ArtifactStore(ManifestDB(f"{root}/m.db"), [LocalBackend(f"{root}/data")],
                             code_version="test")
    cfg = load_config({"storage.data_root": f"{root}/data"})
    cfg["lds"]["T_avg"] = 4
    cfg["lds"]["Q"] = _Q

    base = RunSpec(dataset=_DS, seed=42, process="cfm", conditional=False)
    store.save(K.SUBSET_MASKS, base,
               [np.random.RandomState(m).rand(6) < 0.5 for m in range(_M)])
    g = torch.Generator().manual_seed(9)
    store.save(K.GENERATION, base,
               {"samples": torch.randn(_Q, 3, 32, 32, generator=g).clamp(-1, 1),
                "labels": torch.zeros(_Q, dtype=torch.long),
                "Q": _Q, "gen_seed": 0, "ode_steps": 2, "conditional": False})
    # two subset chains over the SAME masks: distinct tiny models per (chain, m)
    for chain in _CHAINS:
        cspec = base.with_(seed=chain)
        for m in range(_M):
            torch.manual_seed(chain * 10 + m)
            model = UNetCFM(base_ch=8, num_classes=None)
            store.save(K.SUBSET_CHECKPOINT, cspec,
                       checkpoint_state(model, base_ch=8, num_classes=None, step=1,
                                        extra={"model_type": "cfm", "conditional": False}),
                       m=m)
    uc = SubsetsUseCase(store, cfg, device="cpu")
    return store, cfg, uc


def _spec(seed, qt="gen"):
    return RunSpec(dataset=_DS, seed=seed, process="cfm", conditional=False,
                   query_type=qt)


def test_diagonal_chain_normalises_to_legacy_name(env):
    store, cfg, uc = env
    out = uc.compute_losses(_DS, 42, e_seed=0, chain=42, conditional=False)
    assert out.get("chain") is None
    assert store.exists(K.GT_LOSSES, _spec(42), eseed=0)
    assert not store.exists(K.GT_LOSSES, _spec(42), eseed=0, chain=42)


def test_cross_chain_writes_chain_keyed_file_with_different_content(env):
    store, cfg, uc = env
    uc.compute_losses(_DS, 42, e_seed=0, conditional=False)              # diagonal
    out = uc.compute_losses(_DS, 42, e_seed=0, chain=123, conditional=False)
    assert out["chain"] == 123 and out["shape"] == [_M, _Q]
    diag = store.load(K.GT_LOSSES, _spec(42), eseed=0)
    cross = store.load(K.GT_LOSSES, _spec(42), eseed=0, chain=123)
    assert cross.shape == diag.shape
    assert not np.allclose(cross, diag), "different chains must give different losses"
    # idempotent skip
    assert uc.compute_losses(_DS, 42, e_seed=0, chain=123,
                             conditional=False) == {"skipped": True}


def test_val_track_refuses_chain(env):
    _, _, uc = env
    with pytest.raises(ValueError, match="val"):
        uc.compute_losses(_DS, 42, query_type="val", e_seed=0, chain=123,
                          conditional=False)


def test_gt_avg_gen_is_mean_of_diagonal_and_cross(env):
    store, cfg, uc = env
    uc.compute_losses(_DS, 42, e_seed=0, conditional=False)
    uc.compute_losses(_DS, 42, e_seed=0, chain=123, conditional=False)
    out = uc.compute_gt(_DS, 42, chains=[42, 123], e_seeds=[0], conditional=False)
    assert out["gt_avg"] is True and out["chains"] == [42, 123]
    gt = store.load(K.GT_MATRIX, _spec(42))
    diag = store.load(K.GT_LOSSES, _spec(42), eseed=0)
    cross = store.load(K.GT_LOSSES, _spec(42), eseed=0, chain=123)
    assert np.allclose(gt, (diag + cross) / 2)


def test_gt_avg_val_reads_each_chains_own_losses(env):
    store, cfg, uc = env
    for chain in _CHAINS:                       # the zero-new-compute half
        uc.compute_losses(_DS, chain, query_type="val", e_seed=0, conditional=False)
    out = uc.compute_gt(_DS, 42, query_type="val", chains=[42, 123], e_seeds=[0],
                        conditional=False)
    assert out["gt_avg"] is True
    gt = store.load(K.GT_MATRIX, _spec(42, "val"))
    v42 = store.load(K.GT_LOSSES, _spec(42, "val"), eseed=0)
    v123 = store.load(K.GT_LOSSES, _spec(123, "val"), eseed=0)
    assert np.allclose(gt, (v42 + v123) / 2)
    # no cross-keyed val artifact was ever created
    assert not store.exists(K.GT_LOSSES, _spec(42, "val"), eseed=0, chain=123)


def test_gt_avg_is_strict_about_missing_cells(env):
    _, _, uc = env
    with pytest.raises(FileNotFoundError, match="chain"):
        uc.compute_gt(_DS, 123, chains=[42, 123], e_seeds=[0], conditional=False)


def test_gt_avg_refuses_replica(env):
    _, _, uc = env
    with pytest.raises(ValueError, match="replica"):
        uc.compute_gt(_DS, 42, chains=[42, 123], e_seeds=[0], replica=1,
                      conditional=False)


def test_sharded_cross_chain_matches_single_process(env):
    store, cfg, uc = env
    for rank in range(2):
        out = uc.compute_losses(_DS, 42, e_seed=1, chain=123, rank=rank, world=2,
                                conditional=False)
    assert out["finalized"] is True
    sharded = store.load(K.GT_LOSSES, _spec(42), eseed=1, chain=123).copy()
    uc.compute_losses(_DS, 42, e_seed=1, chain=123, conditional=False, force=True)
    single = store.load(K.GT_LOSSES, _spec(42), eseed=1, chain=123)
    assert np.array_equal(sharded, single)
    assert store.exists(K.GT_LOSSROW, _spec(42), eseed=1, m=0, chain=123)


def test_legacy_compute_gt_unchanged(env):
    store, cfg, uc = env
    uc.compute_losses(_DS, 42, e_seed=0, conditional=False)
    out = uc.compute_gt(_DS, 42, e_seeds=[0], conditional=False)
    assert "gt_avg" not in out
    gt = store.load(K.GT_MATRIX, _spec(42))
    assert np.allclose(gt, store.load(K.GT_LOSSES, _spec(42), eseed=0))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
