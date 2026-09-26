"""EK-FAC use-case integration ([n]) — full fit -> score -> evaluate loop on a
real (temporary) store, CPU only.

Covers what the unit suite (test_ekfac.py) cannot: the store plumbing (factors
artifact save/load/skip through the pt codec), the damping-grid selection via
the pipeline's λ-selector, the SCORES artifact landing under method `ekfac_if`,
and `evaluate` consuming it unchanged. Runs in seconds: a toy dataset registered
just for this test, a base_ch=8 UNet, and MC budgets of 1-2 draws.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows.config import load_config  # noqa: E402
from balds.workflows.curvature import EkfacFitUseCase, EkfacScoreUseCase  # noqa: E402
from balds.workflows.usecases import EvaluateUseCase  # noqa: E402
from balds.schema.artifact import ArtifactKind as K  # noqa: E402
from balds.schema.registry import DATASETS  # noqa: E402
from balds.schema.runspec import RunSpec  # noqa: E402
from balds.data.base import ImageDataset  # noqa: E402
from balds.models import checkpoint_state  # noqa: E402
from balds.models.unet import UNetCFM  # noqa: E402
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB  # noqa: E402

_DS = "_ekfac_toy"
_N, _M, _Q = 6, 4, 3


def _toy_loader(data_dir, **_):
    g = torch.Generator().manual_seed(0)
    train = ImageDataset(torch.randn(_N, 3, 32, 32, generator=g).clamp(-1, 1),
                         torch.zeros(_N, dtype=torch.long))
    test = ImageDataset(torch.randn(4, 3, 32, 32, generator=g).clamp(-1, 1),
                        torch.zeros(4, dtype=torch.long))
    return train, test


if not DATASETS.has(_DS):
    DATASETS.add(_DS, _toy_loader)


@pytest.fixture(scope="module")
def env():
    root = tempfile.mkdtemp()
    store = ArtifactStore(ManifestDB(f"{root}/m.db"), [LocalBackend(f"{root}/data")],
                             code_version="test")
    cfg = load_config({
        "storage.data_root": f"{root}/data",
        "ekfac.fit_epochs": 1, "ekfac.eig_epochs": 1,
        "ekfac.fit_batch_size": 3, "ekfac.eig_batch_size": 3,
        "ekfac.mc_loss": 2, "ekfac.mc_measurement": 2, "ekfac.grad_chunk": 2,
        "ekfac.query_coords_dtype": "float32",
    })
    cfg["ekfac"]["damping_grid"] = [1e-8, 1e-4]

    base = RunSpec(dataset=_DS, seed=42, process="cfm", conditional=False)
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
    qspec = base.with_(query_type="gen")
    store.save(K.GT_MATRIX, qspec, np.random.RandomState(7).rand(_M, _Q))
    return store, cfg


def test_fit_then_skip(env):
    store, cfg = env
    fit = EkfacFitUseCase(store, cfg, device="cpu")
    out = fit.run(_DS, 42, process="cfm", conditional=False)
    assert out.get("layers", 0) > 0 and out["n_params"] > 0
    base = RunSpec(dataset=_DS, seed=42, process="cfm", conditional=False)
    assert store.exists(K.EKFAC_FACTORS, base)
    assert fit.run(_DS, 42, process="cfm", conditional=False) == {"skipped": True}


def test_score_selects_damping_and_saves_scores(env):
    store, cfg = env
    EkfacFitUseCase(store, cfg, device="cpu").run(_DS, 42, process="cfm", conditional=False)
    sc = EkfacScoreUseCase(store, cfg, device="cpu")
    out = sc.run(_DS, 42, query_type="gen", process="cfm", conditional=False)
    assert out["shape"] == [_N, _Q]
    assert out["best_damping"] in cfg["ekfac"]["damping_grid"]
    assert set(out["per_damping"]) == {f"{d:g}" for d in cfg["ekfac"]["damping_grid"]}
    spec = RunSpec(dataset=_DS, seed=42, method="ekfac_if", query_type="gen",
                   process="cfm", conditional=False)
    scores = store.load(K.SCORES, spec)
    assert scores.shape == (_N, _Q) and np.isfinite(scores).all()
    # idempotent skip on the score side too
    assert sc.run(_DS, 42, query_type="gen", process="cfm",
                  conditional=False) == {"skipped": True}


def test_evaluate_consumes_ekfac_scores(env):
    store, cfg = env
    EkfacFitUseCase(store, cfg, device="cpu").run(_DS, 42, process="cfm", conditional=False)
    EkfacScoreUseCase(store, cfg, device="cpu").run(_DS, 42, query_type="gen",
                                                   process="cfm", conditional=False)
    out = EvaluateUseCase(store, cfg).run("ekfac_if", _DS, 42, query_type="gen",
                                          process="cfm", conditional=False)
    assert np.isfinite(out["mean_lds"]) and out["n_queries"] == _Q










if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
