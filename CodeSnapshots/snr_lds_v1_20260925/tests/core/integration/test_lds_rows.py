"""M applied at load (CODE_DAS_DDPM_ADAPTER item 8, ledger §6.2-51 ①).

cifar2_das keeps all 128 archive subsets on disk, and LDS consumers use the first
``resolve_M`` = 64 through ``config.lds_rows``. What has to hold:

* evaluate on a 128-row cifar2_das-like store equals the manual masks[:64]/gt[:64]
  computation, and differs from the pooled 128-row number (the fixture carries a
  second-batch GT offset, as the archive does);
* on a platform whose artifacts hold exactly M rows the result is byte-identical
  to the pre-change arithmetic (all rows);
* the filed artifacts are not modified by reading them.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows.config import load_config, lds_rows, resolve_M  # noqa: E402
from balds.workflows.usecases import EvaluateUseCase  # noqa: E402
from balds.schema.artifact import ArtifactKind as K  # noqa: E402
from balds.schema.runspec import RunSpec  # noqa: E402
from balds.evaluation.lds import compute_lds, predicted_influence  # noqa: E402
from balds.evaluation.stats import bootstrap_ci  # noqa: E402
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB  # noqa: E402

N, Q = 40, 6
METHOD = "dtrak_T100"


def _store(tmp: Path):
    return ArtifactStore(ManifestDB(str(tmp / "m.db")), [LocalBackend(str(tmp / "data"))],
                            code_version="test")


def _file(store, dataset: str, n_subsets: int, *, batch_offset: float):
    rng = np.random.RandomState(0)
    spec = RunSpec(dataset=dataset, seed=42, process="ddpm", conditional=True,
                   method=METHOD, query_type="gen")
    masks = [rng.rand(N) < 0.5 for _ in range(n_subsets)]
    gt = rng.rand(n_subsets, Q)
    gt[64:] += batch_offset                    # second batch: a per-query constant
    scores = rng.randn(N, Q).astype(np.float32)
    store.save(K.SUBSET_MASKS, spec, masks)
    store.save(K.GT_MATRIX, spec, gt)
    store.save(K.SCORES, spec, scores)
    return spec, masks, gt, scores


def _manual(scores, masks, gt):
    """The pre-change EvaluateUseCase arithmetic, verbatim."""
    subset_idx = list(range(min(gt.shape[0], len(masks))))
    pred = predicted_influence(scores, masks, subset_idx)
    per_query, mean_lds = compute_lds(gt[:len(subset_idx)], pred)
    mean, lo, hi = bootstrap_ci(per_query)
    return {"mean_lds": mean_lds, "ci_lo": lo, "ci_hi": hi, "n_queries": len(per_query)}


def test_lds_rows_is_a_prefix_and_identity_at_M():
    cfg = load_config({})
    m128, g128 = list(range(128)), np.arange(128 * 2).reshape(128, 2)
    masks, gt = lds_rows(cfg, "cifar2_das", m128, g128)
    assert resolve_M(cfg, "cifar2_das") == 64 and masks == list(range(64))
    assert np.array_equal(gt, g128[:64])
    m64, g64 = list(range(64)), np.arange(64 * 2).reshape(64, 2)
    masks, gt = lds_rows(cfg, "cifar2_5k", m64, g64)
    assert masks == m64 and np.array_equal(gt, g64)
    few, gfew = list(range(4)), np.zeros((4, 2))              # toy stores: fewer than M
    assert lds_rows(cfg, "cifar2_das", few, gfew)[0] == few
    print("  lds_rows: first M on cifar2_das; identity at M and below")


def test_evaluate_uses_first_64_of_128_without_touching_the_files():
    tmp = Path(tempfile.mkdtemp())
    store = _store(tmp)
    cfg = load_config({"storage.data_root": str(tmp / "data")})
    spec, masks, gt, scores = _file(store, "cifar2_das", 128, batch_offset=14.0)
    out = EvaluateUseCase(store, cfg).run(METHOD, "cifar2_das", 42, query_type="gen",
                                          process="ddpm", conditional=True)
    assert out == _manual(scores, masks[:64], gt[:64])
    assert out["mean_lds"] != _manual(scores, masks, gt)["mean_lds"]
    assert len(store.load(K.SUBSET_MASKS, spec)) == 128
    assert np.array_equal(store.load(K.GT_MATRIX, spec), gt)
    print(f"  evaluate cifar2_das: M=64 LDS {out['mean_lds']:.4f} (pooled 128 would be "
          f"{_manual(scores, masks, gt)['mean_lds']:.4f}); files still 128 rows")


def test_exact_M_platform_is_byte_identical_to_the_old_arithmetic():
    tmp = Path(tempfile.mkdtemp())
    store = _store(tmp)
    cfg = load_config({"storage.data_root": str(tmp / "data")})
    _, masks, gt, scores = _file(store, "cifar2_5k", 64, batch_offset=0.0)
    out = EvaluateUseCase(store, cfg).run(METHOD, "cifar2_5k", 42, query_type="gen",
                                          process="ddpm", conditional=True)
    want = _manual(scores, masks, gt)
    assert out.keys() == want.keys()
    for k in want:
        assert np.asarray(out[k]).tobytes() == np.asarray(want[k]).tobytes(), k
    print("  evaluate cifar2_5k (64 filed = M): byte-identical to the pre-change arithmetic")
