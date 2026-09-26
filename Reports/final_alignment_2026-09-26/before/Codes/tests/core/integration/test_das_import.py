"""DAS-archive import (§6.2-33 ⑦) — end to end on a synthetic 20-sample archive.

What has to hold, and is pinned here rather than left to the real 75 GB tree:

* the READER's format conventions — headerless float32 dumps, the error pickle's
  block axis, and above all the ground-truth pickle's ``(timestep, query)``
  orientation. That last one is the archive's single silent trap: the four
  pickled blocks are the query BATCH dimension, and averaging the wrong axis
  produces a full-rank plausible matrix whose LDS is ~0 (d2 report §5). The
  fixtures below make the two readings numerically different, so a transposed
  reader fails instead of quietly scoring nothing;
* the GT CALIBER — GT_LOSSES per noise seed is the mean over retrain replicas,
  GT_MATRIX the mean over noise seeds, both unweighted, so ``balds subsets gt``
  re-derives the matrix from the losses this importer wrote;
* the e_n caliber is THIS repo's (sqrt -> normalise -> mean), not DAS's;
* idempotence: a second run writes nothing;
* the artifact addresses, including the golden train-features path.
"""
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows import das_import as di  # noqa: E402
from balds.workflows.config import load_config, resolve_M, resolve_Q  # noqa: E402
from balds.schema.artifact import ArtifactKind as K  # noqa: E402
from balds.schema.runspec import RunSpec  # noqa: E402
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB, das_archive as da  # noqa: E402

# --- synthetic archive dimensions (all small; T_FEAT stays 100 so the feat name
# --- and therefore the golden path is the production one) --------------------
N, P, SHARDS = 20, 8, da.TRAIN_SHARDS      # 20 train samples over the archive's 5 shards
NQ, M, T_GT, T_ERR = 6, 4, 3, 5            # queries per track, subsets, GT timesteps, error timesteps
REPLICAS, E_SEEDS = (0, 1), (0, 1)
TRAIN_IDS = [17488, 17769, 25061, 7882, 4701, 38681, 11482, 37495, 39588, 5729,
             101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
VAL_IDS = [7405, 5226, 1363, 6615, 7612, 7758]


def _feat_rows(tag: int, n: int) -> np.ndarray:
    """Distinguishable rows: shard/track identity in the 100s, row index in the 10s."""
    return (tag * 100.0 + np.arange(n)[:, None] * 10.0 + np.arange(P)[None, :]).astype(np.float32)


def _gt_value(m, j, e, t, q):
    """Deliberately NOT symmetric in (t, q): a transposed read gives other numbers."""
    return m * 1000.0 + j * 100.0 + e * 10.0 + t * 1.0 + q * 0.01


def _write_archive(root: Path) -> None:
    saved = root / da.SAVED_SUBDIR
    index = root / da.INDEX_SUBDIR
    (index / "lds-val").mkdir(parents=True)
    with open(index / "idx-train.pkl", "wb") as fh:
        pickle.dump(TRAIN_IDS, fh)
    with open(index / "idx-val.pkl", "wb") as fh:
        pickle.dump(VAL_IDS, fh)
    for m in range(M):
        keep = [TRAIN_IDS[i] for i in range(N) if (i + m) % 2 == 0]
        with open(index / "lds-val" / f"sub-idx-{m}.pkl", "wb") as fh:
            pickle.dump(keep, fh)

    for feat, sub in da.READOUT_DIRS.items():
        d = saved / "grad" / sub / str(di.T_FEAT)
        d.mkdir(parents=True)
        rep = di.T_FEAT if feat == "trak" else 1        # `loss` keeps one row per (sample, t)
        for i in range(SHARDS):
            _feat_rows(i, (N // SHARDS) * rep).tofile(d / f"ddpm-train-keys-{i}-{P}.npy")
        _feat_rows(50, NQ * rep).tofile(d / f"ddpm-gen-keys-{P}.npy")
        if feat != "trak":                              # the archive has no val side for `loss`
            _feat_rows(60, NQ * rep).tofile(d / f"ddpm-val-keys-{P}.npy")

    (saved / "error").mkdir(parents=True)
    err = (np.arange(T_ERR)[:, None] + 0.5 + np.arange(N)[None, :] * 0.01).astype(np.float32)
    with open(saved / "error" / f"error-{T_ERR}-train.pkl", "wb") as fh:
        pickle.dump([err[:, :12], err[:, 12:]], fh)     # blocks split the SAMPLE axis

    for m in range(M):
        for j in REPLICAS:
            d = saved / "lds-val" / f"ddpm-sub-{m}-{j}"
            d.mkdir(parents=True)
            for e in E_SEEDS:
                block = np.array([[_gt_value(m, j, e, t, q) for q in range(NQ)]
                                  for t in range(T_GT)], dtype=np.float32)
                for track in ("gen", "val"):
                    # +track offset so gen and val can never be confused
                    b = block + (0.5 if track == "val" else 0.0)
                    with open(d / f"e-{e}-{track}.pkl", "wb") as fh:
                        pickle.dump([b[:, :4], b[:, 4:]], fh)   # blocks split the QUERY axis

    from PIL import Image
    g = (saved / "gen" / "gen")
    g.mkdir(parents=True)
    for i in range(NQ):
        px = np.full((32, 32, 3), i * 10 + 3, dtype=np.uint8)
        Image.fromarray(px).save(g / f"{i}.png")


@pytest.fixture(scope="module")
def env(request):
    tmp = Path(tempfile.mkdtemp())
    _write_archive(tmp / "archive")
    store = ArtifactStore(ManifestDB(str(tmp / "m.db")),
                             [LocalBackend(str(tmp / "data"))], code_version="test")
    cfg = load_config({
        "storage.data_root": str(tmp / "data"),
        "datasets.raw_dirs.das_archive": str(tmp / "archive"),
        "lds.Q_by_dataset.cifar2_das": NQ,
    })
    mp = pytest.MonkeyPatch()
    mp.setattr(di, "M_SUBSETS", M)
    mp.setattr(di, "REPLICAS", REPLICAS)
    mp.setattr(di, "E_SEEDS", E_SEEDS)
    mp.setattr(di, "PROJ_DIM", P)
    mp.setattr(di, "T_ERROR", T_ERR)
    request.addfinalizer(mp.undo)
    request.addfinalizer(lambda: shutil.rmtree(tmp, ignore_errors=True))
    uc = di.DasImportUseCase(store, cfg)
    out = uc.run(feats=("das", "dtrak", "trak", "l1norm"))
    return {"store": store, "cfg": cfg, "uc": uc, "out": out, "root": tmp,
            "spec": RunSpec(dataset=di.DATASET, process="ddpm", seed=42, conditional=True)}


def test_root_resolution_accepts_both_forms(env):
    root = env["root"] / "archive"
    assert da.resolve_root(str(root)) == str(root.resolve())
    assert da.resolve_root(str(root / da.SAVED_SUBDIR)) == str(root.resolve())
    with pytest.raises(FileNotFoundError):
        da.resolve_root(str(env["root"] / "data"))
    print("  root resolution: archive root and saved/ sub-tree both accepted")


def test_masks_are_the_archive_subsets(env):
    masks = env["store"].load(K.SUBSET_MASKS, env["spec"])
    assert len(masks) == M and masks[0].shape == (N,) and masks[0].dtype == np.bool_
    for m in range(M):
        want = np.array([(i + m) % 2 == 0 for i in range(N)])
        assert np.array_equal(np.asarray(masks[m]), want), m
    print(f"  masks: {M} x {N} bool, position mapping via idx-train order ok")


def test_features_row_order_and_query_selection(env):
    st, spec = env["store"], env["spec"]
    want_train = np.concatenate([_feat_rows(i, N // SHARDS) for i in range(SHARDS)], 0)
    for feat in ("das", "dtrak", "l1norm"):
        name = f"{feat}_T{di.T_FEAT}"
        tr = st.load(K.TRAIN_FEATURES, spec, feat=name)
        assert isinstance(tr, torch.Tensor) and tr.dtype == torch.float32
        assert np.array_equal(tr.numpy(), want_train), feat
        for track, tag in (("gen", 50), ("val", 60)):
            q = st.load(K.QUERY_FEATURES, spec.with_(query_type=track), feat=name)
            assert np.array_equal(q.numpy(), _feat_rows(tag, NQ)), (feat, track)
    print("  features: shard concat order + both query tracks bit-identical to the dumps")


def test_trak_per_timestep_rows_are_averaged_and_val_is_absent(env):
    st, spec = env["store"], env["spec"]
    name = f"trak_T{di.T_FEAT}"
    tr = st.load(K.TRAIN_FEATURES, spec, feat=name)
    assert tr.shape == (N, P)
    want = np.concatenate([_feat_rows(i, (N // SHARDS) * di.T_FEAT)
                           .reshape(N // SHARDS, di.T_FEAT, P).mean(1) for i in range(SHARDS)], 0)
    assert np.allclose(tr.numpy(), want, atol=1e-4)
    assert env["out"]["features"]["trak"]["query_val"] == "absent from archive"
    meta = st.load(K.FEATURE_META, spec, feat=name)
    assert "UNVERIFIED" in meta["row_layout"] and meta["caveat"]
    print("  trak: (n*T, p) collapsed to (n, p); no val side; caveat recorded in meta")


def test_ground_truth_axis_and_averaging(env):
    st, spec = env["store"], env["spec"]
    for track, off in (("gen", 0.0), ("val", 0.5)):
        tspec = spec.with_(query_type=track)
        per_e = []
        for e in E_SEEDS:
            got = st.load(K.GT_LOSSES, tspec, eseed=e)
            want = np.array([[np.mean([_gt_value(m, j, e, t, q) for j in REPLICAS
                                       for t in range(T_GT)]) + off
                              for q in range(NQ)] for m in range(M)])
            assert got.shape == (M, NQ)
            assert np.allclose(got, want), (track, e)
            per_e.append(want)
        gt = st.load(K.GT_MATRIX, tspec)
        assert np.allclose(gt, np.mean(per_e, axis=0)), track
        # a query-axis average (the 2026-09-02 first-round error) would be constant
        # across q; the fixture makes that detectably different
        assert gt[:, 0].std() > 0 and abs(gt[0, 1] - gt[0, 0]) > 1e-6
    print("  GT: (timestep, query) axis honoured; losses = mean over replicas, "
          "matrix = mean over noise seeds")


def test_gt_matrix_is_reproducible_from_the_stored_losses(env):
    """`balds subsets gt --e-seeds 0,1` must re-derive exactly what we wrote."""
    from balds.workflows.subsets import SubsetsUseCase
    uc = SubsetsUseCase(env["store"], env["cfg"], device="cpu")
    before = env["store"].load(K.GT_MATRIX, env["spec"].with_(query_type="val")).copy()
    uc.compute_gt(di.DATASET, 42, process="ddpm", query_type="val",
                  e_seeds=list(E_SEEDS))
    after = env["store"].load(K.GT_MATRIX, env["spec"].with_(query_type="val"))
    assert np.array_equal(before, after)
    print("  GT matrix: identical when re-derived by SubsetsUseCase.compute_gt")


def test_error_weight_uses_this_repos_caliber(env):
    e_n = env["store"].load(K.ERROR_WEIGHT, env["spec"], feat="das")
    err = np.sqrt((np.arange(T_ERR)[None, :] + 0.5
                   + np.arange(N)[:, None] * 0.01).astype(np.float32))
    want = (err / (np.linalg.norm(err, axis=1, keepdims=True) + 1e-8)).mean(1)
    assert e_n.shape == (N,) and np.allclose(e_n, want, atol=1e-6)
    # DAS's own order (normalise -> mean -> sqrt) is a DIFFERENT number; make sure
    # we did not silently adopt it
    das_order = np.sqrt((err / (np.linalg.norm(err, axis=1, keepdims=True) + 1e-8)).mean(1))
    assert not np.allclose(e_n, das_order)
    print("  e_n: repo caliber (sqrt -> normalise -> mean), filed under feat='das'")


def test_generation_roundtrip(env):
    gen = env["store"].load(K.GENERATION, env["spec"])
    assert gen["Q"] == NQ and gen["gen_seed"] is None and gen["ode_steps"] is None
    assert gen["images_u8"].dtype == torch.uint8 and gen["images_u8"].shape == (NQ, 3, 32, 32)
    assert torch.equal(gen["labels"], torch.zeros(NQ, dtype=torch.long))
    for i in range(NQ):
        assert int(gen["images_u8"][i, 0, 0, 0]) == i * 10 + 3, i
    assert torch.allclose(gen["samples"], gen["images_u8"].float() / 127.5 - 1.0)
    print("  generations: PNGs read by index (not listdir order), u8 + [-1,1] both stored")


def test_import_is_idempotent(env):
    out = env["uc"].run(feats=("das", "dtrak", "trak", "l1norm"))
    assert out["masks"] == "skipped" and out["generation"] == "skipped"
    assert out["error_weight"] == "skipped"
    for feat, res in out["features"].items():
        for key, val in res.items():
            assert val in ("skipped", "absent from archive"), (feat, key, val)
    assert all(v == "skipped" for k, v in out["ground_truth"].items())
    print("  idempotent: second run writes nothing")


def test_golden_paths_and_per_dataset_tables(env):
    data_root = Path(env["cfg"]["storage"]["data_root"])
    golden = [
        "featurize/das_T100/cifar2_das/ddpm/seed_42/train_features.pt",
        "featurize/das_T100/cifar2_das/ddpm/seed_42/query_features_gen.pt",
        "featurize/das_T100/cifar2_das/ddpm/seed_42/query_features_val.pt",
        "featurize/das_T100/cifar2_das/ddpm/seed_42/meta.json",
        "featurize/das/cifar2_das/ddpm/seed_42/error_train.npy",
        "featurize/dtrak_T100/cifar2_das/ddpm/seed_42/train_features.pt",
        "featurize/l1norm_T100/cifar2_das/ddpm/seed_42/train_features.pt",
        "subsets/cifar2_das_masks.pkl",
        "results/gt_matrix_ddpm_cifar2_das_seed_42.npy",
        "results/gt_matrix_ddpm_cifar2_das_val_seed_42.npy",
        "results/gt_losses_ddpm_cifar2_das_seed_42_eseed_0.npy",
        "generations/cifar2_das/ddpm_cond/seed_42/samples.pt",
    ]
    missing = [p for p in golden if not (data_root / p).is_file()]
    assert not missing, missing
    assert not (data_root / "featurize/trak_T100/cifar2_das/ddpm/seed_42/"
                            "query_features_val.pt").exists()
    cfg = load_config({})
    assert resolve_M(cfg, "cifar2_das") == 64 and resolve_M(cfg, "cifar2_5k") == 64
    # the importer still files every archive subset (M=64 is applied at load, lds_rows)
    assert len(env["store"].load(K.SUBSET_MASKS, env["spec"])) == M
    assert resolve_Q(cfg, "cifar2_das") == 1000 and resolve_Q(cfg, "cifar2_5k") == 100
    print(f"  addressing: {len(golden)} golden paths present; M/Q tables per dataset ok")


def test_unknown_readout_is_refused(env):
    with pytest.raises(ValueError, match="unknown readout"):
        env["uc"].run(feats=("das", "nope"))
    print("  unknown readout refused")


def test_dataset_key_needs_the_archive_and_matches_the_index(env):
    import os
    from balds.schema.registry import DATASETS
    with pytest.raises(ValueError, match="raw_dirs.das_archive"):
        DATASETS.get(di.DATASET)("/tmp")
    cache = Path(__file__).resolve().parents[3] / "_Data" / "hf_cache"
    if not cache.is_dir():
        print("  cifar2_das loader: SKIPPED pixel check (no _Data/hf_cache here)")
        return
    from balds.workflows.common import _load_ds
    cfg = load_config({})                               # the REAL archive + cifar cache
    if not os.path.isdir(cfg["datasets"]["raw_dirs"]["das_archive"]):
        print("  cifar2_das loader: SKIPPED (no DAS archive on this machine)")
        return
    train, test = _load_ds(cfg, di.DATASET)
    root = da.resolve_root(cfg["datasets"]["raw_dirs"]["das_archive"])
    idx_train, idx_val = da.read_split_indices(root)
    assert len(train) == len(idx_train) == 5000 and len(test) == len(idx_val) == 1000
    assert set(int(v) for v in train.labels.tolist()) == set(di.CLASSES)
    # row i IS idx[i]: labels must match the raw split's labels at those indices,
    # in that order (a sorted or re-filtered loader would pass the counts above)
    from datasets import load_dataset
    raw = load_dataset("cifar10", cache_dir=cfg["datasets"]["raw_dirs"]["cifar"])
    for split, ds_obj, index in (("train", train, idx_train), ("test", test, idx_val)):
        want = [int(raw[split][int(i)]["label"]) for i in index[:200]]
        assert [int(v) for v in ds_obj.labels[:200].tolist()] == want, split
    # every mask row keeps exactly the archive subset it came from
    masks_n = [len(da.read_subset_indices(root, m)) for m in range(3)]
    assert masks_n == [2500, 2500, 2500], masks_n
    # val order is the archive's, and at the archive's own Q the repo's balanced
    # selector is the identity — so a re-featurized val track lands on these rows
    from balds.data.base import balanced_query_indices
    assert balanced_query_indices(test.labels, 1000) == list(range(1000))
    print("  cifar2_das loader: N=5000/1000 from the index files, classes {1,7}, "
          "balanced selector is the identity at Q=1000")
