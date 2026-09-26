"""Local artifact addressing, atomic writes and provenance tests."""
from __future__ import annotations

import tempfile

import numpy as np

from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB
from balds.artifacts import addressing



def test_addressing_matches_legacy_layout():
    s = RunSpec(dataset="cifar2", seed=42)
    assert addressing.relpath(K.TRAIN_FEATURES, s, feat="das") == "featurize/das/cifar2/seed_42/train_features.pt"
    assert addressing.relpath(K.ERROR_WEIGHT, s, feat="das") == "featurize/das/cifar2/seed_42/error_train.npy"
    assert addressing.relpath(K.QUERY_FEATURES, s, feat="das") == "featurize/das/cifar2/seed_42/query_features_gen.pt"
    assert addressing.relpath(K.QUERY_FEATURES, s.with_(query_type="val"), feat="das") == "featurize/das/cifar2/seed_42/query_features_val.pt"
    assert addressing.relpath(K.FEATURE_META, s, feat="das") == "featurize/das/cifar2/seed_42/meta.json"
    assert addressing.relpath(K.GT_MATRIX, s) == "results/gt_matrix_fm_cifar2_seed_42.npy"
    assert addressing.relpath(K.GT_MATRIX, s.with_(query_type="val")) == "results/gt_matrix_fm_cifar2_val_seed_42.npy"
    assert addressing.relpath(K.SUBSET_MASKS, s) == "subsets/cifar2_masks.pkl"
    assert addressing.relpath(K.CHECKPOINT, s) == "checkpoints/cifar2/cfm_cond/seed_42/final.pt"
    assert addressing.relpath(K.GENERATION, s) == "generations/cifar2/cfm_cond/seed_42/samples.pt"
    sm = s.with_(method="das")
    assert addressing.relpath(K.SCORES, sm) == "scores/das/cifar2/seed_42/scores.npy"
    assert addressing.relpath(K.SCORES, sm.with_(query_type="val")) == "scores/das/cifar2_val/seed_42/scores.npy"
    assert addressing.relpath(K.SCORES_LAMBDA, sm, lam=1000.0) == "scores/das/cifar2/seed_42/scores_lambda_1.00e+03.npy"


def _store(tmp):
    return ArtifactStore(
        ManifestDB(f"{tmp}/manifest.db"),
        [LocalBackend(tmp)],
        code_version="test",
    )


def test_local_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(tmp)
        spec = RunSpec(dataset="cifar2", seed=999)
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        entry = store.save(K.GT_MATRIX, spec, arr, upstream=("deadbeef",))
        assert entry.codec == "npy" and entry.code_version == "test" and len(entry.blob_hash) == 64
        assert store.exists(K.GT_MATRIX, spec)
        got = store.load(K.GT_MATRIX, spec)
        assert np.array_equal(arr, got)
        assert store.verify_local() == []                       # integrity holds
        assert store.manifest.get(store._item_key(K.GT_MATRIX, spec, {})).upstream == ("deadbeef",)




def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASSED {len(fns)} store tests")


if __name__ == "__main__":
    _run_all()
