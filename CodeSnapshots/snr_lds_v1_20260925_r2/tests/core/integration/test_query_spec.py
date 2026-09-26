"""QUERY-SET SPEC [i] (adjudicated 2026-08-07, experiment.qmd §1.3), 2026-08-10.

Three things the formal ICLR query spec requires that the code did not do:

  * **val = class-balanced fixed indices.** Three call sites (GT losses, query
    featurization, per-protocol scoring) each independently took
    ``test.images[:Q]`` — not balanced, and free to drift apart. They now share
    one deterministic selector.
  * **gen noise is fixed per seed, not shared across seeds.** ``gen_seed`` was
    hardcoded to 123 regardless of ``--seed``, so every replicate's queries
    started from identical noise and probed nearly the same corner of the
    model's output space. It is now ``gen_seed_base + seed``.
  * **Q must not be silently mismatched.** The generation path does not encode
    Q, so an existing artifact at a different Q used to be skipped-and-reused.

Also covers the ``conditional`` plumbing that reaches these paths: unconditional
identities must resolve to the ``*_uncond`` model segment, and their generation
labels must not pretend to carry class information.

Run:  conda run -n da python Codes/tests/integration/test_query_spec.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows.config import load_config
from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.data.base import ImageDataset, balanced_query_indices, select_val_queries
from balds.artifacts.addressing import relpath


def _fake_test_split(per_class, classes=(1, 7)):
    """Deliberately class-clustered, like a filtered HF split: [:Q] is unbalanced."""
    imgs, labs = [], []
    for c in classes:
        for i in range(per_class):
            imgs.append(torch.full((3, 32, 32), float(c) + i / 1000))
            labs.append(c)
    return ImageDataset(torch.stack(imgs), torch.tensor(labs, dtype=torch.long))


def test_val_selection_is_balanced_deterministic_and_shared():
    ds = _fake_test_split(per_class=500)
    # the old behaviour, shown to be unbalanced on a clustered split
    naive = ds.labels[:100]
    assert int((naive == 1).sum()) == 100 and int((naive == 7).sum()) == 0, \
        "precondition: [:Q] on a clustered split takes one class only"

    q1 = select_val_queries(ds, 100)
    q2 = select_val_queries(ds, 100)
    counts = {int(c): int((q1.labels == c).sum()) for c in (1, 7)}
    assert counts == {1: 50, 7: 50}, counts
    assert torch.equal(q1.images, q2.images) and torch.equal(q1.labels, q2.labels), \
        "selection must be deterministic across calls (fixed indices, no RNG)"
    assert balanced_query_indices(ds.labels, 100) == sorted(balanced_query_indices(ds.labels, 100))
    print(f"  val: balanced {counts}, deterministic, replaces the unbalanced [:Q]")


def test_val_selection_scales_to_ten_classes():
    ds = _fake_test_split(per_class=200, classes=tuple(range(10)))
    q = select_val_queries(ds, 100)
    counts = {int(c): int((q.labels == c).sum()) for c in range(10)}
    assert set(counts.values()) == {10}, counts
    print("  val: cifar10 Q=100 -> 10 per class")


def test_val_selection_refuses_unbalanceable_Q():
    ds = _fake_test_split(per_class=100, classes=(1, 7))
    for bad in (99, 101):
        try:
            select_val_queries(ds, bad)
            raise AssertionError(f"Q={bad} should be rejected")
        except ValueError:
            pass
    try:                                    # not enough images to satisfy the split
        select_val_queries(_fake_test_split(per_class=5), 100)
        raise AssertionError("should refuse when a class has too few test images")
    except ValueError:
        pass
    print("  val: refuses indivisible Q and insufficient per-class supply")


def test_gen_seed_is_per_seed_and_reproducible():
    cfg = load_config({})
    base = int(cfg["lds"]["gen_seed_base"])
    seeds = (42, 123, 456)
    gen_seeds = {s: base + s for s in seeds}
    assert len(set(gen_seeds.values())) == len(seeds), "each seed needs its own noise"
    assert base + 42 == 165
    # reproducible: same derivation -> same starting noise
    a = (torch.manual_seed(gen_seeds[42]), torch.randn(4, 3, 32, 32))[1]
    b = (torch.manual_seed(gen_seeds[42]), torch.randn(4, 3, 32, 32))[1]
    c = (torch.manual_seed(gen_seeds[123]), torch.randn(4, 3, 32, 32))[1]
    assert torch.equal(a, b), "same seed must reproduce the same noise"
    assert not torch.equal(a, c), "different model seeds must not share noise"
    print(f"  gen: per-seed noise {gen_seeds}, reproducible and distinct")


def test_query_defaults_are_the_formal_spec():
    lds = load_config({})["lds"]
    assert lds["Q"] == 100, lds["Q"]
    dsets = load_config({})["datasets"]
    for k in ("cifar2_5k", "cifar10_v2"):
        assert k in dsets, f"{k} must be configured or generate raises KeyError"
        assert 100 % len(dsets[k]["classes"]) == 0
    print("  defaults: Q=100 for both tracks; cifar2_5k / cifar10_v2 configured")


def test_uncond_identity_resolves_to_uncond_paths():
    u = RunSpec(dataset="cifar2_5k", process="cfm", seed=42, conditional=False)
    c = RunSpec(dataset="cifar10_v2", process="cfm", seed=42, conditional=True)
    assert relpath(K.CHECKPOINT, u) == "checkpoints/cifar2_5k/cfm_uncond/seed_42/final.pt"
    assert relpath(K.GENERATION, u) == "generations/cifar2_5k/cfm_uncond/seed_42/samples.pt"
    assert relpath(K.GENERATION, c) == "generations/cifar10_v2/cfm_cond/seed_42/samples.pt"
    print("  uncond identities resolve to *_uncond checkpoint and generation paths")


# --- HP-p projection identity (added 2026-08-10) ------------------------------

def test_projection_identity_segments_features_and_scores():
    """HP-p sweeps four p; without a projection segment they overwrite each other
    and whatever survives is silently consumed downstream as the chosen p."""
    s = RunSpec(dataset="cifar2_5k", seed=42, conditional=False)
    sm = RunSpec(dataset="cifar2_5k", seed=42, method="das1squ", query_type="gen",
                 conditional=False)
    paths = {p: relpath(K.TRAIN_FEATURES, s, feat="das", proj=f"p{p}s0")
             for p in (1024, 4096, 16384, 32768)}
    assert len(set(paths.values())) == 4, f"rungs must not collide: {paths}"
    score_paths = {p: relpath(K.SCORES, sm, proj=f"p{p}s0") for p in (1024, 32768)}
    assert len(set(score_paths.values())) == 2, score_paths
    # projection seed is part of the identity too (A2-style grids)
    assert relpath(K.TRAIN_FEATURES, s, feat="das", proj="p4096s0") != \
           relpath(K.TRAIN_FEATURES, s, feat="das", proj="p4096s1")
    # absent proj -> byte-identical to the pre-existing production path
    assert relpath(K.TRAIN_FEATURES, s, feat="das") == \
           "featurize/das/cifar2_5k/seed_42/train_features.pt"
    assert relpath(K.SCORES, sm) == "scores/das1squ/cifar2_5k/seed_42/scores.npy"
    # e_n is a no-grad residual norm: identical across rungs, so it must NOT be
    # segmented (otherwise every rung recomputes the same 5000x1000 forwards).
    assert relpath(K.ERROR_WEIGHT, s, feat="das") == \
           relpath(K.ERROR_WEIGHT, s, feat="das", proj="p32768s0"), \
        "e_n is projection-independent and must be shared across rungs"
    print("  HP-p: 4 rungs get distinct feature+score paths; e_n shared; default path unchanged")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} query-spec tests passed")
