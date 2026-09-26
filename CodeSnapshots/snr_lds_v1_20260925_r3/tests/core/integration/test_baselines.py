"""THE MISSING E1 BASELINES — pixel/CLIP ([k]) and TracInCP/GAS ([f]), 2026-08-16.

Both families exist to make the main table an honest comparison, and both have a
failure mode that produces a plausible number instead of an error:

  * **pixel / CLIP** are not gradients. If they inherit the gradient pipeline's
    readout, timestep count, projection or error weight, they stop being "do the
    images look alike" and become something with no name.
  * **TracInCP / GAS** average a similarity over training checkpoints. If every
    checkpoint resolves to the same artifact path, the average is C copies of the
    final model and TracInCP silently *is* grad_dot — same shape, same magnitude,
    no warning. That is the whole reason the feature path grew a step segment.

Run:  Codes/tests/integration/test_baselines.py   (CLIP part needs the cached HF snapshot)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.schema.artifact import ArtifactKind as K
from balds.schema.estimator import FeatureSet
from balds.schema.registry import METHODS
from balds.schema.runspec import RunSpec
import balds.attribution  # noqa: F401  (registers methods)
from balds.attribution.embed import EMBEDDERS, embed
from balds.attribution.kernel import grad_similarity
from balds.artifacts.addressing import relpath


# --- [f] checkpoint identity -------------------------------------------------

def test_step_segment_separates_checkpoints_and_leaves_production_alone():
    s = RunSpec(dataset="cifar2_5k", seed=42, conditional=False)
    prod = relpath(K.TRAIN_FEATURES, s, feat="trak_T100")
    assert prod == "featurize/trak_T100/cifar2_5k/seed_42/train_features.pt", prod
    paths = {st: relpath(K.TRAIN_FEATURES, s, feat="trak_T100", step=st)
             for st in (1953, 3906, 5859)}
    assert len(set(paths.values())) == 3, paths
    assert prod not in paths.values(), "the final model must not collide with a mid step"
    q = RunSpec(dataset="cifar2_5k", seed=42, query_type="gen", conditional=False)
    assert relpath(K.QUERY_FEATURES, q, feat="trak_T100", step=1953) != \
           relpath(K.QUERY_FEATURES, q, feat="trak_T100")
    # composes with the HP-p segment rather than replacing it
    assert relpath(K.TRAIN_FEATURES, s, feat="trak_T100", proj="p4096s0", step=1953) == \
           "featurize/trak_T100/cifar2_5k/p4096s0/seed_42/step_1953/train_features.pt"
    print("  step segment: 3 checkpoints -> 3 paths, production path byte-identical")


def test_mid_checkpoint_steps_are_derived_from_this_run_s_history():
    """Total steps differ per dataset (7812 vs 78125), so the step numbers cannot
    be a constant — they are round(frac * this run's total)."""
    from balds.workflows.checkpoints import attribution_checkpoints, mid_checkpoint_steps

    cfg = {"train": {"checkpoint_fracs": [0.25, 0.5, 0.75]}}

    class _Store:
        def __init__(self, total): self.total = total
        def exists(self, *a, **k): return self.total is not None
        has_local = exists
        def load(self, *a, **k): return {"steps": self.total}

    base = RunSpec(dataset="cifar2_5k", seed=42, conditional=False)
    assert mid_checkpoint_steps(_Store(7812), cfg, base) == [1953, 3906, 5859]
    assert mid_checkpoint_steps(_Store(78125), cfg, base) == [19531, 39062, 58594]
    assert attribution_checkpoints(_Store(7812), cfg, base) == [1953, 3906, 5859, None]
    # no history -> only the final model, which callers must treat as "no TracIn"
    assert attribution_checkpoints(_Store(None), cfg, base) == [None]
    print("  steps derived per identity (7812->1953.., 78125->19531..), final appended")


# --- [f] TracInCP / GAS ------------------------------------------------------

def _pairs(c, n=40, p=8, q=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return ([torch.randn(n, p, generator=g) for _ in range(c)],
            [torch.randn(q, p, generator=g) for _ in range(c)])


def test_tracin_is_the_mean_over_checkpoints():
    tg, tq = _pairs(4)
    feats = FeatureSet(grads=tg[-1], error=None, feat_method="trak",
                       ckpt_grads=tg, ckpt_query=tq)
    got = METHODS.get("tracincp").score(feats, tq[-1], 0.1, device="cpu")
    want = sum(grad_similarity(g, q, cosine=False, device="cpu")
               for g, q in zip(tg, tq)) / 4
    assert np.allclose(got, want, atol=1e-6), np.abs(got - want).max()
    gas = METHODS.get("gas").score(feats, tq[-1], 0.1, device="cpu")
    want_cos = sum(grad_similarity(g, q, cosine=True, device="cpu")
                   for g, q in zip(tg, tq)) / 4
    assert np.allclose(gas, want_cos, atol=1e-6)
    assert not np.allclose(got, gas), "GAS must be the cosine variant, not a copy"
    print("  tracincp = mean of per-checkpoint dot; gas = mean of per-checkpoint cosine")


def test_tracin_degenerates_to_grad_dot_only_when_checkpoints_are_identical():
    """The bug the step segment prevents: if every checkpoint loads the final
    model's features, TracInCP returns exactly grad_dot and nothing says so."""
    tg, tq = _pairs(4)
    same = FeatureSet(grads=tg[0], error=None, feat_method="trak",
                      ckpt_grads=[tg[0]] * 4, ckpt_query=[tq[0]] * 4)
    collapsed = METHODS.get("tracincp").score(same, tq[0], 0.1, device="cpu")
    plain = METHODS.get("grad_dot").score(same, tq[0], 0.1, device="cpu")
    assert np.allclose(collapsed, plain, atol=1e-6), \
        "precondition: identical checkpoints DO collapse onto grad_dot"
    distinct = FeatureSet(grads=tg[-1], error=None, feat_method="trak",
                          ckpt_grads=tg, ckpt_query=tq)
    real = METHODS.get("tracincp").score(distinct, tq[-1], 0.1, device="cpu")
    assert not np.allclose(real, plain, atol=1e-3), \
        "with real checkpoints TracInCP must differ from the final-model baseline"
    print("  identical checkpoints collapse to grad_dot; distinct ones do not")


def test_tracin_refuses_rather_than_falling_back():
    m = METHODS.get("tracincp_T100")
    assert m.feat_method == "trak_T100"
    bare = FeatureSet(grads=torch.randn(10, 4), error=None, feat_method="trak_T100")
    try:
        m.score(bare, torch.randn(2, 4), 0.1, device="cpu")
        raise AssertionError("must refuse without checkpoints")
    except ValueError as e:
        assert "--ckpt-step" in str(e), str(e)
    tg, tq = _pairs(3)
    try:
        m.score(FeatureSet(grads=tg[0], error=None, feat_method="trak_T100",
                           ckpt_grads=tg, ckpt_query=tq[:2]), tq[0], 0.1, device="cpu")
        raise AssertionError("must refuse mismatched pair counts")
    except ValueError as e:
        assert "pairs must match" in str(e), str(e)
    print("  tracin refuses (loudly) with no checkpoints and with unpaired sides")


# --- [k] pixel / CLIP --------------------------------------------------------

def test_pixel_embedder_is_the_image_itself():
    imgs = torch.rand(7, 3, 32, 32) * 2 - 1          # dataset convention: [-1, 1]
    F = embed("pixel", imgs, device="cpu")
    assert F.shape == (7, 3 * 32 * 32), F.shape
    assert float(F.min()) >= 0.0 and float(F.max()) <= 1.0, "must land in [0,1]"
    ref = (imgs[3].float() / 2 + 0.5).clamp(0, 1).reshape(-1)
    assert torch.allclose(F[3], ref, atol=1e-6), "must be the image, not a transform of it"
    print("  pixel: flattened image in [0,1], no model involved")


def test_embedders_are_registered_and_have_similarity_methods():
    assert set(EMBEDDERS.names()) == {"pixel", "clip"}
    for base in ("pixel", "clip"):
        for suffix, cos in (("dot", False), ("cos", True)):
            m = METHODS.get(f"{base}_{suffix}")
            assert m.feat_method == base, m.feat_method
            assert m._cosine is cos
            assert not getattr(m, "needs_error_weight", False), "an embedding has no e_n"
            assert not getattr(m, "needs_repeats", False), "an embedding has no sigma-hat"
            assert not getattr(m, "needs_checkpoints", False)
    # ...and they must NOT have acquired timestep siblings — there is nothing to
    # average over timesteps in a CLIP embedding
    for bad in ("pixel_T100", "clip_dot_T100", "pixel_dot_T20"):
        assert not METHODS.has(bad), f"{bad} should not exist"
    print("  pixel/clip registered with dot+cos, no e_n / sigma-hat / T siblings")


def test_clip_embedder_runs_from_the_local_cache():
    imgs = torch.rand(4, 3, 32, 32) * 2 - 1
    F = embed("clip", imgs, device="cpu", batch_size=2)
    assert F.shape == (4, 512), F.shape
    assert torch.isfinite(F).all()
    # distinct images must not collapse to one embedding
    assert float((F[0] - F[1]).abs().max()) > 1e-4
    print(f"  clip: {tuple(F.shape)} embeddings from the cached snapshot")


# --- [e] Journey-TRAK --------------------------------------------------------

def test_capturing_the_trajectory_does_not_perturb_the_sampler():
    """The journey has to belong to the samples being attributed, so capture must
    be a pure observation — if it changed the path, the trajectory would describe
    an image nobody scored."""
    from balds.models.sample import euler_solve

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, padding=1))
    wrapped = lambda x, t: model(x)                       # noqa: E731
    x0 = torch.randn(4, 3, 8, 8)
    plain = euler_solve(wrapped, x0, steps=10, device="cpu")
    final, traj = euler_solve(wrapped, x0, steps=10, device="cpu", capture_at=[0, 5, 10])
    assert torch.equal(plain, final), "capture must not change the result"
    assert sorted(traj) == [0, 5, 10]
    assert torch.equal(traj[0], x0), "step 0 is the initial noise, before any update"
    assert torch.equal(traj[10], final), "the last capture is the final sample"
    assert not torch.equal(traj[0], traj[5]), "intermediate states must actually differ"
    print("  trajectory capture is observation-only; endpoints are x0 and the sample")


def test_journey_readout_refuses_l_simple():
    """L_Simple's target needs the (x_0, noise) pair behind x_t. A trajectory
    latent has none, and substituting one would silently not be L_Simple."""
    from balds.schema.registry import PROCESSES
    import balds.models  # noqa: F401

    from balds.attribution import GradFeaturizer

    model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, padding=1))
    fz = GradFeaturizer(model, PROCESSES.get("cfm"), proj_dim=8, T=3, loss_type="mse",
                        batch_size=2, device="cpu", projection="basic")
    try:
        fz.featurize_states(torch.randn(2, 3, 3, 8, 8), [0.0, 0.5, 1.0], torch.zeros(2, dtype=torch.long))
        raise AssertionError("mse must be refused on trajectory latents")
    except ValueError as e:
        assert "x_0" in str(e) and "noise" in str(e), str(e)
    fz2 = GradFeaturizer(model, PROCESSES.get("cfm"), proj_dim=8, T=3, loss_type="mean",
                         batch_size=2, device="cpu", projection="basic")
    try:
        fz2.featurize_states(torch.randn(2, 3, 3, 8, 8), [0.0, 0.5], torch.zeros(2, dtype=torch.long))
        raise AssertionError("must refuse when times do not match the states")
    except ValueError as e:
        assert "trajectory states" in str(e), str(e)
    print("  journey refuses L_Simple, and refuses mismatched state/time counts")


def test_journey_method_reads_two_different_recipes():
    """The only method whose sides differ: ordinary gradients for training
    samples, trajectory gradients for queries."""
    m = METHODS.get("journey_trak_T100")
    assert m.feat_method == "das_T100", m.feat_method
    assert m.query_feat_method == "journey", m.query_feat_method
    # every other method reads one recipe on both sides
    for other in ("das_T100", "dtrak_T100", "trak_T100", "tracincp_T100", "pixel_dot"):
        assert getattr(METHODS.get(other), "query_feat_method", None) is None, other
    s = RunSpec(dataset="cifar2_5k", seed=42, conditional=False)
    assert relpath(K.GEN_TRAJECTORY, s) == \
           "generations/cifar2_5k/cfm_uncond/seed_42/trajectory.pt"
    print("  journey_trak: train=das_T100, query=journey; trajectory is a samples.pt sibling")


def test_cosine_similarity_is_not_a_valid_check_in_this_feature_space():
    """Recorded because it nearly caused a wrong call (2026-08-16): journey
    features sit at cosine 0.992 from the ordinary ones, which looks like "same
    thing relabelled" until you measure the floor. DIFFERENT queries already sit
    at 0.956 — the space is concentrated, so absolute cosine says nothing. The
    real comparison is against a difference known to be genuine: T=10 vs T=100
    pairs at 0.9977, so journey at 0.9923 is further apart than a change that
    moves LDS by 0.03. Any future 'is this baseline distinct' check has to be
    made against a control, not a threshold."""
    a = torch.randn(50, 128)
    concentrated = a * 0.02 + torch.ones(1, 128)      # a deliberately concentrated space
    n = concentrated / concentrated.norm(dim=1, keepdim=True)
    off_diag = (n @ n.T).mean() - (1.0 / n.shape[0])
    assert float(off_diag) > 0.9, "precondition: concentrated spaces give high cosine to everything"
    print("  documented: absolute cosine is meaningless here; compare against the floor")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} baseline tests passed")
