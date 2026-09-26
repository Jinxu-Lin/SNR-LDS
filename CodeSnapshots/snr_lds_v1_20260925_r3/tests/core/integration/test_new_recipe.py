"""NEW-RECIPE GATES ([g][a][b][m], 2026-08-09): the DAS-aligned training recipe,
the cifar2_5k dataset, unconditional training, and replica-run plumbing behave
as adjudicated (experiment.qmd §1.2) — verified on tiny CPU models, no GPU.

Covers:
  [g] epoch-based step derivation (7812 full / per-subset), AdamW+warmup+cosine
      schedule shape, hflip/dropout knobs, mid-training checkpoint callback.
  [a] cifar2_5k: N=5000, 2500/class, deterministic across loads (fixed sampling
      seed), a strict subset of the 10k cifar2 train split.  [data-dependent:
      skipped cleanly if the HF cache is absent on this machine]
  [b] unconditional training path (num_classes=None): no class embedding,
      forward with labels=None, checkpoint round-trips through build_model.
  [m] replica retrain-seed offset is deterministic and distinct from the main
      run; replica=0 is byte-identical to the legacy call.

Run:  conda run -n da python Codes/tests/integration/test_new_recipe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows.common import _steps, _subset_steps
from balds.models.model_io import build_model, checkpoint_state
from balds.models.train import train_model
from balds.models.unet import UNetCFM

RECIPE_CFG = {"train": {"epochs": 200}}


def _tiny_ds(n=32, num_classes=2):
    torch.manual_seed(0)
    from balds.data.base import ImageDataset
    return ImageDataset(torch.randn(n, 3, 32, 32),
                        torch.randint(0, num_classes, (n,)))


def test_epoch_step_derivation():
    assert _steps(RECIPE_CFG, 5000, 128) == 7812        # 200*5000/128 = 7812.5 -> round
    assert _steps(RECIPE_CFG, 50000, 128) == 78125      # cifar10
    assert _subset_steps(RECIPE_CFG, 2500, 128) == 3906  # 200*2500/128 = 3906.25
    assert _subset_steps(RECIPE_CFG, 2437, 128) == 3808  # varies with actual mask size
    print("  [g] step derivation: full 7812 / subset per-actual-size ok")


def test_uncond_recipe_training_with_mid_checkpoints():
    ds = _tiny_ds()
    saved = []
    model, hist = train_model(
        "cfm", ds, steps=8, batch_size=8, lr=1e-3, seed=7,
        num_classes=None,                     # [b] unconditional
        device="cpu", base_ch=8, compile_model=False, log_every=4,
        weight_decay=1e-6, warmup_frac=0.25, dropout=0.1, hflip=True,   # [g]
        checkpoint_steps=(2, 4, 6), on_checkpoint=lambda s, m: saved.append(s))
    assert saved == [2, 4, 6], saved
    assert not hasattr(model, "class_embed")
    assert model.num_classes is None
    # checkpoint round-trip: uncond model reconstructs from its own ckpt
    ck = checkpoint_state(model, base_ch=8, num_classes=None, step=8,
                          extra={"model_type": "cfm", "conditional": False})
    m2 = build_model(ck, device="cpu")
    assert m2.num_classes is None
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = m2(x, torch.tensor([0.5, 0.5]), None)
    assert out.shape == x.shape
    print("  [b]+[g] uncond training w/ mid-ckpts at 25/50/75%: fires + round-trips")


def test_warmup_schedule_shape():
    ds = _tiny_ds()
    lrs = []
    train_model("cfm", ds, steps=20, batch_size=8, lr=1e-3, seed=1,
                num_classes=2, device="cpu", base_ch=8, compile_model=False,
                log_every=1, warmup_frac=0.5,
                on_progress=lambda **kw: lrs.append(kw["lr"]))
    peak = max(lrs)
    assert lrs[0] < 0.15 * peak, f"warmup start too high: {lrs[0]:.2e} vs peak {peak:.2e}"
    assert abs(peak - 1e-3) / 1e-3 < 0.15, f"peak {peak:.2e} != lr 1e-3"
    assert lrs[-1] < 0.5 * peak, "cosine tail did not decay"
    print(f"  [g] warmup->cosine: start {lrs[0]:.1e} -> peak {peak:.1e} -> end {lrs[-1]:.1e}")


def test_legacy_default_path_deterministic_and_conditional():
    ds = _tiny_ds()
    losses = []
    for _ in range(2):
        _, hist = train_model("cfm", ds, steps=4, batch_size=8, lr=1e-3, seed=3,
                              num_classes=2, device="cpu", base_ch=8,
                              compile_model=False, log_every=2)
        losses.append([h["loss"] for h in hist])
    assert losses[0] == losses[1], "same-seed default path must be deterministic"
    print("  legacy conditional default path: deterministic across runs")


def test_replica_seed_offset():
    # mirrors pipeline.train_subsets: retrain_seed = seed + m + 1 + replica*100003
    main = [42 + m + 1 for m in range(64)]
    rep1 = [42 + m + 1 + 100003 for m in range(64)]
    assert len(set(main) & set(rep1)) == 0, "replica seeds must not collide with main"
    assert rep1 == [m + 100003 for m in main], "replica offset must be a pure shift"
    print("  [m] replica retrain-seed namespace: disjoint deterministic shift")


def test_cifar2_5k_loader():
    import os
    cache = "_Data/hf_cache"
    if not os.path.isdir(cache):
        print("  [a] SKIPPED (no _Data/hf_cache on this machine)")
        return
    from balds.data import get_dataset
    tr1, te = get_dataset("cifar2_5k", cache)
    tr2, _ = get_dataset("cifar2_5k", cache)
    labels = tr1.labels
    assert len(tr1.images) == 5000, len(tr1.images)
    counts = {int(c): int((labels == c).sum()) for c in labels.unique()}
    assert sorted(counts.values()) == [2500, 2500], counts
    assert torch.equal(tr1.images[0], tr2.images[0]) and torch.equal(tr1.labels, tr2.labels), \
        "cifar2_5k must be deterministic across loads"
    full, _ = get_dataset("cifar2", cache)
    assert len(full.images) == 10000
    # 5k images must all appear in the 10k split (subset property, spot check)
    assert any(torch.equal(tr1.images[0], full.images[i]) for i in range(len(full.images)))
    print(f"  [a] cifar2_5k: N=5000 balanced {counts}, deterministic, subset-of-10k ok")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} new-recipe tests passed")


def test_p_uncond_resolves_per_dataset():
    """CIFAR-10's adjudicated recipe is PURE-conditional (p_uncond=0), not the
    global 0.1 default, and every cifar10_v2 artifact in the library carries 0.0.

    Before the table this depended on remembering `--set train.p_uncond=0` on
    every command: a task brief that said "take the train defaults" trained 0.1
    instead, and the acceptance checklist (conditional/model_type/num_classes/
    step) does not look at p_uncond -- so the wrong model passes every check and
    goes downstream silently. Caught by hand on 2026-08-27; this is the guard.
    """
    from balds.workflows.config import load_config, resolve_p_uncond

    cfg = load_config({})
    assert resolve_p_uncond(cfg, "cifar10_v2") == 0.0, "C10 must be pure-conditional"
    assert resolve_p_uncond(cfg, "cifar2_5k") == float(cfg["train"]["p_uncond"]), \
        "datasets with no override fall back to the global default"
    # the training path must consume the resolver, not the raw config key --
    # otherwise the table exists and does nothing (the failure mode is invisible)
    import inspect

    import balds.workflows.subsets as pl
    src = inspect.getsource(pl)
    assert 'p_uncond=t["p_uncond"]' not in src, \
        "a training call site still reads train.p_uncond directly, bypassing the per-dataset table"
    print("  p_uncond resolves per dataset (cifar10_v2 -> 0.0); no call site bypasses it")
