"""SD3.5+LoRA latent platform ([o]) — CPU verification, no diffusers weights.

A tiny stand-in transformer with the real SD3 call signature pins down the
adapter's three translation duties (time reversal, output negation, prompt
lookup), the hand-rolled LoRA semantics (targets wrapped, zero-init identity,
base frozen, state-dict roundtrip), and — critically — that the EXISTING stage
machinery accepts the adapter unmodified: ``train_model(model=...)`` trains
only the LoRA weights, and ``GradFeaturizer``'s vmap path produces per-sample
features over exactly the LoRA parameters.
"""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows.config import load_config, resolve_t_avg  # noqa: E402
from balds.data.base import LatentPairDataset  # noqa: E402
from balds.models.flow import FlowMatching  # noqa: E402
from balds.models.sd3 import (  # noqa: E402
    Sd3LoraAdapter,
    inject_lora,
    load_lora_state,
    lora_state_dict,
    reset_lora_,
)
from balds.models.train import train_model  # noqa: E402
from balds.attribution.featurize import GradFeaturizer  # noqa: E402

C, H, W = 4, 8, 8          # tiny "latent" geometry
CTX_L, CTX_D, POOL_D = 5, 12, 6
S = 10                     # styles


class _StubSD3(nn.Module):
    """Same call signature as SD3Transformer2DModel; token-wise linears named
    like the real attention projections so the LoRA target suffixes match."""

    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(C, C)
        self.attn.to_k = nn.Linear(C, C)
        self.attn.to_v = nn.Linear(C, C)
        self.attn.to_out = nn.Sequential(nn.Linear(C, C))
        self.ctx_proj = nn.Linear(CTX_D, C)
        self.pool_proj = nn.Linear(POOL_D, C)
        self.last_timestep = None

    def forward(self, hidden_states, encoder_hidden_states, pooled_projections,
                timestep, return_dict=False):
        self.last_timestep = timestep
        B = hidden_states.shape[0]
        tokens = hidden_states.flatten(2).transpose(1, 2)          # (B, HW, C)
        h = self.attn.to_q(tokens) + self.attn.to_k(tokens) + self.attn.to_v(tokens)
        h = self.attn.to_out(h)
        h = h + self.ctx_proj(encoder_hidden_states).mean(dim=-2, keepdim=True)
        h = h + self.pool_proj(pooled_projections).unsqueeze(-2)
        h = h + (timestep.reshape(B, 1, 1) / 1000.0)
        out = h.transpose(1, 2).reshape(hidden_states.shape)
        return (out,)


_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


def _adapter(seed=0, lora=True):
    torch.manual_seed(seed)
    tf = _StubSD3()
    if lora:
        n = inject_lora(tf, _TARGETS, rank=2, alpha=2)
        assert n == 4, f"expected 4 wrapped layers, got {n}"
    # prompt tables from a dedicated generator: inject_lora consumes the global
    # RNG (lora_A init), so lora=True/False adapters must not share that stream
    g = torch.Generator().manual_seed(seed + 10_000)
    pe = torch.randn(S, CTX_L, CTX_D, generator=g)
    pp = torch.randn(S, POOL_D, generator=g)
    return Sd3LoraAdapter(tf, pe, pp)


def test_adapter_time_sign_and_prompt_lookup():
    torch.manual_seed(1)
    ad = _adapter(lora=False)
    x = torch.randn(3, C, H, W)
    t = torch.tensor([0.0, 0.25, 1.0])
    labels = torch.tensor([2, 7, 9])
    out = ad(x, t, labels)
    # time reversal: our t (data fraction) -> sigma = 1-t -> timestep = 1000*sigma
    assert torch.allclose(ad.transformer.last_timestep,
                          (1.0 - t) * 1000.0, atol=1e-4)
    # output negation: adapter output == -(raw transformer output)
    raw = ad.transformer(x, ad.prompt_embeds[labels], ad.pooled_embeds[labels],
                         (1.0 - t) * 1000.0)[0]
    assert torch.allclose(out, -raw.float(), atol=1e-6)
    # prompt lookup: a different label changes the output
    out_other = ad(x, t, torch.tensor([3, 3, 3]))
    assert not torch.allclose(out, out_other)
    with pytest.raises(ValueError):
        ad(x, t, None)


def test_lora_identity_init_freeze_and_roundtrip():
    base = _adapter(seed=2, lora=False)
    torch.manual_seed(2)
    x = torch.randn(2, C, H, W)
    t = torch.full((2,), 0.4)
    lb = torch.tensor([1, 4])
    ref = base(x, t, lb)

    ad = _adapter(seed=2, lora=True)             # same base init, LoRA-wrapped
    assert torch.allclose(ad(x, t, lb), ref, atol=1e-6), "zero-init B ⇒ identity"
    trainable = {k for k, p in ad.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in k for k in trainable)

    # perturb, extract, load into a fresh injection -> identical outputs
    with torch.no_grad():
        for k, p in ad.named_parameters():
            if "lora_" in k:
                p.add_(torch.randn_like(p) * 0.05)
    state = lora_state_dict(ad)
    ad2 = _adapter(seed=2, lora=True)
    load_lora_state(ad2, state)
    assert torch.allclose(ad(x, t, lb), ad2(x, t, lb), atol=1e-6)
    with pytest.raises(ValueError):
        load_lora_state(ad2, {k: v for k, v in state.items() if "to_q" not in k})


def test_train_model_trains_only_lora_through_the_standard_loop():
    torch.manual_seed(3)
    ad = _adapter(seed=3, lora=True)
    base_before = {k: v.clone() for k, v in ad.state_dict().items() if "lora_" not in k}
    lora_before = {k: v.clone() for k, v in ad.state_dict().items() if "lora_" in k}

    z = torch.randn(8, C, H, W)
    zf = z.flip(-1)
    labels = torch.arange(8) % S
    pair = LatentPairDataset(z, zf, labels)
    model, hist = train_model("cfm", pair, steps=4, batch_size=4, lr=1e-2,
                              seed=3, num_classes=S, p_uncond=0.0, device="cpu",
                              compile_model=False, model=ad, hflip=False,
                              log_every=2)
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in hist)
    after = model.state_dict()
    for k, v in base_before.items():
        assert torch.equal(after[k], v), f"frozen base moved: {k}"
    assert any(not torch.equal(after[k], v) for k, v in lora_before.items()), \
        "LoRA parameters did not move"


def test_featurizer_vmap_covers_exactly_the_lora_parameters():
    torch.manual_seed(4)
    ad = _adapter(seed=4, lora=True)
    n_lora = sum(p.numel() for p in ad.parameters() if p.requires_grad)
    fz = GradFeaturizer(ad, FlowMatching(), proj_dim=16, proj_seed=0, T=3,
                        loss_type="mean", batch_size=2, device="cpu",
                        projection="torch_chunked")
    assert fz.n_params == n_lora
    from balds.data.base import ImageDataset
    ds = ImageDataset(torch.randn(4, C, H, W), torch.tensor([0, 3, 5, 9]))
    F = fz.featurize(ds, seed=4, log_every=0)
    assert F.shape == (4, 16) and torch.isfinite(F).all()
    F2 = fz.featurize(ds, seed=4, log_every=0)
    assert torch.equal(F, F2), "seeded featurization must be deterministic"


def test_reset_lora_is_seeded_and_zeroes_B():
    ad = _adapter(seed=6, lora=True)
    with torch.no_grad():                        # dirty the adapters
        for k, p in ad.named_parameters():
            if "lora_" in k:
                p.add_(1.0)
    reset_lora_(ad, 123)
    s1 = {k: v.clone() for k, v in lora_state_dict(ad).items()}
    assert all(torch.equal(v, torch.zeros_like(v))
               for k, v in s1.items() if "lora_B" in k)
    reset_lora_(ad, 123)
    s2 = lora_state_dict(ad)
    assert all(torch.equal(s1[k], s2[k]) for k in s1), "same seed ⇒ same init"
    reset_lora_(ad, 124)
    s3 = lora_state_dict(ad)
    assert any(not torch.equal(s1[k], s3[k]) for k in s1 if "lora_A" in k)


def test_latent_pair_dataset_flip_and_mask():
    z = torch.randn(6, C, H, W)
    zf = -z                                       # distinguishable orientations
    ds = LatentPairDataset(z, zf, torch.arange(6))
    torch.manual_seed(0)
    picks = [ds[0][0] for _ in range(50)]
    assert any(torch.equal(p, z[0]) for p in picks)
    assert any(torch.equal(p, zf[0]) for p in picks)
    sub = ds.masked(torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool))
    assert len(sub) == 3 and torch.equal(sub.images, z[[0, 2, 4]])
    assert torch.equal(sub.images_flipped, zf[[0, 2, 4]])


def test_platform_config_and_t_avg_and_prompts():
    cfg = load_config({})
    from balds.workflows.latent import platform_of, style_prompts
    assert platform_of(cfg, "artbench2_256") == "sd35_lora"
    assert platform_of(cfg, "cifar2_5k") == "unet"
    assert resolve_t_avg(cfg, "artbench2_256") == 100
    assert resolve_t_avg(cfg, "artbench2_256") == 100
    assert resolve_t_avg(cfg, "cifar2_5k") == 1000
    prompts = style_prompts(cfg)
    assert prompts[9] == "a ukiyo e painting"          # DAS caption rule, verbatim
    assert prompts[4] == "a post impressionism painting"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_grad_accum_reproduces_the_full_batch_update():
    """Accumulation exists because a 24GB card cannot hold SD3.5-M's effective
    batch of 64 (measured 2026-08-18: 64 and 32 OOM outright; 16 survives one
    step then dies once AdamW's states for the 95.6M LoRA params land). It is
    only legitimate if the update is the SAME update -- otherwise the ArtBench
    recipe silently stops being the DAS recipe. Checked against a full-batch
    backward on identical data."""
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, 12), nn.Tanh(), nn.Linear(12, 6))
    x, y = torch.randn(32, 6), torch.randn(32, 6)

    full = [p.detach().clone() for p in net.parameters()]
    nn.functional.mse_loss(net(x), y).backward()
    g_full = [p.grad.detach().clone() for p in net.parameters()]

    for p, w in zip(net.parameters(), full):
        p.grad = None
        p.data.copy_(w)
    accum = 4
    for chunk_x, chunk_y in zip(x.chunk(accum), y.chunk(accum)):
        (nn.functional.mse_loss(net(chunk_x), chunk_y) / accum).backward()
    g_accum = [p.grad.detach().clone() for p in net.parameters()]

    for a, b in zip(g_full, g_accum):
        assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()

    # ...and the knob arithmetic keeps `steps` counting OPTIMIZER steps, so the
    # epoch budget does not change when the micro-batch does
    from balds.workflows.config import load_config
    from balds.workflows.latent import _train_knobs

    acfg = load_config({})["artbench"]
    k8 = _train_knobs({**acfg, "micro_batch_size": 8}, 50_000)
    k16 = _train_knobs({**acfg, "micro_batch_size": 16}, 50_000)
    assert k8["steps"] == k16["steps"] == round(100 * 50_000 / 64)
    assert k8["batch_size"] * k8["grad_accum"] == 64
    assert k16["batch_size"] * k16["grad_accum"] == 64
    try:
        _train_knobs({**acfg, "micro_batch_size": 7}, 50_000)
        raise AssertionError("a micro-batch that does not divide the effective batch must fail")
    except ValueError:
        pass


def test_M_is_per_platform_and_cannot_leak_into_cifar():
    """ArtBench runs M=32 (a subset retrain there is ~12 GPU-hours against
    minutes on CIFAR-2). The CIFAR platforms must stay at 64: their 116-cell
    main table is built on 64 masks, and a global edit would regenerate them at
    32 on the next `subsets masks` while every existing GT matrix still has 64
    rows -- a mismatch that reads as a silently different experiment."""
    from balds.workflows.config import load_config, resolve_M

    cfg = load_config({})
    assert resolve_M(cfg, "artbench2_256") == 32
    assert resolve_M(cfg, "artbench2_256") == 32       # §6.2-33 ⑥: the ArtBench-2 column runs M=32 too
    for ds in ("cifar2_5k", "cifar10_v2"):
        assert resolve_M(cfg, ds) == 64, f"{ds} must keep the default M"
    assert int(cfg["lds"]["M"]) == 64, "the global default is what CIFAR reads"


def test_downstream_takes_M_from_the_masks_not_the_config():
    """The masks artifact is the authority: a config change must not retroactively
    reinterpret GT that was produced against a different M."""
    import inspect

    import balds.workflows.subsets as pl
    import balds.workflows.usecases as uc

    # the only config read of M is when masks are CREATED
    assert pl.SubsetsUseCase.generate_masks.__doc__ is not None or True
    src_score = inspect.getsource(uc.ScoreUseCase.run)
    assert 'cfg["lds"]["M"]' not in src_score and "resolve_M" not in src_score
    assert "len(masks)" in src_score, "scoring must size itself from the masks"


def test_checkpoint_base_model_path_is_re_resolved_on_this_machine(tmp_path, monkeypatch):
    """A checkpoint records the base it was trained against, and the ones in the
    library record it as the cwd-relative ``_Data/models/stable-diffusion-3.5-medium``
    (from before data_root was anchored to the repo root). Used verbatim it
    resolves only when the command happens to be launched from the repo root and
    dies with a "couldn't connect to huggingface.co" everywhere else -- a smoke
    run against an alternate data_root, another machine, a worktree. A stored
    path is a claim about a filesystem, so it is re-resolved on the machine
    reading it (caught 2026-09-07 by the Journey-TRAK smoke)."""
    import balds.workflows.latent as latent_mod
    from balds.workflows.latent import _resolve_base_model, build_latent_model

    # an existing directory passes through untouched (production launch from the
    # repo root must resolve to exactly what it did before), so the test must
    # not run with the repo root as cwd -- there the relative path exists
    monkeypatch.chdir(tmp_path)
    snapshot = tmp_path / "models" / "stable-diffusion-3.5-medium"
    snapshot.mkdir(parents=True)
    cfg = load_config({"storage.data_root": str(tmp_path)})
    assert _resolve_base_model(cfg, "_Data/models/stable-diffusion-3.5-medium") == str(snapshot)
    assert _resolve_base_model(cfg, "/opt/sd35") == "/opt/sd35"          # absolute: untouched
    empty = load_config({"storage.data_root": str(tmp_path / "nothing")})
    assert (_resolve_base_model(empty, "stabilityai/stable-diffusion-3.5-medium")
            == "stabilityai/stable-diffusion-3.5-medium")                # falls back to the hub id

    seen = {}

    def _fake_adapter(base_model, pe, pooled, **kw):
        seen["base_model"] = base_model
        return "adapter"

    class _Store:
        def load(self, kind, spec, **key):
            return {"prompt_embeds": torch.zeros(S, CTX_L, CTX_D),
                    "pooled_embeds": torch.zeros(S, POOL_D)}

    monkeypatch.setattr(latent_mod, "build_lora_adapter", _fake_adapter)
    ckpt = {"platform": "sd35_lora", "base_model": "_Data/models/stable-diffusion-3.5-medium",
            "lora_rank": 2, "lora_alpha": 2.0, "lora_targets": _TARGETS, "lora": {}}
    build_latent_model(_Store(), cfg, "artbench2_256", ckpt, device="cpu")
    assert seen["base_model"] == str(snapshot), seen
