"""Journey-TRAK on the SD3.5 latent platform — §6.2-28 ②, 2026-09-07.

Journey-TRAK attributes over the states the sampler actually passed through, so
the trajectory has to be CAPTURED from a replay of the stored generation, and the
replay has to land on that generation or the journey belongs to some other image.
On the latent platform the replay runs in LATENT space (that is where the sampler
works and where ``samples.pt`` lives), through the SAME ``euler_solve`` the
generate stage used — a second hand-written sampler is exactly how the drift
stops being zero.

T1 the replay lands on the stored latents bit for bit; drift 0, states shaped
   (Q, P, 16, h, w) and taken at the captured step indices
T2 the drift check is a real gate: a different LoRA raises, and the tolerance is
   the config knob `lds.journey_drift_tol`
T3 the journey features come out (Q, p) on the latent platform, at the platform's
   own featurize batch size, and are deterministic
T4 `journey_trak_T100` scores end to end from them
T5 the accumulation shared by featurize/featurize_states is bit-identical to the
   out-of-place reference formula (the refactor that gave the trajectory path the
   LoRA platform's memory profile)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import balds.workflows.latent as latent_mod  # noqa: E402
from balds.workflows.config import load_config  # noqa: E402
from balds.workflows.features import FeaturizeUseCase
from balds.workflows.queries import GenerateUseCase  # noqa: E402
from balds.workflows.usecases import ScoreUseCase  # noqa: E402
from balds.schema.artifact import ArtifactKind as K  # noqa: E402
from balds.schema.registry import DATASETS  # noqa: E402
from balds.schema.runspec import RunSpec  # noqa: E402
from balds.data.base import ImageDataset  # noqa: E402
from balds.models import euler_solve  # noqa: E402
from balds.models.flow import FlowMatching  # noqa: E402
from balds.models.sd3 import Sd3LoraAdapter, inject_lora, lora_state_dict  # noqa: E402
from balds.attribution.featurize import GradFeaturizer, _accumulate  # noqa: E402
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB  # noqa: E402

_DS = "_journey_toy"
N, C, HW = 8, 4, 8            # tiny stand-in for (5000, 16, 32, 32)
CTX_L, CTX_D, POOL_D, S = 5, 12, 6, 10
Q, ODE, POINTS, PDIM = 4, 6, 4, 16
TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


class _StubSD3(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(C, C); self.attn.to_k = nn.Linear(C, C)
        self.attn.to_v = nn.Linear(C, C); self.attn.to_out = nn.Sequential(nn.Linear(C, C))
        self.ctx_proj = nn.Linear(CTX_D, C); self.pool_proj = nn.Linear(POOL_D, C)

    def forward(self, hidden_states, encoder_hidden_states, pooled_projections,
                timestep, return_dict=False):
        B = hidden_states.shape[0]
        tok = hidden_states.flatten(2).transpose(1, 2)
        h = self.attn.to_out(self.attn.to_q(tok) + self.attn.to_k(tok) + self.attn.to_v(tok))
        h = h + self.ctx_proj(encoder_hidden_states).mean(-2, keepdim=True)
        h = h + self.pool_proj(pooled_projections).unsqueeze(-2) + timestep.reshape(B, 1, 1) / 1000
        return (h.transpose(1, 2).reshape(hidden_states.shape),)


def _toy_loader(data_dir, **_):
    g = torch.Generator().manual_seed(0)
    return (ImageDataset(torch.randn(N, 3, 32, 32, generator=g), torch.zeros(N, dtype=torch.long)),
            ImageDataset(torch.randn(2, 3, 32, 32, generator=g), torch.zeros(2, dtype=torch.long)))


if not DATASETS.has(_DS):
    DATASETS.add(_DS, _toy_loader)


def _adapter(pe, pooled, *, perturb: float = 0.0, seed: int = 0):
    torch.manual_seed(0)
    tf = _StubSD3()
    inject_lora(tf, TARGETS, rank=2, alpha=2)
    m = Sd3LoraAdapter(tf, pe, pooled)
    if perturb:
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for k, p in m.named_parameters():
                if "lora_" in k:
                    p.add_(perturb * torch.randn(p.shape, generator=g))
    return m.eval()


@pytest.fixture(scope="module")
def env(module_mocker=None):
    root = tempfile.mkdtemp()
    store = ArtifactStore(ManifestDB(f"{root}/m.db"), [LocalBackend(f"{root}/data")],
                             code_version="test")
    cfg = load_config({"storage.data_root": f"{root}/data"})
    cfg["datasets"][_DS] = {"classes": [4, 9], "num_classes": 10, "platform": "sd35_lora"}
    cfg["artbench"].update({"latent_shape": [C, HW, HW], "featurize_batch_size": 2,
                            "lora": {"rank": 2, "alpha": 2, "targets": TARGETS}})
    cfg["featurize"]["proj_dim"] = PDIM
    cfg["lds"]["journey_points"] = POINTS
    g = torch.Generator().manual_seed(1)
    pe, pooled = torch.randn(S, CTX_L, CTX_D, generator=g), torch.randn(S, POOL_D, generator=g)
    store.save(K.PROMPT_EMBEDS, RunSpec(dataset=_DS),
               {"prompt_embeds": pe, "pooled_embeds": pooled})
    base = RunSpec(dataset=_DS, process="cfm", seed=42, conditional=True)
    model = _adapter(pe, pooled, perturb=0.1, seed=3)
    store.save(K.CHECKPOINT, base, {
        "platform": "sd35_lora", "model_type": "cfm", "conditional": True, "step": 1,
        "base_model": "stub", "lora_rank": 2, "lora_alpha": 2.0, "lora_targets": TARGETS,
        "lora": lora_state_dict(model)})
    # the stored generation, made exactly the way run_latent_generate makes one
    labels = torch.tensor([4, 4, 9, 9])
    torch.manual_seed(165)
    x0 = torch.randn(Q, C, HW, HW)
    with torch.no_grad():
        latents = euler_solve(model, x0, steps=ODE, device="cpu",
                              class_label=labels).cpu()
    store.save(K.GENERATION, base, {
        "samples": latents, "images_u8": ((latents[:, :3] + 1) * 127.5).clamp(0, 255).to(torch.uint8),
        "labels": labels, "Q": Q, "gen_seed": 165, "ode_steps": ODE,
        "conditional": True, "platform": "sd35_lora"})
    return store, cfg, base, pe, pooled


@pytest.fixture(autouse=True)
def stub_model(monkeypatch, env):
    """The real path materialises a 2.24B SD3.5 from disk; this fixture swaps the
    ONE function that does it, so everything else under test is production code."""
    _, _, _, pe, pooled = env

    def _build(store, cfg, dataset, ckpt, *, device):
        from balds.models.sd3 import load_lora_state
        m = _adapter(pe, pooled)
        if ckpt is not None:
            load_lora_state(m, ckpt["lora"])
        return m.eval()

    monkeypatch.setattr(latent_mod, "build_latent_model", _build)


def test_t1_replay_lands_on_the_stored_latents(env):
    store, cfg, base, _, _ = env
    out = GenerateUseCase(store, cfg, device="cpu").trajectory(_DS, 42)
    assert out["replay_drift"] == 0.0, out
    assert out["points"] == POINTS and out["drift_tol"] == 1e-3
    traj = store.load(K.GEN_TRAJECTORY, base)
    assert list(traj["states"].shape) == [Q, POINTS, C, HW, HW]
    steps = traj["steps"].tolist()
    assert steps == sorted(set(steps)) and steps[0] == 0 and steps[-1] == ODE
    # the captured state at step 0 IS the sampler's x0 (same seed, same draw)
    torch.manual_seed(165)
    assert torch.equal(traj["states"][:, 0], torch.randn(Q, C, HW, HW))
    assert GenerateUseCase(store, cfg, device="cpu").trajectory(_DS, 42) == {"skipped": True}


def test_t2_drift_is_a_real_gate(env, monkeypatch):
    store, cfg, base, pe, pooled = env
    # a DIFFERENT LoRA must not reproduce the generation -> hard error
    def _other(store_, cfg_, dataset, ckpt, *, device):
        return _adapter(pe, pooled, perturb=0.5, seed=99)
    monkeypatch.setattr(latent_mod, "build_latent_model", _other)
    with pytest.raises(ValueError, match="diverged from the stored generation"):
        GenerateUseCase(store, cfg, device="cpu").trajectory(_DS, 42, force=True)
    # ...and the tolerance is the config knob, not a literal
    cfg2 = {**cfg, "lds": {**cfg["lds"], "journey_drift_tol": 1e9}}
    out = GenerateUseCase(store, cfg2, device="cpu").trajectory(_DS, 42, force=True)
    assert out["drift_tol"] == 1e9 and out["replay_drift"] > 1e-3
    assert float(load_config({})["lds"]["journey_drift_tol"]) == 1e-3


def test_t3_journey_features_on_the_latent_platform(env, monkeypatch):
    store, cfg, base, _, _ = env
    GenerateUseCase(store, cfg, device="cpu").trajectory(_DS, 42, force=True)
    import balds.workflows.features as features_mod
    seen = []
    def explicit_projector(*args, **kwargs):
        seen.append(kwargs["projection"])
        return GradFeaturizer(*args, **kwargs)
    monkeypatch.setattr(features_mod, "GradFeaturizer", explicit_projector)
    monkeypatch.setitem(cfg["featurize"], "projection", "torch_chunked")
    uc = FeaturizeUseCase(store, cfg, device="cpu")
    assert uc._feat_batch(_DS) == 2                      # the platform's own batch size
    out = uc.run(_DS, 42, feat="journey", split="query", query_type="gen")
    assert out["query_gen"] == [Q, PDIM] and out["points"] == POINTS
    assert seen == ["torch_chunked"]
    F = store.load(K.QUERY_FEATURES, base.with_(query_type="gen"), feat="journey")
    assert F.shape == (Q, PDIM) and torch.isfinite(F).all()
    again = uc.run(_DS, 42, feat="journey", split="query", query_type="gen", force=True)
    assert again["query_gen"] == [Q, PDIM]
    assert torch.equal(F, store.load(K.QUERY_FEATURES, base.with_(query_type="gen"),
                                     feat="journey"))
    # journey stays query-side, gen-track only
    with pytest.raises(ValueError, match="query-side feature only"):
        uc.run(_DS, 42, feat="journey", split="train")
    with pytest.raises(ValueError, match="only for the gen track"):
        uc.run(_DS, 42, feat="journey", split="query", query_type="val")


def test_t4_journey_trak_scores_end_to_end(env):
    store, cfg, base, _, _ = env
    GenerateUseCase(store, cfg, device="cpu").trajectory(_DS, 42, force=True)
    FeaturizeUseCase(store, cfg, device="cpu").run(
        _DS, 42, feat="journey", split="query", query_type="gen", force=True)
    # the train side of journey_trak is the ORDINARY gradient feature
    rng = np.random.RandomState(0)
    store.save(K.TRAIN_FEATURES, base, torch.tensor(rng.randn(N, PDIM), dtype=torch.float32),
               feat="das_T100")
    M = 4
    store.save(K.SUBSET_MASKS, base, [rng.rand(N) < 0.5 for _ in range(M)])
    store.save(K.GT_MATRIX, base, rng.randn(M, Q).astype(np.float64))
    out = ScoreUseCase(store, cfg, device="cpu").run("journey_trak_T100", _DS, 42,
                                                     query_type="gen")
    assert out["shape"] == [N, Q]
    scores = store.load(K.SCORES, base.with_(method="journey_trak_T100", query_type="gen"))
    assert scores.shape == (N, Q) and np.isfinite(scores).all()


def test_t5_shared_accumulation_matches_the_out_of_place_formula():
    """The trajectory featurizer now shares `featurize`'s in-place accumulation
    (1.42 GiB per (B,P) buffer on the LoRA platform). In-place is only legitimate
    if it is the same arithmetic — pinned here against the literal formula the
    two paths used to write out separately."""
    torch.manual_seed(0)
    B, P = 3, 17
    for normalize in (True, False):
        emb_ref = None
        emb_new = None
        w_sum = 0.0
        for t, w_t in enumerate((1.0, 0.75, 0.25)):
            grads = {f"p{i}": torch.randn(B, 4, 2) for i in range(3)}
            grads["tail"] = torch.randn(B, 5)
            ref_flat = torch.cat([v.reshape(B, -1) for v in grads.values()], dim=1).float()
            if normalize:
                ref_flat = ref_flat / (torch.norm(ref_flat, dim=-1, keepdim=True) + 1e-8)
            emb_ref = w_t * ref_flat if emb_ref is None else emb_ref + w_t * ref_flat
            emb_new = _accumulate([{k: v.clone() for k, v in grads.items()}], B,
                                  normalize=normalize, w_t=w_t, emb=emb_new)
            w_sum += w_t
        assert torch.equal(emb_ref / w_sum, emb_new.clone().div_(w_sum)), normalize


def test_t5b_featurize_states_equals_featurize_at_the_same_state():
    """A trajectory point IS an (x_t, t) pair; the only thing that differs from
    the ordinary featurizer is where x_t came from. With one shared accumulation
    the two entry points must agree exactly on the same (x_t, t).

    Built on the CFM identity ``interpolate(x1, eps, t=0) = eps``: at t=0 the
    ordinary featurizer's x_t is its own drawn noise, so handing that same noise
    to ``featurize_states`` as a one-point trajectory asks both paths for the
    gradient at the identical input.
    """
    g = torch.Generator().manual_seed(9)
    pe, pooled = torch.randn(S, CTX_L, CTX_D, generator=g), torch.randn(S, POOL_D, generator=g)
    ad = _adapter(pe, pooled, perturb=0.1, seed=5)
    labels = torch.tensor([4, 4, 9, 9])
    ds = ImageDataset(torch.randn(4, C, HW, HW, generator=g), labels)
    kw = dict(proj_dim=PDIM, proj_seed=0, loss_type="mean", batch_size=2,
              device="cpu", projection="torch_chunked")
    a = GradFeaturizer(ad, FlowMatching(), t_set=[0.0], **kw).featurize(
        ds, seed=7, log_every=0)
    # reproduce the noise `featurize` draws per batch (seeded per timestep index)
    noise = []
    for start in range(0, 4, 2):
        torch.manual_seed(7 * 1000 + 0)
        noise.append(torch.randn_like(ds.images[start:start + 2]))
    states = torch.cat(noise).unsqueeze(1)                 # (4, 1, C, H, W)
    b = GradFeaturizer(ad, FlowMatching(), **kw).featurize_states(
        states, [0.0], labels, log_every=0)
    assert torch.equal(a, b), (a - b).abs().max()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_t6_drift_diagnostics_separate_rounding_from_a_different_sample():
    """`max |Δ|` alone cannot tell "same image, rounded differently" from "a
    different image" — a 100-step bf16 ODE amplifies last-bit noise, and the
    kernel choice moves with the machine/torch/batch. The gate therefore reports
    the per-sample cosine next to it, which does separate the two."""
    from balds.workflows.common import _replay_diagnostics

    torch.manual_seed(0)
    z = torch.randn(8, 16, 4, 4)
    same = _replay_diagnostics(z, z)
    assert same["max_abs_diff"] == 0.0 and same["cos_min"] == pytest.approx(1.0, abs=1e-6)
    assert same["n_cos_below_999"] == 0 and same["Q"] == 8

    rounded = _replay_diagnostics(z + 0.01 * torch.randn_like(z), z)
    assert rounded["max_abs_diff"] > 0 and rounded["cos_min"] > 0.999
    assert rounded["n_cos_below_999"] == 0

    different = _replay_diagnostics(z.flip(0), z)
    assert different["cos_min"] < 0.5 and different["n_cos_below_999"] == 8
