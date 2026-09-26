"""EK-FAC on the SD3.5+LoRA latent platform ([n] × [o], TASK_EKFAC_LORA §4-(3))
— the full fit → score → evaluate → repeats → tweedie chain on a stub latent
platform, CPU only, no diffusers weights.

What only this suite can pin down (the unit suite covers the block math):

* the platform branch of ``EkfacFitUseCase`` / ``_prepare``: model = LoRA
  adapter rebuilt from a LoRA-only checkpoint, data = the latents cache (never
  ``get_dataset``), hflip = the pre-encoded flip-pair cache;
* the two serial K-FAC blocks per adapter flow through the store artifacts and
  the damping-grid selection unchanged (``lam_b`` is None end to end);
* ``ekfac.query_chunk`` is a pure residency knob: a chunked re-stream files a
  bit-identical score matrix (§3-e — on the real platform the resident (Q, P)
  block is ~19 GB and must be chunked);
* the solver-variant trio (stratified_antithetic + blockshrink) runs on this
  platform under its guarded ``<raw>__<tag>`` name — the main table's three
  rows (ekfac_if / __sa_blockshrink / _tweedie) share this one chain;
* the platform refuses ``--uncond`` (prompt-conditional by construction).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows.config import load_config  # noqa: E402
from balds.workflows.curvature import (EkfacFitUseCase, EkfacRepeatsUseCase,  # noqa: E402
                               EkfacScoreUseCase)
from balds.workflows.usecases import EvaluateUseCase  # noqa: E402
from balds.schema.artifact import ArtifactKind as K  # noqa: E402
from balds.schema.runspec import RunSpec  # noqa: E402
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB  # noqa: E402

_DS = "_eklora_toy"
C, H, W = 4, 8, 8            # tiny "latent" geometry
CTX_L, CTX_D, POOL_D = 5, 12, 6
S = 10                       # styles
_N, _M, _Q = 6, 4, 3
RANK, ALPHA = 2, 2.0
_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
_SEED = 42


class _StubSD3(nn.Module):
    """Same call signature as SD3Transformer2DModel (mirrors the stub of
    test_artbench_platform.py); token-wise linears named like the real
    attention projections so the LoRA target suffixes match."""

    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(C, C)
        self.attn.to_k = nn.Linear(C, C)
        self.attn.to_v = nn.Linear(C, C)
        self.attn.to_out = nn.Sequential(nn.Linear(C, C))
        self.ctx_proj = nn.Linear(CTX_D, C)
        self.pool_proj = nn.Linear(POOL_D, C)

    def forward(self, hidden_states, encoder_hidden_states, pooled_projections,
                timestep, return_dict=False):
        B = hidden_states.shape[0]
        tokens = hidden_states.flatten(2).transpose(1, 2)          # (B, HW, C)
        h = self.attn.to_q(tokens) + self.attn.to_k(tokens) + self.attn.to_v(tokens)
        h = self.attn.to_out(h)
        h = h + self.ctx_proj(encoder_hidden_states).mean(dim=-2, keepdim=True)
        h = h + self.pool_proj(pooled_projections).unsqueeze(-2)
        h = h + (timestep.reshape(B, 1, 1) / 1000.0)
        out = h.transpose(1, 2).reshape(hidden_states.shape)
        return (out,)


def _stub_loader(base_model, *, device, dtype):
    """Stand-in for ``load_sd3_transformer``: deterministic weights, mirrored
    freeze/eval/device handling."""
    torch.manual_seed(0)
    tf = _StubSD3().to(dtype)
    tf.requires_grad_(False)
    return tf.to(device).eval()


def _cfg(root, **extra):
    cfg = load_config({
        "storage.data_root": f"{root}/data",
        "datasets._eklora_toy": {"platform": "sd35_lora",
                                 "classes": list(range(S)), "num_classes": S},
        "artbench.base_model": "stub-model",
        "artbench.base_dtype": "float32",
        "artbench.lora": {"rank": RANK, "alpha": ALPHA, "targets": _TARGETS},
        "ekfac.fit_epochs": 1, "ekfac.eig_epochs": 1,
        "ekfac.fit_batch_size": 3, "ekfac.eig_batch_size": 3,
        "ekfac.mc_loss": 2, "ekfac.mc_measurement": 2, "ekfac.grad_chunk": 2,
        "ekfac.query_coords_dtype": "float32",
        "tweedie.repeats": 2, "tweedie.sigma_samples": 4,
        "tweedie.gamma_by_method": {"ekfac_if_tweedie": {_DS: 1.0}},
        **extra,
    })
    cfg["ekfac"]["damping_grid"] = [1e-8, 1e-4]
    return cfg


@pytest.fixture(scope="module")
def env(request):
    mp = pytest.MonkeyPatch()
    request.addfinalizer(mp.undo)
    mp.setattr("balds.models.sd3.load_sd3_transformer", _stub_loader)

    root = tempfile.mkdtemp()
    store = ArtifactStore(ManifestDB(f"{root}/m.db"), [LocalBackend(f"{root}/data")],
                             code_version="test")
    cfg = _cfg(root)

    g = torch.Generator().manual_seed(1)
    ds_spec = RunSpec(dataset=_DS)
    z = torch.randn(_N, C, H, W, generator=g)
    zf = torch.randn(_N, C, H, W, generator=g)     # a DISTINCT orientation, not a spatial flip
    zt = torch.randn(4, C, H, W, generator=g)
    y = torch.arange(_N, dtype=torch.long) % S
    for split, latents, labels, key in (("train", z, y, {}),
                                        ("train", zf, y, {"flip": True}),
                                        ("test", zt, torch.arange(4) % S, {})):
        store.save(K.LATENTS, ds_spec,
                   {"latents": latents.half(), "labels": labels, "split": split,
                    "flip": bool(key), "base_model": "stub-model"},
                   split=split, **key)
    store.save(K.PROMPT_EMBEDS, ds_spec,
               {"prompt_embeds": torch.randn(S, CTX_L, CTX_D, generator=g),
                "pooled_embeds": torch.randn(S, POOL_D, generator=g),
                "prompts": [f"style {i}" for i in range(S)],
                "base_model": "stub-model"})

    from balds.workflows.latent import _lora_checkpoint, artbench_cfg, build_latent_model
    from balds.models.sd3 import reset_lora_
    base = RunSpec(dataset=_DS, seed=_SEED, process="cfm", conditional=True)
    adapter = build_latent_model(store, cfg, _DS, None, device="cpu")
    reset_lora_(adapter, _SEED)
    with torch.no_grad():                          # nonzero up-projections: a zero
        for k, p in adapter.named_parameters():    # delta path has degenerate curvature
            if "lora_B" in k:
                p.copy_(torch.randn(p.shape, generator=g) / RANK)
    store.save(K.CHECKPOINT, base,
               _lora_checkpoint(adapter, artbench_cfg(cfg), step=1, process="cfm"))

    store.save(K.GENERATION, base,
               {"samples": torch.randn(_Q, C, H, W, generator=g),
                "labels": torch.arange(_Q, dtype=torch.long) % S,
                "Q": _Q, "gen_seed": 0, "ode_steps": 2, "conditional": True,
                "platform": "sd35_lora"})
    store.save(K.SUBSET_MASKS, base,
               [np.random.RandomState(m).rand(_N) < 0.5 for m in range(_M)])
    store.save(K.GT_MATRIX, base.with_(query_type="gen"),
               np.random.RandomState(7).rand(_M, _Q))
    return store, cfg, root


def test_fit_covers_the_lora_blocks_and_skips(env):
    store, cfg, _ = env
    fit = EkfacFitUseCase(store, cfg, device="cpu")
    out = fit.run(_DS, _SEED, process="cfm", conditional=True)
    # 4 wrapped Linears × (lora_A + lora_B) = 8 covered blocks, bias-free
    assert out["layers"] == 2 * len(_TARGETS)
    assert out["n_params"] == len(_TARGETS) * 2 * RANK * C
    base = RunSpec(dataset=_DS, seed=_SEED, process="cfm", conditional=True)
    state = store.load(K.EKFAC_FACTORS, base)
    assert state["meta"]["platform"] == "sd35_lora"
    assert state["meta"]["hflip"] is True
    assert state["meta"]["train_mode_noop"] is True
    assert all(ly["lam_b"] is None for ly in state["layers"].values())
    assert all(".lora_A" in n or ".lora_B" in n for n in state["meta"]["layer_order"])
    assert fit.run(_DS, _SEED, process="cfm", conditional=True) == {"skipped": True}


def test_platform_refuses_uncond(env):
    store, cfg, _ = env
    with pytest.raises(ValueError, match="prompt-conditional"):
        EkfacFitUseCase(store, cfg, device="cpu").run(
            _DS, _SEED, process="cfm", conditional=False)
    with pytest.raises(ValueError, match="prompt-conditional"):
        EkfacScoreUseCase(store, cfg, device="cpu").run(
            _DS, _SEED, query_type="gen", process="cfm", conditional=False)


def test_score_streams_selects_and_evaluate_consumes(env):
    store, cfg, _ = env
    EkfacFitUseCase(store, cfg, device="cpu").run(_DS, _SEED, process="cfm", conditional=True)
    out = EkfacScoreUseCase(store, cfg, device="cpu").run(
        _DS, _SEED, query_type="gen", process="cfm", conditional=True)
    assert out["shape"] == [_N, _Q]
    assert out["best_damping"] in cfg["ekfac"]["damping_grid"]
    spec = RunSpec(dataset=_DS, seed=_SEED, method="ekfac_if", query_type="gen",
                   process="cfm", conditional=True)
    scores = store.load(K.SCORES, spec)
    assert scores.shape == (_N, _Q) and np.isfinite(scores).all()
    for d in cfg["ekfac"]["damping_grid"]:
        assert store.exists(K.SCORES_LAMBDA, spec, lam=d)
    ev = EvaluateUseCase(store, cfg).run("ekfac_if", _DS, _SEED, query_type="gen",
                                        process="cfm", conditional=True)
    assert np.isfinite(ev["mean_lds"]) and ev["n_queries"] == _Q




def test_query_chunk_is_a_pure_residency_knob(env):
    """§3-e: chunked query blocks + per-block train re-streams file EXACTLY the
    matrices of the resident run (global query indices key the MC draws)."""
    store, cfg, root = env
    EkfacFitUseCase(store, cfg, device="cpu").run(_DS, _SEED, process="cfm", conditional=True)
    EkfacScoreUseCase(store, cfg, device="cpu").run(
        _DS, _SEED, query_type="gen", process="cfm", conditional=True)
    spec = RunSpec(dataset=_DS, seed=_SEED, method="ekfac_if", query_type="gen",
                   process="cfm", conditional=True)
    before = {d: store.load(K.SCORES_LAMBDA, spec, lam=d).copy()
              for d in cfg["ekfac"]["damping_grid"]}
    chunked = _cfg(root, **{"ekfac.query_chunk": 2})     # 2 < Q=3 -> two streams
    out = EkfacScoreUseCase(store, chunked, device="cpu").run(
        _DS, _SEED, query_type="gen", process="cfm", conditional=True, rescore=True)
    assert out["shape"] == [_N, _Q]
    for d in chunked["ekfac"]["damping_grid"]:
        after = store.load(K.SCORES_LAMBDA, spec, lam=d)
        assert np.array_equal(before[d], after), f"damping {d}: chunking moved scores"


def test_paper_kernel_runs_on_the_platform(env):
    """The paper's kernel `fmas_raw` (stratified_antithetic + blockshrink, bound
    to the name) files under its own name on the latent platform too."""
    store, cfg, root = env
    EkfacFitUseCase(store, cfg, device="cpu").run(_DS, _SEED, process="cfm", conditional=True)
    vcfg = _cfg(root, **{"ekfac.blockshrink_grid": [1e-2, 1e-1]})
    bad = _cfg(root, **{"ekfac.sampling": "stratified_antithetic", "ekfac.blockshrink_grid": [1e-2, 1e-1]})
    with pytest.raises(ValueError, match="bound"):
        EkfacScoreUseCase(store, bad, device="cpu").run(
            _DS, _SEED, query_type="gen", process="cfm", conditional=True)   # ekfac_if + sa --set
    out = EkfacScoreUseCase(store, vcfg, device="cpu").run(
        _DS, _SEED, query_type="gen", process="cfm", conditional=True, method="fmas_raw")
    assert out["best_damping"] in vcfg["ekfac"]["blockshrink_grid"]
    spec = RunSpec(dataset=_DS, seed=_SEED, method="fmas_raw",
                   query_type="gen", process="cfm", conditional=True)
    assert np.isfinite(store.load(K.SCORES, spec)).all()
    ref = RunSpec(dataset=_DS, seed=_SEED, method="ekfac_if", query_type="gen",
                  process="cfm", conditional=True)
    assert not np.array_equal(store.load(K.SCORES, spec), store.load(K.SCORES, ref))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
