"""DAS-archive DDPM adapter (TASK CODE_DAS_DDPM_ADAPTER) — CPU, tiny UNet2DModel.

What has to hold for `balds ekfac fit/score` on cifar2_das:

* the adapter feeds the UNet the timestep index DDPM._t_index used to corrupt x_t;
* build_model routes a das_ddpm checkpoint to the adapter (legacy attention keys
  renamed, strict load, resnet dropout 0.1, every parameter trainable) and a
  UNetCFM checkpoint through the old path unchanged;
* the importer files the archive model at the platform's checkpoint identity and
  refuses a scheduler that is not the registered ddpm process;
* an oversize generation is truncated to Q, and the GT columns follow the same
  queries (gen: first Q; val: balanced indices), with Q in SCORES_META;
* EK-FAC covers exactly the Conv2d + Linear modules of the diffusers UNet.
"""
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("diffusers")
from diffusers import DDPMScheduler, UNet2DModel  # noqa: E402

from balds.workflows import das_import as di  # noqa: E402
from balds.workflows.config import load_config  # noqa: E402
from balds.workflows.curvature import EkfacFitUseCase, EkfacScoreUseCase, _prepare  # noqa: E402
from balds.schema.artifact import ArtifactKind as K  # noqa: E402
from balds.schema.registry import DATASETS, PROCESSES  # noqa: E402
from balds.schema.runspec import RunSpec  # noqa: E402
from balds.data.base import ImageDataset, balanced_query_indices  # noqa: E402
from balds.evaluation.lds import lds_of_scores  # noqa: E402
from balds.models import build_model, checkpoint_state  # noqa: E402
from balds.models.das_ddpm import (DasDdpmAdapter, build_das_ddpm,  # noqa: E402
                                     verify_scheduler_config)
from balds.models.unet import UNetCFM  # noqa: E402
from balds.attribution.ekfac import kfac_target_modules  # noqa: E402
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB, das_archive as da  # noqa: E402

# the archive config's shape, shrunk (two levels, one attention level, 32 channels)
TINY_UNET = {
    "_class_name": "UNet2DModel", "_diffusers_version": "0.16.1", "act_fn": "silu",
    "add_attention": True, "attention_head_dim": None, "block_out_channels": [32, 32],
    "center_input_sample": False, "class_embed_type": None,
    "down_block_types": ["DownBlock2D", "AttnDownBlock2D"], "downsample_padding": 0,
    "flip_sin_to_cos": False, "freq_shift": 1, "in_channels": 3, "layers_per_block": 1,
    "mid_block_scale_factor": 1, "norm_eps": 1e-06, "norm_num_groups": 8,
    "num_class_embeds": None, "out_channels": 3, "resnet_time_scale_shift": "scale_shift",
    "sample_size": 32, "time_embedding_type": "positional",
    "up_block_types": ["AttnUpBlock2D", "UpBlock2D"],
}
# verbatim ddpm/ddpm_42/scheduler/scheduler_config.json
SCHEDULER = {
    "_class_name": "DDPMScheduler", "_diffusers_version": "0.16.1", "beta_end": 0.02,
    "beta_schedule": "linear", "beta_start": 0.0001, "clip_sample": True,
    "clip_sample_range": 1.0, "dynamic_thresholding_ratio": 0.995,
    "num_train_timesteps": 1000, "prediction_type": "epsilon", "sample_max_value": 1.0,
    "thresholding": False, "trained_betas": None, "variance_type": "fixed_small",
}
_TO_LEGACY = {"to_q": "query", "to_k": "key", "to_v": "value", "to_out.0": "proj_attn"}


def _tiny_unet(seed: int = 0) -> UNet2DModel:
    """The diffusers-0.16 architecture of TINY_UNET: its mid-block attention had a
    GroupNorm, which current diffusers only builds when attn_norm_num_groups is set
    (the stored config, like the archive's, predates that key)."""
    torch.manual_seed(seed)
    net = UNet2DModel.from_config({**TINY_UNET, "attn_norm_num_groups": TINY_UNET["norm_num_groups"]})
    assert "mid_block.attentions.0.group_norm.weight" in net.state_dict()
    return net.eval()


def _legacy_state(unet: nn.Module) -> dict:
    """The state dict under diffusers-0.16 attention names, as the archive stores it."""
    out = {}
    for k, v in unet.state_dict().items():
        for new, old in _TO_LEGACY.items():
            k = k.replace(f".{new}.", f".{old}.")
        out[k] = v.clone()
    return out


def _das_ckpt(unet: nn.Module) -> dict:
    return {"platform": "das_ddpm", "model_type": "ddpm", "conditional": False,
            "unet_config": dict(TINY_UNET), "unet_state": _legacy_state(unet),
            "scheduler_config": dict(SCHEDULER), "source": {}, "step": None}


def _store(root: Path):
    return ArtifactStore(ManifestDB(str(root / "m.db")), [LocalBackend(str(root / "data"))],
                            code_version="test")


# --------------------------------------------------------------------------- #
def test_adapter_timestep_is_ddpm_t_index():
    class _Spy(nn.Module):
        def forward(self, sample, timestep, return_dict=True):
            self.seen = timestep
            return SimpleNamespace(sample=sample.double())

    t = torch.cat([torch.linspace(0, 1, 4001),
                   torch.tensor([0.0, 4.999e-4, 5e-4, 5.001e-4, 0.4995, 0.9994, 0.9995, 1.0])])
    ad = DasDdpmAdapter(_Spy())
    out = ad(torch.zeros(len(t), 3, 2, 2), t, torch.arange(len(t)))
    assert torch.equal(ad.unet.seen, PROCESSES.get("ddpm")._t_index(t))
    assert out.dtype == torch.float32
    ad(torch.zeros(2, 3, 2, 2), 0.25)                     # python scalar, no label
    assert torch.equal(ad.unet.seen, torch.tensor([250, 250]))
    print(f"  adapter: timestep == DDPM._t_index on {len(t)} levels incl. rounding edges")


def test_scheduler_contract_is_the_registered_ddpm_process():
    assert verify_scheduler_config(SCHEDULER)["num_train_timesteps"] == 1000
    sched = DDPMScheduler.from_config(dict(SCHEDULER))
    proc = PROCESSES.get("ddpm")
    assert proc.T == sched.config.num_train_timesteps
    assert torch.equal(proc._schedule("cpu").alpha_bars, sched.alphas_cumprod)
    print("  scheduler: archive config's alphas_cumprod == registered ddpm table, bit-exact")


def test_build_model_dispatch_and_unetcfm_untouched():
    ref = _tiny_unet()
    ckpt = _das_ckpt(ref)
    model = build_model(copy.deepcopy(ckpt), device="cpu")
    assert isinstance(model, DasDdpmAdapter) and not model.training
    assert all(p.requires_grad for p in model.parameters())
    drops = {n: m.p for n, m in model.named_modules() if isinstance(m, nn.Dropout)}
    resnet = {n: p for n, p in drops.items() if ".resnets." in n}
    assert resnet and all(p == 0.1 for p in resnet.values())
    assert all(p == 0.0 for n, p in drops.items() if n not in resnet)
    assert any(n.startswith("unet.mid_block.resnets.") for n in resnet)

    x = torch.randn(3, 3, 32, 32, generator=torch.Generator().manual_seed(3))
    t = torch.tensor([0.0, 0.5004, 0.999])
    with torch.no_grad():
        want = ref(x, torch.tensor([0, 500, 999])).sample
        assert torch.equal(model(x, t, torch.tensor([1, 0, 1])), want)
        assert torch.equal(model(x, t), want)         # class_label ignored

    bad = copy.deepcopy(ckpt)
    bad["unet_state"] = {k: v for k, v in bad["unet_state"].items() if ".proj_attn." not in k}
    with pytest.raises(RuntimeError, match="Missing key"):
        build_model(bad, device="cpu")

    torch.manual_seed(1)
    cfm = UNetCFM(base_ch=8, num_classes=None)
    back = build_model(checkpoint_state(cfm, base_ch=8, num_classes=None, step=1,
                                        extra={"model_type": "cfm"}), device="cpu")
    assert type(back) is UNetCFM and not back.training
    sd = back.state_dict()
    assert list(sd) == list(cfm.state_dict())
    assert all(torch.equal(sd[k], v) for k, v in cfm.state_dict().items())
    print(f"  dispatch: das_ddpm -> adapter (outputs bit-equal to the source UNet, "
          f"{len(resnet)} resnet dropouts at 0.1, strict load); UNetCFM path unchanged")


def test_kfac_targets_are_conv2d_and_linear_only():
    model = build_das_ddpm(_das_ckpt(_tiny_unet()), device="cpu")
    mods = kfac_target_modules(model)
    want = [n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    assert list(mods) == want
    assert {type(m) for m in mods.values()} == {nn.Conv2d, nn.Linear}
    for leaf in ("conv_in", "conv_out", "time_embedding.linear_1", "time_embedding.linear_2",
                 "attentions.0.to_q", "attentions.0.to_k", "attentions.0.to_v",
                 "attentions.0.to_out.0", "resnets.0.conv1", "resnets.0.time_emb_proj"):
        assert any(n.endswith(leaf) for n in mods), leaf
    assert not any(isinstance(m, nn.GroupNorm) for m in mods.values())
    print(f"  kfac targets: {len(mods)} modules = every Conv2d/Linear, no GroupNorm")


# --------------------------------------------------------------------------- #
def _write_model_dir(root: Path, state: dict, scheduler: dict) -> None:
    (root / "_code_DAS").mkdir(parents=True, exist_ok=True)
    paths = {k: Path(p) for k, p in da.model_paths(str(root), 42).items()}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    paths["unet_config"].write_text(json.dumps(TINY_UNET))
    paths["scheduler_config"].write_text(json.dumps(scheduler))
    torch.save(state, paths["unet_weights"])


def test_import_model_files_checkpoint_and_refuses_other_schedulers():
    tmp = Path(tempfile.mkdtemp())
    root = tmp / "archive"
    ref = _tiny_unet()
    legacy = _legacy_state(ref)
    _write_model_dir(root, legacy, SCHEDULER)
    store = _store(tmp)
    uc = di.DasImportUseCase(store, load_config({"storage.data_root": str(tmp / "data")}))

    out = uc.import_model(str(root))
    assert out["n_params"] == sum(p.numel() for p in ref.parameters())
    assert (tmp / "data/checkpoints/cifar2_das/ddpm_cond/seed_42/final.pt").is_file()
    spec = RunSpec(dataset="cifar2_das", process="ddpm", seed=42, conditional=True)
    ck = store.load(K.CHECKPOINT, spec)
    assert (ck["platform"], ck["model_type"], ck["conditional"], ck["step"]) == \
        ("das_ddpm", "ddpm", False, None)
    assert ck["unet_config"] == TINY_UNET and ck["scheduler_config"] == SCHEDULER
    assert list(ck["unet_state"]) == list(legacy)
    assert all(torch.equal(ck["unet_state"][k], v) for k, v in legacy.items())
    weights = Path(da.model_paths(str(root), 42)["unet_weights"])
    assert ck["source"]["sha256"]["unet_weights"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert set(ck["source"]["sha256"]) == {"unet_config", "unet_weights", "scheduler_config"}
    assert ck["source"]["paths"]["unet_weights"] == \
        "CIFAR2/saved/5000-0.5/ddpm/ddpm_42/unet/diffusion_pytorch_model.bin"
    assert uc.import_model(str(root)) == "skipped"

    for key, value in (("beta_schedule", "scaled_linear"), ("beta_start", 0.00085),
                       ("beta_end", 0.012), ("num_train_timesteps", 4000),
                       ("prediction_type", "v_prediction"), ("trained_betas", [0.1] * 1000)):
        _write_model_dir(root, legacy, {**SCHEDULER, key: value})
        with pytest.raises(ValueError, match=key):
            uc.import_model(str(root), force=True)
    assert store.load(K.CHECKPOINT, spec)["scheduler_config"] == SCHEDULER
    print("  import: checkpoint at ddpm_cond/seed_42 with archive state + sha256; "
          "skip on rerun; 6 scheduler mismatches refused under --force")


# --------------------------------------------------------------------------- #
_DS = "_das_ddpm_toy"
_N, _M, _NGEN, _NTEST, _Q = 6, 4, 6, 8, 4


def _toy_loader(data_dir, **_):
    g = torch.Generator().manual_seed(0)
    train = ImageDataset(torch.randn(_N, 3, 32, 32, generator=g).clamp(-1, 1),
                         torch.tensor([0, 1] * (_N // 2)))
    test = ImageDataset(torch.randn(_NTEST, 3, 32, 32, generator=g).clamp(-1, 1),
                        torch.tensor([0] * (_NTEST // 2) + [1] * (_NTEST // 2)))
    return train, test


if not DATASETS.has(_DS):
    DATASETS.add(_DS, _toy_loader)


def _cfg(tmp: Path, Q: int):
    cfg = load_config({
        "storage.data_root": str(tmp / "data"),
        f"lds.Q_by_dataset.{_DS}": Q,
        "ekfac.fit_epochs": 1, "ekfac.eig_epochs": 1,
        "ekfac.fit_batch_size": 3, "ekfac.eig_batch_size": 3,
        "ekfac.mc_loss": 2, "ekfac.mc_measurement": 2, "ekfac.grad_chunk": 2,
        "ekfac.query_coords_dtype": "float32",
    })
    cfg["ekfac"]["damping_grid"] = [1e-6, 1e-2]
    return cfg


@pytest.fixture(scope="module")
def toy():
    tmp = Path(tempfile.mkdtemp())
    store = _store(tmp)
    base = RunSpec(dataset=_DS, seed=42, process="ddpm", conditional=True)
    store.save(K.CHECKPOINT, base, _das_ckpt(_tiny_unet()))
    g = torch.Generator().manual_seed(1)
    store.save(K.GENERATION, base,
               {"samples": torch.randn(_NGEN, 3, 32, 32, generator=g).clamp(-1, 1),
                "labels": torch.zeros(_NGEN, dtype=torch.long), "Q": _NGEN,
                "gen_seed": None, "ode_steps": None, "conditional": True})
    store.save(K.SUBSET_MASKS, base, [np.random.RandomState(m).rand(_N) < 0.5 for m in range(_M)])
    store.save(K.GT_MATRIX, base.with_(query_type="gen"), np.random.RandomState(7).rand(_M, _NGEN))
    store.save(K.GT_MATRIX, base.with_(query_type="val"), np.random.RandomState(8).rand(_M, _NTEST))
    cfg = _cfg(tmp, _Q)
    out = EkfacFitUseCase(store, cfg, device="cpu").run(_DS, 42, process="ddpm", conditional=True)
    return SimpleNamespace(tmp=tmp, store=store, cfg=cfg, base=base, fit=out)


def test_fit_runs_on_the_adapter(toy):
    n_target = len(kfac_target_modules(build_das_ddpm(_das_ckpt(_tiny_unet()), device="cpu")))
    assert toy.fit["layers"] == n_target and toy.fit["n_params"] > 0
    print(f"  fit: {toy.fit['layers']} layers, {toy.fit['n_params']} preconditioned params")


def test_prepare_truncates_oversize_generation_only(toy):
    gen = toy.store.load(K.GENERATION, toy.base)
    kw = dict(query_type="gen", process="ddpm", conditional=True)
    ctx = _prepare(toy.store, toy.cfg, "cpu", _DS, 42, **kw)
    assert ctx.Q == _Q
    assert torch.equal(ctx.q_images, gen["samples"][:_Q])
    assert torch.equal(ctx.q_labels, gen["labels"][:_Q])
    exact = _prepare(toy.store, _cfg(toy.tmp, _NGEN), "cpu", _DS, 42, **kw)
    assert exact.Q == _NGEN and torch.equal(exact.q_images, gen["samples"])

    val = _prepare(toy.store, toy.cfg, "cpu", _DS, 42, **{**kw, "query_type": "val"})
    _, test = _toy_loader(None)
    cols = balanced_query_indices(test.labels, _Q)
    assert cols == [0, 1, 4, 5] and torch.equal(val.q_images, test.images[cols])
    print(f"  _prepare: gen {_NGEN}->{_Q} rows (identity at Q={_NGEN}); val rows {cols}")


def test_score_selects_on_the_matching_gt_columns_and_records_Q(toy):
    _, test = _toy_loader(None)
    masks = toy.store.load(K.SUBSET_MASKS, toy.base)
    for track, cols in (("gen", list(range(_Q))),
                        ("val", balanced_query_indices(test.labels, _Q))):
        out = EkfacScoreUseCase(toy.store, toy.cfg, device="cpu").run(
            _DS, 42, query_type=track, process="ddpm", conditional=True)
        assert out["shape"] == [_N, _Q]
        spec = toy.base.with_(method="ekfac_if", query_type=track)
        assert toy.store.load(K.SCORES_META, spec)["Q"] == _Q
        gt = toy.store.load(K.GT_MATRIX, spec)[:, cols]
        for d in toy.cfg["ekfac"]["damping_grid"]:
            raw = toy.store.load(K.SCORES_LAMBDA, spec, lam=d)
            assert raw.shape == (_N, _Q) and np.isfinite(raw).all()
            assert abs(lds_of_scores(raw, gt, masks) - out["per_damping"][f"{d:g}"]) < 1e-12
    print("  score: (N, Q) per-damping LDS == LDS on the GT columns of the scored "
          "queries (gen first Q, val balanced); SCORES_META.Q recorded")


def test_score_selects_on_the_first_M_subsets(toy):
    """Item 8: resolve_M below the filed row count -> the damping pick uses the
    first M masks/GT rows (lds_rows); the filed artifacts keep every row."""
    cfg = _cfg(toy.tmp, _Q)
    cfg["lds"]["M_by_dataset"][_DS] = 2
    out = EkfacScoreUseCase(toy.store, cfg, device="cpu").run(
        _DS, 42, query_type="gen", process="ddpm", conditional=True, rescore=True)
    spec = toy.base.with_(method="ekfac_if", query_type="gen")
    masks = toy.store.load(K.SUBSET_MASKS, toy.base)
    gt = toy.store.load(K.GT_MATRIX, spec)
    assert len(masks) == _M and gt.shape == (_M, _NGEN)
    for d in cfg["ekfac"]["damping_grid"]:
        raw = toy.store.load(K.SCORES_LAMBDA, spec, lam=d)
        want = lds_of_scores(raw, gt[:2, :_Q], masks[:2])
        assert abs(want - out["per_damping"][f"{d:g}"]) < 1e-12
    print(f"  score: M=2 of {_M} filed subsets used for the damping pick; files untouched")
