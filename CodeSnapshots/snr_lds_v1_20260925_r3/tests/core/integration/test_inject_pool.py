from __future__ import annotations

import torch

import balds.workflows.queries as pipeline
from balds.workflows.queries import GenerateUseCase
from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.models import (DDPMSchedule, UNetCFM, checkpoint_state, ddim_sample,
                            euler_solve)
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB
from balds.artifacts.addressing import relpath


def _store(tmp_path):
    return ArtifactStore(
        ManifestDB(str(tmp_path / "manifest.db")),
        [LocalBackend(str(tmp_path / "data"))], code_version="stage2-test")


def _cfg(T=4):
    return {
        "datasets": {"toy": {"classes": list(range(2)), "num_classes": 2}},
        "model": {"num_classes": 2},
        "ddpm": {"T": T},
        "lds": {"Q": 2, "gen_seed_base": 123},
        "inject": {"pool": {"batch_size": 8, "gen_seed": 20260914}},
    }


class _ZeroModel(torch.nn.Module):
    def forward(self, x, t, class_label=None):
        return torch.zeros_like(x)


def test_ddim_and_euler_samplers_are_finite_and_deterministic():
    model = _ZeroModel()
    x = torch.randn(4, 3, 8, 8)
    schedule = DDPMSchedule(T=8)
    a = ddim_sample(model, schedule, x, steps=4, eta=0, device="cpu")
    b = ddim_sample(model, schedule, x, steps=4, eta=0, device="cpu")
    flow = euler_solve(model, x, steps=4, device="cpu")
    assert a.shape == b.shape == flow.shape == x.shape
    assert torch.equal(a, b)
    assert torch.isfinite(a).all() and torch.isfinite(flow).all()


def test_pool_is_batch_independent_balanced_and_refuses_tag(tmp_path):
    store = _store(tmp_path)
    cfg = _cfg()
    spec = RunSpec(dataset="toy", process="cfm", seed=42, conditional=True)
    model = UNetCFM(base_ch=8, num_classes=2)
    store.save(K.CHECKPOINT, spec, checkpoint_state(
        model, base_ch=8, num_classes=2, step=1,
        extra={"model_type": "cfm", "conditional": True, "p_uncond": 0.0}))
    usecase = GenerateUseCase(store, cfg, device="cpu")
    usecase.run("toy", 42, process="cfm", pool=8, sampler="euler",
                steps=2, batch=8, tag="batch8")
    usecase.run("toy", 42, process="cfm", pool=8, sampler="euler",
                steps=2, batch=4, tag="batch4")
    p8 = store.load(K.GEN_POOL, spec, tag="batch8")
    p4 = store.load(K.GEN_POOL, spec, tag="batch4")
    assert torch.equal(p8["images_u8"], p4["images_u8"])
    assert torch.equal(p8["labels"], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
    assert p8["images_u8"].dtype == torch.uint8
    assert p8["guidance"] == 0.0 and p8["p_uncond"] == 0.0
    assert len(p8["checkpoint_sha256"]) == 64
    try:
        usecase.run("toy", 42, process="cfm", pool=8, sampler="euler",
                    steps=2, batch=8, tag="batch8")
        raise AssertionError("existing tags must be refused")
    except FileExistsError:
        pass
    assert relpath(K.GEN_POOL, spec, tag="euler2_n8") == \
        "generations/toy/cfm_cond/seed_42/pool_euler2_n8.pt"


class _CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = []

    def forward(self, x, t, class_label=None):
        self.labels.append(None if class_label is None else class_label.detach().cpu().clone())
        return torch.zeros_like(x)


def test_cfg_gate_uses_checkpoint_p_uncond_and_uncond_has_no_label(tmp_path, monkeypatch):
    store = _store(tmp_path)
    cfg = _cfg(T=3)
    models = []

    def build_stub(_ckpt, device):
        model = _CountingModel()
        models.append(model)
        return model

    monkeypatch.setattr(pipeline, "build_model", build_stub)
    uc = GenerateUseCase(store, cfg, device="cpu")

    conditional = RunSpec(dataset="toy", process="ddpm", seed=1, conditional=True)
    store.save(K.CHECKPOINT, conditional, {"p_uncond": 0.0})
    uc.run("toy", 1, process="ddpm", Q=2)
    assert len(models[-1].labels) == 3
    assert all(value is not None for value in models[-1].labels)

    legacy = RunSpec(dataset="toy", process="ddpm", seed=2, conditional=True)
    store.save(K.CHECKPOINT, legacy, {})  # missing p_uncond is the legacy case
    uc.run("toy", 2, process="ddpm", Q=2)
    assert len(models[-1].labels) == 6
    assert any(torch.equal(value, torch.full((2,), 2)) for value in models[-1].labels)

    trained_cfg = RunSpec(dataset="toy", process="ddpm", seed=4, conditional=True)
    store.save(K.CHECKPOINT, trained_cfg, {"p_uncond": 0.1})
    uc.run("toy", 4, process="ddpm", Q=2)
    assert len(models[-1].labels) == 6
    assert any(torch.equal(value, torch.full((2,), 2)) for value in models[-1].labels)

    unconditional = RunSpec(dataset="toy", process="ddpm", seed=3, conditional=False)
    store.save(K.CHECKPOINT, unconditional, {"p_uncond": 0.1})
    uc.run("toy", 3, process="ddpm", Q=2, conditional=False)
    assert len(models[-1].labels) == 3
    assert models[-1].labels == [None, None, None]


def test_ddpm_sample_accepts_exact_initial_noise_shape():
    from balds.models import ddpm_sample

    x = torch.randn(2, 3, 4, 4)
    out = ddpm_sample(_ZeroModel(), DDPMSchedule(T=2), tuple(x.shape),
                      x_T=x, device="cpu")
    assert out.shape == x.shape and torch.isfinite(out).all()
