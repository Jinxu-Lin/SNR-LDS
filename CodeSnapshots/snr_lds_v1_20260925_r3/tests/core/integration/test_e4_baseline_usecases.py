from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import balds.workflows.baselines as baseline_app
from balds.workflows.baselines import AbuUseCase, NdaUseCase, ParameterWeightingUseCase
from balds.workflows.config import load_config
from balds.cli.main import build_parser
from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.data.base import ImageDataset
from balds.models.flow import FlowMatching
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(1, 2, 1)
        self.second = torch.nn.Conv2d(2, 1, 1)

    def forward(self, x, t, labels):
        return self.second(torch.tanh(self.first(x)))


def _store(tmp_path):
    root = str(tmp_path / "data")
    return ArtifactStore(ManifestDB(str(tmp_path / "manifest.db")),
                            [LocalBackend(root)], code_version="e4-test")


def _data():
    generator = torch.Generator().manual_seed(17)
    train = ImageDataset(torch.randn(3, 1, 2, 2, generator=generator),
                         torch.zeros(3, dtype=torch.long))
    test = ImageDataset(torch.randn(2, 1, 2, 2, generator=generator),
                        torch.zeros(2, dtype=torch.long))
    return train, test


def test_parameter_weighting_prepare_fit_score_and_config_identity(tmp_path, monkeypatch):
    store, (train, test), model = _store(tmp_path), _data(), _TinyModel()
    cfg = load_config({})
    cfg["featurize"].update({"proj_dim": 4, "proj_seed": 2})
    cfg["parameter_weighting"].update({
        "proj_dim": 4, "proj_seed": 2, "projection": "torch_chunked",
        "batch_size": 1, "learning_query_count": 2, "learning_query_ids": [],
        "epochs": 2, "top_k": 1, "ridge": 0.5,
        "contribution_query_chunk": 1,
    })
    base = RunSpec(dataset="toy", seed=4)
    store.save(K.TRAIN_FEATURES, base, torch.randn(3, 4), feat="dtrak_T100")
    learning = {"samples": test.images.clone(), "labels": test.labels.clone()}
    store.save(K.GENERATION, base, learning, name="weight_learning_samples")
    store.save(K.GENERATION, base, learning)
    monkeypatch.setattr(baseline_app, "_platform",
                        lambda helper, dataset, seed, process, conditional:
                        (base, model, train, test, FlowMatching()))

    usecase = ParameterWeightingUseCase(store, cfg, device="cpu")
    prepared = usecase.prepare("toy", 4, config_tag="tiny")
    fitted = usecase.fit("toy", 4, config_tag="tiny")
    scored = usecase.score("toy", 4, config_tag="tiny")
    assert prepared["shape"] == [3, 2, 4]
    assert fitted["groups"] == 4
    assert scored["shape"] == [3, 2]
    spec = RunSpec(dataset="toy", seed=4, method="dtrak_param_weighted_T100")
    scores = store.load(K.SCORES, spec, config="tiny")
    assert scores.shape == (3, 2) and np.isfinite(scores).all()
    manifest = store.load(K.PARAM_CONTRIBUTIONS, spec, config="tiny")
    assert manifest["blocks"] == [[0, 1], [1, 2]]
    assert "contributions" not in manifest
    for q0, q1 in manifest["blocks"]:
        block = store.load(K.PARAM_CONTRIBUTIONS, spec, config="tiny", q0=q0, q1=q1)
        assert block["learning_query_ids"] == [q0]
        assert list(block["contributions"].shape) == [3, 1, 4]

    changed = copy.deepcopy(cfg)
    changed["parameter_weighting"]["ridge"] = 0.75
    with pytest.raises(ValueError, match="new --config-tag"):
        ParameterWeightingUseCase(store, changed, device="cpu").prepare(
            "toy", 4, config_tag="tiny")
    with pytest.raises(ValueError, match="new --config-tag"):
        ParameterWeightingUseCase(store, changed, device="cpu").fit(
            "toy", 4, config_tag="tiny", force=True)
    cfg["parameter_weighting"]["learning_generation"] = "samples"
    with pytest.raises(ValueError, match="independent named generation"):
        usecase.prepare("toy", 4, config_tag="bad")


def test_abu_prepare_score_restores_model(tmp_path, monkeypatch):
    store, (train, test), model = _store(tmp_path), _data(), _TinyModel()
    cfg = load_config({})
    cfg["abu"].update({"fisher_samples": 2, "fisher_mc": 1,
                       "measurement_mc": 2, "query_mc": 1,
                       "damping": 0.2, "step_size": 0.01})
    base = RunSpec(dataset="toy", seed=5)
    store.save(K.GENERATION, base, {"samples": test.images, "labels": test.labels})
    monkeypatch.setattr(baseline_app, "_platform",
                        lambda helper, dataset, seed, process, conditional:
                        (base, model, train, test, FlowMatching()))
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    usecase = AbuUseCase(store, cfg, device="cpu")
    usecase.prepare("toy", 5, config_tag="diag")
    out = usecase.score("toy", 5, config_tag="diag")
    assert out["shape"] == [3, 2]
    assert all(torch.equal(value, original[name]) for name, value in model.named_parameters())
    spec = RunSpec(dataset="toy", seed=5, method="abu_plus")
    assert np.isfinite(store.load(K.SCORES, spec, config="diag")).all()


def test_abu_ekfac_fit_score_real_routing_and_metadata(tmp_path, monkeypatch):
    store, (train, test), model = _store(tmp_path), _data(), _TinyModel()
    cfg = load_config({})
    cfg["abu"].update({"preconditioner": "ekfac", "parameter_domain": "ekfac",
                       "ekfac_fit_epochs": 2, "ekfac_eig_epochs": 2, "ekfac_batch_size": 2,
                       "measurement_mc": 2, "query_mc": 2, "flip_mode": "max"})
    base = RunSpec(dataset="toy", seed=5)
    store.save(K.GENERATION, base, {"samples": test.images, "labels": test.labels, "query_ids": [7, 12]})
    monkeypatch.setattr(baseline_app, "_platform",
                        lambda helper, dataset, seed, process, conditional:
                        (base, model, train, test, FlowMatching()))
    def forbidden(*args, **kwargs):
        raise AssertionError("EK-FAC must not call diagonal Fisher")
    monkeypatch.setattr(baseline_app, "empirical_fisher_diagonal", forbidden)
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    uc = AbuUseCase(store, cfg, device="cpu")
    uc.prepare("toy", 5, config_tag="ek")
    assert uc.score("toy", 5, config_tag="ek")["shape"] == [3, 2]
    assert all(torch.equal(value, original[name]) for name, value in model.named_parameters())
    spec = base.with_(method="abu_plus")
    prep = store.load(K.ABU_PREP, spec, config="ek")
    meta = store.load(K.SCORES_META, spec, config="ek")
    assert "fisher_diagonal" not in prep
    assert meta["preconditioner"] == "ekfac"
    assert meta["curvature"]["n_eigenvalue_draws"] == 6
    assert meta["curvature"]["effective_gain"] == 1.0
    assert meta["curvature"]["damping_mode"] == "relative_layer"
    assert meta["query_ids"] == [7, 12]
    assert not store.exists(K.EKFAC_FACTORS, base)  # IF factors never overwritten
    # Same config can read a compatible project factor package via existing store.
    store.save(K.EKFAC_FACTORS, base, prep["fisher_ekfac"])
    cfg["abu"]["curvature_source"] = "ekfac_factors"
    uc.prepare("toy", 5, config_tag="imported")
    uc.score("toy", 5, config_tag="imported")
    np.testing.assert_array_equal(store.load(K.SCORES, spec, config="ek"),
                                  store.load(K.SCORES, spec, config="imported"))
    legacy = copy.deepcopy(prep["fisher_ekfac"])
    legacy["meta"].pop("curvature_kind")
    store.save(K.EKFAC_FACTORS, base, legacy)
    with pytest.raises(ValueError, match="empirical_fisher"):
        uc.prepare("toy", 5, config_tag="bad-ggn")
    cfg["abu"]["preconditioner"] = "typo"
    with pytest.raises(ValueError, match="preconditioner"):
        uc.prepare("toy", 5, config_tag="typo")


def test_nda_direct_scores_file_under_config(tmp_path, monkeypatch):
    store, (train, test) = _store(tmp_path), _data()
    cfg = load_config({})
    cfg["nda"].update({"patch_size": 1, "timesteps": [1, 4], "schedule_steps": 10,
                       "timestep_recipes": [{"patch_size": 1}, {"patch_size": 3}],
                       "train_chunk": 1, "query_patch_chunk": 1})
    base = RunSpec(dataset="toy", seed=6)
    store.save(K.GENERATION, base, {"samples": test.images, "labels": test.labels})
    monkeypatch.setattr(baseline_app, "_load_ds", lambda cfg, dataset: (train, test))
    out = NdaUseCase(store, cfg, device="cpu").score("toy", 6, config_tag="p1")
    assert out["shape"] == [3, 2]
    spec = RunSpec(dataset="toy", seed=6, method="nda")
    scores = store.load(K.SCORES, spec, config="p1")
    meta = store.load(K.SCORES_META, spec, config="p1")
    assert scores.shape == (3, 2) and np.isfinite(scores).all()
    assert meta["shape"] == [3, 2] and meta["query_ids"] == [0, 1]
    assert [r["patch_size"] for r in meta["timestep_recipes"]] == [1, 3]
    assert meta["timestep_indices"] == [0, 1]






def test_abu_latent_flip_reads_pair_cache_not_spatial_flip(tmp_path):
    store, (train, _) = _store(tmp_path), _data()
    base = RunSpec(dataset="toy")
    encoded_flip = train.images + 9
    store.save(K.LATENTS, base, {"latents": train.images, "labels": train.labels}, split="train")
    store.save(K.LATENTS, base, {"latents": encoded_flip, "labels": train.labels}, split="train", flip=True)
    actual = AbuUseCase(store, {}, device="cpu")._flipped(
        SimpleNamespace(_is_latent=lambda ds: True), "toy", train, {"flip_mode": "max"})
    assert torch.equal(actual, encoded_flip)
    assert not torch.equal(actual, train.images.flip(-1))


@pytest.mark.parametrize("command", ["parameter-weighting", "abu", "nda"])
def test_e4_commands_refuse_non_main_table_query_track(command):
    action = "score"
    with pytest.raises(SystemExit):
        build_parser().parse_args([command, action, "--config-tag", "x",
                                   "--query-type", "inject"])
