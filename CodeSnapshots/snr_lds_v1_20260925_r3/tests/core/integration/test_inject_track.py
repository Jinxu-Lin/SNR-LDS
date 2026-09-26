from __future__ import annotations

import pytest
import torch

import balds.workflows.features as pipeline
from balds.workflows.config import load_config
from balds.workflows.features import FeaturizeUseCase
from balds.cli.main import build_parser
from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.data.base import ImageDataset
from balds.models import UNetCFM, checkpoint_state
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB
from balds.artifacts.addressing import relpath


def _store(tmp_path):
    return ArtifactStore(
        ManifestDB(str(tmp_path / "manifest.db")),
        [LocalBackend(str(tmp_path / "data"))], code_version="stage4-test")


def test_inject_track_addressing_and_ground_truth_refusal():
    cfm = RunSpec(dataset="cifar10_inj4", process="cfm", seed=42,
                  method="das", query_type="inject")
    ddpm = cfm.with_(process="ddpm")
    assert relpath(K.QUERY_FEATURES, cfm, feat="das") == \
        "featurize/das/cifar10_inj4/seed_42/query_features_inject.pt"
    assert relpath(K.QUERY_FEATURES, ddpm, feat="das") == \
        "featurize/das/cifar10_inj4/ddpm/seed_42/query_features_inject.pt"
    assert relpath(K.SCORES, cfm) == \
        "scores/das/cifar10_inj4_inject/seed_42/scores.npy"
    assert relpath(K.SCORES, ddpm) == \
        "scores/das/cifar10_inj4_inject/ddpm/seed_42/scores.npy"
    assert relpath(K.INJECT_RESULT, ddpm) == \
        "scores/das/cifar10_inj4_inject/ddpm/seed_42/inject_result.json"
    for kind, key in ((K.GT_MATRIX, {}), (K.GT_LOSSES, {"eseed": 0}),
                      (K.GT_LOSSROW, {"eseed": 0, "m": 0}),
                      (K.LDS_RESULT, {})):
        with pytest.raises(ValueError, match="no artifact"):
            relpath(kind, ddpm, **key)
    with pytest.raises(ValueError, match="query_type"):
        RunSpec(dataset="x", query_type="other")


def test_tiny_inject_query_featurization_for_das_and_pixel(tmp_path, monkeypatch):
    store = _store(tmp_path)
    cfg = load_config({
        "storage.data_root": str(tmp_path / "data"),
        "featurize.batch_size": 1,
    })
    train = ImageDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
    test = ImageDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))
    monkeypatch.setattr(pipeline, "_load_ds", lambda _cfg, _dataset: (train, test))
    base = RunSpec(dataset="cifar10_inj4", process="cfm", seed=42)
    model = UNetCFM(base_ch=8, num_classes=10)
    store.save(K.CHECKPOINT, base, checkpoint_state(
        model, base_ch=8, num_classes=10, step=1,
        extra={"model_type": "cfm", "conditional": True, "p_uncond": 0.0}))
    query_u8 = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
    store.save(K.INJECT_QUERIES, base, {
        "images_u8": query_u8, "host_labels": torch.tensor([3, 7]),
        "queries_sha256": "q" * 64,
    })
    usecase = FeaturizeUseCase(store, cfg, device="cpu")
    grad = usecase.run(
        "cifar10_inj4", 42, feat="das", split="query", query_type="inject",
        proj_dim=8, proj_seed=0)
    pixel = usecase.run(
        "cifar10_inj4", 42, feat="pixel", split="query", query_type="inject")
    qspec = base.with_(query_type="inject")
    Fg = store.load(K.QUERY_FEATURES, qspec, feat="das", proj="p8s0")
    Fp = store.load(K.QUERY_FEATURES, qspec, feat="pixel")
    assert grad["query_inject"] == [2, 8] and tuple(Fg.shape) == (2, 8)
    assert pixel["query_inject"] == [2, 3072] and tuple(Fp.shape) == (2, 3072)

    train_out = usecase.run(
        "cifar10_inj4", 42, feat="das", split="train", max_samples=1,
        proj_dim=8, proj_seed=1, no_error_weight=True)
    assert train_out["error_weight"] == "skipped (--no-error-weight)"
    assert not store.exists(K.ERROR_WEIGHT, base, feat="das")


def test_cli_accepts_inject_track_and_no_error_weight():
    args = build_parser().parse_args([
        "featurize", "--dataset", "cifar10_inj4", "--query-type", "inject",
        "--split", "both", "--no-error-weight"])
    assert args.query_type == "inject" and args.no_error_weight



def test_cli_accepts_inject_evaluate():
    evaluate = build_parser().parse_args([
        "inject", "evaluate", "--method", "das", "--dataset", "cifar10_inj4",
        "--process", "ddpm", "--split", "test", "--k", "200"])
    assert evaluate.action == "evaluate" and evaluate.method == "das" and evaluate.k == 200
