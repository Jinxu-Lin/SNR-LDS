from __future__ import annotations

import copy

import numpy as np
import torch

import balds.workflows.common as pipeline
import balds.workflows.inject as inject_app
import balds.workflows.usecases as usecases
from balds.workflows.inject import InjectUseCase
from balds.workflows.usecases import ScoreUseCase
from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.data.base import ImageDataset
from balds.report.tables import render_inject
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB


def _store(tmp_path):
    return ArtifactStore(
        ManifestDB(str(tmp_path / "manifest.db")),
        [LocalBackend(str(tmp_path / "data"))], code_version="stage5-test")


def _meta():
    return {"dataset": "toy", "per_class": 2, "train_sha256": "a" * 64,
            "pairs": {
                "zero": {"host_label": 0, "concept_name": "c0", "tier": "coarse",
                         "replaced_cifar10_indices": [0, 1]},
                "one": {"host_label": 1, "concept_name": "c1", "tier": "fine",
                        "replaced_cifar10_indices": [10, 11]},
            }}


def _perfect(host):
    values = np.arange(20, dtype=float) * -1e-4
    start = host * 10
    values[start:start + 2] = [2.0, 1.0]
    return values


class _StubMethod:
    feat_method = "pixel"
    needs_error_weight = False
    needs_repeats = False
    needs_checkpoints = False

    def score(self, feats, query, lam, *, device):
        bad0, bad1 = -_perfect(0), -_perfect(1)
        if float(lam) == 0.1:
            return np.stack([_perfect(0), bad0, _perfect(1), bad1], axis=1)
        return np.stack([bad0, _perfect(0), bad1, _perfect(1)], axis=1)


def test_score_selects_val_then_evaluate_and_compare(tmp_path, monkeypatch):
    store = _store(tmp_path)
    cfg = {
        "featurize": {"proj_dim": 4, "proj_seed": 0},
        "score": {"lambda_sweep": [0.1, 1.0], "lambda_selector": "oracle",
                  "lambda_selector_by_track": {"inject": "inject_val"}},
        "inject": {"k": 2, "bootstrap": {"n": 50, "seed": 42}},
    }
    labels = torch.tensor([0] * 10 + [1] * 10)
    train = ImageDataset(torch.zeros(20, 3, 32, 32), labels)
    monkeypatch.setattr(pipeline, "_load_ds", lambda _cfg, _dataset: (train, train))
    monkeypatch.setattr(inject_app, "_load_ds", lambda _cfg, _dataset: (train, train))
    monkeypatch.setattr(usecases.METHODS, "get", lambda _name: _StubMethod())
    base = RunSpec(dataset="toy", process="ddpm", seed=42)
    qspec = base.with_(method="stubA", query_type="inject")
    store.save(K.TRAIN_FEATURES, qspec, torch.zeros(20, 4), feat="pixel")
    store.save(K.QUERY_FEATURES, qspec, torch.zeros(4, 4), feat="pixel")
    store.save(K.INJECT_META, RunSpec(dataset="toy"), _meta())
    store.save(K.INJECT_QUERIES, base, {
        "images_u8": torch.zeros(4, 3, 32, 32, dtype=torch.uint8),
        "host_labels": torch.tensor([0, 0, 1, 1]),
        "split": torch.tensor([0, 1, 0, 1], dtype=torch.int8),
        "queries_sha256": "q" * 64,
    })

    scored = ScoreUseCase(store, cfg, device="cpu").run(
        "stubA", "toy", 42, process="ddpm", query_type="inject")
    assert scored["best_lam"] == 0.1 and scored["selector"] == "inject_val"
    score_meta = store.load(K.SCORES_META, qspec)
    assert score_meta["best_lam"] == 0.1
    assert score_meta["oracle_test_lam"] == 1.0
    assert score_meta["per_lambda_ap_val"]["0.1"] > score_meta["per_lambda_ap_val"]["1"]
    assert score_meta["per_lambda_ap_test"]["0.1"] < score_meta["per_lambda_ap_test"]["1"]

    app = InjectUseCase(store, cfg, device="cpu")
    result = app.evaluate("stubA", "toy", 42, process="ddpm", split="test")
    assert result["split"] == "test" and len(result["per_query"]) == 2
    assert set(result["per_concept"]) == {"c0", "c1"}
    assert "chance" in result and "overall" in result
    assert "INJECT toy/test stubA" in render_inject(result)

    other = copy.deepcopy(result)
    other["method"] = "stubB"
    for concept, summary in other["per_concept"].items():
        # A-B is +.10 for both concepts; only c1 is fine.
        summary["precision_at_k_pool"]["mean"] -= 0.10
    bspec = qspec.with_(method="stubB")
    store.save(K.INJECT_RESULT, bspec, other)
    compared = app.compare(
        ["stubA", "stubB"], "toy", 42, process="ddpm",
        split="test", metric="precision_at_k_pool")
    assert compared["fine_large_count"] == 1
    assert compared["fine_large_consistent_sign"]
    assert compared["fine_large_consistent_count"] == 1
