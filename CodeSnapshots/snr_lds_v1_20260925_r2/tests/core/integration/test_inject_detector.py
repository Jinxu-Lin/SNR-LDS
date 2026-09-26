from __future__ import annotations

import csv
from pathlib import Path

import torch

from balds.workflows.inject import InjectUseCase
from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.evaluation.detector import predict_proba, screen_probabilities, train_detector
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB
from balds.artifacts.addressing import relpath


def _store(tmp_path):
    return ArtifactStore(
        ManifestDB(str(tmp_path / "manifest.db")),
        [LocalBackend(str(tmp_path / "data"))], code_version="stage3-test")


def test_detector_one_epoch_and_probability_shape():
    generator = torch.Generator().manual_seed(4)
    images = torch.randint(0, 256, (20 * 8, 3, 32, 32),
                           dtype=torch.uint8, generator=generator)
    labels = torch.arange(20).repeat_interleave(8)
    state, report = train_detector(
        images, labels, num_classes=20,
        cfg={"epochs": 1, "batch_size": 160, "lr": 0.01,
             "momentum": 0.9, "weight_decay": 0.0, "holdout_frac": 0.25},
        device="cpu", seed=0)
    probabilities = predict_proba(state, images[:7], batch=4, device="cpu")
    assert probabilities.shape == (7, 20) and probabilities.dtype == torch.float32
    assert torch.allclose(probabilities.sum(1), torch.ones(7), atol=1e-6)
    assert len(report["confusion"]) == 20
    assert report["n_per_class"] == [8] * 20


def test_screen_rates_distinguish_paired_any_and_include_threshold_edge():
    probabilities = torch.zeros(4, 20)
    probabilities[0, 10], probabilities[0, 0] = 0.90, 0.10  # paired h0, edge
    probabilities[1, 11], probabilities[1, 0] = 0.91, 0.09  # any, not paired h0
    probabilities[2, 11], probabilities[2, 1] = 0.90, 0.10  # paired h1, edge
    probabilities[3, 1] = 1.0                              # host prediction
    result = screen_probabilities(
        probabilities, torch.tensor([0, 0, 1, 1]), threshold=0.9)
    assert len(result["flagged"]) == 3
    assert result["per_host"]["0"] == {
        "n_pool": 2, "n_flag_paired": 1, "rate_paired": 0.5,
        "n_flag_any_concept": 2, "rate_any_concept": 1.0,
    }
    assert result["per_host"]["1"]["rate_paired"] == 0.5
    assert result["per_host"]["1"]["rate_any_concept"] == 0.5


def _meta():
    return {
        "dataset": "cifar10_inj4", "train_sha256": "a" * 64,
        "pairs": {
            f"host{host}": {"host_label": host, "concept_name": f"concept{host}",
                             "tier": "fine" if host >= 5 else "coarse"}
            for host in range(10)
        },
    }


def test_review_export_edit_import_round_trip_and_unknown_decision(tmp_path):
    store = _store(tmp_path)
    cfg = {"inject": {"review": {"per_concept_cap": 150},
                      "split": {"val_frac": 0.5, "seed": 0}}}
    usecase = InjectUseCase(store, cfg, device="cpu")
    spec = RunSpec(dataset="cifar10_inj4", process="ddpm", seed=42)
    images = torch.arange(40, dtype=torch.uint8)[:, None, None, None].expand(-1, 3, 32, 32)
    labels = torch.arange(10).repeat_interleave(4)
    store.save(K.GEN_POOL, spec, {
        "images_u8": images, "labels": labels, "n": 40,
    }, tag="tiny")
    store.save(K.INJECT_META, RunSpec(dataset="cifar10_inj4"), _meta())
    flagged = []
    for host in range(10):
        for offset in range(2):
            flagged.append({"pool_index": host * 4 + offset, "host_label": host,
                            "pred_class": 10 + host, "pred_concept_host": host,
                            "prob": 0.99 - offset * 0.01})
    store.save(K.INJECT_CANDIDATES, spec, {
        "threshold": 0.9, "detector_sha256": "d" * 64,
        "pool_sha256": "p" * 64, "flagged": flagged, "per_host": {},
    }, tag="tiny")

    exported = usecase.review_export(
        "cifar10_inj4", 42, process="ddpm", tag="tiny")
    assert exported["rows"] == 20 and exported["sheets"] == 10
    table_rel = relpath(K.INJECT_REVIEW_TABLE, spec, tag="tiny")
    table_path = Path(store.local.abspath(table_rel))
    with table_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if int(row["host_label"]) in (0, 1):
            row["decision"] = "accept"
        else:
            row["decision"] = "reject"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)

    out = usecase.queries(
        "cifar10_inj4", 42, process="ddpm", tag="tiny",
        review=table_rel, version="v1", reviewer="researcher")
    assert out["Q"] == 4 and out["val"] == 2 and out["test"] == 2
    queries = store.load(K.INJECT_QUERIES, spec)
    assert queries["pool_indices"].tolist() == [0, 1, 4, 5]
    assert queries["host_labels"].tolist() == queries["concept_hosts"].tolist()
    assert queries["n_accepted_per_concept"]["0"] == 2
    assert len(queries["review_csv_sha256"]) == len(queries["queries_sha256"]) == 64

    rows[0]["decision"] = "maybe"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    try:
        usecase.queries(
            "cifar10_inj4", 42, process="ddpm", tag="tiny",
            review=table_rel, version="v2", force=True)
        raise AssertionError("unknown decision must be refused")
    except ValueError as exc:
        assert "unknown review decision" in str(exc)

    assert relpath(K.INJECT_DETECTOR, RunSpec(dataset="cifar10_inj4")) == \
        "results/inject/cifar10_inj4/detector.pt"
    assert relpath(K.INJECT_CANDIDATES, spec, tag="tiny") == \
        "generations/cifar10_inj4/ddpm_cond/seed_42/pool_tiny_candidates.json"
    assert relpath(K.INJECT_QUERIES, spec) == \
        "generations/cifar10_inj4/ddpm_cond/seed_42/inject_queries.pt"
