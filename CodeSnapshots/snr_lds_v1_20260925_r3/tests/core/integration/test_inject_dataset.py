from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from balds.workflows.config import load_config, resolve_p_uncond
from balds.workflows.inject import InjectUseCase
from balds.cli.main import build_parser
from balds.schema.artifact import ArtifactKind as K
from balds.schema.registry import DATASETS
from balds.schema.runspec import RunSpec
from balds.data.inject import build_cifar10_injected, detector_training_data
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB
from balds.artifacts.addressing import relpath


HOST_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CONCEPT_NAMES = [
    "sunflower", "castle", "butterfly", "leopard", "cattle",
    "wolf", "mushroom", "camel", "skyscraper", "tractor", "rocket",
]


class _Feature:
    def __init__(self, names):
        self.names = names


class _FakeSplit:
    def __init__(self, labels, *, names, field, offset):
        self._labels = list(labels)
        self._field = field
        self._offset = offset
        self.features = {field: _Feature(names)}

    def __iter__(self):
        for index, label in enumerate(self._labels):
            value = (self._offset + index) % 256
            yield {"img": np.full((32, 32, 3), value, dtype=np.uint8),
                   self._field: label}


def _fake_datasets():
    return {
        "cifar10": {
            "train": _FakeSplit(
                [label for label in range(10) for _ in range(8)],
                names=HOST_NAMES, field="label", offset=0),
            "test": _FakeSplit(
                [label for label in range(10) for _ in range(3)],
                names=HOST_NAMES, field="label", offset=31),
        },
        "cifar100": {
            "train": _FakeSplit(
                [label for label in range(len(CONCEPT_NAMES)) for _ in range(500)],
                names=CONCEPT_NAMES, field="fine_label", offset=67),
            "test": _FakeSplit(
                [label for label in range(len(CONCEPT_NAMES)) for _ in range(100)],
                names=CONCEPT_NAMES, field="fine_label", offset=149),
        },
    }


def _spec():
    return {
        "per_class": 2,
        "index_seed": 20260914,
        "concept_seed": 20260914,
        "cifar100_hf_id": "cifar100",
        "pairs": {
            host: [concept, "fine" if host in {"cat", "deer", "dog", "horse", "truck"}
                   else "coarse"]
            for host, concept in zip(HOST_NAMES, CONCEPT_NAMES)
        },
    }


@pytest.fixture
def fake_hf(monkeypatch):
    sources = _fake_datasets()

    def load_dataset(name, *, cache_dir):
        assert cache_dir
        return sources[name]

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    return sources


def _as_u8(images):
    return torch.round((images + 1.0) * 127.5).to(torch.uint8)


def _clean_u8(split):
    return torch.stack([
        torch.from_numpy(np.asarray(item["img"])).permute(2, 0, 1)
        for item in split
    ])


def test_build_is_exact_deterministic_and_pair_local(fake_hf):
    spec = _spec()
    train1, test1, meta1 = build_cifar10_injected("cache", spec)
    train2, test2, meta2 = build_cifar10_injected("cache", spec)
    assert DATASETS.has("cifar10_inj4")
    assert len(train1) == 80 and len(test1) == 30
    assert torch.equal(train1.labels, train2.labels)
    assert torch.equal(train1.images, train2.images)
    assert torch.equal(test1.images, test2.images)
    assert meta1 == meta2
    assert train1.inject_meta == meta1
    assert meta1["train_sha256"] == meta2["train_sha256"]

    clean = _clean_u8(fake_hf["cifar10"]["train"])
    injected = _as_u8(train1.images)
    replaced_all = []
    c100_train = list(fake_hf["cifar100"]["train"])
    for host, row in meta1["pairs"].items():
        positions = row["replaced_cifar10_indices"]
        injected_ids = row["injected_cifar100_train_ids"]
        detector_train = row["detector_cifar100_train_ids"]
        detector_test = row["detector_cifar100_test_ids"]
        assert len(positions) == len(injected_ids) == 2
        assert len(detector_train) == 300 and len(detector_test) == 100
        assert set(injected_ids).isdisjoint(detector_train)
        expected = torch.stack([
            torch.from_numpy(np.asarray(c100_train[index]["img"])).permute(2, 0, 1)
            for index in injected_ids
        ])
        assert torch.equal(injected[positions], expected), host
        replaced_all.extend(positions)
    keep = torch.ones(len(train1), dtype=torch.bool)
    keep[replaced_all] = False
    assert torch.equal(injected[keep], clean[keep])
    assert torch.equal(train1.labels, torch.tensor(
        [label for label in range(10) for _ in range(8)]))
    assert torch.equal(_as_u8(test1.images), _clean_u8(fake_hf["cifar10"]["test"]))

    swapped = copy.deepcopy(spec)
    swapped["pairs"]["airplane"][0] = "rocket"
    _train3, _test3, meta3 = build_cifar10_injected("cache", swapped)
    for host in HOST_NAMES[1:]:
        assert meta3["pairs"][host] == meta1["pairs"][host]


def test_unknown_concept_and_bad_pair_matrix_are_rejected(fake_hf):
    spec = _spec()
    spec["pairs"]["airplane"][0] = "not-a-cifar100-label"
    with pytest.raises(ValueError, match="unknown CIFAR-100 concept"):
        build_cifar10_injected("cache", spec)
    spec = _spec()
    spec["pairs"].pop("truck")
    with pytest.raises(ValueError, match="every CIFAR-10 host"):
        build_cifar10_injected("cache", spec)


def test_config_registry_address_and_build_idempotence(tmp_path, fake_hf):
    cfg = {
        "datasets": {"raw_dirs": {"cifar": str(tmp_path / "cache")}},
        "inject": {"cifar10_inj4": _spec()},
    }
    store = ArtifactStore(
        ManifestDB(str(tmp_path / "manifest.db")),
        [LocalBackend(str(tmp_path / "data"))], code_version="test")
    usecase = InjectUseCase(store, cfg, device="cpu")
    first = usecase.build("cifar10_inj4")
    assert not first["skipped"] and len(first["pairs"]) == 10
    assert {row["replaced"] for row in first["pairs"]} == {2}
    spec = RunSpec(dataset="cifar10_inj4")
    assert relpath(K.INJECT_META, spec) == "subsets/cifar10_inj4_inject_meta.json"
    stored = store.load(K.INJECT_META, spec)
    assert stored["train_sha256"] == first["train_sha256"]
    assert usecase.build("cifar10_inj4")["skipped"]

    usecase.cfg["inject"]["cifar10_inj4"]["index_seed"] += 1
    with pytest.raises(FileExistsError, match="differing inject_meta"):
        usecase.build("cifar10_inj4")
    assert not usecase.build("cifar10_inj4", force=True)["skipped"]


def test_defaults_resolve_pure_conditional_and_cli_build():
    cfg = load_config({})
    assert resolve_p_uncond(cfg, "cifar10_inj4") == 0.0
    assert set(cfg["inject"]["cifar10_inj4"]["pairs"]) == set(HOST_NAMES)
    args = build_parser().parse_args(["inject", "build", "--dataset", "cifar10_inj4"])
    assert (args.cmd, args.action, args.dataset, args.force) == (
        "inject", "build", "cifar10_inj4", False)


def test_duplicate_concepts_are_rejected(fake_hf):
    spec = _spec()
    spec["pairs"]["truck"][0] = spec["pairs"]["airplane"][0]
    with pytest.raises(ValueError, match="ten distinct"):
        build_cifar10_injected("cache", spec)


def test_detector_source_assembly_excludes_injected_rows_and_ids(fake_hf):
    spec = _spec()
    _train, _test, meta = build_cifar10_injected("cache", spec)
    images, labels, class_names, source = detector_training_data("cache", spec, meta)
    replaced = {index for row in meta["pairs"].values()
                for index in row["replaced_cifar10_indices"]}
    assert replaced.isdisjoint(source["clean_cifar10_indices"])
    assert len(source["clean_cifar10_indices"]) == 80 - 20
    assert len(images) == len(labels) == 60 + 10 * (300 + 100)
    assert len(class_names) == 20
    for host, row in meta["pairs"].items():
        assert set(row["injected_cifar100_train_ids"]).isdisjoint(
            source["cifar100_train_ids"][host])
    assert torch.bincount(labels, minlength=20).tolist() == [6] * 10 + [400] * 10


def test_real_hf_cache_when_cifar100_is_present():
    cfg = load_config({})
    cache = Path(cfg["datasets"]["raw_dirs"]["cifar"])
    cached = list(cache.glob("cifar100")) + list((cache / "hub").glob("datasets--cifar100"))
    if not cached:
        pytest.skip("CIFAR-100 is not present in the local HF cache")
    train, test, meta = build_cifar10_injected(
        str(cache), cfg["inject"]["cifar10_inj4"])
    assert len(train) == 50_000 and len(test) == 10_000
    assert len(meta["pairs"]) == 10 and len(meta["train_sha256"]) == 64
    assert all(len(row["replaced_cifar10_indices"]) == 200
               for row in meta["pairs"].values())
