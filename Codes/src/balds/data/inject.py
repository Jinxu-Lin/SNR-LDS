"""Deterministic CIFAR-10 contamination benchmark construction."""
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from typing import Any

import numpy as np

from balds.schema.registry import DATASETS
from .base import ImageDataset


BUILDER_VERSION = "cifar10-inject-v1"
DETECTOR_TRAIN_PER_CONCEPT = 300
DETECTOR_TEST_PER_CONCEPT = 100


def _feature_names(split: Any, field: str) -> list[str]:
    try:
        names = split.features[field].names
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(f"dataset split has no named {field!r} feature") from exc
    if not names:
        raise ValueError(f"dataset feature {field!r} has no class names")
    return [str(name) for name in names]


def _images_and_labels(split: Any, *, label_field: str) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for item in split:
        image = np.asarray(item["img"])
        if image.shape == (3, 32, 32):
            image = image.transpose(1, 2, 0)
        if image.shape != (32, 32, 3):
            raise ValueError(f"expected CIFAR image shape (32,32,3), got {image.shape}")
        if image.dtype != np.uint8:
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
                raise ValueError("CIFAR pixels must be finite values in [0,255]")
            image = image.astype(np.uint8)
        images.append(np.ascontiguousarray(image.transpose(2, 0, 1)))
        labels.append(int(item[label_field]))
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def _image_dataset(images_u8: np.ndarray, labels: np.ndarray) -> ImageDataset:
    import torch

    images = torch.from_numpy(np.ascontiguousarray(images_u8)).to(torch.float32)
    images = images.div(127.5).sub(1.0)
    return ImageDataset(images, torch.from_numpy(np.ascontiguousarray(labels)).long())


def _train_sha256(images_u8: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(images_u8, dtype=np.uint8).tobytes())
    digest.update(np.ascontiguousarray(labels, dtype=np.int64).tobytes())
    return digest.hexdigest()


def detector_training_data(data_dir: str, spec: dict, meta: dict):
    """Assemble the detector's leakage-free 20-class uint8 training set.

    Host rows come from clean CIFAR-10 with every injected position removed;
    concept rows come only from the detector-specific CIFAR-100 train/test IDs
    filed in ``meta``.  The generated pool is not an input to this function.
    """
    import torch
    from datasets import load_dataset

    cifar10 = load_dataset("cifar10", cache_dir=data_dir)
    cifar100 = load_dataset(str(spec["cifar100_hf_id"]), cache_dir=data_dir)
    host_names = _feature_names(cifar10["train"], "label")
    concept_names = _feature_names(cifar100["train"], "fine_label")
    _validate_spec(spec, host_names)
    clean_u8, host_labels = _images_and_labels(cifar10["train"], label_field="label")
    concept_train_u8, concept_train_labels = _images_and_labels(
        cifar100["train"], label_field="fine_label")
    concept_test_u8, concept_test_labels = _images_and_labels(
        cifar100["test"], label_field="fine_label")
    keep = np.ones(len(host_labels), dtype=bool)
    concept_images, concept_targets = [], []
    source = {"clean_cifar10_indices": [], "cifar100_train_ids": {},
              "cifar100_test_ids": {}}
    class_names = list(host_names)
    for host_label, host_name in enumerate(host_names):
        row = meta["pairs"][host_name]
        if int(row["host_label"]) != host_label:
            raise ValueError(f"metadata host label mismatch for {host_name!r}")
        replaced = np.asarray(row["replaced_cifar10_indices"], dtype=np.int64)
        injected = set(int(v) for v in row["injected_cifar100_train_ids"])
        train_ids = np.asarray(row["detector_cifar100_train_ids"], dtype=np.int64)
        test_ids = np.asarray(row["detector_cifar100_test_ids"], dtype=np.int64)
        if injected.intersection(int(v) for v in train_ids):
            raise ValueError(f"detector train IDs overlap injected IDs for {host_name!r}")
        keep[replaced] = False
        concept_id = int(row["concept_fine_id"])
        if str(row["concept_name"]) != concept_names[concept_id]:
            raise ValueError(f"metadata concept identity mismatch for {host_name!r}")
        if not np.all(concept_train_labels[train_ids] == concept_id):
            raise ValueError(f"detector train IDs leave concept {row['concept_name']!r}")
        if not np.all(concept_test_labels[test_ids] == concept_id):
            raise ValueError(f"detector test IDs leave concept {row['concept_name']!r}")
        concept_images.extend((concept_train_u8[train_ids], concept_test_u8[test_ids]))
        concept_targets.extend((
            np.full(len(train_ids), 10 + host_label, dtype=np.int64),
            np.full(len(test_ids), 10 + host_label, dtype=np.int64),
        ))
        source["cifar100_train_ids"][host_name] = train_ids.tolist()
        source["cifar100_test_ids"][host_name] = test_ids.tolist()
        class_names.append(str(row["concept_name"]))
    source["clean_cifar10_indices"] = np.flatnonzero(keep).tolist()
    images = np.concatenate([clean_u8[keep], *concept_images])
    labels = np.concatenate([host_labels[keep], *concept_targets])
    return (torch.from_numpy(np.ascontiguousarray(images)),
            torch.from_numpy(np.ascontiguousarray(labels)).long(), class_names, source)


def detector_training_data_all_real(data_dir: str, spec: dict, meta: dict):
    """Every real image, for a detector that only screens generated images.

    Researcher ruling 2026-09-15: nothing needs to be held out from a detector
    that never sees training images at screening time, and it is not reported.
    Hosts are all CIFAR-10 train rows (the clean originals at the injected
    positions included) plus all CIFAR-10 test rows, 6,000 per class; concepts
    are every CIFAR-100 train and test row of each paired fine class, 600 each
    (the injected rows included). Labels keep the isolated layout: 0-9 hosts,
    10-19 the paired concepts in host order.
    """
    import torch
    from datasets import load_dataset

    cifar10 = load_dataset("cifar10", cache_dir=data_dir)
    cifar100 = load_dataset(str(spec["cifar100_hf_id"]), cache_dir=data_dir)
    host_names = _feature_names(cifar10["train"], "label")
    concept_names = _feature_names(cifar100["train"], "fine_label")
    _validate_spec(spec, host_names)
    train_u8, train_labels = _images_and_labels(cifar10["train"], label_field="label")
    test_u8, test_labels = _images_and_labels(cifar10["test"], label_field="label")
    concept_train_u8, concept_train_labels = _images_and_labels(
        cifar100["train"], label_field="fine_label")
    concept_test_u8, concept_test_labels = _images_and_labels(
        cifar100["test"], label_field="fine_label")
    concept_images, concept_targets = [], []
    source = {"mode": "all_real",
              "cifar10_rows": {"train": int(len(train_labels)), "test": int(len(test_labels))},
              "cifar100_train_ids": {}, "cifar100_test_ids": {}}
    class_names = list(host_names)
    for host_label, host_name in enumerate(host_names):
        row = meta["pairs"][host_name]
        if int(row["host_label"]) != host_label:
            raise ValueError(f"metadata host label mismatch for {host_name!r}")
        concept_id = int(row["concept_fine_id"])
        if str(row["concept_name"]) != concept_names[concept_id]:
            raise ValueError(f"metadata concept identity mismatch for {host_name!r}")
        train_ids = np.flatnonzero(concept_train_labels == concept_id)
        test_ids = np.flatnonzero(concept_test_labels == concept_id)
        concept_images.extend((concept_train_u8[train_ids], concept_test_u8[test_ids]))
        concept_targets.extend((
            np.full(len(train_ids), 10 + host_label, dtype=np.int64),
            np.full(len(test_ids), 10 + host_label, dtype=np.int64),
        ))
        source["cifar100_train_ids"][host_name] = train_ids.tolist()
        source["cifar100_test_ids"][host_name] = test_ids.tolist()
        class_names.append(str(row["concept_name"]))
    images = np.concatenate([train_u8, test_u8, *concept_images])
    labels = np.concatenate([train_labels, test_labels, *concept_targets])
    source["n_per_class"] = np.bincount(labels, minlength=2 * len(host_names)).tolist()
    return (torch.from_numpy(np.ascontiguousarray(images)),
            torch.from_numpy(np.ascontiguousarray(labels)).long(), class_names, source)


#: detector data modes for the CIFAR injected datasets (``inject.<block>.data_mode``)
DETECTOR_DATA_MODES = {"isolated": detector_training_data,
                       "all_real": detector_training_data_all_real}


def _validate_spec(spec: Mapping[str, Any], host_names: list[str]) -> None:
    required = {"per_class", "index_seed", "concept_seed", "cifar100_hf_id", "pairs"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"injection spec lacks {sorted(missing)}")
    per_class = int(spec["per_class"])
    if per_class <= 0:
        raise ValueError("inject per_class must be positive")
    pairs = spec["pairs"]
    if not isinstance(pairs, Mapping) or set(pairs) != set(host_names):
        raise ValueError("inject pairs must name every CIFAR-10 host class exactly once")
    concepts: list[str] = []
    for host, pair in pairs.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"inject pair for {host!r} must be [concept, tier]")
        concepts.append(str(pair[0]))
        if str(pair[1]) not in {"coarse", "fine"}:
            raise ValueError(f"inject tier for {host!r} must be coarse or fine")
    if len(set(concepts)) != len(concepts):
        raise ValueError("inject pairs must use ten distinct CIFAR-100 concepts")


def build_cifar10_injected(
        data_dir: str, spec: dict) -> tuple[ImageDataset, ImageDataset, dict]:
    """Replace fixed CIFAR-10 rows with paired CIFAR-100 images.

    The returned tensors remain in CIFAR-10 HF order with the original host
    labels. Random draws are keyed independently by host class, so changing one
    pair cannot perturb any other host's positions or concept sample IDs.
    """
    from datasets import load_dataset

    cifar10 = load_dataset("cifar10", cache_dir=data_dir)
    cifar100 = load_dataset(str(spec["cifar100_hf_id"]), cache_dir=data_dir)
    host_names = _feature_names(cifar10["train"], "label")
    concept_names = _feature_names(cifar100["train"], "fine_label")
    _validate_spec(spec, host_names)

    clean_train_u8, train_labels = _images_and_labels(
        cifar10["train"], label_field="label")
    test_u8, test_labels = _images_and_labels(cifar10["test"], label_field="label")
    concept_train_u8, concept_train_labels = _images_and_labels(
        cifar100["train"], label_field="fine_label")
    _concept_test_u8, concept_test_labels = _images_and_labels(
        cifar100["test"], label_field="fine_label")

    injected_u8 = clean_train_u8.copy()
    per_class = int(spec["per_class"])
    # Detector concept draws per pair. The defaults are the 4% benchmark's; the 8%
    # key sets 100 train (every CIFAR-100 train row left after 400 injected) and 100 test.
    detector_train_n = int(spec.get("detector_train_per_concept", DETECTOR_TRAIN_PER_CONCEPT))
    detector_test_n = int(spec.get("detector_test_per_concept", DETECTOR_TEST_PER_CONCEPT))
    if detector_train_n <= 0 or detector_test_n <= 0:
        raise ValueError("detector_train_per_concept and detector_test_per_concept must be positive")
    pairs_meta: dict[str, dict[str, Any]] = {}
    for host_label, host_name in enumerate(host_names):
        concept_name, tier = spec["pairs"][host_name]
        concept_name = str(concept_name)
        if concept_name not in concept_names:
            raise ValueError(f"unknown CIFAR-100 concept {concept_name!r} for host {host_name!r}")
        concept_fine_id = concept_names.index(concept_name)
        host_rows = np.flatnonzero(train_labels == host_label)
        concept_train_rows = np.flatnonzero(concept_train_labels == concept_fine_id)
        concept_test_rows = np.flatnonzero(concept_test_labels == concept_fine_id)
        if len(host_rows) < per_class:
            raise ValueError(f"host {host_name!r} has only {len(host_rows)} rows")
        if len(concept_train_rows) < per_class + detector_train_n:
            raise ValueError(f"concept {concept_name!r} lacks train rows for injection+detector")
        if len(concept_test_rows) < detector_test_n:
            raise ValueError(f"concept {concept_name!r} lacks detector test rows")

        positions_rng = np.random.RandomState(int(spec["index_seed"]) * 100 + host_label)
        concept_rng = np.random.RandomState(int(spec["concept_seed"]) * 100 + host_label)
        replaced = np.sort(positions_rng.choice(host_rows, per_class, replace=False))
        injected_ids = np.sort(
            concept_rng.choice(concept_train_rows, per_class, replace=False))
        remaining_train = np.setdiff1d(concept_train_rows, injected_ids, assume_unique=True)
        detector_train_ids = np.sort(concept_rng.choice(
            remaining_train, detector_train_n, replace=False))
        detector_test_ids = np.sort(concept_rng.choice(
            concept_test_rows, detector_test_n, replace=False))

        injected_u8[replaced] = concept_train_u8[injected_ids]
        pairs_meta[host_name] = {
            "host_label": host_label,
            "concept_name": concept_name,
            "concept_fine_id": concept_fine_id,
            "tier": str(tier),
            "replaced_cifar10_indices": [int(v) for v in replaced],
            "injected_cifar100_train_ids": [int(v) for v in injected_ids],
            "detector_cifar100_train_ids": [int(v) for v in detector_train_ids],
            "detector_cifar100_test_ids": [int(v) for v in detector_test_ids],
        }

    meta = {
        # the registered key (the loader sets it); direct callers keep the original name
        "dataset": str(spec.get("dataset_key", "cifar10_inj4")),
        "per_class": per_class,
        "index_seed": int(spec["index_seed"]),
        "concept_seed": int(spec["concept_seed"]),
        "cifar100_hf_id": str(spec["cifar100_hf_id"]),
        "pairs": pairs_meta,
        "train_sha256": _train_sha256(injected_u8, train_labels),
        "builder_version": BUILDER_VERSION,
    }
    train = _image_dataset(injected_u8, train_labels)
    train.inject_meta = meta
    return train, _image_dataset(test_u8, test_labels), meta


def _load_registered_cifar10(dataset_key: str):
    def load(data_dir: str, *, inject_spec=None, **_):
        if not isinstance(inject_spec, Mapping):
            raise ValueError(f"{dataset_key} requires its inject config block")
        spec = dict(inject_spec)
        spec["dataset_key"] = dataset_key
        train, test, _meta = build_cifar10_injected(data_dir, spec)
        return train, test
    return load


#: 4% pilot and the 8% retrain (§6.2-54); each reads ``inject.<key>``.
for _key in ("cifar10_inj4",):
    DATASETS.add(_key, _load_registered_cifar10(_key))


# --------------------------------------------------------------------------- #
# Shared ArtBench-10 pixel-space injection for both latent backbones
# --------------------------------------------------------------------------- #

