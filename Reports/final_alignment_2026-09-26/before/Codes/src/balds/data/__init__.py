"""Data-Access: dataset loaders + subset-mask generation.

Importing registers the cifar*/artbench* loaders (heavy libs imported lazily, so
this stays import-safe without the raw data present).
"""
from __future__ import annotations

from . import artbench  # noqa: F401  (registers artbench2/5)
from . import cifar  # noqa: F401  (registers cifar2/5/10)
from . import inject  # noqa: F401  (registers cifar10_inj4/_inj8 and the ArtBench inject keys)
from balds.schema.registry import DATASETS
from .base import ImageDataset, SubsetView, balanced_query_indices, select_val_queries
from .masks import generate_subsets


def get_dataset(name: str, data_dir: str, **extra):
    """Load ``(train, test)`` for a registered dataset name.

    ``extra`` carries raw locations a loader needs BEYOND its primary
    ``data_dir``; the app layer passes ``raw_dirs=cfg["datasets"]["raw_dirs"]``
    (see ``balds.workflows.common._load_ds``). Only imported platforms use it —
    ``cifar2_das`` takes its pixels from the CIFAR cache but its *split* from
    index files inside the DAS archive, which is a second raw tree. Every other
    loader takes ``data_dir`` alone and ignores the rest.
    """
    return DATASETS.get(name)(data_dir, **extra)


__all__ = ["get_dataset", "generate_subsets", "ImageDataset", "SubsetView",
           "balanced_query_indices", "select_val_queries"]
