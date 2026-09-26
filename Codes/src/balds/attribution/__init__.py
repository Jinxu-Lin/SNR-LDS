"""Horizontal axis — attribution scores (projected-Gram family + denoising layers).

Importing this package registers all weights, transforms and score methods
(the declarative replacement for the legacy ``07_score.py`` dispatch).
"""
from __future__ import annotations

from . import kernel  # noqa: F401
from . import transforms  # noqa: F401  (registers shape + eb transforms)
from . import methods  # noqa: F401  (registers all score methods)
from .error_weight import compute_error_weight
from .featurize import GradFeaturizer
from .kernel import trak_kernel
from .methods import FEAT_T_GRID, WeightedKernelMethod

__all__ = ["trak_kernel", "WeightedKernelMethod", "GradFeaturizer", "compute_error_weight",
           "FEAT_T_GRID"]
