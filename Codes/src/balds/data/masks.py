"""LDS subset masks — Bernoulli(α) inclusion vectors (pure compute).

The mask *artifact* is loaded/saved through the store (``ArtifactKind.SUBSET_MASKS``);
this module only generates them. The seed is fixed (independent of the model
training seed), matching the legacy ``generate_subsets``.
"""
from __future__ import annotations

import numpy as np


def generate_subsets(n: int, alpha: float, m: int, seed: int = 42) -> list[np.ndarray]:
    """Return ``m`` boolean masks of shape ``(n,)``; sample ``i`` kept w.p. ``alpha``."""
    rng = np.random.RandomState(seed)
    return [rng.rand(n) < alpha for _ in range(m)]
