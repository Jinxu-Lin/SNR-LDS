"""Pure helpers for the AN-2 counterfactual chain (no IO, no model code).

* removal sets from a score matrix (top-k most positive influence per query),
  or random (seeded) — both as boolean KEEP masks, exactly ``k`` False;
* the two distances the analysis reports: per-image L2 on [0, 1]-scaled pixels
  (DAS convention: uint8 -> float / 255, flattened, Euclidean norm) and CLIP
  cosine (computed by the caller from embeddings);
* the hypergeometric expectation used to sanity-check the random arm.
"""
from __future__ import annotations

import numpy as np


def topk_removal_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """KEEP mask with the ``k`` most positive-influence samples removed.

    ``scores`` is one query's column, shape ``(N,)``; larger = more supportive
    (the pipeline's sign convention: removal raises the query's loss). Ties are
    broken by index (stable sort), so the set is a pure function of the column.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not 0 < k < s.shape[0]:
        raise ValueError(f"k={k} must be in (0, N={s.shape[0]})")
    order = np.argsort(-s, kind="stable")
    mask = np.ones(s.shape[0], dtype=bool)
    mask[order[:k]] = False
    return mask


def random_removal_mask(n: int, k: int, seed: int) -> np.ndarray:
    """KEEP mask with ``k`` uniformly random samples removed (seeded)."""
    if not 0 < k < n:
        raise ValueError(f"k={k} must be in (0, N={n})")
    rng = np.random.RandomState(int(seed))
    mask = np.ones(n, dtype=bool)
    mask[rng.choice(n, size=k, replace=False)] = False
    return mask


def removed_indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(~np.asarray(mask, dtype=bool))


def overlap_count(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    """How many removed samples two removal sets share."""
    return int(np.sum(~np.asarray(mask_a, bool) & ~np.asarray(mask_b, bool)))


def hypergeometric_expected_overlap(n: int, k: int) -> float:
    """E[|A ∩ B|] for two independent uniform k-subsets of an n-set = k²/n."""
    return float(k) * float(k) / float(n)


def pixel_l2(img_a_u8: np.ndarray, img_b_u8: np.ndarray) -> float:
    """Euclidean distance between two uint8 images on the [0, 1] scale."""
    a = np.asarray(img_a_u8, dtype=np.float64).reshape(-1) / 255.0
    b = np.asarray(img_b_u8, dtype=np.float64).reshape(-1) / 255.0
    if a.shape != b.shape:
        raise ValueError(f"image shapes differ: {a.shape} vs {b.shape}")
    return float(np.linalg.norm(a - b))


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def auc_vs_reference(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) + 0.5·P(a = b) for independent draws — the Mann–Whitney AUC of
    the method's per-image records against the random-removal records
    (MUCS-style statistic; 0.5 = indistinguishable from random removal)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        raise ValueError("AUC needs at least one record on each side")
    gt = (a[:, None] > b[None, :]).mean()
    eq = (a[:, None] == b[None, :]).mean()
    return float(gt + 0.5 * eq)
