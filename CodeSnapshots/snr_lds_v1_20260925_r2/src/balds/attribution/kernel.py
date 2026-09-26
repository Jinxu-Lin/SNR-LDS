"""The single numeric primitive of the horizontal axis: the TRAK kernel.

``score(n, q) = g_n^T (G^T G + λI)^{-1} g_q``.

Kept on GPU torch (``torch.linalg.inv`` on the 4096×4096 kernel) for performance
and to preserve bit-for-bit parity with the legacy ``07_score.py::_trak_finalize``
(moving it to numpy would diverge in float32). The previously-buried normalization
``K_inv /= K_inv.abs().mean()`` is now a named, documented, swappable kwarg.
"""
from __future__ import annotations

import numpy as np
import torch


def trak_inverse(train_features: torch.Tensor, lam: float, *, normalize="mean_abs"):
    """Native inverse, exposed separately for resumable full-MC experiments."""
    G = train_features.float()
    K = G.T @ G
    K_reg = K + lam * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
    inverse = torch.linalg.inv(K_reg)
    if normalize == "mean_abs":
        return inverse / inverse.abs().mean()
    if normalize != "none":
        raise ValueError(f"unknown normalize mode '{normalize}'")
    return inverse


def trak_kernel(
    train_features: torch.Tensor,
    query_features: torch.Tensor,
    lam: float,
    *,
    normalize: str = "mean_abs",
    row_normalize: str = "none",
    device: str = "cuda",
    extra_features=None,
):
    """Return the ``(N, Q)`` score matrix for regularisation strength ``lam``.

    With ``extra_features`` (a sequence of ``(M, k)`` tensors) it also scores
    those against the SAME ``K⁻¹`` and returns ``(scores, [extra_scores, ...])``.
    That is what the shrinkage layer's σ̂ needs: repeat-featurized copies of a
    few hundred training samples must go through the identical kernel, and
    rebuilding ``K⁻¹`` per repeat would dominate the cost (at p=32768 the inverse
    alone is a 4 GiB matrix, once per λ).

    Parameters
    ----------
    train_features:
        ``(N, k)`` projected training gradients ``G``.
    query_features:
        ``(Q, k)`` projected query gradients.
    lam:
        Ridge regularisation strength λ.
    normalize:
        ``"mean_abs"`` reproduces the legacy ``K_inv /= K_inv.abs().mean()``
        (rank-neutral for LDS); ``"none"`` disables it.
    row_normalize:
        Per-train-row denominator for the influence-normalisation baselines
        (D-TRAK repo ``methods/02_relative_if`` / ``03_norm_if``):
        ``"none"`` (plain TRAK) · ``"h_inv_grad"`` divides row ``i`` by
        ``||K⁻¹ g_i||`` (Relative IF, Barshan et al.) · ``"grad"`` divides by
        ``||g_i||`` (Renormalized IF).
    device:
        Compute device; falls back to CPU when CUDA is unavailable.
    """
    comp = device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
    G = train_features.float().to(comp)
    Q = query_features.float().to(comp)

    K_inv = trak_inverse(G, lam, normalize=normalize)

    features_k = G @ K_inv
    scores = (Q @ features_k.T).T                       # (N, Q)

    extras = []
    for ex in (extra_features or []):
        ex_g = ex.float().to(comp)
        ex_k = ex_g @ K_inv
        ex_scores = (Q @ ex_k.T).T                      # (M, Q), same K_inv
        if row_normalize == "h_inv_grad":
            ex_scores = ex_scores / (ex_k.norm(dim=1, keepdim=True) + 1e-12)
        elif row_normalize == "grad":
            ex_scores = ex_scores / (ex_g.norm(dim=1, keepdim=True) + 1e-12)
        extras.append(ex_scores.cpu().numpy().astype(np.float32))
        del ex_g, ex_k, ex_scores

    if row_normalize == "h_inv_grad":
        scores = scores / (features_k.norm(dim=1, keepdim=True) + 1e-12)
    elif row_normalize == "grad":
        scores = scores / (G.norm(dim=1, keepdim=True) + 1e-12)
    elif row_normalize != "none":
        raise ValueError(f"unknown row_normalize mode '{row_normalize}'")

    result = scores.cpu().numpy().astype(np.float32)
    del G, Q, K_inv, features_k, scores
    if str(comp).startswith("cuda"):
        with torch.cuda.device(comp):
            torch.cuda.empty_cache()
    if extra_features is not None:
        return result, extras
    return result


def grad_similarity(
    train_features: torch.Tensor,
    query_features: torch.Tensor,
    *,
    cosine: bool = False,
    device: str = "cuda",
) -> np.ndarray:
    """Raw gradient similarity ``(N, Q)`` — the Gradient (dot / cosine) baseline.

    No kernel inverse, no λ. Matches the D-TRAK repo convention
    (``methods/01_tracin``, final-checkpoint entry): dot products of projected
    loss gradients; ``cosine=True`` row-normalises both sides first.
    """
    comp = device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
    G = train_features.float().to(comp)
    Q = query_features.float().to(comp)
    if cosine:
        G = G / (G.norm(dim=1, keepdim=True) + 1e-12)
        Q = Q / (Q.norm(dim=1, keepdim=True) + 1e-12)
    scores = (G @ Q.T)                                   # (N, Q)
    result = scores.cpu().numpy().astype(np.float32)
    del G, Q, scores
    if str(comp).startswith("cuda"):
        with torch.cuda.device(comp):
            torch.cuda.empty_cache()
    return result
