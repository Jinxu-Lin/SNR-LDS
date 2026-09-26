"""Baseline-method numerics vs the D-TRAK reference formulas (CPU, tiny tensors).

Reference: github.com/sail-sg/D-TRAK CIFAR2/methods/{01_tracin,02_relative_if,
03_norm_if} —
  * Relative IF: ⟨g_q, K⁻¹g_i⟩ / ||K⁻¹g_i||        (scores / norm(G@K⁻¹, row))
  * Renorm IF:   ⟨g_q, K⁻¹g_i⟩ / ||g_i||           (scores / norm(G, row))
  * Gradient dot/cos: final-checkpoint raw similarity, no kernel, no λ.
Also pins that plain TRAK output is unchanged by the row_normalize addition.

Run: conda run -n da python Codes/tests/integration/test_baseline_methods.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import balds.attribution  # noqa: F401  (register methods)
from balds.schema.estimator import FeatureSet
from balds.schema.registry import METHODS
from balds.attribution.kernel import grad_similarity, trak_kernel

torch.manual_seed(0)
N, Q, k = 20, 5, 8
G = torch.randn(N, k)
Qf = torch.randn(Q, k)
LAM = 0.5


def _reference_kernel_scores() -> tuple[np.ndarray, torch.Tensor]:
    K_inv = torch.linalg.inv(G.T @ G + LAM * torch.eye(k))
    K_inv = K_inv / K_inv.abs().mean()          # legacy mean_abs normalization
    feats_k = G @ K_inv
    return (Qf @ feats_k.T).T.numpy(), feats_k


def test_plain_trak_unchanged():
    ref, _ = _reference_kernel_scores()
    out = trak_kernel(G, Qf, LAM, device="cpu")
    assert np.allclose(out, ref, atol=1e-5), "row_normalize default must not change TRAK"
    print("  ok  test_plain_trak_unchanged")


def test_relative_if_formula():
    ref, feats_k = _reference_kernel_scores()
    expect = ref / (feats_k.norm(dim=1, keepdim=True).numpy() + 1e-12)
    out = trak_kernel(G, Qf, LAM, row_normalize="h_inv_grad", device="cpu")
    assert np.allclose(out, expect, atol=1e-5)
    m = METHODS.get("relative_if")
    assert m.feat_method == "trak" and not m.needs_error_weight
    out2 = m.score(FeatureSet(grads=G, error=None, feat_method="trak"), Qf, LAM, device="cpu")
    assert np.allclose(out2, expect, atol=1e-5)
    print("  ok  test_relative_if_formula")


def test_renorm_if_formula():
    ref, _ = _reference_kernel_scores()
    expect = ref / (G.norm(dim=1, keepdim=True).numpy() + 1e-12)
    out = METHODS.get("renorm_if").score(
        FeatureSet(grads=G, error=None, feat_method="trak"), Qf, LAM, device="cpu")
    assert np.allclose(out, expect, atol=1e-5)
    print("  ok  test_renorm_if_formula")


def test_grad_dot_and_cos():
    expect_dot = (G @ Qf.T).numpy()
    out = grad_similarity(G, Qf, cosine=False, device="cpu")
    assert np.allclose(out, expect_dot, atol=1e-5)
    Gn = G / G.norm(dim=1, keepdim=True)
    Qn = Qf / Qf.norm(dim=1, keepdim=True)
    expect_cos = (Gn @ Qn.T).numpy()
    out_cos = METHODS.get("grad_cos").score(
        FeatureSet(grads=G, error=None, feat_method="trak"), Qf, lam=123.0, device="cpu")
    assert np.allclose(out_cos, expect_cos, atol=1e-5), "lam must be ignored"
    assert np.abs(out_cos).max() <= 1.0 + 1e-6
    print("  ok  test_grad_dot_and_cos")


def test_registry_coverage():
    for name in ("relative_if", "relative_if_T100", "renorm_if", "renorm_if_T100",
                 "grad_dot", "grad_dot_T100", "grad_cos", "grad_cos_T100"):
        m = METHODS.get(name)
        expected_feat = "trak_T100" if name.endswith("_T100") else "trak"
        assert m.feat_method == expected_feat, (name, m.feat_method)
    print("  ok  test_registry_coverage")


if __name__ == "__main__":
    test_plain_trak_unchanged()
    test_relative_if_formula()
    test_renorm_if_formula()
    test_grad_dot_and_cos()
    test_registry_coverage()
    print("PASSED 5 baseline-method tests")
