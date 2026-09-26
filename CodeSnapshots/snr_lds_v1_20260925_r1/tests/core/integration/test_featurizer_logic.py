"""Featurizer + error-weight LOGIC test on a tiny model (no fast_jl needed).

Full-scale featurization of the 35.75M-param U-Net needs the matrix-free
``fast_jl`` CudaProjector. This test validates the featurization *machinery*
(vmap-grad -> L2-normalize -> timestep average -> project; and the e_n RMSE
pipeline) at small scale where the pure-torch ``basic`` projector is feasible.

Run (needs CUDA + trak):  conda run -n da python tests/integration/test_featurizer_logic.py
"""
from __future__ import annotations

import numpy as np
import torch

from balds.schema.registry import PROCESSES
import balds.models  # noqa: F401  (register processes)
from balds.models.unet import UNetCFM
from balds.attribution import GradFeaturizer, compute_error_weight


class _DS:
    def __init__(self, n=8):
        torch.manual_seed(0)
        self.images = torch.randn(n, 3, 32, 32)
        self.labels = torch.tensor([i % 2 for i in range(n)], dtype=torch.long)
    def __len__(self):
        return len(self.images)


def main():
    dev = "cuda:0"
    model = UNetCFM(base_ch=8, num_classes=2).to(dev).eval()
    nparams = sum(p.numel() for p in model.parameters())
    ds = _DS(8)
    process = PROCESSES.get("cfm")
    print(f"  tiny model params={nparams:,}")

    feat = GradFeaturizer(model, process, proj_dim=64, proj_seed=0, T=4,
                          batch_size=4, loss_type="mean", device=dev, projection="basic")
    f1 = feat.featurize(ds, seed=42, log_every=0).numpy()
    f2 = feat.featurize(ds, seed=42, log_every=0).numpy()
    print(f"  projector={feat.projector_kind} features shape={f1.shape} finite={np.isfinite(f1).all()}")
    assert f1.shape == (8, 64)
    assert np.isfinite(f1).all()
    assert np.allclose(f1, f2, atol=1e-5), "featurization must be deterministic for a fixed seed"
    assert np.abs(f1).sum() > 0

    e_n = compute_error_weight(model, process, ds, seed=42, T_error=4, batch_size=4, device=dev)
    print(f"  e_n shape={e_n.shape} finite={np.isfinite(e_n).all()} min={e_n.min():.4f} max={e_n.max():.4f}")
    assert e_n.shape == (8,) and np.isfinite(e_n).all() and (e_n >= 0).all()

    # mse (TRAK) variant also runs
    feat_mse = GradFeaturizer(model, process, proj_dim=64, proj_seed=0, T=2,
                              batch_size=4, loss_type="mse", device=dev, projection="basic")
    fm = feat_mse.featurize(ds, seed=1, log_every=0).numpy()
    assert fm.shape == (8, 64) and np.isfinite(fm).all()

    print("FEATURIZER LOGIC PASSED (vmap-grad + normalize + project + e_n, deterministic)")


if __name__ == "__main__":
    main()
