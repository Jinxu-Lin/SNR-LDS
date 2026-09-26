"""GT CHUNKING PARITY: the 2026-07-09 timestep-batching rewrite of
``compute_query_losses`` (balds/eval/ground_truth.py) reproduces the original
one-timestep-at-a-time loop exactly.

Motivation: on real hardware the original loop was kernel-launch-bound (~10-20%
GPU util at Q<=1000, T_avg=1000 sequential forward calls) rather than
compute-bound, making full GT production take 15h+ per (dataset,seed,process).
The fix batches several timesteps' worth of samples into one forward call. This
test guards that the batching is a pure performance change: same per-step RNG
draws in the same order, same math, same output -- verified against a verbatim
copy of the pre-rewrite implementation on a tiny CPU model (no CUDA/data needed).

Run:  conda run -n da python tests/integration/test_ground_truth_chunking.py
(works without a GPU; only needs torch + the real UNetCFM/DDPMSchedule classes)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.evaluation.ground_truth import compute_query_losses
from balds.models.ddpm import DDPMSchedule
from balds.models.unet import UNetCFM


@torch.no_grad()
def _reference_compute_query_losses(model, queries, labels, model_type, device, *,
                                    schedule=None, T_avg=1000, seed=0,
                                    batch_size=256, num_workers=0):
    """Verbatim copy of the pre-2026-07-09 implementation (one timestep per forward call)."""
    model.eval().to(device)
    Q = queries.shape[0]
    per_sample = torch.zeros(Q)
    loader = DataLoader(TensorDataset(queries, labels), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=False)

    if model_type == "cfm":
        ts = torch.linspace(0.05, 0.95, T_avg)
        for step_idx, t_val in enumerate(ts):
            torch.manual_seed(seed * 1000 + step_idx)
            x0_full = torch.randn(Q, *queries.shape[1:])
            offset = 0
            for q_b, lb in loader:
                bsz = q_b.shape[0]
                q_b = q_b.to(device); lb = lb.to(device)
                x0_b = x0_full[offset:offset + bsz].to(device)
                t_b = torch.full((bsz,), t_val.item(), device=device)
                x_t = (1 - t_val) * x0_b + t_val * q_b
                u_t = q_b - x0_b
                v_pred = model(x_t, t_b, lb)
                per_sample[offset:offset + bsz] += (v_pred - u_t).pow(2).mean(dim=(1, 2, 3)).cpu()
                offset += bsz
    else:
        assert schedule is not None
        T = schedule.T
        ts = torch.linspace(0, T - 1, T_avg).long()
        for t_idx_val in ts:
            torch.manual_seed(seed * 1000 + t_idx_val.item())
            noise_full = torch.randn(Q, *queries.shape[1:])
            offset = 0
            for q_b, lb in loader:
                bsz = q_b.shape[0]
                q_b = q_b.to(device); lb = lb.to(device)
                noise_b = noise_full[offset:offset + bsz].to(device)
                t_idx = torch.full((bsz,), t_idx_val.item(), device=device, dtype=torch.long)
                x_t = schedule.q_sample(q_b, t_idx, noise=noise_b)
                eps_pred = model(x_t, t_idx.float() / T, lb)
                per_sample[offset:offset + bsz] += (eps_pred - noise_b).pow(2).mean(dim=(1, 2, 3)).cpu()
                offset += bsz

    return (per_sample / T_avg).numpy()


def _tiny_model(num_classes=2):
    torch.manual_seed(0)
    return UNetCFM(base_ch=8, ch_mults=(1, 2), attn_resolutions=(), num_res_blocks=1,
                   num_classes=num_classes)


def _tiny_queries(Q=6, num_classes=2):
    torch.manual_seed(1)
    queries = torch.randn(Q, 3, 32, 32)
    labels = torch.randint(0, num_classes, (Q,))
    return queries, labels


def test_cfm_chunking_matches_reference():
    model = _tiny_model()
    queries, labels = _tiny_queries()
    T_avg = 17  # deliberately not a multiple of any chunk size -> exercises a partial last chunk
    ref = _reference_compute_query_losses(model, queries, labels, "cfm", "cpu", T_avg=T_avg, seed=3)
    # target_batch=18 with Q=6 -> chunk=3, so T_avg=17 needs 6 chunks with a partial (2-step) last one
    got = compute_query_losses(model, queries, labels, "cfm", "cpu", T_avg=T_avg, seed=3, target_batch=18)
    assert np.allclose(ref, got, atol=1e-6), f"max diff {np.abs(ref - got).max()}"
    print(f"  [cfm] chunked matches reference, max diff {np.abs(ref - got).max():.2e}")


def test_ddpm_chunking_matches_reference():
    model = _tiny_model()
    queries, labels = _tiny_queries()
    schedule = DDPMSchedule(T=1000)
    T_avg = 17
    ref = _reference_compute_query_losses(model, queries, labels, "ddpm", "cpu",
                                          schedule=schedule, T_avg=T_avg, seed=7)
    got = compute_query_losses(model, queries, labels, "ddpm", "cpu",
                               schedule=schedule, T_avg=T_avg, seed=7, target_batch=18)
    assert np.allclose(ref, got, atol=1e-6), f"max diff {np.abs(ref - got).max()}"
    print(f"  [ddpm] chunked matches reference, max diff {np.abs(ref - got).max():.2e}")


def test_chunk_size_does_not_change_result():
    """Different target_batch (hence different chunk boundaries) must agree with each other."""
    model = _tiny_model()
    queries, labels = _tiny_queries(Q=5)
    T_avg = 23
    a = compute_query_losses(model, queries, labels, "cfm", "cpu", T_avg=T_avg, seed=1, target_batch=5)   # chunk=1
    b = compute_query_losses(model, queries, labels, "cfm", "cpu", T_avg=T_avg, seed=1, target_batch=37)  # chunk=7
    c = compute_query_losses(model, queries, labels, "cfm", "cpu", T_avg=T_avg, seed=1, target_batch=5000)  # chunk=T_avg (single call)
    assert np.allclose(a, b, atol=1e-6) and np.allclose(a, c, atol=1e-6)
    print(f"  [chunk-invariance] chunk=1 vs chunk=7 vs single-call all agree, max diff {max(np.abs(a-b).max(), np.abs(a-c).max()):.2e}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} ground-truth chunking tests passed")
