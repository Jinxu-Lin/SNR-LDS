"""CPU regression for fast_jl kernel dispatch; no CUDA job is launched."""
import sys
from types import SimpleNamespace

import pytest
import torch

from balds.attribution.projection import build_projector


@pytest.mark.parametrize("batch_size,capacity", [(1, 8), (2, 8), (7, 8), (8, 8),
                                                (9, 16), (16, 16), (17, 32),
                                                (32, 32), (50, 32)])
def test_cuda_jl_uses_supported_capacity_without_changing_rows(monkeypatch, batch_size, capacity):
    projectors = pytest.importorskip("trak.projectors")
    calls = []

    # Bypass only CUDA construction. Exercise the installed TRAK project()
    # implementation, including its symbol lookup and seed/model_id handling.
    def cpu_init(self, *, grad_dim, proj_dim, seed, proj_type, max_batch_size, device):
        self.grad_dim, self.proj_dim, self.seed = grad_dim, proj_dim, seed
        self.proj_type, self.max_batch_size = proj_type, max_batch_size
        self.device, self.num_sms = device, 128

    def kernel(size):
        def project(grads, proj_dim, seed, num_sms):
            calls.append((size, grads, proj_dim, seed, num_sms))
            return torch.zeros(grads.shape[0], proj_dim)
        return project

    monkeypatch.setattr(projectors.CudaProjector, "__init__", cpu_init)
    monkeypatch.setitem(sys.modules, "fast_jl", SimpleNamespace(
        **{f"project_normal_{size}": kernel(size) for size in (8, 16, 32)}))
    projector, resolved = build_projector("cuda_jl", grad_dim=11, proj_dim=13,
                                          seed=7, batch_size=batch_size, device="cuda:0")
    grads = torch.arange(batch_size * 11, dtype=torch.float32).reshape(batch_size, 11)
    before = grads.clone()
    result = projector.project(grads, model_id=2)
    assert resolved == "cuda_jl"
    assert projector.max_batch_size == capacity
    assert len(calls) == 1
    actual_capacity, actual_grads, dim, seed, sms = calls[0]
    assert actual_capacity == capacity
    assert actual_grads is grads  # no padding, batching, resampling or copy
    assert (dim, seed, sms) == (13, 20007, 128)
    assert torch.equal(grads, before)
    assert result.shape == (batch_size, 13)
