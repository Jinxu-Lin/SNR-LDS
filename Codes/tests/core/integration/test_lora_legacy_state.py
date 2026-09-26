"""Legacy LoRA checkpoints (raw ``lora_A``/``lora_B`` Parameters, pre-a4c1e6b)
must load into the submodule-style LoRALinear and produce identical outputs.
Surfaced by the ArtBench-2 main model (08-30) on 2026-09-03."""
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.models.sd3 import inject_lora, load_lora_state, lora_state_dict, upgrade_lora_state  # noqa: E402


def _net():
    torch.manual_seed(0)
    m = nn.Module(); m.attn = nn.Module()
    m.attn.to_q = nn.Linear(6, 6); m.attn.to_out = nn.Sequential(nn.Linear(6, 6))
    inject_lora(m, ["to_q", "to_out.0"], rank=2, alpha=2)
    with torch.no_grad():
        for k, p in m.named_parameters():
            if "lora_" in k:
                p.add_(0.3 * torch.randn(p.shape))
    return m


def test_legacy_keys_load_and_match():
    m = _net()
    new_state = lora_state_dict(m)
    legacy = {k[:-len(".weight")]: v for k, v in new_state.items()}       # 08-30 format
    assert all(k.endswith("lora_A") or k.endswith("lora_B") for k in legacy)
    up = upgrade_lora_state(legacy, m)
    assert set(up) == set(new_state) and all(torch.equal(up[k], new_state[k]) for k in up)
    m2 = _net()
    with torch.no_grad():
        for k, p in m2.named_parameters():
            if "lora_" in k:
                p.zero_()
    load_lora_state(m2, legacy)
    x = torch.randn(3, 6)
    assert torch.equal(m2.attn.to_q(x), m.attn.to_q(x))
    # new-format state still loads unchanged
    load_lora_state(m2, new_state)
    assert torch.equal(m2.attn.to_q(x), m.attn.to_q(x))


def test_wrong_shape_is_refused():
    m = _net()
    bad = {k: torch.zeros(7, 7) for k in lora_state_dict(m)}
    with pytest.raises(ValueError, match="shape"):
        load_lora_state(m, bad)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
