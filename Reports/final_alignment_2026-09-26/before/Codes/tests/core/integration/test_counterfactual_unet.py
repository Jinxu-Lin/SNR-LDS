"""Counterfactual chain on the UNet (CIFAR) platform — CPU, tiny model, real temp store.

The LoRA/latent chain is covered by test_counterfactual.py; this exercises the
second branch behind the same seam (2026-09-03, §6.2-33 ③): full-model retrain on
the LDS subset recipe, pixel replay, query-loss change and the AUC against random.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows.config import load_config, resolve_paper_method  # noqa: E402
from balds.workflows.counterfactual import ALL_ARMS, METHOD_ARMS, REFERENCE_ARM, RANDOM_ARM, CounterfactualUseCase, arm_id  # noqa: E402
from balds.schema.artifact import ArtifactKind as K  # noqa: E402
from balds.schema.registry import DATASETS  # noqa: E402
from balds.schema.runspec import RunSpec  # noqa: E402
from balds.data.base import ImageDataset  # noqa: E402
from balds.models import build_model, checkpoint_state, euler_solve  # noqa: E402
from balds.models.unet import UNetCFM  # noqa: E402
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB  # noqa: E402

_DS = "_cf_unet_toy"
N, Q, KREM, NQ, NRAND = 12, 3, 4, 3, 2


def _toy_loader(data_dir, **_):
    g = torch.Generator().manual_seed(0)
    return (ImageDataset(torch.randn(N, 3, 32, 32, generator=g).clamp(-1, 1), torch.zeros(N, dtype=torch.long)),
            ImageDataset(torch.randn(2, 3, 32, 32, generator=g).clamp(-1, 1), torch.zeros(2, dtype=torch.long)))


if not DATASETS.has(_DS):
    DATASETS.add(_DS, _toy_loader)


def _embed(images_u8):
    return images_u8.float().flatten(1)


@pytest.fixture(scope="module")
def env():
    root = tempfile.mkdtemp()
    store = ArtifactStore(ManifestDB(f"{root}/m.db"), [LocalBackend(f"{root}/data")],
                             code_version="test")
    cfg = load_config({"storage.data_root": f"{root}/data"})
    cfg["datasets"][_DS] = {"classes": [1, 7], "num_classes": 2}
    cfg["train"].update({"epochs": 1, "batch_size": 4, "compile": False})
    cfg["model"]["base_ch"] = 8
    cfg["lds"]["T_avg"] = 4
    base = RunSpec(dataset=_DS, process="cfm", seed=42, conditional=False)
    torch.manual_seed(1)
    model = UNetCFM(base_ch=8, num_classes=None)
    store.save(K.CHECKPOINT, base,
               checkpoint_state(model, base_ch=8, num_classes=None, step=1,
                                extra={"model_type": "cfm", "conditional": False}))
    # the stored generation, produced by the same replay path
    gen = {"Q": Q, "gen_seed": 165, "ode_steps": 3, "labels": torch.zeros(Q, dtype=torch.long),
           "samples": torch.zeros(Q, 3, 32, 32), "conditional": False}
    m = build_model(store.load(K.CHECKPOINT, base), device="cpu")
    gen["samples"] = CounterfactualUseCase.replay(m, gen, device="cpu")
    store.save(K.GENERATION, base, gen)
    for i, arm in enumerate(METHOD_ARMS):
        method = resolve_paper_method(cfg, arm)
        sspec = base.with_(method=method, query_type="gen")
        store.save(K.SCORES, sspec, np.random.RandomState(100 + i).randn(N, Q).astype(np.float32))
    uc = CounterfactualUseCase(store, cfg, device="cpu", embed_fn=_embed)
    return store, cfg, uc, base


def test_unet_chain_end_to_end(env):
    store, cfg, uc, base = env
    v = uc.verify(_DS, 42, conditional=False)
    assert v["samples_identical"] and v["self_consistent"] and v["verdict"] == "strict"
    assert store.exists(K.CF_ANALYSIS, base, name="replay_check")
    out = uc.build(_DS, 42, conditional=False, k=KREM, n_queries=NQ, n_random=NRAND)
    assert out[arm_id("fmas", KREM)]["n_sets"] == NQ
    for arm in ALL_ARMS:
        r = uc.retrain(_DS, 42, arm=arm, k=KREM, conditional=False)
        assert r["trained"] == (NQ if arm != RANDOM_ARM else NRAND)
        assert uc.retrain(_DS, 42, arm=arm, k=KREM, conditional=False)["trained"] == 0   # idempotent
        g = uc.regenerate(_DS, 42, arm=arm, k=KREM, conditional=False)
        assert g["regenerated"] == (NQ if arm != RANDOM_ARM else NRAND)
    aid = arm_id(METHOD_ARMS[0], KREM)
    ck = store.load(K.CF_CHECKPOINT, base, arm=aid, qi=0)
    assert ck["counterfactual_arm"] == aid and ck["conditional"] is False
    cf = store.load(K.CF_GENERATION, base, arm=aid, qi=0)
    assert cf["images_u8"].dtype == torch.uint8 and cf["images_u8"].shape == (1, 3, 32, 32)
    # a retrained model regenerates something different from the original
    assert not torch.equal(cf["samples"][0], store.load(K.GENERATION, base)["samples"][0])
    res = uc.analyze(_DS, 42, k=KREM, conditional=False, n_qual=2)
    assert "36M U-Net" in res["caliber"]
    filed = store.load(K.CF_ANALYSIS, base, name=f"summary_k{KREM}")
    assert filed["reference"] == "stored" and filed["replay_floor"]["l2_floor_max"] == 0.0
    assert res["summary"][aid]["n"] == NQ and "dloss_mean" in res["summary"][aid]
    assert res["summary"][arm_id(RANDOM_ARM, KREM)]["n"] == NRAND * NQ
    assert set(res["auc_vs_random"][aid]) == {"l2", "clip", "dloss"}
    # a second k lives in its own arm and never touches the first
    out2 = uc.build(_DS, 42, conditional=False, k=KREM + 1, n_queries=NQ, n_random=NRAND)
    assert out2[arm_id("fmas", KREM + 1)]["n_sets"] == NQ
    assert store.exists(K.CF_MASK, base, arm=arm_id("fmas", KREM + 1), qi=0)
    assert int((~store.load(K.CF_MASK, base, arm=arm_id("fmas", KREM + 1), qi=0)).sum()) == KREM + 1


def test_verify_tolerance_and_reference_replay(env):
    """§6.2-34 (b): a replay that differs from the stored generation by a
    cross-build rounding floor passes as `tolerant`; a real drift fails; and a
    same-machine `reference` replay makes analyze floor-free by construction."""
    store, cfg, uc, base = env
    gen = store.load(K.GENERATION, base)
    pristine = gen["samples"].clone()
    # (1) rounding-level drift (a few % of pixels move by one uint8 level, as the
    #     2026-09-04 torch 2.6-vs-2.12 measurement did) -> tolerant, no raise
    gen["samples"] = pristine + 2e-4
    store.save(K.GENERATION, base, gen)
    v = uc.verify(_DS, 42, conditional=False)
    assert v["verdict"] == "tolerant" and v["self_consistent"] and v["u8_max_delta"] <= 1
    assert 0 < v["l2_floor_max"] <= v["l2_tol"]
    # (2) real drift -> fail (raises under strict, reports under --no-strict)
    gen["samples"] = pristine + 0.5
    store.save(K.GENERATION, base, gen)
    with pytest.raises(ValueError, match="replay verification FAILED"):
        uc.verify(_DS, 42, conditional=False)
    assert uc.verify(_DS, 42, conditional=False, strict=False)["verdict"] == "fail"
    # (3) a huge tolerance cannot rescue a multi-level uint8 delta
    assert uc.verify(_DS, 42, conditional=False, strict=False, l2_tol=1e9)["verdict"] == "fail"
    # (4) the reference arm: undeleted model replayed here, analyze measures against it
    gen["samples"] = pristine
    store.save(K.GENERATION, base, gen)
    uc.verify(_DS, 42, conditional=False)
    r = uc.regenerate(_DS, 42, arm=REFERENCE_ARM, k=KREM, conditional=False)
    assert r["regenerated"] == 1
    assert uc.regenerate(_DS, 42, arm=REFERENCE_ARM, k=KREM, conditional=False)["regenerated"] == 0
    ref = store.load(K.CF_GENERATION, base, arm=REFERENCE_ARM, qi=0)
    assert torch.equal(ref["samples"], pristine) and ref["images_u8"].shape[0] == pristine.shape[0]
    res = uc.analyze(_DS, 42, k=KREM, conditional=False, n_qual=2)
    filed = store.load(K.CF_ANALYSIS, base, name=f"summary_k{KREM}")
    assert filed["reference"] == "replay" and "replay_floor" not in filed
    assert res["summary"][arm_id(METHOD_ARMS[0], KREM)]["n"] == NQ


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
