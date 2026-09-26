"""AN-2 counterfactual chain (task brief 2026-08-30) — CPU, stub SD3, real temp store.

T1 default retrain path byte-identical after the mask-source parametrisation
T2 removal sets: exactly k removed, == the arm's top-k, method arms differ
T3 random arm: sets pairwise different, exactly k removed, overlap ~ k²/N
T4 replay parity: the undeleted model reproduces samples.pt bit for bit
   (and a different LoRA does NOT — the reverse assertion)
T5 rank/world sharding covers every set once; rerun skips
T6 removal-set meta traces to the source scores (relpath/sha256/λ/γ/query/seed)
+  analysis writes summary (with the caliber string) and both figures
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows.config import load_config  # noqa: E402
from balds.workflows.config import resolve_paper_method  # noqa: E402
from balds.workflows.counterfactual import ALL_ARMS, METHOD_ARMS, RANDOM_ARM, CounterfactualUseCase, arm_id  # noqa: E402
from balds.workflows.latent import run_latent_train_subsets  # noqa: E402
from balds.schema.artifact import ArtifactKind as K  # noqa: E402
from balds.schema.registry import DATASETS  # noqa: E402
from balds.schema.runspec import RunSpec  # noqa: E402
from balds.data.base import ImageDataset  # noqa: E402
from balds.evaluation.counterfactual import (  # noqa: E402
    hypergeometric_expected_overlap,
    overlap_count,
    random_removal_mask,
    topk_removal_mask,
)
from balds.models.sd3 import Sd3LoraAdapter, inject_lora, load_lora_state, lora_state_dict  # noqa: E402
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB  # noqa: E402

_DS = "_cf_toy"
N, C, H, W = 8, 4, 8, 8
CTX_L, CTX_D, POOL_D, S = 5, 12, 6, 10
Q, KREM, NQ, NRAND = 4, 3, 4, 3


class _StubSD3(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(C, C); self.attn.to_k = nn.Linear(C, C)
        self.attn.to_v = nn.Linear(C, C); self.attn.to_out = nn.Sequential(nn.Linear(C, C))
        self.ctx_proj = nn.Linear(CTX_D, C); self.pool_proj = nn.Linear(POOL_D, C)

    def forward(self, hidden_states, encoder_hidden_states, pooled_projections,
                timestep, return_dict=False):
        B = hidden_states.shape[0]
        tok = hidden_states.flatten(2).transpose(1, 2)
        h = self.attn.to_out(self.attn.to_q(tok) + self.attn.to_k(tok) + self.attn.to_v(tok))
        h = h + self.ctx_proj(encoder_hidden_states).mean(-2, keepdim=True)
        h = h + self.pool_proj(pooled_projections).unsqueeze(-2) + timestep.reshape(B, 1, 1) / 1000
        return (h.transpose(1, 2).reshape(hidden_states.shape),)


def _toy_loader(data_dir, **_):
    g = torch.Generator().manual_seed(0)
    return (ImageDataset(torch.randn(N, 3, 32, 32, generator=g), torch.zeros(N, dtype=torch.long)),
            ImageDataset(torch.randn(2, 3, 32, 32, generator=g), torch.zeros(2, dtype=torch.long)))


if not DATASETS.has(_DS):
    DATASETS.add(_DS, _toy_loader)


def _factory(pe, pooled):
    def make(ckpt):
        torch.manual_seed(0)
        tf = _StubSD3()
        inject_lora(tf, ["to_q", "to_k", "to_v", "to_out.0"], rank=2, alpha=2)
        m = Sd3LoraAdapter(tf, pe, pooled)
        if ckpt is not None:
            load_lora_state(m, ckpt["lora"])
        return m.eval()
    return make


def _decode(latents):                      # deterministic stand-in for the VAE
    x = latents[:, :3]
    return ((x - x.min()) / (x.max() - x.min() + 1e-8) * 255).to(torch.uint8)


def _embed(images_u8):
    return images_u8.float().flatten(1)


@pytest.fixture(scope="module")
def env():
    root = tempfile.mkdtemp()
    store = ArtifactStore(ManifestDB(f"{root}/m.db"), [LocalBackend(f"{root}/data")],
                             code_version="test")
    cfg = load_config({"storage.data_root": f"{root}/data"})
    cfg["datasets"][_DS] = {"classes": [4, 9], "num_classes": 10, "platform": "sd35_lora"}
    cfg["lds"]["T_avg"] = 4                      # query-loss change stays cheap on the stub
    cfg["artbench"].update({"epochs": 1, "batch_size": 4, "micro_batch_size": 4,
                            "warmup_steps": 0, "lora": {"rank": 2, "alpha": 2,
                            "targets": ["to_q", "to_k", "to_v", "to_out.0"]}})
    g = torch.Generator().manual_seed(1)
    base = RunSpec(dataset=_DS, process="cfm", seed=42, conditional=True)
    z = torch.randn(N, C, H, W, generator=g); labels = torch.tensor([4, 9] * (N // 2))
    store.save(K.LATENTS, RunSpec(dataset=_DS), {"latents": z.half(), "labels": labels}, split="train")
    store.save(K.LATENTS, RunSpec(dataset=_DS), {"latents": z.flip(-1).half(), "labels": labels},
               split="train", flip=True)
    pe, pooled = torch.randn(S, CTX_L, CTX_D, generator=g), torch.randn(S, POOL_D, generator=g)
    store.save(K.PROMPT_EMBEDS, RunSpec(dataset=_DS), {"prompt_embeds": pe, "pooled_embeds": pooled})
    factory = _factory(pe, pooled)
    main = factory(None)
    with torch.no_grad():
        for k, p in main.named_parameters():
            if "lora_" in k:
                p.add_(0.1 * torch.randn(p.shape, generator=g))
    ckpt = {"platform": "sd35_lora", "model_type": "cfm", "conditional": True, "step": 1,
            "base_model": "stub", "lora_rank": 2, "lora_alpha": 2.0,
            "lora_targets": ["to_q", "to_k", "to_v", "to_out.0"], "lora": lora_state_dict(main)}
    store.save(K.CHECKPOINT, base, ckpt)
    # the stored generation, produced by the same replay path (labels balanced 4/9)
    gen_labels = torch.tensor([4, 4, 9, 9])
    gen = {"Q": Q, "gen_seed": 165, "ode_steps": 3, "labels": gen_labels,
           "samples": torch.zeros(Q, C, H, W)}
    lat = CounterfactualUseCase.replay(factory(ckpt), gen, device="cpu")
    gen["samples"] = lat; gen["images_u8"] = _decode(lat); gen["conditional"] = True
    store.save(K.GENERATION, base, gen)
    # keep >= the micro-batch (4) so the loader is never empty: drop 2 of 8 per mask
    sub_masks = []
    for m in range(2):
        mk = np.ones(N, dtype=bool); mk[[m, m + 3]] = False; sub_masks.append(mk)
    store.save(K.SUBSET_MASKS, base, sub_masks)
    # two method arms' scores: distinct matrices -> distinct removal sets
    for i, arm in enumerate(METHOD_ARMS):
        method = resolve_paper_method(cfg, arm)   # fmas -> the adopted layer's tree
        sspec = base.with_(method=method, query_type="gen")
        store.save(K.SCORES, sspec, np.random.RandomState(100 + i).randn(N, Q).astype(np.float32))
        store.save(K.SCORES_META, sspec, {"method": method, "best_lam": 0.5 * (i + 1),
                                          "gamma": 256.0 if i == 0 else None, "selector": "OracleSelector"})
    uc = CounterfactualUseCase(store, cfg, device="cpu", model_factory=factory,
                               decode_fn=_decode, embed_fn=_embed)
    return store, cfg, uc, factory, base


def test_t1_default_retrain_path_is_bitwise_unchanged(env):
    store, cfg, uc, factory, base = env
    run_latent_train_subsets(store, cfg, _DS, 42, process="cfm", device="cpu",
                             model_factory=lambda: factory(None))
    masks = store.load(K.SUBSET_MASKS, base)
    run_latent_train_subsets(store, cfg, _DS, 42, process="cfm", device="cpu", masks=masks,
                             ckpt_kind=K.CF_CHECKPOINT, key_fn=lambda m: {"arm": "t1", "qi": m},
                             progress_tag="t1", model_factory=lambda: factory(None))
    for m in range(len(masks)):
        a = store.load(K.SUBSET_CHECKPOINT, base, m=m)["lora"]
        b = store.load(K.CF_CHECKPOINT, base, arm="t1", qi=m)["lora"]
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), \
            "explicit masks == SUBSET_MASKS must retrain to the same bytes"


def test_t2_method_arm_removal_sets(env):
    store, cfg, uc, factory, base = env
    out = uc.build(_DS, 42, k=KREM, n_queries=NQ, n_random=NRAND)
    assert {a["n_sets"] for k, a in out.items() if k in {arm_id(m, KREM) for m in METHOD_ARMS}} == {NQ}
    sets = {}
    for arm in METHOD_ARMS:
        scores = store.load(K.SCORES, base.with_(method=resolve_paper_method(cfg, arm), query_type="gen"))
        for qi in range(NQ):
            mk = store.load(K.CF_MASK, base, arm=arm_id(arm, KREM), qi=qi)
            assert mk.dtype == np.bool_ and int((~mk).sum()) == KREM
            top = set(np.argsort(-scores[:, qi], kind="stable")[:KREM].tolist())
            assert set(np.flatnonzero(~mk).tolist()) == top
            sets[(arm, qi)] = frozenset(np.flatnonzero(~mk).tolist())
    assert any(sets[(METHOD_ARMS[0], q)] != sets[(METHOD_ARMS[1], q)] for q in range(NQ)), \
        "the two method arms must NOT share removal sets"


def test_t3_random_arm(env):
    store, cfg, uc, factory, base = env
    uc.build(_DS, 42, k=KREM, n_queries=NQ, n_random=NRAND)
    meta = store.load(K.CF_META, base, arm=arm_id(RANDOM_ARM, KREM))
    assert meta["n_sets"] == NRAND and len(set(meta["random_seeds"])) == NRAND
    ms = [store.load(K.CF_MASK, base, arm=arm_id(RANDOM_ARM, KREM), qi=i) for i in range(NRAND)]
    for mk in ms:
        assert int((~mk).sum()) == KREM
    for i in range(NRAND):
        for j in range(i + 1, NRAND):
            assert not np.array_equal(ms[i], ms[j]), "random sets must be pairwise different"
    # the statistical claim at a realistic size (pure function)
    big = [random_removal_mask(5000, 1000, 42 * 100003 + 7919 * (r + 1)) for r in range(5)]
    exp = hypergeometric_expected_overlap(5000, 1000)
    for i in range(5):
        for j in range(i + 1, 5):
            assert abs(overlap_count(big[i], big[j]) - exp) < 60, (i, j)   # ~5 sd
    method = topk_removal_mask(np.random.RandomState(0).randn(5000), 1000)
    assert abs(np.mean([overlap_count(method, b) for b in big]) - exp) < 40


def test_t6_meta_traces_to_source_scores(env):
    store, cfg, uc, factory, base = env
    uc.build(_DS, 42, k=KREM, n_queries=NQ, n_random=NRAND)
    for i, arm in enumerate(METHOD_ARMS):
        meta = store.load(K.CF_META, base, arm=arm_id(arm, KREM))
        src = meta["source"]
        assert src["method"] == resolve_paper_method(cfg, arm) and src["relpath"].endswith("scores.npy")
        assert meta["method"] == arm and meta["arm"] == arm_id(arm, KREM)
        assert len(src["sha256"]) == 64
        assert src["scores_best_lam"] == 0.5 * (i + 1)
        assert meta["k"] == KREM and meta["n_queries"] == NQ and meta["query_type"] == "gen"
    assert store.load(K.CF_META, base, arm=arm_id(RANDOM_ARM, KREM))["source"] is None


def test_t5_sharded_retrain_and_regenerate_cover_all_once(env):
    store, cfg, uc, factory, base = env
    uc.build(_DS, 42, k=KREM, n_queries=NQ, n_random=NRAND)
    arm = METHOD_ARMS[0]; aid = arm_id(arm, KREM)
    r0 = uc.retrain(_DS, 42, arm=arm, k=KREM, rank=0, world=2)
    r1 = uc.retrain(_DS, 42, arm=arm, k=KREM, rank=1, world=2)
    assert r0["trained"] + r1["trained"] == NQ
    assert all(store.exists(K.CF_CHECKPOINT, base, arm=aid, qi=q) for q in range(NQ))
    again = uc.retrain(_DS, 42, arm=arm, k=KREM, rank=0, world=1)   # everything skips
    assert again["trained"] == NQ                                  # shard size; all skipped
    g0 = uc.regenerate(_DS, 42, arm=arm, k=KREM, rank=0, world=2)
    g1 = uc.regenerate(_DS, 42, arm=arm, k=KREM, rank=1, world=2)
    assert g0["regenerated"] + g1["regenerated"] == NQ
    assert uc.regenerate(_DS, 42, arm=arm, k=KREM)["regenerated"] == 0     # idempotent
    for q in range(NQ):
        cf = store.load(K.CF_GENERATION, base, arm=aid, qi=q)
        assert cf["indices"].tolist() == [q] and cf["images_u8"].shape[0] == 1
    # random arm: one LoRA per set, ALL n_queries regenerated under it
    uc.retrain(_DS, 42, arm=RANDOM_ARM, k=KREM)
    uc.regenerate(_DS, 42, arm=RANDOM_ARM, k=KREM)
    cf = store.load(K.CF_GENERATION, base, arm=arm_id(RANDOM_ARM, KREM), qi=0)
    assert cf["indices"].tolist() == list(range(NQ))


def test_t4_replay_parity(env):
    store, cfg, uc, factory, base = env
    out = uc.verify(_DS, 42)
    assert out["samples_identical"] and out["images_identical"]
    # reverse assertion: a different LoRA must NOT be bit-identical
    uc.build(_DS, 42, k=KREM, n_queries=NQ, n_random=NRAND)
    uc.retrain(_DS, 42, arm=METHOD_ARMS[0], k=KREM)
    ckpt = store.load(K.CF_CHECKPOINT, base, arm=arm_id(METHOD_ARMS[0], KREM), qi=0)
    other = CounterfactualUseCase(store, cfg, device="cpu", decode_fn=_decode,
                                  model_factory=lambda _c: factory(ckpt))
    with pytest.raises(ValueError, match="replay verification FAILED"):
        other.verify(_DS, 42)
    bad = other.verify(_DS, 42, strict=False)
    assert bad["samples_identical"] is False and bad["verdict"] == "fail"
    # the last check filed wins: restore the honest one so later analyses are not blocked
    assert uc.verify(_DS, 42)["verdict"] == "strict"


def test_analysis_outputs(env):
    store, cfg, uc, factory, base = env
    uc.build(_DS, 42, k=KREM, n_queries=NQ, n_random=NRAND)
    for arm in ALL_ARMS:
        uc.retrain(_DS, 42, arm=arm, k=KREM)
        uc.regenerate(_DS, 42, arm=arm, k=KREM)
    out = uc.analyze(_DS, 42, k=KREM, n_qual=2)
    m0, rid = arm_id(METHOD_ARMS[0], KREM), arm_id(RANDOM_ARM, KREM)
    assert "NOT comparable" in out["caliber"] and f"k={KREM}" in out["caliber"]
    assert out["summary"][m0]["n"] == NQ and "dloss_mean" in out["summary"][m0]
    assert out["summary"][rid]["n"] == NRAND * NQ
    # AUC against random removal, per metric, in [0, 1]
    for a in (m0, arm_id(METHOD_ARMS[1], KREM)):
        assert set(out["auc_vs_random"][a]) == {"l2", "clip", "dloss"}
        assert all(0.0 <= v <= 1.0 for v in out["auc_vs_random"][a].values())
    res = store.load(K.CF_ANALYSIS, base, name=f"summary_k{KREM}")
    assert res["caliber"] == out["caliber"] and len(res["records"][rid]) == NRAND * NQ
    assert all("dloss" in r and "loss_base" in r for r in res["records"][m0])
    for name in (f"boxplot_k{KREM}", f"qualitative_k{KREM}"):
        png = store.load(K.CF_FIGURE, base, name=name)
        assert isinstance(png, (bytes, bytearray)) and png[:8] == b"\x89PNG\r\n\x1a\n"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
