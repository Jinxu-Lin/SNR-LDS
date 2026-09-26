"""Counterfactual query selection — --query-indices / --balanced (2026-09-07).

``build`` took the FIRST ``n_queries`` columns of the score matrix. The generated
queries are laid out class by class (``_balanced_labels``), so on ArtBench-2's
Q=100 the first 50 are all style 4 and ``--n-queries 10`` buys ten queries of one
class — a class-conditional experiment dressed up as a query sample.

T1 --balanced K takes K/num_classes per class, from the generation's own labels
T2 the chosen indices are stored in the meta as EXPLICIT indices (a rule would
   be re-derived at read time from a labels vector that can move)
T3 --query-indices is honoured verbatim, in the order given
T4 the three selectors are mutually exclusive (call level and CLI level)
T5 the whole chain downstream reads the meta's list: regenerate files the right
   query index for a method arm AND for the random arm
T6 a meta with no query_indices key (the six arms already on disk) still means
   range(n_queries) — byte-for-byte the old behaviour
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows.config import load_config, resolve_paper_method  # noqa: E402
from balds.workflows.counterfactual import (  # noqa: E402
    DEFAULT_N_QUERIES,
    METHOD_ARMS,
    RANDOM_ARM,
    CounterfactualUseCase,
    arm_id,
)
from balds.cli.main import build_parser  # noqa: E402
from balds.schema.artifact import ArtifactKind as K  # noqa: E402
from balds.schema.registry import DATASETS  # noqa: E402
from balds.schema.runspec import RunSpec  # noqa: E402
from balds.data.base import ImageDataset  # noqa: E402
from balds.models import build_model, checkpoint_state  # noqa: E402
from balds.models.unet import UNetCFM  # noqa: E402
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB  # noqa: E402

_DS = "_cf_q_toy"
N, Q, KREM, NRAND = 10, 20, 3, 2
#: two classes, INTERLEAVED: a prefix and a balanced pick are then different sets
GEN_LABELS = torch.tensor([1, 7] * (Q // 2))


def _toy_loader(data_dir, **_):
    g = torch.Generator().manual_seed(0)
    return (ImageDataset(torch.randn(N, 3, 32, 32, generator=g).clamp(-1, 1),
                         torch.zeros(N, dtype=torch.long)),
            ImageDataset(torch.randn(2, 3, 32, 32, generator=g).clamp(-1, 1),
                         torch.zeros(2, dtype=torch.long)))


if not DATASETS.has(_DS):
    DATASETS.add(_DS, _toy_loader)


@pytest.fixture(scope="module")
def env():
    root = tempfile.mkdtemp()
    store = ArtifactStore(ManifestDB(f"{root}/m.db"), [LocalBackend(f"{root}/data")],
                             code_version="test")
    cfg = load_config({"storage.data_root": f"{root}/data"})
    cfg["datasets"][_DS] = {"classes": [1, 7], "num_classes": 2}
    cfg["train"].update({"epochs": 1, "batch_size": 4, "compile": False})
    cfg["model"]["base_ch"] = 8
    cfg["lds"]["T_avg"] = 2
    base = RunSpec(dataset=_DS, process="cfm", seed=42, conditional=False)
    torch.manual_seed(1)
    model = UNetCFM(base_ch=8, num_classes=None)
    ckpt = checkpoint_state(model, base_ch=8, num_classes=None, step=1,
                            extra={"model_type": "cfm", "conditional": False})
    store.save(K.CHECKPOINT, base, ckpt)
    gen = {"Q": Q, "gen_seed": 165, "ode_steps": 2, "labels": GEN_LABELS,
           "conditional": False}
    gen["samples"] = CounterfactualUseCase.replay(build_model(ckpt, device="cpu"),
                                                  {**gen, "samples": torch.zeros(Q, 3, 32, 32)},
                                                  device="cpu")
    gen["images_u8"] = ((gen["samples"] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    store.save(K.GENERATION, base, gen)
    for i, arm in enumerate(METHOD_ARMS):
        sspec = base.with_(method=resolve_paper_method(cfg, arm), query_type="gen")
        store.save(K.SCORES, sspec,
                   np.random.RandomState(100 + i).randn(N, Q).astype(np.float32))
    uc = CounterfactualUseCase(store, cfg, device="cpu",
                               decode_fn=lambda x: ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8))
    return store, cfg, uc, base


def test_t1_t2_balanced_picks_per_class_and_stores_explicit_indices(env):
    store, cfg, uc, base = env
    out = uc.build(_DS, 42, conditional=False, k=KREM, balanced=4, n_random=NRAND, force=True)
    for arm in (*METHOD_ARMS, RANDOM_ARM):
        meta = store.load(K.CF_META, base, arm=arm_id(arm, KREM))
        qi = meta["query_indices"]
        assert qi == [0, 1, 2, 3], (arm, qi)          # interleaved labels -> 2 per class
        assert meta["n_queries"] == 4
        got = [int(GEN_LABELS[i]) for i in qi]
        assert got.count(1) == 2 and got.count(7) == 2
    # method arms get one set per chosen query; random arm gets n_random sets
    assert out[arm_id(METHOD_ARMS[0], KREM)]["n_sets"] == 4
    assert out[arm_id(RANDOM_ARM, KREM)]["n_sets"] == NRAND
    assert out[arm_id(RANDOM_ARM, KREM)]["query_indices"] == [0, 1, 2, 3]


def test_t1b_balanced_beats_a_prefix_on_a_class_sorted_generation(env):
    """The AB2 shape: labels laid out class by class, so a prefix is one class."""
    store, cfg, uc, base = env
    sorted_labels = torch.tensor([1] * (Q // 2) + [7] * (Q // 2))
    gen = store.load(K.GENERATION, base)
    other = RunSpec(dataset=_DS, process="cfm", seed=99, conditional=False)
    store.save(K.GENERATION, other, {**gen, "labels": sorted_labels})
    for arm in METHOD_ARMS:
        store.save(K.SCORES, other.with_(method=resolve_paper_method(cfg, arm),
                                         query_type="gen"),
                   np.random.RandomState(7).randn(N, Q).astype(np.float32))
    uc.build(_DS, 99, conditional=False, k=KREM, balanced=10, n_random=1)
    qi = store.load(K.CF_META, other, arm=arm_id(METHOD_ARMS[0], KREM))["query_indices"]
    assert qi == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
    assert [int(sorted_labels[i]) for i in qi].count(1) == 5
    # ...whereas the prefix of the same size is a single class
    assert set(int(sorted_labels[i]) for i in range(10)) == {1}


def test_t3_explicit_query_indices_are_honoured_in_order(env):
    store, cfg, uc, base = env
    uc.build(_DS, 42, conditional=False, k=KREM, query_indices=[5, 0, 19], n_random=1, force=True)
    meta = store.load(K.CF_META, base, arm=arm_id(METHOD_ARMS[0], KREM))
    assert meta["query_indices"] == [5, 0, 19] and meta["n_sets"] == 3
    # set qi must be the top-k of query query_indices[qi], not of column qi
    scores = np.asarray(store.load(
        K.SCORES, base.with_(method=resolve_paper_method(cfg, METHOD_ARMS[0]),
                             query_type="gen")))
    for qi, col in enumerate([5, 0, 19]):
        mk = store.load(K.CF_MASK, base, arm=arm_id(METHOD_ARMS[0], KREM), qi=qi)
        want = set(np.argsort(-scores[:, col], kind="stable")[:KREM].tolist())
        assert set(np.flatnonzero(~np.asarray(mk, dtype=bool)).tolist()) == want


def test_t4_selectors_are_mutually_exclusive(env):
    store, cfg, uc, base = env
    with pytest.raises(ValueError, match="mutually exclusive"):
        uc.build(_DS, 42, conditional=False, k=KREM, n_queries=4, query_indices=[0, 1])
    with pytest.raises(ValueError, match="mutually exclusive"):
        uc.build(_DS, 42, conditional=False, k=KREM, n_queries=4, balanced=4)
    with pytest.raises(ValueError, match="mutually exclusive"):
        uc.build(_DS, 42, conditional=False, k=KREM, query_indices=[0, 1], balanced=4)
    with pytest.raises(ValueError, match="duplicates"):
        uc.build(_DS, 42, conditional=False, k=KREM, query_indices=[0, 0, 1])
    # and argparse refuses the pair before the use-case is ever reached
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["counterfactual", "build", "--n-queries", "4",
                           "--query-indices", "0,1"])
    args = parser.parse_args(["counterfactual", "build", "--balanced", "10"])
    assert args.balanced == 10 and args.n_queries is None and args.query_indices is None


def test_t5_regenerate_files_the_meta_index_for_both_arm_kinds(env):
    store, cfg, uc, base = env
    uc.build(_DS, 42, conditional=False, k=KREM, query_indices=[5, 0, 19], n_random=1, force=True)
    arm = METHOD_ARMS[0]
    uc.retrain(_DS, 42, arm=arm, k=KREM, conditional=False)
    uc.regenerate(_DS, 42, arm=arm, k=KREM, conditional=False)
    for qi, col in enumerate([5, 0, 19]):
        cf = store.load(K.CF_GENERATION, base, arm=arm_id(arm, KREM), qi=qi)
        assert cf["indices"].tolist() == [col], (qi, col)
    uc.retrain(_DS, 42, arm=RANDOM_ARM, k=KREM, conditional=False)
    uc.regenerate(_DS, 42, arm=RANDOM_ARM, k=KREM, conditional=False)
    cf = store.load(K.CF_GENERATION, base, arm=arm_id(RANDOM_ARM, KREM), qi=0)
    assert cf["indices"].tolist() == [5, 0, 19]


def test_t5b_analyze_reports_the_real_query_indices(env):
    """The whole point is that a sparse, non-prefix index set survives to the
    end: the records, the AUC and the figures must be about queries 5/0/19."""
    store, cfg, uc, base = env
    uc.build(_DS, 42, conditional=False, k=KREM, query_indices=[5, 0, 19],
             n_random=1, force=True)
    for arm in (*METHOD_ARMS, RANDOM_ARM):
        uc.retrain(_DS, 42, arm=arm, k=KREM, conditional=False)
        uc.regenerate(_DS, 42, arm=arm, k=KREM, conditional=False)
    out = uc.analyze(_DS, 42, k=KREM, conditional=False, n_qual=2)
    res = store.load(K.CF_ANALYSIS, base, name=f"summary_k{KREM}")
    for aid, recs in res["records"].items():
        assert recs, aid
        assert set(r["query"] for r in recs) == {5, 0, 19}, (aid, recs)
        # the loss reference is the SAME query's undeleted loss, not column qi's
        assert all("dloss" in r and "loss_base" in r for r in recs), aid
    assert out["summary"][arm_id(METHOD_ARMS[0], KREM)]["n"] == 3
    assert out["summary"][arm_id(RANDOM_ARM, KREM)]["n"] == 3      # 1 set x 3 queries


def test_t6_legacy_meta_without_the_key_means_the_prefix(env):
    """The six arms already on disk carry no query_indices; they must keep
    meaning what they meant, and the default must stay the historic 50."""
    assert DEFAULT_N_QUERIES == 50
    assert CounterfactualUseCase._query_indices({"n_queries": 4}) == [0, 1, 2, 3]
    assert CounterfactualUseCase._query_indices(
        {"n_queries": 3, "query_indices": [7, 2]}) == [7, 2]
    store, cfg, uc, base = env
    uc.build(_DS, 42, conditional=False, k=KREM, n_queries=3, n_random=1, force=True)
    meta = store.load(K.CF_META, base, arm=arm_id(METHOD_ARMS[0], KREM))
    assert meta["query_indices"] == [0, 1, 2]
    stripped = {kk: v for kk, v in meta.items() if kk != "query_indices"}
    assert CounterfactualUseCase._query_indices(stripped) == [0, 1, 2]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
