"""CPU synthetic coverage of source retrieval production; no model execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from balds.artifacts.addressing import relpath
from balds.artifacts.e3c import atomic, read_json, save_tensor
from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.workflows import source_scoring as s
from balds.workflows import curvature as c


def test_current_code_version_and_explicit_legacy_override(tmp_path):
    from balds import __version__

    store = s.ScoreBlocks(tmp_path / "inputs", tmp_path / "outputs")
    assert store.code_version == f"snr-lds-{__version__}"
    legacy = s.ScoreBlocks(tmp_path / "inputs", tmp_path / "outputs", code_version="historical-run")
    assert legacy.code_version == "historical-run"


def queries_and_selection():
    split = np.ones(500, dtype=np.int64)
    val = np.concatenate([np.arange(h * 50, h * 50 + 10) for h in range(10)])
    test = np.concatenate([np.arange(h * 50 + 10, h * 50 + 50) for h in range(10)])
    split[val] = 0
    return ({"split": split, "host_labels": np.repeat(np.arange(10), 50),
             "queries_sha256": "published-query-identity", "review_version": "v2"},
            {"indices_concept_order": test.tolist(), "queries_sha256": "published-query-identity", "review_version": "v2"})


def seed_inputs(root):
    queries, selected = queries_and_selection()
    spec = s._spec("fmas_raw")
    save_tensor(root / relpath(K.INJECT_QUERIES, spec), queries)
    atomic(root / "selection.json", selected)
    for kind in (K.CHECKPOINT, K.EKFAC_FACTORS, K.INJECT_META):
        path = root / relpath(kind, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"input placeholder; must never be loaded during planning")
    for dataset in ("cifar10", "cifar100"):
        path = root / "hf_cache" / dataset / "train.arrow"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"raw placeholder")
    return queries, selected


def test_plans_pin_accepted_250_mc_and_keep_input_and_output_identities_separate(tmp_path, monkeypatch):
    source, output = tmp_path / "data", tmp_path / "scores"
    queries, selected = seed_inputs(source)
    monkeypatch.setattr(c, "_prepare", lambda *a, **k: pytest.fail("planning loaded a model"))
    for method, sampler, damping in (("fmas_raw", "stratified_antithetic", "blockshrink"),
                                     ("ekfac_if", "iid", "global")):
        plan = s.prepare(source, output, method=method, selection="selection.json")
        identity = plan["identity"]
        assert plan["ready"] and not plan["fit"]
        assert identity["ekfac"]["mc_loss"] == identity["ekfac"]["mc_measurement"] == 250
        assert identity["ekfac"]["grad_chunk"] == 125
        assert identity["ekfac"]["row_chunk"] == 1000
        assert identity["ekfac"]["query_chunk"] == 100
        assert identity["ekfac"]["sampling"] == sampler
        assert identity["ekfac"]["damping_mode"] == damping
        assert identity["query_ids"] == np.flatnonzero(queries["split"] == 0).tolist() + selected["indices_concept_order"]
        assert read_json(output / method / "identity/identity.json") == identity
        census = s.status(source, output, method=method, selection="selection.json")
        assert census["blocks_done"] == 0 and census["blocks_total"] == 5
        assert len(census["missing_blocks"]) == 5
    assert not (source / "progress").exists()
    with pytest.raises(ValueError, match="identity conflict"):
        s.prepare(source, output, method="fmas_raw", selection="selection.json", query_chunk=50)
    with pytest.raises(ValueError, match="relative"):
        s.prepare(source, output, method="fmas_raw", selection="../selection.json")


def test_missing_inputs_stop_before_model_preparation_and_invalid_shards_fail_early(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "_prepare", lambda *a, **k: pytest.fail("missing inputs loaded a model"))
    kwargs = dict(method="fmas_raw", selection="selection.json")
    plan = s.prepare(tmp_path / "missing", tmp_path / "out", **kwargs)
    assert not plan["ready"] and len(plan["missing_inputs"]) == 5
    with pytest.raises(FileNotFoundError, match="missing retrieval inputs"):
        s.run(tmp_path / "missing", tmp_path / "out", **kwargs)
    with pytest.raises(ValueError, match="rank"):
        s.run(tmp_path / "missing", tmp_path / "out", world=0, **kwargs)
    with pytest.raises(ValueError, match="positive"):
        s.prepare(tmp_path / "missing", tmp_path / "out", query_chunk=0, **kwargs)
    with pytest.raises(ValueError, match="ancestor"):
        s.ScoreBlocks(tmp_path / "data", tmp_path / "data")
    store = s.ScoreBlocks(tmp_path / "data", tmp_path / "other")
    with pytest.raises(ValueError, match="only isolated"):
        store.save(K.CHECKPOINT, s._spec("fmas_raw"), {})


class FakeScorer:
    def __init__(self):
        self.calls = []

    def sample_gradient(self, image, label, **kwargs):
        self.calls.append((float(image), label, kwargs))
        # The answer depends on the RNG key, so reindexing queries silently
        # changes the expected matrix and this test fails.
        return torch.tensor([float(kwargs["index"] + 1), float(image)], dtype=torch.float32)

    def to_eigencoords(self, value):
        return value


def tiny_context():
    spec = s._spec("ekfac_if")
    factors = SimpleNamespace(meta={"fit_epochs": 125, "eig_epochs": 125},
        damped_inverse_flat=lambda damping, device, mode: torch.tensor([1., 2.]) / damping)
    e = deepcopy(s.EKFAC)
    e.update(query_chunk=2, row_chunk=2)
    return SimpleNamespace(e=e, base=spec, factors=factors, train_ds=SimpleNamespace(
        images=torch.tensor([2., 3., 5., 7.]), labels=torch.tensor([0, 1, 0, 1])),
        N=4, q_images=torch.tensor([11., 13., 17.]), q_labels=torch.tensor([1, 0, 1]), Q=3,
        original_query_ids=np.array([8, 3, 19]), query_source="published-query-identity",
        dampings=[1., 2.], qdtype=torch.float32, mc_meas=250, mc_loss=250, chunk=125,
        hflip=True, train_flipped=None, train_mode=True, conditional=True, seed=42,
        device="cpu", readout="loss", sampling="iid", damping_mode="global", P=2,
        scorer=FakeScorer())


def test_native_block_execution_preserves_rng_query_ids_row_tiles_and_resumes(tmp_path, monkeypatch):
    store = s.ScoreBlocks(tmp_path / "inputs", tmp_path / "scores", "synthetic-test")
    spec = s._spec("ekfac_if")
    ctx = tiny_context()
    config = {"storage": {"data_root": str(tmp_path / "scores")}}
    timing = []
    kwargs = dict(rank=0, world=1, row_rank=0, row_world=1,
                  timing=lambda *args: timing.append(args))
    assert s._score_blocks(store, config, ctx, spec, **kwargs) == 4
    assert len(timing) == 4
    query_calls = [call for call in ctx.scorer.calls if call[2]["phase"] == "ekfac_meas"]
    assert [call[2]["index"] for call in query_calls] == [8, 3, 19]
    assert [call[0] for call in query_calls] == [11., 13., 17.]
    assert all(call[2]["mc"] == 250 and call[2]["chunk"] == 125 for call in ctx.scorer.calls)
    assert all(not call[2]["hflip"] and not call[2]["train_mode"] for call in query_calls)
    training = [call for call in ctx.scorer.calls if call[2]["phase"] == "ekfac_loss"]
    assert [call[2]["index"] for call in training] == [0, 1, 2, 3] * 2
    assert all(call[2]["hflip"] and call[2]["train_mode"] for call in training)
    keys = store.local_blocks(K.SCORES_BLOCK, spec)
    assert len(keys) == 4
    meta = store.load(K.SCORES_BLOCK, spec, **keys[0])
    np.testing.assert_array_equal(meta["original_query_ids"], [8, 3, 19])
    assert str(meta["query_source"]) == "published-query-identity"
    full, info = c._assemble_blocks(store, K.SCORES_BLOCK, spec, [(0, 2), (2, 3)],
        [(0, 2), (2, 4)], 4, meta, s.IDENTITY_KEYS, prefix=(2,), Q=3,
        rank=0, world=1, row_rank=0, row_world=1, what="synthetic retrieval")
    expected = np.outer(np.arange(1, 5), np.array([9, 4, 20]))
    expected += 2 * np.outer(np.array([2, 3, 5, 7]), np.array([11, 13, 17]))
    np.testing.assert_array_equal(full, np.stack([expected, expected / 2]))
    assert info["finalized"]
    monkeypatch.setattr(ctx.scorer, "sample_gradient", lambda *a, **k: pytest.fail("resume recomputed"))
    assert s._score_blocks(store, config, ctx, spec, **kwargs) == 0
    # iid MC draws depend on grad_chunk. Reusing another chunk's partials must fail.
    ctx.chunk = 100
    with pytest.raises(ValueError, match="grad_chunk"):
        s._score_blocks(store, config, ctx, spec, **kwargs)


def test_missing_identity_is_not_fabricated_for_existing_blocks(tmp_path):
    source, output = tmp_path / "data", tmp_path / "scores"
    seed_inputs(source)
    store = s.ScoreBlocks(source, output)
    store.save(K.SCORES_BLOCK, s._spec("fmas_raw"), {"block": np.zeros((1, 1, 1))},
               q0=0, q1=100, r0=0, r1=1000)
    with pytest.raises(ValueError, match="no method identity"):
        s.prepare(source, output, method="fmas_raw", selection="selection.json")
    with pytest.raises(ValueError, match="no method identity"):
        s.status(source, output, method="fmas_raw", selection="selection.json")


def test_run_selects_original_images_and_propagates_the_published_rng_axis(tmp_path, monkeypatch):
    source, output = tmp_path / "data", tmp_path / "scores"
    queries, selected = seed_inputs(source)
    expected = np.flatnonzero(queries["split"] == 0).tolist() + selected["indices_concept_order"]
    def fake_prepare(store, cfg, device, dataset, seed, **kwargs):
        assert cfg["storage"]["data_root"] == str(output)
        assert Path(cfg["datasets"]["raw_dirs"]["cifar"]) == source / "hf_cache"
        assert dataset == "cifar10_inj4" and seed == 42 and device == "cpu"
        assert kwargs["e"]["mc_loss"] == kwargs["e"]["mc_measurement"] == 250
        return SimpleNamespace(N=50000, Q=500, q_images=torch.arange(500),
            q_labels=torch.as_tensor(queries["host_labels"]),
            factors=SimpleNamespace(meta={"fit_epochs": 125, "eig_epochs": 125}))
    def fake_score(store, cfg, ctx, spec, **kwargs):
        assert ctx.Q == 500
        assert ctx.q_images.tolist() == ctx.original_query_ids.tolist() == expected
        assert ctx.query_source == "published-query-identity"
        assert kwargs["rank"] == 1 and kwargs["world"] == 3
        assert kwargs["row_rank"] == 2 and kwargs["row_world"] == 5
        return 0
    monkeypatch.setattr(c, "_prepare", fake_prepare)
    monkeypatch.setattr(s, "_score_blocks", fake_score)
    result = s.run(source, output, method="fmas_raw", selection="selection.json", device="cpu",
                   rank=1, world=3, row_rank=2, row_world=5)
    assert not result["blocks_complete"] and result["assembly_required"]
    assert (output / "fmas_raw/worker_1_2/receipt.json").is_file()
    assert not (output / "fmas_raw/result.json").exists()


def test_tool_run_flags_reach_real_workflow(tmp_path, monkeypatch, capsys):
    script = Path(__file__).parents[1] / "tools/score_retrieval.py"
    spec = importlib.util.spec_from_file_location("retrieval_tool", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    got = {}
    def fake_run(inputs, outputs, **kwargs):
        got.update(inputs=inputs, outputs=outputs, **kwargs)
        return {"blocks_complete": False}
    monkeypatch.setattr(s, "run", fake_run)
    module.main(["run", "--input-data-root", str(tmp_path / "data"), "--output-root", str(tmp_path / "out"),
        "--method", "ekfac_if", "--process", "ddpm", "--selection", "selection.json", "--device", "cpu", "--rank", "1", "--world", "3",
        "--row-rank", "2", "--row-world", "4", "--cpu-threads", "1", "--code-version", "test"])
    assert got["rank"] == 1 and got["world"] == 3 and got["row_rank"] == 2 and got["row_world"] == 4
    assert got["device"] == "cpu" and got["cpu_threads"] == 1 and got["code_version"] == "test"
    assert '"blocks_complete": false' in capsys.readouterr().out


def test_reviewed500_ddpm_identity_is_distinct_and_requires_no_selection_file(tmp_path):
    root = tmp_path / 'inputs'
    queries, _ = seed_inputs(root)
    save_tensor(root / relpath(K.INJECT_QUERIES, s._spec('fmas_raw', 'ddpm')), queries)
    identity = s.scientific_identity(queries, None, 'fmas_raw', process='ddpm')
    assert identity['process'] == 'ddpm' and len(identity['query_ids']) == 500
    assert identity != s.scientific_identity(queries, None, 'fmas_raw', process='cfm')
    plan = s.prepare(root, tmp_path / 'out', method='fmas_raw', process='ddpm')
    assert plan['identity'] == identity
    assert not any('selection' in name for name in plan['inputs'])
