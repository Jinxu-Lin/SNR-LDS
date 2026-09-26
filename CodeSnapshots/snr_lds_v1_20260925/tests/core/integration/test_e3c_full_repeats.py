"""Ordinary CPU tests: actual small models/native operators, no GPU smoke."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from balds.workflows.e3c import (METHODS, TRACKS, Telemetry, dtrak_pass, feature_pass,
                         fmas_pass, mc_stream, repeat_identity, resolve_config, run_repeat,
                         score_ids, status)
from balds.workflows.curvature import _query_pass, _train_row
from balds.schema.registry import PROCESSES
from balds.data.base import ImageDataset
from balds.evaluation.e3c import analyze_track, head_ids, preference_agreement
from balds.evaluation.lds import compute_lds, predicted_influence, support_head_scores
from balds.attribution.ekfac import EkfacFactors, EkfacScorer, _mc_generator
from balds.attribution.featurize import GradFeaturizer
from balds.attribution.kernel import trak_inverse, trak_kernel
from balds.artifacts.e3c import RepeatFiles, atomic, load_score, read_json, save_tensor

ROOT = Path(__file__).resolve().parents[3]
CONFIG = None


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x, t, label=None):
        return self.conv(x) * (1 + t.reshape(-1, 1, 1, 1))


@pytest.fixture
def tiny():
    torch.manual_seed(123)
    cfg = deepcopy(resolve_config(CONFIG))
    cfg.update(n_train=5, n_queries=6)
    cfg["selection"].update(gen=[1, 4], val=[2, 3])
    cfg["fmas"].update(mc_loss=8, mc_measurement=8, grad_chunk=4, query_coords_dtype="float32", row_block=2)
    cfg["dtrak"].update(proj_dim=8, batch_size=2, feature_block=4)
    model = Model()
    proc = PROCESSES.get("cfm")
    train = ImageDataset(torch.randn(5, 1, 2, 2), torch.zeros(5, dtype=torch.long))
    queries = {t: ImageDataset(torch.randn(6, 1, 2, 2), torch.zeros(6, dtype=torch.long)) for t in TRACKS}
    factors = EkfacFactors({"conv": {"Q_A": torch.eye(1), "Q_B": torch.eye(1),
                            "lam_w": torch.ones(1, 1), "lam_b": torch.ones(1)}}, {"layer_order": ["conv"]})
    ctx = SimpleNamespace(model=model, proc=proc, factors=factors,
          scorer=EkfacScorer(model, proc, factors, device="cpu"), P=2, N=5, device="cpu",
          train_ds=train, qdtype=torch.float32, mc_meas=8, mc_loss=8, chunk=4,
          hflip=True, train_mode=True, readout="loss", sampling="stratified_antithetic", dampings=[.01], conditional=False)
    return cfg, ctx, queries


def test_streams_fixed_science_and_distinct_both_sides():
    cfg = resolve_config(CONFIG)
    streams = [mc_stream(cfg, r, m, role, track) for r in range(4) for m in METHODS
               for role, track in (("train", "shared"), ("query", "gen"), ("query", "val"))]
    assert len({(s["seed"], s["phase"]) for s in streams}) == 24
    assert mc_stream(cfg, 0, "fmas_raw", "train")["seed"] == 1000042
    assert mc_stream(cfg, 0, "fmas_raw", "query", "gen")["seed"] == 101000042
    assert cfg["model_seed"] == 42 and cfg["dtrak"]["proj_seed"] == 0
    assert cfg["fmas"]["rho"] == .01 and cfg["dtrak"]["lambda"] == .05
    with pytest.raises(ValueError):
        mc_stream(cfg, 0, "fmas_raw", "train", "val")
    phase, index = "ekfac_meas", 95
    tag = sum(ord(c) * 1009 ** i for i, c in enumerate(phase)) % 1_000_003
    for seed in (42, 123, 456, 1000042, 4000042):
        old = torch.Generator().manual_seed((seed * 1000003 + tag) * 1000003 + index)
        new = _mc_generator(seed, phase, index)
        assert torch.equal(torch.rand(20, generator=old), torch.rand(20, generator=new))
    assert _mc_generator(101000042, phase, index).initial_seed() == ((101000042 * 1000003 + tag) * 1000003 + index) % 2**64


def test_fmas_native_parity_shared_inverse_queries_and_resume(tiny, tmp_path, monkeypatch):
    cfg, ctx, queries = tiny
    files = RepeatFiles(tmp_path, {"repeat": 0})
    progress = Telemetry(files, "cpu")
    calls, inverses = [], []
    sample = ctx.scorer.sample_gradient
    inverse = ctx.factors.damped_inverse_flat
    def record(*a, **kw):
        calls.append((kw["phase"], kw["seed"], kw["index"]))
        return sample(*a, **kw)
    def record_inv(*a, **kw):
        result = inverse(*a, **kw)
        inverses.append(result)
        return result
    monkeypatch.setattr(ctx.scorer, "sample_gradient", record)
    monkeypatch.setattr(ctx.factors, "damped_inverse_flat", record_inv)
    fmas_pass(ctx, queries, cfg, 0, files, progress)
    assert len(inverses) == 1
    assert len(calls) == 9  # five shared train rows, four shared query gradients
    assert [i for phase, _, i in calls if phase == "ekfac_meas"] == [1, 4, 2, 3]
    first_calls = list(calls)
    fmas_pass(ctx, queries, cfg, 0, files, progress)
    assert calls == first_calls  # all row blocks/query cache reused
    expected_coords = []
    for track in TRACKS:
        ctx.q_images, ctx.q_labels, ctx.Q = queries[track].images, queries[track].labels, 6
        ctx.seed = mc_stream(cfg, 0, "fmas_raw", "query", track)["seed"]
        for q in cfg["selection"][track]:
            expected_coords.append(_query_pass(ctx, SimpleNamespace(log=lambda **kw: None), q, q + 1))
    coords = torch.cat(expected_coords)
    expected = np.stack([_train_row(ctx, coords, {.01: inverses[0]}, ctx.train_ds.images[i], None,
                          seed=mc_stream(cfg, 0, "fmas_raw", "train")["seed"], index=i)[.01] for i in range(5)])
    for ti, track in enumerate(TRACKS):
        actual = load_score(tmp_path, "fmas_raw", track, files.identity, score_ids(cfg, track), (5, 2))
        np.testing.assert_array_equal(actual, expected[:, ti * 2:ti * 2 + 2])
    # Real interruption: remove only middle test-owned block, resume computes original IDs 2,3.
    (tmp_path / "fmas_raw/rows/2_4.pt").unlink()
    calls.clear()
    fmas_pass(ctx, queries, cfg, 0, files, progress)
    assert [x[2] for x in calls] == [2, 3]
    before = files.get("fmas_raw/query_coords", [["gen", 1], ["gen", 4], ["val", 2], ["val", 3]])
    other = RepeatFiles(tmp_path / "other", {"repeat": 1})
    fmas_pass(ctx, queries, cfg, 1, other, Telemetry(other, "cpu"))
    after = other.get("fmas_raw/query_coords", [["gen", 1], ["gen", 4], ["val", 2], ["val", 3]])
    assert not torch.equal(before, after)
    assert not np.array_equal(files.get("fmas_raw/rows/0_2", {"train": [0, 1], "query": [["gen", 1], ["gen", 4], ["val", 2], ["val", 3]]}),
                              other.get("fmas_raw/rows/0_2", {"train": [0, 1], "query": [["gen", 1], ["gen", 4], ["val", 2], ["val", 3]]}))


def test_dtrak_native_model_grid_batch_identity_kernel_and_resume(tiny, tmp_path, monkeypatch):
    cfg, ctx, queries = tiny
    files = RepeatFiles(tmp_path, {"repeat": 0})
    dtrak_pass(ctx.model, ctx.proc, ctx.train_ds, queries, cfg, 0, files, Telemetry(files, "cpu"), "cpu")
    fz = GradFeaturizer(ctx.model, ctx.proc, proj_dim=8, proj_seed=0, T=100, loss_type="msl2",
                        batch_size=2, projection="torch_chunked", device="cpu")
    seed = mc_stream(cfg, 0, "dtrak_T100", "train")["seed"]
    train = fz.featurize(ctx.train_ds, seed=seed, log_every=0)
    inv = files.get("dtrak_T100/kernel_inverse", list(range(5)))
    torch.testing.assert_close(inv, trak_inverse(train, .05), rtol=0, atol=0)
    changed = fz.featurize(ctx.train_ds, seed=mc_stream(cfg, 1, "dtrak_T100", "train")["seed"], log_every=0)
    assert not torch.equal(changed, train)
    assert not torch.equal(inv, trak_inverse(changed, .05))
    for track in reversed(TRACKS):
        qs = mc_stream(cfg, 0, "dtrak_T100", "query", track)["seed"]
        full = fz.featurize(queries[track], seed=qs, log_every=0)
        selected = full[cfg["selection"][track]]
        expected = trak_kernel(train, selected, .05, device="cpu")
        actual = load_score(tmp_path, "dtrak_T100", track, files.identity, score_ids(cfg, track), (5, 2))
        np.testing.assert_array_equal(actual, expected)
        short = ImageDataset(queries[track].images[cfg["selection"][track]], queries[track].labels[cfg["selection"][track]])
        if track == "gen":  # gen [1,4] changes original within-batch positions
            assert not torch.equal(selected, fz.featurize(short, seed=qs, log_every=0))
        different = fz.featurize(queries[track], seed=mc_stream(cfg, 1, "dtrak_T100", "query", track)["seed"], log_every=0)
        assert not torch.equal(full, different)
    # No featurization on completed resume; delete a batch-aligned cache then match native full stream.
    def forbidden(*a, **kw):
        raise AssertionError("completed feature unexpectedly recomputed")
    monkeypatch.setattr(GradFeaturizer, "featurize", forbidden)
    dtrak_pass(ctx.model, ctx.proc, ctx.train_ds, queries, cfg, 0, files, Telemetry(files, "cpu"), "cpu")
    monkeypatch.undo()
    (tmp_path / "dtrak_T100/train/0_4.pt").unlink()
    resumed = feature_pass(fz, ctx.train_ds, seed, 4, files, "train", Telemetry(files, "cpu"))
    torch.testing.assert_close(resumed, train, rtol=0, atol=0)
    with pytest.raises(ValueError, match="boundaries"):
        feature_pass(fz, ctx.train_ds, seed, 3, files, "bad", Telemetry(files, "cpu"))


def test_statistics_native_lds_sd_ties_constant_clusters():
    rng = np.random.default_rng(9)
    masks = rng.integers(0, 2, (9, 20)).astype(bool)
    gt = rng.normal(size=(9, 3))
    gt[:, 2] = 1
    scores = {r: {m: rng.normal(size=(20, 3)).astype(np.float32) for m in METHODS} for r in range(4)}
    for r in range(4):
        scores[r]["dtrak_T100"][:, 1] = 0
    result = analyze_track(scores, gt, masks, [7, 8, 99], [.05, 1.0], bootstrap_count=2000)
    assert len(result["head_pairs"]) == 6 * 2 * 3
    assert result["constant_response_count"] == 4 * 2 * 2
    assert result["constant_prediction_count"] >= 4 * 2
    for m in METHODS:
        means = [compute_lds(gt, predicted_influence(scores[r][m], masks, list(range(9))))[1] for r in range(4)]
        assert result["summary"][1][m]["repeat_sd_ddof1"] == pytest.approx(np.std(means, ddof=1))
        assert result["summary"][1][m]["mean_lds"] == pytest.approx(np.mean(means))
    np.testing.assert_array_equal(preference_agreement([0, 0, 1, 1, -1], [0, 1, 0, -1, -1]), [1, .5, .5, 0, 1])
    assert head_ids(np.ones((20, 1)))[0] == {0}
    np.testing.assert_array_equal(support_head_scores(scores[0][METHODS[0]], 1), scores[0][METHODS[0]])
    same = {r: {m: scores[0][METHODS[0]] for m in METHODS} for r in range(4)}
    tied = analyze_track(same, gt, masks, [7, 8, 99], [1.0])
    assert tied["summary"][0]["ties"] == 12
    assert tied["summary"][0]["agreement_ci95"] == [1., 1.]
    assert all(p["jaccard"] == 1 for p in tied["head_pairs"])
    assert tied["summary"][0]["difference_ci95"] == [0., 0.]
    with pytest.raises(ValueError):
        analyze_track({0: {METHODS[0]: scores[0][METHODS[0]]}}, gt, masks, [7, 8, 99], [1.])
    scores[0][METHODS[0]] = scores[0][METHODS[0]][:-1]
    with pytest.raises(ValueError, match="missing rows"):
        analyze_track(scores, gt, masks, [7, 8, 99], [1.])


def test_artifact_conflicts_partial_and_mixed_repeat(tiny, tmp_path):
    cfg, _, _ = tiny
    package = {"config": cfg, "inputs": []}
    atomic(tmp_path / "package.json", package)
    assert not any(s["complete"] for s in status(cfg, tmp_path))
    files = RepeatFiles(tmp_path / "repeat_0", repeat_identity(cfg, package, 0))
    files.put("test", torch.ones(2), [2, 3])
    with pytest.raises(ValueError, match="identity"):
        files.get("test", [0, 1])
    with pytest.raises(ValueError, match="conflict"):
        RepeatFiles(tmp_path / "repeat_0", repeat_identity(cfg, package, 1))
    for m in METHODS:
        for t in TRACKS:
            files.score(m, t, np.ones((5, 2), dtype=np.float32), score_ids(cfg, t))
    assert status(cfg, tmp_path)[0]["complete"]
    meta = tmp_path / "repeat_0/fmas_raw/gen/scores.json"
    item = read_json(meta)
    item["identity"]["repeat_id"] = 1
    atomic(meta, item)
    with pytest.raises(ValueError, match="mixed repeat"):
        status(cfg, tmp_path)


def test_cpu_entrypoint_failure_then_resume_keeps_model_seed_and_completed_work(tiny, tmp_path, monkeypatch):
    from balds.workflows import e3c
    cfg, ctx, queries = tiny
    ctx.q_images, ctx.q_labels = queries["gen"].images, queries["gen"].labels
    package = {"config": cfg, "inputs": [], "dataset_snapshot": "dataset.pt"}
    atomic(tmp_path / "package.json", package)
    save_tensor(tmp_path / "dataset.pt", {"train_images": ctx.train_ds.images, "train_labels": ctx.train_ds.labels,
               "test_images": queries["val"].images, "test_labels": queries["val"].labels})
    def context(store, native, device, dataset, seed, **kw):
        assert seed == 42 and dataset == "cifar2_5k"
        assert kw["conditional"] is False and kw["e"]["blockshrink_grid"] == [.01]
        return ctx
    monkeypatch.setattr(e3c, "_prepare", context)
    monkeypatch.setattr(e3c, "select_val_queries", lambda *a: queries["val"])
    original_put = RepeatFiles.put
    def interrupted(self, name, *a):
        if name == "fmas_raw/rows/2_4":
            raise RuntimeError("simulated ordinary CPU interruption")
        return original_put(self, name, *a)
    monkeypatch.setattr(RepeatFiles, "put", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        run_repeat(cfg, 0, "cpu", tmp_path, tmp_path)
    assert read_json(tmp_path / "repeat_0/progress.json")["failed"]
    saved = tmp_path / "repeat_0/fmas_raw/rows/0_2.pt"
    stamp = saved.stat().st_mtime_ns
    monkeypatch.setattr(RepeatFiles, "put", original_put)
    run_repeat(cfg, 0, "cpu", tmp_path, tmp_path)
    assert saved.stat().st_mtime_ns == stamp
    assert status(cfg, tmp_path)[0]["complete"]
    def forbidden(*a, **kw):
        raise AssertionError("completed repeat loaded model again")
    monkeypatch.setattr(e3c, "_prepare", forbidden)
    run_repeat(cfg, 0, "cpu", tmp_path, tmp_path)
    assert read_json(tmp_path / "repeat_0/progress.json")["stage"] == "complete"




def launcher_module():
    spec = importlib.util.spec_from_file_location("e3c_launch", ROOT / "Codes/tools/e3c_launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


