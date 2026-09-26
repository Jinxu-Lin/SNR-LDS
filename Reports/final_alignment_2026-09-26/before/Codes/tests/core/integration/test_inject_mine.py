"""CIFAR-10 4%: all-data detector and class-by-class mining (CODE_INJECT_C10_MINE r1).

Researcher ruling 2026-09-15: the detector is an aid, trained on every real
image; mining generates one class at a time, keeps only flagged images, and
hands 50 reviewed candidates per class to the query import. What has to hold:

* ``all_real`` data holds every host and concept row in the legacy class order;
* ``resnet18_cifar`` trains exactly as before; ``resnet50_in1k`` trains and
  predicts (no download in tests); holdout 0 works; pair metrics are right;
* tagged detectors never touch the legacy address;
* mining stops at target, respects the cap, resumes identically after an
  interruption, stores no rejected image, regenerates a kept image from its
  recipe, refuses a mismatched rerun, and screens an included pool first;
* review export and query import work from a mined set.
"""
from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import balds.workflows.inject as inject_app  # noqa: E402
from balds.workflows.inject import InjectUseCase  # noqa: E402
from balds.workflows.common import MINE_ROW_KEY, _mine_noise, _pool_noise  # noqa: E402
from balds.cli.main import build_parser  # noqa: E402
from balds.schema.artifact import ArtifactKind as K  # noqa: E402
from balds.schema.runspec import RunSpec  # noqa: E402
from balds.data.inject import (build_cifar10_injected, detector_training_data,  # noqa: E402
                                 detector_training_data_all_real)
from balds.evaluation import detector as det  # noqa: E402
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB  # noqa: E402
from balds.artifacts.addressing import relpath  # noqa: E402

HOST_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]
CONCEPT_NAMES = ["sunflower", "castle", "butterfly", "leopard", "cattle",
                 "wolf", "mushroom", "camel", "skyscraper", "tractor"]
REPO = Path(__file__).resolve().parents[3]


def _store(tmp_path):
    return ArtifactStore(ManifestDB(str(tmp_path / "manifest.db")),
                            [LocalBackend(str(tmp_path / "data"))], code_version="mine-test")


# --------------------------------------------------------------------------- #
# detector data and training
# --------------------------------------------------------------------------- #
class _Feature:
    def __init__(self, names):
        self.names = names


class _FakeSplit:
    def __init__(self, labels, *, names, field, offset):
        self._labels, self._field, self._offset = list(labels), field, offset
        self.features = {field: _Feature(names)}

    def __iter__(self):
        for index, label in enumerate(self._labels):
            yield {"img": np.full((32, 32, 3), (self._offset + index) % 256, dtype=np.uint8),
                   self._field: label}


@pytest.fixture
def fake_hf(monkeypatch):
    sources = {
        "cifar10": {
            "train": _FakeSplit([c for c in range(10) for _ in range(8)],
                                names=HOST_NAMES, field="label", offset=0),
            "test": _FakeSplit([c for c in range(10) for _ in range(3)],
                               names=HOST_NAMES, field="label", offset=31),
        },
        "cifar100": {
            "train": _FakeSplit([c for c in range(10) for _ in range(500)],
                                names=CONCEPT_NAMES, field="fine_label", offset=67),
            "test": _FakeSplit([c for c in range(10) for _ in range(100)],
                               names=CONCEPT_NAMES, field="fine_label", offset=149),
        },
    }
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda name, *, cache_dir: sources[name])
    return sources


def _spec():
    return {"per_class": 2, "index_seed": 20260914, "concept_seed": 20260914,
            "cifar100_hf_id": "cifar100",
            "pairs": {h: [c, "fine"] for h, c in zip(HOST_NAMES, CONCEPT_NAMES)}}


def test_all_real_holds_every_host_and_concept_row_in_legacy_class_order(fake_hf):
    spec = _spec()
    _train, _test, meta = build_cifar10_injected("cache", spec)
    images, labels, names, source = detector_training_data_all_real("cache", spec, meta)
    _i, _l, iso_names, _s = detector_training_data("cache", spec, meta)
    assert names == iso_names == HOST_NAMES + CONCEPT_NAMES
    counts = torch.bincount(labels, minlength=20).tolist()
    assert counts[:10] == [8 + 3] * 10 and counts[10:] == [600] * 10
    assert source["mode"] == "all_real" and source["n_per_class"] == counts
    # hosts are the CLEAN originals, including the rows the injected set replaced
    clean = torch.stack([torch.from_numpy(np.asarray(r["img"])).permute(2, 0, 1)
                         for r in fake_hf["cifar10"]["train"]])
    assert torch.equal(images[:80], clean)
    replaced = [i for row in meta["pairs"].values() for i in row["replaced_cifar10_indices"]]
    assert len(replaced) == 20 and all(int(labels[i]) == i // 8 for i in replaced)
    # every concept row, the injected ids included
    for host, row in meta["pairs"].items():
        assert set(row["injected_cifar100_train_ids"]) <= set(source["cifar100_train_ids"][host])
        assert len(source["cifar100_train_ids"][host]) == 500
        assert len(source["cifar100_test_ids"][host]) == 100
    print("  all_real: 11 host rows (train+test, replaced originals included) and 600 concept "
          "rows per class, legacy order")


def _old_detector_module():
    code = subprocess.run(["git", "-C", str(REPO), "show", "HEAD:Codes/balds/eval/detector.py"],
                          capture_output=True, text=True, check=True).stdout
    spec = importlib.util.spec_from_loader("old_detector", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(code, "old_detector.py", "exec"), module.__dict__)
    return module




def test_resnet50_in1k_trains_and_predicts_without_download():
    generator = torch.Generator().manual_seed(1)
    images = torch.randint(0, 256, (20 * 2, 3, 32, 32), dtype=torch.uint8, generator=generator)
    labels = torch.arange(20).repeat_interleave(2)
    cfg = {"arch": "resnet50_in1k", "weights": None, "input_size": 64, "epochs": 1,
           "batch_size": 40, "lr": 0.01, "holdout_frac": 0.0}
    state, report = det.train_detector(images, labels, num_classes=20, cfg=cfg,
                                       device="cpu", seed=0)
    assert report["arch"] == "resnet50_in1k" and state["fc.weight"].shape == (20, 2048)
    payload = {"state_dict": state, "arch": "resnet50_in1k", "config": cfg}
    probs = det.predictor_for(payload, device="cpu")(images[:5], batch=3)
    assert probs.shape == (5, 20) and torch.allclose(probs.sum(1), torch.ones(5), atol=1e-5)
    assert torch.equal(probs, det.predict_proba(state, images[:5], batch=3, device="cpu",
                                                arch="resnet50_in1k", input_size=64))
    assert det.pretrained_weights_info("resnet50_in1k", None) is None
    info = det.pretrained_weights_info("resnet50_in1k", "IMAGENET1K_V2")
    assert info["url"].endswith("resnet50-11ad3fa6.pth") and info["path"].endswith("resnet50-11ad3fa6.pth")
    print("  resnet50_in1k: 20-way head, 32->64 px here, train + predict on CPU (weights=None)")


def test_pair_metrics_on_known_cases():
    assert det.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert det.auroc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert det.auroc([0.1], [0.9]) == 0.0
    neg = np.linspace(0, 0.99, 100)                     # 1 of 100 negatives above 0.98
    tpr, threshold = det.tpr_at_fpr([0.985, 0.5, 0.995], neg, 0.01)
    assert threshold == pytest.approx(0.98) and tpr == pytest.approx(2 / 3)
    probs = torch.zeros(4, 20)
    probs[:, 10] = torch.tensor([0.9, 0.8, 0.1, 0.2])
    report = det.pair_metrics(probs, torch.tensor([10, 10, 0, 0]))
    assert report["pair_auroc"][0] == 1.0 and report["pair_tpr_at_1pct_host_fpr"][0] == 1.0
    assert np.isnan(report["pair_auroc"][1])


def test_tagged_detector_addresses_and_untagged_non_legacy_recipes_are_refused(tmp_path):
    spec = RunSpec(dataset="cifar10_inj4")
    assert relpath(K.INJECT_DETECTOR, spec) == "results/inject/cifar10_inj4/detector.pt"
    assert relpath(K.INJECT_DETECTOR, spec, tag="final_r50") == \
        "results/inject/cifar10_inj4/detector_final_r50.pt"
    pool = RunSpec(dataset="cifar10_inj4", process="ddpm", seed=42)
    assert relpath(K.INJECT_CANDIDATES, pool, tag="ddim50_n50000") == \
        "generations/cifar10_inj4/ddpm_cond/seed_42/pool_ddim50_n50000_candidates.json"
    assert relpath(K.INJECT_CANDIDATES, pool, tag="ddim50_n50000", detector_tag="final_r50") == \
        "generations/cifar10_inj4/ddpm_cond/seed_42/pool_ddim50_n50000_candidates.dtag-final_r50.json"
    assert relpath(K.INJECT_MINED, pool, tag="v1") == \
        "generations/cifar10_inj4/ddpm_cond/seed_42/mine_v1.pt"
    from balds.workflows.config import load_config
    cfg = load_config({})
    uc = InjectUseCase(_store(tmp_path), cfg, device="cpu")
    for block in ("detector_all_r18", "detector_all_r50"):
        with pytest.raises(ValueError, match="needs --detector-tag"):
            uc.detector_train("cifar10_inj4", detector_config=block)
    assert cfg["inject"]["detector"]["arch"] == "resnet18_cifar"
    assert "data_mode" not in cfg["inject"]["detector"]


# --------------------------------------------------------------------------- #
# mining with a stub sampler and detector
# --------------------------------------------------------------------------- #
def _mine_setup(tmp_path, *, rate=None, mine_cfg=None):
    """Store with meta, checkpoint and detector; stub model/sampler/detector.

    The stub sampler writes the class into channel 0 and a noise-derived score
    into channel 1; the stub detector puts that score on p[10 + class]."""
    store = _store(tmp_path)
    meta = {"dataset": "cifar10_inj4", "train_sha256": "a" * 64,
            "pairs": {h: {"host_label": i, "concept_name": CONCEPT_NAMES[i], "tier": "fine"}
                      for i, h in enumerate(HOST_NAMES)}}
    store.save(K.INJECT_META, RunSpec(dataset="cifar10_inj4"), meta)
    spec = RunSpec(dataset="cifar10_inj4", process="ddpm", seed=42)
    store.save(K.CHECKPOINT, spec, {"model": "stub", "p_uncond": 0.0})
    store.save(K.INJECT_DETECTOR, RunSpec(dataset="cifar10_inj4"),
               {"state_dict": {}, "arch": "resnet18_cifar", "config": {}}, tag="final")
    cfg = {"inject": {"mine": {"gen_seed": 20260915, "sampler": "ddim", "steps": 50,
                               "eta": 0.0, "target": 3, "chunk": 8, "keep_threshold": 0.5,
                               "max_per_class": 40, "batch_size": 3, **(mine_cfg or {})},
                      "review": {"per_concept_cap": 150}, "split": {"val_frac": 0.5, "seed": 0}},
           "ddpm": {"T": 4}}
    rate = rate or {}

    def sampler(model, schedule, x_T, *, sampler, steps, eta, class_label, device):
        assert (sampler, steps, eta) == ("ddim", 50, 0.0)
        out = torch.zeros(len(x_T), 3, 32, 32, dtype=torch.uint8)
        out[:, 0] = class_label.view(-1, 1, 1).to(torch.uint8)
        score = (x_T[:, 0, 0, 0].abs() * 1000).remainder(256).to(torch.uint8)
        out[:, 1] = score.view(-1, 1, 1)
        out[:, 2] = (x_T[:, 1].abs() * 50).clamp(0, 255).to(torch.uint8)
        return out

    class Predictor:
        calls = 0
        fail_after = None

        def __call__(self, images, *, batch):
            Predictor.calls += 1
            if Predictor.fail_after is not None and Predictor.calls > Predictor.fail_after:
                raise RuntimeError("injected interruption")
            images = torch.as_tensor(images)
            probs = torch.zeros(len(images), 20)
            cls = images[:, 0, 0, 0].long()
            score = images[:, 1, 0, 0].float() / 255.0
            scale = torch.tensor([rate.get(int(c), 1.0) for c in cls])
            probs[torch.arange(len(images)), 10 + cls] = score * scale
            probs[torch.arange(len(images)), cls] = 1 - score * scale
            return probs

    return store, cfg, sampler, Predictor


@pytest.fixture
def stubbed(monkeypatch):
    def install(sampler, predictor_cls):
        monkeypatch.setattr(inject_app, "predictor_for", lambda detector, device: predictor_cls())
        monkeypatch.setattr("balds.workflows.common._sample_pool_rows", sampler)
        monkeypatch.setattr("balds.models.build_model", lambda ckpt, device: object())
    return install


def _mine(store, cfg, **kw):
    args = dict(process="ddpm", mine_tag="v1", detector_tag="final", classes="0,1,2")
    args.update(kw)
    return InjectUseCase(store, cfg, device="cpu").mine("cifar10_inj4", 42, **args)


def test_mining_stops_at_target_keeps_only_flagged_and_regenerates_from_recipe(tmp_path, stubbed):
    store, cfg, sampler, Predictor = _mine_setup(tmp_path)
    stubbed(sampler, Predictor)
    out = _mine(store, cfg)
    spec = RunSpec(dataset="cifar10_inj4", process="ddpm", seed=42)
    mined = store.load(K.INJECT_MINED, spec, tag="v1")
    for c in (0, 1, 2):
        rows = (mined["host_labels"] == c).nonzero().flatten()
        assert out["per_class"][HOST_NAMES[c]]["status"] == "done" and len(rows) >= 3
        assert (mined["probs"][rows, 10 + c] >= 0.5).all()          # nothing rejected stored
    assert int((mined["host_labels"] == 3).sum()) == 0             # classes not requested
    progress = mined["progress"]
    assert all(progress[str(c)]["generated"] % 8 == 0 for c in (0, 1, 2))
    # the recipe regenerates every kept image exactly
    for i, source in enumerate(mined["sources"]):
        kind, gen_seed, cls, row = source.split(":")
        assert kind == "mine" and int(gen_seed) == 20260915
        x_T = _mine_noise(int(cls), [int(row)], gen_seed=20260915, shape=(3, 32, 32))
        image = sampler(None, None, x_T, sampler="ddim", steps=50, eta=0.0,
                        class_label=torch.tensor([int(cls)]), device="cpu")
        assert torch.equal(image[0], mined["images_u8"][i])
    assert mined["recipe"]["keep_rule"] == {"probability": "p[10 + class]", "threshold": 0.5}
    assert len(mined["recipe"]["detector_sha256"]) == len(mined["recipe"]["checkpoint_sha256"]) == 64
    # a rerun with nothing left to do is a no-op
    again = _mine(store, cfg)
    assert again["per_class"] == out["per_class"]
    print(f"  mining: {out['kept_total']} kept for 3 classes, all >= threshold, every kept image "
          f"regenerated from its recipe")


def test_mining_respects_max_per_class(tmp_path, stubbed):
    store, cfg, sampler, Predictor = _mine_setup(tmp_path, rate={1: 0.0})   # class 1 never flags
    stubbed(sampler, Predictor)
    out = _mine(store, cfg, max_per_class=20)
    assert out["per_class"]["automobile"] == {"label": 1, "kept": 0, "from_pool": 0,
                                               "generated": 20, "chunks": 3,
                                               "status": "incomplete"}
    assert out["per_class"]["bird"]["status"] == "done"


def test_interrupted_mining_resumes_to_the_uninterrupted_artifact(tmp_path, stubbed):
    spec = RunSpec(dataset="cifar10_inj4", process="ddpm", seed=42)
    store_a, cfg, sampler, Predictor = _mine_setup(tmp_path / "a")
    stubbed(sampler, Predictor)
    _mine(store_a, cfg)
    full = store_a.load(K.INJECT_MINED, spec, tag="v1")

    store_b, cfg_b, _sampler, PredictorB = _mine_setup(tmp_path / "b")
    PredictorB.calls, PredictorB.fail_after = 0, 7                 # dies mid-chunk
    stubbed(sampler, PredictorB)
    with pytest.raises(RuntimeError, match="injected interruption"):
        _mine(store_b, cfg_b)
    partial = store_b.load(K.INJECT_MINED, spec, tag="v1")
    assert len(partial["sources"]) < len(full["sources"])
    PredictorB.fail_after = None
    _mine(store_b, cfg_b)
    resumed = store_b.load(K.INJECT_MINED, spec, tag="v1")
    for key in ("images_u8", "host_labels", "probs"):
        assert torch.equal(resumed[key], full[key]), key
    assert resumed["sources"] == full["sources"] and resumed["progress"] == full["progress"]
    assert resumed["recipe"] == full["recipe"]
    print("  resume: interrupted after 7 detector calls, rerun ends identical to the "
          "uninterrupted artifact")


def test_mismatched_rerun_is_refused(tmp_path, stubbed):
    store, cfg, sampler, Predictor = _mine_setup(tmp_path)
    stubbed(sampler, Predictor)
    _mine(store, cfg, classes="0")
    with pytest.raises(FileExistsError, match="keep_rule"):
        _mine(store, cfg, classes="0", keep_threshold=0.6)
    with pytest.raises(FileExistsError, match="chunk"):
        _mine(store, cfg, classes="0", chunk=16)
    store.save(K.INJECT_DETECTOR, RunSpec(dataset="cifar10_inj4"),
               {"state_dict": {"changed": torch.ones(1)}, "arch": "resnet18_cifar"}, tag="final")
    with pytest.raises(FileExistsError, match="detector_sha256"):
        _mine(store, cfg, classes="0")


def test_include_pool_is_screened_first_and_must_match_the_checkpoint(tmp_path, stubbed):
    store, cfg, sampler, Predictor = _mine_setup(tmp_path)
    stubbed(sampler, Predictor)
    spec = RunSpec(dataset="cifar10_inj4", process="ddpm", seed=42)
    checkpoint_sha = store.describe(K.CHECKPOINT, spec)["sha256"]
    labels = torch.arange(10).repeat_interleave(4)
    x_T = _pool_noise(range(40), gen_seed=20260914, shape=(3, 32, 32))
    images = sampler(None, None, x_T, sampler="ddim", steps=50, eta=0.0,
                     class_label=labels, device="cpu")
    pool = {"images_u8": images, "labels": labels, "n": 40, "gen_seed": 20260914,
            "sampler": "ddim", "steps": 50, "eta": 0.0, "checkpoint_sha256": checkpoint_sha}
    store.save(K.GEN_POOL, spec, pool, tag="ddim50_n40")
    out = _mine(store, cfg, include_pool="ddim50_n40", classes="0,1")
    mined = store.load(K.INJECT_MINED, spec, tag="v1")
    pool_rows = [s for s in mined["sources"] if s.startswith("pool:")]
    assert mined["pool"]["kept_per_class"] == {
        str(c): int(((labels == c) & (images[:, 1, 0, 0].float() / 255 >= 0.5)).sum())
        for c in range(10)}
    assert len(pool_rows) == sum(mined["pool"]["kept_per_class"].values())
    for source in pool_rows:
        row = int(source.split(":")[2])
        index = mined["sources"].index(source)
        assert torch.equal(mined["images_u8"][index], images[row])
    assert out["per_class"]["airplane"]["from_pool"] == mined["pool"]["kept_per_class"]["0"]

    store.save(K.GEN_POOL, spec, {**pool, "checkpoint_sha256": "b" * 64}, tag="other")
    with pytest.raises(ValueError, match="not this mining recipe"):
        _mine(store, cfg, mine_tag="v2", include_pool="other", classes="0")


def test_mining_noise_is_disjoint_from_filed_pool_noise():
    modulus = 2**63 - 1
    pool_seeds = {(20260914 * 1_000_003 + i) % modulus for i in range(50_000)}
    mine_seeds = {(20260915 * 1_000_003 + (c + 1) * MINE_ROW_KEY + r) % modulus
                  for c in range(10) for r in range(0, 300_000, 997)}
    assert pool_seeds.isdisjoint(mine_seeds)
    assert not torch.equal(_mine_noise(0, [0], gen_seed=20260915, shape=(3, 4, 4)),
                           _pool_noise([0], gen_seed=20260915, shape=(3, 4, 4)))


def test_review_export_and_queries_from_a_mined_set(tmp_path, stubbed):
    store, cfg, sampler, Predictor = _mine_setup(tmp_path)
    stubbed(sampler, Predictor)
    _mine(store, cfg, classes="0,1")
    spec = RunSpec(dataset="cifar10_inj4", process="ddpm", seed=42)
    mined = store.load(K.INJECT_MINED, spec, tag="v1")
    uc = InjectUseCase(store, cfg, device="cpu")
    out = uc.review_export("cifar10_inj4", 42, process="ddpm", tag=None, mine_tag="v1",
                           per_concept=2)
    assert out["rows"] == 4 and out["sheets"] == 2
    table = Path(store.local.abspath(out["review"]))
    assert out["review"] == "results/inject/cifar10_inj4/ddpm_cond/seed_42/review_mine-v1/review.csv"
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for host in (0, 1):
        probs = [float(r["prob"]) for r in rows if int(r["host_label"]) == host]
        assert probs == sorted(probs, reverse=True)
    for row in rows:
        i = int(row["candidate_index"])
        assert row["candidate_id"] == mined["sources"][i]
        row["decision"] = "accept"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = uc.queries("cifar10_inj4", 42, process="ddpm", tag=None, review=out["review"],
                        version="v1", reviewer="researcher", mine_tag="v1")
    queries = store.load(K.INJECT_QUERIES, spec)
    assert result["Q"] == 4
    for key in ("images_u8", "host_labels", "concept_hosts", "pool_indices", "probs", "split",
                "review_version", "reviewer", "review_csv_sha256", "detector_sha256", "pool_tag",
                "n_reviewed_per_concept", "n_accepted_per_concept", "queries_sha256",
                "code_version"):
        assert key in queries, key
    assert queries["pool_tag"] == "mine:v1" and queries["mine_tag"] == "v1"
    assert queries["detector_sha256"] == mined["recipe"]["detector_sha256"]
    for q, index in enumerate(queries["pool_indices"].tolist()):
        assert torch.equal(queries["images_u8"][q], mined["images_u8"][index])
        assert queries["candidate_ids"][q] == mined["sources"][index]
    rows[0]["candidate_id"] = "mine:0:0:999999"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="does not match its mined candidate"):
        uc.queries("cifar10_inj4", 42, process="ddpm", tag=None, review=out["review"],
                   version="v2", force=True, mine_tag="v1")
    print("  review-export (per class, sorted, candidate ids) and queries (today's fields) "
          "from a mined set")


def test_screen_with_a_detector_tag_leaves_the_legacy_candidates_untouched(tmp_path, monkeypatch):
    store = _store(tmp_path)
    spec = RunSpec(dataset="cifar10_inj4", process="ddpm", seed=42)
    store.save(K.GEN_POOL, spec, {"images_u8": torch.zeros(10, 3, 32, 32, dtype=torch.uint8),
                                  "labels": torch.arange(10), "n": 10}, tag="tiny")
    store.save(K.INJECT_DETECTOR, RunSpec(dataset="cifar10_inj4"), {"state_dict": {"a": 1}})
    store.save(K.INJECT_DETECTOR, RunSpec(dataset="cifar10_inj4"),
               {"state_dict": {"b": 2}, "arch": "resnet18_cifar"}, tag="final")
    probs = torch.zeros(10, 20)
    probs[:, 10] = 1.0
    monkeypatch.setattr(inject_app, "predict_proba", lambda *a, **k: probs)
    cfg = {"inject": {"screen_threshold": 0.9, "pool": {"batch_size": 4}}}
    uc = InjectUseCase(store, cfg, device="cpu")
    bare = uc.screen("cifar10_inj4", 42, process="ddpm", tag="tiny")
    bare_bytes = Path(store.local.abspath(bare["candidates"])).read_bytes()
    tagged = uc.screen("cifar10_inj4", 42, process="ddpm", tag="tiny", detector_tag="final")
    assert tagged["candidates"].endswith("pool_tiny_candidates.dtag-final.json")
    assert Path(store.local.abspath(bare["candidates"])).read_bytes() == bare_bytes
    assert tagged["detector_sha256"] != bare["detector_sha256"]


def test_cli_parses_the_new_actions():
    args = build_parser().parse_args([
        "inject", "mine", "--dataset", "cifar10_inj4", "--process", "ddpm", "--seed", "42",
        "--mine-tag", "r50v1", "--detector-tag", "final_r50", "--target", "150",
        "--chunk", "5000", "--keep-threshold", "0.5", "--max-per-class", "300000",
        "--include-pool", "ddim50_n50000"])
    assert (args.action, args.mine_tag, args.detector_tag, args.target, args.chunk,
            args.keep_threshold, args.max_per_class, args.include_pool) == \
        ("mine", "r50v1", "final_r50", 150, 5000, 0.5, 300000, "ddim50_n50000")
    train = build_parser().parse_args([
        "inject", "detector-train", "--dataset", "cifar10_inj4",
        "--detector-config", "detector_all_r50", "--detector-tag", "dev_r50"])
    assert (train.detector_config, train.detector_tag) == ("detector_all_r50", "dev_r50")
