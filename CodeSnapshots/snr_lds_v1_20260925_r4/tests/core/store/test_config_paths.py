"""Raw-data directories follow an overridden data root (CODE_DATA_ROOT_ANCHOR r1, §6.2-53).

The incident: a detached checkout ran with ``--set storage.data_root=<master
library>``; artifacts went to the library, but ``datasets.raw_dirs`` still
resolved inside the checkout (and ``inject_foreign`` against the cwd). What has
to hold:

* with ``storage.data_root`` overridden, every raw dir whose default lies under
  the default data root moves under the override, from any working directory,
  and ``manifest_db``/``cas_root`` derive as before;
* an explicit raw-dir override (a key, or the whole block) wins; a relative
  explicit value anchors to REPO_ROOT;
* with no overrides nothing moves, except ``inject_foreign`` is now repo-anchored;
* ``_raw_dir`` / ``_load_ds`` hand the derived paths to the dataset loaders;
* the CLI logs one line naming every resolved path.
"""
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows import common as pipeline  # noqa: E402
from balds.workflows.config import REPO_ROOT, describe_paths, load_config  # noqa: E402

#: the raw dirs defaults.yaml declares, relative to the default data root "_Data"
DEFAULT_REL = {
    "cifar": "hf_cache",
    "artbench": "raw/artbench-10-imagefolder-split",
    "das_archive": "raw/das_archive",
}


@pytest.fixture
def elsewhere(tmp_path, monkeypatch):
    """A working directory that is neither the repo nor the library."""
    cwd = tmp_path / "some_cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return tmp_path


def test_raw_dirs_follow_an_absolute_data_root_override_from_any_cwd(elsewhere):
    root = elsewhere / "library"
    cfg = load_config({"storage.data_root": str(root)})
    raw = cfg["datasets"]["raw_dirs"]
    assert set(DEFAULT_REL) <= set(raw)
    for key in raw:
        assert raw[key] == f"{root}/{DEFAULT_REL[key]}", key
    assert cfg["storage"]["data_root"] == str(root)
    assert cfg["storage"]["manifest_db"] == f"{root}/manifest.db"
    assert cfg["storage"]["cas_root"] == f"{root}/cas"
    assert not any(str(elsewhere / "some_cwd") in v for v in raw.values())
    # trailing slash on the override does not double the separator
    slashed = load_config({"storage.data_root": f"{root}/"})["datasets"]["raw_dirs"]
    assert slashed["cifar"] == f"{root}/hf_cache"


def test_raw_dirs_follow_a_relative_data_root_override_anchored_to_the_repo(elsewhere):
    cfg = load_config({"storage.data_root": "alt_library"})
    assert cfg["storage"]["data_root"] == f"{REPO_ROOT}/alt_library"
    for key, rel in DEFAULT_REL.items():
        assert cfg["datasets"]["raw_dirs"][key] == f"{REPO_ROOT}/alt_library/{rel}", key
    assert cfg["storage"]["manifest_db"] == f"{REPO_ROOT}/alt_library/manifest.db"


def test_explicit_raw_dir_overrides_win(elsewhere):
    root = elsewhere / "library"
    cfg = load_config({
        "storage.data_root": str(root),
        "datasets.raw_dirs.cifar": "/data/shared/hf_cache",      # absolute: kept
        "datasets.raw_dirs.artbench": "local/artbench",          # relative: repo-anchored
    })
    raw = cfg["datasets"]["raw_dirs"]
    assert raw["cifar"] == "/data/shared/hf_cache"
    assert raw["artbench"] == f"{REPO_ROOT}/local/artbench"
    assert raw["das_archive"] == f"{root}/raw/das_archive"

    block = load_config({"storage.data_root": str(root),
                         "datasets.raw_dirs": {"cifar": "rel/cache", "extra": "/abs/extra"}})
    assert block["datasets"]["raw_dirs"] == {"cifar": f"{REPO_ROOT}/rel/cache",
                                            "extra": "/abs/extra"}


def test_no_override_is_unchanged_and_inject_foreign_is_repo_anchored(elsewhere):
    cfg = load_config({})
    assert cfg["storage"]["data_root"] == f"{REPO_ROOT}/_Data"
    assert cfg["storage"]["manifest_db"] == f"{REPO_ROOT}/_Data/manifest.db"
    assert cfg["storage"]["cas_root"] == f"{REPO_ROOT}/_Data/cas"
    for key, rel in DEFAULT_REL.items():
        assert cfg["datasets"]["raw_dirs"][key] == f"{REPO_ROOT}/_Data/{rel}", key
    # overriding something unrelated moves nothing either
    other = load_config({"lds.Q": 7})
    assert other["datasets"]["raw_dirs"] == cfg["datasets"]["raw_dirs"]
    assert other["storage"] == cfg["storage"]


def test_raw_dir_and_load_ds_receive_the_derived_paths(elsewhere, monkeypatch):
    root = elsewhere / "library"
    cfg = load_config({"storage.data_root": str(root)})
    seen = []
    monkeypatch.setattr(pipeline, "get_dataset",
                        lambda name, data_dir, **kw: seen.append((name, data_dir, kw)) or (1, 2))

    assert pipeline._raw_dir(cfg, "cifar2_5k") == f"{root}/hf_cache"
    assert pipeline._raw_dir(cfg, "artbench10") == f"{root}/raw/artbench-10-imagefolder-split"
    pipeline._load_ds(cfg, "cifar2_das")
    pipeline._load_ds(cfg, "artbench2_256")
    (_, das_dir, das_kw), (_, art_dir, art_kw) = seen
    assert das_dir == f"{root}/hf_cache"
    assert das_kw["raw_dirs"]["das_archive"] == f"{root}/raw/das_archive"
    assert art_dir == f"{root}/raw/artbench-10-imagefolder-split"
    assert art_kw["raw_dirs"]["artbench"] == f"{root}/raw/artbench-10-imagefolder-split"
    assert art_kw["inject_spec"] is None


def test_describe_paths_is_one_line_naming_every_path(elsewhere):
    root = elsewhere / "library"
    cfg = load_config({"storage.data_root": str(root)})
    line = describe_paths(cfg)
    assert "\n" not in line and line.startswith("paths: ")
    for fragment in (f"data_root={root}", f"manifest_db={root}/manifest.db",
                     f"cas_root={root}/cas",
                     *(f"raw_dirs.{k}={root}/{rel}" for k, rel in DEFAULT_REL.items())):
        assert fragment in line, fragment


def test_cli_logs_the_resolved_paths_at_start(elsewhere, monkeypatch, caplog):
    import importlib

    # the module, not the ``main`` function that ``balds.cli`` re-exports under the same name
    cli = importlib.import_module("balds.cli.main")

    root = elsewhere / "library"

    class _Stop(Exception):
        pass

    def stub_container(overrides, device):
        cfg = load_config(overrides)
        return SimpleNamespace(cfg=cfg, evaluate_usecase=lambda: (_ for _ in ()).throw(_Stop()))

    monkeypatch.setattr(cli, "build_container", stub_container)
    caplog.set_level(logging.INFO, logger="balds.cli")
    with pytest.raises(_Stop):
        cli.main(["--device", "cpu", "--set", f"storage.data_root={root}", "evaluate",
                  "--method", "dtrak_T100", "--dataset", "cifar2_5k", "--seed", "42"])
    lines = [r.getMessage() for r in caplog.records if r.name == "balds.cli"]
    assert lines == [describe_paths(load_config({"storage.data_root": str(root)}))]
    assert f"raw_dirs.cifar={root}/hf_cache" in lines[0]
