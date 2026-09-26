"""EXISTS SEMANTICS (incident 2026-08-09): a manifest row is NOT proof that an
artifact is materialized, and ``manifest_db``/``cas_root`` follow ``data_root``.

Background — the two failure modes this locks down:
  1. A smoke run with ``--set storage.data_root=<tmp>`` wrote its checkpoint
     rows into the MASTER manifest (manifest_db did not follow data_root) while
     the files landed in a throwaway tree that was then deleted.
  2. ``exists()`` accepted a manifest row as existence, so the next real
     training run for that identity skipped — reporting success, producing
     nothing. The same path guards per-subset skips, i.e. 64 subsets could
     "complete" in seconds with zero files and exit 0.

Run (pure stdlib + torch-free):  python Codes/tests/store/test_exists_semantics.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows.config import load_config
from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from balds.artifacts import ArtifactStore, LocalBackend, ManifestDB


def _store(root):
    return ArtifactStore(ManifestDB(f"{root}/m.db"), [LocalBackend(f"{root}/data")],
                            code_version="t")


def test_orphan_manifest_row_is_not_existence():
    root = tempfile.mkdtemp()
    store = _store(root)
    spec = RunSpec(dataset="cifar2_5k", process="cfm", seed=42, conditional=False)
    store.save(K.LOSS_HISTORY, spec, {"history": []})
    assert store.exists(K.LOSS_HISTORY, spec), "freshly saved artifact must exist"
    # delete the bytes, keep the manifest row -> the orphan-row situation
    from balds.artifacts import addressing
    os.remove(f"{root}/data/{addressing.relpath(K.LOSS_HISTORY, spec)}")
    assert store.manifest.get(store._item_key(K.LOSS_HISTORY, spec, {})) is not None, \
        "precondition: the manifest row must still be there"
    assert not store.exists(K.LOSS_HISTORY, spec), \
        "an orphan manifest row must NOT report the artifact as existing"
    print("  orphan manifest row -> exists()=False (idempotent skips no longer lie)")


def test_manifest_and_cas_follow_data_root():
    cfg = load_config({"storage.data_root": "/tmp/throwaway_root"})
    assert cfg["storage"]["manifest_db"] == "/tmp/throwaway_root/manifest.db", cfg["storage"]
    assert cfg["storage"]["cas_root"] == "/tmp/throwaway_root/cas"
    explicit = load_config({"storage.data_root": "/tmp/a", "storage.manifest_db": "/tmp/b/m.db"})
    assert explicit["storage"]["manifest_db"] == "/tmp/b/m.db", "explicit override must win"
    default = load_config({})
    # the relative default "_Data/manifest.db" is anchored at the REPO root, not
    # the cwd -- a run launched from Codes/ must not fork a second library
    from balds.workflows.config import REPO_ROOT
    assert default["storage"]["manifest_db"] == f"{REPO_ROOT}/_Data/manifest.db", \
        f"default must anchor at repo root, got {default['storage']['manifest_db']}"
    assert default["storage"]["data_root"] == f"{REPO_ROOT}/_Data"
    print("  manifest_db/cas_root follow data_root; explicit override wins; "
          "relative default anchored at repo root")


def test_local_and_remote_still_report_existence():
    root = tempfile.mkdtemp()
    store = _store(root)
    spec = RunSpec(dataset="cifar2_5k", seed=7, conditional=False)
    assert not store.exists(K.LOSS_HISTORY, spec), "nothing saved yet"
    store.save(K.LOSS_HISTORY, spec, {"history": [1]})
    assert store.exists(K.LOSS_HISTORY, spec), "local file must count as existence"
    assert store.load(K.LOSS_HISTORY, spec) == {"history": [1]}
    print("  positive path intact: local bytes -> exists()=True, load round-trips")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} exists-semantics tests passed")
