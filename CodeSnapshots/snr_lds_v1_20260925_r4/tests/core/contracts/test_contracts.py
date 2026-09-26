"""Contracts-layer unit tests. Pure stdlib — runs without torch or pytest.

    python tests/contracts/test_contracts.py     # standalone
    pytest tests/contracts/test_contracts.py      # under pytest
"""
from __future__ import annotations

from balds.schema import (
    ArtifactKind,
    Registry,
    RunSpec,
)


def test_registry_basic():
    reg = Registry("thing")

    @reg.register("a")
    def fa():
        return 1

    reg.add("b", lambda: 2)
    assert reg.get("a") is fa
    assert reg.names() == ["a", "b"]
    assert reg.has("a") and not reg.has("z")
    assert len(reg) == 2


def test_registry_duplicate_raises():
    reg = Registry("thing")
    reg.add("a", 1)
    try:
        reg.add("a", 2)
    except KeyError:
        pass
    else:
        raise AssertionError("duplicate registration must raise KeyError")


def test_registry_unknown_raises():
    reg = Registry("thing")
    try:
        reg.get("missing")
    except KeyError as e:
        assert "missing" in str(e) and "registered" in str(e)
    else:
        raise AssertionError("unknown name must raise KeyError")


def test_runspec_digest_deterministic():
    a = RunSpec(dataset="cifar2", method="das", seed=42)
    b = RunSpec(dataset="cifar2", method="das", seed=42)
    assert a.digest() == b.digest()
    assert len(a.digest()) == 16


def test_runspec_changes_digest():
    a = RunSpec(dataset="cifar2", method="das")
    assert a.with_(method="dtrak").digest() != a.digest()
    assert a.with_(query_type="val").is_val
    assert not a.is_val


def test_artifact_kinds_distinct():
    vals = [k.value for k in ArtifactKind]
    assert len(vals) == len(set(vals))
    assert ArtifactKind.SCORES.value == "scores"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASSED {len(fns)} contracts tests")


if __name__ == "__main__":
    _run_all()
