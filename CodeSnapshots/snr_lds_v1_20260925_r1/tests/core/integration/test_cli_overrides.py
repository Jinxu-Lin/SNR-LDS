"""``--set`` accepts list values (CODE_CLI_LIST_OVERRIDE r1).

D3-C3 B2 extends ``ekfac.blockshrink_grid`` downward. Before this, every ``--set``
value became int, float or str, so a list arrived as a string and
``curvature._grid`` failed. What has to hold:

* ``[1.0e-6,1e-5,...,1]`` reaches the config as seven floats (``1e-6`` without a
  dot included) and ``_grid`` returns them for a blockshrink caliber;
* scalar overrides are coerced exactly as before;
* a malformed list literal raises naming the key.
"""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from balds.workflows.config import load_config  # noqa: E402
from balds.workflows.curvature import _ekfac_cfg, _grid  # noqa: E402

cli = importlib.import_module("balds.cli.main")

GRID = "[1.0e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]"
GRID_FLOATS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def test_list_literal_reaches_config_as_floats_and_grid_returns_them():
    overrides = cli.parse_overrides([f"ekfac.blockshrink_grid={GRID}"])
    grid = overrides["ekfac.blockshrink_grid"]
    assert grid == GRID_FLOATS and all(type(v) is float for v in grid)
    cfg = load_config(overrides)
    assert cfg["ekfac"]["blockshrink_grid"] == GRID_FLOATS
    e = {**_ekfac_cfg(cfg), "damping_mode": "blockshrink"}
    assert _grid(e) == GRID_FLOATS
    # the global-damping grid is untouched by this override
    assert _grid({**_ekfac_cfg(cfg), "damping_mode": "global"}) == \
        [float(d) for d in load_config({})["ekfac"]["damping_grid"]]
    print(f"  blockshrink_grid {GRID} -> {len(grid)} floats; _grid returns them")


def test_scalar_overrides_keep_the_original_coercion():
    got = cli.parse_overrides([
        "lds.Q_by_dataset.cifar2_das=100", "storage.data_root=/some/path",
        "score.alpha=0.5", "ekfac.max_train_samples=1e-6", "paper.fmas_layer=shrink",
        "x.neg=-3", "x.spaces=a b=c"])
    assert got == {"lds.Q_by_dataset.cifar2_das": 100, "storage.data_root": "/some/path",
                   "score.alpha": 0.5, "ekfac.max_train_samples": 1e-6,
                   "paper.fmas_layer": "shrink", "x.neg": -3, "x.spaces": "a b=c"}
    assert type(got["lds.Q_by_dataset.cifar2_das"]) is int and type(got["score.alpha"]) is float
    # the previous nested coercion, verbatim, agrees on every non-list value
    def old(v):
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    for raw in ("100", "/some/path", "0.5", "1e-6", "shrink", "-3", "nan", "inf", "", "1_000"):
        new = cli.coerce_override("k", raw)
        want = old(raw)
        assert type(new) is type(want) and (new == want or (new != new and want != want)), raw
    print("  scalar overrides: int/float/str exactly as before (incl. '', nan, inf, 1_000)")


def test_int_lists_stay_ints_and_mixed_lists_become_floats():
    assert cli.coerce_override("k", "[0,1,2]") == [0, 1, 2]
    assert all(type(v) is int for v in cli.coerce_override("k", "[0,1,2]"))
    assert cli.coerce_override("k", "[1, 2.5]") == [1.0, 2.5]
    assert cli.coerce_override("k", '["a", "b"]') == ["a", "b"]
    assert cli.coerce_override("k", " [ ]") == []


@pytest.mark.parametrize("raw", ["[1e-6,", "[1e-6,,1e-5]", "[a,b]", "[1e-6] extra"])
def test_malformed_list_literal_raises_naming_the_key(raw):
    with pytest.raises(ValueError, match=r"--set ekfac\.blockshrink_grid: malformed list literal"):
        cli.parse_overrides([f"ekfac.blockshrink_grid={raw}"])


def test_cli_hands_the_parsed_list_to_the_container(monkeypatch):
    seen = {}

    class _Stop(Exception):
        pass

    def stub_container(overrides, device):
        seen.update(overrides)
        raise _Stop()

    monkeypatch.setattr(cli, "build_container", stub_container)
    with pytest.raises(_Stop):
        cli.main(["--device", "cpu", "--set", f"ekfac.blockshrink_grid={GRID}",
                  "--set", "lds.Q_by_dataset.cifar2_das=100",
                  "ekfac", "score", "--method", "fmas_raw", "--dataset", "cifar2_das",
                  "--process", "ddpm", "--seed", "42"])
    assert seen == {"ekfac.blockshrink_grid": GRID_FLOATS, "lds.Q_by_dataset.cifar2_das": 100}
    with pytest.raises(ValueError, match="ekfac.blockshrink_grid"):
        cli.main(["--device", "cpu", "--set", "ekfac.blockshrink_grid=[1e-6,",
                  "ekfac", "score", "--method", "fmas_raw", "--dataset", "cifar2_das"])
    print("  CLI: --set list reaches build_container as floats; malformed list stops before it")
