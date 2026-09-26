"""``balds.report`` must export everything the CLI reaches through the package.

``balds/cli/main.py`` does ``from balds import report`` and then calls
``report.render_inject(out)``. ``render_inject`` lived only in
``balds.report.tables``, so ``inject evaluate`` died with ``AttributeError`` after
the scoring work was already done, while the suite stayed green because
``test_inject_evaluate.py`` imports the submodule directly (2026-09-15).

What has to hold:

* every public name of every module under ``balds/report/`` is reachable as
  ``balds.report.<name>`` and listed in ``__all__`` -- a generic guard, so the next
  renderer added to ``tables.py`` cannot repeat this;
* every ``report.<name>`` the CLI calls resolves on the package;
* ``balds.report.render_inject`` renders, and ``inject evaluate`` prints its table
  through the real CLI entry point.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import balds.report as report  # noqa: E402
from balds.workflows.config import load_config  # noqa: E402

cli = importlib.import_module("balds.cli.main")

RESULT = {
    "dataset": "cifar10_inj4", "split": "test", "method": "pixel", "k": 10,
    "best_lam": 0.1, "metrics": ["recall", "precision"],
    "per_concept": {
        "sunflower": {"recall": {"mean": 0.5, "ci_lo": 0.4, "ci_hi": 0.6},
                      "precision": {"mean": 0.25, "ci_lo": 0.2, "ci_hi": 0.3}},
        "castle": {"recall": {"mean": 0.75, "ci_lo": 0.7, "ci_hi": 0.8},
                   "precision": {"mean": 0.5, "ci_lo": 0.45, "ci_hi": 0.55}},
    },
    "per_tier": {
        "fine": {"recall": {"mean": 0.625, "ci_lo": 0.55, "ci_hi": 0.7},
                 "precision": {"mean": 0.375, "ci_lo": 0.3, "ci_hi": 0.45}},
    },
    "overall": {"recall": {"mean": 0.625, "ci_lo": 0.55, "ci_hi": 0.7},
                "precision": {"mean": 0.375, "ci_lo": 0.3, "ci_hi": 0.45}},
    "chance": {"recall": 0.05, "precision": 0.02},
}


def _public_names() -> dict[str, list[str]]:
    """Public functions and classes defined by each module under ``balds/report/``."""
    found: dict[str, list[str]] = {}
    for info in pkgutil.iter_modules(report.__path__):
        module = importlib.import_module(f"balds.report.{info.name}")
        found[info.name] = sorted(
            name for name, value in vars(module).items()
            if not name.startswith("_")
            and (inspect.isfunction(value) or inspect.isclass(value))
            and getattr(value, "__module__", None) == module.__name__)
    return found


def test_every_public_name_under_report_is_exported_by_the_package():
    modules = _public_names()
    assert modules, "no modules found under balds/report/"
    missing_attr, missing_all = [], []
    for module_name, names in modules.items():
        for name in names:
            if getattr(report, name, None) is None:
                missing_attr.append(f"balds.report.{module_name}.{name}")
            elif name not in report.__all__:
                missing_all.append(f"{name} (in balds.report.{module_name})")
    assert not missing_attr, f"not reachable as balds.report.<name>: {missing_attr}"
    assert not missing_all, f"reachable but absent from __all__: {missing_all}"
    assert set(report.__all__) >= {"render_table", "render_lds", "render_inject"}
    print(f"  exported: {sorted(report.__all__)}")


def test_every_report_attribute_the_cli_calls_resolves_on_the_package():
    """Parse ``report.<name>(`` out of the CLI rather than trusting one call site."""
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert "from balds import report" in source
    used = sorted(set(__import__("re").findall(r"\breport\.(\w+)\s*\(", source)))
    assert "render_inject" in used and "render_lds" in used
    unresolved = [name for name in used if not hasattr(report, name)]
    assert not unresolved, f"CLI calls report.{unresolved} but the package lacks it"
    print(f"  CLI reaches: {used}")


def test_render_inject_through_the_package_renders_every_group():
    text = report.render_inject(RESULT)
    assert text == report.tables.render_inject(RESULT)
    for line in ("concept:castle", "concept:sunflower", "tier:fine", "overall", "chance"):
        assert line in text, line
    assert "INJECT cifar10_inj4/test pixel k=10" in text
    assert "0.5000 [0.4000,0.6000]" in text          # sunflower recall
    assert "0.0500" in text                           # chance recall


def test_cli_inject_evaluate_prints_the_table(monkeypatch, capsys):
    """The failing path end to end: ``balds inject evaluate`` through ``main()``."""
    seen = {}

    class _Uc:
        def evaluate(self, method, dataset, seed, *, process, split, k, conditional, force=False):
            seen.update(method=method, dataset=dataset, seed=seed, process=process,
                        split=split, k=k, conditional=conditional)
            return RESULT

    class _Container:
        cfg = load_config({})

        def inject_usecase(self):
            return _Uc()

    monkeypatch.setattr(cli, "build_container", lambda overrides, device: _Container())
    code = cli.main(["--device", "cpu", "inject", "evaluate", "--method", "pixel",
                     "--dataset", "cifar10_inj4", "--process", "ddpm", "--seed", "42"])
    assert code == 0
    assert seen["method"] == "pixel" and seen["dataset"] == "cifar10_inj4"
    out = capsys.readouterr().out
    assert "INJECT cifar10_inj4/test pixel k=10" in out and "concept:castle" in out
    print("  balds inject evaluate printed the INJECT table (no AttributeError)")


def test_inject_evaluate_without_method_still_refuses_before_scoring(monkeypatch):
    """Unchanged behaviour: the use case is built, but nothing is evaluated."""
    class _Uc:
        def evaluate(self, *a, **k):
            raise AssertionError("inject evaluate ran without --method")

    class _Container:
        cfg = load_config({})

        def inject_usecase(self):
            return _Uc()

    monkeypatch.setattr(cli, "build_container", lambda overrides, device: _Container())
    with pytest.raises(SystemExit, match="inject evaluate requires --method"):
        cli.main(["--device", "cpu", "inject", "evaluate", "--dataset", "cifar10_inj4"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
