"""Configuration loading (convention over configuration).

Reads the single bundled ``conf/defaults.yaml`` via ``importlib.resources`` (no
hand-written file IO), then applies dotted-key overrides. Per-run identity lives
in :class:`~balds.schema.runspec.RunSpec` (CLI flags), never in YAML.
"""
from __future__ import annotations

import importlib.resources
import os
from pathlib import Path
from typing import Any

import yaml

def project_root() -> Path:
    """Find an SNR-LDS checkout; installed wheels otherwise use the working directory."""
    package = Path(__file__).resolve()
    for candidate in (*package.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "Codes").is_dir() and (candidate / "Paper").is_dir():
            return candidate
    return Path.cwd().resolve()


REPO_ROOT = str(project_root())


def load_config(overrides: dict[str, Any] | None = None) -> dict:
    """Load defaults and apply ``{"a.b.c": value}`` dotted overrides."""
    text = importlib.resources.files("balds").joinpath("conf", "defaults.yaml").read_text()
    cfg = yaml.safe_load(text)
    default_data_root = str((cfg.get("storage") or {}).get("data_root", ""))
    overrides = dict(overrides or {})
    if "storage.data_root" not in overrides and os.environ.get("BALDS_DATA_ROOT"):
        overrides["storage.data_root"] = os.environ["BALDS_DATA_ROOT"]
    for dotted, value in overrides.items():
        node = cfg
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    _derive_storage_paths(cfg, overrides, default_data_root)
    _anchor_relative_paths(cfg)
    return cfg


#: ``storage.*`` keys whose value is a filesystem path. Every key of
#: ``datasets.raw_dirs`` is a path too, whatever its name.
_STORAGE_PATH_KEYS = ("data_root", "cas_root", "manifest_db")


def _anchored(value):
    if isinstance(value, str) and value and not os.path.isabs(value):
        return os.path.abspath(os.path.join(REPO_ROOT, os.path.expanduser(value)))
    return value


def _raw_dirs(cfg: dict) -> dict | None:
    datasets = cfg.get("datasets")
    raw = datasets.get("raw_dirs") if isinstance(datasets, dict) else None
    return raw if isinstance(raw, dict) else None


def _anchor_relative_paths(cfg: dict) -> None:
    """Resolve relative storage and raw-data paths against the REPO root, not the cwd.

    ``storage.data_root`` defaults to the bare string ``"_Data"``. Interpreted
    relative to the current working directory, that means the artifact library
    is wherever the shell happened to be: a run launched from ``Codes/`` creates
    and silently fills a SECOND library at ``Codes/_Data``, invisible to every
    other run and to the master manifest. That is not hypothetical -- it
    happened (25 duplicated score/GT artifacts plus an empty manifest, removed
    2026-09-01). There is exactly one artifact library per checkout, and it sits
    at ``<repo>/_Data``; anchoring here makes the cwd irrelevant.

    Every ``datasets.raw_dirs`` key is anchored, not a fixed list: a key missing
    from the list (``inject_foreign`` until 2026-09-15) resolved against the cwd.
    Absolute paths (tests, alternate trees, ``--set`` overrides pointing
    off-box) pass through untouched.
    """
    storage = cfg.get("storage")
    if isinstance(storage, dict):
        for key in _STORAGE_PATH_KEYS:
            if key in storage:
                storage[key] = _anchored(storage[key])
    raw = _raw_dirs(cfg)
    if raw is not None:
        for key in list(raw):
            raw[key] = _anchored(raw[key])


def _relative_under(path: str, base: str) -> str | None:
    """``path`` relative to ``base`` when it lies at or under it, else ``None``."""
    if not base:
        return None
    p, b = os.path.normpath(path), os.path.normpath(base)
    if os.path.isabs(p) != os.path.isabs(b):
        return None
    if p == b:
        return ""
    return p[len(b) + 1:] if p.startswith(b + os.sep) else None


def _derive_storage_paths(cfg: dict, overrides: dict[str, Any],
                          default_data_root: str = "") -> None:
    """Keep ``manifest_db``/``cas_root`` and the raw-data dirs inside whatever ``data_root`` points at.

    They are derived quantities, not independent knobs: overriding only
    ``storage.data_root`` (smoke runs, tests, alternate trees) used to leave the
    manifest pointing at the MASTER library while artifacts landed elsewhere, so
    a throwaway run wrote master manifest rows for files it then deleted. Those
    orphan rows made ``exists()`` claim artifacts existed and silently turned a
    real training run into a no-op (incident 2026-08-09, cifar2_5k/seed42).

    The same holds for ``datasets.raw_dirs``: a default that lies under the
    default data root moves to the same relative place under the overridden
    one. Before 2026-09-15 it did not, so a detached checkout run with
    ``--set storage.data_root=<master library>`` wrote artifacts to the library
    but looked for the CIFAR cache, ArtBench and the DAS archive inside the
    checkout (§6.2-53; only ``HF_HUB_OFFLINE=1`` stopped a silent re-download).
    An explicit override of any derived key -- or of the whole ``raw_dirs`` or
    ``datasets`` block -- still wins.
    """
    s = cfg.get("storage")
    if not isinstance(s, dict) or "storage.data_root" not in overrides:
        return
    root = str(s.get("data_root", "")).rstrip("/")
    if not root:
        return
    if "storage.manifest_db" not in overrides:
        s["manifest_db"] = f"{root}/manifest.db"
    if "storage.cas_root" not in overrides:
        s["cas_root"] = f"{root}/cas"
    raw = _raw_dirs(cfg)
    if raw is None or "datasets" in overrides or "datasets.raw_dirs" in overrides:
        return
    for key, value in raw.items():
        if f"datasets.raw_dirs.{key}" in overrides or not isinstance(value, str):
            continue
        rel = _relative_under(value, default_data_root)
        if rel is not None:
            raw[key] = f"{root}/{rel}" if rel else root


def describe_paths(cfg: dict) -> str:
    """One line naming where a run writes artifacts and reads raw data."""
    storage = cfg.get("storage") or {}
    parts = [f"{key}={storage.get(key)}" for key in ("data_root", "manifest_db", "cas_root")]
    parts += [f"raw_dirs.{key}={value}" for key, value in (_raw_dirs(cfg) or {}).items()]
    return "paths: " + " ".join(parts)


def cfg_get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Read a nested config value by dotted path."""
    node = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def resolve_t_avg(cfg: dict, dataset: str) -> int:
    """GT/e_n averaging timepoints for ``dataset``.

    §6.2-16 (researcher decision 2026-08-17): platforms whose GT was already
    produced at 1000 timepoints (cifar2*, cifar10*) keep it; NEW platforms
    (ArtBench) use 100 — aligned with the T=100 feature grid, and 10x cheaper
    where the latent GT is the dominant cost. Table-driven like gamma so the
    per-platform value is one config row, not a per-command --set to forget.
    """
    tbl = (cfg.get("lds", {}) or {}).get("T_avg_by_dataset") or {}
    if dataset in tbl:
        return int(tbl[dataset])
    return int(cfg["lds"]["T_avg"])


def resolve_M(cfg: dict, dataset: str) -> int:
    """Number of GT subsets for ``dataset``.

    Per-platform because M buys statistical power at a price that differs by two
    orders of magnitude between platforms: a CIFAR-2 subset retrain is minutes, an
    ArtBench SD3.5+LoRA subset retrain is ~12 GPU-hours, so M=64 there is ~790
    GPU-hours of the ~1040 for the whole chain (researcher decision 2026-08-18:
    ArtBench runs M=32).

    Table-driven, like gamma and T_avg, specifically so this cannot leak into the
    CIFAR platforms: their 116-cell main table is built on M=64 masks, and a
    global edit would silently regenerate them at 32 on the next `subsets masks`.
    LDS consumers read the first M rows of the filed masks and GT through
    :func:`lds_rows`. Wherever the artifact holds exactly M rows — every platform
    this project generated — that is ``len(masks)``, so the artifact stays the
    authority. An imported archive may file more (``cifar2_das``: 128 filed, 64
    used, researcher ruling 2026-09-14, ledger §6.2-51 ①).
    """
    tbl = (cfg.get("lds", {}) or {}).get("M_by_dataset") or {}
    if dataset in tbl:
        return int(tbl[dataset])
    return int(cfg["lds"]["M"])


def lds_rows(cfg: dict, dataset: str, masks, gt):
    """``(masks[:M], gt[:M])`` with ``M = resolve_M(cfg, dataset)``: the subsets an LDS uses.

    The identity when the artifacts hold at most M rows. ``cifar2_das`` keeps all
    128 archive subsets on disk — they are two separately produced batches whose
    GT differs by a per-query constant, which is the evidence for §6.2-51 and
    what ``Codes/Figure/tab_C3_ddpm.py`` reads in full — while score, evaluate,
    gamma-scan and the curvature path's damping selection use subsets 0..M-1.
    Slicing at load, never refiling, keeps both readings available.
    """
    M = resolve_M(cfg, dataset)
    return masks[:M], gt[:M]


def resolve_Q(cfg: dict, dataset: str) -> int:
    """Number of queries per track for ``dataset``.

    Table-driven like gamma / M / T_avg, and for the same reason: the global
    ``lds.Q`` is 100 because that is how many queries this project *generates*,
    but an IMPORTED platform's query set is not ours to choose — ``cifar2_das``
    ships 1000 gen and 1000 val queries, and its features and ground truth are
    1000 columns wide. Taking the global 100 there would either mismatch the
    imported artifacts or silently subsample them, and the subsample is not free:
    it moves the identity-row LDS by -.002..+.011 relative to the published
    recomputation on the same data (d2_das_linear_2026-09-02).
    """
    tbl = (cfg.get("lds", {}) or {}).get("Q_by_dataset") or {}
    if dataset in tbl:
        return int(tbl[dataset])
    return int(cfg["lds"]["Q"])


def resolve_p_uncond(cfg: dict, dataset: str) -> float:
    """Classifier-free-guidance dropout rate for ``dataset``.

    Table-driven like gamma / M / T_avg, and for the same reason: the global
    ``train.p_uncond`` default is 0.1, but CIFAR-10's adjudicated recipe is the
    PURE-conditional 0.0 — every cifar10_v2 artifact in the library (seed 42's
    main model and its 64 subsets, produced on xc0 2026-08-09) carries
    ``p_uncond=0.0``. Until this table existed the only thing standing between a
    run and a silently mis-trained model was remembering to pass
    ``--set train.p_uncond=0`` on every command; a task brief that said "take the
    train defaults" produced 0.1 instead, and the acceptance checklist
    (conditional/model_type/num_classes/step) does not look at p_uncond, so the
    wrong model would have passed every check and gone downstream
    """
    tbl = (cfg.get("train", {}) or {}).get("p_uncond_by_dataset") or {}
    if dataset in tbl:
        return float(tbl[dataset])
    return float(cfg["train"]["p_uncond"])


def resolve_gamma(cfg: dict, dataset: str, method: str, *,
                  strict: bool = True) -> float | None:
    """The frozen shrinkage knee γ for ``(dataset, method)``.

    Per dataset (2026-08-16 transfer check: cifar10_v2's optimum is 6–11x
    cifar2_5k's; borrowing costs 0.0134) AND per method: every shrinkage
    method reads ``tweedie.gamma_by_method.<method>.<dataset>``. Score scales
    differ per solver (EK-FAC eigenbasis inner products vs a mean_abs-normalised
    projected kernel), so handing one method another's γ quietly crushes it and
    reads as "the method is weak". Hence NO fallback: an uncalibrated cell
    raises (``strict=True``) with the calibration command; ``strict=False``
    returns ``None`` for the tool that produces the value (gamma-scan).
    """
    tw = cfg.get("tweedie", {}) or {}
    table = (tw.get("gamma_by_method") or {}).get(method) or {}
    val, where = table.get(dataset), f"tweedie.gamma_by_method.{method}.{dataset}"
    if val is None:
        if strict:
            raise ValueError(
                f"no calibrated gamma for method {method!r} on dataset {dataset!r} "
                f"({where} is missing or null). Calibrate first: "
                f"`balds gamma-scan --method {method} --dataset {dataset} ...`, then "
                f"write the chosen knee into defaults.yaml. There is deliberately "
                f"no default: a silently inherited gamma reads as a weaker method.")
        return None
    return float(val)


def resolve_paper_method(cfg: dict, method: str) -> str:
    """FMAS denotes the paper's bilinear kernel; its artifact key is fmas_raw."""
    return "fmas_raw" if method == "fmas" else method


def resolve_pixel_resize(cfg: dict, dataset: str) -> int | None:
    """Spatial size the ``pixel`` baseline flattens at, or ``None`` for none.

    Table-driven like gamma / M / T_avg / Q, and for the same reason: the
    baseline is "flatten the raw image", which is a 3072-vector at 32x32 and a
    196,608-vector at 256x256 — 3.9 GB for one 5000-row ArtBench train feature
    and a kernel nobody can hold. The 256px platforms therefore flatten an
    area-averaged 64x64 (12,288 dims, the CIFAR order of magnitude); the CIFAR
    datasets are absent from the table and keep the literal raw pixels, so every
    existing pixel artifact stays what it was.
    """
    tbl = (cfg.get("featurize", {}) or {}).get("pixel_resize_by_dataset") or {}
    if dataset in tbl and tbl[dataset]:
        return int(tbl[dataset])
    return None
