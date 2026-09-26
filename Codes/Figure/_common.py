"""Shared loaders and numerical LDS evaluation for retained paper diagnostics.

Artifact paths follow balds.artifacts.addressing. BALDS_DATA_ROOT selects the
input root; outputs go under results/paper/figures (overridable by BALDS_FIGURE_ROOT).
See README.md and Reports/experiments for input identity and execution order.
"""
from __future__ import annotations

import os
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]                     # repo root
sys.path.insert(0, str(ROOT / "Codes" / "src"))
from balds.evaluation.lds import compute_lds, predicted_influence      # noqa: E402
from balds.evaluation.stats import bootstrap_ci                        # noqa: E402

from _paths import DATA, FIGURE_ROOT, METADATA, figure_dir
OUT = FIGURE_ROOT
TABLES, FIGS, JSONS = figure_dir("tables"), figure_dir("diagnostics"), METADATA

SEEDS = (42, 123, 456)
TRACKS = ("gen", "val")
#: the four "same-source" score constructions of §3.1 / §4.2 / §4.3
CORE = (("fmas_raw", "FMAS"), ("ekfac_if", "EK-FAC IF"),
        ("dtrak_T100", "D-TRAK"), ("das_T100", "DAS"))
#: κ grid of the head protocol (percent of N kept). Researcher 2026-09-14:
#: start at 5 % (1–2 % are statistically meaningless at N=5000), no 50→100 jump.
KAPPA_GRID = (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
KAPPA_MAIN = 5
#: |score|-rank bands of the repeat profile (E0a) and band isolation (E1)
BANDS = ((0, 1), (1, 5), (5, 20), (20, 50), (50, 100))
RHO_MAIN = 1e-2                       # FMAS per-layer damping used everywhere in the paper


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def ds_track(ds: str, track: str) -> str:
    return ds if track == "gen" else f"{ds}_val"


def scores_path(method: str, ds: str, track: str, seed: int) -> Path:
    return DATA / "scores" / method / ds_track(ds, track) / f"seed_{seed}" / "scores.npy"


def gt_path(ds: str, track: str, seed: int) -> Path:
    return DATA / "results" / f"gt_matrix_fm_{ds_track(ds, track)}_seed_{seed}.npy"


def has_scores(method, ds, track, seed) -> bool:
    return scores_path(method, ds, track, seed).exists() and gt_path(ds, track, seed).exists()


def load_scores(method, ds, track, seed) -> np.ndarray:
    return np.load(scores_path(method, ds, track, seed)).astype(np.float64)


def load_gt(ds, track, seed) -> np.ndarray:
    return np.load(gt_path(ds, track, seed)).astype(np.float64)


_MASKS: dict[str, list] = {}


def load_masks(ds: str) -> list:
    if ds not in _MASKS:
        _MASKS[ds] = pickle.load(open(DATA / "subsets" / f"{ds}_masks.pkl", "rb"))
    return _MASKS[ds]






# --------------------------------------------------------------------------- #
# the evaluator (single code path)
# --------------------------------------------------------------------------- #
def lds_per_query(mat: np.ndarray, gt: np.ndarray, masks: list) -> tuple[np.ndarray, float]:
    """Per-query Spearman between subset losses and predicted influence, and the mean.

    ``predicted_influence`` sums the scores of the DELETED rows of each subset
    (``(1-mask) @ scores``); ``compute_lds`` takes Spearman per query column.
    Uses the shared LDS numerical kernel.
    """
    idx = list(range(min(gt.shape[0], len(masks))))
    per_q, mean = compute_lds(gt[: len(idx)], predicted_influence(mat, masks, idx))
    return np.asarray(per_q, dtype=np.float64), float(mean)


def lds(mat, gt, masks) -> float:
    return lds_per_query(mat, gt, masks)[1]


class Cell:
    """One (method, dataset, track, seed) score matrix with its GT and masks."""

    def __init__(self, method, ds, track, seed):
        self.method, self.ds, self.track, self.seed = method, ds, track, seed
        self.a = load_scores(method, ds, track, seed)
        self.gt = load_gt(ds, track, seed)
        self.masks = load_masks(ds)
        self.N, self.Q = self.a.shape

    def lds(self, mat=None) -> float:
        return lds(self.a if mat is None else mat, self.gt, self.masks)

    def per_query(self, mat=None) -> np.ndarray:
        return lds_per_query(self.a if mat is None else mat, self.gt, self.masks)[0]

    def k(self, kappa_pct: float) -> int:
        return int(np.ceil(kappa_pct / 100.0 * self.N))


# --------------------------------------------------------------------------- #
# head / band selections and transforms (all per query, column-wise)
# --------------------------------------------------------------------------- #
def head_mask(a: np.ndarray, k: int, mode: str) -> np.ndarray:
    """Keep k rows per query; ties use ascending training-row index."""
    if mode not in {"abs", "pos"}:
        raise ValueError(f"Unknown head mode: {mode}")
    if not 0 <= k <= a.shape[0]:
        raise ValueError("Head size must be between zero and N")
    key = np.abs(a) if mode == "abs" else a
    mask = np.zeros_like(a, dtype=bool)
    order = np.argsort(-key, axis=0, kind="stable")
    np.put_along_axis(mask, order[:k], True, axis=0)
    return mask


def keep_only(a: np.ndarray, mask: np.ndarray, values=None) -> np.ndarray:
    """``values`` (default τ itself) on the kept rows, 0 elsewhere."""
    return np.where(mask, a if values is None else values, 0.0)


def rank_pct_abs(a: np.ndarray) -> np.ndarray:
    """Per-query rank percentile of |τ| (0 % = largest)."""
    order = np.argsort(-np.abs(a), axis=0, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(a.shape[0])[:, None], order.shape), axis=0)
    return 100.0 * rank / a.shape[0]


def band_mask(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    p = rank_pct_abs(a)
    return (p >= lo) & (p < hi)


def const_tail(a: np.ndarray, head: np.ndarray, head_values: np.ndarray, tail_fn) -> np.ndarray:
    """Head rows keep ``head_values``; every tail row of query q is replaced by
    the per-query mean of ``tail_fn(τ)`` over the tail."""
    tail = ~head
    v = tail_fn(a)
    cnt = np.maximum(tail.sum(axis=0, keepdims=True), 1).astype(np.float64)
    c = np.where(tail, v, 0.0).sum(axis=0, keepdims=True) / cnt
    return np.where(head, head_values, np.broadcast_to(c, a.shape))




# --------------------------------------------------------------------------- #
# statistics and formatting
# --------------------------------------------------------------------------- #
def mean_std(vals) -> tuple[float, float, int]:
    v = np.asarray(list(vals), dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan")), len(v)


def fmt(v: float, nd: int = 4, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def fmt_ms(vals, nd: int = 4, *, single_if_identical: bool = True) -> str:
    """mean±std over seeds; a single value tagged n=1 when there is one seed or
    when all seeds are bit-identical (model-independent val cells)."""
    v = np.asarray(list(vals), dtype=float)
    if len(v) == 1 or (single_if_identical and np.all(v == v[0])):
        return f"{v[0]:.{nd}f}（n=1）"
    return f"{v.mean():.{nd}f}±{v.std(ddof=1):.{nd}f}"


def fmt_ci(per_q, nd: int = 4) -> str:
    m, lo, hi = bootstrap_ci(np.asarray(per_q, dtype=float))
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def md_table(headers, rows, align=None) -> str:
    align = align or ["---"] * len(headers)
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(align) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def write_table(name: str, text: str, title: str = "") -> Path:
    p = TABLES / f"{name}.md"
    p.write_text((f"**{title}**\n\n" if title else "") + text, encoding="utf-8")
    print(f"-> {p}")
    return p


def write_json(name: str, obj) -> Path:
    p = JSONS / f"{name}.json"
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    return p


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
METHOD_COLORS = {"fmas_raw": "#2f6090", "ekfac_if": "#986039", "dtrak_T100": "#537e6b", "das_T100": "#8a6a98"}
#: English labels for figures (DejaVu Sans has no CJK glyphs; tables keep the Chinese labels)
FIG_LABELS = {"fmas_raw": "FMAS", "ekfac_if": "EK-FAC IF", "dtrak_T100": "D-TRAK", "das_T100": "DAS (linear term)",
              "ekfac_mean": "EK-FAC mean readout", "ekfac_msl2": "EK-FAC squared readout", "trak_T100": "TRAK",
              "gas_T100": "GAS", "tracincp_T100": "TracInCP", "relative_if_T100": "Relative IF",
              "renorm_if_T100": "Renorm. IF", "grad_dot_T100": "Gradient dot", "journey_trak_T100": "Journey-TRAK"}


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#66717a", "axes.labelcolor": "#333b43",
                         "xtick.color": "#4c565e", "ytick.color": "#4c565e",
                         "legend.frameon": False, "axes.linewidth": 0.7})
    return plt


def save_fig(fig, name: str):
    fig.savefig(FIGS / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"-> {(FIGS / name)}.svg/.png")
