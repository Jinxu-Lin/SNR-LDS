"""Render the Figure 3(a) table from accepted zero-mean Gaussian SNR selections.

The script reuses the accepted A3 zeta=3 masks, filed CIFAR-2 FM scores,
deletion subsets, and measured responses.  It performs no fitting, scoring, or
model execution.  For every method/query, the SNR mask is fixed while the
absolute-score head and readout are varied.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys

import numpy as np


import argparse
import os

CODE = Path(__file__).resolve().parents[1]
from balds.evaluation.lds import compute_lds

DATA = Path(os.environ.get('BALDS_DATA_ROOT', str(CODE.parent / '_Data')))
REPORT = DATA / 'results/paper/figures/metadata'
A3 = DATA / 'results/snr_lds_20260925/a3/panels'
OUT = DATA / 'results/paper/figures/Fig3'


SEEDS = (42, 123, 456)
HEADS = (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
METHODS = (
    ("fmas_raw", "FMAS", "#668394"),
    ("ekfac_if", "EK-FAC IF", "#B1947B"),
    ("dtrak_T100", "D-TRAK", "#819985"),
    ("das_native_sq", "DAS", "#9B879F"),
)


def absolute_head(scores: np.ndarray, percentage: int) -> np.ndarray:
    """Largest absolute pre-square scores per query, with stable tie order."""
    count = int(np.ceil(len(scores) * percentage / 100.0))
    order = np.argsort(-np.abs(scores), axis=0, kind="stable")
    selected = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(selected, order[:count], True, axis=0)
    return selected


def per_query_lds(values: np.ndarray, deletion: np.ndarray,
                  response: np.ndarray) -> np.ndarray:
    return compute_lds(response, deletion @ values)[0]


def load_seed(method: str, seed: int):
    filed = "das_T100" if method == "das_native_sq" else method
    raw = np.load(DATA / f"scores/{filed}/cifar2_5k/seed_{seed}/scores.npy").astype(np.float64)
    panel = A3 / f"cifar2_5k_s{seed}_gen" / method
    accepted = np.load(panel / "predictions.npz")
    mask = accepted["selected"].astype(bool)
    rows = json.loads((panel / "per_query.json").read_text())
    valid = np.array([row["snr_lds"] is not None for row in rows], dtype=bool)
    if raw.shape != mask.shape or raw.shape[1] != len(valid):
        raise ValueError(f"score/mask/query shape mismatch: {method}/{seed}")
    if not valid.any():
        raise ValueError(f"no valid queries: {method}/{seed}")
    return raw, mask, valid, rows


def main() -> None:
    global DATA, REPORT, A3, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA)
    parser.add_argument('--results', type=Path, help='accepted SNR panels; defaults to DATA/results/snr_lds_20260925/a3/panels')
    parser.add_argument('--output', type=Path, help='figure root; defaults to BALDS_FIGURE_ROOT or DATA/results/paper/figures')
    args = parser.parse_args()
    DATA = args.data_root.resolve()
    figure_root = args.output or Path(os.environ.get('BALDS_FIGURE_ROOT', str(DATA / 'results/paper/figures')))
    OUT, REPORT = figure_root / 'Fig3', figure_root / 'metadata'
    REPORT.mkdir(parents=True, exist_ok=True)
    A3 = args.results or DATA / 'results/snr_lds_20260925/a3/panels'
    with (DATA / "subsets/cifar2_5k_masks.pkl").open("rb") as stream:
        keep = np.asarray(pickle.load(stream), dtype=np.float64)[:64]
    deletion = 1.0 - keep
    records = []
    checks = []

    for method, label, _ in METHODS:
        for seed in SEEDS:
            response = np.load(DATA / f"results/gt_matrix_fm_cifar2_5k_seed_{seed}.npy").astype(np.float64)[:64, :100]
            raw, snr_mask, valid, rows = load_seed(method, seed)
            if raw.shape != (deletion.shape[1], response.shape[1]):
                raise ValueError(f"input alignment differs: {method}/{seed}")

            linear = raw
            squared = raw ** 2
            native = squared if method == "das_native_sq" else linear
            filed_snr = np.array([row["snr_lds"] if row["snr_lds"] is not None else np.nan for row in rows])
            recomputed = per_query_lds(np.where(snr_mask, native, 0.0), deletion, response)
            np.testing.assert_allclose(recomputed[valid], filed_snr[valid], rtol=0, atol=1e-12)
            checks.append({"method": method, "seed": seed, "n_valid": int(valid.sum()),
                           "filed_mean_x100": 100 * float(np.nanmean(filed_snr)),
                           "recomputed_mean_x100": 100 * float(recomputed[valid].mean())})

            for percentage in HEADS:
                support = snr_mask & absolute_head(raw, percentage)
                for readout, values in (("native", native), ("linear", linear),
                                        ("squared", squared)):
                    lds = per_query_lds(np.where(support, values, 0.0), deletion, response)
                    records.append({
                        "method": method,
                        "label": label,
                        "seed": seed,
                        "head_pct": percentage,
                        "readout": readout,
                        "n_valid": int(valid.sum()),
                        "mean_x100": 100 * float(lds[valid].mean()),
                        "per_query_x100": (100 * lds[valid]).tolist(),
                    })

    def average(method: str, percentage: int, readout: str) -> float:
        values = [row["mean_x100"] for row in records
                  if row["method"] == method and row["head_pct"] == percentage
                  and row["readout"] == readout]
        if len(values) != len(SEEDS):
            raise ValueError(f"incomplete seed coverage: {method}/{percentage}/{readout}")
        return float(np.mean(values))

    summary = {}
    for method, label, _ in METHODS:
        summary[method] = {
            "label": label,
            "native": {str(p): average(method, p, "native") for p in HEADS},
            "linear": {str(p): average(method, p, "linear") for p in HEADS},
            "squared": {str(p): average(method, p, "squared") for p in HEADS},
        }

    payload = {
        "rule": "snr_zero_mean_gaussian_v1",
        "zeta": 3.0,
        "dataset": "cifar2_5k",
        "track": "generation",
        "seeds": list(SEEDS),
        "head_grid_pct": list(HEADS),
        "selection": "accepted zeta=3 SNR mask intersected with per-query absolute pre-square score heads",
        "panel_a": "linear readouts on the fixed SNR/head intersection",
        "panel_b": "ordinary-squared readouts on the identical fixed SNR/head intersection",
        "aggregation": "per-query Spearman correlation; valid queries averaged within seed, then seed means averaged equally",
        "accepted_mask_checks": checks,
        "records": records,
        "summary": summary,
    }
    REPORT.joinpath("fig3_snr_controls.json").write_text(json.dumps(payload, indent=2) + "\n")

    OUT.mkdir(parents=True, exist_ok=True)
    columns = (5, 20, 50, 100)
    lines = [
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}",
        r"\toprule",
        r"Readout & 5\% & 20\% & 50\% & 100\% \\",
        r"\midrule",
    ]
    for index, (method, label, _) in enumerate(METHODS):
        for readout, suffix in (("linear", "Lin."), ("squared", "Sq.")):
            values = " & ".join(f"{average(method, p, readout):.2f}" for p in columns)
            lines.append(f"{label} {suffix} & {values} " + r"\\")
        if index + 1 < len(METHODS):
            lines.append(r"\addlinespace[1pt]")
    lines.extend((r"\bottomrule", r"\end{tabular*}"))
    (OUT / "snr-readouts.tex").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "status": "produced",
        "output": str(OUT / "snr-readouts.tex"),
        "endpoints": {
            method: {
                "native_5": average(method, 5, "native"),
                "native_100": average(method, 100, "native"),
                "square_minus_linear_100": average(method, 100, "squared") - average(method, 100, "linear"),
            } for method, _, _ in METHODS
        },
    }, indent=2))


if __name__ == "__main__":
    main()
