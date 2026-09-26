"""Render Figure 1(b) as bin-level relative sampling variation."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

CFA = Path("/path/to/CFA")
PAPER = Path("/path/to/BA-LDS/Paper")
OUT = Path(__file__).resolve().parent
INDEX = CFA / "_Data/reports/narrative_gpu_wave_2026-09-21/VERIFY_M2A_R16_V1.json"
PDF = PAPER / "Figures/Fig1/motivation-repeatability.pdf"

index = json.loads(INDEX.read_text())["score_index"]
paths = sorted(
    (entry for entry in index if entry["method"] == "fmas_raw" and entry["track"] == "gen"),
    key=lambda entry: entry["repeat"],
)
assert [entry["repeat"] for entry in paths] == list(range(16))
scores = np.stack([np.load(entry["path"]).astype(np.float64) for entry in paths])
assert scores.shape == (16, 5000, 16) and np.isfinite(scores).all()

mean = scores.mean(axis=0)
variance = scores.var(axis=0, ddof=1)
order = np.argsort(np.abs(mean), axis=0, kind="stable")
query_ids = np.arange(mean.shape[1])[None, :]
records = []
for band in range(10):
    ids = order[500 * band : 500 * (band + 1)]
    ratios = variance[ids, query_ids].sum(axis=0) / np.square(mean[ids, query_ids]).sum(axis=0)
    records.append(
        {
            "rank_start": 10 * band,
            "rank_end": 10 * (band + 1),
            "queries": 16,
            "samples_per_query": 500,
            "varratio_mean": float(ratios.mean()),
            "varratio_median": float(np.median(ratios)),
            "varratio_min": float(ratios.min()),
            "varratio_max": float(ratios.max()),
        }
    )

with (OUT / "fmas_varratio_deciles.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

# Recompute the pilot-defined appendix bands from the accepted coordinate files.
analysis = CFA / "_Data/results/narrative_wave_20260921/analysis/m2a_r16_20260921"
pilot_records = []
for method in ("fmas_raw", "dtrak_T100"):
    for track in ("gen", "val"):
        stats = torch.load(analysis / method / track / "coordinate_stats.pt", weights_only=False)
        mu = np.asarray(stats["mean"], dtype=np.float64)
        var = np.square(np.asarray(stats["sample_sd"], dtype=np.float64))
        bands = np.asarray(stats["band"])
        for band in range(5):
            per_query = []
            for query in range(mu.shape[1]):
                selected = bands[:, query] == band
                per_query.append(var[selected, query].sum() / np.square(mu[selected, query]).sum())
            pilot_records.append({
                "method": method, "track": track, "band": band,
                "varratio_mean": float(np.mean(per_query)),
            })
with (OUT / "pilot_band_varratio.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=pilot_records[0].keys())
    writer.writeheader()
    writer.writerows(pilot_records)

plt.rcParams.update({
    "font.family": "serif", "font.size": 7, "axes.labelsize": 7,
    "axes.titlesize": 7.4, "axes.linewidth": 0.5, "pdf.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(1.8, 1.65))
fig.subplots_adjust(left=0.23, right=0.91, bottom=0.27, top=0.86)
ax.bar([5 + 10 * i for i in range(10)], [r["varratio_mean"] for r in records],
       width=8.5, color="#668394", alpha=0.85, linewidth=0)
ax.set(xlim=(0, 100), yscale="log", ylim=(0.03, 30),
       xlabel=r"Score rank (low $\rightarrow$ high)",
       ylabel="Relative variation", title=r"(b) Sampling variation $\downarrow$")
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_yticks([0.03, 0.1, 1, 10, 30], labels=[".03", ".1", "1", "10", "30"])
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#EBEBEB", lw=0.4, which="major")
ax.set_axisbelow(True)
ax.tick_params(length=2, pad=1.5)
fig.savefig(PDF)
fig.savefig(PDF.with_suffix(".png"), dpi=300)
plt.close(fig)

print(f"Saved {PDF}")
print("VarRatio means:", [round(r["varratio_mean"], 6) for r in records])
