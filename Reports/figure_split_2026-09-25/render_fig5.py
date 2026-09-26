"""Render Appendix Figure 5 with the manuscript method palette."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


CFA = Path("/path/to/CFA")
PAPER = Path("/path/to/BA-LDS/Paper")
SOURCE = CFA / "Codes/Figure/out/json/fig2_head_profile.json"
DEST = PAPER / "Figures/Fig5"

METHODS = (
    ("fmas_raw", "FMAS", "#668394"),
    ("ekfac_if", "EK-FAC IF", "#B1947B"),
    ("dtrak_T100", "D-TRAK", "#819985"),
    ("das_T100", "DAS (linear)", "#9B879F"),
)
HEADS = (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

data = json.loads(SOURCE.read_text())
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.35,
    "pdf.fonttype": 42,
})

for track in ("gen", "val"):
    fig, ax = plt.subplots(figsize=(2.55, 1.25))
    fig.subplots_adjust(left=0.25, right=0.93, bottom=0.36, top=0.95)
    for method, _, color in METHODS:
        values = np.asarray(data[f"{method}|{track}|abs"], dtype=float)
        ax.plot(HEADS, 100 * values.mean(axis=0), color=color, marker="o", markersize=2)
    ax.set(xlim=(5, 100), xlabel="Retained head (%)", ylabel="LDS ($\\times100$)")
    ax.set_xticks([5, 50, 100])
    ax.set_yticks([38, 42, 46, 50] if track == "gen" else [42, 46, 50, 54])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#E4E8EC", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.savefig(DEST / f"head-{track}.pdf")
    plt.close(fig)

fig = plt.figure(figsize=(7.7, 0.30))
handles = [Line2D([], [], color=color, label=label) for _, label, color in METHODS]
fig.legend(handles=handles, loc="center", ncol=4, frameon=False,
           fontsize=10, handlelength=2.0)
fig.savefig(DEST / "method-legend.pdf")
plt.close(fig)
print(f"Saved Figure 5 assets to {DEST}")
