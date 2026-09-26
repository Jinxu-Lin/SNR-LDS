"""Render Appendix Figure 4 using the linear-density style of Figure 1(a)."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


from _paths import DATA, METADATA, data_path, figure_dir, metadata_input
SETTINGS = json.loads(
    (metadata_input("fig_score_magnitude.json")).read_text()
)
DEST = figure_dir("Fig4") / "density-all.pdf"

METHODS = (
    ("fmas_raw", "FMAS", "#668394"),
    ("ekfac_if", "EK-FAC IF", "#B1947B"),
    ("dtrak_T100", "D-TRAK", "#819985"),
    ("das_T100", "DAS", "#9B879F"),
)
SEEDS = (42, 123, 456)
TRACKS = (("gen", "Generation queries"), ("val", "Validation queries"))

edges = np.linspace(-SETTINGS["plot_range_rms"], SETTINGS["plot_range_rms"], 8002)
width = edges[1] - edges[0]
grid = (edges[:-1] + edges[1:]) / 2


def score_path(method, track, seed):
    dataset = "cifar2_5k" if track == "gen" else "cifar2_5k_val"
    return DATA / f"scores/{method}/{dataset}/seed_{seed}/scores.npy"


densities = {}
central_half = {}
for track, _ in TRACKS:
    for method, _, _ in METHODS:
        curves = []
        pooled_abs = []
        for seed in SEEDS:
            raw = np.load(score_path(method, track, seed)).astype(np.float64)
            scores = raw / np.sqrt(np.mean(raw ** 2, axis=0, keepdims=True))
            pooled_abs.append(np.abs(scores).ravel())
            counts, _ = np.histogram(scores.ravel(), bins=edges)
            curves.append(
                gaussian_filter1d(
                    counts.astype(float),
                    SETTINGS["bandwidth_rms"] / width,
                    mode="constant",
                )
                / (scores.size * width)
            )
        densities[(track, method)] = np.asarray(curves)
        central_half[(track, method)] = float(np.quantile(np.concatenate(pooled_abs), 0.5))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.55,
    "lines.linewidth": 1.1,
    "pdf.fonttype": 42,
})

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.05), sharey=True)
fig.subplots_adjust(left=0.075, right=0.99, bottom=0.25, top=0.86, wspace=0.16)
peak = max(values.mean(axis=0).max() for values in densities.values())
ymax = peak * 1.30

for panel, (ax, (track, title)) in enumerate(zip(axes, TRACKS)):
    for method, label, color in METHODS:
        values = densities[(track, method)]
        mean = values.mean(axis=0)
        ax.fill_between(grid, values.min(axis=0), values.max(axis=0),
                        color=color, alpha=0.10, linewidth=0)
        ax.plot(grid, mean, color=color, label=label)

    bound = central_half[(track, "fmas_raw")]
    ax.axvspan(-bound, bound, color=METHODS[0][2], alpha=0.10, linewidth=0)
    for sign in (-1, 1):
        ax.axvline(sign * bound, color=METHODS[0][2], linestyle=":", linewidth=0.8)
    ax.annotate("", xy=(-bound, ymax * 0.84), xytext=(bound, ymax * 0.84),
                arrowprops=dict(arrowstyle="<->", linewidth=0.65, color="#555555",
                                shrinkA=0, shrinkB=0))
    ax.text(0, ymax * 0.88, "50%", ha="center", va="bottom", fontsize=7)

    ax.set(xlim=(-4, 4), ylim=(0, ymax), xlabel="Score / query RMS",
           title=f"({chr(ord('a') + panel)}) {title}")
    ax.set_xticks([-4, 0, 4])
    ax.set_yticks([0, 0.4, 0.8])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#EBEBEB", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, pad=1.5)

axes[0].set_ylabel("Density")
axes[0].legend(loc="upper left", frameon=False, handlelength=1.4,
               labelspacing=0.25, borderpad=0.1, handletextpad=0.35)

fig.savefig(DEST)
fig.savefig(DEST.with_suffix(".png"), dpi=300)
plt.close(fig)
print(f"Saved {DEST}")
