"""Figure 1(a): accepted per-query RMS score densities, generation track."""
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _paths import DATA, METADATA, figure_dir, metadata_input
METHODS = [("fmas_raw", "FMAS", "#668394"), ("ekfac_if", "EK-FAC IF", "#B1947B"),
           ("dtrak_T100", "D-TRAK", "#819985"), ("das_T100", "DAS", "#9B879F")]
SEEDS = [42, 123, 456]
plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.labelsize": 7,
                     "axes.titlesize": 7.4, "legend.fontsize": 6, "axes.linewidth": .5,
                     "lines.linewidth": 1.05, "pdf.fonttype": 42})
settings = json.loads(metadata_input('fig_score_magnitude.json').read_text())
edges = np.linspace(-settings['plot_range_rms'], settings['plot_range_rms'], 8002)
width = edges[1] - edges[0]
grid = (edges[:-1] + edges[1:]) / 2
densities = {}
central_half = {}
for method, label, color in METHODS:
    curves, halfs, normalized = [], [], []
    for seed in SEEDS:
        raw = np.load(DATA / f'scores/{method}/cifar2_5k/seed_{seed}/scores.npy').astype(np.float64)
        x = raw / np.sqrt(np.mean(raw**2, axis=0, keepdims=True))
        normalized.append(abs(x).ravel())
        counts, _ = np.histogram(x.ravel(), bins=edges)
        curves.append(gaussian_filter1d(counts.astype(float), settings['bandwidth_rms']/width,
                                       mode='constant') / (x.size*width))
    densities[method] = np.mean(curves, axis=0)
    central_half[method] = float(np.quantile(np.concatenate(normalized), .5))

fig, ax = plt.subplots(figsize=(1.8, 1.65))
fig.subplots_adjust(left=.23, right=.95, bottom=.26, top=.86)
for method, label, color in METHODS:
    ax.plot(grid, densities[method], color=color, label=label)
bound = central_half['fmas_raw']
ax.axvspan(-bound, bound, color=METHODS[0][2], alpha=.12, lw=0)
for sign in [-1, 1]:
    ax.axvline(sign*bound, color=METHODS[0][2], ls=':', lw=.75)
ymax = max(d.max() for d in densities.values()) * 1.30
ax.annotate('', xy=(-bound, ymax*.85), xytext=(bound, ymax*.85),
            arrowprops=dict(arrowstyle='<->', lw=.6, color='#555555', shrinkA=0, shrinkB=0))
ax.text(0, ymax*.89, '50%', ha='center', va='bottom', fontsize=6)
ax.set(xlim=(-4, 4), ylim=(0, ymax), xlabel='Score / query RMS', ylabel='Density',
       title='(a) Score distribution')
ax.set_xticks([-4, 0, 4])
ax.set_yticks([0, .4, .8])
ax.legend(loc='upper left', frameon=False, fontsize=4.8, handlelength=1.2,
          labelspacing=.25, borderpad=.1, handletextpad=.3)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color="#EBEBEB", lw=.4)
ax.set_axisbelow(True)
ax.tick_params(length=2, pad=1.5)
dest = figure_dir("Fig1") / "motivation-density.pdf"
fig.savefig(dest)
fig.savefig(dest.with_suffix(".png"), dpi=300)
plt.close(fig)
(METADATA / "fig1_density.json").write_text(json.dumps({
    "bandwidth_rms": settings["bandwidth_rms"], "central_half_width": central_half,
    "normalization": "per-query RMS", "track": "gen", "seeds": SEEDS,
}, indent=2) + "\n")
print(f"Saved {dest}")
