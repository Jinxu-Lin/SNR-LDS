"""Split the mechanism figure using existing accepted arrays (CPU only).

Run from any directory with the da Python environment. Paper assets are PDFs;
audit data and optional variance plots stay beside this script.
"""
from pathlib import Path
import csv
import json
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from matplotlib.transforms import Bbox

ROOT = Path('/path/to/CFA')
OUT = Path(__file__).resolve().parent
FIG = Path('/path/to/BA-LDS/Paper/Figures/section31')
ARCHIVE = ROOT / 'Paper/_archive/ICLR_overleaf_20260922/Figures/section31'
J = ROOT / 'Codes/Figure/out/json'
METHODS = [('fmas_raw', 'FMAS', '#668394'),
           ('ekfac_if', 'EK-FAC IF', '#B1947B'),
           ('dtrak_T100', 'D-TRAK', '#819985'),
           ('das_T100', 'DAS', '#9B879F')]
SEEDS = [42, 123, 456]
plt.rcParams.update({'font.family': 'serif', 'font.size': 7,
                     'axes.labelsize': 7, 'axes.titlesize': 7.4,
                     'legend.fontsize': 6, 'axes.linewidth': .5,
                     'lines.linewidth': 1.05, 'pdf.fonttype': 42})


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color='#EBEBEB', lw=.4)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, pad=1.5)


def save(fig, name, paper=True):
    fig.savefig((FIG if paper else OUT) / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png', dpi=300)
    plt.close(fig)


# Existing LDS values and membership conventions are unchanged.
head = json.loads((J / 'fig2_head_profile.json').read_text())
power = json.loads((J / 'fig1_power_c2.json').read_text())
paired = json.loads((ARCHIVE / 'square_heads.json').read_text())
fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.65))
fig.subplots_adjust(left=.065, right=.985, bottom=.26, top=.86, wspace=.52)
baselines = {}
for method, label, color in METHODS:
    py = 100*np.asarray([power[f'{method}|gen|{seed}|signed']['lds'][:21] for seed in SEEDS]).mean(axis=0)
    axes[0].plot(np.arange(21)/5, py, color=color)
    axes[0].axhline(py[5], color=color, lw=.65, ls=(0, (3, 2)), alpha=.85)
    axes[0].plot([1], [py[5]], marker='o', color=color, ms=2.0)
    baselines[method] = float(py[5])
    rs = [r for r in paired['records'] if r['method'] == method and r['track'] == 'gen']
    for ax, key, ls in [(axes[1], 'native_head', '-'),
                        (axes[2], 'original_same_head', '-'),
                        (axes[2], 'square_same_head', '--')]:
        y = [100*np.mean([r[key]['mean'] for r in rs if r['head_pct'] == pct]) for pct in paired['head_grid']]
        assert min(y) >= 8 and max(y) <= 55
        ax.plot(paired['head_grid'], y, color=color, ls=ls)
    hy = 100*np.asarray(head[f'{method}|gen|abs']).mean(axis=0)
    axes[3].plot([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], hy,
                 color=color, marker='o', ms=1.7)
    np.testing.assert_allclose(py[5], hy[-1], atol=1e-4)
axes[0].set(xlim=(0, 4), xlabel='Signed power p', title='(a) Signed reweighting')
axes[0].set_xticks([0, 1, 2, 3, 4])
axes[0].annotate('p = 1 baselines', xy=(2.2, 38.7), xytext=(2.45, 25),
                 ha='center', fontsize=5.6,
                 arrowprops=dict(arrowstyle='-', lw=.5, color='#666666'))
axes[1].set(xlim=(5, 100), xlabel='Support head (%)', title='(b) Native aggregation')
axes[1].set_xticks([5, 50, 100])
axes[2].set(xlim=(5, 100), xscale='log', xlabel='Support head (%)', title='(c) Matched-head squaring')
axes[2].set_xticks([5, 10, 20, 50, 100])
axes[2].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
axes[2].minorticks_off()
axes[2].legend(handles=[Line2D([], [], color='#555555', ls='-', label='Original'),
                         Line2D([], [], color='#555555', ls='--', label='Squared')],
                loc='lower left', fontsize=5.3, frameon=False, handlelength=1.8,
                labelspacing=.15, borderpad=.1)
axes[3].set(xlim=(5, 100), xlabel='Magnitude head (%)', title='(d) Magnitude aggregation')
axes[3].set_xticks([5, 20, 50, 100])
for panel, ax in enumerate(axes):
    ax.set(ylim=(8, 55) if panel == 0 else (35, 55), ylabel='LDS (×100)')
    ax.set_yticks([10, 30, 50] if panel == 0 else [35, 40, 45, 50, 55])
    style(ax)
for panel, ax in zip('bcd', axes[1:]):
    values = np.concatenate([line.get_ydata() for line in ax.lines])
    print(f'Panel {panel}: data range {values.min():.2f}–{values.max():.2f}; axis 35–55')
# Export each panel independently.  Use the largest tight bounding box for all
# four panels so that equal-width LaTeX inclusion also gives equal heights.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
boxes = [ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
         for ax in axes]
panel_width = max(box.width for box in boxes)
panel_height = max(box.height for box in boxes)
side_padding = 0.01
bottom_padding = 0.005
top_padding = 0.07
for name, box in zip(['reweighting', 'native', 'matched-square', 'magnitude'], boxes):
    cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    crop = Bbox.from_bounds(cx - panel_width / 2 - side_padding,
                            cy - panel_height / 2 - bottom_padding,
                            panel_width + 2 * side_padding,
                            panel_height + bottom_padding + top_padding)
    fig.savefig(FIG / f'aggregation-{name}.pdf', bbox_inches=crop, pad_inches=0)
save(fig, 'aggregation-row')
