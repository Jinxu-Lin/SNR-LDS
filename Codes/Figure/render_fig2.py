"""Render the two Figure 2 curves and its matched absolute-head table."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox

from _paths import METADATA, figure_dir, metadata_input
REPORT = METADATA
DEST = figure_dir("Fig2")
POWER = json.loads(metadata_input("fig1_power_c2.json").read_text())
OLD_HEAD = json.loads(metadata_input("fig2_head_profile.json").read_text())
HEAD = json.loads(metadata_input("abs_head_readouts.json").read_text())
METHODS = [('fmas_raw', 'FMAS', '#668394'),
           ('ekfac_if', 'EK-FAC IF', '#B1947B'),
           ('dtrak_T100', 'D-TRAK', '#819985'),
           ('das_T100', 'DAS', '#9B879F')]
SEEDS = [42, 123, 456]
GRID = HEAD['head_grid_pct']

plt.rcParams.update({'font.family': 'serif', 'font.size': 7,
                     'axes.labelsize': 7, 'axes.titlesize': 7.4,
                     'legend.fontsize': 6, 'axes.linewidth': .5,
                     'lines.linewidth': 1.05, 'pdf.fonttype': 42})


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color='#EBEBEB', lw=.4)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, pad=1.5)


fig, axes = plt.subplots(1, 2, figsize=(3.6, 1.65))
fig.subplots_adjust(left=.10, right=.985, bottom=.26, top=.84, wspace=.58)
baselines = {}
for method, label, color in METHODS:
    h = HEAD['summary'][method]['heads']
    linear = np.asarray([h[str(p)]['linear']['mean_x100'] for p in GRID])
    squared = np.asarray([h[str(p)]['squared']['mean_x100'] for p in GRID])
    # Existing magnitude-head linear values provide an independent equality check.
    np.testing.assert_allclose(
        linear,
        100*np.asarray(OLD_HEAD[f'{method}|gen|abs']).mean(axis=0),
        rtol=0, atol=5e-7,
    )
    native = squared if method == 'das_T100' else linear
    axes[0].plot(GRID, native, color=color, marker='o', ms=1.7)

    py = 100*np.asarray([POWER[f'{method}|gen|{seed}|signed']['lds'][:21]
                         for seed in SEEDS]).mean(axis=0)
    axes[1].plot(np.arange(21)/5, py, color=color)
    axes[1].axhline(py[5], color=color, lw=.65, ls=(0, (3, 2)), alpha=.85)
    axes[1].plot([1], [py[5]], marker='o', color=color, ms=2.0)
    baselines[method] = float(py[5])

axes[0].set(xlim=(100, 5), xscale='log', ylim=(35, 52),
            xlabel='Scores retained (%)', ylabel='LDS (×100)',
            title='(a) Hard truncation')
axes[0].set_xticks([100, 50, 20, 5])
axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
axes[0].minorticks_off()
axes[0].set_yticks([35, 40, 45, 50])

axes[1].set(xlim=(0, 4), ylim=(8, 55),
            xlabel='Signed power p', ylabel='LDS (×100)',
            title='(b) Power reweighting')
axes[1].set_xticks([0, 1, 2, 3, 4])
axes[1].set_yticks([10, 30, 50])
axes[1].annotate('p = 1', xy=(1, 38.7), xytext=(.35, 25),
                 fontsize=5.6, ha='center',
                 arrowprops=dict(arrowstyle='-', lw=.5, color='#666666'))
for ax in axes:
    style(ax)

# Separate, equally sized tight PDFs for compact TeX composition.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
boxes = [ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
         for ax in axes]
w, h = max(b.width for b in boxes), max(b.height for b in boxes)
for name, box in zip(['hard-truncation', 'power'], boxes):
    cx, cy = (box.x0 + box.x1)/2, (box.y0 + box.y1)/2
    crop = Bbox.from_bounds(cx-w/2-.01, cy-h/2-.005,
                            w+.02, h+.075)
    fig.savefig(DEST / f'aggregation-{name}.pdf', bbox_inches=crop, pad_inches=0)
plt.close(fig)

# Compact eight-row table: all four methods are retained because it fits the
# same panel height and directly shows how the squaring gain collapses at 5%.
selected = [5, 20, 50, 100]
lines = [
    r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}',
    r'\toprule',
    r'Readout & 5\% & 20\% & 50\% & 100\% \\',
    r'\midrule',
]
for mi, (method, label, _) in enumerate(METHODS):
    for readout, suffix in [('linear', 'Lin.'), ('squared', 'Sq.')]:
        vals = [HEAD['summary'][method]['heads'][str(p)][readout]['mean_x100']
                for p in selected]
        lines.append(' & '.join([f'{label} {suffix}'] + [f'{v:.2f}' for v in vals]) + r' \\')
    if mi != len(METHODS)-1:
        lines.append(r'\addlinespace[1pt]')
lines += [r'\bottomrule', r'\end{tabular*}']
(DEST / 'absolute-head-readouts.tex').write_text('\n'.join(lines) + '\n')

(REPORT / 'figure2_sources.json').write_text(json.dumps({
    'source': str(REPORT / 'abs_head_readouts.json'),
    'hard_truncation': 'absolute pre-square head; linear readout except native squared DAS',
    'power': 'sign-preserving power of pre-square scores for all methods',
    'table': 'same absolute pre-square head for linear and ordinary-squared readouts',
    'power_p1_x100': baselines,
}, indent=2) + '\n')
print('Rendered Figure 2 panels and eight-row table.')
