"""Render Figure 1(b) from the verified R=16 rank-decile summary."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/path/to/CFA')
TABLE = ROOT / '_Data/reports/author_fig1_split_2026-09-25/rank_deciles.csv'
DEST = Path('/path/to/BA-LDS/Paper/Figures/Fig1/motivation-repeatability.pdf')

with TABLE.open() as f:
    rows = [r for r in csv.DictReader(f) if r['method'] == 'fmas_raw']

# Stored rank 0 is the largest-magnitude decile.  Display in reverse so that
# influence magnitude increases from left to right, matching Figure 1(a).
values = [float(r['repeatability_mean_query_median']) for r in reversed(rows)]

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 7,
    'axes.labelsize': 7, 'axes.titlesize': 7.4,
    'axes.linewidth': .5, 'pdf.fonttype': 42,
})
fig, ax = plt.subplots(figsize=(1.8, 1.65))
fig.subplots_adjust(left=.21, right=.91, bottom=.27, top=.84)
ax.bar([5 + 10*i for i in range(10)], values, width=8.5,
       color='#668394', alpha=.85, linewidth=0)
ax.set(xlim=(0, 100), ylim=(0, 6),
       xlabel=r'Score rank (low $\rightarrow$ high)',
       ylabel='Repeatability',
       title='(b) FMAS repeatability ↑')
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_yticks([0, 2, 4, 6])
ax.spines[['top', 'right']].set_visible(False)
ax.grid(color='#EBEBEB', lw=.4)
ax.set_axisbelow(True)
ax.tick_params(length=2, pad=1.5)
fig.savefig(DEST)
fig.savefig(DEST.with_suffix('.png'), dpi=300)
plt.close(fig)
print(f'Saved {DEST}')
