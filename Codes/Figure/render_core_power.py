"""Render only the retained four-method power controls from existing results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _paths import figure_dir, metadata_input
SOURCE = metadata_input("fig1_power_c2.json")
DEST = figure_dir("Fig6")
data = json.loads(SOURCE.read_text())
methods = [('fmas_raw', 'FMAS'), ('ekfac_if', 'EK-FAC IF'),
           ('dtrak_T100', 'D-TRAK'), ('das_T100', 'DAS')]
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'pdf.fonttype': 42})
fig, axes = plt.subplots(2, 4, figsize=(10, 4.6), sharex=True, sharey=True,
                         constrained_layout=True)
for row, track in enumerate(['gen', 'val']):
    for col, (method, label) in enumerate(methods):
        ax = axes[row, col]
        for family, color, style in [('signed', '#668394', '-'),
                                      ('fold', '#B1947B', '--')]:
            key_family = family
            if f'{method}|{track}|42|{key_family}' not in data:
                key_family = 'folded' if family == 'fold' else family
            values = np.asarray([data[f'{method}|{track}|{seed}|{key_family}']['lds'][:21]
                                 for seed in [42, 123, 456]]) * 100
            assert values.shape == (3, 21) and np.isfinite(values).all()
            ax.plot(np.arange(21)/5, values.mean(axis=0), style, color=color,
                    label='Sign-preserving' if family == 'signed' else 'Folded')
        ax.set_title(label)
        ax.set_xlim(0, 4)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.axvline(1, color='0.7', linewidth=.6)
        ax.grid(alpha=.2)
        if col == 0:
            ax.set_ylabel(('Generation' if track == 'gen' else 'Validation') + '\nLDS (×100)')
        if row == 1:
            ax.set_xlabel('Power p')
axes[0, 0].legend(fontsize=7)
fig.savefig(DEST / 'core-power.pdf')
fig.savefig(DEST / 'core-power.png', dpi=140)

plt.close(fig)
print(f"Saved {DEST}")
