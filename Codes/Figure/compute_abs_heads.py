"""Recompute linear and squared LDS on identical absolute-score heads.

This is a readout analysis of existing accepted CIFAR-2 FM scores, deletion
subsets, and measured subset responses.  It does not train or estimate scores.
"""
from pathlib import Path
import json
import sys
import numpy as np

from _common import Cell, CORE, SEEDS, KAPPA_GRID, head_mask, keep_only
from _paths import METADATA
OUT = METADATA
records = []
for method, label in CORE:
    for seed in SEEDS:
        cell = Cell(method, 'cifar2_5k', 'gen', seed)
        for head_pct in KAPPA_GRID:
            mask = head_mask(cell.a, cell.k(head_pct), 'abs')
            linear = cell.per_query(keep_only(cell.a, mask))
            squared = cell.per_query(keep_only(cell.a, mask, cell.a**2))
            records.append({
                'method': method,
                'label': label.replace(' 一次项', ''),
                'seed': seed,
                'head_pct': head_pct,
                'linear_mean': float(linear.mean()),
                'squared_mean': float(squared.mean()),
            })

summary = {}
for method, label in CORE:
    label = label.replace(' 一次项', '')
    summary[method] = {'label': label, 'heads': {}}
    for head_pct in KAPPA_GRID:
        rows = [r for r in records if r['method'] == method and r['head_pct'] == head_pct]
        summary[method]['heads'][str(head_pct)] = {
            readout: {
                'mean_x100': 100 * float(np.mean([r[f'{readout}_mean'] for r in rows])),
                'sample_std_x100': 100 * float(np.std([r[f'{readout}_mean'] for r in rows], ddof=1)),
            }
            for readout in ['linear', 'squared']
        }

payload = {
    'dataset': 'cifar2_5k',
    'track': 'generation',
    'seeds': list(SEEDS),
    'head_grid_pct': list(KAPPA_GRID),
    'selection': 'per-query largest absolute pre-square scores; ceil(kappa*N), stable training-index ties',
    'linear_readout': 'selected pre-square scores with original signs',
    'squared_readout': 'ordinary square of the same selected scores',
    'aggregation': 'deleted-row score sum; per-query Spearman; equal mean over model seeds',
    'records': records,
    'summary': summary,
}
(OUT / 'abs_head_readouts.json').write_text(json.dumps(payload, indent=2) + '\n')

for method, values in summary.items():
    print(values['label'])
    for head_pct in [5, 20, 50, 100]:
        h = values['heads'][str(head_pct)]
        print(head_pct, f"linear={h['linear']['mean_x100']:.2f}",
              f"squared={h['squared']['mean_x100']:.2f}")
