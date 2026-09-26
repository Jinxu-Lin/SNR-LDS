"""Render final Tables 12/13 from measured native utilities and global selection."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data = Path(os.environ.get('BALDS_DATA_ROOT', str(Path(__file__).resolve().parents[2]/'_Data')))
    parser.add_argument('--utilities', type=Path, default=data/'results/lds_nextwave_20260920/j3/a1/e2/e2_per_query.tsv')
    parser.add_argument('--selection', type=Path, default=data/'results/paper/snr/edel/edel_selection.json')
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('BALDS_FIGURE_ROOT', str(data/'results/paper/figures'))))
    args = parser.parse_args()
    with args.utilities.open() as stream:
        rows = [r for r in csv.DictReader(stream, delimiter='\t') if r['transform'] == 'native']
    by = {(r['method'], int(r['k']), int(r['query'])): float(r['utility']) for r in rows}
    methods = [('fmas_raw', 'FMAS'), ('dtrak_T100', 'D-TRAK'), ('das_native_sq', 'DAS'), ('ekfac_if', 'EK-FAC IF')]
    required = {(m, k, q) for m, _ in methods for k in (300, 1000) for q in range(50)}
    if len(by) != len(rows) or set(by) != required or not np.isfinite(list(by.values())).all():
        raise ValueError('expected 400 unique finite native utility measurements')
    draws = np.random.default_rng(20260920).integers(0, 50, (2000, 50))
    lines = [r'\begin{table}[t]', r'\centering\small', r'\begin{tabular}{lrr}', r'\toprule',
             r'Method & Delete 300 & Delete 1,000 \\', r'\midrule']
    for method, label in methods:
        cells = []
        for k in (300, 1000):
            values = np.array([by[method, k, q] for q in range(50)])
            lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975])
            cells.append(f'{values.mean():.6f} [{lo:.6f}, {hi:.6f}]')
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{Mean query-loss increases on 50 CIFAR-2 generation queries, with pointwise 95\% query-bootstrap intervals. All methods use their native support rankings.}',
              r'\label{tab:app-delete-existing}', r'\end{table}']
    target = args.output/'Table12'
    target.mkdir(parents=True, exist_ok=True)
    (target/'table.tex').write_text('\n'.join(lines)+'\n')
    selection = json.loads(args.selection.read_text())
    if selection['status'] != 'ok':
        raise ValueError('global method selection is unavailable')
    names = dict(methods)
    choices = selection['choices']
    labels = {key: ', '.join(names[m] for m in choices[key]) for key in ('full', 'snr')}
    lines = [r'\begin{table}[t]', r'\centering\small', r'\begin{tabular}{rrr}', r'\toprule',
             f'Deleted & LDS ({labels["full"]}) & SNR-LDS ({labels["snr"]}) '+r'\\', r'\midrule']
    for row in selection['summary']:
        lines.append(f'{row["k"]:,} & {100*row["full_utility"]:.2f} & {100*row["snr_utility"]:.2f} '+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{Evaluator-selected mean query-loss increase ($\times100$), using one global choice per evaluator and native deletion rankings on the same 50 queries.}',
              r'\label{tab:exp-selection}', r'\end{table}']
    target = args.output/'Table13'
    target.mkdir(parents=True, exist_ok=True)
    (target/'table.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output': str(args.output), 'tables': [12, 13], 'choices': choices}))


if __name__ == '__main__':
    main()
