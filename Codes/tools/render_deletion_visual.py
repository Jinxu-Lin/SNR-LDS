"""Render final Figure 3(b) from native and five-set random query measurements.

L2 uses the filed uint8/255 pixels. CLIP is cosine similarity, not distance.
This entry recomputes table means; it does not regenerate images or embeddings.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def summarize(data):
    roots = {name: data / f'results/{name}_20260922' for name in
             ('fmas_das_visual', 'fmas_das_random_visual')}
    with (roots['fmas_das_visual'] / 'per_query.tsv').open() as stream:
        native = list(csv.DictReader(stream, delimiter='\t'))
    with (roots['fmas_das_random_visual'] / 'random_per_set_query.tsv').open() as stream:
        random = list(csv.DictReader(stream, delimiter='\t'))
    by_native = {(r['method'], int(r['k']), int(r['query_id'])): r for r in native}
    by_random = {(int(r['k']), int(r['random_set']), int(r['query_id'])): r for r in random}
    methods = ('das_native_sq', 'fmas_raw')
    if len(by_native) != len(native) or set(by_native) != {
        (m, k, q) for m in methods for k in (300, 1000) for q in range(50)
    }:
        raise ValueError('native visual measurements must cover two methods, two budgets, 50 queries')
    if len(by_random) != len(random) or set(by_random) != {
        (k, s, q) for k in (300, 1000) for s in range(5) for q in range(50)
    }:
        raise ValueError('random measurements must cover five models per budget and 50 queries')
    rows = []
    for k in (300, 1000):
        for metric in ('loss_increase', 'pixel_l2', 'clip_cosine'):
            values = {m: np.array([float(by_native[m, k, q][metric]) for q in range(50)])
                      for m in methods}
            values['random'] = np.array([
                np.mean([float(by_random[k, s, q][metric]) for s in range(5)])
                for q in range(50)])
            if not all(np.isfinite(v).all() for v in values.values()):
                raise ValueError('nonfinite visual metric')
            rows.append(dict(k=k, metric=metric, **{m: float(v.mean()) for m, v in values.items()}))
    return dict(rows=rows, query_ids=list(range(50)), random_sets=5,
                aggregation='average random models within query, then average the same 50 queries',
                pixel_definition='Euclidean distance on flattened common uint8/255 pixels',
                clip_definition='cosine similarity of the filed CLIP image embeddings')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[2] / '_Data'
    parser.add_argument('--data-root', type=Path, default=os.environ.get('BALDS_DATA_ROOT', str(default)))
    parser.add_argument('--output', type=Path, help='figure root; defaults to BALDS_FIGURE_ROOT or DATA/results/paper/figures')
    args = parser.parse_args()
    output = args.output or Path(os.environ.get('BALDS_FIGURE_ROOT', str(args.data_root/'results/paper/figures')))
    result = summarize(args.data_root)
    rows = result['rows']
    lines = [r'\begin{tabular}{@{}rlccc@{}}', r'\toprule',
             r'Deleted & Metric & Random & DAS & FMAS \\', r'\midrule']
    labels = {'loss_increase': r'$\Delta$ loss ($\times100$)',
              'pixel_l2': 'L2 distance', 'clip_cosine': 'CLIP Similarity'}
    for k in (300, 1000):
        if k == 1000:
            lines.append(r'\midrule')
        for row in [r for r in rows if r['k'] == k]:
            metric = row['metric']
            scale, digits = (100, 2) if metric == 'loss_increase' else (1, 4 if metric == 'clip_cosine' else 2)
            values = ' & '.join(f'{row[m]*scale:.{digits}f}' for m in ('random', 'das_native_sq', 'fmas_raw'))
            budget = f'{k:,}' if metric == 'loss_increase' else ''
            lines.append(f'{budget} & {labels[metric]} & {values} ' + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (output/'Fig3').mkdir(parents=True, exist_ok=True)
    (output/'metadata').mkdir(parents=True, exist_ok=True)
    (output/'Fig3/deletion_visual.tex').write_text('\n'.join(lines)+'\n')
    (output/'metadata/deletion_visual.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(output/'Fig3/deletion_visual.tex'), 'cells': 18}))


if __name__ == '__main__':
    main()
