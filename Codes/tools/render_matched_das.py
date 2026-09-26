"""Render final Tables 8/9 from the matched DAS transform analysis."""
import argparse
import json
import os
from pathlib import Path


def main():
    data = Path(os.environ.get('BALDS_DATA_ROOT', str(Path(__file__).resolve().parents[2]/'_Data')))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=data/'results/paper/matched_das')
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('BALDS_FIGURE_ROOT', str(data/'results/paper/figures'))))
    args = parser.parse_args()
    if json.loads((args.input/'missing.json').read_text()):
        raise ValueError('matched DAS analysis has missing inputs')
    key = lambda r: (r['dataset'], r['track'], r['transform'], r['rule'])
    rows = {key(r): r for r in json.loads((args.input/'summaries.json').read_text())}
    gains = {key(r): r for r in json.loads((args.input/'paired_differences.json').read_text())}
    lines8 = [r'\begin{table}[t]', r'\centering\small', r'\begin{tabular}{llrrr}', r'\toprule',
              r'Panel & Readout & Full LDS & LDS@$5\%$ & Fixed-head LDS@$5\%$ \\', r'\midrule']
    lines9 = [r'\begin{table}[t]', r'\centering\footnotesize', r'\begin{tabular}{llrrr}', r'\toprule',
              r'Panel & Transformation & Full gain & Head@$5\%$ gain & Fixed-head gain \\', r'\midrule']
    rules = ('full', 'head_5', 'fixed_linear_head_5')
    for dataset, name in [('cifar2_5k', 'C2'), ('cifar10_v2', 'C10'), ('artbench2_256', 'AB2')]:
        for track in ('gen', 'val'):
            panel = f'{name} {track.title()}'
            for transform, label in [('linear', 'Linear'), ('square', 'Square'), ('signed_square', 'Signed square')]:
                values = [f'{100*rows[dataset, track, transform, rule]["mean"]:.2f}' for rule in rules]
                lines8.append(f'{panel} & {label} & '+' & '.join(values)+r' \\')
                if transform == 'linear':
                    continue
                values = []
                for rule in rules:
                    row = gains[dataset, track, transform, rule]
                    values.append(f'{100*row["mean"]:.2f} [{100*row["ci_low"]:.2f}, {100*row["ci_high"]:.2f}]')
                lines9.append(f'{panel} & {label} & '+' & '.join(values)+r' \\')
    for number, lines, caption, label in [
        (8, lines8, 'Matched DAS readouts under each native configuration. Fixed support-direction heads are selected from linear scores. ArtBench uses native $\\lambda=1$.', 'tab:app-square-comparison'),
        (9, lines9, 'DAS gains relative to the matching linear readout, with pointwise 95\\% paired query-bootstrap intervals. Validation queries are resampled together across seeds; generation queries are resampled within each model.', 'tab:app-square-gains')]:
        lines += [r'\bottomrule', r'\end{tabular}', r'\caption{'+caption+'}', r'\label{'+label+'}', r'\end{table}']
        target = args.output/f'Table{number}'
        target.mkdir(parents=True, exist_ok=True)
        (target/'table.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'tables': [8, 9], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
