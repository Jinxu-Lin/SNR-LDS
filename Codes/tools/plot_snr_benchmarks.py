"""Render the predeclared SNR fit and coverage figures from saved fit artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np
from balds.workflows.benchmark import load_array

CORE = ('fmas_raw', 'dtrak_T100', 'das_native_sq', 'ekfac_if')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--panel', default='cifar2_5k_s42_gen')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    (args.output / 'Fig9').mkdir(parents=True, exist_ok=True)
    (args.output / 'Fig10').mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.manifest.read_text())
    panels = payload.get('panels', payload) if isinstance(payload, dict) else payload
    panel = next(item for item in panels if item['panel_id'] == args.panel)
    sources = {item['method']: item for item in panel['methods']}
    figure, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    for column, method in enumerate(CORE):
        for row, query in enumerate((0, 1)):
            fit_path = args.results / args.panel / method / 'fits' / f'query_{query}.json'
            fit = json.loads(fit_path.read_text())
            axis = axes[row, column]
            source = sources[method]
            path = Path(source['scores'])
            if not path.is_absolute():
                path = args.data_root / path
            raw = load_array(path)[:, source['query_ids'].index(query)]
            x = np.zeros_like(raw, dtype=float) if fit['rms'] == 0 else raw / fit['rms']
            axis.hist(x, bins=50, density=True, alpha=.35, color='tab:blue')
            if fit['status'] == 'ok':
                grid = np.asarray(fit['grid'], dtype=float)
                density = np.exp(np.asarray(fit['log_density'], dtype=float))
                fitted = np.exp(fit['params']['A'] + fit['params']['C'] * grid**2)
                axis.plot(grid, density, color='black', lw=1, label='KDE')
                axis.plot(grid, fitted, color='tab:red', lw=1, label='zero-mean fit')
                bound = 3 * fit['params']['sigma_x']
                axis.axvline(-bound, color='tab:orange', ls='--', lw=1)
                axis.axvline(bound, color='tab:orange', ls='--', lw=1)
            axis.set_title(f'{method} q{query}')
            if method == 'das_native_sq':
                axis.set_xlabel('signed t (aggregation uses t² once)')
    axes[0, 0].legend(fontsize=7)
    figure.savefig(args.output / 'Fig9/fixed-c2-s42-gen-q0-q1.png', dpi=180)
    plt.close(figure)

    sigma, coverage, labels = [], [], []
    for panel in sorted(args.results.iterdir()):
        if not panel.is_dir():
            continue
        for method in sorted(panel.iterdir()):
            per_query = method / 'per_query.json'
            if not per_query.is_file():
                continue
            for record in json.loads(per_query.read_text()):
                if record.get('snr_lds') is None:
                    continue
                sigma.append(record['params'].get('sigma'))
                coverage.append(record['retained_fraction'])
                labels.append(method.name)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    axes[0].hist([v for v in sigma if v is not None], bins=50)
    axes[0].set_title('sigma')
    axes[1].hist(coverage, bins=np.linspace(0, 1, 41))
    axes[1].set_title('zeta=3 retained fraction')
    figure.savefig(args.output / 'Fig10/sigma-coverage-distributions.png', dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    main()
