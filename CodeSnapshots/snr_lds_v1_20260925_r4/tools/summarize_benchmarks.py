"""Aggregate evaluated paper cells across model seeds without refitting."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from balds.workflows.benchmark import write_json

SHARED_VALIDATION = {'pixel_dot', 'pixel_cos', 'clip_dot', 'clip_cos'}
BOOTSTRAP_COUNT = 2000
BOOTSTRAP_SEED = 20260920


def _query_bootstrap(cells, field):
    """Equal-seed query bootstrap; identical query IDs share resample draws."""
    maps = [{int(row['query_id']): row[field] for row in cell['rows']
             if row.get(field) is not None} for cell in cells]
    if not maps:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    shared_ids = sorted(set.intersection(*(set(values) for values in maps)))
    if not shared_ids:
        return None
    draw = rng.integers(0, len(shared_ids), (BOOTSTRAP_COUNT, len(shared_ids)))
    bank = np.zeros(BOOTSTRAP_COUNT)
    for values in maps:
        vector = np.asarray([values[q] for q in shared_ids], dtype=float)
        bank += vector[draw].mean(axis=1) / len(maps)
    return np.quantile(bank, [.025, .975]).tolist()


def summarize(manifest, results, output):
    panels = json.loads(Path(manifest).read_text())
    panels = panels['panels'] if isinstance(panels, dict) else panels
    groups, missing = defaultdict(list), []
    for panel in panels:
        directory = Path(results) / panel['panel_id']
        for method in panel['methods']:
            identity = method['identity']
            key = (identity['dataset'], identity['query_type'], method['method'])
            path = directory / method['method'] / 'summary.json'
            panel_path = directory / 'panel.json'
            if not panel_path.is_file():
                missing.append(dict(panel=panel['panel_id'], method=method['method'], reason='unevaluated panel'))
                continue
            state = json.loads(panel_path.read_text())
            current = next((s for s in state.get('summary', []) if s['method'] == method['method']), None)
            # A stale method directory must not override the current panel's missing state.
            if current is None or current.get('status', '').startswith('missing') or not path.is_file():
                missing.append(dict(panel=panel['panel_id'], method=method['method'], reason='missing input/output'))
                continue
            summary = json.loads(path.read_text())
            rows = json.loads((path.parent / 'per_query.json').read_text())
            if summary['n_valid'] == 0:
                missing.append(dict(panel=panel['panel_id'], method=method['method'], reason='no valid fits'))
            groups[key].append(dict(seed=identity['seed'], summary=summary, rows=rows))
    records = []
    for (dataset, track, method), cells in sorted(groups.items()):
        if len({c['seed'] for c in cells}) != len(cells):
            raise ValueError('duplicate model seed in benchmark group')
        # A valid SNR run can have zero successful fits. Its schema is still
        # SNR, and must not be inferred from the nonempty valid-seed subset.
        fields = {'snr_lds' if 'snr_lds' in c['summary'] else 'ba_lds'
                  for c in cells}
        if len(fields) != 1:
            raise ValueError('mixed BA/SNR schemas within benchmark group')
        adaptive_field = fields.pop()
        shared = track == 'val' and method in SHARED_VALIDATION
        if shared:
            columns = ('query_id', 'status', 'full_lds', adaptive_field)
            reference = [[r[k] for k in columns] for r in cells[0]['rows']]
            if any([[r[k] for k in columns] for r in c['rows']] != reference for c in cells[1:]):
                raise ValueError('shared validation metrics differ across seeds; inspect query/response identity')
            cells = cells[:1]
        valid = [c for c in cells if c['summary']['n_valid'] > 0]
        row = dict(dataset=dataset, track=track, method=method,
                   shared_validation=shared, n_seeds=len(valid), seeds=[c['seed'] for c in valid])
        paired_cells = [dict(rows=[r for r in c['rows']
                                  if r.get(adaptive_field) is not None]) for c in valid]
        for metric in ('full_lds', adaptive_field):
            values = [c['summary'][metric] for c in valid]
            row[metric] = float(np.mean(values)) if values else None
            row[metric + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            row[metric + '_query_ci95'] = _query_bootstrap(paired_cells, metric) if values else None
        for count in ('n_total', 'n_valid', 'n_failed', 'n_fit_failed', 'n_all_zero',
                      'n_empty', 'n_constant_prediction', 'n_constant_response',
                      'n_missing_query'):
            row[count] = sum(c['summary'].get(count, 0) for c in cells)
        records.append(row)
    sensitivity = []
    for (dataset, track, method), cells in sorted(groups.items()):
        if not any(any('by_zeta' in row for row in cell['rows']) for cell in cells):
            continue
        if track == 'val' and method in SHARED_VALIDATION:
            cells = cells[:1]
        for zeta in (1., 2., 3., 4.):
            field = str(zeta)
            seed_means = []
            bootstrap_cells = []
            for cell in cells:
                rows = [dict(row, zeta_lds=row.get('by_zeta', {}).get(field, {}).get('snr_lds'))
                        for row in cell['rows']]
                values = [row['zeta_lds'] for row in rows if row['zeta_lds'] is not None]
                if values:
                    seed_means.append(float(np.mean(values)))
                    bootstrap_cells.append(dict(rows=rows))
            sensitivity.append(dict(
                dataset=dataset, track=track, method=method, zeta=zeta,
                n_seeds=len(seed_means), snr_lds=float(np.mean(seed_means)) if seed_means else None,
                seed_std=(float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else None),
                query_ci95=_query_bootstrap(bootstrap_cells, 'zeta_lds') if bootstrap_cells else None))
    has_legacy = any('ba_lds' in record for record in records)
    if has_legacy and any('snr_lds' in record for record in records):
        raise ValueError('cannot combine legacy BA and SNR result groups in one table')
    result = dict(rule=('legacy_or_mixed_input' if has_legacy else 'snr_zero_mean_gaussian_v1'),
                  aggregation='equal weight over valid model-seed means; sample seed SD; Full uses the same SNR-valid queries',
                  uncertainty=f'{BOOTSTRAP_COUNT} paired query bootstrap draws, seed {BOOTSTRAP_SEED}; seed SD is separate',
                  shared_validation='one estimate after checking identical per-query metrics',
                  records=records, sensitivity=sensitivity, missing=missing)
    output = Path(output)
    write_json(output / 'summary.json', result)
    if records:
        temporary = output / 'summary.csv.tmp'
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(output / 'summary.csv')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.manifest, a.results, a.output)
    print(json.dumps({'groups': len(result['records']), 'missing': len(result['missing'])}))


if __name__ == '__main__':
    main()
