"""CPU diagnostic only: change DAS fit input t -> t^2, keep aggregation t^2.

Uses the accepted A3 manifest/ID alignment and R4 SNR kernels unchanged.
No production outputs, training, features, GPU work, or threshold selection.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SNAPSHOT = Path('/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r4')
sys.path.insert(0, str(SNAPSHOT / 'src'))
from balds.evaluation.background import evaluate_snr
from balds.workflows.benchmark import load_array, write_json

ROOT = Path('/path/to/CFA/_Data')
SOURCE = ROOT / 'results/snr_lds_20260925/a3'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panels', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((SOURCE / 'inputs/benchmark_manifest.json').read_text())
    panels = manifest['panels'] if isinstance(manifest, dict) else manifest
    summaries = []
    for panel in panels:
        pid = panel['panel_id']
        if pid not in args.panels and 'all' not in args.panels:
            continue
        methods = [m for m in panel['methods'] if m['method'] in ('das_native_sq', 'das_T100')]
        if not methods:
            continue
        begin = time.monotonic()
        method = methods[0]
        assert method['score_space'] == 'das-presquare'
        assert method['train_ids'] == panel['train_ids']
        assert panel.get('mask_train_ids', panel['train_ids']) == panel['train_ids']
        ids = panel['query_ids']
        matrix = load_array(ROOT / method['scores'])
        matrix = np.asarray(matrix[:, [method['query_ids'].index(q) for q in ids]], dtype=np.float64)
        native = matrix * matrix
        masks = load_array(ROOT / panel['masks'])
        n = panel.get('n_subsets', len(masks))
        response = load_array(ROOT / panel['response'])
        response = response[:n, [panel['response_query_ids'].index(q) for q in ids]]
        result = evaluate_snr(native, masks[:n], response, device='cpu')
        old = json.loads((SOURCE / 'panels' / pid / method['method'] / 'per_query.json').read_text())
        old = {r['query_id']: r for r in old}
        rows = []
        for q, row in zip(ids, result['per_query']):
            row['query_id'] = q
            assert abs(row['full_lds'] - old[q]['full_lds']) < 1e-10
            rows.append(dict(query_id=q, presquare_fit=old[q], native_square_fit=row))
        summary = dict(panel_id=pid, seconds=time.monotonic()-begin,
                       scope='diagnostic_not_paper_adoption', kernel_snapshot=str(SNAPSHOT),
                       rule='snr_zero_mean_gaussian_v1', fit_space='native_t_squared',
                       aggregation_space='native_t_squared', n_total=len(rows), by_zeta={})
        for z in ('1.0', '2.0', '3.0', '4.0'):
            old_valid = [r for r in rows if r['presquare_fit']['by_zeta'][z]['snr_lds'] is not None]
            new_valid = [r for r in rows if r['native_square_fit']['by_zeta'][z]['snr_lds'] is not None]
            common = [r for r in new_valid if r['presquare_fit']['by_zeta'][z]['snr_lds'] is not None]
            def stats(subset, key):
                if not subset:
                    return dict(n_valid=0, lds=None, retained_median=None, n_empty=0)
                values = [r[key]['by_zeta'][z] for r in subset]
                return dict(n_valid=len(values), lds=float(np.mean([v['snr_lds'] for v in values])),
                            retained_median=float(np.median([v['retained_fraction'] for v in values])),
                            n_empty=sum(v['n_selected'] == 0 for v in values))
            summary['by_zeta'][z] = dict(old=stats(old_valid, 'presquare_fit'),
                                        new=stats(new_valid, 'native_square_fit'),
                                        common_old=stats(common, 'presquare_fit'),
                                        common_new=stats(common, 'native_square_fit'))
        write_json(args.output / pid / 'per_query.json', rows)
        write_json(args.output / pid / 'summary.json', summary)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write_json(args.output / 'summary.json', summaries)

if __name__ == '__main__':
    main()
