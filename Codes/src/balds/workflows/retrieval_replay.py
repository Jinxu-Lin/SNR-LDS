"""Replay archived reviewed500 retrieval from native scores, entirely on CPU."""
from pathlib import Path
import re
import numpy as np

from balds.artifacts.addressing import relpath
from balds.artifacts.e3c import atomic, read_json
from balds.schema.artifact import ArtifactKind as K
from balds.schema.runspec import RunSpec
from .source_assembly import query_selection, report, select_val_only

METHOD_LABELS = {
    'fmas_raw': 'FMAS', 'ekfac_if': 'EK-FAC IF', 'dtrak_T100': 'D-TRAK', 'das_T100': 'DAS',
    'trak_T100': 'TRAK', 'clip_cos': 'CLIP cosine', 'pixel_cos': 'Pixel cosine',
    'clip_dot': 'CLIP dot', 'pixel_dot': 'Pixel dot', 'grad_dot_T100': 'Gradient dot',
    'grad_cos_T100': 'Gradient cosine', 'relative_if_T100': 'Relative IF', 'renorm_if_T100': 'Renormalized IF',
}


def archive_manifest():
    """Relative addresses for the accepted CFM/DDPM score families."""
    result = []
    for process in ('cfm', 'ddpm'):
        base = RunSpec(dataset='cifar10_inj4', process=process, conditional=True, seed=42, query_type='inject')
        panel = dict(process=process, queries=relpath(K.INJECT_QUERIES, base),
                     injection_meta=relpath(K.INJECT_META, base), methods=[])
        for method in METHOD_LABELS:
            spec = base.with_(method=method)
            if method not in ('fmas_raw', 'ekfac_if'):
                item = dict(method=method, scores=relpath(K.SCORES, spec),
                            meta=relpath(K.SCORES_META, spec), axis='full500')
            elif process == 'cfm':
                family = 'xc3-fmas' if method == 'fmas_raw' else 'xc3-if'
                valroot = f'results/cfm_val100_20260923/incoming/{family}'
                out = f'results/cfm_test400_20260923/analysis/{method}'
                item = dict(method=method, scores=f'{out}/test400_scores.npy',
                            meta=f'{out}/result.json', validation=f'{valroot}/{method}/result.json',
                            val_score_root=valroot, axis='test400')
            else:
                root = 'results/runpod_ddpm_source_20260923/score'
                item = dict(method=method, scores=f'{root}/test/{relpath(K.SCORES, spec)}',
                            meta=f'{root}/test/{relpath(K.SCORES_META, spec)}',
                            validation=f'{root}/val/{relpath(K.SCORES_META, spec)}',
                            val_score_root=f'{root}/val', axis='test400')
            panel['methods'].append(item)
        result.append(panel)
    return result


def _path(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('retrieval manifest paths must be relative to the data root')
    resolved = (Path(root) / path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError('retrieval manifest path escapes the data root')
    return resolved


def _assert_source(metadata, query_source):
    found = metadata.get('queries_sha256', metadata.get('query_source', metadata.get('identity', {}).get('query_source')))
    if found != query_source:
        raise ValueError('score metadata query source differs from the reviewed query bundle')


def _selection(metadata):
    curve = metadata.get('val_curve', metadata.get('per_lambda_ap_val'))
    if not curve:
        raise ValueError('validation-only selection evidence is absent')
    curve = {float(key): float(value) for key, value in curve.items()}
    e = metadata.get('identity', {}).get('ekfac', {})
    grid = metadata.get('lambda_sweep') or (e.get('blockshrink_grid') if e.get('damping_mode') == 'blockshrink'
                                          else e.get('damping_grid')) or sorted(curve)
    grid = list(map(float, grid))
    if set(grid) != set(curve) or not np.isfinite(list(curve.values())).all():
        raise ValueError('validation grid and curve disagree')
    best = max(grid, key=lambda damping: curve[damping])
    if best != float(metadata['best_lam']):
        raise ValueError('selected damping differs from the first maximum of validation pool AP')
    return best, grid, curve


def _array(path, shape):
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'expected finite score shape {shape}, got {value.shape}: {path}')
    return value


def replay(data_root, output, *, manifest=None, train_labels=None, paper=None):
    """Recompute all query metrics; reselect curvature damping from val matrices.

    For projected/embedding methods only the selected full500 score is required;
    their archived validation grid/order is checked, and the selected matrix's
    validation pool AP is recomputed. No gradients, fitting or model inference.
    """
    panels = read_json(manifest) if manifest is not None else archive_manifest()
    root, output = Path(data_root), Path(output)
    if train_labels is None:
        from datasets import load_dataset
        # Injection retains the original host labels in the original CIFAR row order.
        labels = np.asarray(load_dataset('cifar10', cache_dir=str(root / 'hf_cache'))['train']['label'])
    else:
        labels = np.load(train_labels, allow_pickle=False)
    if labels.shape != (50000,):
        raise ValueError('retrieval requires all50000 original CIFAR training labels')
    results = {}
    for panel in panels:
        process = panel['process']
        if process not in ('cfm', 'ddpm'):
            raise ValueError('retrieval process must be cfm or ddpm')
        base = RunSpec(dataset='cifar10_inj4', process=process, conditional=True, seed=42, query_type='inject')
        # Use the supplied relative query artifact; codecs preserve the native .pt schema.
        from balds.artifacts.codecs import codec_for
        queries = codec_for(K.INJECT_QUERIES).load(str(_path(root, panel['queries'])))
        val_ids, test_ids = query_selection(queries)
        source = queries['queries_sha256']
        meta = read_json(_path(root, panel['injection_meta']))
        if {x['method'] for x in panel['methods']} != set(METHOD_LABELS) or len(panel['methods']) != 13:
            raise ValueError('final retrieval requires each of the13 paper methods exactly once')
        results[process] = {}
        for item in panel['methods']:
            method = item['method']
            stored = read_json(_path(root, item['meta']))
            _assert_source(stored, source)
            selection = read_json(_path(root, item['validation'])) if 'validation' in item else stored
            _assert_source(selection, source)
            best, grid, curve = _selection(selection)
            if best != float(stored['best_lam']):
                raise ValueError('test scores were selected with a different validation damping')
            if item['axis'] == 'full500':
                matrix = _array(_path(root, item['scores']), (50000, 500))
                checked_best, recomputed = select_val_only({best: matrix[:, val_ids]}, [best],
                    np.asarray(queries['host_labels'])[val_ids], labels, meta)
                selected = matrix[:, test_ids]
                selection_check = 'selected validation matrix and complete archived curve/order'
            elif item['axis'] == 'test400':
                ids = stored.get('query_ids')
                if ids is None:
                    ids = [r['query_id'] for r in stored['test']['per_query']]
                if ids != test_ids:
                    raise ValueError('final test400 scores have a different original query axis')
                val_meta_ids = selection.get('query_ids', selection.get('identity', {}).get('query_ids'))
                if val_meta_ids != val_ids:
                    raise ValueError('validation scores have a different original query axis')
                matrices = {d: _array(_path(root, str(Path(item['val_score_root']) /
                        relpath(K.SCORES_LAMBDA, base.with_(method=method), lam=d))), (50000,100)) for d in grid}
                checked_best, recomputed = select_val_only(matrices, grid,
                    np.asarray(queries['host_labels'])[val_ids], labels, meta)
                selected = _array(_path(root, item['scores']), (50000,400))
                selection_check = 'all validation matrices; test excluded from selection'
            else:
                raise ValueError('score axis must be full500 or test400')
            if checked_best != best or any(not np.isclose(v, curve[d], rtol=1e-10, atol=1e-12)
                                            for d, v in recomputed.items()):
                raise ValueError(f'{process}/{method}: recomputed validation pool AP differs')
            stats = report(selected, test_ids, queries, labels, meta)
            result = dict(method=method, process=process, query_source=source, best_lam=best,
                          selection_check=selection_check, val_curve=curve, scores=item['scores'], **stats)
            results[process][method] = result
            atomic(output / process / f'{method}.json', result)
            print(f"{process}/{method}: R@200={stats['means']['recall_at_k_global']:.6f} AP={stats['means']['ap_global']:.6f}", flush=True)
    summary = {'protocol': 'reviewed-v2-val100-test400', 'methods_per_process': 13,
               'results': {p: {m: dict(best_lam=r['best_lam'], **r['means']) for m,r in methods.items()}
                           for p, methods in results.items()}}
    if paper is not None:
        summary['paper_comparison'] = compare_paper(results, paper)
    atomic(output / 'summary.json', summary)
    return summary


def compare_paper(results, paper):
    """Compare numeric cells at the paper's printed precision; never edit Paper."""
    reverse = {label: method for method, label in METHOD_LABELS.items()}
    differences, boundaries, count = [], [], 0
    concepts = ['leopard','cattle','wolf','camel','tractor','sunflower','castle','butterfly','mushroom','skyscraper']
    for table, process in ((2,None),(18,'cfm'),(19,'ddpm')):
        metric = 'ap_global'
        for line in (Path(paper) / f'Figures/Table{table}/table.tex').read_text().splitlines():
            if 'multicolumn' in line and 'Recall@200' in line:
                metric = 'recall_at_k_global'
            columns = [p.strip() for p in line.split('&')]
            if columns[0] not in reverse:
                continue
            method = reverse[columns[0]]
            printed = [float(re.sub(r'\\\\.*$', '', value).strip()) for value in columns[1:]]
            actual = ([results[p][method]['means'][k] for p in ('cfm','ddpm')
                       for k in ('recall_at_k_global','ap_global')] if table == 2 else
                      [results[process][method]['per_concept'][concept][metric] for concept in concepts])
            precision = 4 if table == 2 else 3
            if len(printed) != len(actual):
                raise ValueError('paper table columns do not match retrieval output')
            for column, (old, new) in enumerate(zip(printed, actual)):
                count += 1
                error, half_unit = abs(new - old), 0.5 * 10 ** (-precision)
                record = dict(table=table, method=method, metric=(
                    ('recall_at_k_global' if column % 2 == 0 else 'ap_global') if table == 2 else metric),
                    column=column, printed=old, reproduced=new)
                if error > half_unit + 1e-12:
                    differences.append(record)
                elif abs(error - half_unit) <= 1e-12:
                    boundaries.append(record)
    return {'status': 'pass' if not differences else 'mismatch', 'numeric_cells': count,
            'differences': differences, 'rounding_boundary_cells': boundaries,
            'rule': 'absolute error <= half the last printed decimal unit + 1e-12'}
