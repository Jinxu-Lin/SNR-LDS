"""Prepare portable paper panels with explicit query IDs; no model execution.

Run after preparing/migrating source arrays. DDPM validation IDs are derived
from the archive's ordered split labels, not guessed from array positions.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from balds.workflows.benchmark import load_array, write_json

METHODS = ['pixel_dot', 'pixel_cos', 'clip_dot', 'clip_cos', 'grad_dot_T100',
           'grad_cos_T100', 'tracincp_T100', 'gas_T100', 'journey_trak_T100',
           'relative_if_T100', 'renorm_if_T100', 'trak_T100', 'dtrak_T100',
           'das_native_sq', 'ekfac_if', 'fmas_raw']


def _save_npy_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def _relative_to_data(path, data):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(data))
    except ValueError as error:
        raise ValueError(f"source must be below data root: {resolved}") from error


def _completed_overrides(paths, data):
    """Index explicitly supplied completed manifests without copying artifacts."""
    result = {}
    for source in paths:
        payload = json.loads(Path(source).read_text())
        panels = payload.get('panels', payload) if isinstance(payload, dict) else payload
        for panel in panels:
            for method in panel.get('methods', []):
                path = Path(method['scores'])
                if path.is_absolute() and not path.is_relative_to(data):
                    # Historical manifests keep their original provenance. Resolve
                    # their data-relative address here, never read the legacy checkout.
                    parts = path.parts
                    if '_Data' not in parts:
                        raise ValueError(f'cannot relocate score address: {path}')
                    path = data.joinpath(*parts[parts.index('_Data') + 1:])
                elif not path.is_absolute():
                    path = data / path
                if not path.is_file():
                    continue
                result[(panel['panel_id'], method['method'])] = dict(
                    method, scores=_relative_to_data(path, data),
                    source_manifest=str(Path(source).resolve().relative_to(data)))
    return result


def prepare(data, datasets, *, restore_ab2_das=False, derived_input_root=None,
            source_manifests=()):
    from balds.schema.artifact import ArtifactKind as K
    from balds.schema.runspec import RunSpec
    from balds.artifacts.addressing import relpath
    from balds.workflows.config import load_config
    from balds.workflows.common import _load_ds
    from balds.data.base import balanced_query_indices

    # ``cifar2_das`` is an imported platform whose existing loader is
    # registered by the DAS workflow module.  The normal application container
    # imports that module, but this standalone CPU prepare entry point does not.
    # Initialize only that registration here before resolving the dataset; do
    # not duplicate the loader or pull in the full training container.
    if 'cifar2_das' in datasets:
        import balds.workflows.das_import  # noqa: F401

    data = Path(data).resolve()
    cfg = load_config({'storage.data_root': str(data)})
    overlays = _completed_overrides(source_manifests, data)
    panels = []
    for dataset in datasets:
        ddpm, ab2 = dataset == 'cifar2_das', dataset == 'artbench2_256'
        seeds = [42] if ddpm or ab2 else [42, 123, 456]
        n = 50000 if dataset == 'cifar10_v2' else 5000
        ddpm_val_ids = None
        if ddpm:
            _, validation = _load_ds(cfg, dataset)
            ddpm_val_ids = list(map(int, balanced_query_indices(validation.labels, 100)))
        for seed in seeds:
            for track in ['gen', 'val']:
                base = dict(dataset=dataset, seed=seed, process='ddpm' if ddpm else 'cfm',
                            conditional=dataset != 'cifar2_5k', query_type=track)
                spec = RunSpec(**base)
                ids = ddpm_val_ids if ddpm and track == 'val' else list(range(100))
                response = relpath(K.GT_MATRIX, spec)
                masks = relpath(K.SUBSET_MASKS, spec)
                response_q = np.load(data/response, mmap_mode='r').shape[1] if (data/response).is_file() else (1000 if ddpm else 100)
                methods = []
                not_applicable = []
                for method in METHODS:
                    if method == 'journey_trak_T100' and track == 'val':
                        not_applicable.append(dict(method=method,
                            reason='Journey-TRAK validation is outside the filed method definition'))
                        continue
                    source_name = 'das_T100' if method == 'das_native_sq' else method
                    score = relpath(K.SCORES, RunSpec(**base, method=source_name))
                    override = overlays.get((f'{dataset}_s{seed}_{track}', method))
                    if ab2 and method == 'das_native_sq' and override is None:
                        if derived_input_root is None:
                            score = f'results/paper/inputs/artbench2_256/{track}/das_presquare_lambda1.npy'
                        else:
                            root = Path(derived_input_root)
                            if root.is_absolute():
                                root = Path(_relative_to_data(root, data))
                            score = str(root / 'artbench2_256' / track / 'das_presquare_lambda1.npy')
                        if restore_ab2_das:
                            if not (data / score).is_file():
                                from balds.workflows.controls import das_linear
                                value = das_linear(data, dataset, track, seed)
                                _save_npy_atomic(data / score, value)
                    if override is not None:
                        score = override['scores']
                    actual_q = load_array(data/score).shape[1] if (data/score).is_file() else len(ids)
                    if ddpm and actual_q == 1000:
                        score_ids = list(range(1000))
                    elif actual_q == 100:
                        score_ids = ids
                    else:
                        raise ValueError(f'unexpected query coverage ({actual_q}): {score}')
                    entry = dict(method=method, scores=score, train_ids=list(range(n)),
                        query_ids=score_ids, score_space='das-presquare' if method == 'das_native_sq' else 'native',
                        identity=dict(**base, method=source_name,
                                      scoring_config={'selection': 'filed native configuration; no SNR retuning'}))
                    if override is not None:
                        if override['train_ids'] != entry['train_ids']:
                            raise ValueError(f'completed source training IDs differ: {method}')
                        entry.update(scores=override['scores'], query_ids=override['query_ids'])
                        entry['identity']['completed_source_manifest'] = override['source_manifest']
                        entry['identity']['completed_source_identity'] = override.get('identity', {})
                    methods.append(entry)
                panels.append(dict(panel_id=f'{dataset}_s{seed}_{track}', train_ids=list(range(n)),
                    mask_train_ids=list(range(n)), query_ids=ids, masks=masks, response=response,
                    response_query_ids=list(range(response_q)), n_subsets=32 if ab2 else 64,
                    methods=methods, not_applicable=not_applicable))
    return panels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', default=['cifar2_5k', 'cifar10_v2', 'cifar2_das', 'artbench2_256'],
                   choices=['cifar2_5k', 'cifar10_v2', 'cifar2_das', 'artbench2_256'])
    p.add_argument('--restore-ab2-das', action='store_true',
                   help='derive lambda1 signed t from existing features, without training or new gradients')
    p.add_argument('--derived-input-root', type=Path,
                   help='caller-owned output below data-root for restored AB2 signed t')
    p.add_argument('--source-manifest', action='append', default=[], type=Path,
                   help='completed manifest to overlay by exact panel/method identity')
    args = p.parse_args()
    panels = prepare(args.data_root, args.datasets, restore_ab2_das=args.restore_ab2_das,
                     derived_input_root=args.derived_input_root,
                     source_manifests=args.source_manifest)
    if args.output.is_file():
        if json.loads(args.output.read_text()) != panels:
            raise ValueError('existing benchmark manifest differs; choose a fresh output root')
    else:
        write_json(args.output, panels)
    print(json.dumps({'panels': len(panels), 'manifest': str(args.output), 'evaluation_computed': False}))


if __name__ == '__main__':
    main()
