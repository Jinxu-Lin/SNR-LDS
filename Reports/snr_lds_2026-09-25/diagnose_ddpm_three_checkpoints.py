"""Three-checkpoint CPU diagnostic; does not modify the four-checkpoint run."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/path/to/CFA/_Data')
OUT = ROOT / 'results/ddpm_tracin_snr_20260925/diagnostics/three_checkpoints'
CODE = Path('/tmp/cfa-ddpm-tracin-gas-j3-snr-v1/Codes')
SNR = Path('/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r4/src')
sys.path[:0] = [str(CODE), str(SNR)]
from cfa.contracts.estimator import FeatureSet
from cfa.contracts.registry import METHODS
import cfa.horizontal.methods
from balds.evaluation.background import evaluate_snr
from balds.workflows.benchmark import load_array, write_json

STEPS = (2000, 4000, 6000)
BASE = ROOT / 'featurize/trak_T100/cifar2_das_retrain_20260924/ddpm/seed_42'


def tensor(step, name, shape):
    value = torch.as_tensor(torch.load(BASE / f'step_{step}' / name,
                                     map_location='cpu', weights_only=False))
    assert tuple(value.shape) == shape and torch.isfinite(value).all()
    return value


def main():
    manifest = json.loads((ROOT / 'results/snr_lds_20260925/a3/inputs/benchmark_manifest.json').read_text())
    panels = manifest['panels'] if isinstance(manifest, dict) else manifest
    train = [tensor(s, 'train_features.pt', (5000, 4096)) for s in STEPS]
    summaries = []
    for track in ('val', 'gen'):
        panel = next(p for p in panels if p['panel_id'] == f'cifar2_das_s42_{track}')
        ids = panel['query_ids']
        for s in STEPS:
            meta = json.loads((BASE / f'step_{s}/meta.json').read_text())
            assert meta['step'] == s and meta['query_ids'][track] == ids
        query = [tensor(s, f'query_features_{track}.pt', (100, 4096)) for s in STEPS]
        feats = FeatureSet(train[-1], None, 'trak_T100', ckpt_grads=train,
                           ckpt_query=query, ckpt_steps=STEPS)
        masks = load_array(ROOT / panel['masks'])
        n = panel.get('n_subsets', len(masks))
        y = load_array(ROOT / panel['response'])[:n, [panel['response_query_ids'].index(q) for q in ids]]
        for method in ('tracincp_T100', 'gas_T100'):
            scores = METHODS.get(method).score(feats, query[-1], 0., device='cpu')
            assert scores.shape == (5000, 100) and np.isfinite(scores).all()
            result = evaluate_snr(scores, masks[:n], y, device='cpu')
            for q, row in zip(ids, result['per_query']):
                row['query_id'] = q
            directory = OUT / track / method
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / 'scores.npy', scores)
            write_json(directory / 'per_query.json', result['per_query'])
            summary = dict(method=method, track=track, steps=STEPS,
                           aggregation='uniform_three_checkpoint_mean', retrained_trajectory=True,
                           rule=result['rule'], core_code='19c31140c300d344f827f15e04d5ff460ad9f6ac',
                           snr_code=str(SNR), diagnostic_only=True,
                           full_all100=float(np.mean([r['full_lds'] for r in result['per_query']])),
                           **result['summary'])
            write_json(directory / 'summary.json', summary)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    write_json(OUT / 'summary.json', summaries)


if __name__ == '__main__':
    main()
