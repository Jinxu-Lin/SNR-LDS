"""Jinxu3 CPU continuation: reuse original scorer, then the accepted SNR evaluator.

No feature extraction, training, generation, or old density-based BA evaluation.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

DATA = Path('/path/to/CFA/_Data')
ROOT = DATA / 'results/ddpm_tracin_snr_20260925/a1'
CFA_CODE = Path('/tmp/cfa-ddpm-tracin-gas-j3-snr-v1')
SNR_CODE = Path('/path/to/BA-LDS/CodeSnapshots/snr_lds_v1_20260925_r4')
SUPERVISOR = Path('/path/to/CFA-worktrees/ba-native-gpu-wave-20260921/Codes/tools')
PY = '/path/to/CFA-envs/e3c-torch260/bin/python'
METHODS = {'tracincp_T100', 'gas_T100'}
SOURCE = DATA / 'results/ddpm_tracin_gas_20260924/ba_manifest.json'
MANIFEST = ROOT / 'inputs/manifest.json'


def convert_manifest(reference, scored):
    """Use accepted response IDs and preserve the retrained trajectory identity."""
    indexed = {p['panel_id']: p for p in scored}
    output = []
    for ref in reference:
        if ref['panel_id'] not in ('cifar2_das_s42_gen', 'cifar2_das_s42_val'):
            continue
        src = indexed[ref['panel_id']]
        if src['query_ids'] != ref['query_ids'] or src['train_ids'] != ref['train_ids']:
            raise ValueError('DDPM training/query IDs differ from accepted SNR panel')
        panel = copy.deepcopy(ref)
        panel['methods'] = []
        panel['not_applicable'] = []
        for item in src['methods']:
            if item['method'] not in METHODS:
                continue
            if item['query_ids'] != ref['query_ids'] or item['train_ids'] != ref['train_ids']:
                raise ValueError('score-column IDs differ from declared panel IDs')
            base = next(m for m in ref['methods'] if m['method'] == item['method'])
            method = copy.deepcopy(base)
            method.update(scores=item['scores'], query_ids=item['query_ids'], score_space='native')
            method['identity']['completed_source_manifest'] = str(SOURCE)
            method['identity']['completed_source_identity'] = item['identity']
            panel['methods'].append(method)
        if {m['method'] for m in panel['methods']} != METHODS or len(panel['methods']) != 2:
            raise ValueError('need exactly TracInCP and GAS on each track')
        output.append(panel)
    if len(output) != 2:
        raise ValueError('need both DDPM generation and validation panels')
    return output


def prepare():
    reference = json.loads((DATA / 'results/snr_lds_20260925/a3/inputs/benchmark_manifest.json').read_text())
    scored = json.loads(SOURCE.read_text())
    panels = convert_manifest(reference, scored)
    for panel in panels:
        for method in panel['methods']:
            if not Path(method['scores']).is_file():
                raise FileNotFoundError(method['scores'])
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists() and json.loads(MANIFEST.read_text()) != panels:
        raise ValueError('existing output manifest has a different input identity')
    tmp = MANIFEST.with_suffix('.tmp')
    tmp.write_text(json.dumps(panels, indent=2) + '\n')
    tmp.replace(MANIFEST)
    print('PREPARED 4 cells / 400 queries; corrected original val IDs retained', flush=True)


def environment(path):
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
                PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                PYTHONPATH=str(path))


def run(actions=('ingest', 'score')):
    old = CFA_CODE / '_Data/reports/ddpm_tracin_gas_2026-09-24/run_j3.py'
    for action in actions:
        print(f'STAGE {action}', flush=True)
        subprocess.run([PY, '-u', str(old), action], cwd=CFA_CODE,
                       env=environment(CFA_CODE / 'Codes'), check=True)
    prepare()
    env = environment(SNR_CODE / 'src')
    commands = [
        [PY, '-m', 'balds.cli.ba', 'batch', '--rule', 'snr_zero_mean_gaussian_v1',
         '--zetas', '1,2,3,4', '--primary-zeta', '3', '--save-fits',
         '--manifest', str(MANIFEST), '--data-root', str(DATA), '--output', str(ROOT/'panels')],
        [PY, str(SNR_CODE/'tools/summarize_benchmarks.py'), '--manifest', str(MANIFEST),
         '--results', str(ROOT/'panels'), '--output', str(ROOT/'tables')],
    ]
    for command in commands:
        print('STAGE ' + ' '.join(command), flush=True)
        subprocess.run(command, cwd=SNR_CODE, env=env, check=True)
    command = [PY, '-m', 'balds.cli.ba', 'verify', '--manifest', str(MANIFEST),
               '--data-root', str(DATA), '--output', str(ROOT/'panels')]
    result = subprocess.run(command, cwd=SNR_CODE, env=env, text=True,
                            stdout=subprocess.PIPE, check=True)
    (ROOT/'verify.json').write_text(result.stdout)
    verify = json.loads(result.stdout)
    if verify['verdict'] != 'pass' or verify['verified_queries'] != 400 or verify['errors']:
        raise ValueError(f'SNR verification failed: {verify}')
    print('DONE 4 DDPM cells / 400 query records; GPU hours=0', flush=True)


def launch():
    sys.path.insert(0, str(SUPERVISOR))
    from e3c_launch import spawn_job
    print(spawn_job([PY, '-u', str(Path(__file__).resolve()), 'run'],
                   Path(__file__).resolve().parent, environment(SNR_CODE/'src'),
                   ROOT/'supervised', -1))


def run_local():
    """Recovery has already registered all features in the master store."""
    run(actions=('score',))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['launch', 'run', 'run_local', 'prepare'])
    globals()[parser.parse_args().action]()
