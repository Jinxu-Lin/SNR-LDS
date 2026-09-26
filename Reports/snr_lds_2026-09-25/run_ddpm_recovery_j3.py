"""Authorized 2026-09-25 takeover: restored features -> GPU gaps -> CPU SNR.

Uses unmodified 19c3114 featurizer/scorer and accepted SNR r4; no training.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CODE = Path('/tmp/cfa-ddpm-tracin-gas-j3-snr-v1')
DATA = Path('/path/to/CFA/_Data')
ROOT = DATA / 'results/ddpm_tracin_snr_20260925/recovery_j3_a1'
PY = '/path/to/CFA-envs/e3c-torch260/bin/python'
LEGACY = CODE / '_Data/reports/ddpm_tracin_gas_2026-09-24/run_j3.py'


def environment(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(CODE/'Codes'),
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                NUMEXPR_NUM_THREADS='2', PYTHONUNBUFFERED='1',
                HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')


def restore():
    import json
    import torch
    sys.path.insert(0, str(CODE/'Codes'))
    loader = importlib.util.spec_from_file_location('ddpm_existing', LEGACY)
    old = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(old)
    from cfa.contracts.artifact import ArtifactKind as K
    c, specs = old.context()
    for step in (2000, 6000):
        directory = DATA / f'results/ddpm_tracin_gas_20260924/recovered_xc3/step_{step}'
        meta = json.loads((directory/'meta.json').read_text())
        assert meta['step'] == step
        assert meta['query_ids'] == {t:p['query_ids'] for t,p in old.panels().items()}
        key = dict(feat='trak_T100', step=step)
        entries = [(K.FEATURE_META, specs['gen'], meta)]
        for kind, track, name, n in [
            (K.TRAIN_FEATURES,'gen','train_features.pt',5000),
            (K.QUERY_FEATURES,'gen','query_features_gen.pt',100),
            (K.QUERY_FEATURES,'val','query_features_val.pt',100),
        ]:
            value = torch.load(directory/name, map_location='cpu', weights_only=False)
            old.checked(value, (n,4096))
            entries.append((kind, specs[track], value))
        for kind, spec, value in entries:
            if c.store.exists(kind, spec, **key):
                existing = c.store.load(kind, spec, **key)
                assert existing == value if isinstance(value, dict) else torch.equal(torch.as_tensor(existing), value)
            else:
                c.store.save(kind, spec, value, **key)
        print(f'RESTORED step={step} 3 matrices + meta', flush=True)


def run():
    restore()
    print('STAGE GPU missing step4000/8000; complete step2000/6000 reused', flush=True)
    subprocess.run([PY,'-u',str(LEGACY),'features'], cwd=CODE, env=environment('0'), check=True)
    print('STAGE CPU score -> SNR batch -> summary -> verify (CUDA hidden)', flush=True)
    subprocess.run([PY,'-u',str(HERE/'run_ddpm_tracin_snr.py'),'run_local'],
                   cwd=HERE, env=environment(''), check=True)
    print('COMPLETE: four score matrices and 400 SNR query records verified', flush=True)


def launch():
    sys.path.insert(0, '/path/to/CFA-worktrees/ba-native-gpu-wave-20260921/Codes/tools')
    from e3c_launch import spawn_job
    print(spawn_job([PY,'-u',str(Path(__file__).resolve()),'run'], HERE,
                   environment(''), ROOT/'supervised', 0))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['launch','run','restore'])
    globals()[parser.parse_args().action]()
