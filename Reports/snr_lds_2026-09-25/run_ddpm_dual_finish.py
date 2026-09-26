"""Checkpoint-boundary handoff: J3 keeps 4000; XC0 computes 8000; J3 CPU join."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

DATA = Path('/path/to/CFA/_Data')
OUT = DATA/'results/ddpm_tracin_snr_20260925'
OLD_JOB = OUT/'recovery_j3_a1/supervised/launcher/repeat_0_2b4f495cee63.json'
FEATURES = DATA/'featurize/trak_T100/cifar2_das_retrain_20260924/ddpm/seed_42'
INCOMING = DATA/'results/ddpm_tracin_gas_20260924/incoming/xc0_step8000'
PY_J3 = '/path/to/CFA-envs/e3c-torch260/bin/python'
PY_XC0 = '/path/to/miniconda3/envs/da/bin/python'
CPU = Path('/path/to/BA-LDS/Reports/snr_lds_2026-09-25/run_ddpm_tracin_snr.py')
SUPERVISOR = Path('/path/to/CFA-worktrees/ba-native-gpu-wave-20260921/Codes/tools')


def process_token(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except (OSError,IndexError):
        return None


def alive(pid,token):
    return token is not None and process_token(pid)==token


def code(remote):
    return Path('/tmp/cfa-ddpm-step8000-xc0' if remote else '/tmp/cfa-ddpm-tracin-gas-j3-snr-v1')


def environment(remote):
    paths=[str(code(remote)/'Codes')]
    if remote:
        paths.append(str(DATA/'results/ddpm_tracin_gas_20260924/python_deps'))
    return dict(os.environ, CUDA_VISIBLE_DEVICES='1' if remote else '',
                PYTHONPATH=os.pathsep.join(paths), OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2',
                PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')


def modules(remote):
    sys.path.insert(0,str(code(remote)/'Codes'))
    spec=importlib.util.spec_from_file_location('legacy_ddpm',code(remote)/'_Data/reports/ddpm_tracin_gas_2026-09-24/run_j3.py')
    old=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    return old


def xc0():
    old=modules(True)
    old.features((8000,))
    # Existing features() validates each completed tensor; transfer only this step.
    subprocess.run(['ssh','jinxu3','mkdir','-p',str(INCOMING)],check=True)
    subprocess.run(['rsync','-a',str(FEATURES/'step_8000'),f'jinxu3:{INCOMING}/'],check=True)
    marker=OUT/'xc0_step8000_a1/RETURNED.json'
    old.write(marker,dict(step=8000,features=3,host='xuchang-lab0',status='returned'))
    subprocess.run(['rsync','-a',str(marker),f'jinxu3:{INCOMING}/'],check=True)
    print('DONE step8000: three features + meta returned to jinxu3',flush=True)


def join():
    old=modules(False)
    job=json.loads(OLD_JOB.read_text())
    logfile=Path(job['log'])
    print('WAIT existing J3 step4000 completion; do not restart it',flush=True)
    while 'DONE_FEATURE step=4000 split=val' not in logfile.read_text():
        if not alive(job['worker_pid'],job['worker_token']):
            raise RuntimeError('original J3 chain exited before step4000 finished; inspect original log')
        time.sleep(10)
    # Only terminate the exact recorded feature child, after all 4000 tensors saved.
    feature_pid=3516458
    proc=Path(f'/proc/{feature_pid}')
    if proc.exists():
        args=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if str(code(False)/'_Data/reports/ddpm_tracin_gas_2026-09-24/run_j3.py') not in args or not args.rstrip().endswith('features'):
            raise RuntimeError('feature PID no longer identifies the original feature worker')
        token=process_token(feature_pid)
        if token != '302259470':
            raise RuntimeError('feature PID was reused; refusing to signal another process')
        os.kill(feature_pid,signal.SIGTERM)
        for _ in range(60):
            if not alive(feature_pid,token):
                break
            time.sleep(1)
        else:
            raise RuntimeError('feature worker did not exit; no forced kill or parallel CPU launch')
    for _ in range(60):
        if not alive(job['worker_pid'],job['worker_token']):
            break
        time.sleep(1)
    else:
        raise RuntimeError('original controller still active; do not duplicate its CPU stage')
    old.write(OUT/'dual_join_a1/HANDOFF.json',dict(
        completed_step=4000,old_job=str(OLD_JOB),reason='researcher requested step8000 on xuchang0',
        note='original chain nonzero exit is intentional checkpoint-boundary handoff'))
    print('J3 step4000 completed; original serial chain stopped at checkpoint boundary',flush=True)
    print('WAIT XC0 step8000 returned marker',flush=True)
    while not (INCOMING/'RETURNED.json').is_file():
        time.sleep(30)
    marker=json.loads((INCOMING/'RETURNED.json').read_text())
    if marker != dict(step=8000,features=3,host='xuchang-lab0',status='returned'):
        raise ValueError('unexpected returned identity')
    import torch
    from cfa.contracts.artifact import ArtifactKind as K
    c,specs=old.context()
    source=INCOMING/'step_8000'
    meta=json.loads((source/'meta.json').read_text())
    assert meta['step']==8000 and meta['query_ids']=={t:p['query_ids'] for t,p in old.panels().items()}
    entries=[(K.FEATURE_META,specs['gen'],meta)]
    for kind,track,name,n in [(K.TRAIN_FEATURES,'gen','train_features.pt',5000),
                              (K.QUERY_FEATURES,'gen','query_features_gen.pt',100),
                              (K.QUERY_FEATURES,'val','query_features_val.pt',100)]:
        value=torch.load(source/name,map_location='cpu',weights_only=False)
        entries.append((kind,specs[track],old.checked(value,(n,4096))))
    for kind,spec,value in entries:
        key=dict(feat='trak_T100',step=8000)
        if c.store.exists(kind,spec,**key):
            existing=c.store.load(kind,spec,**key)
            assert existing==value if isinstance(value,dict) else torch.equal(torch.as_tensor(existing),value)
        else:
            c.store.save(kind,spec,value,**key)
    print('INGESTED XC0 step8000; all four checkpoint sets ready; starting CPU chain',flush=True)
    subprocess.run([PY_J3,'-u',str(CPU),'run_local'],cwd=CPU.parent,env=environment(False),check=True)
    print('COMPLETE four scores and 400 SNR query records verified',flush=True)


def launch(remote):
    sys.path.insert(0,str(SUPERVISOR))
    from e3c_launch import spawn_job
    task='xc0' if remote else 'join'
    root=OUT/('xc0_step8000_a1' if remote else 'dual_join_a1')
    print(spawn_job([PY_XC0 if remote else PY_J3,'-u',str(Path(__file__).resolve()),task],
                   code(remote),environment(remote),root/'supervised',1 if remote else -1))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['xc0','join','launch-xc0','launch-join'])
    action=parser.parse_args().action
    if action.startswith('launch-'):
        launch(action=='launch-xc0')
    else:
        globals()[action]()
