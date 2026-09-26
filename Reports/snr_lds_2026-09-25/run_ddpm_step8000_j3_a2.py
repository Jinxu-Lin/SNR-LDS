"""Researcher-authorized XC0 -> J3 handoff of step8000 only, then CPU closeout."""
import importlib.util
from pathlib import Path
import subprocess
import sys

from run_ddpm_recovery_j3 import CODE, LEGACY, PY, environment

HERE = Path(__file__).resolve().parent
OUT = Path('/path/to/CFA/_Data/results/ddpm_tracin_snr_20260925/recovery_j3_a2')


def run():
    sys.path.insert(0, str(CODE / 'Codes'))
    loader = importlib.util.spec_from_file_location('ddpm_existing', LEGACY)
    old = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(old)
    print('STAGE GPU step8000 only; step2000/4000/6000 retained unchanged', flush=True)
    old.features((8000,))
    print('STAGE CPU four scores -> SNR -> summary -> verify; CUDA hidden', flush=True)
    subprocess.run([PY, '-u', str(HERE / 'run_ddpm_tracin_snr.py'), 'run_local'],
                   cwd=HERE, env=environment(''), check=True)
    print('COMPLETE four DDPM score matrices and 400 verified SNR records', flush=True)


def launch():
    sys.path.insert(0, '/path/to/CFA-worktrees/ba-native-gpu-wave-20260921/Codes/tools')
    from e3c_launch import spawn_job
    print(spawn_job([PY, '-u', str(Path(__file__).resolve()), 'run'],
                   CODE, environment('0'), OUT / 'supervised', 0))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['launch', 'run'])
    globals()[parser.parse_args().action]()
