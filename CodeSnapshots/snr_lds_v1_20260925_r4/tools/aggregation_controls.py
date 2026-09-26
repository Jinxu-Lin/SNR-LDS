"""Reproduce adopted M3 and S1-S3 CPU aggregation controls from stored inputs."""
import argparse
from pathlib import Path
from balds.workflows.controls import e2, snr_transforms, transforms

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('stage',choices=['deletion','transforms','snr-transforms'])
p.add_argument('--data-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--utilities',type=Path,help='native measured utility TSV (deletion stage)')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
if a.stage=='deletion':
    if a.utilities is None:p.error('--utilities required for deletion controls')
    e2(a.data_root,a.output,a.utilities)
elif a.stage=='transforms':transforms(a.data_root,a.output)
else:snr_transforms(a.data_root,a.output)
