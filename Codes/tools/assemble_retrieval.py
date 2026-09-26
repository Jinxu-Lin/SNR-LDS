"""Assemble reviewed-500 CFM/DDPM retrieval from completed curvature blocks."""
import argparse
import json
from pathlib import Path
from balds.workflows.source_assembly import assemble

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root', type=Path, required=True)
p.add_argument('--sources', type=Path, nargs='+', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--method', choices=['fmas_raw','ekfac_if'], required=True)
p.add_argument('--process', choices=['cfm','ddpm'], required=True)
p.add_argument('--selection', help='query selection JSON, relative to data root')
p.add_argument('--comparison-manifest', help='JSON listing comparison methods, relative to data root')
a=p.parse_args()
print(json.dumps(assemble(a.data_root,a.sources,a.output,a.method,a.selection,a.comparison_manifest,process=a.process),indent=2))
