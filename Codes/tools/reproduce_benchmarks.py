"""Run the final paper's array-level SNR pipeline with portable data/output roots.

Stages are explicit: preparing inputs never launches model training. Evaluation
fits the final SNR rule; postprocess reuses existing results without fitting.
"""
import argparse
import json
import os
from pathlib import Path

from balds.workflows.benchmark import deletion, manifest_status, run_manifest, verify_manifest


def main():
    code = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'evaluate', 'postprocess', 'status', 'verify'])
    p.add_argument('--data-root', type=Path, default=os.environ.get('BALDS_DATA_ROOT', str(code.parent/'_Data')))
    p.add_argument('--output', type=Path, help='defaults to DATA/results/paper/snr; relative paths use data root')
    p.add_argument('--manifest', type=Path, help='existing portable input manifest; default OUTPUT/inputs/benchmark_manifest.json')
    p.add_argument('--results', type=Path, help='existing panels for postprocess/status/verify; default OUTPUT/panels')
    p.add_argument('--config', type=Path, default=code/'configs/paper.json')
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    data = a.data_root.expanduser().resolve()
    output = a.output or Path('results/paper/snr')
    output = output if output.is_absolute() else data/output
    manifest = a.manifest or output/'inputs/benchmark_manifest.json'
    panels = a.results or output/'panels'
    config = json.loads(a.config.read_text())
    if a.stage == 'prepare':
        from prepare_benchmarks import prepare
        from balds.workflows.benchmark import write_json
        archived = [data/path for path in config['source_manifests']]
        available = [path for path in archived if path.is_file()]
        result = prepare(data, config['datasets'], restore_ab2_das=True,
                         derived_input_root=output/'inputs',
                         source_manifests=available)
        write_json(manifest, result)
        result = {'manifest': str(manifest), 'panels': len(result),
                  'source_overlays': [str(path.relative_to(data)) for path in available],
                  'unavailable_archived_overlays': [str(path.relative_to(data)) for path in archived if not path.is_file()]}
    elif a.stage == 'evaluate':
        result = run_manifest(manifest, data, panels, device=a.device,
                              rule=config['rule'], zetas=config['zetas'],
                              primary_zeta=config['primary_zeta'], save_fits=True)
    elif a.stage == 'postprocess':
        from summarize_benchmarks import summarize
        result = summarize(manifest, panels, output/'tables')
        deletion(panels, data/config['utility_table'], output/'edel')
        result = {'groups': len(result['records']), 'output': str(output)}
    elif a.stage == 'status':
        result = manifest_status(manifest, panels)
    else:
        result = verify_manifest(manifest, data, panels,
                                 zetas=config['zetas'], primary_zeta=config['primary_zeta'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
