"""Run one independent MC repeat or summarize its fixed-pilot diagnostics."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(prog='balds-repeat', description=__doc__)
    p.add_argument('action', choices=['prepare', 'run', 'status', 'summarize'])
    p.add_argument('--data-root')
    p.add_argument('--output', required=True, help='relative to data root, or absolute within it')
    p.add_argument('--config', help='JSON recipe; defaults to the bundled R16 configuration')
    p.add_argument('--repeat', type=int)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--pilot-manifest', help='independent pilot NPY paths for each method/track')
    p.add_argument('--analysis-output', help='separate output for summarize; never changes repeat inputs')
    args = p.parse_args(argv)
    from balds.workflows.config import load_config
    from balds.workflows import e3c, repeatability
    overrides = {'storage.data_root': args.data_root} if args.data_root else {}
    data = Path(load_config(overrides)['storage']['data_root']).resolve()
    out = Path(args.output)
    out = (data / out).resolve() if not out.is_absolute() else out.resolve()
    if out == data or data not in out.parents:
        p.error('--output must be a subdirectory of the selected data root')
    cfg = e3c.resolve_config(args.config)
    if args.action == 'prepare':
        result = e3c.prepare(cfg, data, out)
    elif args.action == 'run':
        if args.repeat is None:
            p.error('run requires --repeat')
        e3c.run_repeat(cfg, args.repeat, args.device, data, out)
        result = e3c.status(cfg, out)
    elif args.action == 'status':
        result = e3c.status(cfg, out)
    else:
        if not args.pilot_manifest:
            p.error('summarize requires --pilot-manifest')
        analysis = None
        if args.analysis_output:
            analysis = Path(args.analysis_output)
            analysis = (data / analysis).resolve() if not analysis.is_absolute() else analysis.resolve()
            if analysis == data or data not in analysis.parents:
                p.error('--analysis-output must be a subdirectory of the selected data root')
        result = repeatability.summarize(cfg, data, out, args.pilot_manifest, analysis)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
