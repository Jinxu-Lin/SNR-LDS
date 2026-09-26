"""Produce DDPM reference or retrained four-checkpoint scores for the final paper."""
import argparse
import json
from pathlib import Path
from balds.workflows import ddpm_baselines as workflow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('features', 'score'))
    parser.add_argument('--family', required=True, choices=('reference', 'checkpoints'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True, help='explicit DDPM gen/val query axes')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--line', choices=('embedding', 'gradient'), default='gradient')
    parser.add_argument('--checkpoint-root', default='results/ddpm_gaps_20260923/retrain_j2')
    parser.add_argument('--steps', type=int, nargs='+', default=list(workflow.STEPS))
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    kwargs = dict(family=args.family, line=args.line, checkpoint_root=args.checkpoint_root)
    if args.command == 'features':
        result = workflow.features(args.data_root, args.manifest, steps=args.steps, device=args.device, **kwargs)
    else:
        result = workflow.score(args.data_root, args.manifest, args.output, **kwargs)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
