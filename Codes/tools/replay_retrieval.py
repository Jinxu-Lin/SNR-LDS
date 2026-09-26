"""Replay final13-method retrieval metrics from archived scores, without inference."""
import argparse
import json
from pathlib import Path
from balds.workflows.retrieval_replay import replay, archive_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, help='optional override of the relative accepted archive addresses')
    parser.add_argument('--train-labels', type=Path, help='optional50000-row .npy; otherwise read cached CIFAR labels')
    parser.add_argument('--paper', type=Path, help='read-only check against printed Tables2/18/19')
    args = parser.parse_args(argv)
    result = replay(args.data_root, args.output, manifest=args.manifest,
                    train_labels=args.train_labels, paper=args.paper)
    print(json.dumps(result, indent=2))
    return 0 if result.get('paper_comparison', {}).get('status', 'pass') == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
