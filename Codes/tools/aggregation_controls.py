"""Reproduce final Tables 8/9: matched DAS readouts and paired query intervals."""
import argparse
from pathlib import Path
from balds.workflows.controls import transforms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['transforms'])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    transforms(args.data_root, args.output)


if __name__ == '__main__':
    main()
