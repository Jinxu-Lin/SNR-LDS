"""Prepare, compute or inspect CFM common-200 retrieval score blocks locally."""
import argparse
import json
from pathlib import Path

from balds.workflows import source_scoring


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status"))
    parser.add_argument("--input-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection", required=True, help="published selection JSON, relative to input data root")
    parser.add_argument("--method", choices=source_scoring.METHODS, required=True)
    parser.add_argument("--query-chunk", type=int, default=100, help="resident columns; MC stays at the accepted 250/250")
    parser.add_argument("--code-version", help="optional provenance stamp shared by workers")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--row-rank", type=int, default=0)
    parser.add_argument("--row-world", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    common = dict(method=args.method, selection=args.selection, query_chunk=args.query_chunk,
                  code_version=args.code_version)
    if args.command == "run":
        common.update(device=args.device, rank=args.rank, world=args.world,
                      row_rank=args.row_rank, row_world=args.row_world, cpu_threads=args.cpu_threads)
    result = getattr(source_scoring, args.command)(args.input_data_root, args.output_root, **common)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
