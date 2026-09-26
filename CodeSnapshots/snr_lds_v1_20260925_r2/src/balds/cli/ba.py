"""Small CPU-first entry point for the paper's evaluation procedures."""
from __future__ import annotations

import argparse
import json

SNR_RULE = "snr_zero_mean_gaussian_v1"
LEGACY_RULE = "paper_twosided_fit_then_native_aggregate_v1"


def _zetas(value):
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("zetas must be comma-separated numbers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one zeta is required")
    return result


def build_parser():
    parser = argparse.ArgumentParser(prog="balds", description="Background-adaptive Linear Datamodeling Score")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("evaluate", help="evaluate an aligned score / subset-response cell")
    for name in ("scores", "masks", "responses", "output"):
        p.add_argument("--" + name, required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--fit-scores", help="separate fitting/selection values; scores still control aggregation")
    group.add_argument("--score-space", choices=("native", "das-presquare"), default="native",
                       help="das-presquare: input is signed t, fit t and sum native t squared")
    p.add_argument("--n-subsets", type=int)
    p.add_argument("--query-ids", help="comma-separated aligned column positions")
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--pilot", action="store_true", help="separately retained early appendix positive-only pilot")
    p.add_argument("--save-fits", action="store_true")
    p.add_argument("--rule", choices=(SNR_RULE, LEGACY_RULE), required=True,
                   help="scientific rule; SNR runs must name snr_zero_mean_gaussian_v1")
    p.add_argument("--zetas", type=_zetas, default=(1., 2., 3., 4.),
                   help="SNR thresholds from one fit (default: 1,2,3,4)")
    p.add_argument("--primary-zeta", type=float, default=3.)
    p = sub.add_parser("batch", help="evaluate portable paper panels; missing inputs remain explicit")
    for name in ("manifest", "data-root", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--save-fits", action="store_true")
    p.add_argument("--rule", choices=(SNR_RULE,), required=True)
    p.add_argument("--zetas", type=_zetas, default=(1., 2., 3., 4.))
    p.add_argument("--primary-zeta", type=float, default=3.)
    p = sub.add_parser("deletion", help="compare Full/BA method choice on measured deletion utilities")
    for name in ("evaluations", "utilities", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--legacy-per-query", action="store_true",
                   help="archived per-query selector; never use for the SNR main table")
    p = sub.add_parser("status", help="report atomic SNR query/method completion")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("verify", help="verify SNR checkpoint identity, masks and aggregation algebra")
    for name in ("manifest", "data-root", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--zetas", type=_zetas, default=(1., 2., 3., 4.))
    p.add_argument("--primary-zeta", type=float, default=3.)
    return parser


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    command = args.pop("command")
    from balds.workflows.benchmark import (deletion, evaluate_files, json_value,
                                           manifest_status, run_manifest, verify_manifest)
    if command == "evaluate":
        if args["query_ids"] is not None:
            args["query_ids"] = [int(v) for v in args["query_ids"].split(",")]
        result = evaluate_files(**args)
    elif command == "batch":
        result = run_manifest(**args)
    elif command == "deletion":
        args["benchmark"] = not args.pop("legacy_per_query")
        result = deletion(**args)
    elif command == "status":
        result = manifest_status(**args)
    else:
        result = verify_manifest(**args)
    print(json.dumps(json_value(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
