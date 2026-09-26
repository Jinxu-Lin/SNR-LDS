"""Reconstructed three-seed training recipe; dry-run unless --execute is set."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
from datetime import datetime, timezone
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def commands(config, data_root, device, stage, seeds):
    for seed in seeds:
        cmd = ["balds-run", "--data-root", str(data_root), "--device", device]
        for key, value in config["overrides"].items():
            # The current CLI coerces numbers/lists, not JSON boolean literals.
            if isinstance(value, bool):
                value = int(value)
            cmd += ["--set", f"{key}={json.dumps(value)}"]
        cmd += [stage, "--dataset", config["dataset"], "--process", config["process"],
                "--uncond", "--seed", str(seed)]
        if stage == "generate":
            cmd += ["--Q", str(config["generation"]["queries"]), "--ode-steps",
                    str(config["generation"]["ode_steps"])]
        yield seed, cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=("train", "generate"), default="train")
    parser.add_argument("--seed", type=int, choices=(42, 123, 456))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text())
    seeds = [args.seed] if args.seed is not None else config["seeds"]
    data_root = args.data_root.expanduser().resolve()
    jobs = list(commands(config, data_root, args.device, args.stage, seeds))
    for seed, cmd in jobs:
        print(shlex.join(cmd), flush=True)
        if not args.execute:
            continue
        model_dir = Path("checkpoints") / config["dataset"] / "cfm_uncond" / f"seed_{seed}"
        output = model_dir / "final.pt" if args.stage == "train" else (
            Path("generations") / config["dataset"] / "cfm_uncond" / f"seed_{seed}" / "samples.pt")
        if (data_root / output).exists():
            raise SystemExit(f"Refusing to overwrite or skip an existing output: {output}")
        if args.stage == "generate" and not (data_root / model_dir / "final.pt").is_file():
            raise SystemExit(f"Missing trained checkpoint: {model_dir / 'final.pt'}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run = HERE / "runs" / f"{stamp}_{args.stage}_seed{seed}"
        run.mkdir(parents=True, exist_ok=False)
        versions = {}
        for package in ("snr-lds", "torch", "torchvision", "numpy", "scipy", "datasets"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        receipt = {"command": cmd, "config": config, "start_utc": stamp,
                   "python": sys.version, "packages": versions,
                   "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                   "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True),
                   "status": "started"}
        receipt_path = run / "receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        try:
            with (run / "console.log").open("w") as stream:
                result = subprocess.run(cmd, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
            receipt.update(exit_code=result.returncode, status="completed" if result.returncode == 0 else "failed")
        except BaseException as exc:
            receipt.update(status="interrupted_or_failed", error=repr(exc))
            raise
        finally:
            receipt["end_utc"] = datetime.now(timezone.utc).isoformat()
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
