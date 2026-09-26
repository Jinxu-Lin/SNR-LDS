"""Check dossier links, configuration, recorded artifacts, and CLI parsing (no training)."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "Experiments"


def check():
    errors = []
    markdown = [ROOT / "README.md", *EXPERIMENTS.rglob("*.md")]
    links = 0
    for path in markdown:
        text = path.read_text()
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            target = target.split("#")[0]
            if not (path.parent / target).exists():
                errors.append(f"Broken link: {path.relative_to(ROOT)} -> {target}")
            links += 1
    for path in EXPERIMENTS.rglob("*"):
        if path.suffix not in (".md", ".json", ".csv", ".py") or "runs" in path.parts:
            continue
        if re.search(r"/home/[A-Za-z0-9_.-]+", path.read_text()):
            errors.append(f"Personal home path: {path.relative_to(ROOT)}")
    folder = EXPERIMENTS / "01_training/cifar2_cfm"
    spec = importlib.util.spec_from_file_location("training_recipe", folder / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.REPO != ROOT:
        errors.append("Training runner resolves the wrong repository root")
    from balds.cli.main import build_parser, parse_overrides
    config = json.loads((folder / "config.json").read_text())
    command_count = 0
    for stage in ("train", "generate"):
        for seed, cmd in module.commands(config, Path("/path/to/new-data"), "cuda:0", stage, config["seeds"]):
            parsed = build_parser().parse_args(cmd[1:])
            if parsed.seed != seed or parsed.cmd != stage or not parsed.uncond:
                errors.append(f"CLI identity mismatch: {seed}/{stage}")
            if parse_overrides(parsed.overrides) != config["overrides"]:
                errors.append(f"Override mismatch: {seed}/{stage}")
            command_count += 1
    records = json.loads((folder / "evidence/artifacts.json").read_text())["records"]
    if len(records) != 15 or any(not record["exists"] for record in records):
        errors.append("Expected 15 recorded training artifacts")
    for record in records:
        if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
            errors.append(f"Invalid hash: {record['path']}")
    archive = ROOT / "_LocalArchive/experiment_reorganization_2026-09-26"
    if archive.exists():
        result = subprocess.run(["git", "check-ignore", "-q", str(archive)], cwd=ROOT)
        if result.returncode != 0:
            errors.append("Local archive is not ignored")
    return {"status": "pass" if not errors else "fail", "markdown_links": links,
            "parsed_cli_commands": command_count, "recorded_training_artifacts": len(records),
            "gpu_jobs_launched": 0, "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check()
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text)
    raise SystemExit(bool(result["errors"]))
