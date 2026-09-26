"""Inspect existing training artifacts without loading executable pickle objects."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def inspect(data_root):
    import torch
    records = []
    for seed in (42, 123, 456):
        base = Path("checkpoints/cifar2_5k/cfm_uncond") / f"seed_{seed}"
        for name in ("final.pt", "step_1953.pt", "step_3906.pt", "step_5859.pt", "loss_history.json"):
            relative = base / name
            path = data_root / relative
            item = {"seed": seed, "path": relative.as_posix(), "exists": path.is_file()}
            if path.is_file():
                item.update(bytes=path.stat().st_size, sha256=digest(path))
                if path.suffix == ".pt":
                    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                    item["metadata"] = {k: v for k, v in checkpoint.items()
                                        if v is None or isinstance(v, (str, bool, int, float))}
                    del checkpoint
            records.append(item)
    return {"kind": "inspection_of_existing_artifacts", "not_a_training_receipt": True,
            "paths_relative_to": "external_data_root", "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional NEW evidence directory; must not exist")
    args = parser.parse_args()
    report = inspect(args.data_root)
    if args.output is None:
        print(json.dumps(report, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "artifacts.json").write_text(json.dumps(report, indent=2) + "\n")
    for seed in (42, 123, 456):
        source = args.data_root / f"checkpoints/cifar2_5k/cfm_uncond/seed_{seed}/loss_history.json"
        if source.is_file():
            target = args.output / f"seed_{seed}"
            target.mkdir()
            (target / source.name).write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
