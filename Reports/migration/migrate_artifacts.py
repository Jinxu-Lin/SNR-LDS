#!/usr/bin/env python3
"""Metadata-only migration inventory by default; copying is an explicit action.

Uses only the standard library. Never hashes, deletes source files, contacts
hosts, launches experiments, or rewrites historical binary payloads.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import subprocess
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
DEFAULT_PROFILES = "analysis,regeneration"
TRANSFERABLE = {"selected", "reusable_input", "historical_appendix", "provenance_only"}


def safe_path(root: Path, relative: str) -> Path:
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Expected a root-relative path: {relative}")
    candidate = root.joinpath(*p.parts)
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes root: {relative}")
    return candidate


def selected_rows(inventory: dict, profiles: str):
    wanted = set(profiles.split(","))
    return [row for row in inventory["files"]
            if row["profile"] in wanted and row["status"] in TRANSFERABLE]


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def scan(source: Path, rules: dict, output: Path):
    source = source.resolve()
    checkout = {"head": None, "tracked_worktree_changes": None,
                "meaning": "checkout metadata only; individual experiment execution versions remain in receipts and artifact metadata"}
    if shutil.which("git"):
        head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True)
        status = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
        if head.returncode == 0:
            checkout["head"] = head.stdout.strip()
        if status.returncode == 0:
            checkout["tracked_worktree_changes"] = len(status.stdout.splitlines())
    families = [{**f, "file_count": 0, "apparent_bytes": 0} for f in rules["families"]]
    candidates = [(f, tuple({g.split("*")[0].split("?")[0].split("[")[0]
                             for g in f.get("include", [])})) for f in families]
    files, totals, symlinks = [], {}, []
    roots = [source / "_Data", source / "Codes/Figure/out/json"]
    for root in roots:
        if not root.exists():
            continue
        for directory, dirs, names in os.walk(root, followlinks=False):
            dirs.sort()
            for name in sorted(names):
                path = Path(directory) / name
                relative = path.relative_to(source).as_posix()
                if path.is_symlink():
                    symlinks.append({"source_relative": relative, "target": os.readlink(path)})
                    continue
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue  # Live execution may rename temporary files.
                top = "/".join(relative.split("/")[:2])
                total = totals.setdefault(top, {"file_count": 0, "apparent_bytes": 0})
                total["file_count"] += 1
                total["apparent_bytes"] += stat.st_size
                matches = [f for f, prefixes in candidates
                           if relative.startswith(prefixes)
                           and any(fnmatch.fnmatchcase(relative, g) for g in f.get("include", []))
                           and not any(fnmatch.fnmatchcase(relative, g) for g in f.get("exclude", []))]
                for f in matches:
                    f["file_count"] += 1
                    f["apparent_bytes"] += stat.st_size
                eligible = [f for f in matches if f.get("profile")]
                if not eligible:
                    continue
                # Specific rules precede broader rules. Record secondary membership.
                family = eligible[0]
                destination = relative
                if family.get("destination_prefix"):
                    destination = family["destination_prefix"] + relative[len(family["source_prefix"]):]
                files.append({
                    "source_relative": relative, "destination_relative": destination,
                    "family": family["id"], "also_matches": [f["id"] for f in eligible[1:]],
                    "status": family["status"], "profile": family["profile"],
                    "role": family["role"], "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                })
    files.sort(key=lambda row: row["source_relative"])
    indexed, missing = [], []
    manifest = source / "_Data/manifest.db"
    if manifest.exists():
        with sqlite3.connect(manifest.as_uri() + "?mode=ro", uri=True) as connection:
            for row in connection.execute("SELECT item_key,kind,relpath,size,code_version FROM manifest"):
                entry = dict(zip(("item_key", "kind", "relpath", "size", "code_version"), row))
                indexed.append(entry)
                if not safe_path(source / "_Data", entry["relpath"]).exists():
                    missing.append(entry)
    selected_data = {r["source_relative"].removeprefix("_Data/") for r in files
                     if r["source_relative"].startswith("_Data/") and r["status"] in TRANSFERABLE}
    indexed_paths = {r["relpath"] for r in indexed}
    inventory = {
        "schema_version": 1,
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source), "source_root_is_provenance_only": True,
        "source_checkout": checkout,
        "scan": "lstat/stat metadata; live local files; no content hashing; no remote inventory",
        "scientific_status": rules["scientific_status"], "scope_totals": totals,
        "families": families, "files": files, "symlinks_not_selected": symlinks,
        "pending": rules["pending"], "excluded": rules["excluded"],
        "registry": {"entries": len(indexed), "missing_local_entries": len(missing),
                     "missing_local": missing,
                     "selected_data_without_registry_entry": sorted(selected_data - indexed_paths)},
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "inventory.json", inventory)
    with (output / "files.tsv").open("w", newline="") as stream:
        columns = ["source_relative", "destination_relative", "family", "status", "profile", "role", "bytes", "mtime_ns"]
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(files)
    for profile in sorted({r["profile"] for r in files}):
        rows = selected_rows(inventory, profile)
        (output / f"files.{profile}.txt").write_text("".join(r["source_relative"] + "\n" for r in rows))
    (output / "files.txt").write_text("".join(r["source_relative"] + "\n" for r in selected_rows(inventory, DEFAULT_PROFILES)))
    print(json.dumps({"inventory": str(output / "inventory.json"), "listed_files": len(files),
                      "registry_entries": len(indexed), "registry_missing_local": len(missing)}, indent=2))


def assess(rows, source: Path, target: Path, check_target=False):
    counts, issues = {}, []
    for row in rows:
        src = safe_path(source, row["source_relative"])
        dst = safe_path(target, row["destination_relative"])
        if not src.is_file():
            state = "source_missing"
        elif src.stat().st_size != row["bytes"] or src.stat().st_mtime_ns != row["mtime_ns"]:
            state = "source_changed_since_inventory"
        elif not dst.exists():
            state = "destination_missing" if check_target else "would_copy"
        elif dst.stat().st_size != row["bytes"] or dst.stat().st_mtime_ns != row["mtime_ns"]:
            state = "destination_conflict"
        else:
            state = "metadata_matches"
        counts[state] = counts.get(state, 0) + 1
        if state not in {"would_copy", "metadata_matches"}:
            issues.append({"source": row["source_relative"], "destination": row["destination_relative"], "status": state})
    return {"files": len(rows), "apparent_bytes": sum(r["bytes"] for r in rows), "states": counts, "issues": issues,
            "verification_limit": "size/mtime and path presence only; not a content or scientific-equivalence certificate"}


def copy_files(rows, source: Path, target: Path):
    if source.resolve() == target.resolve():
        raise ValueError("Source and target must differ")
    report = assess(rows, source, target)
    if report["issues"]:
        raise ValueError("Source changed or destination conflict: refresh inventory/review plan first")
    for row in rows:
        src, dst = safe_path(source, row["source_relative"]), safe_path(target, row["destination_relative"])
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(json.dumps(assess(rows, source, target, True), indent=2))


def filter_registry(inventory: dict, source: Path, target: Path, execute: bool):
    """Carry existing identities only; unregistered files stay explicitly unregistered."""
    source_db = source / "_Data/manifest.db"
    target_db = target / "_Data/manifest.db"
    selected = {r["source_relative"].removeprefix("_Data/"): r for r in inventory["files"]
                if r["status"] in TRANSFERABLE and r["source_relative"].startswith("_Data/")
                and r["destination_relative"] == r["source_relative"]}
    with sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True) as connection:
        schema = connection.execute("SELECT sql FROM sqlite_master WHERE name='manifest'").fetchone()[0]
        rows = list(connection.execute("SELECT * FROM manifest"))
    kept = [row for row in rows if row[3] in selected
            and safe_path(target / "_Data", row[3]).is_file()
            and safe_path(target / "_Data", row[3]).stat().st_size == row[5]]
    report = {"source_rows": len(rows), "eligible_present_rows": len(kept),
              "excluded_or_not_copied": len(rows) - len(kept), "target": str(target_db),
              "hashes": "existing values retained; no new hash is calculated", "execute": execute}
    if execute:
        if target_db.exists():
            raise ValueError("Target manifest already exists; do not overwrite a running destination registry")
        target_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(target_db) as connection:
            connection.execute(schema)
            connection.executemany("INSERT INTO manifest VALUES (?,?,?,?,?,?,?,?,?,?)", kept)
            connection.execute("CREATE INDEX idx_relpath ON manifest(relpath)")
            connection.execute("CREATE INDEX idx_blob ON manifest(blob_hash)")
    print(json.dumps(report, indent=2))


def relocate_json(inventory: dict, source: Path, target: Path, execute: bool):
    """Create runtime copies; keep original receipts/payloads unchanged."""
    source_root = str(source.resolve())
    old_root = inventory["source_root"]
    destination_root = str(target.resolve())
    selected = {r["source_relative"]: r["destination_relative"] for r in inventory["files"]}
    changes = []
    def visit(value, ba_manifest=False):
        if isinstance(value, dict):
            return {k: visit(v, ba_manifest) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v, ba_manifest) for v in value]
        if isinstance(value, str):
            for prefix in (source_root, old_root):
                if value.startswith(prefix + "/"):
                    relative = value[len(prefix) + 1:]
                    mapped = selected.get(relative, relative)
                    if ba_manifest and mapped.startswith("_Data/"):
                        return mapped.removeprefix("_Data/")
                    return destination_root + "/" + mapped
        return value
    for relative in json.loads((HERE / "selection.json").read_text())["runtime_json_inputs"]:
        src = safe_path(source, relative)
        if not src.exists():
            changes.append({"source": relative, "status": "source_missing"})
            continue
        value = json.loads(src.read_text())
        is_ba = relative.endswith(("/inputs/bench_manifest.json", "/inputs/edel_manifest.json"))
        transformed = visit(value, is_ba)
        if is_ba:
            for panel in transformed:
                if str(panel["panel_id"]).startswith("artbench2_256_"):
                    # Historical min-slicing tolerated n_subsets=64 even though
                    # AB2 has 32 actual retrained subsets; declare the real bank.
                    panel["n_subsets"] = 32
                for method in panel["methods"]:
                    is_das = method["method"] in {"das_T100", "das_native_sq"}
                    method["score_space"] = "das-presquare" if is_das else "native"
                    if is_das:
                        method["scores"] = method.get("presquare", method["scores"])
                        method["display_method"] = "DAS (fit t; aggregate t²)"
                    for legacy_key in ("score_representation", "ba_input_space", "presquare"):
                        method.pop(legacy_key, None)
        dst_rel = "_Data/runtime_manifests/" + relative
        changes.append({"source": relative, "destination": dst_rel,
                        "status": "adapted_runtime_copy", "content_changed": transformed != value,
                        "protocol": "explicit score_space; DAS fit(t)/sum(t²); data-root-relative inputs" if is_ba else "historical paths rebased"})
        if execute:
            dst = safe_path(target, dst_rel)
            if dst.exists():
                raise ValueError(f"Runtime copy already exists: {dst}; choose a fresh destination")
            write_json(dst, transformed)
    print(json.dumps({"execute": execute, "files": changes,
                      "scientific_warning": "BA runtime copies declare the NEW score_space protocol, but no predictions/results are recomputed here. Run balds BA into a new results/paper/... identity; never relabel V3 results."}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["plan", "inventory", "verify", "copy", "registry", "relocate-json"], default="plan")
    parser.add_argument("--source", type=Path, help="Existing CFA project root; needed for source checks/copy")
    parser.add_argument("--target", type=Path, default=HERE.parents[1], help="New project root")
    parser.add_argument("--inventory", type=Path, default=HERE / "inventory.json")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--profiles", default=DEFAULT_PROFILES)
    parser.add_argument("--execute", action="store_true", help="Only copy/registry/relocate-json may mutate the target")
    args = parser.parse_args()
    if args.action == "inventory":
        if not args.source:
            parser.error("inventory requires --source")
        scan(args.source, json.loads((HERE / "selection.json").read_text()), args.output)
        return
    inventory = json.loads(args.inventory.read_text())
    rows = selected_rows(inventory, args.profiles)
    if not args.source:
        if args.action != "plan":
            parser.error(f"{args.action} requires --source")
        counts = {}
        for r in rows:
            entry = counts.setdefault(r["profile"], {"files": 0, "bytes": 0})
            entry["files"] += 1
            entry["bytes"] += r["bytes"]
        print(json.dumps({"mode": "saved_inventory_only", "snapshot": inventory["snapshot_utc"], "profiles": counts,
                          "pending": inventory["pending"], "transfer_executed": False}, ensure_ascii=False, indent=2))
        return
    source, target = args.source.resolve(), args.target.resolve()
    if args.action == "registry":
        filter_registry(inventory, source, target, args.execute)
    elif args.action == "relocate-json":
        relocate_json(inventory, source, target, args.execute)
    elif args.action == "copy" and args.execute:
        copy_files(rows, source, target)
    else:
        report = assess(rows, source, target, args.action == "verify")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["issues"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
