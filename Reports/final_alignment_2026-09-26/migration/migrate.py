"""Final-paper migration, reusing the existing metadata inventory/copy implementation.

The default is a plan. Copying requires --execute and never overwrites existing
files. Scientific outputs are copied, not regenerated. Source data are read-only.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('existing_migration', HERE.parents[1] / 'migration/migrate_artifacts.py')
implementation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(implementation)


def registry(rows, source, target, execute):
    database = target / '_Data/manifest.db'
    if not database.exists():
        implementation.filter_registry({'files': rows}, source, target, execute)
        return
    selected = {row['source_relative'].removeprefix('_Data/') for row in rows
                if row['source_relative'].startswith('_Data/')
                and row['source_relative'] == row['destination_relative']}
    with sqlite3.connect((source / '_Data/manifest.db').as_uri() + '?mode=ro', uri=True) as connection:
        candidates = [row for row in connection.execute('SELECT * FROM manifest')
                      if row[3] in selected and (target / '_Data' / row[3]).is_file()
                      and (target / '_Data' / row[3]).stat().st_size == row[5]]
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        existing = {row[0]: row for row in connection.execute('SELECT * FROM manifest')}
    conflicts = [row[0] for row in candidates if row[0] in existing and row != existing[row[0]]]
    if conflicts:
        raise ValueError(f'Existing destination registry identities differ; nothing overwritten: {conflicts}')
    missing = [row for row in candidates if row[0] not in existing]
    if execute and missing:
        with sqlite3.connect(database) as connection:
            connection.executemany('INSERT INTO manifest VALUES (?,?,?,?,?,?,?,?,?,?)', missing)
    result = {'eligible_source_entries': len(candidates), 'existing_destination_entries': len(existing),
              'new_entries': len(missing), 'execute': execute, 'existing_entries_overwritten': False,
              'hashes': 'Existing identifiers retained; no new hashing'}
    if execute:
        (HERE / 'REGISTRY_RECEIPT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='plan', choices=['inventory', 'plan', 'copy', 'verify', 'registry'])
    parser.add_argument('--source', type=Path, required=True, help='Original project root')
    parser.add_argument('--target', type=Path, default=HERE.parents[2])
    parser.add_argument('--profiles', default='analysis,regeneration')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.action == 'inventory':
        implementation.scan(args.source, json.loads((HERE / 'selection.json').read_text()), HERE)
        return
    inventory = json.loads((HERE / 'inventory.json').read_text())
    rows = implementation.selected_rows(inventory, args.profiles)
    if args.action == 'registry':
        registry(rows, args.source, args.target, args.execute)
        return
    report = implementation.assess(rows, args.source, args.target, args.action == 'verify')
    free = shutil.disk_usage(args.target).free
    needed = sum(row['bytes'] for row in rows if not (args.target / row['destination_relative']).exists())
    report.update(free_bytes=free, required_copy_bytes=needed, remaining_bytes_after_copy=free-needed,
                  profiles=args.profiles, transfer_executed=False)
    if args.action == 'copy' and args.execute:
        if free - needed < 30 * 2**30:
            raise SystemExit('Copy would leave less than30GiB free; no transfer performed.')
        implementation.copy_files(rows, args.source, args.target)
        report = implementation.assess(rows, args.source, args.target, True)
        report.update(transfer_executed=True, profiles=args.profiles,
                      operation='independent shutil.copy2 files; no hardlinks/symlinks; source preserved')
        (HERE / 'COPY_RECEIPT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['issues']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
