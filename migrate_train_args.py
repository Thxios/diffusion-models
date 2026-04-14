"""Migrate outputs/**/train_args.json files to the new TrainArgs schema.

Applies:
  - Rename 'lr_schduler_cfg' → 'lr_scheduler_cfg'  (typo fix)
  - Remove 'device' field                            (Accelerate-managed now)
  - Remove any unrecognised fields                   (e.g. old class_condition → class_conditioning)

Usage:
  python migrate_train_args.py            # dry-run (shows diffs, writes nothing)
  python migrate_train_args.py --apply    # apply in-place
"""

import os
import glob
import json
import dataclasses
import sys
import copy
from typing import Any


def get_known_fields():
    # Import here so this script can run standalone
    sys.path.insert(0, os.path.dirname(__file__))
    from train import TrainArgs
    return {f.name for f in dataclasses.fields(TrainArgs)}


def migrate(d: dict, known_fields: set) -> tuple[dict, list[str]]:
    """Return (migrated_dict, list_of_change_descriptions)."""
    d = copy.deepcopy(d)
    changes = []

    # 1. Typo fix
    if 'lr_schduler_cfg' in d:
        d['lr_scheduler_cfg'] = d.pop('lr_schduler_cfg')
        changes.append("renamed 'lr_schduler_cfg' → 'lr_scheduler_cfg'")

    # 2. Remove device
    if 'device' in d:
        del d['device']
        changes.append("removed 'device'")

    # 3. Remove unknown fields
    unknown = [k for k in list(d.keys()) if k not in known_fields]
    for k in unknown:
        del d[k]
        changes.append(f"removed unknown field '{k}'")

    return d, changes


def main(apply: bool = False):
    known = get_known_fields()

    paths = sorted(glob.glob('outputs/**/train_args.json', recursive=True))
    if not paths:
        print('No train_args.json files found under outputs/.')
        return

    needs_update = []
    for path in paths:
        with open(path) as f:
            try:
                d = json.load(f)
            except json.JSONDecodeError as e:
                print(f'  SKIP (JSON error): {path}: {e}')
                continue

        new_d, changes = migrate(d, known)
        if changes:
            needs_update.append((path, new_d, changes))

    if not needs_update:
        print(f'All {len(paths)} train_args.json files are already up to date.')
        return

    # ── Report ──────────────────────────────────────────────────────────
    print(f'Found {len(needs_update)} file(s) needing migration:\n')
    for path, new_d, changes in needs_update:
        print(f'  {path}')
        for c in changes:
            print(f'    • {c}')

    # ── Validation: each migrated dict must parse without error ─────────
    from train import TrainArgs
    errors = []
    for path, new_d, changes in needs_update:
        try:
            TrainArgs(**new_d)
        except TypeError as e:
            errors.append(f'  {path}: {e}')

    if errors:
        print('\nValidation FAILED for:')
        for e in errors:
            print(e)
        print('\nAborting — fix errors before applying.')
        sys.exit(1)

    print(f'\nValidation passed for all {len(needs_update)} file(s).')

    if not apply:
        print('\nDry-run complete. Pass --apply to write changes.')
        return

    # ── Apply ────────────────────────────────────────────────────────────
    for path, new_d, changes in needs_update:
        with open(path, 'w') as f:
            json.dump(new_d, f, indent=2)
        print(f'  Updated: {path}')

    print(f'\nDone. Migrated {len(needs_update)} file(s).')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
