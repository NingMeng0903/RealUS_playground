#!/usr/bin/env python3
"""Enumerate connected cameras and pin each serial to a stable ``camN`` alias.

Usage::

    conda activate /media/camp/EXT_DRIVE/envs/camera_calib
    python scripts/discover_cameras.py

Behaviour:
- New serials are appended to ``configs/cameras.yaml`` with the next free
  ``camN``. The binding is permanent after that.
- Existing serials keep their alias untouched.
- Prints a table of every camera the config knows about, with online status.

Pass ``--dry-run`` to see what would change without modifying the yaml.
"""
from __future__ import annotations

import argparse
import os
import site
import sys
from pathlib import Path


if os.environ.get("PYTHONNOUSERSITE") != "1":
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

_user_site = site.getusersitepackages()
sys.path = [p for p in sys.path if not p.startswith(_user_site)]

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from multicam_calib.devices.discovery import resolve_roster  # noqa: E402
from multicam_calib.io.config import cameras_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Do not write to cameras.yaml")
    args = ap.parse_args()

    resolved = resolve_roster(mutate_config=not args.dry_run)

    print(f"Roster file: {cameras_path()}")
    print(f"Registered cameras: {len(resolved)}")
    print()
    print(f"{'alias':<10}  {'serial':<16}  {'driver':<10}  {'model':<40}  status")
    print("-" * 96)
    for r in resolved:
        status = "ONLINE" if r.online else "offline"
        model = r.entry.model or (r.discovered.model if r.discovered else "")
        print(f"{r.entry.alias:<10}  {r.entry.serial:<16}  {r.entry.driver:<10}  {model:<40}  {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
