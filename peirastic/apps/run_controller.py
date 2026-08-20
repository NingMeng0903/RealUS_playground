#!/usr/bin/env python3
"""Window A: start C++ inner + peirastic outer. CSV off unless --log-csv."""

from __future__ import annotations

import argparse
from pathlib import Path

from peirastic.realman8dof.daemon import run_service


def main() -> int:
    parser = argparse.ArgumentParser(description="peirastic.realman8dof controller")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/configs/joint_admittance_8dof.yaml"),
    )
    parser.add_argument("--log-csv", default=None, help="off by default")
    parser.add_argument("--shm-prefix", default="", help="test isolation prefix")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-panel", action="store_true")
    args = parser.parse_args()
    return run_service(
        args.config,
        shm_prefix=str(args.shm_prefix),
        log_csv=args.log_csv,
        dry_run=bool(args.dry_run),
        panel=not args.no_panel,
    )


if __name__ == "__main__":
    raise SystemExit(main())
