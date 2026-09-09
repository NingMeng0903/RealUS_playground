#!/usr/bin/env python3
"""Window A: start C++ inner + peirastic outer. CSV off unless --log-csv."""

from __future__ import annotations

import argparse
from pathlib import Path


_DEFAULT_CONTROLLER_YAML = (
    Path(__file__).resolve().parents[1] / "configs" / "controller.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="peirastic.realman8dof controller")
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONTROLLER_YAML,
    )
    parser.add_argument(
        "--log-csv",
        nargs="?",
        const="auto",
        default=None,
        help="200 Hz CSV including force. Bare flag writes apps/logs/peirastic/run_*.csv",
    )
    parser.add_argument("--shm-prefix", default="", help="test isolation prefix")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-panel", action="store_true")
    args = parser.parse_args()

    # Keep the process's startup/main thread on observer CPUs before importing
    # the daemon (which imports NumPy/SDK code and starts helper threads).  The
    # control loop later narrows its own calling thread to timing.control_cpu;
    # the native child keeps its existing explicit native_cpu binding.
    from rm75_control.control.admittance_common.cpu_resources import (
        prepare_background_cpus,
    )
    prepare_background_cpus(config_path=args.config)
    # Controller keeps its scheduling priority; only numerical pools are bounded.
    from rm75_control.control.admittance_common.observer_runtime import (
        limit_numeric_threads,
    )
    limit_numeric_threads()

    from peirastic.realman8dof.daemon import run_service
    from realus_clock import get_clock
    get_clock()
    return run_service(
        args.config,
        shm_prefix=str(args.shm_prefix),
        log_csv=args.log_csv,
        dry_run=bool(args.dry_run),
        panel=not args.no_panel,
    )


if __name__ == "__main__":
    raise SystemExit(main())
