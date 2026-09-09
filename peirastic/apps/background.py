"""Launch an offline job away from control cores with bounded numerical pools.

Usage: python -m peirastic.apps.background -- python offline_job.py [arguments]
Only the launched job is changed. Do not use this for the controller/native solver.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("provide a command after --")
    if any(arg == "peirastic.apps.run_controller" or Path(arg).name in
           ("run_controller.py", "wbc_rt") for arg in command):
        parser.error("do not launch the controller or native solver as a background job")
    from rm75_control.control.admittance_common.observer_runtime import prepare_observer_process
    prepare_observer_process()
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    print(f"[CPU] Background CPUs={cpus}; numeric threads=1.", file=sys.stderr, flush=True)
    try:
        os.execvp(command[0], command)
    except OSError as exc:
        print(f"[ERROR] Cannot launch {command[0]}: {exc}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
