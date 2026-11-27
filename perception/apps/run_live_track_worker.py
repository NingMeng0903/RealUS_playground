#!/usr/bin/env python3
"""Optional live multiview track worker (dynamic updates) for RealUS.

Uses the same tracking config as offline capture. Prefer offline capture for
one-shot quality; use this for continuous pose refresh after betas are fixed.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    repo = Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
    os.chdir(repo)
    sys.path.insert(0, str(repo / "src"))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=repo / "configs/tracking/realus_dwpose_easymocap.yaml")
    ap.add_argument(
        "--connect",
        type=str,
        default=None,
        help="Override camera ZMQ connect (default: ingress.connect from --config).",
    )
    ap.add_argument("--publish-bind", type=str, default="tcp://127.0.0.1:5598")
    args, unknown = ap.parse_known_args()

    connect = args.connect
    if connect is None:
        try:
            from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig

            connect = str(MultiviewRealtimeConfig.load(args.config).ingress.connect)
        except Exception as exc:
            # Preserve the legacy physical-camera default if this thin wrapper
            # cannot inspect a malformed or unavailable config.
            connect = "tcp://127.0.0.1:17356"
            print(
                f"warning: could not resolve ingress.connect from config: {exc}; using {connect}",
                file=sys.stderr,
            )

    # Prefer dedicated live worker if present; else fall back to offline loop note.
    module = "projects.genesis_ue_sync.multiview_realtime.cli.run_multiview_track_worker"
    try:
        __import__(module)
    except Exception:
        print(
            "Live worker module unavailable; use perception/apps/run_smplx_capture.py repeatedly "
            "or install/run Among_US live worker path.",
            file=sys.stderr,
        )
        return 1

    sys.argv = [
        "run_multiview_track_worker",
        "--config",
        str(args.config),
        "--connect",
        str(connect),
        "--pub-bind",
        str(args.publish_bind),
        *unknown,
    ]
    runpy.run_module(module, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
