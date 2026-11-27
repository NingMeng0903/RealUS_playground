#!/usr/bin/env python3
"""Send a reviewed command to the current RealUS UE session and wait for its result."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from projects.genesis_ue_sync.integrations.ue.session import EditorCommand, EditorSessionPaths, wait_for_command_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, default=Path(os.environ.get("SESSION_DIR", ROOT / "outputs/ue_sessions/realus_fullflow_v1")))
    parser.add_argument("--command", required=True, choices=["start_pie", "stop_pie", "resume_camera_preview", "stop_session_watcher"])
    parser.add_argument("--timeout-s", type=float, default=60)
    parser.add_argument(
        "--ready-timeout-s",
        type=float,
        default=None,
        help="For start_pie, wait this many seconds for the UE game world before reporting failure.",
    )
    args = parser.parse_args()
    paths = EditorSessionPaths(args.session_dir.expanduser().resolve())
    payload = {}
    if args.ready_timeout_s is not None and args.command == "start_pie":
        payload["ready_timeout_s"] = float(args.ready_timeout_s)
    command = EditorCommand(command_type=args.command, payload=payload)
    command.write(paths)
    result = wait_for_command_result(paths, command.request_id, timeout_s=args.timeout_s)
    print(json.dumps({"success": result.success, "detail": result.detail, "payload": result.payload}, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
