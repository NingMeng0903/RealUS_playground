#!/usr/bin/env python3
"""Window C: goto_joints or movej."""

from __future__ import annotations

import argparse
import time

import numpy as np

from peirastic.core.ipc import CommandClient, Status
from peirastic.core.modes import Mode, ModeRequest


def main() -> int:
    parser = argparse.ArgumentParser(description="peirastic joint modes")
    parser.add_argument("--shm-prefix", default="")
    parser.add_argument("--mode", choices=("goto", "movej"), default="movej")
    parser.add_argument("--q-deg", required=True, help="8 comma-separated degrees")
    parser.add_argument("--duration-s", type=float, default=None)
    args = parser.parse_args()
    q_deg = np.array([float(x) for x in str(args.q_deg).split(",")], dtype=float)
    if q_deg.size != 8:
        raise SystemExit("need 8 joints (rail m as deg0? use rad via --q-rad)")
    q = q_deg.copy()
    q[0] = float(q_deg[0])  # first number is rail metres if |q0|<2 else treat as mm
    if abs(q[0]) > 2.0:
        q[0] *= 0.001
    q[1:] = np.deg2rad(q_deg[1:])
    mode = Mode.GOTO_JOINTS if args.mode == "goto" else Mode.MOVEJ
    client = CommandClient(prefix=str(args.shm_prefix))
    payload = {"q_target": [float(x) for x in q]}
    if args.duration_s is not None:
        payload["duration_s"] = float(args.duration_s)
    client.set_mode(ModeRequest(mode, payload))
    print(f"[MODE] {mode.name}", flush=True)
    try:
        while True:
            tel = client.snapshot()
            st = int(tel["status"])
            if st in (int(Status.DONE), int(Status.STOPPED)):
                print("[OK] arrived", flush=True)
                return 0
            if st in (int(Status.ERROR), int(Status.ESTOP)):
                print("[WARN] " + str(tel["msg"]), flush=True)
                return 1
            time.sleep(0.05)
    except KeyboardInterrupt:
        client.stop()
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
