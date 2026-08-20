#!/usr/bin/env python3
"""Window C: call track_cartesian with an ellipse reference."""

from __future__ import annotations

import argparse
import time

from peirastic.core.ipc import CommandClient, Status
from peirastic.core.modes import Mode, ModeRequest


def main() -> int:
    parser = argparse.ArgumentParser(description="peirastic ellipse track")
    parser.add_argument("--shm-prefix", default="")
    parser.add_argument("--x-pp-cm", type=float, default=10.0)
    parser.add_argument("--y-pp-cm", type=float, default=30.0)
    parser.add_argument("--max-vel-cm-s", type=float, default=4.0)
    parser.add_argument("--duration-s", type=float, default=40.0)
    args = parser.parse_args()
    client = CommandClient(prefix=str(args.shm_prefix))
    client.set_mode(
        ModeRequest(
            Mode.TRACK_CARTESIAN,
            {
                "reference": "ellipse",
                "x_pp_cm": float(args.x_pp_cm),
                "y_pp_cm": float(args.y_pp_cm),
                "max_vel_cm_s": float(args.max_vel_cm_s),
                "duration_s": float(args.duration_s),
            },
        )
    )
    print("[MODE] TRACK_CARTESIAN ellipse", flush=True)
    try:
        while True:
            tel = client.snapshot()
            st = int(tel["status"])
            if st in (int(Status.DONE), int(Status.STOPPED)):
                print("[OK] done", flush=True)
                return 0
            if st in (int(Status.ERROR), int(Status.ESTOP)):
                print("[WARN] " + str(tel["msg"]), flush=True)
                return 1
            time.sleep(0.05)
    except KeyboardInterrupt:
        client.stop()
        print("[STOP] interrupted", flush=True)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
