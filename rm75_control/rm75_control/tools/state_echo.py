#!/usr/bin/env python3
"""Print shared-memory robot state (rostopic-like debug for state relay)."""

from __future__ import annotations

import argparse
import time

from rm75_control.control.admittance_common.state_relay import RelayStateBus


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subscribe", default="rm75_state", help="SHM segment name")
    ap.add_argument("--hz", type=float, default=2.0, help="Print rate")
    ap.add_argument("--once", action="store_true", help="Print one frame and exit")
    args = ap.parse_args()

    bus = RelayStateBus(args.subscribe)
    period = 1.0 / max(float(args.hz), 0.1)
    last_seq = -1
    try:
        while True:
            snap = bus.read()
            if snap.seq != last_seq and snap.ok:
                last_seq = int(snap.seq)
                q = snap.q_deg if snap.q_deg is not None else []
                p = snap.pose if snap.pose is not None else []
                f = snap.force_raw
                print(
                    f"seq={snap.seq} rail_m={bus.last_rail_m:.4f} "
                    f"q_deg={[round(x, 2) for x in q]} "
                    f"pos={[round(x, 4) for x in p[:3]]} Fz={f[2]:+.2f}N",
                    flush=True,
                )
                if args.once:
                    break
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        bus.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
