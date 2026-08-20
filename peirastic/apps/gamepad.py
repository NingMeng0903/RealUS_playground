#!/usr/bin/env python3
"""Window C: pad → filtered v_cmd → servo_twist. L3 toggles hybrid, R3 e-stop."""

from __future__ import annotations

import argparse
import time

from rm75_control.control.joint_admittance_8dof.teleop.gamepad_twist import GamepadTwistConfig
from peirastic.core.ipc import CommandClient, Status, TwistBus
from peirastic.core.modes import Mode, ModeRequest
from peirastic.sources.gamepad import GamepadTwistSource


def main() -> int:
    parser = argparse.ArgumentParser(description="peirastic gamepad source")
    parser.add_argument("--shm-prefix", default="")
    parser.add_argument("--desired-z", type=float, default=0.0)
    parser.add_argument("--hold", action="store_true", help="use servo_twist_hold")
    parser.add_argument("--trans-m-s", type=float, default=0.10)
    parser.add_argument("--rot-rad-s", type=float, default=0.60)
    args = parser.parse_args()
    prefix = str(args.shm_prefix)
    client = CommandClient(prefix=prefix)
    twist_bus = TwistBus(prefix=prefix, create=False)
    cfg = GamepadTwistConfig(trans_m_s=args.trans_m_s, rot_rad_s=args.rot_rad_s)
    src = GamepadTwistSource(cfg=cfg)
    src.start()
    vel_mode = Mode.SERVO_TWIST_HOLD if args.hold else Mode.SERVO_TWIST
    hybrid = False
    client.set_mode(ModeRequest(vel_mode, {}))
    print("[MODE] SERVO_TWIST" + ("_HOLD" if args.hold else ""), flush=True)
    try:
        while True:
            snap = src.snapshot()
            twist_bus.write(
                snap["twist"],
                axes=snap["axes"],
                buttons=snap["buttons"],
                hz=snap["hz"],
                connected=snap["connected"],
                l3=snap["l3"],
                r3=snap["r3"],
            )
            if snap["r3_edge"]:
                client.estop()
                print("[ESTOP] pad R3", flush=True)
                return 130
            if snap["l3_edge"]:
                hybrid = not hybrid
                if hybrid:
                    client.set_mode(
                        ModeRequest(
                            Mode.TRACK_HYBRID,
                            {"reference": "hold", "desired_z": float(args.desired_z)},
                        )
                    )
                    print("[MODE] TRACK_HYBRID", flush=True)
                else:
                    client.set_mode(ModeRequest(vel_mode, {}))
                    print("[MODE] " + vel_mode.name, flush=True)
            tel = client.snapshot()
            if int(tel["status"]) == int(Status.ESTOP):
                print("[ESTOP] " + str(tel["msg"]), flush=True)
                return 130
            time.sleep(0.005)
    except KeyboardInterrupt:
        client.stop()
        print("[STOP] interrupted", flush=True)
        return 0
    finally:
        src.close()
        twist_bus.close()
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
