#!/usr/bin/env python3
"""Window C: pad → filtered v_cmd → servo_twist.

L3 toggles force-velocity hybrid: tool-Z from peirastic/configs/force.yaml,
other axes stay on the pad. R3 e-stop.

Motion is sent only while a live Bluetooth pad is present (kernel Bus /
SDL GUID). USB and a missing pad are inhibited so their rest axes cannot
alias the xpadneo trigger map. Pygame and kernel names may differ.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

from rm75_control.control.joint_admittance_8dof.teleop.gamepad_twist import (
    GamepadTwistConfig,
)
from peirastic.core.ipc import CommandClient, Status, TwistBus
from peirastic.core.modes import Mode, ModeRequest
from peirastic.realman8dof.force.config import desired_z_n
from peirastic.sources.gamepad import GamepadTwistSource


def _fmt_vec(vec, n=3) -> str:
    return " ".join(f"{float(v):+.3f}" for v in list(vec)[:n])


def main() -> int:
    parser = argparse.ArgumentParser(description="peirastic gamepad source")
    parser.add_argument("--shm-prefix", default="")
    parser.add_argument("--desired-z", type=float, default=None)
    parser.add_argument("--trans-m-s", type=float, default=None)
    parser.add_argument("--rot-rad-s", type=float, default=None)
    parser.add_argument("--hold", action="store_true", help="use servo_twist_hold")
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=True,
        help="no periodic v_cmd log (default)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print a 5 Hz pad heartbeat",
    )
    args = parser.parse_args()
    quiet = bool(args.quiet) and not bool(args.verbose)
    twist_cfg = GamepadTwistConfig()
    if args.trans_m_s is not None or args.rot_rad_s is not None:
        twist_cfg = replace(
            twist_cfg,
            trans_m_s=float(args.trans_m_s) if args.trans_m_s is not None else twist_cfg.trans_m_s,
            rot_rad_s=float(args.rot_rad_s) if args.rot_rad_s is not None else twist_cfg.rot_rad_s,
        )
    fz = float(args.desired_z) if args.desired_z is not None else desired_z_n()
    hybrid_payload = {"reference": "pad", "use_tff_split": True}
    if args.desired_z is not None:
        hybrid_payload["desired_z"] = float(args.desired_z)
    prefix = str(args.shm_prefix)
    client = CommandClient(prefix=prefix)
    twist_bus = TwistBus(prefix=prefix, create=False)
    src = GamepadTwistSource(cfg=twist_cfg)
    src.start()
    vel_mode = Mode.SERVO_TWIST_HOLD if args.hold else Mode.SERVO_TWIST
    hybrid = False
    last_l3_s = 0.0
    last_log_s = 0.0
    client.set_mode(ModeRequest(vel_mode, {}))
    pad = getattr(src, "pad", None)
    describe = getattr(pad, "describe", None)
    print("[MODE] SERVO_TWIST" + ("_HOLD" if args.hold else ""), flush=True)
    print(f"[STATE] L3 hybrid Fz*={fz:.2f}N (peirastic/configs/force.yaml)", flush=True)
    print("[STATE] motion requires live bluetooth pad (usb/missing inhibited)", flush=True)
    if describe is not None:
        print("[STATE] " + str(describe()), flush=True)
    last_live = None
    try:
        while True:
            snap = src.snapshot()
            live = bool(snap["connected"]) and bool(snap["armed"])
            twist_bus.write(
                snap["twist"] if live else [0.0] * 6,
                axes=snap["axes"],
                buttons=snap["buttons"] if live else None,
                hz=snap["hz"] if live else float("nan"),
                connected=live,
                l3=bool(snap["l3"]) if live else False,
                r3=bool(snap["r3"]) if live else False,
            )
            if last_live is None or live != last_live:
                last_live = live
                why = str(snap.get("transport") or "none")
                if live:
                    print(f"[PAD] bluetooth live transport={why} — motion enabled", flush=True)
                else:
                    print(
                        f"[PAD] no live bluetooth (transport={why}) — motion inhibited",
                        flush=True,
                    )
            if live and snap["r3_edge"]:
                print("[ESTOP] pad R3 — zero rail then stop", flush=True)
                client.estop()
                return 130
            now = time.monotonic()
            if live and snap["l3_edge"] and (now - last_l3_s) > 0.15:
                last_l3_s = now
                hybrid = not hybrid
                if hybrid:
                    client.set_mode(ModeRequest(Mode.TRACK_HYBRID, hybrid_payload))
                    print(f"[MODE] TRACK_HYBRID pad+force Z={fz:.2f}N", flush=True)
                else:
                    client.set_mode(ModeRequest(vel_mode, {}))
                    print("[MODE] " + vel_mode.name, flush=True)
            if not quiet and (now - last_log_s) >= 0.20:
                last_log_s = now
                twv = snap["twist"]
                ax = snap["axes"]
                print(
                    f"[PAD] {snap['layout']} hz={snap['hz']:5.1f} "
                    f"v={_fmt_vec(twv[:3])} w={_fmt_vec(twv[3:6])} "
                    f"ax={_fmt_vec(ax, 6)} "
                    f"l3={int(snap['l3'])} r3={int(snap['r3'])} "
                    f"armed={int(snap['armed'])} hybrid={int(hybrid)}",
                    flush=True,
                )
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
