#!/usr/bin/env python3
"""Window C: pad → filtered v_cmd → servo_twist.

L3 toggles force-velocity hybrid: tool-Z from peirastic/configs/force.yaml,
other axes stay on the pad. R3 e-stop. Y starts Window-8 SMPL-X capture
(preview PNGs + Genesis orange mesh), same job as the remote GUI button.

Motion is sent only while a live Bluetooth pad is present (kernel Bus /
SDL GUID). USB and a missing pad are inhibited so their rest axes cannot
alias the xpadneo trigger map. Pygame and kernel names may differ.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

_REPO = Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
for _p in (_REPO, _REPO / "rm75_control", _REPO / "src"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from rm75_control.control.joint_admittance_8dof.teleop.gamepad_twist import (
    GamepadTwistConfig,
)
from peirastic.core.ipc import CommandClient, Status, TwistBus
from peirastic.core.modes import Mode, ModeRequest
from peirastic.realman8dof.force.config import desired_z_n
from peirastic.sources.gamepad import GamepadTwistSource

from perception.capture_flow import (  # noqa: E402
    CaptureResult,
    is_capture_progress_line,
    try_start_smplx_capture,
)


def _fmt_vec(vec, n=3) -> str:
    return " ".join(f"{float(v):+.3f}" for v in list(vec)[:n])


def _green(msg: str) -> str:
    return f"\033[32m{msg}\033[0m"


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
    parser.add_argument(
        "--no-capture-y",
        action="store_true",
        help="do not bind Xbox Y to SMPL-X capture",
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
    if args.no_capture_y:
        print("[STATE] Y capture disabled (--no-capture-y)", flush=True)
    else:
        print("[STATE] Y = SMPL-X capture + preview PNGs + Genesis publish", flush=True)
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
            if snap.get("y_edge") and not args.no_capture_y:
                def _on_capture_done(result: CaptureResult) -> None:
                    if result.ok:
                        print(
                            _green(f"[CAPTURE] done {result.run_name} → {result.moment_dir}"),
                            flush=True,
                        )
                    elif result.quality_rejection is not None:
                        print(
                            _green(
                                f"[CAPTURE] quality rejected {result.run_name} "
                                f"see {result.moment_dir}"
                            ),
                            flush=True,
                        )
                    else:
                        print(
                            _green(
                                f"[CAPTURE] failed {result.run_name} "
                                f"rc={result.returncode} {result.error} "
                                f"log={result.log_path}"
                            ),
                            flush=True,
                        )

                def _on_capture_log(line: str) -> None:
                    if is_capture_progress_line(line):
                        print(_green(f"[CAPTURE] {line.strip()}"), flush=True)

                start = try_start_smplx_capture(
                    label="xbox_y",
                    repo=_REPO,
                    on_done=_on_capture_done,
                    on_log=_on_capture_log,
                )
                if start.started:
                    print(
                        _green(f"[CAPTURE] Y — SMPL-X + Genesis  run={start.run_name}"),
                        flush=True,
                    )
                    print(
                        _green(
                            "[CAPTURE] DWPose TensorRT/CUDA + EasyMocap GPU — "
                            "leave Cam and this window running"
                        ),
                        flush=True,
                    )
                else:
                    print(_green(f"[CAPTURE] Y ignored ({start.reason})"), flush=True)
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
                    f"l3={int(snap['l3'])} r3={int(snap['r3'])} y={int(snap['y'])} "
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
