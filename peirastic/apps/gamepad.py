#!/usr/bin/env python3
"""Window C: pad → filtered v_cmd → servo_twist.

The pad may stay connected. It drives only in pad-owned SERVO_TWIST /
SERVO_TWIST_HOLD or its own L3 pad-hybrid. MOVEJ, Cartesian,
TRACK_CARTESIAN, commanded cartesian_velocity, and a running program
outrank the sticks; R3 e-stop still wins. When the pad does not own
the mode it must not write zeros over the command twist bus.

L3 toggles force-velocity hybrid: tool-Z from peirastic/configs/force.yaml,
other axes stay on the pad. Y starts Window-8 SMPL-X capture. B runs the
R_SUPFEMV program. Motion requires a live Bluetooth pad.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

_REPO = Path(os.environ.get("REALUS_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
for _p in (_REPO, _REPO / "rm75_control", _REPO / "src"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from rm75_control.control.joint_admittance_8dof.teleop.gamepad_twist import (
    GamepadTwistConfig,
)
from peirastic.core.ipc import CommandClient, Status, TwistBus
from peirastic.core.modes import Mode, ModeRequest, try_mode
from peirastic.core.session import pad_may_drive
from peirastic.realman8dof.force.config import desired_z_n
from peirastic.sources.gamepad import GamepadTwistSource

from peirastic.apps.vessel_scan import (  # noqa: E402
    program_is_running,
    try_start_vessel_scan,
)
from perception.capture_flow import (  # noqa: E402
    CaptureResult,
    is_capture_progress_line,
    try_start_smplx_capture,
)


def _fmt_vec(vec, n=3) -> str:
    return " ".join(f"{float(v):+.3f}" for v in list(vec)[:n])


def _green(msg: str) -> str:
    return f"\033[32m{msg}\033[0m"


def pad_link_event(prev: bool | None, live: bool) -> str | None:
    """Announce Bluetooth only on connect, or after a live pad drops."""

    if prev is None:
        return "[PAD] bluetooth live" if live else None
    if live == prev:
        return None
    return "[PAD] bluetooth live" if live else "[PAD] bluetooth lost"


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
    parser.add_argument(
        "--no-vessel-b",
        action="store_true",
        help="do not bind Xbox B to the R_SUPFEMV approach/scan program",
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
    hybrid_payload = {
        "reference": "pad",
        "use_tff_split": True,
        "label": "track_hybrid_pad",
        "filter": False,  # pad already LPF+jerk at source; force Z never filtered
    }
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
    if not quiet:
        print(f"[STATE] L3 hybrid Fz*={fz:.2f}N (peirastic/configs/force.yaml)", flush=True)
        if args.no_capture_y:
            print("[STATE] Y capture disabled (--no-capture-y)", flush=True)
        else:
            print(
                "[STATE] Y = SMPL-X (burst→beta, 1 sync frame→pose) + preview + Genesis",
                flush=True,
            )
        if args.no_vessel_b:
            print("[STATE] B vessel scan disabled (--no-vessel-b)", flush=True)
        else:
            print(
                "[STATE] B = 8DOF TRACK 5cm standoff → hybrid close → 10cm R_SUPFEMV scan",
                flush=True,
            )
        print("[STATE] motion requires live bluetooth pad (usb/missing inhibited)", flush=True)
        pad = getattr(src, "pad", None)
        describe = getattr(pad, "describe", None)
        if describe is not None:
            print("[STATE] " + str(describe()), flush=True)
    else:
        print("[STATE] ready", flush=True)
    last_live = None
    try:
        tel0 = client.snapshot()
        if try_mode(tel0.get("mode")) is None:
            print("[PAD] waiting for Window A", flush=True)
            while try_mode(tel0.get("mode")) is None:
                time.sleep(0.05)
                tel0 = client.snapshot()
        if pad_may_drive(int(tel0.get("mode") or 0), label=str(tel0.get("msg") or "")):
            client.set_mode(ModeRequest(vel_mode, {"secondary": "track", "filter": False}))
        elif not quiet:
            print(
                "[PAD] connected — command mode has priority "
                f"(live={tel0.get('msg') or tel0.get('mode')}); R3 e-stop still live",
                flush=True,
            )
        while True:
            snap = src.snapshot()
            live = bool(snap["connected"]) and bool(snap["armed"])
            program = program_is_running()
            tel = client.snapshot()
            pad_drive = live and pad_may_drive(
                int(tel.get("mode") or 0),
                program=program,
                label=str(tel.get("msg") or ""),
            )
            # Commanded cartesian_velocity shares this bus. Do not write
            # zeros over v* when the pad does not own the mode.
            twist_bus.write(
                snap["twist"] if pad_drive else None,
                axes=snap["axes"],
                buttons=snap["buttons"] if live else None,
                hz=snap["hz"] if live else float("nan"),
                connected=live,
                l3=bool(snap["l3"]) if pad_drive else False,
                r3=bool(snap["r3"]) if live else False,
            )
            link = pad_link_event(last_live, live)
            last_live = live
            if link:
                print(link, flush=True)
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
                            "mesh publishes after pose; preview PNGs keep writing"
                        ),
                        flush=True,
                    )
                else:
                    print(_green(f"[CAPTURE] Y ignored ({start.reason})"), flush=True)
            if snap.get("b_edge") and not args.no_vessel_b:
                if program_is_running():
                    print("[VESSEL] B ignored (busy)", flush=True)
                elif not live:
                    print("[VESSEL] B ignored (no live bluetooth)", flush=True)
                else:
                    print("[VESSEL] B — resolving R_SUPFEMV plan", flush=True)
                    refuse = try_start_vessel_scan(
                        client,
                        repo=_REPO,
                        on_log=lambda line: print(line, flush=True),
                    )
                    if refuse:
                        print(f"[VESSEL] B ignored ({refuse})", flush=True)
                    else:
                        print(
                            "[VESSEL] B — 8DOF TRACK 5cm → hybrid close → 10cm R_SUPFEMV",
                            flush=True,
                        )
            now = time.monotonic()
            if pad_drive and snap["l3_edge"] and (now - last_l3_s) > 0.15:
                last_l3_s = now
                hybrid = not hybrid
                if hybrid:
                    client.set_mode(ModeRequest(Mode.TRACK_HYBRID, hybrid_payload))
                    print(f"[MODE] TRACK_HYBRID pad+force Z={fz:.2f}N", flush=True)
                else:
                    client.set_mode(ModeRequest(vel_mode, {"secondary": "track", "filter": False}))
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
