#!/usr/bin/env python3
"""Window C: pad → filtered v_cmd → servo_twist.

The pad may stay connected. It drives only in pad-owned SERVO_TWIST /
SERVO_TWIST_HOLD or its own L3 pad-hybrid. MOVEJ, Cartesian,
TRACK_CARTESIAN, commanded cartesian_velocity, and a running program
outrank the sticks; R3 e-stop still wins. When the pad does not own
the mode it must not write zeros over the command twist bus.

L3 toggles force-velocity hybrid. Default is the old TFF tool-Z plus
icra torque_tilt My. ``--hybrid-law ultrapoc`` adds delayed-KF α on the
stick and image ωy on top of My. ``--hybrid-log`` writes a jsonl of
stick, α, moment, and image on each L3. Y starts Window-8 SMPL-X
capture. B runs the R_SUPFEMV program. Motion requires a live
Bluetooth pad.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

if __name__ == "__main__":
    from rm75_control.control.admittance_common.observer_runtime import limit_numeric_threads
    from rm75_control.control.admittance_common.cpu_resources import prepare_background_cpus
    # Preserve pad/R3 priority while keeping its libraries off the control cores.
    limit_numeric_threads()
    prepare_background_cpus()

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
from peirastic.realman8dof.modes.pad_visual import (
    PAD_HYBRID_LAWS,
    allocate_pad_log_path,
    default_pad_log_path,
    pad_hybrid_payload,
)
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


def _install_mode(client: CommandClient, req: ModeRequest) -> None:
    """Send a mode and wait for Window A's post-on_enter acknowledgement."""

    seq = client.set_mode(req)
    ret = client.wait_installed(seq, req.mode)
    if ret != 0:
        raise RuntimeError(f"{req.mode.name} install failed ({ret})")


def pad_link_event(prev: bool | None, live: bool) -> str | None:
    """Announce Bluetooth only on connect, or after a live pad drops."""

    if prev is None:
        return "[PAD] bluetooth live" if live else None
    if live == prev:
        return None
    return "[PAD] bluetooth live" if live else "[PAD] bluetooth lost"


def pad_hybrid_active(snapshot: dict) -> bool:
    """Resync L3 after an external program has returned control to the pad."""
    return (try_mode(snapshot.get("mode")) == Mode.TRACK_HYBRID
            and "pad" in str(snapshot.get("msg") or "").lower())


# ac2d TRACK_HYBRID on_enter is slow. A 150 ms debounce expires while
# wait_installed blocks, so a stick-click bounce latched in GamepadTwistSource
# immediately toggles back to SERVO_TWIST_HOLD.
L3_TOGGLE_GUARD_S = 0.40


def l3_accept_toggle(
    *,
    edge: bool,
    held: bool,
    now: float,
    last_s: float,
    released: bool,
) -> tuple[bool, bool]:
    """One physical click → one L3 toggle.

    Re-arm only after L3 has been up for ``L3_TOGGLE_GUARD_S`` since the last
    accepted edge. A brief open during mechanical chatter does not re-arm.
    """
    if (not held) and (now - last_s) >= L3_TOGGLE_GUARD_S:
        released = True
    if edge and released and (now - last_s) >= L3_TOGGLE_GUARD_S:
        return True, False
    return False, released


def drain_pad_edges(src) -> None:
    """Drop edges latched while the main loop was blocked in wait_installed."""
    src.snapshot()


def main() -> int:
    parser = argparse.ArgumentParser(description="peirastic gamepad source")
    parser.add_argument("--shm-prefix", default="")
    parser.add_argument("--desired-z", type=float, default=None)
    parser.add_argument("--trans-m-s", type=float, default=None)
    parser.add_argument("--rot-rad-s", type=float, default=None)
    parser.add_argument("--hold", action="store_true", help="use servo_twist_hold")
    parser.add_argument("--force-profile", choices=("baseline", "icra"), default="baseline",
                        help="legacy L3 tilt overlay; ignored by --hybrid-law ac2d/ultrapoc")
    parser.add_argument(
        "--hybrid-law",
        choices=PAD_HYBRID_LAWS,
        default="legacy",
        help="L3 hybrid: legacy = TFF Z + tilt My (default); "
             "ultrapoc = α·v* and image/My; ac2d = comparison Z+My",
    )
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
    parser.add_argument(
        "--hybrid-log",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help="record pad-hybrid jsonl (stick, α, My, image ω, regions). "
             "default apps/logs/pad_hybrid/<stamp>.jsonl; new file each L3",
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
    hybrid_law = str(args.hybrid_law)
    log_requested = None
    if args.hybrid_log is not None:
        requested = args.hybrid_log
        if str(requested).strip().lower() in ("", "auto"):
            log_requested = default_pad_log_path(_REPO)
        else:
            log_requested = Path(requested).expanduser()
            if not log_requested.is_absolute():
                log_requested = _REPO / log_requested
            log_requested = log_requested.resolve()
        print(f"[PAD-LOG] L3 hybrid jsonl → {log_requested} (new file each L3)", flush=True)

    def make_hybrid_payload():
        log_path = None
        if log_requested is not None:
            allocated = allocate_pad_log_path(log_requested)
            log_path = {
                "path": str(allocated),
                "hybrid_law": hybrid_law,
                "trans_m_s": float(twist_cfg.trans_m_s),
                "rot_rad_s": float(twist_cfg.rot_rad_s),
                "hold": bool(args.hold),
            }
            print(f"[PAD-LOG] writing {allocated}", flush=True)
        return pad_hybrid_payload(
            hybrid_law,
            desired_z=fz,
            log_path=log_path,
            force_profile=args.force_profile,
        )
    prefix = str(args.shm_prefix)
    client = CommandClient(prefix=prefix)
    twist_bus = TwistBus(prefix=prefix, create=False)
    src = GamepadTwistSource(cfg=twist_cfg)
    src.start()
    vel_mode = Mode.SERVO_TWIST_HOLD if args.hold else Mode.SERVO_TWIST
    hybrid = False
    last_l3_s = 0.0
    l3_released = True
    last_log_s = 0.0
    if not quiet:
        print(
            f"[STATE] L3 hybrid law={hybrid_law} Fz*={fz:.2f}N "
            f"(legacy=TFF Z+tilt My; ultrapoc=slew α·v* + My⊕image)",
            flush=True,
        )
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
            try:
                # The pad inherits the persistent session DOF.  It is a mode
                # source, not a structure-selection command.
                _install_mode(client, ModeRequest(vel_mode, {"filter": False}))
            except Exception as exc:
                print(f"[MODE] unable to install {vel_mode.name}: {exc}", flush=True)
                return 2
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
            hybrid = pad_hybrid_active(tel)
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
            fire_l3, l3_released = l3_accept_toggle(
                edge=bool(snap["l3_edge"]),
                held=bool(snap["l3"]),
                now=now,
                last_s=last_l3_s,
                released=l3_released,
            )
            if pad_drive and fire_l3:
                hybrid = not hybrid
                try:
                    if hybrid:
                        _install_mode(client, ModeRequest(Mode.TRACK_HYBRID, make_hybrid_payload()))
                    else:
                        _install_mode(client, ModeRequest(vel_mode, {"filter": False}))
                except Exception as exc:
                    hybrid = not hybrid
                    name = "TRACK_HYBRID" if not hybrid else vel_mode.name
                    print(f"[MODE] {name} install failed: {exc}", flush=True)
                    last_l3_s = time.monotonic()
                    drain_pad_edges(src)
                    continue
                last_l3_s = time.monotonic()
                drain_pad_edges(src)
                if hybrid:
                    if hybrid_law == "ultrapoc":
                        print(
                            f"[MODE] TRACK_HYBRID pad ultrapoc delay-KF α·v + My⊕image  Z={fz:.2f}N",
                            flush=True,
                        )
                    elif hybrid_law == "ac2d":
                        print(
                            f"[MODE] TRACK_HYBRID pad ac2d Z+My  Z={fz:.2f}N",
                            flush=True,
                        )
                    else:
                        print(f"[MODE] TRACK_HYBRID pad+force Z={fz:.2f}N", flush=True)
                else:
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
