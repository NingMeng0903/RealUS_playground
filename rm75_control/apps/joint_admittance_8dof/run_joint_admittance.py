#!/usr/bin/env python3
"""8-DOF controller daemon (window A): UDP + SHM + local WBC when C submits a task.

Window A in the 3-terminal layout: keeps the sole Realman TCP/UDP session,
publishes ``rm75_state`` for the Genesis twin, and **runs the 200 Hz WBC loop
locally** when window C submits a phase program (no per-tick CANFD SHM relay).

  source env.sh
  python apps/joint_admittance_8dof/run_joint_admittance.py \\
      --config configs/joint_admittance_8dof.yaml

Twin (separate terminal):

  python apps/joint_admittance_8dof/run_with_twin.py

Task orchestration (window C):

  python apps/joint_admittance_8dof/d_sin_tool_y.py --config ... --enable-force ...
  python apps/joint_admittance_8dof/d_gamepad_vcmd.py --config ...
  python apps/joint_admittance_8dof/d_ellipse_track.py --config ...
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import signal
import time
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.admittance_common.phase_ipc import PhaseCmd, PhaseCommandHub, PhaseStatus
from rm75_control.control.admittance_common.state_bus import RobotStateBus
from rm75_control.control.admittance_common.observer import CompensatedForceObserver
from rm75_control.control.admittance_common.state_relay import (
    StateRelayPublisher,
    parse_state_relay_config,
)
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import (
    CartesianTrackConfig,
    CartesianTrackOuterLoop,
    JointIkController,
    run_joint_admittance_loop,
)
from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    parse_rail_servo_config,
)
from rm75_control.hw.lw100.rail_calibration import CalValidationError
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.reference import HoldReference
from rm75_control.control.joint_admittance_8dof.gamepad_vcmd_program import (
    close_built_pad,
)
from rm75_control.core.session import RobotSession
from rm75_control.force.compensation.tool_pose import maybe_sync_kin_tcp_from_config


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _reload_task_parsers():
    """Re-import yaml parsers so a long-lived window A picks up new keys.

    Each START already re-reads the yaml file.  Without this, an old
    ``_reject_unknown`` still running from daemon start refuses the task
    (``box_activate_rad`` on tasks #4–#6).
    """
    from rm75_control.control.joint_admittance_8dof.solver import (
        branch_barrier,
        joint_comfort,
        sigma_setbased,
    )
    from rm75_control.control.joint_admittance_8dof.tasks import (
        psi_retarget,
        rail_extension,
    )
    from rm75_control.control.joint_admittance_8dof import config as ik_config
    from rm75_control.control.joint_admittance_8dof import api as ja_api
    from rm75_control.control.joint_admittance_8dof import ellipse_track_program as etp
    from rm75_control.control.joint_admittance_8dof import gamepad_vcmd_program as gvp
    from rm75_control.control.joint_admittance_8dof import reference as ja_ref
    from rm75_control.control.joint_admittance_8dof import sin_tool_y_program as syp

    importlib.reload(branch_barrier)
    importlib.reload(joint_comfort)
    importlib.reload(sigma_setbased)
    importlib.reload(psi_retarget)
    importlib.reload(rail_extension)
    importlib.reload(ik_config)
    importlib.reload(ja_ref)
    importlib.reload(ja_api)
    importlib.reload(gvp)
    importlib.reload(syp)
    importlib.reload(etp)
    return gvp, syp, etp


def _run_controller_service(
    sess,
    bus: RobotStateBus,
    raw: dict,
    *,
    config_path: Path | None = None,
    hub: PhaseCommandHub,
    rail_m_fn,
    rail_bridge: RailServoBridge | None = None,
    relay: StateRelayPublisher | None = None,
    poll_s: float = 0.05,
    verbose: bool = False,
) -> None:
    """Hot-wait for window C; run WBC locally on START (direct UDP + CANFD)."""
    stop = False
    sig_n = 0

    def _on_sig(_signum, _frame) -> None:
        nonlocal stop, sig_n
        sig_n += 1
        # First action: kill rail (non-blocking) so FA24 cannot stay latched.
        if rail_bridge is not None and rail_bridge.enabled:
            try:
                rail_bridge.estop()
            except Exception:
                pass
        try:
            hub.request_stop()
        except Exception:
            pass
        stop = True
        if sig_n == 1:
            print(
                "\nrm75 controller: Ctrl+C — stopping task "
                "(second Ctrl+C forces exit)",
                flush=True,
            )
            return
        # Second+ signal: ProxQP / CANFD may hold the GIL for seconds near
        # singularity — do not wait for a clean Python teardown.
        print("\nrm75 controller: force exit", flush=True)
        os._exit(130)

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    hub.set_idle()
    print("rm75 controller: hot-wait", flush=True)

    while not stop:
        polled = hub.poll()
        if polled is None:
            time.sleep(poll_s)
            continue

        cmd, cmd_seq, params = polled
        if cmd == PhaseCmd.STOP:
            hub.ack(cmd_seq)
            hub.set_stopped(cmd_seq)
            continue

        if cmd != PhaseCmd.START or params is None:
            hub.ack(cmd_seq)
            continue

        task_n = hub.task_n

        # Refuse move→D / FA24 until rail Modbus path is hot (or re-armed after panic).
        if rail_bridge is not None and rail_bridge.enabled:
            if not rail_bridge.calibrated:
                hub.set_error(cmd_seq, "rail NOT CALIBRATED")
                hub.ack(cmd_seq)
                print(
                    f"rm75 controller: task #{task_n} refused — rail not calibrated. "
                    "Run: python apps/lw100_rail_home_limit.py",
                    flush=True,
                )
                if not stop:
                    print("rm75 controller: hot-wait", flush=True)
                continue
            need_rearm = rail_bridge.panicked or not rail_bridge.armed
            if need_rearm and rail_bridge.panicked:
                reason = rail_bridge.panic_reason or "rail PANIC"
                print(
                    f"rm75 controller: task #{task_n} — recovering from "
                    f"{reason}; clear limits then auto-rearm",
                    flush=True,
                )
            if not rail_bridge.ensure_armed(
                timeout_s=float(getattr(rail_bridge.config, "arm_timeout_s", 8.0)),
                rearm=need_rearm,
            ):
                hub.set_error(
                    cmd_seq,
                    "rail NOT READY (clear limit DI / check Modbus, then retry)",
                )
                hub.ack(cmd_seq)
                print(
                    f"rm75 controller: task #{task_n} refused — rail NOT READY "
                    "(if a limit was hit: nudge off the switch and resubmit)",
                    flush=True,
                )
                if not stop:
                    print("rm75 controller: hot-wait", flush=True)
                continue

        hub.set_running(cmd_seq, msg="accepted")
        print(f"rm75 controller: running task #{task_n}", flush=True)

        phase_labels: list[str] = []
        tick_counter = [0]
        phase_idx = [0]
        last_progress_label = [""]

        def _on_step(label, t_phase, step, pose, f_ext, t_wall=float("nan")) -> None:
            tick_counter[0] += 1
            if relay is not None and f_ext is not None:
                relay.set_f_ext(f_ext)
            if label in phase_labels:
                idx = phase_labels.index(label)
            else:
                phase_labels.append(label)
                idx = len(phase_labels) - 1
            phase_idx[0] = idx
            label_s = str(label)
            if label_s != last_progress_label[0]:
                last_progress_label[0] = label_s
                hub.set_progress(
                    cmd_seq,
                    phase_idx=idx,
                    phase_label=label_s,
                    ticks=tick_counter[0],
                )

        try:
            # Window A is long-lived.  Re-read yaml *and* reload the parsers
            # so a new key (e.g. box_activate_rad) cannot refuse START.
            gvp, syp, etp = _reload_task_parsers()
            task_raw = load_yaml(config_path) if config_path is not None else raw
            task_kind = str(getattr(params, "task_kind", "sin_tool_y") or "sin_tool_y")
            if task_kind == "gamepad_vcmd":
                built = gvp.build_gamepad_vcmd_program(params, raw=task_raw)
            elif task_kind == "ellipse_track":
                built = etp.build_ellipse_track_program(params, raw=task_raw)
            else:
                built = syp.build_sin_tool_y_program(params, raw=task_raw)
            rail_m_fn.set_active(built.inner)
            if relay is not None:
                # Prefer task kin (synced gripper TCP) for SHM pose publish.
                relay.set_kin(built.inner.kin)
            if rail_bridge is not None and rail_bridge.enabled:
                rail_csv = getattr(params, "rail_log_csv", None)
                if rail_csv:
                    rail_bridge.enable_log_csv(str(rail_csv))
                # Drop inherited SHM target / standstill latch from prior task.
                rail_bridge.begin_tracking_session()
            result = syp.execute_sin_tool_y_program(
                sess,
                bus,
                params,
                raw=task_raw,
                built=built,
                on_step=_on_step,
                stop_check=hub.should_stop,
                verbose=verbose,
                rail_bridge=rail_bridge,
            )
            if hub.should_stop():
                hub.set_stopped(cmd_seq)
                print(f"rm75 controller: task #{task_n} stopped", flush=True)
            elif result.stop_reason:
                hub.set_error(cmd_seq, result.stop_reason)
                print(
                    f"rm75 controller: task #{task_n} safety stop — "
                    f"{result.stop_reason}",
                    flush=True,
                )
            elif result.stalled:
                hub.set_error(cmd_seq, "control watchdog fired")
                print(
                    f"rm75 controller: task #{task_n} safety stop — "
                    "control watchdog fired",
                    flush=True,
                )
            else:
                hub.set_done(cmd_seq)
                print(
                    f"rm75 controller: task #{task_n} done "
                    f"({result.duration_s:.1f}s, {result.ticks} ticks)",
                    flush=True,
                )
        except KeyboardInterrupt:
            stop = True
            hub.set_stopped(cmd_seq, msg="interrupted")
            print(f"rm75 controller: task #{task_n} interrupted", flush=True)
        except Exception as exc:
            hub.set_error(cmd_seq, str(exc))
            print(f"rm75 controller: task error: {exc}", flush=True)
            if "unknown" in str(exc) and "configuration keys" in str(exc):
                print(
                    "rm75 controller: window A is still using the parser from "
                    "daemon start. Restart run_joint_admittance.py, then resubmit.",
                    flush=True,
                )
        finally:
            close_built_pad(locals().get("built"))
            hub.ack(cmd_seq)
            rail_m_fn.reset_idle()
            if rail_bridge is not None and rail_bridge.enabled:
                # Prefer non-blocking path if abort already set (Ctrl+C).
                # Normal exit: hold (FA24=0) unless residual is large — avoid
                # re-opening follow for sub-mm settle crawls (3–5 r/min hum).
                try:
                    if stop or rail_bridge._abort.is_set():
                        rail_bridge.estop()
                    else:
                        rail_bridge.hold_or_settle_after_task(settle_if_err_mm=2.0)
                except Exception:
                    try:
                        rail_bridge.estop()
                    except Exception:
                        pass
            if not stop:
                print("rm75 controller: hot-wait", flush=True)


class _RailPublisher:
    """Mutable rail source for SHM twin during idle vs active WBC.

    When the LW100 bridge is enabled, publish **encoder** position (poll_hz)
    so the twin mirrors the real carriage. WBC itself uses open-loop ``q_cmd[0]``
    and does not close the loop on this value.
    """

    def __init__(self, default_m: float, bridge: RailServoBridge | None = None) -> None:
        self._default_m = float(default_m)
        self._bridge = bridge
        self._active_inner: JointIkController | None = None

    def reset_idle(self) -> None:
        if self._bridge is not None and self._bridge.enabled:
            self._default_m = float(self._bridge.measured_m)
        elif self._active_inner is not None:
            self._default_m = float(self._active_inner.q_cmd[0])
        self._active_inner = None

    def set_active(self, inner: JointIkController) -> None:
        self._active_inner = inner

    def __call__(self) -> float:
        if self._bridge is not None and self._bridge.enabled:
            # After software zero is applied, always publish the live encoder
            # (true carriage pose). Never emit a fake 0 that makes the twin jump.
            if self._bridge.calibrated:
                m = float(self._bridge.measured_m)
                if math.isfinite(m):
                    self._default_m = m
                    return m
            # Not calibrated yet: hold last good / NaN (relay must not treat as 0).
            return float(self._default_m) if math.isfinite(self._default_m) else float("nan")
        if self._active_inner is not None:
            return float(self._active_inner.q_cmd[0])
        return self._default_m


def main() -> int:
    ap = argparse.ArgumentParser(
        description="8-DOF controller daemon (window A)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", type=Path, default=Path("configs/joint_admittance_8dof.yaml"))
    ap.add_argument(
        "--state-relay",
        default="rm75_state",
        metavar="NAME",
        help="Publish robot state to SHM for twin / window C (default rm75_state)",
    )
    ap.add_argument("--no-state-relay", action="store_true", help="Do not publish SHM")
    ap.add_argument("--relay-hz", type=float, default=None, help="SHM publish rate (default from YAML)")
    ap.add_argument(
        "--hold",
        action="store_true",
        help="Stream CANFD idle hold (teach re-anchor). Do NOT use with d_sin_tool_y.py",
    )
    ap.add_argument("--verbose", "-v", action="store_true", help="Print loop / teach / phase status")
    ap.add_argument("--dry-run", action="store_true", help="build controllers only, do not connect")
    args = ap.parse_args()

    raw = load_yaml(args.config)
    startup = raw.get("startup", {})
    relay_cfg = parse_state_relay_config(raw)
    if args.no_state_relay:
        relay_name = None
    else:
        relay_name = str(args.state_relay or relay_cfg.name or "rm75_state")
    relay_hz = float(args.relay_hz) if args.relay_hz is not None else relay_cfg.hz
    dt = float(raw.get("timing", {}).get("dt_ms", 5.0)) / 1000.0
    # Never seed twin/SHM from q_ref_m (yaml often has 0.0) — that fakes a jump to 0.
    # After bridge.start(), publisher uses live encoder metres.
    rail_default_m = float("nan")
    rail_bridge = RailServoBridge(parse_rail_servo_config(raw))
    if args.verbose and rail_bridge.enabled and not rail_bridge.log_csv_path:
        log_dir = Path(__file__).resolve().parents[1] / "logs" / "rail_servo"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        rail_bridge.enable_log_csv(str(log_dir / f"rail_{ts}.csv"))
    rail_pub = _RailPublisher(rail_default_m, bridge=rail_bridge)

    if args.dry_run:
        mode = "hold+CANFD" if args.hold else "controller+hot-wait"
        print(f"rm75 controller: dry-run OK ({mode})", flush=True)
        return 0

    robot_cfg = raw.get("robot", {})
    relay: StateRelayPublisher | None = None
    inner: JointIkController | None = None
    hub: PhaseCommandHub | None = None
    # Long-lived kin for SHM pose: RealMan UDP pose is often ArmTip/link_7
    # (~220 mm behind gripper TCP). Overwrite with Pinocchio fk_pose.
    pub_kin = RobotKinematics()

    if args.hold:
        kin = pub_kin
        inner_cfg = build_joint_ik_config(raw)
        inner = JointIkController(kin, inner_cfg)
        rail_pub.set_active(inner)

    with RobotSession(
        ip=robot_cfg.get("ip"),
        port=robot_cfg.get("port"),
        config=args.config,
        quiet=True,
    ) as sess:
        try:
            if rail_bridge.enabled:
                try:
                    rail_bridge.start()
                except CalValidationError as exc:
                    print(
                        f"rm75 controller: rail calibration failed — {exc}",
                        flush=True,
                    )
                    return 2
                meas = float(rail_bridge.measured_m)
                rail_pub._default_m = meas
                if inner is not None and math.isfinite(meas):
                    inner.q_cmd[0] = meas
            if inner is not None:
                maybe_sync_kin_tcp_from_config(raw=raw, kin=inner.kin, robot=sess.robot)
            else:
                maybe_sync_kin_tcp_from_config(raw=raw, kin=pub_kin, robot=sess.robot)
            bus = RobotStateBus(sess.robot, raw, robot_ip=sess.ip)
            bus.start()

            if relay_name:
                relay = StateRelayPublisher(
                    bus,
                    name=relay_name,
                    hz=relay_hz,
                    rail_m_fn=rail_pub,
                    kin=inner.kin if inner is not None else pub_kin,
                )
                try:
                    relay.set_force_observer(CompensatedForceObserver.from_yaml(raw))
                except Exception:
                    pass
                relay.start()
                if args.hold:
                    print(
                        f"rm75 controller: hold @ {relay_hz:.0f} Hz",
                        flush=True,
                    )
                else:
                    print(
                        f"rm75 controller: running @ {relay_hz:.0f} Hz",
                        flush=True,
                    )
            elif args.hold:
                print("rm75 controller: hold (no SHM)", flush=True)
            else:
                print("rm75 controller: running (no SHM)", flush=True)

            if args.hold:
                assert inner is not None
                gains = inner.cfg.cartesian_track
                outer = CartesianTrackOuterLoop(
                    HoldReference(),
                    CartesianTrackConfig(
                        k_task=np.array(
                            [
                                gains.k_task_lin,
                                gains.k_task_lin,
                                gains.k_task_lin,
                                gains.k_task_rot,
                                gains.k_task_rot,
                                gains.k_task_rot,
                            ],
                            dtype=float,
                        ),
                        max_pos_err_m=gains.max_pos_err_m,
                        max_rot_err_rad=gains.max_rot_err_rad,
                        euler_order=inner.cfg.euler_order,
                        control_frame=inner.cfg.control_frame,
                    ),
                )
                run_joint_admittance_loop(
                    sess,
                    outer,
                    inner,
                    q_start_deg=None,
                    duration_s=None,
                    dt=dt,
                    force_observer=None,
                    follow=bool(startup.get("follow", True)),
                    move_speed=int(startup.get("move_speed", 20)),
                    realtime=bool(startup.get("realtime", False)),
                    watchdog_timeout_s=float(startup.get("watchdog_timeout_s", 0.1)),
                    state_bus=bus,
                    verbose=args.verbose,
                    rail_bridge=rail_bridge,
                )
            else:
                hub = PhaseCommandHub()
                _run_controller_service(
                    sess,
                    bus,
                    raw,
                    config_path=args.config,
                    hub=hub,
                    rail_m_fn=rail_pub,
                    rail_bridge=rail_bridge,
                    relay=relay,
                    verbose=args.verbose,
                )
        finally:
            if hub is not None:
                hub.close()
            if relay is not None:
                relay.stop()
            rail_bridge.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
