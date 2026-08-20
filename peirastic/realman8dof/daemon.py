"""Window A: start inner (wbc_rt) + 200 Hz outer, swap modes without rebuild."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import numpy as np

from rm75_control.control.admittance_common.observer import CompensatedForceObserver
from rm75_control.control.admittance_common.state_bus import RobotStateBus
from rm75_control.control.admittance_common.state_relay import (
    StateRelayPublisher,
    parse_state_relay_config,
)
from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    parse_rail_servo_config,
)
from rm75_control.control.joint_admittance_8dof.loop import Phase, run_joint_admittance_phases
from rm75_control.core.session import RobotSession
from rm75_control.hw.lw100.rail_calibration import CalValidationError
from peirastic.core.estop import EstopBus
from peirastic.core.ipc import Cmd, CommandHub, Status, TwistBus
from peirastic.core.modes import MODE_LABEL, Mode, ModeRequest
from peirastic.core.panel import Panel
from peirastic.core.session import is_swappable
from peirastic.realman8dof.binding import bind_controller, load_yaml
from peirastic.realman8dof.session import ProxyOuter, compile_request

def default_log_dir() -> Path:
    playground = Path(__file__).resolve().parents[2]
    return playground / "rm75_control" / "apps" / "logs" / "peirastic"


def resolve_log_csv(
    value: str | None,
    *,
    now: float | None = None,
    log_dir: Path | None = None,
) -> str | None:
    """None = off. Bare ``--log-csv`` / ``auto`` = timestamped file under apps/logs."""

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"auto", "1", "true", "yes"}:
        root = Path(log_dir) if log_dir is not None else default_log_dir()
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime(
            "%Y%m%d_%H%M%S",
            time.localtime(now) if now is not None else time.localtime(),
        )
        return str(root / f"run_{stamp}.csv")
    return str(Path(text).expanduser())


_PHASE_COPY = (
    "qdot_ff_provider",
    "scale_qdot_ff_with_governor",
    "governor_err_ok_mm",
    "governor_err_max_mm",
    "governor_scale_min",
    "governor_joint_err_ok_deg",
    "governor_joint_err_max_deg",
    "governor_tau_s",
    "governor_freeze_below",
    "governor_release_above",
    "soft_start_ramp_s",
    "on_enter",
    "on_exit",
    "on_tick",
    "force_observer",
)


class ControllerService:
    def __init__(
        self,
        raw: dict,
        *,
        config_path: Path,
        shm_prefix: str = "",
        log_csv: str | None = None,
        panel: bool = True,
    ) -> None:
        self.raw = raw
        self.config_path = Path(config_path)
        self.log_csv = resolve_log_csv(log_csv)
        self.estop = EstopBus()
        self.panel = Panel(enabled=panel)
        self.hub = CommandHub(prefix=shm_prefix)
        self.twist = TwistBus(prefix=shm_prefix, create=True)
        self.kin, self.inner, self.ctx = bind_controller(raw)
        self.mode = Mode.SERVO_TWIST
        self.ticks = 0
        self._stop = False
        self._pending: ModeRequest | None = None
        self._live: Phase | None = None
        self._mode_t0 = 0.0
        self._finite_duration: float | None = None

    def close(self) -> None:
        self.hub.close()
        self.twist.close()

    def _on_signal(self, rail) -> None:
        def _handler(_signum, _frame) -> None:
            if self._stop:
                os._exit(130)
            self._stop = True
            self.hub.request_stop()
            self.estop.trip("signal")
            if rail is not None and getattr(rail, "enabled", False):
                try:
                    rail.estop()
                except Exception:
                    pass

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _trip_hardware(self, rail, reason: str, *, robot=None) -> None:
        self.estop.trip(reason)
        self.hub.request_stop()
        self.panel.event("ESTOP", reason)
        if rail is not None and getattr(rail, "enabled", False):
            try:
                rail.halt_velocity()
            except Exception:
                pass
            try:
                rail.estop()
            except Exception:
                pass
        if robot is not None:
            try:
                robot.rm_set_arm_slow_stop()
            except Exception:
                pass

    def run(self, sess, bus, rail) -> None:
        self._on_signal(rail)
        dt = float(self.raw.get("timing", {}).get("dt_ms", 5.0)) / 1000.0
        obs = None
        try:
            obs = CompensatedForceObserver.from_yaml(self.raw)
        except Exception:
            obs = None
        if rail is not None and getattr(rail, "enabled", False):
            try:
                rail.halt_if_moving()
            except Exception:
                pass
        if self.log_csv:
            self.panel.event("STATE", f"csv {self.log_csv}")
        try:
            from peirastic.realman8dof.force.config import DEFAULT_FORCE_YAML, desired_z_n

            self.panel.event(
                "STATE",
                f"force yaml {DEFAULT_FORCE_YAML} Fz*={desired_z_n():.2f}N",
            )
        except Exception:
            pass
        self.panel.event("STATE", "inner up; servo_twist(0)")
        self.hub.publish(status=Status.RUNNING, mode=self.mode, msg="ready")

        while not self._stop:
            if self.estop.tripped:
                self.hub.publish(
                    status=Status.ESTOP,
                    mode=self.mode,
                    estop=True,
                    msg=self.estop.reason,
                )
                time.sleep(0.05)
                polled = self.hub.poll()
                if polled is not None and polled[0] == Cmd.RESET:
                    self.estop.reset()
                    self.hub.clear_stop()
                    self.hub.ack(polled[1])
                    self.panel.event("OK", "estop reset")
                continue

            req = self._pending or ModeRequest(Mode.SERVO_TWIST, {})
            self._pending = None
            first = self.hub.poll()
            if first is not None:
                cmd, seq, parsed = first
                self.hub.ack(seq)
                if cmd == Cmd.ESTOP:
                    self._trip_hardware(rail, "ipc estop", robot=getattr(sess, "robot", None))
                    continue
                if cmd == Cmd.STOP:
                    continue
                if cmd == Cmd.SET_MODE and parsed is not None:
                    req = parsed

            try:
                compiled = compile_request(
                    self.ctx,
                    req,
                    raw=self.raw,
                    twist_read=lambda: self.twist.read()["twist"],
                    dt=dt,
                )
            except Exception as exc:
                self.panel.event("WARN", f"compile {exc}")
                self.hub.publish(status=Status.ERROR, mode=req.mode, msg=str(exc)[:90])
                time.sleep(0.05)
                continue

            velocity_loop = is_swappable(req.mode)
            proxy = ProxyOuter(compiled.outer)
            if velocity_loop:
                phase = Phase(outer=proxy, label=compiled.label, duration_s=None)
                for key in _PHASE_COPY:
                    setattr(phase, key, getattr(compiled, key))
            else:
                phase = compiled
            self._live = phase
            self.mode = req.mode
            self._mode_t0 = 0.0
            self._finite_duration = compiled.duration_s if velocity_loop else None
            self.hub.clear_stop()
            self.panel.event("MODE", MODE_LABEL[self.mode])
            self.hub.publish(status=Status.RUNNING, mode=self.mode, msg=phase.label)

            def _install_velocity(
                new: Phase,
                parsed_req: ModeRequest,
                pose,
                t_ref: float,
                *,
                status: Status = Status.RUNNING,
            ) -> None:
                if phase.on_exit is not None:
                    phase.on_exit()
                proxy.bind(new.outer)
                for key in _PHASE_COPY:
                    setattr(phase, key, getattr(new, key))
                phase.label = new.label
                pose_now = np.asarray(pose, dtype=float)
                proxy.set_origin(pose_now, t_s=float(t_ref))
                if phase.on_enter is not None:
                    phase.on_enter()
                if hasattr(new.outer, "begin_hybrid_episode"):
                    q = np.asarray(self.inner.q_cmd, dtype=float)
                    applied_qdot = self.inner.core.qdot_prev
                    applied_twist = self.inner.kin.jacobian(q) @ applied_qdot
                    self.inner.begin_hybrid_episode(q, applied_qdot)
                    proxy.begin_hybrid_episode(applied_twist, pose_now)
                self.mode = parsed_req.mode
                self._mode_t0 = float(t_ref)
                self._finite_duration = new.duration_s
                self.panel.event("MODE", MODE_LABEL[self.mode])
                self.hub.publish(status=status, mode=self.mode, msg=phase.label)

            def _apply(parsed_req: ModeRequest, pose, t_ref: float = 0.0) -> None:
                # Joint PTP runner is not a velocity proxy: any new mode rebuilds.
                if not velocity_loop or not is_swappable(parsed_req.mode):
                    self._pending = parsed_req
                    self.hub.request_stop()
                    return
                new = compile_request(
                    self.ctx,
                    parsed_req,
                    raw=self.raw,
                    twist_read=lambda: self.twist.read()["twist"],
                    dt=dt,
                )
                _install_velocity(new, parsed_req, pose, t_ref)

            def _stop() -> bool:
                if self._stop or self.estop.tripped or self.hub.should_stop():
                    return True
                if self.twist.read()["r3"]:
                    self._trip_hardware(rail, "pad R3", robot=getattr(sess, "robot", None))
                    return True
                return False

            def _on_step(label, t_phase, step, pose, f_ext, t_wall=float("nan")) -> None:
                del label, t_wall
                t_ref = float(t_phase)
                polled = self.hub.poll()
                if polled is not None:
                    cmd, seq, parsed = polled
                    self.hub.ack(seq)
                    if cmd == Cmd.ESTOP:
                        self._trip_hardware(rail, "ipc estop", robot=getattr(sess, "robot", None))
                    elif cmd == Cmd.STOP:
                        self.hub.request_stop()
                    elif cmd == Cmd.SET_MODE and parsed is not None:
                        try:
                            _apply(parsed, pose, t_ref)
                        except Exception as exc:
                            self.panel.event("WARN", str(exc))
                if (
                    velocity_loop
                    and self._finite_duration is not None
                    and (t_ref - self._mode_t0) >= float(self._finite_duration)
                ):
                    self.panel.event("OK", f"{phase.label} done")
                    try:
                        idle = compile_request(
                            self.ctx,
                            ModeRequest(Mode.SERVO_TWIST, {}),
                            raw=self.raw,
                            twist_read=lambda: self.twist.read()["twist"],
                            dt=dt,
                        )
                        _install_velocity(
                            idle,
                            ModeRequest(Mode.SERVO_TWIST, {}),
                            pose,
                            t_ref,
                            status=Status.DONE,
                        )
                    except Exception as exc:
                        self.panel.event("WARN", str(exc))
                self.ticks += 1
                q = np.asarray(getattr(step, "q_send", self.inner.q_cmd), dtype=float)
                err = float(getattr(phase.outer, "last_err_mm", float("nan")))
                slack = float(getattr(step, "slack_norm", float("nan")))
                fz = float(f_ext[2]) if f_ext is not None and len(f_ext) > 2 else float("nan")
                self.hub.publish(
                    status=Status.RUNNING,
                    mode=self.mode,
                    ticks=self.ticks,
                    estop=self.estop.tripped,
                    pad_hz=float(self.twist.read()["hz"]),
                    track_err_mm=err,
                    slack=slack,
                    f_ext_z=fz,
                    msg=phase.label,
                )
                self.panel.update(
                    mode=MODE_LABEL[self.mode],
                    status="RUNNING",
                    ticks=self.ticks,
                    q=list(q.reshape(-1)[:8]),
                    pose=list(np.asarray(pose, dtype=float).reshape(-1)[:6]),
                    f_ext_z=fz,
                    track_err_mm=err,
                    slack=slack,
                    rail_m=float(q[0]) if q.size else float("nan"),
                    wbc_ok=True,
                    pad_hz=float(self.twist.read()["hz"]),
                    estop=self.estop.tripped,
                    estop_reason=self.estop.reason,
                )
                self.panel.maybe_draw()

            result = run_joint_admittance_phases(
                sess,
                [phase],
                self.inner,
                dt=dt,
                force_observer=obs,
                state_bus=bus,
                stop_check=_stop,
                rail_bridge=rail,
                log_csv=self.log_csv,
                verbose=False,
                on_step=_on_step,
            )
            self._live = None
            if self.estop.tripped:
                if rail is not None and getattr(rail, "enabled", False):
                    try:
                        rail.halt_velocity()
                    except Exception:
                        pass
                self.hub.publish(status=Status.ESTOP, mode=self.mode, estop=True)
            elif result.stop_reason:
                self.panel.event("WARN", result.stop_reason)
                self.hub.publish(status=Status.ERROR, mode=self.mode, msg=result.stop_reason[:90])
            elif self._pending is None and not velocity_loop:
                self.hub.publish(status=Status.DONE, mode=self.mode, ticks=self.ticks)


def run_service(
    config_path: Path,
    *,
    shm_prefix: str = "",
    log_csv: str | None = None,
    dry_run: bool = False,
    panel: bool = True,
) -> int:
    log_csv = resolve_log_csv(log_csv)
    if log_csv:
        print(f"[STATE] csv {log_csv}", flush=True)
    raw = load_yaml(config_path)
    if dry_run:
        bind_controller(raw, backend="python")
        print("[STATE] dry-run bind ok", flush=True)
        return 0
    svc = ControllerService(
        raw, config_path=config_path, shm_prefix=shm_prefix, log_csv=log_csv, panel=panel
    )
    robot_cfg = raw.get("robot", {})
    rail = RailServoBridge(parse_rail_servo_config(raw))
    relay_cfg = parse_state_relay_config(raw)
    try:
        if rail.enabled:
            try:
                rail.start()
            except CalValidationError as exc:
                print(f"[WARN] rail calibration failed: {exc}", flush=True)
                return 2
        with RobotSession(
            ip=robot_cfg.get("ip"),
            port=robot_cfg.get("port"),
            config=str(config_path),
            quiet=True,
        ) as sess:
            bus = RobotStateBus(sess.robot, raw, robot_ip=sess.ip)
            bus.start()
            relay = None
            if relay_cfg.enabled:
                relay = StateRelayPublisher(
                    bus, name=relay_cfg.name, hz=relay_cfg.hz, kin=svc.inner.kin
                )
                relay.start()
            try:
                svc.panel.event("STATE", "running")
                svc.run(sess, bus, rail)
            finally:
                if relay is not None:
                    relay.stop()
    finally:
        rail.stop()
        svc.close()
    return 0
