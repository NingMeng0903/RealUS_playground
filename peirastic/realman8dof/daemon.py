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
from rm75_control.control.joint_admittance_8dof.loop import run_joint_admittance_phases
from rm75_control.core.session import RobotSession
from rm75_control.hw.lw100.rail_calibration import CalValidationError
from peirastic.core.estop import EstopBus
from peirastic.core.ipc import Cmd, CommandHub, Status, TwistBus
from peirastic.core.modes import MODE_LABEL, Mode, ModeRequest
from peirastic.core.panel import Panel
from peirastic.realman8dof.binding import bind_controller, load_yaml
from peirastic.realman8dof.session import ProxyOuter, compile_request


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
        self.log_csv = log_csv
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
        _SWAPPABLE = {Mode.SERVO_TWIST, Mode.SERVO_TWIST_HOLD, Mode.TRACK_HYBRID}

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

    def _trip_hardware(self, rail, reason: str) -> None:
        self.estop.trip(reason)
        self.hub.request_stop()
        self.panel.event("ESTOP", reason)
        if rail is not None and getattr(rail, "enabled", False):
            try:
                rail.estop()
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

            req = ModeRequest(Mode.SERVO_TWIST, {})
            first = self.hub.poll()
            if first is not None:
                cmd, seq, parsed = first
                self.hub.ack(seq)
                if cmd == Cmd.ESTOP:
                    self._trip_hardware(rail, "ipc estop")
                    continue
                if cmd == Cmd.STOP:
                    continue
                if cmd == Cmd.SET_MODE and parsed is not None:
                    req = parsed

            try:
                phase = compile_request(
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

            self.mode = req.mode
            self.hub.clear_stop()
            self.panel.event("MODE", MODE_LABEL[self.mode])
            self.hub.publish(status=Status.RUNNING, mode=self.mode, msg=phase.label)

            def _stop() -> bool:
                if self._stop or self.estop.tripped or self.hub.should_stop():
                    return True
                snap = self.twist.read()
                if snap["r3"]:
                    self._trip_hardware(rail, "pad R3")
                    return True
                return False

            def _on_step(label, t_phase, step, pose, f_ext, t_wall=float("nan")) -> None:
                del label, t_phase, t_wall
                self.ticks += 1
                q = np.asarray(getattr(step, "q_send", self.inner.q_cmd), dtype=float)
                self.hub.publish(
                    status=Status.RUNNING,
                    mode=self.mode,
                    ticks=self.ticks,
                    estop=self.estop.tripped,
                    pad_hz=float(self.twist.read()["hz"]),
                    track_err_mm=float(getattr(phase.outer, "last_err_mm", float("nan"))),
                    slack=float(getattr(step, "slack_norm", float("nan"))),
                    f_ext_z=float(f_ext[2]) if f_ext is not None and len(f_ext) > 2 else float("nan"),
                    msg=str(getattr(step, "qp_status", "")),
                )
                self.panel.update(
                    mode=MODE_LABEL[self.mode],
                    status="RUNNING",
                    ticks=self.ticks,
                    q=list(q.reshape(-1)[:8]),
                    pose=list(np.asarray(pose, dtype=float).reshape(-1)[:6]),
                    f_ext_z=float(f_ext[2]) if f_ext is not None and len(f_ext) > 2 else float("nan"),
                    track_err_mm=float(getattr(phase.outer, "last_err_mm", float("nan"))),
                    slack=float(getattr(step, "slack_norm", float("nan"))),
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
            if self.estop.tripped:
                self.hub.publish(status=Status.ESTOP, mode=self.mode, estop=True)
            elif result.stop_reason:
                self.panel.event("WARN", result.stop_reason)
                self.hub.publish(status=Status.ERROR, mode=self.mode, msg=result.stop_reason[:90])
            else:
                self.hub.publish(status=Status.DONE, mode=self.mode, ticks=self.ticks)


def run_service(
    config_path: Path,
    *,
    shm_prefix: str = "",
    log_csv: str | None = None,
    dry_run: bool = False,
    panel: bool = True,
) -> int:
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
