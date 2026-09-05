"""Window A: start inner (wbc_rt) + 200 Hz outer, swap modes without rebuild."""

from __future__ import annotations

import math
import os
import signal
import time
from pathlib import Path

import numpy as np

from rm75_control.control.admittance_common.observer import CompensatedForceObserver
from rm75_control.control.admittance_common.state_bus import (
    RobotStateBus,
    expand_q_meas_8dof,
)
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
from rm75_control.force.compensation.tool_pose import read_tool_offset_cache
from rm75_control.hw.lw100.rail_calibration import CalValidationError
from peirastic.core.estop import EstopBus
from peirastic.core.ipc import Cmd, CommandHub, Status, TwistBus
from peirastic.core.modes import DofRequest, MODE_LABEL, Mode, ModeRequest
from peirastic.core.panel import Panel
from peirastic.core.session import (
    idle_after_finite,
    is_swappable,
    pad_source_present,
    stay_after_duration,
)
from peirastic.realman8dof.binding import bind_controller, load_yaml
from peirastic.realman8dof.session import ProxyOuter, compile_request
from rm75_control.control.joint_admittance_8dof.api import (
    set_controller_dof,
    validate_dof,
)

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


HANDOFF_STOP_REASONS = frozenset({"external_stop_before_send"})


def is_handoff_stop(reason: str | None, *, pending_commanded: bool) -> bool:
    """True when a phase ends only so the next commanded mode can start."""

    return bool(reason) and str(reason) in HANDOFF_STOP_REASONS and bool(pending_commanded)


def ready_state_msg(
    *,
    rail_m: float | None,
    tcp: str | None,
    force_obs: bool | None = None,
) -> str:
    parts = ["ready"]
    if rail_m is not None and math.isfinite(float(rail_m)):
        parts.append(f"rail={float(rail_m) * 1000.0:.1f} mm")
    if tcp:
        parts.append(f"tcp={tcp}")
    if force_obs is not None:
        parts.append("force=on" if force_obs else "force=off")
    return "  ".join(parts)


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


def _reject_legacy_dof_kwargs(legacy: dict) -> None:
    if "secondary" in legacy or "last_secondary" in legacy:
        raise ValueError(
            "secondary was removed from the PEIRASTIC API/IPC; use set_dof(7 or 8)"
        )
    if legacy:
        key = next(iter(legacy))
        raise TypeError(f"unexpected keyword argument {key!r}")


def idle_mode_payload(*, dof: int | None = None, **legacy) -> dict:
    """Payload for post-finite idle; task policy follows session DOF."""
    _reject_legacy_dof_kwargs(legacy)
    if dof is None:
        raise TypeError("idle_mode_payload requires dof=7 or 8")
    value = validate_dof(dof)
    return {
        "filter": False,
        "task_policy": "payload_id" if value == 7 else "track",
    }


def idle_after_command(
    *,
    dof: int | None = None,
    pad_source: bool = False,
    **legacy,
) -> ModeRequest:
    """Idle after a finite move.

    The task policy is derived from the persistent session DOF.  A zero
    velocity command never ends a continuous SERVO task by itself.
    """
    _reject_legacy_dof_kwargs(legacy)
    if dof is None:
        raise TypeError("idle_after_command requires dof=7 or 8")
    value = validate_dof(dof)
    if value == 7:
        return ModeRequest(
            Mode.SERVO_TWIST,
            {
                "task_policy": "payload_id",
                "label": "dof7_idle",
                "filter": False,
                "v_cmd": [0.0] * 6,
                "joint_hold": True,
            },
        )
    return ModeRequest(
        idle_after_finite(pad_source=pad_source),
        idle_mode_payload(dof=value),
    )


def dof_transition_hold(*, dof: int) -> ModeRequest:
    """Keep the current structure quiescent while a DOF switch is pending.

    The request is compiled with the *current* session DOF.  It therefore
    cannot use the target structure's ``payload_id`` policy before the switch
    commits.  An explicit zero twist plus the ``off`` policy prevents the
    coupled 8-DOF rail/posture path from continuing to generate motion while
    the feedback stationarity gate is waiting.
    """
    validate_dof(dof)
    return ModeRequest(
        Mode.SERVO_TWIST_HOLD,
        {
            "v_cmd": [0.0] * 6,
            "task_policy": "off",
            "label": "dof_transition_hold",
            "filter": False,
        },
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
        robot=None,
    ) -> None:
        self.raw = raw
        self.config_path = Path(config_path)
        self.log_csv = resolve_log_csv(log_csv)
        self.estop = EstopBus()
        self.panel = Panel(enabled=panel)
        self.hub = CommandHub(prefix=shm_prefix)
        self.twist = TwistBus(prefix=shm_prefix, create=True)
        self.kin, self.inner, self.ctx, tcp_name = bind_controller(raw, robot=robot)
        self._dof = 8
        self.ctx.dof = self._dof
        set_controller_dof(self.inner, self._dof)
        if tcp_name:
            self.tcp_name = str(tcp_name)
        else:
            cached = read_tool_offset_cache()
            self.tcp_name = str(cached[0] or "") if cached is not None else None
        self.mode = Mode.SERVO_TWIST_HOLD
        self.ticks = 0
        self._stop = False
        self._pending: ModeRequest | None = None
        self._live: Phase | None = None
        self._mode_t0 = 0.0
        self._finite_duration: float | None = None
        self._cmd_seq = 0
        self._pending_commanded = False
        # Command ACK means only that the mailbox was consumed.  Keep the
        # command sequence attached to a queued mode until its on_enter hook
        # has run and the install ACK can be published.
        self._pending_install_seq: int | None = None
        self._pending_dof: tuple[int, int] | None = None
        self._runner_started = False
        # A DOF request waits for this explicit boundary.  A live velocity
        # phase remains open until its finite duration expires or the caller
        # explicitly stops/replaces it; zero SERVO input is not a boundary.
        self._dof_boundary_open = False
        self._fault_sm = "RUNNING"
        self._fault_epoch = 0
        self.force_observer = None
        self.force_observer_error = ""
        try:
            self.force_observer = CompensatedForceObserver.from_yaml(self.raw)
        except Exception as exc:
            self.force_observer_error = str(exc)
            self.panel.event("WARN", f"force observer off: {exc}")

    def close(self) -> None:
        self.hub.close()
        self.twist.close()

    def _pad_row(self) -> dict:
        return self.twist.read()

    def _pad_source_present(self) -> bool:
        row = self._pad_row()
        return pad_source_present(
            float(row.get("stamp") or 0.0),
            hz=row.get("hz"),
            connected=bool(row.get("connected")),
        )

    def _idle_request(self) -> ModeRequest:
        return idle_after_command(
            dof=self._dof,
            pad_source=self._pad_source_present(),
        )

    @property
    def dof(self) -> int:
        return int(self._dof)

    def _dof_stationary(self, bus, rail) -> bool:
        """Use fresh arm/rail feedback before committing a structure switch."""
        # With hardware feedback, the SDK/rail samples below are the source
        # of truth.  ``core.qdot_prev`` is only the last command and can stay
        # nonzero after a coordinated stop, so treating it as actual motion
        # would make a valid stationary boundary impossible to commit.
        if bus is None:
            core = getattr(self.inner, "core", None)
            qdot = getattr(core, "qdot_prev", None)
            if qdot is not None:
                arr = np.asarray(qdot, dtype=float).reshape(-1)
                if arr.size and (
                    not np.all(np.isfinite(arr))
                    or np.any(np.abs(arr) > 0.03)
                ):
                    return False
        if bus is not None:
            try:
                snap = bus.read()
                ok = bool(getattr(snap, "ok", snap.get("ok", False) if isinstance(snap, dict) else False))
                if not ok:
                    return False
                stamp = getattr(snap, "t_s", snap.get("t_s", float("nan")) if isinstance(snap, dict) else float("nan"))
                stamp = float(stamp)
                if not np.isfinite(stamp) or abs(time.monotonic() - stamp) > 0.10:
                    return False
                qd = getattr(snap, "qdot_deg_s", None)
                if qd is None and isinstance(snap, dict):
                    qd = snap.get("qdot_deg_s")
                if qd is not None:
                    arr = np.asarray(qd, dtype=float).reshape(-1)
                    if arr.size < 7 or not np.all(np.isfinite(arr[:7])):
                        return False
                    if float(np.max(np.abs(arr[:7]))) > 2.0:
                        return False
                else:
                    return False
            except Exception:
                return False
        if rail is not None and bool(getattr(rail, "enabled", False)):
            try:
                command = rail.command
                v = float(command.v_ff_m_s)
                # NaN is the bridge's "no velocity feed-forward" marker for
                # position/idle commands; the measured and time-stamped
                # execution sample below remains the stationarity gate.
                if np.isfinite(v) and abs(v) > 1.0e-3:
                    return False
                measured = float(rail.measured_speed_m_s)
                if not np.isfinite(measured) or abs(measured) > 1.0e-3:
                    return False
                feedback = rail.execution_feedback
                sample_t = float(getattr(feedback, "sample_mono_s", float("nan")))
                v_feedback = float(getattr(feedback, "v_meas_m_s", float("nan")))
                age = float(getattr(feedback, "sample_age_s", float("inf")))
                if (
                    not bool(getattr(feedback, "valid", False))
                    or not np.isfinite(sample_t)
                    or not np.isfinite(age)
                    or age > 0.10
                    or abs(time.monotonic() - sample_t) > 0.10
                    or not np.isfinite(v_feedback)
                    or abs(v_feedback) > 1.0e-3
                ):
                    return False
            except Exception:
                return False
        return True

    def _commit_pending_dof(self, bus, rail) -> bool:
        pending = self._pending_dof
        if pending is None or not self._dof_stationary(bus, rail):
            return False
        value, seq = pending
        try:
            q_live = self._live_q8(bus, rail)
            if bus is not None and q_live is None:
                # A structure switch must seed from a fresh arm/rail sample;
                # q_cmd is only an offline fallback when no feedback bus is
                # present in unit tests.
                return False
            # ``set_controller_dof`` performs the single transition reset
            # from this fresh seed before changing rail ownership.  Keep the
            # reset in one place so native/Python mixer state is not cleared
            # twice (the second reset would lose the just-seeded references).
            set_controller_dof(self.inner, value, q_live=q_live)
            self.ctx.dof = value
            self._dof = value
        except Exception as exc:
            self.panel.event("WARN", f"dof {exc}")
            self.hub.publish(
                status=Status.ERROR,
                mode=self.mode,
                msg=str(exc)[:90],
                done_seq=seq,
                err_code=1,
                dof=self._dof,
                dof_pending=-1,
                dof_requested=value,
                dof_effective=self._dof,
                dof_request_seq=seq,
                dof_done_seq=seq,
                dof_status=Status.ERROR,
            )
            self._pending_dof = None
            self._pending = None
            self._pending_commanded = False
            return False
        self._pending_dof = None
        self.hub.publish(
            status=Status.DONE,
            mode=self.mode,
            msg=f"dof={value}",
            done_seq=seq,
            err_code=0,
            dof=value,
            dof_pending=-1,
            dof_requested=value,
            dof_effective=value,
            dof_request_seq=seq,
            dof_done_seq=seq,
            dof_status=Status.DONE,
        )
        return True

    def _live_q8(self, bus, rail) -> np.ndarray | None:
        """Return a fresh measured [rail, arm×7] seed for DOF transitions."""
        if bus is None:
            return None
        try:
            snap = bus.read()
            ok = bool(
                getattr(
                    snap,
                    "ok",
                    snap.get("ok", False) if isinstance(snap, dict) else False,
                )
            )
            if not ok:
                return None
            q_arm = getattr(
                snap,
                "q_deg",
                snap.get("q_deg") if isinstance(snap, dict) else None,
            )
            if q_arm is None:
                return None
            q_arm = np.asarray(q_arm, dtype=float).reshape(-1)
            if q_arm.size < 7 or not np.all(np.isfinite(q_arm[:7])):
                return None
            if rail is not None and bool(getattr(rail, "enabled", False)):
                # Use the position belonging to the same fresh execution
                # sample used by the stationarity gate.  A missing position
                # is rejected.  Using a newer/unrelated encoder read would
                # seed the arm and rail from different instants.
                feedback = rail.execution_feedback
                rail_m = float(getattr(feedback, "position_m", float("nan")))
                if not np.isfinite(rail_m):
                    return None
            else:
                rail_m = float(np.asarray(self.inner.q_cmd, dtype=float).reshape(-1)[0])
            if not np.isfinite(rail_m):
                return None
            q_live = expand_q_meas_8dof(q_arm[:7], rail_m)
            return q_live if np.all(np.isfinite(q_live)) else None
        except Exception:
            return None

    def _queue_dof(self, req: DofRequest, seq: int, bus, rail) -> None:
        value = validate_dof(req.dof)
        if self._pending_dof is not None:
            pending_value, _pending_seq = self._pending_dof
            self.hub.publish(
                status=Status.ERROR,
                mode=self.mode,
                msg=f"dof={pending_value} already pending",
                done_seq=seq,
                err_code=1,
                dof=self._dof,
                dof_pending=pending_value,
                dof_requested=value,
                dof_effective=self._dof,
                dof_request_seq=seq,
                dof_done_seq=seq,
                dof_status=Status.ERROR,
            )
            return
        if value == self._dof and self._pending_dof is None:
            self.hub.publish(
                status=Status.DONE,
                mode=self.mode,
                msg=f"dof={value}",
                done_seq=seq,
                err_code=0,
                dof=value,
                dof_pending=-1,
                dof_requested=value,
                dof_effective=value,
                dof_request_seq=seq,
                dof_done_seq=seq,
                dof_status=Status.DONE,
            )
            return
        self._pending_dof = (value, int(seq))
        self.hub.publish(
            status=Status.RUNNING,
            mode=self.mode,
            msg=f"dof pending={value}",
            dof=self._dof,
            dof_pending=value,
            dof_requested=value,
            dof_effective=self._dof,
            dof_request_seq=seq,
            dof_status=Status.RUNNING,
        )
        if self._live is None:
            self._commit_pending_dof(bus, rail)
        # ``after_current`` is a boundary request, not a stop request.
        # Keep the current finite/continuous phase alive; an explicit STOP
        # or its natural duration boundary lets the daemon commit once the
        # fresh arm/rail feedback is stationary.  In particular, a zero
        # SERVO twist is still a live task and must not be terminated here.

    def _pad_twist(self) -> np.ndarray:
        row = self._pad_row()
        if not bool(row["connected"]):
            return np.zeros(6, dtype=float)
        tw = np.asarray(row["twist"], dtype=float).reshape(-1)
        if tw.size < 6:
            out = np.zeros(6, dtype=float)
            out[: tw.size] = tw
            return out
        return tw[:6].copy()

    def _pad_r3(self) -> bool:
        row = self._pad_row()
        return bool(row["connected"]) and bool(row["r3"])

    def _pad_hz(self) -> float:
        row = self._pad_row()
        if not bool(row["connected"]):
            return float("nan")
        return float(row["hz"])

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

    def _cancel_pending_transitions(
        self,
        reason: str,
        *,
        err_code: int = -6,
        publish: bool = True,
    ) -> None:
        """Drop queued structure/mode changes after a terminal fault.

        A DOF request is deliberately asynchronous, so clearing only the
        current phase is insufficient: a pending request could otherwise be
        committed after ESTOP/RESET and a queued mode could then be compiled
        under that stale structure.  Keep all cancellation in one helper so
        the ESTOP, safety-brake, and generic fault paths have identical
        semantics.
        """

        pending_dof = self._pending_dof
        pending_mode = self._pending
        pending_mode_seq = self._pending_install_seq
        self._pending_dof = None
        self._pending = None
        self._pending_commanded = False
        self._pending_install_seq = None
        self._dof_boundary_open = False
        if not publish:
            return
        if pending_dof is not None:
            value, seq = pending_dof
            self.hub.publish(
                status=Status.ERROR,
                mode=self.mode,
                msg=str(reason)[:90],
                done_seq=seq,
                err_code=err_code,
                dof=self._dof,
                dof_pending=-1,
                dof_requested=value,
                dof_effective=self._dof,
                dof_request_seq=seq,
                dof_done_seq=seq,
                dof_status=Status.ERROR,
            )
        if pending_mode is not None and pending_mode_seq is not None:
            self.hub.publish(
                status=Status.ERROR,
                mode=pending_mode.mode,
                msg=str(reason)[:90],
                done_seq=pending_mode_seq,
                err_code=err_code,
            )

    @staticmethod
    def _coordinated_brake(rail, *, robot=None) -> None:
        """Brake both actuators without latching the controller in ESTOP.

        ``run_joint_admittance_phases`` owns its private safety-stop helper.
        The daemon only reaches this fallback when a STOP was observed before
        the runner published its first replacement tick, so there is no
        runner-local callback to invoke.  Keep the fallback explicitly
        coordinated and leave controller history intact for the next hold.
        """

        if rail is not None and getattr(rail, "enabled", False):
            try:
                rail.hold_current()
            except Exception:
                try:
                    rail.kill_motion()
                except Exception:
                    pass
        if robot is not None:
            try:
                robot.rm_set_arm_slow_stop()
            except Exception:
                pass

    @staticmethod
    def _is_latched_fault(reason: str) -> bool:
        text = str(reason or "").lower()
        tokens = (
            "watchdog",
            "native_timeout",
            "uncertified",
            "p0_conflict",
            "qpik_fault",
            "feedback_stale",
            "rail_feedback",
            "rail_target_rejected",
            "rail_panic",
            "partial_arm",
            "unknown_partial",
            "arm_send_fault",
            "publication",
        )
        return any(tok in text for tok in tokens)

    def _reset_allowed(self, sess, rail, bus=None) -> bool:
        if self._fault_sm not in ("SAFE_HOLD", "AWAIT_RESET", "FAULT_LATCHED", "STOPPING"):
            if not bool(getattr(self.estop, "tripped", False)):
                return False
            self._fault_sm = "AWAIT_RESET"
        if rail is not None and getattr(rail, "enabled", False):
            if bool(getattr(rail, "panicked", False)):
                return False
            age = float(getattr(rail, "last_encoder_age_s", float("nan")) or float("nan"))
            if np.isfinite(age) and age > 0.05:
                return False
            if abs(float(getattr(rail, "measured_vel_m_s", 0.0) or 0.0)) >= 0.002:
                return False
        if bus is not None:
            try:
                snap = bus.read()
                ok = bool(getattr(snap, "ok", False))
                stamp = float(getattr(snap, "t_s", float("nan")))
                if not ok or not np.isfinite(stamp) or abs(time.monotonic() - stamp) > 0.10:
                    return False
            except Exception:
                return False
        return True

    def _trip_hardware(self, rail, reason: str, *, robot=None) -> None:
        self._fault_sm = "FAULT_LATCHED"
        self._fault_epoch += 1
        abort = getattr(self.inner, "abort_publication", None)
        if callable(abort):
            try:
                abort()
            except Exception:
                pass
        self._cancel_pending_transitions(reason)
        self.estop.trip(reason)
        self.hub.request_stop()
        self.panel.event("ESTOP", reason)
        self._fault_sm = "STOPPING"
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
        self._fault_sm = "SAFE_HOLD"
        self._fault_sm = "AWAIT_RESET"

    def run(self, sess, bus, rail) -> None:
        self._on_signal(rail)
        dt = float(self.raw.get("timing", {}).get("dt_ms", 5.0)) / 1000.0
        obs = self.force_observer
        if rail is not None and getattr(rail, "enabled", False):
            try:
                rail.halt_if_moving()
            except Exception:
                pass
        if self.log_csv:
            self.panel.event("STATE", f"csv {self.log_csv}")
        self.hub.publish(
            status=Status.RUNNING,
            mode=self.mode,
            msg="ready",
            dof=self._dof,
            dof_pending=-1,
            dof_requested=self._dof,
            dof_effective=self._dof,
            dof_status=Status.IDLE,
        )

        while not self._stop:
            if self.estop.tripped:
                # A fault can be raised asynchronously (signal, watchdog,
                # or a safety layer) without passing through
                # ``_trip_hardware``.  Make RESET unable to revive stale
                # queued DOF/mode requests as well.
                self._cancel_pending_transitions(
                    self.estop.reason or "estop",
                    publish=False,
                )
                self.hub.publish(
                    status=Status.ESTOP,
                    mode=self.mode,
                    estop=True,
                    msg=self.estop.reason,
                )
                time.sleep(0.05)
                polled = self.hub.poll()
                if polled is not None and polled[0] == Cmd.RESET:
                    if not self._reset_allowed(sess, rail, bus):
                        self.hub.ack(polled[1])
                        self.panel.event("WARN", "reset refused: axes not still/fresh")
                        continue
                    self._cancel_pending_transitions("reset", publish=False)
                    q_meas = self._live_q8(bus, rail)
                    if q_meas is None:
                        q_meas = np.asarray(self.inner.q_cmd, dtype=float).copy()
                    try:
                        self.inner.reset(q_meas)
                    except Exception:
                        pass
                    self.estop.reset()
                    self.hub.clear_stop()
                    self._fault_sm = "RUNNING"
                    self._fault_epoch += 1
                    self.hub.ack(polled[1])
                    self.panel.event("OK", "estop reset")
                continue

            commanded = False
            install_seq: int | None = None
            if self._pending_dof is not None:
                # A DOF boundary needs an actual zero-motion phase.  Keeping
                # this transition inside the normal runner preserves command
                # polling, arm/rail tail compensation, and the fresh
                # feedback stationarity gate while a queued replacement mode
                # remains uncompiled.
                self._dof_boundary_open = True
                req = dof_transition_hold(dof=self._dof)
            elif self._pending is not None:
                commanded = bool(self._pending_commanded)
                req = self._pending
                self._pending = None
                self._pending_commanded = False
                install_seq = self._pending_install_seq
                self._pending_install_seq = None
            else:
                req = self._idle_request()
            first = self.hub.poll()
            if first is not None:
                cmd, seq, parsed = first
                self.hub.ack(seq)
                self._cmd_seq = int(seq)
                if cmd == Cmd.ESTOP:
                    self._trip_hardware(rail, "ipc estop", robot=getattr(sess, "robot", None))
                    continue
                if cmd == Cmd.STOP:
                    continue
                if cmd == Cmd.SET_DOF and isinstance(parsed, DofRequest):
                    try:
                        self._queue_dof(parsed, seq, bus, rail)
                    except Exception as exc:
                        self.hub.publish(
                            status=Status.ERROR,
                            mode=self.mode,
                            msg=str(exc)[:90],
                            done_seq=seq,
                            err_code=1,
                            dof=self._dof,
                            dof_pending=-1,
                            dof_requested=(parsed.dof if isinstance(parsed, DofRequest) else self._dof),
                            dof_effective=self._dof,
                            dof_request_seq=seq,
                            dof_done_seq=seq,
                            dof_status=Status.ERROR,
                        )
                    if self._live is None and self._pending_dof is None:
                        req = self._idle_request()
                    elif self._live is None:
                        # No phase owns the command loop, but feedback may
                        # still show a short stop tail.  Run the current
                        # structure's zero-motion hold so the normal
                        # feedback gate can retry; do not start a posture
                        # idle that could keep the rail moving.
                        self._dof_boundary_open = True
                        req = dof_transition_hold(dof=self._dof)
                    elif self._live is not None:
                        continue
                if cmd == Cmd.SET_MODE and parsed is not None:
                    if self._pending_dof is not None:
                        # The DOF request owns the next boundary.  Keep this
                        # mode request queued and run a zero-motion hold
                        # until the structure commit has completed; compiling
                        # it now would capture the old ``ctx.dof``.
                        self._pending = parsed
                        self._pending_commanded = True
                        self._pending_install_seq = int(seq)
                        self._dof_boundary_open = True
                        req = dof_transition_hold(dof=self._dof)
                    else:
                        req = parsed
                        commanded = True
                        install_seq = int(seq)

            try:
                compiled = compile_request(
                    self.ctx,
                    req,
                    raw=self.raw,
                    twist_read=self._pad_twist,
                    dt=dt,
                )
            except Exception as exc:
                self.panel.event("WARN", f"compile {exc}")
                self.hub.publish(
                    status=Status.ERROR,
                    mode=req.mode,
                    msg=str(exc)[:90],
                    done_seq=self._cmd_seq if commanded else 0,
                    err_code=1,
                )
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
            if commanded:
                self.panel.event("MODE", MODE_LABEL[self.mode])
            self.hub.publish(status=Status.RUNNING, mode=self.mode, msg=phase.label)

            def _arm_install_ack(target: Phase, mode: Mode, seq: int | None) -> None:
                """Publish install_seq only after this phase's on_enter runs."""

                if seq is None or int(seq) <= 0:
                    return
                previous = target.on_enter

                def _entered() -> None:
                    if previous is not None:
                        previous()
                    self.hub.publish(
                        status=Status.RUNNING,
                        mode=mode,
                        msg=target.label,
                        install_seq=int(seq),
                    )

                target.on_enter = _entered

            _arm_install_ack(phase, req.mode, install_seq if commanded else None)

            def _install_velocity(
                new: Phase,
                parsed_req: ModeRequest,
                pose,
                t_ref: float,
                *,
                status: Status = Status.RUNNING,
                done_seq: int | None = None,
                err_code: int | None = None,
                announce: bool = True,
                install_seq: int | None = None,
            ) -> None:
                _arm_install_ack(new, parsed_req.mode, install_seq)
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
                if announce:
                    self.panel.event("MODE", MODE_LABEL[self.mode])
                self.hub.publish(
                    status=status,
                    mode=self.mode,
                    msg=phase.label,
                    done_seq=done_seq,
                    err_code=err_code,
                )

            def _commit_dof_at_boundary(pose, t_ref: float) -> bool:
                """Commit a queued structure and rebind the post-boundary idle task."""
                if not self._dof_boundary_open or self._pending_dof is None:
                    return False
                if not self._commit_pending_dof(bus, rail):
                    return False
                # The old phase has ended.  Rebind idle from the newly
                # committed session DOF so a 7-DOF commit cannot continue
                # running an 8-DOF track outer (or vice versa).
                try:
                    idle_req = self._idle_request()
                    idle = compile_request(
                        self.ctx,
                        idle_req,
                        raw=self.raw,
                        twist_read=self._pad_twist,
                        dt=dt,
                    )
                    _install_velocity(
                        idle,
                        idle_req,
                        pose,
                        t_ref,
                        announce=False,
                    )
                    self._dof_boundary_open = False
                    return True
                except Exception as exc:
                    self.panel.event("WARN", f"dof idle rebind {exc}")
                    return False

            def _apply(
                parsed_req: ModeRequest,
                pose,
                t_ref: float = 0.0,
                *,
                install_seq: int | None = None,
            ) -> None:
                # Joint PTP runner is not a velocity proxy: any new mode rebuilds.
                if self._pending_dof is not None:
                    # A mode received while a structure request is waiting
                    # must be installed under the new structure, never
                    # compiled against the old ``ctx.dof``.  Queue it as an
                    # explicit replacement boundary; the outer runner will
                    # brake, commit the DOF, then the next outer iteration
                    # compiles this request.
                    self._pending = parsed_req
                    self._pending_commanded = True
                    self._pending_install_seq = install_seq
                    self.hub.request_stop()
                    return
                if not velocity_loop or not is_swappable(parsed_req.mode):
                    self._pending = parsed_req
                    self._pending_commanded = True
                    self._pending_install_seq = install_seq
                    self.hub.request_stop()
                    return
                new = compile_request(
                    self.ctx,
                    parsed_req,
                    raw=self.raw,
                    twist_read=self._pad_twist,
                    dt=dt,
                )
                _install_velocity(
                    new,
                    parsed_req,
                    pose,
                    t_ref,
                    install_seq=install_seq,
                )

            def _stop() -> bool:
                if self._stop or self.estop.tripped or self.hub.should_stop():
                    return True
                # A queued replacement is installed by the outer daemon
                # iteration after the DOF boundary has committed.  End this
                # transition-hold runner so that replacement compilation
                # cannot happen inside the old phase.
                if (
                    self._pending is not None
                    and self._pending_dof is None
                    and not self._dof_boundary_open
                ):
                    return True
                if self._pad_r3():
                    self._trip_hardware(rail, "pad R3", robot=getattr(sess, "robot", None))
                    return True
                return False

            def _on_step(label, t_phase, step, pose, f_ext, t_wall=float("nan")) -> None:
                del label
                t_ref = float(t_phase)
                polled = self.hub.poll()
                if polled is not None:
                    cmd, seq, parsed = polled
                    self.hub.ack(seq)
                    self._cmd_seq = int(seq)
                    if cmd == Cmd.ESTOP:
                        self._trip_hardware(rail, "ipc estop", robot=getattr(sess, "robot", None))
                    elif cmd == Cmd.STOP:
                        self.hub.request_stop()
                    elif cmd == Cmd.SET_DOF and isinstance(parsed, DofRequest):
                        try:
                            self._queue_dof(parsed, seq, bus, rail)
                        except Exception as exc:
                            self.panel.event("WARN", str(exc))
                            self.hub.publish(
                                status=Status.ERROR,
                                mode=self.mode,
                                msg=str(exc)[:90],
                                done_seq=seq,
                                err_code=1,
                                dof=self._dof,
                                dof_pending=-1,
                                dof_requested=(parsed.dof if isinstance(parsed, DofRequest) else self._dof),
                                dof_effective=self._dof,
                                dof_request_seq=seq,
                                dof_done_seq=seq,
                                dof_status=Status.ERROR,
                            )
                    elif cmd == Cmd.SET_MODE and parsed is not None:
                        try:
                            _apply(parsed, pose, t_ref, install_seq=int(seq))
                        except Exception as exc:
                            self.panel.event("WARN", str(exc))
                            self.hub.publish(
                                status=Status.ERROR,
                                mode=parsed.mode,
                                msg=str(exc)[:90],
                                done_seq=self._cmd_seq,
                                err_code=1,
                            )
                if (
                    velocity_loop
                    and self._finite_duration is not None
                    and (t_ref - self._mode_t0) >= float(self._finite_duration)
                ):
                    self.panel.event("OK", f"{phase.label} done")
                    try:
                        if stay_after_duration(self.mode) and self._pending_dof is None:
                            self._finite_duration = None
                            phase.label = "servo_twist"
                            self.hub.publish(
                                status=Status.DONE,
                                mode=self.mode,
                                msg=phase.label,
                                done_seq=self._cmd_seq,
                                err_code=0,
                            )
                        else:
                            # If a DOF request is queued, keep the currently
                            # committed structure quiescent while waiting for
                            # the fresh feedback stationarity gate.  The
                            # normal post-task idle policy may keep d*/rail
                            # posture active and would prevent that gate from
                            # ever becoming true.
                            idle_req = (
                                dof_transition_hold(dof=self._dof)
                                if self._pending_dof is not None
                                else self._idle_request()
                            )
                            idle = compile_request(
                                self.ctx,
                                idle_req,
                                raw=self.raw,
                                twist_read=self._pad_twist,
                                dt=dt,
                            )
                            _install_velocity(
                                idle,
                                idle_req,
                                pose,
                                t_ref,
                                status=Status.DONE,
                                done_seq=self._cmd_seq,
                                err_code=0,
                                announce=False,
                            )
                            self._dof_boundary_open = True
                    except Exception as exc:
                        self.panel.event("WARN", str(exc))
                if not self.estop.tripped:
                    _commit_dof_at_boundary(pose, t_ref)
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
                    pad_hz=float(self._pad_hz()),
                    track_err_mm=err,
                    slack=slack,
                    f_ext_z=fz,
                    msg=phase.label,
                )
                self.hub.motion.publish(
                    v_tcp_z=float(getattr(step, "v_tcp_z_actual", float("nan"))),
                    a_tcp_z_plus=float(getattr(step, "a_tcp_z_plus", 0.0)),
                    feedback_age_s=float(
                        getattr(step, "feedback_age_s", float("inf"))
                    ),
                    t_wall_s=float(t_wall),
                    valid=bool(getattr(step, "feedback_velocity_valid", False)),
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
                    pad_hz=float(self._pad_hz()),
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
                preserve_controller_state=bool(
                    getattr(self, "_runner_started", False)
                ),
            )
            self._runner_started = True
            self._live = None
            if self.hub.should_stop() and not self.estop.tripped and not result.stop_reason:
                # A STOP can be observed before the loop reaches its
                # publication gate.  Still execute the coordinated hold so a
                # later idle/DOF transition cannot resume with stale motion.
                self.panel.event("STOP", "external stop")
                self._coordinated_brake(
                    rail,
                    robot=getattr(sess, "robot", None),
                )
            if self.estop.tripped:
                if rail is not None and getattr(rail, "enabled", False):
                    try:
                        rail.halt_velocity()
                    except Exception:
                        pass
                self.hub.publish(
                    status=Status.ESTOP,
                    mode=self.mode,
                    estop=True,
                    done_seq=self._cmd_seq,
                    err_code=-6,
                )
            elif result.stop_reason == "uncertified_brake":
                self._trip_hardware(
                    rail,
                    result.stop_reason,
                    robot=getattr(sess, "robot", None),
                )
                self.hub.publish(
                    status=Status.ESTOP,
                    mode=self.mode,
                    estop=True,
                    msg=result.stop_reason,
                    done_seq=self._cmd_seq,
                    err_code=-6,
                )
            elif self._pending_dof is not None:
                if result.stop_reason and not is_handoff_stop(
                    result.stop_reason, pending_commanded=True
                ):
                    self.panel.event("WARN", result.stop_reason)
                    # A replacement mode queued behind a failed structure
                    # switch must not silently run under the old DOF.
                    self._cancel_pending_transitions(
                        result.stop_reason,
                        err_code=1,
                    )
                else:
                    self._dof_boundary_open = True
                    if self._commit_pending_dof(bus, rail):
                        self._dof_boundary_open = False
            elif is_handoff_stop(
                result.stop_reason, pending_commanded=self._pending_commanded
            ):
                pass
            elif result.stop_reason:
                if self._is_latched_fault(result.stop_reason):
                    self._trip_hardware(
                        rail,
                        result.stop_reason,
                        robot=getattr(sess, "robot", None),
                    )
                    self.hub.publish(
                        status=Status.ESTOP,
                        mode=self.mode,
                        estop=True,
                        msg=result.stop_reason[:90],
                        done_seq=self._cmd_seq,
                        err_code=-6,
                    )
                    continue
                self._cancel_pending_transitions(
                    result.stop_reason,
                    err_code=1,
                )
                self.panel.event("WARN", result.stop_reason)
                self.hub.publish(
                    status=Status.ERROR,
                    mode=self.mode,
                    msg=result.stop_reason[:90],
                    done_seq=self._cmd_seq,
                    err_code=1,
                )
            elif not velocity_loop:
                self.hub.publish(
                    status=Status.DONE,
                    mode=self.mode,
                    ticks=self.ticks,
                    done_seq=self._cmd_seq,
                    err_code=0,
                )
                if self._pending is None:
                    self._pending = self._idle_request()
                    self._pending_commanded = False


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
    robot_cfg = raw.get("robot", {})
    rail = RailServoBridge(parse_rail_servo_config(raw))
    relay_cfg = parse_state_relay_config(raw)
    svc = None
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
            # Bind after the SDK session exists so planning + force-Z use the
            # pendant/web tool, not outputs/rm75_tool_offset.json (often gripper2).
            svc = ControllerService(
                raw,
                config_path=config_path,
                shm_prefix=shm_prefix,
                log_csv=log_csv,
                panel=panel,
                robot=sess.robot,
            )
            bus = RobotStateBus(sess.robot, raw, robot_ip=sess.ip)
            bus.start()
            relay = None
            if relay_cfg.enabled:
                def _rail_m_fn() -> float:
                    # Twin must show LW100 encoder, not URDF 0 / WBC q_cmd.
                    if not rail.enabled:
                        return float("nan")
                    if rail.calibrated:
                        m = float(rail.measured_m)
                        if math.isfinite(m):
                            return m
                    return float("nan")

                relay = StateRelayPublisher(
                    bus,
                    name=relay_cfg.name,
                    hz=relay_cfg.hz,
                    kin=svc.inner.kin,
                    rail_m_fn=_rail_m_fn,
                )
                relay.start()
            else:
                print(
                    "[WARN] state_relay.enabled=false — "
                    "Genesis twin will stay at URDF default (no hardware DW)",
                    flush=True,
                )
            enc = float(rail.measured_m) if rail.enabled else float("nan")
            svc.panel.event(
                "STATE",
                ready_state_msg(
                    rail_m=enc if math.isfinite(enc) else None,
                    tcp=svc.tcp_name or None,
                    force_obs=svc.force_observer is not None,
                ),
            )
            try:
                svc.run(sess, bus, rail)
            finally:
                if relay is not None:
                    relay.stop()
    finally:
        rail.stop()
        if svc is not None:
            svc.close()
    return 0
