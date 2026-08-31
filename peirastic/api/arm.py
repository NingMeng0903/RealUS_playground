"""RM_API2-shaped facade over peirastic modes. SI units; int return codes."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from peirastic.api.codes import (
    ERR_CONTROLLER,
    ERR_NO_ACK,
    ERR_SEND,
    ERR_STOPPED,
    ERR_TIMEOUT,
    ERR_UNIMPLEMENTED,
    OK,
)
from peirastic.api.payloads import (
    HfpcPayload,
    HfvcPayload,
    MoveJPayload,
    MoveLPayload,
    ServoTwistPayload,
    TrackCartesianPayload,
)
from peirastic.core.ipc import CommandClient, Status, TwistBus
from peirastic.core.modes import Mode, ModeRequest

ACK_TIMEOUT_S = 2.0
DEFAULT_BLOCK_S = 60.0
CONTACT_POLL_S = 0.02


def _as_q(q) -> list[float]:
    arr = np.asarray(q, dtype=float).reshape(-1)
    if arr.size != 8:
        raise ValueError(f"q must be 8-vec (rail + 7 arm rad), got {arr.size}")
    return arr.tolist()


def _as_pose(pose) -> list[float]:
    arr = np.asarray(pose, dtype=float).reshape(-1)
    if arr.size != 6:
        raise ValueError(f"pose must be 6-vec [x,y,z,rx,ry,rz], got {arr.size}")
    return arr.tolist()


def _as_poses(poses) -> list[list[float]]:
    arr = np.asarray(poses, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = arr.reshape(-1, 6)
    return arr.tolist()


def _as_twist(twist) -> list[float]:
    arr = np.asarray(twist, dtype=float).reshape(-1)
    if arr.size != 6:
        raise ValueError(f"twist must be 6-vec, got {arr.size}")
    return arr.tolist()


def _check_v(v: float) -> float:
    v = float(v)
    if not 0.0 < v <= 1.0:
        raise ValueError(f"v must be in (0, 1], got {v}")
    return v


def poll_force_contact(
    snapshot_fn,
    *,
    enter_n: float,
    confirm_s: float,
    timeout_s: float,
    want_mode: int | None = int(Mode.TRACK_HYBRID),
    want_label: str | None = None,
) -> bool:
    """Latch Fz after the hybrid phase is live and the probe has been in air."""

    hold_s = 0.0
    last = time.monotonic()
    deadline = last + float(timeout_s)
    saw_air = False
    air_n = 0.5 * float(enter_n)
    while time.monotonic() < deadline:
        try:
            tel = snapshot_fn()
        except (TypeError, AttributeError) as exc:
            raise RuntimeError("interrupted") from exc
        if tel is None:
            raise RuntimeError("interrupted")
        st = int(tel.get("status", -1))
        if st in (int(Status.ERROR), int(Status.ESTOP)):
            raise RuntimeError(str(tel.get("msg") or "force mode failed"))
        if want_mode is not None and int(tel.get("mode", -1)) != int(want_mode):
            time.sleep(CONTACT_POLL_S)
            last = time.monotonic()
            continue
        if want_label and want_label not in str(tel.get("msg") or ""):
            time.sleep(CONTACT_POLL_S)
            last = time.monotonic()
            continue
        fz = float(tel.get("f_ext_z", float("nan")))
        now = time.monotonic()
        dt = now - last
        last = now
        if not saw_air:
            if np.isfinite(fz) and abs(fz) < air_n:
                saw_air = True
            time.sleep(CONTACT_POLL_S)
            continue
        if np.isfinite(fz) and abs(fz) >= float(enter_n):
            hold_s += dt
            if hold_s >= float(confirm_s):
                return True
        else:
            hold_s = 0.0
        time.sleep(CONTACT_POLL_S)
    return False


class _ClientMixin:
    client: CommandClient | None
    inner: Any
    ctx: Any
    engine: Any
    last_request: ModeRequest | None
    last_seq: int
    _qp_aux: dict[str, Any]
    _force_extra: dict[str, Any]

    def _check_tail(self, r: float, connect: int) -> int | None:
        if float(r) != 0.0 or int(connect) != 0:
            return ERR_UNIMPLEMENTED
        return None

    def _with_aux(self, payload: dict[str, Any]) -> dict[str, Any]:
        out = dict(payload)
        if self._qp_aux:
            merged = dict(out.get("qp_aux") or {})
            merged.update(self._qp_aux)
            out["qp_aux"] = merged
        if self._force_extra:
            for key, val in self._force_extra.items():
                out.setdefault(key, val)
        return out

    def _compile_local(self, req: ModeRequest):
        from peirastic.realman8dof.session import compile_request

        if self.engine is not None:
            return self.engine.set_mode(req)
        if self.ctx is not None:
            return compile_request(self.ctx, req)
        return None

    def _send(self, mode: Mode, payload: dict[str, Any], *, block: float | int = 0) -> int:
        req = ModeRequest(mode, self._with_aux(payload))
        self.last_request = req
        if self.client is None:
            try:
                self._compile_local(req)
            except ValueError:
                return ERR_CONTROLLER
            except Exception:
                return ERR_CONTROLLER
            return OK
        try:
            seq = self.client.set_mode(req)
        except ValueError:
            return ERR_SEND
        except FileNotFoundError:
            return ERR_SEND
        self.last_seq = int(seq)
        if not block:
            return OK
        return self._wait_done(int(seq), block)

    def _snapshot(self) -> dict[str, Any]:
        if self.client is None:
            return {}
        return self.client.snapshot()

    def _wait_done(self, seq: int, block: float | int) -> int:
        timeout = DEFAULT_BLOCK_S if float(block) == 1.0 else float(block)
        deadline = time.monotonic() + timeout
        ack_deadline = time.monotonic() + min(ACK_TIMEOUT_S, timeout)
        saw_ack = False
        while time.monotonic() < deadline:
            try:
                snap = self._snapshot()
            except (TypeError, AttributeError, FileNotFoundError):
                return ERR_SEND
            st = int(snap.get("status", -1))
            if int(snap.get("ack_seq", 0)) >= seq:
                saw_ack = True
            if st == int(Status.ESTOP):
                return ERR_STOPPED
            if st == int(Status.STOPPED):
                return ERR_STOPPED
            if int(snap.get("done_seq", 0)) >= seq:
                err = int(snap.get("err_code", 0))
                return OK if err == 0 else int(err)
            if st == int(Status.ERROR) and saw_ack:
                err = int(snap.get("err_code", 0))
                return ERR_CONTROLLER if err == 0 else int(err)
            if not saw_ack and time.monotonic() >= ack_deadline:
                return ERR_NO_ACK
            time.sleep(0.02)
        if not saw_ack:
            return ERR_NO_ACK
        return ERR_TIMEOUT


class _MovePlanMixin(_ClientMixin):
    _max_line_speed: float
    _max_angular_speed: float
    _max_joint_speed: float

    def set_max_line_speed(self, m_s: float) -> int:
        self._max_line_speed = float(m_s)
        return OK

    def set_max_angular_speed(self, rad_s: float) -> int:
        self._max_angular_speed = float(rad_s)
        return OK

    def set_max_joint_speed(self, rad_s: float) -> int:
        self._max_joint_speed = float(rad_s)
        return OK

    def movej(self, q, v: float = 0.2, r: float = 0, connect: int = 0, block: int = 1) -> int:
        bad = self._check_tail(r, connect)
        if bad is not None:
            return bad
        v = _check_v(v)
        payload = MoveJPayload(q_target=_as_q(q), v=v, label="movej").to_json()
        return self._send(Mode.MOVEJ, payload, block=block)

    def movej_p(
        self,
        pose,
        v: float = 0.2,
        r: float = 0,
        connect: int = 0,
        block: int = 1,
        *,
        rail_m: float | None = None,
    ) -> int:
        bad = self._check_tail(r, connect)
        if bad is not None:
            return bad
        v = _check_v(v)
        payload = MoveJPayload(
            pose=_as_pose(pose), v=v, rail_m=rail_m, label="movej_p"
        ).to_json()
        return self._send(Mode.MOVEJ, payload, block=block)

    def movel(self, pose, v: float = 0.2, r: float = 0, connect: int = 0, block: int = 1) -> int:
        return self.cartesian(pose, v=v, r=r, connect=connect, block=block)

    def cartesian(self, pose, v: float = 0.2, r: float = 0, connect: int = 0, block: int = 1) -> int:
        """Pose-to-pose: IK the goal, then joint-space smooth PTP (not a TCP line)."""

        bad = self._check_tail(r, connect)
        if bad is not None:
            return bad
        v = _check_v(v)
        payload = MoveLPayload(
            pose=_as_pose(pose),
            v=v,
            max_lin_vel_m_s=float(self._max_line_speed) * v,
            label="cartesian",
        ).to_json()
        return self._send(Mode.MOVEL, payload, block=block)

    def moves(self, poses, v: float = 0.2, r: float = 0, connect: int = 0, block: int = 1) -> int:
        bad = self._check_tail(r, connect)
        if bad is not None:
            return bad
        v = _check_v(v)
        payload = MoveLPayload(
            poses=_as_poses(poses),
            v=v,
            speed_m_s=float(self._max_line_speed) * v,
            max_lin_vel_m_s=float(self._max_line_speed) * v,
            label="moves",
        ).to_json()
        return self._send(Mode.MOVES, payload, block=block)


class _TrackMixin(_ClientMixin):
    def track_pose(
        self,
        pose,
        *,
        duration_s: float | None = None,
        block: int = 0,
        label: str = "track_hold",
        soft_start: bool = False,
    ) -> int:
        # Single-pose polyline so QPIK tracks the given TCP, not the live origin.
        payload = TrackCartesianPayload(
            reference="polyline",
            poses=[_as_pose(pose)],
            soft_start=bool(soft_start),
            duration_s=duration_s,
            label=label,
        ).to_json()
        return self._send(Mode.TRACK_CARTESIAN, payload, block=block)

    def track_polyline(
        self,
        poses,
        *,
        speed_m_s: float,
        soft_start: bool = True,
        ramp_s: float = 0.4,
        duration_s: float | None = None,
        block: int = 0,
        label: str = "track_polyline",
        max_lin_vel_m_s: float | None = None,
        move_kp: float | None = None,
    ) -> int:
        payload = TrackCartesianPayload(
            reference="polyline",
            poses=_as_poses(poses),
            speed_m_s=float(speed_m_s),
            soft_start=bool(soft_start),
            ramp_s=float(ramp_s),
            duration_s=duration_s,
            max_lin_vel_m_s=max_lin_vel_m_s,
            move_kp=move_kp,
            label=label,
        ).to_json()
        return self._send(Mode.TRACK_CARTESIAN, payload, block=block)

    def track_ellipse(
        self,
        *,
        amplitude_x_m: float,
        amplitude_y_m: float,
        period_s: float | None = None,
        max_vel_m_s: float | None = None,
        soft_start: bool = True,
        ramp_s: float = 2.0,
        duration_s: float | None = None,
        label: str = "track_ellipse",
    ) -> int:
        payload = TrackCartesianPayload(
            reference="ellipse",
            amplitude_x_m=float(amplitude_x_m),
            amplitude_y_m=float(amplitude_y_m),
            period_s=period_s,
            max_vel_m_s=max_vel_m_s,
            soft_start=bool(soft_start),
            ramp_s=float(ramp_s),
            duration_s=duration_s,
            label=label,
        ).to_json()
        return self._send(Mode.TRACK_CARTESIAN, payload, block=0)

    def movep_canfd(self, pose, follow: bool = False) -> int:
        del follow
        return self.track_pose(pose, block=0, label="movep_canfd")


class _VelocityMixin(_ClientMixin):
    _twist_frame: str
    _twist_dt_ms: float

    def set_movev_canfd_init(self, *, frame_type: str = "tool", dt_ms: float = 5) -> int:
        self._twist_frame = str(frame_type)
        self._twist_dt_ms = float(dt_ms)
        return OK

    def movev_canfd(self, twist) -> int:
        payload = ServoTwistPayload(v_cmd=_as_twist(twist), label="movev_canfd").to_json()
        return self._send(Mode.SERVO_TWIST, payload, block=0)

    def track_twist(
        self,
        twist=None,
        *,
        hold: bool = False,
        duration_s: float | None = None,
        label: str | None = None,
    ) -> int:
        payload = ServoTwistPayload(
            v_cmd=None if twist is None else _as_twist(twist),
            duration_s=duration_s,
            label=label or ("servo_twist_hold" if hold else "servo_twist"),
        ).to_json()
        mode = Mode.SERVO_TWIST_HOLD if hold else Mode.SERVO_TWIST
        return self._send(mode, payload, block=0)


class _ForceMixin(_ClientMixin):
    _force_extra: dict[str, Any]
    _contact_enter_n: float | None
    _enter_confirm_s: float | None

    def set_force_control(
        self,
        *,
        force_axes=None,
        track_axes=None,
        desired_force=None,
        desired_force_ramp_s: float | None = None,
        control_frame: str | None = None,
        kp_pos=None,
        pos_err_deadband_m: float | None = None,
        pos_correction_max_m_s: float | None = None,
        admittance_mass: float | None = None,
        admittance_damping: float | None = None,
        admittance_stiffness: float | None = None,
        contact_enter_n: float | None = None,
        contact_exit_n: float | None = None,
        enter_confirm_s: float | None = None,
        exit_confirm_s: float | None = None,
        hard_enter_n: float | None = None,
        deadband_n: float | None = None,
        deadband_width_n: float | None = None,
        max_velocity=None,
        max_acceleration=None,
        max_vz_tool_m_s: float | None = None,
        sensor: str | None = None,
    ) -> int:
        """Persist task-level force defaults. yaml stays the file-level baseline."""

        extra = self._force_extra
        if force_axes is not None:
            extra["force_axes"] = list(np.asarray(force_axes, dtype=float).reshape(6))
        if track_axes is not None:
            extra["track_axes"] = list(np.asarray(track_axes, dtype=float).reshape(6))
        if desired_force is not None:
            if isinstance(desired_force, (int, float)):
                extra["desired_z"] = float(desired_force)
                extra["desired_force"] = float(desired_force)
            else:
                extra["desired_force"] = list(np.asarray(desired_force, dtype=float).reshape(-1))
        for key, val in (
            ("desired_force_ramp_s", desired_force_ramp_s),
            ("control_frame", control_frame),
            ("pos_err_deadband_m", pos_err_deadband_m),
            ("pos_correction_max_m_s", pos_correction_max_m_s),
            ("admittance_mass", admittance_mass),
            ("admittance_damping", admittance_damping),
            ("admittance_stiffness", admittance_stiffness),
            ("contact_enter_n", contact_enter_n),
            ("contact_exit_n", contact_exit_n),
            ("enter_confirm_s", enter_confirm_s),
            ("exit_confirm_s", exit_confirm_s),
            ("hard_enter_n", hard_enter_n),
            ("deadband_n", deadband_n),
            ("deadband_width_n", deadband_width_n),
            ("max_vz_tool_m_s", max_vz_tool_m_s),
            ("sensor", sensor),
        ):
            if val is not None:
                extra[key] = val
        if kp_pos is not None:
            extra["kp_pos"] = list(np.asarray(kp_pos, dtype=float).reshape(6))
        if max_velocity is not None:
            extra["max_velocity"] = list(np.asarray(max_velocity, dtype=float).reshape(6))
        if max_acceleration is not None:
            extra["max_acceleration"] = list(np.asarray(max_acceleration, dtype=float).reshape(6))
        if contact_enter_n is not None:
            self._contact_enter_n = float(contact_enter_n)
        if enter_confirm_s is not None:
            self._enter_confirm_s = float(enter_confirm_s)
        return OK

    def set_force_raw_override(self, raw: dict[str, Any]) -> int:
        """Escape hatch for research/safety blocks that stay in force.yaml."""

        self._force_extra.update(dict(raw))
        return OK

    def hfpc(
        self,
        poses,
        *,
        speed_m_s: float | None = None,
        law: str = "tff",
        force=None,
        force_axes=None,
        duration_s: float | None = None,
        v: float = 0.2,
        r: float = 0,
        connect: int = 0,
        block: int = 1,
        label: str = "hfpc",
        soft_start: bool | None = None,
        ramp_s: float | None = None,
    ) -> int:
        bad = self._check_tail(r, connect)
        if bad is not None:
            return bad
        v = _check_v(v)
        arr = _as_poses(poses)
        payload = HfpcPayload(
            reference="polyline",
            poses=arr,
            speed_m_s=speed_m_s if speed_m_s is not None else float(self._max_line_speed) * v,
            law=law,
            force=force,
            force_axes=None if force_axes is None else list(np.asarray(force_axes, dtype=float).reshape(6)),
            duration_s=duration_s,
            label=label,
            soft_start=soft_start,
            ramp_s=ramp_s,
        ).to_json()
        return self._send(Mode.TRACK_HYBRID, payload, block=block)

    def hfpc_ellipse(
        self,
        *,
        amplitude_x_m: float,
        amplitude_y_m: float,
        law: str = "tff",
        force=None,
        force_axes=None,
        period_s: float | None = None,
        max_vel_m_s: float | None = None,
        duration_s: float | None = None,
        label: str = "hfpc_ellipse",
    ) -> int:
        payload = HfpcPayload(
            reference="ellipse",
            law=law,
            force=force,
            force_axes=None if force_axes is None else list(np.asarray(force_axes, dtype=float).reshape(6)),
            amplitude_x_m=float(amplitude_x_m),
            amplitude_y_m=float(amplitude_y_m),
            period_s=period_s,
            max_vel_m_s=max_vel_m_s,
            duration_s=duration_s,
            label=label,
        ).to_json()
        return self._send(Mode.TRACK_HYBRID, payload, block=0)

    def hfvc(
        self,
        twist=None,
        *,
        source: str = "pad",
        force=None,
        force_axes=None,
        duration_s: float | None = None,
        label: str = "hfvc",
    ) -> int:
        payload = HfvcPayload(
            reference=source,
            v_cmd=None if twist is None else _as_twist(twist),
            force=force,
            force_axes=None if force_axes is None else list(np.asarray(force_axes, dtype=float).reshape(6)),
            duration_s=duration_s,
            label=label,
        ).to_json()
        return self._send(Mode.TRACK_HYBRID, payload, block=0)

    def stop_force(self) -> int:
        return self.track_twist()

    def wait_contact(
        self,
        *,
        enter_n: float | None = None,
        confirm_s: float | None = None,
        timeout_s: float,
        want_label: str | None = None,
    ) -> int:
        enter = self._contact_enter_n if enter_n is None else float(enter_n)
        if enter is None:
            from peirastic.realman8dof.force.config import load_force_raw

            hm = (load_force_raw().get("hybrid_motion") or {})
            pc = hm.get("physical_contact") or {}
            enter = float(pc.get("enter_n", hm.get("contact_threshold_n", 0.8)))
        confirm = self._enter_confirm_s if confirm_s is None else float(confirm_s)
        if confirm is None:
            confirm = 0.02
        try:
            ok = poll_force_contact(
                self._snapshot,
                enter_n=float(enter),
                confirm_s=float(confirm),
                timeout_s=float(timeout_s),
                want_label=want_label,
            )
        except RuntimeError:
            return ERR_CONTROLLER
        return OK if ok else ERR_TIMEOUT


class _QpAuxMixin(_ClientMixin):
    inner: Any
    _qp_aux: dict[str, Any]

    def _apply_inner_aux(self) -> None:
        if self.inner is None:
            return
        from peirastic.realman8dof.session import apply_qp_aux

        apply_qp_aux(self.inner, {"qp_aux": self._qp_aux})

    def set_collision_avoidance(self, enable: bool) -> int:
        self._qp_aux["collision"] = bool(enable)
        self._apply_inner_aux()
        return OK

    def set_nullspace(
        self,
        centering: bool | None = None,
        arm_angle: bool | None = None,
        manipulability: bool | None = None,
    ) -> int:
        if centering is not None:
            self._qp_aux["centering"] = bool(centering)
        if arm_angle is not None:
            self._qp_aux["arm_angle"] = bool(arm_angle)
        if manipulability is not None:
            self._qp_aux["manipulability"] = bool(manipulability)
        self._apply_inner_aux()
        return OK

    def set_singularity_escape(self, enable: bool) -> int:
        self._qp_aux["singularity_escape"] = bool(enable)
        self._apply_inner_aux()
        return OK

    def get_task_slack(self) -> tuple[int, float]:
        snap = self._snapshot()
        if not snap:
            return ERR_CONTROLLER, float("nan")
        val = float(snap.get("slack", float("nan")))
        return OK, val


class _StateMixin(_ClientMixin):
    state: Any
    force_bus: Any

    def get_joint_radian(self) -> tuple[int, list[float]]:
        if self.state is None:
            return ERR_CONTROLLER, []
        try:
            q = self.state.q_meas_8dof()
        except Exception:
            return ERR_CONTROLLER, []
        if q is None or not np.isfinite(q).all():
            return ERR_CONTROLLER, []
        return OK, np.asarray(q, dtype=float).reshape(-1).tolist()

    def get_current_arm_state(self) -> tuple[int, dict[str, Any]]:
        ret_q, q = self.get_joint_radian()
        pose: list[float] = []
        err = ""
        if self.state is not None:
            try:
                snap = self.state.read()
                if getattr(snap, "pose", None) is not None:
                    pose = np.asarray(snap.pose, dtype=float).reshape(-1).tolist()
                if not getattr(snap, "ok", True):
                    err = "stale"
            except Exception as exc:
                err = str(exc)
        if ret_q != OK and not pose:
            return ERR_CONTROLLER, {"joint": [], "pose": [], "err": err or "no state"}
        return OK, {"joint": q, "pose": pose, "err": err}

    def get_force_data(self) -> tuple[int, list[float]]:
        if self.force_bus is not None:
            try:
                ok, _seq, _t, wrench = self.force_bus.read()
                if ok:
                    return OK, np.asarray(wrench, dtype=float).reshape(-1).tolist()
            except Exception:
                pass
        snap = self._snapshot()
        if snap:
            fz = float(snap.get("f_ext_z", float("nan")))
            return OK, [0.0, 0.0, fz, 0.0, 0.0, 0.0]
        return ERR_CONTROLLER, []

    def get_controller_state(self) -> tuple[int, dict[str, Any]]:
        snap = self._snapshot()
        if not snap:
            return ERR_CONTROLLER, {}
        return OK, dict(snap)

    def set_arm_stop(self) -> int:
        if self.client is None:
            return ERR_SEND
        try:
            self.client.stop()
        except Exception:
            return ERR_SEND
        return OK

    def set_arm_estop(self) -> int:
        if self.client is None:
            return ERR_SEND
        try:
            self.client.estop()
        except Exception:
            return ERR_SEND
        return OK

    def reset(self) -> int:
        if self.client is None:
            return ERR_SEND
        try:
            self.client.reset()
        except Exception:
            return ERR_SEND
        return OK


class PeirasticArm(
    _MovePlanMixin,
    _TrackMixin,
    _VelocityMixin,
    _ForceMixin,
    _QpAuxMixin,
    _StateMixin,
):
    """Window-C handle. SI units. Methods return RM_API2-style int codes."""

    def __init__(
        self,
        *,
        prefix: str = "",
        client: CommandClient | None = None,
        twist: TwistBus | None = None,
        state=None,
        force_bus=None,
        inner=None,
        ctx=None,
        engine=None,
        attach: bool | None = None,
    ) -> None:
        self.prefix = str(prefix)
        self.inner = inner
        self.ctx = ctx
        self.engine = engine
        self.last_request: ModeRequest | None = None
        self.last_seq = 0
        self._qp_aux: dict[str, Any] = {}
        self._force_extra: dict[str, Any] = {}
        self._contact_enter_n: float | None = None
        self._enter_confirm_s: float | None = None
        self._max_line_speed = 0.4
        self._max_angular_speed = 1.0
        self._max_joint_speed = 1.0
        self._twist_frame = "tool"
        self._twist_dt_ms = 5.0
        if client is not None:
            self.client = client
        elif attach is False:
            self.client = None
        else:
            self.client = CommandClient(prefix=prefix)
        self.twist = twist
        if twist is None and attach is not False:
            try:
                self.twist = TwistBus(prefix=prefix, create=False)
            except FileNotFoundError:
                self.twist = None
        self.state = state
        if state is None and attach is not False:
            try:
                from rm75_control.control.admittance_common.state_relay import RelayStateBus

                self.state = RelayStateBus()
                self.state.ensure_attached()
            except Exception:
                self.state = None
        self.force_bus = force_bus
        if force_bus is None and attach is not False:
            try:
                from rm75_control.control.admittance_common.state_relay import ForceExtBus

                self.force_bus = ForceExtBus()
                self.force_bus.ensure_attached()
            except Exception:
                self.force_bus = None

    def close(self) -> None:
        if self.twist is not None:
            try:
                self.twist.close()
            except Exception:
                pass
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        if self.state is not None and hasattr(self.state, "stop"):
            try:
                self.state.stop()
            except Exception:
                pass
        if self.force_bus is not None and hasattr(self.force_bus, "stop"):
            try:
                self.force_bus.stop()
            except Exception:
                pass
