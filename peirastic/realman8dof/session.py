"""Compile ModeRequest → Phase and hold the live outer for offline sample()."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from rm75_control.control.joint_admittance_8dof.api import CompileContext, SecondaryPolicy
from rm75_control.control.joint_admittance_8dof.loop import Phase
from rm75_control.control.joint_admittance_8dof.reference import (
    EllipseToolXYReference,
    HoldReference,
    WorldPolylineReference,
)
from peirastic.core.modes import Mode, ModeRequest
from peirastic.realman8dof.modes.cartesian import (
    build_cartesian_ptp_phase,
    build_moves_phase,
    resolve_pose_q,
)
from peirastic.realman8dof.modes.joint import build_goto_joints_phase, build_movej_phase
from peirastic.realman8dof.modes.servo import ServoTwistHoldOuter, ServoTwistOuter, slew_kwargs
from peirastic.realman8dof.modes.track import (
    build_pad_hybrid_phase,
    build_track_cartesian_phase,
    build_track_hybrid_phase,
)


def _twist_source(payload: dict, twist_read: Callable | None):
    if "v_cmd" in payload:
        v = np.asarray(payload["v_cmd"], dtype=float).reshape(6)
        return lambda: v
    if twist_read is not None:
        return twist_read
    z = np.zeros(6, dtype=float)
    return lambda: z.copy()


def _ellipse_ref(payload: dict, euler_order: str) -> EllipseToolXYReference:
    ax = float(payload.get("amplitude_x_m", 0.5 * 0.01 * float(payload.get("x_pp_cm", 10.0))))
    ay = float(payload.get("amplitude_y_m", 0.5 * 0.01 * float(payload.get("y_pp_cm", 30.0))))
    period = payload.get("period_s")
    vmax = payload.get("max_vel_m_s")
    if vmax is None and payload.get("max_vel_cm_s") is not None:
        vmax = 0.01 * float(payload["max_vel_cm_s"])
    stop = payload.get("stop_ramp_s")
    scan = payload.get("duration_s")
    rot = payload.get("rot_amp_rad")
    if rot is None and payload.get("rot_amp_deg") is not None:
        rot = np.deg2rad(np.asarray(payload["rot_amp_deg"], dtype=float))
    return EllipseToolXYReference(
        ax,
        ay,
        period_s=None if period is None else float(period),
        max_vel_m_s=None if vmax is None else float(vmax),
        soft_start=bool(payload.get("soft_start", True)),
        ramp_s=float(payload.get("ramp_s", 2.0)),
        duration_s=None if scan is None else float(scan),
        stop_ramp_s=None if stop is None else float(stop),
        euler_order=euler_order,
        rot_amp_rad=rot,
    )


def _polyline_ref(payload: dict, euler_order: str) -> WorldPolylineReference:
    points = payload.get("points")
    poses = payload.get("poses")
    rpy = payload.get("rpy")
    plan_path = payload.get("plan_path")
    phase = str(payload.get("phase") or "scan")
    if plan_path and (points is None and poses is None):
        import json
        from pathlib import Path

        raw = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        if not bool(raw.get("ok", True)):
            raise ValueError(str(raw.get("reason") or "vessel plan is not ok"))
        if phase == "close":
            poses = [raw.get("contact_pose") or (raw.get("scan_poses") or [[]])[0]]
        else:
            poses = raw.get("scan_poses")
            points = raw.get("world_xyz")
            if rpy is None and poses:
                rpy = [row[3:6] for row in poses]
    if poses is not None:
        arr = np.asarray(poses, dtype=float).reshape(-1, 6)
        points = arr[:, :3]
        rpy = arr[:, 3:6]
    if points is None:
        raise ValueError("polyline reference needs points, poses, or plan_path")
    speed = payload.get("speed_m_s")
    if speed is None and payload.get("speed_cm_s") is not None:
        speed = 0.01 * float(payload["speed_cm_s"])
    return WorldPolylineReference(
        points,
        rpy=rpy,
        speed_m_s=0.02 if speed is None else float(speed),
        soft_start=bool(payload.get("soft_start", True)),
        ramp_s=float(payload.get("ramp_s", 0.4)),
        euler_order=euler_order,
    )


_SERVO_SECONDARY = frozenset({"track", "hold", "payload_id"})
_RAIL_STILL_M_S = 0.002
_RAIL_STALE_S = 0.050


def _secondary_preset(payload: dict) -> str:
    """SERVO_TWIST / HOLD require an explicit rail/nullspace policy."""

    value = payload.get("secondary")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "SERVO_TWIST / SERVO_TWIST_HOLD requires explicit secondary: "
            "'track', 'hold', or 'payload_id'"
        )
    preset = value.strip()
    if preset not in _SERVO_SECONDARY:
        raise ValueError(
            f"Unsupported servo secondary={preset!r}. "
            "'off' only disables soft tasks; it does not lock the rail. "
            "Use 'payload_id' for rail-locked identification."
        )
    return preset


def _rail_speed_m_s(inner) -> float:
    speeds: list[float] = []
    core = getattr(inner, "core", None)
    qdot = getattr(core, "qdot_prev", None)
    if qdot is not None and np.size(qdot) > 0:
        vel = float(np.asarray(qdot, dtype=float).reshape(-1)[0])
        if np.isfinite(vel):
            speeds.append(vel)
    for key in ("last_v_r_ref", "last_rail_exec_velocity_m_s"):
        raw = getattr(inner, key, None)
        if raw is None:
            continue
        try:
            vel = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(vel):
            speeds.append(vel)
    if not speeds:
        return 0.0
    return float(max(abs(vel) for vel in speeds))


def _rail_feedback_age_s(inner) -> float:
    core = getattr(inner, "core", None)
    for obj in (inner, core):
        if obj is None:
            continue
        raw = getattr(obj, "last_rail_feedback_age_s", None)
        if raw is None:
            raw = getattr(obj, "rail_feedback_age_s", None)
        if raw is None:
            continue
        try:
            age = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(age):
            return age
    return float("nan")


def _require_still_rail_for_payload_id(inner) -> None:
    """Refuse to lock a moving or stale rail into payload_id."""

    if _rail_speed_m_s(inner) > _RAIL_STILL_M_S:
        raise ValueError(
            "payload_id requires a still rail "
            f"(|v_r|≤{1e3 * _RAIL_STILL_M_S:.0f} mm/s). Stop first."
        )
    age = _rail_feedback_age_s(inner)
    if np.isfinite(age) and age > _RAIL_STALE_S:
        raise ValueError(
            f"payload_id refused: rail feedback stale ({1e3 * age:.0f} ms)"
        )


def apply_qp_aux(inner, payload: dict | None) -> None:
    """Apply optional QP auxiliary overrides carried on a mode payload."""

    pay = dict(payload or {})
    aux = dict(pay.get("qp_aux") or {})
    if pay.get("collision_avoidance") is not None:
        aux["collision"] = pay["collision_avoidance"]
    if not aux:
        return
    core = getattr(inner, "core", None)
    if "collision" in aux and core is not None and hasattr(core, "set_collision_enabled"):
        core.set_collision_enabled(bool(aux["collision"]))
    if "centering" in aux:
        inner.set_centering_suppressed(not bool(aux["centering"]))
    if "arm_angle" in aux:
        inner.set_arm_task_suppressed(not bool(aux["arm_angle"]))
    if "manipulability" in aux:
        inner.set_manipulability_active(bool(aux["manipulability"]))
    if "singularity_escape" in aux and core is not None:
        cfg = getattr(getattr(core, "cfg", None), "sigma_setbased", None)
        if cfg is not None:
            cfg.enabled = bool(aux["singularity_escape"])
        tracker = getattr(core, "sigma_setbased", None)
        tcfg = getattr(tracker, "cfg", None)
        if tcfg is not None:
            tcfg.enabled = bool(aux["singularity_escape"])


def compile_request(
    ctx: CompileContext,
    req: ModeRequest,
    *,
    raw: dict | None = None,
    twist_read: Callable | None = None,
    dt: float = 0.005,
) -> Phase:
    payload = dict(req.payload)
    raw = raw or {}
    if req.mode in (Mode.SERVO_TWIST, Mode.SERVO_TWIST_HOLD):
        payload["secondary"] = _secondary_preset(payload)
        if payload["secondary"] == "payload_id":
            _require_still_rail_for_payload_id(ctx.inner)
    if req.mode == Mode.SERVO_TWIST:
        outer = ServoTwistOuter(
            _twist_source(payload, twist_read),
            control_frame=ctx.control_frame,
            euler_order=ctx.euler_order,
            **slew_kwargs(payload),
        )
        phase = Phase(
            outer=outer,
            label=str(payload.get("label") or "servo_twist"),
            duration_s=payload.get("duration_s"),
        )
        return _finish_phase(ctx, payload, phase)
    if req.mode == Mode.SERVO_TWIST_HOLD:
        outer = ServoTwistHoldOuter(
            _twist_source(payload, twist_read),
            control_frame=ctx.control_frame,
            euler_order=ctx.euler_order,
            dt=dt,
            **slew_kwargs(payload),
        )
        phase = Phase(
            outer=outer,
            label=str(payload.get("label") or "servo_twist_hold"),
            duration_s=payload.get("duration_s"),
        )
        return _finish_phase(ctx, payload, phase)
    if req.mode == Mode.TRACK_CARTESIAN:
        kind = str(payload.get("reference", "ellipse"))
        if kind == "hold":
            ref = HoldReference()
        elif kind == "polyline":
            ref = _polyline_ref(payload, ctx.euler_order)
        else:
            ref = _ellipse_ref(payload, ctx.euler_order)
        track_dur = payload.get("duration_s")
        if kind == "ellipse":
            track_dur = getattr(ref, "duration_s", track_dur)
        return _finish_phase(
            ctx,
            payload,
            build_track_cartesian_phase(
                ctx,
                ref,
                duration_s=track_dur,
                label=str(payload.get("label", "track_cartesian")),
                max_lin_vel_m_s=payload.get("max_lin_vel_m_s"),
                move_kp=payload.get("move_kp"),
            ),
        )
    if req.mode == Mode.TRACK_HYBRID:
        kind = str(payload.get("reference", "hold"))
        if kind in ("pad", "twist", "servo"):
            return _finish_phase(
                ctx,
                payload,
                build_pad_hybrid_phase(
                    ctx,
                    twist_read=_twist_source(payload, twist_read),
                    duration_s=payload.get("duration_s"),
                    dt=dt,
                    label=str(payload.get("label", "track_hybrid_pad")),
                    payload=payload,
                ),
            )
        if kind == "ellipse":
            ref = _ellipse_ref(payload, ctx.euler_order)
        elif kind == "polyline":
            ref = _polyline_ref(payload, ctx.euler_order)
        else:
            ref = HoldReference()
        hybrid_dur = payload.get("duration_s")
        if kind == "ellipse":
            hybrid_dur = getattr(ref, "duration_s", hybrid_dur)
        return _finish_phase(
            ctx,
            payload,
            build_track_hybrid_phase(
                ctx,
                ref,
                duration_s=hybrid_dur,
                dt=dt,
                use_tff_split=bool(payload.get("use_tff_split", False)),
                payload=payload,
            ),
        )
    if req.mode in (Mode.GOTO_JOINTS, Mode.MOVEJ):
        if payload.get("q_target") is None:
            if payload.get("pose") is None:
                raise ValueError("MOVEJ needs q_target or pose")
            q = resolve_pose_q(
                ctx,
                payload["pose"],
                q_seed=None if payload.get("q_start") is None else np.asarray(payload["q_start"], dtype=float),
                rail_m=payload.get("rail_m"),
                require_path=False,
            )
        else:
            q = np.asarray(payload["q_target"], dtype=float).reshape(-1)
        q0 = payload.get("q_start")
        q_start = None if q0 is None else np.asarray(q0, dtype=float)
        dur = payload.get("duration_s")
        v = payload.get("v")
        if req.mode == Mode.GOTO_JOINTS:
            return _finish_phase(
                ctx,
                payload,
                build_goto_joints_phase(ctx, q, q_start=q_start, duration_s=dur, v=v),
            )
        return _finish_phase(
            ctx,
            payload,
            build_movej_phase(
                ctx,
                q,
                q_start=q_start,
                duration_s=dur,
                v=v,
                secondary=payload.get("secondary"),
            ),
        )
    if req.mode == Mode.CARTESIAN_PTP:
        return _finish_phase(ctx, payload, build_cartesian_ptp_phase(ctx, payload, dt=dt))
    if req.mode == Mode.MOVES:
        return _finish_phase(ctx, payload, build_moves_phase(ctx, payload, dt=dt))
    raise ValueError(f"unknown mode {req.mode}")


def _truthy(value) -> bool:
    return value is True or value in (1, "1", "true", "True", "yes")


def _attach_joint_hold(phase: Phase, inner) -> None:
    """Freeze q after a payload-ID PTP. QPIK/CBF must not shove the TCP."""

    prev_on_enter = phase.on_enter
    prev_on_exit = phase.on_exit

    def _enter() -> None:
        if prev_on_enter is not None:
            prev_on_enter()
        inner.set_direct_joint_ptp(True)
        inner.set_plan_drives_rail(False)

    def _exit() -> None:
        inner.set_direct_joint_ptp(False)
        if prev_on_exit is not None:
            prev_on_exit()

    phase.on_enter = _enter
    phase.on_exit = _exit
    phase.qdot_ff_provider = lambda _t: np.zeros(8, dtype=float)


def _finish_phase(ctx: CompileContext, payload: dict, phase: Phase) -> Phase:
    preset = payload.get("secondary")
    if preset or payload.get("qp_aux") or payload.get("collision_avoidance") is not None:
        prev = phase.on_enter

        def _enter() -> None:
            if prev is not None:
                prev()
            if preset is not None:
                SecondaryPolicy(preset=str(preset)).apply(ctx.inner)
            apply_qp_aux(ctx.inner, payload)

        phase.on_enter = _enter
    if _truthy(payload.get("joint_hold") or payload.get("freeze_joints")):
        _attach_joint_hold(phase, ctx.inner)
    return phase


class ProxyOuter:
    """Swap the live outer without leaving the 200 Hz runner."""

    def __init__(self, child) -> None:
        self._child = child

    def bind(self, child) -> None:
        self._child = child

    @property
    def child(self):
        return self._child

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        if not hasattr(self._child, "set_origin"):
            return
        import inspect

        fn = self._child.set_origin
        params = inspect.signature(fn).parameters
        if "t_s" in params:
            fn(pose0, t_s=t_s)
        else:
            fn(pose0)

    def begin_hybrid_episode(self, applied_twist_base, current_pose) -> None:
        if hasattr(self._child, "begin_hybrid_episode"):
            self._child.begin_hybrid_episode(applied_twist_base, current_pose)

    def sample(
        self,
        t_s: float,
        current_pose: np.ndarray,
        f_ext: np.ndarray,
        *,
        q_meas=None,
        f_ext_raw=None,
        dt_actual=None,
        v_tcp_z_actual=None,
        sensor_age_s=None,
        feedback_age_s=None,
        feedback_fresh_tick=None,
        feedback_velocity_valid=None,
        **kwargs,
    ):
        import inspect

        fn = self._child.sample
        params = inspect.signature(fn).parameters
        extra = {
            "q_meas": q_meas,
            "f_ext_raw": f_ext_raw,
            "dt_actual": dt_actual,
            "v_tcp_z_actual": v_tcp_z_actual,
            "sensor_age_s": sensor_age_s,
            "feedback_age_s": feedback_age_s,
            "feedback_fresh_tick": feedback_fresh_tick,
            "feedback_velocity_valid": feedback_velocity_valid,
        }
        extra.update(kwargs)
        kw = {
            key: val
            for key, val in extra.items()
            if key in params and val is not None
        }
        return fn(t_s, current_pose, f_ext, **kw)

    def __getattr__(self, name):
        return getattr(self._child, name)


class ModeEngine:
    """Offline / unit-test engine: compile and sample without hardware."""

    def __init__(self, ctx: CompileContext, *, raw: dict | None = None, dt: float = 0.005) -> None:
        self.ctx = ctx
        self.raw = raw or {}
        self.dt = float(dt)
        self.phase: Phase | None = None
        self.mode = Mode.SERVO_TWIST

    def set_mode(self, req: ModeRequest, *, twist_read=None) -> Phase:
        self.mode = req.mode
        self.phase = compile_request(
            self.ctx, req, raw=self.raw, twist_read=twist_read, dt=self.dt
        )
        return self.phase

    def sample(
        self,
        t_s: float,
        pose: np.ndarray,
        f_ext: np.ndarray,
        *,
        q_meas: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.phase is None:
            raise RuntimeError("set_mode first")
        kwargs = {}
        import inspect

        params = inspect.signature(self.phase.outer.sample).parameters
        if "q_meas" in params and q_meas is not None:
            kwargs["q_meas"] = q_meas
        return np.asarray(
            self.phase.outer.sample(t_s, pose, f_ext, **kwargs), dtype=float
        )
