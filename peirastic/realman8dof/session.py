"""Compile ModeRequest → Phase and hold the live outer for offline sample()."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from rm75_control.control.joint_admittance_8dof.api import CompileContext, SecondaryPolicy
from rm75_control.control.joint_admittance_8dof.loop import Phase
from rm75_control.control.joint_admittance_8dof.reference import (
    EllipseToolXYReference,
    HoldReference,
    JointSmoothMoveReference,
)
from peirastic.core.modes import Mode, ModeRequest
from peirastic.realman8dof.modes.joint import build_goto_joints_phase, build_movej_phase
from peirastic.realman8dof.modes.servo import ServoTwistHoldOuter, ServoTwistOuter
from peirastic.realman8dof.modes.track import (
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
    return EllipseToolXYReference(
        ax,
        ay,
        period_s=None if period is None else float(period),
        max_vel_m_s=None if vmax is None else float(vmax),
        soft_start=bool(payload.get("soft_start", True)),
        ramp_s=float(payload.get("ramp_s", 2.0)),
        euler_order=euler_order,
    )


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
    if req.mode == Mode.SERVO_TWIST:
        outer = ServoTwistOuter(
            _twist_source(payload, twist_read),
            control_frame=ctx.control_frame,
            euler_order=ctx.euler_order,
        )
        phase = Phase(outer=outer, label="servo_twist", duration_s=payload.get("duration_s"))
        phase.on_enter = lambda: SecondaryPolicy(preset="track").apply(ctx.inner)
        return phase
    if req.mode == Mode.SERVO_TWIST_HOLD:
        outer = ServoTwistHoldOuter(
            _twist_source(payload, twist_read),
            control_frame=ctx.control_frame,
            euler_order=ctx.euler_order,
            dt=dt,
        )
        phase = Phase(
            outer=outer, label="servo_twist_hold", duration_s=payload.get("duration_s")
        )
        phase.on_enter = lambda: SecondaryPolicy(preset="track").apply(ctx.inner)
        return phase
    if req.mode == Mode.TRACK_CARTESIAN:
        kind = str(payload.get("reference", "ellipse"))
        if kind == "hold":
            ref = HoldReference()
        else:
            ref = _ellipse_ref(payload, ctx.euler_order)
        return build_track_cartesian_phase(
            ctx,
            ref,
            duration_s=payload.get("duration_s"),
            label=str(payload.get("label", "track_cartesian")),
        )
    if req.mode == Mode.TRACK_HYBRID:
        kind = str(payload.get("reference", "hold"))
        if kind == "ellipse":
            ref = _ellipse_ref(payload, ctx.euler_order)
        else:
            ref = HoldReference()
        return build_track_hybrid_phase(
            ctx,
            ref,
            raw=raw,
            desired_z=float(payload.get("desired_z", 0.0)),
            duration_s=payload.get("duration_s"),
            dt=dt,
            use_tff_split=bool(payload.get("use_tff_split", False)),
        )
    if req.mode in (Mode.GOTO_JOINTS, Mode.MOVEJ):
        q = np.asarray(payload["q_target"], dtype=float).reshape(-1)
        q0 = payload.get("q_start")
        q_start = None if q0 is None else np.asarray(q0, dtype=float)
        dur = payload.get("duration_s")
        if req.mode == Mode.GOTO_JOINTS:
            return build_goto_joints_phase(ctx, q, q_start=q_start, duration_s=dur)
        return build_movej_phase(ctx, q, q_start=q_start, duration_s=dur)
    raise ValueError(f"unknown mode {req.mode}")


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
        if hasattr(self._child, "set_origin"):
            self._child.set_origin(pose0, t_s=t_s)

    def begin_hybrid_episode(self, applied_twist_base, current_pose) -> None:
        if hasattr(self._child, "begin_hybrid_episode"):
            self._child.begin_hybrid_episode(applied_twist_base, current_pose)

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray, **kwargs):
        import inspect

        fn = self._child.sample
        params = inspect.signature(fn).parameters
        kw = {key: val for key, val in kwargs.items() if key in params}
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
