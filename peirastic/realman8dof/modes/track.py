"""Cartesian track and force-position hybrid (TFF + ForceLaw)."""

from __future__ import annotations

import numpy as np

from rm75_control.control.admittance_common.reference import MotionReferenceSource
from rm75_control.control.joint_admittance_8dof.api import (
    CompileContext,
    SecondaryPolicy,
    compile_phase,
    phase_cartesian_track,
    phase_hybrid_track,
)
from rm75_control.control.joint_admittance_8dof.loop import (
    AdmittanceOuterLoop,
    CartesianTrackOuterLoop,
    Phase,
)
from peirastic.realman8dof.force.config import build_force_controller
from peirastic.realman8dof.force.legacy import LegacyForceLaw
from peirastic.realman8dof.force.tff import SELECTION_TOOL_Z_FORCE, compose_tff
from peirastic.realman8dof.modes.servo import ServoTwistOuter


class HybridTffOuter:
    """Position axes: k e + v_ff. Force axes: ForceLaw. Compose with TFF."""

    def __init__(
        self,
        position: CartesianTrackOuterLoop,
        force_law,
        *,
        desired_force: np.ndarray,
        selection: np.ndarray | None = None,
        dt: float = 0.005,
        mask_force_from_path: bool = True,
    ) -> None:
        self.position = position
        self.force_law = force_law
        self.desired_force = np.asarray(desired_force, dtype=float).reshape(6)
        self.selection = (
            np.asarray(SELECTION_TOOL_Z_FORCE, dtype=float)
            if selection is None
            else np.asarray(selection, dtype=float).reshape(6)
        )
        self.dt = float(dt)
        self.mask_force_from_path = bool(mask_force_from_path)
        self.last_err_mm = 0.0
        self.last_vel_ff = np.zeros(6, dtype=float)
        self.last_pose_d: np.ndarray | None = None
        self.last_path_twist = np.zeros(6, dtype=float)
        self.last_feedback_twist = np.zeros(6, dtype=float)
        self.controller = getattr(force_law, "controller", None)

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        self.position.set_origin(pose0, t_s=t_s)
        self.force_law.reset(pose=pose0, f_ext=np.zeros(6))

    def begin_hybrid_episode(self, applied_twist_base, current_pose) -> None:
        del applied_twist_base
        self.force_law.reset(pose=current_pose, f_ext=np.zeros(6))

    def sample(
        self,
        t_s: float,
        current_pose: np.ndarray,
        f_ext: np.ndarray,
        *,
        contact: bool | None = None,
        f_ext_raw: np.ndarray | None = None,
        dt_actual: float | None = None,
        sensor_age_s: float | None = None,
        feedback_age_s: float | None = None,
        feedback_fresh_tick: bool | None = None,
        feedback_velocity_valid: bool | None = None,
        v_tcp_z_actual: float | None = None,
    ) -> np.ndarray:
        del feedback_fresh_tick
        velocity_valid = (
            bool(feedback_velocity_valid)
            if feedback_velocity_valid is not None
            else v_tcp_z_actual is not None
        )
        v_actual = v_tcp_z_actual if velocity_valid else None
        v_pos = np.asarray(
            self.position.sample(t_s, current_pose, f_ext), dtype=float
        ).reshape(6)
        path = np.asarray(self.position.last_path_twist, dtype=float).reshape(6)
        if self.mask_force_from_path:
            path = path * self.selection
            v_pos = v_pos * self.selection
        fout = self.force_law.update(
            dt_s=float(dt_actual) if dt_actual is not None else self.dt,
            pose=current_pose,
            f_ext=np.asarray(f_ext, dtype=float).reshape(6),
            f_des=self.desired_force,
            path_twist=path,
            contact=contact,
            f_ext_raw=f_ext_raw,
            dt_actual=dt_actual,
            sensor_age_s=sensor_age_s,
            feedback_age_s=feedback_age_s,
            v_tcp_z_actual=v_actual,
        )
        v_star = compose_tff(v_pos, fout.v_force, self.selection)
        self.last_err_mm = float(self.position.last_err_mm)
        self.last_vel_ff = np.asarray(self.position.last_vel_ff, dtype=float).copy()
        self.last_pose_d = (
            None
            if self.position.last_pose_d is None
            else np.asarray(self.position.last_pose_d, dtype=float).copy()
        )
        self.last_path_twist = path
        self.last_feedback_twist = np.asarray(
            self.position.last_feedback_twist, dtype=float
        ).copy()
        return v_star


def build_track_cartesian_phase(
    ctx: CompileContext,
    reference: MotionReferenceSource,
    *,
    duration_s: float | None = None,
    label: str = "track_cartesian",
    max_lin_vel_m_s: float | None = None,
    move_kp: float | None = None,
) -> Phase:
    kwargs = {}
    if max_lin_vel_m_s is not None:
        kwargs["max_lin_vel_m_s"] = float(max_lin_vel_m_s)
    if move_kp is not None:
        kwargs["move_kp"] = float(move_kp)
    spec = phase_cartesian_track(
        reference, label=label, duration_s=duration_s, **kwargs
    )
    return compile_phase(spec, ctx).phase


def _hybrid_controller(dt: float, payload: dict | None = None):
    controller, _raw, desired_z = build_force_controller(dt, payload=payload)
    f_des = np.zeros(6, dtype=float)
    f_des[2] = float(desired_z)
    return controller, f_des


def build_pad_hybrid_phase(
    ctx: CompileContext,
    *,
    twist_read,
    duration_s: float | None = None,
    dt: float = 0.005,
    label: str = "track_hybrid_pad",
    payload: dict | None = None,
) -> Phase:
    if twist_read is None:
        raise ValueError("pad hybrid needs a live twist source")
    controller, f_des = _hybrid_controller(dt, payload)
    pos = ServoTwistOuter(
        twist_read,
        control_frame=ctx.control_frame,
        euler_order=ctx.euler_order,
    )
    outer = HybridTffOuter(
        pos,
        LegacyForceLaw(controller),
        desired_force=f_des,
        dt=dt,
        mask_force_from_path=True,
    )
    phase = Phase(outer=outer, label=label, duration_s=duration_s)
    phase.on_enter = lambda: SecondaryPolicy(preset="track").apply(ctx.inner)
    return phase


def build_track_hybrid_phase(
    ctx: CompileContext,
    reference: MotionReferenceSource,
    *,
    duration_s: float | None = None,
    dt: float = 0.005,
    label: str = "track_hybrid",
    use_tff_split: bool = False,
    payload: dict | None = None,
) -> Phase:
    controller, f_des = _hybrid_controller(dt, payload)
    if not use_tff_split:
        spec = phase_hybrid_track(
            reference,
            controller,
            desired_force=f_des,
            label=label,
            duration_s=duration_s,
        )
        return compile_phase(spec, ctx).phase
    cart = compile_phase(
        phase_cartesian_track(reference, label=label, duration_s=duration_s),
        ctx,
    )
    outer = HybridTffOuter(
        cart.outer,
        LegacyForceLaw(controller),
        desired_force=f_des,
        dt=dt,
    )
    phase = cart.phase
    phase.outer = outer
    phase.label = label
    return phase


def wrap_admittance(reference, controller, desired_force) -> AdmittanceOuterLoop:
    return AdmittanceOuterLoop(controller, reference, desired_force=desired_force)
