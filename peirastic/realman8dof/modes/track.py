"""Cartesian track and force-position hybrid (TFF + ForceLaw)."""

from __future__ import annotations

import numpy as np

from rm75_control.control.admittance_common.controller import AdmittanceController
from rm75_control.control.admittance_common.reference import MotionReferenceSource
from rm75_control.control.admittance_common.scaling import scale_admittance_for_desired_z
from rm75_control.control.joint_admittance_8dof.api import (
    CompileContext,
    compile_phase,
    phase_cartesian_track,
    phase_hybrid_track,
)
from rm75_control.control.joint_admittance_8dof.loop import (
    AdmittanceOuterLoop,
    CartesianTrackOuterLoop,
    Phase,
)
from peirastic.realman8dof.force.legacy import LegacyForceLaw
from peirastic.realman8dof.force.tff import SELECTION_TOOL_Z_FORCE, compose_tff


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
        self.last_err_mm = 0.0
        self.last_vel_ff = np.zeros(6, dtype=float)
        self.last_pose_d: np.ndarray | None = None
        self.last_path_twist = np.zeros(6, dtype=float)
        self.last_feedback_twist = np.zeros(6, dtype=float)
        self.controller = getattr(force_law, "controller", None)

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        self.position.set_origin(pose0, t_s=t_s)
        self.force_law.reset(pose=pose0, f_ext=np.zeros(6))

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        v_pos = np.asarray(
            self.position.sample(t_s, current_pose, f_ext), dtype=float
        ).reshape(6)
        path = np.asarray(self.position.last_path_twist, dtype=float).reshape(6)
        fout = self.force_law.update(
            dt_s=self.dt,
            pose=current_pose,
            f_ext=np.asarray(f_ext, dtype=float).reshape(6),
            f_des=self.desired_force,
            path_twist=path,
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
) -> Phase:
    spec = phase_cartesian_track(reference, label=label, duration_s=duration_s)
    return compile_phase(spec, ctx).phase


def build_track_hybrid_phase(
    ctx: CompileContext,
    reference: MotionReferenceSource,
    *,
    raw: dict,
    desired_z: float,
    duration_s: float | None = None,
    dt: float = 0.005,
    label: str = "track_hybrid",
    use_tff_split: bool = False,
) -> Phase:
    cfg = scale_admittance_for_desired_z(raw, float(desired_z))
    controller = AdmittanceController(dt, cfg)
    f_des = np.zeros(6, dtype=float)
    f_des[2] = float(desired_z)
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
