"""Cartesian track and force-position hybrid (TFF + ForceLaw)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from rm75_control.control.admittance_common.reference import MotionReferenceSource
from rm75_control.control.joint_admittance_8dof.api import (
    CompileContext,
    SecondaryPolicy,
    compile_phase,
    phase_cartesian_track,
    phase_hybrid_track,
)
from rm75_control.control.admittance_common.pose_math import pose_track_error_mm_deg
from rm75_control.control.joint_admittance_8dof.loop import (
    AdmittanceOuterLoop,
    CartesianTrackOuterLoop,
    Phase,
)
from peirastic.realman8dof.force.config import build_force_controller
from peirastic.realman8dof.force.fce import FceAdmittanceLaw, use_fce_law
from peirastic.realman8dof.force.legacy import LegacyForceLaw
from peirastic.realman8dof.force.tff import SELECTION_TOOL_Z_FORCE, compose_tff
from peirastic.realman8dof.force.torque_tilt import (
    LegacyForceWithTilt,
    TorqueTilt,
    TorqueTiltConfig,
    apply_tilt_selection,
)
from peirastic.realman8dof.modes.servo import ServoTwistOuter, slew_kwargs


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
        self.last_force_err_mm = 0.0
        self.last_vel_ff = np.zeros(6, dtype=float)
        self.last_pose_d: np.ndarray | None = None
        self.last_path_twist = np.zeros(6, dtype=float)
        self.last_feedback_twist = np.zeros(6, dtype=float)
        self.last_tau_y = float("nan")
        self.last_tau_error_y = float("nan")
        self.last_omega_y = float("nan")
        self.last_theta_tilt = float("nan")
        self.last_tilt_engaged = False
        self.last_tilt_frozen = False
        self.last_tilt_capped = False
        self.last_tilt_stalled = False
        self.last_tilt_stop_reason = ""
        self.last_tilt_deadband_nm = float("nan")
        self.last_cop_x = float("nan")
        self.last_cop_y = float("nan")
        self.last_cop_r = float("nan")
        self.last_on_tube = False
        self.controller = getattr(force_law, "controller", None)
        cfg = getattr(self.position, "cfg", None)
        if cfg is not None and hasattr(cfg, "track_axes"):
            cfg.track_axes = self.selection.copy()

    def _euler_order(self) -> str:
        cfg = getattr(self.position, "cfg", None)
        if cfg is not None:
            return str(getattr(cfg, "euler_order", "xyz"))
        return str(getattr(self.position, "euler_order", "xyz"))

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
        slack_norm: float | None = None,
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
            slack_norm=slack_norm,
            euler_order=self._euler_order(),
        )
        v_star = compose_tff(v_pos, fout.v_force, self.selection)
        telemetry = dict(getattr(fout, "telemetry", None) or {})
        self.last_tau_y = float(telemetry.get("tau_y", float("nan")))
        self.last_tau_error_y = float(telemetry.get("tau_error_y", float("nan")))
        self.last_omega_y = float(telemetry.get("omega_y", float("nan")))
        self.last_theta_tilt = float(telemetry.get("theta_tilt", float("nan")))
        self.last_tilt_engaged = bool(telemetry.get("tilt_engaged", False))
        self.last_tilt_frozen = bool(telemetry.get("tilt_frozen", False))
        self.last_tilt_capped = bool(telemetry.get("tilt_capped", False))
        self.last_tilt_stalled = bool(telemetry.get("tilt_stalled", False))
        self.last_tilt_stop_reason = str(telemetry.get("tilt_stop_reason", ""))
        self.last_tilt_deadband_nm = float(telemetry.get("tilt_deadband_nm", float("nan")))
        self.last_cop_x = float(telemetry.get("cop_x", float("nan")))
        self.last_cop_y = float(telemetry.get("cop_y", float("nan")))
        self.last_cop_r = float(telemetry.get("cop_r", float("nan")))
        self.last_on_tube = bool(telemetry.get("on_tube", False))
        pose_d = getattr(self.position, "last_pose_d", None)
        if pose_d is not None:
            euler = self._euler_order()
            self.last_err_mm, _ = pose_track_error_mm_deg(
                pose_d,
                current_pose,
                track_axes=self.selection,
                euler_order=euler,
            )
            self.last_force_err_mm, _ = pose_track_error_mm_deg(
                pose_d,
                current_pose,
                track_axes=1.0 - self.selection,
                euler_order=euler,
            )
        else:
            self.last_err_mm = float(getattr(self.position, "last_err_mm", 0.0))
            self.last_force_err_mm = 0.0
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


def _desired_force(payload: dict | None, desired_z: float) -> np.ndarray:
    f_des = np.zeros(6, dtype=float)
    f_des[2] = float(desired_z)
    pay = dict(payload or {})
    if pay.get("desired_force") is not None:
        df = np.asarray(pay["desired_force"], dtype=float).reshape(-1)
        if df.size == 1:
            f_des[2] = float(df[0])
        elif df.size >= 6:
            f_des = df[:6].astype(float)
        elif df.size >= 3:
            f_des[2] = float(df[2])
    return f_des


def _hybrid_controller(dt: float, payload: dict | None = None):
    controller, raw, desired_z = build_force_controller(dt, payload=payload)
    return controller, _desired_force(payload, desired_z), raw


def _hybrid_force_law(
    dt: float, payload: dict | None = None, *, control_frame: str = "tool"
):
    if use_fce_law(payload):
        from peirastic.realman8dof.force.config import desired_z_n

        law = FceAdmittanceLaw.from_payload(dt, payload)
        return law, _desired_force(payload, desired_z_n(payload=payload)), TorqueTiltConfig(enabled=False)
    controller, f_des, raw = _hybrid_controller(dt, payload)
    law = LegacyForceLaw(controller)
    tilt_cfg = TorqueTiltConfig.from_dict(raw)
    selection = selection_from_payload(payload)
    if selection is not None and selection[tilt_cfg.axis] >= 1.0:
        tilt_cfg = replace(tilt_cfg, enabled=False)
    if tilt_cfg.enabled:
        if str(control_frame).lower() != "tool":
            raise ValueError("torque_tilt requires tool-frame hybrid control")
        law = LegacyForceWithTilt(law, TorqueTilt(tilt_cfg))
    return law, f_des, tilt_cfg


def selection_from_payload(payload: dict | None) -> np.ndarray | None:
    """Build TFF selection S (1=track, 0=force) from an optional payload."""

    pay = dict(payload or {})
    if pay.get("selection") is not None:
        return np.asarray(pay["selection"], dtype=float).reshape(6)
    if pay.get("track_axes") is not None:
        return np.asarray(pay["track_axes"], dtype=float).reshape(6)
    if pay.get("force_axes") is not None:
        force = np.asarray(pay["force_axes"], dtype=float).reshape(6)
        return np.clip(1.0 - force, 0.0, 1.0)
    return None


def _mask_force_from_path(payload: dict | None, default: bool) -> bool:
    pay = dict(payload or {})
    if pay.get("mask_force_from_path") is None:
        return bool(default)
    return bool(pay["mask_force_from_path"])


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
    force_law, f_des, tilt_cfg = _hybrid_force_law(dt, payload, control_frame=ctx.control_frame)
    selection = apply_tilt_selection(selection_from_payload(payload), tilt_cfg)
    pos = ServoTwistOuter(
        twist_read,
        control_frame=ctx.control_frame,
        euler_order=ctx.euler_order,
        filter_default=True,
        filter_mask=selection,
        **slew_kwargs(payload),
    )
    outer = HybridTffOuter(
        pos,
        force_law,
        desired_force=f_des,
        selection=selection,
        dt=dt,
        mask_force_from_path=_mask_force_from_path(payload, True),
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
    if not use_tff_split:
        controller, f_des, _raw = _hybrid_controller(dt, payload)
        spec = phase_hybrid_track(
            reference,
            controller,
            desired_force=f_des,
            label=label,
            duration_s=duration_s,
        )
        return compile_phase(spec, ctx).phase
    force_law, f_des, tilt_cfg = _hybrid_force_law(dt, payload, control_frame=ctx.control_frame)
    cart = compile_phase(
        phase_cartesian_track(reference, label=label, duration_s=duration_s),
        ctx,
    )
    outer = HybridTffOuter(
        cart.outer,
        force_law,
        desired_force=f_des,
        selection=apply_tilt_selection(selection_from_payload(payload), tilt_cfg),
        dt=dt,
        mask_force_from_path=_mask_force_from_path(payload, True),
    )
    phase = cart.phase
    phase.outer = outer
    phase.label = label
    return phase


def wrap_admittance(reference, controller, desired_force) -> AdmittanceOuterLoop:
    return AdmittanceOuterLoop(controller, reference, desired_force=desired_force)
