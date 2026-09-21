"""Cartesian track and force-position hybrid (TFF + ForceLaw)."""

from __future__ import annotations

from dataclasses import replace
import time

import numpy as np

from rm75_control.control.admittance_common.reference import MotionReferenceSource
from rm75_control.control.joint_admittance_8dof.api import (
    CompileContext,
    SecondaryPolicy,
    compile_phase,
    phase_cartesian_track,
    phase_hybrid_track,
    stroke_amplitude_m,
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
from peirastic.realman8dof.modes.contact_reference import ContactGatedReference
from peirastic.realman8dof.modes.servo import ServoTwistOuter, slew_kwargs


def _pad_visual_gamma(guide, force_n, cfg):
    """Pad smoothstep γ when pad.force_gate_* is set; otherwise the v8 linear gate."""

    pad = dict((getattr(guide, "config", None) or {}).get("pad") or {})
    start = pad.get("force_gate_start_n")
    end = pad.get("force_gate_end_n")
    if start is not None and end is not None:
        from peirastic.realman8dof.modes.pad_scan_config import pad_visual_force_gate
        return pad_visual_force_gate(force_n, start, end)
    return cfg.differential_repair.force_gate(force_n)


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
        self.last_path_twist_raw = np.zeros(6, dtype=float)
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
        self.last_pad_alpha = 1.0
        self.last_contact_present = False
        self.last_facts = {}
        self._pad_scan_qp = None
        self._pad_previous_twist = np.zeros(6, dtype=float)
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
        lifecycle = getattr(self, "command_lifecycle", None)
        if lifecycle is not None:
            lifecycle.abort("set_origin", definitely_not_sent=True, latch=False)
        self.position.set_origin(pose0, t_s=t_s)
        self.force_law.reset(pose=pose0, f_ext=np.zeros(6))
        self._pad_previous_twist = np.zeros(6, dtype=float)
        self.last_facts = {}
        lifecycle = getattr(self, "command_lifecycle", None)
        if lifecycle is not None:
            lifecycle.reset(pose0)

    def begin_hybrid_episode(self, applied_twist_base, current_pose) -> None:
        lifecycle = getattr(self, "command_lifecycle", None)
        if lifecycle is not None:
            lifecycle.abort("episode_reset", definitely_not_sent=True, latch=False)
        self.force_law.reset(pose=current_pose, f_ext=np.zeros(6))
        self._pad_previous_twist = np.zeros(6, dtype=float)
        self.last_facts = {}
        lifecycle = getattr(self, "command_lifecycle", None)
        if lifecycle is not None:
            lifecycle.reset(current_pose, applied_twist_base=applied_twist_base)

    def sample(
        self, t_s, current_pose, f_ext, *, contact=None, f_ext_raw=None,
        dt_actual=None, sensor_age_s=None, feedback_age_s=None,
        feedback_fresh_tick=None, feedback_velocity_valid=None,
        v_tcp_z_actual=None, slack_norm=None, wrench_source_time_s=None,
        wrench_source_id=None, wrench_source_wall_time_ns=None,
        measured_twist_base=None, measured_twist_valid=False,
        measured_twist_fresh=False, measurement_time_s=None,
        measured_twist_metadata=None,
    ):
        kwargs = dict(contact=contact, f_ext_raw=f_ext_raw, dt_actual=dt_actual,
            sensor_age_s=sensor_age_s, feedback_age_s=feedback_age_s,
            feedback_fresh_tick=feedback_fresh_tick,
            feedback_velocity_valid=feedback_velocity_valid,
            v_tcp_z_actual=v_tcp_z_actual, slack_norm=slack_norm)
        lifecycle = getattr(self, "command_lifecycle", None)
        if lifecycle is None:
            return self._sample(t_s, current_pose, f_ext, **kwargs)
        return lifecycle.sample(self._sample, t_s, current_pose, f_ext, kwargs,
            wrench_source_time_s=wrench_source_time_s, wrench_source_id=wrench_source_id,
            wrench_source_wall_time_ns=wrench_source_wall_time_ns,
            measured_twist_base=measured_twist_base, measured_twist_valid=measured_twist_valid,
            measured_twist_fresh=measured_twist_fresh, measurement_time_s=measurement_time_s,
            measured_twist_metadata=measured_twist_metadata)

    def _sample(
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
        gate = getattr(self, "contact_gate", None)
        if gate is not None and not gate.started and gate.start_force_n is not None:
            gate.reanchor_hold(current_pose)
        if gate is not None and gate.start_force_n is not None:
            gate.guard_approach(current_pose)
            age = max(float(sensor_age_s or 0), float(feedback_age_s or 0))
            if feedback_fresh_tick is not False:
                gate.observe_force(float(f_ext[2]), time.monotonic() - age,
                                   valid=np.isfinite(age) and 0 <= age <= 0.1)
        velocity_valid = (
            bool(feedback_velocity_valid)
            if feedback_velocity_valid is not None
            else v_tcp_z_actual is not None
        )
        v_actual = v_tcp_z_actual if velocity_valid else None
        v_pos = np.asarray(
            self.position.sample(t_s, current_pose, f_ext), dtype=float
        ).reshape(6)
        path_raw = np.asarray(self.position.last_path_twist, dtype=float).reshape(6)
        path = path_raw
        if self.mask_force_from_path:
            path = path_raw * self.selection
            v_pos = v_pos * self.selection
        dt_s = float(dt_actual) if dt_actual is not None else self.dt
        visual_omega = None
        guide = getattr(self, "pad_visual", None)
        present = contact
        if present is None:
            present = bool(getattr(getattr(self.force_law, "controller", None),
                                   "contact_present", False))
        self.last_contact_present = bool(present)
        if guide is not None:
            lifecycle = getattr(self, "command_lifecycle", None)
            preview_kwargs = dict(
                force_n=float(np.asarray(f_ext, dtype=float).reshape(6)[2]),
                stick_speed_m_s=float(np.linalg.norm(path_raw[:2])),
            )
            if lifecycle is not None:
                preview_kwargs['force_n'] = lifecycle.force_n
            visual_omega = guide.preview(
                pose=current_pose,
                dt_s=dt_s,
                contact_present=(bool(present) if lifecycle is None else
                                 lifecycle._pending['preview_contact']),
                tilt=getattr(self.force_law, "tilt", None),
                **preview_kwargs,
            )
        fuse = (
            guide is not None
            and bool(getattr(guide, "progress", False))
            and getattr(self.controller, "cfg", None) is not None
        )
        alpha = 1.0
        published = getattr(guide, "last_alpha", None) if guide is not None else None
        if published is not None:
            alpha = float(published)
            if not np.isfinite(alpha):
                alpha = 1.0
            else:
                alpha = float(np.clip(alpha, 0.0, 1.0))
            if alpha != 1.0 and not fuse:
                v_pos = v_pos * alpha
                path = path * alpha
        self.last_pad_alpha = alpha
        lifecycle = getattr(self, "command_lifecycle", None)
        force_update = self.force_law.update if lifecycle is None else lifecycle.prepare_force
        fout = force_update(
            dt_s=dt_s,
            pose=current_pose,
            f_ext=np.asarray(f_ext, dtype=float).reshape(6),
            f_des=self.desired_force,
            path_twist=path if fuse else path,
            path_twist_raw=path_raw,
            contact=contact,
            f_ext_raw=f_ext_raw,
            dt_actual=dt_actual,
            sensor_age_s=sensor_age_s,
            feedback_age_s=feedback_age_s,
            v_tcp_z_actual=v_actual,
            slack_norm=slack_norm,
            euler_order=self._euler_order(),
            visual_velocity_target_rad_s=None if fuse else visual_omega,
        )
        allocate = self._allocate_pad_scan_qp if lifecycle is None else lifecycle.allocate
        allocated = allocate(
            current_pose, f_ext, fout, path_raw, guide, bool(present), dt_s,
        ) if fuse else None
        if allocated is not None:
            v_star = allocated
        else:
            failed_facts = dict(self.last_facts)
            added = getattr(guide, "last_visual_omega", None) if guide is not None else None
            if added is not None:
                force = np.asarray(fout.v_force, dtype=float).reshape(6).copy()
                force[4] = float(force[4]) + float(added)
                tilt = getattr(self.force_law, "tilt", None)
                vmax = getattr(getattr(tilt, "cfg", None), "vmax_rad_s", None)
                if vmax is None:
                    vmax = getattr(guide, "omega_vmax", 0.28)
                limit = max(float(vmax), 1.0e-6)
                force[4] = float(np.clip(force[4], -limit, limit))
                fout.v_force = force
            if fuse and alpha != 1.0:
                v_pos = v_pos * alpha
                path = path * alpha
            v_star = compose_tff(v_pos, fout.v_force, self.selection)
            self.last_facts = failed_facts
        if guide is not None and lifecycle is None:
            from peirastic.realman8dof.force.torque_tilt import rotation_from_pose
            guide.commit(v_star, rotation_from_pose(current_pose, euler_order=self._euler_order()))
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
        if allocated is None:
            self.last_path_twist = path
        else:
            self.last_vel_ff = np.asarray(self.last_path_twist, dtype=float).copy()
        self.last_path_twist_raw = path_raw
        self.last_feedback_twist = np.asarray(
            self.position.last_feedback_twist, dtype=float
        ).copy()
        if lifecycle is None:
            self._pad_previous_twist = np.asarray(v_star, dtype=float).reshape(6).copy()
        return v_star

    def _pad_scan_solver(self, guide):
        if self._pad_scan_qp is not None:
            return self._pad_scan_qp
        from dataclasses import fields
        from peirastic.contact_qp.qp import ContactQp, QpConfig
        raw = dict((guide.config or {}).get("qp") or {})
        feature = dict((guide.config or {}).get("feature") or {})
        settings = {key: raw[key] for key in raw if key in {item.name for item in fields(QpConfig)}}
        settings["allocation_policy"] = "delay_kf_cop_v1"
        settings["enable_aperture"] = False
        if feature.get("c_min") is not None:
            settings["c_min"] = feature["c_min"]
        if feature.get("quality_policy_version"):
            settings["quality_policy_version"] = feature["quality_policy_version"]
        vmax = np.asarray(self.controller.cfg.max_velocity, dtype=float).copy()
        vmax[2] = min(vmax[2], self.controller.cfg.max_vz_tool_m_s)
        tilt = self.force_law.tilt.cfg
        vmax[4] = min(vmax[4], tilt.vmax_rad_s)
        accel = np.asarray(self.controller.cfg.max_acceleration, dtype=float).copy()
        accel[4] = min(accel[4], tilt.a_max)
        pad = dict((guide.config or {}).get("pad") or {})
        if pad.get("angular_max_rad_s") is not None:
            vmax[4] = min(vmax[4], float(pad["angular_max_rad_s"]))
        if pad.get("angular_acceleration_rad_s2") is not None:
            accel[4] = min(accel[4], float(pad["angular_acceleration_rad_s2"]))
        settings["max_velocity"] = vmax
        settings["max_acceleration"] = accel
        settings["angle_limit_rad"] = tilt.theta_max_rad
        self._pad_scan_qp = ContactQp(QpConfig(**settings))
        return self._pad_scan_qp

    def _allocate_pad_scan_qp(self, pose, f_ext, fout, path_raw, guide, contact, dt_s):
        """Same delay-KF COP / My / α / visual allocation as phantom Lissajous.

        Publish the resulting tool twist on the 994742bb handheld path. Do not
        re-inject Cartesian rows into native IK.
        """
        try:
            from peirastic.contact_qp.geometry import (
                contact_cop_from_wrench, twist_tcp_to_face, wrench_tcp_to_face,
            )
            from peirastic.contact_qp.qp import QpInput
            solver = self._pad_scan_solver(guide)
            cfg = solver.config
            geometry = guide.geometry
            transform = twist_tcp_to_face(geometry)
            wrench = np.asarray(f_ext, dtype=float).reshape(6)
            env = wrench_tcp_to_face(-wrench, geometry)
            cop, cop_reason = contact_cop_from_wrench(
                env, min_force_n=cfg.cop_min_force_n,
                max_abs_m=min(geometry.half_length_m, cfg.cop_max_m),
            )
            mechanical = np.asarray(fout.v_force, dtype=float).reshape(6)
            feedback = np.asarray(self.position.last_feedback_twist, dtype=float).reshape(6).copy()
            moving = np.asarray(path_raw, dtype=float).reshape(6).copy()
            if self.mask_force_from_path:
                feedback = feedback * self.selection
                moving = moving * self.selection
            feedback[[2, 4]] = 0.
            moving[[2, 4]] = 0.
            if np.linalg.norm(moving[:2]) < float(np.min(cfg.subspace_tolerance[:2])):
                moving[:2] = 0.
            hold = min(float(self.dt), float(dt_s) if np.isfinite(dt_s) and dt_s > 0 else self.dt)
            slew = max(hold, float(dt_s) if np.isfinite(dt_s) and dt_s > 0 else hold)
            # Slewed pad α (994742bb). Image ω is the scan request; QP slews Ω.
            alpha_des = float(np.clip(getattr(guide, "last_alpha", getattr(guide, "alpha_cmd", 1.)), 0., 1.))
            request = float(getattr(guide, "omega_target", 0.) or 0.)
            visual_valid = bool(contact) and bool(np.isfinite(request))
            tilt = getattr(self.force_law, "tilt", None)
            measured = float(getattr(tilt, "theta_tilt", 0.) or 0.)
            stick = feedback + alpha_des * moving
            previous = np.asarray(self._pad_previous_twist, dtype=float).reshape(6).copy()
            # Scan's 5 ms accel box would clip a 20 mm/s stick to a_max*dt.
            # Keep the handheld path instant; only Z / My slew from last publish.
            if not np.any(np.abs(previous) > 1e-12):
                previous = compose_tff(stick, mechanical, self.selection)
            else:
                previous = np.where(self.selection > 0.5, stick, previous)
            result = solver.solve(QpInput(
                geometry=geometry, nominal_twist=mechanical, path_twist=moving,
                force_n=float(-env[2]), dt_s=hold, now_s=time.monotonic(),
                previous_twist=previous, measured_angle=measured,
                acceleration_dt_s=slew, alpha_preferred=alpha_des,
                mechanical_normal_m_s=float((transform @ mechanical)[2]),
                mechanical_omega_rad_s=float((transform @ mechanical)[4]),
                path_feedback_contact=transform @ feedback,
                path_feedforward_contact=transform @ moving,
                visual_request_rad_s=request if visual_valid else 0.,
                visual_task_valid=visual_valid,
                visual_gamma=_pad_visual_gamma(guide, float(-env[2]), cfg),
                cop_m=cop,
                repair_execution_enabled=bool(contact),
            ), online=True, deadline_s=time.monotonic() + min(0.004, hold))
            if result.qp_twist is None:
                self.last_facts = dict(getattr(guide, "last_facts", None) or {},
                    visual_policy="pad_scan_delay_kf_cop_v1",
                    allocation_diagnostics=dict(result.diagnostics))
                return None
            self.last_pad_alpha = float(result.alpha)
            self.last_path_twist = feedback + float(result.alpha) * moving
            self.last_facts = dict(getattr(guide, "last_facts", None) or {},
                cop_m=cop, cop_valid=cop is not None, cop_reason=cop_reason,
                alpha=self.last_pad_alpha, alpha_target=alpha_des,
                visual_request_rad_s=request, visual_task_valid=visual_valid,
                allocation_diagnostics=dict(result.diagnostics),
                visual_policy="pad_scan_delay_kf_cop_v1")
            return np.asarray(result.qp_twist, dtype=float).reshape(6)
        except (TypeError, ValueError, RuntimeError) as exc:
            self.last_facts = dict(getattr(guide, "last_facts", None) or {},
                visual_policy="pad_scan_delay_kf_cop_v1",
                allocation_diagnostics={"reason": type(exc).__name__, "detail": str(exc)})
            return None


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
    law_name = str(dict(payload or {}).get("law") or "").lower()
    if law_name in ("admittance_1d", "ac2d", "tafac"):
        from peirastic.realman8dof.force.comparison_laws import comparison_law_from_payload
        from peirastic.realman8dof.force.config import desired_z_n

        law = comparison_law_from_payload(dt, payload)
        return law, _desired_force(payload, desired_z_n(payload=payload)), TorqueTiltConfig(enabled=False)
    if use_fce_law(payload):
        from peirastic.realman8dof.force.config import desired_z_n

        law = FceAdmittanceLaw.from_payload(dt, payload)
        return law, _desired_force(payload, desired_z_n(payload=payload)), TorqueTiltConfig(enabled=False)
    controller, f_des, raw = _hybrid_controller(dt, payload)
    law = LegacyForceLaw(controller)
    tilt_cfg = TorqueTiltConfig.from_dict(raw)
    selection = None if _contact_qp_payload(payload) else selection_from_payload(payload)
    if selection is not None and selection[tilt_cfg.axis] >= 1.0:
        tilt_cfg = replace(tilt_cfg, enabled=False)
    if tilt_cfg.enabled:
        if str(control_frame).lower() != "tool":
            raise ValueError("torque_tilt requires tool-frame hybrid control")
        law = LegacyForceWithTilt(law, TorqueTilt(tilt_cfg))
    return law, f_des, tilt_cfg


def _contact_qp_payload(payload: dict | None) -> bool:
    return dict(payload or {}).get("contact_qp") is not None


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


def _contact_gated_reference(
    reference: MotionReferenceSource,
    payload: dict | None,
) -> tuple[MotionReferenceSource, ContactGatedReference | None]:
    """Optionally defer trajectory time until the compiled force loop confirms contact."""

    if not bool(dict(payload or {}).get("wait_for_contact", False)):
        return reference, None
    pay = dict(payload or {})
    if pay.get("scan_contact_n") is not None and not bool(pay.get("use_tff_split", False)):
        raise ValueError("independent scan force gate requires TFF")
    gate = ContactGatedReference(
        reference,
        start_force_n=pay.get("scan_contact_n"),
        start_force_s=pay.get("scan_contact_s", 0.1),
        start_blend_s=pay.get("scan_start_blend_s", 0.5),
        max_lateral_drift_m=pay.get("scan_max_lateral_drift_m", 0.03),
    )
    return gate, gate


def _bind_contact_gate(
    phase: Phase,
    gate: ContactGatedReference,
    *,
    duration_s: float | None,
    inner=None,
) -> Phase:
    """Bind a gate to the compiled controller and install its duration predicate."""

    outer = phase.outer
    controller = getattr(outer, "controller", None)
    if controller is None or not hasattr(controller, "contact_present"):
        raise ValueError(
            "wait_for_contact requires a compiled hybrid controller exposing "
            "outer.controller.contact_present"
        )
    gate.bind_contact_present(
        lambda controller=controller: bool(controller.contact_present)
    )
    # Keep the gate discoverable by the runner/telemetry without changing the
    # existing outer class hierarchy.
    outer.contact_gate = gate
    if inner is not None:
        previous_enter = phase.on_enter
        previous_exit = phase.on_exit

        planned = []

        def _plan_scan_stroke():
            child = getattr(gate, "reference", None)
            if hasattr(child, "poses"):
                poses = np.asarray(child.poses, dtype=float).reshape(-1, 6)
                y_center = 0.5 * float(poses[0, 1] + poses[1, 1])
            else:
                y_center = float(inner.kin.fk_pose(inner.q_cmd)[1])
            planned[:] = inner.plan_scan_stroke(
                y_center, stroke_amplitude_m(gate), apply=False
            )

        def _enter():
            if previous_enter is not None:
                previous_enter()
            inner.set_rail_posture_frozen(False)
            if hasattr(inner, "release_planned_stroke"):
                inner.release_planned_stroke()
            # IRD on enter for the lease, but never publish: a pinned d* let
            # J4 walk to 126° on the 005 L retract (005 L_DtP/001).
            _plan_scan_stroke()

        def _exit():
            try:
                inner.set_rail_posture_frozen(False)
                if hasattr(inner, "release_planned_stroke"):
                    inner.release_planned_stroke()
            finally:
                if previous_exit is not None:
                    previous_exit()

        def _on_start():
            # Keep homotopy live through contact, path, and the retract handoff.
            inner.set_rail_posture_frozen(False)
            if hasattr(inner, "release_planned_stroke"):
                inner.release_planned_stroke()

        phase.on_enter = _enter
        phase.on_exit = _exit
        gate.on_start = _on_start
    if duration_s is not None:
        duration = float(duration_s)

        def _arrived(_pose) -> bool:
            return bool(gate.started and gate.elapsed_s >= duration)

        phase.wait_until = _arrived
    return phase


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
    from .pad_visual import attach_pad_visual, wrap_pad_hybrid_log
    pad_visual = attach_pad_visual(payload, tilt_cfg, force_law=force_law)
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
    if pad_visual is not None:
        outer.pad_visual = pad_visual
        previous_exit = phase.on_exit

        def _exit():
            try:
                pad_visual.close(wait=False)
            finally:
                if previous_exit is not None:
                    previous_exit()

        phase.on_exit = _exit
    wrap_pad_hybrid_log(phase, payload)
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
    from .contact_qp import wrap_study_phase
    if dict(payload or {}).get("contact_qp") is not None and not use_tff_split:
        raise ValueError("contact_qp study requires the existing TFF split")
    gated_reference, gate = _contact_gated_reference(reference, payload)
    if not use_tff_split:
        controller, f_des, _raw = _hybrid_controller(dt, payload)
        spec = phase_hybrid_track(
            gated_reference,
            controller,
            desired_force=f_des,
            label=label,
            duration_s=duration_s,
        )
        phase = compile_phase(spec, ctx).phase
        if gate is not None:
            return _bind_contact_gate(phase, gate, duration_s=duration_s, inner=ctx.inner)
        return phase
    force_law, f_des, tilt_cfg = _hybrid_force_law(dt, payload, control_frame=ctx.control_frame)
    cart = compile_phase(
        phase_cartesian_track(gated_reference, label=label, duration_s=duration_s),
        ctx,
    )
    cart.outer.cfg.feedback_accel_limit_m_s2 = 1.0
    if dict(payload or {}).get("contact_qp") is not None:
        cart.outer.cfg.max_lin_vel_m_s = 0.02
    outer = HybridTffOuter(
        cart.outer,
        force_law,
        desired_force=f_des,
        selection=apply_tilt_selection(
            None if _contact_qp_payload(payload) else selection_from_payload(payload),
            tilt_cfg,
        ),
        dt=dt,
        mask_force_from_path=_mask_force_from_path(payload, True),
    )
    phase = cart.phase
    phase.outer = outer
    phase.label = label
    if gate is not None:
        _bind_contact_gate(phase, gate, duration_s=duration_s, inner=ctx.inner)
    law_name = str(dict(payload or {}).get("law") or "").lower()
    if law_name in ("admittance_1d", "ac2d", "tafac"):
        from peirastic.realman8dof.force.comparison_laws import wrap_comparison_phase
        return wrap_comparison_phase(phase, payload or {})
    return wrap_study_phase(phase,payload or {},ctx)


def wrap_admittance(reference, controller, desired_force) -> AdmittanceOuterLoop:
    return AdmittanceOuterLoop(controller, reference, desired_force=desired_force)
