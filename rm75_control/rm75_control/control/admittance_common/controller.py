"""Stable tool-frame force/motion decoupling and trajectory tracking.

This controller keeps the current setpoint-normalized proactive reference and
8-DOF integration while restoring the contact-stability structure validated in
``7dde980``: hysteretic contact release, constant free-space seek, implicit
Euler admittance, predictive force-space velocity damping, and Dimeas variable
inertia.

Tool-Z force axis:

    M(t) * v_dot + D(t) * (v - v_r) = F_des - F_ext

The force direction remains the TCP/tool Z axis supplied by the existing
RealMan TCP synchronisation path.  The proactive reference is not the complete
force/motion observer controller from Li et al. (2022).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, lfilter
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.adaptive_ke import (
    AdaptiveKeConfig,
    EnvironmentStiffnessEstimator,
)
from rm75_control.control.admittance_common.force_barrier import (
    ForceBarrierConfig,
    ForceSpaceVelocityDamper,
)
from rm75_control.control.admittance_common.pose_math import pose_error, wrap_pi
from rm75_control.control.admittance_common.proactive_force_ff import (
    ProactiveFfConfig,
    ProactiveForceIntegrator,
)


def smooth_deadband_eff(f_err: float, deadband_n: float, width_n: float) -> float:
    """Apply a C1 deadband to the force error."""
    if width_n <= 0.0:
        if abs(f_err) <= deadband_n:
            return 0.0
        return f_err - math.copysign(deadband_n, f_err)
    af = abs(f_err)
    if af <= deadband_n:
        return 0.0
    if af >= deadband_n + width_n:
        return f_err - math.copysign(deadband_n + 0.5 * width_n, f_err)
    t = (af - deadband_n) / width_n
    gain = t * t * (3.0 - 2.0 * t)
    return math.copysign(gain * (af - deadband_n), f_err)


@dataclass
class AdmittanceConfig:
    """Configuration for the single stable force/motion controller."""

    euler_order: str = "xyz"
    force_axes: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    )
    control_frame: str = "tool"
    kp_pos: np.ndarray = field(default_factory=lambda: np.zeros(6))
    track_axes: np.ndarray = field(default_factory=lambda: np.ones(6))
    system_delay_s: float = 0.015
    contact_threshold_n: float = 0.5
    contact_use_fz_only: bool = True
    contact_release_n: float = 0.25
    contact_release_ticks: int = 40
    deadband_n: float = 0.3
    deadband_width_n: float = 0.2
    max_velocity: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.2, 0.05, 0.5, 0.5, 0.5])
    )
    max_acceleration: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 0.8, 2.0, 2.0, 2.0])
    )
    max_vz_tool_m_s: float = 0.05
    open_loop: bool = False
    desired_force_ramp_s: float = 0.5
    admittance_mass_z: float = 3.0
    admittance_damping_z: float = 60.0
    damping_law: str = "trend"
    damping_base_z: float = 15.0
    damping_alpha_e: float = 1.0
    damping_beta_e_edot: float = 2.5
    damping_max_z: float = 200.0
    edot_lpf_s: float = 0.02
    seek_vz_m_s: float = 0.012
    seek_force_sat_n: float = 1.0
    proactive_ff: ProactiveFfConfig = field(default_factory=ProactiveFfConfig)
    force_barrier: ForceBarrierConfig = field(default_factory=ForceBarrierConfig)
    pos_err_deadband_m: float = 0.0
    pos_correction_max_m_s: float = 0.0
    adaptive_ke: AdaptiveKeConfig = field(default_factory=AdaptiveKeConfig)
    var_damping_enabled: bool = True
    var_damping_omega_c_hz: float = 3.5
    var_damping_lambda: float = 0.951
    var_damping_f_max_n: float = 7.0
    var_damping_d_u: float = 0.0
    var_damping_m_u: float = 4.0
    var_damping_m_max: float = 5.0
    var_damping_dc_alpha: float = 0.02
    # Iₛ below this is treated as 0 (noise). 0 disables the floor.
    var_damping_is_floor: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict) -> AdmittanceConfig:
        c = raw.get("hybrid_motion", raw.get("controller", raw))
        frames = raw.get("frames", {})
        traj = raw.get("trajectory_demo", raw.get("trajectory", {}))
        force_axes = np.asarray(
            c.get("force_axes", [0, 0, 1, 0, 0, 0]),
            dtype=float,
        )
        open_loop = bool(
            c.get(
                "open_loop",
                c.get("open_loop_scan", traj.get("open_loop", False)),
            )
        )
        return cls(
            euler_order=str(frames.get("euler_order", "xyz")),
            control_frame=str(
                frames.get("control_frame", c.get("control_frame", "tool"))
            ),
            force_axes=force_axes,
            kp_pos=np.asarray(
                c.get("kp_pos", [0, 0, 0, 0, 0, 0]),
                dtype=float,
            ),
            track_axes=np.asarray(
                c.get("track_axes", [1, 1, 1, 1, 1, 1]),
                dtype=float,
            ),
            system_delay_s=float(c.get("system_delay_s", 0.015)),
            contact_threshold_n=float(c.get("contact_threshold_n", 0.5)),
            contact_use_fz_only=bool(c.get("contact_use_fz_only", True)),
            contact_release_n=float(c.get("contact_release_n", 0.25)),
            contact_release_ticks=int(c.get("contact_release_ticks", 40)),
            deadband_n=float(c.get("deadband_n", 0.3)),
            deadband_width_n=float(c.get("deadband_width_n", 0.2)),
            max_velocity=np.asarray(
                c.get("max_velocity", [0.2, 0.2, 0.10, 0.5, 0.5, 0.5]),
                dtype=float,
            ),
            max_acceleration=np.asarray(
                c.get("max_acceleration", [1.0, 1.0, 0.8, 2.0, 2.0, 2.0]),
                dtype=float,
            ),
            max_vz_tool_m_s=float(c.get("max_vz_tool_m_s", 0.05)),
            open_loop=open_loop,
            desired_force_ramp_s=float(c.get("desired_force_ramp_s", 0.5)),
            admittance_mass_z=float(c.get("admittance_mass_z", 3.0)),
            admittance_damping_z=float(c.get("admittance_damping_z", 60.0)),
            damping_law=str(c.get("damping_law", "trend")).lower(),
            damping_base_z=float(
                c.get("damping_base_z", c.get("admittance_damping_z", 15.0))
            ),
            damping_alpha_e=float(c.get("damping_alpha_e", 1.0)),
            damping_beta_e_edot=float(c.get("damping_beta_e_edot", 2.5)),
            damping_max_z=float(c.get("damping_max_z", 200.0)),
            edot_lpf_s=float(c.get("edot_lpf_s", 0.02)),
            seek_vz_m_s=float(c.get("seek_vz_m_s", 0.012)),
            seek_force_sat_n=float(c.get("seek_force_sat_n", 1.0)),
            proactive_ff=ProactiveFfConfig.from_dict(c),
            force_barrier=ForceBarrierConfig.from_dict(c),
            pos_err_deadband_m=float(c.get("pos_err_deadband_m", 0.0)),
            pos_correction_max_m_s=float(
                c.get("pos_correction_max_m_s", 0.0)
            ),
            adaptive_ke=AdaptiveKeConfig.from_dict(raw, c),
            var_damping_enabled=bool(c.get("var_damping_enabled", True)),
            var_damping_omega_c_hz=float(
                c.get("var_damping_omega_c_hz", 3.5)
            ),
            var_damping_lambda=float(c.get("var_damping_lambda", 0.951)),
            var_damping_f_max_n=float(c.get("var_damping_f_max_n", 7.0)),
            var_damping_d_u=float(c.get("var_damping_d_u", 0.0)),
            var_damping_m_u=float(c.get("var_damping_m_u", 4.0)),
            var_damping_m_max=float(c.get("var_damping_m_max", 5.0)),
            var_damping_dc_alpha=float(
                c.get("var_damping_dc_alpha", 0.02)
            ),
            var_damping_is_floor=float(
                c.get("var_damping_is_floor", 0.0)
            ),
        )


class AdmittanceController:
    """Tool-frame hybrid controller with TCP-Z force admittance."""

    def __init__(
        self,
        dt: float,
        config: AdmittanceConfig | None = None,
    ) -> None:
        self.dt = dt
        self.cfg = config or AdmittanceConfig()
        # A fixed identifier is retained in CSV logs; it is not a mode switch.
        self.controller_mode = "legacy_symmetric"
        self.last_v_cmd = np.zeros(6)
        self._in_contact_latched = False
        self._release_count = 0
        self.contact_present = False
        self.time_scale = 1.0
        self.v_force_z = 0.0
        self.v_r_z = 0.0
        self._proactive_ff = ProactiveForceIntegrator(self.cfg.proactive_ff)
        self._force_barrier = ForceSpaceVelocityDamper(self.cfg.force_barrier)
        self.force_reference_scale_n = float("nan")
        self.force_reference_drive = 0.0
        self.force_reference_gate_scale = 1.0
        self.force_reference_accel_m_s2 = 0.0
        self.force_reference_reversal_reset = False
        self._contact_time_s = 0.0
        self._d_z_smooth = float(self.cfg.damping_base_z)
        self.f_des_z_eff = 0.0
        self._ke_estimator = EnvironmentStiffnessEstimator(
            self.cfg.adaptive_ke,
            dt=dt,
            mass_z=self.cfg.admittance_mass_z,
        )
        self.ke_est = float(self.cfg.adaptive_ke.ke_initial)
        self.adaptive_bd = float(self.cfg.admittance_damping_z)
        self.zeta_eff = float(self.cfg.adaptive_ke.zeta)
        self.damping_z_eff = float(self.cfg.damping_base_z)
        self.damping_ke_z = float(self.cfg.admittance_damping_z)
        self.damping_trend_z = float(self.cfg.damping_base_z)
        self.damping_dimeas_z = 0.0
        self.instability_index = 0.0
        self.instability_index_raw = 0.0
        self._m_z_now = float(self.cfg.admittance_mass_z)
        self.mass_z_eff = self._m_z_now
        self._f_dc = 0.0
        self._p_hi = 0.0
        self._p_ac = 0.0
        self._eff_prev = 0.0
        self._eff_dot = 0.0
        self.cap_press_z = 0.0
        self.cap_retract_z = 0.0
        self.force_pred_z = 0.0
        self.force_dot_z = 0.0
        self._init_hp_filter()

    def _init_hp_filter(self) -> None:
        fs = 1.0 / self.dt if self.dt > 0 else 100.0
        wn = min(
            max(self.cfg.var_damping_omega_c_hz / (0.5 * fs), 1e-3),
            0.99,
        )
        b, a = butter(2, wn, btype="high")
        self._hp_b = np.asarray(b, dtype=np.float64)
        self._hp_a = np.asarray(a, dtype=np.float64)
        self._hp_zi = np.zeros(
            max(len(self._hp_a), len(self._hp_b)) - 1,
            dtype=np.float64,
        )
        self._is_energy_alpha = (
            float(min(1.0, self.dt / 0.2)) if self.dt > 0 else 0.05
        )

    def set_time_scale(self, scale: float) -> None:
        self.time_scale = float(np.clip(scale, 0.0, 1.0))

    def reset(self, *, clear_velocity: bool = False) -> None:
        self._in_contact_latched = False
        self._release_count = 0
        self.contact_present = False
        self.v_force_z = 0.0
        self.v_r_z = 0.0
        self._proactive_ff.reset()
        self._force_barrier.reset()
        self.force_reference_scale_n = float("nan")
        self.force_reference_drive = 0.0
        self.force_reference_gate_scale = 1.0
        self.force_reference_accel_m_s2 = 0.0
        self.force_reference_reversal_reset = False
        self._contact_time_s = 0.0
        self._d_z_smooth = float(self.cfg.damping_base_z)
        self.f_des_z_eff = 0.0
        self.damping_z_eff = float(self.cfg.damping_base_z)
        self.damping_ke_z = float(self.cfg.admittance_damping_z)
        self.damping_trend_z = float(self.cfg.damping_base_z)
        self.damping_dimeas_z = 0.0
        self.instability_index = 0.0
        self.instability_index_raw = 0.0
        self._m_z_now = float(self.cfg.admittance_mass_z)
        self.mass_z_eff = self._m_z_now
        self._f_dc = 0.0
        self._p_hi = 0.0
        self._p_ac = 0.0
        self._eff_prev = 0.0
        self._eff_dot = 0.0
        self.cap_press_z = 0.0
        self.cap_retract_z = 0.0
        self.force_pred_z = 0.0
        self.force_dot_z = 0.0
        self._hp_zi.fill(0.0)
        self._ke_estimator.reset()
        self.ke_est = self._ke_estimator.ke_est
        self.adaptive_bd = self._ke_estimator.bd
        self.zeta_eff = self._ke_estimator.zeta_eff
        if clear_velocity:
            self.last_v_cmd.fill(0.0)

    def _v_z_cap(self) -> float:
        cap = float(self.cfg.max_vz_tool_m_s)
        max_velocity_z = (
            float(self.cfg.max_velocity[2])
            if self.cfg.max_velocity.size >= 3
            else cap
        )
        if max_velocity_z > 0.0:
            cap = min(cap, max_velocity_z)
        return max(cap, 0.0)

    def _contact_signal_n(self, f_ext: np.ndarray) -> float:
        force = np.asarray(f_ext[:3], dtype=float)
        if self.cfg.contact_use_fz_only:
            return abs(float(force[2]))
        return float(np.linalg.norm(force))

    def _update_contact_latched(self, f_ext: np.ndarray) -> bool:
        signal = self._contact_signal_n(f_ext)
        enter_n = float(self.cfg.contact_threshold_n)
        release_n = min(float(self.cfg.contact_release_n), enter_n)
        release_ticks = max(int(self.cfg.contact_release_ticks), 1)

        if not self._in_contact_latched:
            self._release_count = 0
            if signal >= enter_n:
                self._in_contact_latched = True
                self._contact_time_s = 0.0
                self._proactive_ff.reset()
            return self._in_contact_latched

        if signal < release_n:
            self._release_count += 1
            if self._release_count >= release_ticks:
                self._in_contact_latched = False
                self._release_count = 0
                self._contact_time_s = 0.0
                self._proactive_ff.reset()
        else:
            self._release_count = 0
        return self._in_contact_latched

    def _update_proactive_v_r(
        self,
        eff: float,
        in_contact: bool,
        dt_eff: float,
        *,
        rising_edge: bool,
        desired_force_n: float = 0.0,
        force_velocity_cap: float | None = None,
    ) -> float:
        # Clear either sign on a new contact episode. Keeping a retract-only
        # residue was one source of the previous press/retract asymmetry.
        if rising_edge:
            self._proactive_ff.reset()
        self.v_r_z = self._proactive_ff.update(
            eff,
            in_contact=in_contact,
            dt_eff=dt_eff,
            instability_index=self.instability_index,
            v_force_z=self.v_force_z,
            v_z_cap=(
                self._v_z_cap()
                if force_velocity_cap is None
                else max(float(force_velocity_cap), 0.0)
            ),
            desired_force_n=desired_force_n,
        )
        self.force_reference_scale_n = float(
            self._proactive_ff.last_force_scale_n
        )
        self.force_reference_drive = float(self._proactive_ff.last_drive)
        self.force_reference_gate_scale = float(
            self._proactive_ff.last_instability_scale
        )
        self.force_reference_accel_m_s2 = float(
            self._proactive_ff.last_reference_accel_m_s2
        )
        self.force_reference_reversal_reset = bool(
            self._proactive_ff.last_reversal_reset
        )
        return self.v_r_z

    @staticmethod
    def fuse_tool_sleeve(
        v_pos_base: np.ndarray,
        v_force_tool: np.ndarray,
        r_mat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        v_pos_tool = np.zeros(6, dtype=float)
        v_pos_tool[:3] = r_mat.T @ np.asarray(v_pos_base[:3], dtype=float)
        v_pos_tool[3:6] = r_mat.T @ np.asarray(v_pos_base[3:6], dtype=float)
        v_cmd_tool = v_pos_tool.copy()
        v_cmd_tool[2] = float(v_force_tool[2])
        v_cmd_base = np.zeros(6, dtype=float)
        v_cmd_base[:3] = r_mat @ v_cmd_tool[:3]
        v_cmd_base[3:] = r_mat @ v_cmd_tool[3:6]
        return v_cmd_tool, v_cmd_base

    def compute_velocity_command(
        self,
        current_pose: np.ndarray,
        desired_pose: np.ndarray,
        desired_vel_ff: np.ndarray,
        f_ext: np.ndarray,
        desired_force: np.ndarray,
        *,
        in_contact: bool | None = None,
        enable_pbac: bool | None = None,
        f_ext_raw: np.ndarray | None = None,
        dt_actual: float | None = None,
        v_tcp_z_actual: float | None = None,
        sensor_age_s: float | None = None,
    ) -> np.ndarray:
        # dt_actual, v_tcp_z_actual and sensor_age_s remain accepted for
        # telemetry/API compatibility. The stable 2965fea loop uses fixed dt.
        del dt_actual, v_tcp_z_actual, sensor_age_s
        cfg = self.cfg
        r_mat = Rsc.from_euler(
            cfg.euler_order,
            current_pose[3:6],
            degrees=False,
        ).as_matrix()

        pose_predicted = np.asarray(current_pose, dtype=float).copy()
        if cfg.system_delay_s > 0.0:
            if cfg.control_frame == "tool":
                pose_predicted[:3] += (
                    r_mat @ self.last_v_cmd[:3] * cfg.system_delay_s
                )
            else:
                pose_predicted[:3] += (
                    self.last_v_cmd[:3] * cfg.system_delay_s
                )

        err_pose = pose_error(
            desired_pose,
            pose_predicted,
            cfg.euler_order,
        )
        vel_ff = np.asarray(desired_vel_ff, dtype=float).copy()
        use_pbac = (
            (not cfg.open_loop)
            if enable_pbac is None
            else bool(enable_pbac)
        )
        if not use_pbac:
            err_pose[:] = 0.0

        err_tool = r_mat.T @ err_pose[:3]
        err_tool[2] = 0.0
        if cfg.pos_err_deadband_m > 0.0:
            for index in (0, 1):
                if abs(err_tool[index]) <= cfg.pos_err_deadband_m:
                    err_tool[index] = 0.0
        kp_xy = np.array(
            [
                cfg.kp_pos[0] * cfg.track_axes[0],
                cfg.kp_pos[1] * cfg.track_axes[1],
                0.0,
            ]
        )
        v_corr_tool = kp_xy * err_tool
        if cfg.pos_correction_max_m_s > 0.0:
            v_corr_tool[:2] = np.clip(
                v_corr_tool[:2],
                -cfg.pos_correction_max_m_s,
                cfg.pos_correction_max_m_s,
            )
        v_corr = np.zeros(6, dtype=float)
        v_corr[:3] = r_mat @ v_corr_tool
        err_rot_tool = r_mat.T @ err_pose[3:6]
        kp_rot = cfg.kp_pos[3:6] * cfg.track_axes[3:6]
        v_corr[3:6] = r_mat @ (kp_rot * err_rot_tool)
        v_pos_base = vel_ff + v_corr

        f_ext = np.asarray(f_ext, dtype=float)
        f_des = np.asarray(desired_force, dtype=float)
        f_ext_z = float(f_ext[2])
        was_latched = self._in_contact_latched
        if in_contact is None:
            in_contact = self._update_contact_latched(f_ext)
        else:
            in_contact = bool(in_contact)
            if in_contact and not was_latched:
                self._contact_time_s = 0.0
                self._proactive_ff.reset()
            if not in_contact:
                self._release_count = 0
            self._in_contact_latched = in_contact
        self.contact_present = bool(in_contact)

        dt_eff = self.dt * self.time_scale
        if in_contact:
            self._contact_time_s += dt_eff
        rising_edge = bool(in_contact) and not was_latched

        raw_z = (
            float(f_ext_raw[2])
            if f_ext_raw is not None
            else f_ext_z
        )
        self._update_instability_index(raw_z)
        # The force loop itself uses the causal low-pass signal, while the
        # predictive barrier watches the raw compensated normal force.  Using
        # the delayed loop signal here defeats the 30 ms reaction horizon.
        self.force_dot_z = self._force_barrier.update_fdot(raw_z, dt_eff)

        mass_z = (
            cfg.admittance_mass_z
            + cfg.var_damping_m_u * self.instability_index
        )
        if cfg.var_damping_m_max > 0.0:
            mass_z = min(mass_z, cfg.var_damping_m_max)
        self._m_z_now = max(mass_z, 1e-3)
        self.mass_z_eff = self._m_z_now

        f_des_z = self._effective_desired_z(float(f_des[2]))
        f_err_z = f_des_z - f_ext_z
        v_lateral_m_s = float(
            np.linalg.norm((r_mat.T @ v_pos_base[:3])[:2])
        )
        if cfg.adaptive_ke.enabled:
            self.ke_est, self.adaptive_bd = self._ke_estimator.update(
                f_ext_z,
                current_pose,
                in_contact=bool(in_contact),
                mass_z=self._m_z_now,
                v_force_z=self.v_force_z,
                v_lateral_m_s=v_lateral_m_s,
                f_err_z=f_err_z,
                f_des_z=f_des_z,
                instability_index=self.instability_index_raw,
                euler_order=cfg.euler_order,
                allow_impact_init=rising_edge,
            )
            self.zeta_eff = self._ke_estimator.zeta_eff

        v_force_tool = np.zeros(6, dtype=float)
        v_force_tool[2] = self._admittance_z(
            f_err_z,
            bool(in_contact),
            dt_eff=dt_eff,
            rising_edge=rising_edge,
            desired_force_n=f_des_z,
            f_ext_z=f_ext_z,
            f_barrier_z=raw_z,
        )
        v_cmd_tool, v_cmd_base = self.fuse_tool_sleeve(
            v_pos_base,
            v_force_tool,
            r_mat,
        )
        v_z_cap = self._v_z_cap()
        if v_z_cap > 0.0:
            v_cmd_tool[2] = float(
                np.clip(v_cmd_tool[2], -v_z_cap, v_z_cap)
            )
            if cfg.control_frame == "base":
                v_cmd_base[:3] = r_mat @ v_cmd_tool[:3]
                v_cmd_base[3:] = r_mat @ v_cmd_tool[3:6]

        v_out = (
            v_cmd_tool
            if cfg.control_frame == "tool"
            else v_cmd_base
        )
        v_clamp = np.clip(v_out, -cfg.max_velocity, cfg.max_velocity)
        dv_max = cfg.max_acceleration * self.dt
        v_final = np.asarray(v_clamp, dtype=float).copy()
        for index in range(6):
            if cfg.force_axes[index] > 0.5:
                continue
            v_final[index] = float(
                np.clip(
                    v_final[index],
                    self.last_v_cmd[index] - dv_max[index],
                    self.last_v_cmd[index] + dv_max[index],
                )
            )
        self.last_v_cmd = v_final.copy()
        return v_final

    def _effective_desired_z(self, f_des_z: float) -> float:
        cfg = self.cfg
        if cfg.desired_force_ramp_s > 1e-6 and f_des_z > 0.0:
            ramp = float(
                np.clip(
                    self._contact_time_s / cfg.desired_force_ramp_s,
                    0.0,
                    1.0,
                )
            )
            f_start = min(
                f_des_z,
                max(
                    cfg.contact_threshold_n
                    + cfg.deadband_n
                    + cfg.deadband_width_n
                    + 0.2,
                    0.35 * f_des_z,
                ),
            )
            f_eff = f_start + (f_des_z - f_start) * ramp
        else:
            f_eff = f_des_z
        self.f_des_z_eff = float(f_eff)
        return float(f_eff)

    def _update_instability_index(self, f_z: float) -> None:
        cfg = self.cfg
        if not cfg.var_damping_enabled:
            self.instability_index = 0.0
            self.instability_index_raw = 0.0
            return
        filtered, self._hp_zi = lfilter(
            self._hp_b,
            self._hp_a,
            np.asarray([f_z], dtype=np.float64),
            zi=self._hp_zi,
        )
        high_pass = float(filtered[0])
        self._f_dc += cfg.var_damping_dc_alpha * (f_z - self._f_dc)
        f_ac = f_z - self._f_dc
        alpha = self._is_energy_alpha
        self._p_hi += alpha * (
            high_pass * high_pass - self._p_hi
        )
        self._p_ac += alpha * (f_ac * f_ac - self._p_ac)
        i_omega = min(
            max(self._p_hi / (self._p_ac + 1e-6), 0.0),
            1.0,
        )
        i_rms = min(
            math.sqrt(max(self._p_ac, 0.0))
            / max(cfg.var_damping_f_max_n, 1e-6),
            1.0,
        )
        self.instability_index_raw = (
            i_omega * i_rms
            + cfg.var_damping_lambda * self.instability_index_raw
        )
        # Keep the Dimeas accumulator continuous.  Only its control activation
        # is dead-zoned, with a smooth transition from floor to 2*floor.
        floor = float(cfg.var_damping_is_floor)
        if floor <= 0.0:
            self.instability_index = self.instability_index_raw
        elif self.instability_index_raw <= floor:
            self.instability_index = 0.0
        elif self.instability_index_raw >= 2.0 * floor:
            self.instability_index = self.instability_index_raw
        else:
            u = (self.instability_index_raw - floor) / floor
            smooth = u * u * (3.0 - 2.0 * u)
            self.instability_index = self.instability_index_raw * smooth

    def _trend_damping(
        self,
        eff: float,
        dt_eff: float,
        *,
        desired_force_n: float = 0.0,
    ) -> float:
        cfg = self.cfg
        if dt_eff > 0.0:
            raw_dot = (eff - self._eff_prev) / dt_eff
            tau = max(float(cfg.edot_lpf_s), 1e-6)
            alpha = min(1.0, dt_eff / tau)
            self._eff_dot += alpha * (raw_dot - self._eff_dot)
        self._eff_prev = float(eff)
        damping = float(cfg.damping_base_z)
        if abs(float(desired_force_n)) > 1e-6:
            damping += float(cfg.damping_alpha_e) * abs(eff)
            damping += float(cfg.damping_beta_e_edot) * max(
                0.0,
                eff * self._eff_dot,
            )
        if cfg.damping_max_z > 0.0:
            damping = min(damping, float(cfg.damping_max_z))
        return max(damping, 0.0)

    def _admittance_z(
        self,
        f_err: float,
        in_contact: bool,
        *,
        dt_eff: float,
        rising_edge: bool,
        desired_force_n: float = 0.0,
        f_ext_z: float = 0.0,
        f_barrier_z: float | None = None,
    ) -> float:
        cfg = self.cfg
        eff = smooth_deadband_eff(
            f_err,
            cfg.deadband_n,
            cfg.deadband_width_n,
        )
        if not in_contact and cfg.seek_force_sat_n > 0.0:
            eff = float(
                np.clip(
                    eff,
                    -float(cfg.seek_force_sat_n),
                    float(cfg.seek_force_sat_n),
                )
            )
        mass_z = max(float(self._m_z_now), 1e-3)
        if cfg.adaptive_ke.enabled and in_contact:
            damping_ke = float(self.adaptive_bd)
        else:
            damping_ke = float(cfg.admittance_damping_z)
        damping_dimeas = (
            cfg.var_damping_d_u * self.instability_index
            if cfg.var_damping_enabled
            else 0.0
        )
        if cfg.damping_law == "ke_critical":
            damping_target = damping_ke + damping_dimeas
            if cfg.adaptive_ke.bd_max > 0.0:
                damping_target = min(
                    damping_target,
                    float(cfg.adaptive_ke.bd_max),
                )
            self.damping_trend_z = float(cfg.damping_base_z)
        else:
            damping_trend = self._trend_damping(
                eff,
                dt_eff,
                desired_force_n=desired_force_n,
            )
            self.damping_trend_z = damping_trend
            # Trend damping supplies the light hand-guidance baseline, while
            # the identified critical damping remains a lower bound on a hard
            # contact.  Use max(), not a sum: this preserves transparency on
            # compliant surfaces and keeps Dimeas as inertia-only (d_u=0).
            damping_target = max(damping_trend, damping_ke) + damping_dimeas
            if cfg.damping_max_z > 0.0:
                damping_target = min(
                    damping_target,
                    float(cfg.damping_max_z),
                )
        if dt_eff > 0.0:
            # Keep the light 7dde980 trend envelope: the trend term itself is
            # capped at 200 N s/m, while the 100 ms smoothing avoids making a
            # fast hand push feel like a permanently high-damping mode.
            tau_d = 0.10
            blend = min(1.0, dt_eff / tau_d)
            self._d_z_smooth += blend * (
                damping_target - self._d_z_smooth
            )
        else:
            self._d_z_smooth = damping_target
        damping = max(float(self._d_z_smooth), 1e-6)
        self.damping_ke_z = damping_ke
        self.damping_dimeas_z = damping_dimeas
        self.damping_z_eff = float(damping)

        v_z_cap = self._v_z_cap()
        cap_press, cap_retract = self._force_barrier.caps(
            f_z=(f_ext_z if f_barrier_z is None else f_barrier_z),
            f_des_z=desired_force_n,
            in_contact=in_contact,
            v_z_cap=v_z_cap,
            seek_vz_m_s=cfg.seek_vz_m_s,
            contact_enter_n=cfg.contact_threshold_n,
        )
        self.cap_press_z = float(cap_press)
        self.cap_retract_z = float(cap_retract)
        self.force_pred_z = float(self._force_barrier.f_pred_z)
        if not in_contact and abs(float(desired_force_n)) > 1e-6:
            # Seeking is a geometric contact-acquisition action, not a force
            # response.  Keep it independent of the 1/3/5 N setpoint and do
            # not let the admittance mass create a slow free-space ramp.
            seek = min(max(float(cfg.seek_vz_m_s), 0.0), v_z_cap)
            velocity = math.copysign(seek, desired_force_n)
            self.v_force_z = float(velocity)
            self.v_r_z = 0.0
            return self.v_force_z
        v_reference = self._update_proactive_v_r(
            eff,
            in_contact,
            dt_eff,
            rising_edge=rising_edge,
            desired_force_n=desired_force_n,
            force_velocity_cap=(
                self.cap_press_z if eff >= 0.0 else self.cap_retract_z
            ),
        )
        eff_limited = self._force_barrier.clamp_eff(eff, damping)
        if dt_eff > 0.0:
            velocity = (
                mass_z * self.v_force_z
                + dt_eff * (eff_limited + damping * v_reference)
            ) / (mass_z + damping * dt_eff)
        else:
            velocity = self.v_force_z
        velocity = self._force_barrier.clamp_velocity(float(velocity))
        if v_z_cap > 0.0:
            velocity = float(
                np.clip(velocity, -v_z_cap, v_z_cap)
            )
        self.v_force_z = velocity
        return velocity


HybridMotionConfig = AdmittanceConfig
HybridMotionController = AdmittanceController
