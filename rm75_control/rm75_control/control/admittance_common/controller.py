"""Tool-frame force/motion decoupling: PBAC track + admittance on force axes.

Tool-Z: M·v̇ + D·(v − v_r) = F_des − F_ext with K=0. Phase-1 bounce fixes:
fixed M, higher fixed D, contact enter/exit hysteresis, wall-clock force dt,
ForceBarrier clamp, asymmetric accel, rate-limited f_des.
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
from rm75_control.control.admittance_common.pose_math import pose_error
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
    contact_enter_n: float = 0.0
    contact_enter_ticks: int = 1
    contact_release_n: float = 0.30
    contact_release_ticks: int = 8
    contact_use_fz_only: bool = True
    deadband_n: float = 0.3
    deadband_width_n: float = 0.2
    max_velocity: np.ndarray = field(
        default_factory=lambda: np.array([0.2, 0.2, 0.05, 0.5, 0.5, 0.5])
    )
    max_acceleration: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 0.8, 2.0, 2.0, 2.0])
    )
    force_accel_press_m_s2: float = 0.8
    force_accel_retract_m_s2: float = 2.0
    max_vz_tool_m_s: float = 0.05
    seek_vz_m_s: float = 0.0
    open_loop: bool = False
    desired_force_ramp_s: float = 1.0
    desired_force_fall_n_s: float = 4.0
    admittance_mass_z: float = 3.0
    admittance_damping_z: float = 60.0
    proactive_ff: ProactiveFfConfig = field(default_factory=ProactiveFfConfig)
    force_barrier: ForceBarrierConfig = field(default_factory=ForceBarrierConfig)
    pos_err_deadband_m: float = 0.0
    pos_correction_max_m_s: float = 0.0
    adaptive_ke: AdaptiveKeConfig = field(default_factory=AdaptiveKeConfig)
    var_damping_enabled: bool = True
    var_damping_omega_c_hz: float = 3.5
    var_damping_lambda: float = 0.951
    var_damping_f_max_n: float = 7.0
    var_damping_d_u: float = 2.0
    var_damping_m_u: float = 4.0
    var_damping_m_max: float = 7.0
    var_damping_dc_alpha: float = 0.02
    force_dt_min_s: float = 0.002
    force_dt_max_s: float = 0.015
    velocity_backcalc_tau_s: float = 0.03
    time_scale_decay_tau_s: float = 0.04
    time_scale_decay_below: float = 0.20
    sensor_age_block_press_s: float = 0.025
    sensor_age_zero_normal_s: float = 0.050

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
        contact_threshold = float(c.get("contact_threshold_n", 0.5))
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
            contact_threshold_n=contact_threshold,
            contact_enter_n=float(
                c.get("contact_enter_n", contact_threshold)
            ),
            contact_enter_ticks=int(c.get("contact_enter_ticks", 1)),
            contact_release_n=float(c.get("contact_release_n", 0.30)),
            contact_release_ticks=int(c.get("contact_release_ticks", 8)),
            contact_use_fz_only=bool(c.get("contact_use_fz_only", True)),
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
            force_accel_press_m_s2=float(
                c.get("force_accel_press_m_s2", 0.8)
            ),
            force_accel_retract_m_s2=float(
                c.get("force_accel_retract_m_s2", 2.0)
            ),
            max_vz_tool_m_s=float(c.get("max_vz_tool_m_s", 0.05)),
            seek_vz_m_s=float(c.get("seek_vz_m_s", 0.0)),
            open_loop=open_loop,
            desired_force_ramp_s=float(c.get("desired_force_ramp_s", 1.0)),
            desired_force_fall_n_s=float(
                c.get("desired_force_fall_n_s", 4.0)
            ),
            admittance_mass_z=float(c.get("admittance_mass_z", 3.0)),
            admittance_damping_z=float(c.get("admittance_damping_z", 60.0)),
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
            var_damping_d_u=float(c.get("var_damping_d_u", 2.0)),
            var_damping_m_u=float(c.get("var_damping_m_u", 4.0)),
            var_damping_m_max=float(c.get("var_damping_m_max", 7.0)),
            var_damping_dc_alpha=float(
                c.get("var_damping_dc_alpha", 0.02)
            ),
            force_dt_min_s=float(c.get("force_dt_min_s", 0.002)),
            force_dt_max_s=float(c.get("force_dt_max_s", 0.015)),
            velocity_backcalc_tau_s=float(
                c.get("velocity_backcalc_tau_s", 0.03)
            ),
            time_scale_decay_tau_s=float(
                c.get("time_scale_decay_tau_s", 0.04)
            ),
            time_scale_decay_below=float(
                c.get("time_scale_decay_below", 0.20)
            ),
            sensor_age_block_press_s=float(
                c.get("sensor_age_block_press_s", 0.025)
            ),
            sensor_age_zero_normal_s=float(
                c.get("sensor_age_zero_normal_s", 0.050)
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
        self.controller_mode = "legacy_symmetric"
        self.last_v_cmd = np.zeros(6)
        self._in_contact_latched = False
        self.contact_present = False
        self._enter_count = 0
        self._exit_count = 0
        self.time_scale = 1.0
        self.v_force_z = 0.0
        self.v_r_z = 0.0
        self._proactive_ff = ProactiveForceIntegrator(self.cfg.proactive_ff)
        self._barrier = ForceSpaceVelocityDamper(self.cfg.force_barrier)
        self.force_pred_z = float("nan")
        self.force_dot_z = float("nan")
        self.cap_press_z = float("nan")
        self.cap_retract_z = float("nan")
        self.force_reference_scale_n = float("nan")
        self.force_reference_drive = 0.0
        self.force_reference_gate_scale = 1.0
        self.force_reference_accel_m_s2 = 0.0
        self.force_reference_reversal_reset = False
        self._contact_time_s = 0.0
        self._d_z_smooth = float(self.cfg.admittance_damping_z)
        self.f_des_z_eff = 0.0
        self._f_des_target = 0.0
        self._force_active_prev = False
        self._ke_estimator = EnvironmentStiffnessEstimator(
            self.cfg.adaptive_ke,
            dt=dt,
            mass_z=self.cfg.admittance_mass_z,
        )
        self.ke_est = float(self.cfg.adaptive_ke.ke_initial)
        self.adaptive_bd = float(self.cfg.admittance_damping_z)
        self.zeta_eff = float(self.cfg.adaptive_ke.zeta)
        self.damping_z_eff = float(self.cfg.admittance_damping_z)
        self.damping_ke_z = float(self.cfg.admittance_damping_z)
        self.damping_dimeas_z = 0.0
        self.instability_index = 0.0
        self._m_z_now = float(self.cfg.admittance_mass_z)
        self.mass_z_eff = self._m_z_now
        self._f_dc = 0.0
        self._p_hi = 0.0
        self._p_ac = 0.0
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
        """Governor scale for trajectory/FF; force loop uses wall-clock dt."""
        self.time_scale = float(np.clip(scale, 0.0, 1.0))

    def reset(self, *, clear_velocity: bool = False) -> None:
        self._in_contact_latched = False
        self.contact_present = False
        self._enter_count = 0
        self._exit_count = 0
        self.v_force_z = 0.0
        self.v_r_z = 0.0
        self._proactive_ff.reset()
        self._barrier.reset()
        self.force_pred_z = float("nan")
        self.force_dot_z = float("nan")
        self.cap_press_z = float("nan")
        self.cap_retract_z = float("nan")
        self.force_reference_scale_n = float("nan")
        self.force_reference_drive = 0.0
        self.force_reference_gate_scale = 1.0
        self.force_reference_accel_m_s2 = 0.0
        self.force_reference_reversal_reset = False
        self._contact_time_s = 0.0
        self._d_z_smooth = float(self.cfg.admittance_damping_z)
        self.f_des_z_eff = 0.0
        self._f_des_target = 0.0
        self._force_active_prev = False
        self.damping_z_eff = float(self.cfg.admittance_damping_z)
        self.damping_ke_z = float(self.cfg.admittance_damping_z)
        self.damping_dimeas_z = 0.0
        self.instability_index = 0.0
        self._m_z_now = float(self.cfg.admittance_mass_z)
        self.mass_z_eff = self._m_z_now
        self._f_dc = 0.0
        self._p_hi = 0.0
        self._p_ac = 0.0
        self._hp_zi.fill(0.0)
        self._ke_estimator.reset()
        self.ke_est = self._ke_estimator.ke_est
        self.adaptive_bd = self._ke_estimator.bd
        self.zeta_eff = self._ke_estimator.zeta_eff
        if clear_velocity:
            self.last_v_cmd.fill(0.0)

    def _v_z_cap_press(self) -> float:
        cap = float(self.cfg.max_vz_tool_m_s)
        max_velocity_z = (
            float(self.cfg.max_velocity[2])
            if self.cfg.max_velocity.size >= 3
            else cap
        )
        if max_velocity_z > 0.0:
            cap = min(cap, max_velocity_z)
        return max(cap, 0.0)

    def _v_z_cap_retract(self) -> float:
        press = self._v_z_cap_press()
        v_ref = float(self.cfg.force_barrier.v_ref_m_s)
        cap = max(press, v_ref)
        max_velocity_z = (
            float(self.cfg.max_velocity[2])
            if self.cfg.max_velocity.size >= 3
            else cap
        )
        if max_velocity_z > 0.0:
            cap = min(cap, max_velocity_z)
        return max(cap, 0.0)

    def _v_z_cap(self) -> float:
        """Symmetric helper used by tests / proactive; press-side authority."""
        return self._v_z_cap_press()

    def _compression_force_n(self, f_ext: np.ndarray) -> float:
        """Signed compression along tool +z (positive = pressing into surface)."""
        force = np.asarray(f_ext[:3], dtype=float)
        if self.cfg.contact_use_fz_only:
            return float(force[2])
        return float(np.linalg.norm(force))

    def _contact_signal_n(self, f_ext: np.ndarray) -> float:
        """Legacy magnitude signal (enter uses signed compression now)."""
        return abs(self._compression_force_n(f_ext))

    def _contact_enter_n(self) -> float:
        enter = float(self.cfg.contact_enter_n)
        if enter <= 0.0:
            enter = float(self.cfg.contact_threshold_n)
        return enter

    def _on_contact_exit(self) -> None:
        self._in_contact_latched = False
        self._enter_count = 0
        self._exit_count = 0
        self.v_force_z = 0.0
        self.v_r_z = 0.0
        self._proactive_ff.reset()
        self._barrier.reset()
        self.f_des_z_eff = 0.0
        self._contact_time_s = 0.0
        self._force_active_prev = False
        self.force_pred_z = float("nan")
        self.force_dot_z = float("nan")
        self.cap_press_z = float("nan")
        self.cap_retract_z = float("nan")

    def _update_contact_latched(self, f_ext: np.ndarray) -> bool:
        f_comp = self._compression_force_n(f_ext)
        enter_n = self._contact_enter_n()
        exit_n = float(self.cfg.contact_release_n)
        enter_ticks = max(int(self.cfg.contact_enter_ticks), 1)
        exit_ticks = max(int(self.cfg.contact_release_ticks), 1)

        if self._in_contact_latched:
            if f_comp < exit_n:
                self._exit_count += 1
                if self._exit_count >= exit_ticks:
                    self._on_contact_exit()
                    return False
            else:
                self._exit_count = 0
            return True

        if f_comp >= enter_n:
            self._enter_count += 1
            if self._enter_count >= enter_ticks:
                self._in_contact_latched = True
                self._enter_count = 0
                self._exit_count = 0
                return True
        else:
            self._enter_count = 0
        return False

    def _update_proactive_v_r(
        self,
        eff: float,
        in_contact: bool,
        dt_eff: float,
        *,
        rising_edge: bool,
        desired_force_n: float = 0.0,
    ) -> float:
        if rising_edge:
            self._proactive_ff.reset()
        self.v_r_z = self._proactive_ff.update(
            eff,
            in_contact=in_contact,
            dt_eff=dt_eff,
            instability_index=self.instability_index,
            v_force_z=self.v_force_z,
            v_z_cap=self._v_z_cap_press(),
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

    def _force_dt(self, dt_actual: float | None) -> float:
        dt = float(self.dt if dt_actual is None else dt_actual)
        return float(
            np.clip(
                dt,
                float(self.cfg.force_dt_min_s),
                float(self.cfg.force_dt_max_s),
            )
        )

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
        cfg = self.cfg
        dt_force = self._force_dt(dt_actual)
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
        # Trajectory / FF follow governor clock; force loop does not.
        vel_ff = vel_ff * float(self.time_scale)
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
                self._in_contact_latched = True
                self._enter_count = 0
                self._exit_count = 0
            elif (not in_contact) and was_latched:
                self._on_contact_exit()
            else:
                self._in_contact_latched = in_contact
        self.contact_present = bool(in_contact)
        # Run force admittance once compression crosses enter even before latch
        # completes, so seek cannot keep driving into a hard surface.
        f_comp = self._compression_force_n(f_ext)
        force_active = bool(in_contact) or (
            f_comp >= self._contact_enter_n()
        )
        was_force_active = bool(getattr(self, "_force_active_prev", False))

        if in_contact:
            self._contact_time_s += dt_force
        rising_edge = bool(in_contact) and not was_latched
        if force_active and not was_force_active:
            # Seed setpoint from measured compression to avoid retract spike.
            self.f_des_z_eff = float(
                np.clip(f_ext_z, 0.0, max(float(f_des[2]), 0.0))
            )
        self._force_active_prev = bool(force_active)

        if (
            v_tcp_z_actual is not None
            and math.isfinite(float(v_tcp_z_actual))
            and cfg.velocity_backcalc_tau_s > 1e-9
        ):
            tau = float(cfg.velocity_backcalc_tau_s)
            alpha = dt_force / (tau + dt_force)
            self.v_force_z = (1.0 - alpha) * self.v_force_z + alpha * float(
                v_tcp_z_actual
            )

        if (
            self.time_scale < float(cfg.time_scale_decay_below)
            and cfg.time_scale_decay_tau_s > 1e-9
        ):
            decay = math.exp(
                -dt_force / max(float(cfg.time_scale_decay_tau_s), 1e-6)
            )
            self.v_force_z *= decay

        raw_z = (
            float(f_ext_raw[2])
            if f_ext_raw is not None
            else f_ext_z
        )
        self._update_instability_index(raw_z)

        mass_z = (
            cfg.admittance_mass_z
            + cfg.var_damping_m_u * self.instability_index
        )
        if cfg.var_damping_m_max > 0.0:
            mass_z = min(mass_z, cfg.var_damping_m_max)
        self._m_z_now = max(mass_z, 1e-3)
        self.mass_z_eff = self._m_z_now

        self._f_des_target = float(f_des[2])
        f_des_z = self._effective_desired_z(
            self._f_des_target,
            in_contact=bool(force_active),
            dt_force=dt_force,
        )
        f_err_z = f_des_z - f_ext_z
        v_lateral_m_s = float(
            np.linalg.norm((r_mat.T @ v_pos_base[:3])[:2])
        )
        if cfg.adaptive_ke.enabled:
            self.ke_est, self.adaptive_bd = self._ke_estimator.update(
                f_ext_z,
                current_pose,
                in_contact=bool(force_active),
                mass_z=self._m_z_now,
                v_force_z=self.v_force_z,
                v_lateral_m_s=v_lateral_m_s,
                f_err_z=f_err_z,
                f_des_z=f_des_z,
                instability_index=self.instability_index,
                euler_order=cfg.euler_order,
                allow_impact_init=rising_edge,
            )
            self.zeta_eff = self._ke_estimator.zeta_eff

        v_force_tool = np.zeros(6, dtype=float)
        v_force_tool[2] = self._admittance_z(
            f_err_z,
            bool(force_active),
            dt_force=dt_force,
            rising_edge=rising_edge,
            desired_force_n=(
                self._f_des_target if not force_active else f_des_z
            ),
            f_ext_z=f_ext_z,
        )

        age = (
            float(sensor_age_s)
            if sensor_age_s is not None and math.isfinite(float(sensor_age_s))
            else 0.0
        )
        if age > float(cfg.sensor_age_zero_normal_s):
            v_force_tool[2] = 0.0
            self.v_force_z = 0.0
        elif age > float(cfg.sensor_age_block_press_s):
            if v_force_tool[2] > 0.0:
                v_force_tool[2] = 0.0
            if self.v_force_z > 0.0:
                self.v_force_z = 0.0

        v_cmd_tool, v_cmd_base = self.fuse_tool_sleeve(
            v_pos_base,
            v_force_tool,
            r_mat,
        )
        # Soft symmetric envelope before accel; barrier already clamped vz.
        v_press = self._v_z_cap_press()
        v_retract = self._v_z_cap_retract()
        v_cmd_tool[2] = float(
            np.clip(v_cmd_tool[2], -v_retract, v_press)
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
        v_final = np.asarray(v_clamp, dtype=float).copy()
        for index in range(6):
            is_force_z = (
                index == 2 and float(cfg.force_axes[index]) > 0.5
            )
            if is_force_z:
                prev = float(self.last_v_cmd[2])
                cur = float(v_final[2])
                if cur >= prev:
                    a_max = float(cfg.force_accel_press_m_s2)
                else:
                    a_max = float(cfg.force_accel_retract_m_s2)
                dv = a_max * dt_force
                v_final[2] = float(np.clip(cur, prev - dv, prev + dv))
                # Accel must not defeat ForceBarrier caps.
                if math.isfinite(self.cap_press_z) and math.isfinite(
                    self.cap_retract_z
                ):
                    v_final[2] = float(
                        np.clip(
                            v_final[2],
                            -float(self.cap_retract_z),
                            float(self.cap_press_z),
                        )
                    )
                continue
            if cfg.force_axes[index] > 0.5:
                continue
            dv_max = float(cfg.max_acceleration[index]) * self.dt
            v_final[index] = float(
                np.clip(
                    v_final[index],
                    self.last_v_cmd[index] - dv_max,
                    self.last_v_cmd[index] + dv_max,
                )
            )
        # Keep integrator consistent with commanded normal after accel.
        if cfg.control_frame == "tool":
            self.v_force_z = float(v_final[2])
        self.last_v_cmd = v_final.copy()
        return v_final

    def _effective_desired_z(
        self,
        f_des_z: float,
        *,
        in_contact: bool,
        dt_force: float,
    ) -> float:
        cfg = self.cfg
        if not in_contact or f_des_z <= 0.0:
            self.f_des_z_eff = 0.0
            return 0.0

        target = float(f_des_z)
        current = float(self.f_des_z_eff)
        if cfg.desired_force_ramp_s > 1e-6:
            rise_rate = abs(target) / float(cfg.desired_force_ramp_s)
        else:
            rise_rate = float("inf")
        fall_rate = max(float(cfg.desired_force_fall_n_s), 0.0)

        if current < target:
            if math.isfinite(rise_rate):
                current = min(target, current + rise_rate * dt_force)
            else:
                current = target
        elif current > target:
            if fall_rate > 0.0:
                current = max(target, current - fall_rate * dt_force)
            else:
                current = target

        self.f_des_z_eff = float(current)
        return float(current)

    def _update_instability_index(self, f_z: float) -> None:
        cfg = self.cfg
        if not cfg.var_damping_enabled:
            self.instability_index = 0.0
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
        self.instability_index = (
            i_omega * i_rms
            + cfg.var_damping_lambda * self.instability_index
        )

    def _admittance_z(
        self,
        f_err: float,
        in_contact: bool,
        *,
        dt_force: float | None = None,
        rising_edge: bool,
        desired_force_n: float = 0.0,
        f_ext_z: float = 0.0,
        dt_eff: float | None = None,
    ) -> float:
        # Back-compat alias for older tests calling dt_eff=.
        if dt_force is None:
            dt_force = float(dt_eff) if dt_eff is not None else float(self.dt)
        else:
            dt_force = float(dt_force)
        cfg = self.cfg
        v_press = self._v_z_cap_press()
        v_retract = self._v_z_cap_retract()

        if not in_contact:
            seek = max(float(cfg.seek_vz_m_s), 0.0)
            velocity = seek if desired_force_n > 0.0 and seek > 0.0 else 0.0
            self._barrier.update_fdot(f_ext_z, dt_force)
            self._barrier.caps(
                f_z=f_ext_z,
                f_des_z=desired_force_n,
                in_contact=False,
                v_z_cap=v_press,
                seek_vz_m_s=seek,
                contact_enter_n=self._contact_enter_n(),
                v_z_cap_retract=v_retract,
            )
            velocity = self._barrier.clamp_velocity(velocity)
            self.v_force_z = velocity
            self.force_pred_z = float(self._barrier.f_pred_z)
            self.force_dot_z = float(self._barrier.f_dot_z)
            self.cap_press_z = float(self._barrier.cap_press_z)
            self.cap_retract_z = float(self._barrier.cap_retract_z)
            self.damping_ke_z = float(cfg.admittance_damping_z)
            self.damping_dimeas_z = 0.0
            self.damping_z_eff = float(cfg.admittance_damping_z)
            self.v_r_z = 0.0
            return velocity

        eff = smooth_deadband_eff(
            f_err,
            cfg.deadband_n,
            cfg.deadband_width_n,
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
        damping_target = damping_ke + damping_dimeas
        if cfg.adaptive_ke.bd_max > 0.0:
            damping_target = min(
                damping_target,
                float(cfg.adaptive_ke.bd_max),
            )
        if dt_force > 0.0:
            tau_d = 0.025 if self.instability_index > 0.5 else 0.10
            blend = min(1.0, dt_force / tau_d)
            self._d_z_smooth += blend * (
                damping_target - self._d_z_smooth
            )
        else:
            self._d_z_smooth = damping_target
        damping = self._d_z_smooth
        self.damping_ke_z = damping_ke
        self.damping_dimeas_z = damping_dimeas
        self.damping_z_eff = float(damping)

        v_reference = self._update_proactive_v_r(
            eff,
            in_contact,
            dt_force,
            rising_edge=rising_edge,
            desired_force_n=desired_force_n,
        )
        velocity = self.v_force_z + (dt_force / mass_z) * (
            eff - damping * (self.v_force_z - v_reference)
        )
        # Soft envelope before barrier (barrier may tighten further).
        velocity = float(np.clip(velocity, -v_retract, v_press))

        self._barrier.update_fdot(f_ext_z, dt_force)
        self._barrier.caps(
            f_z=f_ext_z,
            f_des_z=desired_force_n,
            in_contact=True,
            v_z_cap=v_press,
            seek_vz_m_s=float(cfg.seek_vz_m_s),
            contact_enter_n=self._contact_enter_n(),
            v_z_cap_retract=v_retract,
        )
        velocity = self._barrier.clamp_velocity(velocity)
        self.v_force_z = velocity
        self.force_pred_z = float(self._barrier.f_pred_z)
        self.force_dot_z = float(self._barrier.f_dot_z)
        self.cap_press_z = float(self._barrier.cap_press_z)
        self.cap_retract_z = float(self._barrier.cap_retract_z)
        return velocity


HybridMotionConfig = AdmittanceConfig
HybridMotionController = AdmittanceController
