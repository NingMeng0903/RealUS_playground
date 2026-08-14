"""Stable tool-frame force/motion decoupling and trajectory tracking.

Tool-Z force axis (implicit Euler):

    M0 · v̇ + (D0 + ΔD_hf) · (v − v_r) = e_f + u_DOB

* Low baseline ``D0`` preserves light feel and fast under-/over-force chase.
* Short-lived ``ΔD_hf(Iₛ)`` dissipates contact chatter without sticky steady D.
* ``u_DOB`` removes steady force offset (DOSMAC-lite) without raising D.
* Proactive ``v_r`` chases under-force; over-force retract is never Iₛ-gated.
* Recontact after flight uses a temporary press-speed cap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.adaptive_ke import (
    AdaptiveKeConfig,
    EnvironmentStiffnessEstimator,
)
from rm75_control.control.admittance_common.contact_state import (
    PhysicalContactConfig,
    PhysicalContactTracker,
)
from rm75_control.control.admittance_common.fast_retract_guard import (
    FastRetractGuard,
    FastRetractGuardConfig,
)
from rm75_control.control.admittance_common.force_barrier import (
    ForceSpaceVelocityDamper,
    ForceBarrierConfig,
)
from rm75_control.control.admittance_common.force_dob import (
    ForceDisturbanceObserver,
    ForceDobConfig,
)
from rm75_control.control.admittance_common.bidirectional_flow import (
    BidirectionalFlowConfig,
    BidirectionalFlowController,
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
class SurfaceForceModulationConfig:
    """Optional Piedra-style reduction of normal force while sliding.

    This is a velocity-interface adaptation, not a passivity mechanism.  It
    is disabled by default and only becomes eligible after physical contact
    has remained stable for ``stable_contact_s``.
    """

    enabled: bool = False
    min_force_scale: float = 0.25
    beta_per_m: float = 80.0
    stable_contact_s: float = 0.20
    attack_s: float = 0.05
    release_s: float = 0.15

    @classmethod
    def from_dict(cls, raw: dict) -> "SurfaceForceModulationConfig":
        root = raw if isinstance(raw, dict) else {}
        controller = root.get("hybrid_motion", root.get("controller", root))
        if not isinstance(controller, dict):
            controller = root
        section = controller.get(
            "surface_force_modulation",
            root.get("surface_force_modulation", {}),
        )
        if not isinstance(section, dict):
            section = {}
        return cls(
            enabled=bool(section.get("enabled", False)),
            min_force_scale=float(section.get("min_force_scale", 0.25)),
            beta_per_m=float(section.get("beta_per_m", 80.0)),
            stable_contact_s=float(section.get("stable_contact_s", 0.20)),
            attack_s=float(section.get("attack_s", 0.05)),
            release_s=float(section.get("release_s", 0.15)),
        )


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
    physical_contact: PhysicalContactConfig = field(
        default_factory=PhysicalContactConfig
    )
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
    desired_force_ramp_s: float = 1.0
    admittance_mass_z: float = 3.0
    admittance_damping_z: float = 60.0
    proactive_ff: ProactiveFfConfig = field(default_factory=ProactiveFfConfig)
    fast_retract_guard: FastRetractGuardConfig = field(
        default_factory=FastRetractGuardConfig
    )
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
    # Short-lived high-frequency dissipation (Dimeas detect, ΔD actuate).
    var_damping_hf_attack_s: float = 0.02
    var_damping_hf_hold_s: float = 0.15
    var_damping_hf_release_s: float = 0.12
    # Faster dump when |e_f| > hf_err (hand release / chase, not chatter hold).
    var_damping_hf_release_fast_s: float = 0.04
    var_damping_hf_on: float = 0.25
    var_damping_hf_off: float = 0.12
    # Only add ΔD_hf near the force setpoint so large under/over-force
    # chase is not slowed by a step-response Is spike.
    var_damping_hf_err_n: float = 0.8
    # Temporary press-speed limit after DETACHED → RECONTACT.
    recontact_vz_cap_m_s: float = 0.008
    recontact_hold_s: float = 0.20
    # Soften under-force chase / DOB when tool-XY speed is near a scan turnaround.
    force_lateral_soft_m_s: float = 0.006
    force_lateral_full_m_s: float = 0.018
    force_lateral_gain_floor: float = 1.0
    force_dob: ForceDobConfig = field(default_factory=ForceDobConfig)
    # Optional scalar proxy/real-port energy-flow adaptation.  ``off`` is
    # the safe legacy default; observe/active are opt-in and require the
    # caller to provide a verified force/velocity sign before press can be
    # modulated.
    bidirectional_flow: BidirectionalFlowConfig = field(
        default_factory=BidirectionalFlowConfig
    )
    # Predictive force-space velocity damper.  Its telemetry is populated even
    # when the flow adapter is disabled so existing loggers can consume the
    # same fields in all modes.
    force_barrier: ForceBarrierConfig = field(default_factory=ForceBarrierConfig)
    # Force-axis slew is intentionally asymmetric.  A zero value preserves
    # the historical uncapped force-axis path; positive values are applied
    # after the safety caps and before the normal-axis command is returned.
    force_axis_slew_press_m_s2: float = 0.0
    force_axis_slew_retract_m_s2: float = 0.0
    force_axis_slew_reverse_m_s2: float = 0.0
    surface_force_modulation: SurfaceForceModulationConfig = field(
        default_factory=SurfaceForceModulationConfig
    )
    # Contact episode re-arm is distinct from a physical contact reacquire.
    contact_episode_release_s: float = 0.30
    contact_episode_release_force_n: float = 0.15

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
            physical_contact=PhysicalContactConfig.from_dict(raw),
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
            desired_force_ramp_s=float(c.get("desired_force_ramp_s", 1.0)),
            admittance_mass_z=float(c.get("admittance_mass_z", 3.0)),
            admittance_damping_z=float(c.get("admittance_damping_z", 60.0)),
            proactive_ff=ProactiveFfConfig.from_dict(c),
            fast_retract_guard=FastRetractGuardConfig.from_dict(raw),
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
            var_damping_hf_attack_s=float(
                c.get("var_damping_hf_attack_s", 0.02)
            ),
            var_damping_hf_hold_s=float(
                c.get("var_damping_hf_hold_s", 0.15)
            ),
            var_damping_hf_release_s=float(
                c.get("var_damping_hf_release_s", 0.12)
            ),
            var_damping_hf_release_fast_s=float(
                c.get("var_damping_hf_release_fast_s", 0.04)
            ),
            var_damping_hf_on=float(c.get("var_damping_hf_on", 0.25)),
            var_damping_hf_off=float(c.get("var_damping_hf_off", 0.12)),
            var_damping_hf_err_n=float(
                c.get("var_damping_hf_err_n", 0.8)
            ),
            recontact_vz_cap_m_s=float(
                c.get("recontact_vz_cap_m_s", 0.008)
            ),
            recontact_hold_s=float(c.get("recontact_hold_s", 0.20)),
            force_lateral_soft_m_s=float(
                c.get("force_lateral_soft_m_s", 0.006)
            ),
            force_lateral_full_m_s=float(
                c.get("force_lateral_full_m_s", 0.018)
            ),
            force_lateral_gain_floor=float(
                c.get("force_lateral_gain_floor", 1.0)
            ),
            force_dob=ForceDobConfig.from_dict(c),
            bidirectional_flow=BidirectionalFlowConfig.from_dict(raw),
            force_barrier=ForceBarrierConfig.from_dict(raw),
            force_axis_slew_press_m_s2=float(
                c.get("force_axis_slew_press_m_s2", c.get("force_slew_press_m_s2", 0.0))
            ),
            force_axis_slew_retract_m_s2=float(
                c.get(
                    "force_axis_slew_retract_m_s2",
                    c.get("force_slew_retract_m_s2", 0.0),
                )
            ),
            force_axis_slew_reverse_m_s2=float(
                c.get(
                    "force_axis_slew_reverse_m_s2",
                    c.get("force_slew_reverse_m_s2", 0.0),
                )
            ),
            surface_force_modulation=SurfaceForceModulationConfig.from_dict(raw),
            contact_episode_release_s=float(
                c.get("contact_episode_release_s", 0.30)
            ),
            contact_episode_release_force_n=float(
                c.get("contact_episode_release_force_n", 0.15)
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
        self.last_path_twist = np.zeros(6)
        self.last_feedback_twist = np.zeros(6)
        self._in_contact_latched = False
        self.force_task_latched = False
        self.contact_present = False
        self.physical_contact_state = PhysicalContactTracker.FREE
        self.physical_contact_loss_event = False
        self.physical_contact_reacquire_event = False
        self.physical_contact_acquire_event = False
        self.physical_contact_low_timer_s = 0.0
        self.physical_contact_high_timer_s = 0.0
        self._physical_contact = PhysicalContactTracker(
            self.cfg.physical_contact
        )
        self.time_scale = 1.0
        self.v_force_z = 0.0
        self.v_r_z = 0.0
        self._proactive_ff = ProactiveForceIntegrator(self.cfg.proactive_ff)
        self.force_reference_scale_n = float("nan")
        self.force_reference_drive = 0.0
        self.force_reference_gate_scale = 1.0
        self.force_reference_accel_m_s2 = 0.0
        self.force_reference_reversal_reset = False
        self.force_reference_fast_clear = False
        self._fast_retract_guard = FastRetractGuard(
            self.cfg.fast_retract_guard
        )
        self.force_fast_z = float("nan")
        self.retract_guard_armed = False
        self.retract_fast_hold = False
        self.retract_fast_stop_count = 0
        self.retract_fast_rearm_count = 0
        self._contact_time_s = 0.0
        self._d_z_smooth = float(self.cfg.admittance_damping_z)
        self.f_des_z_eff = 0.0
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
        self._delta_d_hf = 0.0
        self._hf_hold_s = 0.0
        self._hf_active = False
        self._recontact_timer_s = 0.0
        self._force_dob = ForceDisturbanceObserver(self.cfg.force_dob)
        self.u_dob_z = 0.0
        self._force_barrier = ForceSpaceVelocityDamper(self.cfg.force_barrier)
        self.force_pred_z = 0.0
        self.force_dot_z = 0.0
        self.force_barrier_contact_active = False
        self._precontact_barrier_hold_s = 0.0
        self._precontact_peak_force_n = 0.0
        self.cap_press_z = self._v_z_cap()
        self.cap_retract_z = self._v_z_cap()
        self._bidirectional_flow = BidirectionalFlowController(
            dt,
            self.cfg.bidirectional_flow,
        )
        # Public alias retained for integration code and telemetry adapters.
        self.bidirectional_flow = self._bidirectional_flow
        self.flow_mode = self.cfg.bidirectional_flow.mode
        self.flow_alpha = 1.0
        self.flow_tank_energy = float(self.cfg.bidirectional_flow.T0)
        self.flow_fc = 0.0
        self.flow_v_track = 0.0
        self.flow_v_aux = 0.0
        self.flow_retract_through = 0.0
        self.flow_press = 0.0
        self.flow_gamma_effective = 0.0
        self.flow_feedback_stale = True
        self.flow_sign_verified = bool(self.cfg.bidirectional_flow.sign_verified)
        # A physical reacquire is telemetry only until the tool has stayed
        # detached at low raw force for the full episode-release interval.
        self._episode_detached_s = 0.0
        self._episode_rearm_armed = False
        self._episode_seen = False
        self.contact_episode_rearm_event = False
        self.contact_episode_release_s = 0.0
        self._surface_contact_s = 0.0
        self.surface_force_scale = 1.0
        self.surface_force_alpha = 0.0
        self.surface_xy_error_m = 0.0
        # Arm lateral chase softener only after real tool-XY scan motion.
        self._lat_soften_hold_s = 0.0
        self._episode_filter_seed_pending = False
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
        self.force_task_latched = False
        self.contact_present = False
        self.physical_contact_state = PhysicalContactTracker.FREE
        self.physical_contact_loss_event = False
        self.physical_contact_reacquire_event = False
        self.physical_contact_acquire_event = False
        self.physical_contact_low_timer_s = 0.0
        self.physical_contact_high_timer_s = 0.0
        self._physical_contact.reset()
        self.v_force_z = 0.0
        self.v_r_z = 0.0
        self._proactive_ff.reset()
        self.force_reference_scale_n = float("nan")
        self.force_reference_drive = 0.0
        self.force_reference_gate_scale = 1.0
        self.force_reference_accel_m_s2 = 0.0
        self.force_reference_reversal_reset = False
        self.force_reference_fast_clear = False
        self._fast_retract_guard.reset()
        self.force_fast_z = float("nan")
        self.retract_guard_armed = False
        self.retract_fast_hold = False
        self.retract_fast_stop_count = 0
        self.retract_fast_rearm_count = 0
        self._contact_time_s = 0.0
        self._d_z_smooth = float(self.cfg.admittance_damping_z)
        self.f_des_z_eff = 0.0
        self.damping_z_eff = float(self.cfg.admittance_damping_z)
        self.damping_ke_z = float(self.cfg.admittance_damping_z)
        self.damping_dimeas_z = 0.0
        self.instability_index = 0.0
        self._m_z_now = float(self.cfg.admittance_mass_z)
        self.mass_z_eff = self._m_z_now
        self._f_dc = 0.0
        self._p_hi = 0.0
        self._p_ac = 0.0
        self._delta_d_hf = 0.0
        self._hf_hold_s = 0.0
        self._hf_active = False
        self._recontact_timer_s = 0.0
        self._force_dob.reset()
        self.u_dob_z = 0.0
        self._force_barrier.reset()
        self.force_pred_z = 0.0
        self.force_dot_z = 0.0
        self.force_barrier_contact_active = False
        self._precontact_barrier_hold_s = 0.0
        self._precontact_peak_force_n = 0.0
        self.cap_press_z = self._v_z_cap()
        self.cap_retract_z = self._v_z_cap()
        self._bidirectional_flow.reset()
        self.flow_mode = self.cfg.bidirectional_flow.mode
        self.flow_alpha = 1.0
        self.flow_tank_energy = float(self.cfg.bidirectional_flow.T0)
        self.flow_fc = 0.0
        self.flow_v_track = 0.0
        self.flow_v_aux = 0.0
        self.flow_retract_through = 0.0
        self.flow_press = 0.0
        self.flow_gamma_effective = 0.0
        self.flow_feedback_stale = True
        self.flow_sign_verified = bool(self.cfg.bidirectional_flow.sign_verified)
        self._episode_detached_s = 0.0
        self._episode_rearm_armed = False
        self._episode_seen = False
        self.contact_episode_rearm_event = False
        self.contact_episode_release_s = 0.0
        self._surface_contact_s = 0.0
        self.surface_force_scale = 1.0
        self.surface_force_alpha = 0.0
        self.surface_xy_error_m = 0.0
        self._lat_soften_hold_s = 0.0
        self._hp_zi.fill(0.0)
        self._ke_estimator.reset()
        self.ke_est = self._ke_estimator.ke_est
        self.adaptive_bd = self._ke_estimator.bd
        self.zeta_eff = self._ke_estimator.zeta_eff
        if clear_velocity:
            self.last_v_cmd.fill(0.0)

    def begin_hybrid_episode(self, applied_twist: np.ndarray) -> None:
        """Start a force task continuously without resetting passivity energy."""

        seed = np.asarray(applied_twist, dtype=float).reshape(-1)
        if seed.size != 6 or not np.all(np.isfinite(seed)):
            raise ValueError("applied_twist must be a finite six-vector")
        tank = float(self._bidirectional_flow.tank_energy)
        energy_phys = float(self._bidirectional_flow.energy_phys_j)
        energy_mismatch = float(self._bidirectional_flow.energy_mismatch_j)
        # Reuse the established reset list for non-passivity episode state,
        # then restore the energy account through the dedicated flow API.
        self.reset(clear_velocity=False)
        flow_sign = 1.0 if float(self.cfg.bidirectional_flow.normal_sign) >= 0.0 else -1.0
        self._bidirectional_flow.begin_episode(
            flow_sign * float(seed[2]),
            tank_energy=tank,
            energy_phys_j=energy_phys,
            energy_mismatch_j=energy_mismatch,
        )
        self.last_v_cmd = seed.copy()
        self.v_force_z = float(seed[2])
        self.v_r_z = 0.0
        self.time_scale = 1.0
        self.flow_tank_energy = float(self._bidirectional_flow.tank_energy)
        self.flow_alpha = float(self._bidirectional_flow.alpha)
        self.flow_v_track = float(self._bidirectional_flow.v_track)
        self.flow_v_aux = 0.0
        self.flow_retract_through = float(self._bidirectional_flow.retract_through)
        self.flow_press = float(self._bidirectional_flow.press)
        self.flow_feedback_stale = True
        # The first synchronized force sample seeds the high-pass filter at
        # steady state, so a constant contact load is not interpreted as HF.
        self._episode_filter_seed_pending = True

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

    def _press_vz_cap(self) -> float:
        """Symmetric tool-Z cap, optionally tightened on press after recontact."""
        cap = self._v_z_cap()
        if (
            self._recontact_timer_s > 0.0
            and self.cfg.recontact_vz_cap_m_s > 0.0
        ):
            cap = min(cap, float(self.cfg.recontact_vz_cap_m_s))
        return max(cap, 0.0)

    def _update_delta_d_hf(
        self,
        dt_eff: float,
        *,
        abs_eff_n: float = 0.0,
    ) -> float:
        """Fast-attack / hold / fast-release ΔD from the Dimeas index."""
        cfg = self.cfg
        if not cfg.var_damping_enabled or dt_eff <= 0.0:
            self._delta_d_hf = 0.0
            self._hf_hold_s = 0.0
            self._hf_active = False
            return 0.0
        is_now = float(self.instability_index)
        ramp_s = float(cfg.desired_force_ramp_s)
        ramp_done = ramp_s <= 1e-6 or self._contact_time_s >= ramp_s
        near_setpoint = (
            ramp_done
            and abs(float(abs_eff_n)) <= float(cfg.var_damping_hf_err_n)
        )
        target = float(cfg.var_damping_d_u) * is_now
        if (
            (not self._hf_active)
            and near_setpoint
            and is_now >= float(cfg.var_damping_hf_on)
        ):
            self._hf_active = True
            self._hf_hold_s = float(cfg.var_damping_hf_hold_s)
        if self._hf_active:
            if is_now >= float(cfg.var_damping_hf_off) and near_setpoint:
                self._hf_hold_s = max(
                    self._hf_hold_s, float(cfg.var_damping_hf_hold_s)
                )
            else:
                self._hf_hold_s = max(0.0, self._hf_hold_s - dt_eff)
            if self._hf_hold_s <= 0.0 and (
                is_now < float(cfg.var_damping_hf_off) or not near_setpoint
            ):
                self._hf_active = False
                target = 0.0
            if not near_setpoint:
                # Large force error: prefer chase / escape over HF damping.
                target = 0.0
            tau = max(float(cfg.var_damping_hf_attack_s), 1e-4)
        else:
            target = 0.0
            tau = max(float(cfg.var_damping_hf_release_s), 1e-4)
            # Hand-release / large force error: dump ΔD faster than the
            # chatter-hold release so retract does not feel sticky.
            if abs(float(abs_eff_n)) > float(cfg.var_damping_hf_err_n):
                tau = min(tau, max(float(cfg.var_damping_hf_release_fast_s), 1e-4))
        blend = min(1.0, dt_eff / tau)
        self._delta_d_hf += blend * (target - self._delta_d_hf)
        if not self._hf_active and abs(self._delta_d_hf) < 1e-3:
            self._delta_d_hf = 0.0
        return float(self._delta_d_hf)

    def _lateral_chase_scale(
        self,
        v_lateral_m_s: float,
        *,
        dt_s: float = 0.0,
    ) -> float:
        """1 at full scan speed → ``force_lateral_gain_floor`` near turnaround.

        Softening arms only after sustained tool-XY motion (a real scan).  Pure
        force-hold / Z-surface tracking keeps full under-force chase.
        """
        cfg = self.cfg
        soft = max(float(cfg.force_lateral_soft_m_s), 0.0)
        full = max(float(cfg.force_lateral_full_m_s), soft + 1e-6)
        floor = float(np.clip(cfg.force_lateral_gain_floor, 0.0, 1.0))
        v_lat = float(v_lateral_m_s)
        if v_lat >= 0.5 * full:
            # Keep softener armed through short end-dwells.
            self._lat_soften_hold_s = max(self._lat_soften_hold_s, 1.0)
        if dt_s > 0.0 and self._lat_soften_hold_s > 0.0:
            self._lat_soften_hold_s = max(0.0, self._lat_soften_hold_s - dt_s)
        if self._lat_soften_hold_s <= 0.0:
            return 1.0
        u = float(np.clip((v_lat - soft) / (full - soft), 0.0, 1.0))
        blend = u * u * (3.0 - 2.0 * u)
        return float(floor + (1.0 - floor) * blend)

    def _update_proactive_v_r(
        self,
        eff: float,
        in_contact: bool,
        dt_eff: float,
        *,
        rising_edge: bool,
        desired_force_n: float = 0.0,
        retract_fast_hold: bool = False,
        chase_scale: float = 1.0,
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
            v_z_cap=self._v_z_cap(),
            desired_force_n=desired_force_n,
            retract_fast_hold=retract_fast_hold,
            chase_scale=chase_scale,
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
        self.force_reference_fast_clear = bool(
            self._proactive_ff.last_fast_retract_clear
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
        feedback_age_s: float | None = None,
        feedback_freshness: bool | float | None = None,
        feedback_fresh: bool | float | None = None,
    ) -> np.ndarray:
        # Use the measured wall-clock period for force/proxy dynamics and
        # safety timers.  Trajectory governor scaling remains a reference-path
        # concern and does not alter physical-time integration.
        if dt_actual is not None and np.isfinite(dt_actual):
            dt_contact = float(np.clip(dt_actual, 1.0e-4, 0.10))
        else:
            dt_contact = float(self.dt)
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
        if cfg.control_frame == "tool":
            path_task = np.concatenate((r_mat.T @ vel_ff[:3], r_mat.T @ vel_ff[3:]))
            feedback_task = np.concatenate(
                (r_mat.T @ v_corr[:3], r_mat.T @ v_corr[3:])
            )
        else:
            path_task = vel_ff.copy()
            feedback_task = v_corr.copy()
        # QPIK consumes these two sources independently.  Bound each source
        # before the legacy combined-command clamp so saturation is not
        # misreported as high-priority tracking feedback.
        task_limit = np.asarray(cfg.max_velocity, dtype=float)
        self.last_path_twist = np.clip(path_task, -task_limit, task_limit)
        self.last_feedback_twist = np.clip(
            feedback_task, -task_limit, task_limit
        )

        f_ext = np.asarray(f_ext, dtype=float)
        f_des = np.asarray(desired_force, dtype=float)
        f_ext_z = float(f_ext[2])
        raw_z = (
            float(f_ext_raw[2])
            if f_ext_raw is not None
            else f_ext_z
        )
        normal_sign = 1.0 if float(f_des[2]) >= 0.0 else -1.0
        if in_contact is None:
            contact_update = self._physical_contact.update(
                normal_sign * f_ext_z,
                normal_sign * raw_z,
                dt_s=dt_contact,
            )
            if contact_update.acquired:
                self._in_contact_latched = True
        else:
            physical_override = bool(in_contact)
            if physical_override:
                # The force task is enter-only.  Even an explicit physical
                # contact override cannot end it; only reset() starts a new
                # task/ramp episode.
                self._in_contact_latched = True
            contact_update = self._physical_contact.force_state(
                physical_override
            )
        force_task_active = bool(self._in_contact_latched)
        physical_contact = bool(contact_update.present)
        self.force_task_latched = force_task_active
        self.contact_present = physical_contact
        self.physical_contact_state = str(contact_update.state)
        self.physical_contact_loss_event = bool(contact_update.lost)
        self.physical_contact_reacquire_event = bool(
            contact_update.reacquired
        )
        self.physical_contact_acquire_event = bool(contact_update.acquired)
        self.physical_contact_low_timer_s = float(
            self._physical_contact.low_timer_s
        )
        self.physical_contact_high_timer_s = float(
            self._physical_contact.high_timer_s
        )

        # Force/proxy dynamics and all contact timers use wall-clock dt.  The
        # governor still scales the trajectory/reference path above, but it
        # must not silently slow the physical force state or make feedback
        # freshness time-scale dependent.
        dt_flow = dt_contact
        dt_eff = dt_flow
        if force_task_active:
            self._contact_time_s += dt_flow

        self.contact_episode_rearm_event = False
        low_raw = normal_sign * raw_z < float(cfg.contact_episode_release_force_n)
        if not physical_contact and self._episode_seen:
            if not self._episode_rearm_armed:
                if low_raw:
                    self._episode_detached_s += dt_flow
                else:
                    # A detached interval only counts when raw force remains
                    # low; this prevents a noisy trough from re-arming the
                    # episode.  Once armed, keep the latch through the
                    # contact tracker confirmation window.
                    self._episode_detached_s = 0.0
                self._episode_rearm_armed = (
                    self._episode_detached_s
                    >= max(float(cfg.contact_episode_release_s), 0.0)
                )
        elif not physical_contact:
            # Free-space startup is not a detached contact episode.  Arming
            # here made the very first acquisition look like a re-contact.
            self._episode_detached_s = 0.0
            self._episode_rearm_armed = False
        elif contact_update.acquired:
            # Physical reacquire is intentionally telemetry-only.  ``rising``
            # is reserved for first contact or an explicitly re-armed episode.
            self.contact_episode_rearm_event = bool(self._episode_rearm_armed)
            if self._episode_rearm_armed:
                self._episode_detached_s = 0.0
                self._episode_rearm_armed = False
            else:
                self._episode_detached_s = 0.0
        if physical_contact and not contact_update.acquired:
            self._episode_detached_s = 0.0

        rising_edge = bool(contact_update.acquired) and (
            (not self._episode_seen) or self.contact_episode_rearm_event
        )
        if contact_update.acquired:
            self._episode_seen = True
        self.contact_episode_release_s = float(self._episode_detached_s)
        # Physical reacquire is telemetry only.  The temporary press cap is
        # re-armed on first contact or a true episode re-arm, never on every
        # short contact trough.
        if rising_edge:
            self._recontact_timer_s = max(
                self._recontact_timer_s,
                float(cfg.recontact_hold_s),
            )
        if self._recontact_timer_s > 0.0:
            self._recontact_timer_s = max(
                0.0, self._recontact_timer_s - dt_contact
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

        f_des_z = self._effective_desired_z(float(f_des[2]))
        # Piedra-style surface modulation is an optional tracking aid only;
        # it changes the requested force smoothly after stable contact but is
        # not credited by the passivity/energy account.
        surface_scale = self._update_surface_force_scale(
            float(np.linalg.norm(err_tool[:2])),
            physical_contact=physical_contact,
            dt_s=dt_flow,
        )
        f_des_z *= surface_scale
        self.f_des_z_eff = float(f_des_z)
        # Deliberately unfiltered.  Raw fz moves 0.16 N per tick, but the
        # force-axis slew limiter already bounds the command to ~4.9 mm/s per
        # tick and the measured v_force_z step is only 2.8 mm/s p95 — the
        # noise never reaches the joints.  A low-pass here bought nothing and
        # cost twice: 12 ms of phase took the stiff-surface impact from 8 N to
        # 12.2 N, and it starved the proactive feedforward (v_r 6.97 -> 5.89
        # mm/s on a receding surface, tracking error 0.18 -> 0.28 N).
        f_err_z = f_des_z - f_ext_z
        v_lateral_m_s = float(
            np.linalg.norm((r_mat.T @ v_pos_base[:3])[:2])
        )
        chase_scale = self._lateral_chase_scale(
            v_lateral_m_s, dt_s=dt_contact
        )
        if cfg.adaptive_ke.enabled:
            self.ke_est, self.adaptive_bd = self._ke_estimator.update(
                f_ext_z,
                current_pose,
                in_contact=physical_contact,
                mass_z=self._m_z_now,
                v_force_z=self.v_force_z,
                v_lateral_m_s=v_lateral_m_s,
                f_err_z=f_err_z,
                f_des_z=f_des_z,
                instability_index=self.instability_index,
                euler_order=cfg.euler_order,
                allow_impact_init=rising_edge,
                allow_idle_decay=(
                    self.physical_contact_state
                    == PhysicalContactTracker.CONTACT
                    and normal_sign * f_ext_z
                    >= float(cfg.adaptive_ke.contact_force_n)
                ),
            )
            self.zeta_eff = self._ke_estimator.zeta_eff

        # Predictive force-space damper is the primary hard-contact impact
        # limiter.  It runs on wall time and uses the newest stiffness/mass
        # estimate.  In active BEFM mode only, the previous verified tank
        # balance may further tighten press; observe/off never alter behavior
        # through the tank.
        # The barrier always runs in a press-positive normal coordinate.  The
        # rest of the legacy force loop may use either tool-Z sign, so map at
        # this boundary and map velocity back after clamping.
        force_normal_filtered = normal_sign * f_ext_z
        force_normal_raw = normal_sign * raw_z
        force_normal_desired = abs(float(f_des_z))
        self.force_dot_z = float(
            self._force_barrier.update_fdot(force_normal_raw, dt_flow)
        )
        energy_available_j = None
        if cfg.bidirectional_flow.mode == "active":
            energy_available_j = max(
                float(self._bidirectional_flow.tank_energy)
                - float(cfg.bidirectional_flow.Tmin),
                0.0,
            )
        precontact_trigger = max(
            float(cfg.force_barrier.precontact_raw_trigger_n), 0.0
        )
        precontact_impact = (
            not physical_contact
            and precontact_trigger > 0.0
            and force_normal_raw >= precontact_trigger
        )
        if precontact_impact:
            self._precontact_barrier_hold_s = max(
                self._precontact_barrier_hold_s,
                max(float(cfg.physical_contact.enter_confirm_s), dt_flow),
            )
            self._precontact_peak_force_n = max(
                self._precontact_peak_force_n,
                force_normal_raw,
                force_normal_filtered,
            )
        elif self._precontact_barrier_hold_s > 0.0:
            self._precontact_peak_force_n = max(
                self._precontact_peak_force_n,
                force_normal_raw,
                force_normal_filtered,
            )

        # Keep the impact guard active throughout the filtered contact
        # confirmation window.  This is deliberately separate from the
        # physical/force-task latch: a raw air spike can pause press briefly,
        # but it cannot create a sticky contact episode.
        precontact_candidate = (
            not physical_contact
            and float(self._physical_contact.high_timer_s) > 0.0
        )
        precontact_guard = bool(
            not physical_contact
            and (
                precontact_impact
                or self._precontact_barrier_hold_s > 0.0
                or precontact_candidate
            )
        )
        barrier_contact = bool(physical_contact or precontact_guard)
        self.force_barrier_contact_active = barrier_contact
        if physical_contact:
            barrier_force_n = force_normal_filtered
            barrier_desired_n = force_normal_desired
            self._precontact_barrier_hold_s = 0.0
            self._precontact_peak_force_n = 0.0
        else:
            barrier_force_n = max(
                force_normal_filtered,
                force_normal_raw,
                self._precontact_peak_force_n,
            )
            # Before confirmation, treat the acquire threshold as the safe
            # force target.  Continuing toward a 2--5 N setpoint immediately
            # after the first impact defeated the purpose of this guard.
            barrier_desired_n = min(
                force_normal_desired,
                max(float(cfg.physical_contact.enter_n), 0.0),
            )
        self.cap_press_z, self.cap_retract_z = self._force_barrier.caps(
            f_z=barrier_force_n,
            f_des_z=barrier_desired_n,
            in_contact=barrier_contact,
            v_z_cap=self._v_z_cap(),
            seek_vz_m_s=self._v_z_cap(),
            contact_enter_n=float(cfg.contact_threshold_n),
            v_z_cap_retract=self._v_z_cap(),
            ke_est_n_m=float(self.ke_est),
            mass_eq_kg=float(self._m_z_now),
            energy_available_j=energy_available_j,
        )
        if precontact_guard:
            # A deterministic low-speed confirmation sleeve closes the gap
            # between the raw impact tick and the debounced filtered latch.
            confirm_cap = max(float(cfg.recontact_vz_cap_m_s), 0.0)
            if confirm_cap > 0.0:
                self.cap_press_z = min(self.cap_press_z, confirm_cap)
                self._force_barrier.cap_press_z = self.cap_press_z
            self._precontact_barrier_hold_s = max(
                0.0, self._precontact_barrier_hold_s - dt_flow
            )
            if self._precontact_barrier_hold_s <= 0.0 and not precontact_candidate:
                self._precontact_peak_force_n = 0.0
        self.force_pred_z = float(self._force_barrier.f_pred_z)

        v_force_tool = np.zeros(6, dtype=float)
        sensor_age_eff = (
            feedback_age_s if feedback_age_s is not None else sensor_age_s
        )
        v_force_tool[2] = self._admittance_z(
            f_err_z,
            force_task_active,
            dt_eff=dt_eff,
            rising_edge=rising_edge,
            desired_force_n=f_des_z,
            raw_force_z=(
                normal_sign * raw_z
                if f_ext_raw is not None
                else None
            ),
            dt_contact=dt_contact,
            sensor_age_s=sensor_age_eff,
            chase_scale=chase_scale,
        )
        # True air: command the free-space seek, not the tiny admittance
        # crawl from a 0.5 N residual / D=25 (measured vz_achieved ~6 mm/s).
        # Precontact / impact sleeve still wins via the barrier cap below.
        if (not physical_contact) and (not precontact_guard):
            seek = max(float(cfg.force_barrier.v_seek_free_m_s), 0.0)
            v_hi = self._v_z_cap()
            if seek <= 0.0:
                seek = v_hi
            elif v_hi > 0.0:
                seek = min(seek, v_hi)
            v_n = normal_sign * float(v_force_tool[2])
            if seek > 0.0 and v_n < seek:
                v_force_tool[2] = normal_sign * seek
        # Optional scalar bidirectional-flow adapter.  The adapter sees a
        # press-positive normal coordinate; ``normal_sign`` maps the tool
        # force convention into that coordinate and back.
        flow_cfg = cfg.bidirectional_flow
        flow_sign = 1.0 if float(flow_cfg.normal_sign) >= 0.0 else -1.0
        flow_feedback_age = (
            feedback_age_s if feedback_age_s is not None else sensor_age_s
        )
        flow_speed_actual = (
            None
            if v_tcp_z_actual is None
            else flow_sign * float(v_tcp_z_actual)
        )
        flow_command = self._bidirectional_flow.update(
            flow_sign * float(v_force_tool[2]),
            # current_pose[2] is base-Z and is not a normal displacement when
            # the tool is tilted.  Until the loop supplies a projected
            # normal position, let the flow core integrate xa from the fresh
            # tool-normal velocity instead of feeding it base-Z.
            x_actual=None,
            v_actual=flow_speed_actual,
            force=flow_sign * f_ext_z,
            dt_actual=dt_flow,
            feedback_age_s=flow_feedback_age,
            feedback_fresh=(
                feedback_freshness
                if feedback_freshness is not None
                else feedback_fresh
            ),
            # Reconstruct the actual uncoupled implicit proxy RHS with the
            # same total damping used by _admittance_z.  Tank credit remains
            # limited to nominal_damping below, so Dimeas/impact damping is
            # never used as fictitious energy income.
            nominal_damping=float(cfg.admittance_damping_z),
            proxy_mass=float(self._m_z_now),
            proxy_damping=float(self.damping_z_eff),
            active_effort_n=float(
                max(float(f_des_z), 0.0)
                + max(float(self.u_dob_z), 0.0)
                + max(float(self.damping_ke_z * max(self.v_r_z, 0.0)), 0.0)
            ),
        )
        if flow_cfg.mode == "active":
            # The coupled proxy, not the uncoupled legacy Euler result, is the
            # state carried into the next tick.  Without this assignment the
            # -lambda*alpha*Fc branch would be forgotten every cycle.
            self.v_force_z = flow_sign * float(self._bidirectional_flow.vp)
            v_force_tool[2] = flow_sign * flow_command
        self.flow_mode = str(flow_cfg.mode)
        self.flow_alpha = float(self._bidirectional_flow.alpha)
        self.flow_tank_energy = float(self._bidirectional_flow.tank_energy)
        self.flow_fc = float(self._bidirectional_flow.fc)
        self.flow_v_track = float(self._bidirectional_flow.v_track)
        self.flow_v_aux = float(self._bidirectional_flow.v_aux)
        self.flow_retract_through = float(
            self._bidirectional_flow.retract_through
        )
        self.flow_press = float(self._bidirectional_flow.press)
        self.flow_gamma_effective = float(
            self._bidirectional_flow.gamma_effective
        )
        self.flow_feedback_stale = bool(
            self._bidirectional_flow.feedback_stale
        )
        self.flow_sign_verified = bool(
            self._bidirectional_flow.sign_verified
        )
        v_cmd_tool, v_cmd_base = self.fuse_tool_sleeve(
            v_pos_base,
            v_force_tool,
            r_mat,
        )
        # Recontact cap only limits press (+z); over-force retract stays open.
        v_z_cap = self._v_z_cap()
        press_cap = self._press_vz_cap()
        if v_z_cap > 0.0:
            lo = -v_z_cap
            hi = press_cap if press_cap > 0.0 else v_z_cap
            v_normal = normal_sign * float(v_cmd_tool[2])
            v_normal = float(np.clip(v_normal, lo, hi))
            # Force-space barrier caps are directional: a predicted force
            # rise can close press while retract remains available.
            v_normal = self._force_barrier.clamp_velocity(
                v_normal
            )
            # Under-force must press.  Retract while chasing was the
            # "hold" feel; the barrier still brakes over-force above.
            under_force = (normal_sign * float(f_err_z)) > max(
                float(cfg.deadband_n), 0.0
            )
            if physical_contact and under_force and v_normal < 0.0:
                v_normal = 0.0
            v_cmd_tool[2] = normal_sign * v_normal
            if cfg.control_frame == "base":
                v_cmd_base[:3] = r_mat @ v_cmd_tool[:3]
                v_cmd_base[3:] = r_mat @ v_cmd_tool[3:6]

        v_out = (
            v_cmd_tool
            if cfg.control_frame == "tool"
            else v_cmd_base
        )
        v_clamp = np.clip(v_out, -cfg.max_velocity, cfg.max_velocity)
        dv_max = cfg.max_acceleration * dt_flow
        v_final = np.asarray(v_clamp, dtype=float).copy()
        for index in range(6):
            if cfg.force_axes[index] > 0.5:
                if index == 2:
                    desired_normal = normal_sign * float(v_final[index])
                    previous_normal = normal_sign * float(self.last_v_cmd[index])
                    press_slew = max(
                        float(cfg.force_axis_slew_press_m_s2), 0.0
                    )
                    retract_slew = max(
                        float(cfg.force_axis_slew_retract_m_s2), 0.0
                    )
                    reverse_slew = max(
                        float(cfg.force_axis_slew_reverse_m_s2), 0.0
                    )
                    if desired_normal >= previous_normal:
                        if press_slew > 0.0:
                            desired_normal = float(
                                min(
                                    desired_normal,
                                    previous_normal + press_slew * dt_flow,
                                )
                            )
                    else:
                        # Crossing from press to retract is a safety escape,
                        # so it has its own faster allowance.  Once already
                        # retracting, use the regular retract slew.
                        slew = (
                            reverse_slew
                            if previous_normal > 0.0
                            and desired_normal <= 0.0
                            and reverse_slew > 0.0
                            else retract_slew
                        )
                        if slew <= 0.0:
                            continue
                        desired_normal = float(
                            max(
                                desired_normal,
                                previous_normal - slew * dt_flow,
                            )
                        )
                    v_final[index] = normal_sign * desired_normal
                continue
            v_final[index] = float(
                np.clip(
                    v_final[index],
                    self.last_v_cmd[index] - dv_max[index],
                    self.last_v_cmd[index] + dv_max[index],
                )
            )
        if cfg.bidirectional_flow.mode == "active":
            requested_press = max(normal_sign * float(v_final[2]), 0.0)
            paid_press = self._bidirectional_flow.settle_applied_press(
                requested_press
            )
            if requested_press > paid_press:
                v_final[2] = normal_sign * paid_press
            self.flow_tank_energy = float(self._bidirectional_flow.tank_energy)
        self.last_v_cmd = v_final.copy()
        return v_final

    def _update_surface_force_scale(
        self,
        xy_error_m: float,
        *,
        physical_contact: bool,
        dt_s: float,
    ) -> float:
        """Return the optional elastic-surface desired-force scale.

        The normal force is reduced as tangential tracking error grows, using
        ``alpha = 1-exp(-beta*||e_xy||)``.  This layer is deliberately
        independent from BEFM/tank accounting and defaults to unity.
        """

        cfg = self.cfg.surface_force_modulation
        self.surface_xy_error_m = max(float(xy_error_m), 0.0)
        if physical_contact:
            self._surface_contact_s += max(float(dt_s), 0.0)
        else:
            self._surface_contact_s = 0.0

        eligible = (
            bool(cfg.enabled)
            and physical_contact
            and self._surface_contact_s >= max(float(cfg.stable_contact_s), 0.0)
        )
        if eligible:
            alpha_target = 1.0 - math.exp(
                -max(float(cfg.beta_per_m), 0.0) * self.surface_xy_error_m
            )
            min_scale = float(np.clip(cfg.min_force_scale, 0.0, 1.0))
            target = alpha_target * min_scale + (1.0 - alpha_target)
        else:
            alpha_target = 0.0
            target = 1.0

        target = float(np.clip(target, 0.0, 1.0))
        tau = float(cfg.attack_s if target < self.surface_force_scale else cfg.release_s)
        if tau <= 1.0e-9:
            self.surface_force_scale = target
        else:
            blend = float(np.clip(max(float(dt_s), 0.0) / tau, 0.0, 1.0))
            self.surface_force_scale += blend * (
                target - self.surface_force_scale
            )
        self.surface_force_scale = float(
            np.clip(self.surface_force_scale, 0.0, 1.0)
        )
        self.surface_force_alpha = float(np.clip(alpha_target, 0.0, 1.0))
        return self.surface_force_scale

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
            return
        if self._episode_filter_seed_pending:
            self._hp_zi = lfilter_zi(self._hp_b, self._hp_a) * float(f_z)
            self._f_dc = float(f_z)
            self._p_hi = 0.0
            self._p_ac = 0.0
            self.instability_index = 0.0
            self._episode_filter_seed_pending = False
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
        dt_eff: float,
        rising_edge: bool,
        desired_force_n: float = 0.0,
        raw_force_z: float | None = None,
        dt_contact: float | None = None,
        sensor_age_s: float | None = None,
        chase_scale: float = 1.0,
    ) -> float:
        cfg = self.cfg
        eff = smooth_deadband_eff(
            f_err,
            cfg.deadband_n,
            cfg.deadband_width_n,
        )
        mass_z = max(float(self._m_z_now), 1e-3)
        # Steady damping: D0 unless legacy drive_damping keeps Keemink b_d.
        if (
            cfg.adaptive_ke.enabled
            and cfg.adaptive_ke.drive_damping
            and in_contact
        ):
            damping_ke = float(self.adaptive_bd)
        else:
            damping_ke = float(cfg.admittance_damping_z)
        damping_dimeas = self._update_delta_d_hf(
            dt_eff, abs_eff_n=abs(float(eff))
        )
        # Impact burst: on rising edge, briefly allow critical-damping level
        # even when drive_damping is False (stiff-first without sticky steady D).
        if (
            rising_edge
            and cfg.adaptive_ke.enabled
            and not cfg.adaptive_ke.drive_damping
            and in_contact
        ):
            damping_ke = max(damping_ke, float(self.adaptive_bd))
        damping_target = damping_ke + damping_dimeas
        if cfg.adaptive_ke.bd_max > 0.0:
            damping_target = min(
                damping_target,
                float(cfg.adaptive_ke.bd_max),
            )
        if rising_edge and damping_target > self._d_z_smooth:
            self._d_z_smooth = damping_target
        elif dt_eff > 0.0:
            if damping_target >= self._d_z_smooth:
                tau_d = max(float(cfg.var_damping_hf_attack_s), 0.01)
            else:
                tau_d = max(float(cfg.var_damping_hf_release_s), 0.05)
            blend = min(1.0, dt_eff / tau_d)
            self._d_z_smooth += blend * (
                damping_target - self._d_z_smooth
            )
        else:
            self._d_z_smooth = damping_target
        damping_total = self._d_z_smooth
        # Keep the nominal/base damping attached to (v-v_r), but make every
        # extra dissipative channel zero-centred.  In particular Dimeas must
        # not multiply the proactive reference and thereby amplify a stale
        # press/retract anchor.
        damping_base = max(float(damping_ke), 0.0)
        damping_extra = max(damping_total - damping_base, 0.0)
        damping = damping_base + damping_extra
        self.damping_ke_z = damping_ke
        self.damping_dimeas_z = damping_dimeas
        self.damping_z_eff = float(damping)

        v_z_cap = self._v_z_cap()
        press_cap = self._press_vz_cap()
        retract_fast_hold = self._fast_retract_guard.update(
            raw_force_n=raw_force_z,
            desired_force_n=desired_force_n,
            filtered_eff_n=eff,
            active_reference_m_s=self.v_r_z,
            dt_s=self.dt if dt_contact is None else dt_contact,
            sensor_age_s=sensor_age_s,
            instability_index=self.instability_index,
        )
        self.force_fast_z = float(self._fast_retract_guard.fast_force_n)
        self.retract_guard_armed = bool(self._fast_retract_guard.armed)
        self.retract_fast_hold = bool(retract_fast_hold)
        self.retract_fast_stop_count = int(
            self._fast_retract_guard.stop_count
        )
        self.retract_fast_rearm_count = int(
            self._fast_retract_guard.rearm_count
        )
        v_reference = self._update_proactive_v_r(
            eff,
            in_contact,
            dt_eff,
            rising_edge=rising_edge,
            desired_force_n=desired_force_n,
            retract_fast_hold=retract_fast_hold,
            chase_scale=chase_scale,
        )
        self.u_dob_z = self._force_dob.update(
            eff,
            dt_eff=dt_eff,
            in_contact=in_contact,
            instability_index=self.instability_index,
            chase_scale=chase_scale,
        )
        drive = float(eff) + float(self.u_dob_z)
        if dt_eff <= 0.0:
            velocity = float(self.v_force_z)
        else:
            # Implicit Euler with split damping:
            # (M/dt + D0 + D_extra)v+ = M/dt*v + D0*v_r + drive.
            # D_extra is zero-centred and therefore cannot amplify v_r.
            denom = mass_z / dt_eff + max(damping, 0.0)
            velocity = (
                (mass_z / dt_eff) * self.v_force_z
                + max(damping_base, 0.0) * v_reference
                + drive
            ) / max(denom, 1e-6)
        if v_z_cap > 0.0:
            lo = -v_z_cap
            hi = press_cap if press_cap > 0.0 else v_z_cap
            velocity = float(np.clip(velocity, lo, hi))
        self.v_force_z = velocity
        return velocity


HybridMotionConfig = AdmittanceConfig
HybridMotionController = AdmittanceController
