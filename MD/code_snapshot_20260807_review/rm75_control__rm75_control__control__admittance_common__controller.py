"""Stable tool-frame force/motion decoupling and trajectory tracking.

Tool-Z force axis (implicit Euler, zero-centered extras):

    M · v̇ + D0 · (v − v_r) + D_extra · v = e_f + u_DOB

* ``D0`` sets steady feel (light scan / hand push).
* ``D_extra = D_impact (+ delay/HF)`` only multiplies ``v`` — never ``v_r`` —
  so impact damping cannot amplify active press reference (passivity).
* Bounce cut: continuous impact-danger Di + Teff press budget + SUSPECT_LOSS.
* Predictive retract stop + unified zero-centered ``D_extra`` (impact/brake).
* Overshoot-episode interlock gates press *drive* only (never hard-zeros vf).
* Press tank (4 mJ) + bidirectional port PO/PC on real Δx.
* Free-seek (air) vs contact press caps; raw-force exit → 3 mm/s.
* Direction-safe TCP anti-windup (default off for A/B).
* DOB only in confirmed CONTACT; cleared on SUSPECT_LOSS.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, lfilter
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
    ForceBarrierConfig,
    ForceSpaceVelocityDamper,
)
from rm75_control.control.admittance_common.force_dob import (
    ForceDisturbanceObserver,
    ForceDobConfig,
)
from rm75_control.control.admittance_common.pose_math import pose_error, wrap_pi
from rm75_control.control.admittance_common.press_energy_tank import (
    PortPassivityConfig,
    PortPassivityObserver,
    PressEnergyTank,
    PressEnergyTankConfig,
)
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


def soft_tanh_eff(f_err: float, eps_n: float) -> float:
    """Continuous soft zone: e − ε tanh(e/ε). No sticky exact-zero band."""
    eps = max(float(eps_n), 1e-6)
    return float(f_err) - eps * math.tanh(float(f_err) / eps)


def smoothstep01(x: float) -> float:
    t = float(np.clip(x, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


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
    # True: e−εtanh(e/ε) (no sticky zero band). False: classic deadband.
    deadband_soft_tanh: bool = False
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
    force_lateral_gain_floor: float = 0.35
    force_dob: ForceDobConfig = field(default_factory=ForceDobConfig)
    force_barrier: ForceBarrierConfig = field(default_factory=ForceBarrierConfig)
    # Delay-aware contact damping: D_delay = κ · K̂b · T_dead · contact_conf.
    delay_damping_enabled: bool = True
    delay_damping_kappa: float = 0.8
    delay_damping_bd_max: float = 600.0
    # "impact_only": apply delay damping solely as a short post-acquire burst
    # (keeps steady-contact D = D0 for light hand feel).  "always": old
    # continuous contact D_delay (heavy; caused sticky shake in 160926).
    delay_damping_mode: str = "impact_only"
    # Post-acquire / re-hit impact burst (seconds).
    impact_damping_hold_s: float = 0.12
    impact_damping_release_s: float = 0.10
    # Critical-damping impact: D0+Di = clip(2 ζ √(M Ke), d_min, d_max).
    impact_damping_zeta: float = 0.9
    impact_damping_d_min: float = 60.0
    impact_damping_d_max: float = 100.0
    impact_ke_floor: float = 800.0
    # Force-rise impact arm (do NOT require delayed v_tcp > 0).
    impact_compress_v_min_m_s: float = 0.003  # optional confidence only
    impact_fdot_arm_n_s: float = 12.0
    impact_fpred_over_n: float = 0.15
    impact_fpred_horizon_s: float = 0.050
    impact_arm_confirm_s: float = 0.010
    impact_fdot_rearm_n_s: float = 5.0
    impact_rearm_f_frac: float = 0.9
    # Zero-centered retract brake after predictive stop (total D target).
    retract_brake_damping_ns_m: float = 70.0
    retract_brake_hold_s: float = 0.060
    retract_brake_release_s: float = 0.040
    # Continuous reverse-interlock blend (gate velocity, not hard bool).
    reverse_interlock_enter_m_s: float = 0.004
    reverse_interlock_exit_m_s: float = 0.0015
    reverse_interlock_enter_confirm_s: float = 0.010
    reverse_interlock_exit_confirm_s: float = 0.010
    v_gate_window_s: float = 0.030
    # Unified D_extra smoothing (impact + brake share one target).
    d_extra_attack_s: float = 0.005
    d_extra_release_s: float = 0.120
    d_extra_min_hold_s: float = 0.040
    # Continuous impact danger → Di (renew while unsafe).
    impact_danger_f_over_n: float = 0.15
    impact_danger_fdot_n_s: float = 5.0
    impact_danger_f_over_fdot_n: float = 0.10
    impact_safe_f_over_n: float = 0.25
    impact_safe_confirm_s: float = 0.025
    impact_pred_span_n: float = 1.5
    # Force-axis asymmetric slew [m/s²] — soft contact chase vs air seek.
    force_slew_press_m_s2: float = 0.60
    force_slew_retract_m_s2: float = 1.20
    force_slew_press_to_retract_m_s2: float = 2.0
    force_slew_zero_cross_m_s2: float = 0.70
    # Free-space seek (air) vs contact press — separate budgets.
    free_seek_vz_m_s: float = 0.080
    free_seek_accel_m_s2: float = 0.80
    free_seek_exit_force_n: float = 0.15
    free_seek_exit_fdot_n_s: float = 5.0
    contact_press_cap_m_s: float = 0.005
    press_energy_tank: PressEnergyTankConfig = field(
        default_factory=PressEnergyTankConfig
    )
    port_passivity: PortPassivityConfig = field(
        default_factory=PortPassivityConfig
    )
    # Measured end-to-end cmd→TCP delay for press budget (≈ 60–90 ms).
    t_eff_s: float = 0.070
    # Press-only Teff budget (works with barrier disabled).
    delay_press_budget_enabled: bool = True
    delay_press_budget_min_n: float = 0.40
    delay_press_budget_frac: float = 0.25
    delay_press_v_floor_m_s: float = 0.003
    delay_press_cap_lpf_tau_s: float = 0.18
    # SUSPECT_LOSS / low-force recovery (do not wait for confirmed LOST).
    suspect_recovery_enabled: bool = True
    suspect_recovery_vz_cap_m_s: float = 0.012
    suspect_recovery_vr_press_max_m_s: float = 0.003
    suspect_recovery_f_abs_n: float = 0.3
    suspect_recovery_f_frac: float = 0.35
    suspect_recovery_hold_s: float = 0.20
    # Force-task latched + under setpoint: soft-chase press ceiling [m/s].
    low_force_press_cap_m_s: float = 0.020
    low_force_press_enter_n: float = 1.80
    # Contact + wrong-sign Fz below −threshold → kill press (frame/fault).
    force_sign_fault_n: float = 3.5
    # Filtered TCP velocity → direction-safe admittance anti-windup.
    v_tcp_lpf_tau_s: float = 0.012
    v_tcp_clip_m_s: float = 0.12
    v_force_aw_enabled: bool = False
    v_force_aw_tau_s: float = 0.040
    # Short desired-force ramp re-armed on each reacquire [s].
    recontact_force_ramp_s: float = 0.25
    # Max slew on ΔD_hf [N·s/m per second] — kills bounce-freq param excitation.
    var_damping_hf_slew_max: float = 800.0
    # Wrist singularity: attenuate tool-wz tracking near q6 ≈ 0.
    wrist_relax_enabled: bool = True
    wrist_relax_enter_rad: float = 0.175  # ~10 deg
    wrist_relax_exit_rad: float = 0.35    # ~20 deg
    wrist_relax_floor: float = 0.30
    wrist_relax_lpf_tau_s: float = 0.08

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
            deadband_soft_tanh=bool(c.get("deadband_soft_tanh", False)),
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
                c.get("force_lateral_gain_floor", 0.35)
            ),
            force_dob=ForceDobConfig.from_dict(c),
            force_barrier=ForceBarrierConfig.from_dict(c),
            delay_damping_enabled=bool(
                c.get("delay_damping_enabled", True)
            ),
            delay_damping_kappa=float(c.get("delay_damping_kappa", 0.8)),
            delay_damping_bd_max=float(
                c.get("delay_damping_bd_max", 600.0)
            ),
            delay_damping_mode=str(
                c.get("delay_damping_mode", "impact_only")
            ),
            impact_damping_hold_s=float(
                c.get("impact_damping_hold_s", 0.12)
            ),
            impact_damping_release_s=float(
                c.get("impact_damping_release_s", 0.10)
            ),
            impact_damping_zeta=float(
                c.get("impact_damping_zeta", 0.9)
            ),
            impact_damping_d_min=float(
                c.get("impact_damping_d_min", 60.0)
            ),
            impact_damping_d_max=float(
                c.get("impact_damping_d_max", 100.0)
            ),
            impact_ke_floor=float(c.get("impact_ke_floor", 800.0)),
            impact_compress_v_min_m_s=float(
                c.get("impact_compress_v_min_m_s", 0.003)
            ),
            impact_fdot_arm_n_s=float(
                c.get("impact_fdot_arm_n_s", 12.0)
            ),
            impact_fpred_over_n=float(
                c.get("impact_fpred_over_n", 0.15)
            ),
            impact_fpred_horizon_s=float(
                c.get("impact_fpred_horizon_s", 0.050)
            ),
            impact_arm_confirm_s=float(
                c.get("impact_arm_confirm_s", 0.010)
            ),
            impact_fdot_rearm_n_s=float(
                c.get("impact_fdot_rearm_n_s", 5.0)
            ),
            impact_rearm_f_frac=float(
                c.get("impact_rearm_f_frac", 0.9)
            ),
            retract_brake_damping_ns_m=float(
                c.get("retract_brake_damping_ns_m", 70.0)
            ),
            retract_brake_hold_s=float(
                c.get("retract_brake_hold_s", 0.060)
            ),
            retract_brake_release_s=float(
                c.get("retract_brake_release_s", 0.040)
            ),
            reverse_interlock_enter_m_s=float(
                c.get("reverse_interlock_enter_m_s", 0.004)
            ),
            reverse_interlock_exit_m_s=float(
                c.get("reverse_interlock_exit_m_s", 0.0015)
            ),
            reverse_interlock_enter_confirm_s=float(
                c.get("reverse_interlock_enter_confirm_s", 0.010)
            ),
            reverse_interlock_exit_confirm_s=float(
                c.get("reverse_interlock_exit_confirm_s", 0.010)
            ),
            v_gate_window_s=float(c.get("v_gate_window_s", 0.030)),
            d_extra_attack_s=float(c.get("d_extra_attack_s", 0.010)),
            d_extra_release_s=float(c.get("d_extra_release_s", 0.120)),
            d_extra_min_hold_s=float(c.get("d_extra_min_hold_s", 0.040)),
            impact_danger_f_over_n=float(
                c.get("impact_danger_f_over_n", 0.15)
            ),
            impact_danger_fdot_n_s=float(
                c.get("impact_danger_fdot_n_s", 5.0)
            ),
            impact_danger_f_over_fdot_n=float(
                c.get("impact_danger_f_over_fdot_n", 0.10)
            ),
            impact_safe_f_over_n=float(
                c.get("impact_safe_f_over_n", 0.25)
            ),
            impact_safe_confirm_s=float(
                c.get("impact_safe_confirm_s", 0.025)
            ),
            impact_pred_span_n=float(c.get("impact_pred_span_n", 1.5)),
            force_slew_press_m_s2=float(
                c.get("force_slew_press_m_s2", 0.60)
            ),
            force_slew_retract_m_s2=float(
                c.get("force_slew_retract_m_s2", 1.20)
            ),
            force_slew_press_to_retract_m_s2=float(
                c.get("force_slew_press_to_retract_m_s2", 2.0)
            ),
            force_slew_zero_cross_m_s2=float(
                c.get("force_slew_zero_cross_m_s2", 0.70)
            ),
            free_seek_vz_m_s=float(c.get("free_seek_vz_m_s", 0.080)),
            free_seek_accel_m_s2=float(
                c.get("free_seek_accel_m_s2", 0.80)
            ),
            free_seek_exit_force_n=float(
                c.get("free_seek_exit_force_n", 0.15)
            ),
            free_seek_exit_fdot_n_s=float(
                c.get("free_seek_exit_fdot_n_s", 5.0)
            ),
            contact_press_cap_m_s=float(
                c.get("contact_press_cap_m_s", 0.005)
            ),
            press_energy_tank=PressEnergyTankConfig.from_dict(raw),
            port_passivity=PortPassivityConfig.from_dict(raw),
            t_eff_s=float(c.get("t_eff_s", 0.070)),
            delay_press_budget_enabled=bool(
                c.get("delay_press_budget_enabled", True)
            ),
            delay_press_budget_min_n=float(
                c.get("delay_press_budget_min_n", 0.40)
            ),
            delay_press_budget_frac=float(
                c.get("delay_press_budget_frac", 0.25)
            ),
            delay_press_v_floor_m_s=float(
                c.get("delay_press_v_floor_m_s", 0.003)
            ),
            delay_press_cap_lpf_tau_s=float(
                c.get("delay_press_cap_lpf_tau_s", 0.18)
            ),
            suspect_recovery_enabled=bool(
                c.get("suspect_recovery_enabled", True)
            ),
            suspect_recovery_vz_cap_m_s=float(
                c.get("suspect_recovery_vz_cap_m_s", 0.012)
            ),
            suspect_recovery_vr_press_max_m_s=float(
                c.get("suspect_recovery_vr_press_max_m_s", 0.003)
            ),
            suspect_recovery_f_abs_n=float(
                c.get("suspect_recovery_f_abs_n", 0.3)
            ),
            suspect_recovery_f_frac=float(
                c.get("suspect_recovery_f_frac", 0.35)
            ),
            suspect_recovery_hold_s=float(
                c.get("suspect_recovery_hold_s", 0.20)
            ),
            low_force_press_cap_m_s=float(
                c.get("low_force_press_cap_m_s", 0.020)
            ),
            low_force_press_enter_n=float(
                c.get("low_force_press_enter_n", 1.80)
            ),
            force_sign_fault_n=float(c.get("force_sign_fault_n", 3.5)),
            v_tcp_lpf_tau_s=float(c.get("v_tcp_lpf_tau_s", 0.012)),
            v_tcp_clip_m_s=float(c.get("v_tcp_clip_m_s", 0.12)),
            v_force_aw_enabled=bool(c.get("v_force_aw_enabled", False)),
            v_force_aw_tau_s=float(c.get("v_force_aw_tau_s", 0.040)),
            recontact_force_ramp_s=float(
                c.get("recontact_force_ramp_s", 0.25)
            ),
            var_damping_hf_slew_max=float(
                c.get("var_damping_hf_slew_max", 800.0)
            ),
            wrist_relax_enabled=bool(c.get("wrist_relax_enabled", True)),
            wrist_relax_enter_rad=float(
                c.get("wrist_relax_enter_rad", 0.175)
            ),
            wrist_relax_exit_rad=float(
                c.get("wrist_relax_exit_rad", 0.35)
            ),
            wrist_relax_floor=float(c.get("wrist_relax_floor", 0.30)),
            wrist_relax_lpf_tau_s=float(
                c.get("wrist_relax_lpf_tau_s", 0.08)
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
        # Smooth only zero-centered D_extra (never fold Di into D0).
        self._d_extra_smooth = 0.0
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
        self.cap_press_z = float("nan")
        self.cap_retract_z = float("nan")
        self.force_pred_z = float("nan")
        self.force_dot_z = float("nan")
        self.ke_barrier = float(self.cfg.force_barrier.ke_seek_default)
        self.damping_delay_z = 0.0
        self.damping_impact_z = 0.0
        self._impact_timer_s = 0.0
        self._impact_rearm_ready = True
        self._impact_arm_confirm_s = 0.0
        self._impact_safe_timer_s = 0.0
        self._retract_brake_timer_s = 0.0
        self._prev_retract_fast_hold = False
        self._reverse_interlock = False
        self._interlock_enter_timer_s = 0.0
        self._interlock_exit_timer_s = 0.0
        self.damping_retract_brake_z = 0.0
        self.reverse_interlock_active = False
        self.reverse_interlock_gate = 1.0
        self.impact_danger = False
        self.d_extra_target_z = 0.0
        self._d_extra_hold_s = 0.0
        self.f_err_raw = 0.0
        self.f_err_eff = 0.0
        self.v_force_raw = 0.0
        self.v_tcp_z_gate = float("nan")
        self.tank_energy_j = 0.0
        self.tank_gamma = 1.0
        self._press_tank = PressEnergyTank(self.cfg.press_energy_tank)
        self._port_po = PortPassivityObserver(self.cfg.port_passivity)
        self.port_energy_j = float(self.cfg.port_passivity.e_initial_j)
        self.port_excess_j = 0.0
        self.damping_pc_z = 0.0
        self.free_seek_active = True
        self._energy_limit_active = False
        self._tank_pc_active = False
        self.force_sign_fault = False
        self._overshoot_episode_s = 0.0
        self._gate_hist: deque[tuple[float, float]] = deque()
        self._gate_x = 0.0
        self._gate_t = 0.0
        self._x_tcp_integ = 0.0
        self._x_tcp_prev = 0.0
        self._have_tcp_x = False
        self._recontact_ramp_timer_s = 0.0
        self._wrist_relax = 1.0
        self.wrist_relax_scale = 1.0
        self.v_tcp_z_actual = float("nan")
        self.v_tcp_z_filt = float("nan")
        self._v_tcp_med: list[float] = []
        self.suspect_recovery_active = False
        self.dob_frozen = False
        self._suspect_recovery_timer_s = 0.0
        self._prev_suspect_recovery = False
        self._press_budget_filt: float | None = None
        # Arm lateral chase softener only after real tool-XY scan motion.
        self._lat_soften_hold_s = 0.0
        self._v_lateral_for_hf = 0.0
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
        self._d_extra_smooth = 0.0
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
        self.cap_press_z = float("nan")
        self.cap_retract_z = float("nan")
        self.force_pred_z = float("nan")
        self.force_dot_z = float("nan")
        self.ke_barrier = float(self.cfg.force_barrier.ke_seek_default)
        self.damping_delay_z = 0.0
        self.damping_impact_z = 0.0
        self._impact_timer_s = 0.0
        self._impact_rearm_ready = True
        self._impact_arm_confirm_s = 0.0
        self._impact_safe_timer_s = 0.0
        self._retract_brake_timer_s = 0.0
        self._prev_retract_fast_hold = False
        self._reverse_interlock = False
        self._interlock_enter_timer_s = 0.0
        self._interlock_exit_timer_s = 0.0
        self.damping_retract_brake_z = 0.0
        self.reverse_interlock_active = False
        self.reverse_interlock_gate = 1.0
        self.impact_danger = False
        self.d_extra_target_z = 0.0
        self._d_extra_hold_s = 0.0
        self.f_err_raw = 0.0
        self.f_err_eff = 0.0
        self.v_force_raw = 0.0
        self.v_tcp_z_gate = float("nan")
        self.tank_energy_j = 0.0
        self.tank_gamma = 1.0
        self._press_tank.reset()
        self._port_po.reset()
        self.port_energy_j = float(self._port_po.energy_j)
        self.port_excess_j = 0.0
        self.damping_pc_z = 0.0
        self.free_seek_active = True
        self._energy_limit_active = False
        self._tank_pc_active = False
        self.force_sign_fault = False
        self._overshoot_episode_s = 0.0
        self._gate_hist.clear()
        self._gate_x = 0.0
        self._gate_t = 0.0
        self._x_tcp_integ = 0.0
        self._x_tcp_prev = 0.0
        self._have_tcp_x = False
        self._recontact_ramp_timer_s = 0.0
        self._wrist_relax = 1.0
        self.wrist_relax_scale = 1.0
        self.v_tcp_z_actual = float("nan")
        self.v_tcp_z_filt = float("nan")
        self._v_tcp_med.clear()
        self.suspect_recovery_active = False
        self.dob_frozen = False
        self._suspect_recovery_timer_s = 0.0
        self._prev_suspect_recovery = False
        self._press_budget_filt = None
        self._lat_soften_hold_s = 0.0
        self._v_lateral_for_hf = 0.0
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

    def _press_vz_cap(self) -> float:
        """Press (+z) cap: free-seek vs contact / recovery ceilings.

        Tight ``contact_press_cap`` only during energy-limit episodes
        (impact / overshoot / over-force). Steady under-force tracking uses
        the normal ``max_vz`` / low-force ceilings so 2 N can be held.
        """
        if self.free_seek_active:
            cap = max(
                float(self.cfg.free_seek_vz_m_s),
                float(self.cfg.max_vz_tool_m_s),
            )
        else:
            cap = self._v_z_cap()
            if (
                float(self.cfg.contact_press_cap_m_s) > 0.0
                and bool(getattr(self, "_energy_limit_active", False))
            ):
                cap = min(cap, float(self.cfg.contact_press_cap_m_s))
        if (
            self._recontact_timer_s > 0.0
            and self.cfg.recontact_vz_cap_m_s > 0.0
        ):
            cap = min(cap, float(self.cfg.recontact_vz_cap_m_s))
        if (
            (
                self.suspect_recovery_active
                or self._suspect_recovery_timer_s > 0.0
            )
            and self.cfg.suspect_recovery_vz_cap_m_s > 0.0
        ):
            cap = min(cap, float(self.cfg.suspect_recovery_vz_cap_m_s))
        return max(cap, 0.0)

    def _update_v_tcp_filt(self, dt_eff: float) -> float:
        """Median + clip + LPF on measured tool-Z TCP velocity."""
        cfg = self.cfg
        if not np.isfinite(self.v_tcp_z_actual):
            return (
                float(self.v_tcp_z_filt)
                if np.isfinite(self.v_tcp_z_filt)
                else float(self.v_force_z)
            )
        clip = max(float(cfg.v_tcp_clip_m_s), 1e-3)
        raw = float(np.clip(self.v_tcp_z_actual, -clip, clip))
        self._v_tcp_med.append(raw)
        if len(self._v_tcp_med) > 3:
            self._v_tcp_med.pop(0)
        med = float(np.median(self._v_tcp_med))
        tau = max(float(cfg.v_tcp_lpf_tau_s), 0.0)
        if not np.isfinite(self.v_tcp_z_filt) or tau <= 1e-9 or dt_eff <= 0.0:
            self.v_tcp_z_filt = med
        else:
            blend = min(1.0, dt_eff / tau)
            self.v_tcp_z_filt += blend * (med - self.v_tcp_z_filt)
        self._update_v_tcp_gate(raw, dt_eff)
        return float(self.v_tcp_z_filt)

    def _update_v_tcp_gate(self, v_raw: float, dt_eff: float) -> float:
        """Slow reliable TCP-Z velocity via position regression (mode gating)."""
        cfg = self.cfg
        dt = max(float(dt_eff), 0.0)
        self._gate_t += dt
        self._gate_x += float(v_raw) * dt
        self._gate_hist.append((self._gate_t, self._gate_x))
        win = max(float(cfg.v_gate_window_s), dt)
        t_cut = self._gate_t - win
        while len(self._gate_hist) > 2 and self._gate_hist[0][0] < t_cut:
            self._gate_hist.popleft()
        if len(self._gate_hist) < 2:
            self.v_tcp_z_gate = float(v_raw)
            return self.v_tcp_z_gate
        t0, x0 = self._gate_hist[0]
        t1, x1 = self._gate_hist[-1]
        den = t1 - t0
        if den <= 1e-6:
            self.v_tcp_z_gate = float(v_raw)
        else:
            # OLS slope for uniform-ish samples in the window.
            ts = np.asarray([p[0] for p in self._gate_hist], dtype=float)
            xs = np.asarray([p[1] for p in self._gate_hist], dtype=float)
            t_mean = float(np.mean(ts))
            x_mean = float(np.mean(xs))
            var_t = float(np.sum((ts - t_mean) ** 2))
            if var_t < 1e-12:
                self.v_tcp_z_gate = float((x1 - x0) / den)
            else:
                self.v_tcp_z_gate = float(
                    np.sum((ts - t_mean) * (xs - x_mean)) / var_t
                )
        return float(self.v_tcp_z_gate)

    def _force_axis_slew(
        self, v_target: float, v_prev: float, dt_s: float
    ) -> float:
        """Asymmetric slew: air seek fast; contact press soft; retract open."""
        cfg = self.cfg
        dt = max(float(dt_s), 1e-6)
        if v_target >= 0.0:
            if self.free_seek_active:
                a_lim = float(cfg.free_seek_accel_m_s2)
            else:
                a_lim = float(cfg.force_slew_press_m_s2)
        elif v_prev > 0.0:
            a_lim = float(cfg.force_slew_press_to_retract_m_s2)
        else:
            a_lim = float(cfg.force_slew_retract_m_s2)
        if v_target * v_prev < 0.0 and not self.free_seek_active:
            a_lim = min(a_lim, float(cfg.force_slew_zero_cross_m_s2))
        dv = a_lim * dt
        return float(np.clip(v_target, v_prev - dv, v_prev + dv))

    def _arm_impact_burst(self) -> None:
        self._impact_timer_s = max(
            self._impact_timer_s,
            float(self.cfg.impact_damping_hold_s)
            + float(self.cfg.impact_damping_release_s),
        )
        self._impact_rearm_ready = False

    def _critical_impact_delta(
        self, *, mass_z: float, damping_ke: float
    ) -> float:
        """Extra damping so D0+ΔD ≈ clip(2ζ√(M Ke), d_min, d_max)."""
        cfg = self.cfg
        ke = max(
            float(self.ke_est),
            float(self.ke_barrier),
            float(cfg.impact_ke_floor),
            1.0,
        )
        d_total = (
            2.0
            * float(cfg.impact_damping_zeta)
            * math.sqrt(max(float(mass_z), 1e-3) * ke)
        )
        d_total = float(
            min(
                max(d_total, float(cfg.impact_damping_d_min)),
                float(cfg.impact_damping_d_max),
            )
        )
        return max(0.0, d_total - float(damping_ke))

    def _delay_press_budget_cap(
        self,
        *,
        f_ext_z: float,
        f_des_z: float,
        v_hi: float,
        dt_eff: float,
    ) -> float:
        """Press-only Teff budget: v ≤ (Fdes+budget−f)/(Ke·Teff)."""
        cfg = self.cfg
        if not cfg.delay_press_budget_enabled or v_hi <= 0.0:
            return v_hi
        ke = max(
            float(self.ke_est),
            float(self.ke_barrier),
            float(cfg.impact_ke_floor) * 0.5,
            200.0,
        )
        teff = max(float(cfg.t_eff_s), 1e-4)
        budget = max(
            float(cfg.delay_press_budget_min_n),
            float(cfg.delay_press_budget_frac) * abs(float(f_des_z)),
            1e-6,
        )
        f_pred = (
            float(self.force_pred_z)
            if np.isfinite(self.force_pred_z)
            else float(f_ext_z)
        )
        raw = (float(f_des_z) + budget - f_pred) / (ke * teff)
        floor = float(cfg.delay_press_v_floor_m_s)
        raw = float(min(max(raw, floor), v_hi))
        tau = max(float(cfg.delay_press_cap_lpf_tau_s), 0.0)
        if self._press_budget_filt is None or tau <= 1e-9 or dt_eff <= 0.0:
            self._press_budget_filt = raw
        else:
            # Fast attack when tightening (over-force); slow release when opening.
            tau_use = tau if raw >= self._press_budget_filt else max(tau * 0.25, 0.01)
            blend = min(1.0, dt_eff / tau_use)
            self._press_budget_filt += blend * (
                raw - self._press_budget_filt
            )
        return float(min(max(self._press_budget_filt, floor), v_hi))

    def _update_delta_d_hf(
        self,
        dt_eff: float,
        *,
        abs_eff_n: float = 0.0,
        v_lateral_m_s: float = 0.0,
    ) -> float:
        """Fast-attack / hold / fast-release ΔD from the Dimeas index.

        Large |e_f| blocks *new* arming so chase is not choked, but does not
        force-zero an already-active hold (that used to swing ΔD 0↔380 at the
        bounce frequency and feed the limit cycle).  A slew limit caps the
        same parametric excitation.
        """
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
        soft = max(float(cfg.force_lateral_soft_m_s), 0.0)
        full = max(float(cfg.force_lateral_full_m_s), soft + 1e-6)
        moving = float(v_lateral_m_s) >= full
        strong_chatter = is_now >= max(1.2, 3.0 * float(cfg.var_damping_hf_on))
        target = float(cfg.var_damping_d_u) * is_now
        if (
            (not self._hf_active)
            and near_setpoint
            and is_now >= float(cfg.var_damping_hf_on)
            and (moving or strong_chatter)
        ):
            self._hf_active = True
            self._hf_hold_s = float(cfg.var_damping_hf_hold_s)
        if self._hf_active:
            if (
                is_now >= float(cfg.var_damping_hf_off)
                and (moving or strong_chatter)
            ):
                # Renew hold even during large |e_f|; only arming is gated.
                self._hf_hold_s = max(
                    self._hf_hold_s, float(cfg.var_damping_hf_hold_s)
                )
            else:
                self._hf_hold_s = max(0.0, self._hf_hold_s - dt_eff)
            if self._hf_hold_s <= 0.0 and (
                is_now < float(cfg.var_damping_hf_off)
                or not (moving or strong_chatter)
            ):
                self._hf_active = False
                target = 0.0
            if not moving and not strong_chatter:
                target = 0.0
            tau = max(float(cfg.var_damping_hf_attack_s), 1e-4)
        else:
            target = 0.0
            tau = max(float(cfg.var_damping_hf_release_s), 1e-4)
            if abs(float(abs_eff_n)) > float(cfg.var_damping_hf_err_n):
                tau = min(
                    tau, max(float(cfg.var_damping_hf_release_fast_s), 1e-4)
                )
            if not moving:
                tau = min(
                    tau, max(float(cfg.var_damping_hf_release_fast_s), 1e-4)
                )
        blend = min(1.0, dt_eff / tau)
        next_d = self._delta_d_hf + blend * (target - self._delta_d_hf)
        slew = float(cfg.var_damping_hf_slew_max)
        if slew > 0.0 and dt_eff > 0.0:
            max_step = slew * dt_eff
            next_d = float(
                np.clip(
                    next_d,
                    self._delta_d_hf - max_step,
                    self._delta_d_hf + max_step,
                )
            )
        self._delta_d_hf = next_d
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
        q_meas: np.ndarray | None = None,
    ) -> np.ndarray:
        # The hardware-proven admittance dynamics retain the nominal fixed dt.
        # Wall-clock dt is used only by contact/fast-force confirmation timers.
        if v_tcp_z_actual is not None and np.isfinite(v_tcp_z_actual):
            self.v_tcp_z_actual = float(v_tcp_z_actual)
        else:
            self.v_tcp_z_actual = float("nan")
        if dt_actual is not None and np.isfinite(dt_actual):
            dt_contact = float(np.clip(dt_actual, 0.0025, 0.020))
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
        kp_rot = np.asarray(cfg.kp_pos[3:6] * cfg.track_axes[3:6], dtype=float)
        wrist_scale = self._update_wrist_relax(q_meas, dt_contact)
        kp_rot[2] *= wrist_scale
        v_corr[3:6] = r_mat @ (kp_rot * err_rot_tool)
        v_pos_base = vel_ff + v_corr

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

        # Force dynamics always run at nominal dt.  Governor time_scale freezes
        # trajectory / FF progression only (see loop.py); freezing the force
        # integrator mid-contact would leave an over-force uncorrected.
        dt_eff = float(self.dt)
        if force_task_active:
            self._contact_time_s += dt_eff
        rising_edge = bool(contact_update.acquired)
        reacquired = bool(contact_update.reacquired)

        # SUSPECT_LOSS / low-force: immediate slow re-press (don't wait LOST).
        f_des_abs = abs(float(f_des[2]))
        f_low = max(
            float(cfg.suspect_recovery_f_abs_n),
            float(cfg.suspect_recovery_f_frac) * f_des_abs,
        )
        self.suspect_recovery_active = bool(
            cfg.suspect_recovery_enabled
            and force_task_active
            and (
                contact_update.state
                == PhysicalContactTracker.SUSPECT_LOSS
                or abs(f_ext_z) < f_low
            )
        )
        if self.suspect_recovery_active:
            self._suspect_recovery_timer_s = max(
                self._suspect_recovery_timer_s,
                float(cfg.suspect_recovery_hold_s),
            )
        elif self._suspect_recovery_timer_s > 0.0:
            self._suspect_recovery_timer_s = max(
                0.0, self._suspect_recovery_timer_s - dt_eff
            )

        if reacquired or rising_edge:
            self._recontact_timer_s = max(
                self._recontact_timer_s,
                float(cfg.recontact_hold_s),
            )
            self._arm_impact_burst()
            self._press_tank.seed()
        # Re-hit after bounce trough: force rising out of suspect recovery.
        if (
            force_task_active
            and self._prev_suspect_recovery
            and not self.suspect_recovery_active
            and abs(f_ext_z) >= max(0.5, 0.5 * f_des_abs)
        ):
            self._arm_impact_burst()
            self._press_tank.seed()
            self._recontact_timer_s = max(
                self._recontact_timer_s,
                float(cfg.recontact_hold_s),
            )
        self._prev_suspect_recovery = bool(self.suspect_recovery_active)

        if reacquired and float(cfg.recontact_force_ramp_s) > 1e-6:
            # Re-arm a short desired-force ramp so re-hits don't demand full F.
            self._recontact_ramp_timer_s = float(cfg.recontact_force_ramp_s)
        if self._recontact_timer_s > 0.0:
            self._recontact_timer_s = max(
                0.0, self._recontact_timer_s - dt_contact
            )
        if self._recontact_ramp_timer_s > 0.0:
            self._recontact_ramp_timer_s = max(
                0.0, self._recontact_ramp_timer_s - dt_eff
            )
        if self._impact_timer_s > 0.0:
            self._impact_timer_s = max(
                0.0, self._impact_timer_s - dt_eff
            )
        # Dimeas only while confirmed CONTACT (not bounce trough / air).
        if contact_update.state == PhysicalContactTracker.CONTACT:
            self._update_instability_index(raw_z)
        else:
            # Bleed Is in air / suspect so post-lift D doesn't stick high.
            self.instability_index *= 0.90

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
        # 4d15c1d: do not mute tool-Y in free space — that felt like air hitch
        # at scan start. Contact latch / force axes already gate Z press.
        v_lateral_m_s = float(
            np.linalg.norm((r_mat.T @ v_pos_base[:3])[:2])
        )
        self._v_lateral_for_hf = v_lateral_m_s
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

        v_force_tool = np.zeros(6, dtype=float)
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
            sensor_age_s=sensor_age_s,
            chase_scale=chase_scale,
            physical_contact=physical_contact,
            f_ext_z=f_ext_z,
        )
        v_cmd_tool, v_cmd_base = self.fuse_tool_sleeve(
            v_pos_base,
            v_force_tool,
            r_mat,
        )
        # Barrier + recontact cap only limits press (+z); over-force retract
        # stays open unless the barrier itself schedules a retract floor.
        v_z_cap = self._v_z_cap()
        press_cap = self._press_vz_cap()
        if np.isfinite(self.cap_press_z):
            press_cap = (
                min(press_cap, self.cap_press_z)
                if press_cap > 0.0
                else self.cap_press_z
            )
        retract_cap = v_z_cap
        if np.isfinite(self.cap_retract_z) and self.cap_retract_z >= 0.0:
            retract_cap = min(retract_cap, self.cap_retract_z)
        if v_z_cap > 0.0:
            lo = -retract_cap if retract_cap > 0.0 else -v_z_cap
            hi = press_cap if press_cap > 0.0 else v_z_cap
            v_cmd_tool[2] = float(np.clip(v_cmd_tool[2], lo, hi))
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
                # Asymmetric force-axis slew (was previously skipped).
                v_final[index] = self._force_axis_slew(
                    float(v_final[index]),
                    float(self.last_v_cmd[index]),
                    float(self.dt),
                )
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
        # Reacquire: briefly pull desired force back toward a soft start so
        # every re-hit does not demand full F_des into a stiff surface.
        ramp_s = float(cfg.recontact_force_ramp_s)
        if ramp_s > 1e-6 and self._recontact_ramp_timer_s > 0.0 and f_des_z > 0.0:
            alpha = float(
                np.clip(1.0 - self._recontact_ramp_timer_s / ramp_s, 0.0, 1.0)
            )
            f_soft = min(f_des_z, max(0.35 * f_des_z, cfg.contact_threshold_n))
            f_eff = f_soft + (f_eff - f_soft) * alpha
        self.f_des_z_eff = float(f_eff)
        return float(f_eff)

    def _update_wrist_relax(
        self,
        q_meas: np.ndarray | None,
        dt_s: float,
    ) -> float:
        """Attenuate tool-wz tracking near wrist singularity (q6 ≈ 0)."""
        cfg = self.cfg
        if not cfg.wrist_relax_enabled:
            self._wrist_relax = 1.0
            self.wrist_relax_scale = 1.0
            return 1.0
        target = 1.0
        if q_meas is not None:
            q = np.asarray(q_meas, dtype=float).reshape(-1)
            # 8-DoF: rail + j1..j7 → q6 is index 6; 7-DoF arm-only → index 5.
            idx = 6 if q.size >= 8 else (5 if q.size >= 6 else -1)
            if idx >= 0:
                aq6 = abs(float(q[idx]))
                enter = max(float(cfg.wrist_relax_enter_rad), 1e-6)
                exit_ = max(float(cfg.wrist_relax_exit_rad), enter + 1e-6)
                floor = float(np.clip(cfg.wrist_relax_floor, 0.0, 1.0))
                if aq6 <= enter:
                    target = floor
                elif aq6 >= exit_:
                    target = 1.0
                else:
                    u = (aq6 - enter) / (exit_ - enter)
                    blend = u * u * (3.0 - 2.0 * u)
                    target = floor + (1.0 - floor) * blend
        tau = max(float(cfg.wrist_relax_lpf_tau_s), 0.0)
        if tau > 1e-9 and dt_s > 0.0:
            blend = min(1.0, dt_s / tau)
            self._wrist_relax += blend * (target - self._wrist_relax)
        else:
            self._wrist_relax = target
        self.wrist_relax_scale = float(self._wrist_relax)
        return self.wrist_relax_scale

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
        dt_eff: float,
        rising_edge: bool,
        desired_force_n: float = 0.0,
        raw_force_z: float | None = None,
        dt_contact: float | None = None,
        sensor_age_s: float | None = None,
        chase_scale: float = 1.0,
        physical_contact: bool = False,
        f_ext_z: float = 0.0,
    ) -> float:
        cfg = self.cfg
        self.f_err_raw = float(f_err)
        if cfg.deadband_soft_tanh:
            eps = max(float(cfg.deadband_n), 1e-4)
            eff = soft_tanh_eff(f_err, eps)
        else:
            eff = smooth_deadband_eff(
                f_err,
                cfg.deadband_n,
                cfg.deadband_width_n,
            )
        self.f_err_eff = float(eff)
        mass_z = max(float(self._m_z_now), 1e-3)
        dt_c = self.dt if dt_contact is None else float(dt_contact)

        # --- Barrier telemetry + Teff press budget inputs.
        self._force_barrier.update_fdot(float(f_ext_z), dt_eff)
        v_tcp_filt = self._update_v_tcp_filt(dt_eff)
        v_tcp_gate = (
            float(self.v_tcp_z_gate)
            if np.isfinite(self.v_tcp_z_gate)
            else float(v_tcp_filt)
        )
        # Position increment for tank (integrate measured TCP vel).
        if np.isfinite(self.v_tcp_z_actual) and dt_eff > 0.0:
            self._x_tcp_integ += float(self.v_tcp_z_actual) * dt_eff
        if not self._have_tcp_x:
            self._x_tcp_prev = self._x_tcp_integ
            self._have_tcp_x = True
        dx_tcp = self._x_tcp_integ - self._x_tcp_prev
        self._x_tcp_prev = self._x_tcp_integ

        self._force_barrier.note_contact_edge(bool(physical_contact))
        self._force_barrier.update_ke(
            f_z=float(f_ext_z),
            v_tcp_z=v_tcp_filt,
            in_contact=bool(physical_contact),
            dt_eff=dt_eff,
            f_des_z=float(desired_force_n),
        )
        self.ke_barrier = float(self._force_barrier.ke_barrier)
        self.force_dot_z = float(self._force_barrier.f_dot_z)
        # One-sided Teff-horizon prediction (press safety; ignore negative ḟ).
        pred_horizon = max(
            float(cfg.t_eff_s),
            float(cfg.force_barrier.t_pred_s),
            0.0,
        )
        self.force_pred_z = float(
            float(f_ext_z) + max(self.force_dot_z, 0.0) * pred_horizon
        )

        # Fast retract guard early: predictive stop + ḟ_fast for impact arm.
        retract_fast_hold = self._fast_retract_guard.update(
            raw_force_n=raw_force_z,
            desired_force_n=desired_force_n,
            filtered_eff_n=eff,
            active_reference_m_s=self.v_r_z,
            dt_s=dt_c,
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
        if retract_fast_hold and not self._prev_retract_fast_hold:
            self._retract_brake_timer_s = max(
                self._retract_brake_timer_s,
                float(cfg.retract_brake_hold_s)
                + float(cfg.retract_brake_release_s),
            )
        self._prev_retract_fast_hold = bool(retract_fast_hold)

        # Free-seek exit on fast raw force (before confirmed contact FSM).
        fdot_exit = (
            float(self._fast_retract_guard.force_dot_fast)
            if self._fast_retract_guard.valid
            else (
                float(self.force_dot_z)
                if np.isfinite(self.force_dot_z)
                else 0.0
            )
        )
        f_fast_exit = (
            float(self.force_fast_z)
            if np.isfinite(self.force_fast_z)
            else float(f_ext_z)
        )
        if self.free_seek_active and (
            abs(f_fast_exit) >= float(cfg.free_seek_exit_force_n)
            or fdot_exit >= float(cfg.free_seek_exit_fdot_n_s)
            or bool(physical_contact)
        ):
            self.free_seek_active = False
        elif (
            not physical_contact
            and abs(float(f_ext_z)) < max(0.30, float(cfg.free_seek_exit_force_n))
            and self.physical_contact_state
            in (PhysicalContactTracker.FREE, PhysicalContactTracker.LOST)
        ):
            self.free_seek_active = True

        use_fast = bool(self._fast_retract_guard.valid) and np.isfinite(
            self.force_fast_z
        )
        f_fast = float(self.force_fast_z) if use_fast else float(f_ext_z)
        if use_fast and np.isfinite(self._fast_retract_guard.force_dot_fast):
            fdot_fast = float(self._fast_retract_guard.force_dot_fast)
        else:
            fdot_fast = (
                float(self.force_dot_z)
                if np.isfinite(self.force_dot_z)
                else 0.0
            )
        impact_pred_h = max(float(cfg.impact_fpred_horizon_s), 0.0)
        force_pred_up = f_fast + max(fdot_fast, 0.0) * impact_pred_h
        f_des = float(desired_force_n)
        # Continuous impact danger (renew while unsafe — not a one-shot pulse).
        compressing_actual = (
            np.isfinite(v_tcp_gate) and float(v_tcp_gate) > 0.002
        )
        impact_danger = bool(physical_contact) and (
            force_pred_up > f_des + float(cfg.impact_danger_f_over_n)
            or (
                fdot_fast > float(cfg.impact_danger_fdot_n_s)
                and float(f_ext_z) > f_des + float(cfg.impact_danger_f_over_fdot_n)
            )
            or (
                fdot_fast > float(cfg.impact_fdot_arm_n_s)
                and self._impact_rearm_ready
            )
        )
        self.impact_danger = bool(impact_danger)
        # Energy limit (tank γ / PO-PC / tight press cap): over-force &
        # bounce only — never starve legitimate under-force tracking.
        over_force = float(f_ext_z) > f_des + 0.25
        self._energy_limit_active = bool(
            impact_danger
            or self._impact_timer_s > 0.0
            or over_force
            or retract_fast_hold
            or self._retract_brake_timer_s > 0.0
            or self._overshoot_episode_s > 0.0
        )
        if impact_danger:
            self._impact_timer_s = max(
                self._impact_timer_s,
                float(cfg.impact_damping_hold_s),
            )
            self._impact_safe_timer_s = 0.0
            self._impact_rearm_ready = False
            self._impact_arm_confirm_s = 0.0
        else:
            safe_to_release = (
                fdot_fast <= 0.0
                and (not compressing_actual)
                and float(f_ext_z) < f_des + float(cfg.impact_safe_f_over_n)
            )
            if safe_to_release:
                self._impact_safe_timer_s += max(dt_eff, 0.0)
            else:
                self._impact_safe_timer_s = 0.0
            if (
                fdot_fast < float(cfg.impact_fdot_rearm_n_s)
                and abs(float(f_ext_z))
                < float(cfg.impact_rearm_f_frac) * abs(f_des)
            ):
                self._impact_rearm_ready = True

        # Steady damping: D0 (+ optional legacy Keemink b_d).
        if (
            cfg.adaptive_ke.enabled
            and cfg.adaptive_ke.drive_damping
            and in_contact
        ):
            damping_ke = float(self.adaptive_bd)
        else:
            damping_ke = float(cfg.admittance_damping_z)

        d_delay_mag = 0.0
        if cfg.delay_damping_enabled and physical_contact:
            t_dead = max(
                float(cfg.t_eff_s),
                float(cfg.force_barrier.t_dead_s),
                1e-4,
            )
            d_delay_mag = (
                float(cfg.delay_damping_kappa)
                * float(self.ke_barrier)
                * t_dead
                * float(self._force_barrier.contact_conf)
            )
        mode = str(cfg.delay_damping_mode).lower()
        d_delay = d_delay_mag if mode == "always" else 0.0
        self.damping_delay_z = float(d_delay)

        d_impact_peak = self._critical_impact_delta(
            mass_z=mass_z, damping_ke=damping_ke
        )
        # Continuous Di from predicted over-force (hard surfaces stay high).
        span = max(float(cfg.impact_pred_span_n), 1e-3)
        r_f = smoothstep01(
            (force_pred_up - f_des - float(cfg.impact_danger_f_over_n)) / span
        )
        r_dot = smoothstep01(
            (fdot_fast - float(cfg.impact_danger_fdot_n_s))
            / max(float(cfg.impact_fdot_arm_n_s), 1.0)
        )
        r_impact = max(r_f, r_dot) if bool(physical_contact) else 0.0
        # Timer hold (acquire / renewed danger): full Di until safe-confirmed.
        if self._impact_timer_s > 0.0 and not (
            self._impact_safe_timer_s + 1e-12
            >= max(float(cfg.impact_safe_confirm_s), 0.0)
        ):
            r_impact = 1.0
        d_impact = float(d_impact_peak) * float(r_impact)
        if (
            rising_edge
            and cfg.adaptive_ke.enabled
            and not cfg.adaptive_ke.drive_damping
            and not cfg.delay_damping_enabled
            and in_contact
        ):
            damping_ke = max(damping_ke, float(self.adaptive_bd))
        self.damping_impact_z = float(d_impact)

        d_base = float(damping_ke)
        d_brake = 0.0
        brake_release = float(cfg.retract_brake_release_s)
        brake_total = float(cfg.retract_brake_damping_ns_m)
        if self._retract_brake_timer_s > 0.0 and brake_total > 0.0:
            d_brake_peak = max(0.0, brake_total - d_base)
            if self._retract_brake_timer_s > brake_release:
                d_brake = d_brake_peak
            elif brake_release > 1e-9:
                d_brake = d_brake_peak * (
                    self._retract_brake_timer_s / brake_release
                )
            self._retract_brake_timer_s = max(
                0.0, self._retract_brake_timer_s - max(dt_eff, 0.0)
            )
        self.damping_retract_brake_z = float(d_brake)

        confirmed_contact = (
            self.physical_contact_state == PhysicalContactTracker.CONTACT
        )
        if confirmed_contact:
            damping_dimeas = self._update_delta_d_hf(
                dt_eff,
                abs_eff_n=abs(float(eff)),
                v_lateral_m_s=float(
                    getattr(self, "_v_lateral_for_hf", 0.0)
                ),
            )
        else:
            self._delta_d_hf = 0.0
            self._hf_hold_s = 0.0
            self._hf_active = False
            damping_dimeas = 0.0

        # One unified zero-centered D_extra target (no 25/60/70 boolean fight).
        d_extra_target = float(
            max(d_delay, d_impact, d_brake, damping_dimeas)
        )
        bd_max = float(cfg.delay_damping_bd_max)
        if bd_max <= 0.0 and cfg.adaptive_ke.bd_max > 0.0:
            bd_max = float(cfg.adaptive_ke.bd_max)
        if bd_max > 0.0:
            d_extra_target = min(d_extra_target, max(0.0, bd_max - d_base))
        self.d_extra_target_z = float(d_extra_target)
        if d_extra_target > self._d_extra_smooth + 1e-6:
            self._d_extra_hold_s = max(
                self._d_extra_hold_s, float(cfg.d_extra_min_hold_s)
            )
        if self._d_extra_hold_s > 0.0:
            self._d_extra_hold_s = max(
                0.0, self._d_extra_hold_s - max(dt_eff, 0.0)
            )
        if dt_eff > 0.0:
            if d_extra_target >= self._d_extra_smooth:
                # Fast attack: snap within one tick when attack_s ≤ dt.
                tau_d = max(float(cfg.d_extra_attack_s), 1e-4)
                if tau_d <= dt_eff + 1e-12:
                    self._d_extra_smooth = d_extra_target
                else:
                    blend = min(1.0, dt_eff / tau_d)
                    self._d_extra_smooth += blend * (
                        d_extra_target - self._d_extra_smooth
                    )
            elif self._d_extra_hold_s > 0.0:
                pass  # hold
            else:
                tau_d = max(float(cfg.d_extra_release_s), 0.02)
                blend = min(1.0, dt_eff / tau_d)
                self._d_extra_smooth += blend * (
                    d_extra_target - self._d_extra_smooth
                )
        else:
            self._d_extra_smooth = d_extra_target
        d_extra = float(max(self._d_extra_smooth, 0.0))
        self.damping_ke_z = d_base
        self.damping_dimeas_z = damping_dimeas
        self.damping_z_eff = float(d_base + d_extra)

        v_z_cap = self._v_z_cap()
        press_cap = self._press_vz_cap()
        # Latched force-task + under contact enter: hard 10–20 mm/s press.
        if (
            bool(in_contact)
            and (not self.free_seek_active)
            and float(cfg.low_force_press_cap_m_s) > 0.0
            and abs(float(f_ext_z))
            < float(cfg.low_force_press_enter_n)
        ):
            press_cap = min(press_cap, float(cfg.low_force_press_cap_m_s))
        # Teff delay budget (press-only) — not during free-space seek.
        if (
            (bool(in_contact) or bool(physical_contact))
            and (not self.free_seek_active)
        ):
            press_cap = min(
                press_cap,
                self._delay_press_budget_cap(
                    f_ext_z=float(f_ext_z),
                    f_des_z=float(desired_force_n),
                    v_hi=v_z_cap if v_z_cap > 0.0 else press_cap,
                    dt_eff=dt_eff,
                ),
            )

        # Optional barrier (usually disabled); retract stays open in press_only.
        self._force_barrier.caps(
            f_z=float(f_ext_z),
            f_des_z=float(desired_force_n),
            in_contact=bool(physical_contact),
            v_z_cap=v_z_cap,
            seek_vz_m_s=v_z_cap,
            v_z_cap_retract=v_z_cap,
            retract_fast_hold=bool(retract_fast_hold),
            dt_eff=dt_eff,
        )
        # Publish the *effective* press cap used this tick (budget / recovery).
        self.cap_press_z = float(press_cap)
        self.cap_retract_z = float(
            v_z_cap
            if not np.isfinite(self._force_barrier.cap_retract_z)
            else max(self._force_barrier.cap_retract_z, v_z_cap)
        )
        if (
            cfg.force_barrier.enabled
            and np.isfinite(self._force_barrier.cap_press_z)
            and self._force_barrier.cap_press_z >= 0.0
        ):
            press_cap = min(press_cap, float(self._force_barrier.cap_press_z))
            self.cap_press_z = float(press_cap)

        # Press FF: require real force (blocks free-flight windup at fz≈0) but
        # allow brief suspect_loss dips.  Retract (eff<0) stays open whenever
        # the force task is latched so over-force escape is never gated.
        press_ok = bool(physical_contact) or abs(float(f_ext_z)) >= 0.3
        ff_in_contact = (
            bool(in_contact) if float(eff) < 0.0 else press_ok
        )
        if self.suspect_recovery_active:
            # Block proactive press windup while bouncing off the surface.
            ff_in_contact = bool(in_contact) if float(eff) < 0.0 else False
        v_reference = self._update_proactive_v_r(
            eff,
            ff_in_contact,
            dt_eff,
            rising_edge=rising_edge,
            desired_force_n=desired_force_n,
            retract_fast_hold=retract_fast_hold,
            chase_scale=chase_scale,
        )
        # Clamp +vr in recovery; retract side stays open.
        vr_press_max = press_cap
        if self.suspect_recovery_active or self._suspect_recovery_timer_s > 0.0:
            vr_press_max = min(
                vr_press_max,
                float(cfg.suspect_recovery_vr_press_max_m_s),
            )
        self.v_r_z = float(
            np.clip(self.v_r_z, -v_z_cap if v_z_cap > 0.0 else -1.0, vr_press_max)
        )
        self._proactive_ff.v_r = self.v_r_z
        v_reference = self.v_r_z

        # DOB only in confirmed CONTACT (not latched force-task / suspect).
        dob_contact = confirmed_contact
        if self.suspect_recovery_active and self._force_dob.u_dob > 0.0:
            self._force_dob.u_dob = 0.0
        self.u_dob_z = self._force_dob.update(
            eff,
            dt_eff=dt_eff,
            in_contact=dob_contact,
            instability_index=self.instability_index,
            chase_scale=chase_scale,
        )
        if self.suspect_recovery_active and self.u_dob_z > 0.0:
            self.u_dob_z = 0.0
            self._force_dob.u_dob = 0.0
        self.dob_frozen = bool(self._force_dob.frozen)

        drive = float(eff) + float(self.u_dob_z)
        drive_retract = min(drive, 0.0)
        drive_press = max(drive, 0.0)

        # Wrong-sign Fz while in contact: never chase +vz toward Fd (144217).
        fault_n = max(float(cfg.force_sign_fault_n), 0.0)
        self.force_sign_fault = bool(
            fault_n > 0.0
            and bool(physical_contact)
            and float(f_ext_z) < -fault_n
        )
        if self.force_sign_fault:
            drive_press = 0.0
            v_reference = min(float(v_reference), 0.0)
            drive = drive_retract + drive_press

        # Overshoot-recovery episode only — never permanently lock slow push.
        if retract_fast_hold or self._retract_brake_timer_s > 0.0:
            self._overshoot_episode_s = max(
                self._overshoot_episode_s,
                float(cfg.retract_brake_hold_s)
                + float(cfg.retract_brake_release_s),
            )
        if self._overshoot_episode_s > 0.0:
            self._overshoot_episode_s = max(
                0.0, self._overshoot_episode_s - max(dt_eff, 0.0)
            )
        overshoot_recovery = self._overshoot_episode_s > 0.0
        over_force = float(f_ext_z) > float(desired_force_n) + 0.25
        self._energy_limit_active = bool(
            bool(self.impact_danger)
            or self._impact_timer_s > 0.0
            or over_force
            or overshoot_recovery
            or retract_fast_hold
            or self._retract_brake_timer_s > 0.0
        )
        # Tank/PO only on live danger / over-force / overshoot — not a lingering
        # impact_timer while already under Fd (else empty tank blocks re-press).
        self._tank_pc_active = bool(
            bool(self.impact_danger)
            or over_force
            or overshoot_recovery
            or (
                self._energy_limit_active
                and float(f_ext_z) >= float(desired_force_n) - 0.05
            )
        )

        # Gate velocity (regression) → press-drive blend during overshoot only.
        enter_v = max(float(cfg.reverse_interlock_enter_m_s), 0.0)
        exit_v = max(float(cfg.reverse_interlock_exit_m_s), 0.0)
        gate = 1.0
        if overshoot_recovery and enter_v > 0.0:
            exit_v = min(exit_v, enter_v)
            if np.isfinite(v_tcp_gate) and float(v_tcp_gate) < -enter_v:
                self._interlock_enter_timer_s += max(dt_eff, 0.0)
                self._interlock_exit_timer_s = 0.0
            elif np.isfinite(v_tcp_gate) and float(v_tcp_gate) > -exit_v:
                self._interlock_exit_timer_s += max(dt_eff, 0.0)
                self._interlock_enter_timer_s = 0.0
            if (
                self._interlock_enter_timer_s + 1e-12
                >= max(float(cfg.reverse_interlock_enter_confirm_s), 0.0)
            ):
                self._reverse_interlock = True
            if (
                self._interlock_exit_timer_s + 1e-12
                >= max(float(cfg.reverse_interlock_exit_confirm_s), 0.0)
            ):
                self._reverse_interlock = False
            if np.isfinite(v_tcp_gate):
                gate = smoothstep01(
                    (float(v_tcp_gate) + enter_v)
                    / max(enter_v - exit_v, 1e-6)
                )
            if self._reverse_interlock:
                gate = 0.0
        else:
            self._reverse_interlock = False
            self._interlock_enter_timer_s = 0.0
            self._interlock_exit_timer_s = 0.0
            gate = 1.0
        self.reverse_interlock_gate = float(np.clip(gate, 0.0, 1.0))
        if overshoot_recovery:
            drive_press *= float(self.reverse_interlock_gate)
            v_reference = (
                min(float(v_reference), 0.0)
                + float(self.reverse_interlock_gate)
                * max(float(v_reference), 0.0)
            )
        drive = drive_retract + drive_press
        self.reverse_interlock_active = bool(
            overshoot_recovery and self.reverse_interlock_gate < 0.99
        )

        # Press-side tank: scale only in energy-limit episodes.
        # Under-force tracking must keep γ=1 (else Et→0 permanently kills press).
        u_press = max(
            drive_press
            + max(d_base, 0.0) * max(float(v_reference), 0.0),
            0.0,
        )
        v_press_est = max(float(self.v_force_z), 0.0)
        if self._tank_pc_active:
            gamma = self._press_tank.observe_and_scale(
                f_ext_z=float(f_ext_z),
                dx_m=float(dx_tcp),
                u_press=u_press,
                v_press_est_m_s=v_press_est,
                dt_s=dt_eff,
            )
        else:
            # Book-keep credit only; refill toward initial for next bounce.
            self._press_tank.observe_and_scale(
                f_ext_z=float(f_ext_z),
                dx_m=float(dx_tcp),
                u_press=0.0,
                v_press_est_m_s=0.0,
                dt_s=dt_eff,
            )
            e0 = float(self.cfg.press_energy_tank.e_initial_j)
            emax = float(self.cfg.press_energy_tank.e_max_j)
            if float(f_ext_z) < float(desired_force_n):
                self._press_tank.energy_j = min(
                    emax, max(float(self._press_tank.energy_j), e0)
                )
            gamma = 1.0
            self._press_tank.gamma = 1.0
        self.tank_gamma = float(gamma)
        self.tank_energy_j = float(self._press_tank.energy_j)
        drive_press *= float(gamma)
        v_reference = (
            min(float(v_reference), 0.0)
            + float(gamma) * max(float(v_reference), 0.0)
        )
        drive = drive_retract + drive_press

        # Continuous real-port PO: always integrate F·dx; D_PC only on excess.
        # Do NOT clear excess when danger gates off (that forgave bounce energy).
        v_a = (
            float(v_tcp_gate)
            if np.isfinite(v_tcp_gate)
            else (
                float(v_tcp_filt)
                if np.isfinite(v_tcp_filt)
                else float(self.v_force_z)
            )
        )
        d_pc = self._port_po.update(
            f_ext_z=float(f_ext_z),
            dx_m=float(dx_tcp),
            v_actual_m_s=v_a,
            dt_s=dt_eff,
        )
        self.damping_pc_z = float(d_pc)
        self.port_energy_j = float(self._port_po.energy_j)
        self.port_excess_j = float(self._port_po.excess_j)

        if dt_eff <= 0.0:
            velocity = float(self.v_force_z)
        else:
            # Implicit Euler, zero-centered extras + PO/PC on measured v_a:
            #   (M/dt+D0+D_extra) v+ = M/dt·v + D0·vr + drive − D_PC·v_a
            denom = mass_z / dt_eff + max(d_base, 0.0) + max(d_extra, 0.0)
            velocity = (
                (mass_z / dt_eff) * self.v_force_z
                + max(d_base, 0.0) * v_reference
                + drive
                - max(d_pc, 0.0) * v_a
            ) / max(denom, 1e-6)
        # Telemetry: pre-clip admittance solution (not post-interlock zero).
        self.v_force_raw = float(velocity)
        if v_z_cap > 0.0:
            lo = -v_z_cap
            hi = press_cap if press_cap > 0.0 else v_z_cap
            velocity = float(np.clip(velocity, lo, hi))

        if (
            cfg.v_force_aw_enabled
            and dt_eff > 0.0
            and np.isfinite(v_tcp_filt)
        ):
            tau_aw = max(float(cfg.v_force_aw_tau_s), 1e-3)
            alpha = 1.0 - math.exp(-dt_eff / tau_aw)
            weaken_retract = (
                float(eff) < 0.0
                and velocity < 0.0
                and float(v_tcp_filt) >= 0.0
            )
            same_direction = velocity * float(v_tcp_filt) > 0.0
            reduces_magnitude = abs(float(v_tcp_filt)) < abs(velocity)
            if (
                (not weaken_retract)
                and same_direction
                and reduces_magnitude
            ):
                velocity = float(
                    velocity + alpha * (float(v_tcp_filt) - velocity)
                )
            if v_z_cap > 0.0:
                lo = -v_z_cap
                hi = press_cap if press_cap > 0.0 else v_z_cap
                velocity = float(np.clip(velocity, lo, hi))

        self.v_force_z = float(velocity)
        return float(velocity)


HybridMotionConfig = AdmittanceConfig
HybridMotionController = AdmittanceController
