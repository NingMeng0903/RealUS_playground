# Controller dump (post-cleanup)

Generated: 2026-08-03T16:09:22+08:00

Source root: `/media/camp/EXT_DRIVE/RealUS_playground/rm75_control`

Verbatim snapshot of controller yaml + Python after hesitation fixes, J6 barrier removal, and garbage/API/print/comment cleanup. Scan CSV logging retained.

Kept control features of note:
- Centering σ-yield when manip active (J6 exempt from yield)
- J5/J6 centering weights 2.5; `q_nominal` J6=45°
- Centering recovery latch (do not force manip while recovering)
- No J6 soft singularity barrier

---

# Verbatim controller code and parameters

## `configs/joint_admittance_8dof.yaml`

```yaml
# Joint-space 8-DOF inner loop (rail_y + RM75 arm) — configs/joint_admittance_8dof.yaml
#
# URDF: rm75_control/assets/robots/rm75_6f_8dof/RM75-6F-8dof.urdf
# Genesis viz: python -m rm75_control.control.joint_admittance_8dof.viewer.demo --show-viewer
# Param spec: joint_admittance_8dof/config/slider_rail.yaml (default viewer scene)

robot:
  ip: "192.168.1.18"
  port: 8080
  thread_mode: 2

timing:
  dt_ms: 5.0

# UDP arm-state push (rm_set_realtime_push). Requires robot.thread_mode: 2.
realtime_push:
  cycle: 1              # broadcast period = cycle * 5 ms (1 -> 200 Hz)
  port: 8098
  ip: "192.168.1.80"    # PC NIC on robot subnet — do not auto-detect on multi-NIC hosts
  force_coordinate: 0   # 0=sensor frame (matches rm_get_force_data force_data)

# Shared-memory state relay for split-process Genesis twin (same host).
# Match realtime_push (cycle=1 -> 200 Hz) so attach-mode WBC does not stair-step.
state_relay:
  enabled: false
  name: rm75_state
  hz: 200

inner:
  control_frame: tool
  euler_order: xyz
  # Sync RealMan active tool into Pinocchio link_7→tcp (force-hybrid / tool-Z).
  sync_tcp_from_robot: true

  v_scale: 0.8             # fraction of URDF joint velocity limit
  # Accel limits are unit-explicit (arm rad/s^2 vs rail m/s^2).
  a_max_arm: 18.0          # rad/s^2 per arm joint (1..7)
  a_max_rail_m_s2: 0.30    # m/s^2 for the rail
  position_margin_deg: 2.0
  # Keep 0: non-zero margin teleports q_cmd off the rail end-stop.
  position_margin_rail_mm: 0.0
  # QP velocity bound: stop q_cmd leading encoders (0 disables).
  resync_err_deg: 6.0            # arm joints 1..7 (degrees)
  resync_err_rail_mm: 20.0       # rail joint 0 (millimetres — units matter!)

  qp:
    # Escande slack QP: task_weight >> reg (~1e4) so secondary stays in nullspace.
    task_weight: [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
    # Effort allocation: rail 1e-3 (mass-exempt), shoulder/elbow 1e-2, wrist 5e-3.
    reg: [1.0e-3, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 5.0e-3, 5.0e-3, 5.0e-3]
    backend: proxqp
    eps_abs: 1.0e-6
    max_iter: 400             # realtime ProxQP cap (long solves freeze MoveJ)
    max_iter_cap: 400
    max_solve_ms: 8.0         # skip retry if first attempt already burned wall budget
    fail_qdot_decay: 0.85
    twist_sigma_floor: 0.08   # scale Cartesian/force twist when σ < sigma_ref
    # Avoidance onset vs sigma_ref (escape must lead the twist brake).
    sigma_escape_ref_scale: 1.25
    warn_on_fail: false
    # Chiaverini 1997 SR damping for nullspace projection.
    sr_damping:
      lam0: 0.05
      sigma_ref: 0.08
      sigma_floor: 1.0e-6
    # Soften W_task toward task_weight_min_frac as σ_min drops (LPF-smoothed).
    task_weight_min_frac: 0.05
    task_weight_lpf_tau_s: 0.25
    # Mass-weighted reg: reg[i] *= max(diag(M)[i], mass_reg_floor).
    use_mass_weighted_reg: true
    mass_reg_floor: 0.05
    # Rail exempt: otherwise diag(M)[0]≈carriage+arm over-prices rail motion.
    mass_weight_exempt_rail: true
    # LPF on mass-weighted reg diagonal (0 disables).
    mass_reg_lpf_tau_s: 0.2
    # Prefer kinematic nullspace (dyn N_dyn is oblique).
    use_dyn_nullspace: false
    # Faverjon/Tournassoud limit damper band (arm rad, rail m — units differ).
    limit_damper_band_rad: 0.15      # arm joints 1..7 (rad)
    limit_damper_band_rail_m: 0.02   # rail joint 0 (m)

  collision:
    enabled: true          # CBF self-collision (low-poly STL)
    d_safe: 0.01
    d_activate: 0.04
    gamma: 5.0
    max_pairs: 8

  nullspace:
    k_center: 1.5
    k_limit: 2.0
    activation: 0.85
    # Index 5 = J5: resist flip when escape-zone ∇μ is also active.
    # Index 6 = J6: prefer nominal 45° (wrist away from |J6|≈0).
    weights: [0.0, 1.0, 1.0, 1.0, 1.0, 2.5, 2.5, 1.0]
    # Comfortable RM75 posture attractor (independent of taught D).
    q_nominal_deg: [0.0, 0.0, -45.0, 0.0, 90.0, 0.0, 45.0, 0.0]
    # Move-phase nullspace: ascend Yoshikawa μ (see manipulability_task.py).
    manipulability:
      k_mu: 0.8
      eps_rad: 5.0e-4
      sigma_fade_ref: 0.12

  # Viscous damping on composed secondary qdot (1/s); scales up near limits.
  nullspace_d_null: 0.5
  nullspace_d_null_adaptive: 1.0
  # Cap soft secondary qdot as a fraction of URDF v_max (near σ, N→I).
  nullspace_max_qdot_frac: 0.2
  # Keep recovering the active posture after singularity escape.
  centering_recovery_gain: 3.0
  centering_recovery_max_qdot_frac: 0.35
  centering_recovery_tol: 0.12

  arm_angle:
    enabled: true
    k_psi: 1.0
    psi_ref_deg: null      # null -> capture at reset / set by the app after IK

  # Preferred-extension rail coordination (Yamamoto & Yun 1994), COUPLED only:
  # (1) velocity-gated FF, (2) extension-gated reach. Idle → rail stays put.
  rail_extension:
    enabled: true
    k_ext: 2.0             # reach push-back gain (1/s per m of extension error)
    k_ff: 1.0              # 100% vel_ff → rail column projection
    v_ff_thr_m_s: 0.005    # FF silent below 5 mm/s (micro-adjust / jitter)
    v_ff_span_m_s: 0.015   # FF fully on by ~20 mm/s (2 cm/s scan is "awake")
    e0_m: 0.02             # reach dead zone: arm handles ±2 cm inside d_pref
    e1_m: 0.08             # full reach authority by 8 cm drift
    w_max: 2.0             # QP weight cap (≪ W_task=100)
    v_max_m_s: 0.08        # cap on the task's desired rail velocity
    limit_margin_m: 0.08   # C¹ smoothstep handoff band before physical stop (8 cm)
    # σ-escape: keep w_max*(1+k_sigma_boost)=6 ≪ W_task=100 (slack > rail > arm).
    k_sigma_boost: 2.0     # w_ext boosts up to 3x as σ → 0
    k_esc: 0.5             # σ-escape velocity gain (m/s per unit σ gradient)
    w_sigma_floor: 1.0     # baseline w inside dead zone when σ depressed
    # move→D pose attractor (preset=move → mode=pose_attract).
    k_pose: 2.0            # 1/s soft P on (y_target - y_rail)
    pose_e0_m: 0.005       # settle dead-zone (stop hunting at target)
    pose_e1_m: 0.04        # full pose-attract weight by 4 cm
    pose_w_max: 4.0        # ≪ W_task=100
    sigma_guard_enter: 0.45
    sigma_guard_exit: 0.70
    v_guard_max_m_s: 0.04  # guardrail cannot yank rail off the pose path
    v_lpf_tau_s: 0.12      # macro-micro LPF on desired rail velocity

  rail:
    # mode: coupled | locked; locked_style: hold | rail_only | tcp_fixed.
    mode: coupled              # scan: rail joins QP (set locked for pin-only scan)
    locked_style: hold
    q_ref_m: 0.0
    # HOLD-only lock knobs:
    lock_gain: 200.0
    lock_reg_scale: 100.0     # tempered from 500 -> 100 (HOLD still very rigid)
    lock_vel_eps_m_s: 0.0
    lock_hard_pin: true
    # Geometry / limits (also mirrored in URDF; keep in sync):
    # rail_y = 0 at -Y end stop, rail_y = travel_m at +Y end (0..800 mm).
    v_max_m_s: 0.20           # 20 cm/s (matches motor + URDF velocity limit)
    travel_m: 0.80            # [0, 0.80] m

frames:
  # Prefer inner.control_frame / inner.euler_order; this block only supplies
  # euler_order fallback for older loaders.
  euler_order: xyz

force:
  desired_z_n: 1.0
  phi_source: phi_recommended
  fc_hz: 6.0
  min_samples: 22
  causal_fc_hz: 6.0
  causal_order: 2
  causal_history: 5
  # Inertia compensation off on the joint stream (re-enable only with telemetry).
  use_inertia: false

hybrid_motion:
  # Single force controller (2965fea): tool-frame masks, no runtime mode switch.
  force_axes: [0, 0, 1, 0, 0, 0]
  track_axes: [1, 1, 0, 1, 1, 1]
  # Tool-frame PBAC: v = vel_ff + kp*err (De Schutter 1988); X==Y tangential.
  kp_pos: [2.0, 2.0, 0.0, 1.5, 1.5, 1.5]
  pos_err_deadband_m: 0.0005
  pos_correction_max_m_s: 0.08
  system_delay_s: 0.015
  # fz-only enter-only contact latch: lateral shear must not flip contact.
  contact_threshold_n: 0.8
  contact_use_fz_only: true
  # Fixed sensor-noise deadband (never scale with desired_z).
  deadband_n: 0.10
  deadband_width_n: 0.10
  # Caps: tangential X==Y; tool-Z matches max_vz_tool_m_s (single vz authority).
  max_velocity: [0.22, 0.22, 0.10, 0.6, 0.6, 0.6]
  max_acceleration: [1.0, 1.0, 0.8, 2.0, 2.0, 2.0]
  admittance_mass_z: 1.5
  admittance_damping_z: 25.0   # fixed normal damping in and out of contact
  # One press/retract cap; bounce handled by Dimeas variable inertia.
  max_vz_tool_m_s: 0.10
  # Ramp desired_z from ~contact threshold over this many seconds of latch.
  desired_force_ramp_s: 1.0
  # Dimeas variable-INERTIA on tool-Z (inflate M with Iₛ, not D).
  var_damping_enabled: true
  # ~3.5 Hz splits hand guidance (<2 Hz) from contact resonance (~4.9 Hz).
  var_damping_omega_c_hz: 3.5
  # Iₛ EWMA ≈ paper τ~0.1 s at 200 Hz (λ=exp(-dt/0.1)).
  var_damping_lambda: 0.951
  var_damping_f_max_n: 7.0
  var_damping_d_u: 0.0           # do not inflate damping with Iₛ (Dimeas)
  var_damping_m_u: 3.0           # M = mass_z + m_u·Iₛ (hard-contact anti-bounce)
  # Hardware safety cap on virtual-mass authority.
  var_damping_m_max: 5.0
  var_damping_dc_alpha: 0.02
  adaptive_ke:
    # Off: K̂_e unreliable on moving tissue. Keys kept for A/B re-enable.
    enabled: false
    zeta: 0.9
    ke_initial: 80.0
    ke_min: 40.0
    ke_max: 2500.0
    ke_impact_initial: 1500.0
    ke_forgetting: 0.995
    ke_forgetting_inc: 0.88
    ke_idle_decay_s: 2.0
    ke_soft_floor: 300.0
    ke_detach_decay_s: 1.0
    displacement_source: admittance
    dx_threshold_m: 0.00008
    contact_force_n: 0.8
    settle_ticks: 10
    gate_lateral_velocity: true
    lateral_vel_gate_m_s: 0.02
    gate_df_spike: true
    df_spike_n: 4.0
    f_err_gate_n: 1.2
    f_err_gate_frac: 0.35
    bd_min: 25.0
    bd_max: 200.0
    bd_slew_max: 400.0
    ke_slew_max: 1200.0
  # Energy-aware leaky force reference: Dimeas gates press only; retract stays open.
  proactive_feedforward: true
  proactive_retract_only: false
  proactive_gain: 0.10
  proactive_retract_gain: 0.10
  proactive_leak_s: 0.3
  v_r_max_m_s: 0.06
  # Fade press authority for Is∈[start, gate]; Dimeas M(t) + fixed D stay active.
  proactive_press_is_gate_start: 0.20
  proactive_press_is_gate: 0.60
  proactive_press_drive_max: 1.0
  proactive_retract_drive_max: 1.0
  proactive_reset_on_reversal: true
  # Same normalized small-error law across setpoints (deadband rejects noise first).
  force_scale_min_n: 0.20
  force_scale_fraction: 0.15

# LW100 rail servo (Modbus RTU over USR-TCP232).
# Soft CSP: WBC q_cmd[0] → soft PD → FA24; encoder feeds WBC q_meas[0].
# Soft faults → HOLD; hard PANIC only on garbage encoder. OOB targets rejected.
# Workflow: manually seat carriage at -Y, zero_mode=current, home_on_exit=false.
hw:
  lw100:
    enabled: true
    host: 192.168.0.7
    port: 8234
    slave: 1
    lead_mm: 10.0
    zero_mode: current
    counts0: 0
    sign: 1
    enable_settle_s: 0.3
    # Cold start: prove worker Modbus read+FA24=0 before any set_target / move→D.
    arm_good_reads: 30          # ~0.6 s @ 50 Hz consecutive healthy polls
    arm_settle_s: 0.8           # extra FA24=0 hold after good polls
    arm_max_span_mm: 2.0
    arm_timeout_s: 10.0
    fault_margin_m: 0.05
    poll_hz: 50
    inter_frame_delay_s: 0.0005
    timeout_s: 0.06             # poll-budget; was 0.15 / class-default 1.0
    retries: 1
    deadband_mm: 0.5
    max_speed_rpm: 900       # FA23: 0.15 m/s @ 10 mm/rev (gentler vs Er-01 on move→D)
    # Soft CSP via FA24 (see apps/lw100_vel_pos_follow_demo.py).
    vel_kp: 18.0
    vel_kd: 0.22
    vel_ff_gain: 1.0
    vel_max_m_s: 0.15
    vel_amax_m_s2: 0.8
    vel_deadband_mm: 0.02
    target_timeout_s: 0.25
    encoder_freeze_s: 1.0
    encoder_freeze_min_v_m_s: 0.02
    encoder_freeze_min_move_mm: 0.5
    accel_ms: 200            # FA40 — manual: Er-01 if accel too short at start
    decel_ms: 200            # FA41
    scurve_ms: 30            # FA42
    busy_speed_rpm: 1
    home_on_exit: false
    home_speed_rpm: 900
    home_approach_mm: 40
    home_timeout_s: 60
    verbose: false

startup:
  # Used by window A / C bring-up (not 6-DOF pose_slot).
  enable_force: false
  follow: true
  move_speed: 20
  realtime: false
  watchdog_timeout_s: 0.1
```

## `rm75_control/control/admittance_common/controller.py`

```python
"""Tool-frame force/motion decoupling (2965fea): PBAC track + admittance on force axes.

Tool-Z: M(t)·v̇ + D(t)·(v − v_r) = F_des − F_ext. Dimeas gates press only.
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
        self.contact_present = False
        self.time_scale = 1.0
        self.v_force_z = 0.0
        self.v_r_z = 0.0
        self._proactive_ff = ProactiveForceIntegrator(self.cfg.proactive_ff)
        self.force_reference_scale_n = float("nan")
        self.force_reference_drive = 0.0
        self.force_reference_gate_scale = 1.0
        self.force_reference_accel_m_s2 = 0.0
        self.force_reference_reversal_reset = False
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
        self.contact_present = False
        self.v_force_z = 0.0
        self.v_r_z = 0.0
        self._proactive_ff.reset()
        self.force_reference_scale_n = float("nan")
        self.force_reference_drive = 0.0
        self.force_reference_gate_scale = 1.0
        self.force_reference_accel_m_s2 = 0.0
        self.force_reference_reversal_reset = False
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
        if self._in_contact_latched:
            return True
        if self._contact_signal_n(f_ext) >= float(
            self.cfg.contact_threshold_n
        ):
            self._in_contact_latched = True
        return self._in_contact_latched

    def _update_proactive_v_r(
        self,
        eff: float,
        in_contact: bool,
        dt_eff: float,
        *,
        rising_edge: bool,
        desired_force_n: float = 0.0,
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
                instability_index=self.instability_index,
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
    ) -> float:
        cfg = self.cfg
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
        if dt_eff > 0.0:
            tau_d = 0.025 if self.instability_index > 0.5 else 0.10
            blend = min(1.0, dt_eff / tau_d)
            self._d_z_smooth += blend * (
                damping_target - self._d_z_smooth
            )
        else:
            self._d_z_smooth = damping_target
        damping = self._d_z_smooth
        self.damping_ke_z = damping_ke
        self.damping_dimeas_z = damping_dimeas
        self.damping_z_eff = float(damping)

        v_z_cap = self._v_z_cap()
        v_reference = self._update_proactive_v_r(
            eff,
            in_contact,
            dt_eff,
            rising_edge=rising_edge,
            desired_force_n=desired_force_n,
        )
        velocity = self.v_force_z + (dt_eff / mass_z) * (
            eff - damping * (self.v_force_z - v_reference)
        )
        if v_z_cap > 0.0:
            velocity = float(
                np.clip(velocity, -v_z_cap, v_z_cap)
            )
        self.v_force_z = velocity
        return velocity


HybridMotionConfig = AdmittanceConfig
HybridMotionController = AdmittanceController
```

## `rm75_control/control/admittance_common/force_barrier.py`

```python
"""Force-space velocity damper for tool-Z press and retract motion.

The damper predicts near-future force from a filtered force derivative and
limits normal velocity before the delayed admittance loop can build a large
over-force transient.  It deliberately does not depend on the environment
stiffness estimate, which is least reliable at first impact.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ForceBarrierConfig:
    enabled: bool = True
    t_react_s: float = 0.030
    budget_min_n: float = 1.0
    budget_frac: float = 0.20
    f_keep_n: float = 0.5
    v_ref_m_s: float = 0.05
    v_min_retract_m_s: float = 0.002
    fdot_lpf_s: float = 0.040

    @classmethod
    def from_dict(cls, raw: dict) -> "ForceBarrierConfig":
        barrier = raw.get("force_barrier", raw)
        if not isinstance(barrier, dict):
            barrier = {}
        return cls(
            enabled=bool(barrier.get("enabled", True)),
            t_react_s=float(barrier.get("t_react_s", 0.030)),
            budget_min_n=float(barrier.get("budget_min_n", 1.0)),
            budget_frac=float(barrier.get("budget_frac", 0.20)),
            f_keep_n=float(barrier.get("f_keep_n", 0.5)),
            v_ref_m_s=float(barrier.get("v_ref_m_s", 0.05)),
            v_min_retract_m_s=float(barrier.get("v_min_retract_m_s", 0.002)),
            fdot_lpf_s=float(barrier.get("fdot_lpf_s", 0.040)),
        )


class ForceSpaceVelocityDamper:
    def __init__(self, cfg: ForceBarrierConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.f_dot_z = 0.0
        self._f_prev: float | None = None
        self.cap_press_z = 0.0
        self.cap_retract_z = 0.0
        self.f_pred_z = 0.0

    def update_fdot(self, f_z: float, dt_eff: float) -> float:
        if dt_eff <= 0.0:
            return self.f_dot_z
        if self._f_prev is None:
            self._f_prev = float(f_z)
            self.f_dot_z = 0.0
            return self.f_dot_z
        raw = (float(f_z) - self._f_prev) / dt_eff
        self._f_prev = float(f_z)
        tau = max(float(self.cfg.fdot_lpf_s), 1e-6)
        alpha = min(1.0, dt_eff / tau)
        self.f_dot_z += alpha * (raw - self.f_dot_z)
        return self.f_dot_z

    def caps(
        self,
        *,
        f_z: float,
        f_des_z: float,
        in_contact: bool,
        v_z_cap: float,
        seek_vz_m_s: float,
        contact_enter_n: float,
    ) -> tuple[float, float]:
        cfg = self.cfg
        v_hi = max(float(v_z_cap), 0.0)
        if not cfg.enabled:
            self.cap_press_z = v_hi
            self.cap_retract_z = v_hi
            self.f_pred_z = float(f_z)
            return self.cap_press_z, self.cap_retract_z

        if not in_contact:
            seek = max(float(seek_vz_m_s), 0.0)
            if v_hi > 0.0:
                seek = min(seek, v_hi) if seek > 0.0 else v_hi
            del contact_enter_n
            self.cap_press_z = seek if seek > 0.0 else v_hi
            self.cap_retract_z = v_hi
            self.f_pred_z = float(f_z)
            return self.cap_press_z, self.cap_retract_z

        if abs(float(f_des_z)) < 1e-6:
            self.cap_press_z = v_hi
            self.cap_retract_z = v_hi
            self.f_pred_z = float(f_z)
            return self.cap_press_z, self.cap_retract_z

        budget = max(
            float(cfg.budget_min_n),
            float(cfg.budget_frac) * abs(float(f_des_z)),
            1e-6,
        )
        f_pred = float(f_z) + self.f_dot_z * max(float(cfg.t_react_s), 0.0)
        self.f_pred_z = f_pred
        v_ref = max(float(cfg.v_ref_m_s), 0.0)

        cap_press = max(
            0.0,
            ((float(f_des_z) + budget) - f_pred) / budget * v_ref,
        )
        if v_hi > 0.0:
            cap_press = min(cap_press, v_hi)

        cap_retract = max(
            float(cfg.v_min_retract_m_s),
            (f_pred - float(cfg.f_keep_n)) / budget * v_ref,
        )
        if v_hi > 0.0:
            cap_retract = min(cap_retract, v_hi)

        self.cap_press_z = float(cap_press)
        self.cap_retract_z = float(cap_retract)
        return self.cap_press_z, self.cap_retract_z

    def clamp_eff(self, eff: float, damping: float) -> float:
        damping = max(float(damping), 1e-6)
        return float(
            min(
                max(float(eff), -damping * self.cap_retract_z),
                damping * self.cap_press_z,
            )
        )

    def clamp_velocity(self, velocity: float) -> float:
        if velocity >= 0.0:
            return float(min(velocity, self.cap_press_z))
        return float(max(velocity, -self.cap_retract_z))
```

## `rm75_control/control/admittance_common/proactive_force_ff.py`

```python
"""Energy-aware leaky force-error reference for the tool-Z ``v_r`` slot.

This is an engineering complement to the 2nd-order admittance loop:

    M · v̇ + D · (v − v_r) = F_err

It is **not** the human-input observer or Eq. (23)/(35) controller from
Li et al. (2022): it has no human dynamics model or observer-error dynamics.
It keeps the hardware-tested 0.3 s short-memory structure and a
setpoint-normalized drive.  The two signs have the same small-error gain, but
their safety treatment follows contact power:

* ``eff > 0`` presses farther into the surface and can inject contact energy,
  so Dimeas attenuates this branch as high-frequency instability rises;
* ``eff < 0`` releases an over-force contact, so Dimeas must not suppress the
  escape direction.  Its drive is still bounded, and the virtual
  mass/critical damping remain active in the passive admittance layer.

Bidirectional integration (``retract_only=False``) gives the "error-large →
proactive chase" hand feel on both press and retract.  Its guards are:

* leaky decay toward zero (``leak_s``);
* |v_r| ≤ ``v_r_max_m_s`` (< unified tool-Z cap — leaves headroom for D·v);
* only energy-injecting press fades as Dimeas Iₛ → ``press_is_gate``;
* bounded normalized drive on both signs;
* same-contact error reversal projects away an old, opposing ``v_r``;
* Åström anti-windup at both the reference and force-velocity caps;
* the caller clears either sign on contact re-acquire.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProactiveFfConfig:
    enabled: bool = True
    retract_only: bool = False
    # Small-error normalized gains [m/s²].  They default equal; the
    # directional difference comes from the press-only energy gate and the
    # over-force branch not being closed by the instability gate.
    gain: float = 0.10
    retract_gain: float = 0.10
    leak_s: float = 0.3         # leak time constant [s]
    v_r_max_m_s: float = 0.06
    # Energy-injecting press stays fully available below ``gate_start``, then
    # fades linearly to zero at ``press_is_gate``.  Retraction is an
    # over-force escape and is deliberately not gated.
    press_is_gate_start: float = 0.0
    press_is_gate: float = 0.5
    force_scale_min_n: float = 0.30
    force_scale_fraction: float = 0.15
    press_drive_max: float = 1.0
    retract_drive_max: float = 1.0
    reset_on_reversal: bool = True

    @classmethod
    def from_dict(cls, raw: dict) -> ProactiveFfConfig:
        p = raw.get("proactive_ff", raw)
        if not isinstance(p, dict):
            p = raw
        gain = float(p.get("gain", p.get("proactive_gain", 0.10)))
        return cls(
            enabled=bool(p.get("enabled", p.get("proactive_feedforward", True))),
            retract_only=bool(p.get("retract_only", p.get("proactive_retract_only", False))),
            gain=gain,
            retract_gain=float(
                p.get(
                    "retract_gain",
                    p.get("proactive_retract_gain", gain),
                )
            ),
            leak_s=float(p.get("leak_s", p.get("proactive_leak_s", 0.3))),
            v_r_max_m_s=float(p.get("v_r_max_m_s", 0.06)),
            press_is_gate_start=float(
                p.get(
                    "press_is_gate_start",
                    p.get("proactive_press_is_gate_start", 0.0),
                )
            ),
            press_is_gate=float(p.get("press_is_gate", p.get("proactive_press_is_gate", 0.5))),
            force_scale_min_n=float(p.get("force_scale_min_n", 0.30)),
            force_scale_fraction=float(p.get("force_scale_fraction", 0.15)),
            press_drive_max=float(
                p.get(
                    "press_drive_max",
                    p.get("proactive_press_drive_max", 1.0),
                )
            ),
            retract_drive_max=float(
                p.get(
                    "retract_drive_max",
                    p.get("proactive_retract_drive_max", 1.0),
                )
            ),
            reset_on_reversal=bool(
                p.get(
                    "reset_on_reversal",
                    p.get("proactive_reset_on_reversal", True),
                )
            ),
        )


class ProactiveForceIntegrator:
    """Leaky normalized reference integrator with contact-power guards."""

    def __init__(self, cfg: ProactiveFfConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.v_r = 0.0
        self.last_force_scale_n = float("nan")
        self.last_drive = 0.0
        self.last_instability_scale = 1.0
        self.last_reference_accel_m_s2 = 0.0
        self.last_reversal_reset = False

    def update(
        self,
        eff: float,
        *,
        in_contact: bool,
        dt_eff: float,
        instability_index: float,
        v_force_z: float,
        v_z_cap: float,
        desired_force_n: float = 0.0,
    ) -> float:
        cfg = self.cfg
        if not cfg.enabled:
            self.v_r = 0.0
            self.last_drive = 0.0
            self.last_instability_scale = 1.0
            self.last_reference_accel_m_s2 = 0.0
            self.last_reversal_reset = False
            return 0.0
        if dt_eff <= 0.0:
            return self.v_r

        force_scale = max(
            cfg.force_scale_min_n,
            cfg.force_scale_fraction * abs(float(desired_force_n)),
            1e-6,
        )
        drive_unclamped = float(eff) / force_scale
        if eff < 0.0:
            drive = float(
                np.clip(
                    drive_unclamped,
                    -max(cfg.retract_drive_max, 0.0),
                    0.0,
                )
            )
        else:
            drive = float(
                np.clip(
                    drive_unclamped,
                    0.0,
                    max(cfg.press_drive_max, 0.0),
                )
            )
        self.last_force_scale_n = force_scale
        self.last_drive = drive
        self.last_instability_scale = 1.0
        self.last_reference_accel_m_s2 = 0.0
        self.last_reversal_reset = False

        has_effective_error = in_contact and abs(eff) > 1e-12
        integrate = has_effective_error
        if integrate and cfg.retract_only and eff > 0.0:
            integrate = False

        # Do not let the previous direction spend 0.2--0.5 s fighting a new
        # force error.  The passive admittance velocity is intentionally not
        # reset; M and D still make the actual TCP-Z reversal continuous.
        if (
            has_effective_error
            and cfg.reset_on_reversal
            and self.v_r * float(eff) < 0.0
        ):
            self.v_r = 0.0
            self.last_reversal_reset = True

        if cfg.leak_s > 1e-6:
            self.v_r -= (dt_eff / cfg.leak_s) * self.v_r

        if integrate:
            if eff < 0.0:
                # Over-force retraction releases contact energy.  Never let an
                # instability detector close the escape route.
                step = cfg.retract_gain * drive
            else:
                step = cfg.gain * drive
            if step > 0.0 and cfg.press_is_gate > 1e-9:
                gate_stop = max(float(cfg.press_is_gate), 1e-9)
                gate_start = float(
                    np.clip(cfg.press_is_gate_start, 0.0, gate_stop)
                )
                if instability_index <= gate_start:
                    self.last_instability_scale = 1.0
                elif gate_stop <= gate_start + 1e-9:
                    self.last_instability_scale = 0.0
                else:
                    self.last_instability_scale = float(
                        np.clip(
                            1.0
                            - (instability_index - gate_start)
                            / (gate_stop - gate_start),
                            0.0,
                            1.0,
                        )
                    )
                step *= self.last_instability_scale

            # Conditional integration at both saturation layers.  Motion back
            # toward the admissible set is always allowed.
            v_r_cap = max(float(cfg.v_r_max_m_s), 0.0)
            at_negative_cap = (
                (v_z_cap > 0.0 and v_force_z <= -v_z_cap + 1e-6)
                or (v_r_cap > 0.0 and self.v_r <= -v_r_cap + 1e-6)
            )
            at_positive_cap = (
                (v_z_cap > 0.0 and v_force_z >= v_z_cap - 1e-6)
                or (v_r_cap > 0.0 and self.v_r >= v_r_cap - 1e-6)
            )
            if (step < 0.0 and at_negative_cap) or (
                step > 0.0 and at_positive_cap
            ):
                step = 0.0
            self.last_reference_accel_m_s2 = float(step)
            self.v_r += dt_eff * step

        if cfg.v_r_max_m_s > 0.0:
            self.v_r = float(np.clip(self.v_r, -cfg.v_r_max_m_s, cfg.v_r_max_m_s))
        if v_z_cap > 0.0:
            self.v_r = float(np.clip(self.v_r, -v_z_cap, v_z_cap))
        return self.v_r
```

## `rm75_control/control/admittance_common/adaptive_ke.py`

```python
"""Online environment stiffness estimation + critical-damping admittance.

Coupled contact model on the normal (tool-Z) admittance axis:

    m_d · ẍ + b_d · ẋ + K_e · x = F_ext

Damping ratio ζ = b_d / (2√(m_d K_e)). Holding ζ fixed while K_e changes
requires (Keemink et al. 2018 §III.C):

    b_d(t) = 2 ζ √(m_d · K̂_e(t))

Learning rule (Duan, Gan, Chen & Dai, RAS 102 (2018) eq. 14, asymmetric
EWMA on |ΔF/Δx| — the 27c1689 shape that hardware confirmed keeps hard
surfaces stable):

    if in_contact and gates_pass and |Δx| >= dx_threshold:
        ke_inst = |ΔF/Δx|
        λ       = ke_forgetting_inc  if ke_inst > K̂_e   (fast track up)
                  ke_forgetting      otherwise           (slow forget down)
        K̂_e   ← λ · K̂_e + (1 − λ) · ke_inst

Stiff-first impact initialisation: on a contact rising edge we jump K̂_e up to
``ke_impact_initial`` (b_d follows immediately, no slew). Underdamped first
few ticks on a hard surface is exactly what starts a bounce cascade; jumping
to overdamped and then learning DOWN on soft surfaces avoids that.

Idle / detach soft decay: neither hold-last (previous refactor: K̂_e climbs
monotonically) nor hard reset (older refactor: b_d drops to ~16 N·s/m on
every re-impact and re-starts the bounce) is safe. Both branches decay
K̂_e toward ``ke_initial`` with a time constant (τ_idle in steady contact
with small |f_err|, τ_detach out of contact). A 50 ms bounce flight keeps
almost all of the stiffness the estimator just learned about the surface
it will re-hit; a long steady press on soft tissue eventually relaxes
K̂_e so b_d drops back and the press regains bandwidth to chase a receding
surface.

Direction-agnostic tangential gate: ``gate_lateral_velocity`` acts on the
**magnitude** of the tangential (tool-XY) commanded velocity — any spatial
trajectory (Y sweep, X, arc, spline, teleop) gates learning the same way.

``reset()`` seeds K̂_e = ke_initial (new-session semantics only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rsc


@dataclass
class AdaptiveKeConfig:
    enabled: bool = False
    zeta: float = 1.0
    ke_initial: float = 80.0
    # Asymmetric EWMA: fast track up when the surface reads stiffer than we
    # believe (impact-safe), slow forget down when it reads softer (avoid
    # over-reacting to a single quiet tick). Two λ's, not one.
    ke_forgetting: float = 0.995      # slow forget (surface softens)
    ke_forgetting_inc: float = 0.88   # fast track  (surface stiffens)
    ke_min: float = 40.0
    ke_max: float = 2500.0
    dx_threshold_m: float = 8e-5
    contact_force_n: float = 0.8
    # Stiff-first impact initialisation. On a contact rising edge K̂_e jumps
    # UP to this value (b_d follows, no slew). Underdamped-at-impact starts
    # bounce cascades; overdamped-at-impact is safe and learns down on soft
    # surfaces. 0 disables. 27c1689: 1500.
    ke_impact_initial: float = 1500.0
    # Soft-decay time constants toward ke_initial (see module docstring).
    # ke_detach_decay_s: out of contact. 1.0 s keeps ~95 % of learned K̂_e
    # through a 50 ms bounce flight and returns to seed over ~5 s.
    ke_detach_decay_s: float = 1.0
    # ke_idle_decay_s: in steady contact with no learning update AND
    # |f_err|_envelope inside the gate (steady tracking, not over-force).
    # 2.0 s keeps enough stiffness for chase while letting soft tissue relax.
    ke_idle_decay_s: float = 2.0
    # Soft-tissue idle-decay floor: decay target is max(ke_initial, ke_soft_floor)
    # instead of ke_initial alone. Impact stiff-first (ke_impact_initial) is
    # unchanged; only the downward chase decay is prevented from reaching the
    # ~16 N·s/m underdamped band (Ke=80) on a compliant surface — Phase B1.
    # 0 disables (legacy: decay all the way to ke_initial).
    ke_soft_floor: float = 300.0
    bd_max: float = 200.0
    bd_min: float = 25.0
    bd_slew_max: float = 400.0
    ke_slew_max: float = 1200.0
    displacement_source: str = "admittance"
    # Trajectory-agnostic tangential-speed gate (magnitude of tool-XY vel).
    gate_lateral_velocity: bool = True
    lateral_vel_gate_m_s: float = 0.02
    # |ΔF| spike gate: a single-tick jump above df_spike_n N is likely a
    # sensor spike or geometric coupling, not a real stiffness sample.
    gate_df_spike: bool = True
    df_spike_n: float = 4.0
    # |f_err| gate: during an over-force transient the instantaneous
    # ΔF/Δx is dominated by the loop response, not the environment.
    # Effective gate = max(f_err_gate_n, f_err_gate_frac * |f_des_z|):
    # f_err_gate_n is the small-setpoint noise floor; the relative term keeps
    # the "steady vs transient" judgement self-similar at any setpoint. A
    # fixed 1.2 N gate at a 5 N hold froze K̂_e at ke_impact_initial forever
    # (normal hand-interaction ripple > 1.2 N) — b_d stayed ~70+ N·s/m and
    # the retract felt heavily damped.
    f_err_gate_n: float = 1.2
    f_err_gate_frac: float = 0.35
    # Hold K̂_e (no learning) this many ticks after contact acquisition so
    # the first-impact transient doesn't dominate the estimator.
    settle_ticks: int = 10

    @classmethod
    def from_dict(cls, raw: dict, parent: dict) -> AdaptiveKeConfig:
        a = raw.get("adaptive_ke", parent.get("adaptive_ke", {}))
        if not isinstance(a, dict):
            a = {}
        return cls(
            enabled=bool(a.get("enabled", parent.get("adaptive_ke_enabled", False))),
            zeta=float(a.get("zeta", parent.get("adaptive_zeta", 1.0))),
            ke_initial=float(a.get("ke_initial", parent.get("ke_initial", 80.0))),
            ke_forgetting=float(a.get("ke_forgetting", parent.get("ke_forgetting", 0.995))),
            ke_forgetting_inc=float(
                a.get("ke_forgetting_inc", parent.get("ke_forgetting_inc", 0.88))
            ),
            ke_min=float(a.get("ke_min", parent.get("ke_min", 40.0))),
            ke_max=float(a.get("ke_max", parent.get("ke_max", 2500.0))),
            dx_threshold_m=float(a.get("dx_threshold_m", parent.get("ke_dx_threshold_m", 8e-5))),
            contact_force_n=float(
                a.get("contact_force_n", parent.get("adaptive_contact_force_n", 0.8))
            ),
            ke_impact_initial=float(a.get("ke_impact_initial", 1500.0)),
            ke_detach_decay_s=float(a.get("ke_detach_decay_s", 1.0)),
            ke_idle_decay_s=float(a.get("ke_idle_decay_s", 2.0)),
            ke_soft_floor=float(a.get("ke_soft_floor", 300.0)),
            bd_max=float(a.get("bd_max", parent.get("adaptive_bd_max", 200.0))),
            bd_min=float(a.get("bd_min", parent.get("adaptive_bd_min", 25.0))),
            bd_slew_max=float(a.get("bd_slew_max", parent.get("adaptive_bd_slew_max", 400.0))),
            ke_slew_max=float(a.get("ke_slew_max", parent.get("ke_slew_max", 1200.0))),
            displacement_source=str(
                a.get("displacement_source", parent.get("ke_displacement_source", "admittance"))
            ).lower(),
            gate_lateral_velocity=bool(a.get("gate_lateral_velocity", True)),
            lateral_vel_gate_m_s=float(
                a.get("lateral_vel_gate_m_s", a.get("scan_vel_gate_m_s", 0.02))
            ),
            gate_df_spike=bool(a.get("gate_df_spike", True)),
            df_spike_n=float(a.get("df_spike_n", 4.0)),
            f_err_gate_n=float(a.get("f_err_gate_n", 1.2)),
            f_err_gate_frac=float(a.get("f_err_gate_frac", 0.35)),
            settle_ticks=int(a.get("settle_ticks", 10)),
        )


class EnvironmentStiffnessEstimator:
    """Asymmetric-λ EWMA of |ΔF/Δx| on the normal admittance axis with
    stiff-first impact + soft idle/detach decays (see module docstring).

    Outputs (K̂_e, b_d) with b_d = 2ζ√(m_d K̂_e) (Keemink 2018 critical-damping),
    slewed at ``bd_slew_max`` per second so the send path never sees a step.
    """

    def __init__(self, cfg: AdaptiveKeConfig, *, dt: float, mass_z: float = 3.0) -> None:
        self.cfg = cfg
        self.dt = max(dt, 1e-6)
        self._mass_z = max(mass_z, 1e-3)
        self.ke_est = float(cfg.ke_initial)
        self.bd = self._critical_bd(self._mass_z)
        self._x_adm = 0.0
        self._last_f_z = 0.0
        self._last_x = 0.0
        self._have_prev = False
        self._contact_ref_pose: np.ndarray | None = None
        self._in_contact = False
        self._update_gated = False
        self._contact_ticks = 0
        self.last_dx_m = 0.0
        self.last_df_n = 0.0
        self.update_count = 0
        # |f_err| envelope (peak-hold with ~0.3 s release) gating the idle
        # decay: an oscillation crosses f_err=0 twice per cycle, so the
        # instantaneous |f_err| under-reports over-force by ~100 %.
        self._f_err_env = 0.0

    def reset(self) -> None:
        self.ke_est = float(self.cfg.ke_initial)
        self.bd = self._critical_bd(self._mass_z)
        self._x_adm = 0.0
        self._last_f_z = 0.0
        self._last_x = 0.0
        self._have_prev = False
        self._contact_ref_pose = None
        self._in_contact = False
        self._update_gated = False
        self._contact_ticks = 0
        self.last_dx_m = 0.0
        self.last_df_n = 0.0
        self.update_count = 0
        self._f_err_env = 0.0

    def _critical_bd(self, mass_z: float) -> float:
        ke = max(self.ke_est, self.cfg.ke_min)
        bd = 2.0 * self.cfg.zeta * math.sqrt(max(mass_z, 1e-3) * ke)
        lo = self.cfg.bd_min if self.cfg.bd_min > 0.0 else 0.0
        return float(np.clip(bd, lo, self.cfg.bd_max))

    def _slew_ke(self, ke_target: float) -> float:
        max_dke = self.cfg.ke_slew_max * self.dt
        delta = float(np.clip(ke_target - self.ke_est, -max_dke, max_dke))
        return self.ke_est + delta

    def _slew_damping(self, bd_target: float) -> float:
        max_dbd = self.cfg.bd_slew_max * self.dt
        delta = float(np.clip(bd_target - self.bd, -max_dbd, max_dbd))
        return self.bd + delta

    @staticmethod
    def tool_z_displacement_m(
        pose: np.ndarray,
        ref_pose: np.ndarray,
        *,
        euler_order: str = "xyz",
    ) -> float:
        pose = np.asarray(pose, dtype=float)
        ref = np.asarray(ref_pose, dtype=float)
        d_base = pose[:3] - ref[:3]
        r_mat = Rsc.from_euler(euler_order, pose[3:6], degrees=False).as_matrix()
        return float((r_mat.T @ d_base)[2])

    def _normal_displacement_m(
        self,
        pose: np.ndarray,
        *,
        v_force_z: float,
        euler_order: str = "xyz",
    ) -> float:
        if self.cfg.displacement_source == "pose" and self._contact_ref_pose is not None:
            return self.tool_z_displacement_m(pose, self._contact_ref_pose, euler_order=euler_order)
        self._x_adm += float(v_force_z) * self.dt
        return self._x_adm

    def _f_err_gate_eff_n(self, f_des_z: float) -> float:
        """Setpoint-relative |f_err| gate with a small-force noise floor.

        max(f_err_gate_n, f_err_gate_frac·|f_des_z|) keeps the "steady vs
        transient" judgement self-similar at any desired force instead of
        freezing K̂_e whenever the setpoint outgrows a fixed absolute gate.
        """
        cfg = self.cfg
        return max(float(cfg.f_err_gate_n), float(cfg.f_err_gate_frac) * abs(f_des_z))

    def _should_update_ke(
        self,
        f_ext_z: float,
        f_err_z: float,
        v_lateral_m_s: float,
        df: float,
        f_err_gate_n: float,
    ) -> bool:
        cfg = self.cfg
        if abs(f_ext_z) < cfg.contact_force_n:
            return False
        if abs(f_err_z) > f_err_gate_n:
            return False
        if cfg.gate_lateral_velocity and abs(v_lateral_m_s) > cfg.lateral_vel_gate_m_s:
            return False
        if cfg.gate_df_spike and abs(df) > cfg.df_spike_n:
            return False
        return True

    def update(
        self,
        f_ext_z: float,
        pose: np.ndarray,
        *,
        in_contact: bool,
        mass_z: float,
        v_force_z: float = 0.0,
        v_lateral_m_s: float = 0.0,
        f_err_z: float = 0.0,
        f_des_z: float = 0.0,
        instability_index: float = 0.0,
        euler_order: str = "xyz",
        allow_impact_init: bool = True,
        allow_idle_decay: bool = True,
    ) -> tuple[float, float]:
        """Return (ke_est, bd) after one tick.

        ``v_lateral_m_s`` is the magnitude (>=0) of the tangential (tool-XY)
        speed, direction-agnostic — see module docstring.
        ``f_des_z`` is the (ramped) tool-Z force setpoint; it sizes the
        relative |f_err| gate (see ``_f_err_gate_eff_n``).
        ``instability_index`` is the raw Dimeas Iₛ contact-resonance index.
        ``allow_idle_decay`` lets callers veto the quiet-contact decay when
        their physical-contact state is uncertain.
        ``allow_impact_init``: caller sets this False on a contact rising
        edge that follows only a brief flicker (turnaround dip), so the
        stiff-first K̂_e jump fires on genuine impacts only.
        """
        cfg = self.cfg
        self._mass_z = max(mass_z, 1e-3)
        if not cfg.enabled:
            return self.ke_est, self.bd

        # Peak-hold envelope of |f_err| (~0.3 s release).
        self._f_err_env = max(abs(f_err_z), self._f_err_env * (1.0 - self.dt / 0.3))

        # Contact rising edge: stiff-first init (safe overdamped at impact).
        if in_contact and not self._in_contact:
            self._contact_ref_pose = np.asarray(pose, dtype=float).copy()
            self._x_adm = 0.0
            self._have_prev = False
            self._contact_ticks = 0
            if (
                allow_impact_init
                and cfg.ke_impact_initial > 0.0
                and self.ke_est < cfg.ke_impact_initial
            ):
                self.ke_est = min(float(cfg.ke_impact_initial), cfg.ke_max)
                # b_d jumps with K̂_e immediately: an underdamped first few
                # ticks on a hard surface is what starts a bounce cascade.
                self.bd = self._critical_bd(self._mass_z)

        if not in_contact:
            self._in_contact = False
            self._contact_ref_pose = None
            self._x_adm = 0.0
            self._have_prev = False
            self._update_gated = False
            self._contact_ticks = 0
            tau = max(float(cfg.ke_detach_decay_s), 1e-3)
            self.ke_est += (self.dt / tau) * (float(cfg.ke_initial) - self.ke_est)
            self.ke_est = float(np.clip(self.ke_est, cfg.ke_min, cfg.ke_max))
            bd_target = self._critical_bd(mass_z)
            self.bd = self._slew_damping(bd_target)
            return self.ke_est, self.bd

        self._in_contact = True
        self._contact_ticks += 1
        if self._contact_ref_pose is None:
            self._contact_ref_pose = np.asarray(pose, dtype=float).copy()

        x = self._normal_displacement_m(pose, v_force_z=v_force_z, euler_order=euler_order)

        f_err_gate_n = self._f_err_gate_eff_n(f_des_z)

        gated = True
        learned = False
        if self._contact_ticks <= max(cfg.settle_ticks, 0):
            gated = True
        elif self._have_prev:
            df = f_ext_z - self._last_f_z
            dx = x - self._last_x
            self.last_df_n = float(df)
            self.last_dx_m = float(dx)
            gated = not self._should_update_ke(
                f_ext_z, f_err_z, v_lateral_m_s, df, f_err_gate_n
            )
            if not gated and abs(dx) >= cfg.dx_threshold_m:
                ke_inst = abs(df / dx)
                ke_inst = float(np.clip(ke_inst, cfg.ke_min, cfg.ke_max))
                lam = (
                    cfg.ke_forgetting_inc if ke_inst > self.ke_est else cfg.ke_forgetting
                )
                ke_target = lam * self.ke_est + (1.0 - lam) * ke_inst
                self.ke_est = self._slew_ke(ke_target)
                learned = True
                self.update_count += 1

        # Stiff-first closure (idle decay): steady tracking with no ΔF/Δx
        # update this tick lets the impact-initialised K̂_e relax toward
        # ke_initial so the press regains bandwidth to chase a receding
        # surface. The force-error envelope gates transient samples. Dimeas
        # independently raises virtual inertia; coupling its index into this
        # decay kept K_e and velocity-dependent damping high after hand pushes.
        if (
            not learned
            and allow_idle_decay
            and cfg.ke_idle_decay_s > 1e-6
            and self._contact_ticks > max(cfg.settle_ticks, 0)
            and self._f_err_env <= f_err_gate_n
        ):
            self.ke_est += (self.dt / cfg.ke_idle_decay_s) * (
                max(float(cfg.ke_initial), float(cfg.ke_soft_floor)) - self.ke_est
            )
            self.ke_est = float(np.clip(self.ke_est, cfg.ke_min, cfg.ke_max))

        self._update_gated = gated
        self._last_f_z = f_ext_z
        self._last_x = x
        self._have_prev = True

        bd_target = self._critical_bd(mass_z)
        self.bd = self._slew_damping(bd_target)
        return self.ke_est, self.bd

    @property
    def zeta_eff(self) -> float:
        denom = 2.0 * math.sqrt(max(self._mass_z, 1e-3) * max(self.ke_est, self.cfg.ke_min))
        if denom < 1e-9:
            return 0.0
        return self.bd / denom

    @property
    def update_gated(self) -> bool:
        return self._update_gated
```

## `rm75_control/control/admittance_common/observer.py`

```python
"""Compensated external wrench from rolling pose/force buffer + phi."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, lfilter, lfilter_zi

from rm75_control.force.compensation import regressor as fid
from rm75_control.force.compensation.paths import CONFIG_FORCE, PHI_JSON


@dataclass
class ForceObserverConfig:
    phi_path: Path = PHI_JSON
    phi_source: str = "phi_recommended"
    force_sensor: Path = CONFIG_FORCE
    fc_hz: float = 2.5
    buffer_s: float = 4.0
    min_samples: int = 35
    use_inertia: bool = False
    poll_hz: float = 100.0
    # Causal online estimator (Keemink 2018 G2: keep filter order low and the
    # cutoff high to avoid the phase lag that destabilises the marginally passive
    # virtual-inertia model). 10 Hz / order-2 ≈ half the 6 Hz group delay
    # (Keemink G2); keep below the virtual-mass passivity floor.
    causal_fc_hz: float = 10.0
    causal_order: int = 2
    causal_history: int = 5


@dataclass
class ForceSampleBuffer:
    max_len: int
    t: deque = field(default_factory=deque)
    pose: deque = field(default_factory=deque)
    force: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.t = deque(maxlen=self.max_len)
        self.pose = deque(maxlen=self.max_len)
        self.force = deque(maxlen=self.max_len)

    def append(self, t_s: float, pose6: np.ndarray, force6: np.ndarray) -> None:
        self.t.append(t_s)
        self.pose.append(np.asarray(pose6, dtype=float))
        self.force.append(np.asarray(force6, dtype=float))

    def __len__(self) -> int:
        return len(self.t)


class CompensatedForceObserver:
    def __init__(self, cfg: ForceObserverConfig) -> None:
        self._fid = fid
        self.cfg = cfg
        self.phi = self._load_phi(cfg.phi_path, cfg.phi_source)
        self.frame = fid.FrameConfig.from_yaml(cfg.force_sensor)
        max_len = max(cfg.min_samples + 5, int(cfg.buffer_s * cfg.poll_hz) + 5)
        self.buf = ForceSampleBuffer(max_len=max_len)

        # --- causal online estimator state (O(1) per tick) ---
        k = max(2, int(cfg.causal_history))
        self._pose_ring: deque = deque(maxlen=k)
        self._t_ring: deque = deque(maxlen=k)
        self._n_updates = 0
        fs = float(cfg.poll_hz)
        wn = min(float(cfg.causal_fc_hz) / (0.5 * fs), 0.99)
        self._lpf_b, self._lpf_a = butter(int(cfg.causal_order), wn, btype="low")
        self._lpf_zi_unit = lfilter_zi(self._lpf_b, self._lpf_a)  # (order,)
        self._lpf_zi: np.ndarray | None = None  # (order, 6), lazily warm-started
        self._f_ext_last = np.zeros(6, dtype=float)
        # Compensated but UNfiltered wrench from the latest update(): the
        # Dimeas instability index must see the 5.8-20 Hz band the 6 Hz
        # control LPF removes (feed this to the index, f_ext_filt to control).
        self.f_ext_raw_last = np.zeros(6, dtype=float)

    @staticmethod
    def _load_phi(path: Path, source: str) -> np.ndarray:
        data = json.loads(path.read_text())
        if source not in data:
            raise SystemExit(f"Key '{source}' not in {path}")
        return np.array([data[source][k] for k in fid.PHI_NAMES])

    def append(self, t_s: float, pose6: np.ndarray, force_raw: np.ndarray) -> None:
        self.buf.append(t_s, pose6, force_raw)

    def ready(self) -> bool:
        return len(self.buf) >= self.cfg.min_samples

    def latest_wrench(self) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Return (signed_filtered_raw, f_ext).

        Return (signed_filtered_raw, f_ext) in the link_7 / sensor frame.
        """
        if not self.ready():
            return None
        t = np.asarray(self.buf.t)
        pose = np.asarray(self.buf.pose)
        force = np.asarray(self.buf.force)
        W, Y = self._fid.build_dataset(
            pose, force, t, self.frame, fc=self.cfg.fc_hz, use_inertia=self.cfg.use_inertia
        )
        k = len(t) - 1
        sl = slice(6 * k, 6 * k + 6)
        raw_show = Y[sl].copy()
        f_ext = (Y[sl] - W[sl] @ self.phi).reshape(6)
        return raw_show, f_ext

    def update(
        self,
        t_s: float,
        regressor_pose: np.ndarray,
        force_raw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Causal link_7-frame external wrench (before ``wrench_link7_to_tcp``)."""
        self._pose_ring.append(np.asarray(regressor_pose, dtype=float).reshape(6).copy())
        self._t_ring.append(float(t_s))
        self._n_updates += 1

        poses = np.asarray(self._pose_ring, dtype=float)
        times = np.asarray(self._t_ring, dtype=float)
        W_row, _g_s = self._fid.regressor_row_causal(
            poses, times, self.frame, use_inertia=self.cfg.use_inertia
        )

        signed = self._fid.apply_sign(
            np.asarray(force_raw, dtype=float), self.frame.force_sign
        )
        f_ext_raw = signed - W_row @ self.phi  # (6,)
        self.f_ext_raw_last = f_ext_raw.copy()

        if self._lpf_zi is None:
            # Warm-start each channel at its first value → no startup transient.
            self._lpf_zi = np.outer(self._lpf_zi_unit, f_ext_raw)
        f_ext_filt, self._lpf_zi = lfilter(
            self._lpf_b, self._lpf_a, f_ext_raw[None, :], axis=0, zi=self._lpf_zi
        )
        f_ext_filt = f_ext_filt.reshape(6)
        self._f_ext_last = f_ext_filt
        return signed, f_ext_filt

    def ready_causal(self) -> bool:
        """Warm-up gate for the causal path (filter settled + history filled)."""
        return self._n_updates >= self.cfg.min_samples

    @property
    def n_samples(self) -> int:
        """Number of causal update() calls seen (for warm-up progress messages)."""
        return self._n_updates

    @classmethod
    def from_yaml(cls, raw: dict) -> CompensatedForceObserver:
        f = raw.get("force", {})
        fc_cfg = float(yaml.safe_load(CONFIG_FORCE.read_text()).get("filtfilt_cutoff_hz", 2.5))
        fc_hz = float(f.get("fc_hz", fc_cfg))
        timing = raw.get("timing", {})
        dt_ms = float(timing.get("dt_ms", 10.0))
        rp = raw.get("realtime_push", {})
        cycle = int(rp.get("cycle", max(1, int(round(dt_ms / 5.0)))))
        poll_hz = 1000.0 / (cycle * 5.0)
        return cls(
            ForceObserverConfig(
                phi_path=PHI_JSON,
                phi_source=str(f.get("phi_source", "phi_recommended")),
                fc_hz=fc_hz,
                buffer_s=float(f.get("buffer_s", 4.0)),
                min_samples=int(f.get("min_samples", 35)),
                use_inertia=bool(f.get("use_inertia", False)),
                poll_hz=poll_hz,
                causal_fc_hz=float(f.get("causal_fc_hz", 10.0)),
                causal_order=int(f.get("causal_order", 2)),
                causal_history=int(f.get("causal_history", 5)),
            )
        )
```

## `rm75_control/control/joint_admittance_8dof/config.py`

```python
"""YAML -> JointIkConfig loader for the joint-space inner loop.

Keeps the inner-loop tuning (QP weights, CBF, nullspace/arm-angle, safety
limits) in one config section so bring-up is a matter of editing yaml, not
code.  The outer admittance loop is configured via admittance_common keys and built via AdmittanceConfig.from_dict.
"""

from __future__ import annotations

import math

import numpy as np

from rm75_control.control.joint_admittance_8dof.loop import JointIkConfig
from rm75_control.control.joint_admittance_8dof.ik_types import SrDampingConfig
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import ArmAngleTaskConfig
from rm75_control.control.joint_admittance_8dof.tasks.manipulability_task import ManipulabilityTaskConfig
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import NullspaceTaskConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import RailExtensionConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_lock import RailLockConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode


def _arr(v, default):
    return np.asarray(v if v is not None else default, dtype=float)


def _resolve_rail_mode(r: dict) -> tuple[RailMode, LockedStyle]:
    """Read (mode, locked_style) from yaml.

    Schema::
        rail:
          mode: coupled | locked
          locked_style: hold | rail_only | tcp_fixed   # only if mode=locked
    """
    mode_str = str(r.get("mode", "coupled")).lower()
    raw_style = r.get("locked_style", "hold")
    if mode_str == "coupled":
        return RailMode.COUPLED, LockedStyle.HOLD
    if mode_str == "locked":
        style = LockedStyle(str(raw_style).lower()) if raw_style else LockedStyle.HOLD
        return RailMode.LOCKED, style
    raise ValueError(f"unknown inner.rail.mode: {r.get('mode')!r}")


def build_joint_ik_config(raw: dict) -> JointIkConfig:
    timing = raw.get("timing", {})
    dt = float(timing.get("dt_ms", 5.0)) / 1000.0

    inner = raw.get("inner", {})
    euler_order = str(raw.get("frames", {}).get("euler_order", inner.get("euler_order", "xyz")))

    c = inner.get("qp", {})
    reg = c.get("reg", None)
    if isinstance(reg, (list, tuple)):
        reg_arr = _arr(reg, [1e-2] * 8)
    elif reg is None:
        reg_arr = None  # let QpConfig defaults through
    else:
        reg_arr = np.full(8, float(reg))

    coll = inner.get("collision", {})
    collision = CollisionConfig(
        enabled=bool(coll.get("enabled", True)),
        d_safe=float(coll.get("d_safe", 0.03)),
        d_activate=float(coll.get("d_activate", 0.08)),
        gamma=float(coll.get("gamma", 5.0)),
        max_pairs=int(coll.get("max_pairs", 8)),
    )

    sr = c.get("sr_damping", {})
    sr_damping = SrDampingConfig(
        lam0=float(sr.get("lam0", 0.05)),
        sigma_ref=float(sr.get("sigma_ref", 0.08)),
        sigma_floor=float(sr.get("sigma_floor", 1e-6)),
    )

    qp_kwargs: dict = dict(
        task_weight=_arr(c.get("task_weight"), [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]),
        backend=str(c.get("backend", "proxqp")),
        eps_abs=float(c.get("eps_abs", 1e-6)),
        max_iter=int(c.get("max_iter", 200)),
        euler_order=euler_order,
        collision=collision,
        sr_damping=sr_damping,
        use_dyn_nullspace=bool(c.get("use_dyn_nullspace", False)),
        limit_damper_band_rad=float(c.get("limit_damper_band_rad", 0.15)),
        limit_damper_band_rail_m=float(c.get("limit_damper_band_rail_m", 0.05)),
        warn_on_fail=bool(c.get("warn_on_fail", True)),
        mass_reg_floor=float(c.get("mass_reg_floor", 0.05)),
        mass_weight_exempt_rail=bool(c.get("mass_weight_exempt_rail", True)),
        mass_reg_lpf_tau_s=float(c.get("mass_reg_lpf_tau_s", 0.2)),
        task_weight_min_frac=float(c.get("task_weight_min_frac", 0.05)),
        task_weight_lpf_tau_s=float(c.get("task_weight_lpf_tau_s", 0.25)),
        max_iter_cap=int(c.get("max_iter_cap", 400)),
        fail_qdot_decay=float(c.get("fail_qdot_decay", 0.85)),
        max_solve_ms=float(c.get("max_solve_ms", 8.0)),
        twist_sigma_floor=float(c.get("twist_sigma_floor", 0.08)),
        sigma_escape_ref_scale=float(c.get("sigma_escape_ref_scale", 1.25)),
    )
    if reg_arr is not None:
        qp_kwargs["reg"] = reg_arr
    if "use_mass_weighted_reg" in c:
        qp_kwargs["use_mass_weighted_reg"] = bool(c["use_mass_weighted_reg"])
    qp = QpConfig(**qp_kwargs)

    n = inner.get("nullspace", {})
    q_nominal_deg = n.get("q_nominal_deg")
    nullspace = NullspaceTaskConfig(
        k_center=float(n.get("k_center", 0.5)),
        k_limit=float(n.get("k_limit", 2.0)),
        activation=float(n.get("activation", 0.85)),
        weights=(np.asarray(n["weights"], dtype=float) if n.get("weights") is not None else None),
        q_nominal_rad=(
            np.radians(np.asarray(q_nominal_deg, dtype=float)) if q_nominal_deg is not None else None
        ),
    )

    m = n.get("manipulability", {})
    manipulability = ManipulabilityTaskConfig(
        k_mu=float(m.get("k_mu", 0.8)),
        eps_rad=float(m.get("eps_rad", 1e-4)),
        sigma_fade_ref=float(m.get("sigma_fade_ref", 0.12)),
    )

    a = inner.get("arm_angle", {})
    psi_ref_deg = a.get("psi_ref_deg")
    psi_home_deg = a.get("psi_home_deg")
    psi_hard_lower_deg = a.get("psi_hard_lower_deg")
    psi_hard_upper_deg = a.get("psi_hard_upper_deg")
    arm_angle = ArmAngleTaskConfig(
        enabled=bool(a.get("enabled", False)),
        k_psi=float(a.get("k_psi", 1.0)),
        psi_ref_rad=(math.radians(float(psi_ref_deg)) if psi_ref_deg is not None else None),
        psi_home_rad=(math.radians(float(psi_home_deg)) if psi_home_deg is not None else None),
        max_psi_swing_rad=math.radians(float(a.get("max_psi_swing_deg", 150.0))),
        psi_hard_lower_rad=(
            math.radians(float(psi_hard_lower_deg)) if psi_hard_lower_deg is not None else None
        ),
        psi_hard_upper_rad=(
            math.radians(float(psi_hard_upper_deg)) if psi_hard_upper_deg is not None else None
        ),
    )

    margin_deg = float(inner.get("position_margin_deg", 1.0))
    resync_deg = float(inner.get("resync_err_deg", 6.0))
    resync_rail_mm = float(inner.get("resync_err_rail_mm", 20.0))

    a_max_arm = float(inner.get("a_max_arm", 20.0))
    a_max_rail = float(inner.get("a_max_rail_m_s2", 0.5))

    r = inner.get("rail", {})
    rail_mode, locked_style = _resolve_rail_mode(r)
    re_cfg = inner.get("rail_extension", {})
    rail_extension = RailExtensionConfig(
        enabled=bool(re_cfg.get("enabled", True)),
        k_ext=float(re_cfg.get("k_ext", 2.0)),
        k_ff=float(re_cfg.get("k_ff", 1.0)),
        v_ff_thr_m_s=float(re_cfg.get("v_ff_thr_m_s", 0.005)),
        v_ff_span_m_s=float(re_cfg.get("v_ff_span_m_s", 0.015)),
        e0_m=float(re_cfg.get("e0_m", 0.02)),
        e1_m=float(re_cfg.get("e1_m", 0.08)),
        w_max=float(re_cfg.get("w_max", 2.0)),
        v_max_m_s=float(re_cfg.get("v_max_m_s", 0.08)),
        limit_margin_m=float(re_cfg.get("limit_margin_m", 0.08)),
        k_sigma_boost=float(re_cfg.get("k_sigma_boost", 2.0)),
        k_esc=float(re_cfg.get("k_esc", 0.5)),
        w_sigma_floor=float(re_cfg.get("w_sigma_floor", 1.0)),
        k_pose=float(re_cfg.get("k_pose", 2.0)),
        pose_e0_m=float(re_cfg.get("pose_e0_m", 0.005)),
        pose_e1_m=float(re_cfg.get("pose_e1_m", 0.04)),
        pose_w_max=float(re_cfg.get("pose_w_max", 4.0)),
        sigma_guard_enter=float(re_cfg.get("sigma_guard_enter", 0.45)),
        sigma_guard_exit=float(re_cfg.get("sigma_guard_exit", 0.70)),
        v_guard_max_m_s=float(re_cfg.get("v_guard_max_m_s", 0.04)),
        v_lpf_tau_s=float(re_cfg.get("v_lpf_tau_s", 0.12)),
    )

    rail = RailLockConfig(
        mode=rail_mode,
        locked_style=locked_style,
        q_ref_m=(float(r["q_ref_m"]) if r.get("q_ref_m") is not None else None),
        lock_gain=float(r.get("lock_gain", 200.0)),
        lock_reg_scale=float(r.get("lock_reg_scale", 100.0)),
        lock_vel_eps_m_s=float(r.get("lock_vel_eps_m_s", 0.0)),
        lock_hard_pin=bool(r.get("lock_hard_pin", True)),
        v_max_m_s=(float(r["v_max_m_s"]) if r.get("v_max_m_s") is not None else None),
        travel_m=float(r.get("travel_m", 0.80)),
    )

    return JointIkConfig(
        dt=dt,
        control_frame=str(inner.get("control_frame", "tool")),
        euler_order=euler_order,
        qp=qp,
        nullspace=nullspace,
        manipulability=manipulability,
        arm_angle=arm_angle,
        rail=rail,
        rail_extension=rail_extension,
        v_scale=float(inner.get("v_scale", 0.5)),
        a_max_arm_rad_s2=a_max_arm,
        a_max_rail_m_s2=a_max_rail,
        position_margin_rad=math.radians(margin_deg),
        position_margin_rail_m=float(inner.get("position_margin_rail_mm", 0.0)) / 1000.0,
        resync_err_rad=math.radians(resync_deg),
        resync_err_rail_m=resync_rail_mm / 1000.0,
        nullspace_d_null=float(inner.get("nullspace_d_null", 0.0)),
        nullspace_d_null_adaptive=float(inner.get("nullspace_d_null_adaptive", 1.0)),
        nullspace_max_qdot_frac=float(inner.get("nullspace_max_qdot_frac", 0.2)),
        centering_recovery_gain=float(inner.get("centering_recovery_gain", 3.0)),
        centering_recovery_max_qdot_frac=float(
            inner.get("centering_recovery_max_qdot_frac", 0.35)
        ),
        centering_recovery_tol=float(inner.get("centering_recovery_tol", 0.12)),
    )
```

## `rm75_control/control/joint_admittance_8dof/loop.py`

```python
"""Joint-space inner loop: Cartesian twist -> absolute joint angles (rm_movej_canfd).

``JointIkController``: hardware-free WBC QP IK + safety clamp (no send-path LPF).
``run_joint_admittance_phases``: on-robot orchestration closing on FK(q_meas).
"""

from __future__ import annotations

import csv
import inspect
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig, QpIkController
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    arm_q_from_full,
    deg2rad,
    full_q_from_arm,
    max_joint_err_deg,
    pose_distance,
    pose_error,
    pose_track_error_mm_deg,
    rad2deg,
    wrap_joint_delta,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_lock import (
    RailLockConfig,
    RailLockTask,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import (
    LockedStyle,
    RailMode,
)
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import (
    ArmAngleTask,
    ArmAngleTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.manipulability_task import (
    ManipulabilityTask,
    ManipulabilityTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)
from rm75_control.control.joint_admittance_8dof.tasks.secondary_composer import SecondaryComposer
from rm75_control.control.joint_admittance_8dof.ik_types import saturate_error
from rm75_control.control.joint_admittance_8dof.utils.safety import (
    SafetyLimiter,
    SafetyLimits,
    Watchdog,
)


# ---------------------------------------------------------------------------
# Inner loop (hardware-free)
# ---------------------------------------------------------------------------
@dataclass
class JointIkConfig:
    dt: float = 0.005
    control_frame: str = "tool"        # frame the incoming twist is expressed in
    euler_order: str = "xyz"
    qp: QpConfig = field(default_factory=QpConfig)
    nullspace: NullspaceTaskConfig = field(default_factory=NullspaceTaskConfig)
    manipulability: ManipulabilityTaskConfig = field(default_factory=ManipulabilityTaskConfig)
    arm_angle: ArmAngleTaskConfig = field(default_factory=ArmAngleTaskConfig)
    rail: RailLockConfig = field(default_factory=RailLockConfig)
    # Preferred-extension rail coordination (COUPLED mode only).
    rail_extension: RailExtensionConfig = field(default_factory=RailExtensionConfig)
    v_scale: float = 0.5               # fraction of URDF joint velocity limit allowed
    # Accel limits are unit-separated: rail m/s^2, arm rad/s^2.
    a_max_arm_rad_s2: float = 20.0     # rad/s^2 per arm joint (1..7)
    a_max_rail_m_s2: float = 0.30      # m/s^2 for prismatic rail (0)
    position_margin_rad: float = 0.017
    position_margin_rail_m: float = 0.0  # metres (do not reuse arm rad margin)
    # QP velocity bound: stop q_cmd leading q_meas (0 disables; never a teleport).
    resync_err_rad: float = 0.10       # arm joints 1..7 (radians)
    resync_err_rail_m: float = 0.020   # rail joint 0 (metres; 20 mm)
    nullspace_d_null: float = 0.0          # viscous damping on secondary qdot (1/s)
    nullspace_d_null_adaptive: float = 1.0 # scale d_null up near joint limits
    # Cap soft secondary qdot as a fraction of URDF v_max (near σ, N→I).
    nullspace_max_qdot_frac: float = 0.2
    # Latched stronger posture pull after singularity escape until near target.
    centering_recovery_gain: float = 3.0
    centering_recovery_max_qdot_frac: float = 0.35
    centering_recovery_tol: float = 0.12


@dataclass
class JointIkStep:
    q_send: np.ndarray          # commanded joint position (rad) after clamp
    qdot: np.ndarray            # joint velocity (rad/s)
    twist_base: np.ndarray      # requested task twist in the base frame
    sigma_min: float
    manip: float
    slack_norm: float
    n_cbf_active: int
    follow_err_rad: float       # max |q_meas - q_cmd| this tick (0 if no q_meas)
    cart_err_mm: float = 0.0    # outer-loop tracking error, filled by the caller
    qdot_ff_norm: float = 0.0
    arm_singularity_smooth: float = 1.0
    limit_activation: float = 0.0
    vel_clamped: bool = False
    acc_clamped: bool = False
    pos_clamped: bool = False
    tcp_jump_mm: float = 0.0
    rail_ext_err_m: float = 0.0
    rail_ext_weight: float = 0.0
    rail_vel_pin: float = float("nan")      # m/s hard pin, or NaN if free
    rail_qdot_ff: float = float("nan")      # plan qdot_ff[0] before strip
    plan_drives_rail: bool = False


class JointIkController:
    """Reusable inner loop: (q_cmd, q_meas, twist) -> next joint command (rad)."""

    def __init__(self, kin: RobotKinematics, cfg: JointIkConfig | None = None) -> None:
        self.kin = kin
        self.cfg = cfg or JointIkConfig()
        self.cfg.qp.euler_order = self.cfg.euler_order
        self.centering_task = JointCenteringTask.from_kinematics(kin, self.cfg.nullspace)
        self.manipulability_task = (
            ManipulabilityTask(kin, self.cfg.manipulability)
            if self.cfg.manipulability.k_mu > 0.0
            else None
        )
        self.arm_task = (
            ArmAngleTask(kin, self.cfg.arm_angle) if self.cfg.arm_angle.enabled else None
        )
        self.rail_task = RailLockTask(self.cfg.rail)
        self.rail_ext_task = (
            RailExtensionTask(kin, self.cfg.rail_extension)
            if self.cfg.rail_extension.enabled
            else None
        )
        # Preset-gated: pose_attract (move), reach (scan), off (hold).
        self._rail_ext_active = True
        # σ-escape gradient cache (~20 Hz at dt=5 ms via RailGoodness).
        from rm75_control.control.joint_admittance_8dof.tasks.rail_goodness import (
            CachedRailGoodness,
            SigmaMinGoodness,
        )

        self._rail_goodness = CachedRailGoodness(
            SigmaMinGoodness(kin), period_ticks=10
        )
        self._sigma_grad_rail_cached: float = 0.0
        self._sigma_grad_tick: int = 0
        self._sigma_grad_period: int = 10
        # Build an 8-vector a_max: rail is m/s^2, arm joints 1..7 are rad/s^2.
        a_max_vec = np.full(kin.nv, float(self.cfg.a_max_arm_rad_s2))
        a_max_vec[0] = float(self.cfg.a_max_rail_m_s2)
        # Position margin is unit-separated too: arm rad, rail metres.
        margin_vec = np.full(kin.nv, float(self.cfg.position_margin_rad))
        margin_vec[0] = float(self.cfg.position_margin_rail_m)
        self.limits = SafetyLimits.from_kinematics(
            kin,
            v_scale=self.cfg.v_scale,
            a_max=a_max_vec,
            position_margin=margin_vec,
        )
        if self.cfg.rail.v_max_m_s is not None:
            self.limits.v_max[0] = min(
                float(self.limits.v_max[0]),
                float(self.cfg.rail.v_max_m_s),
            )
        self.core = QpIkController(self.kin, self.limits, self.cfg.qp)
        self.safety = SafetyLimiter(self.limits)
        self.q_cmd = np.zeros(kin.nv, dtype=float)
        self._arm_task_suppressed = False
        self._centering_suppressed = False
        self._manipulability_active = False
        self.secondary = SecondaryComposer.from_controller_parts(
            self.centering_task,
            self.arm_task,
            self.cfg.nullspace,
            manipulability=self.manipulability_task,
            rail_lock=self.rail_task,
            d_null=self.cfg.nullspace_d_null,
            adaptive_d_null_gain=self.cfg.nullspace_d_null_adaptive,
            v_max=kin.v_max,
            max_qdot_frac=self.cfg.nullspace_max_qdot_frac,
        )
        self.last_secondary_norm: float = 0.0
        self.last_sigma_min: float = float(self.cfg.qp.sr_damping.sigma_ref)
        self._singularity_escape_seen: bool = False
        self._centering_recovery_active: bool = False
        self._rail_mode: RailMode = self.cfg.rail.mode
        self._locked_style: LockedStyle = self.cfg.rail.locked_style
        # Immutable yaml rail mode (live cfg.rail.mode is mutated by locks).
        self._configured_rail_mode: RailMode = self.cfg.rail.mode
        # When True, plan owns rail velocity via qdot_ff pin.
        self._plan_drives_rail: bool = False
        # Direct joint PTP: integrate plan (+fb); skip Cartesian ProxQP.
        self._direct_joint_ptp: bool = False
        self._apply_rail_mode_side_effects()

    @property
    def rail_mode(self) -> RailMode:
        return self._rail_mode

    def set_plan_drives_rail(self, enabled: bool) -> None:
        """Pin rail to plan qdot_ff[0] (SRS move→D); clear on scan/hold exit."""
        self._plan_drives_rail = bool(enabled)

    def set_direct_joint_ptp(self, enabled: bool) -> None:
        """Enable joint-space PTP (no Cartesian ProxQP primary)."""
        self._direct_joint_ptp = bool(enabled)

    @property
    def configured_rail_mode(self) -> RailMode:
        """Yaml rail mode (immutable); live cfg.rail.mode is mutated by locks."""
        return self._configured_rail_mode

    @property
    def locked_style(self) -> LockedStyle:
        """Active LockedStyle (only meaningful when rail_mode == LOCKED)."""
        return self._locked_style

    @property
    def is_locked_hold(self) -> bool:
        return (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.HOLD
        )

    def set_arm_task_suppressed(self, suppressed: bool) -> None:
        """Pause arm-angle nullspace (e.g. during joint-space move)."""
        self._arm_task_suppressed = bool(suppressed)

    def set_centering_suppressed(self, suppressed: bool) -> None:
        """Pause joint-centering nullspace (e.g. during joint-space move)."""
        self._centering_suppressed = bool(suppressed)

    def set_manipulability_active(self, active: bool) -> None:
        """Use ∇μ ascent in the nullspace instead of Liegeois centering."""
        self._manipulability_active = bool(active) and self.manipulability_task is not None

    def set_rail_extension_active(self, active: bool) -> None:
        """Gate preferred-extension / pose-attract rail task (COUPLED)."""
        self._rail_ext_active = bool(active)

    def set_rail_extension_mode(self, mode: str) -> None:
        """Select ``reach`` (scan) or ``pose_attract`` (move→D)."""
        if self.rail_ext_task is not None:
            self.rail_ext_task.set_mode(mode)  # type: ignore[arg-type]

    def set_rail_pose_target(self, y_rail_m: float | None) -> None:
        """Soft-attract target for pose_attract mode (metres on the rail)."""
        if self.rail_ext_task is not None:
            self.rail_ext_task.set_rail_pose_target(y_rail_m)

    def capture_rail_extension_ref(self) -> None:
        """Capture preferred rail extension from the current scan-entry posture."""
        if self.rail_ext_task is not None:
            self.rail_ext_task.capture_reference(self.q_cmd)

    def reset(self, q0_rad: np.ndarray) -> None:
        self.q_cmd = np.asarray(q0_rad, dtype=float).copy()
        self.core.reset()
        self.safety.reset(self.q_cmd)
        if self.arm_task is not None:
            self.arm_task.reset(self.q_cmd)
        self.rail_task.reset(self.q_cmd)
        if self.rail_ext_task is not None:
            self.rail_ext_task.reset(self.q_cmd)
        self._singularity_escape_seen = False
        self._centering_recovery_active = False
        self._apply_rail_mode_side_effects()

    def _centering_recovery_scale(
        self,
        q: np.ndarray,
        sigma_min: float,
        sigma_escape_ref: float,
    ) -> float:
        """Latch strong posture recovery after leaving the singularity zone."""
        if sigma_escape_ref > 1e-9 and sigma_min < sigma_escape_ref:
            self._singularity_escape_seen = True
            self._centering_recovery_active = False
            return 1.0

        if not self._singularity_escape_seen:
            self._centering_recovery_active = False
            return 1.0

        target_error = np.abs(
            (np.asarray(q, dtype=float) - self.centering_task.q_target)
            / self.centering_task.half
        )
        weighted_error = float(np.max(target_error * self.centering_task.w))
        if weighted_error <= max(float(self.cfg.centering_recovery_tol), 0.0):
            self._singularity_escape_seen = False
            self._centering_recovery_active = False
            return 1.0

        self._centering_recovery_active = True
        return max(float(self.cfg.centering_recovery_gain), 1.0)

    def set_rail_mode(
        self,
        mode: RailMode | str,
        *,
        q_ref_m: float | None = None,
        locked_style: LockedStyle | str | None = None,
    ) -> None:
        """Set rail mode (COUPLED / LOCKED) and optional locked_style."""
        if isinstance(mode, str):
            mode = RailMode(mode)
        self._rail_mode = mode
        if locked_style is not None:
            if isinstance(locked_style, str):
                locked_style = LockedStyle(locked_style)
            self._locked_style = locked_style
        if q_ref_m is not None:
            self.rail_task.set_reference(q_ref_m)
        elif mode == RailMode.LOCKED and self._locked_style == LockedStyle.HOLD:
            # HOLD without explicit ref = pin at current command (never yaml 0.0).
            self.rail_task.set_reference(float(self.q_cmd[0]))
        self._apply_rail_mode_side_effects()

    def set_coupled(self) -> None:
        """Convenience: switch to RailMode.COUPLED (rail participates in QP)."""
        self.set_rail_mode(RailMode.COUPLED)

    def set_locked(
        self,
        style: LockedStyle | str = LockedStyle.HOLD,
        *,
        q_ref_m: float | None = None,
    ) -> None:
        """Convenience: switch to RailMode.LOCKED with a specific style."""
        self.set_rail_mode(RailMode.LOCKED, q_ref_m=q_ref_m, locked_style=style)

    def _apply_rail_mode_side_effects(self) -> None:
        self.rail_task.cfg.mode = self._rail_mode
        self.rail_task.cfg.locked_style = self._locked_style

    def _pin_rail_if_locked_hold(self) -> None:
        """Freeze rail_y when LOCKED+HOLD (RAIL_ONLY/TCP_FIXED drive via qdot_ff)."""
        if not self.is_locked_hold or not self.cfg.rail.lock_hard_pin:
            return
        if self.rail_task.q_ref is None:
            return
        self.q_cmd[0] = float(self.rail_task.q_ref)
        self.core.qdot_prev[0] = 0.0

    def _twist_to_base(self, twist: np.ndarray, q_for_rot: np.ndarray) -> np.ndarray:
        twist = np.asarray(twist, dtype=float)
        if self.cfg.control_frame != "tool":
            return twist
        R = self.kin.fk_placement(q_for_rot).rotation
        out = np.zeros(6, dtype=float)
        out[:3] = R @ twist[:3]
        out[3:6] = R @ twist[3:6]
        return out

    def _secondary(
        self,
        q: np.ndarray,
        qdot_ff: np.ndarray | None,
        *,
        manipulability_active: bool | None = None,
        centering_sigma_fade: bool = True,
        sigma_min: float | None = None,
        sigma_escape_ref: float | None = None,
        centering_gain_scale: float = 1.0,
        max_qdot_frac_override: float | None = None,
    ) -> np.ndarray:
        sigma_ref = float(self.cfg.qp.sr_damping.sigma_ref)
        if sigma_escape_ref is None:
            sigma_escape_ref = sigma_ref * float(
                getattr(self.cfg.qp, "sigma_escape_ref_scale", 1.25)
            )
        qdot0 = self.secondary.compose(
            q,
            qdot_ff,
            self.core.qdot_prev,
            arm_suppressed=self._arm_task_suppressed,
            # Prefer this tick's σ (stale σ delays ∇μ escape by one period).
            sigma_min=(
                self.last_sigma_min if sigma_min is None else float(sigma_min)
            ),
            sigma_ref=sigma_ref,
            sigma_escape_ref=float(sigma_escape_ref),
            centering_suppressed=self._centering_suppressed,
            centering_sigma_fade=centering_sigma_fade,
            manipulability_active=(
                self._manipulability_active
                if manipulability_active is None
                else manipulability_active
            ),
            centering_gain_scale=centering_gain_scale,
            max_qdot_frac_override=max_qdot_frac_override,
        )
        self.last_secondary_norm = float(np.linalg.norm(qdot0))
        return qdot0

    def update(
        self,
        twist: np.ndarray,
        dt: float | None = None,
        q_meas: np.ndarray | None = None,
        qdot_ff: np.ndarray | None = None,
        *,
        vel_ff: np.ndarray | None = None,
    ) -> JointIkStep:
        """One Cartesian-tracking WBC step.

        ``q_meas`` rotates tool→base twist and bounds command lead via QP
        velocity constraints (never a position teleport). ``qdot_ff`` feeds
        the nullspace with centering / arm-angle tasks.
        """
        dt = self.cfg.dt if dt is None else dt
        q_prev = self.q_cmd
        follow_err = 0.0 if q_meas is None else float(np.max(np.abs(q_prev - q_meas)))
        q_rot = q_meas if q_meas is not None else q_prev
        twist_base = self._twist_to_base(twist, q_rot)

        # Two-threshold σ policy: sigma_ref brakes twist; sigma_escape_ref
        # starts avoidance earlier (must lead the brake).
        sigma_ref = float(self.cfg.qp.sr_damping.sigma_ref)
        sigma_escape_ref = sigma_ref * float(
            getattr(self.cfg.qp, "sigma_escape_ref_scale", 1.25)
        )
        J_pre = self.kin.jacobian(q_prev)
        sigma_pre = float(self.kin.singular_values(J_pre).min())
        centering_gain_scale = self._centering_recovery_scale(
            q_prev, sigma_pre, sigma_escape_ref
        )
        if sigma_ref > 1e-9 and sigma_pre < sigma_ref:
            floor = float(getattr(self.cfg.qp, "twist_sigma_floor", 0.08))
            twist_scale = max(float(sigma_pre / sigma_ref), floor)
            # Below half σ_ref, square scale so force retract cannot collapse posture.
            if sigma_pre < 0.5 * sigma_ref:
                twist_scale = max(twist_scale * twist_scale, 0.5 * floor)
            twist_base = twist_base * twist_scale

        locked_hold = self.is_locked_hold
        rail_only = (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.RAIL_ONLY
        )
        tcp_fixed = (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.TCP_FIXED
        )
        # Clamp qdot_ff to v_max (plan/anchor must not exceed hardware limits).
        if qdot_ff is not None:
            v_lim_ff = np.asarray(self.safety.lim.v_max, dtype=float)
            qdot_ff = np.clip(np.asarray(qdot_ff, dtype=float), -v_lim_ff, v_lim_ff)

        # Direct joint PTP: integrate plan (+fb); skip Cartesian ProxQP.
        if self._direct_joint_ptp and qdot_ff is not None:
            qdot_cmd = np.asarray(qdot_ff, dtype=float).copy()
            q_next = q_prev + qdot_cmd * dt
            rep = self.safety.clamp(q_prev, q_next, dt)
            self.q_cmd = rep.q_safe
            if dt > 1e-9:
                self.core.qdot_prev = (self.q_cmd - q_prev) / dt
            else:
                self.core.qdot_prev = qdot_cmd
            if q_meas is not None:
                lead_max = float(self.cfg.resync_err_rail_m)
                if lead_max > 0.0:
                    q0_meas = float(np.asarray(q_meas, dtype=float)[0])
                    q0_cmd = float(self.q_cmd[0])
                    if q0_cmd > q0_meas + lead_max:
                        self.q_cmd[0] = q0_meas + lead_max
                        if dt > 1e-9:
                            self.core.qdot_prev[0] = (self.q_cmd[0] - q_prev[0]) / dt
                    elif q0_cmd < q0_meas - lead_max:
                        self.q_cmd[0] = q0_meas - lead_max
                        if dt > 1e-9:
                            self.core.qdot_prev[0] = (self.q_cmd[0] - q_prev[0]) / dt
            J = self.kin.jacobian(q_prev)
            sigma = self.kin.singular_values(J)
            sigma_min = float(sigma.min())
            self.last_sigma_min = sigma_min
            qdot_out = self.core.qdot_prev.copy()
            return JointIkStep(
                q_send=self.q_cmd.copy(),
                qdot=qdot_out,
                twist_base=twist_base,
                sigma_min=sigma_min,
                manip=float(np.prod(sigma)),
                slack_norm=0.0,
                n_cbf_active=0,
                follow_err_rad=follow_err,
                qdot_ff_norm=float(np.linalg.norm(qdot_ff)),
                arm_singularity_smooth=1.0,
                limit_activation=0.0,
                vel_clamped=rep.vel_clamped,
                acc_clamped=rep.acc_clamped,
                pos_clamped=rep.pos_clamped,
                rail_ext_err_m=0.0,
                rail_ext_weight=0.0,
                rail_vel_pin=float(qdot_ff[0]),
                rail_qdot_ff=float(qdot_ff[0]),
                plan_drives_rail=True,
            )

        # Pin rail vel only for LOCKED (RAIL_ONLY/TCP_FIXED) or plan ownership.
        plan_drives_rail = rail_only or tcp_fixed or bool(self._plan_drives_rail)

        qdot_ff_sec = qdot_ff
        rail_vel_pin: float | None = None
        rail_qdot_ff_val = float("nan")
        if qdot_ff is not None:
            qdot_ff_arr = np.asarray(qdot_ff, dtype=float)
            v_rail = float(qdot_ff_arr[0])
            rail_qdot_ff_val = v_rail
            # Secondary tasks act on the arm; strip rail from qdot_ff_sec.
            qdot_ff_sec = qdot_ff_arr.copy()
            qdot_ff_sec[0] = 0.0
            if plan_drives_rail:
                rail_vel_pin = v_rail

        # Command-lead anti-windup: arm rad, rail m (units matter).
        resync_vec = np.full(self.kin.nv, float(self.cfg.resync_err_rad))
        resync_vec[0] = float(self.cfg.resync_err_rail_m)

        # Preferred-extension rail coordination (COUPLED only).
        rail_task_vel: float | None = None
        rail_task_weight = 0.0
        rail_ext_err = 0.0
        # Posture recovery takes nullspace once σ recovers from escape.
        manip_for_saturation = (
            self._manipulability_active and not self._centering_recovery_active
        )
        if (
            self.rail_ext_task is not None
            and self._rail_ext_active
            and self._rail_mode == RailMode.COUPLED
        ):
            sigma_now = float(sigma_pre)
            # sig_scale fades scan FF; sig_escape drives escape (separate thresholds).
            sig_scale = 1.0
            if sigma_ref > 1e-9 and sigma_now < sigma_ref:
                sig_scale = max(sigma_now / sigma_ref, 0.25)
            sig_escape = 1.0
            if sigma_escape_ref > 1e-9 and sigma_now < sigma_escape_ref:
                sig_escape = max(sigma_now / sigma_escape_ref, 0.0)
            # Refresh σ-escape gradient every _sigma_grad_period ticks.
            self._sigma_grad_tick += 1
            if (
                self._sigma_grad_tick % self._sigma_grad_period == 0
                or self._sigma_grad_tick == 1
            ):
                _g, self._sigma_grad_rail_cached = self._rail_goodness.refresh(
                    q_prev, force=True
                )
                del _g
            v_ext, w_ext = self.rail_ext_task(
                q_prev,
                sigma_scale=sig_scale,
                sigma_escape_scale=sig_escape,
                sigma_grad_rail=self._sigma_grad_rail_cached,
                vel_ff=vel_ff,
                dt_s=float(dt),
            )
            rail_ext_err = self.rail_ext_task.last_err_m
            if w_ext > 0.0:
                rail_task_vel = v_ext
                rail_task_weight = w_ext
            # Arm ∇μ escape when σ is depressed (skip if centering recovery latched).
            if (
                sigma_escape_ref > 1e-9
                and sigma_now < sigma_escape_ref
                and not self._centering_recovery_active
            ):
                manip_for_saturation = True

        r = self.core.step(
            q_prev,
            twist_base,
            dt,
            secondary_qdot=self._secondary(
                q_prev,
                qdot_ff_sec,
                manipulability_active=manip_for_saturation,
                centering_sigma_fade=not (
                    self._rail_ext_active and self._rail_mode == RailMode.COUPLED
                ),
                sigma_min=sigma_pre,
                sigma_escape_ref=sigma_escape_ref,
                centering_gain_scale=centering_gain_scale,
                max_qdot_frac_override=(
                    self.cfg.centering_recovery_max_qdot_frac
                    if self._centering_recovery_active
                    else None
                ),
            ),
            q_meas=q_meas,
            resync_err=resync_vec,
            rail_locked=locked_hold,
            rail_lock_reg_scale=self.cfg.rail.lock_reg_scale,
            rail_lock_vel_eps_m_s=self.cfg.rail.lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin,
            zero_secondary_rail=not locked_hold,
            rail_task_vel_m_s=rail_task_vel,
            rail_task_weight=rail_task_weight,
        )

        rep = self.safety.clamp(q_prev, r.q_next, dt)
        self.q_cmd = rep.q_safe
        if dt > 1e-9 and (rep.vel_clamped or rep.acc_clamped or rep.pos_clamped):
            self.core.qdot_prev = rep.dq / dt
        # Hard rail command-lead cap vs encoder (resync_err_rail_m).
        if q_meas is not None:
            lead_max = float(self.cfg.resync_err_rail_m)
            if lead_max > 0.0:
                q0_meas = float(np.asarray(q_meas, dtype=float)[0])
                q0_cmd = float(self.q_cmd[0])
                if q0_cmd > q0_meas + lead_max:
                    self.q_cmd[0] = q0_meas + lead_max
                    if dt > 1e-9:
                        self.core.qdot_prev[0] = (self.q_cmd[0] - q_prev[0]) / dt
                elif q0_cmd < q0_meas - lead_max:
                    self.q_cmd[0] = q0_meas - lead_max
                    if dt > 1e-9:
                        self.core.qdot_prev[0] = (self.q_cmd[0] - q_prev[0]) / dt
        # Plan-owned rail: integrate q_cmd[0] from qdot_ff (QP pin alone is insufficient).
        if plan_drives_rail and qdot_ff is not None and dt > 1e-9:
            v_rail = float(np.asarray(qdot_ff)[0])
            y = float(q_prev[0] + v_rail * dt)
            y_lo = float(self.limits.q_lower[0])
            y_hi = float(self.limits.q_upper[0])
            self.q_cmd[0] = float(np.clip(y, y_lo, y_hi))
            self.core.qdot_prev[0] = (self.q_cmd[0] - q_prev[0]) / dt
            if rail_only:
                self.q_cmd[1:] = q_prev[1:]
                self.core.qdot_prev[1:] = 0.0
        else:
            self._pin_rail_if_locked_hold()
        qdot_out = r.qdot.copy()
        if locked_hold and self.cfg.rail.lock_hard_pin:
            qdot_out[0] = 0.0
        elif plan_drives_rail and qdot_ff is not None:
            qdot_out[0] = float(np.asarray(qdot_ff)[0])
            if rail_only:
                qdot_out[1:] = 0.0
        self.last_sigma_min = r.sigma_min
        return JointIkStep(
            q_send=self.q_cmd.copy(),
            qdot=qdot_out,
            twist_base=twist_base,
            sigma_min=r.sigma_min,
            manip=r.manip,
            slack_norm=r.slack_norm,
            n_cbf_active=r.n_cbf_active,
            follow_err_rad=follow_err,
            qdot_ff_norm=float(np.linalg.norm(qdot_ff)) if qdot_ff is not None else 0.0,
            arm_singularity_smooth=self.secondary.last_arm_smooth,
            limit_activation=self.secondary.last_limit_activation,
            vel_clamped=rep.vel_clamped,
            acc_clamped=rep.acc_clamped,
            pos_clamped=rep.pos_clamped,
            rail_ext_err_m=rail_ext_err,
            rail_ext_weight=rail_task_weight,
            rail_vel_pin=(
                float(rail_vel_pin) if rail_vel_pin is not None else float("nan")
            ),
            rail_qdot_ff=rail_qdot_ff_val,
            plan_drives_rail=bool(plan_drives_rail),
        )


# ---------------------------------------------------------------------------
# Outer loops
# ---------------------------------------------------------------------------
class OuterLoop(Protocol):
    """Task-space controller producing a Cartesian twist each tick."""

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        """Return a 6D twist in the inner loop's control_frame."""
        ...


class AdmittanceOuterLoop:
    """Wrap AdmittanceController + a MotionReferenceSource (force-position hybrid)."""

    def __init__(self, controller, reference_source, *, desired_force: np.ndarray | None = None):
        self.controller = controller
        self.reference = reference_source
        self.desired_force = (
            np.zeros(6) if desired_force is None else np.asarray(desired_force, dtype=float)
        )
        self.last_err_mm: float = 0.0
        self.last_track_rot_deg: float = 0.0
        self.last_vel_ff: np.ndarray | None = None

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        if hasattr(self.reference, "set_origin"):
            try:
                self.reference.set_origin(pose0, t_s=t_s)
            except TypeError:
                self.reference.set_origin(pose0)

    def set_time_scale(self, scale: float) -> None:
        """Governor scale (0..1); pause force integrator with the reference clock."""
        if hasattr(self.controller, "set_time_scale"):
            self.controller.set_time_scale(scale)

    def sample(
        self,
        t_s: float,
        current_pose: np.ndarray,
        f_ext: np.ndarray,
        f_ext_raw: np.ndarray | None = None,
        dt_actual: float | None = None,
        v_tcp_z_actual: float | None = None,
        sensor_age_s: float | None = None,
    ) -> np.ndarray:
        ref = self.reference.sample(t_s)
        # Track-axis-only error (force axis excluded).
        tr_mm, tr_deg = pose_track_error_mm_deg(
            ref.pose_d,
            current_pose,
            track_axes=self.controller.cfg.track_axes,
            euler_order=self.controller.cfg.euler_order,
        )
        self.last_err_mm = tr_mm
        self.last_track_rot_deg = tr_deg
        self.last_vel_ff = np.asarray(ref.vel_ff, dtype=float).copy()
        return self.controller.compute_velocity_command(
            current_pose,
            ref.pose_d,
            ref.vel_ff,
            f_ext,
            self.desired_force,
            f_ext_raw=f_ext_raw,
            dt_actual=dt_actual,
            v_tcp_z_actual=v_tcp_z_actual,
            sensor_age_s=sensor_age_s,
        )


@dataclass
class CartesianTrackConfig:
    """PD + feedforward Cartesian tracking (no force axis)."""

    k_task: np.ndarray = field(default_factory=lambda: np.full(6, 2.0))
    max_pos_err_m: float = 0.05
    max_rot_err_rad: float = 0.35
    max_lin_vel_m_s: float = 0.4
    max_ang_vel_rad_s: float = 1.5
    euler_order: str = "xyz"
    # Must match JointIkConfig.control_frame (tool twist is rotated by R @ twist).
    control_frame: str = "tool"


class CartesianTrackOuterLoop:
    """PD + feedforward Cartesian tracking against measured pose (no force)."""

    def __init__(self, reference, cfg: CartesianTrackConfig | None = None) -> None:
        self.reference = reference
        self.cfg = cfg or CartesianTrackConfig()
        self.last_err_mm: float = 0.0
        self.time_scale: float = 1.0
        self.last_vel_ff: np.ndarray | None = None

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        if hasattr(self.reference, "set_origin"):
            try:
                self.reference.set_origin(pose0, t_s=t_s)
            except TypeError:
                self.reference.set_origin(pose0)

    def set_time_scale(self, scale: float) -> None:
        """Governor scale (0..1): scale trajectory vel_ff only, not the PD term."""
        self.time_scale = float(np.clip(scale, 0.0, 1.0))

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        del f_ext
        cfg = self.cfg
        ref = self.reference.sample(t_s)
        self.last_vel_ff = np.asarray(ref.vel_ff, dtype=float).copy()
        err = pose_error(ref.pose_d, current_pose, cfg.euler_order)
        self.last_err_mm = float(np.linalg.norm(err[:3]) * 1000.0)
        err_sat = saturate_error(err, cfg.max_pos_err_m, cfg.max_rot_err_rad)
        v_ff = np.asarray(ref.vel_ff, dtype=float) * self.time_scale
        v = v_ff + cfg.k_task * err_sat  # base-frame twist

        lin_n = float(np.linalg.norm(v[:3]))
        if cfg.max_lin_vel_m_s > 0.0 and lin_n > cfg.max_lin_vel_m_s:
            v[:3] *= cfg.max_lin_vel_m_s / lin_n
        ang_n = float(np.linalg.norm(v[3:6]))
        if cfg.max_ang_vel_rad_s > 0.0 and ang_n > cfg.max_ang_vel_rad_s:
            v[3:6] *= cfg.max_ang_vel_rad_s / ang_n

        if cfg.control_frame == "tool":
            R = Rsc.from_euler(cfg.euler_order, current_pose[3:6], degrees=False).as_matrix()
            out = np.zeros(6, dtype=float)
            out[:3] = R.T @ v[:3]
            out[3:6] = R.T @ v[3:6]
            return out
        return v


@dataclass
class JointTrackConfig:
    """Joint-space PD + feedforward tracking (MoveJ-like; no Cartesian stall)."""

    k_joint: float = 2.0
    max_joint_err_rad: float = 0.35
    sigma_ref: float = 0.08
    # σ-adaptive floor: k_eff = k_joint * max(σ/σ_ref, floor).
    k_joint_sigma_min_frac: float = 0.2
    control_frame: str = "tool"
    euler_order: str = "xyz"
    # Rise-only slew on k_eff (1/s); fall is immediate for singularity protection.
    k_joint_rise_per_s: float = 1.2
    # LPF on last_qdot_fb (s); damps QP dual chatter when secondary ≈ slack·W_task.
    fb_lpf_tau_s: float = 0.015
    # Scale fb secondary pull (0..1); keeps QP reg well-conditioned.
    fb_secondary_gain: float = 0.4


class JointTrackOuterLoop:
    """MoveJ-like outer loop: track joint plan via J(q)·(qdot_plan + k·q_err)."""

    def __init__(
        self,
        reference,
        kin: RobotKinematics,
        cfg: JointTrackConfig | None = None,
        *,
        v_max_rad_s: np.ndarray | None = None,
    ) -> None:
        self.reference = reference
        self.kin = kin
        self.cfg = cfg or JointTrackConfig()
        self.v_max = (
            np.asarray(v_max_rad_s, dtype=float)
            if v_max_rad_s is not None
            else np.asarray(kin.v_max, dtype=float)
        )
        self.last_err_mm: float = 0.0
        self.last_joint_err_deg: float = 0.0
        self.last_sigma_min: float = 0.0
        # Feedback-only term for QP secondary (plan ff is governor-scaled separately).
        self.last_qdot_fb: np.ndarray | None = None
        self._qdot_fb_lpf: np.ndarray | None = None  # LPF state, unscaled
        self._k_eff_prev: float | None = None
        self._t_prev: float | None = None

    def set_origin(self, pose0: np.ndarray) -> None:
        if hasattr(self.reference, "set_origin"):
            self.reference.set_origin(pose0)

    def sample(
        self,
        t_s: float,
        current_pose: np.ndarray,
        f_ext: np.ndarray,
        *,
        q_meas: np.ndarray | None = None,
    ) -> np.ndarray:
        del f_ext
        if q_meas is None:
            raise RuntimeError("JointTrackOuterLoop.sample requires q_meas")
        cfg = self.cfg
        q_ref, qdot_plan = self.reference.sample_q(t_s)
        q_meas = np.asarray(q_meas, dtype=float)
        q_err = np.clip(
            wrap_joint_delta(q_meas, q_ref),
            -cfg.max_joint_err_rad,
            cfg.max_joint_err_rad,
        )
        self.last_joint_err_deg = max_joint_err_deg(q_meas, q_ref)
        J = self.kin.jacobian(q_meas)
        sigma = self.kin.singular_values(J)
        sigma_min = float(sigma.min())
        self.last_sigma_min = sigma_min
        if cfg.sigma_ref > 1e-9:
            k_target = cfg.k_joint * float(
                np.clip(sigma_min / cfg.sigma_ref, cfg.k_joint_sigma_min_frac, 1.0)
            )
        else:
            k_target = cfg.k_joint
        # Rise-only slew on k_eff (fall is immediate).
        if (
            self._k_eff_prev is None
            or self._t_prev is None
            or cfg.k_joint_rise_per_s <= 0.0
            or k_target <= self._k_eff_prev
        ):
            k_eff = k_target
        else:
            dt_eff = max(0.0, t_s - self._t_prev)
            k_eff = min(k_target, self._k_eff_prev + cfg.k_joint_rise_per_s * dt_eff)
        dt_eff_lpf = 0.005 if self._t_prev is None else max(1e-4, t_s - self._t_prev)
        self._k_eff_prev = k_eff
        self._t_prev = t_s
        qdot_fb_raw = k_eff * q_err
        if self._qdot_fb_lpf is None or cfg.fb_lpf_tau_s <= 0.0:
            self._qdot_fb_lpf = qdot_fb_raw.copy()
        else:
            alpha = dt_eff_lpf / (cfg.fb_lpf_tau_s + dt_eff_lpf)
            self._qdot_fb_lpf = self._qdot_fb_lpf + alpha * (qdot_fb_raw - self._qdot_fb_lpf)
        # Scale secondary fb only; primary v_cmd still uses full qdot_fb_raw.
        self.last_qdot_fb = self._qdot_fb_lpf * float(cfg.fb_secondary_gain)
        qdot_cmd = qdot_plan + qdot_fb_raw
        v_lim = np.asarray(self.v_max, dtype=float)
        qdot_cmd = np.clip(qdot_cmd, -v_lim, v_lim)
        v_base = J @ qdot_cmd
        # Soften primary twist near σ or with large residual q_err.
        q_err_deg = float(np.max(np.abs(np.rad2deg(q_err))))
        feas = 1.0
        if cfg.sigma_ref > 1e-9 and sigma_min < cfg.sigma_ref:
            feas = float(
                np.clip(sigma_min / cfg.sigma_ref, cfg.k_joint_sigma_min_frac, 1.0)
            )
        if q_err_deg > 8.0 and sigma_min < cfg.sigma_ref * 1.5:
            feas *= min(1.0, 8.0 / q_err_deg)
        if feas < 1.0:
            v_base = feas * v_base
        pose_ref = self.kin.fk_pose(q_ref)
        err = pose_error(pose_ref, current_pose, cfg.euler_order)
        self.last_err_mm = float(np.linalg.norm(err[:3]) * 1000.0)

        if cfg.control_frame == "tool":
            R = Rsc.from_euler(cfg.euler_order, current_pose[3:6], degrees=False).as_matrix()
            out = np.zeros(6, dtype=float)
            out[:3] = R.T @ v_base[:3]
            out[3:6] = R.T @ v_base[3:6]
            return out
        return v_base


# ---------------------------------------------------------------------------
# On-robot orchestration
# ---------------------------------------------------------------------------
def _set_realtime_priority(priority: int = 80) -> bool:
    """Best-effort SCHED_FIFO for the control thread (needs CAP_SYS_NICE / root)."""
    try:
        param = os.sched_param(priority)
        os.sched_setscheduler(0, os.SCHED_FIFO, param)
        return True
    except (PermissionError, OSError, AttributeError):
        return False


# Spin the last ~1 ms of the period (sleep often wakes 1–3 ms late at 200 Hz).
_SPIN_MARGIN_S = 0.001


def _wait_until(deadline: float) -> None:
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > _SPIN_MARGIN_S:
            time.sleep(remaining - _SPIN_MARGIN_S)


def _resync_late_tick(next_tick: float, now: float, dt: float) -> tuple[float, float]:
    """If we missed a whole period, jump the schedule forward instead of bursting.

    Returns ``(next_tick, late_ms)`` where ``late_ms`` is how far ``now`` was
    past the scheduled tick start (always >= 0).
    """
    late_s = now - next_tick
    if late_s > dt:
        return now, late_s * 1000.0
    return next_tick, max(0.0, late_s * 1000.0)


@dataclass
class LoopResult:
    ticks: int
    duration_s: float
    max_jitter_ms: float
    stalled: bool
    stutter_count: int = 0
    stop_reason: str = ""


@dataclass
class Phase:
    """One leg of a multi-phase on-robot run (shared inner loop / watchdog).

    ``t_ref`` advances by ``dt * governor_scale``; qdot_ff is sampled at the
    same governed ``t_ref``. Set ``governor_err_max_mm=0`` to disable Cartesian
    governor (typical for MoveJ-like joint moves).
    """

    outer: OuterLoop
    label: str = ""
    duration_s: float | None = None          # None -> run until wait_until (or max_duration_s)
    max_duration_s: float | None = None      # wall-clock safety cap
    wait_until: object | None = None         # Callable pose or (pose, q_meas) -> bool
    qdot_ff_provider: object | None = None   # Callable[[float], qdot_ff_rad_s] sampled at t_ref
    scale_qdot_ff_with_governor: bool = True # False keeps plan-anchor alive when t_ref frozen
    require_arrival: bool = False            # abort later phases if wait_until never fires
    governor_err_ok_mm: float = 5.0
    governor_err_max_mm: float = 25.0
    # Joint-space governor: enable with governor_joint_err_max_deg > 0.
    governor_joint_err_ok_deg: float = 3.0
    governor_joint_err_max_deg: float = 0.0
    governor_tau_s: float = 0.2
    governor_freeze_below: float = 0.02
    governor_release_above: float = 0.10
    soft_start_ramp_s: float = 0.0           # governor soft-start at phase entry (s)
    force_observer: object | None = None     # None -> reuse the loop-level force_observer
    on_enter: object | None = None           # Callable[[], None], fired right after set_origin
    on_exit: object | None = None            # Callable[[], None], fired when phase completes
    on_tick: object | None = None            # Callable[[float, JointIkStep, np.ndarray], None]


class _TickLogger:
    """Async per-tick CSV telemetry (background writer; no sync flush in the RT loop)."""

    _HEADER = (
        ["t_wall_s", "phase", "controller_mode", "t_ref_s"]
        + [f"q_cmd_{i}" for i in range(0, 8)]
        + [f"q_meas_{i}" for i in range(0, 8)]
        + [f"pose_{a}" for a in ("x", "y", "z", "rx", "ry", "rz")]
        # twist_* = deprecated alias of twist_requested_*; achieved = J(q)qdot.
        + [f"twist_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + [f"twist_requested_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + [f"twist_achieved_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + ["track_err_mm", "follow_err_deg", "slack_norm", "n_cbf",
           "vel_clamped", "acc_clamped", "pos_clamped", "fx", "fy", "fz",
           "instability_idx", "instability_idx_raw", "instability_idx_active",
           "damping_z_eff",
           "damping_ke_z", "damping_dimeas_z",
           "v_force_z", "ke_est",
           "f_des_z_eff", "v_r_z",
           "force_reference_scale_n", "force_reference_drive",
           "force_reference_gate_scale",
           "force_reference_accel_m_s2",
           "force_reference_reversal_reset",
           "mass_z_eff", "takeover",
           "dt_actual_s", "sensor_age_s",
           "fx_raw_comp", "fy_raw_comp", "fz_raw_comp",
           "vz_achieved_tool", "contact_present",
           "force_pred_z", "force_dot_z", "cap_press_z", "cap_retract_z",
           "ke_update_gated", "ke_dx_m", "ke_df_n", "ke_update_count",
           "governor_scale", "governor_scale_raw", "sigma_min",
           "qdot_norm", "qdot_max_frac_vmax",
           "qdot_ff_norm", "arm_singularity_smooth", "limit_activation",
           "tcp_jump_mm",
           "rail_ext_err_m", "rail_ext_w",
           "rail_target_sent_m", "rail_meas_m",
           "rail_vel_pin", "plan_drives_rail", "rail_qdot_ff"]
    )

    def __init__(self, path: str) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._run,
            args=(path,),
            name="joint-admittance-csv",
            daemon=True,
        )
        self._worker.start()

    def _run(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self._HEADER)
            n = 0
            while True:
                if self._stop.is_set() and self._q.empty():
                    break
                try:
                    row = self._q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if row is None:
                    break
                w.writerow(row)
                n += 1
                if n % 200 == 0:
                    f.flush()

    def write(
        self,
        t_wall,
        label,
        t_ref,
        step: JointIkStep,
        q_meas,
        pose,
        f_ext,
        outer=None,
        *,
        governor_scale: float = float("nan"),
        governor_scale_raw: float = float("nan"),
        v_max: np.ndarray | None = None,
        rail_meas_m: float = float("nan"),
        dt_actual_s: float = float("nan"),
        sensor_age_s: float = float("nan"),
        f_ext_raw: np.ndarray | None = None,
        twist_achieved_base: np.ndarray | None = None,
        v_tcp_z_actual: float = float("nan"),
    ) -> None:
        qm = q_meas if q_meas is not None else np.full(8, np.nan)
        ctrl = getattr(outer, "controller", None)
        is_idx = getattr(ctrl, "instability_index", float("nan"))
        is_idx_raw = getattr(ctrl, "instability_index_raw", float("nan"))
        d_eff = getattr(ctrl, "damping_z_eff", float("nan"))
        d_ke = getattr(ctrl, "damping_ke_z", float("nan"))
        d_dimeas = getattr(ctrl, "damping_dimeas_z", float("nan"))
        v_fz = getattr(ctrl, "v_force_z", float("nan"))
        ke_est = getattr(ctrl, "ke_est", float("nan"))
        f_des_eff = getattr(ctrl, "f_des_z_eff", float("nan"))
        v_r_z = getattr(ctrl, "v_r_z", float("nan"))
        force_reference_scale = getattr(
            ctrl, "force_reference_scale_n", float("nan")
        )
        force_reference_drive = getattr(
            ctrl, "force_reference_drive", float("nan")
        )
        force_reference_gate = getattr(
            ctrl, "force_reference_gate_scale", float("nan")
        )
        force_reference_accel = getattr(
            ctrl, "force_reference_accel_m_s2", float("nan")
        )
        force_reference_reversal_reset = getattr(
            ctrl, "force_reference_reversal_reset", False
        )
        mass_z_eff = getattr(ctrl, "mass_z_eff", float("nan"))
        takeover = getattr(ctrl, "takeover_active", False)
        contact_present = getattr(ctrl, "contact_present", False)
        cap_press_z = getattr(ctrl, "cap_press_z", float("nan"))
        cap_retract_z = getattr(ctrl, "cap_retract_z", float("nan"))
        force_pred_z = getattr(ctrl, "force_pred_z", float("nan"))
        force_dot_z = getattr(ctrl, "force_dot_z", float("nan"))
        ke_tracker = getattr(ctrl, "_ke_estimator", None)
        ke_update_gated = getattr(ke_tracker, "update_gated", False)
        ke_dx_m = getattr(ke_tracker, "last_dx_m", float("nan"))
        ke_df_n = getattr(ke_tracker, "last_df_n", float("nan"))
        ke_update_count = getattr(ke_tracker, "update_count", 0)
        raw_comp = (
            np.asarray(f_ext_raw, dtype=float)
            if f_ext_raw is not None
            else np.full(6, np.nan)
        )
        twist_achieved = (
            np.asarray(twist_achieved_base, dtype=float)
            if twist_achieved_base is not None
            else np.full(6, np.nan)
        )
        qdot_norm = float(np.linalg.norm(step.qdot))
        # Max |qdot|/v_max (1.0 = saturated on at least one joint).
        if v_max is not None and np.any(v_max > 1e-9):
            qdot_max_frac = float(np.max(np.abs(step.qdot) / np.maximum(v_max, 1e-9)))
        else:
            qdot_max_frac = float("nan")
        rail_sent = float(step.q_send[0]) if step.q_send is not None else float("nan")
        self._q.put(
            [
                f"{t_wall:.4f}",
                label,
                str(getattr(ctrl, "controller_mode", "none")),
                f"{t_ref:.4f}",
            ]
            + [f"{v:.6f}" for v in step.q_send]
            + [f"{v:.6f}" for v in qm]
            + [f"{v:.6f}" for v in pose]
            + [f"{v:.5f}" for v in step.twist_base]
            + [f"{v:.5f}" for v in step.twist_base]
            + [f"{v:.5f}" for v in twist_achieved]
            + [f"{step.cart_err_mm:.3f}", f"{np.degrees(step.follow_err_rad):.4f}",
               f"{step.slack_norm:.5f}", step.n_cbf_active,
               int(step.vel_clamped), int(step.acc_clamped), int(step.pos_clamped),
               f"{f_ext[0]:.3f}", f"{f_ext[1]:.3f}", f"{f_ext[2]:.3f}",
               f"{is_idx:.4f}", f"{is_idx_raw:.4f}", f"{is_idx:.4f}",
               f"{d_eff:.2f}",
               f"{d_ke:.2f}", f"{d_dimeas:.2f}",
               f"{v_fz:.5f}", f"{ke_est:.1f}",
               f"{f_des_eff:.3f}", f"{v_r_z:.5f}",
               f"{force_reference_scale:.4f}",
               f"{force_reference_drive:.6f}",
               f"{force_reference_gate:.4f}",
               f"{force_reference_accel:.6f}",
               int(bool(force_reference_reversal_reset)),
               f"{mass_z_eff:.4f}",
               int(bool(takeover)),
               f"{dt_actual_s:.6f}", f"{sensor_age_s:.6f}",
               f"{raw_comp[0]:.3f}", f"{raw_comp[1]:.3f}", f"{raw_comp[2]:.3f}",
               f"{v_tcp_z_actual:.6f}", int(bool(contact_present)),
               f"{force_pred_z:.4f}", f"{force_dot_z:.4f}",
               f"{cap_press_z:.6f}", f"{cap_retract_z:.6f}",
               int(bool(ke_update_gated)), f"{ke_dx_m:.8f}", f"{ke_df_n:.5f}",
               int(ke_update_count),
               f"{governor_scale:.4f}", f"{governor_scale_raw:.4f}",
               f"{step.sigma_min:.5f}",
               f"{qdot_norm:.5f}", f"{qdot_max_frac:.4f}",
               f"{step.qdot_ff_norm:.5f}", f"{step.arm_singularity_smooth:.4f}",
               f"{step.limit_activation:.4f}",
               f"{step.tcp_jump_mm:.3f}",
               f"{step.rail_ext_err_m:.5f}", f"{step.rail_ext_weight:.4f}",
               f"{rail_sent:.6f}",
               f"{rail_meas_m:.6f}" if np.isfinite(rail_meas_m) else "",
               f"{step.rail_vel_pin:.6f}" if np.isfinite(step.rail_vel_pin) else "",
               int(bool(step.plan_drives_rail)),
               f"{step.rail_qdot_ff:.6f}" if np.isfinite(step.rail_qdot_ff) else ""]
        )

    def close(self) -> None:
        self._q.put(None)
        self._stop.set()
        self._worker.join(timeout=1.0)


def _expand_q_meas(q_deg_or_rad: np.ndarray, rail_m: float) -> np.ndarray:
    """Realman feedback is 7 arm joints; prepend rail position for 8-DOF FK."""
    q = np.asarray(q_deg_or_rad, dtype=float)
    if q.size >= 8:
        return q[:8]
    if q.size == 7:
        return full_q_from_arm(q, rail_m)
    raise ValueError(f"expected 7 or 8 joint values, got {q.size}")


def _rail_m_for_init(rail_bridge, inner: JointIkController) -> float:
    """Seed WBC ``q_cmd[0]`` from encoder so the first set_target is near reality."""
    if rail_bridge is not None and rail_bridge.enabled:
        return float(rail_bridge.measured_m)
    return float(inner.q_cmd[0])


def _rail_m_for_feedback(rail_bridge, inner: JointIkController) -> float:
    """Rail ``q_meas[0]`` from encoder (not q_cmd); OOB/garbage → fall back to q_cmd."""
    if rail_bridge is None or not getattr(rail_bridge, "enabled", False):
        return float(inner.q_cmd[0])
    try:
        meas = float(rail_bridge.measured_m)
    except Exception:
        return float(inner.q_cmd[0])
    sane = getattr(rail_bridge, "_encoder_sane", None)
    if callable(sane):
        if not sane(meas):
            return float(inner.q_cmd[0])
    elif not (np.isfinite(meas)):
        return float(inner.q_cmd[0])
    return meas


def _joint_plan_err_deg(outer: OuterLoop, t_ref: float, q_meas: np.ndarray) -> float | None:
    """Max |q_ref(t_ref) - q_meas| in deg from the outer loop's joint reference."""
    ref = getattr(outer, "reference", None)
    if ref is None or not hasattr(ref, "sample_q"):
        return None
    q_ref, _ = ref.sample_q(t_ref)
    return max_joint_err_deg(q_meas, q_ref)


def _reference_governor_scale(
    phase: Phase,
    *,
    outer_err_mm: float | None,
    joint_err_deg: float | None,
) -> float:
    """Raw governor scale in [0, 1] (min of active bands); filter in GovernorFilter."""
    scales: list[float] = []

    if phase.governor_joint_err_max_deg > 0.0 and joint_err_deg is not None:
        e0, e1 = phase.governor_joint_err_ok_deg, phase.governor_joint_err_max_deg
        if e1 > e0:
            scales.append(float(np.clip((e1 - joint_err_deg) / (e1 - e0), 0.0, 1.0)))
        else:
            scales.append(1.0)

    if phase.governor_err_max_mm > 0.0 and outer_err_mm is not None:
        e0, e1 = phase.governor_err_ok_mm, phase.governor_err_max_mm
        if e1 > e0:
            scales.append(float(np.clip((e1 - outer_err_mm) / (e1 - e0), 0.0, 1.0)))

    return min(scales) if scales else 1.0


class GovernorFilter:
    """First-order LPF + freeze hysteresis on the governor scale."""

    def __init__(
        self,
        tau_s: float = 0.2,
        freeze_below: float = 0.02,
        release_above: float = 0.10,
    ) -> None:
        self.tau_s = float(tau_s)
        self.freeze_below = float(freeze_below)
        self.release_above = float(release_above)
        self.scale = 1.0
        self.frozen = False

    def update(self, raw: float, dt: float) -> float:
        raw = float(np.clip(raw, 0.0, 1.0))
        alpha = 1.0 if self.tau_s <= 0.0 else min(1.0, dt / self.tau_s)
        self.scale += alpha * (raw - self.scale)
        if self.frozen:
            if raw >= self.release_above and self.scale >= self.release_above:
                self.frozen = False
        elif self.scale <= self.freeze_below:
            self.frozen = True
        return 0.0 if self.frozen else self.scale


def _send_joint_canfd_cmd(robot, q_deg, follow: bool, canfd_proxy=None) -> None:
    from rm75_control.motion.canfd import send_joint_canfd

    q = np.asarray(q_deg, dtype=float).reshape(-1)[:7]
    if canfd_proxy is not None:
        canfd_proxy.write(q, follow=follow)
        return
    if robot is None:
        raise RuntimeError("no robot handle and no CANFD proxy configured")
    send_joint_canfd(robot, list(q), follow=follow)


def run_joint_admittance_phases(
    session,
    phases: list[Phase],
    inner: JointIkController,
    *,
    q_start_deg: np.ndarray | None = None,
    dt: float | None = None,
    force_observer=None,
    follow: bool = True,
    move_speed: int = 20,
    realtime: bool = False,
    watchdog_timeout_s: float = 0.1,
    on_step=None,
    log_csv: str | None = None,
    verbose: bool = True,
    state_bus=None,
    canfd_proxy=None,
    stop_check=None,
    rail_bridge=None,
) -> LoopResult:
    """Run ``Phase`` objects on the robot as one continuous CANFD stream."""
    from rm75_control.control.admittance_common.state_bus import RobotStateBus

    dt = inner.cfg.dt if dt is None else dt
    robot = session.robot

    if q_start_deg is not None:
        if robot is None:
            raise RuntimeError("q_start_deg move_j requires a local robot session")
        session.move_joints(list(np.asarray(q_start_deg, dtype=float)), velocity_percent=move_speed, block=1)
        time.sleep(0.5)

    own_bus = state_bus is None
    if own_bus:
        state_bus = RobotStateBus(robot, session.config, robot_ip=session.ip)
        state_bus.start()
    async_obs = state_bus.observer
    if verbose and own_bus:
        print(
            f"  feedback: UDP push {async_obs.push_period_ms:.0f}ms "
            f"port={async_obs.config.port} ip={async_obs._target_ip}",
            flush=True,
        )
    ticks = 0
    max_jitter_ms = 0.0
    stutter_count = 0
    stalled = False
    total_t0 = time.perf_counter()
    logger = _TickLogger(log_csv) if log_csv else None
    try:
        _pose0_rm = async_obs.wait_first_pose(timeout_s=5.0)
        snap0 = async_obs.read()
        if snap0.q_deg is None:
            raise RuntimeError("no joint feedback from robot")
        q0_rad = _expand_q_meas(
            deg2rad(snap0.q_deg),
            _rail_m_for_init(rail_bridge, inner),
        )
        # Cartesian loop uses Pinocchio TCP (may differ from RealMan FK).
        pose0 = inner.kin.fk_pose(q0_rad)
        inner.reset(q0_rad)

        if realtime and not _set_realtime_priority():
            if verbose:
                print("  (SCHED_FIFO unavailable - running at normal priority)", flush=True)

        def _hold() -> None:
            # watchdog stall action: hold at the last commanded joint state
            try:
                _send_joint_canfd_cmd(
                    robot,
                    rad2deg(arm_q_from_full(inner.q_cmd)),
                    False,
                    canfd_proxy,
                )
            except Exception:
                if robot is not None:
                    try:
                        robot.rm_set_arm_slow_stop()
                    except Exception:
                        pass

        wd = Watchdog(watchdog_timeout_s, _hold)
        wd.start()
        try:
            pose_rm = _pose0_rm
            q_meas = q0_rad
            pose_pin = pose0
            jump_warn_t = 0.0
            phase_stopped = False
            stop_reason = ""
            try:
                for phase_idx, phase in enumerate(phases):
                    if stop_check is not None and stop_check():
                        phase_stopped = True
                        if verbose:
                            print("  stopped by external request", flush=True)
                        break
                    if verbose:
                        print(f"-- phase: {phase.label or phase.outer.__class__.__name__} --", flush=True)
                    # Phase origin from encoders (never from the command integrator).
                    snap = async_obs.read()
                    if snap.q_deg is not None:
                        rail_seed = _rail_m_for_init(rail_bridge, inner)
                        q_meas = _expand_q_meas(deg2rad(snap.q_deg), rail_seed)
                    pose_pin = inner.kin.fk_pose(q_meas)
                    # Soft-start: reseed plan from live encoders (no tick-0 lurch).
                    ref = getattr(phase.outer, "reference", None)
                    if ref is not None:
                        try:
                            q_live = np.asarray(q_meas, dtype=float).reshape(-1)
                            if hasattr(ref, "reseed_start"):
                                ref.reseed_start(q_live)
                            elif hasattr(ref, "q_start") and hasattr(ref, "q_target"):
                                if q_live.size == int(np.asarray(ref.q_start).size):
                                    ref.q_start = q_live.copy()
                        except Exception:
                            pass
                    if hasattr(phase.outer, "set_origin"):
                        phase.outer.set_origin(pose_pin)
                    if phase.on_enter is not None:
                        phase.on_enter()

                    obs = phase.force_observer if phase.force_observer is not None else force_observer
                    phase_t0 = time.perf_counter()
                    next_tick = phase_t0
                    last_tick_time = phase_t0
                    t_ref = 0.0
                    gov_filter = GovernorFilter(
                        tau_s=phase.governor_tau_s,
                        freeze_below=phase.governor_freeze_below,
                        release_above=phase.governor_release_above,
                    )
                    scale = 1.0
                    phase_arrived = False
                    prev_pose_cmd = inner.kin.fk_pose(inner.q_cmd)
                    # Encoder TCP velocity: update only on a fresh UDP sequence.
                    last_feedback_seq = int(getattr(snap, "seq", 0))
                    last_feedback_t = float(getattr(snap, "t_s", 0.0))
                    last_feedback_q = np.asarray(q_meas, dtype=float).copy()
                    twist_achieved_base = np.zeros(6, dtype=float)
                    v_tcp_z_actual = 0.0
                    while True:
                        if stop_check is not None and stop_check():
                            phase_stopped = True
                            break
                        now = time.perf_counter()
                        dt_raw = now - last_tick_time
                        last_tick_time = now
                        # The first phase tick occurs immediately after setup;
                        # use the nominal period rather than a near-zero dt.
                        if dt_raw < 0.002:
                            dt_raw = dt
                        dt_actual = float(np.clip(dt_raw, 0.002, 0.015))
                        next_tick, late_ms = _resync_late_tick(next_tick, now, dt)
                        if late_ms > dt * 1000.0:
                            stutter_count += 1
                        max_jitter_ms = max(max_jitter_ms, late_ms)
                        t_wall = now - phase_t0
                        if phase.duration_s is not None and t_ref >= phase.duration_s:
                            break
                        if phase.max_duration_s is not None and t_wall >= phase.max_duration_s:
                            break
    
                        snap = async_obs.read()
                        if snap.pose is not None:
                            pose_rm = snap.pose
                        if snap.q_deg is not None:
                            q_new = _expand_q_meas(
                                deg2rad(snap.q_deg),
                                _rail_m_for_feedback(rail_bridge, inner),
                            )
                            snap_seq = int(getattr(snap, "seq", 0))
                            snap_t = float(getattr(snap, "t_s", 0.0))
                            if (
                                snap_seq != last_feedback_seq
                                and snap_t > last_feedback_t
                            ):
                                dt_feedback = snap_t - last_feedback_t
                                if 0.001 <= dt_feedback <= 0.050:
                                    qdot_meas = (
                                        wrap_joint_delta(last_feedback_q, q_new)
                                        / dt_feedback
                                    )
                                    twist_achieved_base = (
                                        inner.kin.jacobian(q_new) @ qdot_meas
                                    )
                                    pose_for_velocity = inner.kin.fk_pose(q_new)
                                    r_velocity = Rsc.from_euler(
                                        inner.cfg.euler_order,
                                        pose_for_velocity[3:6],
                                        degrees=False,
                                    ).as_matrix()
                                    v_tcp_z_actual = float(
                                        (r_velocity.T @ twist_achieved_base[:3])[2]
                                    )
                                last_feedback_seq = snap_seq
                                last_feedback_t = snap_t
                                last_feedback_q = q_new.copy()
                            q_meas = q_new
                            pose_pin = inner.kin.fk_pose(q_meas)

                        sensor_age_s = (
                            max(0.0, time.monotonic() - float(snap.t_s))
                            if float(getattr(snap, "t_s", 0.0)) > 0.0
                            else float("inf")
                        )

                        f_ext = np.zeros(6)
                        f_ext_raw = None
                        if obs is not None:
                            pose_l7 = inner.kin.frame_pose(q_meas, "link_7")
                            _signed, f_ext = obs.update(now - total_t0, pose_l7, snap.force_raw)
                            f_ext_raw = getattr(obs, "f_ext_raw_last", None)
                            f_ext = inner.kin.wrench_link7_to_tcp(f_ext)
                            if f_ext_raw is not None:
                                f_ext_raw = inner.kin.wrench_link7_to_tcp(f_ext_raw)
    
                        q_prev = inner.q_cmd.copy()
                        # Pause force integrator with the governed reference clock.
                        if hasattr(phase.outer, "set_time_scale"):
                            phase.outer.set_time_scale(scale)
                        sample_params = inspect.signature(phase.outer.sample).parameters
                        sample_kwargs: dict = {}
                        if "q_meas" in sample_params:
                            sample_kwargs["q_meas"] = q_meas
                        if "f_ext_raw" in sample_params and f_ext_raw is not None:
                            # Unfiltered wrench for Dimeas (6 Hz LPF hides the band).
                            sample_kwargs["f_ext_raw"] = f_ext_raw
                        if "dt_actual" in sample_params:
                            sample_kwargs["dt_actual"] = dt_actual
                        if "v_tcp_z_actual" in sample_params:
                            sample_kwargs["v_tcp_z_actual"] = v_tcp_z_actual
                        if "sensor_age_s" in sample_params:
                            sample_kwargs["sensor_age_s"] = sensor_age_s
                        twist = np.asarray(
                            phase.outer.sample(t_ref, pose_pin, f_ext, **sample_kwargs),
                            dtype=float,
                        )
                        qdot_ff = (
                            phase.qdot_ff_provider(t_ref)
                            if phase.qdot_ff_provider is not None
                            else None
                        )
                        if qdot_ff is not None:
                            qdot_ff = np.asarray(qdot_ff, dtype=float)
                            if phase.scale_qdot_ff_with_governor:
                                qdot_ff = qdot_ff * scale
                        # Additive joint fb (not governor-scaled) closes nullspace q_err.
                        qdot_fb = getattr(phase.outer, "last_qdot_fb", None)
                        if qdot_fb is not None:
                            qdot_fb = np.asarray(qdot_fb, dtype=float)
                            qdot_ff = qdot_fb if qdot_ff is None else (qdot_ff + qdot_fb)
                        vel_ff_ref = getattr(phase.outer, "last_vel_ff", None)
                        control_dt = dt
                        step = inner.update(
                            twist,
                            control_dt,
                            q_meas=q_meas,
                            qdot_ff=qdot_ff,
                            vel_ff=vel_ff_ref,
                        )
                        if rail_bridge is not None:
                            rail_bridge.set_target_m(float(inner.q_cmd[0]))
                        outer_err_mm = getattr(phase.outer, "last_err_mm", None)
                        if outer_err_mm is not None:
                            step.cart_err_mm = outer_err_mm
                        pose_cmd = inner.kin.fk_pose(step.q_send)
                        step.tcp_jump_mm = float(
                            np.linalg.norm(pose_cmd[:3] - prev_pose_cmd[:3]) * 1000.0
                        )
                        if verbose and step.tcp_jump_mm > 8.0 and now - jump_warn_t >= 1.0:
                            jump_warn_t = now
                            print(
                                f"  warn: TCP jump {step.tcp_jump_mm:.1f}mm/tick",
                                flush=True,
                            )
                        prev_pose_cmd = pose_cmd
                        _send_joint_canfd_cmd(
                            robot,
                            rad2deg(arm_q_from_full(step.q_send)),
                            follow,
                            canfd_proxy,
                        )
                        wd.beat()
    
                        # Reference-clock governor: reference waits for the arm.
                        joint_err_deg = getattr(phase.outer, "last_joint_err_deg", None)
                        if joint_err_deg is None:
                            joint_err_deg = _joint_plan_err_deg(phase.outer, t_ref, q_meas)
                        raw_scale = _reference_governor_scale(
                            phase,
                            outer_err_mm=outer_err_mm,
                            joint_err_deg=joint_err_deg,
                        )
                        scale = gov_filter.update(raw_scale, control_dt)
                        # Soft-start ramp: first ~0.3s cannot command near-vmax.
                        ramp_s = float(getattr(phase, "soft_start_ramp_s", 0.0) or 0.0)
                        if ramp_s > 1e-6:
                            scale *= float(np.clip(t_wall / ramp_s, 0.0, 1.0))
                        t_ref += control_dt * scale
    
                        if phase.on_tick is not None:
                            phase.on_tick(t_ref, step, q_meas)
    
                        dq_deg = np.abs(rad2deg(step.q_send - q_prev))
                        if verbose and now - jump_warn_t >= 1.0 and np.any(dq_deg > 1.5):
                            jump_warn_t = now
                            j = int(np.argmax(dq_deg)) + 1
                            print(
                                f"  warn: joint jump J{j} {dq_deg.max():.2f}deg/tick "
                                f"(>{1.5:.1f} @ {dt*1000:.0f}ms)",
                                flush=True,
                            )
    
                        if logger is not None:
                            rail_meas = float("nan")
                            if rail_bridge is not None and rail_bridge.enabled:
                                try:
                                    rail_meas = float(rail_bridge.measured_m)
                                except Exception:
                                    rail_meas = float("nan")
                            logger.write(
                                now - total_t0, phase.label, t_ref, step, q_meas, pose_pin, f_ext,
                                outer=phase.outer,
                                governor_scale=scale,
                                governor_scale_raw=raw_scale,
                                v_max=inner.limits.v_max,
                                rail_meas_m=rail_meas,
                                dt_actual_s=dt_actual,
                                sensor_age_s=sensor_age_s,
                                f_ext_raw=f_ext_raw,
                                twist_achieved_base=twist_achieved_base,
                                v_tcp_z_actual=v_tcp_z_actual,
                            )
                        if on_step is not None:
                            on_step(phase.label, t_ref, step, pose_pin, f_ext, t_wall)

                        if phase.wait_until is not None:
                            n_wait = len(inspect.signature(phase.wait_until).parameters)
                            if n_wait >= 2:
                                phase_arrived = bool(phase.wait_until(pose_pin, q_meas))
                            else:
                                phase_arrived = bool(phase.wait_until(pose_pin))
                            if phase_arrived:
                                break
    
                        ticks += 1
                        next_tick += dt
                        _wait_until(next_tick)

                    if phase.on_exit is not None:
                        phase.on_exit()

                    if phase_stopped:
                        break

                    if phase.require_arrival and not phase_arrived:
                        err_mm = getattr(phase.outer, "last_err_mm", float("nan"))
                        jq = getattr(phase.outer, "last_joint_err_deg", float("nan"))
                        d_mm = d_deg = float("nan")
                        try:
                            pt = getattr(phase, "pose_target", None)
                            if pt is None:
                                ref = getattr(phase.outer, "reference", None)
                                pt = getattr(ref, "pose_d", None) or getattr(ref, "pose_target", None)
                            if pt is not None and q_meas is not None:
                                d_mm, d_deg = pose_distance(
                                    pose_pin, pt, inner.cfg.euler_order
                                )
                        except Exception:
                            pass
                        print(
                            f"  ERROR: phase {phase.label!r} did not reach target "
                            f"(t_ref={t_ref:.2f}s, wall={t_wall:.1f}s, "
                            f"track={err_mm:.0f}mm, poseΔ={d_mm:.1f}mm/{d_deg:.1f}deg, "
                            f"jq={jq:.1f}deg) "
                            f"— skipping remaining phases",
                            flush=True,
                        )
                        break
            except KeyboardInterrupt:
                if verbose:
                    print("\nStopped.", flush=True)
        finally:
            wd.stop()
            stalled = wd.fired
    finally:
        if own_bus:
            state_bus.stop()
        if logger is not None:
            logger.close()

    total_s = time.perf_counter() - total_t0
    if verbose:
        stutter_note = f", {stutter_count} stutter(s)" if stutter_count else ""
        print(
            f"  joint-admittance loop: {ticks} ticks, {total_s:.1f}s, "
            f"max jitter {max_jitter_ms:.2f} ms{stutter_note}"
            f"{' [WATCHDOG FIRED]' if stalled else ''}",
            flush=True,
        )
    return LoopResult(
        ticks=ticks,
        duration_s=total_s,
        max_jitter_ms=max_jitter_ms,
        stalled=stalled,
        stutter_count=stutter_count,
        stop_reason=stop_reason,
    )


def run_joint_admittance_loop(
    session,
    outer: OuterLoop,
    inner: JointIkController,
    *,
    q_start_deg: np.ndarray | None = None,
    duration_s: float = 10.0,
    dt: float | None = None,
    force_observer=None,
    follow: bool = True,
    move_speed: int = 20,
    realtime: bool = False,
    watchdog_timeout_s: float = 0.1,
    on_step=None,
    log_csv: str | None = None,
    verbose: bool = True,
    state_bus=None,
    rail_bridge=None,
) -> LoopResult:
    """Single-phase convenience wrapper around ``run_joint_admittance_phases``."""
    phase = Phase(
        outer=outer,
        label="run",
        duration_s=duration_s,
    )
    on_step_1 = None if on_step is None else (lambda label, t, step, pose, f_ext: on_step(t, step, pose, f_ext))
    return run_joint_admittance_phases(
        session,
        [phase],
        inner,
        q_start_deg=q_start_deg,
        dt=dt,
        force_observer=force_observer,
        follow=follow,
        move_speed=move_speed,
        realtime=realtime,
        watchdog_timeout_s=watchdog_timeout_s,
        on_step=on_step_1,
        log_csv=log_csv,
        verbose=verbose,
        state_bus=state_bus,
        rail_bridge=rail_bridge,
    )
```

## `rm75_control/control/joint_admittance_8dof/api.py`

```python
"""Phase factories + compile: JointPhaseSpec → runtime Phase for the 8-DOF loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal

import numpy as np

from rm75_control.control.admittance_common.controller import AdmittanceController
from rm75_control.control.admittance_common.reference import MotionReferenceSource
from rm75_control.control.joint_admittance_8dof.loop import (
    AdmittanceOuterLoop,
    CartesianTrackConfig,
    CartesianTrackOuterLoop,
    JointIkController,
    JointTrackConfig,
    JointTrackOuterLoop,
    Phase,
)
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import _wrap_pi
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    auto_move_duration_s,
    pose_distance,
    wrap_joint_delta,
)
from rm75_control.control.joint_admittance_8dof.reference import (
    HoldReference,
    JointSmoothMoveReference,
    RailSmoothMoveReference,
    SrsSmoothMoveReference,
    auto_rail_move_duration_s,
    srs_move_duration_s,
)


class TaskMode(str, Enum):
    JOINT_RESET = "joint_reset"
    CARTESIAN_GOTO = "cartesian_goto"
    CARTESIAN_TRACK = "cartesian_track"
    HYBRID_TRACK = "hybrid_track"
    # LOCKED_MOVE == plan drives the rail while the top-level mode is LOCKED;
    # the substyle (RAIL_ONLY vs TCP_FIXED) is carried on JointPhaseSpec.
    LOCKED_MOVE = "locked_move"


@dataclass
class ArmAngleSpec:
    """Arm-angle nullspace target applied on phase entry (scan/handoff)."""

    psi_rad: float | None = None


@dataclass
class SecondaryPolicy:
    """Nullspace / secondary-task preset: off | move | track | hold."""

    preset: Literal["off", "move", "track", "hold"] = "track"
    arm_angle: ArmAngleSpec | None = None
    qdot_ff: Literal["off", "plan", "plan_joint"] = "off"

    def _set_arm_angle_reference(
        self,
        inner: JointIkController,
        psi_rad: float | None,
    ) -> None:
        if psi_rad is None or inner.arm_task is None:
            return
        psi_live = float(inner.arm_task.arm_angle(inner.q_cmd))
        psi_set = float(psi_live + _wrap_pi(float(psi_rad) - psi_live))
        inner.arm_task.set_reference(psi_set)

    def apply(self, inner: JointIkController, *, psi_rad: float | None = None) -> None:
        psi = psi_rad
        if self.arm_angle is not None and self.arm_angle.psi_rad is not None:
            psi = self.arm_angle.psi_rad

        if self.preset == "move":
            # Plan owns posture; suppress secondary fights with the planner.
            inner.set_coupled()
            inner.set_arm_task_suppressed(True)
            inner.set_centering_suppressed(True)
            inner.set_manipulability_active(False)
            inner.set_rail_extension_mode("pose_attract")
            inner.set_rail_extension_active(True)
        elif self.preset == "track":
            inner.set_plan_drives_rail(False)
            inner.set_manipulability_active(False)
            inner.set_centering_suppressed(False)
            inner.set_arm_task_suppressed(False)
            # Restore yaml rail mode (live cfg.rail.mode is mutated by locks).
            if inner.configured_rail_mode == RailMode.COUPLED:
                inner.set_coupled()
                inner.set_rail_extension_mode("reach")
                inner.capture_rail_extension_ref()
                inner.set_rail_extension_active(True)
            else:
                inner.set_locked(
                    LockedStyle.HOLD, q_ref_m=float(inner.q_cmd[0])
                )
                inner.set_rail_extension_active(False)
            if psi is not None and inner.arm_task is not None:
                self._set_arm_angle_reference(inner, psi)
        elif self.preset == "hold":
            inner.set_plan_drives_rail(False)
            inner.set_manipulability_active(False)
            # Suppress centering (fights teach/force hold); keep arm_angle branch.
            inner.set_centering_suppressed(True)
            inner.set_arm_task_suppressed(False)
            # Pin rail at current q_cmd[0] (never yaml q_ref_m=0.0).
            inner.set_locked(LockedStyle.HOLD, q_ref_m=float(inner.q_cmd[0]))
            inner.set_rail_extension_active(False)
            if psi is not None and inner.arm_task is not None:
                self._set_arm_angle_reference(inner, psi)
        elif self.preset == "off":
            inner.set_arm_task_suppressed(True)
            inner.set_centering_suppressed(True)
            inner.set_manipulability_active(False)
            inner.set_rail_extension_active(False)

    def make_qdot_ff_provider(
        self,
        inner: JointIkController,
        move_ref: JointSmoothMoveReference | SrsSmoothMoveReference | None,
    ) -> Callable[[float], np.ndarray] | None:
        if self.qdot_ff == "off" or move_ref is None:
            return None
        if self.qdot_ff == "plan":
            return lambda t: move_ref.sample_q(t)[1]
        if self.qdot_ff == "plan_joint":

            def _joint_ff(t: float) -> np.ndarray:
                q_plan, dq_plan = move_ref.sample_q(t)
                return dq_plan + 1.0 * wrap_joint_delta(inner.q_cmd, q_plan)

            return _joint_ff
        return None


@dataclass
class GovernorSpec:
    err_ok_mm: float = 5.0
    err_max_mm: float = 25.0
    joint_err_ok_deg: float = 3.0
    joint_err_max_deg: float = 0.0
    tau_s: float = 0.2
    freeze_below: float = 0.02
    release_above: float = 0.10


@dataclass
class JointPhaseSpec:
    mode: TaskMode
    label: str = ""
    secondary: SecondaryPolicy = field(default_factory=SecondaryPolicy)
    governor: GovernorSpec = field(default_factory=GovernorSpec)
    duration_s: float | None = None
    max_duration_s: float | None = None
    wait_until: Callable[..., bool] | None = None
    require_arrival: bool = False
    force_observer: Any = None
    scale_qdot_ff_with_governor: bool = True
    # Move / goto
    move_ref: JointSmoothMoveReference | SrsSmoothMoveReference | None = None
    pose_target: np.ndarray | None = None
    q_target_rad: np.ndarray | None = None
    move_kp: float = 2.0
    move_mode: Literal["joint", "cartesian"] = "cartesian"
    max_lin_vel_m_s: float = 0.4
    sigma_ref: float = 0.08
    # Track / hybrid
    reference: MotionReferenceSource | None = None
    controller: AdmittanceController | None = None
    desired_force: np.ndarray | None = None
    # Locked-move (LOCKED + RAIL_ONLY / TCP_FIXED): external plan drives rail
    rail_ref: RailSmoothMoveReference | None = None
    locked_style: LockedStyle = LockedStyle.RAIL_ONLY
    q_rail_target_m: float | None = None


@dataclass
class CompileContext:
    kin: RobotKinematics
    inner: JointIkController
    euler_order: str = "xyz"
    control_frame: str = "tool"
    v_scale: float = 0.5


@dataclass
class CompiledPhase:
    phase: Phase
    label: str
    outer: Any = None
    move_ref: JointSmoothMoveReference | SrsSmoothMoveReference | None = None
    rail_ref: RailSmoothMoveReference | None = None
    reference: MotionReferenceSource | None = None


def make_srs_move_reference(
    kin: RobotKinematics,
    q_start_rad: np.ndarray,
    pose_target: np.ndarray,
    q_target_rad: np.ndarray,
    duration_s: float,
    *,
    euler_order: str = "xyz",
) -> SrsSmoothMoveReference:
    """Build a branch-locked SRS move reference (plan pose = FK(q_target))."""
    from rm75_control.kinematics.srs_ik import d_wt_from_kin, psi_from_q

    q_start = np.asarray(q_start_rad, dtype=float)
    q_target = np.asarray(q_target_rad, dtype=float)
    # Plan uses live FK(q_target), not a cached pose_target.
    pose_from_q = np.asarray(kin.fk_pose(q_target), dtype=float).reshape(6)
    v_max = kin.v_max * 0.5  # match inner v_scale default
    T_rate = srs_move_duration_s(q_start, q_target, max_qdot_rad_s=v_max)
    T = max(float(duration_s), T_rate)
    d_wt = float(d_wt_from_kin(kin))
    return SrsSmoothMoveReference(
        kin,
        q_start,
        pose_from_q,
        y_rail_target_m=float(q_target[0]),
        psi_target_rad=float(psi_from_q(q_target[1:])),
        duration_s=T,
        euler_order=euler_order,
        d_wt=d_wt,
    )


def attach_joint_move_rail(
    phase: Phase,
    inner: JointIkController,
) -> None:
    """Pin rail to the joint plan and enable direct joint PTP (no Cartesian QP)."""
    prev_on_enter = phase.on_enter
    prev_on_exit = phase.on_exit

    def _enter() -> None:
        if prev_on_enter is not None:
            prev_on_enter()
        inner.set_rail_extension_active(False)
        inner.set_plan_drives_rail(True)
        inner.set_direct_joint_ptp(True)

    def _exit() -> None:
        inner.set_direct_joint_ptp(False)
        inner.set_plan_drives_rail(False)
        if prev_on_exit is not None:
            prev_on_exit()

    phase.on_enter = _enter
    phase.on_exit = _exit


def attach_srs_move_tracking(
    phase: Phase,
    inner: JointIkController,
    move_ref: SrsSmoothMoveReference,
    q_target_rad: np.ndarray,
) -> None:
    """Wire ψ_ref(t) + centering for SRS move; pin rail to y_rail(t) plan."""
    q_target = np.asarray(q_target_rad, dtype=float)
    prev_on_enter = phase.on_enter
    prev_on_tick = phase.on_tick
    prev_on_exit = phase.on_exit

    def _enter() -> None:
        if prev_on_enter is not None:
            prev_on_enter()
        if not inner._centering_suppressed:
            inner.centering_task.set_q_target(q_target)
        if inner.arm_task is not None and not inner._arm_task_suppressed:
            inner.arm_task.set_reference(move_ref.psi_start)
        inner.set_plan_drives_rail(True)

    def _tick(t_ref: float, step, q_meas: np.ndarray) -> None:
        if inner.arm_task is not None and not inner._arm_task_suppressed:
            inner.arm_task.set_reference(move_ref.sample_psi(t_ref))
        if prev_on_tick is not None:
            prev_on_tick(t_ref, step, q_meas)

    def _exit() -> None:
        inner.set_plan_drives_rail(False)
        # Restore yaml posture attractor (do not keep D-point as scan target).
        inner.centering_task.set_q_target(None)
        if prev_on_exit is not None:
            prev_on_exit()

    phase.on_enter = _enter
    phase.on_tick = _tick
    phase.on_exit = _exit


@dataclass
class MovePlan:
    duration_s: float
    move_mode: Literal["joint", "cartesian"]
    gov_joint_max_deg: float
    meta: dict


def compute_move_plan(
    kin: RobotKinematics,
    q0_rad: np.ndarray,
    q_target_rad: np.ndarray,
    pose_target: np.ndarray,
    *,
    v_scale: float,
    duration_s: float | None = None,
    move_mode: Literal["joint", "cartesian"] = "joint",
    peak_joint_v_frac: float = 0.80,
    max_lin_vel_m_s: float = 0.4,
    duration_min_s: float = 2.5,
    duration_max_s: float = 20.0,
    approach_dz_m: float | None = None,
    sigma_ref: float = 0.08,
    euler_order: str = "xyz",
) -> MovePlan:
    """Duration and joint governor cap for an explicit PTP mode (no auto-switch)."""
    auto_duration, meta = auto_move_duration_s(
        kin,
        q0_rad,
        q_target_rad,
        pose_target,
        v_scale=v_scale,
        v_max_rad_s=kin.v_max,
        peak_joint_v_frac=peak_joint_v_frac,
        max_lin_vel_m_s=max_lin_vel_m_s,
        duration_min_s=duration_min_s,
        duration_max_s=duration_max_s,
        approach_dz_m=approach_dz_m,
        sigma_ref=sigma_ref,
        euler_order=euler_order,
    )
    max_dq_deg = float(meta["max_dq_deg"])
    gov_joint_max_deg = float(np.clip(1.15 * max_dq_deg, 25.0, 90.0))
    duration = float(duration_s) if duration_s is not None else auto_duration
    meta["user_override"] = duration_s is not None
    return MovePlan(
        duration_s=duration,
        move_mode=move_mode,
        gov_joint_max_deg=gov_joint_max_deg,
        meta=meta,
    )


def make_move_arrived(
    pose_target: np.ndarray,
    q_target_rad: np.ndarray,
    *,
    tol_mm: float = 3.0,
    tol_deg: float = 1.5,
    joint_tol_deg: float = 3.0,
    rail_tol_mm: float = 5.0,
    joint_only: bool = False,
    require_joints: bool = True,
    euler_order: str = "xyz",
) -> Callable[[np.ndarray, np.ndarray], bool]:
    """Arrival gate for move→D (pose and/or joint tolerances)."""

    def _joints_ok(q_meas: np.ndarray) -> bool:
        qa = np.asarray(q_meas, dtype=float).reshape(-1)
        qt = np.asarray(q_target_rad, dtype=float).reshape(-1)
        n = int(min(qa.size, qt.size))
        if n < 1:
            return False
        if abs(float(qa[0]) - float(qt[0])) * 1000.0 > float(rail_tol_mm):
            return False
        if n > 1:
            from rm75_control.control.joint_admittance_8dof.model import wrap_joint_delta

            d = wrap_joint_delta(qa[:n], qt[:n])
            arm_err = float(np.rad2deg(np.max(np.abs(d[1:]))))
            if arm_err > float(joint_tol_deg):
                return False
        return True

    def _fn(pose_meas: np.ndarray, q_meas: np.ndarray) -> bool:
        if joint_only:
            return _joints_ok(q_meas)
        d_mm, d_deg = pose_distance(pose_meas, pose_target, euler_order)
        if d_mm > tol_mm or d_deg > tol_deg:
            return False
        if not require_joints:
            return True
        return _joints_ok(q_meas)

    return _fn


def make_rail_arrived(
    q_target_m: float,
    *,
    tol_mm: float = 0.5,
) -> Callable[[np.ndarray, np.ndarray], bool]:
    def _fn(pose_meas: np.ndarray, q_meas: np.ndarray) -> bool:
        del pose_meas
        return abs(float(q_meas[0]) - float(q_target_m)) * 1000.0 <= tol_mm

    return _fn


def phase_rail_reposition(
    q_target_m: float,
    q_start_rad: np.ndarray,
    kin: RobotKinematics,
    *,
    label: str = "rail_reposition",
    style: LockedStyle | str = LockedStyle.RAIL_ONLY,
    duration_s: float | None = None,
    max_duration_s: float | None = None,
    require_arrival: bool = True,
    force_observer: Any = None,
    v_max_m_s: float | None = None,
) -> JointPhaseSpec:
    """Smoothstep rail_y to ``q_target_m`` (RAIL_ONLY or TCP_FIXED)."""
    if isinstance(style, str):
        style = LockedStyle(style)
    if style not in (LockedStyle.RAIL_ONLY, LockedStyle.TCP_FIXED):
        raise ValueError(
            f"phase_rail_reposition style must be RAIL_ONLY or TCP_FIXED, got {style}"
        )
    q_start = np.asarray(q_start_rad, dtype=float)
    rail_v = float(v_max_m_s if v_max_m_s is not None else kin.v_max[0])
    if duration_s is None:
        duration_s = auto_rail_move_duration_s(
            float(q_start[0]),
            float(q_target_m),
            v_max_m_s=rail_v,
            peak_v_frac=1.0,
        )
    rail_ref = RailSmoothMoveReference(q_start, float(q_target_m), float(duration_s))
    # Suppress secondary posture tasks during rail reposition.
    sec = SecondaryPolicy(preset="off", qdot_ff="plan")
    return JointPhaseSpec(
        mode=TaskMode.LOCKED_MOVE,
        label=label,
        rail_ref=rail_ref,
        q_rail_target_m=float(q_target_m),
        locked_style=style,
        duration_s=float(duration_s),
        max_duration_s=max_duration_s,
        require_arrival=require_arrival,
        force_observer=force_observer,
        secondary=sec,
        governor=GovernorSpec(err_max_mm=0.0),
        scale_qdot_ff_with_governor=False,
        wait_until=make_rail_arrived(q_target_m),
        move_kp=2.0 if style == LockedStyle.TCP_FIXED else 0.0,
        max_lin_vel_m_s=0.10 if style == LockedStyle.TCP_FIXED else 0.4,
    )


def phase_hold_at_pose(
    duration_s: float,
    *,
    label: str = "hold",
    move_kp: float = 1.0,
    force_observer: Any = None,
) -> JointPhaseSpec:
    """Hold current TCP pose for ``duration_s`` (rail locked via preset hold)."""
    return JointPhaseSpec(
        mode=TaskMode.CARTESIAN_TRACK,
        label=label,
        reference=HoldReference(),
        duration_s=float(duration_s),
        move_kp=float(move_kp),
        force_observer=force_observer,
        secondary=SecondaryPolicy(preset="hold", qdot_ff="off"),
        governor=GovernorSpec(err_ok_mm=15.0, err_max_mm=80.0),
    )


def phase_cartesian_goto(
    move_ref: JointSmoothMoveReference | SrsSmoothMoveReference,
    *,
    label: str = "cartesian_goto",
    pose_target: np.ndarray | None = None,
    q_target_rad: np.ndarray | None = None,
    move_kp: float = 2.0,
    move_mode: Literal["joint", "cartesian"] = "cartesian",
    max_lin_vel_m_s: float = 0.4,
    max_duration_s: float | None = None,
    gov_joint_max_deg: float = 25.0,
    require_arrival: bool = True,
    force_observer: Any = None,
) -> JointPhaseSpec:
    sec = SecondaryPolicy(
        preset="move",
        qdot_ff="plan_joint",
    )
    gov = (
        GovernorSpec(
            err_max_mm=0.0,
            joint_err_ok_deg=12.0,
            joint_err_max_deg=max(float(gov_joint_max_deg), 60.0),
        )
        if move_mode == "joint"
        else GovernorSpec(
            err_ok_mm=10.0,
            err_max_mm=60.0,
            joint_err_ok_deg=5.0,
            joint_err_max_deg=0.0,
        )
    )
    return JointPhaseSpec(
        mode=TaskMode.CARTESIAN_GOTO if move_mode == "cartesian" else TaskMode.JOINT_RESET,
        label=label,
        move_ref=move_ref,
        pose_target=pose_target,
        q_target_rad=q_target_rad,
        move_kp=move_kp,
        move_mode=move_mode,
        max_lin_vel_m_s=max_lin_vel_m_s,
        max_duration_s=max_duration_s,
        require_arrival=require_arrival,
        force_observer=force_observer,
        secondary=sec,
        governor=gov,
        scale_qdot_ff_with_governor=False,
        wait_until=(
            make_move_arrived(
                pose_target,
                q_target_rad,
                joint_only=(move_mode == "joint"),
                # Cartesian/SRS: TCP pose is the goal; joint residual is OK.
                require_joints=(move_mode == "joint"),
                tol_mm=5.0 if move_mode == "cartesian" else 3.0,
                tol_deg=3.0 if move_mode == "cartesian" else 1.5,
            )
            if pose_target is not None and q_target_rad is not None
            else None
        ),
    )


def phase_hybrid_track(
    reference: MotionReferenceSource,
    controller: AdmittanceController,
    *,
    desired_force: np.ndarray,
    label: str = "hybrid_track",
    duration_s: float | None = None,
    force_observer: Any = None,
    psi_rad_on_enter: float | None = None,
    governor: GovernorSpec | None = None,
    secondary: SecondaryPolicy | None = None,
) -> JointPhaseSpec:
    sec = secondary or SecondaryPolicy(preset="track", qdot_ff="off")
    if psi_rad_on_enter is not None and sec.arm_angle is None:
        sec.arm_angle = ArmAngleSpec(psi_rad=psi_rad_on_enter)
    return JointPhaseSpec(
        mode=TaskMode.HYBRID_TRACK,
        label=label,
        reference=reference,
        controller=controller,
        desired_force=np.asarray(desired_force, dtype=float),
        duration_s=duration_s,
        force_observer=force_observer,
        secondary=sec,
        governor=governor or GovernorSpec(err_ok_mm=10.0, err_max_mm=40.0),
    )


def _make_on_enter(spec: JointPhaseSpec, ctx: CompileContext) -> Callable[[], None] | None:
    psi = None
    if spec.secondary.arm_angle is not None:
        psi = spec.secondary.arm_angle.psi_rad

    def _enter() -> None:
        spec.secondary.apply(ctx.inner, psi_rad=psi)
        # move→D: soft-attract rail to the target pose's rail coordinate.
        if (
            spec.secondary.preset == "move"
            and spec.q_target_rad is not None
            and len(np.asarray(spec.q_target_rad).reshape(-1)) > 0
        ):
            y_tgt = float(np.asarray(spec.q_target_rad, dtype=float).reshape(-1)[0])
            ctx.inner.set_rail_pose_target(y_tgt)
            ctx.inner.set_rail_extension_mode("pose_attract")
            ctx.inner.set_rail_extension_active(True)
        if spec.mode == TaskMode.LOCKED_MOVE and spec.q_rail_target_m is not None:
            ctx.inner.set_locked(spec.locked_style, q_ref_m=spec.q_rail_target_m)

    return _enter


def _make_on_exit(spec: JointPhaseSpec, ctx: CompileContext) -> Callable[[], None] | None:
    if spec.mode != TaskMode.LOCKED_MOVE:
        return None

    def _exit() -> None:
        ctx.inner.set_locked(LockedStyle.HOLD, q_ref_m=float(ctx.inner.q_cmd[0]))

    return _exit


def compile_phase(spec: JointPhaseSpec, ctx: CompileContext) -> CompiledPhase:
    """Build a runtime ``Phase`` from a ``JointPhaseSpec``."""
    gov = spec.governor
    on_enter = _make_on_enter(spec, ctx)
    on_exit = _make_on_exit(spec, ctx)
    ff_ref = spec.rail_ref if spec.mode == TaskMode.LOCKED_MOVE else spec.move_ref
    qdot_ff = spec.secondary.make_qdot_ff_provider(ctx.inner, ff_ref)

    if spec.mode in (TaskMode.JOINT_RESET, TaskMode.CARTESIAN_GOTO):
        if spec.move_ref is None:
            raise ValueError(f"{spec.mode}: move_ref is required")
        v_max_scaled = ctx.kin.v_max * ctx.v_scale
        if spec.move_mode == "joint":
            outer = JointTrackOuterLoop(
                spec.move_ref,
                ctx.kin,
                JointTrackConfig(
                    k_joint=float(spec.move_kp),
                    max_joint_err_rad=0.35,
                    sigma_ref=spec.sigma_ref,
                    control_frame=ctx.control_frame,
                    euler_order=ctx.euler_order,
                ),
                v_max_rad_s=v_max_scaled,
            )
        else:
            outer = CartesianTrackOuterLoop(
                spec.move_ref,
                CartesianTrackConfig(
                    k_task=np.full(6, spec.move_kp),
                    max_pos_err_m=0.05,
                    max_rot_err_rad=0.35,
                    max_lin_vel_m_s=spec.max_lin_vel_m_s,
                    control_frame=ctx.control_frame,
                    euler_order=ctx.euler_order,
                ),
            )
        phase = Phase(
            outer=outer,
            label=spec.label or spec.mode.value,
            duration_s=spec.duration_s,
            max_duration_s=spec.max_duration_s,
            wait_until=spec.wait_until,
            on_enter=on_enter,
            on_exit=on_exit,
            require_arrival=spec.require_arrival,
            governor_err_ok_mm=gov.err_ok_mm,
            governor_err_max_mm=gov.err_max_mm,
            governor_joint_err_ok_deg=gov.joint_err_ok_deg,
            governor_joint_err_max_deg=gov.joint_err_max_deg,
            governor_tau_s=gov.tau_s,
            governor_freeze_below=gov.freeze_below,
            governor_release_above=gov.release_above,
            soft_start_ramp_s=(0.3 if spec.secondary.preset == "move" else 0.0),
            qdot_ff_provider=qdot_ff,
            scale_qdot_ff_with_governor=spec.scale_qdot_ff_with_governor,
            force_observer=spec.force_observer,
        )
        if spec.move_mode == "joint":
            attach_joint_move_rail(phase, ctx.inner)
            phase.scale_qdot_ff_with_governor = True
        # Wire ψ_ref(t) + centering when the move plan is SRS.
        if (
            isinstance(spec.move_ref, SrsSmoothMoveReference)
            and spec.q_target_rad is not None
        ):
            attach_srs_move_tracking(
                phase, ctx.inner, spec.move_ref, spec.q_target_rad
            )
            # Governor must scale the rail pin (avoid full-rate y_dot at soft-start).
            phase.scale_qdot_ff_with_governor = True
        return CompiledPhase(
            phase=phase,
            label=phase.label,
            outer=outer,
            move_ref=spec.move_ref,
        )

    if spec.mode == TaskMode.LOCKED_MOVE:
        if spec.rail_ref is None:
            raise ValueError("locked_move: rail_ref is required")
        hold = HoldReference()
        kp = (
            float(spec.move_kp)
            if spec.locked_style == LockedStyle.TCP_FIXED
            else 0.0
        )
        outer = CartesianTrackOuterLoop(
            hold,
            CartesianTrackConfig(
                k_task=np.full(6, kp),
                max_pos_err_m=0.05,
                max_rot_err_rad=0.35,
                max_lin_vel_m_s=spec.max_lin_vel_m_s,
                control_frame=ctx.control_frame,
                euler_order=ctx.euler_order,
            ),
        )
        phase = Phase(
            outer=outer,
            label=spec.label or spec.mode.value,
            duration_s=spec.duration_s,
            max_duration_s=spec.max_duration_s,
            wait_until=spec.wait_until,
            on_enter=on_enter,
            on_exit=on_exit,
            require_arrival=spec.require_arrival,
            governor_err_ok_mm=gov.err_ok_mm,
            governor_err_max_mm=gov.err_max_mm,
            governor_joint_err_ok_deg=gov.joint_err_ok_deg,
            governor_joint_err_max_deg=gov.joint_err_max_deg,
            governor_tau_s=gov.tau_s,
            governor_freeze_below=gov.freeze_below,
            governor_release_above=gov.release_above,
            qdot_ff_provider=qdot_ff,
            scale_qdot_ff_with_governor=spec.scale_qdot_ff_with_governor,
            force_observer=spec.force_observer,
        )
        return CompiledPhase(
            phase=phase,
            label=phase.label,
            outer=outer,
            rail_ref=spec.rail_ref,
        )

    if spec.mode == TaskMode.CARTESIAN_TRACK:
        if spec.reference is None:
            raise ValueError(f"{spec.mode}: reference is required")
        outer = CartesianTrackOuterLoop(
            spec.reference,
            CartesianTrackConfig(
                k_task=np.full(6, spec.move_kp),
                max_pos_err_m=0.05,
                max_rot_err_rad=0.35,
                max_lin_vel_m_s=spec.max_lin_vel_m_s,
                control_frame=ctx.control_frame,
                euler_order=ctx.euler_order,
            ),
        )
        phase = Phase(
            outer=outer,
            label=spec.label or spec.mode.value,
            duration_s=spec.duration_s,
            max_duration_s=spec.max_duration_s,
            wait_until=spec.wait_until,
            on_enter=on_enter,
            on_exit=on_exit,
            require_arrival=spec.require_arrival,
            governor_err_ok_mm=gov.err_ok_mm,
            governor_err_max_mm=gov.err_max_mm,
            governor_joint_err_ok_deg=gov.joint_err_ok_deg,
            governor_joint_err_max_deg=gov.joint_err_max_deg,
            governor_tau_s=gov.tau_s,
            governor_freeze_below=gov.freeze_below,
            governor_release_above=gov.release_above,
            force_observer=spec.force_observer,
            scale_qdot_ff_with_governor=spec.scale_qdot_ff_with_governor,
        )
        return CompiledPhase(
            phase=phase,
            label=phase.label,
            outer=outer,
            reference=spec.reference,
        )

    if spec.mode == TaskMode.HYBRID_TRACK:
        if spec.reference is None or spec.controller is None:
            raise ValueError("hybrid_track: reference and controller are required")
        desired = spec.desired_force if spec.desired_force is not None else np.zeros(6)
        outer = AdmittanceOuterLoop(spec.controller, spec.reference, desired_force=desired)
        phase = Phase(
            outer=outer,
            label=spec.label or spec.mode.value,
            duration_s=spec.duration_s,
            max_duration_s=spec.max_duration_s,
            wait_until=spec.wait_until,
            on_enter=on_enter,
            on_exit=on_exit,
            require_arrival=spec.require_arrival,
            governor_err_ok_mm=gov.err_ok_mm,
            governor_err_max_mm=gov.err_max_mm,
            governor_joint_err_ok_deg=gov.joint_err_ok_deg,
            governor_joint_err_max_deg=gov.joint_err_max_deg,
            governor_tau_s=gov.tau_s,
            governor_freeze_below=gov.freeze_below,
            governor_release_above=gov.release_above,
            force_observer=spec.force_observer,
        )
        return CompiledPhase(
            phase=phase,
            label=phase.label,
            outer=outer,
            reference=spec.reference,
        )

    raise ValueError(f"unknown TaskMode: {spec.mode}")


def compile_phases(
    specs: list[JointPhaseSpec],
    ctx: CompileContext,
) -> list[CompiledPhase]:
    return [compile_phase(s, ctx) for s in specs]


from rm75_control.control.admittance_common.scaling import scale_admittance_for_desired_z
```

## `rm75_control/control/joint_admittance_8dof/sin_tool_y_program.py`

```python
"""Shared sin-tool-Y program builder and executor (window A and C)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.controller import AdmittanceController
from rm75_control.control.admittance_common.phase_ipc import SinToolYTaskParams
from rm75_control.control.joint_admittance_8dof.api import (
    ArmAngleSpec,
    CompileContext,
    GovernorSpec,
    SecondaryPolicy,
    compile_phases,
    phase_hold_at_pose,
    phase_hybrid_track,
    phase_rail_reposition,
    scale_admittance_for_desired_z,
)
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkController,
    LoopResult,
    run_joint_admittance_phases,
)
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    deg2rad,
    full_q_from_arm,
    wrap_joint_delta,
)
from rm75_control.control.joint_admittance_8dof.reference import (
    HoldReference,
    SinToolYReference,
)
from rm75_control.control.joint_admittance_8dof.wbc_arm import WbcArm
from rm75_control.control.joint_admittance_8dof.pose_ik import solve_pose_ik
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import _wrap_pi
from rm75_control.force.compensation import excitation as ex
from rm75_control.force.compensation.id_config import load_config as load_force_id_config
from rm75_control.force.compensation.paths import CONFIG_ID
from rm75_control.force.compensation.tool_pose import maybe_sync_kin_tcp_from_config


@dataclass
class ScanTargetD:
    """Planned move->D target from taught joints (Pinocchio FK / pose IK)."""

    q_slot_deg: np.ndarray
    pose_d: np.ndarray
    pose_id: np.ndarray
    q_target_rad: np.ndarray


def load_slot_joints_only(slot: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load taught ``q_deg`` / ``pose_base`` from poses.yaml without RealMan FK."""
    fid = load_force_id_config(CONFIG_ID)
    data = ex.load_poses_yaml(fid.poses_yaml)
    rec = ex.get_slot_record(data, slot)
    if rec is None:
        raise RuntimeError(f"Pose slot {slot!r} missing in {fid.poses_yaml}")
    q_deg = np.asarray(rec["q_deg"], dtype=float)
    pose_id = np.asarray(rec["pose_base"], dtype=float)
    return q_deg, pose_id, rec


def resolve_scan_target_at_d(
    slot: str,
    kin: RobotKinematics,
    *,
    euler_order: str = "xyz",
    rail_m: float = 0.0,
    qp_cfg=None,
    nullspace_cfg=None,
) -> ScanTargetD:
    """Resolve scan pose D and joint target for the move->D phase.

    Joints-only: taught ``q_deg`` with j7+90° (ArmTip +X → TCP +Z), fold
    approach into a world-vertical plane, optional pose IK. Move execution is
    still ``--move-mode`` (joint MoveJ or cartesian/SRS).
    """
    travel = 0.80
    try:
        travel = float(kin.q_upper[0])
    except Exception:
        pass
    return _resolve_scan_target_joints(
        slot,
        kin,
        rail_m=rail_m,
        travel_m=travel,
        qp_cfg=qp_cfg,
        nullspace_cfg=nullspace_cfg,
        euler_order=euler_order,
    )


def _pick_wellconditioned_rail_m(
    kin: RobotKinematics,
    q_arm_rad: np.ndarray,
    *,
    travel_m: float,
    prefer_m: float | None = None,
    n_samples: int = 21,
) -> tuple[float, float]:
    """Pick rail_y that maximizes σ_min for a fixed taught arm posture.

    ``prefer_m`` (e.g. mid-stroke from ``--rail-scan-center``) breaks ties and
    softly biases toward the caller's prior when σ is nearly flat.
    Returns ``(y_rail_m, sigma_min)``.
    """
    travel = max(float(travel_m), 1e-3)
    prefer = float(prefer_m) if prefer_m is not None else 0.5 * travel
    prefer = float(np.clip(prefer, 0.0, travel))
    best_y = prefer
    best_sig = -1.0
    best_score = -1e9
    q_arm = np.asarray(q_arm_rad, dtype=float).reshape(-1)
    for i in range(max(3, int(n_samples))):
        y = travel * i / (n_samples - 1)
        q = full_q_from_arm(q_arm, float(y))
        try:
            sig = float(kin.singular_values(kin.jacobian(q)).min())
        except Exception:
            continue
        # Soft prefer prior: 2 cm of rail ≈ 0.01 of σ_min (tie-break only).
        score = sig - 0.5 * abs(y - prefer)
        if score > best_score:
            best_score = score
            best_sig = sig
            best_y = y
    return float(best_y), float(best_sig)


def _remap_taught_q_armtip_x_to_tcp_z(q_arm_rad: np.ndarray) -> np.ndarray:
    """Map ArmTip-+X approach teach onto probe TCP-+Z (= ArmTip -Y).

    Slot ``d`` was taught with ArmTip +X oblique-down in the symmetry plane.
    Probe URDF TCP has +Z = ArmTip -Y, so the same joint vector leaves the tip
    sideways.  Adding +π/2 on wrist joint 7 is ``R ← R·Rz(+π/2)`` and makes
    ArmTip -Y (and TCP +Z) inherit the old +X world direction.
    """
    q = np.asarray(q_arm_rad, dtype=float).reshape(-1).copy()
    if q.size < 7:
        raise ValueError(f"expected 7 arm joints, got {q.size}")
    q[6] = float(q[6] + 0.5 * np.pi)
    # Keep a principal value so SRS / limit checks stay sane.
    q[6] = float(np.arctan2(np.sin(q[6]), np.cos(q[6])))
    return q


def _fold_flange_into_world_vertical_plane(R_l7: np.ndarray) -> tuple[np.ndarray, float]:
    """Fold link_7 so TCP+Z (= -Y) and flange +Z lie in a world-vertical plane.

    Taught D's ArmTip +X already had ~16° of world-Y lean; j7+90° kept that lean
    on TCP+Z.  Project approach into a constant-Y vertical plane (normal = ê_y),
    rebuild a right-handed flange frame with +Z also in that plane.
    Returns ``(R_l7_new, approach_fold_deg)``.
    """
    R = np.asarray(R_l7, dtype=float).reshape(3, 3)
    ey = np.array([0.0, 1.0, 0.0])
    # TCP +Z = ArmTip -Y
    approach = -R[:, 1]
    n = float(np.linalg.norm(approach))
    if n < 1e-9:
        return R.copy(), 0.0
    approach = approach / n
    a_proj = approach - (approach @ ey) * ey
    na = float(np.linalg.norm(a_proj))
    if na < 1e-9:
        return R.copy(), 0.0
    a_proj = a_proj / na
    fold_deg = float(np.degrees(np.arccos(np.clip(approach @ a_proj, -1.0, 1.0))))

    y_axis = -a_proj  # ArmTip +Y after fold
    # Flange +Z in the same vertical plane, ⊥ Y; pick the branch near the old Z.
    z_axis = np.cross(ey, y_axis)
    nz = float(np.linalg.norm(z_axis))
    if nz < 1e-9:
        return R.copy(), fold_deg
    z_axis = z_axis / nz
    if float(z_axis @ R[:, 2]) < 0.0:
        z_axis = -z_axis
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-12)
    # Re-orthogonalize Z in case of drift.
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / max(float(np.linalg.norm(z_axis)), 1e-12)
    R_new = np.column_stack((x_axis, y_axis, z_axis))
    return R_new, fold_deg


def _tcp_pose_from_link7(
    kin: RobotKinematics,
    p_l7: np.ndarray,
    R_l7: np.ndarray,
    *,
    euler_order: str = "xyz",
) -> np.ndarray:
    """Compose world TCP pose from link_7 pose and URDF link_7→tcp offset."""
    R_off = np.asarray(kin._R_link7_tcp, dtype=float).reshape(3, 3)
    t_off = np.asarray(kin._r_link7_tcp, dtype=float).reshape(3)
    R_tcp = R_l7 @ R_off
    p_tcp = np.asarray(p_l7, dtype=float).reshape(3) + R_l7 @ t_off
    pose = np.zeros(6, dtype=float)
    pose[:3] = p_tcp
    pose[3:6] = Rsc.from_matrix(R_tcp).as_euler(euler_order, degrees=False)
    return pose


def _resolve_scan_target_joints(
    slot: str,
    kin: RobotKinematics,
    *,
    rail_m: float = 0.0,
    travel_m: float = 0.80,
    refine_rail: bool = True,
    qp_cfg=None,
    nullspace_cfg=None,
    euler_order: str = "xyz",
) -> ScanTargetD:
    q_deg_taught, pose_id, _rec = load_slot_joints_only(slot)
    q_arm = _remap_taught_q_armtip_x_to_tcp_z(deg2rad(q_deg_taught))
    y_rail = float(rail_m)
    if refine_rail:
        y_rail, _sig = _pick_wellconditioned_rail_m(
            kin,
            q_arm,
            travel_m=float(travel_m),
            prefer_m=float(rail_m),
        )
    q_seed = full_q_from_arm(q_arm, y_rail)

    Ml7 = kin.frame_placement(q_seed, "link_7")
    R_fold, _fold_deg = _fold_flange_into_world_vertical_plane(Ml7.rotation)
    pose_d = _tcp_pose_from_link7(
        kin, Ml7.translation, R_fold, euler_order=euler_order
    )

    q_target_rad = q_seed
    if qp_cfg is not None:
        q_target_rad, _ok, _rep = solve_pose_ik(
            kin,
            q_seed,
            pose_d,
            qp_cfg=qp_cfg,
            nullspace_cfg=nullspace_cfg,
            attractor_q=q_seed,
        )
        # Keep Cartesian target as FK of the solved q (consistent with build()).
        pose_d = np.asarray(kin.fk_pose(q_target_rad), dtype=float)
    else:
        # No QP: still publish the folded Cartesian; SRS will pull toward it.
        pass

    q_deg = np.rad2deg(q_target_rad[1:])
    return ScanTargetD(
        q_slot_deg=q_deg,
        pose_d=pose_d,
        pose_id=pose_id,
        q_target_rad=q_target_rad,
    )



def load_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_psi_sides(
    psi_center: float,
    *,
    side_offset_rad: float = np.deg2rad(90.5),
    psi_left_rad: float | None = None,
    psi_right_rad: float | None = None,
) -> tuple[float, float, float]:
    """Center swivel at pose D; left/right = center ± offset (same branch)."""
    center = float(_wrap_pi(psi_center))
    if psi_left_rad is not None and psi_right_rad is not None:
        return (
            center,
            float(_wrap_pi(psi_left_rad)),
            float(_wrap_pi(psi_right_rad)),
        )
    off = abs(float(side_offset_rad))
    return center, float(_wrap_pi(center + off)), float(_wrap_pi(center - off))


def resolve_psi_sides_live(
    psi_center: float,
    psi_live: float,
    *,
    fallback_offset_rad: float = np.deg2rad(90.5),
    min_offset_rad: float = np.deg2rad(10.0),
) -> tuple[float, float, float]:
    """Center @ D; left = live Realman psi; right mirrored: center - (left - center)."""
    center = float(_wrap_pi(psi_center))
    left = float(_wrap_pi(psi_live))
    delta = _wrap_pi(left - center)
    if abs(delta) < min_offset_rad:
        return resolve_psi_sides(center, side_offset_rad=fallback_offset_rad)
    right = float(_wrap_pi(center - delta))
    return center, left, right


def plan_q_toggle_at_pose(
    kin: RobotKinematics,
    pose_d: np.ndarray,
    q_center_rad: np.ndarray,
    q_live_rad: np.ndarray,
    *,
    qp_cfg,
    nullspace_cfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IK @ pose D: center, left (seed=live teach), right (mirror joint delta)."""
    q_center = np.asarray(q_center_rad, dtype=float).reshape(-1)
    pose_d = np.asarray(pose_d, dtype=float).reshape(6)
    q_left, ok_l, rep_l = solve_pose_ik(
        kin, q_live_rad, pose_d, qp_cfg=qp_cfg, nullspace_cfg=nullspace_cfg
    )
    if not ok_l or rep_l.pos_err_mm > 5.0:
        return q_center, q_center, q_center
    delta = q_left - q_center
    q_right, ok_r, rep_r = solve_pose_ik(
        kin, q_center - delta, pose_d, qp_cfg=qp_cfg, nullspace_cfg=nullspace_cfg
    )
    if not ok_r or rep_r.pos_err_mm > 5.0:
        q_right = q_center - delta
    return q_center, np.asarray(q_left, dtype=float).reshape(-1), np.asarray(q_right, dtype=float).reshape(-1)


def plan_psi_sides_ik_at_pose(
    kin: RobotKinematics,
    inner: JointIkController,
    pose_d: np.ndarray,
    q_center_rad: np.ndarray,
    q_live_rad: np.ndarray,
    *,
    qp_cfg,
    nullspace_cfg,
    side_offset_rad: float = np.deg2rad(90.5),
) -> tuple[float, float, float]:
    """ψ labels for logging (from IK q targets at fixed TCP)."""
    q_c, q_l, q_r = plan_q_toggle_at_pose(
        kin, pose_d, q_center_rad, q_live_rad, qp_cfg=qp_cfg, nullspace_cfg=nullspace_cfg
    )
    if inner.arm_task is None:
        return resolve_psi_sides(0.0, side_offset_rad=side_offset_rad)
    psi_c = float(inner.arm_task.arm_angle(q_c))
    if np.max(np.abs(q_l - q_c)) < 1e-6:
        return resolve_psi_sides(psi_c, side_offset_rad=side_offset_rad)
    psi_l = float(inner.arm_task.arm_angle(q_l))
    psi_r = float(inner.arm_task.arm_angle(q_r))
    return psi_c, psi_l, psi_r


def plan_psi_toggle_sides(
    inner: JointIkController,
    q_live_rad: np.ndarray,
    psi_center: float,
    *,
    side_offset_rad: float = np.deg2rad(90.5),
    psi_left_rad: float | None = None,
    psi_right_rad: float | None = None,
    psi_live_left: bool = True,
    kin: RobotKinematics | None = None,
    pose_d: np.ndarray | None = None,
    q_center_rad: np.ndarray | None = None,
    qp_cfg=None,
    nullspace_cfg=None,
) -> tuple[float, float, float]:
    """Plan center/left/right ψ for hybrid @ D (IK-feasible at fixed TCP when possible)."""
    if psi_left_rad is not None and psi_right_rad is not None:
        return resolve_psi_sides(
            psi_center,
            psi_left_rad=psi_left_rad,
            psi_right_rad=psi_right_rad,
        )
    if (
        psi_live_left
        and kin is not None
        and pose_d is not None
        and q_center_rad is not None
        and qp_cfg is not None
        and nullspace_cfg is not None
        and inner.arm_task is not None
    ):
        return plan_psi_sides_ik_at_pose(
            kin,
            inner,
            pose_d,
            q_center_rad,
            q_live_rad,
            qp_cfg=qp_cfg,
            nullspace_cfg=nullspace_cfg,
            side_offset_rad=side_offset_rad,
        )
    if psi_live_left and inner.arm_task is not None:
        psi_live = float(inner.arm_task.arm_angle(q_live_rad))
        return resolve_psi_sides_live(
            psi_center,
            psi_live,
            fallback_offset_rad=side_offset_rad,
        )
    return resolve_psi_sides(psi_center, side_offset_rad=side_offset_rad)


def attach_hybrid_posture_toggle(
    phases: list,
    inner: JointIkController,
    *,
    q_center: np.ndarray,
    q_left: np.ndarray,
    q_right: np.ndarray,
    period_s: float,
    verbose: bool = False,
    filter_alpha: float = 0.02,
    ramp_duration_s: float = 4.0,
    k_center_scale: float = 2.5,
    max_qdot_frac: float = 0.35,
) -> None:
    """Ramp joint centering targets (same TCP) — visible multi-DOF posture change."""
    if period_s <= 0.0:
        return
    q_center = np.asarray(q_center, dtype=float).reshape(-1)
    q_left = np.asarray(q_left, dtype=float).reshape(-1)
    q_right = np.asarray(q_right, dtype=float).reshape(-1)

    inner.set_arm_task_suppressed(True)
    k_saved = float(inner.centering_task.cfg.k_center)
    inner.centering_task.cfg.k_center = k_saved * float(k_center_scale)
    frac_saved = float(inner.secondary.max_qdot_frac)
    inner.secondary.max_qdot_frac = float(max_qdot_frac)
    inner.centering_task.q_target = q_center.copy()

    ramp_s = max(0.5, min(float(ramp_duration_s), float(period_s) * 0.85))
    toggle_state = {
        "last_bucket": -1,
        "current_q": q_center.copy(),
        "ramp_from": q_center.copy(),
        "ramp_to": q_center.copy(),
        "ramp_t0": 0.0,
    }

    def _q_for_bucket(bucket: int) -> tuple[np.ndarray, str]:
        if bucket == 0:
            return q_center, "center"
        if bucket % 2 == 1:
            return q_left, "left"
        return q_right, "right"

    def on_tick(t_ref: float, _step, _q_meas: np.ndarray) -> None:
        bucket = int(t_ref / period_s)
        if bucket != toggle_state["last_bucket"]:
            toggle_state["last_bucket"] = bucket
            target, _tag = _q_for_bucket(bucket)
            toggle_state["ramp_from"] = toggle_state["current_q"].copy()
            toggle_state["ramp_to"] = target.copy()
            toggle_state["ramp_t0"] = t_ref

        dt_ramp = t_ref - toggle_state["ramp_t0"]
        u = float(np.clip(dt_ramp / ramp_s, 0.0, 1.0))
        u2, u3, u4, u5 = u * u, u * u * u, u * u * u * u, u * u * u * u * u
        s = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
        delta = wrap_joint_delta(toggle_state["ramp_from"], toggle_state["ramp_to"])
        target_q = toggle_state["ramp_from"] + s * delta

        diff = wrap_joint_delta(toggle_state["current_q"], target_q)
        toggle_state["current_q"] = toggle_state["current_q"] + filter_alpha * diff
        toggle_state["current_q"][0] = q_center[0]
        inner.centering_task.q_target = toggle_state["current_q"].copy()

    hybrid_labels = ("scan", "hybrid@D")
    for phase in phases:
        if phase.label in hybrid_labels:
            phase.on_tick = on_tick
            return
    raise RuntimeError(f"attach_hybrid_posture_toggle: no phase in {hybrid_labels}")


def attach_scan_psi_toggle(
    phases: list,
    inner: JointIkController,
    *,
    psi_center: float,
    psi_left: float,
    psi_right: float,
    period_s: float,
    verbose: bool = False,
    filter_alpha: float = 0.01,
    ramp_duration_s: float = 4.0,
    k_psi_scale: float = 0.35,
) -> None:
    """Hybrid phase: hold center, then quintic-ramp left / right arm-angle targets."""
    if period_s <= 0.0:
        return
    if inner.arm_task is None:
        raise RuntimeError("psi toggle requires arm_angle secondary task")

    k_psi_saved = float(inner.arm_task.cfg.k_psi)
    inner.arm_task.cfg.k_psi = k_psi_saved * float(k_psi_scale)

    ramp_s = max(0.5, min(float(ramp_duration_s), float(period_s) * 0.85))
    toggle_state = {
        "last_bucket": -1,
        "current_psi": psi_center,
        "ramp_from": psi_center,
        "ramp_to": psi_center,
        "ramp_t0": 0.0,
    }

    def _target_for_bucket(bucket: int) -> tuple[float, str]:
        if bucket == 0:
            return psi_center, "center"
        if bucket % 2 == 1:
            return psi_left, "left"
        return psi_right, "right"

    def on_tick(t_ref: float, _step, _q_meas: np.ndarray) -> None:
        bucket = int(t_ref / period_s)
        if bucket != toggle_state["last_bucket"]:
            toggle_state["last_bucket"] = bucket
            target, _tag = _target_for_bucket(bucket)
            toggle_state["ramp_from"] = toggle_state["current_psi"]
            toggle_state["ramp_to"] = target
            toggle_state["ramp_t0"] = t_ref

        dt_ramp = t_ref - toggle_state["ramp_t0"]
        u = float(np.clip(dt_ramp / ramp_s, 0.0, 1.0))
        u2, u3, u4, u5 = u * u, u * u * u, u * u * u * u, u * u * u * u * u
        s = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
        delta = _wrap_pi(toggle_state["ramp_to"] - toggle_state["ramp_from"])
        target = _wrap_pi(toggle_state["ramp_from"] + s * delta)

        current = toggle_state["current_psi"]
        diff = _wrap_pi(target - current)
        toggle_state["current_psi"] = _wrap_pi(current + filter_alpha * diff)
        inner.arm_task.set_reference(toggle_state["current_psi"])

    hybrid_labels = ("scan", "hybrid@D")
    for phase in phases:
        if phase.label in hybrid_labels:
            phase.on_tick = on_tick
            return
    raise RuntimeError(
        f"attach_scan_psi_toggle: no phase in {hybrid_labels}"
    )


@dataclass
class BuiltSinToolYProgram:
    phases: list
    compiled: list
    inner: JointIkController
    kin: RobotKinematics
    force_observer: Any


def build_sin_tool_y_program(
    params: SinToolYTaskParams,
    *,
    raw: dict | None = None,
) -> BuiltSinToolYProgram:
    """Build phase list from precomputed task params (same on C and A)."""
    raw = raw if raw is not None else load_yaml(params.config_path)
    kin = RobotKinematics()
    maybe_sync_kin_tcp_from_config(
        kin,
        raw,
        tcp_offset_pose=params.tcp_offset_pose if params.tcp_offset_pose else None,
    )
    inner_cfg = build_joint_ik_config(raw)
    inner = JointIkController(kin, inner_cfg)
    max_lin = (
        float(params.cartesian_max_lin_vel)
        if params.cartesian_max_lin_vel is not None
        else 0.4
    )
    rail_m = float(inner_cfg.rail.q_ref_m)

    q_target_rad = np.asarray(params.q_target_rad, dtype=float).reshape(-1)
    q0_rad = np.asarray(params.q0_rad, dtype=float).reshape(-1)
    # Wait/SRS target must be FK(q_target) after TCP sync — raw params.pose_d can
    # still carry an ArmTip/IK residual orientation that blocks arrival forever
    # while track_err_mm (position-only) looks fine.
    pose_d = np.asarray(kin.fk_pose(q_target_rad), dtype=float).reshape(6)
    move_mode = str(params.plan_move_mode)
    if move_mode == "joint":
        move_phase = WbcArm.make_movej_phase(
            kin,
            q0_rad,
            q_target_rad,
            duration_s=float(params.plan_duration_s),
            label=f"movej->{params.slot}",
            move_kp=float(params.move_kp),
            gov_joint_max_deg=float(params.plan_gov_joint_max_deg),
            force_observer=None,
        )
    else:
        move_phase = WbcArm.make_movel_phase(
            kin,
            q0_rad,
            pose_d,
            q_target_rad,
            duration_s=float(params.plan_duration_s),
            label=f"movel->{params.slot}",
            move_kp=float(params.move_kp),
            max_lin_vel_m_s=max_lin,
            gov_joint_max_deg=float(params.plan_gov_joint_max_deg),
            force_observer=None,
            euler_order=inner_cfg.euler_order,
        )

    force_observer = None
    if params.enable_force and params.scan_duration > 0.0:
        from rm75_control.control.admittance_common.observer import CompensatedForceObserver

        force_observer = CompensatedForceObserver.from_yaml(raw)
        move_phase.force_observer = force_observer

    ctx = CompileContext(
        kin=kin,
        inner=inner,
        euler_order=inner_cfg.euler_order,
        control_frame=inner_cfg.control_frame,
        v_scale=inner_cfg.v_scale,
    )

    specs = [move_phase]

    if params.hold_at_d_s > 0.0:
        specs.append(
            phase_hold_at_pose(
                params.hold_at_d_s,
                label="hold@D",
                force_observer=force_observer,
            )
        )

    if params.rail_move_cm > 0.0:
        sign = 1.0 if params.rail_move_dir == "+y" else -1.0
        rail0 = float(inner_cfg.rail.q_ref_m if inner_cfg.rail.q_ref_m is not None else 0.0)
        delta_m = sign * float(params.rail_move_cm) * 0.01
        rail_target = rail0 + delta_m
        lo, hi = 0.0, float(inner_cfg.rail.travel_m)
        if not (lo <= rail_target <= hi):
            raise RuntimeError(
                f"rail target {rail_target * 100:.1f}cm outside travel "
                f"[{lo * 100:.0f}, {hi * 100:.0f}]cm"
            )
        q_rail_start = full_q_from_arm(q_target_rad, rail_m=rail0)
        rail_style = str(params.rail_move_mode)
        specs.append(
            phase_rail_reposition(
                rail_target,
                q_rail_start,
                kin,
                label=f"rail{params.rail_move_dir}{params.rail_move_cm:.0f}cm_{rail_style}",
                style=rail_style,
                force_observer=force_observer,
                v_max_m_s=inner_cfg.rail.v_max_m_s,
            )
        )

    if params.scan_duration > 0.0:
        dt = float(raw.get("timing", {}).get("dt_ms", 5.0)) / 1000.0
        outer_ctrl = AdmittanceController(
            dt, scale_admittance_for_desired_z(raw, float(params.desired_z))
        )
        desired_force = np.zeros(6)
        desired_force[2] = float(params.desired_z)
        psi = None if params.psi_tgt is None or not np.isfinite(params.psi_tgt) else float(params.psi_tgt)
        if params.scan_hybrid_hold:
            # TCP force hold (no Y sin), but keep rail COUPLED + reach so
            # σ-escape / preferred-extension can still slide the carriage.
            hybrid_ref: HoldReference | SinToolYReference = HoldReference()
            hybrid_label = "hybrid@D"
            hybrid_sec = SecondaryPolicy(
                preset="track",
                arm_angle=ArmAngleSpec(psi_rad=psi) if psi is not None else None,
                qdot_ff="off",
            )
            hybrid_gov = GovernorSpec(err_ok_mm=15.0, err_max_mm=80.0)
        else:
            amplitude_m = float(params.y_pp_cm) * 0.01 / 2.0
            max_vel_m_s = float(params.max_vel_cm_s) * 0.01
            hybrid_ref = SinToolYReference(
                amplitude_m,
                period_s=params.period_s,
                max_vel_m_s=None if params.period_s is not None else max_vel_m_s,
                soft_start=True,
                ramp_s=2.0,
                euler_order=inner_cfg.euler_order,
            )
            hybrid_label = "scan"
            # COUPLED: let the QP-IK freely distribute the tool-Y sweep between the
            # rail and the arm (rail slides, arm reaches out) — exactly the old
            # controller-driven-rail behaviour. The velocity-mode motor just follows
            # the resulting smooth q_cmd[0]; no rail pinning, no arm-only contortion.
            hybrid_sec = SecondaryPolicy(preset="track", qdot_ff="off")
            hybrid_gov = GovernorSpec(err_ok_mm=10.0, err_max_mm=40.0)
        specs.append(
            phase_hybrid_track(
                hybrid_ref,
                outer_ctrl,
                desired_force=desired_force,
                label=hybrid_label,
                duration_s=float(params.scan_duration),
                force_observer=force_observer,
                psi_rad_on_enter=psi,
                secondary=hybrid_sec,
                governor=hybrid_gov,
            )
        )

    compiled = compile_phases(specs, ctx)
    phases = [c.phase for c in compiled]
    if params.psi_toggle_period_s > 0.0 and params.scan_duration > 0.0:
        q_c = np.asarray(params.q_target_rad, dtype=float).reshape(-1)
        has_q = (
            len(params.q_toggle_left_rad) >= q_c.size
            and len(params.q_toggle_right_rad) >= q_c.size
        )
        if has_q:
            attach_hybrid_posture_toggle(
                phases,
                inner,
                q_center=q_c,
                q_left=np.asarray(params.q_toggle_left_rad, dtype=float).reshape(-1),
                q_right=np.asarray(params.q_toggle_right_rad, dtype=float).reshape(-1),
                period_s=float(params.psi_toggle_period_s),
                filter_alpha=float(params.psi_filter_alpha),
                ramp_duration_s=float(params.psi_ramp_s),
            )
        elif params.psi_tgt is not None and np.isfinite(params.psi_tgt):
            psi_center, psi_left, psi_right = resolve_psi_sides(
                float(params.psi_tgt),
                side_offset_rad=float(params.psi_side_offset_rad),
                psi_left_rad=params.psi_left_rad,
                psi_right_rad=params.psi_right_rad,
            )
            attach_scan_psi_toggle(
                phases,
                inner,
                psi_center=psi_center,
                psi_left=psi_left,
                psi_right=psi_right,
                period_s=float(params.psi_toggle_period_s),
                filter_alpha=float(params.psi_filter_alpha),
                ramp_duration_s=float(params.psi_ramp_s),
            )
        else:
            raise RuntimeError("psi toggle requires q_toggle_left/right or psi_tgt")
    return BuiltSinToolYProgram(
        phases=phases,
        compiled=compiled,
        inner=inner,
        kin=kin,
        force_observer=force_observer,
    )


def execute_sin_tool_y_program(
    session,
    state_bus,
    params: SinToolYTaskParams,
    *,
    raw: dict | None = None,
    built: BuiltSinToolYProgram | None = None,
    on_step: Callable | None = None,
    stop_check: Callable[[], bool] | None = None,
    verbose: bool = False,
    rail_bridge=None,
) -> LoopResult:
    """Run WBC on window A (direct UDP feedback + direct CANFD)."""
    raw = raw if raw is not None else load_yaml(params.config_path)
    startup = raw.get("startup", {})
    dt = float(raw.get("timing", {}).get("dt_ms", 5.0)) / 1000.0
    if built is None:
        built = build_sin_tool_y_program(params, raw=raw)

    return run_joint_admittance_phases(
        session,
        built.phases,
        built.inner,
        q_start_deg=None,
        dt=dt,
        follow=bool(startup.get("follow", True)),
        move_speed=int(startup.get("move_speed", 20)),
        realtime=bool(startup.get("realtime", False)),
        watchdog_timeout_s=float(startup.get("watchdog_timeout_s", 0.1)),
        on_step=on_step,
        log_csv=params.log_csv,
        state_bus=state_bus,
        canfd_proxy=None,
        stop_check=stop_check,
        verbose=verbose,
        rail_bridge=rail_bridge,
    )


def make_task_params_from_args(
    args,
    *,
    config_path: str,
    q0_rad: np.ndarray,
    q_target_rad: np.ndarray,
    pose_d: np.ndarray,
    plan,
    psi_tgt: float | None,
    desired_z: float,
    enable_force: bool,
    psi_left_rad: float | None = None,
    psi_right_rad: float | None = None,
    q_toggle_left_rad: np.ndarray | None = None,
    q_toggle_right_rad: np.ndarray | None = None,
    tcp_offset_pose: np.ndarray | None = None,
) -> SinToolYTaskParams:
    return SinToolYTaskParams(
        config_path=config_path,
        slot=str(args.slot),
        move_kp=float(args.move_kp),
        y_pp_cm=float(args.y_pp_cm),
        max_vel_cm_s=float(args.max_vel_cm_s),
        period_s=args.period_s,
        desired_z=float(desired_z),
        scan_duration=float(args.scan_duration),
        hold_at_d_s=float(args.hold_at_d_s),
        rail_move_cm=float(args.rail_move_cm),
        rail_move_mode=str(args.rail_move_mode),
        rail_move_dir=str(args.rail_move_dir),
        enable_force=bool(enable_force),
        log_csv=args.log_csv,
        rail_log_csv=getattr(args, "rail_log_csv", None),
        cartesian_max_lin_vel=args.cartesian_max_lin_vel,
        q0_rad=np.asarray(q0_rad, dtype=float).reshape(-1).tolist(),
        q_target_rad=np.asarray(q_target_rad, dtype=float).reshape(-1).tolist(),
        pose_d=np.asarray(pose_d, dtype=float).reshape(6).tolist(),
        plan_duration_s=float(plan.duration_s),
        plan_move_mode=str(plan.move_mode),
        plan_gov_joint_max_deg=float(plan.gov_joint_max_deg),
        psi_tgt=psi_tgt,
        psi_toggle_period_s=float(getattr(args, "psi_toggle_period", 0.0) or 0.0),
        psi_side_offset_rad=np.deg2rad(
            float(getattr(args, "psi_side_offset_deg", 90.5))
        ),
        psi_left_rad=(
            float(psi_left_rad)
            if psi_left_rad is not None
            else (
                np.deg2rad(float(args.psi_left_deg))
                if getattr(args, "psi_left_deg", None) is not None
                else None
            )
        ),
        psi_right_rad=(
            float(psi_right_rad)
            if psi_right_rad is not None
            else (
                np.deg2rad(float(args.psi_right_deg))
                if getattr(args, "psi_right_deg", None) is not None
                else None
            )
        ),
        psi_filter_alpha=float(getattr(args, "psi_toggle_alpha", 0.02)),
        psi_ramp_s=float(getattr(args, "psi_ramp_s", 4.0)),
        scan_hybrid_hold=bool(getattr(args, "hybrid_hold_at_d", False)),
        q_toggle_left_rad=(
            np.asarray(q_toggle_left_rad, dtype=float).reshape(-1).tolist()
            if q_toggle_left_rad is not None
            else []
        ),
        q_toggle_right_rad=(
            np.asarray(q_toggle_right_rad, dtype=float).reshape(-1).tolist()
            if q_toggle_right_rad is not None
            else []
        ),
        tcp_offset_pose=(
            np.asarray(tcp_offset_pose, dtype=float).reshape(6).tolist()
            if tcp_offset_pose is not None
            else []
        ),
    )
```

## `rm75_control/control/joint_admittance_8dof/pose_ik.py`

```python
"""One-shot pose IK for the 8-DOF stack (no vendor rm_algo_inverse_kinematics).

``resolve_pose_ik_srs``: preferred closed-form SRS + ψ enum + path check.
``solve_pose_ik``: legacy iterative WBC IK for tools / reachability scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as Rsc
from scipy.spatial.transform import Slerp

from rm75_control.control.joint_admittance_8dof.ik_types import saturate_error
from rm75_control.control.joint_admittance_8dof.model import (
    RAIL_INDEX,
    RobotKinematics,
    full_q_from_arm,
    pose_error,
)
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig, QpIkController
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import (
    ArmAngleTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits
from rm75_control.kinematics.srs_ik import (
    Q_LOWER,
    Q_UPPER,
    branch_from_q,
    d_wt_from_kin,
    psi_from_q,
    srs_ik,
)


class UnreachablePathError(RuntimeError):
    """No ψ candidate yields a globally reachable path (fail loud; re-teach)."""


@dataclass
class PoseIkReport:
    """Convergence diagnostics from ``solve_pose_ik`` / ``resolve_pose_ik_srs``."""
    pos_err_mm: float
    rot_err_deg: float
    sigma_min: float
    iters: int
    within_limits: bool
    psi_deg: float = float("nan")
    psi_home_deg: float = float("nan")
    path_ok: bool = True


@dataclass
class PlannerGoalWeights:
    """Weights for the SRS planner's goal_score (higher = better posture).

    The score is
        s = -w_home · ((ψ − ψ_home)/π)²
            -w_sigma_floor · max(0, sigma_safe − sigma_min)
            -w_limit · Σ ((q_i − q_mid_i)/q_range_i)²
            -w_wrist · exp(-8 · sin²(q5))
            -w_elbow · max(0, 0.3 − sin(q4))

    ψ_home is the PRIMARY attractor; sigma / limit / wrist / elbow are
    thresholds that keep the candidate feasible / comfortable but do not
    compete with ψ_home unless ψ_home itself lands in trouble.
    """
    w_home: float = 1.0
    sigma_safe: float = 0.08
    w_sigma_floor: float = 100.0
    w_limit: float = 0.5
    w_wrist: float = 0.3
    w_elbow: float = 0.5


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _slerp_pose(p0: np.ndarray, p1: np.ndarray, s: float, euler_order: str = "xyz") -> np.ndarray:
    """Constant-speed SE(3) interpolation: position lerp + rotation SLERP.

    Both endpoints are 6-vec ``[x, y, z, rx, ry, rz]`` (matches fk_pose).
    ``s`` in [0, 1].
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    R_stack = Rsc.from_euler(euler_order, np.stack([p0[3:6], p1[3:6]]), degrees=False)
    key_times = [0.0, 1.0]
    slerp = Slerp(key_times, R_stack)
    R_s = slerp([float(np.clip(s, 0.0, 1.0))])[0]
    pos = (1.0 - s) * p0[:3] + s * p1[:3]
    out = np.zeros(6, dtype=float)
    out[:3] = pos
    out[3:6] = R_s.as_euler(euler_order, degrees=False)
    return out


def _goal_score(
    q_arm: np.ndarray,
    q_full: np.ndarray,
    psi: float,
    psi_home: float,
    sigma_min: float,
    kin: RobotKinematics,
    weights: PlannerGoalWeights,
) -> float:
    """Higher = more desirable posture.  See PlannerGoalWeights docstring."""
    d_home = _wrap_pi(psi - psi_home) / np.pi          # ∈ [-1, 1]
    home_penalty = weights.w_home * d_home * d_home

    sigma_penalty = weights.w_sigma_floor * max(0.0, weights.sigma_safe - sigma_min)

    q_range = Q_UPPER - Q_LOWER
    q_mid = 0.5 * (Q_UPPER + Q_LOWER)
    u = (q_arm - q_mid) / np.maximum(q_range, 1e-6)
    limit_penalty = weights.w_limit * float(np.sum(u * u))

    # Wrist singularity proxy: exp(-8·sin²(q5)) is ~1 at q5 ≈ 0 / ±π, ~0 elsewhere.
    wrist_penalty = weights.w_wrist * float(np.exp(-8.0 * np.sin(q_arm[4]) ** 2))

    # Straight-elbow penalty: sin(q4) < 0.3 means elbow bent < ~17.5°, i.e.
    # near-straight arm — dangerous (approaches the SRS shoulder-arm-wrist
    # collinear singularity used by the arm_angle observability decay).
    elbow_penalty = weights.w_elbow * max(0.0, 0.3 - float(np.sin(q_arm[3])))

    return -(home_penalty + sigma_penalty + limit_penalty + wrist_penalty + elbow_penalty)


def _path_reachable(
    kin: RobotKinematics,
    pose_seed: np.ndarray,
    pose_target: np.ndarray,
    psi_seed: float,
    psi_target: float,
    branch: int,
    y_rail_seed: float,
    y_rail_target: float,
    *,
    n_samples: int = 10,
    euler_order: str = "xyz",
    d_wt: float | None = None,
) -> bool:
    """True iff srs_ik succeeds at every interior sample of the (pose, ψ, y_rail)
    interpolation.  Endpoints are excluded: they are guaranteed by the seed
    (feasibility already verified for the seed) and by the enumeration itself.
    """
    # Unwrap ψ so linear interpolation goes the short way and does not cross ±π.
    psi_target_unwrapped = psi_seed + _wrap_pi(psi_target - psi_seed)
    for i in range(1, n_samples + 1):
        s = i / (n_samples + 1)                           # 1/(n+1) ... n/(n+1)
        pose_s = _slerp_pose(pose_seed, pose_target, s, euler_order)
        psi_s = psi_seed + s * (psi_target_unwrapped - psi_seed)
        y_rail_s = y_rail_seed + s * (y_rail_target - y_rail_seed)
        q_arm = srs_ik(
            pose_s,
            psi_s,
            branch,
            y_rail=y_rail_s,
            euler_order=euler_order,
            d_wt=d_wt,
        )
        if q_arm is None:
            return False
    return True


def resolve_pose_ik_srs(
    kin: RobotKinematics,
    q_seed: np.ndarray,
    pose_target: np.ndarray,
    *,
    q_branch_seed: np.ndarray | None = None,
    y_rail_target: float | None = None,
    psi_home_rad: float | None = None,
    max_psi_swing_rad: float = 150.0 * np.pi / 180.0,
    psi_hard_lower_rad: float | None = None,
    psi_hard_upper_rad: float | None = None,
    planner_weights: PlannerGoalWeights | None = None,
    psi_grid_step_rad: float = 5.0 * np.pi / 180.0,
    path_check_samples: int = 10,
    top_k_for_path_check: int = 5,
    require_path: bool = True,
    euler_order: str = "xyz",
) -> tuple[np.ndarray, bool, PoseIkReport]:
    """SRS closed-form IK + 1-D ψ grid enumeration + path reachability check.

    Returns ``(q_target_full_rad, ok, report)`` where ``q_target_full_rad``
    is an 8-vec with the rail entry set to ``y_rail_target`` (or
    ``q_seed[0]`` if the caller left it None).

    Enumeration rules (in priority order):

    1. Reject ψ candidates outside ``[psi_hard_lower_rad, psi_hard_upper_rad]``
       (if provided) and outside ``|wrap(ψ − ψ_seed)| ≤ max_psi_swing_rad``.
    2. Reject candidates whose srs_ik is None (branch unreachable / hits
       shoulder or wrist singularity / violates URDF joint limits).
    3. Rank surviving candidates by :func:`_goal_score` and take the top-K.
    4. For each top-K candidate, verify the whole interpolation path
       ``(pose_seed, ψ_seed) → (pose_target, ψ_candidate)`` is srs_ik-solvable
       at ``path_check_samples`` interior points.
    5. Return the highest-scoring candidate whose path check passes.

    Raises
    ------
    UnreachablePathError
        If no candidate survives the path check.  The caller must re-teach
        the target pose or the seed rather than silently accepting a plan
        that will stall mid-move.
    """
    weights = planner_weights or PlannerGoalWeights()
    q_seed = np.asarray(q_seed, dtype=float).copy()
    if q_seed.size != 8:
        raise ValueError(f"q_seed must be 8-vec, got size {q_seed.size}")
    q_arm_seed = q_seed[1:]
    q_branch_src = (
        np.asarray(q_branch_seed, dtype=float).copy()
        if q_branch_seed is not None
        else q_seed
    )
    if q_branch_src.size != 8:
        raise ValueError(f"q_branch_seed must be 8-vec, got size {q_branch_src.size}")
    y_rail_seed = float(q_seed[RAIL_INDEX])
    y_rail_target = float(q_seed[RAIL_INDEX] if y_rail_target is None else y_rail_target)

    pose_seed = kin.fk_pose(q_seed)
    psi_seed = psi_from_q(q_arm_seed)
    branch_seed = branch_from_q(q_branch_src[1:])
    psi_home = float(psi_seed if psi_home_rad is None else psi_home_rad)
    d_wt = float(d_wt_from_kin(kin))

    # Candidate ψ grid on (-π, π].  max_psi_swing is measured from ψ_home
    # (the posture attractor), NOT from ψ_seed — so a live q0 at ψ≈72° can
    # still pick a ψ near 72° even when the taught slot branch differs.
    psi_grid = np.arange(-np.pi, np.pi, float(psi_grid_step_rad))
    scored: list[tuple[float, float, np.ndarray, float]] = []  # (score, psi, q_arm, sigma_min)
    for psi in psi_grid:
        d_home = abs(_wrap_pi(float(psi) - psi_home))
        if d_home > float(max_psi_swing_rad):
            continue
        # Hard bounds (cable-carrier / cabin envelope):
        if psi_hard_lower_rad is not None and float(psi) < float(psi_hard_lower_rad):
            continue
        if psi_hard_upper_rad is not None and float(psi) > float(psi_hard_upper_rad):
            continue

        q_arm = srs_ik(
            pose_target, float(psi), branch_seed,
            y_rail=y_rail_target, euler_order=euler_order, d_wt=d_wt,
        )
        if q_arm is None:
            continue
        q_full = full_q_from_arm(q_arm, rail_m=y_rail_target)
        J = kin.jacobian(q_full)
        sigma_min = float(kin.singular_values(J).min())
        score = _goal_score(q_arm, q_full, float(psi), psi_home, sigma_min, kin, weights)
        scored.append((score, float(psi), q_arm, sigma_min))

    if not scored:
        raise UnreachablePathError(
            "SRS IK found no reachable ψ candidate for pose_target — "
            "check max_psi_swing_rad, psi_hard_*, or re-teach the target pose."
        )

    scored.sort(key=lambda x: x[0], reverse=True)     # highest score first
    top_k = scored[: max(1, int(top_k_for_path_check))]

    def _report_from(
        psi: float,
        q_arm: np.ndarray,
        sigma_min: float,
        *,
        path_ok: bool,
    ) -> tuple[np.ndarray, bool, PoseIkReport]:
        q_full = full_q_from_arm(q_arm, rail_m=y_rail_target)
        pose_ach = kin.fk_pose(q_full)
        err = pose_error(pose_target, pose_ach, euler_order)
        pos_err_m = float(np.linalg.norm(err[:3]))
        rot_err_rad = float(np.linalg.norm(err[3:6]))
        within = bool(
            np.all(q_full[1:] >= Q_LOWER - 1e-6)
            and np.all(q_full[1:] <= Q_UPPER + 1e-6)
        )
        report = PoseIkReport(
            pos_err_mm=pos_err_m * 1000.0,
            rot_err_deg=float(np.degrees(rot_err_rad)),
            sigma_min=sigma_min,
            iters=0,
            within_limits=within,
            psi_deg=float(np.degrees(psi)),
            psi_home_deg=float(np.degrees(psi_home)),
            path_ok=path_ok,
        )
        ok = path_ok and within and pos_err_m <= 0.005 and rot_err_rad <= np.deg2rad(2.0)
        return q_full, ok, report

    if not require_path:
        score, psi, q_arm, sigma_min = top_k[0]
        return _report_from(psi, q_arm, sigma_min, path_ok=False)

    # Path reachability check on the top-K candidates.
    for score, psi, q_arm, sigma_min in top_k:
        if _path_reachable(
            kin,
            pose_seed=pose_seed,
            pose_target=pose_target,
            psi_seed=psi_seed,
            psi_target=psi,
            branch=branch_seed,
            y_rail_seed=y_rail_seed,
            y_rail_target=y_rail_target,
            n_samples=int(path_check_samples),
            euler_order=euler_order,
            d_wt=d_wt,
        ):
            return _report_from(psi, q_arm, sigma_min, path_ok=True)

    # None of the top-K candidates has a fully reachable path.
    _, psi_best, q_arm_best, sigma_best = top_k[0]
    raise UnreachablePathError(
        f"pose IK: top-{len(top_k)} ψ candidates all fail path reachability. "
        f"Best ψ={np.degrees(psi_best):.1f}° from ψ_seed={np.degrees(psi_seed):.1f}° "
        f"(branch from {'branch_seed' if q_branch_seed is not None else 'q_seed'}) — "
        f"either the pose is too far from the seed or ψ_home is unreachable at this pose. "
        f"Please re-teach the target pose or adjust psi_home_deg / max_psi_swing_deg."
    )


def solve_pose_ik(
    kin: RobotKinematics,
    q_seed: np.ndarray,
    pose_target: np.ndarray,
    *,
    max_iters: int = 500,
    pos_tol_m: float = 1e-3,
    rot_tol_rad: float = 0.02,
    dt: float = 0.02,
    k_gain: float = 3.0,
    max_pos_err_m: float = 0.05,
    max_rot_err_rad: float = 0.20,
    qp_cfg: QpConfig | None = None,
    nullspace_cfg: NullspaceTaskConfig | None = None,
    attractor_q: np.ndarray | None = None,
    trace: list[dict] | None = None,
) -> tuple[np.ndarray, bool, PoseIkReport]:
    """Iterative WBC IK (legacy): ``q_seed`` → ``q`` with fk(q) ≈ pose_target.

    ``attractor_q=None`` centers on ``q_seed`` (not yaml zeros). Prefer SRS IK.
    """
    cfg = qp_cfg or QpConfig()
    limits = SafetyLimits.from_kinematics(kin, v_scale=0.9, a_max=50.0)
    ctrl = QpIkController(kin, limits, cfg)

    task: JointCenteringTask | None = None
    if nullspace_cfg is not None:
        # Default attractor is q_seed (teach posture), not yaml q_nominal zeros.
        target = np.asarray(
            attractor_q if attractor_q is not None else q_seed,
            dtype=float,
        )
        cfg_used = NullspaceTaskConfig(
            k_center=nullspace_cfg.k_center,
            k_limit=nullspace_cfg.k_limit,
            activation=nullspace_cfg.activation,
            weights=nullspace_cfg.weights,
            q_nominal_rad=target,
        )
        task = JointCenteringTask.from_kinematics(kin, cfg_used)

    q = np.clip(np.asarray(q_seed, dtype=float).copy(), kin.q_lower, kin.q_upper)
    pose_target = np.asarray(pose_target, dtype=float)
    ctrl.reset(q)

    sigma_last = float("nan")
    pos_err_m = float("nan")
    rot_err_rad = float("nan")
    for it in range(max_iters):
        err = pose_error(pose_target, kin.fk_pose(q), cfg.euler_order)
        pos_err_m = float(np.linalg.norm(err[:3]))
        rot_err_rad = float(np.linalg.norm(err[3:6]))
        if pos_err_m < pos_tol_m and rot_err_rad < rot_tol_rad:
            report = _make_report(q, kin, ctrl, pos_err_m, rot_err_rad, it, sigma_last)
            if trace is not None:
                trace.append(
                    {
                        "iter": it,
                        "pos_err_mm": pos_err_m * 1000.0,
                        "rot_err_deg": np.degrees(rot_err_rad),
                        "v_cmd_norm": 0.0,
                        "slack_norm": None,
                        "n_cbf_active": None,
                        "sigma_min": report.sigma_min,
                        "converged": True,
                    }
                )
            return q, True, report
        err_sat = saturate_error(err, max_pos_err_m, max_rot_err_rad)
        v_cmd = k_gain * err_sat
        secondary = task(q) if task is not None else None
        r = ctrl.step(q, v_cmd, dt, secondary_qdot=secondary)
        sigma_last = r.sigma_min
        if trace is not None:
            trace.append(
                {
                    "iter": it,
                    "pos_err_mm": pos_err_m * 1000.0,
                    "rot_err_deg": np.degrees(rot_err_rad),
                    "v_cmd_norm": float(np.linalg.norm(v_cmd)),
                    "slack_norm": r.slack_norm,
                    "n_cbf_active": r.n_cbf_active,
                    "sigma_min": r.sigma_min,
                    "converged": False,
                }
            )
        q = np.clip(r.q_next, kin.q_lower, kin.q_upper)

    report = _make_report(q, kin, ctrl, pos_err_m, rot_err_rad, max_iters, sigma_last)
    return q, False, report


def _make_report(
    q: np.ndarray,
    kin: RobotKinematics,
    ctrl: QpIkController,
    pos_err_m: float,
    rot_err_rad: float,
    iters: int,
    sigma_last: float,
) -> PoseIkReport:
    try:
        sigma_min = float(kin.singular_values(kin.jacobian(q)).min())
    except Exception:
        sigma_min = float(sigma_last)
    margin = float(ctrl.constraints.lim.position_margin)
    lo = kin.q_lower + margin
    hi = kin.q_upper - margin
    within = bool(np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9))
    return PoseIkReport(
        pos_err_mm=pos_err_m * 1000.0,
        rot_err_deg=float(np.degrees(rot_err_rad)),
        sigma_min=sigma_min,
        iters=int(iters),
        within_limits=within,
    )


__all__ = [
    "PlannerGoalWeights",
    "PoseIkReport",
    "UnreachablePathError",
    "resolve_pose_ik_srs",
    "solve_pose_ik",
]
```

## `rm75_control/control/joint_admittance_8dof/reference.py`

```python
"""Motion references for the joint-admittance loop (pure kinematics / scipy).

HoldReference, JointSmoothMoveReference, SrsSmoothMoveReference (branch-locked
quintic in pose/ψ), RailSmoothMoveReference, SinToolYReference.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.reference import MotionReference


class HoldReference:
    """Hold the start pose: pose_d = pose0, vel_ff = 0."""

    def __init__(self) -> None:
        self._pose0: np.ndarray | None = None

    def set_origin(self, pose0: np.ndarray) -> None:
        self._pose0 = np.asarray(pose0, dtype=float).copy()

    def sample(self, t_s: float) -> MotionReference:
        if self._pose0 is None:
            raise RuntimeError("HoldReference.set_origin must be called first")
        return MotionReference.from_pose_hold(self._pose0)


def smoothstep_scalar(t_s: float, duration_s: float) -> tuple[float, float]:
    """Quintic smoothstep s(u), ds/dt with s''(0)=s''(1)=0 (C² endpoints)."""
    if duration_s <= 0.0:
        return 1.0, 0.0
    u = float(np.clip(t_s / duration_s, 0.0, 1.0))
    u2 = u * u
    u3 = u2 * u
    u4 = u3 * u
    u5 = u4 * u
    s = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
    ds_du = 30.0 * u2 - 60.0 * u3 + 30.0 * u4
    ds_dt = ds_du / duration_s
    return s, ds_dt


class JointSmoothMoveReference:
    """Joint-space smoothstep (q_start→q_target) exposed via FK/J as Cartesian ref.

    Does no IK — interpolate a pre-resolved ``q_target`` only.
    """

    def __init__(
        self,
        kin,
        q_start_rad: np.ndarray,
        q_target_rad: np.ndarray,
        duration_s: float,
    ) -> None:
        self.kin = kin
        self.q_start = np.asarray(q_start_rad, dtype=float).copy()
        self.q_target = np.asarray(q_target_rad, dtype=float).copy()
        self.duration_s = float(duration_s)

    def set_origin(self, pose0: np.ndarray) -> None:
        # q_start already anchors this reference; pose0 is implied by FK(q_start).
        del pose0

    def sample_q(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Joint-space (q_ref(t), qdot_ff(t)) for Phase.qdot_ff_provider."""
        from rm75_control.control.joint_admittance_8dof.model import wrap_joint_delta

        s, ds_dt = smoothstep_scalar(t_s, self.duration_s)
        dq = wrap_joint_delta(self.q_start, self.q_target)
        q = self.q_start + s * dq
        qdot = ds_dt * dq
        return q, qdot

    def sample(self, t_s: float) -> MotionReference:
        """Cartesian (pose, vel_ff) via FK/Jacobian."""
        q, qdot = self.sample_q(t_s)
        pose = self.kin.fk_pose(q)
        vel = self.kin.jacobian(q) @ qdot
        return MotionReference(pose, vel, t_ref=t_s)

    def done(self, t_s: float) -> bool:
        return t_s >= self.duration_s


def srs_move_duration_s(
    q_start_rad: np.ndarray,
    q_target_rad: np.ndarray,
    *,
    max_qdot_rad_s: float | np.ndarray = 1.0,
    peak_v_frac: float = 0.60,
    duration_min_s: float = 0.5,
) -> float:
    """Auto duration so quintic peak ``1.875·|dq|/T`` stays under ``peak_v_frac·v_max``."""
    from rm75_control.control.joint_admittance_8dof.model import wrap_joint_delta

    dq = np.abs(wrap_joint_delta(q_start_rad, q_target_rad))
    if np.isscalar(max_qdot_rad_s):
        vmax_vec = np.full_like(dq, float(max_qdot_rad_s))
    else:
        vmax_vec = np.asarray(max_qdot_rad_s, dtype=float)
    vmax_vec = np.maximum(vmax_vec * float(peak_v_frac), 1e-6)
    t_per_joint = 1.875 * dq / vmax_vec
    return max(float(duration_min_s), float(np.max(t_per_joint)))


class SrsSmoothMoveReference:
    """Branch-locked quintic in (pose, ψ, y_rail); each tick ``srs_ik`` on q_start branch.

    Cartesian path is line+slerp; ψ is C²-smooth; no mid-move J1/J4 flip.
    """

    def __init__(
        self,
        kin,
        q_start_rad: np.ndarray,
        pose_target: np.ndarray,
        *,
        y_rail_target_m: float,
        psi_target_rad: float,
        duration_s: float,
        branch_id: int | None = None,
        euler_order: str = "xyz",
        d_wt: float | None = None,
        max_ik_fail_streak: int = 5,
    ) -> None:
        from rm75_control.kinematics.srs_ik import branch_from_q, d_wt_from_kin, psi_from_q

        self.kin = kin
        self.q_start = np.asarray(q_start_rad, dtype=float).copy()
        self.pose_start = np.asarray(self.kin.fk_pose(self.q_start), dtype=float)
        self.pose_target = np.asarray(pose_target, dtype=float).copy()
        self.y_start = float(self.q_start[0])
        self.y_target = float(y_rail_target_m)
        self.duration_s = float(duration_s)
        q_arm_start = self.q_start[1:]
        self.branch_id = int(branch_id) if branch_id is not None else int(branch_from_q(q_arm_start))
        self.psi_start = float(psi_from_q(q_arm_start))
        self.psi_target = float(psi_target_rad)
        # Shortest-arc unwrap so ψ does not travel the long way around ±π.
        self.psi_delta = float(
            (self.psi_target - self.psi_start + np.pi) % (2.0 * np.pi) - np.pi
        )
        self.euler_order = str(euler_order)
        self.d_wt = float(d_wt_from_kin(kin) if d_wt is None else d_wt)
        R_start = Rsc.from_euler(self.euler_order, self.pose_start[3:])
        R_target = Rsc.from_euler(self.euler_order, self.pose_target[3:])
        self._R_start = R_start
        self._delta_rotvec = (R_target * R_start.inv()).as_rotvec()
        self._last_q = self.q_start.copy()
        self._ik_fail_streak = 0
        self._max_ik_fail_streak = int(max(1, max_ik_fail_streak))

    def reseed_start(self, q_start_rad: np.ndarray) -> None:
        """Re-anchor start from live encoders; keep pose/y/ψ targets."""
        from rm75_control.kinematics.srs_ik import branch_from_q, psi_from_q

        self.q_start = np.asarray(q_start_rad, dtype=float).copy()
        self.pose_start = np.asarray(self.kin.fk_pose(self.q_start), dtype=float)
        self.y_start = float(self.q_start[0])
        q_arm_start = self.q_start[1:]
        self.branch_id = int(branch_from_q(q_arm_start))
        self.psi_start = float(psi_from_q(q_arm_start))
        self.psi_delta = float(
            (self.psi_target - self.psi_start + np.pi) % (2.0 * np.pi) - np.pi
        )
        R_start = Rsc.from_euler(self.euler_order, self.pose_start[3:])
        R_target = Rsc.from_euler(self.euler_order, self.pose_target[3:])
        self._R_start = R_start
        self._delta_rotvec = (R_target * R_start.inv()).as_rotvec()
        self._last_q = self.q_start.copy()
        self._ik_fail_streak = 0

    def _pose_at(self, s: float) -> np.ndarray:
        pos = self.pose_start[:3] + s * (self.pose_target[:3] - self.pose_start[:3])
        R_at = Rsc.from_rotvec(s * self._delta_rotvec) * self._R_start
        pose = np.zeros(6)
        pose[:3] = pos
        pose[3:] = R_at.as_euler(self.euler_order)
        return pose

    def _q_at(self, s: float) -> np.ndarray:
        from rm75_control.kinematics.srs_ik import srs_ik

        pose_s = self._pose_at(s)
        psi_s = self.psi_start + s * self.psi_delta
        y_s = self.y_start + s * (self.y_target - self.y_start)
        q_arm = srs_ik(
            pose_s,
            psi_s,
            self.branch_id,
            y_rail=y_s,
            euler_order=self.euler_order,
            check_limits=False,
            d_wt=self.d_wt,
        )
        q = np.zeros_like(self.q_start)
        q[0] = y_s
        if q_arm is None:
            self._ik_fail_streak += 1
            if self._ik_fail_streak >= self._max_ik_fail_streak:
                raise RuntimeError(
                    f"SrsSmoothMoveReference: srs_ik returned None for "
                    f"{self._ik_fail_streak} consecutive samples "
                    f"(s={s:.3f}, branch={self.branch_id}, "
                    f"psi={np.degrees(psi_s):.1f}deg). "
                    f"Refusing silent joint hold (would freeze TCP governor). "
                    f"Use joint PTP recovery for cross-branch moves."
                )
            q = self._last_q.copy()
            q[0] = y_s
        else:
            self._ik_fail_streak = 0
            q[1:] = q_arm
            self._last_q = q.copy()
        return q

    def sample_q(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        s, _ds_dt = smoothstep_scalar(t_s, self.duration_s)
        q = self._q_at(s)
        # qdot_ff via central-diff on the smoothstep clock so the loop's
        # Phase.qdot_ff_provider gets a consistent (q, qdot) pair even at t=0/T.
        h = 1.0e-3
        s_plus, _ = smoothstep_scalar(min(t_s + h, self.duration_s), self.duration_s)
        s_minus, _ = smoothstep_scalar(max(t_s - h, 0.0), self.duration_s)
        q_plus = self._q_at(s_plus)
        q_minus = self._q_at(s_minus)
        denom = max(1e-9, (min(t_s + h, self.duration_s) - max(t_s - h, 0.0)))
        qdot = (q_plus - q_minus) / denom
        return q, qdot

    def sample(self, t_s: float) -> MotionReference:
        s, _ = smoothstep_scalar(t_s, self.duration_s)
        pose = self._pose_at(s)
        h = 1.0e-3
        s_plus, _ = smoothstep_scalar(min(t_s + h, self.duration_s), self.duration_s)
        s_minus, _ = smoothstep_scalar(max(t_s - h, 0.0), self.duration_s)
        pose_plus = self._pose_at(s_plus)
        pose_minus = self._pose_at(s_minus)
        denom = max(1e-9, (min(t_s + h, self.duration_s) - max(t_s - h, 0.0)))
        vel = np.zeros(6)
        vel[:3] = (pose_plus[:3] - pose_minus[:3]) / denom
        R_plus = Rsc.from_euler(self.euler_order, pose_plus[3:])
        R_minus = Rsc.from_euler(self.euler_order, pose_minus[3:])
        vel[3:] = (R_plus * R_minus.inv()).as_rotvec() / denom
        return MotionReference(pose_d=pose, vel_ff=vel, t_ref=t_s)

    def sample_psi(self, t_s: float) -> float:
        s, _ = smoothstep_scalar(t_s, self.duration_s)
        return float(self.psi_start + s * self.psi_delta)

    def set_origin(self, pose0: np.ndarray) -> None:
        del pose0  # q_start anchors this reference

    def done(self, t_s: float) -> bool:
        return t_s >= self.duration_s


def auto_rail_move_duration_s(
    q_start_m: float,
    q_target_m: float,
    *,
    v_max_m_s: float,
    peak_v_frac: float = 0.50,
    duration_min_s: float = 0.5,
) -> float:
    """Duration for quintic rail smoothstep (peak speed 1.875·|dq|/T)."""
    dq = abs(float(q_target_m) - float(q_start_m))
    v_lim = max(float(v_max_m_s) * float(peak_v_frac), 1e-6)
    from_rail = 1.875 * dq / v_lim
    return max(float(duration_min_s), from_rail)


class RailSmoothMoveReference:
    """Quintic smoothstep on rail_y only; arm joints held at q_start[1:]."""

    def __init__(
        self,
        q_start: np.ndarray,
        q_target_m: float,
        duration_s: float,
    ) -> None:
        self.q_start = np.asarray(q_start, dtype=float).copy()
        self.q_target_m = float(q_target_m)
        self.duration_s = float(duration_s)
        self._q_arm = self.q_start[1:].copy()

    @property
    def q_target(self) -> np.ndarray:
        q = self.q_start.copy()
        q[0] = self.q_target_m
        return q

    def sample_q(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        s, ds_dt = smoothstep_scalar(t_s, self.duration_s)
        dq_rail = self.q_target_m - float(self.q_start[0])
        q = np.zeros_like(self.q_start)
        q[0] = float(self.q_start[0]) + s * dq_rail
        q[1:] = self._q_arm
        qdot = np.zeros_like(self.q_start)
        qdot[0] = ds_dt * dq_rail
        return q, qdot

    def done(self, t_s: float) -> bool:
        return t_s >= self.duration_s


def sin_period_for_peak_vel(amplitude_m: float, max_vel_m_s: float) -> float:
    if amplitude_m <= 0.0 or max_vel_m_s <= 0.0:
        return 1.0
    return 2.0 * math.pi * amplitude_m / max_vel_m_s


def sin_y_motion(
    t_s: float,
    amplitude_m: float,
    omega: float,
    *,
    soft_start: bool,
    ramp_s: float = 2.0,
) -> tuple[float, float]:
    """(dy, vy) with C1 soft start via time-warp tau(t) (pose/vel stay consistent)."""
    if soft_start and ramp_s > 0.0:
        if t_s < ramp_s:
            # tau(t) = int_0^t sin(pi*u/(2*ramp)) du
            tau = (2.0 * ramp_s / math.pi) * (1.0 - math.cos(0.5 * math.pi * t_s / ramp_s))
            tau_dot = math.sin(0.5 * math.pi * t_s / ramp_s)
        else:
            tau = t_s - ramp_s + (2.0 * ramp_s / math.pi)
            tau_dot = 1.0
    else:
        tau = t_s
        tau_dot = 1.0
    dy = amplitude_m * math.sin(omega * tau)
    vy = amplitude_m * omega * math.cos(omega * tau) * tau_dot
    return dy, vy


class SinToolYReference:
    """Tool-frame Y sinusoid about a fixed origin (orientation held)."""

    def __init__(
        self,
        amplitude_m: float,
        *,
        period_s: float | None = None,
        max_vel_m_s: float | None = None,
        soft_start: bool = True,
        ramp_s: float = 2.0,
        euler_order: str = "xyz",
    ) -> None:
        if period_s is None:
            if max_vel_m_s is None:
                raise ValueError("provide either period_s or max_vel_m_s")
            period_s = sin_period_for_peak_vel(amplitude_m, max_vel_m_s)
        self.amplitude_m = float(amplitude_m)
        self.period_s = float(period_s)
        self.omega = 2.0 * math.pi / self.period_s if self.period_s > 0 else 0.0
        self.soft_start = soft_start
        self.ramp_s = ramp_s
        self.euler_order = euler_order
        self._origin: np.ndarray | None = None
        # Phase anchor for teach re-origin: sample uses (t_s - _t_anchor) so a
        # mid-scan set_origin() does not double-apply the accumulated sin offset.
        self._t_anchor: float = 0.0

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        self._origin = np.asarray(pose0, dtype=float).copy()
        if t_s is not None:
            self._t_anchor = float(t_s)

    def sample(self, t_s: float) -> MotionReference:
        if self._origin is None:
            raise RuntimeError("SinToolYReference.set_origin must be called first")
        t_eff = float(t_s) - float(self._t_anchor)
        dy, vy = sin_y_motion(
            t_eff, self.amplitude_m, self.omega, soft_start=self.soft_start, ramp_s=self.ramp_s
        )
        r_mat = Rsc.from_euler(self.euler_order, self._origin[3:6], degrees=False).as_matrix()
        pose = self._origin.copy()
        pose[:3] = self._origin[:3] + r_mat @ np.array([0.0, dy, 0.0])
        vel = np.zeros(6, dtype=float)
        vel[:3] = r_mat @ np.array([0.0, vy, 0.0])
        return MotionReference(pose_d=pose, vel_ff=vel, t_ref=t_s)

```

## `rm75_control/control/joint_admittance_8dof/wbc_arm.py`

```python
"""Industrial motion facade over local ProxQP admittance (RM_API2-style).

Mirrors RealMan ``MovePlan.rm_movej`` / ``rm_movel`` / ``rm_movej_p`` signatures
(``v``, ``r``, ``connect``, ``block`` → ``int`` status) but drives the local
WBC stack — it does **not** forward to vendor ``rm_movej`` (that would drop
collision CBF / admittance / rail coupling).

Also exposes:
  * ``algo_fk`` / ``algo_ik`` — kinematics

Typical use (window C → window A phase IPC)::

    arm = WbcArm(config_path="configs/joint_admittance_8dof.yaml")
    arm.connect()
    tag = arm.movej(q_deg, v=20, r=0, connect=0, block=1)
    # then start force scan / movel explicitly — no auto distance switch
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from rm75_control.control.admittance_common.phase_ipc import (
    PhaseCommandClient,
    PhaseStatus,
    SinToolYTaskParams,
)
from rm75_control.control.joint_admittance_8dof.api import (
    MovePlan,
    compute_move_plan,
    make_srs_move_reference,
    phase_cartesian_goto,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.reference import JointSmoothMoveReference

_LOG = logging.getLogger(__name__)

# Status codes aligned with RM_API2 Robotic_Arm MovePlan conventions.
OK = 0
ERR_PARAM = 1
ERR_SEND = -1
ERR_RECV = -2
ERR_ARRIVAL = -4
ERR_TIMEOUT = -5


def _clamp_v(v: int) -> int:
    return int(np.clip(int(v), 1, 100))


def _v_to_scale(v: int) -> float:
    """Map RM-style speed percent 1..100 onto a duration scale factor."""
    return float(np.clip(_clamp_v(v) / 100.0, 0.05, 1.0))


def _warn_stub(r: int, connect: int) -> None:
    if int(r) != 0 or int(connect) != 0:
        _LOG.info(
            "WbcArm: r=%s connect=%s ignored this release (no blend / multi-seg)",
            r,
            connect,
        )


class WbcArm:
    """Unified MoveJ / MoveL API over the local ProxQP admittance controller."""

    def __init__(
        self,
        config_path: str | Path = "configs/joint_admittance_8dof.yaml",
        *,
        phase_client: PhaseCommandClient | None = None,
        kin: RobotKinematics | None = None,
        default_timeout_s: float = 120.0,
    ) -> None:
        self.config_path = str(config_path)
        self._client = phase_client
        self.kin = kin or RobotKinematics()
        self.default_timeout_s = float(default_timeout_s)

    def connect(self, *, timeout_s: float = 30.0) -> int:
        """Attach to window A phase IPC hub. Returns 0 on success, -1 on failure."""
        if self._client is None:
            self._client = PhaseCommandClient()
        try:
            self._client.wait_for_hub(timeout_s=timeout_s)
            return OK
        except TimeoutError:
            return ERR_SEND

    # ------------------------------------------------------------------ builders
    @staticmethod
    def make_movej_phase(
        kin: RobotKinematics,
        q_start_rad: np.ndarray,
        q_target_rad: np.ndarray,
        *,
        duration_s: float,
        label: str = "movej",
        move_kp: float = 2.0,
        gov_joint_max_deg: float = 25.0,
        max_duration_s: float | None = None,
        require_arrival: bool = True,
        force_observer: Any = None,
    ):
        """Build a joint-space PTP phase (MoveJ semantics, same ProxQP)."""
        q0 = np.asarray(q_start_rad, dtype=float).reshape(-1)
        qt = np.asarray(q_target_rad, dtype=float).reshape(-1)
        move_ref = JointSmoothMoveReference(kin, q0, qt, float(duration_s))
        pose_tgt = np.asarray(kin.fk_pose(qt), dtype=float).reshape(6)
        T = float(duration_s)
        return phase_cartesian_goto(
            move_ref,
            label=label,
            pose_target=pose_tgt,
            q_target_rad=qt,
            move_kp=float(move_kp),
            move_mode="joint",
            max_duration_s=float(max_duration_s) if max_duration_s is not None else T * 2.5 + 15.0,
            gov_joint_max_deg=float(gov_joint_max_deg),
            require_arrival=require_arrival,
            force_observer=force_observer,
        )

    @staticmethod
    def make_movel_phase(
        kin: RobotKinematics,
        q_start_rad: np.ndarray,
        pose_target: np.ndarray,
        q_target_rad: np.ndarray,
        *,
        duration_s: float,
        label: str = "movel",
        move_kp: float = 2.0,
        max_lin_vel_m_s: float = 0.4,
        gov_joint_max_deg: float = 25.0,
        max_duration_s: float | None = None,
        require_arrival: bool = True,
        force_observer: Any = None,
        euler_order: str = "xyz",
    ):
        """Build a Cartesian straight-line SRS phase (MoveL semantics)."""
        q0 = np.asarray(q_start_rad, dtype=float).reshape(-1)
        qt = np.asarray(q_target_rad, dtype=float).reshape(-1)
        pose = np.asarray(pose_target, dtype=float).reshape(6)
        move_ref = make_srs_move_reference(
            kin, q0, pose, qt, float(duration_s), euler_order=euler_order
        )
        T = float(move_ref.duration_s)
        return phase_cartesian_goto(
            move_ref,
            label=label,
            pose_target=np.asarray(kin.fk_pose(qt), dtype=float).reshape(6),
            q_target_rad=qt,
            move_kp=float(move_kp),
            move_mode="cartesian",
            max_lin_vel_m_s=float(max_lin_vel_m_s),
            max_duration_s=float(max_duration_s) if max_duration_s is not None else T * 2.5 + 15.0,
            gov_joint_max_deg=float(gov_joint_max_deg),
            require_arrival=require_arrival,
            force_observer=force_observer,
        )

    def algo_ik(
        self,
        pose: list[float] | np.ndarray,
        q_seed: list[float] | np.ndarray | None = None,
        *,
        q_seed_deg: bool = True,
    ) -> tuple[int, list[float]]:
        """Solve pose → joints.

        Returns:
            (0, [rail_mm, j1..j7 °]) on success, (1, []) on failure.

        ``q_seed``: if ``q_seed_deg`` then industrial list ``[rail_mm, °…]`` /
        7-arm °; else full ``q`` in rad (8).
        """
        from rm75_control.control.joint_admittance_8dof.pose_ik import solve_pose_ik

        pose_a = np.asarray(pose, dtype=float).reshape(6)
        if q_seed is None:
            q0 = np.zeros(self.kin.nv, dtype=float)
            q0[0] = 0.4
        elif q_seed_deg:
            try:
                q0 = self._joint_list_to_rad(q_seed)
            except ValueError:
                return ERR_PARAM, []
        else:
            q0 = np.asarray(q_seed, dtype=float).reshape(-1)
            if q0.size != self.kin.nv:
                return ERR_PARAM, []
        try:
            q_sol, ok, _rep = solve_pose_ik(self.kin, q0, pose_a)
        except Exception:
            return ERR_PARAM, []
        if not ok or q_sol is None:
            return ERR_PARAM, []
        q = np.asarray(q_sol, dtype=float).reshape(-1)
        out = [float(q[0]) * 1000.0, *np.rad2deg(q[1:]).tolist()]
        return OK, out

    def algo_fk(
        self,
        joint: list[float] | np.ndarray,
        *,
        q_deg: bool = True,
    ) -> tuple[int, list[float]]:
        """关节 → TCP 位姿 (FK).

        Args:
            joint: ``q_deg=True`` 时工业列表 ``[rail_mm, j1..j7 °]`` 或 7 臂角 °；
                ``False`` 时为 8 维 rad。
        Returns:
            (0, [x,y,z,rx,ry,rz]) 位置 m、姿态 rad；失败 (1, [])。
        """
        try:
            q = (
                self._joint_list_to_rad(joint)
                if q_deg
                else np.asarray(joint, dtype=float).reshape(-1)
            )
        except ValueError:
            return ERR_PARAM, []
        if q.size != self.kin.nv:
            return ERR_PARAM, []
        pose = np.asarray(self.kin.fk_pose(q), dtype=float).reshape(6)
        return OK, pose.tolist()

    # ------------------------------------------------------------------ motion
    def movej(
        self,
        joint: list[float],
        v: int,
        r: int,
        connect: int,
        block: int,
        *,
        q0_deg: list[float] | None = None,
        timeout_s: float | None = None,
    ) -> int:
        """关节空间运动 (MoveJ).

        Args:
            joint: 目标构型。长度 8：``[rail_mm, j1..j7 °]``；长度 7：仅臂角 °（rail=0.4 m）。
            v: 速度百分比 1~100
            r: 交融半径（本轮忽略）
            connect: 轨迹连接（本轮忽略）
            block: 0 非阻塞；1 阻塞至到位；>1 阻塞并作超时秒数

        Returns:
            0 成功；1 参数/规划失败；-1 IPC 失败；-2 未到位/停止；-4 到位校验失败；-5 超时。
        """
        _warn_stub(r, connect)
        try:
            q_tgt = self._joint_list_to_rad(joint)
        except ValueError:
            return ERR_PARAM
        q0 = self._resolve_q0_rad(q0_deg)
        plan = self._plan_duration(q0, q_tgt, move_mode="joint", v=v)
        params = self._make_move_params(
            q0_rad=q0,
            q_target_rad=q_tgt,
            pose_d=self.kin.fk_pose(q_tgt),
            plan=plan,
            move_mode="joint",
            v=v,
        )
        return self._submit(params, block=block, timeout_s=timeout_s)

    def movel(
        self,
        pose: list[float],
        v: int,
        r: int,
        connect: int,
        block: int,
        *,
        q0_deg: list[float] | None = None,
        q_target_deg: list[float] | None = None,
        timeout_s: float | None = None,
    ) -> int:
        """笛卡尔空间直线运动 (MoveL / SRS)。

        Args:
            pose: [x,y,z,rx,ry,rz]，位置 m，姿态 rad（xyz 欧拉）。
            v/r/connect/block: 同 ``movej``。
            q_target_deg: 可选预解关节；缺省则 ``algo_ik``。
        """
        _warn_stub(r, connect)
        pose_a = np.asarray(pose, dtype=float).reshape(6)
        q0 = self._resolve_q0_rad(q0_deg)
        if q_target_deg is not None:
            try:
                q_tgt = self._joint_list_to_rad(q_target_deg)
            except ValueError:
                return ERR_PARAM
        else:
            code, q_list = self.algo_ik(
                pose_a, q_seed=q0, q_seed_deg=False
            )
            if code != OK:
                return code
            try:
                q_tgt = self._joint_list_to_rad(q_list)
            except ValueError:
                return ERR_PARAM
        plan = self._plan_duration(q0, q_tgt, move_mode="cartesian", v=v, pose=pose_a)
        params = self._make_move_params(
            q0_rad=q0,
            q_target_rad=q_tgt,
            pose_d=pose_a,
            plan=plan,
            move_mode="cartesian",
            v=v,
        )
        return self._submit(params, block=block, timeout_s=timeout_s)

    def movej_p(
        self,
        pose: list[float],
        v: int,
        r: int,
        connect: int,
        block: int,
        *,
        q0_deg: list[float] | None = None,
        timeout_s: float | None = None,
    ) -> int:
        """位姿目标 → IK → 关节空间运动（对应 RM ``rm_movej_p``）。"""
        _warn_stub(r, connect)
        pose_a = np.asarray(pose, dtype=float).reshape(6)
        q0 = self._resolve_q0_rad(q0_deg)
        code, q_list = self.algo_ik(pose_a, q_seed=q0, q_seed_deg=False)
        if code != OK:
            return code
        return self.movej(
            q_list, v, r, connect, block, q0_deg=self._rad_to_joint_list(q0), timeout_s=timeout_s
        )

    # ------------------------------------------------------------------ helpers
    def _rad_to_joint_list(self, q_rad: np.ndarray) -> list[float]:
        q = np.asarray(q_rad, dtype=float).reshape(-1)
        return [float(q[0]) * 1000.0, *np.rad2deg(q[1:]).tolist()]

    def _joint_list_to_rad(self, joint: list[float] | np.ndarray) -> np.ndarray:
        j = np.asarray(joint, dtype=float).reshape(-1)
        if j.size == self.kin.nv:
            q = np.zeros(self.kin.nv, dtype=float)
            q[0] = float(j[0]) * 0.001
            q[1:] = np.deg2rad(j[1:])
            return q
        if j.size == self.kin.nv - 1:
            q = np.zeros(self.kin.nv, dtype=float)
            q[0] = 0.4
            q[1:] = np.deg2rad(j)
            return q
        raise ValueError(f"joint size {j.size} != {self.kin.nv} or {self.kin.nv - 1}")

    def _resolve_q0_rad(self, q0_deg: list[float] | None) -> np.ndarray:
        if q0_deg is not None:
            return self._joint_list_to_rad(q0_deg)
        q = np.zeros(self.kin.nv, dtype=float)
        q[0] = 0.4
        return q
    def _plan_duration(
        self,
        q0: np.ndarray,
        q_tgt: np.ndarray,
        *,
        move_mode: str,
        v: int,
        pose: np.ndarray | None = None,
    ) -> MovePlan:
        pose_d = pose if pose is not None else self.kin.fk_pose(q_tgt)
        plan = compute_move_plan(
            self.kin,
            q0,
            q_tgt,
            pose_d,
            v_scale=_v_to_scale(v),
            move_mode=move_mode,  # type: ignore[arg-type]
        )
        return plan

    def _make_move_params(
        self,
        *,
        q0_rad: np.ndarray,
        q_target_rad: np.ndarray,
        pose_d: np.ndarray,
        plan: MovePlan,
        move_mode: str,
        v: int,
    ) -> SinToolYTaskParams:
        del v  # speed already baked into plan.duration_s via v_scale
        return SinToolYTaskParams(
            config_path=self.config_path,
            slot="wbc_arm",
            scan_duration=0.0,
            hold_at_d_s=0.0,
            rail_move_cm=0.0,
            enable_force=False,
            q0_rad=np.asarray(q0_rad, dtype=float).reshape(-1).tolist(),
            q_target_rad=np.asarray(q_target_rad, dtype=float).reshape(-1).tolist(),
            pose_d=np.asarray(pose_d, dtype=float).reshape(6).tolist(),
            plan_duration_s=float(plan.duration_s),
            plan_move_mode=move_mode,
            plan_gov_joint_max_deg=float(plan.gov_joint_max_deg),
            move_kp=2.0,
        )

    def _submit(
        self,
        params: SinToolYTaskParams,
        *,
        block: int,
        timeout_s: float | None,
    ) -> int:
        if self._client is None:
            if self.connect() != OK:
                return ERR_SEND
        assert self._client is not None
        try:
            cmd_seq = self._client.start(params)
        except Exception:
            return ERR_SEND
        if int(block) == 0:
            return OK
        # RM single-thread: block>1 means timeout seconds; else use default.
        to = float(timeout_s) if timeout_s is not None else self.default_timeout_s
        if int(block) > 1:
            to = float(block)
        deadline = time.monotonic() + to
        while time.monotonic() < deadline:
            st = self._client.read_status()
            if st is not None and int(st["status_seq"]) == int(cmd_seq):
                status = st["status"]
                if status == PhaseStatus.DONE:
                    return OK
                if status == PhaseStatus.ERROR:
                    return ERR_ARRIVAL
                if status == PhaseStatus.STOPPED:
                    return ERR_RECV
            time.sleep(0.05)
        try:
            self._client.stop()
        except Exception:
            pass
        return ERR_TIMEOUT
```

## `rm75_control/control/joint_admittance_8dof/tasks/secondary_composer.py`

```python
"""Priority-aware composition of nullspace secondary tasks.

Joint limit repulsion always runs; the arm-angle task fades out CONTINUOUSLY
as any joint approaches its physical limit (no on/off switch - a binary gate
at a fixed activation chattered against the limit-repulsion task when the
nullspace parked the arm right on the threshold).

The composed soft-task velocity (centering + arm-angle + viscous damping) is
magnitude-capped per joint: near a kinematic singularity the SR-damped
projector opens up (N -> I), and an uncapped centering gradient - large when
the posture is far from q_nominal, e.g. a straight arm at start-up - would
otherwise drive the whole arm at rad/s scale while the Cartesian task is soft.
The joint-plan feedforward ``qdot_ff`` is added AFTER the cap: it is the
primary content of a joint-space move and is already velocity-limited by the
plan itself and by the QP box.

Rail behaviour is decoupled from this composer: RailMode.COUPLED lets the QP
resolve rail motion normally; LOCKED + HOLD applies the RailLockTask below;
LOCKED + RAIL_ONLY / TCP_FIXED are driven by qdot_ff[0] plus the QP rail-vel
pin in constraint_mgr — the composer only forwards the arm portion of qdot_ff.
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import ArmAngleTask
from rm75_control.control.joint_admittance_8dof.tasks.manipulability_task import (
    ManipulabilityTask,
)
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_lock import RailLockTask


def max_limit_activation(
    q_rad: np.ndarray,
    q_mid: np.ndarray,
    half: np.ndarray,
    *,
    activation: float,
) -> float:
    """Peak limit-repulsion activation in [0, 1] (same metric as JointCenteringTask)."""
    q = np.asarray(q_rad, dtype=float)
    u_limit = (q - q_mid) / half
    span = max(1.0 - activation, 1e-6)
    over = np.clip((np.abs(u_limit) - activation) / span, 0.0, 1.0)
    return float(np.max(over))


def _smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class SecondaryComposer:
    """Compose centering + arm-angle + feedforward with limit priority."""

    def __init__(
        self,
        centering: JointCenteringTask,
        arm_task: ArmAngleTask | None,
        *,
        manipulability: ManipulabilityTask | None = None,
        rail_lock: RailLockTask | None = None,
        arm_activation_limit: float = 0.92,
        arm_fade_band: float = 0.05,
        d_null: float = 0.0,
        adaptive_d_null_gain: float = 1.0,
        v_max: np.ndarray | None = None,
        max_qdot_frac: float = 0.2,
    ) -> None:
        self.centering = centering
        self.arm_task = arm_task
        self.manipulability = manipulability
        self.rail_lock = rail_lock
        self.arm_activation_limit = float(arm_activation_limit)
        self.arm_fade_band = float(arm_fade_band)
        self.d_null = float(d_null)
        self.adaptive_d_null_gain = float(adaptive_d_null_gain)
        self.v_max = None if v_max is None else np.asarray(v_max, dtype=float)
        self.max_qdot_frac = float(max_qdot_frac)
        self.last_limit_activation: float = 0.0
        self.last_arm_smooth: float = 1.0

    @classmethod
    def from_controller_parts(
        cls,
        centering: JointCenteringTask,
        arm_task: ArmAngleTask | None,
        nullspace_cfg: NullspaceTaskConfig,
        *,
        manipulability: ManipulabilityTask | None = None,
        rail_lock: RailLockTask | None = None,
        d_null: float = 0.0,
        adaptive_d_null_gain: float = 1.0,
        v_max: np.ndarray | None = None,
        max_qdot_frac: float = 0.2,
    ) -> "SecondaryComposer":
        return cls(
            centering,
            arm_task,
            manipulability=manipulability,
            rail_lock=rail_lock,
            arm_activation_limit=nullspace_cfg.activation + 0.07,
            d_null=d_null,
            adaptive_d_null_gain=adaptive_d_null_gain,
            v_max=v_max,
            max_qdot_frac=max_qdot_frac,
        )

    def _arm_weight(self, u_max: float) -> float:
        """Continuous arm-task weight vs peak limit activation.

        1.0 while well clear of limits, smoothstep-fading to 0.0 across
        ``[arm_activation_limit - band, arm_activation_limit + band]``.  A
        continuous function of u_max cannot chatter the way the old binary
        ``u_max < limit`` gate did.
        """
        band = max(self.arm_fade_band, 1e-6)
        return _smoothstep01((self.arm_activation_limit + band - u_max) / (2.0 * band))

    def compose(
        self,
        q_rad: np.ndarray,
        qdot_ff: np.ndarray | None,
        qdot_prev: np.ndarray | None,
        *,
        arm_suppressed: bool,
        sigma_min: float = 1.0,
        sigma_ref: float = 0.08,
        sigma_escape_ref: float = 0.0,
        centering_suppressed: bool = False,
        manipulability_active: bool = False,
        centering_sigma_fade: bool = True,
        centering_gain_scale: float = 1.0,
        max_qdot_frac_override: float | None = None,
    ) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        cfg = self.centering.cfg
        u_max = max_limit_activation(
            q,
            self.centering.q_mid,
            self.centering.half,
            activation=cfg.activation,
        )
        self.last_limit_activation = u_max

        qdot_soft = np.zeros_like(q)
        rail_hold = self.rail_lock is not None and self.rail_lock.active
        # Nullspace attractor (centering) and ∇μ escape must COEXIST when σ
        # dips: replacing centering with manip alone let J5 drift to the
        # flipped wrist on long hard-contact scans.  Always apply centering
        # first (unless suppressed); add manipulability on top.  When manip is
        # active in the escape band, σ-yield proximal centering (floor 0.25) so
        # ∇μ can lead without a full tug-of-war; never kill the attractor.
        # J6 (index 6) is exempt from yield: |J6|≈0 is wrist singularity and
        # yielding the 45° nominal pull let scans park there and jitter.
        if not centering_suppressed:
            qdot_center = float(max(centering_gain_scale, 0.0)) * self.centering(q)
            escape_ref = float(sigma_escape_ref) if float(sigma_escape_ref) > 1e-9 else float(sigma_ref)
            if (
                manipulability_active
                and escape_ref > 1e-9
                and float(sigma_min) < escape_ref
            ):
                yield_scale = float(
                    np.clip(float(sigma_min) / escape_ref, 0.25, 1.0)
                )
                qdot_full = qdot_center
                qdot_center = yield_scale * qdot_center
                if qdot_center.shape[0] > 6:
                    qdot_center[6] = qdot_full[6]
            qdot_soft = qdot_center
        # Rail is a base translation: ∂μ/∂q0 is analytically zero, but the FD
        # gradient in ManipulabilityTask can produce small numerical residuals
        # that get unit-normalised to k_mu.  Always exclude rail from the
        # manipulability push — its purpose is to escape ARM singularities,
        # never to be a stealth rail driver behind the primary QP's back.
        if manipulability_active and self.manipulability is not None:
            qdot_soft = qdot_soft + self.manipulability(
                q, sigma_min=sigma_min, exclude_rail=True
            )
        if rail_hold:
            qdot_soft = qdot_soft + self.rail_lock(q)

        d_eff = self.d_null
        if self.adaptive_d_null_gain > 0.0 and u_max > 0.0:
            d_eff = d_eff * (1.0 + self.adaptive_d_null_gain * u_max)
        if d_eff > 0.0 and qdot_prev is not None:
            qdot_soft = qdot_soft - d_eff * np.asarray(qdot_prev, dtype=float)

        # Per-joint magnitude cap on the soft tasks (see module docstring).
        cap_frac = (
            self.max_qdot_frac
            if max_qdot_frac_override is None
            else float(max_qdot_frac_override)
        )
        if self.v_max is not None and cap_frac > 0.0:
            cap = cap_frac * self.v_max
            qdot_soft = np.clip(qdot_soft, -cap, cap)

        if not rail_hold:
            qdot_soft[0] = 0.0

        # Near σ≈0 attenuate centering/manip/damping — NOT arm_angle.
        # Disabled during COUPLED rail-extension scan: rail carries base
        # translation, centering keeps arm posture (Yamamoto & Yun split).
        if (
            centering_sigma_fade
            and not manipulability_active
            and sigma_min < sigma_ref
        ):
            fade = sigma_min / max(sigma_ref, 1e-6)
            qdot_soft = qdot_soft * fade

        qdot0 = qdot_soft
        if self.arm_task is not None and not arm_suppressed:
            w_arm = self._arm_weight(u_max)
            if w_arm > 0.0:
                qdot_arm = self.arm_task(q)
                self.last_arm_smooth = w_arm * float(self.arm_task.last_singularity_smooth)
                qdot0 = qdot0 + w_arm * qdot_arm
            else:
                self.last_arm_smooth = 0.0
        else:
            self.last_arm_smooth = 1.0 if self.arm_task is None else 0.0

        if qdot_ff is not None:
            qdot0 = qdot0 + np.asarray(qdot_ff, dtype=float)

        return qdot0
```

## `rm75_control/control/joint_admittance_8dof/tasks/nullspace_task.py`

```python
"""Nullspace secondary task: joint centering + limit avoidance (Liegeois 1977).

Produces a desired joint velocity `qdot0` that the CLIK/QP core projects into the
nullspace of the primary Cartesian task, so it never perturbs TCP tracking.  It
uses the redundancy of the 7-DOF arm to (a) pull joints toward the middle of
their range and (b) repel them harder as they approach a limit.

The cost being descended is the classic Liegeois manipulability/limit criterion
    H(q) = 1/2 * sum_i w_i * ((q_i - q_mid_i) / half_range_i)^2
    qdot0 = -k * dH/dq
plus a smooth activation term that grows near the limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


@dataclass
class NullspaceTaskConfig:
    k_center: float = 1.0        # centering velocity gain (rad/s per normalized unit)
    k_limit: float = 2.0         # extra repulsion gain near a limit
    activation: float = 0.8      # |u| beyond which limit repulsion ramps in (u in [-1,1])
    weights: np.ndarray | None = None   # optional per-joint weighting (len 7)
    # Centering target (rad). Defaults to the midpoint of each joint's position
    # limits, which for a symmetric elbow limit (e.g. J4 +-135deg) is 0deg - a
    # dead-straight arm. Set this to a natural "elbow bent" posture instead
    # (e.g. J4 ~ 90deg) so the redundant DOF doesn't fight the primary task by
    # trying to snap the elbow straight; see JointCenteringTask.__call__.
    q_nominal_rad: np.ndarray | None = None


class JointCenteringTask:
    """Callable secondary task: q (rad) -> qdot0 (rad/s)."""

    def __init__(
        self,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
        cfg: NullspaceTaskConfig | None = None,
    ) -> None:
        self.q_lower = np.asarray(q_lower, dtype=float)
        self.q_upper = np.asarray(q_upper, dtype=float)
        self.cfg = cfg or NullspaceTaskConfig()
        # Geometric mid/half-range: ALWAYS from the true limits, used only for the
        # limit-repulsion term below - do not confuse with the centering target.
        self.q_mid = 0.5 * (self.q_lower + self.q_upper)
        self.half = 0.5 * (self.q_upper - self.q_lower)
        # guard against zero-range joints
        self.half = np.where(self.half > 1e-9, self.half, 1.0)
        # Centering target: nominal "comfortable" posture if given, else the
        # geometric mid (which, on a symmetric elbow limit, is a straight arm).
        self.q_target = (
            self.q_mid.copy()
            if self.cfg.q_nominal_rad is None
            else np.asarray(self.cfg.q_nominal_rad, dtype=float)
        )
        self._q_target_default = self.q_target.copy()
        self.w = (
            np.ones_like(self.q_mid)
            if self.cfg.weights is None
            else np.asarray(self.cfg.weights, dtype=float)
        )

    def set_q_target(self, q_rad: np.ndarray | None = None) -> None:
        """Override the centering attractor (e.g. move-phase plan target).

        ``None`` restores the yaml ``q_nominal_deg`` default (comfortable
        posture).  Taught scan pose D is NOT the centering target — only the
        Cartesian + ψ tasks hold TCP at D; nullspace pulls toward nominal.
        For the 8-DOF controller this accepts all eight coordinates at runtime,
        but a zero rail weight means the attractor acts on the seven arm joints
        only.  RAIL pose remains owned by ``set_rail_pose_target()``.
        """
        if q_rad is None:
            self.q_target = self._q_target_default.copy()
        else:
            target = np.asarray(q_rad, dtype=float).reshape(-1)
            if target.shape != self.q_target.shape:
                raise ValueError(
                    f"q target shape {target.shape} != {self.q_target.shape}"
                )
            if not np.all(np.isfinite(target)):
                raise ValueError("q target must contain only finite values")
            self.q_target = target.copy()

    @classmethod
    def from_kinematics(
        cls, kin: RobotKinematics, cfg: NullspaceTaskConfig | None = None
    ) -> "JointCenteringTask":
        return cls(kin.q_lower, kin.q_upper, cfg)

    def __call__(self, q_rad: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        q = np.asarray(q_rad, dtype=float)

        # gradient-descent centering toward q_target (nominal posture, or geometric mid)
        u_target = (q - self.q_target) / self.half
        qdot0 = -cfg.k_center * self.w * u_target

        # smooth limit repulsion beyond activation band - always relative to the
        # TRUE joint range, independent of the centering target above.
        if cfg.k_limit > 0.0 and cfg.activation < 1.0:
            u_limit = (q - self.q_mid) / self.half     # normalized position in [-1, 1]
            span = max(1.0 - cfg.activation, 1e-6)
            over = np.clip((np.abs(u_limit) - cfg.activation) / span, 0.0, 1.0)
            qdot0 = qdot0 - cfg.k_limit * np.sign(u_limit) * (over * over)
        return qdot0
```

## `rm75_control/control/joint_admittance_8dof/tasks/manipulability_task.py`

```python
"""Nullspace secondary task: ascend Yoshikawa manipulability ∇μ(q).

During a large joint-space move near a kinematic singularity, Liegeois centering
pulls toward q_nominal (often a straight arm) and fights the plan.  This task
instead commands joint velocity along +∇μ so the redundant DOF bends away from
singular postures while the primary Cartesian / joint tracking task runs in the
task space.  The gradient is computed by central finite differences on
``RobotKinematics.manipulability`` — cheap enough at 200 Hz for nv=7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


@dataclass
class ManipulabilityTaskConfig:
    k_mu: float = 0.8          # rad/s per unit ∂μ/∂q (scaled by typical |∇μ|)
    eps_rad: float = 1e-4      # finite-difference step per joint
    # Fade manipulability ascent when σ is already healthy (avoid fighting scan).
    sigma_fade_ref: float = 0.12


class ManipulabilityTask:
    """Callable secondary task: q (rad) -> qdot0 (rad/s) along +∇μ."""

    def __init__(self, kin: RobotKinematics, cfg: ManipulabilityTaskConfig | None = None) -> None:
        self.kin = kin
        self.cfg = cfg or ManipulabilityTaskConfig()
        self.last_mu: float = 0.0
        self.last_grad_norm: float = 0.0

    def gradient(self, q_rad: np.ndarray, *, exclude_rail: bool = False) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        eps = max(float(self.cfg.eps_rad), 1e-6)
        mu0 = self.kin.manipulability(self.kin.jacobian(q))
        grad = np.zeros(self.kin.nv, dtype=float)
        for i in range(self.kin.nv):
            qp = q.copy()
            qm = q.copy()
            qp[i] += eps
            qm[i] -= eps
            mu_p = self.kin.manipulability(self.kin.jacobian(qp))
            mu_m = self.kin.manipulability(self.kin.jacobian(qm))
            grad[i] = (mu_p - mu_m) / (2.0 * eps)
        if exclude_rail:
            grad[0] = 0.0
        self.last_mu = mu0
        self.last_grad_norm = float(np.linalg.norm(grad))
        return grad

    def __call__(self, q_rad: np.ndarray, *, sigma_min: float = 1.0, exclude_rail: bool = False) -> np.ndarray:
        grad = self.gradient(q_rad, exclude_rail=exclude_rail)
        if self.last_grad_norm < 1e-12:
            return np.zeros(self.kin.nv, dtype=float)
        # Unit direction × gain; typical |∇μ| is O(0.01–0.1) near singularities.
        qdot0 = self.cfg.k_mu * grad / self.last_grad_norm
        ref = max(float(self.cfg.sigma_fade_ref), 1e-6)
        if sigma_min >= ref:
            fade = max(0.0, 1.0 - (sigma_min - ref) / ref)
            qdot0 = qdot0 * fade
        return qdot0
```

## `rm75_control/control/joint_admittance_8dof/tasks/rail_extension.py`

```python
"""Preferred arm-extension / pose-attract rail task: proactive base-arm coordination.

Two operating modes (selected by the phase preset):

* ``reach`` (scan / track) — Yamamoto & Yun 1994 preferred arm extension
  ``e = (y_tcp - y_rail) - d_pref`` plus scan feedforward; σ-escape boosts
  authority when the arm nears singularity.
* ``pose_attract`` (move→D) — soft position attractor to the *target pose's*
  rail coordinate ``y_rail_target = q_target[0]``.  Monotonic, settles and
  *stops* (no hunting).  σ_min is a *guardrail only*: with dead-zone + rate
  limit it temporarily pushes along ∂σ/∂y_rail when σ drops below a
  threshold, then hands control back to the pose attractor.  Continuous
  gradient climbing is intentionally *not* used (that caused limit cycles).

Macro-micro (Khatib/Seraji): the desired rail velocity is low-pass filtered
so the rail only absorbs the slow large-displacement component; the arm
nullspace eats the fast residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, RAIL_INDEX
from rm75_control.control.joint_admittance_8dof.tasks.rail_goodness import (
    RailGoodness,
    SigmaMinGoodness,
)


RailExtMode = Literal["reach", "pose_attract"]


def rail_vel_ff_from_reference(
    vel_ff: np.ndarray,
    kin: RobotKinematics,
    q_rad: np.ndarray,
    *,
    k_ff: float = 1.0,
) -> float:
    """Scalar rail speed from any reference ``vel_ff`` (base-frame linear vel).

    Projects the reference linear velocity onto the rail Jacobian column —
    works for sin, spline, hold-to-move, or any ``MotionReference`` that
    populates ``vel_ff[:3]`` in the base frame (as all current sources do).
    """
    v_lin = np.asarray(vel_ff[:3], dtype=float)
    j_rail = kin.jacobian(q_rad)[:3, RAIL_INDEX]
    denom = float(np.dot(j_rail, j_rail))
    if denom < 1e-12:
        return 0.0
    return float(k_ff) * float(np.dot(j_rail, v_lin) / denom)


@dataclass
class RailExtensionConfig:
    enabled: bool = True
    k_ext: float = 1.0
    # Base-frame reference linear velocity feedforward (Yamamoto & Yun 1996):
    # callers pass ``MotionReference.vel_ff``; the rail column projection is
    # trajectory-agnostic (sin, spline, segment, ...).
    k_ff: float = 1.0
    v_ff_thr_m_s: float = 0.01
    v_ff_span_m_s: float = 0.03
    e0_m: float = 0.05
    e1_m: float = 0.15
    w_max: float = 1.5
    v_max_m_s: float = 0.08
    # Fade the task to zero within this distance (m) of a rail travel limit
    # when the desired velocity points into the limit.
    limit_margin_m: float = 0.08
    # σ-escape: boost rail weight and add non-reach velocity along σ ascent
    # when σ_min dips. Keep w_max*(1+k_sigma_boost) ≪ W_task.
    k_sigma_boost: float = 2.0
    # Escape velocity scale [m/s per unit σ].
    k_esc: float = 0.5
    # Baseline weight when reach error is in the dead zone but σ is low.
    w_sigma_floor: float = 1.0
    # --- move→D pose attractor (primary during preset="move") ---
    k_pose: float = 2.0          # 1/s soft P on (y_target - y_rail)
    pose_e0_m: float = 0.005     # settle dead-zone (m); stops hunting at target
    pose_e1_m: float = 0.04      # full pose-attract weight by this error
    pose_w_max: float = 4.0      # ≪ W_task=100
    # σ guardrail (pose_attract): only engages below enter, clears above exit.
    sigma_guard_enter: float = 0.45
    sigma_guard_exit: float = 0.70
    # Cap on guardrail velocity so it cannot yank the rail off the pose path.
    v_guard_max_m_s: float = 0.04
    # Macro-micro LPF on the *desired* rail velocity (seconds).
    v_lpf_tau_s: float = 0.12


def _smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class RailExtensionTask:
    """Callable: q (rad/m) -> (v_rail_des m/s, w_ext) for the WBC QP."""

    def __init__(
        self,
        kin: RobotKinematics,
        cfg: RailExtensionConfig | None = None,
        *,
        goodness: RailGoodness | None = None,
    ) -> None:
        self.kin = kin
        self.cfg = cfg or RailExtensionConfig()
        self.goodness: RailGoodness = goodness or SigmaMinGoodness(kin)
        self.d_pref_m: float | None = None
        self.y_rail_target_m: float | None = None
        self.mode: RailExtMode = "reach"
        self.last_err_m: float = 0.0
        self.last_weight: float = 0.0
        self.last_limit_saturated: bool = False
        self._guard_active: bool = False
        self._v_lpf: float = 0.0
        self._v_lpf_initialized: bool = False

    def set_mode(self, mode: RailExtMode) -> None:
        mode_s = str(mode).strip().lower()
        if mode_s not in ("reach", "pose_attract"):
            raise ValueError(f"unknown rail extension mode {mode!r}")
        if mode_s != self.mode:
            # Reset LPF on mode switch so a scan FF residue does not kick move.
            self._v_lpf = 0.0
            self._v_lpf_initialized = False
            self._guard_active = False
        self.mode = mode_s  # type: ignore[assignment]

    def set_rail_pose_target(self, y_rail_m: float | None) -> None:
        """Set / clear the move→D soft attractor target (metres)."""
        if y_rail_m is None:
            self.y_rail_target_m = None
            return
        lo = float(self.kin.q_lower[RAIL_INDEX])
        hi = float(self.kin.q_upper[RAIL_INDEX])
        self.y_rail_target_m = float(np.clip(float(y_rail_m), lo, hi))

    def extension(self, q_rad: np.ndarray) -> float:
        """Arm Y-extension: base-frame TCP y minus rail position (m)."""
        q = np.asarray(q_rad, dtype=float)
        y_tcp = float(self.kin.fk_placement(q).translation[1])
        return y_tcp - float(q[RAIL_INDEX])

    def capture_reference(self, q_rad: np.ndarray) -> None:
        self.d_pref_m = self.extension(q_rad)

    def reset(self, q_rad: np.ndarray) -> None:
        self.capture_reference(q_rad)
        self.last_err_m = 0.0
        self.last_weight = 0.0
        self.last_limit_saturated = False
        self._guard_active = False
        self._v_lpf = 0.0
        self._v_lpf_initialized = False

    def _limit_saturation(self, q_rail: float, v: float) -> float:
        """Return 0..1 scale; C¹ smoothstep fade before a directional hard stop.

        Fades only when moving *into* a limit so reversing away from a pinned
        rail recovers authority immediately.  At the physical stop the scale is
        0; with a wide enough ``limit_margin_m`` the fade completes before pin.
        """
        margin = float(self.cfg.limit_margin_m)
        if margin <= 1e-6:
            self.last_limit_saturated = False
            return 1.0

        lo = float(self.kin.q_lower[RAIL_INDEX])
        hi = float(self.kin.q_upper[RAIL_INDEX])

        if v > 1e-9:
            if q_rail >= hi:
                self.last_limit_saturated = True
                return 0.0
            if q_rail > hi - margin:
                u = float(np.clip((hi - q_rail) / margin, 0.0, 1.0))
                self.last_limit_saturated = False
                return _smoothstep01(u)

        elif v < -1e-9:
            if q_rail <= lo:
                self.last_limit_saturated = True
                return 0.0
            if q_rail < lo + margin:
                u = float(np.clip((q_rail - lo) / margin, 0.0, 1.0))
                self.last_limit_saturated = False
                return _smoothstep01(u)

        self.last_limit_saturated = False
        return 1.0

    def _macro_lpf(self, v: float, *, dt_s: float | None) -> float:
        """First-order LPF so the rail only takes the slow (macro) component."""
        tau = float(self.cfg.v_lpf_tau_s)
        if tau <= 1e-6 or dt_s is None or dt_s <= 0.0:
            self._v_lpf = float(v)
            self._v_lpf_initialized = True
            return float(v)
        if not self._v_lpf_initialized:
            self._v_lpf = float(v)
            self._v_lpf_initialized = True
            return float(v)
        alpha = float(dt_s) / (tau + float(dt_s))
        self._v_lpf = (1.0 - alpha) * self._v_lpf + alpha * float(v)
        return float(self._v_lpf)

    def _sigma_guard_hold(self, sigma_scale: float) -> bool:
        """Update and return the σ-guard latch (enter/exit hysteresis).

        True means "σ is unhealthy enough that escape outranks the primary".
        Shared by both modes so ``reach`` and ``pose_attract`` switch over at
        the same σ instead of each inventing their own rule.
        """
        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        if self._guard_active:
            if sig >= float(self.cfg.sigma_guard_exit):
                self._guard_active = False
        elif sig < float(self.cfg.sigma_guard_enter):
            self._guard_active = True
        return self._guard_active

    def _sigma_guard_velocity(
        self,
        *,
        sigma_scale: float,
        sigma_grad_rail: float,
        v_primary: float,
    ) -> float:
        """Dead-zoned σ guardrail: engage only when σ is unhealthy.

        Hysteresis (enter/exit) prevents chatter.  Never fights a strong
        primary attractor (same anti-oppose rule as the old σ-escape).
        """
        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        if not self._sigma_guard_hold(sig):
            return 0.0
        v_g = float(self.cfg.k_esc) * (1.0 - sig) * float(sigma_grad_rail)
        v_g = float(np.clip(v_g, -self.cfg.v_guard_max_m_s, self.cfg.v_guard_max_m_s))
        if v_g * v_primary < 0.0 and abs(v_primary) > 1.0e-4:
            return 0.0
        return v_g

    def _call_pose_attract(
        self,
        q: np.ndarray,
        *,
        sigma_escape_scale: float,
        sigma_grad_rail: float,
        dt_s: float | None,
    ) -> tuple[float, float]:
        sigma_scale = sigma_escape_scale
        if self.y_rail_target_m is None:
            self.last_err_m = 0.0
            self.last_weight = 0.0
            self.last_limit_saturated = False
            return 0.0, 0.0
        y = float(q[RAIL_INDEX])
        err = float(self.y_rail_target_m) - y  # +err → move rail toward target
        self.last_err_m = err
        e0 = float(self.cfg.pose_e0_m)
        e1 = max(float(self.cfg.pose_e1_m), e0 + 1e-6)
        span = e1 - e0
        w_pose = float(self.cfg.pose_w_max) * _smoothstep01((abs(err) - e0) / span)
        v_pose = float(
            np.clip(self.cfg.k_pose * err, -self.cfg.v_max_m_s, self.cfg.v_max_m_s)
        )
        # Inside settle dead-zone: primary is exactly zero (stop hunting).
        if abs(err) <= e0:
            v_pose = 0.0
        v_guard = self._sigma_guard_velocity(
            sigma_scale=sigma_scale,
            sigma_grad_rail=sigma_grad_rail,
            v_primary=v_pose,
        )
        v_total = v_pose + v_guard
        v_total = float(np.clip(v_total, -self.cfg.v_max_m_s, self.cfg.v_max_m_s))
        v_total = self._macro_lpf(v_total, dt_s=dt_s)
        lim = self._limit_saturation(y, v_total)
        self.last_limit_saturated = lim < 1e-6
        v_total *= lim
        # Guardrail alone still needs a floor weight so the QP can act when
        # the pose error is already inside the dead-zone but σ is bad.
        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        w_guard = float(self.cfg.w_sigma_floor) * (1.0 - sig) if self._guard_active else 0.0
        w = (w_pose + w_guard) * lim
        self.last_weight = w
        return v_total, w

    def _call_reach(
        self,
        q: np.ndarray,
        *,
        sigma_scale: float,
        sigma_escape_scale: float,
        sigma_grad_rail: float,
        vel_ff: np.ndarray | None,
        dt_s: float | None,
    ) -> tuple[float, float]:
        if self.d_pref_m is None:
            self.capture_reference(q)
        err = self.extension(q) - float(self.d_pref_m)
        span = max(float(self.cfg.e1_m) - float(self.cfg.e0_m), 1e-6)
        # Reach term (unchanged Yamamoto-Yun coordination).
        w_reach = float(self.cfg.w_max) * _smoothstep01(
            (abs(err) - float(self.cfg.e0_m)) / span
        )
        v_reach = float(
            np.clip(self.cfg.k_ext * err, -self.cfg.v_max_m_s, self.cfg.v_max_m_s)
        )
        # Mirror pose_attract: inside the dead zone the primary is exactly
        # zero.  A residual v_reach there is never large enough to move the
        # rail, but it is large enough to veto the σ-escape below.
        if abs(err) <= float(self.cfg.e0_m):
            v_reach = 0.0
        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        sig_esc = float(np.clip(sigma_escape_scale, 0.0, 1.0))
        v_ff = (
            rail_vel_ff_from_reference(vel_ff, self.kin, q, k_ff=self.cfg.k_ff)
            if vel_ff is not None
            else 0.0
        )
        v_ff *= sig
        # σ-escape: extra rail velocity along the TCP-preserving σ-ascent
        # direction; kicks in even when |err| < e0 (dead zone) if σ drops.
        v_escape = float(self.cfg.k_esc) * (1.0 - sig_esc) * float(sigma_grad_rail)
        v_primary = v_ff + v_reach
        # Anti-oppose, σ-gated exactly like the pose_attract guardrail: while
        # σ is healthy the escape is a soft preference and must not hunt
        # against reach/FF, but once σ drops past the guard threshold escape
        # is the whole point and has to be allowed to win.  This used to be
        # unconditional, so at hybrid@D — where the force axis keeps a small
        # reach error alive — the escape was vetoed on every tick no matter
        # how far σ fell.
        if self._sigma_guard_hold(sig_esc):
            pass
        elif v_escape * v_primary < 0.0 and abs(v_primary) > 1.0e-4:
            v_escape = 0.0
        v_total = v_primary + v_escape
        v = float(np.clip(v_total, -self.cfg.v_max_m_s, self.cfg.v_max_m_s))
        v = self._macro_lpf(v, dt_s=dt_s)
        # Rail-limit fade (applies to the combined velocity).
        lim = self._limit_saturation(float(q[RAIL_INDEX]), v)
        self.last_limit_saturated = lim < 1e-6
        v *= lim
        thr = float(self.cfg.v_ff_thr_m_s)
        span_ff = max(float(self.cfg.v_ff_span_m_s), 1e-6)
        w_ff = float(self.cfg.w_max) * _smoothstep01((abs(v_ff) - thr) / span_ff) * sig
        # Weight: reach + scan feedforward + σ-baseline floor, then σ-boost.
        w = (w_reach + w_ff + float(self.cfg.w_sigma_floor) * (1.0 - sig_esc)) * lim
        sig_boost = 1.0 + float(self.cfg.k_sigma_boost) * (1.0 - sig_esc)
        w *= sig_boost
        self.last_err_m = float(err)
        self.last_weight = w
        return v, w

    def __call__(
        self,
        q_rad: np.ndarray,
        *,
        sigma_scale: float = 1.0,
        sigma_escape_scale: float | None = None,
        sigma_grad_rail: float = 0.0,
        vel_ff: np.ndarray | None = None,
        dt_s: float | None = None,
    ) -> tuple[float, float]:
        """Return ``(v_rail_des, w_ext)`` for the QP.

        Args
        ----
        q_rad : current command joint vector.
        sigma_scale : 1.0 when σ_min is healthy (``≥ sigma_ref``), 0.0 at
            deep singularity.  This is the σ-health scalar computed by the
            loop, NOT the raw σ_min.  Fades the *scan feedforward*: as the arm
            gets ill-conditioned the rail stops chasing the scan reference.
        sigma_escape_scale : the same kind of scalar but measured against the
            earlier ``sigma_escape_ref`` (default 2·σ_ref); drives σ-escape
            velocity, the ``w_sigma_floor`` baseline, the w-boost and the
            guard latch.  Avoidance has to start before the loop's twist
            brake does — the rail is accel-limited and cannot respond within
            the tick the brake fires.  Defaults to ``sigma_scale``.
        sigma_grad_rail : ``d σ_min / d y_rail`` under TCP-preservation
            (:mod:`rm75_control.control.joint_admittance_8dof.solver.sigma_grad`).
            Sign tells us which rail direction escapes the singularity.
            Prefer sourcing this from a :class:`RailGoodness` implementation
            (default: :class:`SigmaMinGoodness`).
        dt_s : optional control period for the macro-micro LPF.
        """
        if not self.cfg.enabled:
            self.last_err_m = 0.0
            self.last_weight = 0.0
            self.last_limit_saturated = False
            return 0.0, 0.0
        q = np.asarray(q_rad, dtype=float)
        sig_esc = (
            float(sigma_scale)
            if sigma_escape_scale is None
            else float(sigma_escape_scale)
        )
        if self.mode == "pose_attract":
            return self._call_pose_attract(
                q,
                sigma_escape_scale=sig_esc,
                sigma_grad_rail=sigma_grad_rail,
                dt_s=dt_s,
            )
        return self._call_reach(
            q,
            sigma_scale=sigma_scale,
            sigma_escape_scale=sig_esc,
            sigma_grad_rail=sigma_grad_rail,
            vel_ff=vel_ff,
            dt_s=dt_s,
        )
```

## `rm75_control/control/joint_admittance_8dof/tasks/rail_lock.py`

```python
"""Rail prismatic DOF hold task (used only in RailMode.LOCKED + LockedStyle.HOLD).

The other LOCKED styles (RAIL_ONLY / TCP_FIXED) do not use this task: they let
the external plan drive ``qdot_ff[0]`` and the QP box pin the rail velocity to
that value.  RailMode.COUPLED lets the QP decide rail motion itself (subject to
reg / v_max / a_max / resync from the standard SafetyLimits path).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RAIL_INDEX
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode


@dataclass
class RailLockConfig:
    """Rail control configuration.

    Fields under "lock_*" only take effect in ``LOCKED + HOLD``.  ``v_max_m_s``
    and travel/visual metadata apply to all modes.
    """

    mode: RailMode = RailMode.LOCKED
    locked_style: LockedStyle = LockedStyle.HOLD
    q_ref_m: float | None = None
    # HOLD-only knobs
    lock_gain: float = 200.0
    lock_reg_scale: float = 100.0  # multiply qp.reg[0] when HOLD-locked
    lock_vel_eps_m_s: float = 0.0  # rail velocity box in HOLD (m/s)
    lock_hard_pin: bool = True     # after QP, pin q_cmd[0] = q_ref every tick
    # Rail speed / geometry (used by planners and safety limits)
    v_max_m_s: float | None = None
    travel_m: float = 0.80         # [0, travel_m] m (rail_y=0 at -Y end)

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = RailMode(self.mode)
        if isinstance(self.locked_style, str):
            self.locked_style = LockedStyle(self.locked_style)

    @property
    def is_locked_hold(self) -> bool:
        return self.mode == RailMode.LOCKED and self.locked_style == LockedStyle.HOLD


class RailLockTask:
    """When ``LOCKED + HOLD``, pull rail_y toward q_ref (m/s per m error)."""

    def __init__(self, cfg: RailLockConfig | None = None) -> None:
        self.cfg = cfg or RailLockConfig()
        self.q_ref = self.cfg.q_ref_m

    def reset(self, q_rad: np.ndarray) -> None:
        # Always re-capture: yaml may seed q_ref_m=0.0 as a placeholder, and HOLD
        # without an explicit ref means "hold where we are now".
        self.q_ref = float(np.asarray(q_rad, dtype=float)[RAIL_INDEX])

    def set_reference(self, q_ref_m: float) -> None:
        self.q_ref = float(q_ref_m)

    @property
    def active(self) -> bool:
        """Task is only meaningful in LOCKED + HOLD."""
        return self.cfg.is_locked_hold and self.q_ref is not None

    def __call__(self, q_rad: np.ndarray) -> np.ndarray:
        qdot0 = np.zeros_like(np.asarray(q_rad, dtype=float))
        if not self.active:
            return qdot0
        err = float(q_rad[RAIL_INDEX]) - float(self.q_ref)
        qdot0[RAIL_INDEX] = -self.cfg.lock_gain * err
        return qdot0
```

## `rm75_control/control/joint_admittance_8dof/tasks/arm_angle.py`

```python
"""S-R-S arm-angle (swivel) redundancy parametrization for the RM75-F.

The RM75 is a spherical-shoulder (J1-J3), elbow (J4), spherical-wrist (J5-J7)
arm, so its single redundant DOF has an exact geometric coordinate: the swivel
angle psi - the rotation of the elbow point E about the shoulder-wrist axis SW,
measured from a fixed reference plane (Shimizu et al. 2008; Kreutz-Delgado).

Using psi as an explicit nullspace coordinate is more deterministic than a
joint-space posture attractor: holding psi_ref pins the elbow branch exactly
(the value observed at the IK solution / teach pose), while the primary
Cartesian task and the joint-limit repulsion stay untouched.

Frames used (verified against the URDF: |S-E| = 256 mm, |E-W| = 210 mm,
joint_1/2, joint_3/4, joint_5/6 pairs are coincident):

    S = origin of joint_2  (shoulder center, fixed in base)
    E = origin of joint_4  (elbow center)
    W = origin of joint_6  (wrist center)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from rm75_control.control.joint_admittance_8dof.ik_types import project_onto_task_nullspace
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics

_SHOULDER_JOINT = "joint_2"
_ELBOW_JOINT = "joint_4"
_WRIST_JOINT = "joint_6"

# Reference direction defining psi = 0: the base -Z axis projected off the SW
# axis ("elbow hanging down" plane).  Any fixed vector not parallel to SW works;
# base Z is a good choice for a table-mounted arm.
_V_REF = np.array([0.0, 0.0, -1.0])


@dataclass
class ArmAngleTaskConfig:
    enabled: bool = False
    k_psi: float = 1.0            # swivel tracking gain (1/s)
    psi_ref_rad: float | None = None   # None -> capture at reset()
    fd_eps_rad: float = 1e-4      # central-difference step for the gradient
    safe_denom_eps: float = 1e-4  # floor on grad_psi . gN to prevent blow-up
    # exp(-gain * obs^2) attenuation near the algorithmic singularity, with
    # obs = (ne / |E-S|) * nr, both DIMENSIONLESS in [0, 1].  (An earlier
    # version used ne in meters (~0.16 max on the RM75) with gain 100, which
    # attenuated the task to ~3% in EVERY posture - psi looked "stuck".)
    obs_decay_gain: float = 400.0
    max_qdot_frac: float = 0.15   # clip |qdot| to this fraction of v_max per joint
    # Global posture attractor for the SRS planner (pose_ik.resolve_pose_ik_srs).
    # ψ_home is the target the enumeration pulls toward on every new pose so
    # the arm stays in a consistent "posture family" (elbow always to the
    # same side, no random re-branching).  ``None`` means "capture the swivel
    # angle of the controller's very first reset()" — the arm defaults to
    # whatever posture the operator taught.
    psi_home_rad: float | None = None
    # Hard-cap the ψ swing allowed by the planner (used by resolve_pose_ik_srs).
    # An IK candidate whose ψ is more than this away from ψ_seed is dropped
    # before the goal-score ranking — this is the anti-twist guard.
    max_psi_swing_rad: float = 150.0 * np.pi / 180.0
    # Optional absolute ψ envelope (e.g. cable-carrier protection).  None
    # disables the hard limit; if set, both ends are checked.
    psi_hard_lower_rad: float | None = None
    psi_hard_upper_rad: float | None = None


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class ArmAngleTask:
    """Callable secondary task: q (rad) -> qdot0 (rad/s) tracking psi_ref.

    The raw gradient grad_psi mostly points along directions that also move the
    TCP (psi changes when the wrist moves), so a naive gradient step nearly
    vanishes after nullspace projection.  The step is therefore normalized with
    the TASK-NULLSPACE-projected gradient gN = N(J) grad_psi:

        qdot0 = k_psi * wrap(psi_ref - psi(q)) * gN / (grad_psi . gN)

    which gives d(psi)/dt = k_psi * err exactly on the self-motion manifold
    (the QP re-projects, which is idempotent on gN).
    """

    def __init__(self, kin: RobotKinematics, cfg: ArmAngleTaskConfig | None = None) -> None:
        self.kin = kin
        self.cfg = cfg or ArmAngleTaskConfig()
        self.psi_ref = self.cfg.psi_ref_rad
        # Continuous (unwrapped) swivel target — arctan2 reports psi in
        # (-pi, pi]; crossing the branch fires a 2pi jump in the raw angle
        # while the arm barely moves, which makes wrap(psi_ref-psi) look like
        # a huge nullspace error and drives violent swivel chatter on hardware.
        self._psi_ref_unwrapped: float | None = None
        # NOTE: psi is analytically invariant to the rail position q[0]: S, E
        # and W all translate together with the base, so SW / SE (and hence
        # psi) are unchanged.  An earlier patch froze the rail coordinate for
        # the psi geometry ("_rail_ref_m"); it was a no-op and has been
        # removed (see tests/test_arm_angle_rail_invariance.py).
        self._model = kin.model
        self._data = self._model.createData()
        self._jids = tuple(
            self._model.getJointId(n) for n in (_SHOULDER_JOINT, _ELBOW_JOINT, _WRIST_JOINT)
        )
        self.last_singularity_smooth: float = 1.0

    def _sw_observability(self, q_rad: np.ndarray) -> tuple[float, float, float]:
        """Return (ne_norm, nr, obs) for algorithmic-singularity attenuation.

        ``ne_norm`` = elbow off-axis offset normalized by the upper-arm length
        |E-S| (sin of the shoulder-elbow angle off the SW axis) and ``nr`` =
        |V_REF x w_hat| (sin of the SW-vs-reference-vector angle) are both
        dimensionless in [0, 1]; their product is the observability measure.
        """
        q = np.asarray(q_rad, dtype=float)
        pin.forwardKinematics(self._model, self._data, q)
        s_id, e_id, w_id = self._jids
        S = np.asarray(self._data.oMi[s_id].translation)
        E = np.asarray(self._data.oMi[e_id].translation)
        W = np.asarray(self._data.oMi[w_id].translation)
        sw = W - S
        n_sw = float(np.linalg.norm(sw))
        se = E - S
        n_se = float(np.linalg.norm(se))
        if n_sw < 1e-9 or n_se < 1e-9:
            return 0.0, 0.0, 0.0
        w_hat = sw / n_sw
        e_perp = se - np.dot(se, w_hat) * w_hat
        r_perp = _V_REF - np.dot(_V_REF, w_hat) * w_hat
        ne = float(np.linalg.norm(e_perp)) / n_se
        nr = float(np.linalg.norm(r_perp))
        return ne, nr, ne * nr

    # ---- geometry ----------------------------------------------------------
    def arm_angle(self, q_rad: np.ndarray) -> float:
        """Swivel angle psi(q) in (-pi, pi] (invariant to the rail q[0])."""
        q = np.asarray(q_rad, dtype=float)
        pin.forwardKinematics(self._model, self._data, q)
        s_id, e_id, w_id = self._jids
        S = np.asarray(self._data.oMi[s_id].translation)
        E = np.asarray(self._data.oMi[e_id].translation)
        W = np.asarray(self._data.oMi[w_id].translation)

        sw = W - S
        n_sw = float(np.linalg.norm(sw))
        if n_sw < 1e-9:
            return 0.0
        w_hat = sw / n_sw

        # Elbow direction and reference direction, both projected off the SW axis.
        e_perp = (E - S) - np.dot(E - S, w_hat) * w_hat
        r_perp = _V_REF - np.dot(_V_REF, w_hat) * w_hat
        ne = float(np.linalg.norm(e_perp))
        nr = float(np.linalg.norm(r_perp))
        if ne < 1e-9 or nr < 1e-9:
            # Arm fully stretched (elbow on the SW axis) or SW parallel to the
            # reference vector: psi is undefined; report 0 (gradient ~0 too).
            return 0.0
        e_u = e_perp / ne
        r_u = r_perp / nr
        return float(np.arctan2(np.dot(np.cross(r_u, e_u), w_hat), np.dot(r_u, e_u)))

    def _psi_unwrapped(self, q_rad: np.ndarray) -> float:
        """Swivel angle continuous near the active reference (no ±pi branch flip)."""
        psi = self.arm_angle(q_rad)
        if self._psi_ref_unwrapped is None:
            return psi
        return float(self._psi_ref_unwrapped + _wrap_pi(psi - self._psi_ref_unwrapped))

    def grad_arm_angle(self, q_rad: np.ndarray) -> np.ndarray:
        """d psi / d q via central differences on arm joints only (rail excluded)."""
        q = np.asarray(q_rad, dtype=float)
        eps = self.cfg.fd_eps_rad
        g = np.zeros_like(q)
        for i in range(1, q.size):
            qp = q.copy()
            qm = q.copy()
            qp[i] += eps
            qm[i] -= eps
            g[i] = (self._psi_unwrapped(qp) - self._psi_unwrapped(qm)) / (2.0 * eps)
        return g

    # ---- task interface ------------------------------------------------------
    def reset(self, q_rad: np.ndarray) -> None:
        """Capture psi_ref from the current configuration if not already set
        (by config or an explicit set_reference from the application)."""
        q = np.asarray(q_rad, dtype=float)
        if self.psi_ref is None:
            self.psi_ref = self.arm_angle(q)
        self._psi_ref_unwrapped = float(self.psi_ref)

    def set_reference(self, psi_ref_rad: float) -> None:
        psi_ref_rad = float(psi_ref_rad)
        if self._psi_ref_unwrapped is not None:
            self._psi_ref_unwrapped = float(
                self._psi_ref_unwrapped + _wrap_pi(psi_ref_rad - self._psi_ref_unwrapped)
            )
        else:
            self._psi_ref_unwrapped = psi_ref_rad
        self.psi_ref = psi_ref_rad

    def __call__(self, q_rad: np.ndarray) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        if self.psi_ref is None:
            self.reset(q)
        psi = self._psi_unwrapped(q)
        g = self.grad_arm_angle(q)
        if float(np.dot(g, g)) < 1e-10:
            return np.zeros_like(q)
        J = self.kin.jacobian(q)
        sigma = self.kin.singular_values(J)
        sigma_min = float(sigma.min())
        # Kinematic nullspace only — must match qp.use_dyn_nullspace (off on
        # RM75) so d(psi)/dt ~= k_psi * err in the executed QP solution.
        gN = project_onto_task_nullspace(J, g, sigma_min=sigma_min)
        denom = float(np.dot(g, gN))
        if denom < 1e-10:
            # psi not controllable within the task nullspace at this q
            return np.zeros_like(q)
        err = float(self._psi_ref_unwrapped) - psi
        _, _, obs = self._sw_observability(q)
        smooth = 1.0 - np.exp(-self.cfg.obs_decay_gain * obs * obs)
        self.last_singularity_smooth = float(smooth)
        safe_denom = denom + self.cfg.safe_denom_eps
        qdot = smooth * self.cfg.k_psi * err * gN / safe_denom
        v_cap = self.cfg.max_qdot_frac * np.asarray(self.kin.v_max, dtype=float)
        return np.clip(qdot, -v_cap, v_cap)
```

## `rm75_control/control/joint_admittance_8dof/solver/qp_builder.py`

```python
"""WBC velocity-IK core: slack-variable QP + CBF self-collision constraints.

Formulation (Escande et al. 2014 slack task + Faverjon velocity damper / Khazoom CBF):

    x = [qdot; w]  in R^{nv+6}

    min  0.5 (qdot - qdot_nom)^T W_reg (qdot - qdot_nom) + 0.5 w^T W_task w
    s.t. J_tcp qdot - w = v_cmd                     (equality)
         l_box <= qdot <= u_box                     (joint boxes)
         J_col qdot >= v_safe                       (CBF, optional)

H is block-diagonal (no J^T J).  ProxQP warm-started each tick.

This layer consumes a *given* task twist ``v_cmd`` verbatim (Escande et al. 2014
Sec. III): the position-feedback loop that produces the twist lives exactly once
in the caller (outer loop / pose_ik), never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import (
    CollisionConfig,
    CollisionModel,
)
from rm75_control.control.joint_admittance_8dof.ik_types import (
    IkStepResult,
    SrDampingConfig,
    project_onto_task_nullspace,
    sr_damping_lambda,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import (
    CbfSlotTracker,
    build_cbf_rows,
)
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    VelocityBoxConstraints,
    build_wbc_inequalities,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits

N_SLACK = 6


@dataclass
class QpConfig:
    task_weight: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=float)
    )
    # Effort allocation for ultrasound scanning on a 7-DOF arm + rail:
    #
    #   idx 0   rail (prismatic, m)      1.0e-2  — same as shoulder; primary
    #                                              task recruits rail for base-Y
    #                                              when sigma dips. Secondary
    #                                              rail drive is zeroed in qp;
    #                                              patient limits are v_max /
    #                                              a_max_rail, not a 5x reg tax.
    #   idx 1-4 shoulder/elbow           1.0e-2  — base motion is fine for
    #                                              gross pose adjustments.
    #   idx 5-7 wrist 1/2/3              5.0e-3  — cheapest: fine-scale
    #                                              orientation (probe tilt)
    #                                              is exactly what a scan
    #                                              wants to do with the
    #                                              wrist, not the shoulder.
    #
    # With ``use_mass_weighted_reg=True`` these baseline weights are further
    # multiplied by ``max(diag(M(q)), mass_reg_floor)`` — heavier joints
    # (shoulder) become naturally more expensive than the wrist even inside
    # the arm cluster.  Mass weighting keeps shoulder dearer than wrist; rail
    # joins the primary equality when the arm Jacobian is ill-conditioned.
    reg: np.ndarray = field(
        default_factory=lambda: np.array(
            [1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 5.0e-3, 5.0e-3, 5.0e-3],
            dtype=float,
        )
    )
    backend: str = "proxqp"
    eps_abs: float = 1e-6
    max_iter: int = 200
    # Clamp applied in ProxQP backend so a yaml typo (e.g. 3000) cannot freeze
    # the 200 Hz loop for seconds near singularities / CBF.
    max_iter_cap: int = 400
    euler_order: str = "xyz"
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    # Chiaverini 1997 SR damping for nullspace projection.
    sr_damping: SrDampingConfig = field(default_factory=SrDampingConfig)
    # Soften W_task as σ_min drops so slack absorbs infeasible twists.
    task_weight_min_frac: float = 0.05
    task_weight_lpf_tau_s: float = 0.25
    # Weight QP reg by diag(M(q)) for dynamics-consistent nullspace resolution.
    use_mass_weighted_reg: bool = True
    # Floor on diag(M) in the mass-weighted reg: wrist inertias are ~1e-3,
    # which drove the effective reg to ~1e-6 x task_weight and ill-conditioned
    # the QP (occasional ProxQP failures = one-tick freezes).
    mass_reg_floor: float = 0.05
    # Exempt the rail (joint 0) from mass weighting.  diag(M)[0] is the full
    # carriage + arm mass (~9.8 kg on the RM75 rig), which priced rail motion
    # 30-400x above the arm joints: the QP stretched the arm to near-straight
    # (sigma_arm ~ 0.03) before rail motion became marginally cheaper.  With
    # the exemption the rail's effective reg is exactly ``reg[0]`` — an
    # absolute, yaml-tunable cost, sized against the arm's mass-weighted regs.
    mass_weight_exempt_rail: bool = True
    # LPF time constant (s) on the mass-weighted reg diagonal.  diag(M(q))
    # re-evaluated every tick makes H change tick-to-tick, degrading ProxQP
    # warm starts (a vibration input near singular poses where iteration
    # counts already spike).  0 disables (legacy per-tick behaviour).
    mass_reg_lpf_tau_s: float = 0.2
    # Use Khatib N_dyn instead of kinematic N in secondary projection.
    use_dyn_nullspace: bool = True
    # Faverjon/Tournassoud joint-limit velocity damper band: allowed speed
    # toward a limit ramps to 0 across this zone before the margin.  Units are
    # PER JOINT: rad for the arm, metres for the prismatic rail.  The old
    # scalar band applied 0.15 "rad" = 0.15 m to the rail — the damper started
    # throttling rail velocity from |y| > 6.5 cm (60% of the ±0.25 m travel),
    # exactly where the rail is needed most to rescue arm singularities.
    limit_damper_band_rad: float = 0.15      # arm joints 1..7 (rad)
    limit_damper_band_rail_m: float = 0.05   # rail joint 0 (metres)
    warn_on_fail: bool = True
    # On ProxQP failure: qdot ← fail_qdot_decay * qdot_prev (not a hard 0.5
    # chop — that was a one-tick jerk when the solver hiccupped).
    fail_qdot_decay: float = 0.85
    # Hard wall-clock budget for one ProxQP attempt+retry (ms).  Exceeding
    # this skips the retry and returns fail — prevents GIL freezes of
    # multiple seconds near σ→0 that starve the rail Modbus loop (PANIC).
    max_solve_ms: float = 8.0
    # Below this σ_min, Cartesian twist (incl. force) is scaled down so
    # nullspace escape / rail recruitment can win over force-driven collapse.
    twist_sigma_floor: float = 0.08
    # Avoidance onset = sigma_ref * scale.  Must lead the twist brake (>1) so
    # rail/∇μ can accelerate before Cartesian is clamped, but stay below the
    # healthy-D band (σ≈0.11–0.13) — 2.0 kept D permanently escaping.
    sigma_escape_ref_scale: float = 1.25


class _ProxQpWbcBackend:
    def __init__(self, nv: int, max_cbf: int, cfg: QpConfig) -> None:
        import proxsuite

        self._px = proxsuite
        self.nv = nv
        self.n_slack = N_SLACK
        self.n_var = nv + self.n_slack
        self.n_eq = N_SLACK
        self.n_in = nv + max_cbf
        self.qp = proxsuite.proxqp.dense.QP(self.n_var, self.n_eq, self.n_in)
        self._eps_tight = float(cfg.eps_abs)
        # Retry tolerance near singularities: ProxQP hits MAX_ITER when the
        # equality Jqdot=w+v_cmd is nearly rank-deficient (σ→0).  A ~100x
        # looser eps on the retry lets the solver accept "good enough" without
        # a full-stop fallback; typical converged residuals are already
        # 1e-5..1e-4 in this regime.
        self._eps_loose = max(self._eps_tight * 100.0, 1.0e-4)
        # Store max_iter locally — do NOT keep self.cfg (retry must not touch it).
        # Cap for realtime: yaml historically had 3000 and a single failed tick
        # could hold the GIL for >10 s (looks like mid-MoveJ freeze, no fault).
        cap = int(getattr(cfg, "max_iter_cap", 400) or 400)
        self._max_iter = int(min(max(int(cfg.max_iter), 1), max(cap, 1)))
        self.qp.settings.eps_abs = self._eps_tight
        self.qp.settings.max_iter = self._max_iter
        self.qp.settings.initial_guess = (
            proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        )
        self._initialized = False
        self.fail_count = 0
        self._warn_on_fail = bool(cfg.warn_on_fail)
        # Rate-limit MAX_ITER warnings: at 200 Hz a singular pose can spam
        # thousands of identical lines and itself starve the control loop.
        self._warn_every = 25
        self._warn_seen = 0
        self._max_solve_s = max(1.0e-3, float(getattr(cfg, "max_solve_ms", 8.0)) * 1.0e-3)

    def _status(self):
        return self.qp.results.info.status

    def _solved(self) -> bool:
        return self._status() == self._px.proxqp.QPSolverOutput.PROXQP_SOLVED

    def solve(
        self,
        H: np.ndarray,
        g: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        C: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> np.ndarray:
        import time as _time

        if not self._initialized:
            self.qp.init(H, g, A, b, C, lo, hi)
            self._initialized = True
        else:
            # Warm-start fuse: reusing multipliers from a failed tick poisons the
            # next solve (MAX_ITER death spiral from tick 1 onward).  Cold-start
            # only while recovering; restore warm-start after a clean solve.
            if self.fail_count > 0:
                self.qp.settings.initial_guess = (
                    self._px.proxqp.InitialGuess.NO_INITIAL_GUESS
                )
            else:
                self.qp.settings.initial_guess = (
                    self._px.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
                )
            self.qp.settings.eps_abs = self._eps_tight
            self.qp.settings.max_iter = self._max_iter
            self.qp.update(H=H, g=g, A=A, b=b, C=C, l=lo, u=hi)

        t0 = _time.perf_counter()
        self.qp.solve()
        elapsed = _time.perf_counter() - t0

        if not self._solved():
            # First retry: cold-start + loose eps + fewer iters.  Skip the
            # retry if the first attempt already burned the wall budget —
            # near σ→0 a second full solve can hold the GIL for seconds
            # (rail Modbus starves → encoder freeze → PANIC; Ctrl+C feels dead).
            remaining = self._max_solve_s - elapsed
            if remaining > 1.0e-3:
                self.qp.settings.initial_guess = (
                    self._px.proxqp.InitialGuess.NO_INITIAL_GUESS
                )
                self.qp.settings.eps_abs = self._eps_loose
                retry_iters = int(
                    min(max(int(self._max_iter), 1), 200, max(int(remaining / 0.00005), 20))
                )
                self.qp.settings.max_iter = retry_iters
                self.qp.solve()
                self.qp.settings.max_iter = int(self._max_iter)

        if not self._solved():
            self.fail_count += 1
            self._warn_seen += 1
            if self._warn_on_fail and self._warn_seen % self._warn_every == 1:
                print(
                    f"[WBC WARN] ProxQP {self._status()} "
                    f"(fail_count={self.fail_count}, "
                    f"suppressing next {self._warn_every - 1})",
                    flush=True,
                )
            return None

        self.fail_count = 0
        self._warn_seen = 0
        return np.asarray(self.qp.results.x, dtype=float)


class _OsqpWbcBackend:
    """Fallback when ProxQP unavailable (no warm equality+ineq resize)."""

    def __init__(self, nv: int, max_cbf: int, cfg: QpConfig) -> None:
        import osqp
        import scipy.sparse as sp

        self._osqp = osqp
        self._sp = sp
        self.nv = nv
        self.n_slack = N_SLACK
        self.n_var = nv + self.n_slack
        self.n_in = nv + max_cbf
        self.cfg = cfg
        self.prob = None

    def solve(self, H, g, A, b, C, lo, hi):
        sp = self._sp
        A_full = np.vstack([C, A])
        l_full = np.concatenate([lo, b])
        u_full = np.concatenate([hi, b])
        P = sp.csc_matrix(np.triu(H))
        A_csc = sp.csc_matrix(A_full)
        if self.prob is None:
            self.prob = self._osqp.OSQP()
            self.prob.setup(
                P, g, A_csc, l_full, u_full,
                verbose=False, warm_start=True,
                eps_abs=self.cfg.eps_abs, eps_rel=self.cfg.eps_abs,
                max_iter=self.cfg.max_iter,
            )
        else:
            self.prob.update(Px=P.data, q=g, Ax=A_csc.data, l=l_full, u=u_full)
        res = self.prob.solve()
        if res.x is None or np.any(np.isnan(res.x)):
            return None
        return np.asarray(res.x, dtype=float)


class QpIkController:
    """Slack-variable WBC velocity-IK core: (q, v_cmd) -> qdot."""

    def __init__(
        self,
        kin: RobotKinematics,
        limits: SafetyLimits,
        cfg: QpConfig | None = None,
        collision: CollisionModel | None = None,
    ) -> None:
        self.kin = kin
        self.cfg = cfg or QpConfig()
        # Per-joint damper band: arm in rad, prismatic rail (joint 0) in m.
        damper_band = np.full(kin.nv, float(self.cfg.limit_damper_band_rad))
        damper_band[0] = float(self.cfg.limit_damper_band_rail_m)
        self.constraints = VelocityBoxConstraints(
            limits, damper_band_rad=damper_band
        )
        self.collision_cfg = self.cfg.collision
        self._max_cbf = max(1, int(self.collision_cfg.max_pairs))
        self.collision = collision
        if self.collision_cfg.enabled and self.collision is None:
            self.collision = CollisionModel(kin.model)
        self._cbf_slots = CbfSlotTracker(max_pairs=self._max_cbf)
        self.qdot_prev = np.zeros(kin.nv, dtype=float)
        self._m_diag_lpf: np.ndarray | None = None
        self._task_scale_lpf: float = 1.0
        self.backend = self._make_backend(kin.nv)

        w_reg = np.asarray(self.cfg.reg, dtype=float)
        if w_reg.ndim == 0 or w_reg.size == 1:
            w_reg = np.full(kin.nv, float(w_reg))
        self._w_reg = w_reg
        self._w_task = np.asarray(self.cfg.task_weight, dtype=float)

    def _make_backend(self, nv: int):
        want = self.cfg.backend.lower()
        if want == "proxqp":
            try:
                return _ProxQpWbcBackend(nv, self._max_cbf, self.cfg)
            except Exception:
                pass
        if want in ("osqp", "proxqp"):
            try:
                return _OsqpWbcBackend(nv, self._max_cbf, self.cfg)
            except Exception as exc:
                raise RuntimeError(
                    "No QP backend available (install proxsuite or osqp)"
                ) from exc
        raise ValueError(f"unknown QP backend {self.cfg.backend!r}")

    @property
    def backend_name(self) -> str:
        return type(self.backend).__name__.replace("_", "").replace("Backend", "").lower()

    def reset(self, q0_rad: np.ndarray | None = None) -> None:
        del q0_rad  # QP state is velocity history / LPF only
        self.qdot_prev = np.zeros(self.kin.nv, dtype=float)
        self._m_diag_lpf = None
        self._task_scale_lpf = 1.0

    def _task_scale_sigma(self, sigma_min: float, dt: float) -> float:
        """LPF-smoothed W_task scale in [min_frac, 1] from σ_min."""
        sigma_ref = float(self.cfg.sr_damping.sigma_ref)
        raw = 1.0
        if sigma_ref > 1e-9 and sigma_min < sigma_ref:
            frac = float(sigma_min) / sigma_ref
            raw = max(frac * frac, float(self.cfg.task_weight_min_frac))
        tau = float(self.cfg.task_weight_lpf_tau_s)
        if tau > 1e-9 and dt > 1e-9:
            alpha = min(1.0, dt / tau)
            self._task_scale_lpf += alpha * (raw - self._task_scale_lpf)
            return float(self._task_scale_lpf)
        self._task_scale_lpf = float(raw)
        return float(raw)

    def set_collision_enabled(self, enabled: bool) -> None:
        self.collision_cfg.enabled = bool(enabled)

    def step(
        self,
        q_prev: np.ndarray,
        twist_ref: np.ndarray,
        dt: float,
        secondary_qdot: np.ndarray | None = None,
        *,
        q_meas: np.ndarray | None = None,
        resync_err: float | np.ndarray = 0.0,
        rail_locked: bool = False,
        rail_lock_reg_scale: float = 1.0,
        rail_lock_vel_eps_m_s: float = 0.0,
        rail_vel_pin_m_s: float | None = None,
        zero_secondary_rail: bool = False,
        rail_task_vel_m_s: float | None = None,
        rail_task_weight: float = 0.0,
    ) -> IkStepResult:
        q_prev = np.asarray(q_prev, dtype=float)
        v_cmd = np.asarray(twist_ref, dtype=float)

        J = self.kin.jacobian(q_prev)
        sigma = self.kin.singular_values(J)
        sigma_min = float(sigma.min())

        nv = self.kin.nv
        ns = N_SLACK
        n_var = nv + ns

        # Chiaverini SR projection: λ(σ) grows as σ→0 so N→I and secondary
        # tasks / qdot_ff keep control of singular directions.
        proj_damping = sr_damping_lambda(sigma_min, self.cfg.sr_damping)
        M = self.kin.mass_matrix(q_prev) if self.cfg.use_dyn_nullspace or self.cfg.use_mass_weighted_reg else None
        qdot_nom = (
            project_onto_task_nullspace(
                J,
                secondary_qdot,
                damping=proj_damping,
                sigma_min=sigma_min,
                sr_cfg=self.cfg.sr_damping,
                M=M,
                use_dyn=self.cfg.use_dyn_nullspace and M is not None,
            )
            if secondary_qdot is not None
            else np.zeros(nv, dtype=float)
        )
        # Rail bleed guard: the SR-damped nullspace basis N couples all joints,
        # so even a rail-clean secondary_qdot (composer zeroes [0]) is smeared
        # by the projection into a nonzero qdot_nom[0].  In COUPLED / RAIL_ONLY
        # / TCP_FIXED we do not want secondary tasks (centering / manip /
        # arm_task / damping) to drive rail via this projection back-door —
        # rail motion is recruited only by the primary Cartesian equality
        # Jqdot = v_cmd and by the EXPLICIT preferred-extension rail task
        # (rail_task_vel_m_s / rail_task_weight below), never by projected
        # nullspace velocities.  Zero the rail bias here.
        if zero_secondary_rail and qdot_nom.shape[0] > 0:
            qdot_nom[0] = 0.0

        w_reg = self._w_reg.copy()
        w_task = self._w_task.copy()
        if rail_locked and rail_lock_reg_scale > 1.0:
            w_reg[0] *= float(rail_lock_reg_scale)
        w_task *= self._task_scale_sigma(sigma_min, dt)

        # rail_extension hint stays at its full weight (the task itself scales
        # by σ_scale via Bug 2 — do NOT double-schedule here).
        rail_w_eff = float(rail_task_weight)

        H = np.zeros((n_var, n_var), dtype=float)
        if self.cfg.use_mass_weighted_reg and M is not None:
            m_diag = np.maximum(np.diag(M), self.cfg.mass_reg_floor)
            if self.cfg.mass_weight_exempt_rail:
                # Rail cost is reg[0] verbatim: diag(M)[0] is the ~10 kg
                # carriage+arm mass, which over-priced rail motion 30-400x
                # vs the arm and starved rail recruitment (arm stretched to
                # near-straight before the rail moved).
                m_diag[0] = 1.0
            tau = float(self.cfg.mass_reg_lpf_tau_s)
            if tau > 1e-9 and dt > 1e-9:
                if self._m_diag_lpf is None:
                    self._m_diag_lpf = m_diag.copy()
                else:
                    alpha = min(1.0, dt / tau)
                    self._m_diag_lpf += alpha * (m_diag - self._m_diag_lpf)
                m_diag = self._m_diag_lpf
            H[:nv, :nv] = np.diag(w_reg * m_diag)
        else:
            H[:nv, :nv] = np.diag(w_reg)
        H[nv:, nv:] = np.diag(w_task)
        g = np.zeros(n_var, dtype=float)
        g[:nv] = -np.diag(H[:nv, :nv]) * qdot_nom if self.cfg.use_mass_weighted_reg and M is not None else -w_reg * qdot_nom

        # Preferred-extension rail task (Yamamoto & Yun 1994 base-arm
        # coordination): a soft scalar task w/2*(qdot[0] - v_rail)^2 added
        # directly to the cost.  The Cartesian equality rows (much heavier)
        # keep the TCP on the reference while the arm absorbs the rail motion,
        # so tracking is NOT sacrificed — unlike a nullspace-projected rail
        # drive, which the SR-damped projector smears near singularities
        # (Dietrich et al. 2015).  Weight is scheduled continuously by the
        # caller (0 in the extension dead zone: the rail does not wander).
        if (
            rail_task_vel_m_s is not None
            and rail_w_eff > 0.0
            and not rail_locked
            and rail_vel_pin_m_s is None
        ):
            H[0, 0] += rail_w_eff
            g[0] -= rail_w_eff * float(rail_task_vel_m_s)

        A = np.zeros((ns, n_var), dtype=float)
        A[:, :nv] = J
        A[:, nv:] = -np.eye(ns)
        b = v_cmd

        lo_box, hi_box = self.constraints.bounds(
            q_prev,
            dt,
            self.qdot_prev,
            q_meas=q_meas,
            resync_err=resync_err,
            rail_locked=rail_locked,
            rail_lock_vel_eps_m_s=rail_lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin_m_s,
        )
        if self.collision is not None and self.collision_cfg.enabled:
            cbf = build_cbf_rows(
                self.collision,
                self.kin,
                q_prev,
                self.collision_cfg,
                tracker=self._cbf_slots,
            )
        else:
            from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import CbfRows

            cbf = CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))
            self._cbf_slots = CbfSlotTracker(max_pairs=self._max_cbf)

        C, lo, hi = build_wbc_inequalities(
            nv, ns, lo_box, hi_box, cbf, self._max_cbf
        )

        x = self.backend.solve(
            np.ascontiguousarray(H),
            np.ascontiguousarray(g),
            np.ascontiguousarray(A),
            np.ascontiguousarray(b),
            np.ascontiguousarray(C),
            np.ascontiguousarray(lo),
            np.ascontiguousarray(hi),
        )
        if x is None:
            # Solver failure: exponential decay of previous velocity.  Near
            # σ→0 decay harder — keeping a large qdot_prev is what drove the
            # elbow straight in force-hybrid retract before the next tick
            # burned seconds in ProxQP.
            decay = float(self.cfg.fail_qdot_decay)
            sigma_ref = float(self.cfg.sr_damping.sigma_ref)
            if sigma_ref > 1e-9 and sigma_min < sigma_ref:
                decay = min(decay, 0.4)
            qdot = decay * self.qdot_prev
            slack = np.zeros(ns, dtype=float)
        else:
            qdot = x[:nv]
            slack = x[nv:]
        self.qdot_prev = qdot
        q_next = q_prev + qdot * dt
        return IkStepResult(
            q_next=q_next,
            qdot=qdot,
            sigma_min=sigma_min,
            manip=self.kin.manipulability(J),
            slack_norm=float(np.linalg.norm(slack)),
            n_cbf_active=int(cbf.jacobian.shape[0]),
        )
```

## `rm75_control/control/joint_admittance_8dof/solver/constraint_mgr.py`

```python
"""Per-tick inequality constraints for the WBC QP inner loop.

Joint velocity box (velocity / position look-ahead / acceleration) plus optional
CBF self-collision rows stacked into ProxQP's l <= C x <= u form.
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import CbfRows
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


def _collapse_to(
    lo: np.ndarray,
    hi: np.ndarray,
    keep_lo: np.ndarray,
    keep_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve an empty ``[lo, hi]`` by projecting onto ``[keep_lo, keep_hi]``.

    ``lo > hi`` means the stage just applied is infeasible against the
    higher-priority interval ``[keep_lo, keep_hi]``.  The requested value is
    whichever endpoint caused the crossing (``lo`` when the new stage pushed
    the floor up, ``hi`` when it pushed the ceiling down); clamping it back
    into the priority interval keeps the intent of the lower-priority stage
    while guaranteeing the returned box is always executable.

    Rows that are already feasible are returned untouched.
    """
    crossed = lo > hi
    if not np.any(crossed):
        return lo, hi
    # ``lo`` rose above ``hi``: the new stage wants at least ``lo``.
    # Otherwise ``hi`` fell below ``lo``: the new stage wants at most ``hi``.
    want = np.where(lo > keep_hi, lo, hi)
    # ``+ 0.0`` normalises -0.0 so a collapsed box compares as lo == hi.
    pinned = np.clip(want, keep_lo, keep_hi) + 0.0
    return np.where(crossed, pinned, lo), np.where(crossed, pinned, hi)


class VelocityBoxConstraints:
    def __init__(
        self,
        limits: SafetyLimits,
        *,
        damper_band_rad: float | np.ndarray = 0.15,
    ) -> None:
        self.lim = limits
        # Faverjon/Tournassoud velocity-damper influence zone before each
        # (margin-backed) joint limit; see bounds() below.  Scalar or per-joint
        # vector — units are per joint (rad for revolute, m for the prismatic
        # rail), so a scalar rad band must NOT be applied to the rail.
        self.damper_band_rad = np.asarray(damper_band_rad, dtype=float)

    def bounds(
        self,
        q: np.ndarray,
        dt: float,
        qdot_prev: np.ndarray | None = None,
        *,
        q_meas: np.ndarray | None = None,
        resync_err: float | np.ndarray = 0.0,
        rail_locked: bool = False,
        rail_lock_vel_eps_m_s: float = 0.0,
        rail_vel_pin_m_s: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        lim = self.lim
        q = np.asarray(q, dtype=float)

        # Staged/prioritised clamp: v_max + position margin are hard safety
        # bounds and are always honoured.  a_max and the resync anti-windup
        # bound are secondary - each is applied only if it doesn't render the
        # box infeasible against the *previous* (higher-priority) stage; a
        # single combined "crossed -> discard everything" check would let a
        # transient accel/resync conflict silently drop the resync bound
        # (or worse, both) for the rest of the move, which is exactly what
        # let the command lead run away unbounded instead of saturating.
        #
        # When a stage IS infeasible against the previous one, the conflict is
        # resolved by PROJECTING onto the higher-priority interval (see
        # ``_collapse_to``), never by averaging the two.  An unclamped midpoint
        # silently inverts the documented priority: at the rail's 0 m end stop
        # it produced lo == hi == +0.925 m/s against v_max = 0.16 m/s (a forced
        # 6x over-speed the servo answers with Er-01), and an inbound rail
        # arriving at the stop was pinned at a negative velocity - i.e. forced
        # to keep driving INTO the stop - because a_max could not decelerate
        # inside the damper band.
        lo = -lim.v_max.copy()
        hi = lim.v_max.copy()

        m = lim.position_margin

        # Faverjon & Tournassoud (1987) velocity damper toward each joint
        # limit: the allowed speed TOWARD a limit ramps linearly to zero over
        # the last ``damper_band_rad`` before the (margin-backed) limit, while
        # motion AWAY stays unconstrained.  This replaces the old binary
        # "|u| > 0.95 -> zero bound" rule, which flipped the box between
        # +-v_max and 0 in a single tick and chattered against the soft
        # centering / arm-angle tasks whenever the nullspace parked a joint on
        # the threshold.  The ramp is continuous in q and always keeps 0
        # inside the box.  The damper never restricts motion AWAY from a
        # limit, so it can never block a margin recovery.
        band = np.broadcast_to(self.damper_band_rad, q.shape)
        if np.any(band > 1e-9):
            b = np.maximum(band, 1e-9)
            d_hi = np.clip(((lim.q_upper - m) - q) / b, 0.0, 1.0)
            d_lo = np.clip((q - (lim.q_lower + m)) / b, 0.0, 1.0)
            # Joints with band <= 0 keep the full velocity box.
            d_hi = np.where(band > 1e-9, d_hi, 1.0)
            d_lo = np.where(band > 1e-9, d_lo, 1.0)
            hi = np.minimum(hi, lim.v_max * d_hi)
            lo = np.maximum(lo, -lim.v_max * d_lo)

        # v_max + damper is the *executable envelope*: every later stage is
        # projected back into it, so the returned box can never ask for a
        # velocity the joint cannot run or one that points into a hard stop.
        v_lo, v_hi = lo, hi

        if lim.a_max is not None and qdot_prev is not None:
            qdot_prev = np.asarray(qdot_prev, dtype=float)
            a = lim.a_max * dt
            # a_max is secondary: honour it whenever it intersects the
            # envelope, drop it when it does not.  A rail decelerating into an
            # end stop cannot brake inside the damper band at a_max_rail
            # (0.3 m/s^2 needs ~17 mm from 0.1 m/s, band is 20 mm), so the
            # intersection goes empty exactly there — and "keep the envelope"
            # is the only safe answer.
            a_lo = np.maximum(v_lo, qdot_prev - a)
            a_hi = np.minimum(v_hi, qdot_prev + a)
            lo, hi = _collapse_to(a_lo, a_hi, v_lo, v_hi)

        # Position look-ahead, applied last so a margin recovery ramps up
        # under a_max instead of stepping to v_max.  Inside the margin the
        # push-back ``p_lo`` is margin/dt, routinely tens of times v_max: it
        # is a direction, not an achievable speed, so it is clamped into the
        # box the earlier stages left.
        p_lo = (lim.q_lower + m - q) / dt
        p_hi = (lim.q_upper - m - q) / dt
        lo, hi = _collapse_to(
            np.maximum(lo, p_lo), np.minimum(hi, p_hi), lo, hi
        )

        # Vectorised command-lead damper: resync_err is either scalar (legacy;
        # arm-only, radians) or an nv-vector with per-joint bounds — arm rad
        # for joints 1..7 and metres for joint 0 (rail).  Using a scalar rad
        # bound for the prismatic joint was a silent unit bug: 0.10 rad =
        # 100 mm of lead allowed on the rail, and the QP would happily plan
        # multiple centimetres ahead of the encoder before anti-windup engaged.
        if q_meas is not None:
            re = np.broadcast_to(
                np.asarray(resync_err, dtype=float), q.shape
            ).astype(float)
            active = re > 0.0
            if np.any(active):
                q_meas = np.asarray(q_meas, dtype=float)
                lead = q - q_meas
                band = np.maximum(re * 0.5, 1e-6)
                d_hi = np.clip((re - lead) / band, 0.0, 1.0)
                d_lo = np.clip((re + lead) / band, 0.0, 1.0)
                hi_new = np.where(hi > 0.0, hi * d_hi, hi)
                lo_new = np.where(lo < 0.0, lo * d_lo, lo)
                hi = np.where(active, hi_new, hi)
                lo = np.where(active, lo_new, lo)
                # Scaling a one-sided box toward 0 can push a bound past the
                # other one; keep the interval non-empty.
                lo, hi = _collapse_to(lo, hi, v_lo, v_hi)

        if rail_vel_pin_m_s is not None:
            v = float(rail_vel_pin_m_s)
            lo[0] = v
            hi[0] = v
        elif rail_locked:
            eps = max(float(rail_lock_vel_eps_m_s), 0.0)
            lo[0] = -eps
            hi[0] = eps

        return lo, hi


def build_wbc_inequalities(
    nv: int,
    n_slack: int,
    lo_box: np.ndarray,
    hi_box: np.ndarray,
    cbf: CbfRows,
    max_cbf_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack [I_nv, 0; J_cbf, 0] with box + CBF lower bounds.

    Returns C (n_in, nv+n_slack), l, u for l <= C x <= u.
    Inactive CBF slots are l=-inf, u=+inf.
    """
    n_in = nv + max_cbf_rows
    n_var = nv + n_slack
    C = np.zeros((n_in, n_var), dtype=float)
    C[:nv, :nv] = np.eye(nv)
    l = np.full(n_in, -np.inf, dtype=float)
    u = np.full(n_in, np.inf, dtype=float)
    l[:nv] = lo_box
    u[:nv] = hi_box

    n_active = cbf.jacobian.shape[0]
    if cbf.slot_index is not None and cbf.slot_index.size == n_active:
        for k in range(n_active):
            i = int(cbf.slot_index[k])
            if i < 0 or i >= max_cbf_rows:
                continue
            C[nv + i, :nv] = cbf.jacobian[k]
            l[nv + i] = cbf.lower[k]
    else:
        for i in range(min(n_active, max_cbf_rows)):
            C[nv + i, :nv] = cbf.jacobian[i]
            l[nv + i] = cbf.lower[i]
    return C, l, u
```

## `rm75_control/control/joint_admittance_8dof/solver/sigma_grad.py`

```python
"""TCP-preserving directional derivative of σ_min w.r.t. rail translation.

World-frame J is independent of q_rail here, so ∂σ/∂q_rail is zero.  Instead
move rail by δy with arm δq_arm = -J_arm⁺ e_rail δy (hold TCP); σ under that
coordinated move is what σ-escape needs.  Central difference, 2 Jacobians.
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RAIL_INDEX, RobotKinematics


def _sigma_min(J: np.ndarray) -> float:
    return float(np.linalg.svd(J, compute_uv=False).min())


def sigma_min_grad_rail(
    kin: RobotKinematics,
    q_rad: np.ndarray,
    eps: float = 1.0e-3,
) -> float:
    """Directional derivative ``d σ_min / d y_rail`` under TCP-preservation.

    Positive value → moving the rail in +Y increases the arm's conditioning
    (helps escape a singularity); negative → −Y direction helps instead.
    Returns 0.0 when ``J_arm`` is itself rank-deficient (rare — happens only
    at deep singularities where the whole task is already infeasible).
    """
    q = np.asarray(q_rad, dtype=float)
    J = kin.jacobian(q)
    # J_arm: columns 1..7 (the 7-DOF arm), J_rail: column 0.
    J_arm = np.delete(J, RAIL_INDEX, axis=1)
    e_rail = J[:, RAIL_INDEX]
    # Damped least-squares pseudoinverse (small damping keeps this smooth
    # near singularities — the analytical J_arm^+ blows up right where we
    # want the escape term most).
    lam = 5.0e-3
    try:
        dq_arm = -np.linalg.solve(
            J_arm.T @ J_arm + lam * lam * np.eye(J_arm.shape[1]),
            J_arm.T @ e_rail,
        )
    except np.linalg.LinAlgError:
        return 0.0
    # Central difference under the coordinated move.
    q_p = q.copy()
    q_m = q.copy()
    q_p[RAIL_INDEX] += eps
    q_m[RAIL_INDEX] -= eps
    # scatter dq_arm into the non-rail slots
    arm_slots = [i for i in range(q.shape[0]) if i != RAIL_INDEX]
    for k, slot in enumerate(arm_slots):
        q_p[slot] += eps * dq_arm[k]
        q_m[slot] -= eps * dq_arm[k]
    sig_p = _sigma_min(kin.jacobian(q_p))
    sig_m = _sigma_min(kin.jacobian(q_m))
    return float((sig_p - sig_m) / (2.0 * eps))
```

## `apps/joint_admittance_8dof/d_sin_tool_y.py`

```python
#!/usr/bin/env python3
"""8-DOF task orchestration (window C): IK/planning, submit program to window A.

  source env.sh
  python apps/joint_admittance_8dof/d_sin_tool_y.py --dry-run
  # same taught q_deg → pose_d = Pin FK; default MoveJ (WbcArm) then force scan
  python apps/joint_admittance_8dof/d_sin_tool_y.py --enable-force --desired-z 3.0 --scan-duration 600
  # explicit MoveL/SRS instead of MoveJ:
  python apps/joint_admittance_8dof/d_sin_tool_y.py --move-mode cartesian --enable-force --desired-z 1.0
  # move->D by taught joint angles (ignore RealMan TCP; for gripper-Z rotation tests):
  python apps/joint_admittance_8dof/d_sin_tool_y.py \\
      --d-target joints --move-mode joint --enable-force --desired-z 1.0 \\
      --hybrid-hold-at-d --scan-duration 60
  # move to D, hold 5s, tcp_fixed rail +Y 15cm (no scan):
  python apps/joint_admittance_8dof/d_sin_tool_y.py \\
      --scan-duration 0 --hold-at-d-s 5 --rail-move-cm 15 --rail-move-mode tcp_fixed
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.admittance_common.phase_ipc import PhaseCommandClient, PhaseStatus
from rm75_control.control.admittance_common.state_bus import RobotStateBus
from rm75_control.control.admittance_common.state_relay import (
    RelayStateBus,
    parse_state_relay_config,
    relay_shm_has_publisher,
)
from rm75_control.control.joint_admittance_8dof.api import compute_move_plan
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.sin_tool_y_program import (
    execute_sin_tool_y_program,
    make_task_params_from_args,
    plan_psi_toggle_sides,
    plan_q_toggle_at_pose,
    resolve_scan_target_at_d,
)
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    deg2rad,
    full_q_from_arm,
    rad2deg,
)
from rm75_control.core.session import RobotSession
from rm75_control.force.compensation.tool_pose import maybe_sync_kin_tcp_from_config


@dataclass
class _AttachSession:
    """Minimal session stand-in when window A owns the Realman TCP."""

    config: dict
    ip: str
    robot: object = None

    def move_joints(self, *args, **kwargs) -> None:
        raise RuntimeError("move_j is unavailable in attach mode (window A owns TCP)")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/joint_admittance_8dof.yaml"))
    ap.add_argument("--slot", type=str, default="d")
    ap.add_argument(
        "--d-target",
        choices=("joints",),
        default="joints",
        help="Move→D target source (joints only): taught q + j7+90° so ArmTip +X → "
        "TCP +Z, then Pin FK / pose IK. Execution follows --move-mode "
        "(joint MoveJ or cartesian/SRS).",
    )
    ap.add_argument(
        "--approach-dz-mm",
        type=float,
        default=0.220 * 1000.0,
        help="Standoff used only to size auto move duration (not for pose_d).",
    )
    ap.add_argument("--move-duration", type=float, default=None)
    ap.add_argument(
        "--move-duration-margin",
        type=float,
        default=0.80,
        help="Peak joint speed fraction of (URDF·v_scale) used to size auto "
             "move duration (was 0.50; higher = faster move→D).",
    )
    ap.add_argument("--move-duration-min", type=float, default=2.5)
    ap.add_argument(
        "--move-duration-max",
        type=float,
        default=20.0,
        help="Cap on auto move duration (s). Was 5s and crushed 13s joint moves into a jerk.",
    )
    ap.add_argument("--move-kp", type=float, default=2.0)
    ap.add_argument("--move-mode", choices=("cartesian", "joint"), default="joint",
                    help="PTP to D: joint=MoveJ (default, industrial PTP); "
                         "cartesian=MoveL/SRS. Scan/track always Cartesian. "
                         "No auto detect-and-switch.")
    ap.add_argument("--y-pp-cm", type=float, default=16.0,
                    help="Tool-Y scan peak-to-peak (cm). 90 = 900 mm stroke.")
    ap.add_argument("--max-vel-cm-s", type=float, default=2.0)
    ap.add_argument("--period-s", type=float, default=None)
    ap.add_argument("--desired-z", type=float, default=None)
    ap.add_argument("--scan-duration", type=float, default=30.0)
    ap.add_argument(
        "--rail-scan-center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plan pose D / scan origin at rail mid-stroke (travel/2), not at rail_y=0. "
        "Start rail may still be 0 after manual home; move->D carries rail to center. "
        "Y stroke is then ±(y_pp/2) about the rail-center pose (default: on).",
    )
    ap.add_argument(
        "--psi-toggle-period",
        type=float,
        default=0.0,
        help="During hybrid scan, alternate swivel psi every N seconds (0=off)",
    )
    ap.add_argument(
        "--psi-side-offset-deg",
        type=float,
        default=90.5,
        help="Fallback ± offset from center when live left unavailable (default: 90.5)",
    )
    ap.add_argument(
        "--psi-left-deg",
        type=float,
        default=None,
        help="Explicit left swivel target in degrees (overrides live Realman read)",
    )
    ap.add_argument(
        "--psi-right-deg",
        type=float,
        default=None,
        help="Explicit right swivel target in degrees (requires --psi-left-deg)",
    )
    ap.add_argument(
        "--no-psi-live-left",
        action="store_true",
        help="Do not use current Realman joints as left target; use ±offset only",
    )
    ap.add_argument(
        "--psi-toggle-alpha",
        type=float,
        default=0.02,
        help="LPF polish on posture ramp per tick (default 0.02)",
    )
    ap.add_argument(
        "--psi-ramp-s",
        type=float,
        default=4.0,
        help="Quintic ramp duration for each psi target change (default 4s)",
    )
    ap.add_argument(
        "--hybrid-hold-at-d",
        action="store_true",
        help=(
            "At D: force-position hold (no Y sin scan); rail stays COUPLED "
            "so σ-escape can slide the carriage"
        ),
    )
    ap.add_argument(
        "--hold-s",
        type=float,
        default=0.0,
        help="After move (and scan if any), keep running N seconds for Genesis/FK check",
    )
    ap.add_argument(
        "--hold-at-d-s",
        type=float,
        default=0.0,
        help="After move->D, hold TCP at D for N seconds (rail locked)",
    )
    ap.add_argument(
        "--rail-move-cm",
        type=float,
        default=0.0,
        help="After hold, unlock rail and move this distance (cm)",
    )
    ap.add_argument(
        "--rail-move-mode",
        choices=("rail_only", "tcp_fixed"),
        default="rail_only",
        help="rail_only: arm still, TCP rides rail; tcp_fixed: hold TCP, arm compensates",
    )
    ap.add_argument(
        "--rail-move-dir",
        choices=("+y", "-y"),
        default="+y",
        help="Rail travel direction for --rail-move-cm",
    )
    ap.add_argument("--enable-force", action="store_true", default=None)
    ap.add_argument("--log-interval", type=float, default=2.0)
    ap.add_argument("--verbose", "-v", action="store_true", help="Detailed IK / WBC logs + auto CSV")
    ap.add_argument(
        "--log-csv",
        type=str,
        default=None,
        help="WBC tick CSV path (A writes it). Default with -v: logs/sin_tool_y/run_<ts>.csv",
    )
    ap.add_argument(
        "--rail-log-csv",
        type=str,
        default=None,
        help="LW100 soft-loop CSV path (A writes it). Default with -v: logs/rail_servo/rail_<ts>.csv",
    )
    ap.add_argument("--cartesian-max-lin-vel", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--no-attach-state",
        action="store_true",
        help="Own robot TCP/UDP locally (debug only; do not run with window A)",
    )
    args = ap.parse_args()

    if args.verbose and not args.log_csv:
        log_dir = Path(__file__).resolve().parents[1] / "logs" / "sin_tool_y"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.log_csv = str(log_dir / f"run_{ts}.csv")
    # Pair rail servo CSV with WBC CSV (same timestamp when auto).
    rail_log_csv = getattr(args, "rail_log_csv", None)
    if args.verbose and not rail_log_csv:
        rail_dir = Path(__file__).resolve().parents[1] / "logs" / "rail_servo"
        rail_dir.mkdir(parents=True, exist_ok=True)
        if args.log_csv:
            stem = Path(args.log_csv).stem.replace("run_", "rail_", 1)
            if stem == Path(args.log_csv).stem:
                stem = f"rail_{time.strftime('%Y%m%d_%H%M%S')}"
            rail_log_csv = str(rail_dir / f"{stem}.csv")
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            rail_log_csv = str(rail_dir / f"rail_{ts}.csv")
    args.rail_log_csv = rail_log_csv
    if args.verbose and float(args.log_interval) >= 1.999:
        args.log_interval = 0.5
    if args.verbose and args.log_csv:
        print(f"debug log CSV (written by window A): {args.log_csv}", flush=True)
    if args.verbose and args.rail_log_csv:
        print(f"rail servo CSV (written by window A): {args.rail_log_csv}", flush=True)

    raw = load_yaml(args.config)
    startup = raw.get("startup", {})
    relay_cfg = parse_state_relay_config(raw)

    kin = RobotKinematics()
    inner_cfg = build_joint_ik_config(raw)
    inner = JointIkController(kin, inner_cfg)
    travel_m = float(inner_cfg.rail.travel_m)
    rail_center_m = 0.5 * travel_m
    rail_plan_m = (
        rail_center_m
        if bool(args.rail_scan_center)
        else float(inner_cfg.rail.q_ref_m if inner_cfg.rail.q_ref_m is not None else 0.0)
    )
    rail_m = rail_plan_m

    desired_z = args.desired_z if args.desired_z is not None else float(raw.get("force", {}).get("desired_z_n", 0.0))
    enable_force = args.enable_force if args.enable_force is not None else bool(startup.get("enable_force", False))

    if args.dry_run:
        print("dry-run: controllers built OK, not connecting.", flush=True)
        return 0

    robot_cfg = raw.get("robot", {})
    max_lin = float(args.cartesian_max_lin_vel) if args.cartesian_max_lin_vel is not None else 0.4
    sigma_ref = float(inner_cfg.qp.sr_damping.sigma_ref)

    local_bus: RobotStateBus | None = None
    state_bus: RobotStateBus | RelayStateBus | None = None
    phase_client: PhaseCommandClient | None = None
    attach_mode = not args.no_attach_state
    shm_name = str(relay_cfg.name or "rm75_state")

    if attach_mode:
        print("rm75 task: connecting to window A …", flush=True)
        attach_bus = RelayStateBus(shm_name)
        try:
            attach_bus.wait_first_pose(timeout_s=30.0)
        except TimeoutError as exc:
            raise RuntimeError(
                f"no live relay on shm {shm_name!r} — start window A first "
                f"(run_joint_admittance.py)"
            ) from exc
        state_bus = attach_bus
        phase_client = PhaseCommandClient()
        try:
            phase_client.wait_for_hub(timeout_s=30.0)
        except TimeoutError as exc:
            raise RuntimeError(
                "window A phase IPC not ready — restart run_joint_admittance.py"
            ) from exc
        print("rm75 task: connected", flush=True)
        session_cm = nullcontext(_AttachSession(config=raw, ip=str(robot_cfg.get("ip", ""))))
    else:
        if relay_shm_has_publisher(shm_name):
            raise RuntimeError(
                f"window A is already publishing shm {shm_name!r}. "
                "Drop --no-attach-state or stop window A."
            )
        session_cm = RobotSession(
            ip=robot_cfg.get("ip"),
            port=robot_cfg.get("port"),
            config=args.config,
            quiet=True,
        )

    with session_cm as sess:
        maybe_sync_kin_tcp_from_config(
            kin,
            raw,
            robot=getattr(sess, "robot", None),
            attach_mode=attach_mode,
        )
        if not attach_mode:
            local_bus = RobotStateBus(sess.robot, raw, robot_ip=sess.ip)
            local_bus.start()
            state_bus = local_bus
            print("rm75 task: CANFD + local UDP (standalone)", flush=True)

        scan_target = resolve_scan_target_at_d(
            args.slot,
            kin,
            euler_order=inner_cfg.euler_order,
            rail_m=rail_m,
            qp_cfg=inner_cfg.qp,
            nullspace_cfg=inner_cfg.nullspace,
        )
        pose_d = scan_target.pose_d
        q_target_rad = np.asarray(scan_target.q_target_rad, dtype=float)

        if attach_mode:
            snap0 = state_bus.read()
            if snap0.q_deg is None:
                raise RuntimeError("no joint feedback on attach bus")
            rail_start_m = float(getattr(state_bus, "last_rail_m", 0.0))
            q0_rad = full_q_from_arm(deg2rad(snap0.q_deg), rail_start_m)
        else:
            ret0, st0 = sess.robot.rm_get_current_arm_state()
            if ret0 != 0:
                raise RuntimeError(f"rm_get_current_arm_state failed: {ret0}")
            rail_start_m = 0.0
            q0_rad = full_q_from_arm(
                deg2rad(np.asarray(st0["joint"][:7], dtype=float)),
                rail_start_m,
            )
        psi_tgt = None
        if inner.arm_task is not None:
            psi_tgt = inner.arm_task.arm_angle(q_target_rad)

        # PTP mode is explicit (--move-mode); scan/track stays Cartesian/hybrid.
        move_mode = str(args.move_mode)
        plan = compute_move_plan(
            kin,
            q0_rad,
            q_target_rad,
            pose_d,
            v_scale=inner_cfg.v_scale,
            duration_s=args.move_duration,
            move_mode=move_mode,
            peak_joint_v_frac=float(args.move_duration_margin),
            max_lin_vel_m_s=max_lin,
            duration_min_s=float(args.move_duration_min),
            duration_max_s=float(args.move_duration_max),
            approach_dz_m=float(args.approach_dz_mm) * 0.001,
            sigma_ref=sigma_ref,
            euler_order=inner_cfg.euler_order,
        )

        psi_left = None
        psi_right = None
        q_toggle_left = None
        q_toggle_right = None
        if args.scan_duration > 0.0 and args.psi_toggle_period > 0.0:
            if psi_tgt is None and inner.arm_task is None:
                raise RuntimeError("--psi-toggle-period requires arm_angle task (psi at D)")
            q_toggle_center, q_toggle_left, q_toggle_right = plan_q_toggle_at_pose(
                kin,
                pose_d,
                q_target_rad,
                q0_rad,
                qp_cfg=inner_cfg.qp,
                nullspace_cfg=inner_cfg.nullspace,
            )
            if inner.arm_task is not None and psi_tgt is not None:
                _psi_center, psi_left, psi_right = plan_psi_toggle_sides(
                    inner,
                    q0_rad,
                    psi_tgt,
                    side_offset_rad=np.deg2rad(float(args.psi_side_offset_deg)),
                    psi_left_rad=(
                        np.deg2rad(float(args.psi_left_deg))
                        if args.psi_left_deg is not None
                        else None
                    ),
                    psi_right_rad=(
                        np.deg2rad(float(args.psi_right_deg))
                        if args.psi_right_deg is not None
                        else None
                    ),
                    psi_live_left=not args.no_psi_live_left,
                    kin=kin,
                    pose_d=pose_d,
                    q_center_rad=q_target_rad,
                    qp_cfg=inner_cfg.qp,
                    nullspace_cfg=inner_cfg.nullspace,
                )
            max_l = float(
                np.max(np.abs(rad2deg(q_toggle_left[1:] - q_toggle_center[1:])))
            )
            if max_l < 15.0 and args.psi_left_deg is None:
                print(
                    "  WARN: left Δq < 15deg — park arm in LEFT teach pose, "
                    "then submit (q0 read at task start, before move->D)",
                    flush=True,
                )

        task_params = make_task_params_from_args(
            args,
            config_path=str(args.config.resolve()),
            q0_rad=q0_rad,
            q_target_rad=q_target_rad,
            pose_d=pose_d,
            plan=plan,
            psi_tgt=psi_tgt,
            desired_z=desired_z,
            enable_force=enable_force,
            psi_left_rad=psi_left,
            psi_right_rad=psi_right,
            q_toggle_left_rad=q_toggle_left,
            q_toggle_right_rad=q_toggle_right,
            tcp_offset_pose=kin.tcp_offset_pose,
        )

        last_status_msg = [""]

        def _poll_attach_status(cmd_seq: int) -> PhaseStatus:
            assert phase_client is not None
            skip_msgs = {
                "accepted",
                "running",
                "done",
                "stopped",
                "waiting for task",
                "shutdown",
                "interrupted",
            }
            last_status_msg[0] = ""
            stop_n = [0]

            def _on_sig(_signum, _frame) -> None:
                stop_n[0] += 1
                try:
                    phase_client.stop()
                except Exception:
                    pass
                if stop_n[0] == 1:
                    print(
                        "\nrm75 task: Ctrl+C — stop requested on window A "
                        "(second Ctrl+C forces exit)",
                        flush=True,
                    )
                    return
                print("\nrm75 task: force exit", flush=True)
                os._exit(130)

            prev_int = signal.signal(signal.SIGINT, _on_sig)
            prev_term = signal.signal(signal.SIGTERM, _on_sig)
            try:
                while True:
                    st = phase_client.read_status()
                    if st is not None and st["status_seq"] == cmd_seq:
                        msg = str(st["msg"])
                        status = st["status"]
                        if (
                            args.log_interval > 0
                            and status == PhaseStatus.RUNNING
                            and msg
                            and msg not in skip_msgs
                            and msg != last_status_msg[0]
                        ):
                            last_status_msg[0] = msg
                            print(f"rm75 task: {msg}", flush=True)
                        if status in (
                            PhaseStatus.DONE,
                            PhaseStatus.ERROR,
                            PhaseStatus.STOPPED,
                        ):
                            return status
                    time.sleep(0.05)
            finally:
                signal.signal(signal.SIGINT, prev_int)
                signal.signal(signal.SIGTERM, prev_term)

        try:
            if attach_mode:
                assert phase_client is not None
                cmd_seq = phase_client.start(task_params)
                print(f"rm75 task: submitted task #{cmd_seq}", flush=True)
                final = _poll_attach_status(cmd_seq)
                if final == PhaseStatus.ERROR:
                    st = phase_client.read_status()
                    raise RuntimeError(f"window A task failed: {st['msg'] if st else 'unknown'}")
                if final == PhaseStatus.STOPPED:
                    print("rm75 task: stopped", flush=True)
                else:
                    print("rm75 task: done", flush=True)
            else:
                execute_sin_tool_y_program(
                    sess,
                    state_bus,
                    task_params,
                    raw=raw,
                    verbose=bool(args.verbose),
                )
            if args.hold_s > 0:
                print(
                    f"holding {args.hold_s:.0f}s @ D — Ctrl+C to exit early",
                    flush=True,
                )
                t_hold = time.monotonic() + float(args.hold_s)
                try:
                    while time.monotonic() < t_hold:
                        time.sleep(0.2)
                except KeyboardInterrupt:
                    print("\nStopped.", flush=True)
        except KeyboardInterrupt:
            if attach_mode and phase_client is not None:
                phase_client.stop()
            print("\nStopped.", flush=True)
        finally:
            if phase_client is not None:
                phase_client.close()
            if attach_mode and state_bus is not None:
                state_bus.stop()
            elif local_bus is not None:
                local_bus.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## `apps/joint_admittance_8dof/run_joint_admittance.py`

```python
#!/usr/bin/env python3
"""8-DOF controller daemon (window A): UDP + SHM + local WBC when C submits a task.

Window A in the 3-terminal layout: keeps the sole Realman TCP/UDP session,
publishes ``rm75_state`` for the Genesis twin, and **runs the 200 Hz WBC loop
locally** when window C submits a phase program (no per-tick CANFD SHM relay).

  source env.sh
  python apps/joint_admittance_8dof/run_joint_admittance.py \\
      --config configs/joint_admittance_8dof.yaml

Twin (separate terminal):

  python apps/joint_admittance_8dof/run_with_twin.py

Task orchestration (window C):

  python apps/joint_admittance_8dof/d_sin_tool_y.py --config ... --enable-force ...
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.admittance_common.phase_ipc import PhaseCmd, PhaseCommandHub, PhaseStatus
from rm75_control.control.admittance_common.state_bus import RobotStateBus
from rm75_control.control.admittance_common.state_relay import (
    StateRelayPublisher,
    parse_state_relay_config,
)
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import (
    CartesianTrackConfig,
    CartesianTrackOuterLoop,
    JointIkController,
    run_joint_admittance_loop,
)
from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    parse_rail_servo_config,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.reference import HoldReference
from rm75_control.control.joint_admittance_8dof.sin_tool_y_program import (
    build_sin_tool_y_program,
    execute_sin_tool_y_program,
)
from rm75_control.core.session import RobotSession
from rm75_control.force.compensation.tool_pose import maybe_sync_kin_tcp_from_config


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_controller_service(
    sess,
    bus: RobotStateBus,
    raw: dict,
    *,
    config_path: Path | None = None,
    hub: PhaseCommandHub,
    rail_m_fn,
    rail_bridge: RailServoBridge | None = None,
    relay: StateRelayPublisher | None = None,
    poll_s: float = 0.05,
    verbose: bool = False,
) -> None:
    """Hot-wait for window C; run WBC locally on START (direct UDP + CANFD)."""
    stop = False
    sig_n = 0

    def _on_sig(_signum, _frame) -> None:
        nonlocal stop, sig_n
        sig_n += 1
        # First action: kill rail (non-blocking) so FA24 cannot stay latched.
        if rail_bridge is not None and rail_bridge.enabled:
            try:
                rail_bridge.estop()
            except Exception:
                pass
        try:
            hub.request_stop()
        except Exception:
            pass
        stop = True
        if sig_n == 1:
            print(
                "\nrm75 controller: Ctrl+C — stopping task "
                "(second Ctrl+C forces exit)",
                flush=True,
            )
            return
        # Second+ signal: ProxQP / CANFD may hold the GIL for seconds near
        # singularity — do not wait for a clean Python teardown.
        print("\nrm75 controller: force exit", flush=True)
        os._exit(130)

    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    hub.set_idle()
    print("rm75 controller: hot-wait", flush=True)

    while not stop:
        polled = hub.poll()
        if polled is None:
            time.sleep(poll_s)
            continue

        cmd, cmd_seq, params = polled
        if cmd == PhaseCmd.STOP:
            hub.ack(cmd_seq)
            hub.set_stopped(cmd_seq)
            continue

        if cmd != PhaseCmd.START or params is None:
            hub.ack(cmd_seq)
            continue

        task_n = hub.task_n

        # Refuse move→D / FA24 until rail Modbus path is hot (or re-armed after panic).
        if rail_bridge is not None and rail_bridge.enabled:
            need_rearm = rail_bridge.panicked or not rail_bridge.armed
            if not rail_bridge.ensure_armed(
                timeout_s=float(getattr(rail_bridge.config, "arm_timeout_s", 8.0)),
                rearm=need_rearm,
            ):
                hub.set_error(cmd_seq, "rail NOT READY (arming failed)")
                hub.ack(cmd_seq)
                print(
                    f"rm75 controller: task #{task_n} refused — rail NOT READY",
                    flush=True,
                )
                if not stop:
                    print("rm75 controller: hot-wait", flush=True)
                continue

        hub.set_running(cmd_seq, msg="accepted")
        print(f"rm75 controller: running task #{task_n}", flush=True)

        phase_labels: list[str] = []
        tick_counter = [0]
        phase_idx = [0]
        last_progress_label = [""]

        def _on_step(label, t_phase, step, pose, f_ext, t_wall=float("nan")) -> None:
            tick_counter[0] += 1
            if label in phase_labels:
                idx = phase_labels.index(label)
            else:
                phase_labels.append(label)
                idx = len(phase_labels) - 1
            phase_idx[0] = idx
            label_s = str(label)
            if label_s != last_progress_label[0]:
                last_progress_label[0] = label_s
                hub.set_progress(
                    cmd_seq,
                    phase_idx=idx,
                    phase_label=label_s,
                    ticks=tick_counter[0],
                )

        try:
            # Window A is long-lived while force-controller tuning happens in
            # YAML. Re-read it for every submitted task so Window C cannot
            # silently run a controller snapshot left over from daemon start.
            task_raw = load_yaml(config_path) if config_path is not None else raw
            built = build_sin_tool_y_program(params, raw=task_raw)
            rail_m_fn.set_active(built.inner)
            if relay is not None:
                # Prefer task kin (synced gripper TCP) for SHM pose publish.
                relay.set_kin(built.inner.kin)
            if rail_bridge is not None and rail_bridge.enabled:
                rail_csv = getattr(params, "rail_log_csv", None)
                if rail_csv:
                    rail_bridge.enable_log_csv(str(rail_csv))
            result = execute_sin_tool_y_program(
                sess,
                bus,
                params,
                raw=task_raw,
                built=built,
                on_step=_on_step,
                stop_check=hub.should_stop,
                verbose=verbose,
                rail_bridge=rail_bridge,
            )
            if hub.should_stop():
                hub.set_stopped(cmd_seq)
                print(f"rm75 controller: task #{task_n} stopped", flush=True)
            elif result.stop_reason:
                hub.set_error(cmd_seq, result.stop_reason)
                print(
                    f"rm75 controller: task #{task_n} safety stop — "
                    f"{result.stop_reason}",
                    flush=True,
                )
            elif result.stalled:
                hub.set_error(cmd_seq, "control watchdog fired")
                print(
                    f"rm75 controller: task #{task_n} safety stop — "
                    "control watchdog fired",
                    flush=True,
                )
            else:
                hub.set_done(cmd_seq)
                print(
                    f"rm75 controller: task #{task_n} done "
                    f"({result.duration_s:.1f}s, {result.ticks} ticks)",
                    flush=True,
                )
        except KeyboardInterrupt:
            stop = True
            hub.set_stopped(cmd_seq, msg="interrupted")
            print(f"rm75 controller: task #{task_n} interrupted", flush=True)
        except Exception as exc:
            hub.set_error(cmd_seq, str(exc))
            print(f"rm75 controller: task error: {exc}", flush=True)
        finally:
            hub.ack(cmd_seq)
            rail_m_fn.reset_idle()
            if rail_bridge is not None and rail_bridge.enabled:
                # Prefer non-blocking path if abort already set (Ctrl+C).
                try:
                    if stop or rail_bridge._abort.is_set():
                        rail_bridge.estop()
                    else:
                        rail_bridge.hold_current()
                except Exception:
                    try:
                        rail_bridge.estop()
                    except Exception:
                        pass
            if not stop:
                print("rm75 controller: hot-wait", flush=True)


class _RailPublisher:
    """Mutable rail source for SHM twin during idle vs active WBC.

    When the LW100 bridge is enabled, publish **encoder** position (poll_hz)
    so the twin mirrors the real carriage. WBC itself uses open-loop ``q_cmd[0]``
    and does not close the loop on this value.
    """

    def __init__(self, default_m: float, bridge: RailServoBridge | None = None) -> None:
        self._default_m = float(default_m)
        self._bridge = bridge
        self._active_inner: JointIkController | None = None

    def reset_idle(self) -> None:
        if self._bridge is not None and self._bridge.enabled:
            self._default_m = float(self._bridge.measured_m)
        elif self._active_inner is not None:
            self._default_m = float(self._active_inner.q_cmd[0])
        self._active_inner = None

    def set_active(self, inner: JointIkController) -> None:
        self._active_inner = inner

    def __call__(self) -> float:
        if self._bridge is not None and self._bridge.enabled:
            return float(self._bridge.measured_m)
        if self._active_inner is not None:
            return float(self._active_inner.q_cmd[0])
        return self._default_m


def main() -> int:
    ap = argparse.ArgumentParser(
        description="8-DOF controller daemon (window A)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", type=Path, default=Path("configs/joint_admittance_8dof.yaml"))
    ap.add_argument(
        "--state-relay",
        default="rm75_state",
        metavar="NAME",
        help="Publish robot state to SHM for twin / window C (default rm75_state)",
    )
    ap.add_argument("--no-state-relay", action="store_true", help="Do not publish SHM")
    ap.add_argument("--relay-hz", type=float, default=None, help="SHM publish rate (default from YAML)")
    ap.add_argument(
        "--hold",
        action="store_true",
        help="Stream CANFD idle hold (teach re-anchor). Do NOT use with d_sin_tool_y.py",
    )
    ap.add_argument("--verbose", "-v", action="store_true", help="Print loop / teach / phase status")
    ap.add_argument("--dry-run", action="store_true", help="build controllers only, do not connect")
    args = ap.parse_args()

    raw = load_yaml(args.config)
    startup = raw.get("startup", {})
    relay_cfg = parse_state_relay_config(raw)
    if args.no_state_relay:
        relay_name = None
    else:
        relay_name = str(args.state_relay or relay_cfg.name or "rm75_state")
    relay_hz = float(args.relay_hz) if args.relay_hz is not None else relay_cfg.hz
    dt = float(raw.get("timing", {}).get("dt_ms", 5.0)) / 1000.0
    rail_default_m = float(raw.get("inner", {}).get("rail", {}).get("q_ref_m", 0.0))
    rail_bridge = RailServoBridge(parse_rail_servo_config(raw))
    if args.verbose and rail_bridge.enabled and not rail_bridge.log_csv_path:
        log_dir = Path(__file__).resolve().parents[1] / "logs" / "rail_servo"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        rail_bridge.enable_log_csv(str(log_dir / f"rail_{ts}.csv"))
    rail_pub = _RailPublisher(rail_default_m, bridge=rail_bridge)

    if args.dry_run:
        mode = "hold+CANFD" if args.hold else "controller+hot-wait"
        print(f"rm75 controller: dry-run OK ({mode})", flush=True)
        return 0

    robot_cfg = raw.get("robot", {})
    relay: StateRelayPublisher | None = None
    inner: JointIkController | None = None
    hub: PhaseCommandHub | None = None
    # Long-lived kin for SHM pose: RealMan UDP pose is often ArmTip/link_7
    # (~220 mm behind gripper TCP). Overwrite with Pinocchio fk_pose.
    pub_kin = RobotKinematics()

    if args.hold:
        kin = pub_kin
        inner_cfg = build_joint_ik_config(raw)
        inner = JointIkController(kin, inner_cfg)
        rail_pub.set_active(inner)

    with RobotSession(
        ip=robot_cfg.get("ip"),
        port=robot_cfg.get("port"),
        config=args.config,
        quiet=True,
    ) as sess:
        try:
            if rail_bridge.enabled:
                rail_bridge.start()
                rail_pub._default_m = float(rail_bridge.measured_m)
            if inner is not None:
                maybe_sync_kin_tcp_from_config(raw=raw, kin=inner.kin, robot=sess.robot)
            else:
                maybe_sync_kin_tcp_from_config(raw=raw, kin=pub_kin, robot=sess.robot)
            bus = RobotStateBus(sess.robot, raw, robot_ip=sess.ip)
            bus.start()

            if relay_name:
                relay = StateRelayPublisher(
                    bus,
                    name=relay_name,
                    hz=relay_hz,
                    rail_m_fn=rail_pub,
                    kin=inner.kin if inner is not None else pub_kin,
                )
                relay.start()
                if args.hold:
                    print(
                        f"rm75 controller: hold @ {relay_hz:.0f} Hz",
                        flush=True,
                    )
                else:
                    print(
                        f"rm75 controller: running @ {relay_hz:.0f} Hz",
                        flush=True,
                    )
            elif args.hold:
                print("rm75 controller: hold (no SHM)", flush=True)
            else:
                print("rm75 controller: running (no SHM)", flush=True)

            if args.hold:
                assert inner is not None
                outer = CartesianTrackOuterLoop(
                    HoldReference(),
                    CartesianTrackConfig(
                        k_task=np.full(6, 2.0),
                        euler_order=inner.cfg.euler_order,
                        control_frame=inner.cfg.control_frame,
                    ),
                )
                run_joint_admittance_loop(
                    sess,
                    outer,
                    inner,
                    q_start_deg=None,
                    duration_s=None,
                    dt=dt,
                    force_observer=None,
                    follow=bool(startup.get("follow", True)),
                    move_speed=int(startup.get("move_speed", 20)),
                    realtime=bool(startup.get("realtime", False)),
                    watchdog_timeout_s=float(startup.get("watchdog_timeout_s", 0.1)),
                    state_bus=bus,
                    verbose=args.verbose,
                    rail_bridge=rail_bridge,
                )
            else:
                hub = PhaseCommandHub()
                _run_controller_service(
                    sess,
                    bus,
                    raw,
                    config_path=args.config,
                    hub=hub,
                    rail_m_fn=rail_pub,
                    rail_bridge=rail_bridge,
                    relay=relay,
                    verbose=args.verbose,
                )
        finally:
            if hub is not None:
                hub.close()
            if relay is not None:
                relay.stop()
            rail_bridge.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

