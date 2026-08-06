# Controller log — passivity / force-hold (2026-08-04)

## Run `233408` (broken — user report)

Command:
```bash
cd rm75_control && source env.sh
python apps/joint_admittance_8dof/d_sin_tool_y.py \
  --d-target joints --move-mode joint \
  --enable-force --desired-z 2 --scan-duration 3000 -v
```

Logs:
- `MD/logs_sin_tool_y/run_20260804_233408.csv`
- `MD/logs_sin_tool_y/rail_20260804_233408.csv`
- `MD/logs_sin_tool_y/run_20260804_233408_key_ds10.csv`

User: still shakes; hand cannot press down; **no active force hold**.

### Scan metrics (~40 s)

| metric | value |
|--------|-------|
| fz med / p95 / max | **0.49** / 3.04 / 5.28 N (Fd=2) |
| underforce frac (F<Fd−0.5) | **77.7%** |
| tank_gamma med / frac≈0 | **0** / **54%** (underforce: **66%**) |
| tank_E med | **0 mJ** (empty) |
| cap_press med | **3.00 mm/s** always |
| free_seek frac | 0.27% |
| v_r med | **0** (retract_only) |
| v_force med (underforce) | 1.7 mm/s |
| hard peaks fz>4 | 23 |

### Root cause

Passivity layer fought **force tracking**:

1. **Tank** drained on every press (`e_req = u·v·dt`) and never refilled under underforce → `γ→0` kills press drive (~52–66% of underforce ticks).
2. **PO/PC** treated `F·Δx>0` while chasing Fd as excess → occasional `D_PC=120`.
3. **`contact_press_cap=3 mm/s`** always after first touch → cannot re-acquire / hold 2 N; hand press feels locked.
4. Free-seek rarely re-armed after contact FSM stays latched.

Bounce still present (~23 hard peaks) but secondary to "压不动 / 不主动下压".

## Fix (post-233408)

Energy-limit gate: tank γ / PO-PC / tight `contact_press_cap` apply **only** when
`impact_danger | impact_timer | F>Fd+0.25 | overshoot/retract-brake`.

Else: `γ=1`, `D_PC=0`, refill tank toward `e_initial`, press cap = normal / `low_force_press_cap` (10 mm/s).

Tank/PO use `_tank_pc_active` (live impact_danger / over_force / overshoot), not a lingering impact_timer while F≪Fd — so bounce hold does not block re-press.

YAML: `contact_press_cap_m_s: 0.012` (tight only inside energy-limit).
Free-seek re-arm threshold: `max(0.30, free_seek_exit_force_n)` when FREE/LOST.

Code snapshot: `MD/code_snapshot_passivity/`

### Retest

```bash
cd rm75_control && source env.sh
# Window A: run_joint_admittance.py --config configs/joint_admittance_8dof.yaml
python apps/joint_admittance_8dof/d_sin_tool_y.py \
  --d-target joints --move-mode joint \
  --enable-force --desired-z 2 --scan-duration 3000 -v
```

Expect: fz med near 2 N; tank_gamma≈1 while underforce; cap_press ≥10 mm/s when F≪Fd; hand can push; still check hard bounce.

---
\nUpdated: 2026-08-04T23:38:13\n\n# Verbatim snapshot (post-233408 energy-limit gate)\n\n## `MD/code_snapshot_passivity/joint_admittance_8dof.yaml`\n\n```\n# Joint-space 8-DOF inner loop (rail_y + RM75 arm) — configs/joint_admittance_8dof.yaml
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
    twist_sigma_floor: 0.25   # keep ≥25% scan/force twist near σ (was 0.08 → stuck)
    twist_scale_lpf_tau_s: 0.08  # smooth σ-brake (kills single-tick 2× jumps)
    # Avoidance onset vs sigma_ref (rail escape). 1.25 keeps pose-D scan out of
    # a permanent escape latch (2.0 was too sticky).
    sigma_escape_ref_scale: 1.25
    warn_on_fail: false
    # Chiaverini 1997 SR damping for nullspace projection.
    sr_damping:
      lam0: 0.05
      sigma_ref: 0.08
      sigma_floor: 1.0e-6
    # σ-adaptive W_task (4d15c1d defaults).
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
    # 4d15c1d defaults (Jul-30 hw OK): uniform arm weights, k_center=1.
    k_center: 1.0
    k_limit: 2.0
    activation: 0.85
    weights: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    q_nominal_deg: [0.0, 0.0, -45.0, 0.0, 90.0, 0.0, 45.0, 0.0]
    manipulability:
      k_mu: 0.8
      eps_rad: 5.0e-4
      sigma_fade_ref: 0.12

  nullspace_d_null: 0.5
  nullspace_d_null_adaptive: 1.0
  nullspace_max_qdot_frac: 0.2

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
    k_sigma_boost: 2.0     # w_ext boosts up to 3x as σ → 0 (4d15c1d)
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
  causal_fc_hz: 12.0
  causal_order: 1
  causal_history: 5
  # Inertia compensation off on the joint stream (re-enable only with telemetry).
  use_inertia: false

hybrid_motion:
  force_axes:
  - 0
  - 0
  - 1
  - 0
  - 0
  - 0
  track_axes:
  - 1
  - 1
  - 0
  - 1
  - 1
  - 1
  kp_pos:
  - 2.0
  - 2.0
  - 0.0
  - 1.5
  - 1.5
  - 1.5
  pos_err_deadband_m: 0.0005
  pos_correction_max_m_s: 0.08
  system_delay_s: 0.015
  contact_threshold_n: 0.8
  contact_use_fz_only: true
  physical_contact:
    enabled: true
    enter_n: 0.8
    hard_enter_n: 1.5
    exit_n: 0.35
    # 50 ms rejects air inertia spikes (≤20 ms); true press stays for seconds.
    enter_confirm_s: 0.05
    exit_confirm_s: 0.1
  # Slightly wider band: blunt single-tick contact dips without sticky D.
  # Soft zone: no sticky exact-zero band (slow-push derder fix).
  deadband_n: 0.025
  deadband_width_n: 0.075
  deadband_soft_tanh: true
  max_velocity:
  - 0.22
  - 0.22
  - 0.1
  - 0.6
  - 0.6
  - 0.6
  max_acceleration:
  - 1.0
  - 1.0
  - 0.8
  - 2.0
  - 2.0
  - 2.0
  # Low baseline MD for light feel + fast under/over-force chase.
  # Bounce: stiffness-scheduled barrier + D_delay=κ·K̂b·T_dead.
  # Chatter: short-lived ΔD_hf(Is) on measured TCP vel. Steady offset: force_dob.
  admittance_mass_z: 1.0
  admittance_damping_z: 25.0
  max_vz_tool_m_s: 0.08
  desired_force_ramp_s: 0.30          # air seek ramp (was 0.8 — felt slow in air)
  recontact_force_ramp_s: 0.25
  # Dimeas OFF for passivity A/B (not impact protection).
  var_damping_enabled: false
  var_damping_omega_c_hz: 1.8
  var_damping_lambda: 0.951
  var_damping_f_max_n: 30.0
  var_damping_d_u: 60.0
  var_damping_m_u: 0.0
  var_damping_m_max: 3.0
  var_damping_dc_alpha: 0.02
  var_damping_hf_attack_s: 0.02
  var_damping_hf_hold_s: 0.12
  var_damping_hf_release_s: 0.10
  var_damping_hf_release_fast_s: 0.04
  var_damping_hf_on: 0.35
  var_damping_hf_off: 0.18
  var_damping_hf_err_n: 0.8
  var_damping_hf_slew_max: 800.0
  recontact_vz_cap_m_s: 0.02
  recontact_hold_s: 0.12
  # Passivity baseline (204355 lesson): cut active press energy, keep D0=25.
  t_eff_s: 0.070
  delay_press_budget_enabled: true
  delay_press_budget_min_n: 0.40
  delay_press_budget_frac: 0.25
  delay_press_v_floor_m_s: 0.003
  delay_press_cap_lpf_tau_s: 0.18
  suspect_recovery_enabled: true
  suspect_recovery_vz_cap_m_s: 0.012
  suspect_recovery_vr_press_max_m_s: 0.003
  suspect_recovery_f_abs_n: 0.3
  suspect_recovery_f_frac: 0.35
  suspect_recovery_hold_s: 0.20
  low_force_press_cap_m_s: 0.010
  low_force_press_enter_n: 1.80
  v_tcp_lpf_tau_s: 0.012
  v_tcp_clip_m_s: 0.12
  v_force_aw_enabled: false       # A/B: was weakening retract vs delayed TCP
  v_force_aw_tau_s: 0.040
  force_barrier:
    enabled: false
    press_only: true
    t_dead_s: 0.070
    t_pred_s: 0.070              # match Teff (was 30 ms under-predict)
    budget_min_n: 0.40
    budget_frac: 0.25
    f_keep_n: 0.3
    v_floor_press_m_s: 0.010
    v_floor_retract_m_s: 0.0
    f_panic_n: 20.0
    yield_overforce_n: 1.5
    yield_fdot_max_n_s: 60.0
    ke_seek_default: 300.0
    ke_min: 200.0
    ke_max: 4000.0
    ke_attack_s: 0.20
    ke_release_s: 0.8
    ke_free_hold_s: 0.5
    ke_v_press_min_m_s: 0.012
    ke_f_err_gate_n: 1.5
    ke_slew_up_max: 5000.0
    ke_impact_seed: 0.0
    cap_lpf_tau_s: 0.08
    limit_free_seek: false
    fdot_taps: 3
  # Impact: zero-centered Di; D0+Di in [60,100]; compress-rise re-arm.
  delay_damping_enabled: true
  delay_damping_mode: impact_only
  delay_damping_kappa: 0.8
  delay_damping_bd_max: 150.0
  impact_damping_hold_s: 0.12
  impact_damping_release_s: 0.10
  impact_damping_zeta: 0.9
  impact_damping_d_min: 60.0
  impact_damping_d_max: 100.0
  impact_ke_floor: 800.0
  # Impact: force-rise arm (not delayed v_tcp>0); confirm ~10 ms.
  impact_compress_v_min_m_s: 0.003
  impact_fdot_arm_n_s: 12.0
  impact_fpred_over_n: 0.15
  impact_fpred_horizon_s: 0.050
  impact_arm_confirm_s: 0.010
  impact_fdot_rearm_n_s: 5.0
  impact_rearm_f_frac: 0.9
  # Predictive retract stop → short zero-centered brake on vf.
  retract_brake_damping_ns_m: 70.0
  retract_brake_hold_s: 0.060
  retract_brake_release_s: 0.040
  # Interlock only during retract-overshoot episode; gate press drive (not vf).
  reverse_interlock_enter_m_s: 0.004
  reverse_interlock_exit_m_s: 0.0015
  reverse_interlock_enter_confirm_s: 0.010
  reverse_interlock_exit_confirm_s: 0.010
  v_gate_window_s: 0.030
  # Unified D_extra (impact/brake share one target).
  d_extra_attack_s: 0.005
  d_extra_release_s: 0.120
  d_extra_min_hold_s: 0.040
  # Impact danger renew while hard contact still unsafe.
  impact_danger_f_over_n: 0.15
  impact_danger_fdot_n_s: 5.0
  impact_danger_f_over_fdot_n: 0.10
  impact_safe_f_over_n: 0.25
  impact_safe_confirm_s: 0.025
  impact_pred_span_n: 1.5
  # Contact press slew (gentle). Air seek uses free_seek_accel_m_s2.
  force_slew_press_m_s2: 0.30
  force_slew_retract_m_s2: 1.20
  force_slew_press_to_retract_m_s2: 2.0
  force_slew_zero_cross_m_s2: 0.40
  # Air seek vs contact press. Tight contact_press_cap only in energy-limit
  # (impact / overshoot / F>Fd+0.25); under-force tracking uses low_force_cap.
  free_seek_vz_m_s: 0.080
  free_seek_accel_m_s2: 0.80
  free_seek_exit_force_n: 0.15
  free_seek_exit_fdot_n_s: 5.0
  contact_press_cap_m_s: 0.012
  # Tank must be small vs ½M v² (3.2 mJ @ 80 mm/s) — was 80 mJ (inert).
  press_energy_tank:
    enabled: true
    e_max_j: 0.004
    e_initial_j: 0.001
    e_min_j: 0.0
    credit_gain: 0.20
    dx_deadband_m: 2.0e-6
    seed_on_acquire: false
  # Bidirectional real-port PO/PC (elastic rebound / delayed inject).
  port_passivity:
    enabled: true
    e_max_j: 0.004
    e_initial_j: 0.002
    eps_v2dt: 1.0e-8
    d_pc_max: 120.0
    leak_s: 0.5
  wrist_relax_enabled: true
  wrist_relax_enter_rad: 0.175
  wrist_relax_exit_rad: 0.35
  wrist_relax_floor: 0.30
  wrist_relax_lpf_tau_s: 0.08
  force_dob:
    enabled: false               # A/B: active energy source
    ki: 8.0
    leak_s: 0.4
    u_max_n: 1.5
    freeze_is: 0.45
    reset_on_reversal: true
  # Soften under-force chase near scan turnaround (tool-XY slow).
  force_lateral_soft_m_s: 0.006
  force_lateral_full_m_s: 0.018
  force_lateral_gain_floor: 0.35
  adaptive_ke:
    enabled: true
    # Telemetry / impact seed only — barrier K̂b drives contact D_delay.
    drive_damping: false
    zeta: 0.9
    ke_initial: 80.0
    ke_min: 40.0
    ke_max: 2500.0
    ke_impact_initial: 500.0
    ke_forgetting: 0.995
    ke_forgetting_inc: 0.88
    ke_idle_decay_s: 2.0
    ke_soft_floor: 120.0
    ke_detach_decay_s: 1.0
    displacement_source: admittance
    dx_threshold_m: 8.0e-05
    contact_force_n: 0.8
    settle_ticks: 10
    gate_lateral_velocity: true
    lateral_vel_gate_m_s: 0.02
    gate_df_spike: true
    df_spike_n: 4.0
    f_err_gate_n: 1.2
    f_err_gate_frac: 0.35
    bd_min: 25.0
    bd_max: 600.0
    bd_slew_max: 400.0
    ke_slew_max: 1200.0
  proactive_feedforward: true
  # A/B: retract-only — cut Dv_r press injection; under-force via e_f / D0.
  proactive_retract_only: true
  proactive_gain: 0.24
  proactive_retract_gain: 0.30
  proactive_leak_s: 0.25
  v_r_max_m_s: 0.06
  proactive_gate_press_on_is: false
  proactive_press_is_gate_start: 0.2
  proactive_press_is_gate: 0.6
  # Soften press when Is high (never hard-kill); slew-limit rising v_r.
  proactive_press_is_soft_floor: 0.45
  proactive_press_is_soft_stop: 0.85
  proactive_press_slew_max_m_s2: 0.35
  proactive_press_drive_max: 1.2
  proactive_retract_drive_max: 1.4
  proactive_reset_on_reversal: true
  force_scale_min_n: 0.18
  force_scale_fraction: 0.12
  fast_retract_guard:
    enabled: true
    cutoff_hz: 20.0
    stop_margin_n: 0.25
    stop_margin_fraction: 0.05
    rearm_margin_n: 0.45
    rearm_margin_fraction: 0.1
    stop_confirm_s: 0.015
    rearm_confirm_s: 0.01
    min_hold_s: 0.025
    max_sensor_age_s: 0.02
    # Stop retract when F_pred,down reaches Fd (before low-side 1.75 N).
    # Armed + |ḟ| gate: bounce overshoot only, not slow surface follow.
    retract_stop_prediction_s: 0.045
    retract_stop_margin_n: 0.10
    retract_stop_confirm_s: 0.005
    retract_stop_fdot_n_s: 15.0
hw:
  lw100:
    enabled: true
    host: 192.168.0.7
    port: 8234
    slave: 1
    lead_mm: 10.0
    zero_mode: current
    counts0: 0
    # Host rail_y ↔ motor: -1 flips FA24 RPM (+ encoder map in rail_servo).
    sign: -1
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
\n```\n\n## `MD/code_snapshot_passivity/controller.py`\n\n```\n"""Stable tool-frame force/motion decoupling and trajectory tracking.

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
    # Force-axis asymmetric slew [m/s²] — contact press kept gentle.
    force_slew_press_m_s2: float = 0.30
    force_slew_retract_m_s2: float = 1.20
    force_slew_press_to_retract_m_s2: float = 2.0
    force_slew_zero_cross_m_s2: float = 0.40
    # Free-space seek (air) vs contact press — separate budgets.
    free_seek_vz_m_s: float = 0.080
    free_seek_accel_m_s2: float = 0.80
    free_seek_exit_force_n: float = 0.15
    free_seek_exit_fdot_n_s: float = 5.0
    contact_press_cap_m_s: float = 0.012
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
    # Force-task latched + under setpoint: hard press ceiling [m/s].
    low_force_press_cap_m_s: float = 0.010
    low_force_press_enter_n: float = 1.80
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
                c.get("force_slew_press_m_s2", 0.30)
            ),
            force_slew_retract_m_s2=float(
                c.get("force_slew_retract_m_s2", 1.20)
            ),
            force_slew_press_to_retract_m_s2=float(
                c.get("force_slew_press_to_retract_m_s2", 2.0)
            ),
            force_slew_zero_cross_m_s2=float(
                c.get("force_slew_zero_cross_m_s2", 0.40)
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
                c.get("contact_press_cap_m_s", 0.012)
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
                c.get("low_force_press_cap_m_s", 0.010)
            ),
            low_force_press_enter_n=float(
                c.get("low_force_press_enter_n", 1.80)
            ),
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
        if self._energy_limit_active:
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

        # Bidirectional real-port PO/PC — only when energy-limit active.
        # Force-tracking work (F·Δx>0 while under Fd) is not a passivity fault.
        v_a = (
            float(v_tcp_gate)
            if np.isfinite(v_tcp_gate)
            else (
                float(v_tcp_filt)
                if np.isfinite(v_tcp_filt)
                else float(self.v_force_z)
            )
        )
        if self._energy_limit_active:
            d_pc = self._port_po.update(
                f_ext_z=float(f_ext_z),
                dx_m=float(dx_tcp),
                v_actual_m_s=v_a,
                dt_s=dt_eff,
            )
        else:
            self._port_po.update(
                f_ext_z=float(f_ext_z),
                dx_m=float(dx_tcp),
                v_actual_m_s=v_a,
                dt_s=dt_eff,
            )
            # Leak observer back toward initial; suppress D_PC while tracking.
            e0 = float(self.cfg.port_passivity.e_initial_j)
            emax = float(self.cfg.port_passivity.e_max_j)
            self._port_po.energy_j = min(
                emax, max(float(self._port_po.energy_j), 0.5 * e0)
            )
            self._port_po.excess_j = 0.0
            self._port_po.d_pc = 0.0
            d_pc = 0.0
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
\n```\n\n## `MD/code_snapshot_passivity/press_energy_tank.py`\n\n```\n"""Contact-port energy tank + bidirectional passivity observer (PO/PC).

Tank (press budget):
    ΔE = F_mid · Δx   (residual-accumulated position increments)
    scales only active press drive; retract unrestricted.

PO/PC (bidirectional):
    tracks excess energy injected into the real TCP port and adds a short
    dissipative term D_PC · v_a when the observer goes negative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PressEnergyTankConfig:
    enabled: bool = True
    e_max_j: float = 0.004
    e_initial_j: float = 0.001
    e_min_j: float = 0.0
    credit_gain: float = 0.20
    # Residual accumulator floor [m] — small steps accumulate, not discarded.
    dx_deadband_m: float = 2.0e-6
    # Re-seeding on bounce reacquire refills the tank — keep false.
    seed_on_acquire: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> PressEnergyTankConfig:
        c = raw.get("hybrid_motion", raw.get("controller", raw))
        p = c.get("press_energy_tank", {})
        if not isinstance(p, dict):
            p = {}
        return cls(
            enabled=bool(p.get("enabled", True)),
            e_max_j=float(p.get("e_max_j", 0.004)),
            e_initial_j=float(p.get("e_initial_j", 0.001)),
            e_min_j=float(p.get("e_min_j", 0.0)),
            credit_gain=float(p.get("credit_gain", 0.20)),
            dx_deadband_m=float(p.get("dx_deadband_m", 2.0e-6)),
            seed_on_acquire=bool(p.get("seed_on_acquire", False)),
        )


@dataclass
class PortPassivityConfig:
    """Bidirectional real-port passivity observer / controller."""

    enabled: bool = True
    e_max_j: float = 0.004
    e_initial_j: float = 0.002
    # Floor on v_a²·dt in D_PC denominator.
    eps_v2dt: float = 1.0e-8
    d_pc_max: float = 120.0
    # Leak excess back toward zero when dissipating [1/s].
    leak_s: float = 0.5

    @classmethod
    def from_dict(cls, raw: dict) -> PortPassivityConfig:
        c = raw.get("hybrid_motion", raw.get("controller", raw))
        p = c.get("port_passivity", {})
        if not isinstance(p, dict):
            p = {}
        return cls(
            enabled=bool(p.get("enabled", True)),
            e_max_j=float(p.get("e_max_j", 0.004)),
            e_initial_j=float(p.get("e_initial_j", 0.002)),
            eps_v2dt=float(p.get("eps_v2dt", 1.0e-8)),
            d_pc_max=float(p.get("d_pc_max", 120.0)),
            leak_s=float(p.get("leak_s", 0.5)),
        )


class PressEnergyTank:
    def __init__(self, cfg: PressEnergyTankConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.energy_j = float(self.cfg.e_initial_j)
        self.gamma = 1.0
        self._f_prev = float("nan")
        self._dx_residual = 0.0

    def seed(self) -> None:
        if self.cfg.seed_on_acquire:
            self.energy_j = float(self.cfg.e_initial_j)
            self.gamma = 1.0

    def _consume_dx(self, dx_m: float) -> float:
        self._dx_residual += float(dx_m)
        dead = max(float(self.cfg.dx_deadband_m), 0.0)
        if abs(self._dx_residual) < dead:
            return 0.0
        dx_used = self._dx_residual
        self._dx_residual = 0.0
        return dx_used

    def observe_and_scale(
        self,
        *,
        f_ext_z: float,
        dx_m: float,
        u_press: float,
        v_press_est_m_s: float,
        dt_s: float,
    ) -> float:
        """Credit/debit tank; return γ ∈ [0, 1] for active press scaling."""
        cfg = self.cfg
        if not cfg.enabled:
            self.gamma = 1.0
            return 1.0

        f = float(f_ext_z)
        dx = self._consume_dx(dx_m)
        f_prev = self._f_prev if self._f_prev == self._f_prev else f
        f_mid = 0.5 * (f + float(f_prev))
        self._f_prev = f

        dW = f_mid * dx
        # Partial credit only — elastic return must not fully refill the tank.
        if dW < 0.0 and f_mid > 0.0:
            self.energy_j = min(
                float(cfg.e_max_j),
                self.energy_j + (-dW) * float(cfg.credit_gain),
            )

        u_p = max(float(u_press), 0.0)
        v_p = max(float(v_press_est_m_s), 0.0)
        dt = max(float(dt_s), 0.0)
        e_req = u_p * v_p * dt
        if e_req <= 1e-12 or u_p <= 1e-12:
            self.gamma = 1.0
            return 1.0

        e_avail = max(float(self.energy_j) - float(cfg.e_min_j), 0.0)
        gamma = min(1.0, e_avail / e_req)
        self.energy_j = max(
            float(cfg.e_min_j),
            self.energy_j - gamma * e_req,
        )
        self.gamma = float(gamma)
        return self.gamma


class PortPassivityObserver:
    """Bidirectional PO/PC on the real TCP contact port."""

    def __init__(self, cfg: PortPassivityConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.energy_j = float(self.cfg.e_initial_j)
        self.excess_j = 0.0
        self.d_pc = 0.0
        self._f_prev = float("nan")
        self._dx_residual = 0.0

    def seed(self) -> None:
        self.energy_j = float(self.cfg.e_initial_j)
        self.excess_j = 0.0
        self.d_pc = 0.0

    def update(
        self,
        *,
        f_ext_z: float,
        dx_m: float,
        v_actual_m_s: float,
        dt_s: float,
    ) -> float:
        """Observe port work; return D_PC ≥ 0 (zero-centered on v_a)."""
        cfg = self.cfg
        if not cfg.enabled:
            self.d_pc = 0.0
            self.excess_j = 0.0
            return 0.0

        f = float(f_ext_z)
        self._dx_residual += float(dx_m)
        # Use any accumulated motion; tiny residual stays for next tick.
        dx = self._dx_residual
        if abs(dx) < 1e-9:
            dx_used = 0.0
        else:
            dx_used = dx
            self._dx_residual = 0.0

        f_prev = self._f_prev if self._f_prev == self._f_prev else f
        f_mid = 0.5 * (f + float(f_prev))
        self._f_prev = f
        dt = max(float(dt_s), 0.0)

        # Work done ON the environment by the tip.
        dW_env = f_mid * dx_used
        # Robot energy storage relative to port: decreases when env is loaded.
        self.energy_j -= dW_env
        if self.energy_j > float(cfg.e_max_j):
            self.energy_j = float(cfg.e_max_j)

        excess = max(0.0, -self.energy_j)  # negative tank ⇒ injected too much
        self.excess_j = float(excess)
        if excess <= 1e-12 or dt <= 0.0:
            self.d_pc = 0.0
            # Mild leak toward initial when passive.
            if self.energy_j < float(cfg.e_initial_j):
                leak = max(float(cfg.leak_s), 0.0)
                self.energy_j += (
                    (float(cfg.e_initial_j) - self.energy_j)
                    * (1.0 - math.exp(-leak * dt))
                )
            return 0.0

        v_a = float(v_actual_m_s)
        denom = max(v_a * v_a * dt, float(cfg.eps_v2dt))
        d_pc = min(float(cfg.d_pc_max), excess / denom)
        self.d_pc = float(d_pc)
        # Apply dissipation credit: E += D_PC · v_a² · dt
        self.energy_j += d_pc * v_a * v_a * dt
        if self.energy_j > 0.0:
            self.energy_j = min(self.energy_j, float(cfg.e_max_j))
        return self.d_pc
\n```\n\n## `MD/code_snapshot_passivity/fast_retract_guard.py`\n\n```\n"""Low-latency veto for stale active over-force retraction.

The 6 Hz force used by the passive admittance remains untouched.  Compensated
raw force clears/holds a negative active reference when a drop is observed or
*predicted* (cmd→TCP stop delay).  This guard never commands a press.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class FastRetractGuardConfig:
    enabled: bool = True
    cutoff_hz: float = 20.0
    stop_margin_n: float = 0.25
    stop_margin_fraction: float = 0.05
    rearm_margin_n: float = 0.45
    rearm_margin_fraction: float = 0.10
    stop_confirm_s: float = 0.015
    rearm_confirm_s: float = 0.010
    min_hold_s: float = 0.025
    max_sensor_age_s: float = 0.020
    # Predict force at TCP-stop horizon; stop retract before low-side cross.
    # Requires high-side arm + fast fall so slow surface tracking is untouched.
    retract_stop_prediction_s: float = 0.045
    retract_stop_margin_n: float = 0.10
    retract_stop_confirm_s: float = 0.005
    retract_stop_fdot_n_s: float = 15.0

    @classmethod
    def from_dict(cls, raw: dict) -> FastRetractGuardConfig:
        c = raw.get("hybrid_motion", raw.get("controller", raw))
        p = c.get("fast_retract_guard", {})
        if not isinstance(p, dict):
            p = {}
        return cls(
            enabled=bool(p.get("enabled", True)),
            cutoff_hz=float(p.get("cutoff_hz", 20.0)),
            stop_margin_n=float(p.get("stop_margin_n", 0.25)),
            stop_margin_fraction=float(p.get("stop_margin_fraction", 0.05)),
            rearm_margin_n=float(p.get("rearm_margin_n", 0.45)),
            rearm_margin_fraction=float(p.get("rearm_margin_fraction", 0.10)),
            stop_confirm_s=float(p.get("stop_confirm_s", 0.015)),
            rearm_confirm_s=float(p.get("rearm_confirm_s", 0.010)),
            min_hold_s=float(p.get("min_hold_s", 0.025)),
            max_sensor_age_s=float(p.get("max_sensor_age_s", 0.020)),
            retract_stop_prediction_s=float(
                p.get("retract_stop_prediction_s", 0.045)
            ),
            retract_stop_margin_n=float(
                p.get("retract_stop_margin_n", 0.10)
            ),
            retract_stop_confirm_s=float(
                p.get("retract_stop_confirm_s", 0.005)
            ),
            retract_stop_fdot_n_s=float(
                p.get("retract_stop_fdot_n_s", 15.0)
            ),
        )


class FastRetractGuard:
    def __init__(self, cfg: FastRetractGuardConfig) -> None:
        self.cfg = cfg
        self._raw_window: deque[float] = deque(maxlen=3)
        self.reset()

    def reset(self) -> None:
        self._raw_window.clear()
        self.fast_force_n = float("nan")
        self.force_dot_fast = 0.0
        self.force_pred_down = float("nan")
        self.armed = False
        self.hold = False
        self.valid = False
        self._stop_timer_s = 0.0
        self._rearm_timer_s = 0.0
        self._hold_timer_s = 0.0
        self._prev_fast_force = float("nan")
        self.stop_count = 0
        self.rearm_count = 0
        self.predictive_stop_count = 0

    def _update_fast_force(self, raw_force_n: float, dt_s: float) -> float:
        self._raw_window.append(float(raw_force_n))
        raw_median = float(np.median(np.asarray(self._raw_window, dtype=float)))
        if not np.isfinite(self.fast_force_n):
            self.fast_force_n = raw_median
            self._prev_fast_force = raw_median
            self.force_dot_fast = 0.0
            return self.fast_force_n
        fc = max(float(self.cfg.cutoff_hz), 0.0)
        alpha = (
            1.0 - math.exp(-2.0 * math.pi * fc * max(dt_s, 0.0))
            if fc > 0.0
            else 1.0
        )
        self.fast_force_n += float(np.clip(alpha, 0.0, 1.0)) * (
            raw_median - self.fast_force_n
        )
        if dt_s > 1e-9 and np.isfinite(self._prev_fast_force):
            raw_dot = (self.fast_force_n - self._prev_fast_force) / dt_s
            # Mild LPF on ḟ_fast (~8 ms) to reject single-tick spikes.
            tau = 0.008
            a_dot = 1.0 - math.exp(-dt_s / tau)
            self.force_dot_fast += a_dot * (raw_dot - self.force_dot_fast)
        self._prev_fast_force = self.fast_force_n
        return self.fast_force_n

    def update(
        self,
        *,
        raw_force_n: float | None,
        desired_force_n: float,
        filtered_eff_n: float,
        active_reference_m_s: float,
        dt_s: float,
        sensor_age_s: float | None,
        instability_index: float,
    ) -> bool:
        cfg = self.cfg
        dt = max(float(dt_s), 0.0)
        age_valid = (
            sensor_age_s is None
            or (
                np.isfinite(sensor_age_s)
                and float(sensor_age_s) <= max(cfg.max_sensor_age_s, 0.0)
            )
        )
        self.valid = bool(
            cfg.enabled
            and raw_force_n is not None
            and np.isfinite(raw_force_n)
            and np.isfinite(desired_force_n)
            and age_valid
        )
        if not self.valid:
            # Fail open: the established passive + active escape law remains.
            # Discard stale fast-path history as well; after a sensor dropout
            # the first fresh sample must prime a new filter episode rather
            # than blend with pre-dropout force.
            self._raw_window.clear()
            self.fast_force_n = float("nan")
            self.force_dot_fast = 0.0
            self.force_pred_down = float("nan")
            self._prev_fast_force = float("nan")
            self.armed = False
            self.hold = False
            self._stop_timer_s = 0.0
            self._rearm_timer_s = 0.0
            self._hold_timer_s = 0.0
            return False

        fast_force = self._update_fast_force(float(raw_force_n), dt)
        target = abs(float(desired_force_n))
        stop_margin = max(
            float(cfg.stop_margin_n),
            float(cfg.stop_margin_fraction) * target,
        )
        rearm_margin = max(
            float(cfg.rearm_margin_n),
            float(cfg.rearm_margin_fraction) * target,
        )
        # Crossing guard: arm on the high side; legacy stop on the low side.
        # Predictive stop uses falling ḟ × T_stop so retract ends near Fd
        # *before* the delayed TCP finishes lifting off.
        arm_level = target + stop_margin
        stop_level = max(target - stop_margin, 0.0)
        rearm_level = target + rearm_margin
        pred_horizon = max(float(cfg.retract_stop_prediction_s), 0.0)
        self.force_pred_down = float(
            fast_force + min(self.force_dot_fast, 0.0) * pred_horizon
        )
        pred_stop_level = target + max(float(cfg.retract_stop_margin_n), 0.0)
        retract_episode = (
            float(filtered_eff_n) < 0.0
            and float(active_reference_m_s) <= 0.0
        )

        if self.hold:
            self._hold_timer_s += dt
            if fast_force >= rearm_level:
                self._rearm_timer_s += dt
            else:
                self._rearm_timer_s = 0.0

            can_leave = self._hold_timer_s + 1e-12 >= max(
                cfg.min_hold_s,
                0.0,
            )
            rearm_confirm = max(
                cfg.rearm_confirm_s,
                0.015 if instability_index > 0.6 else 0.0,
            )
            if can_leave and (
                self._rearm_timer_s + 1e-12 >= rearm_confirm
                or float(filtered_eff_n) >= 0.0
            ):
                self.hold = False
                self.armed = fast_force >= arm_level
                self._rearm_timer_s = 0.0
                self._hold_timer_s = 0.0
                self.rearm_count += 1
            return self.hold

        if not retract_episode:
            self.armed = False
            self._stop_timer_s = 0.0
            return False

        if fast_force >= arm_level:
            self.armed = True
            self._stop_timer_s = 0.0
            return False

        # Predictive: armed over-force episode + fast fall whose Teff-horizon
        # prediction reaches Fd.  Slow surface tracking (small |ḟ|) is ignored.
        fdot_gate = max(float(cfg.retract_stop_fdot_n_s), 0.0)
        predictive = (
            self.armed
            and pred_horizon > 0.0
            and self.force_dot_fast <= -fdot_gate
            and self.force_pred_down <= pred_stop_level
        )
        low_side = self.armed and fast_force <= stop_level
        if predictive or low_side:
            self._stop_timer_s += dt
            if predictive:
                confirm = max(float(cfg.retract_stop_confirm_s), 0.0)
            else:
                confirm = max(
                    float(cfg.stop_confirm_s),
                    0.020 if instability_index > 0.6 else 0.0,
                )
            if self._stop_timer_s + 1e-12 >= confirm:
                self.hold = True
                self._hold_timer_s = 0.0
                self._stop_timer_s = 0.0
                self.stop_count += 1
                if predictive:
                    self.predictive_stop_count += 1
        else:
            # Stay armed throughout the hysteresis band, but confirmation is
            # continuous-time only while a stop condition is true.
            self._stop_timer_s = 0.0
        return self.hold
\n```\n\n## `MD/code_snapshot_passivity/loop.py`\n\n```\n"""Joint-space inner loop: Cartesian twist -> absolute joint angles (rm_movej_canfd).

``JointIkController``: hardware-free WBC QP IK + safety clamp (no send-path LPF).
``run_joint_admittance_phases``: on-robot orchestration closing on FK(q_meas).
"""

from __future__ import annotations

import csv
import inspect
import math
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
    # Leave escape only above enter·scale (hysteresis); kills ∇μ↔3×center chatter.
    # Keep near 1.1 so pose D (σ≈0.10–0.12) can clear the latch.
    centering_recovery_exit_scale: float = 1.12
    # Soft-ramp recovery gain (s); 0 = hard step (old behaviour).
    centering_recovery_blend_tau_s: float = 0.25


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


def twist_scale_target(
    sigma_min: float,
    sigma_ref: float,
    floor: float = 0.25,
) -> float:
    """Cartesian/force twist scale vs σ (monotonic, floor-clamped, no kink).

    Deep-σ square branch was removed: it dropped scale to ~0.09 and produced
    single-tick 2× jumps on recovery (run_20260804_145958).
    """
    if float(sigma_ref) <= 1e-9 or float(sigma_min) >= float(sigma_ref):
        return 1.0
    return float(max(float(sigma_min) / float(sigma_ref), float(floor)))


def twist_scale_lpf_step(
    filt: float,
    target: float,
    *,
    dt: float,
    tau_s: float,
) -> float:
    """One first-order LPF step on twist_scale (tau<=0 → hard step)."""
    if float(tau_s) <= 1e-6 or float(dt) <= 1e-9:
        return float(target)
    a = float(np.clip(float(dt) / float(tau_s), 0.0, 1.0))
    return float(filt) + a * (float(target) - float(filt))


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
        self._sigma_grad_period: int = 10  # 4d15c1d: ~50 ms refresh @ 200 Hz
        self._twist_scale_filt: float = 1.0
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
        self._twist_scale_filt = 1.0
        self._apply_rail_mode_side_effects()

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
    ) -> np.ndarray:
        # 4d15c1d: ∇μ XOR centering; optional this-tick σ for faster escape.
        qdot0 = self.secondary.compose(
            q,
            qdot_ff,
            self.core.qdot_prev,
            arm_suppressed=self._arm_task_suppressed,
            sigma_min=(
                self.last_sigma_min if sigma_min is None else float(sigma_min)
            ),
            sigma_ref=self.cfg.qp.sr_damping.sigma_ref,
            centering_suppressed=self._centering_suppressed,
            centering_sigma_fade=centering_sigma_fade,
            manipulability_active=(
                self._manipulability_active
                if manipulability_active is None
                else manipulability_active
            ),
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
        f_ext_z: float | None = None,
        f_des_z: float | None = None,
    ) -> JointIkStep:
        """One Cartesian-tracking WBC step.

        ``q_meas`` rotates tool→base twist and bounds command lead via QP
        velocity constraints (never a position teleport). ``qdot_ff`` feeds
        the nullspace with centering / arm-angle tasks.
        """
        _ = f_ext_z, f_des_z  # call-site compat; 4d15 path does not special-case
        dt = self.cfg.dt if dt is None else dt
        q_prev = self.q_cmd
        follow_err = 0.0 if q_meas is None else float(np.max(np.abs(q_prev - q_meas)))
        q_rot = q_meas if q_meas is not None else q_prev
        twist_base = self._twist_to_base(twist, q_rot)

        # Soften Cartesian (incl. force) before the QP when already near
        # singularity. Floor + LPF (no square kink) so deep-σ twist stays
        # usable and recovery does not punch the scan (run_20260804_145958).
        sigma_ref = float(self.cfg.qp.sr_damping.sigma_ref)
        J_pre = self.kin.jacobian(q_prev)
        sigma_pre = float(self.kin.singular_values(J_pre).min())
        floor = float(getattr(self.cfg.qp, "twist_sigma_floor", 0.25))
        twist_scale = twist_scale_target(sigma_pre, sigma_ref, floor)
        tau = float(getattr(self.cfg.qp, "twist_scale_lpf_tau_s", 0.0) or 0.0)
        self._twist_scale_filt = twist_scale_lpf_step(
            self._twist_scale_filt,
            twist_scale,
            dt=float(dt),
            tau_s=tau,
        )
        if self._twist_scale_filt < 1.0 - 1e-9:
            twist_base = twist_base * float(self._twist_scale_filt)

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

        # Preferred-extension rail coordination (COUPLED only) — 4d15c1d.
        rail_task_vel: float | None = None
        rail_task_weight = 0.0
        rail_ext_err = 0.0
        manip_for_saturation = bool(self._manipulability_active)
        if (
            self.rail_ext_task is not None
            and self._rail_ext_active
            and self._rail_mode == RailMode.COUPLED
        ):
            sigma_now = float(sigma_pre)
            sig_scale = 1.0
            if sigma_ref > 1e-9 and sigma_now < sigma_ref:
                sig_scale = max(sigma_now / sigma_ref, 0.25)
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
                sigma_grad_rail=self._sigma_grad_rail_cached,
                vel_ff=vel_ff,
                dt_s=float(dt),
            )
            rail_ext_err = self.rail_ext_task.last_err_m
            if w_ext > 0.0:
                rail_task_vel = v_ext
                rail_task_weight = w_ext
            # Escape arm singularities in nullspace whenever σ is depressed
            # (∇μ XOR centering in SecondaryComposer).
            if sigma_ref > 1e-9 and sigma_now < sigma_ref:
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
        """Governor scale (0..1) for trajectory/FF; force loop stays on wall clock."""
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
        q_meas: np.ndarray | None = None,
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
            q_meas=q_meas,
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
           "force_reference_fast_clear",
           "force_fast_z",
           "retract_guard_armed", "retract_fast_hold",
           "retract_fast_stop_count", "retract_fast_rearm_count",
           "force_task_latched",
           "physical_contact_state",
           "physical_contact_acquire_event", "physical_contact_loss_event",
           "physical_contact_reacquire_event",
           "physical_contact_low_timer_s", "physical_contact_high_timer_s",
           "mass_z_eff", "takeover",
           "dt_actual_s", "sensor_age_s",
           "fx_raw_comp", "fy_raw_comp", "fz_raw_comp",
           "vz_achieved_tool", "contact_present",
           "force_pred_z", "force_dot_z", "cap_press_z", "cap_retract_z",
           "ke_barrier", "damping_delay_z", "damping_impact_z",
           "damping_retract_brake_z", "reverse_interlock_active",
           "reverse_interlock_gate", "impact_danger", "d_extra_target_z",
           "f_err_raw", "f_err_eff", "v_force_raw", "v_tcp_z_gate",
           "tank_energy_j", "tank_gamma",
           "port_energy_j", "port_excess_j", "damping_pc_z",
           "free_seek_active",
           "wrist_relax_scale",
           "u_dob_z", "dob_frozen", "suspect_recovery", "v_tcp_z_filt",
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
        force_reference_fast_clear = getattr(
            ctrl, "force_reference_fast_clear", False
        )
        force_fast_z = getattr(ctrl, "force_fast_z", float("nan"))
        retract_guard_armed = getattr(ctrl, "retract_guard_armed", False)
        retract_fast_hold = getattr(ctrl, "retract_fast_hold", False)
        retract_fast_stop_count = getattr(
            ctrl, "retract_fast_stop_count", 0
        )
        retract_fast_rearm_count = getattr(
            ctrl, "retract_fast_rearm_count", 0
        )
        force_task_latched = getattr(ctrl, "force_task_latched", False)
        physical_contact_state = getattr(
            ctrl, "physical_contact_state", ""
        )
        physical_contact_acquire_event = getattr(
            ctrl, "physical_contact_acquire_event", False
        )
        physical_contact_loss_event = getattr(
            ctrl, "physical_contact_loss_event", False
        )
        physical_contact_reacquire_event = getattr(
            ctrl, "physical_contact_reacquire_event", False
        )
        physical_contact_tracker = getattr(ctrl, "_physical_contact", None)
        physical_contact_low_timer = getattr(
            ctrl,
            "physical_contact_low_timer_s",
            getattr(physical_contact_tracker, "low_timer_s", float("nan")),
        )
        physical_contact_high_timer = getattr(
            ctrl,
            "physical_contact_high_timer_s",
            getattr(physical_contact_tracker, "high_timer_s", float("nan")),
        )
        mass_z_eff = getattr(ctrl, "mass_z_eff", float("nan"))
        takeover = getattr(ctrl, "takeover_active", False)
        contact_present = getattr(ctrl, "contact_present", False)
        cap_press_z = getattr(ctrl, "cap_press_z", float("nan"))
        cap_retract_z = getattr(ctrl, "cap_retract_z", float("nan"))
        force_pred_z = getattr(ctrl, "force_pred_z", float("nan"))
        force_dot_z = getattr(ctrl, "force_dot_z", float("nan"))
        ke_barrier = getattr(ctrl, "ke_barrier", float("nan"))
        damping_delay_z = getattr(ctrl, "damping_delay_z", float("nan"))
        damping_impact_z = getattr(ctrl, "damping_impact_z", float("nan"))
        damping_retract_brake_z = getattr(
            ctrl, "damping_retract_brake_z", float("nan")
        )
        reverse_interlock_active = getattr(
            ctrl, "reverse_interlock_active", False
        )
        reverse_interlock_gate = getattr(
            ctrl, "reverse_interlock_gate", float("nan")
        )
        impact_danger = getattr(ctrl, "impact_danger", False)
        d_extra_target_z = getattr(ctrl, "d_extra_target_z", float("nan"))
        f_err_raw = getattr(ctrl, "f_err_raw", float("nan"))
        f_err_eff = getattr(ctrl, "f_err_eff", float("nan"))
        v_force_raw = getattr(ctrl, "v_force_raw", float("nan"))
        v_tcp_z_gate = getattr(ctrl, "v_tcp_z_gate", float("nan"))
        tank_energy_j = getattr(ctrl, "tank_energy_j", float("nan"))
        tank_gamma = getattr(ctrl, "tank_gamma", float("nan"))
        port_energy_j = getattr(ctrl, "port_energy_j", float("nan"))
        port_excess_j = getattr(ctrl, "port_excess_j", float("nan"))
        damping_pc_z = getattr(ctrl, "damping_pc_z", float("nan"))
        free_seek_active = getattr(ctrl, "free_seek_active", False)
        wrist_relax_scale = getattr(ctrl, "wrist_relax_scale", float("nan"))
        u_dob_z = getattr(ctrl, "u_dob_z", float("nan"))
        dob_frozen = getattr(ctrl, "dob_frozen", False)
        suspect_recovery = getattr(ctrl, "suspect_recovery_active", False)
        v_tcp_z_filt = getattr(ctrl, "v_tcp_z_filt", float("nan"))
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
               int(bool(force_reference_fast_clear)),
               f"{force_fast_z:.3f}",
               int(bool(retract_guard_armed)),
               int(bool(retract_fast_hold)),
               int(retract_fast_stop_count),
               int(retract_fast_rearm_count),
               int(bool(force_task_latched)),
               str(physical_contact_state),
               int(bool(physical_contact_acquire_event)),
               int(bool(physical_contact_loss_event)),
               int(bool(physical_contact_reacquire_event)),
               f"{float(physical_contact_low_timer):.6f}",
               f"{float(physical_contact_high_timer):.6f}",
               f"{mass_z_eff:.4f}",
               int(bool(takeover)),
               f"{dt_actual_s:.6f}", f"{sensor_age_s:.6f}",
               f"{raw_comp[0]:.3f}", f"{raw_comp[1]:.3f}", f"{raw_comp[2]:.3f}",
               f"{v_tcp_z_actual:.6f}", int(bool(contact_present)),
               f"{force_pred_z:.4f}", f"{force_dot_z:.4f}",
               f"{cap_press_z:.6f}", f"{cap_retract_z:.6f}",
               f"{float(ke_barrier):.1f}",
               f"{float(damping_delay_z):.2f}",
               f"{float(damping_impact_z):.2f}",
               f"{float(damping_retract_brake_z):.2f}",
               int(bool(reverse_interlock_active)),
               f"{float(reverse_interlock_gate):.4f}",
               int(bool(impact_danger)),
               f"{float(d_extra_target_z):.2f}",
               f"{float(f_err_raw):.4f}",
               f"{float(f_err_eff):.4f}",
               f"{float(v_force_raw):.5f}",
               f"{float(v_tcp_z_gate):.6f}",
               f"{float(tank_energy_j):.6f}",
               f"{float(tank_gamma):.4f}",
               f"{float(port_energy_j):.6f}",
               f"{float(port_excess_j):.6f}",
               f"{float(damping_pc_z):.2f}",
               int(bool(free_seek_active)),
               f"{float(wrist_relax_scale):.4f}",
               f"{float(u_dob_z):.4f}",
               int(bool(dob_frozen)),
               int(bool(suspect_recovery)),
               f"{float(v_tcp_z_filt):.6f}",
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
                        # Governor scales trajectory/FF only; force loop uses wall clock.
                        if hasattr(phase.outer, "set_time_scale"):
                            phase.outer.set_time_scale(scale)
                        sample_params = inspect.signature(phase.outer.sample).parameters
                        sample_kwargs: dict = {}
                        if "q_meas" in sample_params:
                            sample_kwargs["q_meas"] = q_meas
                        if "f_ext_raw" in sample_params and f_ext_raw is not None:
                            # Unfiltered wrench for Dimeas (LPF hides the band).
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
                        ctrl = getattr(phase.outer, "controller", None)
                        f_des_z = float(
                            getattr(ctrl, "f_des_z_eff", float("nan"))
                        ) if ctrl is not None else float("nan")
                        f_ext_z = (
                            float(f_ext[2])
                            if f_ext is not None and len(f_ext) > 2
                            else float("nan")
                        )
                        step = inner.update(
                            twist,
                            control_dt,
                            q_meas=q_meas,
                            qdot_ff=qdot_ff,
                            vel_ff=vel_ff_ref,
                            f_ext_z=f_ext_z if math.isfinite(f_ext_z) else None,
                            f_des_z=f_des_z if math.isfinite(f_des_z) else None,
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
                        # Near singularity, TCP lag is expected — do not starve
                        # the reference clock (hw: straight-elbow recovery stuck
                        # 7s at gov≈0.33 while track≈25–30 mm).
                        sigma_ref_gov = float(
                            inner.cfg.qp.sr_damping.sigma_ref
                        )
                        if (
                            sigma_ref_gov > 1e-9
                            and float(step.sigma_min) < sigma_ref_gov
                        ):
                            raw_scale = max(float(raw_scale), 0.55)
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
\n```\n\n## Key CSV head (`run_20260804_233408_key_ds10.csv`)\n\n```csv\nt_wall_s,phase,fz,v_force_z,v_force_raw,v_r_z,tank_gamma,tank_energy_j,damping_pc_z,port_energy_j,port_excess_j,cap_press_z,free_seek_active,damping_z_eff,damping_impact_z,impact_danger,reverse_interlock_active,physical_contact_state\n0.0006,movej->d,0.523,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.0506,movej->d,0.503,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.1006,movej->d,0.497,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.1506,movej->d,0.509,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.2006,movej->d,0.505,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.2506,movej->d,0.514,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.3006,movej->d,0.528,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.3506,movej->d,0.493,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.4006,movej->d,0.511,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.4506,movej->d,0.515,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.5006,movej->d,0.513,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.5506,movej->d,0.540,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.6006,movej->d,0.512,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.6506,movej->d,0.502,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.7006,movej->d,0.532,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.7506,movej->d,0.580,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.8006,movej->d,0.514,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.8506,movej->d,0.477,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.9006,movej->d,0.544,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n0.9506,movej->d,0.505,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.0006,movej->d,0.532,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.0506,movej->d,0.515,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.1006,movej->d,0.489,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.1506,movej->d,0.555,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.2006,movej->d,0.571,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.2506,movej->d,0.469,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.3006,movej->d,0.521,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.3506,movej->d,0.444,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.4006,movej->d,0.548,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.4506,movej->d,0.507,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.5006,movej->d,0.472,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.5506,movej->d,0.521,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.6006,movej->d,0.524,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.6506,movej->d,0.513,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.7006,movej->d,0.524,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.7506,movej->d,0.468,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.8006,movej->d,0.475,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.8506,movej->d,0.526,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.9006,movej->d,0.563,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n1.9506,movej->d,0.515,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n2.0006,movej->d,0.515,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n2.0506,movej->d,0.522,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,nan,nan,0,0,\n2.0990,scan,0.556,0.00300,0.00497,0.00000,1.0000,0.000978,0.00,0.001976,0.000000,0.003000,0,25.00,0.00,0,0,free\n2.1587,scan,0.385,0.00300,0.00573,0.00000,1.0000,0.000893,0.00,0.001874,0.000000,0.003000,0,25.00,0.00,0,0,free\n2.2160,scan,0.478,0.00300,0.00532,0.00000,1.0000,0.000808,0.00,0.001788,0.000000,0.003000,0,25.00,0.00,0,0,free\n2.2742,scan,0.550,0.00300,0.00500,0.00000,1.0000,0.000726,0.00,0.001732,0.000000,0.003000,0,25.00,0.00,0,0,free\n2.3345,scan,0.539,0.00300,0.00505,0.00000,1.0000,0.000647,0.00,0.001707,0.000000,0.003000,0,25.00,0.00,0,0,free\n2.3943,scan,0.548,0.00300,0.00501,0.00000,1.0000,0.000571,0.00,0.001611,0.000000,0.003000,0,25.00,0.00,0,0,free\n2.4538,scan,0.491,0.00300,0.00526,0.00000,1.0000,0.000489,0.00,0.001595,0.000000,0.003000,0,25.00,0.00,0,0,free\n2.5117,scan,0.534,0.00300,0.00507,0.00000,1.0000,0.000406,0.00,0.001561,0.000000,0.003000,0,25.00,0.00,0,0,free\n2.5793,scan,0.508,0.00300,0.00519,0.00000,1.0000,0.000321,0.00,0.001463,0.000000,0.003000,0,25.00,0.00,0,0,free\n2.6382,scan,0.527,0.00300,0.00510,0.00000,1.0000,0.000239,0.00,0.001353,0.000000,0.003000,0,25.00,0.00,0,0,free\n2.6966,scan,0.508,0.00300,0.00519,0.00000,1.0000,0.000153,0.00,0.001288,0.000000,0.003000,0,25.00,0.00,0,0,free\n2.7546,scan,0.529,0.00300,0.00509,0.00000,1.0000,0.000069,0.00,0.001258,0.000000,0.003000,0,25.00,0.00,0,0,free\n2.8166,scan,0.517,0.00267,0.00267,0.00000,0.0000,0.000000,0.00,0.001163,0.000000,0.003000,0,25.00,0.00,0,0,free\n2.8798,scan,0.473,0.00132,0.00132,0.00000,0.0000,0.000000,0.00,0.001107,0.000000,0.003000,0,25.00,0.00,0,0,free\n2.9414,scan,0.533,0.00177,0.00177,0.00000,0.0000,0.000000,0.00,0.001082,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.0014,scan,0.528,0.00267,0.00267,0.00000,0.0000,0.000000,0.00,0.001064,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.0619,scan,0.526,0.00182,0.00182,0.00000,0.0000,0.000000,0.00,0.001027,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.1206,scan,0.537,0.00179,0.00179,0.00000,0.0000,0.000000,0.00,0.000902,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.1797,scan,0.461,0.00167,0.00167,0.00000,0.0000,0.000000,0.00,0.000902,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.2395,scan,0.490,0.00300,0.00330,0.00000,0.4585,0.000000,0.00,0.000911,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.2979,scan,0.517,0.00092,0.00092,0.00000,0.0000,0.000000,0.00,0.000837,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.3583,scan,0.587,0.00155,0.00155,0.00000,0.0000,0.000000,0.00,0.000844,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.4181,scan,0.524,0.00245,0.00245,0.00000,0.0000,0.000000,0.00,0.000838,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.4768,scan,0.554,0.00121,0.00121,0.00000,0.0000,0.000000,0.00,0.000829,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.5356,scan,0.546,0.00199,0.00199,0.00000,0.0000,0.000000,0.00,0.000842,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.5960,scan,0.508,0.00157,0.00157,0.00000,0.0000,0.000000,0.00,0.000810,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.6544,scan,0.529,0.00150,0.00150,0.00000,0.0000,0.000000,0.00,0.000792,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.7216,scan,0.507,0.00046,0.00046,0.00000,0.0000,0.000000,0.00,0.000749,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.7792,scan,0.555,0.00211,0.00211,0.00000,0.0000,0.000000,0.00,0.000761,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.8397,scan,0.496,0.00118,0.00118,0.00000,0.0000,0.000000,0.00,0.000743,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.9091,scan,0.483,0.00165,0.00165,0.00000,0.0735,0.000000,0.00,0.000757,0.000000,0.003000,0,25.00,0.00,0,0,free\n3.9679,scan,0.518,0.00185,0.00185,0.00000,0.0000,0.000000,0.00,0.000757,0.000000,0.003000,0,25.00,0.00,0,0,free\n4.0253,scan,0.473,0.00147,0.00147,0.00000,0.0000,0.000000,0.00,0.000753,0.000000,0.003000,0,25.00,0.00,0,0,free\n4.0841,scan,0.584,0.00100,0.00100,0.00000,0.0000,0.000000,0.00,0.000742,0.000000,0.003000,0,25.00,0.00,0,0,free\n4.1456,scan,0.540,0.00128,0.00128,0.00000,0.0000,0.000000,0.00,0.000626,0.000000,0.003000,0,25.00,0.00,0,0,free\n4.2045,scan,0.541,0.00228,0.00228,0.00000,0.0000,0.000000,0.00,0.000639,0.000000,0.003000,0,25.00,0.00,0,0,free\n4.2644,scan,0.493,0.00182,0.00182,0.00000,0.3039,0.000000,0.00,0.000619,0.000000,0.003000,0,25.00,0.00,0,0,free\n```\n