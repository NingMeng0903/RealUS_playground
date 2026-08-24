# 8DOF nullspace 吸引子 — 设计 / 代码 / 扫描 CSV 快照

- 生成时间：2026-08-24 23:06:55
- 仓库根：`/media/camp/EXT_DRIVE/RealUS_playground`
- git HEAD：`6340184379a151aee5bf49ad3a77cde46e96c3f9`
- git branch：`codex/stage1-male-retarget-v4`
- 生产路径：`inner.backend: native`（`peirastic/configs/controller.yaml`）
- 触发 log：`python -m peirastic.apps.run_controller --log-csv`
- 原始 CSV：`rm75_control/apps/logs/peirastic/run_20260824_223425.csv`
- 本文档副本 CSV：`MD/todo_controller_logs/run_20260824_223425.csv`
- 吸引子列全时序：`MD/todo_controller_logs/run_20260824_223425_attractor.csv`
- 10 Hz 抽稀：`MD/todo_controller_logs/run_20260824_223425_attractor_10hz.csv`

原始 CSV 约 29 MB、11792 行、434 列，**不能整文件塞进 markdown**。全文已复制到 `MD/todo_controller_logs/`；下面是设计说明、完整吸引子源码，以及这次左右扫的数值结论。

缺失列（若有）：无

---

## 1. 吸引子实际钉的是什么

吸引子 **不是** 「J4 = 90°」这一条关节任务。yaml 里是一条侧卧族 `(d*, ψ*) → q*`（SRS IK），再加一个 **不能动导轨** 的关节向心。

| 量 | yaml / 代码 | 这次 log |
|---|---|---|
| 设计族 J4 | `q_nominal_deg[4] = 96.0°`（注释写 J4≈96°，不是 90°） | 全程 **20.9°–130°**，起步 129°，停住 ≈95° |
| SEW 转角 ψ* | `psi_attr_deg: 68` | ψ* 一直 68°；live ψ 从约 42° 锁到 68° |
| 伸出 d* | `d_attr_m: -0.185` | d* **没有冻在 −185 mm**，大约 −265…−68 mm |
| 肘带 | `elbow_lo/hi ≈ 70°/115°`，中心 95° | 两端贴墙后 J4 出带（21° / 130°） |
| 导轨向心权重 | `nullspace.weights[0] = 0` | J4 误差 **不能下令滑台去补** |
| 导轨 LPF | `rail_allocator.f_c_hz: 2.0` | 轨在跟 Y（corr≈0.98），但慢于手柄 Y |
| 停杆 secondary | C++ `if (quiescent_) sec.setZero()` | 停住后不会弹回 96° |

优先级（高→低）：

1. **QP1 TCP**（手柄世界 Y 瞬时最便宜的是 J4/J6）
2. `allocate_rail` + 2 Hz 低通 + `u_mid`（把滑台放到 `y_tcp − d*` 下面）
3. ψ 半平面 / 包络（这次生效了）
4. 同伦 `s` 上的 `(d*, ψ*, q*)`；`q*` 是 **当前 TCP 上的 SRS**，不是 t=0 的照片构型
5. 关节向心（`weights[0]=0`，且 idle 时 secondary 清零）

所以左右扫时 J4 跟着行程单调变、反向再变回来，是架构而不是「nullspace 没开」。冗余优先保证 TCP，不是钉死照片构型。守 J4≈96° 的条件是滑台 1:1 跟在 `y_tcp − d*` 下，且 TCP Y 留在固定 d*=−185 mm、可用轨约 3–76 cm 时大约 **72 cm** 的舒适带里。这次 Y 扫了约 **1.15 m**，两端贴墙后两边都会拧。

同伦还会 **改 d***（`select_d_for_elbow`）：轨慢时先改伸出量，肘照样被拉开。不要把 `d_center_rate` 提到和手柄同级。ψ 约束不要拆。

---

## 2. 这次扫描 CSV 数值

- 行数：11792
- `t_ref_s`：0.005 → 59.391 s
- `t_wall_s`：0.003 → 59.389 s（若远小于墙钟，用 `row_index × 0.005`）
- 估计时长：58.95 s
- J4 `q_cmd_4`：20.9° … 130.0°（起 129.4° / 止 94.6°）
- 滑台 `q_cmd_0`：0.005 … 0.780 m
- TCP Y：-0.301 … 0.849 m（跨度 1.150 m）
- d*：-0.265 … -0.068 m
- d_live = pose_y − rail：-0.307 … 0.069 m
- ψ：42.0 … 77.3°；ψ*：68.0 … 68.0°
- homotopy_s：0.00 … 1.00
- corr(J4, pose_y) = -0.933
- corr(pose_y, rail) = 0.983
- rail_motion_share 中位：0.703
- wall_active 占比：23.2 %
- u_mid 范围：-0.0156 … 0.0340 m/s
- u_alloc 范围：-0.1009 … 0.1153 m/s
- max |pose_d_y − pose_y| = 0.0000 m（live-Y 已跟 TCP，不是冻在原点）

原始 CSV sha256：`817b74bae9e0e9e9e1bae92f91a2c50e1ffe581caf301b69edff526576ebee17`
副本 sha256：`817b74bae9e0e9e9e1bae92f91a2c50e1ffe581caf301b69edff526576ebee17`

### 关键帧

| i | t_est_s | J4° | rail m | pose_y | d_live | d* | ψ | ψ* | s | share | u_mid | u_alloc | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.00 | 129.4 | 0.020 | -0.169 | -0.189 | -0.189 | 42.0 | 68.0 | 0.00 | 0.73 | 0.000 | 0.000 | 1 |
| 7158 | 35.79 | 22.8 | 0.779 | 0.849 | 0.069 | -0.096 | 68.0 | 68.0 | 1.00 | 0.18 | 0.000 | -0.055 | 1 |
| 9967 | 49.84 | 121.3 | 0.005 | -0.301 | -0.307 | -0.242 | 68.0 | 68.0 | 1.00 | 0.00 | 0.000 | -0.006 | 1 |
| 4078 | 20.39 | 130.0 | 0.162 | -0.068 | -0.230 | -0.146 | 68.0 | 68.0 | 1.00 | 0.90 | -0.009 | -0.082 | 0 |
| 2240 | 11.20 | 20.9 | 0.779 | 0.826 | 0.047 | -0.135 | 68.0 | 68.0 | 1.00 | 0.75 | 0.000 | 0.001 | 1 |
| 2288 | 11.44 | 22.0 | 0.780 | 0.825 | 0.046 | -0.135 | 68.0 | 68.0 | 0.00 | 0.09 | 0.000 | -0.012 | 1 |
| 10050 | 50.25 | 122.1 | 0.005 | -0.293 | -0.299 | -0.250 | 68.0 | 68.0 | 1.00 | 0.00 | 0.001 | -0.007 | 1 |
| 11791 | 58.95 | 94.6 | 0.098 | 0.011 | -0.087 | -0.087 | 68.0 | 68.0 | 0.39 | 0.50 | 0.000 | 0.000 | 0 |

含义：+Y 尽头轨贴 soft_max、d_live 甚至变号、J4 塌到约 20°；−Y 尽头轨贴 soft_min、d_live≈−30 cm、J4≈120°。轨在跟，但行程不够，同伦还在改 d*。

---

## 3. 文件索引（下面均为全文）

1. [`peirastic/configs/controller.yaml`](#1-peirasticconfigscontrolleryaml) — 25313 bytes, 824 lines, sha256 `aa543f7aa41f5280038cab1033e33292ef9ba591fed72e26ffbcf5afd6a1580e`
2. [`rm75_control/configs/joint_admittance_8dof.yaml`](#2-rm75_controlconfigsjoint_admittance_8dofyaml) — 25237 bytes, 823 lines, sha256 `14c58a9f57d66134f208f9e8287c81022956fc209a97c7ce7fcbb0f52bc8dd38`
3. [`rm75_control/rm75_control/control/joint_admittance_8dof/tasks/psi_retarget.py`](#3-rm75_controlrm75_controlcontroljoint_admittance_8doftaskspsi_retargetpy) — 39041 bytes, 1053 lines, sha256 `9720790933b2db8c54e2cc22c1b6ff25461fb91c20edd5a0b11947a0b31dcd7e`
4. [`rm75_control/rm75_control/control/joint_admittance_8dof/tasks/rail_allocator.py`](#4-rm75_controlrm75_controlcontroljoint_admittance_8doftasksrail_allocatorpy) — 16254 bytes, 484 lines, sha256 `4dfc442f40ff579dad3c9bf656aa0ef22273857a88ed8a7e7c4eee679cea757b`
5. [`rm75_control/rm75_control/control/joint_admittance_8dof/tasks/rail_extension.py`](#5-rm75_controlrm75_controlcontroljoint_admittance_8doftasksrail_extensionpy) — 33518 bytes, 818 lines, sha256 `7b4aa9103d9960b1f012fcdc3abc2db05dfd10b96baacd320701343203a98c05`
6. [`rm75_control/rm75_control/control/joint_admittance_8dof/tasks/secondary_composer.py`](#6-rm75_controlrm75_controlcontroljoint_admittance_8doftaskssecondary_composerpy) — 13190 bytes, 347 lines, sha256 `9bb6bfff31106a33a13ed66ccd75ff0f380ab8d11a6c19bc5e13fd808bd713e2`
7. [`rm75_control/rm75_control/control/joint_admittance_8dof/tasks/nullspace_task.py`](#7-rm75_controlrm75_controlcontroljoint_admittance_8doftasksnullspace_taskpy) — 4506 bytes, 102 lines, sha256 `8a409a2d25c451825f659d99a4def201456e2d18a5a789ebe81e0d0750ba69e0`
8. [`rm75_control/rm75_control/control/joint_admittance_8dof/tasks/arm_angle.py`](#8-rm75_controlrm75_controlcontroljoint_admittance_8doftasksarm_anglepy) — 11033 bytes, 236 lines, sha256 `5396029757b8733529d3684bba8959495f0102e50e35ec7292674c5073cf73f0`
9. [`rm75_control/tests/test_side_posture_attractor.py`](#9-rm75_controlteststest_side_posture_attractorpy) — 8953 bytes, 256 lines, sha256 `2534498aaab17bb0c67dab936347908f0044f6c39f00f5925e386683ceb154f8`
10. [`rm75_control/native/wbc_rt/include/wbc_rt/posture.hpp`](#10-rm75_controlnativewbc_rtincludewbc_rtposturehpp) — 3031 bytes, 80 lines, sha256 `d2d370271414b20c24f99dd5986edb0d3911326f27a265a98c11f0cfc0f02641`
11. [`rm75_control/native/wbc_rt/src/posture.cpp`](#11-rm75_controlnativewbc_rtsrcposturecpp) — 13092 bytes, 343 lines, sha256 `a8c48ca43aa1026792758a2749ac7827e05da3d0a0d88ae7ea82d56c3eec7d48`
12. [`rm75_control/native/wbc_rt/include/wbc_rt/inner.hpp`](#12-rm75_controlnativewbc_rtincludewbc_rtinnerhpp) — 6736 bytes, 234 lines, sha256 `5db6009cd8779d2c7190fe284d9ae9a0f7be571413110a9f57dda8d9b9adca1e`
13. [`rm75_control/native/wbc_rt/src/inner.cpp`](#13-rm75_controlnativewbc_rtsrcinnercpp) — 44266 bytes, 1182 lines, sha256 `08b400fa0ec2c15179adbc5159555277657116a105c682e8cb3f91014cc66c23`
14. [`rm75_control/native/wbc_rt/include/wbc_rt/config.hpp`](#14-rm75_controlnativewbc_rtincludewbc_rtconfighpp) — 6104 bytes, 210 lines, sha256 `6f58d62e253d95aa4628db1d3e3005e777788c9cf120366e07d710c4e7cbf6d9`
15. [`rm75_control/native/wbc_rt/include/wbc_rt/protocol.hpp`](#15-rm75_controlnativewbc_rtincludewbc_rtprotocolhpp) — 3296 bytes, 165 lines, sha256 `186ac815f6305575bd3fe65712b1e6417d7ec2fc8b35cea673f549754ec89f66`
16. `loop.py` 吸引子相关摘录（全文太大）

---

## 1. `peirastic/configs/controller.yaml`

- sha256：`aa543f7aa41f5280038cab1033e33292ef9ba591fed72e26ffbcf5afd6a1580e`
- 行数：824

```yaml
# peirastic Window A machine config — peirastic/configs/controller.yaml
# Inner / rail / robot. Force-axis law is peirastic/configs/force.yaml (loaded separately).
#
# URDF: rm75_control/assets/robots/rm75_6f_8dof/RM75-6F-8dof.urdf
# Genesis viz: python -m rm75_control.control.joint_admittance_8dof.viewer.demo --show-viewer
# Param spec: joint_admittance_8dof/config/slider_rail.yaml (default viewer scene)

robot:
  ip: "192.168.1.18"
  port: 8080
  thread_mode: 2

timing:
  # 5.0 ms target.  t_ref advances by wall time; integration clips a late
  # tick to [dt_nom, 1.25*dt_nom].  If deadline_slack_s > 0 on <99% of
  # ticks, raise this back to 7.0.
  dt_ms: 5.0
  # Post-solve gate re-reads UDP; 80 ms still fails closed on a true push gap.
  feedback_timeout_ms: 80.0
  # Consecutive rejected/stale feedback before abort.  One hitch coasts.
  feedback_coast_ms: 300.0
  rt_disable_gc: true
  verbose_json: false
  # Best-effort RT: pin the control thread; hold /dev/cpu_dma_latency at 0.
  control_cpu: 2
  disable_cstates: true

# UDP arm-state push (rm_set_realtime_push). Requires robot.thread_mode: 2.
realtime_push:
  cycle: 1              # broadcast period = cycle * 5 ms (1 -> 200 Hz)
  port: 8098
  ip: "192.168.1.80"    # PC NIC on robot subnet — do not auto-detect on multi-NIC hosts
  force_coordinate: 0   # 0=sensor frame (matches rm_get_force_data force_data)

# Shared-memory state relay for split-process Genesis twin (same host).
# Match realtime_push (cycle=1 -> 200 Hz) so attach-mode WBC does not stair-step.
# Window B (`run_with_twin.py`) only mirrors hardware if this is on.
state_relay:
  enabled: true
  name: rm75_state
  hz: 200

# Slack-QP inner loop (Escande). Physical q/v/a/collision live under hard_limits;
# Cartesian/nullspace/rail-extension tuning lives under inner.
qpik:
  hard_limits:
    v_scale: 0.8
    a_max_arm_rad_s2: 3.0
    a_max_rail_m_s2: 0.60
    position_margin_deg: 0.3
    position_margin_rail_mm: 0.0
    command_lead_arm_deg: 6.0
    command_lead_rail_mm: 20.0
    velocity_damper:
      arm_band_rad: 0.25
      rail_band_m: 0.025
    collision:
      enabled: true
      d_safe: 0.01
      d_activate: 0.04
      gamma: 5.0
      max_pairs: 8
    rail:
      mode: coupled
      locked_style: hold
      lock_vel_eps_m_s: 0.0
      v_max_m_s: 0.15
      travel_m: 0.80
      # Linear taper inner edge (must match rail_band_m).  Stick-speed
      # braking is the stopping envelope, not a step at this line.
      soft_min_m: 0.030
      soft_max_m: 0.755
      # QP / servo box.  5–780 mm is the full travel; 780 is reachable.
      hard_min_m: 0.005
      hard_max_m: 0.78

inner:
  # python = in-process QPIK. native = separate wbc_rt process (SHM).
  backend: native
  native_bin: /media/camp/EXT_DRIVE/RealUS_playground/rm75_control/native/wbc_rt/build/wbc_rt
  control_frame: tool
  euler_order: xyz
  # Sync RealMan active tool into Pinocchio link_7→tcp (force-hybrid / tool-Z).
  sync_tcp_from_robot: true

  qp:
    task_weight: [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
    reg: [1.0e-3, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.2e-2, 1.2e-2, 1.2e-2]
    backend: proxqp
    use_cpp_kernel: true
    eps_abs: 1.0e-6
    max_iter: 400
    max_iter_cap: 400
    max_solve_ms: 5.0
    fail_qdot_decay: 0.85
    twist_sigma_floor: 0.02
    warn_on_fail: false
    sr_damping:
      lam0: 0.05
      sigma_ref: 0.08
      sigma_floor: 1.0e-6
    task_weight_min_frac: 0.05
    task_weight_lpf_tau_s: 0.25
    aniso_task_damping: true
    use_mass_weighted_reg: true
    mass_reg_floor: 0.05
    mass_weight_exempt_rail: true
    mass_reg_lpf_tau_s: 0.2
    limit_damper_band_rad: 0.25
    limit_damper_band_rail_m: 0.025
    near_arm_margin_rad: 0.08
    # Rail continuity is the hard a/j box + macro filter, not a soft glue term.
    smoothness_weight: [0.0, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    # Third-order box: bounds how fast the commanded acceleration may turn.
    # Measured jerk RMS was 250-570 rad/s³ with the acceleration flipping sign
    # on ~half the ticks while the reference twist was smooth.
    j_max_arm_rad_s3: 300.0
    # 3.0 made a full rail acceleration reversal take 2*a_max/j = 0.2 s versus
    # 0.02 s on the arm; 60 keeps 2*a_max/j = 0.02 s after a_max rose to 0.60.
    j_max_rail_m_s3: 60.0
    sigma_setbased:
      enabled: true
      activate: 0.12
      safe: 0.06
      exit: 0.16
      gamma: 8.0
      slack_weight: 200.0
      grad_period_ticks: 10
    branch_barrier:
      enabled: true
      # Soft preference at 30°.  The hard velocity box starts at 50° so
      # J4 cannot blast through 0 when QP1 is holding TCP (035411).
      activate_rad: 0.52
      box_activate_rad: 0.87
      eps_rad: 0.35
      # J4 ±135° damper (open travel only).  Do not reuse eps_rad=20°
      # or the upper wall sits at 115° and vertical press dies.
      j4_limit_eps_rad: 0.08726646259971647   # 5° → zero at ~130°
      j4_limit_activate_rad: 0.4363323129985824  # 25° → taper from ~110°
      # J1 same-sign over-fold.  Zero at 140°; 0→−90° startup stays free.
      j1_overfold_abs_rad: 2.443460952792061   # 140°
      j1_overfold_activate_rad: 0.4363323129985824  # 25° → taper from ~115°
      gamma: 6.0
      slack_weight: 80.0
      dwell_free_s: 0.3
      dwell_ramp_s: 1.0
      dwell_scale_max: 5.0
    # Moe/Kanoun set-based comfort: each arm joint, own slack, 15–25° band.
    joint_comfort:
      enabled: true
      m_comfort_deg: 15.0
      activate_deg: 25.0
      gamma: 6.0
      slack_weight: 80.0

  collision:
    enabled: true
    d_safe: 0.01
    d_activate: 0.04
    gamma: 5.0
    max_pairs: 8

  nullspace:
    k_center: 1.0
    k_limit: 2.0
    activation: 0.75
    weights: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # Side-lying family (Pin–Culioli minmax on the photo seed): ψ=68°,
    # d=−0.185, J4≈96°.  Same branch as the taught lean; the 104° seed
    # already hit 124° under a typical Δx/roll shake.  Signs are fixed.
    q_nominal_deg: [0.0, -89.5, -94.5, 65.2, 96.0, 89.3, 61.0, 94.6]
    manipulability:
      k_mu: 0.8
      eps_rad: 5.0e-4
      sigma_fade_ref: 0.12
      grad_period_ticks: 10
      qdot_tau_s: 0.05

  nullspace_d_null: 0.5
  nullspace_d_null_adaptive: 1.0
  nullspace_max_qdot_frac: 0.2
  # Continuous nullspace fade from task slack.  Not a binary latch.
  saturation:
    slack_enter: 0.15
    slack_exit: 0.03
    secondary_scale: 0.15
    secondary_scale_tau_s: 0.10

  arm_angle:
    enabled: true
    k_psi: 1.5
    psi_ref_deg: null
    obs_smooth_floor: 0.3

  psi_retarget:
    enabled: true
    n_y: 9
    n_d: 8
    n_psi: 9
    w_sigma: 0.5
    # Band around the 60° attractor — not |q6|/128°, which sacrificed J2.
    w_wrist: 0.5
    margin_floor_deg: 20.0
    psi_rate_deg_s: 25.0
    # Design split: J4≈96° (band center 95°).  Unplanned step shares one
    # progress s across (d*, ψ*, q*); q* is srs_ik at the live TCP, not
    # the yaml photo at t=0.
    psi_attr_deg: 68.0
    d_attr_m: -0.185
    d_center_rate_m_s: 0.02
    psi_cmd_lead_deg: 18.0
    psi_replan_period_s: 0.1
    psi_search_half_span_deg: 45.0
    psi_search_n: 9
    psi_wrist_ok_deg: 40.0
    psi_return_dwell_s: 1.0
    # >110° collapses J6 on this family.  Never cross ψ=0.
    psi_envelope_deg: [40.0, 110.0]
    require_design_family: false
    rail_margin_m: 0.02
    # Reject cells whose wrist sits on the ~20° branch-barrier floor.
    wrist_min_deg: 30.0

  # Signed IRD field (ird_playground).  One-shot d* at plan_scan_stroke only.
  # Hot-path RailGoodness is σ_min (autograd IRD caused 127 ms hitches).
  # Queries rebuild probe45 TCP from link_7 so gripper2 is ok.
  ird:
    enabled: true
    device: cpu
    allow_stale: true

  rail_extension:
    enabled: true
    k_ext: 2.0
    k_ff: 1.0
    v_ff_thr_m_s: 0.005
    v_ff_span_m_s: 0.015
    e0_m: 0.02
    e1_m: 0.08
    w_max: 2.0
    v_max_m_s: 0.08
    limit_margin_m: 0.15
    pin_margin_m: 0.008
    escape_leave_m: 0.04
    healthy_sigma_mute: 0.08
    press_v_force_min_m_s: 0.02
    press_dz_max_m: 0.002
    press_y_err_m: 0.005
    press_stall_s: 0.5
    d_band_m: 0.005
    k_sigma_boost: 2.0
    k_esc: 0.5
    w_sigma_floor: 1.0
    k_pose: 2.0
    pose_e0_m: 0.005
    pose_e1_m: 0.04
    pose_w_max: 4.0
    sigma_guard_enter: 0.45
    sigma_guard_exit: 0.70
    v_guard_max_m_s: 0.04
    v_lpf_tau_s: 0.05
    v_lpf_fc_hz: 5.0
    v_lpf_tau_escape_s: 0.04
    # Narrow latch: deep σ or true near-limit only (healthy = FF + allocator).
    sigma_escape_enter: 0.55
    sigma_escape_exit: 0.80
    margin_escape_enter: 0.12
    margin_escape_exit: 0.25
    sigma_drop_rate: 0.0
    escape_enter_dwell_s: 0.05
    k_escape_boost: 1.2
    escape_grad_floor: 0.0
    k_margin_boost: 4.0
    w_ext_cap: 24.0
    d_star_err0_m: 0.01
    d_star_err1_m: 0.04
    d_star_w_mult: 6.0
    d_star_reg_mult: 20.0
    # Escape sign: auto = open travel / σ gradient.  minus/plus force a side.
    escape_sign_policy: auto

  rail_allocator:
    v0_m_s: 0.05
    w0_rad_s: 0.30
    k_margin: 4.0
    kp_mid: 1.2
    ki_mid: 0.80
    u_mid_max_m_s: 0.12
    k_err_rail: 4.0
    e_ref_m: 0.08
    # Rail velocity LPF after the live-Y e_mid fix. 2 Hz follows Y faster than
    # 1 Hz without the 5–10 Hz hunting the reversal tests guard against.
    f_c_hz: 2.0
    kaw_mid: 8.0
    rho_mirror_a: 0.50
    rho_mirror_j: 0.30
    reaction_s: 0.06
    observer_pos_gain: 0.35
    observer_vel_gain: 2.0
    observer_vel_lpf_hz: 8.0


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

# Outer-loop Cartesian P for CARTESIAN_TRACK / GOTO.  CLI --move-kp
# overrides k_task_lin only.  Rotation stays here.
# fb_lpf_tau_s filters k*e only; vel_ff is never filtered.  0 = off.
cartesian_track:
  k_task_lin: 12.0
  k_task_rot: 2.0
  max_pos_err_m: 0.05
  max_rot_err_rad: 0.35
  fb_lpf_tau_s: 0.0

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
  - 1
  - 1
  - 1
  - 1
  kp_pos:
  - 10.0
  - 10.0
  - 5.0
  - 1.5
  - 1.5
  - 1.5
  pos_err_deadband_m: 0.0005
  pos_correction_max_m_s: 0.08
  system_delay_s: 0.055
  contact_threshold_n: 0.8
  contact_use_fz_only: true
  physical_contact:
    enabled: true
    # Initial acquire uses filtered force only.  Replaying 162413 with
    # 0.85 N / 20 ms moves the false 3.49 s acquire to the stable load at
    # 4.14 s, while remaining reachable below the shipped 1 N target.
    enter_n: 0.85
    hard_enter_n: 1.5
    # Force task may stay armed across a bounce.  Physical CONTACT must
    # still be allowed to go LOST, or 80 mm/s press reopens in air.
    hold_until_reset: false
    exit_n: 0.70
    enter_confirm_s: 0.02
    exit_confirm_s: 0.1
  # Slightly wider band: blunt single-tick contact dips without sticky D.
  deadband_n: 0.08
  deadband_width_n: 0.10
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
  # 022208: continuous damping for the residual 2.73 Hz base-admittance mode.
  # Chatter: short-lived ΔD_hf(Is). Steady offset: force_dob. Not sticky Ke·D.
  admittance_mass_z: 1.0
  admittance_damping_z: 40.0
  max_vz_tool_m_s: 0.08
  desired_force_ramp_s: 0.30
  var_damping_enabled: true
  var_damping_omega_c_hz: 2.5
  var_damping_lambda: 0.951
  var_damping_f_max_n: 7.0
  # Dimeas Iₛ raises virtual mass briefly; ΔD_hf authority is off.
  var_damping_d_u: 0.0
  var_damping_m_u: 0.0
  var_damping_m_max: 3.0
  var_damping_dc_alpha: 0.02
  var_damping_hf_attack_s: 0.02
  var_damping_hf_hold_s: 0.08
  var_damping_hf_release_s: 0.12
  var_damping_hf_release_fast_s: 0.04  # dump ΔD on hand-release / large |e_f|
  var_damping_hf_on: 0.30
  var_damping_hf_off: 0.15
  var_damping_hf_err_n: 0.8
  recontact_vz_cap_m_s: 0.012
  # Safety recontact speed latches on contact *loss* / suspect_loss, not on
  # reacquire.  hold / episode_release only manage estimator episodes.
  recontact_hold_s: 0.80
  recontact_settle_m_s: 0.003
  recontact_settle_hold_s: 0.050
  contact_episode_release_s: 0.80
  contact_episode_release_force_n: 0.75
  # Restored from e85c9ab.  Steady under-force offset rejection; 1bfe98b
  # disabled it as part of the anti-bounce sweep, and the force barrier below
  # now provides that brake instead.
  force_dob:
    enabled: false
    ki: 8.0
    leak_s: 0.4
    u_max_n: 1.5
    freeze_is: 0.45
    reset_on_reversal: true
  # Contact impact is limited before BEFM/tank intervention.  In free space
  # this preserves the 80 mm/s approach; after contact F+Fdot*T and the
  # stiffness estimate continuously tighten positive press speed.
  force_barrier:
    enabled: true
    t_react_s: 0.055
    budget_min_n: 1.0
    budget_frac: 0.20
    f_keep_n: 0.5
    f_escape_n: 0.5
    v_ref_m_s: 0.08
    v_min_retract_m_s: 0.0
    v_min_press_m_s: 0.0
    v_underforce_press_m_s: 0.010
    underforce_band_n: 0.20
    # Extra clip only.  First contact is K_ub T_stop, not this number.
    v_seek_free_m_s: 0.020
    tau_stop_s: 0.080
    e_x_m: 0.0004
    e_f_n: 0.20
    bar_f_n: 0.15
    fdot_lpf_s: 0.040
    precontact_raw_trigger_n: 1.50 # short impact sleeve; never latches contact
    stiffness_cap_enabled: true
    ke_floor_n_m: 50.0
    mass_floor_kg: 0.05
  # Independent chirps 031234/031605 select T0=30 ms, Tp=12 ms.  Do not
  # invert Γd or claim Y=A when any downstream constraint is active.
  cdyob:
    mode: shadow
    omega_q_hz: 0.75
    t0_s: 0.030
    tp_s: 0.012
    tau_s: 0.030
    t_n_s: 0.012
    pn_m: 0.0
    v_corr_max_m_s: 0.003
    blend_s: 0.30
    active_press_max_m_s: 0.010
    active_retract_max_m_s: 0.010
    active_q_max_hz: 1.0
    active_force_ratio: 0.90
    active_settle_speed_m_s: 0.003
    active_settle_hold_s: 0.20
    # 032041 A-only contact shadow passes polarity/clip + settled-apply gates.
    active_model_validated: true
  # Normal-port delay-aware shield. Stay in observe until the same solver
  # is feasible on hardware. force/observe are empirical peak guards:
  # require_contact_free_terminal is false until a release model exists.
  # Do not enable passive/ospf/CDYOB from this file.
  safety_shield:
    enabled: true
    mode: observe
    terminal_invariance_proven: false
    energy_sign_verified: false
    # Certificate domain.  Declared without numeric bounds is still false.
    pose_domain_declared: false
    payload_domain_declared: false
    max_feedback_age_s: 0.015
    k_ub_n_m: 8000.0
    recovery_hold_s: 0.050
    require_contact_free_terminal: false
    r_f_n_s: 8.0
    r_f_window_steps: 20
    f_release_n: 0.70
    u_retract_m_s: 0.040
    a_max_m_s2: 1.20
    j_max_m_s3: 40.0
    rho: 0.15
    e0_j: 0.004
    eps_j: 0.0005
    # One-pose free-space fit from run_20260821_000011 identify tail.
    # Not a passivity certificate; another pose/load still needed.
    plant:
      t0_s: 0.050
      tp_s: 0.020
      horizon_steps: 40
    # stop_dx_ub is written by --analyze-stop --write-yaml.
    # certified: false until 200 Hz motion-SHM backup replay + independent
    # val covers every stop event.  Tail lookup uses the worst successor
    # (v_{1,q}, a_{1,q}), not the nominal (v̂_1, â_1).

    velocity_error_ub_m_s:
    - 0.002810
    - 0.002800
    - 0.002800
    - 0.002410
    - 0.002800
    - 0.002820
    - 0.002822
    - 0.002820
    - 0.002800
    - 0.002834
    - 0.002810
    - 0.002800
    - 0.002800
    - 0.002410
    - 0.002800
    - 0.002820
    - 0.002820
    - 0.002829
    - 0.002800
    - 0.002800
    - 0.002810
    - 0.002800
    - 0.002800
    - 0.002410
    - 0.002800
    - 0.002820
    - 0.002820
    - 0.002820
    - 0.002800
    - 0.002800
    - 0.002810
    - 0.002800
    - 0.002800
    - 0.002410
    - 0.002800
    - 0.002820
    - 0.002820
    - 0.002820
    - 0.002800
    - 0.002800
    position_error_ub_m:
    - 0.0000140
    - 0.0000280
    - 0.0000420
    - 0.0000541
    - 0.0000681
    - 0.0000822
    - 0.0000963
    - 0.0001104
    - 0.0001244
    - 0.0001386
    - 0.0001526
    - 0.0001666
    - 0.0001806
    - 0.0001927
    - 0.0002067
    - 0.0002208
    - 0.0002349
    - 0.0002490
    - 0.0002630
    - 0.0002770
    - 0.0002911
    - 0.0003051
    - 0.0003191
    - 0.0003311
    - 0.0003451
    - 0.0003592
    - 0.0003733
    - 0.0003874
    - 0.0004014
    - 0.0004154
    - 0.0004295
    - 0.0004435
    - 0.0004575
    - 0.0004695
    - 0.0004835
    - 0.0004976
    - 0.0005117
    - 0.0005258
    - 0.0005398
    - 0.0005538
  # Historical keys. Force-axis a/j now live inside the shield; these
  # values are not applied after u_sent.
  force_axis_slew_press_m_s2: 1.20
  force_axis_slew_retract_m_s2: 1.20
  force_axis_slew_reverse_m_s2: 2.00
  force_axis_jerk_max_m_s3: 40.0
  # Lee-structure speed-level engineering adapter.  Observe is deliberately
  # non-mutating until the slow press/retract sign check and 2/5/10 mm/s
  # no-contact delay identification have been recorded.
  bidirectional_flow:
    mode: observe
    normal_sign: 1.0
    sign_verified: false
    feedback_delay_verified: false
    require_sign_verification: true
    require_delay_verification: true
    # Lee Sec. V-C: alpha is zero in free space.  Below this |fz| the gate is
    # held off and the tank charges from proxy damping.
    free_space_force_n: 0.5
    Dtrack: 25.0
    Kd: 25.0
    Kp: 250.0              # Dtrack / 0.10 s
    Ki: 0.0
    lambda_gain: 0.25
    track_correction_max_m_s: 0.020
    M_p: 1.0
    D_p: 25.0
    # Provisional conservative auxiliary values; retune only after the
    # velocity-step identification.  This branch can hold/retract, never press.
    M_a: 0.05
    D_a: 5.0
    K_a: 50.0
    B_a: 5.0
    u_retract_n: 0.0
    aux_max_retract_m_s: 0.050
    alpha_attack_s: 0.020
    alpha_release_s: 0.150
    max_feedback_age_s: 0.020
    T0: 0.0010
    Tmax: 0.0040
    Tmin: 0.0001
    mu_power_w: 0.0
    positive_switching_cost_j: 0.0
  # Optional Piedra-style elastic-surface force reduction.  Disabled until
  # stable-contact hardware validation; it is not a passivity guarantee.
  surface_force_modulation:
    enabled: false
    min_force_scale: 0.25
    beta_per_m: 80.0
    stable_contact_s: 0.20
    attack_s: 0.05
    release_s: 0.15
  # Soften under-force chase near scan turnaround (tool-XY slow).
  force_lateral_soft_m_s: 0.006
  force_lateral_full_m_s: 0.018
  force_lateral_gain_floor: 0.35
  adaptive_ke:
    enabled: true
    # Observe Ke / impact burst only — do not hold high critical D in steady contact.
    drive_damping: false
    zeta: 0.9
    ke_initial: 80.0
    ke_min: 40.0
    ke_max: 2500.0
    ke_impact_initial: 0.0
    ke_cap_ub_n_m: 2000.0
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
    bd_max: 180.0
    bd_slew_max: 400.0
    ke_slew_max: 1200.0
  proactive_feedforward: false
  # Legacy parameters are retained for isolated non-CDYOB tests only.
  proactive_retract_only: false
  # 014140 showed why this inactive v_r branch must not contaminate shadow.
  proactive_gain: 0.06
  proactive_retract_gain: 0.06
  proactive_leak_s: 0.50
  v_r_max_m_s: 0.02
  proactive_gate_press_on_is: false
  proactive_press_is_gate_start: 0.2
  proactive_press_is_gate: 0.6
  proactive_press_is_soft_floor: 0.45
  proactive_press_is_soft_stop: 0.85
  proactive_press_slew_max_m_s2: 0.35
  proactive_retract_slew_max_m_s2: 0.35
  proactive_press_drive_max: 1.4
  proactive_retract_drive_max: 1.4
  proactive_reset_on_reversal: true
  proactive_in_band_n: 0.08
  proactive_in_band_leak_s: 0.12
  force_scale_min_n: 0.18
  force_scale_fraction: 0.0
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
hw:
  lw100:
    enabled: true
    host: 192.168.0.7
    port: 8234
    slave: 1
    lead_mm: 10.0
    # calibrated_file: load var/lw100_rail_zero.json (run apps/lw100_rail_home_limit.py first).
    # current/fixed are debug-only; with require_calibration true, current is refused.
    zero_mode: calibrated_file
    counts0: 0
    calibration_path: var/lw100_rail_zero.json
    require_calibration: true
    home_di: di4             # −Y home switch (confirmed on HW; was swapped vs DI3)
    plus_di: di3             # +Y end (run-time e-stop if hit)
    di_nc: true
    di_debounce_n: 3
    soft_min_m: 0.030        # full-speed inner edge; must match qpik.hard_limits.rail
    soft_max_m: 0.755
    hard_min_m: 0.005        # travel box 5–780 mm
    hard_max_m: 0.78
    post_home_m: 0.025
    # Home-script only (controller does not auto-home on start):
    home_search_m_s: 0.020
    home_creep_m_s: 0.003
    home_backoff_mm: 5.0
    home_touch_count: 3
    home_search_timeout_s: 60.0
    home_to_post_m_s: 0.030
    limit_poll_every: 5
    # Host rail_y ↔ motor: -1 flips FA24 RPM (+ encoder map in rail_servo).
    sign: -1
    enable_settle_s: 0.3
    # Cold start: prove worker Modbus read+FA24=0 before any set_target / move→D.
    arm_good_reads: 30          # ~0.6 s @ 50 Hz consecutive healthy polls
    arm_settle_s: 0.8           # extra FA24=0 hold after good polls
    arm_max_span_mm: 2.0
    arm_timeout_s: 10.0
    fault_margin_m: 0.05
    # 205605: t_read med 5.8 ms, FA24 write usually skipped.  43 Hz left
    # ~17 ms of sleep.  60 Hz (16.7 ms) still has ~11 ms median headroom;
    # 80 Hz is tight on p95.  Do not change FA72 or the USR-TCP232-304 baud
    # alone — both ends must match, then power-cycle the drive.  Confirm no
    # DI is mapped to 7 (ZCLAMP) and FC-15 bit6 is 0; do not query while A
    # is live.  FA40/FA41 = time 0→1000 r/min, not 0→vel_max.
    poll_hz: 60
    inter_frame_delay_s: 0.0005
    timeout_s: 0.06             # poll-budget; was 0.15 / class-default 1.0
    retries: 1
    deadband_mm: 0.5
    # FA23 overspeed trip: must sit ABOVE commanded peak (0.15 m/s ≈ 900 rpm
    # @ 10 mm/rev). Equal FA23=cmd caused Er-01 on scan overshoot (151334).
    max_speed_rpm: 1200
    # Soft CSP via FA24 (see apps/lw100_vel_pos_follow_demo.py).
    # POSITION scan/home PD.  Coupled QPIK is pure velocity (no FA24 P).
    # Loaded PD scan BEST (400±40 mm): kp=14/kd=0.22.
    vel_kp: 14.0
    vel_kd: 0.22
    vel_kd_max_m_s: 0.005
    # Matches QP box rail.v_max_m_s 0.15 × v_scale 0.8.  parse_rail_servo_config
    # also caps hw.vel_max by that product so the two cannot drift apart.
    vel_max_m_s: 0.12
    vel_amax_m_s2: 1.2
    # Coupled-mode catch-up of x_ref toward x_goal while moving.  20 mm/s
    # clears a 20 mm standing offset in ~1 s without outrunning FF.
    catch_v_max_m_s: 0.02
    catch_k: 5.0
    catch_frac: 0.3
    decel_request_margin_m_s: 0.005
    match_drive_accel: true
    fa24_rpm_deadband: 0   # write every worker tick; skip only if rpm is unchanged
    vel_deadband_mm: 0.05   # tight tracking band (not a permanent accuracy sacrifice)
    # Standstill hysteresis: freeze FA24 after tight settle; wake only if pushed.
    standstill_enter_mm: 0.05
    standstill_exit_mm: 0.25
    standstill_dwell_s: 0.08
    approach_m: 0.040
    latch_watch_s: 0.12
    target_timeout_s: 0.25
    # Extra coast after timeout before FA24=0.  A 127 ms hitch must not brake.
    target_stale_coast_s: 0.35
    encoder_freeze_s: 1.0
    encoder_freeze_min_v_m_s: 0.02
    encoder_freeze_min_move_mm: 0.15
    # End-of-stream / task-end settle before releasing follow (closes latched overshoot).
    settle_tol_mm: 0.05
    settle_v_m_s: 0.006
    settle_timeout_s: 1.5
    # Stall-safe speed: worst-case latched FA24 overshoot ≤ |err| for max_stall_s.
    max_stall_s: 0.06
    stall_v_floor_m_s: 0.004
    # Soft-reject above v_max·gap + margin; wipe cal only on ≥50 mm or 2 soft jumps.
    jump_margin_mm: 3.0
    jump_hard_mm: 50.0
    jump_soft_streak_panic: 2
    # FA40/41: 120 ms → drive a ≈ 1.0 m/s².  Host a_max is min(this, QP
    # a_max_rail_m_s2 0.60, 0.85 × vel_max/accel_s) so the servo cannot
    # outrun the QP model.
    accel_ms: 120
    decel_ms: 120
    scurve_ms: 30            # FA42
    busy_speed_rpm: 1
    home_on_exit: false
    release_son_on_exit: false  # stop with FA24=0 and keep SON; avoids enable-edge frame wipe
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
  # Control-loop stall timeout.  QP backend pulses the watchdog during ProxQP.
  watchdog_timeout_s: 0.50
```

## 2. `rm75_control/configs/joint_admittance_8dof.yaml`

- sha256：`14c58a9f57d66134f208f9e8287c81022956fc209a97c7ce7fcbb0f52bc8dd38`
- 行数：823

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
  # 5.0 ms target.  t_ref advances by wall time; integration clips a late
  # tick to [dt_nom, 1.25*dt_nom].  If deadline_slack_s > 0 on <99% of
  # ticks, raise this back to 7.0.
  dt_ms: 5.0
  # Post-solve gate re-reads UDP; 80 ms still fails closed on a true push gap.
  feedback_timeout_ms: 80.0
  # Consecutive rejected/stale feedback before abort.  One hitch coasts.
  feedback_coast_ms: 300.0
  rt_disable_gc: true
  verbose_json: false
  # Best-effort RT: pin the control thread; hold /dev/cpu_dma_latency at 0.
  control_cpu: 2
  disable_cstates: true

# UDP arm-state push (rm_set_realtime_push). Requires robot.thread_mode: 2.
realtime_push:
  cycle: 1              # broadcast period = cycle * 5 ms (1 -> 200 Hz)
  port: 8098
  ip: "192.168.1.80"    # PC NIC on robot subnet — do not auto-detect on multi-NIC hosts
  force_coordinate: 0   # 0=sensor frame (matches rm_get_force_data force_data)

# Shared-memory state relay for split-process Genesis twin (same host).
# Match realtime_push (cycle=1 -> 200 Hz) so attach-mode WBC does not stair-step.
# Window B (`run_with_twin.py`) only mirrors hardware if this is on.
state_relay:
  enabled: true
  name: rm75_state
  hz: 200

# Slack-QP inner loop (Escande). Physical q/v/a/collision live under hard_limits;
# Cartesian/nullspace/rail-extension tuning lives under inner.
qpik:
  hard_limits:
    v_scale: 0.8
    a_max_arm_rad_s2: 3.0
    a_max_rail_m_s2: 0.60
    position_margin_deg: 0.3
    position_margin_rail_mm: 0.0
    command_lead_arm_deg: 6.0
    command_lead_rail_mm: 20.0
    velocity_damper:
      arm_band_rad: 0.25
      rail_band_m: 0.025
    collision:
      enabled: true
      d_safe: 0.01
      d_activate: 0.04
      gamma: 5.0
      max_pairs: 8
    rail:
      mode: coupled
      locked_style: hold
      lock_vel_eps_m_s: 0.0
      v_max_m_s: 0.15
      travel_m: 0.80
      # Linear taper inner edge (must match rail_band_m).  Stick-speed
      # braking is the stopping envelope, not a step at this line.
      soft_min_m: 0.030
      soft_max_m: 0.755
      # QP / servo box.  5–780 mm is the full travel; 780 is reachable.
      hard_min_m: 0.005
      hard_max_m: 0.78

inner:
  # python = in-process QPIK. native = separate wbc_rt process (SHM).
  backend: native
  native_bin: /media/camp/EXT_DRIVE/RealUS_playground/rm75_control/native/wbc_rt/build/wbc_rt
  control_frame: tool
  euler_order: xyz
  # Sync RealMan active tool into Pinocchio link_7→tcp (force-hybrid / tool-Z).
  sync_tcp_from_robot: true

  qp:
    task_weight: [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
    reg: [1.0e-3, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 1.2e-2, 1.2e-2, 1.2e-2]
    backend: proxqp
    use_cpp_kernel: true
    eps_abs: 1.0e-6
    max_iter: 400
    max_iter_cap: 400
    max_solve_ms: 5.0
    fail_qdot_decay: 0.85
    twist_sigma_floor: 0.02
    warn_on_fail: false
    sr_damping:
      lam0: 0.05
      sigma_ref: 0.08
      sigma_floor: 1.0e-6
    task_weight_min_frac: 0.05
    task_weight_lpf_tau_s: 0.25
    aniso_task_damping: true
    use_mass_weighted_reg: true
    mass_reg_floor: 0.05
    mass_weight_exempt_rail: true
    mass_reg_lpf_tau_s: 0.2
    limit_damper_band_rad: 0.25
    limit_damper_band_rail_m: 0.025
    near_arm_margin_rad: 0.08
    # Rail continuity is the hard a/j box + macro filter, not a soft glue term.
    smoothness_weight: [0.0, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    # Third-order box: bounds how fast the commanded acceleration may turn.
    # Measured jerk RMS was 250-570 rad/s³ with the acceleration flipping sign
    # on ~half the ticks while the reference twist was smooth.
    j_max_arm_rad_s3: 300.0
    # 3.0 made a full rail acceleration reversal take 2*a_max/j = 0.2 s versus
    # 0.02 s on the arm; 60 keeps 2*a_max/j = 0.02 s after a_max rose to 0.60.
    j_max_rail_m_s3: 60.0
    sigma_setbased:
      enabled: true
      activate: 0.12
      safe: 0.06
      exit: 0.16
      gamma: 8.0
      slack_weight: 200.0
      grad_period_ticks: 10
    branch_barrier:
      enabled: true
      # Soft preference at 30°.  The hard velocity box starts at 50° so
      # J4 cannot blast through 0 when QP1 is holding TCP (035411).
      activate_rad: 0.52
      box_activate_rad: 0.87
      eps_rad: 0.35
      # J4 ±135° damper (open travel only).  Do not reuse eps_rad=20°
      # or the upper wall sits at 115° and vertical press dies.
      j4_limit_eps_rad: 0.08726646259971647   # 5° → zero at ~130°
      j4_limit_activate_rad: 0.4363323129985824  # 25° → taper from ~110°
      # J1 same-sign over-fold.  Zero at 140°; 0→−90° startup stays free.
      j1_overfold_abs_rad: 2.443460952792061   # 140°
      j1_overfold_activate_rad: 0.4363323129985824  # 25° → taper from ~115°
      gamma: 6.0
      slack_weight: 80.0
      dwell_free_s: 0.3
      dwell_ramp_s: 1.0
      dwell_scale_max: 5.0
    # Moe/Kanoun set-based comfort: each arm joint, own slack, 15–25° band.
    joint_comfort:
      enabled: true
      m_comfort_deg: 15.0
      activate_deg: 25.0
      gamma: 6.0
      slack_weight: 80.0

  collision:
    enabled: true
    d_safe: 0.01
    d_activate: 0.04
    gamma: 5.0
    max_pairs: 8

  nullspace:
    k_center: 1.0
    k_limit: 2.0
    activation: 0.75
    weights: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # Side-lying family (Pin–Culioli minmax on the photo seed): ψ=68°,
    # d=−0.185, J4≈96°.  Same branch as the taught lean; the 104° seed
    # already hit 124° under a typical Δx/roll shake.  Signs are fixed.
    q_nominal_deg: [0.0, -89.5, -94.5, 65.2, 96.0, 89.3, 61.0, 94.6]
    manipulability:
      k_mu: 0.8
      eps_rad: 5.0e-4
      sigma_fade_ref: 0.12
      grad_period_ticks: 10
      qdot_tau_s: 0.05

  nullspace_d_null: 0.5
  nullspace_d_null_adaptive: 1.0
  nullspace_max_qdot_frac: 0.2
  # Continuous nullspace fade from task slack.  Not a binary latch.
  saturation:
    slack_enter: 0.15
    slack_exit: 0.03
    secondary_scale: 0.15
    secondary_scale_tau_s: 0.10

  arm_angle:
    enabled: true
    k_psi: 1.5
    psi_ref_deg: null
    obs_smooth_floor: 0.3

  psi_retarget:
    enabled: true
    n_y: 9
    n_d: 8
    n_psi: 9
    w_sigma: 0.5
    # Band around the 60° attractor — not |q6|/128°, which sacrificed J2.
    w_wrist: 0.5
    margin_floor_deg: 20.0
    psi_rate_deg_s: 25.0
    # Design split: J4≈96° (band center 95°).  Unplanned step shares one
    # progress s across (d*, ψ*, q*); q* is srs_ik at the live TCP, not
    # the yaml photo at t=0.
    psi_attr_deg: 68.0
    d_attr_m: -0.185
    d_center_rate_m_s: 0.02
    psi_cmd_lead_deg: 18.0
    psi_replan_period_s: 0.1
    psi_search_half_span_deg: 45.0
    psi_search_n: 9
    psi_wrist_ok_deg: 40.0
    psi_return_dwell_s: 1.0
    # >110° collapses J6 on this family.  Never cross ψ=0.
    psi_envelope_deg: [40.0, 110.0]
    require_design_family: false
    rail_margin_m: 0.02
    # Reject cells whose wrist sits on the ~20° branch-barrier floor.
    wrist_min_deg: 30.0

  # Signed IRD field (ird_playground).  One-shot d* at plan_scan_stroke only.
  # Hot-path RailGoodness is σ_min (autograd IRD caused 127 ms hitches).
  # Queries rebuild probe45 TCP from link_7 so gripper2 is ok.
  ird:
    enabled: true
    device: cpu
    allow_stale: true

  rail_extension:
    enabled: true
    k_ext: 2.0
    k_ff: 1.0
    v_ff_thr_m_s: 0.005
    v_ff_span_m_s: 0.015
    e0_m: 0.02
    e1_m: 0.08
    w_max: 2.0
    v_max_m_s: 0.08
    limit_margin_m: 0.15
    pin_margin_m: 0.008
    escape_leave_m: 0.04
    healthy_sigma_mute: 0.08
    press_v_force_min_m_s: 0.02
    press_dz_max_m: 0.002
    press_y_err_m: 0.005
    press_stall_s: 0.5
    d_band_m: 0.005
    k_sigma_boost: 2.0
    k_esc: 0.5
    w_sigma_floor: 1.0
    k_pose: 2.0
    pose_e0_m: 0.005
    pose_e1_m: 0.04
    pose_w_max: 4.0
    sigma_guard_enter: 0.45
    sigma_guard_exit: 0.70
    v_guard_max_m_s: 0.04
    v_lpf_tau_s: 0.05
    v_lpf_fc_hz: 5.0
    v_lpf_tau_escape_s: 0.04
    # Narrow latch: deep σ or true near-limit only (healthy = FF + allocator).
    sigma_escape_enter: 0.55
    sigma_escape_exit: 0.80
    margin_escape_enter: 0.12
    margin_escape_exit: 0.25
    sigma_drop_rate: 0.0
    escape_enter_dwell_s: 0.05
    k_escape_boost: 1.2
    escape_grad_floor: 0.0
    k_margin_boost: 4.0
    w_ext_cap: 24.0
    d_star_err0_m: 0.01
    d_star_err1_m: 0.04
    d_star_w_mult: 6.0
    d_star_reg_mult: 20.0
    # Escape sign: auto = open travel / σ gradient.  minus/plus force a side.
    escape_sign_policy: auto

  rail_allocator:
    v0_m_s: 0.05
    w0_rad_s: 0.30
    k_margin: 4.0
    kp_mid: 1.2
    ki_mid: 0.80
    u_mid_max_m_s: 0.12
    k_err_rail: 4.0
    e_ref_m: 0.08
    # Rail velocity LPF after the live-Y e_mid fix. 2 Hz follows Y faster than
    # 1 Hz without the 5–10 Hz hunting the reversal tests guard against.
    f_c_hz: 2.0
    kaw_mid: 8.0
    rho_mirror_a: 0.50
    rho_mirror_j: 0.30
    reaction_s: 0.06
    observer_pos_gain: 0.35
    observer_vel_gain: 2.0
    observer_vel_lpf_hz: 8.0


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

# Outer-loop Cartesian P for CARTESIAN_TRACK / GOTO.  CLI --move-kp
# overrides k_task_lin only.  Rotation stays here.
# fb_lpf_tau_s filters k*e only; vel_ff is never filtered.  0 = off.
cartesian_track:
  k_task_lin: 12.0
  k_task_rot: 2.0
  max_pos_err_m: 0.05
  max_rot_err_rad: 0.35
  fb_lpf_tau_s: 0.0

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
  - 1
  - 1
  - 1
  - 1
  kp_pos:
  - 10.0
  - 10.0
  - 5.0
  - 1.5
  - 1.5
  - 1.5
  pos_err_deadband_m: 0.0005
  pos_correction_max_m_s: 0.08
  system_delay_s: 0.055
  contact_threshold_n: 0.8
  contact_use_fz_only: true
  physical_contact:
    enabled: true
    # Initial acquire uses filtered force only.  Replaying 162413 with
    # 0.85 N / 20 ms moves the false 3.49 s acquire to the stable load at
    # 4.14 s, while remaining reachable below the shipped 1 N target.
    enter_n: 0.85
    hard_enter_n: 1.5
    # Force task may stay armed across a bounce.  Physical CONTACT must
    # still be allowed to go LOST, or 80 mm/s press reopens in air.
    hold_until_reset: false
    exit_n: 0.70
    enter_confirm_s: 0.02
    exit_confirm_s: 0.1
  # Slightly wider band: blunt single-tick contact dips without sticky D.
  deadband_n: 0.08
  deadband_width_n: 0.10
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
  # 022208: continuous damping for the residual 2.73 Hz base-admittance mode.
  # Chatter: short-lived ΔD_hf(Is). Steady offset: force_dob. Not sticky Ke·D.
  admittance_mass_z: 1.0
  admittance_damping_z: 40.0
  max_vz_tool_m_s: 0.08
  desired_force_ramp_s: 0.30
  var_damping_enabled: true
  var_damping_omega_c_hz: 2.5
  var_damping_lambda: 0.951
  var_damping_f_max_n: 7.0
  # Dimeas Iₛ raises virtual mass briefly; ΔD_hf authority is off.
  var_damping_d_u: 0.0
  var_damping_m_u: 0.0
  var_damping_m_max: 3.0
  var_damping_dc_alpha: 0.02
  var_damping_hf_attack_s: 0.02
  var_damping_hf_hold_s: 0.08
  var_damping_hf_release_s: 0.12
  var_damping_hf_release_fast_s: 0.04  # dump ΔD on hand-release / large |e_f|
  var_damping_hf_on: 0.30
  var_damping_hf_off: 0.15
  var_damping_hf_err_n: 0.8
  recontact_vz_cap_m_s: 0.012
  # Safety recontact speed latches on contact *loss* / suspect_loss, not on
  # reacquire.  hold / episode_release only manage estimator episodes.
  recontact_hold_s: 0.80
  recontact_settle_m_s: 0.003
  recontact_settle_hold_s: 0.050
  contact_episode_release_s: 0.80
  contact_episode_release_force_n: 0.75
  # Restored from e85c9ab.  Steady under-force offset rejection; 1bfe98b
  # disabled it as part of the anti-bounce sweep, and the force barrier below
  # now provides that brake instead.
  force_dob:
    enabled: false
    ki: 8.0
    leak_s: 0.4
    u_max_n: 1.5
    freeze_is: 0.45
    reset_on_reversal: true
  # Contact impact is limited before BEFM/tank intervention.  In free space
  # this preserves the 80 mm/s approach; after contact F+Fdot*T and the
  # stiffness estimate continuously tighten positive press speed.
  force_barrier:
    enabled: true
    t_react_s: 0.055
    budget_min_n: 1.0
    budget_frac: 0.20
    f_keep_n: 0.5
    f_escape_n: 0.5
    v_ref_m_s: 0.08
    v_min_retract_m_s: 0.0
    v_min_press_m_s: 0.0
    v_underforce_press_m_s: 0.010
    underforce_band_n: 0.20
    # Extra clip only.  First contact is K_ub T_stop, not this number.
    v_seek_free_m_s: 0.020
    tau_stop_s: 0.080
    e_x_m: 0.0004
    e_f_n: 0.20
    bar_f_n: 0.15
    fdot_lpf_s: 0.040
    precontact_raw_trigger_n: 1.50 # short impact sleeve; never latches contact
    stiffness_cap_enabled: true
    ke_floor_n_m: 50.0
    mass_floor_kg: 0.05
  # Independent chirps 031234/031605 select T0=30 ms, Tp=12 ms.  Do not
  # invert Γd or claim Y=A when any downstream constraint is active.
  cdyob:
    mode: shadow
    omega_q_hz: 0.75
    t0_s: 0.030
    tp_s: 0.012
    tau_s: 0.030
    t_n_s: 0.012
    pn_m: 0.0
    v_corr_max_m_s: 0.003
    blend_s: 0.30
    active_press_max_m_s: 0.010
    active_retract_max_m_s: 0.010
    active_q_max_hz: 1.0
    active_force_ratio: 0.90
    active_settle_speed_m_s: 0.003
    active_settle_hold_s: 0.20
    # 032041 A-only contact shadow passes polarity/clip + settled-apply gates.
    active_model_validated: true
  # Normal-port delay-aware shield. Stay in observe until the same solver
  # is feasible on hardware. force/observe are empirical peak guards:
  # require_contact_free_terminal is false until a release model exists.
  # Do not enable passive/ospf/CDYOB from this file.
  safety_shield:
    enabled: true
    mode: observe
    terminal_invariance_proven: false
    energy_sign_verified: false
    # Certificate domain.  Declared without numeric bounds is still false.
    pose_domain_declared: false
    payload_domain_declared: false
    max_feedback_age_s: 0.015
    k_ub_n_m: 8000.0
    recovery_hold_s: 0.050
    require_contact_free_terminal: false
    r_f_n_s: 8.0
    r_f_window_steps: 20
    f_release_n: 0.70
    u_retract_m_s: 0.040
    a_max_m_s2: 1.20
    j_max_m_s3: 40.0
    rho: 0.15
    e0_j: 0.004
    eps_j: 0.0005
    # One-pose free-space fit from run_20260821_000011 identify tail.
    # Not a passivity certificate; another pose/load still needed.
    plant:
      t0_s: 0.050
      tp_s: 0.020
      horizon_steps: 40
    # stop_dx_ub is written by --analyze-stop --write-yaml.
    # certified: false until 200 Hz motion-SHM backup replay + independent
    # val covers every stop event.  Tail lookup uses the worst successor
    # (v_{1,q}, a_{1,q}), not the nominal (v̂_1, â_1).

    velocity_error_ub_m_s:
    - 0.002810
    - 0.002800
    - 0.002800
    - 0.002410
    - 0.002800
    - 0.002820
    - 0.002822
    - 0.002820
    - 0.002800
    - 0.002834
    - 0.002810
    - 0.002800
    - 0.002800
    - 0.002410
    - 0.002800
    - 0.002820
    - 0.002820
    - 0.002829
    - 0.002800
    - 0.002800
    - 0.002810
    - 0.002800
    - 0.002800
    - 0.002410
    - 0.002800
    - 0.002820
    - 0.002820
    - 0.002820
    - 0.002800
    - 0.002800
    - 0.002810
    - 0.002800
    - 0.002800
    - 0.002410
    - 0.002800
    - 0.002820
    - 0.002820
    - 0.002820
    - 0.002800
    - 0.002800
    position_error_ub_m:
    - 0.0000140
    - 0.0000280
    - 0.0000420
    - 0.0000541
    - 0.0000681
    - 0.0000822
    - 0.0000963
    - 0.0001104
    - 0.0001244
    - 0.0001386
    - 0.0001526
    - 0.0001666
    - 0.0001806
    - 0.0001927
    - 0.0002067
    - 0.0002208
    - 0.0002349
    - 0.0002490
    - 0.0002630
    - 0.0002770
    - 0.0002911
    - 0.0003051
    - 0.0003191
    - 0.0003311
    - 0.0003451
    - 0.0003592
    - 0.0003733
    - 0.0003874
    - 0.0004014
    - 0.0004154
    - 0.0004295
    - 0.0004435
    - 0.0004575
    - 0.0004695
    - 0.0004835
    - 0.0004976
    - 0.0005117
    - 0.0005258
    - 0.0005398
    - 0.0005538
  # Historical keys. Force-axis a/j now live inside the shield; these
  # values are not applied after u_sent.
  force_axis_slew_press_m_s2: 1.20
  force_axis_slew_retract_m_s2: 1.20
  force_axis_slew_reverse_m_s2: 2.00
  force_axis_jerk_max_m_s3: 40.0
  # Lee-structure speed-level engineering adapter.  Observe is deliberately
  # non-mutating until the slow press/retract sign check and 2/5/10 mm/s
  # no-contact delay identification have been recorded.
  bidirectional_flow:
    mode: observe
    normal_sign: 1.0
    sign_verified: false
    feedback_delay_verified: false
    require_sign_verification: true
    require_delay_verification: true
    # Lee Sec. V-C: alpha is zero in free space.  Below this |fz| the gate is
    # held off and the tank charges from proxy damping.
    free_space_force_n: 0.5
    Dtrack: 25.0
    Kd: 25.0
    Kp: 250.0              # Dtrack / 0.10 s
    Ki: 0.0
    lambda_gain: 0.25
    track_correction_max_m_s: 0.020
    M_p: 1.0
    D_p: 25.0
    # Provisional conservative auxiliary values; retune only after the
    # velocity-step identification.  This branch can hold/retract, never press.
    M_a: 0.05
    D_a: 5.0
    K_a: 50.0
    B_a: 5.0
    u_retract_n: 0.0
    aux_max_retract_m_s: 0.050
    alpha_attack_s: 0.020
    alpha_release_s: 0.150
    max_feedback_age_s: 0.020
    T0: 0.0010
    Tmax: 0.0040
    Tmin: 0.0001
    mu_power_w: 0.0
    positive_switching_cost_j: 0.0
  # Optional Piedra-style elastic-surface force reduction.  Disabled until
  # stable-contact hardware validation; it is not a passivity guarantee.
  surface_force_modulation:
    enabled: false
    min_force_scale: 0.25
    beta_per_m: 80.0
    stable_contact_s: 0.20
    attack_s: 0.05
    release_s: 0.15
  # Soften under-force chase near scan turnaround (tool-XY slow).
  force_lateral_soft_m_s: 0.006
  force_lateral_full_m_s: 0.018
  force_lateral_gain_floor: 0.35
  adaptive_ke:
    enabled: true
    # Observe Ke / impact burst only — do not hold high critical D in steady contact.
    drive_damping: false
    zeta: 0.9
    ke_initial: 80.0
    ke_min: 40.0
    ke_max: 2500.0
    ke_impact_initial: 0.0
    ke_cap_ub_n_m: 2000.0
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
    bd_max: 180.0
    bd_slew_max: 400.0
    ke_slew_max: 1200.0
  proactive_feedforward: false
  # Legacy parameters are retained for isolated non-CDYOB tests only.
  proactive_retract_only: false
  # 014140 showed why this inactive v_r branch must not contaminate shadow.
  proactive_gain: 0.06
  proactive_retract_gain: 0.06
  proactive_leak_s: 0.50
  v_r_max_m_s: 0.02
  proactive_gate_press_on_is: false
  proactive_press_is_gate_start: 0.2
  proactive_press_is_gate: 0.6
  proactive_press_is_soft_floor: 0.45
  proactive_press_is_soft_stop: 0.85
  proactive_press_slew_max_m_s2: 0.35
  proactive_retract_slew_max_m_s2: 0.35
  proactive_press_drive_max: 1.4
  proactive_retract_drive_max: 1.4
  proactive_reset_on_reversal: true
  proactive_in_band_n: 0.08
  proactive_in_band_leak_s: 0.12
  force_scale_min_n: 0.18
  force_scale_fraction: 0.0
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
hw:
  lw100:
    enabled: true
    host: 192.168.0.7
    port: 8234
    slave: 1
    lead_mm: 10.0
    # calibrated_file: load var/lw100_rail_zero.json (run apps/lw100_rail_home_limit.py first).
    # current/fixed are debug-only; with require_calibration true, current is refused.
    zero_mode: calibrated_file
    counts0: 0
    calibration_path: var/lw100_rail_zero.json
    require_calibration: true
    home_di: di4             # −Y home switch (confirmed on HW; was swapped vs DI3)
    plus_di: di3             # +Y end (run-time e-stop if hit)
    di_nc: true
    di_debounce_n: 3
    soft_min_m: 0.030        # full-speed inner edge; must match qpik.hard_limits.rail
    soft_max_m: 0.755
    hard_min_m: 0.005        # travel box 5–780 mm
    hard_max_m: 0.78
    post_home_m: 0.025
    # Home-script only (controller does not auto-home on start):
    home_search_m_s: 0.020
    home_creep_m_s: 0.003
    home_backoff_mm: 5.0
    home_touch_count: 3
    home_search_timeout_s: 60.0
    home_to_post_m_s: 0.030
    limit_poll_every: 5
    # Host rail_y ↔ motor: -1 flips FA24 RPM (+ encoder map in rail_servo).
    sign: -1
    enable_settle_s: 0.3
    # Cold start: prove worker Modbus read+FA24=0 before any set_target / move→D.
    arm_good_reads: 30          # ~0.6 s @ 50 Hz consecutive healthy polls
    arm_settle_s: 0.8           # extra FA24=0 hold after good polls
    arm_max_span_mm: 2.0
    arm_timeout_s: 10.0
    fault_margin_m: 0.05
    # 205605: t_read med 5.8 ms, FA24 write usually skipped.  43 Hz left
    # ~17 ms of sleep.  60 Hz (16.7 ms) still has ~11 ms median headroom;
    # 80 Hz is tight on p95.  Do not change FA72 or the USR-TCP232-304 baud
    # alone — both ends must match, then power-cycle the drive.  Confirm no
    # DI is mapped to 7 (ZCLAMP) and FC-15 bit6 is 0; do not query while A
    # is live.  FA40/FA41 = time 0→1000 r/min, not 0→vel_max.
    poll_hz: 60
    inter_frame_delay_s: 0.0005
    timeout_s: 0.06             # poll-budget; was 0.15 / class-default 1.0
    retries: 1
    deadband_mm: 0.5
    # FA23 overspeed trip: must sit ABOVE commanded peak (0.15 m/s ≈ 900 rpm
    # @ 10 mm/rev). Equal FA23=cmd caused Er-01 on scan overshoot (151334).
    max_speed_rpm: 1200
    # Soft CSP via FA24 (see apps/lw100_vel_pos_follow_demo.py).
    # POSITION scan/home PD.  Coupled QPIK is pure velocity (no FA24 P).
    # Loaded PD scan BEST (400±40 mm): kp=14/kd=0.22.
    vel_kp: 14.0
    vel_kd: 0.22
    vel_kd_max_m_s: 0.005
    # Matches QP box rail.v_max_m_s 0.15 × v_scale 0.8.  parse_rail_servo_config
    # also caps hw.vel_max by that product so the two cannot drift apart.
    vel_max_m_s: 0.12
    vel_amax_m_s2: 1.2
    # Coupled-mode catch-up of x_ref toward x_goal while moving.  20 mm/s
    # clears a 20 mm standing offset in ~1 s without outrunning FF.
    catch_v_max_m_s: 0.02
    catch_k: 5.0
    catch_frac: 0.3
    decel_request_margin_m_s: 0.005
    match_drive_accel: true
    fa24_rpm_deadband: 0   # write every worker tick; skip only if rpm is unchanged
    vel_deadband_mm: 0.05   # tight tracking band (not a permanent accuracy sacrifice)
    # Standstill hysteresis: freeze FA24 after tight settle; wake only if pushed.
    standstill_enter_mm: 0.05
    standstill_exit_mm: 0.25
    standstill_dwell_s: 0.08
    approach_m: 0.040
    latch_watch_s: 0.12
    target_timeout_s: 0.25
    # Extra coast after timeout before FA24=0.  A 127 ms hitch must not brake.
    target_stale_coast_s: 0.35
    encoder_freeze_s: 1.0
    encoder_freeze_min_v_m_s: 0.02
    encoder_freeze_min_move_mm: 0.15
    # End-of-stream / task-end settle before releasing follow (closes latched overshoot).
    settle_tol_mm: 0.05
    settle_v_m_s: 0.006
    settle_timeout_s: 1.5
    # Stall-safe speed: worst-case latched FA24 overshoot ≤ |err| for max_stall_s.
    max_stall_s: 0.06
    stall_v_floor_m_s: 0.004
    # Soft-reject above v_max·gap + margin; wipe cal only on ≥50 mm or 2 soft jumps.
    jump_margin_mm: 3.0
    jump_hard_mm: 50.0
    jump_soft_streak_panic: 2
    # FA40/41: 120 ms → drive a ≈ 1.0 m/s².  Host a_max is min(this, QP
    # a_max_rail_m_s2 0.60, 0.85 × vel_max/accel_s) so the servo cannot
    # outrun the QP model.
    accel_ms: 120
    decel_ms: 120
    scurve_ms: 30            # FA42
    busy_speed_rpm: 1
    home_on_exit: false
    release_son_on_exit: false  # stop with FA24=0 and keep SON; avoids enable-edge frame wipe
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
  # Control-loop stall timeout.  QP backend pulses the watchdog during ProxQP.
  watchdog_timeout_s: 0.50
```

## 3. `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/psi_retarget.py`

- sha256：`9720790933b2db8c54e2cc22c1b6ff25461fb91c20edd5a0b11947a0b31dcd7e`
- 行数：1053

```python
"""One-shot min-max (d*, ψ*) planner for a known scan stroke.

Online hill-climb of instantaneous elbow margin is a double-well: both rail
ends score high and the interior (rail facing the TCP) scores low, so a
greedy climber parks the carriage on a stop.  For a periodic scan the
literature answer (Pin–Culioli minimax / Vahrenkamp ORM_tr) is to pick the
offset that maximises the *worst* joint margin over the whole stroke, then
hold it.

Call :meth:`PostureRetarget.plan_stroke` once when the scan starts.  After
that :meth:`step` only slews ψ toward ψ* with a single rate limit (no LPF)
and holds the planned d* constant.

Unplanned ``step`` homes ``(d*, ψ*, q*)`` on one progress ``s``.  ``T``
is the slower of the existing ψ and d rates; ``q*`` is ``srs_ik`` at the
current TCP (same branch), not the yaml photo at t=0.  Hunt ``d*`` /
``ψ*`` while moving; freeze ``hold_setpoint`` only when the command and
TCP are both quiet (or slack is high).  Local ψ search takes over only
while the wrist is collapsed and the elbow is still open (SEW is
undefined near the J4 floor).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import (
    RAIL_INDEX,
    RobotKinematics,
    full_q_from_arm,
)
from rm75_control.kinematics.srs_ik import (
    Q_LOWER,
    Q_UPPER,
    branch_from_q,
    flange_tcp_from_kin,
    psi_from_q,
    shoulder_y_from_q_rail,
    srs_ik,
)


class StrokeInfeasibleError(RuntimeError):
    """Raised when no (d, ψ) covers the requested stroke inside rail travel."""


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def nearest_planar_psi(psi_rad: float) -> float:
    """Quantize swivel to the nearer SEW plane ``{0, ±π}``.

    The taught home (J1≈0, J6≈90°) sits at ψ=π; yaml ``q_nominal``
    (J6=45°) sits at ψ=0.  Those are opposite elbow orbits.  Snap once
    at reset so swivel returns to the start family, not the other plane.
    """
    a = _wrap_pi(float(psi_rad))
    if abs(a) <= 0.5 * np.pi:
        return 0.0
    # ±π are the same SEW plane; keep +π so CSV ψ* reads 180°.
    return float(np.pi)


def fold_psi_to_positive(psi_rad: float) -> float:
    """Map ψ into ``[0, π]`` so the one-sided envelope is well-defined.

    ``−π`` and ``+π`` are the same SEW plane; the negative half-plane is
    folded across 0 so the attractor never asks the arm to cross ψ = 0.
    """
    a = abs(_wrap_pi(float(psi_rad)))
    return min(a, float(np.pi))


def clamp_psi_to_envelope(
    psi_rad: float,
    lo_rad: float,
    hi_rad: float,
) -> float:
    """Fold onto the positive family, then clamp to ``[lo, hi] ⊂ (0, π)``."""
    lo = max(float(lo_rad), 1.0e-6)
    hi = min(float(hi_rad), float(np.pi) - 1.0e-6)
    if lo > hi:
        lo, hi = hi, lo
    return float(np.clip(fold_psi_to_positive(psi_rad), lo, hi))


def psi_err_avoiding_zero(cur_rad: float, target_rad: float) -> float:
    """Signed ψ error that never takes the short path through 0."""
    cur = _wrap_pi(float(cur_rad))
    target = _wrap_pi(float(target_rad))
    err = _wrap_pi(target - cur)
    nxt = cur + err
    if cur * nxt < 0.0 and abs(cur) < 0.5 * np.pi and abs(target) < 0.5 * np.pi:
        if err > 0.0:
            err -= 2.0 * np.pi
        else:
            err += 2.0 * np.pi
    return float(err)


# Half-width of the first ``plan_stroke`` search around the taught plane.
# Opposite-family search uses the same width only when this band is empty.
_PLAN_FAMILY_HALF_SPAN_RAD = 40.0 * np.pi / 180.0


def _arm7(q_arm: np.ndarray) -> np.ndarray:
    q = np.asarray(q_arm, dtype=float).reshape(-1)
    return q[1:] if q.size == 8 else q


def d_from_q(kin: RobotKinematics, q_rad: np.ndarray) -> float:
    """Arm Y-reach ``d = y_tcp − q0``.  Invariant to the rail coordinate."""
    q = np.asarray(q_rad, dtype=float).reshape(-1)
    if q.size == 7:
        q = np.concatenate([[0.0], q])
    if q.size != 8:
        raise ValueError(f"q must be length 7 or 8, got {q.size}")
    return float(kin.fk_placement(q).translation[1]) - float(q[RAIL_INDEX])


def joint_margin_frac(q_arm: np.ndarray) -> float:
    """Normalised per-joint slack in (0, 1]; return the worst joint."""
    q = _arm7(q_arm)
    half = 0.5 * (Q_UPPER - Q_LOWER)
    half = np.maximum(half, 1.0e-6)
    lo = (q - Q_LOWER) / half
    hi = (Q_UPPER - q) / half
    return float(np.min(np.minimum(lo, hi)))


def wrist_band_frac(
    q6: float,
    *,
    peak_rad: float = 60.0 * np.pi / 180.0,
) -> float:
    """1 at |q6|≈45°, 0 at a straight wrist and at the J6 stop."""
    a = abs(float(q6))
    q6_max = max(abs(float(Q_LOWER[5])), abs(float(Q_UPPER[5])), 1.0e-6)
    peak = min(max(float(peak_rad), 1.0e-6), q6_max)
    if a <= peak:
        return a / peak
    return max(0.0, 1.0 - (a - peak) / (q6_max - peak))


def design_family_ok(
    q_meas: np.ndarray,
    q_nominal: np.ndarray,
    *,
    psi_tol_rad: float = 45.0 * np.pi / 180.0,
) -> bool:
    """True if measured q is the same SEW family as the design attractor."""
    qm = np.asarray(q_meas, dtype=float).reshape(-1)
    qn = np.asarray(q_nominal, dtype=float).reshape(-1)
    if qm.size == 7:
        qm = np.concatenate([[0.0], qm])
    if qn.size == 7:
        qn = np.concatenate([[0.0], qn])
    if qm.size != 8 or qn.size != 8:
        return False
    psi_m = fold_psi_to_positive(psi_from_q(qm))
    psi_n = fold_psi_to_positive(psi_from_q(qn))
    if abs(psi_m - psi_n) > float(psi_tol_rad):
        return False
    if int(branch_from_q(qm)) != int(branch_from_q(qn)):
        return False
    if abs(float(qn[1])) > 1.0e-3 and abs(float(qm[1])) > 1.0e-3:
        if float(qm[1]) * float(qn[1]) < 0.0:
            return False
    return True


def arm_respects_floor(q_arm: np.ndarray, floor_rad: float) -> bool:
    """True iff every arm joint is at least ``floor_rad`` from a stop."""
    if float(floor_rad) <= 0.0:
        return True
    q = _arm7(q_arm)
    margin = np.minimum(q - Q_LOWER, Q_UPPER - q)
    return bool(np.all(margin >= float(floor_rad) - 1.0e-9))


def stroke_score(
    q_arm: np.ndarray,
    sigma: float,
    *,
    w_sigma: float,
    w_wrist: float,
) -> float:
    """One-shot cell score: worst-joint margin + σ + J6 band around 45°.

    ``|q6|/q6_max`` rewarded opening the wrist all the way to ±128° and
    parked J2 on a stop.  The band peaks at the yaml attractor (45°).
    """
    q = _arm7(q_arm)
    return (
        joint_margin_frac(q)
        + float(w_sigma) * float(sigma)
        + float(w_wrist) * wrist_band_frac(float(q[5]))
    )


@dataclass
class PsiRetargetConfig:
    enabled: bool = True
    n_y: int = 9
    n_d: int = 8
    n_psi: int = 9
    w_sigma: float = 0.5
    # Same scale as w_sigma.  Scores a 45° wrist band, not |q6|/q6_max.
    w_wrist: float = 0.5
    # Reject a (d, ψ) cell if any arm joint is closer than this to a stop.
    margin_floor_rad: float = 15.0 * np.pi / 180.0
    # Used only when ψ* changes (new scan segment).  No LPF on top.
    psi_rate_rad_s: float = 25.0 * np.pi / 180.0
    # Unplanned d* is a band around the design split, not a chasing point.
    d_center_rate_m_s: float = 0.02
    # Do not let ψ_cmd run more than this ahead of live ψ.
    psi_cmd_lead_rad: float = 18.0 * np.pi / 180.0
    # Design family (side-lying).  Unplanned homotopy and plan_stroke.
    psi_attr_rad: float = 68.0 * np.pi / 180.0
    d_attr_m: float = -0.185
    # Runtime elbow band.  Open rail travel must not pick J4≈135°.
    elbow_center_rad: float = 95.0 * np.pi / 180.0
    elbow_lo_rad: float = 70.0 * np.pi / 180.0
    elbow_hi_rad: float = 115.0 * np.pi / 180.0
    elbow_hi_illegal_rad: float = 130.0 * np.pi / 180.0
    psi_return_dwell_s: float = 1.0
    require_design_family: bool = False
    # Local ψ search (unplanned).  9 srs_ik × 0.09 ms ≈ 0.8 ms at 10 Hz.
    psi_replan_period_s: float = 0.1
    psi_search_half_span_rad: float = 45.0 * np.pi / 180.0
    psi_search_n: int = 9
    psi_wrist_ok_rad: float = 40.0 * np.pi / 180.0
    psi_envelope_lo_rad: float = 40.0 * np.pi / 180.0
    psi_envelope_hi_rad: float = 110.0 * np.pi / 180.0
    # Soft travel used by the planner (must cover the whole stroke).
    rail_margin_m: float = 0.02
    # Reject a cell whose wrist sits on the branch-barrier floor (~20°).
    wrist_min_rad: float = 30.0 * np.pi / 180.0


class _SrsEval:
    """Cached flange TCP + one srs_ik + Jacobian/σ evaluation."""

    def __init__(self, kin: RobotKinematics) -> None:
        self.kin = kin
        self._R, self._t = flange_tcp_from_kin(kin)

    def evaluate(
        self,
        pose: np.ndarray,
        psi: float,
        branch: int,
        y_rail: float,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        q_arm = srs_ik(
            pose,
            float(psi),
            int(branch),
            y_rail=shoulder_y_from_q_rail(float(y_rail)),
            R_flange_tcp=self._R,
            t_flange_tcp=self._t,
        )
        if q_arm is None:
            return None
        q_full = full_q_from_arm(q_arm, rail_m=float(y_rail))
        sigma = float(self.kin.singular_values(self.kin.jacobian(q_full)).min())
        return q_arm, q_full, sigma


class PostureRetarget:
    """Stroke min-max planner; ``step`` holds (d*, ψ*) after ``plan_stroke``."""

    def __init__(
        self,
        kin: RobotKinematics,
        cfg: PsiRetargetConfig | None = None,
        *,
        euler_order: str = "xyz",
    ) -> None:
        self.kin = kin
        self.cfg = cfg or PsiRetargetConfig()
        self.euler_order = str(euler_order)
        self._eval = _SrsEval(kin)
        self._psi_cmd: float | None = None
        self._psi_star: float | None = None
        self._d_star: float | None = None
        self._d_center_target: float | None = None
        self._s: float = 0.0
        self._d0: float = float("nan")
        self._psi0: float = float("nan")
        self._branch: int = 0
        self.q_star_rad: np.ndarray | None = None
        self.homotopy_s: float = 0.0
        self._search_age_s: float = 0.0
        self.last_psi_search_count: int = 0
        self.last_search_j6_rad: float = float("nan")
        self._planned: bool = False
        self._z_plan: float = float("nan")
        self._y_center_m: float = float("nan")
        self._amplitude_m: float = float("nan")
        self._rail_lo: float = float("nan")
        self._rail_hi: float = float("nan")
        self.last_psi_score: float = float("nan")
        self.last_dpref_score: float = float("nan")
        self.last_minmax_margin: float = float("nan")
        self.last_elbow_margin_rad: float = float("nan")
        self.last_wrist_open_rad: float = float("nan")
        self.d_star_m: float = float("nan")
        self.psi_star_rad: float = float("nan")
        self.last_psi_family_degraded: bool = False
        self._healthy_dwell_s: float = 0.0
        self._held_prev: bool = False
        self._ird = None

    @property
    def planned(self) -> bool:
        return bool(self._planned)

    def reset(self, q_rad: np.ndarray) -> None:
        q = np.asarray(q_rad, dtype=float)
        # ±π are the same SEW plane.  Stay on the positive half so the
        # command slews 180°→70°, never −180°→−290° through ψ = 0.
        psi = fold_psi_to_positive(float(psi_from_q(q)))
        psi_star = clamp_psi_to_envelope(
            float(self.cfg.psi_attr_rad),
            self.cfg.psi_envelope_lo_rad,
            self.cfg.psi_envelope_hi_rad,
        )
        self._psi_cmd = psi
        self._psi_star = psi_star
        # Start at the live split.  q* is the live configuration — not the
        # yaml photo — so J1 is not pinned to −90° while d* is still here.
        d_live = d_from_q(self.kin, q)
        self._d_star = d_live
        self._d_center_target = float(self.cfg.d_attr_m)
        self._s = 0.0
        self._d0 = float(d_live)
        self._psi0 = float(psi)
        self._branch = int(branch_from_q(q))
        self.q_star_rad = np.asarray(q, dtype=float).reshape(-1).copy()
        self.homotopy_s = 0.0
        self._search_age_s = 0.0
        self._healthy_dwell_s = 0.0
        self.last_psi_search_count = 0
        self.last_search_j6_rad = float("nan")
        self._planned = False
        self._z_plan = float("nan")
        self._held_prev = False
        self.d_star_m = float(self._d_star)
        self.psi_star_rad = float(psi_star)

    def _update_margins(self, q: np.ndarray) -> None:
        q_arm = np.asarray(q, dtype=float).reshape(-1)
        if q_arm.size == 8:
            q_arm = q_arm[1:]
        q4 = float(q_arm[3])
        q6 = float(q_arm[5])
        self.last_elbow_margin_rad = float(
            min(q4 - float(Q_LOWER[3]), float(Q_UPPER[3]) - q4)
        )
        self.last_wrist_open_rad = float(abs(q6))

    def plan_stroke(
        self,
        q_rad: np.ndarray,
        *,
        y_center_m: float,
        amplitude_m: float,
        rail_lo: float,
        rail_hi: float,
    ) -> tuple[float, float]:
        """Grid-search ``(d*, ψ*)`` over the scan stroke.  Raises if empty.

        Search the taught SEW family first.  The opposite plane is used only
        when that family has no feasible cell (singularity / travel).
        """
        q = np.asarray(q_rad, dtype=float)
        self.last_psi_family_degraded = False
        pose0 = np.asarray(self.kin.fk_pose(q), dtype=float).reshape(6)
        branch = int(branch_from_q(q))
        amp = abs(float(amplitude_m))
        y_c = float(y_center_m)
        y_lo = y_c - amp
        y_hi = y_c + amp
        margin = max(float(self.cfg.rail_margin_m), 0.0)
        rail_lo_s = float(rail_lo) + margin
        rail_hi_s = float(rail_hi) - margin
        # y - d ∈ [rail_lo_s, rail_hi_s] for every y in the stroke.
        d_min = y_hi - rail_hi_s
        d_max = y_lo - rail_lo_s
        if d_min > d_max + 1.0e-9:
            raise StrokeInfeasibleError(
                f"scan stroke [{y_lo:.3f}, {y_hi:.3f}] m does not fit rail "
                f"[{rail_lo_s:.3f}, {rail_hi_s:.3f}] m; reduce amplitude"
            )
        n_y = max(int(self.cfg.n_y), 3)
        n_d = max(int(self.cfg.n_d), 3)
        n_psi = max(int(self.cfg.n_psi), 3)
        y_samples = np.linspace(y_lo, y_hi, n_y)
        d_grid = np.linspace(d_min, d_max, n_d)
        d_samples = d_grid
        if self._ird is not None and getattr(self._ird, "available", False):
            T_ird0 = self._ird.tcp_ird_from_q(self.kin, q)
            d_ird = self._ird.query_d_star(
                T_ird0,
                y_tcp0_m=float(pose0[1]),
                y_samples_m=y_samples,
                d_samples_m=d_grid,
                rail_lo=rail_lo_s,
                rail_hi=rail_hi_s,
            )
            if d_ird is not None and d_min - 1.0e-9 <= d_ird <= d_max + 1.0e-9:
                rails = y_samples - float(d_ird)
                if np.all(rails >= rail_lo_s - 1.0e-9) and np.all(
                    rails <= rail_hi_s + 1.0e-9
                ):
                    d_samples = np.array([float(d_ird)], dtype=float)
        psi0 = float(psi_from_q(q))
        # Unplanned home (psi_attr) must not steal the stroke family.
        if self._planned and self._psi_star is not None:
            psi_family = float(self._psi_star)
        else:
            psi_family = nearest_planar_psi(psi0)
        half = float(_PLAN_FAMILY_HALF_SPAN_RAD)
        family_samples = psi_family + np.linspace(-half, half, n_psi)
        opposite = _wrap_pi(psi_family + np.pi)
        opposite_samples = opposite + np.linspace(-half, half, n_psi)
        w_sigma = float(self.cfg.w_sigma)
        w_wrist = float(self.cfg.w_wrist)
        floor = float(self.cfg.margin_floor_rad)

        def _search(
            d_list: np.ndarray, psi_list: np.ndarray
        ) -> tuple[bool, float, float, float]:
            best_s = -np.inf
            best_dv = float(self._d_star if self._d_star is not None else 0.0)
            best_pv = psi_family
            found = False
            for d in d_list:
                for psi in psi_list:
                    worst = np.inf
                    feasible = True
                    last_q: np.ndarray | None = None
                    for y in y_samples:
                        y_rail = float(y) - float(d)
                        if y_rail < rail_lo_s - 1.0e-9 or y_rail > rail_hi_s + 1.0e-9:
                            feasible = False
                            break
                        pose = pose0.copy()
                        pose[1] = float(y)
                        pack = self._eval.evaluate(pose, float(psi), branch, y_rail)
                        if pack is None:
                            feasible = False
                            break
                        q_arm, q_full, sigma = pack
                        last_q = q_full
                        if not arm_respects_floor(q_arm, floor):
                            feasible = False
                            break
                        if abs(float(q_arm[5])) < float(self.cfg.wrist_min_rad) - 1.0e-9:
                            feasible = False
                            break
                        score_y = stroke_score(
                            q_arm, sigma, w_sigma=w_sigma, w_wrist=w_wrist
                        )
                        if score_y < worst:
                            worst = score_y
                    if not feasible or not np.isfinite(worst):
                        continue
                    found = True
                    if worst > best_s:
                        best_s = float(worst)
                        best_dv = float(d)
                        best_pv = float(psi)
                        if last_q is not None:
                            self._update_margins(last_q)
            return found, best_s, best_dv, best_pv

        def _search_d(psi_list: np.ndarray) -> tuple[bool, float, float, float]:
            found, score, d_v, p_v = _search(d_samples, psi_list)
            if not found and d_samples.size == 1 and d_grid.size > 1:
                found, score, d_v, p_v = _search(d_grid, psi_list)
            return found, score, d_v, p_v

        any_feasible, best_score, best_d, best_psi = _search_d(family_samples)
        degraded = False
        if not any_feasible:
            degraded = True
            any_feasible, best_score, best_d, best_psi = _search_d(opposite_samples)
        if not any_feasible:
            raise StrokeInfeasibleError(
                "no feasible (d, ψ) covers the scan stroke; reduce amplitude "
                "or choose a less extended start pose"
            )
        # Family grids are already near 0 or ±π; wrap so CSV ψ* stays readable.
        best_psi = _wrap_pi(best_psi)
        self.last_psi_family_degraded = bool(degraded)
        self._d_star = float(best_d)
        self._d_center_target = float(best_d)
        self._psi_star = float(best_psi)
        self._planned = True
        self._z_plan = float(pose0[2])
        self._y_center_m = y_c
        self._amplitude_m = amp
        self._rail_lo = float(rail_lo)
        self._rail_hi = float(rail_hi)
        self.d_star_m = float(best_d)
        self.psi_star_rad = float(best_psi)
        self.last_minmax_margin = float(best_score)
        self.last_dpref_score = float(best_score)
        self.last_psi_score = float(best_score)
        if self._psi_cmd is None:
            self._psi_cmd = float(best_psi)
        return float(best_d), float(best_psi)

    def step(
        self,
        q_rad: np.ndarray,
        dt_s: float,
        *,
        rail_lo: float,
        rail_hi: float,
        q_nominal: np.ndarray | None = None,
        hold_setpoint: bool = False,
    ) -> tuple[float, float]:
        """Slew (d*, ψ*, q*) on one s; planned strokes only slew ψ."""
        del q_nominal
        q = np.asarray(q_rad, dtype=float)
        if self._psi_cmd is None or self._d_star is None:
            self.reset(q)
        dt = max(float(dt_s), 0.0)
        live_psi = fold_psi_to_positive(float(psi_from_q(q)))
        if self._planned:
            psi_out = self._rate_limit_psi(dt, live_psi=live_psi)
            self._update_margins(q)
            return float(psi_out), float(self._d_star)
        if self._held_prev and not hold_setpoint:
            if self._d_star is not None and np.isfinite(float(self._d_star)):
                self._d0 = float(self._d_star)
            if self._psi_cmd is not None and np.isfinite(float(self._psi_cmd)):
                self._psi0 = float(self._psi_cmd)
            self._s = 0.0
            self.homotopy_s = 0.0
        self._held_prev = bool(hold_setpoint)
        if hold_setpoint:
            psi_out = self._rate_limit_psi(dt, live_psi=live_psi)
            self._update_margins(q)
            return float(psi_out), float(self._d_star)
        self._maybe_retarget_psi(
            q,
            dt_s=dt,
            rail_lo=float(rail_lo),
            rail_hi=float(rail_hi),
        )
        self._advance_homotopy(
            q,
            dt,
            rail_lo=float(rail_lo),
            rail_hi=float(rail_hi),
            live_psi=live_psi,
        )
        self._update_margins(q)
        return float(self._psi_cmd), float(self._d_star)

    def _advance_homotopy(
        self,
        q: np.ndarray,
        dt_s: float,
        *,
        rail_lo: float,
        rail_hi: float,
        live_psi: float,
    ) -> None:
        psi_goal = fold_psi_to_positive(
            float(self._psi_star if self._psi_star is not None else self._psi0)
        )
        pose = np.asarray(self.kin.fk_pose(q), dtype=float).reshape(6)
        d_goal = self._select_d_for_elbow(
            q,
            pose=pose,
            psi=psi_goal,
            rail_lo=float(rail_lo),
            rail_hi=float(rail_hi),
        )
        if d_goal is None or not np.isfinite(float(d_goal)):
            self._rate_limit_psi(float(dt_s), live_psi=live_psi)
            return
        d0 = float(self._d0) if np.isfinite(self._d0) else float(self._d_star)
        psi0 = float(self._psi0) if np.isfinite(self._psi0) else float(self._psi_cmd)
        T = self._homotopy_T(d0, float(d_goal), psi0, psi_goal)
        s_try = min(1.0, float(self._s) + float(dt_s) / T)
        d_try = float(d0 + s_try * (float(d_goal) - d0))
        y_tcp = float(pose[1])
        d_try = self._clip_d_to_travel(
            d_try,
            y_tcp=y_tcp,
            rail_lo=float(rail_lo),
            rail_hi=float(rail_hi),
            d_live=y_tcp - float(q[RAIL_INDEX]),
        )
        if d_try is None:
            self._rate_limit_psi(float(dt_s), live_psi=live_psi)
            return
        d_step = max(float(self.cfg.d_center_rate_m_s), 0.0) * max(float(dt_s), 0.0)
        d_prev = (
            float(self._d_star)
            if self._d_star is not None and np.isfinite(float(self._d_star))
            else float(d_try)
        )
        d_try = max(d_prev - d_step, min(d_prev + d_step, float(d_try)))
        psi_s = fold_psi_to_positive(
            float(psi0) + s_try * psi_err_avoiding_zero(psi0, psi_goal)
        )
        pack = self._eval_at_split(pose, float(psi_s), float(d_try))
        if pack is None or not self._q_star_acceptable(pack[0], q, rail_lo, rail_hi):
            self._rate_limit_psi(float(dt_s), live_psi=live_psi)
            return
        self._s = float(s_try)
        self.homotopy_s = float(s_try)
        self._d_star = float(d_try)
        self.d_star_m = float(d_try)
        q_arm, q_full, _sigma = pack
        self.q_star_rad = np.asarray(q_full, dtype=float).copy()
        self._update_margins(q_full)
        del q_arm
        self._rate_limit_psi(float(dt_s), live_psi=live_psi)

    def _homotopy_T(
        self,
        d0: float,
        d_goal: float,
        psi0: float,
        psi_goal: float,
    ) -> float:
        d_rate = max(float(self.cfg.d_center_rate_m_s), 1.0e-9)
        psi_rate = max(float(self.cfg.psi_rate_rad_s), 1.0e-9)
        t_d = abs(float(d_goal) - float(d0)) / d_rate
        t_psi = abs(psi_err_avoiding_zero(float(psi0), float(psi_goal))) / psi_rate
        return max(t_d, t_psi, 1.0e-6)

    def _j4_in_design_band(self, j4_rad: float, *, loose: bool = False) -> bool:
        lo = float(self.cfg.elbow_lo_rad)
        hi = float(self.cfg.elbow_hi_rad)
        if loose:
            lo -= np.deg2rad(5.0)
            hi += np.deg2rad(7.0)
        return bool(lo - 1.0e-9 <= float(j4_rad) <= hi + 1.0e-9)

    def _j4_illegal_at_stop(self, j4_rad: float, *, has_travel: bool) -> bool:
        if not has_travel:
            return False
        return bool(abs(float(j4_rad)) >= float(self.cfg.elbow_hi_illegal_rad) - 1.0e-9)

    def _rail_window(
        self, y_tcp: float, rail_lo: float, rail_hi: float
    ) -> tuple[float, float] | None:
        margin = max(float(self.cfg.rail_margin_m), 0.0)
        y_lo = float(rail_lo) + margin
        y_hi = float(rail_hi) - margin
        if y_lo > y_hi + 1.0e-12:
            return None
        d_lo = float(y_tcp) - y_hi
        d_hi = float(y_tcp) - y_lo
        if d_lo > d_hi + 1.0e-12:
            return None
        return float(d_lo), float(d_hi)

    def _clip_d_to_travel(
        self,
        d: float,
        *,
        y_tcp: float,
        rail_lo: float,
        rail_hi: float,
        d_live: float | None,
    ) -> float | None:
        window = self._rail_window(float(y_tcp), float(rail_lo), float(rail_hi))
        if window is None:
            if d_live is not None and np.isfinite(float(d_live)):
                return float(d_live)
            return None
        return float(np.clip(float(d), window[0], window[1]))

    def _eval_at_split(
        self,
        pose: np.ndarray,
        psi: float,
        d: float,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        y_rail = float(pose[1]) - float(d)
        return self._eval.evaluate(pose, float(psi), int(self._branch), y_rail)

    def _q_star_acceptable(
        self,
        q_arm: np.ndarray,
        q_live: np.ndarray,
        rail_lo: float,
        rail_hi: float,
    ) -> bool:
        j4 = float(np.asarray(q_arm, dtype=float).reshape(-1)[3])
        window = self._rail_window(
            float(self.kin.fk_placement(q_live).translation[1]),
            float(rail_lo),
            float(rail_hi),
        )
        has_travel = window is not None and (window[1] - window[0]) > 0.01
        if self._j4_illegal_at_stop(j4, has_travel=has_travel):
            return False
        return True

    def _select_d_for_elbow(
        self,
        q: np.ndarray,
        *,
        pose: np.ndarray,
        psi: float,
        rail_lo: float,
        rail_hi: float,
    ) -> float | None:
        """Split at ``psi`` whose IK J4 stays in the design band, near d_attr."""
        y_tcp = float(pose[1])
        window = self._rail_window(y_tcp, float(rail_lo), float(rail_hi))
        if window is None:
            return None
        d_lo, d_hi = window
        d_pref = (
            float(self._d_center_target)
            if self._d_center_target is not None
            else float(self.cfg.d_attr_m)
        )
        has_travel = (d_hi - d_lo) > 0.01
        samples = list(np.linspace(d_lo, d_hi, 11))
        for extra in (d_pref, float(self._d_star), float(self._d0)):
            if extra is None or not np.isfinite(float(extra)):
                continue
            if d_lo - 1.0e-9 <= float(extra) <= d_hi + 1.0e-9:
                samples.append(float(extra))
        samples = [float(x) for x in np.unique(np.asarray(samples, dtype=float))]
        # Prefer the yaml family (J1 < 0).  Do not freeze s on a live/IK
        # sign mismatch — that locked d* while ψ already folded J1.
        sign_pref = -1.0
        j4_c = float(self.cfg.elbow_center_rad)
        best_d: float | None = None
        best_cost = float("inf")
        fallback_d: float | None = None
        fallback_cost = float("inf")
        for d in samples:
            pack = self._eval_at_split(pose, float(psi), float(d))
            if pack is None:
                continue
            q_arm = pack[0]
            j4 = float(q_arm[3])
            j1 = float(q_arm[0])
            if self._j4_illegal_at_stop(j4, has_travel=has_travel):
                continue
            sign_pen = 0.0
            if abs(j1) > np.deg2rad(10.0) and j1 * sign_pref < 0.0:
                sign_pen = 10.0
            cost = abs(float(d) - d_pref) + 0.15 * abs(j4 - j4_c) + sign_pen
            if cost < fallback_cost:
                fallback_cost = float(cost)
                fallback_d = float(d)
            if not self._j4_in_design_band(j4, loose=False):
                continue
            if cost < best_cost:
                best_cost = float(cost)
                best_d = float(d)
        if best_d is not None:
            return float(best_d)
        return fallback_d

    def _maybe_retarget_psi(
        self,
        q: np.ndarray,
        *,
        dt_s: float,
        rail_lo: float,
        rail_hi: float,
    ) -> None:
        dt = max(float(dt_s), 0.0)
        self._search_age_s += dt
        period = max(float(self.cfg.psi_replan_period_s), 0.0)
        due = self._search_age_s + 1.0e-12 >= period
        q_arm = np.asarray(q, dtype=float).reshape(-1)
        if q_arm.size == 8:
            q_arm = q_arm[1:]
        j4 = abs(float(q_arm[3]))
        j6 = abs(float(q_arm[5]))
        attr = clamp_psi_to_envelope(
            float(self.cfg.psi_attr_rad),
            self.cfg.psi_envelope_lo_rad,
            self.cfg.psi_envelope_hi_rad,
        )
        # SEW is undefined near a straight elbow; searching ψ there flipped
        # the family on 035411 (J4 through 0, ψ 39°→−141°).
        if j4 < float(self.cfg.psi_envelope_lo_rad):
            return
        wrist_bad = j6 < float(self.cfg.psi_wrist_ok_rad)
        if wrist_bad:
            self._healthy_dwell_s = 0.0
            if not due:
                return
            self._search_age_s = 0.0
            found = self.search_psi_at_pose(q, rail_lo=rail_lo, rail_hi=rail_hi)
            self.last_psi_search_count += 1
            if found is None:
                return
            self._psi_star = float(found)
            self.psi_star_rad = float(found)
            return
        self._healthy_dwell_s += dt
        if due:
            self._search_age_s = 0.0
        dwell = max(float(self.cfg.psi_return_dwell_s), 0.0)
        if self._healthy_dwell_s + 1.0e-12 >= dwell:
            self._psi_star = float(attr)
            self.psi_star_rad = float(attr)

    def _psi_infeasible_at(
        self,
        q_rad: np.ndarray,
        psi: float,
        *,
        rail_lo: float,
        rail_hi: float,
    ) -> bool:
        q = np.asarray(q_rad, dtype=float)
        pose = np.asarray(self.kin.fk_pose(q), dtype=float).reshape(6)
        d_c = (
            float(self._d_star)
            if self._d_star is not None
            else d_from_q(self.kin, q)
        )
        y_rail = float(pose[1]) - d_c
        margin = max(float(self.cfg.rail_margin_m), 0.0)
        if y_rail < float(rail_lo) + margin or y_rail > float(rail_hi) - margin:
            return True
        pack = self._eval.evaluate(
            pose, float(psi), int(branch_from_q(q)), y_rail
        )
        return pack is None

    def search_psi_at_pose(
        self,
        q_rad: np.ndarray,
        *,
        rail_lo: float,
        rail_hi: float,
    ) -> float | None:
        """Best ψ in the local envelope window at the current TCP, or None.

        Score is wrist openness plus joint margin.  Samples stay inside
        ``[psi_envelope_lo, psi_envelope_hi]`` so the family never crosses 0.
        """
        q = np.asarray(q_rad, dtype=float)
        pose = np.asarray(self.kin.fk_pose(q), dtype=float).reshape(6)
        branch = int(branch_from_q(q))
        d_c = (
            float(self._d_star)
            if self._d_star is not None
            else d_from_q(self.kin, q)
        )
        y_rail = float(pose[1]) - d_c
        margin = max(float(self.cfg.rail_margin_m), 0.0)
        if y_rail < float(rail_lo) + margin or y_rail > float(rail_hi) - margin:
            return None
        lo = float(self.cfg.psi_envelope_lo_rad)
        hi = float(self.cfg.psi_envelope_hi_rad)
        center = (
            float(self._psi_star)
            if self._psi_star is not None
            else clamp_psi_to_envelope(float(psi_from_q(q)), lo, hi)
        )
        center = clamp_psi_to_envelope(center, lo, hi)
        half = max(float(self.cfg.psi_search_half_span_rad), 0.0)
        n = max(int(self.cfg.psi_search_n), 3)
        raw = np.linspace(center - half, center + half, n)
        local = np.unique(
            np.array([clamp_psi_to_envelope(p, lo, hi) for p in raw], dtype=float)
        )
        best_psi, best_j6 = self._score_psi_samples(
            local, pose=pose, branch=branch, y_rail=y_rail
        )
        wrist_ok = float(self.cfg.psi_wrist_ok_rad)
        if best_psi is None or not np.isfinite(best_j6) or best_j6 < wrist_ok:
            full = np.linspace(lo, hi, n)
            best_full, j6_full = self._score_psi_samples(
                full, pose=pose, branch=branch, y_rail=y_rail
            )
            if best_full is not None and (
                best_psi is None or j6_full > best_j6 + 1.0e-9
            ):
                best_psi, best_j6 = best_full, j6_full
        self.last_search_j6_rad = best_j6
        return best_psi

    def _score_psi_samples(
        self,
        samples: np.ndarray,
        *,
        pose: np.ndarray,
        branch: int,
        y_rail: float,
    ) -> tuple[float | None, float]:
        best_s = -np.inf
        best_psi: float | None = None
        best_j6 = float("nan")
        for psi in samples:
            pack = self._eval.evaluate(pose, float(psi), branch, y_rail)
            if pack is None:
                continue
            q_arm, q_full, _sigma = pack
            j6 = abs(float(q_arm[5]))
            if j6 < float(self.cfg.wrist_min_rad) - 1.0e-9:
                continue
            marg = float(np.min(np.minimum(q_arm - Q_LOWER, Q_UPPER - q_arm)))
            score = min(j6 / (60.0 * np.pi / 180.0), 1.0) + 0.8 * min(
                marg / (30.0 * np.pi / 180.0), 1.0
            )
            if score > best_s + 1.0e-9:
                best_s = float(score)
                best_psi = float(psi)
                best_j6 = float(j6)
                self._update_margins(q_full)
                self.last_dpref_score = float(score)
                self.last_psi_score = float(score)
        return best_psi, best_j6

    def nudge_d_star(
        self,
        delta_m: float,
        *,
        y_des_m: float,
        rail_lo: float,
        rail_hi: float,
        dt_s: float = 0.005,
    ) -> float:
        """Shift d* so rail_ff = y_des − d* stays inside the soft travel.

        The clip is a bound, not a step.  ``d_center_rate_m_s`` then slews.
        """
        if self._d_star is None:
            return float("nan")
        y_des = float(y_des_m)
        lo = float(rail_lo)
        hi = float(rail_hi)
        d_lo = y_des - hi
        d_hi = y_des - lo
        if d_lo > d_hi:
            d_lo, d_hi = d_hi, d_lo
        d_new = float(np.clip(float(self._d_star) + float(delta_m), d_lo, d_hi))
        self._d_center_target = d_new
        return self._rate_limit_d(float(dt_s))

    def _rate_limit_d(
        self,
        dt_s: float,
        *,
        y_tcp: float | None = None,
        rail_lo: float | None = None,
        rail_hi: float | None = None,
        d_live: float | None = None,
    ) -> float:
        if self._d_star is None:
            return float("nan")
        target = (
            float(self._d_center_target)
            if self._d_center_target is not None
            else float(self._d_star)
        )
        cur = float(self._d_star)
        err = target - cur
        max_step = max(float(self.cfg.d_center_rate_m_s), 0.0) * max(float(dt_s), 0.0)
        if max_step > 0.0 and abs(err) > max_step:
            err = float(np.clip(err, -max_step, max_step))
        new_d = float(cur + err)
        if (
            y_tcp is not None
            and rail_lo is not None
            and rail_hi is not None
            and np.isfinite(float(y_tcp))
        ):
            margin = max(float(self.cfg.rail_margin_m), 0.0)
            y_lo = float(rail_lo) + margin
            y_hi = float(rail_hi) - margin
            if y_lo > y_hi + 1.0e-12:
                if d_live is not None and np.isfinite(float(d_live)):
                    self._d_star = float(d_live)
                self.d_star_m = float(self._d_star)
                return float(self._d_star)
            d_lo = float(y_tcp) - y_hi
            d_hi = float(y_tcp) - y_lo
            if d_lo > d_hi + 1.0e-12:
                if d_live is not None and np.isfinite(float(d_live)):
                    self._d_star = float(d_live)
                self.d_star_m = float(self._d_star)
                return float(self._d_star)
            new_d = float(np.clip(new_d, d_lo, d_hi))
        self._d_star = new_d
        self.d_star_m = float(self._d_star)
        return float(self._d_star)

    def _rate_limit_psi(
        self, dt_s: float, live_psi: float | None = None
    ) -> float:
        target = fold_psi_to_positive(
            float(self._psi_star if self._psi_star is not None else 0.0)
        )
        cur = fold_psi_to_positive(
            float(self._psi_cmd if self._psi_cmd is not None else target)
        )
        err = psi_err_avoiding_zero(cur, target)
        max_step = float(self.cfg.psi_rate_rad_s) * dt_s
        if max_step > 0.0 and abs(err) > max_step:
            err = float(np.clip(err, -max_step, max_step))
        nxt = float(cur + err)
        # Never publish a command that sits on the wrong side of 0.
        if cur * nxt < 0.0 and abs(cur) > 1.0e-6:
            nxt = float(np.sign(cur) * 1.0e-6)
        nxt = fold_psi_to_positive(nxt)
        lead = max(float(self.cfg.psi_cmd_lead_rad), 0.0)
        if (
            lead > 0.0
            and live_psi is not None
            and np.isfinite(float(live_psi))
        ):
            live = fold_psi_to_positive(float(live_psi))
            lead_nxt = abs(psi_err_avoiding_zero(live, nxt))
            lead_cur = abs(psi_err_avoiding_zero(live, cur))
            if lead_nxt > lead + 1.0e-12 and lead_nxt > lead_cur + 1.0e-12:
                nxt = cur
        self._psi_cmd = nxt
        return float(self._psi_cmd)


__all__ = [
    "PostureRetarget",
    "PsiRetargetConfig",
    "StrokeInfeasibleError",
    "arm_respects_floor",
    "clamp_psi_to_envelope",
    "d_from_q",
    "design_family_ok",
    "fold_psi_to_positive",
    "joint_margin_frac",
    "nearest_planar_psi",
    "psi_err_avoiding_zero",
    "stroke_score",
    "wrist_band_frac",
]
```

## 4. `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/rail_allocator.py`

- sha256：`4dfc442f40ff579dad3c9bf656aa0ef22273857a88ed8a7e7c4eee679cea757b`
- 行数：484

```python
"""Closed-form 8-DoF rail allocation, 20 Hz reference model, and 200 Hz observer.

L1 produces a committed rail velocity ``v_r,ref``.  It is *not* a TCP
closed loop: the arm still solves ``J_a q̇_a = v_d − J_r v̂_r`` in QP1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.filters import (
    first_order_lpf,
    lpf_tau_from_fc,
)
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    stopping_velocity,
    wall_cap,
)


@dataclass
class RailAllocatorConfig:
    """L1 rail allocation + VPC mid-ranging.  Always on in COUPLED mode."""

    # Task-side scale: metres/s and rad/s so v and ω share one residual.
    v0_m_s: float = 0.05
    w0_rad_s: float = 0.30
    # Chan-Dubey: near-limit joints get larger margin_weight → smaller W^{-1}.
    k_margin: float = 4.0
    # VPC mid-ranging (Ma 2015 C_s).  Error is Cartesian d = y_tcp − y_rail − d*.
    kp_mid: float = 1.2
    ki_mid: float = 0.80
    u_mid_max_m_s: float = 0.12
    # Haviland 2022 eq (14): cheapen the rail when |e_mid| is large.
    k_err_rail: float = 4.0
    e_ref_m: float = 0.08
    # Reference-model cutoff.  τ = 1/(2π f_c).  Rail is a low-frequency
    # actuator. Yaml is 2 Hz after the live-Y e_mid fix; do not jump to 5–10 Hz.
    f_c_hz: float = 1.0
    kaw_mid: float = 8.0
    rho_mirror_a: float = 0.50
    rho_mirror_j: float = 0.30
    # One-sided braking envelope (same formula as the worker override).
    reaction_s: float = 0.06
    observer_pos_gain: float = 0.35
    observer_vel_gain: float = 2.0
    observer_vel_lpf_hz: float = 8.0


def allocate_rail(
    J: np.ndarray,
    v_d: np.ndarray,
    *,
    qdot_scale: np.ndarray,
    margin_weight: np.ndarray,
    lam: float,
    v0_m_s: float = 0.05,
    w0_rad_s: float = 0.30,
    e_mid: float = 0.0,
    k_err: float = 0.0,
    e_ref: float = 0.08,
) -> tuple[float, np.ndarray]:
    """Weighted damped least-norm: ``q̇ = W⁻¹ J_nᵀ (J_n W⁻¹ J_nᵀ + λ²I)⁻¹ v_n``.

    ``qdot_scale`` is ``[v_r_max, q̇_max_1..7]``.  ``margin_weight`` is
    Chan-Dubey (≥1); larger means more expensive.  Returns ``(u_r, q̇)``.
    """
    J = np.asarray(J, dtype=float)
    v = np.asarray(v_d, dtype=float).reshape(-1)
    if J.shape[0] != 6 or v.size != 6:
        raise ValueError("allocate_rail expects a 6×n Jacobian and a 6-vector v_d")
    s = np.asarray(qdot_scale, dtype=float).reshape(-1)
    mw = np.asarray(margin_weight, dtype=float).reshape(-1)
    if s.size != J.shape[1] or mw.size != J.shape[1]:
        raise ValueError("qdot_scale / margin_weight must match Jacobian columns")
    scale = np.array(
        [v0_m_s, v0_m_s, v0_m_s, w0_rad_s, w0_rad_s, w0_rad_s], dtype=float
    )
    scale = np.maximum(scale, 1.0e-9)
    v_n = v / scale
    J_n = J / scale[:, None]
    Winv_diag = (s * s) / np.maximum(mw, 1.0e-9)
    # Haviland 2022 eq (14): base cheap when the mid-ranging error is large.
    if float(k_err) > 0.0:
        gain = 1.0 + float(k_err) * min(
            abs(float(e_mid)) / max(float(e_ref), 1.0e-9), 1.0
        )
        Winv_diag[0] *= gain * gain
    JW = J_n * Winv_diag[None, :]
    a = JW @ J_n.T
    lam2 = float(lam) * float(lam)
    a.flat[::7] += lam2
    try:
        y = np.linalg.solve(a, v_n)
    except np.linalg.LinAlgError:
        y = np.linalg.lstsq(a, v_n, rcond=None)[0]
    qdot = Winv_diag * (J_n.T @ y)
    return float(qdot[0]), qdot


@dataclass
class RailReferenceState:
    v: float = 0.0
    a: float = 0.0
    initialized: bool = False


class RailReferenceModel:
    """Δt-adaptive first-order LPF, then hard |a| / |j| boxes, then wall cap.

    History is the *committed* ``v_r,ref`` so the next tick's boxes stay
    consistent with what the worker actually received.
    """

    def __init__(
        self,
        *,
        f_c_hz: float = 1.0,
        a_max: float = 0.60,
        j_max: float = 60.0,
        v_max: float = 0.12,
        reaction_s: float = 0.06,
        soft_min_m: float = 0.015,
        soft_max_m: float = 0.77,
        hard_min_m: float | None = None,
        hard_max_m: float | None = None,
    ) -> None:
        self.f_c_hz = float(f_c_hz)
        self.a_max = float(a_max)
        self.j_max = float(j_max)
        self.v_max = float(v_max)
        self.reaction_s = float(reaction_s)
        self.soft_min_m = float(soft_min_m)
        self.soft_max_m = float(soft_max_m)
        self.hard_min_m = float(soft_min_m if hard_min_m is None else hard_min_m)
        self.hard_max_m = float(soft_max_m if hard_max_m is None else hard_max_m)
        self.state = RailReferenceState()
        self.last_wall_override = False

    def reset(self, v0: float = 0.0) -> None:
        self.state = RailReferenceState(v=float(v0), a=0.0, initialized=False)
        self.last_wall_override = False

    def step(
        self,
        u_r: float,
        dt_s: float,
        *,
        x_m: float,
        apply_wall: bool = True,
        a_max: float | None = None,
        j_max: float | None = None,
    ) -> float:
        dt = float(dt_s)
        if dt <= 1.0e-9:
            return float(self.state.v)
        a_lim = float(self.a_max if a_max is None else min(self.a_max, abs(float(a_max))))
        j_lim = float(self.j_max if j_max is None else min(self.j_max, abs(float(j_max))))
        tau = lpf_tau_from_fc(self.f_c_hz)
        u = float(u_r)
        if not self.state.initialized:
            v_f = u
            self.state.initialized = True
        elif tau <= 1.0e-9:
            v_f = u
        else:
            v_f = first_order_lpf(float(self.state.v), u, dt, tau)
        v_prev = float(self.state.v)
        a_prev = float(self.state.a)
        a_raw = (v_f - v_prev) / dt
        da_max = float(j_lim) * dt
        a = float(np.clip(a_raw, a_prev - da_max, a_prev + da_max))
        a = float(np.clip(a, -a_lim, a_lim))
        v = v_prev + a * dt
        v = float(np.clip(v, -self.v_max, self.v_max))
        self.last_wall_override = False
        if apply_wall:
            lo_cap, hi_cap = wall_cap(
                float(x_m),
                lo=self.hard_min_m,
                hi=self.hard_max_m,
                a_max=a_lim,
                reaction_s=self.reaction_s,
            )
            v_clamped = float(np.clip(v, lo_cap, hi_cap))
            if abs(v_clamped - v) > 1.0e-9:
                self.last_wall_override = True
            v = v_clamped
            a = (v - v_prev) / dt
        if abs(v) < 5.0e-4 and abs(u) < 5.0e-4:
            v = 0.0
            a = 0.0
        self.state.v = float(v)
        self.state.a = float(a)
        return float(v)


class RailStateObserver:
    """200 Hz output: predict with ``v_r,ref``, correct on timestamped encoder.

    This estimates 0–10 Hz rail motion.  It is not a 50 Hz velocity sensor.
    """

    def __init__(
        self,
        *,
        pos_gain: float = 0.35,
        vel_gain: float = 2.0,
        vel_lpf_hz: float = 8.0,
        v_max: float = 0.30,
    ) -> None:
        self.pos_gain = float(pos_gain)
        self.vel_gain = float(vel_gain)
        self.vel_lpf_hz = float(vel_lpf_hz)
        self.v_max = float(v_max)
        self.q_hat = 0.0
        self.v_hat = 0.0
        self._last_sample_t: float | None = None
        self._initialized = False

    def reset(self, q0: float = 0.0, v0: float = 0.0) -> None:
        self.q_hat = float(q0)
        self.v_hat = float(v0)
        self._last_sample_t = None
        self._initialized = True

    def update(
        self,
        *,
        now_s: float,
        dt_s: float,
        v_r_ref: float,
        q_meas: float,
        sample_t: float,
        v_meas: float | None = None,
        v_written: float | None = None,
    ) -> tuple[float, float]:
        if not self._initialized:
            self.reset(q_meas, float(v_meas) if v_meas is not None else 0.0)
            self._last_sample_t = float(sample_t)
            return float(self.q_hat), float(self.v_hat)
        dt = max(float(dt_s), 1.0e-6)
        # Predict with the last written FA24 / measured RPM, never the
        # internal v_r,ref.  Using v_r_ref made the observer optimistic and
        # the arm compensated a rail that had not actually moved.
        if v_written is not None and np.isfinite(float(v_written)):
            v_pred = float(v_written)
        elif v_meas is not None and np.isfinite(float(v_meas)):
            v_pred = float(v_meas)
        else:
            v_pred = float(self.v_hat)
        del v_r_ref
        self.q_hat = float(self.q_hat) + v_pred * dt
        tau = lpf_tau_from_fc(self.vel_lpf_hz)
        if tau <= 1.0e-9:
            self.v_hat = v_pred
        else:
            self.v_hat = first_order_lpf(float(self.v_hat), v_pred, dt, tau)
        if np.isfinite(sample_t) and (
            self._last_sample_t is None or float(sample_t) > float(self._last_sample_t) + 1.0e-9
        ):
            age = max(0.0, float(now_s) - float(sample_t))
            q_pred_at_sample = float(self.q_hat) - v_pred * age
            innov = float(q_meas) - q_pred_at_sample
            self.q_hat += self.pos_gain * innov
            self.v_hat += self.vel_gain * innov
            if v_meas is not None and np.isfinite(float(v_meas)):
                blend = min(1.0, dt * 8.0)
                self.v_hat = (1.0 - blend) * float(self.v_hat) + blend * float(v_meas)
            self._last_sample_t = float(sample_t)
        self.v_hat = float(np.clip(self.v_hat, -self.v_max, self.v_max))
        return float(self.q_hat), float(self.v_hat)


def margin_weight_from_activation(
    q: np.ndarray,
    q_mid: np.ndarray,
    half: np.ndarray,
    *,
    k_margin: float,
    activation: float,
) -> np.ndarray:
    """Per-joint Chan-Dubey weight.  Rail uses the same formula in metres."""
    q = np.asarray(q, dtype=float)
    mid = np.asarray(q_mid, dtype=float)
    h = np.maximum(np.asarray(half, dtype=float), 1.0e-9)
    u = np.clip(np.abs(q - mid) / h, 0.0, 1.0)
    span = max(1.0 - float(activation), 1.0e-6)
    over = np.clip((u - float(activation)) / span, 0.0, 1.0)
    return 1.0 + float(k_margin) * over * over


def soft_saturate(value: float, limit: float) -> float:
    """``limit * tanh(value / limit)``.  Keeps a gradient at the cap."""

    lim = max(float(limit), 1.0e-9)
    return float(lim * np.tanh(float(value) / lim))


class MidrangingController:
    """PI on Cartesian mid-ranging error ``e_mid = (y_tcp − y_rail) − d*``."""

    def __init__(
        self,
        *,
        kp: float = 1.2,
        ki: float = 0.80,
        v_max: float = 0.12,
        kaw: float = 8.0,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.v_max = float(v_max)
        self.kaw = float(kaw)
        self.integ = 0.0
        self.last_raw = 0.0
        self.last_projected = False

    def reset(self) -> None:
        self.integ = 0.0
        self.last_raw = 0.0
        self.last_projected = False

    def step(
        self,
        err_m: float,
        dt_s: float,
        *,
        freeze: bool = False,
        leave_only_sign: float = 0.0,
        u_committed: float | None = None,
    ) -> float:
        """Return saturated mid-ranging velocity.

        ``leave_only_sign`` > 0 at the plus hard wall (only negative u_mid),
        < 0 at the minus hard wall (only positive u_mid).  0 leaves u_mid
        unconstrained.  Integrator anti-windup uses back-calculation against
        the committed rail command when one is supplied.
        """
        err = float(err_m) if np.isfinite(err_m) else 0.0
        dt = max(float(dt_s), 0.0)
        if not freeze and dt > 0.0:
            self.integ += self.ki * err * dt
        raw = self.kp * err + self.integ
        sat = soft_saturate(raw, self.v_max)
        sign = float(leave_only_sign)
        projected = False
        if sign > 0.0 and sat > 0.0:
            sat = 0.0
            projected = True
        elif sign < 0.0 and sat < 0.0:
            sat = 0.0
            projected = True
        self.last_raw = float(raw)
        self.last_projected = bool(projected)
        if freeze:
            return float(sat)
        if dt > 0.0 and u_committed is not None and np.isfinite(float(u_committed)):
            self.integ += self.kaw * (float(u_committed) - float(raw)) * dt
        elif not freeze and abs(raw) > self.v_max:
            self.integ -= self.ki * err * dt
        if projected and dt > 0.0:
            # Do not keep integrating a command that the wall already killed.
            self.integ -= self.ki * err * dt
        return float(sat)


def wall_leave_only_sign(
    x_m: float,
    *,
    hard_min_m: float,
    hard_max_m: float,
    band_m: float,
) -> float:
    """+1 near the plus hard wall (only leave/negative u), -1 near minus, else 0."""
    x = float(x_m)
    band = max(float(band_m), 0.0)
    hi = float(hard_max_m)
    lo = float(hard_min_m)
    if x >= hi - band:
        return 1.0
    if x <= lo + band:
        return -1.0
    return 0.0


def arm_mirror_rail_limits(
    J: np.ndarray,
    a_arm_max: np.ndarray,
    j_arm_max: np.ndarray,
    *,
    rho_a: float = 0.50,
    rho_j: float = 0.30,
) -> tuple[float, float]:
    """Max |a_r|, |j_r| the arm can still mirror: qa = −Ja# Jr vr."""
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or J.shape[0] < 1 or J.shape[1] < 2:
        return float("inf"), float("inf")
    ja = J[:, 1:]
    jr = J[:, 0]
    try:
        p, *_ = np.linalg.lstsq(ja, jr, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf"), float("inf")
    p = np.abs(np.asarray(p, dtype=float).reshape(-1))
    a_arm = np.abs(np.asarray(a_arm_max, dtype=float).reshape(-1))
    j_arm = np.abs(np.asarray(j_arm_max, dtype=float).reshape(-1))
    a_lim = float("inf")
    j_lim = float("inf")
    n = min(p.size, a_arm.size)
    for i in range(n):
        if p[i] <= 1.0e-6:
            continue
        a_lim = min(a_lim, float(rho_a) * float(a_arm[i]) / float(p[i]))
    n_j = min(p.size, j_arm.size)
    for i in range(n_j):
        if p[i] <= 1.0e-6:
            continue
        j_lim = min(j_lim, float(rho_j) * float(j_arm[i]) / float(p[i]))
    if not np.isfinite(a_lim):
        a_lim = float("inf")
    if not np.isfinite(j_lim):
        j_lim = float("inf")
    return float(max(a_lim, 0.0)), float(max(j_lim, 0.0))


def project_arm_compensation(
    J: np.ndarray,
    delta_v_req: np.ndarray,
    q: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    *,
    activation: float = 0.80,
    alpha: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Tu 2022 eq. (22): drop compensation that drives the arm into limits."""

    J = np.asarray(J, dtype=float)
    req = np.asarray(delta_v_req, dtype=float).reshape(-1)
    if J.ndim != 2 or J.shape[0] != req.size or J.shape[1] < 2:
        return req.copy(), 0.0
    J_a = J[:, 1:]
    try:
        qdot_a, *_ = np.linalg.lstsq(J_a, req, rcond=None)
    except np.linalg.LinAlgError:
        return req.copy(), 0.0
    q_a = np.asarray(q, dtype=float).reshape(-1)[1 : 1 + qdot_a.size]
    lo = np.asarray(q_lower, dtype=float).reshape(-1)[1 : 1 + qdot_a.size]
    hi = np.asarray(q_upper, dtype=float).reshape(-1)[1 : 1 + qdot_a.size]
    if q_a.size != qdot_a.size:
        return req.copy(), 0.0
    half = np.maximum(0.5 * (hi - lo), 1.0e-9)
    mid = 0.5 * (hi + lo)
    u = (q_a - mid) / half
    toward_limit = (u * qdot_a) > 0.0
    near = np.abs(u) >= float(activation)
    mask = toward_limit & near
    qdot_p = np.asarray(qdot_a, dtype=float).copy()
    qdot_p[mask] *= 1.0 - float(np.clip(alpha, 0.0, 1.0))
    cmp = J_a @ qdot_p
    nreq = float(np.linalg.norm(req))
    frac = 0.0 if nreq < 1.0e-12 else float(1.0 - np.linalg.norm(cmp) / nreq)
    return np.asarray(cmp, dtype=float), float(np.clip(frac, 0.0, 1.0))


__all__ = (
    "MidrangingController",
    "RailAllocatorConfig",
    "RailReferenceModel",
    "RailReferenceState",
    "RailStateObserver",
    "allocate_rail",
    "arm_mirror_rail_limits",
    "lpf_tau_from_fc",
    "margin_weight_from_activation",
    "project_arm_compensation",
    "soft_saturate",
    "stopping_velocity",
    "wall_cap",
    "wall_leave_only_sign",
)
```

## 5. `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/rail_extension.py`

- sha256：`7b4aa9103d9960b1f012fcdc3abc2db05dfd10b96baacd320701343203a98c05`
- 行数：818

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

from rm75_control.control.joint_admittance_8dof.filters import smoothstep01
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
    jacobian: np.ndarray | None = None,
) -> float:
    """Scalar rail speed from any reference ``vel_ff`` (base-frame linear vel).

    Projects the reference linear velocity onto the rail Jacobian column —
    works for sin, spline, hold-to-move, or any ``MotionReference`` that
    populates ``vel_ff[:3]`` in the base frame (as all current sources do).
    """
    v_lin = np.asarray(vel_ff[:3], dtype=float)
    if jacobian is not None:
        j_rail = np.asarray(jacobian, dtype=float)[:3, RAIL_INDEX]
    else:
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
    limit_margin_m: float = 0.15
    # Hard pin / +q0 end-flip only this close to a soft stop (not the fade).
    pin_margin_m: float = 0.008
    # Stop driving +q0 this far from soft_max so escape cannot dump the carriage
    # onto the +stop (174417 sat at 774 mm for 52 s, Y error 340 mm).
    escape_leave_m: float = 0.04
    # Host soft travel (not URDF 0/0.8). Fade and end-flip use these.
    soft_min_m: float = 0.025
    soft_max_m: float = 0.78
    # Reach may oppose MotionReference FF, but only this much (m/s) so the
    # rail can still re-extend the elbow without re-triggering LW100 Er-01.
    v_reach_cap_m_s: float = 0.05
    # Operator idle: posture reach must not drag the rail at 50 mm/s.
    v_reach_idle_cap_m_s: float = 0.010
    # Raw σ at or above this mutes escape unless a press stall needs Y.
    healthy_sigma_mute: float = 0.08
    # Dead-zone around d_center.  Coupled mode is velocity-authoritative,
    # so this is the only Cartesian position term on the rail axis.  Keep
    # it small enough that a few millimetres of track error still produce
    # v_reach; 80 mm used to kill the term on every healthy scan.
    d_band_m: float = 0.005
    # Bug 2: σ-escape.  When σ_min ↘ the rail should BOOST authority (not
    # cut it — the old ``w *= sigma_scale`` was backwards) and add a
    # non-reaching velocity component along the TCP-preserving σ-ascent
    # direction so the rail acts even inside the reach dead zone.
    #
    # Invariant kept by callers: ``w_max * (1 + k_sigma_boost) ≪ W_task``
    # (default 1.5 * 3 = 4.5 vs W_task = 100 in yaml → 22:1 ratio).  This is
    # what keeps the QP preference order  ``slack > rail > free-arm``
    # untouched even during σ dips (§3 test 1 & 2 in the plan pin this).
    k_sigma_boost: float = 2.0
    # k_esc [m/s per unit σ]: scales the σ-escape velocity component.
    # sigma_grad_rail has units 1/m, so k_esc·(1-sig)·grad has units of m/s.
    # Healthy path uses continuous soft bias (dbb/4d); latch uses same gain.
    k_esc: float = 0.5
    # Baseline w that lets the rail act even when the reach error is inside
    # the dead zone (|e| < e0), provided σ is depressed.  Fades with σ.
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
    v_lpf_tau_s: float = 0.05
    # When > 0, ``v_lpf_tau_s`` is derived as 1/(2π f_c).  0 keeps the raw tau.
    v_lpf_fc_hz: float = 0.0
    # Faster LPF while escape is latched (commit without hunting).
    v_lpf_tau_escape_s: float = 0.04
    # Narrow latch: only deep σ (scale) or truly near joint soft limits.
    sigma_escape_enter: float = 0.55
    sigma_escape_exit: float = 0.80
    margin_escape_enter: float = 0.12
    margin_escape_exit: float = 0.25
    # Latch when raw arm σ falls faster than this (1/s); 0 disables.
    sigma_drop_rate: float = 0.0
    # Require sustained want_enter before latching (blocks turnaround flashes).
    escape_enter_dwell_s: float = 0.05
    # Extra weight multiplier while escape latched (still capped by w_ext_cap).
    k_escape_boost: float = 1.2
    # Floor |grad| when latched without a usable grad; 0 = never invent |grad|.
    escape_grad_floor: float = 0.0
    # Boost rail soft weight when any arm joint is near its soft limit [0,1].
    k_margin_boost: float = 4.0
    w_ext_cap: float = 24.0  # still ≪ W_task=100
    # When |err_band| exceeds err0, raise rail Cartesian reg and fade k_ff
    # so the arm takes Y instead of stretching the split further.
    d_star_err0_m: float = 0.01
    d_star_err1_m: float = 0.04
    d_star_w_mult: float = 6.0
    d_star_reg_mult: float = 20.0
    # Press-stall lateral escape: keep σ-escape / d* nudge alive when Z is
    # still demanding and the carriage still has travel.  Y error is not a
    # gate — mid-stroke stalls often track Y to < 5 mm.
    press_v_force_min_m_s: float = 0.02
    press_dz_max_m: float = 0.002
    press_y_err_m: float = 0.005
    press_stall_s: float = 0.5
    d_star_nudge_m: float = 0.01
    open_travel_min_m: float = 0.01
    # One-sided lateral escape.  ``minus`` drives −q0 until the minus pin.
    escape_sign_policy: str = "auto"
    # Budget for the reach path's ``v_ff + v_reach + v_escape`` sum.  It must
    # leave room for a legal FF *plus* reach, or the two saturate together and
    # reach never runs: gamepad demands 120 mm/s of FF against a 80 mm/s
    # ``v_max_m_s``, so the 40 mm/s shortfall grew the posture error at
    # 39 mm/s until the stick was released and it dumped in one 1 s slide.
    # ``None`` keeps the old shared cap.  The real speed limit is the QP rail
    # box and the FA24 clamp, not this.
    v_reach_total_max_m_s: float | None = None

    def reach_budget_m_s(self) -> float:
        """Total velocity budget for the reach path."""
        if self.v_reach_total_max_m_s is None:
            return float(self.v_max_m_s)
        return max(float(self.v_reach_total_max_m_s), float(self.v_max_m_s))


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
        self.last_in_limit_band: bool = False
        self._guard_active: bool = False
        self._escape_active: bool = False
        self._escape_sign: float = 0.0
        self._escape_flipped_at_end: bool = False
        self._escape_enter_timer_s: float = 0.0
        self._sigma_raw_prev: float | None = None
        self._v_lpf: float = 0.0
        self._v_lpf_initialized: bool = False
        self.last_v_ff: float = 0.0
        self.last_v_escape: float = 0.0
        self.last_v_reach: float = 0.0
        self.last_e_mid_m: float = 0.0
        self._escape_grad_hint: float = 0.0
        self.last_rail_ff_m: float = float("nan")
        self.last_track_err_m: float = 0.0
        self.last_d_star_reg_scale: float = 1.0
        self.last_k_ff_scale: float = 1.0

    def set_mode(self, mode: RailExtMode) -> None:
        mode_s = str(mode).strip().lower()
        if mode_s not in ("reach", "pose_attract"):
            raise ValueError(f"unknown rail extension mode {mode!r}")
        if mode_s != self.mode:
            # Reset LPF on mode switch so a scan FF residue does not kick move.
            self._v_lpf = 0.0
            self._v_lpf_initialized = False
            self._guard_active = False
            self._escape_active = False
            self._escape_sign = 0.0
            self._escape_flipped_at_end = False
            self._sigma_raw_prev = None
        self.mode = mode_s  # type: ignore[assignment]

    def _soft_travel(self) -> tuple[float, float]:
        """Usable rail band: host soft limits ∩ URDF, never the raw URDF stop."""
        urdf_lo = float(self.kin.q_lower[RAIL_INDEX])
        urdf_hi = float(self.kin.q_upper[RAIL_INDEX])
        lo = max(urdf_lo, float(self.cfg.soft_min_m))
        hi = min(urdf_hi, float(self.cfg.soft_max_m))
        if not (lo < hi):
            return urdf_lo, urdf_hi
        return lo, hi

    def set_rail_pose_target(self, y_rail_m: float | None) -> None:
        """Set / clear the move→D soft attractor target (metres)."""
        if y_rail_m is None:
            self.y_rail_target_m = None
            return
        lo, hi = self._soft_travel()
        self.y_rail_target_m = float(np.clip(float(y_rail_m), lo, hi))

    def set_d_pref(self, d_pref_m: float) -> None:
        """Update the preferred arm-extension offset (metres)."""
        self.d_pref_m = float(d_pref_m)

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
        self.last_e_mid_m = 0.0
        self.last_weight = 0.0
        self.last_limit_saturated = False
        self.last_in_limit_band = False
        self._guard_active = False
        self._escape_active = False
        self._escape_sign = 0.0
        self._escape_flipped_at_end = False
        self._sigma_raw_prev = None
        self._v_lpf = 0.0
        self._v_lpf_initialized = False

    def _rail_in_limit_band(self, q_rail: float) -> bool:
        """True while the carriage sits inside either soft-limit fade band."""
        margin = float(self.cfg.limit_margin_m)
        if margin <= 1.0e-9:
            return False
        lo, hi = self._soft_travel()
        return bool(q_rail <= lo + margin or q_rail >= hi - margin)

    def _open_side_travel_m(self, q_rail: float) -> float:
        lo, hi = self._soft_travel()
        return float(max(q_rail - lo, hi - q_rail))

    def _leave_margin_m(self) -> float:
        return max(float(self.cfg.escape_leave_m), float(self.cfg.pin_margin_m))

    def _policy_escape_sign(self, q_rail: float | None = None) -> float:
        raw = str(getattr(self.cfg, "escape_sign_policy", "auto")).strip().lower()
        if raw in ("minus", "-", "neg", "negative"):
            return -1.0
        if raw in ("plus", "+", "pos", "positive"):
            return 1.0
        if raw not in ("auto", "open", "grad", "gradient"):
            raise ValueError(f"unknown rail_extension.escape_sign_policy: {raw!r}")
        # Hold the latched sign so a σ-gradient flicker cannot reverse a
        # committed escape (monotonic latch).  Open travel / pin logic in
        # ``_preferred_escape_sign`` may still reverse at a dead end.
        if abs(float(self._escape_sign)) > 1.0e-9:
            return 1.0 if self._escape_sign > 0.0 else -1.0
        grad = float(self._escape_grad_hint)
        if abs(grad) > 1.0e-9:
            return 1.0 if grad > 0.0 else -1.0
        y = float(q_rail) if q_rail is not None else float("nan")
        if not np.isfinite(y):
            return 0.0
        lo, hi = self._soft_travel()
        plus_room = hi - y
        minus_room = y - lo
        if plus_room > minus_room + 1.0e-9:
            return 1.0
        if minus_room > plus_room + 1.0e-9:
            return -1.0
        return 0.0

    def _in_leave_band(self, q_rail: float, sign: float = 0.0) -> bool:
        lo, hi = self._soft_travel()
        leave = self._leave_margin_m()
        s = float(sign)
        if abs(s) < 1.0e-12:
            s = self._policy_escape_sign(q_rail)
        if s > 0.0:
            return bool(q_rail >= hi - leave)
        if s < 0.0:
            return bool(q_rail <= lo + leave)
        return False

    def _in_plus_leave(self, q_rail: float) -> bool:
        return self._in_leave_band(q_rail, +1.0)

    def _preferred_escape_sign(
        self,
        q_rail: float,
        *,
        backoff: bool = False,
        unload_sign: float = 0.0,
    ) -> float:
        """Policy-side escape; 0 in that leave band; reverse on the policy pin.

        When the elbow is past the design band and the rail still has travel,
        ``unload_sign`` overrides the fixed minus/plus policy so the macro
        pulls live d toward the feasible split.
        """
        sign = self._policy_escape_sign(q_rail)
        if abs(float(unload_sign)) > 1.0e-12:
            sign = 1.0 if float(unload_sign) > 0.0 else -1.0
        lo, hi = self._soft_travel()
        pin = float(self.cfg.pin_margin_m)
        leave = self._leave_margin_m()
        if sign < 0.0:
            if pin > 1.0e-9 and q_rail <= lo + pin:
                return 1.0
            if q_rail <= lo + leave:
                return 1.0 if backoff else 0.0
            return -1.0
        if pin > 1.0e-9 and q_rail >= hi - pin:
            return -1.0
        if q_rail >= hi - leave:
            return -1.0 if backoff else 0.0
        return 1.0

    def _rail_has_open_travel(self, q_rail: float) -> bool:
        return self._open_side_travel_m(q_rail) > float(self.cfg.open_travel_min_m)

    def _rail_end_blocks(self, q_rail: float, sign: float) -> bool:
        """True if moving with ``sign`` (+1/−1) points into the pin band."""
        margin = float(self.cfg.pin_margin_m)
        lo, hi = self._soft_travel()
        if margin <= 1e-9:
            return False
        if sign > 0.0 and q_rail >= hi - margin:
            return True
        if sign < 0.0 and q_rail <= lo + margin:
            return True
        return False

    def _maybe_flip_escape_at_rail_end(self, q_rail: float) -> None:
        """If latched into a dead end, flip sign once (still monotonic)."""
        if not self._escape_active or abs(self._escape_sign) < 1e-9:
            return
        if not self._rail_end_blocks(q_rail, self._escape_sign):
            return
        alt = -self._escape_sign
        if self._rail_end_blocks(q_rail, alt):
            # Both ends blocked — drop escape; L0 box + softσ handle the rest.
            self._escape_active = False
            self._escape_sign = 0.0
            return
        if not self._escape_flipped_at_end:
            self._escape_sign = alt
            self._escape_flipped_at_end = True

    def _clear_escape_latch(self) -> None:
        self._escape_active = False
        self._escape_sign = 0.0
        self._escape_flipped_at_end = False
        self._escape_enter_timer_s = 0.0

    def _escape_latched(
        self,
        *,
        sigma_scale: float,
        sigma_grad_rail: float,
        joint_margin_frac: float,
        sigma_raw: float | None,
        dt_s: float | None,
        q_rail: float,
        trajectory_owns: bool = False,
        unload_sign: float = 0.0,
    ) -> float:
        """Narrow hysteresis latch: deep σ ∪ true near-limit (optional dσ/dt).

        While the MotionReference owns the rail (``|v_ff|>thr``), never enter or
        keep the latch — sticky escape fighting the path caused scan stutter and
        LW100 Er-01 (overspeed) on run_20260813_151334.
        """
        if trajectory_owns:
            self._clear_escape_latch()
            if sigma_raw is not None:
                self._sigma_raw_prev = float(sigma_raw)
            return 0.0

        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        mfrac = float(np.clip(joint_margin_frac, 0.0, 1.0))
        enter = float(self.cfg.sigma_escape_enter)
        exit_ = max(float(self.cfg.sigma_escape_exit), enter)
        m_enter = float(self.cfg.margin_escape_enter)
        m_exit = max(float(self.cfg.margin_escape_exit), m_enter)

        dropping = False
        if (
            sigma_raw is not None
            and dt_s is not None
            and float(dt_s) > 1e-9
            and float(self.cfg.sigma_drop_rate) > 0.0
            and self._sigma_raw_prev is not None
        ):
            dsigma = (float(sigma_raw) - float(self._sigma_raw_prev)) / float(dt_s)
            dropping = dsigma < -float(self.cfg.sigma_drop_rate)
        if sigma_raw is not None:
            self._sigma_raw_prev = float(sigma_raw)

        want_enter = (sig < enter) or (mfrac < m_enter) or dropping
        healthy_exit = (sig >= exit_) and (mfrac >= m_exit)
        dt = float(dt_s) if dt_s is not None and float(dt_s) > 0.0 else 0.0
        dwell = max(float(self.cfg.escape_enter_dwell_s), 0.0)

        if self._escape_active:
            if healthy_exit:
                self._clear_escape_latch()
            else:
                pref = self._preferred_escape_sign(q_rail, unload_sign=unload_sign)
                if abs(pref) < 1.0e-12:
                    self._clear_escape_latch()
                elif pref * self._escape_sign < 0.0:
                    self._escape_sign = pref
        else:
            if want_enter:
                self._escape_enter_timer_s += dt
                if self._escape_enter_timer_s + 1.0e-12 >= dwell:
                    self._escape_active = True
                    self._escape_flipped_at_end = False
                    self._escape_enter_timer_s = 0.0
                    self._escape_sign = self._preferred_escape_sign(
                        q_rail, unload_sign=unload_sign
                    )
                    if abs(self._escape_sign) < 1.0e-12:
                        self._clear_escape_latch()
                        if sigma_raw is not None:
                            self._sigma_raw_prev = float(sigma_raw)
                        return 0.0
            else:
                self._escape_enter_timer_s = 0.0

        if not self._escape_active:
            return 0.0
        self._maybe_flip_escape_at_rail_end(q_rail)
        if not self._escape_active:
            return 0.0
        floor = float(self.cfg.escape_grad_floor)
        mag = abs(float(sigma_grad_rail))
        if floor > 0.0:
            mag = max(mag, floor)
        if mag < 1.0e-12:
            return 0.0
        return self._escape_sign * mag

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

        lo, hi = self._soft_travel()

        if v > 1e-9:
            if q_rail >= hi:
                self.last_limit_saturated = True
                return 0.0
            if q_rail > hi - margin:
                u = float(np.clip((hi - q_rail) / margin, 0.0, 1.0))
                self.last_limit_saturated = False
                return smoothstep01(u)

        elif v < -1e-9:
            if q_rail <= lo:
                self.last_limit_saturated = True
                return 0.0
            if q_rail < lo + margin:
                u = float(np.clip((q_rail - lo) / margin, 0.0, 1.0))
                self.last_limit_saturated = False
                return smoothstep01(u)

        self.last_limit_saturated = False
        return 1.0

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
        enter = float(self.cfg.sigma_guard_enter)
        exit_ = float(self.cfg.sigma_guard_exit)
        if self._guard_active:
            if sig >= exit_:
                self._guard_active = False
        else:
            if sig < enter:
                self._guard_active = True
        if not self._guard_active:
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
        sigma_scale: float,
        sigma_grad_rail: float,
        dt_s: float | None,
    ) -> tuple[float, float]:
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
        w_pose = float(self.cfg.pose_w_max) * smoothstep01((abs(err) - e0) / span)
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
        sigma_grad_rail: float,
        vel_ff: np.ndarray | None,
        dt_s: float | None,
        joint_margin_frac: float = 1.0,
        sigma_raw: float | None = None,
        y_tcp_d: float | None = None,
        press_stalled: bool = False,
        tool_y_err_m: float = 0.0,
        stroke_limiters: bool = True,
        apply_d_band: bool | None = None,
        block_escape: bool = False,
        unload_sign: float = 0.0,
        jacobian: np.ndarray | None = None,
    ) -> tuple[float, float]:
        if self.d_pref_m is None:
            self.capture_reference(q)
        d_star = float(self.d_pref_m)
        y = float(q[RAIL_INDEX])
        self._escape_grad_hint = float(sigma_grad_rail)
        if y_tcp_d is not None and np.isfinite(float(y_tcp_d)):
            y_des = float(y_tcp_d)
        else:
            y_des = float(self.kin.fk_placement(q).translation[1])
        rail_ff = y_des - d_star
        err_raw = rail_ff - y
        band = max(float(getattr(self.cfg, "d_band_m", 0.0)), 0.0)
        use_band = (not stroke_limiters) if apply_d_band is None else bool(apply_d_band)
        if not use_band:
            band = 0.0
        err = float(err_raw - np.clip(err_raw, -band, band))
        self.last_e_mid_m = float(err)
        self.last_rail_ff_m = float(rail_ff)
        self.last_track_err_m = float(err_raw)
        span = max(float(self.cfg.e1_m) - float(self.cfg.e0_m), 1e-6)
        w_reach = float(self.cfg.w_max) * smoothstep01(
            (abs(err) - float(self.cfg.e0_m)) / span
        )
        v_reach = 0.0
        sig = float(np.clip(sigma_scale, 0.0, 1.0))
        err_abs = abs(err)
        e0 = max(float(self.cfg.d_star_err0_m), 0.0)
        e1 = max(float(self.cfg.d_star_err1_m), e0 + 1.0e-6)
        drift = smoothstep01((err_abs - e0) / (e1 - e0)) if e0 > 0.0 else 0.0
        # Haviland eq (14) cheapens the rail in allocate_rail; do not also
        # make the QP rail *more* expensive when |e_mid| is large.
        self.last_d_star_reg_scale = 1.0
        v_ff_measured = (
            rail_vel_ff_from_reference(
                vel_ff, self.kin, q, k_ff=self.cfg.k_ff, jacobian=jacobian
            )
            if vel_ff is not None
            else 0.0
        )
        # Legacy FF is retired: allocate_rail owns task-side rail velocity.
        # Still record the measured feedforward for telemetry / escape latch.
        thr = float(self.cfg.v_ff_thr_m_s)
        ff_owns = abs(v_ff_measured) > thr
        if ff_owns:
            self.last_k_ff_scale = 1.0
            v_ff_att = float(v_ff_measured)
        else:
            self.last_k_ff_scale = 1.0 - drift
            v_ff_att = float(v_ff_measured) * self.last_k_ff_scale
        v_ff = 0.0
        # Trajectory owns rail direction: clear sticky latch (not merely mute v).
        grad_latched = self._escape_latched(
            sigma_scale=sig,
            sigma_grad_rail=sigma_grad_rail,
            joint_margin_frac=joint_margin_frac,
            sigma_raw=sigma_raw,
            dt_s=dt_s,
            q_rail=y,
            trajectory_owns=ff_owns,
            unload_sign=float(unload_sign),
        )
        # Demoted: healthy σ (raw ≥ 0.08) never lets escape drive the rail
        # unless a press stall still needs a lateral Y offset.
        healthy_sigma = (
            sigma_raw is not None
            and float(sigma_raw) >= float(self.cfg.healthy_sigma_mute)
        )
        use_limiters = bool(stroke_limiters)
        in_band = self._rail_in_limit_band(y) if use_limiters else False
        self.last_in_limit_band = bool(in_band)
        y_thr = max(float(self.cfg.press_y_err_m), 0.0)
        policy_sign = self._policy_escape_sign(y)
        backoff = bool(
            use_limiters
            and self._in_leave_band(y, policy_sign)
            and abs(float(tool_y_err_m)) >= y_thr
        )
        allow_press_escape = bool(
            (press_stalled or backoff) and self._rail_has_open_travel(y)
        )
        if block_escape and not allow_press_escape:
            self._clear_escape_latch()
            v_escape = 0.0
        elif in_band and not allow_press_escape:
            self._clear_escape_latch()
            v_escape = 0.0
        elif healthy_sigma and not allow_press_escape:
            self._escape_active = False
            v_escape = 0.0
        elif self._escape_active:
            v_escape = 0.25 * float(self.cfg.k_esc) * float(grad_latched)
        else:
            v_escape = (
                0.25 * float(self.cfg.k_esc) * (1.0 - sig) * float(sigma_grad_rail)
            )
            if allow_press_escape:
                pref = self._preferred_escape_sign(
                    y, backoff=backoff, unload_sign=float(unload_sign)
                )
                v_escape = (
                    0.25
                    * float(self.cfg.k_esc)
                    * pref
                    * max(abs(float(sigma_grad_rail)), 1.0)
                )
                if abs(v_escape) > 1.0e-12:
                    self._escape_active = True
                    self._escape_sign = pref
        v_escape = float(
            np.clip(v_escape, -self.cfg.v_max_m_s, self.cfg.v_max_m_s)
        )
        v = float(v_escape)
        if use_limiters:
            lim = self._limit_saturation(y, v)
        else:
            lim = 1.0
            self.last_limit_saturated = False
        self.last_limit_saturated = lim < 1e-6
        v *= lim
        span_ff = max(float(self.cfg.v_ff_span_m_s), 1e-6)
        w_ff = float(self.cfg.w_max) * smoothstep01(abs(v_ff_att) / span_ff)
        w_sigma = float(self.cfg.w_sigma_floor) * (1.0 - sig)
        w = (w_reach + w_ff + w_sigma) * lim
        sig_boost = 1.0 + float(self.cfg.k_sigma_boost) * (1.0 - sig)
        w *= sig_boost
        mfrac = float(np.clip(joint_margin_frac, 0.0, 1.0))
        w *= 1.0 + float(self.cfg.k_margin_boost) * (1.0 - mfrac)
        if self._escape_active:
            w *= float(self.cfg.k_escape_boost)
        w *= 1.0 + drift * max(float(self.cfg.d_star_w_mult) - 1.0, 0.0)
        w = min(w, float(self.cfg.w_ext_cap))
        self.last_err_m = float(err)
        self.last_weight = w
        self.last_v_ff = float(v_ff_att)
        self.last_v_escape = float(v_escape)
        self.last_v_reach = 0.0
        return v, w

    def __call__(
        self,
        q_rad: np.ndarray,
        *,
        sigma_scale: float = 1.0,
        sigma_grad_rail: float = 0.0,
        vel_ff: np.ndarray | None = None,
        dt_s: float | None = None,
        joint_margin_frac: float = 1.0,
        sigma_raw: float | None = None,
        y_tcp_d: float | None = None,
        press_stalled: bool = False,
        tool_y_err_m: float = 0.0,
        stroke_limiters: bool = True,
        apply_d_band: bool | None = None,
        block_escape: bool = False,
        unload_sign: float = 0.0,
        jacobian: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Return ``(v_rail_des, w_ext)`` for the QP."""
        if not self.cfg.enabled:
            self.last_err_m = 0.0
            self.last_e_mid_m = 0.0
            self.last_weight = 0.0
            self.last_limit_saturated = False
            self.last_in_limit_band = False
            self.last_d_star_reg_scale = 1.0
            self.last_k_ff_scale = 1.0
            return 0.0, 0.0
        q = np.asarray(q_rad, dtype=float)
        self.last_in_limit_band = self._rail_in_limit_band(float(q[RAIL_INDEX]))
        if self.mode == "pose_attract":
            self.last_d_star_reg_scale = 1.0
            self.last_k_ff_scale = 1.0
            self.last_e_mid_m = 0.0
            return self._call_pose_attract(
                q,
                sigma_scale=sigma_scale,
                sigma_grad_rail=sigma_grad_rail,
                dt_s=dt_s,
            )
        return self._call_reach(
            q,
            sigma_scale=sigma_scale,
            sigma_grad_rail=sigma_grad_rail,
            vel_ff=vel_ff,
            dt_s=dt_s,
            y_tcp_d=y_tcp_d,
            joint_margin_frac=joint_margin_frac,
            sigma_raw=sigma_raw,
            press_stalled=press_stalled,
            tool_y_err_m=tool_y_err_m,
            stroke_limiters=stroke_limiters,
            apply_d_band=apply_d_band,
            block_escape=block_escape,
            unload_sign=unload_sign,
            jacobian=jacobian,
        )
```

## 6. `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/secondary_composer.py`

- sha256：`9bb6bfff31106a33a13ed66ccd75ff0f380ab8d11a6c19bc5e13fd808bd713e2`
- 行数：347

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

from rm75_control.control.joint_admittance_8dof.filters import smoothstep01
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


def _as_weight(flag) -> float:
    if isinstance(flag, bool):
        return 1.0 if flag else 0.0
    try:
        value = float(flag)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _soft_cap_per_joint(
    qdot: np.ndarray,
    cap: np.ndarray,
    *,
    band_frac: float = 0.15,
) -> np.ndarray:
    """Per-joint C1 fade into ``cap``; ``cap`` remains a hard ceiling."""

    out = np.asarray(qdot, dtype=float).copy()
    lim = np.asarray(cap, dtype=float)
    n = min(out.size, lim.size)
    if n == 0:
        return out
    mag = np.abs(out[:n])
    hi = np.maximum(lim[:n], 0.0)
    lo = hi * max(0.0, 1.0 - float(band_frac))
    span = np.maximum(hi - lo, 1.0e-12)
    s = np.clip((mag - lo) / span, 0.0, 1.0)
    t = s * s * (3.0 - 2.0 * s)
    blended = mag * (1.0 - t) + hi * t
    desired = np.where(mag <= lo, mag, np.minimum(blended, hi))
    sign = np.sign(out[:n])
    sign = np.where(sign == 0.0, 1.0, sign)
    out[:n] = sign * desired
    return out


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
        self.last_soft_scale: float = 1.0
        self.last_centering_norm: float = 0.0
        self.last_manip_norm: float = 0.0
        self.last_arm_angle_norm: float = 0.0
        self.last_damping_norm: float = 0.0
        self.last_rail_lock_norm: float = 0.0

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
        return smoothstep01((self.arm_activation_limit + band - u_max) / (2.0 * band))

    def compose(
        self,
        q_rad: np.ndarray,
        qdot_ff: np.ndarray | None,
        qdot_prev: np.ndarray | None,
        *,
        arm_suppressed: bool,
        sigma_min: float = 1.0,
        sigma_ref: float = 0.08,
        centering_suppressed: bool = False,
        manipulability_active: bool | float = False,
        centering_sigma_fade: bool = True,
        soft_scale: float = 1.0,
        dt_s: float | None = None,
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
        qdot_center = np.zeros_like(q)
        qdot_mu = np.zeros_like(q)
        qdot_lock = np.zeros_like(q)
        qdot_damp = np.zeros_like(q)
        rail_hold = self.rail_lock is not None and self.rail_lock.active
        # Lillo dual soft layer: q* centering stays on; manipulability ADDS when
        # active (never XOR-replaces the attractor — that forgot the branch).
        if not centering_suppressed:
            qdot_center = np.asarray(self.centering(q), dtype=float)
            qdot_soft = qdot_center
        w_mu = _as_weight(manipulability_active)
        if w_mu > 0.0 and self.manipulability is not None:
            # Rail is a base translation: always exclude from manip push.
            qdot_mu = np.asarray(
                self.manipulability(
                    q, sigma_min=sigma_min, exclude_rail=True, dt_s=dt_s
                ),
                dtype=float,
            )
            sig_ref = max(float(sigma_ref), 1e-6)
            alpha = 1.0
            if sigma_min < sig_ref:
                # Blend up as σ drops so escape grows without dumping q*.
                alpha = 1.0 + (1.0 - float(sigma_min) / sig_ref)
            qdot_soft = qdot_soft + w_mu * float(alpha) * qdot_mu
        if rail_hold:
            qdot_lock = np.asarray(self.rail_lock(q), dtype=float)
            qdot_soft = qdot_soft + qdot_lock

        d_eff = self.d_null
        if self.adaptive_d_null_gain > 0.0 and u_max > 0.0:
            d_eff = d_eff * (1.0 + self.adaptive_d_null_gain * u_max)
        if d_eff > 0.0 and qdot_prev is not None:
            qdot_damp = d_eff * np.asarray(qdot_prev, dtype=float)
            qdot_soft = qdot_soft - qdot_damp
        self.last_centering_norm = float(np.linalg.norm(qdot_center))
        self.last_manip_norm = float(np.linalg.norm(qdot_mu)) * w_mu
        self.last_rail_lock_norm = float(np.linalg.norm(qdot_lock))
        self.last_damping_norm = float(np.linalg.norm(qdot_damp))

        # Per-joint magnitude cap on the soft tasks (see module docstring).
        if self.v_max is not None and self.max_qdot_frac > 0.0:
            cap = self.max_qdot_frac * self.v_max
            qdot_soft = _soft_cap_per_joint(qdot_soft, cap)

        if not rail_hold:
            qdot_soft[0] = 0.0

        # Near σ≈0 mildly attenuate soft tasks — NOT arm_angle, and not
        # J4/J6.  Those two *are* the posture that opens the wrist / keeps
        # the elbow off the stop; fading them 4× is what parked J6 at 2.8°.
        if centering_sigma_fade and sigma_min < sigma_ref:
            fade = max(float(sigma_min) / max(sigma_ref, 1e-6), 0.25)
            scaled = qdot_soft * fade
            if scaled.size > 6:
                scaled[4] = qdot_soft[4]
                scaled[6] = qdot_soft[6]
            qdot_soft = scaled

        qdot0 = qdot_soft
        qdot_arm = np.zeros_like(q)
        if self.arm_task is not None and not arm_suppressed:
            w_arm = self._arm_weight(u_max)
            if w_arm > 0.0:
                qdot_arm = np.asarray(self.arm_task(q), dtype=float)
                self.last_arm_smooth = w_arm * float(self.arm_task.last_singularity_smooth)
                add = w_arm * qdot_arm
                # Drop the part of the later posture that fights the earlier
                # soft stack (centering + manip + damping).
                nb2 = float(np.dot(qdot0, qdot0))
                if nb2 > 1.0e-12 and float(np.dot(qdot0, add)) < 0.0:
                    add = add - (float(np.dot(add, qdot0)) / nb2) * qdot0
                qdot0 = qdot0 + add
            else:
                self.last_arm_smooth = 0.0
        else:
            self.last_arm_smooth = 1.0 if self.arm_task is None else 0.0
        self.last_arm_angle_norm = float(np.linalg.norm(qdot_arm))

        scale = float(np.clip(soft_scale, 0.0, 1.0)) if np.isfinite(soft_scale) else 1.0
        self.last_soft_scale = scale
        qdot0 = qdot0 * scale

        if qdot_ff is not None:
            qdot0 = qdot0 + np.asarray(qdot_ff, dtype=float)

        return qdot0


class SecondaryRateFilter:
    """200 Hz jerk-limited tracker of a slower (15 Hz) secondary target.

    ``j = clip(wn² (target − qdot) − 2 ζ wn a, ±j_max)``, then integrate.
    The filtered vector is a QP2 preference, never added after the QP.
    """

    def __init__(
        self,
        n: int,
        *,
        wn_rad_s: float = 2.0 * np.pi * 8.0,
        zeta: float = 1.0,
        target_hz: float = 15.0,
    ) -> None:
        self.n = int(n)
        self.wn = float(wn_rad_s)
        self.zeta = float(zeta)
        self.target_hz = float(target_hz)
        self.qdot = np.zeros(self.n, dtype=float)
        self.acc = np.zeros(self.n, dtype=float)
        self.target = np.zeros(self.n, dtype=float)
        self._age_s = float("inf")

    def reset(self) -> None:
        self.qdot[:] = 0.0
        self.acc[:] = 0.0
        self.target[:] = 0.0
        self._age_s = float("inf")

    def step(
        self,
        raw: np.ndarray,
        dt_s: float,
        j_max: np.ndarray,
        *,
        force_target: bool = False,
    ) -> np.ndarray:
        dt = max(float(dt_s), 0.0)
        raw_a = np.asarray(raw, dtype=float).reshape(-1)
        if raw_a.size != self.n:
            padded = np.zeros(self.n, dtype=float)
            n = min(raw_a.size, self.n)
            padded[:n] = raw_a[:n]
            raw_a = padded
        period = 1.0 / max(float(self.target_hz), 1.0e-6)
        self._age_s += dt
        if force_target or self._age_s + 1.0e-12 >= period:
            self.target = raw_a.copy()
            self._age_s = 0.0
        if dt <= 1.0e-9:
            return self.qdot.copy()
        j_lim = np.abs(np.asarray(j_max, dtype=float).reshape(-1))
        if j_lim.size == 1:
            j_lim = np.full(self.n, float(j_lim[0]))
        elif j_lim.size != self.n:
            filled = np.full(self.n, float(j_lim[0]) if j_lim.size else 0.0)
            n = min(j_lim.size, self.n)
            filled[:n] = j_lim[:n]
            j_lim = filled
        wn = float(self.wn)
        zeta = float(self.zeta)
        j = wn * wn * (self.target - self.qdot) - 2.0 * zeta * wn * self.acc
        j = np.clip(j, -j_lim, j_lim)
        self.acc = self.acc + j * dt
        self.qdot = self.qdot + self.acc * dt
        return self.qdot.copy()
```

## 7. `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/nullspace_task.py`

- sha256：`8a409a2d25c451825f659d99a4def201456e2d18a5a789ebe81e0d0750ba69e0`
- 行数：102

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
        """
        if q_rad is None:
            self.q_target = self._q_target_default.copy()
        else:
            self.q_target = np.asarray(q_rad, dtype=float).copy()

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

## 8. `rm75_control/rm75_control/control/joint_admittance_8dof/tasks/arm_angle.py`

- sha256：`5396029757b8733529d3684bba8959495f0102e50e35ec7292674c5073cf73f0`
- 行数：236

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
from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import (
    fold_psi_to_positive,
)

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
    # Floor on the observability fade so a stretched arm still tracks ψ.
    obs_smooth_floor: float = 0.3
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
            self.psi_ref = fold_psi_to_positive(self.arm_angle(q))
        # Same SEW plane as −π; keep the tracker on the positive half so a
        # later set_reference(70°) slews 180°→70°, not −180°→−290°.
        self._psi_ref_unwrapped = fold_psi_to_positive(float(self.psi_ref))

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
        # Kinematic nullspace only so d(psi)/dt ~= k_psi * err in the
        # executed QP solution.
        gN = project_onto_task_nullspace(J, g, sigma_min=sigma_min)
        denom = float(np.dot(g, gN))
        err = float(self._psi_ref_unwrapped) - psi
        _, _, obs = self._sw_observability(q)
        smooth = 1.0 - np.exp(-self.cfg.obs_decay_gain * obs * obs)
        floor = float(np.clip(self.cfg.obs_smooth_floor, 0.0, 1.0))
        smooth = max(float(smooth), floor)
        self.last_singularity_smooth = float(smooth)
        safe_denom = max(denom, 0.0) + self.cfg.safe_denom_eps
        qdot = smooth * self.cfg.k_psi * err * gN / safe_denom
        v_cap = self.cfg.max_qdot_frac * np.asarray(self.kin.v_max, dtype=float)
        return np.clip(qdot, -v_cap, v_cap)
```

## 9. `rm75_control/tests/test_side_posture_attractor.py`

- sha256：`2534498aaab17bb0c67dab936347908f0044f6c39f00f5925e386683ceb154f8`
- 行数：256

```python
"""Side-lying attractor, minus escape, and rail host-loop contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.joint_admittance_8dof.config import (
    assert_design_attractor_consistent,
    build_joint_ik_config,
)
from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    live_host_accel_m_s2,
    next_poll_deadline,
)
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import (
    PostureRetarget,
    PsiRetargetConfig,
    d_from_q,
    fold_psi_to_positive,
    psi_err_avoiding_zero,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)
from rm75_control.kinematics.srs_ik import Q_LOWER, Q_UPPER, psi_from_q


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml"
_Q_NOM_DEG = np.array([0.0, -89.5, -94.5, 65.2, 96.0, 89.3, 61.0, 94.6])


def _raw() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_design_q_nominal_matches_psi_attr_and_comfort() -> None:
    cfg = build_joint_ik_config(_raw())
    kin = RobotKinematics()
    qn = np.asarray(cfg.nullspace.q_nominal_rad, dtype=float)
    assert fold_psi_to_positive(psi_from_q(qn)) == pytest.approx(
        float(cfg.psi_retarget.psi_attr_rad), abs=np.deg2rad(1.0)
    )
    assert d_from_q(kin, qn) == pytest.approx(float(cfg.psi_retarget.d_attr_m), abs=0.005)
    q_arm = qn[1:]
    margin = float(np.min(np.minimum(q_arm - Q_LOWER, Q_UPPER - q_arm)))
    assert margin >= float(cfg.qp.joint_comfort.activate_rad) - 1.0e-9
    assert_design_attractor_consistent(cfg, kin=kin)


def test_inconsistent_yaml_is_rejected() -> None:
    raw = deepcopy(_raw())
    raw["inner"]["nullspace"]["q_nominal_deg"] = [
        0.0, 0.0, -45.0, 0.0, 90.0, 40.0, 60.0, 0.0
    ]
    with pytest.raises(ValueError, match="q_nominal"):
        build_joint_ik_config(raw)

    raw = deepcopy(_raw())
    raw["inner"]["psi_retarget"]["psi_attr_deg"] = 0.0
    with pytest.raises(ValueError, match="psi_attr"):
        build_joint_ik_config(raw)


def test_centering_and_q_star_keep_signed_nominal() -> None:
    cfg = build_joint_ik_config(_raw())
    cfg.ird.enabled = False
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    inner = JointIkController(RobotKinematics(), cfg)
    q = np.deg2rad(_Q_NOM_DEG).copy()
    q[0] = 0.40
    q[1] = abs(float(q[1]))
    inner.reset(q)
    assert inner.core.q_star_signs is not None
    assert float(inner.core.q_star_signs[1]) < 0.0
    assert inner.core.q_star is not None
    assert float(inner.core.q_star[1]) > 0.0
    assert float(inner.centering_task.q_target[1]) > 0.0
    assert inner._family_ok is False


def test_planar_start_keeps_design_j1_sign() -> None:
    cfg = build_joint_ik_config(_raw())
    cfg.ird.enabled = False
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    inner = JointIkController(RobotKinematics(), cfg)
    q = np.array(
        [0.31, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    inner.reset(q)
    yaml_j1 = float(cfg.nullspace.q_nominal_rad[1])
    assert inner.core.q_star_signs is not None
    assert float(inner.core.q_star_signs[1]) < 0.0
    assert inner.core.q_star is not None
    assert abs(float(inner.core.q_star[1]) - yaml_j1) > np.deg2rad(20.0)
    assert abs(float(inner.core.q_star[1])) < np.deg2rad(10.0)
    assert abs(float(inner.centering_task.q_target[1])) < np.deg2rad(10.0)


def test_homotopy_done_pins_centering_to_yaml() -> None:
    cfg = build_joint_ik_config(_raw())
    cfg.ird.enabled = False
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    inner = JointIkController(RobotKinematics(), cfg)
    q = np.array(
        [0.31, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    inner.reset(q)
    yanked = q.copy()
    yanked[1] = np.deg2rad(-160.0)
    assert inner.posture_retarget is not None
    inner.posture_retarget.q_star_rad = yanked
    inner.posture_retarget.homotopy_s = 0.4
    inner._publish_homotopy_centering()
    assert float(inner.centering_task.q_target[1]) == pytest.approx(
        np.deg2rad(-160.0)
    )
    inner.posture_retarget.homotopy_s = 1.0
    inner._publish_homotopy_centering()
    yaml_j1 = float(cfg.nullspace.q_nominal_rad[1])
    assert float(inner.centering_task.q_target[1]) == pytest.approx(yaml_j1)
    assert inner.core.q_star is not None
    assert float(inner.core.q_star[1]) == pytest.approx(yaml_j1)
    assert inner.core.q_star_signs is not None
    assert float(inner.core.q_star_signs[1]) < 0.0


def test_psi_star_returns_home_after_healthy_dwell() -> None:
    kin = RobotKinematics()
    cfg = PsiRetargetConfig(
        enabled=True,
        psi_rate_rad_s=np.deg2rad(25.0),
        psi_return_dwell_s=1.0,
        psi_replan_period_s=0.1,
    )
    rt = PostureRetarget(kin, cfg)
    q = np.deg2rad(_Q_NOM_DEG).copy()
    q[0] = 0.40
    rt.reset(q)
    assert rt.psi_star_rad == pytest.approx(float(cfg.psi_attr_rad), abs=1e-9)
    hijack = float(np.deg2rad(100.0))
    rt._psi_star = hijack
    rt.psi_star_rad = hijack
    rt._healthy_dwell_s = 0.0
    dt = 0.05
    prev = float(rt._psi_cmd)
    for _ in range(10):
        psi, _d = rt.step(q, dt, rail_lo=0.005, rail_hi=0.78)
        assert abs(psi_err_avoiding_zero(prev, psi)) <= cfg.psi_rate_rad_s * dt + 1e-9
        assert psi * prev >= -1e-9 or abs(prev) > 0.5 * np.pi
        prev = psi
    assert rt.psi_star_rad == pytest.approx(hijack, abs=1e-9)
    for _ in range(16):
        psi, _d = rt.step(q, dt, rail_lo=0.005, rail_hi=0.78)
        assert abs(psi_err_avoiding_zero(prev, psi)) <= cfg.psi_rate_rad_s * dt + 1e-9
        assert psi * prev >= -1e-9 or abs(prev) > 0.5 * np.pi
        prev = psi
    assert rt.psi_star_rad == pytest.approx(float(cfg.psi_attr_rad), abs=1e-9)


def test_collapsed_wrist_search_then_home() -> None:
    kin = RobotKinematics()
    cfg = PsiRetargetConfig(
        enabled=True,
        psi_replan_period_s=0.0,
        psi_return_dwell_s=1.0,
        psi_rate_rad_s=np.deg2rad(25.0),
    )
    rt = PostureRetarget(kin, cfg)
    q_bad = np.array(
        [0.360018, 2.534646, -0.341951, -2.812693, 2.084567, 2.844237, 0.329491, -1.621615]
    )
    q_good = np.deg2rad(_Q_NOM_DEG).copy()
    q_good[0] = 0.36
    rt.reset(q_bad)
    rt.step(q_bad, 0.1, rail_lo=0.005, rail_hi=0.78)
    assert rt.last_psi_search_count >= 1
    prev = float(rt._psi_cmd)
    dt = 0.05
    for _ in range(25):
        psi, _d = rt.step(q_good, dt, rail_lo=0.005, rail_hi=0.78)
        assert abs(psi_err_avoiding_zero(prev, psi)) <= cfg.psi_rate_rad_s * dt + 1e-9
        assert not (prev * psi < 0.0 and abs(prev) < 0.5 * np.pi)
        prev = psi
    assert rt.psi_star_rad == pytest.approx(float(cfg.psi_attr_rad), abs=1e-9)


def test_preferred_escape_sign_is_minus_except_min_pin() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            soft_min_m=0.10,
            soft_max_m=0.70,
            pin_margin_m=0.008,
            escape_leave_m=0.04,
            escape_sign_policy="minus",
        ),
    )
    for y in np.linspace(0.15, 0.69, 28):
        assert task._preferred_escape_sign(float(y)) <= 0.0
    assert task._preferred_escape_sign(0.105) == pytest.approx(1.0)


def test_step_reference_velocity_step_lag_is_at_most_3_2_mm() -> None:
    a_max = live_host_accel_m_s2(
        vel_max_m_s=0.15, accel_ms=120.0, configured_m_s2=1.2, lead_mm=10.0
    )
    assert a_max == pytest.approx(min(1.2, 0.85 * (1000.0 / 60.0) * 0.010 / 0.12))
    dt = 0.02
    x_ref = 0.400
    v_ref = 0.0
    v_goal = 0.08
    max_lag = 0.0
    for i in range(80):
        now = i * dt
        x_goal = 0.400 + v_goal * now
        x_ref, v_ref, _a = RailServoBridge._step_reference(
            x_ref,
            v_ref,
            x_goal,
            v_goal,
            stationary=False,
            dt=dt,
            v_max=0.15,
            a_max=a_max,
        )
        max_lag = max(max_lag, abs(x_goal - x_ref))
    assert max_lag <= 0.0032 + 1.0e-6


def test_next_poll_deadline_does_not_accumulate_overrun_debt() -> None:
    period = 0.023
    next_t = 0.0
    now = 0.0
    for _ in range(5):
        now += 0.040
        next_t = next_poll_deadline(next_t, now, period)
        assert next_t == pytest.approx(now)
    now += 0.010
    next_t = next_poll_deadline(next_t, now, period)
    assert next_t == pytest.approx(0.200 + period)
    on_time = next_poll_deadline(1.0, 1.010, period)
    assert on_time == pytest.approx(1.0 + period)
```

## 10. `rm75_control/native/wbc_rt/include/wbc_rt/posture.hpp`

- sha256：`d2d370271414b20c24f99dd5986edb0d3911326f27a265a98c11f0cfc0f02641`
- 行数：80

```cpp
#pragma once

#include <optional>

#include "wbc_rt/config.hpp"
#include "wbc_rt/kinematics.hpp"
#include "wbc_rt/srs_ik.hpp"
#include "wbc_rt/types.hpp"

namespace wbc_rt {

class PostureRetarget {
 public:
  explicit PostureRetarget(const Config& cfg);

  void reset(const Vec8& q, const Vec6& pose);
  void set_planned_stroke(double d_star, double psi_star);
  void begin_unplanned(const Vec8& q, const Vec6& pose);
  double nudge_d_star(double delta_m, double y_des_m, double rail_lo, double rail_hi,
                      double dt);

  // Matches Python PostureRetarget.step.  Updates d*, ψ*, q*.
  void step(const Vec8& q, const Vec6& pose, double dt, double rail_lo, double rail_hi,
            bool hold_setpoint);

  bool planned() const { return planned_; }
  double d_star() const { return d_star_; }
  double psi_cmd() const { return psi_cmd_; }
  double psi_star() const { return psi_star_; }
  double homotopy_s() const { return s_; }
  const Vec8& q_star() const { return q_star_; }

 private:
  struct Pack {
    Eigen::Matrix<double, 7, 1> q_arm;
    Vec8 q_full;
  };

  std::optional<Pack> eval_at(const Vec6& pose, double psi, double d) const;
  std::optional<Pack> eval_at(const Vec6& pose, double psi, double d, int branch) const;
  std::optional<double> select_d_for_elbow(const Vec8& q, const Vec6& pose, double psi,
                                           double rail_lo, double rail_hi) const;
  void advance_homotopy(const Vec8& q, const Vec6& pose, double dt, double rail_lo,
                        double rail_hi, double live_psi);
  void maybe_retarget_psi(const Vec8& q, const Vec6& pose, double dt, double rail_lo,
                          double rail_hi);
  std::optional<double> search_psi(const Vec8& q, const Vec6& pose, double rail_lo,
                                   double rail_hi) const;
  double rate_limit_psi(double dt, double live_psi);
  double rate_limit_d(double dt);
  std::optional<std::pair<double, double>> rail_window(double y_tcp, double rail_lo,
                                                       double rail_hi) const;
  std::optional<double> clip_d(double d, double y_tcp, double rail_lo, double rail_hi,
                               double d_live) const;
  bool j4_in_design(double j4, bool loose) const;
  bool j4_illegal(double j4, bool has_travel) const;
  bool q_star_ok(const Eigen::Matrix<double, 7, 1>& q_arm, const Vec8& q_live, const Vec6& pose,
                 double rail_lo, double rail_hi) const;
  double homotopy_T(double d0, double d_goal, double psi0, double psi_goal) const;

  const Eigen::Matrix3d* flange_R() const;
  const Eigen::Vector3d* flange_t() const;

  Config cfg_;
  int branch_ = 0;
  bool planned_ = false;
  bool held_prev_ = false;
  double d_star_ = 0.0;
  double d0_ = 0.0;
  double d_center_target_ = 0.0;
  double psi_cmd_ = 0.0;
  double psi0_ = 0.0;
  double psi_star_ = 0.0;
  double s_ = 0.0;
  double search_age_s_ = 0.0;
  double healthy_dwell_s_ = 0.0;
  Vec8 q_star_ = Vec8::Zero();
};

}  // namespace wbc_rt
```

## 11. `rm75_control/native/wbc_rt/src/posture.cpp`

- sha256：`a8c48ca43aa1026792758a2749ac7827e05da3d0a0d88ae7ea82d56c3eec7d48`
- 行数：343

```cpp
#include "wbc_rt/posture.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace wbc_rt {

PostureRetarget::PostureRetarget(const Config& cfg) : cfg_(cfg) {}

const Eigen::Matrix3d* PostureRetarget::flange_R() const {
  return cfg_.have_flange_tcp ? &cfg_.R_flange_tcp : nullptr;
}
const Eigen::Vector3d* PostureRetarget::flange_t() const {
  return cfg_.have_flange_tcp ? &cfg_.t_flange_tcp : nullptr;
}

void PostureRetarget::reset(const Vec8& q, const Vec6& pose) {
  const double psi = fold_psi_to_positive(srs::psi_from_q(q));
  psi_cmd_ = psi;
  psi0_ = psi;
  psi_star_ = clamp_psi_to_envelope(cfg_.psi_attr, cfg_.psi_env_lo, cfg_.psi_env_hi);
  d_star_ = pose[1] - q[0];
  d0_ = d_star_;
  d_center_target_ = cfg_.d_attr;
  s_ = 0.0;
  branch_ = srs::branch_from_q(q);
  q_star_ = q;
  planned_ = false;
  held_prev_ = false;
  search_age_s_ = 0.0;
  healthy_dwell_s_ = 0.0;
}

void PostureRetarget::set_planned_stroke(double d_star, double psi_star) {
  d_star_ = d_star;
  d_center_target_ = d_star;
  psi_star_ = psi_star;
  planned_ = true;
}

void PostureRetarget::begin_unplanned(const Vec8& q, const Vec6& pose) {
  reset(q, pose);
}

std::optional<PostureRetarget::Pack> PostureRetarget::eval_at(const Vec6& pose, double psi,
                                                             double d) const {
  return eval_at(pose, psi, d, branch_);
}

std::optional<PostureRetarget::Pack> PostureRetarget::eval_at(const Vec6& pose, double psi,
                                                             double d, int branch) const {
  const double y_urdf = pose[1] - d;
  const double y_s = srs::shoulder_y_from_q_rail(y_urdf);
  const auto q_arm = srs::srs_ik(pose, psi, branch, y_s, flange_R(), flange_t());
  if (!q_arm) return std::nullopt;
  Pack p;
  p.q_arm = *q_arm;
  p.q_full[0] = y_urdf;
  p.q_full.tail<7>() = *q_arm;
  return p;
}

std::optional<std::pair<double, double>> PostureRetarget::rail_window(double y_tcp, double rail_lo,
                                                                     double rail_hi) const {
  const double margin = std::max(cfg_.rail_margin, 0.0);
  const double y_lo = rail_lo + margin;
  const double y_hi = rail_hi - margin;
  if (y_lo > y_hi + 1e-12) return std::nullopt;
  const double d_lo = y_tcp - y_hi;
  const double d_hi = y_tcp - y_lo;
  if (d_lo > d_hi + 1e-12) return std::nullopt;
  return std::make_pair(d_lo, d_hi);
}

std::optional<double> PostureRetarget::clip_d(double d, double y_tcp, double rail_lo, double rail_hi,
                                             double d_live) const {
  const auto w = rail_window(y_tcp, rail_lo, rail_hi);
  if (!w) {
    if (std::isfinite(d_live)) return d_live;
    return std::nullopt;
  }
  return clip(d, w->first, w->second);
}

bool PostureRetarget::j4_in_design(double j4, bool loose) const {
  double lo = cfg_.elbow_lo;
  double hi = cfg_.elbow_hi;
  if (loose) {
    lo -= 5.0 * M_PI / 180.0;
    hi += 7.0 * M_PI / 180.0;
  }
  return lo - 1e-9 <= j4 && j4 <= hi + 1e-9;
}

bool PostureRetarget::j4_illegal(double j4, bool has_travel) const {
  if (!has_travel) return false;
  return std::abs(j4) >= cfg_.elbow_illegal - 1e-9;
}

bool PostureRetarget::q_star_ok(const Eigen::Matrix<double, 7, 1>& q_arm, const Vec8& /*q_live*/,
                               const Vec6& pose, double rail_lo, double rail_hi) const {
  const auto w = rail_window(pose[1], rail_lo, rail_hi);
  const bool has_travel = w && (w->second - w->first) > 0.01;
  return !j4_illegal(q_arm[3], has_travel);
}

double PostureRetarget::homotopy_T(double d0, double d_goal, double psi0, double psi_goal) const {
  const double d_rate = std::max(cfg_.d_center_rate, 1e-9);
  const double psi_rate = std::max(cfg_.psi_rate, 1e-9);
  const double t_d = std::abs(d_goal - d0) / d_rate;
  const double t_psi = std::abs(psi_err_avoiding_zero(psi0, psi_goal)) / psi_rate;
  return std::max({t_d, t_psi, 1e-6});
}

std::optional<double> PostureRetarget::select_d_for_elbow(const Vec8& q, const Vec6& pose,
                                                         double psi, double rail_lo,
                                                         double rail_hi) const {
  const auto w = rail_window(pose[1], rail_lo, rail_hi);
  if (!w) return std::nullopt;
  const double d_lo = w->first;
  const double d_hi = w->second;
  const double d_pref = std::isfinite(d_center_target_) ? d_center_target_ : cfg_.d_attr;
  const bool has_travel = (d_hi - d_lo) > 0.01;
  std::vector<double> samples;
  for (int i = 0; i < 11; ++i) {
    const double a = static_cast<double>(i) / 10.0;
    samples.push_back(d_lo + a * (d_hi - d_lo));
  }
  for (double extra : {d_pref, d_star_, d0_}) {
    if (std::isfinite(extra) && d_lo - 1e-9 <= extra && extra <= d_hi + 1e-9) samples.push_back(extra);
  }
  std::sort(samples.begin(), samples.end());
  samples.erase(std::unique(samples.begin(), samples.end(),
                            [](double a, double b) { return std::abs(a - b) < 1e-12; }),
                samples.end());
  const double sign_pref = -1.0;
  const double j4_c = cfg_.elbow_center;
  double best_d = std::numeric_limits<double>::quiet_NaN();
  double best_cost = 1e300;
  double fallback_d = std::numeric_limits<double>::quiet_NaN();
  double fallback_cost = 1e300;
  for (double d : samples) {
    const auto pack = eval_at(pose, psi, d);
    if (!pack) continue;
    const double j4 = pack->q_arm[3];
    const double j1 = pack->q_arm[0];
    if (j4_illegal(j4, has_travel)) continue;
    double sign_pen = 0.0;
    if (std::abs(j1) > 10.0 * M_PI / 180.0 && j1 * sign_pref < 0.0) sign_pen = 10.0;
    const double cost = std::abs(d - d_pref) + 0.15 * std::abs(j4 - j4_c) + sign_pen;
    if (cost < fallback_cost) {
      fallback_cost = cost;
      fallback_d = d;
    }
    if (!j4_in_design(j4, false)) continue;
    if (cost < best_cost) {
      best_cost = cost;
      best_d = d;
    }
  }
  (void)q;
  if (std::isfinite(best_d)) return best_d;
  if (std::isfinite(fallback_d)) return fallback_d;
  return std::nullopt;
}

double PostureRetarget::rate_limit_d(double dt) {
  const double target = std::isfinite(d_center_target_) ? d_center_target_ : d_star_;
  double err = target - d_star_;
  const double max_step = std::max(cfg_.d_center_rate, 0.0) * std::max(dt, 0.0);
  if (max_step > 0.0 && std::abs(err) > max_step) err = clip(err, -max_step, max_step);
  d_star_ = d_star_ + err;
  return d_star_;
}

double PostureRetarget::nudge_d_star(double delta_m, double y_des_m, double rail_lo,
                                    double rail_hi, double dt) {
  double d_lo = y_des_m - rail_hi;
  double d_hi = y_des_m - rail_lo;
  if (d_lo > d_hi) std::swap(d_lo, d_hi);
  d_center_target_ = clip(d_star_ + delta_m, d_lo, d_hi);
  return rate_limit_d(dt);
}

double PostureRetarget::rate_limit_psi(double dt, double live_psi) {
  const double target = fold_psi_to_positive(psi_star_);
  const double cur = fold_psi_to_positive(psi_cmd_);
  double err = psi_err_avoiding_zero(cur, target);
  const double max_step = cfg_.psi_rate * std::max(dt, 0.0);
  if (max_step > 0.0 && std::abs(err) > max_step) err = clip(err, -max_step, max_step);
  double nxt = cur + err;
  if (cur * nxt < 0.0 && std::abs(cur) > 1e-6) nxt = std::copysign(1e-6, cur);
  nxt = fold_psi_to_positive(nxt);
  const double lead = std::max(cfg_.psi_cmd_lead, 0.0);
  if (lead > 0.0 && std::isfinite(live_psi)) {
    const double live = fold_psi_to_positive(live_psi);
    const double lead_nxt = std::abs(psi_err_avoiding_zero(live, nxt));
    const double lead_cur = std::abs(psi_err_avoiding_zero(live, cur));
    if (lead_nxt > lead + 1e-12 && lead_nxt > lead_cur + 1e-12) nxt = cur;
  }
  psi_cmd_ = nxt;
  return psi_cmd_;
}

void PostureRetarget::advance_homotopy(const Vec8& q, const Vec6& pose, double dt, double rail_lo,
                                      double rail_hi, double live_psi) {
  const double psi_goal = fold_psi_to_positive(psi_star_);
  const auto d_goal = select_d_for_elbow(q, pose, psi_goal, rail_lo, rail_hi);
  if (!d_goal) {
    rate_limit_psi(dt, live_psi);
    return;
  }
  const double T = homotopy_T(d0_, *d_goal, psi0_, psi_goal);
  const double s_try = std::min(1.0, s_ + dt / T);
  const double d_live = pose[1] - q[0];
  auto d_try = clip_d(d0_ + s_try * (*d_goal - d0_), pose[1], rail_lo, rail_hi, d_live);
  if (!d_try) {
    rate_limit_psi(dt, live_psi);
    return;
  }
  const double d_step = std::max(cfg_.d_center_rate, 0.0) * std::max(dt, 0.0);
  *d_try = clip(*d_try, d_star_ - d_step, d_star_ + d_step);
  const double psi_s = fold_psi_to_positive(psi0_ + s_try * psi_err_avoiding_zero(psi0_, psi_goal));
  const auto pack = eval_at(pose, psi_s, *d_try);
  if (!pack || !q_star_ok(pack->q_arm, q, pose, rail_lo, rail_hi)) {
    rate_limit_psi(dt, live_psi);
    return;
  }
  s_ = s_try;
  d_star_ = *d_try;
  q_star_ = pack->q_full;
  rate_limit_psi(dt, live_psi);
}

void PostureRetarget::maybe_retarget_psi(const Vec8& q, const Vec6& pose, double dt, double rail_lo,
                                        double rail_hi) {
  search_age_s_ += std::max(dt, 0.0);
  const double period = std::max(cfg_.psi_replan_period, 0.0);
  const bool due = search_age_s_ + 1e-12 >= period;
  const double j4 = std::abs(q[4]);
  const double j6 = std::abs(q[6]);
  const double attr = clamp_psi_to_envelope(cfg_.psi_attr, cfg_.psi_env_lo, cfg_.psi_env_hi);
  if (j4 < cfg_.psi_env_lo) return;
  if (j6 < cfg_.psi_wrist_ok) {
    healthy_dwell_s_ = 0.0;
    if (!due) return;
    search_age_s_ = 0.0;
    const auto found = search_psi(q, pose, rail_lo, rail_hi);
    if (found) psi_star_ = *found;
    return;
  }
  healthy_dwell_s_ += dt;
  if (due) search_age_s_ = 0.0;
  if (healthy_dwell_s_ + 1e-12 >= std::max(cfg_.psi_return_dwell, 0.0)) psi_star_ = attr;
}

std::optional<double> PostureRetarget::search_psi(const Vec8& q, const Vec6& pose, double rail_lo,
                                                 double rail_hi) const {
  const int branch = srs::branch_from_q(q);
  const double d_c = d_star_;
  const double y_rail = pose[1] - d_c;
  const double margin = std::max(cfg_.rail_margin, 0.0);
  if (y_rail < rail_lo + margin || y_rail > rail_hi - margin) return std::nullopt;
  const double lo = cfg_.psi_env_lo;
  const double hi = cfg_.psi_env_hi;
  double center = clamp_psi_to_envelope(psi_star_, lo, hi);
  const double half = std::max(cfg_.psi_search_half, 0.0);
  const int n = std::max(cfg_.psi_search_n, 3);
  auto score = [&](const std::vector<double>& samples) -> std::pair<double, double> {
    double best_s = -1e300;
    double best_psi = std::numeric_limits<double>::quiet_NaN();
    double best_j6 = std::numeric_limits<double>::quiet_NaN();
    for (double psi : samples) {
      const auto pack = eval_at(pose, psi, d_c, branch);
      if (!pack) continue;
      const double j6 = std::abs(pack->q_arm[5]);
      if (j6 < cfg_.wrist_min - 1e-9) continue;
      double marg = 1e9;
      for (int i = 0; i < 7; ++i) {
        marg = std::min(marg, std::min(pack->q_arm[i] - srs::q_lo()[i],
                                       srs::q_hi()[i] - pack->q_arm[i]));
      }
      const double score_v =
          std::min(j6 / (60.0 * M_PI / 180.0), 1.0) + 0.8 * std::min(marg / (30.0 * M_PI / 180.0), 1.0);
      if (score_v > best_s + 1e-9) {
        best_s = score_v;
        best_psi = psi;
        best_j6 = j6;
      }
    }
    return {best_psi, best_j6};
  };
  std::vector<double> local;
  for (int i = 0; i < n; ++i) {
    const double a = (n == 1) ? 0.0 : static_cast<double>(i) / (n - 1);
    local.push_back(clamp_psi_to_envelope(center - half + a * 2.0 * half, lo, hi));
  }
  std::sort(local.begin(), local.end());
  local.erase(std::unique(local.begin(), local.end(),
                          [](double a, double b) { return std::abs(a - b) < 1e-12; }),
              local.end());
  auto [best_psi, best_j6] = score(local);
  if (!std::isfinite(best_psi) || !std::isfinite(best_j6) || best_j6 < cfg_.psi_wrist_ok) {
    std::vector<double> full;
    for (int i = 0; i < n; ++i) {
      const double a = (n == 1) ? 0.0 : static_cast<double>(i) / (n - 1);
      full.push_back(lo + a * (hi - lo));
    }
    auto [bp, bj] = score(full);
    if (std::isfinite(bp) && (!std::isfinite(best_psi) || bj > best_j6 + 1e-9)) {
      best_psi = bp;
      best_j6 = bj;
    }
  }
  if (std::isfinite(best_psi)) return best_psi;
  return std::nullopt;
}

void PostureRetarget::step(const Vec8& q, const Vec6& pose, double dt, double rail_lo,
                           double rail_hi, bool hold_setpoint) {
  dt = std::max(dt, 0.0);
  const double live_psi = fold_psi_to_positive(srs::psi_from_q(q));
  if (planned_) {
    rate_limit_psi(dt, live_psi);
    return;
  }
  if (held_prev_ && !hold_setpoint) {
    d0_ = d_star_;
    psi0_ = psi_cmd_;
    s_ = 0.0;
  }
  held_prev_ = hold_setpoint;
  if (hold_setpoint) {
    rate_limit_psi(dt, live_psi);
    return;
  }
  maybe_retarget_psi(q, pose, dt, rail_lo, rail_hi);
  advance_homotopy(q, pose, dt, rail_lo, rail_hi, live_psi);
}

}  // namespace wbc_rt
```

## 12. `rm75_control/native/wbc_rt/include/wbc_rt/inner.hpp`

- sha256：`5db6009cd8779d2c7190fe284d9ae9a0f7be571413110a9f57dda8d9b9adca1e`
- 行数：234

```cpp
#pragma once

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pinocchio/multibody/geometry.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>

#include "wbc_rt/config.hpp"
#include "wbc_rt/kinematics.hpp"
#include "wbc_rt/posture.hpp"
#include "wbc_rt/protocol.hpp"

namespace wbc_rt {

struct TickIn {
  Vec6 v_cmd = Vec6::Zero();
  Vec8 q_meas = Vec8::Zero();
  Vec8 qdot_ff = Vec8::Zero();
  Vec6 pose_d = Vec6::Zero();
  Vec6 vel_ff = Vec6::Zero();
  Vec6 path_twist = Vec6::Zero();
  Vec6 feedback_twist = Vec6::Zero();
  double dt_nom = 0.005;
  double dt_wall = 0.005;
  double t_mono = 0.0;
  double rail_v = 0.0;
  double v_force_z = 0.0;
  double posture_d = std::numeric_limits<double>::quiet_NaN();
  double posture_psi = std::numeric_limits<double>::quiet_NaN();
  Vec8 posture_q = Vec8::Zero();
  uint32_t flags = 0;
};

struct TickOut {
  Vec8 q_cmd = Vec8::Zero();
  Vec8 qdot = Vec8::Zero();
  Vec6 v_recv = Vec6::Zero();
  Vec6 v_feas = Vec6::Zero();
  Vec6 v_tcp = Vec6::Zero();
  Vec6 residual = Vec6::Zero();
  double slack = 0.0;
  double e_qp = 0.0;
  double u_alloc = 0.0;
  double u_mid = 0.0;
  double v_r_ref = 0.0;
  double psi = 0.0;
  double d_star = 0.0;
  double d_pref = 0.0;
  double solve_ms = 0.0;
  double sigma_min = 0.0;
  double sigma_arm = 0.0;
  uint32_t flags = 0;
  uint32_t joint_limited = 0;
  uint32_t rail_limited = 0;
  uint32_t wall_active = 0;
  uint32_t secondary_suppressed = 0;
  uint32_t status = kStatusOk;
  double ns_norm = 0.0;
  double ns_centering = 0.0;
  double ns_manip = 0.0;
  double ns_arm_angle = 0.0;
  double ns_damping = 0.0;
  double ns_rail_lock = 0.0;
  double sat_scale = 1.0;
  double sec_target_norm = 0.0;
  double homotopy_s = 0.0;
  double psi_star = 0.0;
  double rail_motion_share = std::numeric_limits<double>::quiet_NaN();
};

class Collision {
 public:
  Collision(pinocchio::Model& model, const Config& cfg);
  void update(const Vec8& q, pinocchio::Data& data);
  int build_rows(pinocchio::Data& data, MatX* jac, VecX* lower, std::vector<int>* slots);

 private:
  pinocchio::Model* model_ = nullptr;
  pinocchio::GeometryModel geom_model_;
  pinocchio::GeometryData geom_data_;
  Config cfg_;
  std::vector<int> slots_;
};

class InnerLoop {
 public:
  explicit InnerLoop(const Config& cfg);

  void enable();
  void stop();
  void reset(const Vec8& q0);
  void begin_hybrid(const Vec8& q_meas, const Vec8& qdot_applied);
  void set_rail_mode(uint32_t mode, uint32_t style, double q_ref, bool has_ref);
  void set_flags(uint32_t bits);
  void set_stroke(double d_star, double psi_star);
  std::pair<double, double> plan_stroke(const Vec8& q, double y_center, double amp);
  void set_rail_pose_target(double y, bool valid);
  void capture_rail_ext_ref(const Vec8& q);
  void set_rail_ext_mode(int pose_attract);

  TickOut step(const TickIn& in);

  const Vec8& q_cmd() const { return q_cmd_; }

 private:
  void apply_velocity_box(const Vec8& q_geom, const Vec8& q_cmd, const Vec8& q_meas,
                          double dt, double h1, double h2, bool rail_locked,
                          double rail_pin, bool has_pin, bool lead_exempt,
                          Vec8* lo, Vec8* hi);
  void tighten_branch(const Vec8& q, bool rail_open, Vec8* lo, Vec8* hi);
  bool solve_hqp(const Mat6x8& J, const Vec6& v_cmd, const Vec8& q_geom,
                 const Vec8& q_prev, const Vec8& qdot_nom, double rail_exec,
                 bool has_rail_exec, double rail_task_vel, double rail_w,
                 bool rail_locked, double dt, double h1, double h2,
                 bool rail_open, double rail_pin, bool has_pin, bool lead_exempt,
                 Vec8* qdot, Vec6* residual, double* slack);

  Config cfg_;
  Kinematics kin_;
  PostureRetarget posture_;
  std::unique_ptr<Collision> collision_;
  std::unique_ptr<proxsuite::proxqp::dense::QP<double>> qp1_;
  std::unique_ptr<proxsuite::proxqp::dense::QP<double>> qp2_;
  bool qp1_inited_ = false;
  bool qp2_inited_ = false;
  bool qp1_last_ok_ = false;
  bool qp2_last_ok_ = false;

  Vec8 q_cmd_ = Vec8::Zero();
  Vec8 qdot_prev_ = Vec8::Zero();
  Vec8 qdot_seen_ = Vec8::Zero();
  Vec8 qdot_prev2_ = Vec8::Zero();
  Vec8 dq_prev_ = Vec8::Zero();
  bool have_dq_prev_ = false;
  Vec8 q_lo_ = Vec8::Zero();
  Vec8 q_hi_ = Vec8::Zero();
  Vec8 v_max_ = Vec8::Ones();
  Vec8 a_max_ = Vec8::Ones();
  Vec8 j_max_ = Vec8::Ones();
  Vec8 q_mid_ = Vec8::Zero();
  Vec8 half_ = Vec8::Ones();
  Vec8 q_star_ = Vec8::Zero();
  Vec8 q_star_signs_ = Vec8::Zero();
  Vec8 q_nominal_ = Vec8::Zero();
  Vec8 m_diag_lpf_ = Vec8::Ones();
  bool m_diag_init_ = false;

  Vec8 sec_qdot_ = Vec8::Zero();
  Vec8 sec_acc_ = Vec8::Zero();
  Vec8 sec_target_ = Vec8::Zero();
  Vec8 sec_lpf_ = Vec8::Zero();
  Vec8 gN_lpf_ = Vec8::Zero();
  bool gN_lpf_init_ = false;
  double sec_age_ = 1e9;

  double v_r_ref_ = 0.0;
  double v_r_a_ = 0.0;
  bool v_r_init_ = false;
  double u_alloc_ = 0.0;
  double u_mid_ = 0.0;
  double u_mid_committed_ = 0.0;
  double mid_integ_ = 0.0;
  double q_hat_ = 0.0;
  double v_hat_ = 0.0;
  bool obs_init_ = false;
  double last_sample_t_ = -1.0;

  double last_slack_ = 0.0;
  double sat_scale_ = 1.0;
  double last_sigma_ = 0.08;
  double quiet_s_ = 0.0;
  double cmd_quiet_s_ = 0.0;
  bool quiescent_ = false;
  bool hold_d_prev_ = false;
  bool enabled_ = true;
  Vec6 last_tcp_est_ = Vec6::Zero();

  int rail_mode_ = 0;
  int locked_style_ = 0;
  double rail_q_ref_ = 0.0;
  bool has_rail_ref_ = false;
  bool plan_drives_rail_ = false;
  bool direct_ptp_ = false;
  bool arm_suppress_ = false;
  bool center_suppress_ = false;
  bool manip_active_ = false;
  bool rail_ext_active_ = true;
  int rail_ext_mode_ = 0;
  double y_rail_target_ = 0.0;
  bool has_y_target_ = false;

  double d_star_ = 0.0;
  double d_pref_ = 0.0;
  double psi_cmd_ = 0.0;
  double psi_star_ = 0.0;
  double homotopy_s_ = 0.0;
  bool planned_ = false;
  double d0_ = 0.0;
  double psi0_ = 0.0;

  double press_z_mark_ = std::numeric_limits<double>::quiet_NaN();
  double press_stall_s_ = 0.0;
  double nudge_cool_s_ = 0.0;

  bool escape_active_ = false;
  double escape_sign_ = 0.0;
  double last_e_mid_ = 0.0;
  double last_v_escape_ = 0.0;
  double last_v_ff_ = 0.0;
  double last_ext_w_ = 0.0;
  bool last_limit_sat_ = false;
  double last_d_star_reg_ = 1.0;

  double dwell_s_ = 0.0;
  double dwell_scale_ = 1.0;
  bool sigma_row_active_ = false;
  Vec8 sigma_grad_ = Vec8::Zero();
  int sigma_tick_ = 0;
  bool box_t_init_ = false;
  double box_last_t_ = 0.0;
  double box_h1_ = 0.005;

  Mat6x8 last_lock_J_ = Mat6x8::Zero();
  Vec6 last_lock_v_ = Vec6::Zero();
  MatX last_C_;
  VecX last_lo_;
  VecX last_hi_;
};

}  // namespace wbc_rt
```

## 13. `rm75_control/native/wbc_rt/src/inner.cpp`

- sha256：`08b400fa0ec2c15179adbc5159555277657116a105c682e8cb3f91014cc66c23`
- 行数：1182

```cpp
#include "wbc_rt/inner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/collision/distance.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace wbc_rt {
namespace {

constexpr int kNVar = kNv + kNTaskSlack + kNPref;
constexpr int kNIn = kNv + kMaxCbf + kMaxPrefRows + kNPref;
constexpr double kRailDriveCap = 0.12;
constexpr double kRailPrefW = 64.0;
constexpr double kQuietLin = 0.005;
constexpr double kQuietRot = 0.05;
constexpr double kQuietTcp = 0.010;
constexpr double kQuietHold = 0.15;

void solve_dense_qp(proxsuite::proxqp::dense::QP<double>& qp,
                    bool* inited,
                    bool last_ok,
                    const MatX& H,
                    const VecX& g,
                    const MatX& A,
                    const VecX& b,
                    const MatX& C,
                    const VecX& lo,
                    const VecX& hi) {
  using IG = proxsuite::proxqp::InitialGuessStatus;
  if (!*inited) {
    qp.init(H, g, A, b, C, lo, hi, true);
    *inited = true;
  } else {
    qp.settings.initial_guess =
        last_ok ? IG::WARM_START_WITH_PREVIOUS_RESULT : IG::NO_INITIAL_GUESS;
    qp.update(H, g, A, b, C, lo, hi, true);
  }
  qp.solve();
}

Vec6 twist_to_base(const Vec6& twist, const Eigen::Matrix3d& R, bool tool) {
  if (!tool) return twist;
  Vec6 out;
  out.head<3>() = R * twist.head<3>();
  out.tail<3>() = R * twist.tail<3>();
  return out;
}

}  // namespace

Collision::Collision(pinocchio::Model& model, const Config& cfg)
    : model_(&model), geom_data_(pinocchio::GeometryModel()), cfg_(cfg) {
  if (!cfg.collision_enabled || cfg.collision_urdf.empty()) return;
  std::vector<std::string> dirs;
  const auto slash = cfg.collision_urdf.find_last_of('/');
  if (slash != std::string::npos) dirs.push_back(cfg.collision_urdf.substr(0, slash));
  pinocchio::urdf::buildGeom(model, cfg.collision_urdf, pinocchio::COLLISION, geom_model_,
                             dirs, ::coal::MeshLoaderPtr());
  geom_model_.addAllCollisionPairs();
  if (!cfg.pair_config.empty()) {
    std::ifstream in(cfg.pair_config);
    std::string line;
    while (std::getline(in, line)) {
      auto a0 = line.find('[');
      auto a1 = line.find(',');
      auto a2 = line.find(']');
      if (a0 == std::string::npos || a1 == std::string::npos || a2 == std::string::npos) continue;
      std::string na = line.substr(a0 + 1, a1 - a0 - 1);
      std::string nb = line.substr(a1 + 1, a2 - a1 - 1);
      auto trim = [](std::string s) {
        while (!s.empty() && (s.front() == ' ' || s.front() == '"')) s.erase(s.begin());
        while (!s.empty() && (s.back() == ' ' || s.back() == '"')) s.pop_back();
        return s;
      };
      na = trim(na);
      nb = trim(nb);
      int ia = -1, ib = -1;
      for (std::size_t i = 0; i < geom_model_.geometryObjects.size(); ++i) {
        if (geom_model_.geometryObjects[i].name == na) ia = static_cast<int>(i);
        if (geom_model_.geometryObjects[i].name == nb) ib = static_cast<int>(i);
      }
      if (ia >= 0 && ib >= 0) {
        pinocchio::CollisionPair cp(static_cast<pinocchio::GeomIndex>(ia),
                                    static_cast<pinocchio::GeomIndex>(ib));
        if (geom_model_.existCollisionPair(cp)) geom_model_.removeCollisionPair(cp);
      }
    }
  }
  geom_data_ = pinocchio::GeometryData(geom_model_);
  slots_.assign(static_cast<std::size_t>(cfg.max_pairs), -1);
}

void Collision::update(const Vec8& q, pinocchio::Data& data) {
  if (geom_model_.ngeoms == 0) return;
  pinocchio::updateGeometryPlacements(*model_, data, geom_model_, geom_data_, q);
  pinocchio::computeDistances(geom_model_, geom_data_);
}

int Collision::build_rows(pinocchio::Data& data, MatX* jac, VecX* lower, std::vector<int>* slots) {
  jac->resize(0, kNv);
  lower->resize(0);
  slots->clear();
  if (geom_model_.ngeoms == 0) return 0;
  auto skew = [](const Eigen::Vector3d& r) {
    Eigen::Matrix3d s;
    s << 0, -r.z(), r.y(), r.z(), 0, -r.x(), -r.y(), r.x(), 0;
    return s;
  };
  struct Hit {
    double d;
    Eigen::Vector3d n, pa, pb;
    int ga, gb;
  };
  std::vector<Hit> hits;
  for (std::size_t i = 0; i < geom_model_.collisionPairs.size(); ++i) {
    const auto& res = geom_data_.distanceResults[i];
    const double d = res.min_distance;
    if (d > cfg_.d_activate + 0.01) continue;
    Hit h;
    h.d = d;
    h.pa = Eigen::Vector3d(res.nearest_points[0][0], res.nearest_points[0][1],
                           res.nearest_points[0][2]);
    h.pb = Eigen::Vector3d(res.nearest_points[1][0], res.nearest_points[1][1],
                           res.nearest_points[1][2]);
    // Same witness normal as Python: (p_a - p_b).  coal's res.normal can flip.
    const Eigen::Vector3d n_ab = h.pa - h.pb;
    const double n_norm = n_ab.norm();
    h.n = (n_norm > 1e-9) ? (n_ab / n_norm) : Eigen::Vector3d(0.0, 0.0, 1.0);
    h.ga = static_cast<int>(geom_model_.collisionPairs[i].first);
    h.gb = static_cast<int>(geom_model_.collisionPairs[i].second);
    hits.push_back(h);
  }
  std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) { return a.d < b.d; });
  const int n = std::min(static_cast<int>(hits.size()), cfg_.max_pairs);
  if (n == 0) return 0;
  jac->resize(n, kNv);
  lower->resize(n);
  jac->setZero();
  for (int i = 0; i < n; ++i) {
    const auto& h = hits[static_cast<std::size_t>(i)];
    const auto& goa = geom_model_.geometryObjects[static_cast<std::size_t>(h.ga)];
    const auto& gob = geom_model_.geometryObjects[static_cast<std::size_t>(h.gb)];
    pinocchio::Data::Matrix6x Ja(6, model_->nv), Jb(6, model_->nv);
    pinocchio::getFrameJacobian(*model_, data, goa.parentFrame, pinocchio::LOCAL_WORLD_ALIGNED, Ja);
    pinocchio::getFrameJacobian(*model_, data, gob.parentFrame, pinocchio::LOCAL_WORLD_ALIGNED, Jb);
    const Eigen::Vector3d oa = data.oMf[goa.parentFrame].translation();
    const Eigen::Vector3d ob = data.oMf[gob.parentFrame].translation();
    const Eigen::Matrix<double, 3, 8> Ja_lin =
        Ja.topRows<3>() - skew(h.pa - oa) * Ja.bottomRows<3>();
    const Eigen::Matrix<double, 3, 8> Jb_lin =
        Jb.topRows<3>() - skew(h.pb - ob) * Jb.bottomRows<3>();
    jac->row(i) = h.n.transpose() * (Ja_lin - Jb_lin);
    (*lower)[i] = -cfg_.cbf_gamma * (h.d - cfg_.d_safe);
    slots->push_back(i);
  }
  return n;
}

InnerLoop::InnerLoop(const Config& cfg)
    : cfg_(cfg), kin_(cfg.urdf), posture_(cfg) {
  q_lo_ = kin_.q_lower();
  q_hi_ = kin_.q_upper();
  v_max_ = kin_.v_max() * cfg.v_scale;
  v_max_[0] = std::min({v_max_[0], cfg.rail_v_max, kRailDriveCap});
  q_lo_[0] = std::max(q_lo_[0], cfg.hard_min);
  q_hi_[0] = std::min(q_hi_[0], cfg.hard_max);
  a_max_.setConstant(cfg.a_max_arm);
  a_max_[0] = cfg.a_max_rail;
  j_max_.setConstant(cfg.j_max_arm);
  j_max_[0] = cfg.j_max_rail;
  q_mid_ = 0.5 * (q_lo_ + q_hi_);
  half_ = (0.5 * (q_hi_ - q_lo_)).cwiseMax(1e-9);
  q_nominal_ = cfg.q_nominal;
  if (q_nominal_.norm() < 1e-12) q_nominal_ = q_mid_;
  q_star_ = q_nominal_;
  q_star_signs_ = q_nominal_;
  rail_mode_ = cfg.rail_mode;
  locked_style_ = cfg.locked_style;
  if (cfg.collision_enabled) {
    try {
      collision_ = std::make_unique<Collision>(kin_.model(), cfg);
    } catch (...) {
      collision_.reset();
    }
  }
  qp1_ = std::make_unique<proxsuite::proxqp::dense::QP<double>>(kNVar, kNTaskSlack, kNIn);
  qp2_ = std::make_unique<proxsuite::proxqp::dense::QP<double>>(kNVar, kNTaskSlack, kNIn);
  qp1_->settings.eps_abs = cfg.eps_abs;
  qp2_->settings.eps_abs = cfg.eps_abs;
  qp1_->settings.max_iter = std::min(cfg.max_iter, cfg.max_iter_cap);
  qp2_->settings.max_iter = std::min(cfg.max_iter, cfg.max_iter_cap);
  qp1_->settings.verbose = false;
  qp2_->settings.verbose = false;
  qp1_->settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  qp2_->settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
}

void InnerLoop::enable() { enabled_ = true; }

void InnerLoop::stop() {
  enabled_ = false;
  quiescent_ = true;
  quiet_s_ = kQuietHold;
  v_r_ref_ = 0.0;
  v_r_a_ = 0.0;
  v_r_init_ = false;
  mid_integ_ = 0.0;
  sec_qdot_.setZero();
  sec_acc_.setZero();
  sec_target_.setZero();
  sec_lpf_.setZero();
  gN_lpf_.setZero();
  gN_lpf_init_ = false;
}

void InnerLoop::reset(const Vec8& q0) {
  q_cmd_ = q0;
  qdot_prev_.setZero();
  qdot_seen_.setZero();
  qdot_prev2_.setZero();
  dq_prev_.setZero();
  have_dq_prev_ = false;
  v_r_ref_ = 0.0;
  v_r_a_ = 0.0;
  v_r_init_ = false;
  u_alloc_ = u_mid_ = u_mid_committed_ = mid_integ_ = 0.0;
  q_hat_ = q0[0];
  v_hat_ = 0.0;
  obs_init_ = true;
  last_sample_t_ = -1.0;
  last_slack_ = 0.0;
  sat_scale_ = 1.0;
  quiet_s_ = 0.0;
  cmd_quiet_s_ = 0.0;
  quiescent_ = false;
  hold_d_prev_ = false;
  enabled_ = true;
  plan_drives_rail_ = false;
  direct_ptp_ = false;
  press_stall_s_ = 0.0;
  nudge_cool_s_ = 0.0;
  press_z_mark_ = std::numeric_limits<double>::quiet_NaN();
  escape_active_ = false;
  escape_sign_ = 0.0;
  sec_qdot_.setZero();
  sec_acc_.setZero();
  sec_target_.setZero();
  sec_lpf_.setZero();
  gN_lpf_.setZero();
  gN_lpf_init_ = false;
  sec_age_ = 1e9;
  m_diag_init_ = false;
  box_t_init_ = false;
  kin_.update(q0);
  posture_.reset(q0, kin_.fk_pose_at(q0));
  d_star_ = posture_.d_star();
  d_pref_ = d_star_;
  d0_ = d_star_;
  psi_cmd_ = posture_.psi_cmd();
  psi_star_ = posture_.psi_star();
  psi0_ = psi_cmd_;
  homotopy_s_ = 0.0;
  planned_ = false;
  q_star_ = q0;
  q_star_signs_ = q_nominal_;
}

void InnerLoop::begin_hybrid(const Vec8& q_meas, const Vec8& qdot_applied) {
  qdot_prev_ = qdot_applied;
  qdot_seen_ = qdot_applied;
  q_star_ = q_meas;
  q_star_signs_ = q_nominal_;
  kin_.update(q_meas);
  posture_.begin_unplanned(q_meas, kin_.fk_pose_at(q_meas));
  d_star_ = posture_.d_star();
  d_pref_ = d_star_;
  d0_ = d_star_;
  psi_cmd_ = posture_.psi_cmd();
  psi_star_ = posture_.psi_star();
  psi0_ = psi_cmd_;
  homotopy_s_ = 0.0;
  planned_ = false;
  hold_d_prev_ = false;
}

void InnerLoop::set_rail_mode(uint32_t mode, uint32_t style, double q_ref, bool has_ref) {
  rail_mode_ = static_cast<int>(mode);
  locked_style_ = static_cast<int>(style);
  if (has_ref) {
    rail_q_ref_ = q_ref;
    has_rail_ref_ = true;
  } else if (mode == kRailLocked && style == kStyleHold) {
    rail_q_ref_ = q_cmd_[0];
    has_rail_ref_ = true;
  }
}

void InnerLoop::set_flags(uint32_t bits) {
  plan_drives_rail_ = bits & kFlagPlanDrivesRail;
  direct_ptp_ = bits & kFlagDirectPtp;
  arm_suppress_ = bits & kFlagArmSuppress;
  center_suppress_ = bits & kFlagCenterSuppress;
  manip_active_ = bits & kFlagManipActive;
  rail_ext_active_ = bits & kFlagRailExtActive;
}

void InnerLoop::set_stroke(double d_star, double psi_star) {
  d_star_ = d_star;
  d_pref_ = d_star;
  psi_star_ = psi_star;
  planned_ = true;
  posture_.set_planned_stroke(d_star, psi_star);
}

std::pair<double, double> InnerLoop::plan_stroke(const Vec8& q, double y_center, double amp) {
  kin_.update(q);
  const double y_lo = y_center - std::abs(amp);
  const double y_hi = y_center + std::abs(amp);
  const double rail_lo_s = cfg_.hard_min + cfg_.rail_margin;
  const double rail_hi_s = cfg_.hard_max - cfg_.rail_margin;
  const double d_min = y_hi - rail_hi_s;
  const double d_max = y_lo - rail_lo_s;
  if (d_min > d_max + 1e-9) {
    return {std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
  }
  const double d = 0.5 * (d_min + d_max);
  const double psi = fold_psi_to_positive(kin_.sew_psi(q));
  set_stroke(d, psi);
  return {d, psi};
}

void InnerLoop::set_rail_pose_target(double y, bool valid) {
  has_y_target_ = valid;
  y_rail_target_ = y;
}

void InnerLoop::capture_rail_ext_ref(const Vec8& q) {
  kin_.update(q);
  d_pref_ = kin_.tcp_xyz()[1] - q[0];
}

void InnerLoop::set_rail_ext_mode(int pose_attract) { rail_ext_mode_ = pose_attract; }

void InnerLoop::apply_velocity_box(const Vec8& q_geom, const Vec8& q_cmd, const Vec8& q_meas,
                                   double dt, double h1, double h2, bool rail_locked,
                                   double rail_pin, bool has_pin, bool lead_exempt,
                                   Vec8* lo, Vec8* hi) {
  *lo = -v_max_;
  *hi = v_max_;
  Vec8 band = Vec8::Constant(cfg_.damper_band_rad);
  band[0] = cfg_.damper_band_rail;
  const Vec8 m = (Vec8() << cfg_.position_margin_rail_m,
                  cfg_.position_margin_rad, cfg_.position_margin_rad, cfg_.position_margin_rad,
                  cfg_.position_margin_rad, cfg_.position_margin_rad, cfg_.position_margin_rad,
                  cfg_.position_margin_rad)
                     .finished();
  double q_rail_hi = std::max(q_geom[0], q_cmd[0]);
  double q_rail_lo = std::min(q_geom[0], q_cmd[0]);
  for (int i = 0; i < kNv; ++i) {
    const double b = std::max(band[i], 1e-9);
    double d_hi = clip(((q_hi_[i] - m[i]) - q_geom[i]) / b, 0.0, 1.0);
    double d_lo = clip((q_geom[i] - (q_lo_[i] + m[i])) / b, 0.0, 1.0);
    if (band[i] <= 1e-9) {
      d_hi = 1.0;
      d_lo = 1.0;
    }
    (*hi)[i] = std::min((*hi)[i], v_max_[i] * d_hi);
    (*lo)[i] = std::max((*lo)[i], -v_max_[i] * d_lo);
  }
  if (band[0] > 1e-9) {
    const double b0 = band[0];
    const double d_hi = clip((q_hi_[0] - m[0] - q_rail_hi) / b0, 0.0, 1.0);
    const double d_lo = clip((q_rail_lo - q_lo_[0] - m[0]) / b0, 0.0, 1.0);
    (*hi)[0] = std::min((*hi)[0], v_max_[0] * d_hi);
    (*lo)[0] = std::max((*lo)[0], -v_max_[0] * d_lo);
  }
  double lo_cap, hi_cap;
  wall_cap(q_geom[0], q_lo_[0] + m[0], q_hi_[0] - m[0], a_max_[0], cfg_.rail_reaction_s, &lo_cap,
           &hi_cap);
  double lo_hi, hi_hi, lo_lo, hi_lo;
  wall_cap(q_rail_hi, q_lo_[0] + m[0], q_hi_[0] - m[0], a_max_[0], cfg_.rail_reaction_s, &lo_hi,
           &hi_hi);
  wall_cap(q_rail_lo, q_lo_[0] + m[0], q_hi_[0] - m[0], a_max_[0], cfg_.rail_reaction_s, &lo_lo,
           &hi_lo);
  (*hi)[0] = std::min({(*hi)[0], hi_cap, hi_hi, hi_lo});
  (*lo)[0] = std::max({(*lo)[0], lo_cap, lo_hi, lo_lo});
  for (int i = 0; i < kNv; ++i) {
    double p_lo = (q_lo_[i] + m[i] - q_geom[i]) / dt;
    double p_hi = (q_hi_[i] - m[i] - q_geom[i]) / dt;
    if (i == 0) {
      if (q_geom[0] < q_lo_[0] + m[0]) p_lo = std::min(p_lo, 0.0);
      if (q_geom[0] > q_hi_[0] - m[0]) p_hi = std::max(p_hi, 0.0);
    }
    (*lo)[i] = std::max((*lo)[i], p_lo);
    (*hi)[i] = std::min((*hi)[i], p_hi);
  }
  collapse_interval(lo, hi, &qdot_prev_, &a_max_, dt);
  const double a_dt = h1;
  for (int i = 0; i < kNv; ++i) {
    (*lo)[i] = std::max((*lo)[i], qdot_prev_[i] - a_max_[i] * a_dt);
    (*hi)[i] = std::min((*hi)[i], qdot_prev_[i] + a_max_[i] * a_dt);
  }
  collapse_interval(lo, hi, &qdot_prev_, &a_max_, dt);
  if (std::isfinite(h2) && h2 > 1e-9) {
    for (int i = 0; i < kNv; ++i) {
      const double centre = qdot_prev_[i] + (a_dt / h2) * (qdot_prev_[i] - qdot_prev2_[i]);
      const double span = j_max_[i] * a_dt * a_dt;
      (*lo)[i] = std::max((*lo)[i], centre - span);
      (*hi)[i] = std::min((*hi)[i], centre + span);
    }
    collapse_interval(lo, hi, &qdot_prev_, &a_max_, dt);
  }
  Vec8 re = Vec8::Constant(cfg_.resync_err_rad);
  re[0] = cfg_.resync_err_rail_m;
  for (int i = 0; i < kNv; ++i) {
    if (re[i] <= 0.0) continue;
    double lead = q_cmd[i] - q_meas[i];
    if (lead_exempt && i == 0) lead = 0.0;
    const double reaction = (i == 0) ? cfg_.rail_reaction_s : dt;
    const double toward_hi = stopping_velocity(re[i] - lead, a_max_[i], reaction);
    const double toward_lo = -stopping_velocity(re[i] + lead, a_max_[i], reaction);
    double chi = std::min((*hi)[i], toward_hi);
    double clo = std::max((*lo)[i], toward_lo);
    if (clo > chi) {
      if (lead >= 0.0) chi = clo;
      else clo = chi;
    }
    (*hi)[i] = chi;
    (*lo)[i] = clo;
  }
  collapse_interval(lo, hi, &qdot_prev_, &a_max_, dt);
  if (has_pin) {
    const double v = clip(rail_pin, (*lo)[0], (*hi)[0]);
    (*lo)[0] = v;
    (*hi)[0] = v;
  } else if (rail_locked) {
    const double prev = qdot_prev_[0];
    double target = 0.0;
    if (std::abs(prev) <= cfg_.lock_vel_eps && (*lo)[0] <= 0.0 && 0.0 <= (*hi)[0]) {
      target = 0.0;
    } else {
      target = std::copysign(std::max(std::abs(prev) - a_max_[0] * dt, 0.0), prev);
      target = clip(target, (*lo)[0], (*hi)[0]);
    }
    (*lo)[0] = target;
    (*hi)[0] = target;
  }
}

void InnerLoop::tighten_branch(const Vec8& q, bool rail_open, Vec8* lo, Vec8* hi) {
  if (!cfg_.branch_enabled) return;
  double act = cfg_.branch_box_activate;
  if (act <= 1e-9) act = cfg_.branch_activate;
  const double eps = cfg_.branch_eps;
  const double band = std::max(act - eps, 1e-6);
  for (int i = 1; i < kNv; ++i) {
    const double qs = q_star_signs_[i];
    if (std::abs(qs) <= cfg_.branch_target_eps) continue;
    const double sign = qs >= 0.0 ? 1.0 : -1.0;
    const double margin = sign * q[i];
    const double d = clip((margin - eps) / band, 0.0, 1.0);
    if (sign > 0.0) (*lo)[i] = std::max((*lo)[i], -v_max_[i] * d);
    else (*hi)[i] = std::min((*hi)[i], v_max_[i] * d);
  }
  const double qs1 = q_star_signs_[1];
  if (std::abs(qs1) > cfg_.branch_target_eps) {
    const double sign1 = qs1 >= 0.0 ? 1.0 : -1.0;
    if (sign1 * q[1] > 0.0) {
      const double wall = std::max(cfg_.j1_overfold_abs, 1e-6);
      const double j1_eps = std::max(cfg_.j1_overfold_eps, 0.0);
      const double j1_act = std::max(cfg_.j1_overfold_activate, j1_eps + 1e-6);
      const double j1_band = std::max(j1_act - j1_eps, 1e-6);
      const double d_ov = clip((wall - std::abs(q[1]) - j1_eps) / j1_band, 0.0, 1.0);
      if (sign1 < 0.0) (*lo)[1] = std::max((*lo)[1], -v_max_[1] * d_ov);
      else (*hi)[1] = std::min((*hi)[1], v_max_[1] * d_ov);
    }
  }
  if (rail_open && q.size() > 4) {
    const double j4_eps = cfg_.j4_limit_eps;
    const double j4_act = std::max(cfg_.j4_limit_activate, j4_eps + 1e-6);
    const double j4_band = std::max(j4_act - j4_eps, 1e-6);
    const double d_hi = clip(((q_hi_[4] - q[4]) - j4_eps) / j4_band, 0.0, 1.0);
    const double d_lo = clip(((q[4] - q_lo_[4]) - j4_eps) / j4_band, 0.0, 1.0);
    (*hi)[4] = std::min((*hi)[4], v_max_[4] * d_hi);
    (*lo)[4] = std::max((*lo)[4], -v_max_[4] * d_lo);
  }
}

bool InnerLoop::solve_hqp(const Mat6x8& J, const Vec6& v_cmd, const Vec8& q_geom,
                          const Vec8& q_prev, const Vec8& qdot_nom, double rail_exec,
                          bool has_rail_exec, double rail_task_vel, double rail_w,
                          bool rail_locked, double dt, double h1, double h2, bool rail_open,
                          double rail_pin, bool has_pin, bool lead_exempt, Vec8* qdot,
                          Vec6* residual, double* slack) {
  Mat6x8 J_task = J;
  Vec6 b_task = v_cmd;
  if (has_rail_exec) {
    const Vec6 rail_contrib = J.col(0) * rail_exec;
    J_task.col(0).setZero();
    auto [cmp, frac] = project_arm_compensation(J, -rail_contrib, q_geom, q_lo_, q_hi_, 0.8, 1.0);
    (void)frac;
    b_task = v_cmd + cmp;
  }
  Vec8 lo_box, hi_box;
  apply_velocity_box(q_geom, q_prev, q_geom, dt, h1, h2, rail_locked, rail_pin, has_pin,
                     lead_exempt, &lo_box, &hi_box);
  tighten_branch(q_geom, rail_open, &lo_box, &hi_box);
  collapse_interval(&lo_box, &hi_box, &qdot_prev_, &a_max_, dt);

  MatX C = MatX::Zero(kNIn, kNVar);
  VecX lo = VecX::Constant(kNIn, -1e20);
  VecX hi = VecX::Constant(kNIn, 1e20);
  C.block(0, 0, kNv, kNv) = Mat8::Identity();
  lo.head<kNv>() = lo_box;
  hi.head<kNv>() = hi_box;
  if (collision_) {
    collision_->update(q_geom, kin_.data());
    MatX cj;
    VecX cl;
    std::vector<int> slots;
    const int n = collision_->build_rows(kin_.data(), &cj, &cl, &slots);
    for (int i = 0; i < n; ++i) {
      C.block(kNv + i, 0, 1, kNv) = cj.row(i);
      lo[kNv + i] = cl[i];
      if (has_rail_exec) {
        lo[kNv + i] -= cj(i, 0) * rail_exec;
        C(kNv + i, 0) = 0.0;
      }
    }
  }
  for (int k = 0; k < kNPref; ++k) {
    C(kNv + kMaxCbf + kMaxPrefRows + k, kNv + kNTaskSlack + k) = 1.0;
    lo[kNv + kMaxCbf + kMaxPrefRows + k] = 0.0;
  }

  Eigen::Matrix<double, kNTaskSlack, kNTaskSlack> W =
      cfg_.task_weight.asDiagonal();
  double scale = 1.0;
  if (last_sigma_ < cfg_.sr_sigma_ref) {
    scale = std::max(cfg_.task_weight_min_frac,
                     (last_sigma_ / cfg_.sr_sigma_ref) * (last_sigma_ / cfg_.sr_sigma_ref));
  }
  W *= scale;
  MatX H1 = MatX::Zero(kNVar, kNVar);
  H1.block(kNv, kNv, kNTaskSlack, kNTaskSlack) = W;
  VecX g1 = VecX::Zero(kNVar);
  MatX A1 = MatX::Zero(kNTaskSlack, kNVar);
  A1.leftCols<kNv>() = J_task;
  A1.block(0, kNv, kNTaskSlack, kNTaskSlack) = -Eigen::Matrix<double, 6, 6>::Identity();

  const double inset = std::max(2.0 * cfg_.eps_abs, 1e-8);
  VecX lo1 = lo, hi1 = hi;
  for (int i = 0; i < kNIn; ++i) {
    if (std::isfinite(lo1[i]) && std::isfinite(hi1[i]) && (hi1[i] - lo1[i]) > 2 * inset) {
      lo1[i] += inset;
      hi1[i] -= inset;
    }
  }

  solve_dense_qp(*qp1_, &qp1_inited_, qp1_last_ok_, H1, g1, A1, b_task, C, lo1, hi1);
  const bool qp1_ok = qp1_->results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED ||
                      qp1_->results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_MAX_ITER_REACHED;
  qp1_last_ok_ = qp1_ok;
  if (!qp1_ok) {
    *qdot = Vec8::Zero();
    *residual = b_task;
    *slack = residual->norm();
    return false;
  }
  VecX x1 = qp1_->results.x;
  Vec8 qdot1 = x1.head<kNv>();
  if (has_rail_exec) {
    double seed = rail_exec;
    if (std::isfinite(rail_task_vel) && !rail_locked) seed = rail_task_vel;
    qdot1[0] = clip(seed, lo_box[0], hi_box[0]);
    x1[0] = qdot1[0];
  }
  const Vec6 t1 = J_task * qdot1;
  last_lock_J_ = has_rail_exec ? J_task : J;
  last_lock_v_ = t1;

  Vec8 w_reg = cfg_.reg;
  if (rail_locked) w_reg[0] *= cfg_.lock_reg_scale;
  Vec8 h_reg = w_reg;
  if (cfg_.use_mass_weighted_reg) {
    Vec8 mdiag = kin_.mass().diagonal().cwiseMax(cfg_.mass_reg_floor);
    if (cfg_.mass_weight_exempt_rail) mdiag[0] = 1.0;
    if (cfg_.mass_reg_lpf_tau_s > 1e-9) {
      if (!m_diag_init_) {
        m_diag_lpf_ = mdiag;
        m_diag_init_ = true;
      } else {
        m_diag_lpf_ = first_order_lpf_vec(m_diag_lpf_, mdiag, dt, cfg_.mass_reg_lpf_tau_s);
      }
      mdiag = m_diag_lpf_;
    }
    h_reg = w_reg.cwiseProduct(mdiag);
  }
  VecX slack_w = VecX::Zero(kNPref);
  slack_w[0] = cfg_.sigma_slack_w;
  slack_w[1] = cfg_.branch_slack_w * dwell_scale_;
  for (int k = 2; k < kNPref; ++k) slack_w[k] = cfg_.comfort_slack_w;

  MatX H2 = MatX::Zero(kNVar, kNVar);
  H2.topLeftCorner<kNv, kNv>() = h_reg.asDiagonal();
  H2.block(kNv, kNv, kNTaskSlack, kNTaskSlack) =
      1e-10 * Eigen::Matrix<double, 6, 6>::Identity();
  for (int k = 0; k < kNPref; ++k) H2(kNv + kNTaskSlack + k, kNv + kNTaskSlack + k) = slack_w[k];
  VecX g2 = VecX::Zero(kNVar);
  g2.head<kNv>() = -h_reg.cwiseProduct(qdot_nom);
  if (rail_w > 0.0 && !rail_locked && !has_pin) {
    H2(0, 0) += rail_w;
    g2[0] -= rail_w * rail_task_vel;
  }
  if (cfg_.smoothness.maxCoeff() > 0.0) {
    H2.topLeftCorner<kNv, kNv>() += cfg_.smoothness.cwiseMax(0.0).asDiagonal();
    g2.head<kNv>() -= cfg_.smoothness.cwiseMax(0.0).cwiseProduct(qdot_prev_);
  }

  // Comfort pref (J4)
  int pref_n = 0;
  MatX pref_J = MatX::Zero(kMaxPrefRows, kNv);
  VecX pref_lo = VecX::Zero(kMaxPrefRows);
  Eigen::VectorXi pref_s = Eigen::VectorXi::Zero(kMaxPrefRows);
  if (cfg_.comfort_enabled) {
    const double d_hi = q_hi_[4] - q_geom[4];
    const double d_lo = q_geom[4] - q_lo_[4];
    const double margin = std::min(d_hi, d_lo);
    const double band = std::max(cfg_.comfort_activate - cfg_.comfort_m, 1e-6);
    const double w = smoothstep01((cfg_.comfort_activate - margin) / band);
    if (w > 1e-6) {
      pref_J(pref_n, 4) = (d_hi <= d_lo) ? -w : w;
      pref_lo[pref_n] = -cfg_.comfort_gamma * (margin - cfg_.comfort_m) * w;
      pref_s[pref_n] = 2 + 3;  // J4 slack
      ++pref_n;
    }
  }
  if (cfg_.sigma_enabled) {
    if (last_sigma_ < cfg_.sigma_activate) sigma_row_active_ = true;
    if (last_sigma_ >= cfg_.sigma_exit) sigma_row_active_ = false;
    if (sigma_row_active_ && sigma_grad_.norm() > 1e-12) {
      pref_J.row(pref_n) = sigma_grad_.transpose();
      pref_lo[pref_n] = -cfg_.sigma_gamma * (last_sigma_ - cfg_.sigma_safe);
      pref_s[pref_n] = 0;
      ++pref_n;
    }
  }
  const int pref_base = kNv + kMaxCbf;
  for (int k = 0; k < pref_n; ++k) {
    C.block(pref_base + k, 0, 1, kNv) = pref_J.row(k);
    C(pref_base + k, kNv + kNTaskSlack + pref_s[k]) = 1.0;
    lo[pref_base + k] = pref_lo[k];
  }

  MatX A2 = MatX::Zero(kNTaskSlack, kNVar);
  A2.leftCols<kNv>() = last_lock_J_;
  VecX x2_seed = VecX::Zero(kNVar);
  x2_seed.head<kNv>() = qdot1;
  if (!qp2_inited_) {
    qp2_->init(H2, g2, A2, last_lock_v_, C, lo, hi, true);
    qp2_inited_ = true;
    qp2_->results.x = x2_seed;
    qp2_->solve();
  } else {
    solve_dense_qp(*qp2_, &qp2_inited_, qp2_last_ok_, H2, g2, A2, last_lock_v_, C, lo, hi);
  }
  const bool qp2_ok = qp2_->results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED ||
                      qp2_->results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_MAX_ITER_REACHED;
  qp2_last_ok_ = qp2_ok;
  Vec8 qdot_out = qdot1;
  if (qp2_ok) qdot_out = qp2_->results.x.head<kNv>();
  *qdot = qdot_out;
  *residual = b_task - J_task * qdot_out;
  *slack = residual->norm();
  last_C_ = C;
  last_lo_ = lo;
  last_hi_ = hi;
  last_lock_v_ = last_lock_J_ * qdot_out;
  return true;
}

TickOut InnerLoop::step(const TickIn& in) {
  const auto t0 = std::chrono::steady_clock::now();
  TickOut out;
  Vec6 twist = in.v_cmd;
  if (!enabled_ || (in.flags & kInStale)) {
    twist.setZero();
    out.flags |= kOutStale;
  }
  const double dt_nom = (std::isfinite(in.dt_nom) && in.dt_nom > 0.0) ? in.dt_nom : cfg_.dt;
  const double dt = integration_period(dt_nom, in.dt_wall);
  if (in.flags & kInSeedQcmd) {
    q_cmd_ = in.q_meas;
    q_hat_ = in.q_meas[0];
    if (in.flags & kInHasQdotFf) {
      qdot_prev_ = in.qdot_ff;
      qdot_seen_ = in.qdot_ff;
      dq_prev_ = in.qdot_ff * dt;
      have_dq_prev_ = true;
    }
  }
  Vec8 q_prev = q_cmd_;
  if (rail_mode_ == kRailCoupled && obs_init_ && last_sample_t_ >= 0.0) {
    q_prev[0] = q_hat_;
    q_cmd_[0] = q_hat_;
  }
  const Vec8 q_state = in.q_meas;
  kin_.update(q_state);
  const Mat6x8 J = kin_.jacobian();
  last_sigma_ = kin_.sigma_min();
  const double sigma_arm = kin_.sigma_arm();
  const Eigen::Matrix3d R = kin_.tcp_R();
  const Vec6 twist_base = twist_to_base(twist, R, cfg_.control_frame == "tool");
  out.v_recv = twist_base;

  const bool locked_hold = rail_mode_ == kRailLocked && locked_style_ == kStyleHold;
  const bool rail_only = rail_mode_ == kRailLocked && locked_style_ == kStyleRailOnly;
  const bool tcp_fixed = rail_mode_ == kRailLocked && locked_style_ == kStyleTcpFixed;

  qdot_prev2_ = qdot_seen_;
  qdot_seen_ = qdot_prev_;

  if (direct_ptp_ && (in.flags & kInHasQdotFf)) {
    Vec8 qdot = in.qdot_ff;
    for (int i = 0; i < kNv; ++i) qdot[i] = clip(qdot[i], -v_max_[i], v_max_[i]);
    if (rail_only) qdot.tail<7>().setZero();
    q_cmd_ = q_prev + qdot * dt;
    qdot_prev_ = qdot;
    out.q_cmd = q_cmd_;
    out.qdot = qdot;
    out.sigma_min = last_sigma_;
    out.sigma_arm = sigma_arm;
    return out;
  }

  double rail_exec = qdot_prev_[0];
  bool has_rail_exec = false;
  if (in.flags & kInHasRailV) {
    rail_exec = in.rail_v;
    has_rail_exec = true;
  } else if (obs_init_ && last_sample_t_ >= 0.0) {
    rail_exec = v_hat_;
    has_rail_exec = true;
  }

  const double now = in.t_mono > 0.0 ? in.t_mono : 0.0;
  if (!obs_init_) {
    q_hat_ = q_state[0];
    v_hat_ = 0.0;
    obs_init_ = true;
  } else {
    const double v_pred = (in.flags & kInHasRailV) ? in.rail_v : v_hat_;
    q_hat_ += v_pred * dt;
    v_hat_ = first_order_lpf(v_hat_, v_pred, dt, lpf_tau_from_fc(cfg_.observer_vel_lpf_hz));
    if (now > last_sample_t_ + 1e-9) {
      const double age = std::max(0.0, now - (last_sample_t_ < 0 ? now : last_sample_t_));
      const double innov = q_state[0] - (q_hat_ - v_pred * age);
      q_hat_ += cfg_.observer_pos_gain * innov;
      v_hat_ += cfg_.observer_vel_gain * innov;
      last_sample_t_ = now;
    }
    v_hat_ = clip(v_hat_, -v_max_[0], v_max_[0]);
  }

  double h1 = dt_nom;
  double h2 = std::numeric_limits<double>::quiet_NaN();
  if (!box_t_init_) {
    box_t_init_ = true;
    box_last_t_ = now;
    box_h1_ = dt_nom;
  } else {
    h2 = box_h1_;
    h1 = clip(now - box_last_t_, 0.8 * dt_nom, 1.0 * dt_nom);
    if (!(std::isfinite(h1) && h1 > 0.0)) h1 = dt_nom;
    box_last_t_ = now;
    box_h1_ = h1;
  }

  const double z_now = kin_.tcp_xyz()[2];
  const double y_tcp = kin_.tcp_xyz()[1];
  double y_tcp_d = y_tcp;
  bool has_pose_d = in.flags & kInHasPoseD;
  if (has_pose_d && std::isfinite(in.pose_d[1])) y_tcp_d = in.pose_d[1];
  const double tool_y_err = y_tcp_d - y_tcp;

  const bool contact = in.flags & kInContact;
  const bool has_vf = in.flags & kInHasVForce;
  const bool demanding = contact && has_vf && std::abs(in.v_force_z) >= cfg_.press_v_force_min;
  if (demanding) {
    if (!std::isfinite(press_z_mark_)) press_z_mark_ = z_now;
    if (std::abs(z_now - press_z_mark_) > cfg_.press_dz_max) {
      press_z_mark_ = z_now;
      press_stall_s_ = 0.0;
    } else {
      press_stall_s_ += dt;
    }
  } else {
    press_z_mark_ = std::numeric_limits<double>::quiet_NaN();
    press_stall_s_ = 0.0;
  }
  const bool press_stalled = press_stall_s_ + 1e-12 >= cfg_.press_stall_s;
  const auto soft = std::pair<double, double>{
      std::max(q_lo_[0], cfg_.soft_min), std::min(q_hi_[0], cfg_.soft_max)};
  const double open_travel = std::max(q_state[0] - soft.first, soft.second - q_state[0]);
  const bool has_travel = open_travel > cfg_.open_travel_min;
  const bool j4_blocked =
      (q_hi_[4] - q_prev[4]) <= cfg_.comfort_m || (q_prev[4] - q_lo_[4]) <= cfg_.comfort_m;
  const bool allow_press =
      demanding && has_travel &&
      (press_stalled || j4_blocked || (std::abs(tool_y_err) >= cfg_.press_y_err));

  const double lin = twist_base.head<3>().norm();
  const double rot = twist_base.tail<3>().norm();
  const double tcp_lin = last_tcp_est_.head<3>().norm();
  const bool cmd_quiet = lin < kQuietLin && rot < kQuietRot;
  const bool tcp_quiet = tcp_lin < kQuietTcp;
  if (cmd_quiet) {
    cmd_quiet_s_ += dt;
    if (tcp_quiet) quiet_s_ += dt;
    else quiet_s_ = 0.0;
  } else {
    cmd_quiet_s_ = 0.0;
    quiet_s_ = 0.0;
  }
  quiescent_ = quiet_s_ + 1e-12 >= kQuietHold;
  const bool slack_high = last_slack_ >= cfg_.slack_enter;
  const bool hold_d = quiescent_ || slack_high;
  hold_d_prev_ = hold_d;

  if (cfg_.psi_enabled && rail_mode_ == kRailCoupled) {
    const Vec6 pose = kin_.fk_pose_at(q_prev);
    posture_.step(q_prev, pose, dt, q_lo_[0], q_hi_[0], hold_d);
    d_star_ = posture_.d_star();
    psi_cmd_ = posture_.psi_cmd();
    psi_star_ = posture_.psi_star();
    homotopy_s_ = posture_.homotopy_s();
    planned_ = posture_.planned();
    d_pref_ = d_star_;
    if (homotopy_s_ + 1e-12 >= 1.0) q_star_ = q_nominal_;
    else q_star_ = posture_.q_star();
  }
  if (press_stalled && allow_press && contact && has_vf && nudge_cool_s_ <= 0.0) {
    const double away = (q_prev[0] > 0.5 * (soft.first + soft.second)) ? 1.0 : -1.0;
    const double y_des = has_pose_d ? y_tcp_d : y_tcp;
    const double d_n = posture_.nudge_d_star(-away * cfg_.d_star_nudge, y_des, soft.first,
                                            soft.second, dt);
    if (std::isfinite(d_n)) {
      d_star_ = d_n;
      d_pref_ = d_n;
    }
    nudge_cool_s_ = cfg_.press_stall_s;
  } else {
    nudge_cool_s_ = std::max(0.0, nudge_cool_s_ - dt);
  }

  double rail_task_vel = 0.0;
  double rail_task_w = 0.0;
  bool have_rail_vel = false;
  last_v_escape_ = 0.0;
  last_e_mid_ = 0.0;
  if (cfg_.rail_ext_enabled && rail_ext_active_ && rail_mode_ == kRailCoupled) {
    const double y = q_state[0];
    // e_mid = (y_tcp − d*) − y_rail. SERVO_TWIST latches pose_d at set_origin;
    // using that Y pulls the rail back to the start instead of tracking d*.
    const double y_des = y_tcp;
    const double rail_ff = y_des - d_pref_;
    const double err_raw = rail_ff - y;
    double band = planned_ ? 0.0 : cfg_.d_band;
    const double err = err_raw - clip(err_raw, -band, band);
    last_e_mid_ = err;
    double v_ff = 0.0;
    if (in.flags & kInHasVelFf) {
      const Eigen::Vector3d j_rail = J.topLeftCorner<3, 1>();
      const double den = j_rail.squaredNorm();
      if (den > 1e-12) v_ff = cfg_.k_ff * j_rail.dot(in.vel_ff.head<3>()) / den;
    }
    last_v_ff_ = v_ff;
    const bool healthy = sigma_arm >= cfg_.healthy_sigma_mute;
    if (!healthy || allow_press) {
      double sign = 0.0;
      if (cfg_.escape_sign_policy == "minus") sign = -1.0;
      else if (cfg_.escape_sign_policy == "plus") sign = 1.0;
      else sign = (soft.second - y > y - soft.first) ? 1.0 : -1.0;
      last_v_escape_ = clip(0.25 * cfg_.k_esc * sign, -cfg_.v_max_ext, cfg_.v_max_ext);
      escape_active_ = std::abs(last_v_escape_) > 1e-12;
      escape_sign_ = sign;
    } else {
      escape_active_ = false;
      last_v_escape_ = 0.0;
    }
    last_ext_w_ = cfg_.w_max_ext;
    if (std::abs(last_v_escape_) > 1e-4) {
      rail_task_vel = last_v_escape_;
      have_rail_vel = true;
    }
    rail_task_w = last_ext_w_;
  }

  if (rail_mode_ == kRailCoupled && !locked_hold) {
    const double lam = sr_damping_lambda(last_sigma_, cfg_.sr_lam0, cfg_.sr_sigma_ref,
                                         cfg_.sr_sigma_floor);
    const Vec8 mw = margin_weight_from_activation(q_prev, q_mid_, half_, cfg_.k_margin,
                                                  cfg_.ns_activation);
    auto [u_a, qall] = allocate_rail(J, twist_base, v_max_, mw, lam, cfg_.v0, cfg_.w0,
                                     last_e_mid_, cfg_.k_err_rail, cfg_.e_ref);
    (void)qall;
    u_alloc_ = u_a;
    const double leave = wall_leave_only_sign(q_state[0], q_lo_[0], q_hi_[0], cfg_.damper_band_rail);
    const bool freeze = quiescent_;
    if (!freeze) mid_integ_ += cfg_.ki_mid * last_e_mid_ * dt;
    double raw = cfg_.kp_mid * last_e_mid_ + mid_integ_;
    double sat = soft_saturate(raw, cfg_.u_mid_max);
    if (leave > 0.0 && sat > 0.0) sat = 0.0;
    if (leave < 0.0 && sat < 0.0) sat = 0.0;
    if (dt > 0.0) mid_integ_ += cfg_.kaw_mid * (u_mid_committed_ - raw) * dt;
    u_mid_ = sat;
    auto [a_mir, j_mir] = arm_mirror_rail_limits(J, a_max_, j_max_, cfg_.rho_a, cfg_.rho_j);
    const double tau = lpf_tau_from_fc(cfg_.f_c_hz);
    double v_f = u_alloc_;
    if (v_r_init_ && tau > 1e-9) v_f = first_order_lpf(v_r_ref_, u_alloc_, dt, tau);
    v_r_init_ = true;
    double a_raw = (v_f - v_r_ref_) / dt;
    const double a_lim = std::min(cfg_.a_max_rail, a_mir);
    const double j_lim = std::min(cfg_.j_max_rail, j_mir);
    double a = clip(a_raw, v_r_a_ - j_lim * dt, v_r_a_ + j_lim * dt);
    a = clip(a, -a_lim, a_lim);
    double v = clip(v_r_ref_ + a * dt, -v_max_[0], v_max_[0]);
    double lo_c, hi_c;
    wall_cap(q_state[0], cfg_.hard_min, cfg_.hard_max, a_lim, cfg_.rail_reaction_s, &lo_c, &hi_c);
    v = clip(v, lo_c, hi_c);
    if (std::abs(v) < 5e-4 && std::abs(u_alloc_) < 5e-4) v = 0.0;
    if (planned_ && q_state[0] >= cfg_.soft_max - cfg_.escape_leave && v > 0.0) v = 0.0;
    v_r_ref_ = v;
    v_r_a_ = (v - (v_r_ref_ - a * dt)) / dt;
    rail_task_vel = v;
    have_rail_vel = true;
    rail_task_w = std::max(rail_task_w, kRailPrefW);
  } else {
    u_alloc_ = 0.0;
    u_mid_ = 0.0;
  }

  Vec8 qdot_center = Vec8::Zero();
  if (!center_suppress_) {
    const Vec8 u_t = (q_prev - q_star_).cwiseQuotient(half_);
    qdot_center = -cfg_.k_center * u_t;
    if (cfg_.k_limit > 0.0) {
      const Vec8 u_l = (q_prev - q_mid_).cwiseQuotient(half_);
      const double span = std::max(1.0 - cfg_.ns_activation, 1e-6);
      for (int i = 0; i < kNv; ++i) {
        const double over = clip((std::abs(u_l[i]) - cfg_.ns_activation) / span, 0.0, 1.0);
        qdot_center[i] -= cfg_.k_limit * ((u_l[i] >= 0) ? 1.0 : -1.0) * over * over;
      }
    }
  }
  Vec8 qdot_damp = Vec8::Zero();
  if (cfg_.d_null > 0.0) qdot_damp = cfg_.d_null * qdot_prev_;
  Vec8 sec = qdot_center - qdot_damp;
  sec[0] = 0.0;
  Vec8 qdot_mu = Vec8::Zero();
  if (manip_active_ && cfg_.k_mu > 0.0 && sigma_grad_.norm() > 1e-12) {
    const double fade = clip(last_sigma_ / std::max(cfg_.sigma_fade_ref, 1e-9), 0.0, 1.0);
    qdot_mu = cfg_.k_mu * fade * sigma_grad_;
    sec += qdot_mu;
  }
  Vec8 qdot_arm = Vec8::Zero();
  if (cfg_.arm_enabled && !arm_suppress_) {
    const double psi = kin_.sew_psi(q_prev);
    const double err = wrap_pi(psi_cmd_ - fold_psi_to_positive(psi));
    Vec8 g = Vec8::Zero();
    const double eps = std::max(cfg_.fd_eps, 1e-5);
    for (int i = 1; i < kNv; ++i) {
      Vec8 qp = q_prev, qm = q_prev;
      qp[i] += eps;
      qm[i] -= eps;
      g[i] = (kin_.sew_psi(qp) - kin_.sew_psi(qm)) / (2 * eps);
    }
    kin_.update(q_state);
    const double lam = sr_damping_lambda(last_sigma_, cfg_.sr_lam0, cfg_.sr_sigma_ref,
                                         cfg_.sr_sigma_floor);
    const Vec8 gN = project_nullspace(J, g, lam);
    if (!gN_lpf_init_) {
      gN_lpf_ = gN;
      gN_lpf_init_ = true;
    } else {
      gN_lpf_ = first_order_lpf_vec(gN_lpf_, gN, dt, lpf_tau_from_fc(cfg_.ns_grad_lpf_hz));
    }
    const double den = std::max(g.dot(gN_lpf_), cfg_.safe_denom_eps);
    qdot_arm = cfg_.k_psi * err * gN_lpf_ / den;
    sec += qdot_arm;
  }
  {
    const double enter = cfg_.slack_enter;
    const double exit_ = cfg_.slack_exit;
    const double lo = clip(cfg_.secondary_scale, 0.0, 1.0);
    const double span = std::max(enter - exit_, 1e-9);
    const double s = clip((last_slack_ - exit_) / span, 0.0, 1.0);
    const double target = 1.0 + smoothstep01(s) * (lo - 1.0);
    const double tau = cfg_.secondary_scale_tau_s;
    if (tau <= 1e-9 || dt <= 0.0) {
      sat_scale_ = target;
    } else {
      const double alpha = std::min(1.0, dt / tau);
      sat_scale_ += alpha * (target - sat_scale_);
    }
    sec.tail<7>() *= sat_scale_;
  }
  if (rail_mode_ == kRailCoupled && !locked_hold) sec[0] = u_mid_ + last_v_escape_;
  out.ns_centering = qdot_center.norm();
  out.ns_manip = qdot_mu.norm();
  out.ns_arm_angle = qdot_arm.norm();
  out.ns_damping = qdot_damp.norm();
  out.ns_rail_lock = 0.0;
  if (quiescent_) sec.setZero();
  sec_lpf_ = first_order_lpf_vec(sec_lpf_, sec, dt, lpf_tau_from_fc(cfg_.sec_input_lpf_hz));
  const double period = 1.0 / std::max(cfg_.sec_target_hz, 1.0e-6);
  sec_age_ += dt;
  if (quiescent_ || sec_age_ >= period) {
    sec_target_ = sec_lpf_;
    sec_age_ = 0.0;
  }
  if (dt > 1e-9) {
    const double wn = 2.0 * M_PI * 8.0;
    const double zeta = std::max(cfg_.sec_filter_zeta, 0.0);
    Vec8 j = wn * wn * (sec_target_ - sec_qdot_) - 2.0 * zeta * wn * sec_acc_;
    for (int i = 0; i < kNv; ++i) j[i] = clip(j[i], -j_max_[i], j_max_[i]);
    sec_acc_ += j * dt;
    sec_qdot_ += sec_acc_ * dt;
  }
  Vec8 sec_filt = sec_qdot_;
  if (cfg_.max_qdot_frac > 0.0) {
    Vec8 cap = kin_.v_max() * cfg_.max_qdot_frac;
    cap[0] = v_max_[0];
    for (int i = 0; i < kNv; ++i) sec_filt[i] = clip(sec_filt[i], -cap[i], cap[i]);
    sec_qdot_ = sec_filt;
  }

  ++sigma_tick_;
  if (cfg_.sigma_enabled && (sigma_tick_ % std::max(cfg_.sigma_grad_period, 1) == 0)) {
    const double eps = cfg_.sigma_grad_eps;
    sigma_grad_.setZero();
    for (int i = 1; i < kNv; ++i) {
      Vec8 qp = q_state;
      qp[i] += eps;
      kin_.update(qp);
      const double sp = kin_.sigma_arm();
      sigma_grad_[i] = (sp - sigma_arm) / eps;
    }
    kin_.update(q_state);
  }

  const bool plan_rail = rail_only || tcp_fixed || plan_drives_rail_;
  double rail_pin = 0.0;
  bool has_pin = false;
  if ((in.flags & kInHasQdotFf) && plan_rail) {
    rail_pin = in.qdot_ff[0];
    has_pin = true;
  }
  const bool lead_exempt = std::abs(q_prev[0] - q_state[0]) > cfg_.resync_err_rail_m;
  Vec8 qdot;
  Vec6 residual;
  double slack = 0.0;
  const bool ok = solve_hqp(J, twist_base, q_state, q_prev, sec_filt, rail_exec, has_rail_exec,
                            have_rail_vel ? rail_task_vel : 0.0, rail_task_w, locked_hold, dt, h1,
                            h2, rail_mode_ == kRailCoupled && has_travel && !locked_hold, rail_pin,
                            has_pin, lead_exempt, &qdot, &residual, &slack);
  if (!ok) {
    qdot = qdot_prev_ * cfg_.fail_qdot_decay;
    for (int i = 0; i < kNv; ++i) qdot[i] = clip(qdot[i], -v_max_[i], v_max_[i]);
  }
  q_cmd_ = q_prev + qdot * dt;
  q_cmd_[0] = clip(q_cmd_[0], q_lo_[0], q_hi_[0]);
  if (q_cmd_[0] <= q_lo_[0] + 1e-4 && qdot[0] < 0.0) {
    q_cmd_[0] = q_lo_[0];
    qdot[0] = 0.0;
  } else if (q_cmd_[0] >= q_hi_[0] - 1e-4 && qdot[0] > 0.0) {
    q_cmd_[0] = q_hi_[0];
    qdot[0] = 0.0;
  }
  if (locked_hold && cfg_.lock_hard_pin && has_rail_ref_) {
    q_cmd_[0] = rail_q_ref_;
    qdot[0] = 0.0;
  }
  if (plan_rail && (in.flags & kInHasQdotFf)) {
    q_cmd_[0] = clip(q_prev[0] + in.qdot_ff[0] * dt, q_lo_[0], q_hi_[0]);
    qdot[0] = (q_cmd_[0] - q_prev[0]) / dt;
    if (rail_only) {
      q_cmd_.tail<7>() = q_prev.tail<7>();
      qdot.tail<7>().setZero();
    }
  }
  Vec8 q_shadow, dq_s;
  bool would = false;
  clamp_command_step(q_prev, q_cmd_, have_dq_prev_ ? &dq_prev_ : nullptr, a_max_, dt, &q_shadow,
                     &dq_s, &would);
  if (would) {
    const Vec8 qdot_s = (q_shadow - q_prev) / dt;
    const Vec6 lock_err = last_lock_J_ * qdot_s - last_lock_v_;
    if (lock_err.norm() <= std::max(10.0 * cfg_.eps_abs, 1e-5)) {
      q_cmd_ = q_shadow;
      qdot = qdot_s;
    }
  }
  dq_prev_ = q_cmd_ - q_prev;
  have_dq_prev_ = true;
  qdot_prev_ = qdot;
  last_slack_ = slack;
  last_tcp_est_ = J * qdot;
  u_mid_committed_ = qdot[0] - v_r_ref_;

  const auto t1 = std::chrono::steady_clock::now();
  out.q_cmd = q_cmd_;
  out.qdot = qdot;
  out.v_feas = last_lock_v_;
  out.v_tcp = last_tcp_est_;
  out.residual = residual;
  out.slack = slack;
  out.e_qp = residual.norm();
  out.u_alloc = u_alloc_;
  out.u_mid = u_mid_;
  out.v_r_ref = v_r_ref_;
  out.psi = psi_cmd_;
  out.d_star = d_star_;
  out.d_pref = d_pref_;
  out.sigma_min = last_sigma_;
  out.sigma_arm = sigma_arm;
  out.solve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  for (int i = 1; i < kNv; ++i) {
    if (qdot[i] >= v_max_[i] - 1e-6 || qdot[i] <= -v_max_[i] + 1e-6) out.joint_limited = 1;
  }
  out.rail_limited = (std::abs(qdot[0]) >= v_max_[0] - 1e-6) ? 1 : 0;
  out.wall_active =
      (q_state[0] <= q_lo_[0] + cfg_.damper_band_rail || q_state[0] >= q_hi_[0] - cfg_.damper_band_rail)
          ? 1
          : 0;
  out.secondary_suppressed = (quiescent_ || slack_high) ? 1 : 0;
  out.ns_norm = sec_filt.norm();
  out.sat_scale = sat_scale_;
  out.sec_target_norm = sec_target_.norm();
  out.homotopy_s = homotopy_s_;
  out.psi_star = psi_star_;
  {
    const Vec6 twist_rail = J.col(0) * rail_exec;
    const Vec6 twist_arm = J.rightCols<7>() * qdot.tail<7>();
    Eigen::Vector3d motion = twist_base.head<3>();
    if ((in.flags & kInHasVelFf) && in.vel_ff.head<3>().norm() > 1e-6) {
      motion = in.vel_ff.head<3>();
    }
    double n_dir = motion.norm();
    if (n_dir <= 1e-9) {
      motion = J.topLeftCorner<3, 1>();
      n_dir = motion.norm();
    }
    out.rail_motion_share = std::numeric_limits<double>::quiet_NaN();
    if (n_dir > 1e-9) {
      const Eigen::Vector3d u = motion / n_dir;
      const double rc = twist_rail.head<3>().dot(u);
      const double ac = twist_arm.head<3>().dot(u);
      const double den = std::abs(rc) + std::abs(ac);
      if (den > 1e-9) out.rail_motion_share = std::abs(rc) / den;
    }
  }
  if (out.joint_limited) out.flags |= kOutJointLimited;
  if (out.rail_limited) out.flags |= kOutRailLimited;
  if (out.wall_active) out.flags |= kOutWallActive;
  if (out.secondary_suppressed) out.flags |= kOutSecSuppressed;
  out.status = kStatusOk;
  return out;
}

}  // namespace wbc_rt
```

## 14. `rm75_control/native/wbc_rt/include/wbc_rt/config.hpp`

- sha256：`6f58d62e253d95aa4628db1d3e3005e777788c9cf120366e07d710c4e7cbf6d9`
- 行数：210

```cpp
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "wbc_rt/types.hpp"

namespace wbc_rt {

struct Config {
  std::string urdf;
  std::string collision_urdf;
  std::string pair_config;
  std::string control_frame = "tool";
  std::string euler_order = "xyz";
  std::string escape_sign_policy = "auto";

  double dt = 0.005;
  double feedback_timeout_s = 0.08;
  double v_scale = 0.8;
  double a_max_arm = 3.0;
  double a_max_rail = 0.60;
  double position_margin_rad = 0.005236;
  double position_margin_rail_m = 0.0;
  double resync_err_rad = 0.10472;
  double resync_err_rail_m = 0.020;
  double d_null = 0.5;
  double d_null_adaptive = 1.0;
  double max_qdot_frac = 0.2;
  double sec_target_hz = 15.0;
  double sec_input_lpf_hz = 0.0;
  double ns_grad_lpf_hz = 0.0;
  double ns_hold_fade_v = 0.0;
  double ns_hold_fade_v0 = 0.0;
  double sec_filter_zeta = 1.0;

  Vec6 task_weight = (Vec6() << 100, 100, 100, 50, 50, 50).finished();
  Vec8 reg = (Vec8() << 1e-3, 1e-2, 1e-2, 1e-2, 1e-2, 1.2e-2, 1.2e-2, 1.2e-2).finished();
  Vec8 smoothness = (Vec8() << 0, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15).finished();
  double eps_abs = 1e-6;
  int max_iter = 400;
  int max_iter_cap = 400;
  double max_solve_ms = 5.0;
  double fail_qdot_decay = 0.85;
  double twist_sigma_floor = 0.02;
  double task_weight_min_frac = 0.05;
  double task_weight_lpf_tau_s = 0.25;
  bool aniso_task_damping = true;
  bool use_mass_weighted_reg = true;
  double mass_reg_floor = 0.05;
  bool mass_weight_exempt_rail = true;
  double mass_reg_lpf_tau_s = 0.2;
  double damper_band_rad = 0.25;
  double damper_band_rail = 0.025;
  double rail_reaction_s = 0.06;
  double near_arm_margin_rad = 0.08;
  double j_max_arm = 300.0;
  double j_max_rail = 60.0;
  double sr_lam0 = 0.05;
  double sr_sigma_ref = 0.08;
  double sr_sigma_floor = 1e-6;

  bool sigma_enabled = true;
  double sigma_activate = 0.12;
  double sigma_safe = 0.06;
  double sigma_exit = 0.16;
  double sigma_gamma = 8.0;
  double sigma_slack_w = 200.0;
  double sigma_grad_eps = 1e-4;
  int sigma_grad_period = 10;

  bool branch_enabled = true;
  double branch_activate = 0.52;
  double branch_box_activate = 0.87;
  double branch_eps = 0.35;
  double j4_limit_eps = 0.087266;
  double j4_limit_activate = 0.436332;
  double j1_overfold_abs = 2.44346;
  double j1_overfold_activate = 0.436332;
  double j1_overfold_eps = 0.0;
  double branch_gamma = 6.0;
  double branch_slack_w = 80.0;
  double branch_target_eps = 1e-3;
  double dwell_free_s = 0.3;
  double dwell_ramp_s = 1.0;
  double dwell_scale_max = 5.0;

  bool comfort_enabled = true;
  double comfort_m = 0.261799;
  double comfort_activate = 0.436332;
  double comfort_gamma = 6.0;
  double comfort_slack_w = 80.0;

  bool collision_enabled = true;
  double d_safe = 0.01;
  double d_activate = 0.04;
  double cbf_gamma = 5.0;
  int max_pairs = 8;

  double k_center = 1.0;
  double k_limit = 2.0;
  double ns_activation = 0.8;
  Vec8 q_nominal = Vec8::Zero();

  bool arm_enabled = true;
  double k_psi = 1.0;
  double fd_eps = 1e-4;
  double safe_denom_eps = 1e-4;
  double obs_decay_gain = 400.0;
  double obs_smooth_floor = 0.3;
  double arm_max_qdot_frac = 0.15;

  bool psi_enabled = true;
  double psi_attr = 1.1868238913561442;          // 68°
  double d_attr = -0.185;
  double psi_env_lo = 0.6981317007977318;        // 40°
  double psi_env_hi = 1.9198621771937625;        // 110°
  double d_center_rate = 0.02;
  double psi_rate = 0.4363323129985824;          // 25°/s
  double rail_margin = 0.02;
  double elbow_hi = 2.007128639793479;       // 115°
  double elbow_illegal = 2.2689280275926285;  // 130°
  double elbow_lo = 1.2217304763960306;      // 70°
  double elbow_center = 1.6580627893946132;  // 95°
  double psi_cmd_lead = 0.3141592653589793;  // 18°
  double psi_return_dwell = 1.0;
  double psi_replan_period = 0.1;
  double psi_search_half = 0.7853981633974483;  // 45°
  int psi_search_n = 9;
  double psi_wrist_ok = 0.6981317007977318;  // 40°
  double wrist_min = 0.5235987755982988;     // 30°
  Eigen::Matrix3d R_flange_tcp = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_flange_tcp = Eigen::Vector3d(0.0, 0.0, 0.22);
  bool have_flange_tcp = false;
  bool ird_enabled = false;

  double k_mu = 0.0;
  double sigma_fade_ref = 0.08;

  int rail_mode = 0;
  int locked_style = 0;
  double lock_vel_eps = 0.0;
  double rail_v_max = 0.15;
  double soft_min = 0.030;
  double soft_max = 0.755;
  double hard_min = 0.005;
  double hard_max = 0.78;
  bool lock_hard_pin = true;
  double lock_reg_scale = 100.0;

  double v0 = 0.05;
  double w0 = 0.30;
  double k_margin = 4.0;
  double kp_mid = 1.2;
  double ki_mid = 0.80;
  double u_mid_max = 0.12;
  double k_err_rail = 4.0;
  double e_ref = 0.08;
  double f_c_hz = 1.0;
  double kaw_mid = 8.0;
  double rho_a = 0.50;
  double rho_j = 0.30;
  double observer_pos_gain = 0.35;
  double observer_vel_gain = 2.0;
  double observer_vel_lpf_hz = 8.0;

  bool rail_ext_enabled = true;
  double k_ext = 1.0;
  double k_ff = 1.0;
  double v_ff_thr = 0.01;
  double v_ff_span = 0.03;
  double e0_m = 0.05;
  double e1_m = 0.15;
  double w_max_ext = 1.5;
  double v_max_ext = 0.08;
  double limit_margin = 0.15;
  double pin_margin = 0.008;
  double escape_leave = 0.04;
  double healthy_sigma_mute = 0.08;
  double d_band = 0.005;
  double k_sigma_boost = 2.0;
  double k_esc = 0.5;
  double w_sigma_floor = 1.0;
  double k_pose = 2.0;
  double pose_e0 = 0.005;
  double pose_e1 = 0.04;
  double pose_w_max = 4.0;
  double k_escape_boost = 1.2;
  double k_margin_boost = 4.0;
  double w_ext_cap = 24.0;
  double d_star_err0 = 0.01;
  double d_star_err1 = 0.04;
  double d_star_w_mult = 6.0;
  double press_v_force_min = 0.02;
  double press_dz_max = 0.002;
  double press_y_err = 0.005;
  double press_stall_s = 0.5;
  double d_star_nudge = 0.01;
  double open_travel_min = 0.01;

  double slack_enter = 0.15;
  double slack_exit = 0.03;
  double secondary_scale = 0.15;
  double secondary_scale_tau_s = 0.10;

  static Config load(const std::string& path);
};

}  // namespace wbc_rt
```

## 15. `rm75_control/native/wbc_rt/include/wbc_rt/protocol.hpp`

- sha256：`186ac815f6305575bd3fe65712b1e6417d7ec2fc8b35cea673f549754ec89f66`
- 行数：165

```cpp
#pragma once

#include <cstdint>
#include <cstring>

namespace wbc_rt {

static constexpr uint32_t kMagic = 0x57424331u;  // 'WBC1'
static constexpr uint32_t kVersion = 3;

enum Cmd : uint32_t {
  kCmdNone = 0,
  kCmdStep = 1,
  kCmdEnable = 2,
  kCmdStop = 3,
  kCmdReset = 4,
  kCmdBeginHybrid = 5,
  kCmdSetRailMode = 6,
  kCmdSetFlags = 7,
  kCmdPlanStroke = 8,
  kCmdSetStroke = 9,
  kCmdSetRailPoseTarget = 10,
  kCmdCaptureRailExtRef = 11,
  kCmdSetRailExtMode = 12,
  kCmdShutdown = 13,
};

enum InFlag : uint32_t {
  kInContact = 1u << 0,
  kInStale = 1u << 1,
  kInHasQdotFf = 1u << 2,
  kInHasPoseD = 1u << 3,
  kInHasVelFf = 1u << 4,
  kInHasRailV = 1u << 5,
  kInHasVForce = 1u << 6,
  kInHasPathTwist = 1u << 7,
  kInHasFeedbackTwist = 1u << 8,
  kInSeedQcmd = 1u << 9,
  kInHasPosture = 1u << 10,
  kInHasQStar = 1u << 11,
};

enum CtrlFlag : uint32_t {
  kFlagPlanDrivesRail = 1u << 0,
  kFlagDirectPtp = 1u << 1,
  kFlagArmSuppress = 1u << 2,
  kFlagCenterSuppress = 1u << 3,
  kFlagManipActive = 1u << 4,
  kFlagRailExtActive = 1u << 5,
};

enum OutFlag : uint32_t {
  kOutJointLimited = 1u << 0,
  kOutRailLimited = 1u << 1,
  kOutWallActive = 1u << 2,
  kOutSecSuppressed = 1u << 3,
  kOutStale = 1u << 4,
  kOutReady = 1u << 5,
  kOutFailed = 1u << 6,
  kOutSecondarySuppressed = 1u << 3,
};

enum Status : uint32_t {
  kStatusBoot = 0,
  kStatusReady = 1,
  kStatusOk = 2,
  kStatusFail = 3,
  kStatusShutdown = 4,
};

enum RailModeU : uint32_t {
  kRailCoupled = 0,
  kRailLocked = 1,
};

enum LockedStyleU : uint32_t {
  kStyleHold = 0,
  kStyleRailOnly = 1,
  kStyleTcpFixed = 2,
};

#pragma pack(push, 1)
struct WbcIn {
  uint32_t magic;
  uint32_t version;
  uint64_t seq;
  uint64_t cmd_seq;
  uint32_t cmd;
  uint32_t flags;
  double t_mono;
  double dt_wall;
  double dt_nom;
  double v_cmd[6];
  double q_meas[8];
  double rail_q;
  double rail_v;
  double v_force_z;
  double pose_d[6];
  double vel_ff[6];
  double qdot_ff[8];
  double path_twist[6];
  double feedback_twist[6];
  double cmd_f[16];
  uint32_t cmd_u[8];
};

struct WbcOut {
  uint32_t magic;
  uint32_t version;
  uint64_t seq;
  uint64_t cmd_ack;
  uint32_t status;
  uint32_t flags;
  double q_cmd[8];
  double qdot[8];
  double v_cmd_received[6];
  double v_cmd_feasible[6];
  double v_tcp_estimated[6];
  double task_residual[6];
  double slack;
  double e_qp;
  double u_alloc;
  double u_mid;
  double v_r_ref;
  double psi;
  double d_star;
  double d_pref;
  double solve_ms;
  double sigma_min;
  double sigma_arm;
  double cmd_f[8];
  uint32_t joint_limited;
  uint32_t rail_limited;
  uint32_t wall_active;
  uint32_t secondary_suppressed;
  double ns_norm;
  double ns_centering;
  double ns_manip;
  double ns_arm_angle;
  double ns_damping;
  double ns_rail_lock;
  double sat_scale;
  double sec_target_norm;
  double homotopy_s;
  double psi_star;
  double rail_motion_share;
};
#pragma pack(pop)

static_assert(sizeof(WbcIn) == 608, "WbcIn layout drift");
static_assert(sizeof(WbcOut) == 608, "WbcOut layout drift");

inline void clear_in(WbcIn* s) {
  std::memset(s, 0, sizeof(WbcIn));
  s->magic = kMagic;
  s->version = kVersion;
}

inline void clear_out(WbcOut* s) {
  std::memset(s, 0, sizeof(WbcOut));
  s->magic = kMagic;
  s->version = kVersion;
}

}  // namespace wbc_rt
```

## loop.py 吸引子相关摘录

- 文件：`rm75_control/rm75_control/control/joint_admittance_8dof/loop.py`
- sha256：`b78ab622bc1789cfe45546c9397cdb9f84cae9b5184eeb10344cb251761eaee2`
- 总行数：5654

### lines 108–120: homotopy comment

```python
    Watchdog,
    clamp_command_step,
    integration_period,
)

# Pure rotation used to skip hold_setpoint (only vff[:3] was checked), so
# homotopy q* chased live IK while the stick twisted J1 to −163°.
_HOLD_ROT_THR_RAD_S = 0.05
_QUIESCENT_LIN_M_S = 0.005
_QUIESCENT_ROT_RAD_S = 0.05
_QUIESCENT_TCP_M_S = 0.010
_QUIESCENT_HOLD_S = 0.15
_RAIL_V_DRIVE_CAP_M_S = 0.12
```
### lines 868–904: _latch_attractor_from_q / _publish_homotopy_centering

```python

    def _latch_attractor_from_q(self, q_meas: np.ndarray) -> None:
        """Yaml signs for the branch barrier; homotopy q* starts at live q.

        Publishing the yaml photo as q* at t=0 pinned J1 to −90° while d*
        was still the live split.  Barrier signs stay on the design family
        so a planar J1≈0 start can still fold toward −90°.
        """
        q = np.asarray(q_meas, dtype=float).reshape(-1)
        if q.size != self.kin.nv or not np.all(np.isfinite(q)):
            return
        q_nominal = np.asarray(self.centering_task._q_target_default, dtype=float)
        self.core.set_q_star_signs(q_nominal)
        q_star = q.copy()
        if self.posture_retarget is not None and self.posture_retarget.q_star_rad is not None:
            qh = np.asarray(self.posture_retarget.q_star_rad, dtype=float).reshape(-1)
            if qh.size == q.size:
                q_star = qh
        self.centering_task.set_q_target(q_star)
        self.core.set_q_star(q_star.copy())
        if self.arm_task is not None:
            self.arm_task.reset(q)
        self._check_design_family(q)

    def _publish_homotopy_centering(self) -> None:
        """Homotopy q* while s<1; yaml nominal after s≈1."""
        if self.posture_retarget is None:
            return
        if float(self.posture_retarget.homotopy_s) >= 1.0 - 1.0e-6:
            self.centering_task.set_q_target(None)
            self.core.set_q_star(np.asarray(self.centering_task.q_target, dtype=float))
            return
        qh = self.posture_retarget.q_star_rad
        if qh is not None and np.asarray(qh).size == self.kin.nv:
            self.centering_task.set_q_target(np.asarray(qh, dtype=float))
            self.core.set_q_star(np.asarray(qh, dtype=float))
```
### lines 1800–1870: rail e_mid live TCP Y + allocate_rail

```python
                    )
            v_ext, w_ext = self.rail_ext_task(
                q_prev,
                sigma_scale=sig_scale,
                sigma_grad_rail=self._sigma_grad_rail_cached,
                vel_ff=vel_ff,
                dt_s=float(dt),
                joint_margin_frac=joint_margin_frac,
                sigma_raw=sigma_now,
                # Mid-range e_mid = (y_tcp − d*) − y_rail. SERVO_TWIST pose_d.y
                # stays at set_origin; that latch must not become y_des.
                y_tcp_d=float(pose_now[1]),
                press_stalled=allow_press_escape,
                tool_y_err_m=tool_y_err_m,
                stroke_limiters=stroke_planned,
                apply_d_band=not homing_split,
                block_escape=block_escape,
                unload_sign=unload_sign,
                jacobian=J_pre,
            )
            rail_ext_err = self.rail_ext_task.last_err_m
            rail_escape_active = bool(self.rail_ext_task._escape_active)
            # Prefer projected MotionReference FF over joint-plan rail FF.
            if np.isfinite(getattr(self.rail_ext_task, "last_v_ff", float("nan"))):
                rail_qdot_ff_val = float(self.rail_ext_task.last_v_ff)
            rail_task_weight = w_ext
            # Escape (and only escape) still comes from the extension task.
            # Cartesian mid-ranging and allocate_rail own the committed
            # rail velocity below; w_ext only sets QP2 preference strength.
            if abs(float(self.rail_ext_task.last_v_escape)) > 1.0e-4:
                rail_task_vel = v_ext
            if not self._manipulability_active and sigma_esc_ref > 1e-9:
                manip_weight = smoothstep01(
                    (sigma_esc_ref - float(sigma_now)) / sigma_esc_ref
                )

        arm_qdot_pref = None
        if self._rail_mode == RailMode.COUPLED and not locked_hold:
            lam = sr_damping_lambda(sigma_pre, self.cfg.qp.sr_damping)
            mw = margin_weight_from_activation(
                q_prev,
                self.centering_task.q_mid,
                self.centering_task.half,
                k_margin=float(self.rail_allocator_cfg.k_margin),
                activation=self.centering_task.cfg.activation,
            )
            u_alloc, _q_all = allocate_rail(
                J_pre,
                twist_base,
                qdot_scale=np.asarray(self.limits.v_max, dtype=float),
                margin_weight=mw,
                lam=lam,
                v0_m_s=float(self.rail_allocator_cfg.v0_m_s),
                w0_rad_s=float(self.rail_allocator_cfg.w0_rad_s),
                e_mid=(
                    float(self.rail_ext_task.last_e_mid_m)
                    if self.rail_ext_task is not None
                    else 0.0
                ),
                k_err=float(self.rail_allocator_cfg.k_err_rail),
                e_ref=float(self.rail_allocator_cfg.e_ref_m),
            )
            u_escape = 0.0
            if self.rail_ext_task is not None:
                u_escape = float(self.rail_ext_task.last_v_escape)
                cap = max(float(self.rail_allocator_cfg.u_mid_max_m_s), 0.0)
                if cap > 0.0:
                    u_escape = float(np.clip(u_escape, -cap, cap))
            e_mid = (
                float(self.rail_ext_task.last_e_mid_m)
                if self.rail_ext_task is not None
```
### lines 3074–3082: CSV attractor columns

```python
           "rail_exec_for_qp",
           # Chan-Dubey reg multipliers: rail first, then the worst arm joint.
           "wln_scale_rail", "wln_scale_arm_max",
           "waste_ratio", "rail_ff_m", "rail_posture_err_m",
           "rail_escape_active",
           "psi_deg", "psi_ref_deg", "psi_retarget_score", "d_pref_m",
           "d_star_m", "psi_star_deg", "homotopy_s", "minmax_margin",
           "elbow_margin_rad", "wrist_open_rad", "family_ok",
           "tool_y_des_m", "tool_y_err_mm",
```

## 附录：10 Hz 吸引子列 CSV

全时序吸引子列见 `MD/todo_controller_logs/run_20260824_223425_attractor.csv`（2890826 bytes）。下面是每 20 行抽 1 行（约 10 Hz）。

```csv
t_ref_s,phase,controller_mode,q_cmd_0,q_cmd_1,q_cmd_4,q_cmd_6,q_meas_0,q_meas_4,pose_y,pose_d_y,d_star_m,d_pref_m,psi_deg,psi_star_deg,homotopy_s,rail_motion_share,u_alloc,u_mid,u_posture,wall_active,rail_limited,qpik_nullspace_norm,qpik_nullspace_centering_norm,qpik_nullspace_arm_angle_norm,pad_vy,pad_vcmd_base_vy,rail_meas_m,elbow_margin_rad,wrist_open_rad,family_ok,quiescent,secondary_suppressed,sigma_min,comfort_slack_j4
0.0050,servo_twist,qpik,0.020180,-1.296048,2.259084,1.591062,0.020180,2.259067,-0.168641,-0.168641,-0.188950,-0.188950,41.9982,68.0000,0.000169,0.7263,0.000000,0.000000,,1,0,0.007506,1.992432,0.004358,,,0.020181,,,1,0,0,0.14518,0.000000000e+00
0.1049,servo_twist,qpik,0.020180,-1.296426,2.259084,1.591602,0.020180,2.259084,-0.168661,-0.168661,-0.190951,-0.190951,44.4985,68.0000,0.003543,0.4999,0.000000,0.000000,,1,0,0.629426,1.993930,0.092677,,,0.020180,,,1,0,0,0.14518,0.000000000e+00
0.2049,servo_twist,qpik,0.020180,-1.297204,2.259084,1.592716,0.020181,2.259067,-0.168622,-0.168622,-0.190951,-0.190951,47.0103,68.0000,0.003543,0.7268,0.000000,0.000000,,1,0,0.628143,1.993929,0.182639,,,0.020181,,,1,0,1,0.14518,0.000000000e+00
0.3051,servo_twist,qpik,0.020181,-1.298432,2.259084,1.594472,0.020181,2.259084,-0.168708,-0.168708,-0.190951,-0.190951,49.5241,68.0000,0.003543,0.7717,0.000000,0.000000,,1,0,0.058012,1.993929,0.273997,,,0.020181,,,1,0,1,0.14522,0.000000000e+00
0.4049,servo_twist,qpik,0.020180,-1.300161,2.259084,1.596942,0.020180,2.259119,-0.168820,-0.168820,-0.190951,-0.190951,52.0374,68.0000,0.003543,0.4998,0.000000,0.000000,,1,0,0.002069,1.993931,0.366905,,,0.020180,,,1,0,1,0.14528,0.000000000e+00
0.5049,servo_twist,qpik,0.020181,-1.302381,2.259084,1.600111,0.020180,2.259119,-0.168929,-0.168929,-0.190951,-0.190951,54.6065,68.0000,0.003543,0.5000,0.000000,0.000000,,1,0,0.000069,1.993933,0.463248,,,0.020180,,,1,0,1,0.14538,0.000000000e+00
0.6049,servo_twist,qpik,0.020180,-1.304948,2.259084,1.603767,0.020180,2.259084,-0.168928,-0.168928,-0.190951,-0.190951,57.1291,68.0000,0.003543,0.4986,0.000000,0.000000,,1,0,0.000002,1.993940,0.559128,,,0.020180,,,1,0,1,0.14549,0.000000000e+00
0.7049,servo_twist,qpik,0.020181,-1.307838,2.259084,1.607877,0.020181,2.259067,-0.168864,-0.168864,-0.190951,-0.190951,58.8892,68.0000,0.003543,0.7199,0.000000,0.000000,,1,0,0.000000,1.993948,0.629345,,,0.020181,,,1,0,1,0.14562,0.000000000e+00
0.8049,servo_twist,qpik,0.020181,-1.311029,2.259084,1.612405,0.020181,2.259084,-0.168883,-0.168883,-0.190951,-0.190951,58.8892,68.0000,0.003543,0.4979,0.000000,0.000000,,1,0,0.000000,1.993962,0.638955,,,0.020180,,,1,0,1,0.14575,0.000000000e+00
0.9049,servo_twist,qpik,0.020181,-1.314484,2.259084,1.617296,0.020181,2.259067,-0.168891,-0.168891,-0.190951,-0.190951,58.8892,68.0000,0.003543,0.7087,0.000000,0.000000,,1,0,0.000000,1.993979,0.649347,,,0.020181,,,1,0,1,0.14590,0.000000000e+00
1.0049,servo_twist,qpik,0.020181,-1.317373,2.257292,1.622252,0.020180,2.258962,-0.168746,-0.168746,-0.190951,-0.190951,58.8892,68.0000,0.000000,0.0000,0.000766,0.000000,,1,0,0.000000,1.987585,0.662089,,,0.020180,,,1,0,0,0.14607,0.000000000e+00
1.1059,servo_twist,qpik,0.020196,-1.311898,2.243196,1.621016,0.020200,2.248473,-0.166312,-0.166312,-0.190843,-0.190843,58.8892,68.0000,0.084898,0.0093,0.003056,0.000000,,1,0,0.871906,1.949698,0.677360,,,0.020181,,,1,0,0,0.14776,0.000000000e+00
1.2056,servo_twist,qpik,0.020356,-1.296011,2.218602,1.609133,0.020341,2.226621,-0.159655,-0.159655,-0.189046,-0.189046,58.8892,68.0000,0.361044,0.0310,0.005157,0.003743,,1,0,1.057374,1.868602,0.682190,,,0.020181,,,1,0,0,0.15088,0.000000000e+00
1.3049,servo_twist,qpik,0.020523,-1.274183,2.187214,1.590288,0.020474,2.198382,-0.151904,-0.151904,-0.187169,-0.187169,58.8892,68.0000,0.635500,0.0436,0.009005,0.006790,,1,0,1.049654,1.780354,0.675429,,,0.020184,,,1,0,0,0.15457,0.000000000e+00
1.4049,servo_twist,qpik,0.020887,-1.249151,2.148953,1.567269,0.020815,2.159461,-0.141737,-0.141737,-0.185517,-0.185517,58.8892,68.0000,0.913123,0.0731,0.013829,0.010628,,1,0,1.040475,1.690990,0.660483,,,0.020655,,,1,0,0,0.15945,0.000000000e+00
1.5049,servo_twist,qpik,0.021871,-1.224108,2.109105,1.542238,0.021809,2.119947,-0.131130,-0.131130,-0.185000,-0.185000,59.3893,68.0000,1.000000,0.1140,0.018929,0.013644,,1,0,0.990488,1.600787,0.653561,,,0.021524,,,1,0,0,0.16408,0.000000000e+00
1.6049,servo_twist,qpik,0.023410,-1.199838,2.068122,1.516450,0.023323,2.078251,-0.119742,-0.119742,-0.185000,-0.185000,60.3895,68.0000,1.000000,0.1530,0.024480,0.016104,,1,0,0.925192,1.465866,0.657301,,,0.023102,,,1,0,0,0.16867,0.000000000e+00
1.7049,servo_twist,qpik,0.025502,-1.176552,2.025987,1.490578,0.025386,2.037358,-0.107702,-0.107702,-0.185000,-0.185000,61.3895,68.0000,1.000000,0.1895,0.029979,0.018064,,1,0,0.810289,1.331029,0.657026,,,0.025336,,,1,0,0,0.17291,0.000000000e+00
1.8049,servo_twist,qpik,0.028162,-1.154364,1.983022,1.465031,0.028030,1.995993,-0.096193,-0.096193,-0.183793,-0.183793,62.5147,68.0000,1.000000,0.2235,0.034081,0.017069,,1,0,0.735978,1.195535,0.658153,,,0.027613,,,1,0,0,0.17709,0.000000000e+00
1.9049,servo_twist,qpik,0.031397,-1.132954,1.938784,1.439729,0.031237,1.950109,-0.082576,-0.082576,-0.181766,-0.181766,63.7649,68.0000,1.000000,0.2506,0.038736,0.017576,,0,0,0.674048,1.058297,0.661009,,,0.030965,,,1,0,0,0.18164,0.000000000e+00
2.0064,servo_twist,qpik,0.035079,-1.112368,1.893562,1.414968,0.034868,1.906807,-0.069021,-0.069021,-0.179731,-0.179731,65.0269,68.0000,1.000000,0.2707,0.043360,0.017999,,0,0,0.650191,0.923673,0.663199,,,0.034805,,,1,0,0,0.18596,0.000000000e+00
2.1061,servo_twist,qpik,0.039131,-1.095009,1.850886,1.392793,0.038910,1.862476,-0.055077,-0.055077,-0.177796,-0.177796,66.2937,68.0000,0.009572,0.3735,0.048461,0.018511,,0,0,0.638455,1.086399,0.662821,,,0.038403,,,1,0,0,0.19045,0.000000000e+00
2.2049,servo_twist,qpik,0.043687,-1.078832,1.818780,1.372026,0.043451,1.827377,-0.043674,-0.043674,-0.175777,-0.175777,67.5627,68.0000,0.028641,0.4572,0.051028,0.016548,,0,0,0.707060,0.986128,0.661154,,,0.043212,,,1,0,0,0.19394,0.000000000e+00
2.3049,servo_twist,qpik,0.048566,-1.060958,1.794669,1.349834,0.048325,1.801756,-0.033593,-0.033593,-0.173775,-0.173775,68.0000,68.0000,0.045838,0.5135,0.053535,0.014594,,0,0,0.688539,0.883165,0.624066,,,0.047358,,,1,0,0,0.19638,0.000000000e+00
2.4049,servo_twist,qpik,0.053797,-1.042592,1.772078,1.326719,0.053530,1.778333,-0.023567,-0.023567,-0.173537,-0.173537,68.0000,68.0000,0.071170,0.5417,0.056452,0.015339,,0,0,0.630484,0.787306,0.566601,,,0.052915,,,1,0,0,0.19798,0.000000000e+00
2.5049,servo_twist,qpik,0.059269,-1.024762,1.750495,1.303975,0.058994,1.756744,-0.013526,-0.013526,-0.172398,-0.172398,68.0000,68.0000,0.103607,0.5721,0.059553,0.014455,,0,0,0.581450,0.695527,0.507743,,,0.058763,,,1,0,0,0.19892,0.000000000e+00
2.6049,servo_twist,qpik,0.065090,-1.007820,1.730214,1.282357,0.064784,1.736149,-0.003618,-0.003618,-0.170375,-0.170375,68.0000,68.0000,0.131409,0.6020,0.062744,0.013283,,0,0,0.526375,0.608678,0.449569,,,0.063680,,,1,0,0,0.19967,0.000000000e+00
2.7049,servo_twist,qpik,0.071158,-0.991801,1.711008,1.262120,0.070847,1.717212,0.006611,0.006611,-0.168345,-0.168345,68.0000,68.0000,0.155947,0.6348,0.065799,0.013044,,0,0,0.487491,0.533469,0.393061,,,0.070164,,,1,0,0,0.20011,0.000000000e+00
2.8049,servo_twist,qpik,0.077576,-0.976769,1.693179,1.243494,0.077248,1.698083,0.016541,0.016541,-0.166340,-0.166340,68.0000,68.0000,0.177581,0.6667,0.068696,0.012540,,0,0,0.442231,0.470624,0.338698,,,0.076945,,,1,0,0,0.20066,0.000000000e+00
2.9049,servo_twist,qpik,0.084281,-0.962773,1.676381,1.226724,0.083943,1.680665,0.026984,0.026984,-0.164339,-0.164339,68.0000,68.0000,0.197079,0.6964,0.071190,0.012580,,0,0,0.417976,0.422791,0.287339,,,0.082687,,,1,0,0,0.20102,0.000000000e+00
3.0049,servo_twist,qpik,0.091248,-0.949990,1.659431,1.212572,0.090913,1.664311,0.036762,0.036762,-0.162339,-0.162339,68.0000,68.0000,0.214844,0.7193,0.073071,0.012107,,0,0,0.396631,0.391715,0.240375,,,0.090183,,,1,0,0,0.20161,0.000000000e+00
3.1049,servo_twist,qpik,0.098422,-0.937072,1.645336,1.198099,0.098055,1.649162,0.046805,0.046805,-0.160339,-0.160339,68.0000,68.0000,0.231153,0.7432,0.074213,0.011784,,0,0,0.375276,0.376054,0.192179,,,0.097975,,,1,0,0,0.20192,0.000000000e+00
3.2049,servo_twist,qpik,0.105760,-0.926023,1.630478,1.187196,0.105390,1.635199,0.056713,0.056713,-0.161944,-0.161944,68.0000,68.0000,0.265104,0.7112,0.074627,0.014426,,0,0,0.365334,0.387754,0.150841,,,0.104189,,,1,0,0,0.20224,0.000000000e+00
3.3049,servo_twist,qpik,0.113200,-0.917992,1.608408,1.182400,0.112832,1.614674,0.067317,0.067317,-0.159941,-0.159941,68.0000,68.0000,0.293962,0.6905,0.074918,0.013493,,0,0,0.425294,0.393715,0.118162,,,0.111782,,,1,0,0,0.20376,0.000000000e+00
3.4049,servo_twist,qpik,0.120674,-0.909452,1.593183,1.175247,0.120302,1.596837,0.077793,0.077793,-0.157941,-0.157941,68.0000,68.0000,0.318933,0.7708,0.075208,0.012738,,0,0,0.395209,0.396506,0.084417,,,0.120077,,,1,0,0,0.20484,0.000000000e+00
3.5049,servo_twist,qpik,0.128168,-0.901428,1.580553,1.169686,0.127792,1.584026,0.087771,0.087771,-0.155941,-0.155941,68.0000,68.0000,0.341181,0.7694,0.075441,0.012249,,0,0,0.364685,0.397845,0.055809,,,0.127328,,,1,0,0,0.20540,0.000000000e+00
3.6049,servo_twist,qpik,0.135704,-0.894953,1.567248,1.166660,0.135318,1.570709,0.097408,0.097408,-0.153937,-0.153937,68.0000,68.0000,0.361278,0.7770,0.075690,0.011591,,0,0,0.364811,0.399605,0.033210,,,0.135307,,,1,0,0,0.20628,0.000000000e+00
3.7055,servo_twist,qpik,0.143381,-0.889277,1.553900,1.165583,0.142919,1.557479,0.107161,0.107161,-0.151879,-0.151879,68.0000,68.0000,0.380065,0.7793,0.075898,0.011832,,0,0,0.364559,0.401135,0.015493,,,0.141971,,,1,0,0,0.20726,0.000000000e+00
3.8052,servo_twist,qpik,0.150900,-0.884520,1.541058,1.166341,0.150526,1.544355,0.117173,0.117173,-0.149872,-0.149872,68.0000,68.0000,0.396893,0.7828,0.076132,0.011749,,0,0,0.363017,0.402349,0.003032,,,0.150580,,,1,0,0,0.20828,0.000000000e+00
3.9049,servo_twist,qpik,0.158475,-0.880680,1.527904,1.168682,0.158095,1.531439,0.127043,0.127043,-0.151073,-0.151073,68.0000,68.0000,0.428481,0.7840,0.076371,0.014509,,0,0,0.365936,0.414265,0.005308,,,0.157704,,,1,0,0,0.20936,0.000000000e+00
4.0049,servo_twist,qpik,0.166094,-0.877487,1.514568,1.172308,0.165710,1.518489,0.136609,0.136609,-0.149921,-0.149921,68.0000,68.0000,0.459434,0.7864,0.076607,0.013252,,0,0,0.371480,0.418409,0.010256,,,0.165696,,,1,0,0,0.21060,0.000000000e+00
4.1049,servo_twist,qpik,0.173753,-0.874819,1.501089,1.176979,0.173364,1.504369,0.146576,0.146576,-0.147895,-0.147895,68.0000,68.0000,0.486509,0.7891,0.076860,0.012514,,0,0,0.371941,0.419643,0.012457,,,0.172905,,,1,0,0,0.21197,0.000000000e+00
4.2049,servo_twist,qpik,0.181271,-0.872728,1.487400,1.182433,0.180799,1.491244,0.155898,0.155898,-0.145882,-0.145882,68.0000,68.0000,0.510319,0.7498,0.077094,0.010284,,0,0,0.372050,0.420942,0.012467,,,0.179966,,,1,0,0,0.21333,0.000000000e+00
4.3049,servo_twist,qpik,0.188848,-0.871361,1.472932,1.188669,0.188468,1.476165,0.165765,0.165765,-0.143882,-0.143882,68.0000,68.0000,0.531562,0.7915,0.077352,0.011155,,0,0,0.379672,0.422778,0.010711,,,0.187216,,,1,0,0,0.21492,0.000000000e+00
4.4049,servo_twist,qpik,0.196592,-0.869841,1.459305,1.195137,0.196180,1.462219,0.175448,0.175448,-0.141868,-0.141868,68.0000,68.0000,0.550936,0.7979,0.077605,0.011403,,0,0,0.374842,0.423883,0.008001,,,0.195708,,,1,0,0,0.21637,0.000000000e+00
4.5049,servo_twist,qpik,0.204148,-0.868896,1.444958,1.202065,0.203764,1.448554,0.184873,0.184873,-0.139868,-0.139868,68.0000,68.0000,0.568547,0.7931,0.077877,0.010613,,0,0,0.382604,0.425402,0.004388,,,0.203429,,,1,0,0,0.21778,0.000000000e+00
4.6049,servo_twist,qpik,0.211679,-0.868434,1.429778,1.209438,0.211295,1.432985,0.194724,0.194724,-0.139855,-0.139855,68.0000,68.0000,0.596093,0.7917,0.078179,0.012403,,0,0,0.381201,0.433945,0.000224,,,0.211181,,,1,0,0,0.21936,0.000000000e+00
4.7049,servo_twist,qpik,0.219389,-0.867601,1.415377,1.216783,0.219004,1.419284,0.204354,0.204354,-0.140319,-0.140319,68.0000,68.0000,0.630495,0.7946,0.078447,0.013478,,0,0,0.389230,0.443512,0.003976,,,0.218693,,,1,0,0,0.22073,0.000000000e+00
4.8049,servo_twist,qpik,0.227263,-0.866727,1.401156,1.224148,0.226824,1.404658,0.214103,0.214103,-0.138262,-0.138262,68.0000,68.0000,0.660210,0.8069,0.078749,0.012488,,0,0,0.390848,0.444498,0.008077,,,0.226023,,,1,0,0,0.22218,0.000000000e+00
4.9059,servo_twist,qpik,0.235159,-0.865880,1.387281,1.231356,0.234766,1.390783,0.223863,0.223863,-0.136241,-0.136241,68.0000,68.0000,0.685786,0.8108,0.079055,0.012149,,0,0,0.391583,0.445216,0.011816,,,0.234196,,,1,0,0,0.22350,0.000000000e+00
5.0051,servo_twist,qpik,0.243011,-0.865068,1.373364,1.238558,0.242594,1.377100,0.233593,0.233593,-0.134217,-0.134217,68.0000,68.0000,0.708589,0.8139,0.079346,0.011855,,0,0,0.392951,0.445979,0.015296,,,0.241567,,,1,0,0,0.22473,0.000000000e+00
5.1070,servo_twist,qpik,0.251066,-0.864279,1.359248,1.245806,0.250677,1.362317,0.243501,0.243501,-0.132168,-0.132168,68.0000,68.0000,0.729371,0.8171,0.079660,0.011809,,0,0,0.394748,0.446496,0.018317,,,0.250187,,,1,0,0,0.22605,0.000000000e+00
5.2113,servo_twist,qpik,0.259429,-0.863334,1.345166,1.253057,0.259020,1.347569,0.253767,0.253767,-0.133011,-0.133011,68.0000,68.0000,0.766890,0.8253,0.079984,0.014551,,0,0,0.398052,0.456767,0.020913,,,0.258356,,,1,0,0,0.22730,0.000000000e+00
5.3113,servo_twist,qpik,0.267400,-0.862569,1.331413,1.259993,0.267004,1.334269,0.263305,0.263305,-0.134211,-0.134211,68.0000,68.0000,0.804556,0.8234,0.080298,0.015718,,0,0,0.405058,0.467897,0.022883,,,0.267129,,,1,0,0,0.22841,0.000000000e+00
5.4113,servo_twist,qpik,0.275422,-0.861833,1.317564,1.266875,0.275014,1.321005,0.272889,0.272889,-0.134416,-0.134416,68.0000,68.0000,0.843391,0.8257,0.080595,0.015374,,0,0,0.413429,0.475680,0.024317,,,0.274572,,,1,0,0,0.22947,0.000000000e+00
5.5113,servo_twist,qpik,0.283153,-0.862048,1.302182,1.273949,0.282807,1.306135,0.282696,0.282696,-0.135205,-0.135205,68.0000,68.0000,0.884848,0.7973,0.083840,0.014881,,0,0,0.426034,0.486038,0.024997,,,0.282092,,,1,0,0,0.23060,0.000000000e+00
5.6113,servo_twist,qpik,0.291441,-0.864140,1.280722,1.285765,0.291015,1.288036,0.292703,0.292703,-0.137006,-0.137006,68.0000,68.0000,0.930794,0.7500,0.100028,0.016870,,0,0,0.462215,0.499155,0.026277,,,0.290882,,,1,0,0,0.23166,0.000000000e+00
5.7113,servo_twist,qpik,0.300827,-0.867288,1.249696,1.309561,0.300265,1.260040,0.302715,0.302715,-0.139080,-0.139080,68.0000,68.0000,0.981703,0.4920,0.107707,0.017066,,0,0,0.526803,0.509898,0.037798,,,0.298520,,,1,0,0,0.23261,0.000000000e+00
5.8113,servo_twist,qpik,0.310996,-0.869854,1.219780,1.335699,0.310483,1.228694,0.312468,0.312468,-0.139699,-0.139699,68.0000,68.0000,1.000000,0.4922,0.110287,0.015038,,0,0,0.535450,0.895610,0.051559,,,0.309251,,,1,0,0,0.23241,0.000000000e+00
5.9113,servo_twist,qpik,0.321755,-0.871714,1.191281,1.362460,0.321214,1.200612,0.322216,0.322216,-0.141719,-0.141719,68.0000,68.0000,1.000000,0.4925,0.112747,0.015094,,0,0,0.597838,0.927004,0.068573,,,0.320740,,,1,0,0,0.23152,0.000000000e+00
6.0113,servo_twist,qpik,0.332902,-0.873828,1.163011,1.390303,0.332338,1.171831,0.331289,0.331289,-0.142167,-0.142167,68.0000,68.0000,1.000000,0.4930,0.115297,0.012727,,0,0,0.617377,0.959341,0.091100,,,0.332283,,,1,0,0,0.23009,0.000000000e+00
6.1113,servo_twist,qpik,0.344185,-0.875468,1.138100,1.415639,0.343635,1.144604,0.341045,0.341045,-0.143767,-0.143767,68.0000,68.0000,1.000000,0.7267,0.101136,0.013296,,0,0,0.622087,0.991441,0.115080,,,0.341900,,,1,0,0,0.22845,0.000000000e+00
6.2113,servo_twist,qpik,0.354829,-0.876256,1.120122,1.432915,0.354304,1.125458,0.350484,0.350484,-0.145770,-0.145770,68.0000,68.0000,1.000000,0.9868,0.084854,0.014203,,0,0,0.567742,1.020349,0.129430,,,0.353439,,,1,0,0,0.22750,0.000000000e+00
6.3113,servo_twist,qpik,0.363560,-0.878704,1.107368,1.442913,0.363070,1.110972,0.359495,0.359495,-0.147840,-0.147840,68.0000,68.0000,1.000000,0.7991,0.084682,0.010214,,0,0,0.508686,1.042639,0.142584,,,0.363176,,,1,0,0,0.22758,0.000000000e+00
6.4113,servo_twist,qpik,0.372033,-0.883272,1.093076,1.451938,0.371614,1.096905,0.369256,0.369256,-0.149841,-0.149841,68.0000,68.0000,1.000000,0.8523,0.085014,0.015065,,0,0,0.504508,1.063960,0.157476,,,0.370242,,,1,0,0,0.22840,0.000000000e+00
6.5113,servo_twist,qpik,0.380543,-0.887250,1.079869,1.460710,0.380107,1.083483,0.379001,0.379001,-0.151842,-0.151842,68.0000,68.0000,1.000000,0.8466,0.085357,0.017381,,0,0,0.508916,1.085336,0.172198,,,0.378989,,,1,0,0,0.22913,0.000000000e+00
6.6113,servo_twist,qpik,0.389089,-0.891303,1.066351,1.469422,0.388647,1.070044,0.389066,0.389066,-0.153842,-0.153842,68.0000,68.0000,1.000000,0.8466,0.085738,0.018842,,0,0,0.516361,1.106902,0.186482,,,0.387942,,,1,0,0,0.22980,0.000000000e+00
6.7113,servo_twist,qpik,0.397649,-0.895310,1.052611,1.478057,0.397210,1.056570,0.399102,0.399102,-0.155859,-0.155859,68.0000,68.0000,1.000000,0.8498,0.086099,0.019173,,0,0,0.526727,1.128609,0.200197,,,0.395625,,,1,0,0,0.23049,0.000000000e+00
6.8113,servo_twist,qpik,0.406193,-0.899676,1.037647,1.486906,0.405699,1.040967,0.409300,0.409300,-0.157933,-0.157933,68.0000,68.0000,1.000000,0.8495,0.086521,0.019850,,0,0,0.535874,1.150489,0.213331,,,0.404935,,,1,0,0,0.23132,0.000000000e+00
6.9113,servo_twist,qpik,0.414718,-0.903378,1.023915,1.495050,0.414294,1.027877,0.419242,0.419242,-0.159944,-0.159944,68.0000,68.0000,1.000000,0.8566,0.086859,0.018758,,0,0,0.544142,1.172305,0.225059,,,0.413388,,,1,0,0,0.23197,0.000000000e+00
7.0120,servo_twist,qpik,0.423421,-0.907695,1.008671,1.503622,0.422851,1.011907,0.429259,0.429259,-0.162001,-0.162001,68.0000,68.0000,1.000000,0.8598,0.087275,0.020293,,0,0,0.559378,1.194578,0.236838,,,0.422128,,,1,0,0,0.23278,0.000000000e+00
7.1116,servo_twist,qpik,0.432150,-0.910570,0.996668,1.511265,0.431681,0.999934,0.439181,0.439181,-0.164010,-0.164010,68.0000,68.0000,1.000000,0.8698,0.087607,0.021497,,0,0,0.559753,1.216888,0.248387,,,0.431213,,,1,0,0,0.23329,0.000000000e+00
7.2113,servo_twist,qpik,0.440844,-0.913819,0.983511,1.518900,0.440409,0.986373,0.449262,0.449262,-0.166010,-0.166010,68.0000,68.0000,1.000000,0.8683,0.087969,0.021613,,0,0,0.567604,1.239177,0.258768,,,0.439855,,,1,0,0,0.23387,0.000000000e+00
7.3113,servo_twist,qpik,0.449719,-0.916580,0.970926,1.526087,0.449266,0.973894,0.459138,0.459138,-0.168019,-0.168019,68.0000,68.0000,1.000000,0.8783,0.088307,0.021836,,0,0,0.574561,1.261909,0.267668,,,0.449256,,,1,0,0,0.23435,0.000000000e+00
7.4113,servo_twist,qpik,0.458540,-0.919324,0.958012,1.533031,0.458110,0.961397,0.469036,0.469036,-0.170020,-0.170020,68.0000,68.0000,1.000000,0.8759,0.088642,0.021843,,0,0,0.580844,1.284600,0.275133,,,0.457629,,,1,0,0,0.23480,0.000000000e+00
7.5113,servo_twist,qpik,0.467472,-0.921779,0.945323,1.539673,0.467007,0.948586,0.478892,0.478892,-0.172034,-0.172034,68.0000,68.0000,1.000000,0.8857,0.088973,0.022129,,0,0,0.587362,1.307588,0.281325,,,0.466280,,,1,0,0,0.23525,0.000000000e+00
7.6113,servo_twist,qpik,0.476364,-0.924125,0.932603,1.546069,0.475912,0.935933,0.488876,0.488876,-0.174034,-0.174034,68.0000,68.0000,1.000000,0.8817,0.089333,0.022280,,0,0,0.592576,1.330527,0.286350,,,0.475868,,,1,0,0,0.23565,0.000000000e+00
7.7113,servo_twist,qpik,0.485278,-0.926385,0.919800,1.552288,0.484835,0.922965,0.498839,0.498839,-0.176034,-0.176034,68.0000,68.0000,1.000000,0.8851,0.089694,0.022511,,0,0,0.598641,1.353573,0.290375,,,0.484327,,,1,0,0,0.23602,0.000000000e+00
7.8113,servo_twist,qpik,0.494183,-0.928681,0.906567,1.558486,0.493741,0.910434,0.508805,0.508805,-0.178059,-0.178059,68.0000,68.0000,1.000000,0.8863,0.090022,0.022307,,0,0,0.607204,1.376685,0.293572,,,0.492880,,,1,0,0,0.23642,0.000000000e+00
7.9116,servo_twist,qpik,0.503408,-0.930480,0.894052,1.564436,0.502803,0.896139,0.518970,0.518970,-0.180122,-0.180122,68.0000,68.0000,1.000000,0.9223,0.090416,0.022998,,0,0,0.612373,1.400473,0.295992,,,0.502323,,,1,0,0,0.23678,0.000000000e+00
8.0113,servo_twist,qpik,0.512401,-0.931768,0.882730,1.569951,0.511977,0.885475,0.528923,0.528923,-0.182138,-0.182138,68.0000,68.0000,1.000000,0.9142,0.090728,0.025809,,0,0,0.610644,1.423601,0.297936,,,0.510103,,,1,0,0,0.23694,0.000000000e+00
8.1115,servo_twist,qpik,0.521853,-0.931580,0.874509,1.574678,0.521381,0.876871,0.538826,0.538826,-0.184157,-0.184157,68.0000,68.0000,1.000000,0.9100,0.090948,0.024973,,0,0,0.601407,1.447729,0.299375,,,0.520171,,,1,0,0,0.23713,0.000000000e+00
8.2121,servo_twist,qpik,0.531030,-0.933056,0.862477,1.580302,0.530582,0.864933,0.548961,0.548961,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.9036,0.091299,0.022841,,0,0,0.620210,1.471463,0.300565,,,0.529556,,,1,0,0,0.23731,0.000000000e+00
8.3113,servo_twist,qpik,0.539975,-0.934935,0.849593,1.586036,0.539509,0.852314,0.558799,0.558799,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.9013,0.091651,0.021442,,0,0,0.630334,1.494811,0.301413,,,0.538929,,,1,0,0,0.23759,0.000000000e+00
8.4113,servo_twist,qpik,0.549120,-0.936505,0.837347,1.591598,0.548664,0.839887,0.568762,0.568762,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.9086,0.091980,0.021228,,0,0,0.633610,1.518566,0.301945,,,0.547155,,,1,0,0,0.23783,0.000000000e+00
8.5113,servo_twist,qpik,0.558304,-0.937937,0.825254,1.597152,0.557848,0.827862,0.578644,0.578644,-0.184200,-0.184200,68.0000,68.0000,1.000000,0.9123,0.092295,0.020223,,0,0,0.638319,1.542423,0.302319,,,0.557812,,,1,0,0,0.23805,0.000000000e+00
8.6134,servo_twist,qpik,0.567557,-0.939934,0.811486,1.603236,0.567146,0.815016,0.588727,0.588727,-0.182114,-0.182114,68.0000,68.0000,1.000000,0.9070,0.092640,0.017798,,0,0,0.642454,1.566614,0.302564,,,0.566570,,,1,0,0,0.23833,0.000000000e+00
8.7113,servo_twist,qpik,0.576345,-0.942653,0.796531,1.609490,0.575907,0.799849,0.598380,0.598380,-0.180113,-0.180113,68.0000,68.0000,1.000000,0.9117,0.093051,0.016752,,0,0,0.658370,1.589906,0.302691,,,0.574116,,,1,0,0,0.23865,0.000000000e+00
8.8121,servo_twist,qpik,0.585646,-0.944105,0.784304,1.615105,0.585200,0.787161,0.608437,0.608437,-0.178096,-0.178096,68.0000,68.0000,1.000000,0.9230,0.093395,0.017318,,0,0,0.657064,1.614069,0.302660,,,0.584867,,,1,0,0,0.23884,0.000000000e+00
8.9113,servo_twist,qpik,0.594931,-0.945288,0.772697,1.620620,0.594451,0.775345,0.618393,0.618393,-0.176094,-0.176094,68.0000,68.0000,1.000000,0.9281,0.093723,0.017706,,0,0,0.656838,1.638256,0.302666,,,0.594489,,,1,0,0,0.23899,0.000000000e+00
9.0113,servo_twist,qpik,0.604335,-0.946313,0.761393,1.626101,0.603851,0.763547,0.628364,0.628364,-0.174066,-0.174066,68.0000,68.0000,1.000000,0.9291,0.094034,0.014988,,0,0,0.661336,1.662743,0.302734,,,0.602124,,,1,0,0,0.23911,0.000000000e+00
9.1116,servo_twist,qpik,0.613325,-0.948978,0.746622,1.632378,0.612860,0.749636,0.638177,0.638177,-0.172051,-0.172051,68.0000,68.0000,1.000000,0.9257,0.094409,0.015922,,0,0,0.679768,1.686554,0.302914,,,0.611926,,,1,0,0,0.23936,0.000000000e+00
9.2113,servo_twist,qpik,0.622282,-0.951923,0.730893,1.639040,0.621855,0.734801,0.648074,0.648074,-0.170019,-0.170019,68.0000,68.0000,1.000000,0.9052,0.094812,0.014729,,0,0,0.691287,1.710225,0.303068,,,0.620795,,,1,0,0,0.23958,0.000000000e+00
9.3113,servo_twist,qpik,0.631777,-0.953212,0.718743,1.644902,0.631304,0.721711,0.658159,0.658159,-0.167974,-0.167974,68.0000,68.0000,1.000000,0.9528,0.095168,0.016032,,0,0,0.682752,1.735026,0.303097,,,0.629873,,,1,0,0,0.23973,0.000000000e+00
9.4113,servo_twist,qpik,0.641288,-0.954028,0.707879,1.650429,0.640791,0.710209,0.668025,0.668025,-0.165968,-0.165968,68.0000,68.0000,1.000000,0.9453,0.095471,0.016104,,0,0,0.681198,1.759677,0.303282,,,0.640832,,,1,0,0,0.23976,0.000000000e+00
9.5116,servo_twist,qpik,0.650554,-0.955850,0.694791,1.656543,0.649978,0.699423,0.677607,0.677607,-0.163921,-0.163921,68.0000,68.0000,1.000000,0.8825,0.095780,0.016308,,0,0,0.685698,1.783709,0.303760,,,0.648730,,,1,0,0,0.23974,0.000000000e+00
9.6113,servo_twist,qpik,0.660402,-0.957297,0.681741,1.663262,0.659921,0.683436,0.688282,0.688282,-0.161921,-0.161921,68.0000,68.0000,1.000000,0.9645,0.096187,0.017752,,0,0,0.704966,1.809853,0.304245,,,0.659318,,,1,0,0,0.23996,0.000000000e+00
9.7113,servo_twist,qpik,0.670022,-0.957586,0.671223,1.669127,0.669532,0.673453,0.698117,0.698117,-0.159896,-0.159896,68.0000,68.0000,1.000000,0.9603,0.096483,0.016599,,0,0,0.698301,1.834822,0.305031,,,0.668417,,,1,0,0,0.23988,0.000000000e+00
9.8113,servo_twist,qpik,0.679702,-0.958019,0.660157,1.675288,0.679132,0.661899,0.707993,0.707993,-0.157870,-0.157870,68.0000,68.0000,1.000000,0.9650,0.096809,0.015739,,0,0,0.703579,1.859968,0.306161,,,0.677707,,,1,0,0,0.23983,0.000000000e+00
9.9113,servo_twist,qpik,0.689389,-0.957767,0.650797,1.681170,0.688896,0.653329,0.717724,0.717724,-0.155861,-0.155861,68.0000,68.0000,1.000000,0.9719,0.096920,0.015733,,0,0,0.703479,1.891810,0.308100,,,0.688075,,,1,0,0,0.23977,0.000000000e+00
10.0134,servo_twist,qpik,0.699306,-0.957819,0.640629,1.687388,0.698656,0.642875,0.727751,0.727751,-0.153795,-0.153795,68.0000,68.0000,1.000000,0.9516,0.096570,0.014867,,0,0,0.712282,1.959748,0.310647,,,0.698197,,,1,0,0,0.23965,0.000000000e+00
10.1113,servo_twist,qpik,0.709321,-0.956743,0.632777,1.693240,0.708725,0.632612,0.737952,0.737952,-0.151757,-0.151757,68.0000,68.0000,1.000000,0.9645,0.095607,0.017094,,0,0,0.708753,2.070446,0.313236,,,0.708951,,,1,0,0,0.23956,0.000000000e+00
10.2124,servo_twist,qpik,0.719022,-0.956147,0.624395,1.698875,0.718522,0.626957,0.747371,0.747371,-0.149729,-0.149729,68.0000,68.0000,1.000000,0.9592,0.094055,0.015346,,0,0,0.710622,2.215512,0.315849,,,0.718657,,,1,0,0,0.23940,0.000000000e+00
10.3194,servo_twist,qpik,0.729591,-0.957132,0.611436,1.706243,0.729005,0.612506,0.758298,0.758298,-0.147569,-0.147569,68.0000,68.0000,1.000000,0.9963,0.092270,0.016042,,0,0,0.740947,2.420852,0.318733,,,0.727037,,,1,0,0,0.23920,0.000000000e+00
10.4180,servo_twist,qpik,0.738819,-0.958466,0.598629,1.713138,0.738307,0.602488,0.767697,0.767697,-0.145546,-0.145546,68.0000,68.0000,1.000000,0.9329,0.090348,0.015003,,0,0,0.741981,2.638195,0.321496,,,0.737189,,,1,0,0,0.23906,0.000000000e+00
10.5179,servo_twist,qpik,0.747971,-0.961720,0.581463,1.721158,0.747522,0.585733,0.777375,0.777375,-0.143533,-0.143533,68.0000,68.0000,1.000000,0.9308,0.088566,0.016107,,0,0,0.776204,2.891339,0.324040,,,0.747053,,,1,0,0,0.23919,0.000000000e+00
10.6189,servo_twist,qpik,0.756016,-0.967876,0.558518,1.730012,0.755573,0.565818,0.786262,0.786262,-0.141504,-0.141504,68.0000,68.0000,1.000000,0.8060,0.087292,0.000000,,1,0,0.795520,3.145061,0.326179,,,0.755608,,,1,0,0,0.23942,0.000000000e+00
10.7179,servo_twist,qpik,0.762532,-0.979861,0.523474,1.741136,0.762179,0.536165,0.794722,0.794722,-0.139451,-0.139451,68.0000,68.0000,1.000000,0.6990,0.087131,0.000000,,1,0,0.855020,3.372955,0.327610,,,0.761364,,,1,0,0,0.24020,0.000000000e+00
10.8188,servo_twist,qpik,0.767901,-0.997537,0.476392,1.754575,0.767625,0.494748,0.802974,0.802974,-0.137394,-0.137394,68.0000,68.0000,1.000000,0.5929,0.088171,0.000000,,1,0,0.871589,3.576666,0.328158,,,0.767287,,,1,0,0,0.24148,0.000000000e+00
10.9193,servo_twist,qpik,0.772123,-1.018684,0.422992,1.768181,0.771909,0.439509,0.810966,0.810966,-0.135358,-0.135358,68.0000,68.0000,1.000000,0.5328,0.090607,0.000000,,1,0,0.889095,3.747535,0.326821,,,0.771891,,,1,0,0,0.24333,0.000000000e+00
11.0217,servo_twist,qpik,0.775277,-1.032464,0.390100,1.773079,0.775108,0.398546,0.817553,0.817553,-0.135058,-0.135058,68.0000,68.0000,1.000000,0.4893,0.092504,0.000000,,1,0,0.863840,3.879057,0.320182,,,0.774623,,,1,0,0,0.24500,0.000000000e+00
11.1295,servo_twist,qpik,0.777419,-1.041163,0.372236,1.772006,0.777317,0.374879,0.822606,0.822606,-0.135058,-0.135058,68.0000,68.0000,1.000000,0.4575,0.059025,0.000000,,1,0,0.813070,3.971276,0.314055,,,0.777107,,,1,0,0,0.24660,0.000000000e+00
11.2398,servo_twist,qpik,0.778667,-1.043930,0.364683,1.772791,0.778601,0.363238,0.825549,0.825549,-0.135058,-0.135058,68.0000,68.0000,1.000000,0.7521,0.000555,0.000000,,1,0,0.796262,4.025796,0.306384,,,0.778624,,,1,0,0,0.24742,0.000000000e+00
11.3466,servo_twist,qpik,0.779275,-1.040057,0.371652,1.770753,0.779248,0.365437,0.825802,0.825802,-0.135058,-0.135058,68.0000,68.0000,1.000000,0.4922,-0.000160,0.000000,,1,0,0.750921,4.052266,0.296758,,,0.779259,,,1,0,0,0.24720,0.000000000e+00
11.4517,servo_twist,qpik,0.779522,-1.036571,0.378586,1.768209,0.779511,0.375525,0.825209,0.825209,-0.135058,-0.135058,68.0000,68.0000,1.000000,0.3733,-0.001141,0.000000,,1,0,0.081170,4.062609,0.288175,,,0.779522,,,1,0,1,0.24675,0.000000000e+00
11.5557,servo_twist,qpik,0.779410,-1.028057,0.396736,1.763874,0.779430,0.387201,0.824423,0.824423,-0.135058,-0.135058,68.0000,68.0000,0.000000,0.0861,-0.053648,0.000000,,1,0,0.222740,1.666593,0.280928,,,0.779552,,,1,0,0,0.24641,0.000000000e+00
11.6622,servo_twist,qpik,0.778749,-1.011880,0.429828,1.760702,0.778920,0.417361,0.821042,0.821042,-0.135058,-0.135058,68.0000,68.0000,0.000000,0.1857,-0.090512,0.000000,,1,0,0.567139,1.635758,0.280838,,,0.779275,,,1,0,0,0.24480,0.000000000e+00
11.7726,servo_twist,qpik,0.777263,-0.990299,0.474525,1.755273,0.777401,0.459493,0.814727,0.814727,-0.135058,-0.135058,68.0000,68.0000,0.000000,0.3075,-0.087300,0.000000,,1,0,0.507680,1.569963,0.286322,,,0.777689,,,1,0,0,0.24222,0.000000000e+00
11.8819,servo_twist,qpik,0.774391,-0.962253,0.538970,1.741760,0.774558,0.517089,0.806843,0.806843,-0.133401,-0.133401,68.0000,68.0000,0.009012,0.3671,-0.084061,0.000000,,1,0,0.479524,1.466391,0.296564,,,0.774849,,,1,0,0,0.23960,0.000000000e+00
11.9814,servo_twist,qpik,0.770551,-0.931361,0.610652,1.724179,0.770754,0.587129,0.797609,0.797609,-0.131361,-0.131361,68.0000,68.0000,0.020625,0.4142,-0.080562,0.000000,,1,0,0.467884,1.336512,0.304180,,,0.770780,,,1,0,0,0.23682,0.000000000e+00
12.0794,servo_twist,qpik,0.765527,-0.903421,0.680028,1.704296,0.765814,0.650275,0.787635,0.787635,-0.129346,-0.129346,68.0000,68.0000,0.032798,0.4985,-0.078286,0.000000,,1,0,0.469351,1.173150,0.308164,,,0.766582,,,1,0,0,0.23457,0.000000000e+00
12.1800,servo_twist,qpik,0.759108,-0.875730,0.746370,1.684175,0.759434,0.715323,0.775724,0.775724,-0.127320,-0.127320,68.0000,68.0000,0.045928,0.5390,-0.076718,0.000000,,1,0,0.447311,0.983422,0.306854,,,0.759924,,,1,0,0,0.23209,0.000000000e+00
12.2807,servo_twist,qpik,0.751751,-0.847263,0.809461,1.664859,0.752150,0.782274,0.762059,0.762059,-0.125249,-0.125249,68.0000,68.0000,0.060600,0.5623,-0.075979,0.002449,,0,0,0.421477,0.784778,0.301148,,,0.752155,,,1,0,0,0.22902,0.000000000e+00
12.3822,servo_twist,qpik,0.744119,-0.826841,0.862151,1.644388,0.744510,0.842052,0.749153,0.749153,-0.123184,-0.123184,68.0000,68.0000,0.076710,0.6843,-0.076100,0.002294,,0,0,0.403038,0.611647,0.290918,,,0.744475,,,1,0,0,0.22655,0.000000000e+00
12.4894,servo_twist,qpik,0.735991,-0.814178,0.896007,1.629157,0.736375,0.883922,0.737432,0.737432,-0.121057,-0.121057,68.0000,68.0000,0.094957,0.7657,-0.077620,0.003277,,0,0,0.397786,0.464292,0.276578,,,0.736354,,,1,0,0,0.22508,0.000000000e+00
12.5893,servo_twist,qpik,0.728300,-0.804452,0.920216,1.616595,0.728681,0.908636,0.727498,0.727498,-0.119055,-0.119055,68.0000,68.0000,0.113807,0.7605,-0.079709,0.004305,,0,0,0.382782,0.373642,0.257419,,,0.729735,,,1,0,0,0.22397,0.000000000e+00
12.6976,servo_twist,qpik,0.719723,-0.793851,0.944882,1.602347,0.720137,0.935723,0.716342,0.716342,-0.116888,-0.116888,68.0000,68.0000,0.136449,0.7981,-0.081859,0.004491,,0,0,0.334280,0.330030,0.231924,,,0.721258,,,1,0,0,0.22278,0.000000000e+00
12.8073,servo_twist,qpik,0.710786,-0.784279,0.966647,1.587822,0.711294,0.958762,0.705482,0.705482,-0.114685,-0.114685,68.0000,68.0000,0.162415,0.8131,-0.083986,0.004818,,0,0,0.298226,0.333774,0.201682,,,0.712563,,,1,0,0,0.22180,0.000000000e+00
12.9159,servo_twist,qpik,0.701770,-0.775947,0.984768,1.574226,0.702200,0.981242,0.694272,0.694272,-0.112550,-0.112550,68.0000,68.0000,0.191281,0.8335,-0.085514,0.004162,,0,0,0.275130,0.357276,0.170844,,,0.703386,,,1,0,0,0.22081,0.000000000e+00
13.0148,servo_twist,qpik,0.693188,-0.767846,0.999589,1.562707,0.693697,0.995431,0.684367,0.684367,-0.110511,-0.110511,68.0000,68.0000,0.223320,0.8408,-0.086406,0.003752,,0,0,0.245098,0.374128,0.141551,,,0.694084,,,1,0,0,0.22010,0.000000000e+00
13.1150,servo_twist,qpik,0.684606,-0.758653,1.013735,1.552513,0.685034,1.008818,0.674294,0.674294,-0.108468,-0.108468,68.0000,68.0000,0.261416,0.8445,-0.086591,0.004315,,0,0,0.223548,0.373876,0.114277,,,0.684992,,,1,0,0,0.21928,0.000000000e+00
13.2171,servo_twist,qpik,0.675773,-0.749042,1.027670,1.542874,0.676338,1.023583,0.664082,0.664082,-0.106377,-0.106377,68.0000,68.0000,0.309777,0.8308,-0.086224,0.006261,,0,0,0.200062,0.359118,0.087424,,,0.676263,,,1,0,0,0.21817,0.000000000e+00
13.3209,servo_twist,qpik,0.667152,-0.738366,1.043651,1.533110,0.667628,1.039169,0.653466,0.653466,-0.105309,-0.105309,68.0000,68.0000,0.373719,0.8324,-0.085777,0.006952,,0,0,0.165779,0.346192,0.062893,,,0.667520,,,1,0,0,0.21715,0.000000000e+00
13.4329,servo_twist,qpik,0.658176,-0.726706,1.061572,1.523070,0.658613,1.057739,0.642072,0.642072,-0.107492,-0.107492,68.0000,68.0000,0.473907,0.8072,-0.085260,0.010961,,0,0,0.160685,0.343607,0.040129,,,0.658936,,,1,0,0,0.21595,0.000000000e+00
13.5373,servo_twist,qpik,0.649740,-0.714910,1.080280,1.513597,0.650215,1.076589,0.631453,0.631453,-0.109608,-0.109608,68.0000,68.0000,0.681502,0.8255,-0.084732,0.010204,,0,0,0.121182,0.340073,0.020935,,,0.650997,,,1,0,0,0.21460,0.000000000e+00
13.6396,servo_twist,qpik,0.641318,-0.704102,1.094618,1.507531,0.641730,1.091267,0.620945,0.620945,-0.111673,-0.111673,68.0000,68.0000,1.000000,0.8468,-0.082655,0.010078,,0,0,0.320519,1.739344,0.007553,,,0.642062,,,1,0,0,0.21353,0.000000000e+00
13.7410,servo_twist,qpik,0.632742,-0.693085,1.105405,1.506178,0.633183,1.103258,0.610342,0.610342,-0.113723,-0.113723,68.0000,68.0000,1.000000,0.9071,-0.079202,0.009235,,0,0,0.537012,1.718264,0.003658,,,0.633451,,,1,0,0,0.21238,0.000000000e+00
13.8492,servo_twist,qpik,0.623658,-0.681942,1.113534,1.509668,0.624018,1.112438,0.599644,0.599644,-0.115882,-0.115882,68.0000,68.0000,1.000000,0.9357,-0.074808,0.005703,,0,0,0.549186,1.696395,0.012062,,,0.624536,,,1,0,0,0.21151,0.000000000e+00
13.9582,servo_twist,qpik,0.614509,-0.673312,1.116596,1.517221,0.614983,1.115510,0.589083,0.589083,-0.118050,-0.118050,68.0000,68.0000,1.000000,0.9658,-0.073215,0.006061,,0,0,0.553316,1.674288,0.029636,,,0.615132,,,1,0,0,0.21099,0.000000000e+00
14.0617,servo_twist,qpik,0.606766,-0.661995,1.125681,1.524300,0.607079,1.123329,0.578723,0.578723,-0.120141,-0.120141,68.0000,68.0000,1.000000,0.9069,-0.072427,0.006343,,0,0,0.542539,1.655294,0.053079,,,0.607205,,,1,0,0,0.21004,0.000000000e+00
14.1659,servo_twist,qpik,0.599052,-0.650347,1.136176,1.532154,0.599523,1.133155,0.568803,0.568803,-0.122247,-0.122247,68.0000,68.0000,1.000000,0.9010,-0.071818,0.008168,,0,0,0.537749,1.636413,0.081326,,,0.599389,,,1,0,0,0.20895,0.000000000e+00
14.2690,servo_twist,qpik,0.591885,-0.638416,1.148020,1.540308,0.592319,1.144814,0.558707,0.558707,-0.124318,-0.124318,68.0000,68.0000,1.000000,0.8592,-0.070795,0.009146,,0,0,0.537000,1.618866,0.112299,,,0.593231,,,1,0,0,0.20788,0.000000000e+00
14.3779,servo_twist,qpik,0.584265,-0.626279,1.158010,1.551111,0.584598,1.156473,0.547923,0.547923,-0.126475,-0.126475,68.0000,68.0000,1.000000,0.9483,-0.065976,0.008056,,0,0,0.547560,1.600662,0.147217,,,0.585814,,,1,0,0,0.20659,0.000000000e+00
14.4812,servo_twist,qpik,0.576966,-0.615147,1.163980,1.564127,0.577324,1.163367,0.538124,0.538124,-0.128552,-0.128552,68.0000,68.0000,1.000000,0.8961,-0.063922,0.005531,,0,0,0.566654,1.583348,0.183524,,,0.578386,,,1,0,0,0.20555,0.000000000e+00
14.5864,servo_twist,qpik,0.569771,-0.604522,1.170219,1.576683,0.570129,1.168550,0.528446,0.528446,-0.130642,-0.130642,68.0000,68.0000,1.000000,0.9346,-0.063578,0.006129,,0,0,0.578547,1.566433,0.218820,,,0.571058,,,1,0,0,0.20465,0.000000000e+00
14.6885,servo_twist,qpik,0.563107,-0.594959,1.180000,1.585572,0.563436,1.177661,0.519240,0.519240,-0.132703,-0.132703,68.0000,68.0000,1.000000,0.8245,-0.066884,0.006419,,0,0,0.574627,1.550486,0.250017,,,0.564176,,,1,0,0,0.20359,0.000000000e+00
14.7882,servo_twist,qpik,0.556539,-0.584453,1.195807,1.590582,0.556876,1.190105,0.509963,0.509963,-0.134720,-0.134720,68.0000,68.0000,1.000000,0.6956,-0.075925,0.006796,,0,0,0.572498,1.534427,0.277420,,,0.557139,,,1,0,0,0.20252,0.000000000e+00
14.8919,servo_twist,qpik,0.549040,-0.574774,1.215079,1.592311,0.549432,1.208937,0.499092,0.499092,-0.136808,-0.136808,68.0000,68.0000,1.000000,0.7198,-0.079551,0.006537,,0,0,0.570288,1.515692,0.301778,,,0.549484,,,1,0,0,0.20128,0.000000000e+00
14.9943,servo_twist,qpik,0.541309,-0.567804,1.232995,1.592006,0.541746,1.227298,0.488371,0.488371,-0.138872,-0.138872,68.0000,68.0000,1.000000,0.7187,-0.079850,0.008699,,0,0,0.563786,1.496221,0.321441,,,0.543058,,,1,0,0,0.20012,0.000000000e+00
15.0986,servo_twist,qpik,0.533314,-0.561199,1.250808,1.590846,0.533781,1.245816,0.477388,0.477388,-0.140974,-0.140974,68.0000,68.0000,1.000000,0.7269,-0.079264,0.010045,,0,0,0.565358,1.476103,0.338444,,,0.534877,,,1,0,0,0.19898,0.000000000e+00
15.2012,servo_twist,qpik,0.525457,-0.554535,1.268130,1.589240,0.525844,1.263409,0.466635,0.466635,-0.143040,-0.143040,68.0000,68.0000,1.000000,0.7674,-0.078455,0.009724,,0,0,0.571702,1.456372,0.352867,,,0.527271,,,1,0,0,0.19795,0.000000000e+00
15.3159,servo_twist,qpik,0.517096,-0.546479,1.287490,1.587017,0.517610,1.284266,0.455028,0.455028,-0.145270,-0.145270,68.0000,68.0000,1.000000,0.6380,-0.077349,0.013082,,0,0,0.578059,1.435560,0.366283,,,0.517771,,,1,0,0,0.19679,0.000000000e+00
15.4175,servo_twist,qpik,0.509557,-0.537768,1.306153,1.584882,0.509944,1.303307,0.444361,0.444361,-0.147318,-0.147318,68.0000,68.0000,1.000000,0.7479,-0.075980,0.009400,,0,0,0.588285,1.416999,0.377704,,,0.510121,,,1,0,0,0.19552,0.000000000e+00
15.5229,servo_twist,qpik,0.501521,-0.529251,1.321979,1.583962,0.501910,1.318823,0.433776,0.433776,-0.149426,-0.149426,68.0000,68.0000,1.000000,0.7352,-0.075085,0.008260,,0,0,0.594429,1.397697,0.388096,,,0.502088,,,1,0,0,0.19439,0.000000000e+00
15.6311,servo_twist,qpik,0.493393,-0.519723,1.338318,1.583018,0.493758,1.335247,0.422760,0.422760,-0.151575,-0.151575,68.0000,68.0000,1.000000,0.7279,-0.074225,0.007361,,0,0,0.604391,1.378002,0.397891,,,0.494180,,,1,0,0,0.19320,0.000000000e+00
15.7357,servo_twist,qpik,0.485464,-0.509551,1.357378,1.580132,0.485760,1.352211,0.411786,0.411786,-0.153675,-0.153675,68.0000,68.0000,1.000000,0.8345,-0.081889,0.005040,,0,0,0.607105,1.358738,0.405959,,,0.486277,,,1,0,0,0.19203,0.000000000e+00
15.8420,servo_twist,qpik,0.476697,-0.496174,1.395243,1.563496,0.477160,1.384098,0.400454,0.400454,-0.155815,-0.155815,68.0000,68.0000,1.000000,0.5356,-0.099807,0.005457,,0,0,0.567667,1.335286,0.403420,,,0.477671,,,1,0,0,0.19141,0.000000000e+00
15.9464,servo_twist,qpik,0.467157,-0.484196,1.439581,1.539858,0.467643,1.427173,0.389500,0.389500,-0.157914,-0.157914,68.0000,68.0000,1.000000,0.4930,-0.098527,0.007388,,0,0,0.548352,1.308703,0.395351,,,0.467752,,,1,0,0,0.19169,0.000000000e+00
16.0420,servo_twist,qpik,0.457910,-0.476059,1.476141,1.516931,0.458406,1.462883,0.379914,0.379914,-0.159914,-0.159914,68.0000,68.0000,1.000000,0.4776,-0.095224,0.008933,,0,0,0.520228,1.282707,0.382859,,,0.459921,,,1,0,0,0.19215,0.000000000e+00
16.1430,servo_twist,qpik,0.448240,-0.471555,1.505581,1.494095,0.448747,1.494525,0.371212,0.371212,-0.161950,-0.161950,68.0000,68.0000,1.000000,0.3790,-0.079248,0.010829,,0,0,0.495469,1.255917,0.366225,,,0.449621,,,1,0,0,0.19263,0.000000000e+00
16.2507,servo_twist,qpik,0.439073,-0.472895,1.524828,1.471017,0.439543,1.518349,0.364583,0.364583,-0.164097,-0.164097,68.0000,68.0000,1.000000,0.3168,-0.061975,0.013264,,0,0,0.441399,1.230059,0.343318,,,0.440167,,,1,0,0,0.19389,0.000000000e+00
16.3496,servo_twist,qpik,0.432137,-0.474591,1.534376,1.454612,0.432487,1.531195,0.360231,0.360231,-0.166112,-0.166112,68.0000,68.0000,1.000000,0.6755,-0.046414,0.014478,,0,0,0.424017,1.210914,0.321507,,,0.432459,,,1,0,0,0.19494,0.000000000e+00
16.4564,servo_twist,qpik,0.426164,-0.473020,1.540472,1.445253,0.426467,1.535611,0.355759,0.355759,-0.168242,-0.168242,68.0000,68.0000,1.000000,0.7173,-0.056167,0.014161,,0,0,0.449607,1.195956,0.303863,,,0.426577,,,1,0,0,0.19511,0.000000000e+00
16.5551,servo_twist,qpik,0.420809,-0.461640,1.563486,1.434744,0.421110,1.553919,0.347731,0.347731,-0.169157,-0.169157,68.0000,68.0000,1.000000,0.5797,-0.072382,0.009586,,0,0,0.496296,1.183583,0.292631,,,0.422122,,,1,0,0,0.19350,0.000000000e+00
16.6559,servo_twist,qpik,0.414287,-0.450750,1.586494,1.424972,0.414653,1.577900,0.337142,0.337142,-0.169157,-0.169157,68.0000,68.0000,1.000000,0.6539,-0.073704,0.006200,,0,0,0.510271,1.168462,0.284614,,,0.415505,,,1,0,0,0.19133,0.000000000e+00
16.7536,servo_twist,qpik,0.407374,-0.442493,1.605690,1.416643,0.407738,1.598565,0.326798,0.326798,-0.169157,-0.169157,68.0000,68.0000,1.000000,0.7117,-0.073472,0.004686,,0,0,0.504016,1.152338,0.278915,,,0.408051,,,1,0,0,0.18945,0.000000000e+00
16.8530,servo_twist,qpik,0.400132,-0.435574,1.623253,1.409141,0.400505,1.616454,0.316888,0.316888,-0.169157,-0.169157,68.0000,68.0000,1.000000,0.7238,-0.073116,0.004408,,0,0,0.499568,1.135413,0.275062,,,0.400986,,,1,0,0,0.18780,0.000000000e+00
16.9581,servo_twist,qpik,0.392549,-0.428723,1.641500,1.401798,0.392893,1.636386,0.306188,0.306188,-0.169157,-0.169157,68.0000,68.0000,1.000000,0.7150,-0.071512,0.004196,,0,0,0.500266,1.117712,0.272832,,,0.393620,,,1,0,0,0.18606,0.000000000e+00
17.0654,servo_twist,qpik,0.384798,-0.421919,1.660187,1.394790,0.385211,1.654171,0.295709,0.295709,-0.169590,-0.169590,68.0000,68.0000,1.000000,0.7088,-0.069983,0.004459,,0,0,0.506593,1.099821,0.272259,,,0.385776,,,1,0,0,0.18448,0.000000000e+00
17.1647,servo_twist,qpik,0.377729,-0.415821,1.677642,1.388559,0.378040,1.673509,0.285355,0.285355,-0.170492,-0.170492,68.0000,68.0000,1.000000,0.7140,-0.068411,0.003051,,0,0,0.509195,1.083526,0.272999,,,0.378095,,,1,0,0,0.18278,0.000000000e+00
17.2696,servo_twist,qpik,0.370365,-0.409411,1.696137,1.382372,0.370705,1.691870,0.274982,0.274982,-0.170492,-0.170492,68.0000,68.0000,1.000000,0.6825,-0.066406,0.003229,,0,0,0.515678,1.066752,0.274928,,,0.370703,,,1,0,0,0.18111,0.000000000e+00
17.3770,servo_twist,qpik,0.363181,-0.401968,1.716703,1.376124,0.363578,1.712779,0.264320,0.264320,-0.170492,-0.170492,68.0000,68.0000,1.000000,0.6551,-0.063970,0.003034,,0,0,0.525544,1.050645,0.278410,,,0.363575,,,1,0,0,0.17916,0.000000000e+00
17.4858,servo_twist,qpik,0.356035,-0.394314,1.737918,1.370012,0.356409,1.734875,0.253427,0.253427,-0.170492,-0.170492,68.0000,68.0000,1.000000,0.6455,-0.061278,0.001502,,0,0,0.538054,1.034874,0.283026,,,0.356490,,,1,0,0,0.17710,0.000000000e+00
17.5962,servo_twist,qpik,0.349121,-0.386172,1.759986,1.364048,0.349459,1.757791,0.242387,0.242387,-0.169081,-0.169081,68.0000,68.0000,1.000000,0.6141,-0.057252,-0.000657,,0,0,0.549286,1.019904,0.288633,,,0.349790,,,1,0,0,0.17496,0.000000000e+00
17.7051,servo_twist,qpik,0.342677,-0.377180,1.783498,1.358116,0.343001,1.780463,0.231825,0.231825,-0.167683,-0.167683,68.0000,68.0000,1.000000,0.5775,-0.052966,-0.001712,,0,0,0.567331,1.006377,0.295431,,,0.343412,,,1,0,0,0.17279,0.000000000e+00
17.8156,servo_twist,qpik,0.336598,-0.367188,1.808775,1.352311,0.336903,1.804723,0.221159,0.221159,-0.166355,-0.166355,68.0000,68.0000,1.000000,0.5341,-0.048246,-0.002711,,0,0,0.586900,0.994541,0.303730,,,0.337354,,,1,0,0,0.17044,0.000000000e+00
17.9288,servo_twist,qpik,0.330881,-0.359918,1.836264,1.352397,0.331143,1.832159,0.209855,0.209855,-0.164935,-0.164935,68.0000,68.0000,1.000000,0.4879,-0.042488,-0.004270,,0,0,0.633626,0.984667,0.331796,,,0.331754,,,1,0,0,0.16799,0.000000000e+00
18.0370,servo_twist,qpik,0.325958,-0.356070,1.864874,1.358792,0.326215,1.858200,0.200000,0.200000,-0.163525,-0.163525,68.0000,68.0000,1.000000,0.4343,-0.037172,-0.004751,,0,0,0.720007,0.978552,0.381163,,,0.326735,,,1,0,0,0.16587,0.000000000e+00
18.1631,servo_twist,qpik,0.321018,-0.353815,1.898412,1.370287,0.321225,1.896632,0.187378,0.187378,-0.161978,-0.161978,68.0000,68.0000,1.000000,0.3707,-0.028998,-0.007296,,0,0,0.817815,0.977794,0.449985,,,0.321497,,,1,0,0,0.16303,0.000000000e+00
18.2698,servo_twist,qpik,0.317535,-0.352702,1.930913,1.383468,0.317715,1.924540,0.178061,0.178061,-0.160594,-0.160594,68.0000,68.0000,1.000000,0.3059,-0.022651,-0.007953,,0,0,0.948640,0.986754,0.520333,,,0.317883,,,1,0,0,0.16104,0.000000000e+00
18.3696,servo_twist,qpik,0.314947,-0.352012,1.963868,1.397795,0.315081,1.955345,0.168744,0.168744,-0.159277,-0.159277,68.0000,68.0000,1.000000,0.2441,-0.015557,-0.009432,,0,0,1.060430,1.007297,0.593611,,,0.315051,,,1,0,0,0.15890,0.000000000e+00
18.4715,servo_twist,qpik,0.312974,-0.351557,1.999444,1.413920,0.313071,1.986377,0.159765,0.159765,-0.157835,-0.157835,68.0000,68.0000,1.000000,0.1771,-0.009042,-0.010329,,0,0,1.200053,1.044279,0.673048,,,0.313286,,,1,0,0,0.15673,0.000000000e+00
18.5752,servo_twist,qpik,0.311600,-0.351279,2.037230,1.431396,0.311670,2.022523,0.150038,0.150038,-0.156269,-0.156269,68.0000,68.0000,1.000000,0.1196,-0.009889,-0.004831,,0,0,1.248994,1.102846,0.757543,,,0.311802,,,1,0,0,0.15418,0.000000000e+00
18.6769,servo_twist,qpik,0.310449,-0.351098,2.074783,1.448219,0.310516,2.059436,0.140355,0.140355,-0.154817,-0.154817,68.0000,68.0000,1.000000,0.1172,-0.020760,-0.010692,,0,0,1.303286,1.182317,0.840592,,,0.310644,,,1,0,0,0.15142,0.000000000e+00
18.7832,servo_twist,qpik,0.308644,-0.350939,2.111148,1.462885,0.308739,2.096315,0.129935,0.129935,-0.153215,-0.153215,68.0000,68.0000,1.000000,0.1913,-0.033595,-0.013800,,0,0,1.330386,1.278682,0.917534,,,0.308913,,,1,0,0,0.14847,0.000000000e+00
18.8876,servo_twist,qpik,0.305774,-0.350796,2.142529,1.473247,0.305923,2.131309,0.118550,0.118550,-0.151844,-0.151844,68.0000,68.0000,1.000000,0.3016,-0.045747,-0.015424,,0,0,1.347040,1.379149,0.981079,,,0.306161,,,1,0,0,0.14542,0.000000000e+00
18.9922,servo_twist,qpik,0.301673,-0.350663,2.169133,1.479543,0.301874,2.159758,0.107452,0.107452,-0.150494,-0.150494,68.0000,68.0000,1.000000,0.4159,-0.055070,-0.015464,,0,0,1.351008,1.474314,1.030011,,,0.302122,,,1,0,0,0.14270,0.000000000e+00
19.0924,servo_twist,qpik,0.296684,-0.350530,2.190900,1.482374,0.296964,2.182779,0.096868,0.096868,-0.149378,-0.149378,68.0000,68.0000,1.000000,0.5097,-0.061773,-0.014677,,0,0,1.354085,1.558770,1.065322,,,0.296962,,,1,0,0,0.14033,0.000000000e+00
19.1926,servo_twist,qpik,0.290986,-0.350417,2.209822,1.482872,0.291317,2.202431,0.086328,0.086328,-0.148237,-0.148237,68.0000,68.0000,1.000000,0.5700,-0.067026,-0.013831,,0,0,1.356586,1.636891,1.091899,,,0.292138,,,1,0,0,0.13817,0.000000000e+00
19.2927,servo_twist,qpik,0.284651,-0.350307,2.226428,1.481586,0.285032,2.219727,0.076144,0.076144,-0.147116,-0.147116,68.0000,68.0000,1.000000,0.6284,-0.070998,-0.012390,,0,0,1.359089,1.709020,1.111497,,,0.285556,,,1,0,0,0.13619,0.000000000e+00
19.4006,servo_twist,qpik,0.277392,-0.350195,2.242280,1.478766,0.277716,2.236919,0.064911,0.064911,-0.145921,-0.145921,68.0000,68.0000,1.000000,0.6822,-0.074566,-0.011582,,0,0,1.361622,1.780165,1.126635,,,0.278360,,,1,0,0,0.13414,0.000000000e+00
19.5009,servo_twist,qpik,0.270543,-0.350391,2.256723,1.475980,0.270900,2.251614,0.054781,0.054781,-0.144903,-0.144903,68.0000,68.0000,1.000000,0.7174,-0.077344,-0.010880,,0,0,1.365519,1.850364,1.140313,,,0.271053,,,1,0,0,0.13234,0.000000000e+00
19.6049,servo_twist,qpik,0.262655,-0.350983,2.264278,1.473553,0.263070,2.262540,0.044387,0.044387,-0.144584,-0.144584,68.0000,68.0000,1.000000,0.8250,-0.078944,-0.010691,,0,0,1.365044,1.885913,1.146088,,,0.263118,,,1,0,0,0.13094,0.000000000e+00
19.7149,servo_twist,qpik,0.254111,-0.351252,2.267170,1.471523,0.254520,2.266816,0.034092,0.034092,-0.144584,-0.144584,68.0000,68.0000,1.000000,0.8826,-0.079736,-0.012049,,0,0,1.365357,1.894957,1.146866,,,0.254730,,,1,0,0,0.13038,0.000000000e+00
19.8200,servo_twist,qpik,0.245009,-0.351944,2.267968,1.468315,0.245425,2.268247,0.024095,0.024095,-0.144684,-0.144684,68.0000,68.0000,1.000000,0.9483,-0.080042,-0.013488,,0,0,1.356250,1.891467,1.141522,,,0.245499,,,1,0,0,0.13016,0.000000000e+00
19.9352,servo_twist,qpik,0.236085,-0.351910,2.268142,1.466061,0.236520,2.268701,0.014322,0.014322,-0.144684,-0.144684,68.0000,68.0000,1.000000,0.8515,-0.080326,-0.005313,,0,0,1.352041,1.884891,1.137494,,,0.237220,,,1,0,0,0.13003,0.000000000e+00
20.0422,servo_twist,qpik,0.228083,-0.351387,2.268182,1.465223,0.228458,2.268701,0.004884,0.004884,-0.144684,-0.144684,68.0000,68.0000,1.000000,0.8811,-0.080759,-0.005756,,0,0,1.358021,1.878833,1.137807,,,0.229557,,,1,0,0,0.13002,0.000000000e+00
20.1436,servo_twist,qpik,0.220040,-0.350840,2.268208,1.462989,0.220454,2.268684,-0.004028,-0.004028,-0.144684,-0.144684,68.0000,68.0000,1.000000,0.9012,-0.081032,-0.007412,,0,0,1.352462,1.872691,1.134239,,,0.221138,,,1,0,0,0.12999,0.000000000e+00
20.2440,servo_twist,qpik,0.211974,-0.350424,2.268236,1.460515,0.212360,2.268701,-0.012886,-0.012886,-0.144684,-0.144684,68.0000,68.0000,1.000000,0.9114,-0.081252,-0.008381,,0,0,1.346213,1.866716,1.129683,,,0.212858,,,1,0,0,0.12994,0.000000000e+00
20.3622,servo_twist,qpik,0.202451,-0.350260,2.268305,1.457921,0.202969,2.268631,-0.023165,-0.023165,-0.144684,-0.144684,68.0000,68.0000,1.000000,0.8668,-0.081487,-0.007187,,0,0,1.344242,1.860152,1.125007,,,0.204192,,,1,0,0,0.12987,0.000000000e+00
20.4663,servo_twist,qpik,0.194084,-0.350433,2.268461,1.456106,0.194527,2.268527,-0.032578,-0.032578,-0.144684,-0.144684,68.0000,68.0000,1.000000,0.8923,-0.081757,-0.007311,,0,0,1.340831,1.855181,1.122713,,,0.195737,,,1,0,0,0.12985,0.000000000e+00
20.5664,servo_twist,qpik,0.186051,-0.350317,2.268648,1.454103,0.186485,2.268492,-0.041437,-0.041437,-0.144684,-0.144684,68.0000,68.0000,1.000000,0.9008,-0.081984,-0.008011,,0,0,1.340533,1.850943,1.119979,,,0.186590,,,1,0,0,0.12982,0.000000000e+00
20.6655,servo_twist,qpik,0.177814,-0.350193,2.268828,1.451342,0.178210,2.268492,-0.050589,-0.050589,-0.145728,-0.145728,68.0000,68.0000,1.000000,0.9268,-0.081834,-0.008452,,0,0,1.334591,1.846635,1.115119,,,0.179555,,,1,0,0,0.12976,0.000000000e+00
20.7659,servo_twist,qpik,0.169355,-0.350118,2.269015,1.448230,0.169849,2.268492,-0.059640,-0.059640,-0.145950,-0.145950,68.0000,68.0000,1.000000,0.9112,-0.081937,-0.008535,,0,0,1.331172,1.842439,1.109147,,,0.170585,,,1,0,0,0.12970,0.000000000e+00
20.8660,servo_twist,qpik,0.161375,-0.351427,2.269100,1.447480,0.161813,2.268754,-0.068643,-0.068643,-0.145950,-0.145950,68.0000,68.0000,1.000000,0.9014,-0.082204,-0.009278,,0,0,1.330759,1.838411,1.109324,,,0.162534,,,1,0,0,0.12964,0.000000000e+00
20.9650,servo_twist,qpik,0.153229,-0.351886,2.269100,1.445399,0.153670,2.269015,-0.077660,-0.077660,-0.145950,-0.145950,68.0000,68.0000,1.000000,0.9035,-0.082302,-0.009180,,0,0,1.328787,1.833894,1.106474,,,0.154358,,,1,0,0,0.12958,0.000000000e+00
21.0678,servo_twist,qpik,0.144658,-0.351370,2.269100,1.442269,0.145068,2.269085,-0.087115,-0.087115,-0.145950,-0.145950,68.0000,68.0000,1.000000,0.9289,-0.082213,-0.010477,,0,0,1.325295,1.829420,1.100689,,,0.145235,,,1,0,0,0.12951,0.000000000e+00
21.1687,servo_twist,qpik,0.136170,-0.351005,2.269100,1.438820,0.136600,2.269155,-0.096287,-0.096287,-0.145950,-0.145950,68.0000,68.0000,1.000000,0.9058,-0.082134,-0.010284,,0,0,1.320005,1.825114,1.093606,,,0.136658,,,1,0,0,0.12943,0.000000000e+00
21.2723,servo_twist,qpik,0.127961,-0.351139,2.269099,1.437026,0.128323,2.269155,-0.105445,-0.105445,-0.145950,-0.145950,68.0000,68.0000,1.000000,0.8774,-0.082042,-0.007577,,0,0,1.322679,1.821430,1.091044,,,0.128830,,,1,0,0,0.12936,0.000000000e+00
21.3830,servo_twist,qpik,0.119070,-0.352042,2.269099,1.435655,0.119547,2.269102,-0.115251,-0.115251,-0.145950,-0.145950,68.0000,68.0000,1.000000,0.8446,-0.081946,-0.007814,,0,0,1.321735,1.817712,1.090020,,,0.120708,,,1,0,0,0.12935,0.000000000e+00
21.4880,servo_twist,qpik,0.109619,-0.361640,2.269099,1.440761,0.110005,2.269155,-0.126033,-0.126033,-0.147004,-0.147004,68.0000,68.0000,1.000000,0.9953,-0.081859,-0.012441,,0,0,1.322835,1.813689,1.105774,,,0.111682,,,1,0,0,0.12942,0.000000000e+00
21.5863,servo_twist,qpik,0.101430,-0.369439,2.269099,1.444122,0.101846,2.269068,-0.134927,-0.134927,-0.149055,-0.149055,68.0000,68.0000,1.000000,0.9019,-0.081835,-0.009001,,0,0,1.322052,1.810410,1.118294,,,0.102814,,,1,0,0,0.12959,0.000000000e+00
21.6864,servo_twist,qpik,0.093187,-0.373837,2.269099,1.445662,0.093635,2.269137,-0.143655,-0.143655,-0.151098,-0.151098,68.0000,68.0000,1.000000,0.8919,-0.081309,-0.006843,,0,0,1.325670,1.805765,1.125877,,,0.094957,,,1,0,0,0.12966,0.000000000e+00
21.7863,servo_twist,qpik,0.085239,-0.376314,2.269098,1.445604,0.085573,2.269085,-0.152379,-0.152379,-0.153148,-0.153148,68.0000,68.0000,1.000000,0.8624,-0.079748,-0.006409,,0,0,1.323308,1.799251,1.128898,,,0.086462,,,1,0,0,0.12971,0.000000000e+00
21.8864,servo_twist,qpik,0.077106,-0.377480,2.269098,1.444490,0.077538,2.269068,-0.161465,-0.161465,-0.155180,-0.155180,68.0000,68.0000,1.000000,0.8733,-0.076929,-0.006132,,0,0,1.324716,1.793507,1.129269,,,0.077875,,,1,0,0,0.12972,0.000000000e+00
21.9863,servo_twist,qpik,0.069334,-0.376914,2.269098,1.442032,0.069686,2.269068,-0.170503,-0.170503,-0.157210,-0.157210,68.0000,68.0000,1.000000,0.8538,-0.073476,-0.004924,,0,0,1.323394,1.792860,1.125957,,,0.070960,,,1,0,0,0.12969,0.000000000e+00
22.0863,servo_twist,qpik,0.062320,-0.374352,2.269098,1.439841,0.062686,2.269068,-0.179315,-0.179315,-0.159240,-0.159240,68.0000,68.0000,1.000000,0.8120,-0.070125,-0.005134,,0,0,1.330332,1.800823,1.124084,,,0.063532,,,1,0,0,0.12965,0.000000000e+00
22.1869,servo_twist,qpik,0.055024,-0.370334,2.269098,1.435955,0.055438,2.269102,-0.188338,-0.188338,-0.161271,-0.161271,68.0000,68.0000,1.000000,0.7953,-0.066261,-0.006409,,0,0,1.328729,1.821914,1.117904,,,0.055751,,,1,0,0,0.12956,0.000000000e+00
22.2872,servo_twist,qpik,0.048016,-0.362796,2.269097,1.429214,0.048389,2.269102,-0.197122,-0.197122,-0.163310,-0.163310,68.0000,68.0000,1.000000,0.7895,-0.062274,-0.007472,,0,0,1.329965,1.859598,1.103942,,,0.048276,,,1,0,0,0.12938,0.000000000e+00
22.3863,servo_twist,qpik,0.041685,-0.357119,2.269097,1.421673,0.042019,2.269085,-0.205297,-0.205297,-0.165322,-0.165322,68.0000,68.0000,1.000000,0.8028,-0.058654,-0.007538,,0,0,1.322205,1.911481,1.087190,,,0.042890,,,1,0,0,0.12921,0.000000000e+00
22.4875,servo_twist,qpik,0.035642,-0.353814,2.269097,1.413613,0.036072,2.269068,-0.212845,-0.212845,-0.167407,-0.167407,68.0000,68.0000,1.000000,0.7764,-0.055077,-0.004519,,0,0,1.313142,1.979993,1.069124,,,0.036325,,,1,0,0,0.12901,0.000000000e+00
22.5867,servo_twist,qpik,0.030412,-0.352044,2.269097,1.404907,0.030688,2.269102,-0.219650,-0.219650,-0.169468,-0.169468,68.0000,68.0000,1.000000,0.7932,-0.051906,-0.003817,,0,0,1.300678,2.056130,1.049242,,,0.030677,,,1,0,0,0.12881,0.000000000e+00
22.6863,servo_twist,qpik,0.025145,-0.351116,2.269097,1.396444,0.025410,2.269085,-0.226074,-0.226074,-0.171269,-0.171269,68.0000,68.0000,1.000000,0.7973,-0.048841,0.000000,,1,0,1.290809,2.149478,1.030368,,,0.026183,,,1,0,0,0.12862,0.000000000e+00
22.7868,servo_twist,qpik,0.020118,-0.350564,2.269096,1.387698,0.020373,2.269137,-0.232229,-0.232229,-0.171269,-0.171269,68.0000,68.0000,1.000000,0.7842,-0.046950,0.000000,,1,0,1.282465,2.254714,1.010857,,,0.020622,,,1,0,0,0.12842,0.000000000e+00
22.8874,servo_twist,qpik,0.015255,-0.350303,2.269096,1.378561,0.015511,2.269102,-0.238274,-0.238274,-0.172297,-0.172297,68.0000,68.0000,1.000000,0.7739,-0.044698,0.000000,,1,0,1.270563,2.372186,0.990533,,,0.015515,,,1,0,0,0.12822,0.000000000e+00
22.9863,servo_twist,qpik,0.011003,-0.350192,2.269096,1.369045,0.011172,2.269085,-0.243915,-0.243915,-0.172297,-0.172297,68.0000,68.0000,1.000000,0.7364,-0.043347,0.000000,,1,0,1.255190,2.490637,0.969294,,,0.011800,,,1,0,0,0.12801,0.000000000e+00
23.0867,servo_twist,qpik,0.008287,-0.350154,2.269096,1.357909,0.008425,2.269155,-0.248227,-0.248227,-0.172297,-0.172297,68.0000,68.0000,1.000000,0.5409,-0.042837,0.001922,,1,0,1.224840,2.572724,0.944879,,,0.008546,,,1,0,0,0.12776,0.000000000e+00
23.1863,servo_twist,qpik,0.006865,-0.350108,2.269096,1.345113,0.006960,2.269068,-0.251655,-0.251655,-0.172297,-0.172297,68.0000,68.0000,1.000000,0.3652,-0.042399,0.003165,,1,0,1.170843,2.617741,0.916665,,,0.006983,,,1,0,0,0.12752,0.000000000e+00
23.2866,servo_twist,qpik,0.006110,-0.350064,2.269095,1.331282,0.006162,2.269137,-0.254498,-0.254498,-0.172297,-0.172297,68.0000,68.0000,1.000000,0.2391,-0.041926,0.001276,,1,0,1.097280,2.642293,0.886259,,,0.006266,,,1,0,0,0.12722,0.000000000e+00
23.3863,servo_twist,qpik,0.005685,-0.350026,2.269095,1.316869,0.005713,2.269102,-0.256975,-0.256975,-0.172297,-0.172297,68.0000,68.0000,1.000000,0.1533,-0.041597,0.000000,,1,0,1.029489,2.656395,0.854790,,,0.005737,,,1,0,0,0.12690,0.000000000e+00
23.4863,servo_twist,qpik,0.005404,-0.350017,2.269095,1.302260,0.005420,2.269085,-0.259380,-0.259380,-0.172297,-0.172297,68.0000,68.0000,1.000000,0.1084,-0.037140,0.000000,,1,0,0.970697,2.665863,0.822951,,,0.005465,,,1,0,0,0.12659,0.000000000e+00
23.5863,servo_twist,qpik,0.005264,-0.350016,2.269094,1.291621,0.005274,2.269137,-0.261212,-0.261212,-0.172297,-0.172297,68.0000,68.0000,1.000000,0.1774,-0.009226,0.000000,,1,0,1.001087,2.670675,0.799044,,,0.005309,,,1,0,0,0.12632,0.000000000e+00
23.6867,servo_twist,qpik,0.005186,-0.350648,2.267715,1.291652,0.005173,2.268963,-0.261757,-0.261757,-0.172297,-0.172297,68.0000,68.0000,1.000000,0.0550,0.016715,0.000000,,1,0,1.123390,2.669205,0.795653,,,0.005184,,,1,0,0,0.12625,0.000000000e+00
23.7863,servo_twist,qpik,0.005916,-0.360548,2.251678,1.305914,0.005817,2.258474,-0.257857,-0.257857,-0.172297,-0.172297,68.0000,68.0000,1.000000,0.1224,0.040057,0.000000,,1,0,1.129501,2.589557,0.799350,,,0.005176,,,1,0,0,0.12757,0.000000000e+00
23.8863,servo_twist,qpik,0.007915,-0.375433,2.226365,1.323832,0.007760,2.234493,-0.247599,-0.247599,-0.172497,-0.172497,68.0000,68.0000,1.000000,0.2759,0.039895,0.000597,,1,0,1.116297,2.433093,0.806396,,,0.006663,,,1,0,0,0.13049,0.000000000e+00
23.9863,servo_twist,qpik,0.011090,-0.385947,2.199035,1.339586,0.010910,2.206847,-0.236114,-0.236114,-0.174539,-0.174539,68.0000,68.0000,1.000000,0.3289,0.034951,0.004822,,1,0,1.131777,2.238494,0.816127,,,0.010245,,,1,0,0,0.13381,0.000000000e+00
24.0868,servo_twist,qpik,0.014677,-0.394265,2.170563,1.353464,0.014450,2.178939,-0.224824,-0.224824,-0.176580,-0.176580,68.0000,68.0000,1.000000,0.3139,0.030167,0.006734,,1,0,1.135113,2.037168,0.823287,,,0.014102,,,1,0,0,0.13703,0.000000000e+00
24.1863,servo_twist,qpik,0.017819,-0.401503,2.141649,1.365605,0.017666,2.149216,-0.213530,-0.213530,-0.178580,-0.178580,68.0000,68.0000,1.000000,0.2832,0.024845,0.008625,,1,0,1.130259,1.859163,0.825971,,,0.017696,,,1,0,0,0.14026,0.000000000e+00
24.2863,servo_twist,qpik,0.020636,-0.408200,2.111561,1.376403,0.020504,2.121727,-0.203055,-0.203055,-0.180581,-0.180581,68.0000,68.0000,1.000000,0.2427,0.019637,0.009466,,1,0,1.123376,1.698801,0.823242,,,0.020215,,,1,0,0,0.14310,0.000000000e+00
24.3863,servo_twist,qpik,0.022950,-0.414719,2.079623,1.386004,0.022840,2.089159,-0.191776,-0.191776,-0.182629,-0.182629,68.0000,68.0000,1.000000,0.1975,0.013755,0.011285,,1,0,1.097730,1.560499,0.814062,,,0.022738,,,1,0,0,0.14627,0.000000000e+00
24.4863,servo_twist,qpik,0.024632,-0.421172,2.046788,1.394173,0.024539,2.055579,-0.180687,-0.180687,-0.184651,-0.184651,68.0000,68.0000,1.000000,0.1467,0.008248,0.013397,,1,0,1.070718,1.450903,0.798483,,,0.024278,,,1,0,0,0.14934,0.000000000e+00
24.5865,servo_twist,qpik,0.025757,-0.427953,2.012249,1.401093,0.025700,2.021650,-0.170136,-0.170136,-0.186694,-0.186694,68.0000,68.0000,1.000000,0.0986,0.003870,0.014919,,1,0,1.000951,1.367311,0.776199,,,0.025642,,,1,0,0,0.15229,0.000000000e+00
24.6868,servo_twist,qpik,0.026425,-0.435136,1.976150,1.406755,0.026390,1.987476,-0.160210,-0.160210,-0.186697,-0.186697,68.0000,68.0000,1.000000,0.0572,0.002598,0.009008,,1,0,0.898902,1.307907,0.747484,,,0.026405,,,1,0,0,0.15518,0.000000000e+00
24.7871,servo_twist,qpik,0.026813,-0.442560,1.938879,1.411189,0.026791,1.949219,-0.149541,-0.149541,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.0354,0.003476,0.007778,,1,0,0.801992,1.267937,0.713532,,,0.026760,,,1,0,0,0.15835,0.000000000e+00
24.8863,servo_twist,qpik,0.027175,-0.449985,1.900408,1.414525,0.027155,1.912287,-0.139729,-0.139729,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.0372,0.006395,0.011824,,1,0,0.716136,1.238284,0.676210,,,0.027113,,,1,0,0,0.16142,0.000000000e+00
24.9863,servo_twist,qpik,0.027716,-0.457446,1.860413,1.417080,0.027691,1.872668,-0.129371,-0.129371,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.0614,0.010129,0.014158,,1,0,0.654090,1.210338,0.635757,,,0.027664,,,1,0,0,0.16465,0.000000000e+00
25.0868,servo_twist,qpik,0.028579,-0.464703,1.821069,1.418968,0.028539,1.833189,-0.119045,-0.119045,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.0954,0.014430,0.015330,,1,0,0.605948,1.177929,0.593311,,,0.028352,,,1,0,0,0.16791,0.000000000e+00
25.1903,servo_twist,qpik,0.029879,-0.471832,1.781466,1.420536,0.029812,1.793500,-0.108495,-0.108495,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.1368,0.019423,0.016266,,1,0,0.572070,1.135665,0.550209,,,0.029591,,,1,0,0,0.17116,0.000000000e+00
25.2952,servo_twist,qpik,0.031665,-0.478900,1.742126,1.422086,0.031575,1.752764,-0.097640,-0.097640,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.1873,0.025207,0.016743,,0,0,0.543696,1.081512,0.507612,,,0.031270,,,1,0,0,0.17448,0.000000000e+00
25.4051,servo_twist,qpik,0.034104,-0.485818,1.703407,1.423872,0.034010,1.711610,-0.086203,-0.086203,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.2463,0.031678,0.017559,,0,0,0.522578,1.012249,0.466430,,,0.033554,,,1,0,0,0.17783,0.000000000e+00
25.5176,servo_twist,qpik,0.037353,-0.492522,1.665657,1.426101,0.037195,1.669181,-0.074312,-0.074312,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.3122,0.038750,0.017523,,0,0,0.507348,0.923881,0.426262,,,0.036601,,,1,0,0,0.18126,0.000000000e+00
25.6161,servo_twist,qpik,0.040853,-0.498291,1.632501,1.428669,0.040678,1.637189,-0.064766,-0.064766,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.3782,0.044457,0.016602,,0,0,0.497878,0.837285,0.392617,,,0.040263,,,1,0,0,0.18381,0.000000000e+00
25.7161,servo_twist,qpik,0.044994,-0.503675,1.600878,1.431858,0.044779,1.606645,-0.054961,-0.054961,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.4377,0.050182,0.016256,,0,0,0.492099,0.745582,0.361661,,,0.044512,,,1,0,0,0.18623,0.000000000e+00
25.8214,servo_twist,qpik,0.049919,-0.508801,1.570320,1.435820,0.049651,1.576678,-0.044646,-0.044646,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.4928,0.053539,0.016198,,0,0,0.489145,0.653117,0.333581,,,0.049609,,,1,0,0,0.18852,0.000000000e+00
25.9214,servo_twist,qpik,0.055082,-0.513485,1.542753,1.440321,0.054809,1.550027,-0.034844,-0.034844,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.5375,0.057008,0.015711,,0,0,0.488593,0.576111,0.310006,,,0.054012,,,1,0,0,0.19057,0.000000000e+00
26.0214,servo_twist,qpik,0.060558,-0.518109,1.516577,1.445556,0.060269,1.522939,-0.024722,-0.024722,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.5750,0.060605,0.015887,,0,0,0.489543,0.519628,0.290120,,,0.059754,,,1,0,0,0.19261,0.000000000e+00
26.1214,servo_twist,qpik,0.066427,-0.522561,1.492145,1.451368,0.066110,1.499622,-0.015174,-0.015174,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.6108,0.064187,0.015518,,0,0,0.491795,0.486775,0.274022,,,0.065891,,,1,0,0,0.19435,0.000000000e+00
26.2215,servo_twist,qpik,0.072637,-0.526882,1.468733,1.457862,0.072310,1.476775,-0.005416,-0.005416,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.6474,0.067692,0.015296,,0,0,0.494643,0.478471,0.260975,,,0.071153,,,1,0,0,0.19602,0.000000000e+00
26.3221,servo_twist,qpik,0.079240,-0.530826,1.447179,1.464724,0.078883,1.453493,0.004660,0.004660,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.6834,0.070965,0.015437,,0,0,0.497849,0.489603,0.251112,,,0.078099,,,1,0,0,0.19770,0.000000000e+00
26.4244,servo_twist,qpik,0.086337,-0.534523,1.426396,1.472232,0.085967,1.433526,0.014633,0.014633,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7189,0.073771,0.015056,,0,0,0.501429,0.512196,0.243968,,,0.085301,,,1,0,0,0.19909,0.000000000e+00
26.5227,servo_twist,qpik,0.093440,-0.537780,1.407471,1.479888,0.093031,1.412809,0.024584,0.024584,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7474,0.075821,0.015078,,0,0,0.505304,0.535583,0.239446,,,0.092997,,,1,0,0,0.20053,0.000000000e+00
26.6244,servo_twist,qpik,0.101002,-0.540822,1.389320,1.487992,0.100596,1.395408,0.034452,0.034452,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7712,0.076952,0.014681,,0,0,0.509153,0.553865,0.237229,,,0.099312,,,1,0,0,0.20171,0.000000000e+00
26.7267,servo_twist,qpik,0.108726,-0.543829,1.371790,1.496384,0.108315,1.376594,0.044579,0.044579,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7768,0.077487,0.014549,,0,0,0.514021,0.566321,0.236777,,,0.107317,,,1,0,0,0.20290,0.000000000e+00
26.8284,servo_twist,qpik,0.116521,-0.546946,1.354582,1.505023,0.116116,1.358878,0.054454,0.054454,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7948,0.077974,0.014638,,0,0,0.518439,0.579048,0.237803,,,0.115632,,,1,0,0,0.20403,0.000000000e+00
26.9288,servo_twist,qpik,0.124312,-0.549970,1.337823,1.513864,0.123916,1.341739,0.064284,0.064284,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7996,0.078445,0.014867,,0,0,0.524387,0.592101,0.240070,,,0.123561,,,1,0,0,0.20505,0.000000000e+00
27.0340,servo_twist,qpik,0.132534,-0.553183,1.320777,1.523053,0.132135,1.324129,0.074581,0.074581,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8060,0.078957,0.014909,,0,0,0.529207,0.606591,0.243249,,,0.131303,,,1,0,0,0.20615,0.000000000e+00
27.1396,servo_twist,qpik,0.140929,-0.556481,1.303572,1.532502,0.140481,1.306396,0.084963,0.084963,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8171,0.079484,0.015336,,0,0,0.536020,0.621701,0.247075,,,0.139494,,,1,0,0,0.20722,0.000000000e+00
27.2472,servo_twist,qpik,0.149425,-0.559787,1.286289,1.542128,0.149012,1.288664,0.095373,0.095373,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8164,0.080005,0.015195,,0,0,0.541364,0.637458,0.251333,,,0.147590,,,1,0,0,0.20827,0.000000000e+00
27.3497,servo_twist,qpik,0.157439,-0.563465,1.268886,1.551536,0.157061,1.272555,0.105235,0.105235,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8199,0.080500,0.015261,,0,0,0.549011,0.652705,0.255509,,,0.155873,,,1,0,0,0.20923,0.000000000e+00
27.4535,servo_twist,qpik,0.166097,-0.566002,1.253073,1.560527,0.165565,1.255869,0.115595,0.115595,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8487,0.080999,0.016661,,0,0,0.554416,0.669633,0.259320,,,0.164762,,,1,0,0,0.21014,0.000000000e+00
27.5558,servo_twist,qpik,0.174509,-0.568082,1.238052,1.569137,0.174045,1.241383,0.125357,0.125357,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8374,0.081423,0.016570,,0,0,0.557311,0.686528,0.262424,,,0.173305,,,1,0,0,0.21091,0.000000000e+00
27.6599,servo_twist,qpik,0.183020,-0.570800,1.221827,1.578101,0.182570,1.225675,0.135465,0.135465,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8379,0.081912,0.015931,,0,0,0.563313,0.703920,0.265640,,,0.181475,,,1,0,0,0.21167,0.000000000e+00
27.7604,servo_twist,qpik,0.191271,-0.573460,1.205923,1.586902,0.190935,1.209269,0.145472,0.145472,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8813,0.082430,0.016103,,0,0,0.571792,0.721246,0.268803,,,0.190178,,,1,0,0,0.21248,0.000000000e+00
27.8702,servo_twist,qpik,0.200012,-0.577374,1.186971,1.596668,0.199538,1.190960,0.155906,0.155906,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8281,0.083005,0.014636,,0,0,0.579831,0.739860,0.271900,,,0.198902,,,1,0,0,0.21336,0.000000000e+00
27.9725,servo_twist,qpik,0.208426,-0.580395,1.170197,1.605640,0.207995,1.173629,0.165931,0.165931,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8528,0.083532,0.015360,,0,0,0.585451,0.758063,0.274694,,,0.206881,,,1,0,0,0.21422,0.000000000e+00
28.0737,servo_twist,qpik,0.216849,-0.583148,1.153876,1.614472,0.216367,1.157851,0.175542,0.175542,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8432,0.084031,0.015336,,0,0,0.589869,0.776536,0.277398,,,0.215459,,,1,0,0,0.21491,0.000000000e+00
28.1799,servo_twist,qpik,0.225397,-0.586692,1.135682,1.623815,0.224936,1.140416,0.185838,0.185838,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7972,0.084606,0.014006,,0,0,0.597313,0.795645,0.279858,,,0.224176,,,1,0,0,0.21563,0.000000000e+00
28.2834,servo_twist,qpik,0.233790,-0.590586,1.117002,1.633077,0.233339,1.122107,0.195861,0.195861,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8026,0.085188,0.012901,,0,0,0.606118,0.814804,0.281830,,,0.232679,,,1,0,0,0.21643,0.000000000e+00
28.3832,servo_twist,qpik,0.242201,-0.593622,1.099988,1.641866,0.241782,1.103764,0.205962,0.205962,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8723,0.085784,0.015038,,0,0,0.612990,0.833999,0.283575,,,0.241294,,,1,0,0,0.21718,0.000000000e+00
28.4832,servo_twist,qpik,0.250676,-0.596375,1.083475,1.650493,0.250240,1.088352,0.215532,0.215532,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8779,0.086278,0.015163,,0,0,0.614745,0.853553,0.285245,,,0.249375,,,1,0,0,0.21776,0.000000000e+00
28.5866,servo_twist,qpik,0.259644,-0.598559,1.067636,1.659104,0.259213,1.072819,0.225644,0.225644,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8995,0.086797,0.015575,,0,0,0.620056,0.874202,0.286918,,,0.258374,,,1,0,0,0.21827,0.000000000e+00
28.6904,servo_twist,qpik,0.268644,-0.601096,1.051315,1.667694,0.268140,1.055697,0.235787,0.235787,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8793,0.086778,0.016624,,0,0,0.627590,0.895122,0.288297,,,0.266806,,,1,0,0,0.21882,0.000000000e+00
28.7913,servo_twist,qpik,0.277403,-0.604749,1.038555,1.672812,0.276909,1.042171,0.245595,0.245595,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8365,0.082659,0.016460,,0,0,0.616073,0.914780,0.287904,,,0.275763,,,1,0,0,0.21933,0.000000000e+00
28.8990,servo_twist,qpik,0.286409,-0.611884,1.032620,1.670207,0.285862,1.032624,0.256136,0.256136,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.9233,0.070271,0.016096,,0,0,0.570464,0.932733,0.283355,,,0.284652,,,1,0,0,0.22027,0.000000000e+00
29.0014,servo_twist,qpik,0.295034,-0.619353,1.036123,1.660961,0.294555,1.032188,0.266411,0.266411,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8098,0.061243,0.023141,,0,0,0.516489,0.948049,0.276176,,,0.293131,,,1,0,0,0.22112,0.000000000e+00
29.1071,servo_twist,qpik,0.303609,-0.626545,1.046087,1.647488,0.303167,1.041979,0.276242,0.276242,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7597,0.058289,0.026282,,0,0,0.483260,0.962337,0.267560,,,0.302005,,,1,0,0,0.22176,0.000000000e+00
29.2071,servo_twist,qpik,0.309830,-0.638954,1.047759,1.634907,0.309491,1.047843,0.284346,0.284346,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8309,0.057363,0.021676,,0,0,0.470676,0.972254,0.257202,,,0.308993,,,1,0,0,0.22275,0.000000000e+00
29.3091,servo_twist,qpik,0.315775,-0.655388,1.042232,1.623799,0.315468,1.044126,0.293433,0.293433,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.8037,0.055764,0.020578,,0,0,0.480016,0.982485,0.245153,,,0.315183,,,1,0,0,0.22442,0.000000000e+00
29.4128,servo_twist,qpik,0.321631,-0.672254,1.037454,1.611762,0.321387,1.038837,0.302699,0.302699,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.6803,0.052226,0.020969,,0,0,0.471767,0.992506,0.232048,,,0.321349,,,1,0,0,0.22614,0.000000000e+00
29.5147,servo_twist,qpik,0.327170,-0.688206,1.036871,1.597761,0.326900,1.036586,0.311368,0.311368,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.5118,0.045610,0.020997,,0,0,0.451133,1.001389,0.218470,,,0.326025,,,1,0,0,0.22765,0.000000000e+00
29.6195,servo_twist,qpik,0.332292,-0.706112,1.035696,1.582618,0.332031,1.036429,0.319439,0.319439,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.4586,0.044364,0.020637,,0,0,0.434995,1.009394,0.203375,,,0.331289,,,1,0,0,0.22928,0.000000000e+00
29.7214,servo_twist,qpik,0.336962,-0.725392,1.032025,1.568197,0.336707,1.033183,0.327484,0.327484,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.4355,0.044563,0.021186,,0,0,0.433119,1.016731,0.188385,,,0.336248,,,1,0,0,0.23095,0.000000000e+00
29.8216,servo_twist,qpik,0.341447,-0.744947,1.027351,1.554468,0.341220,1.028854,0.335109,0.335109,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.4091,0.044360,0.021420,,0,0,0.431851,1.023965,0.173874,,,0.340727,,,1,0,0,0.23265,0.000000000e+00
29.9317,servo_twist,qpik,0.346348,-0.766154,1.023018,1.539397,0.346101,1.023566,0.343514,0.343514,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.3446,0.041715,0.021921,,0,0,0.426120,1.031945,0.158569,,,0.345773,,,1,0,0,0.23480,0.000000000e+00
30.0355,servo_twist,qpik,0.350754,-0.786133,1.021910,1.523556,0.350548,1.021436,0.350850,0.350850,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.2555,0.036620,0.021966,,0,0,0.409365,1.038695,0.143757,,,0.350346,,,1,0,0,0.23634,0.000000000e+00
30.1377,servo_twist,qpik,0.354745,-0.806751,1.021254,1.507298,0.354549,1.021332,0.357232,0.357232,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.2149,0.035304,0.021493,,0,0,0.397794,1.044535,0.128879,,,0.354588,,,1,0,0,0.23773,0.000000000e+00
30.2394,servo_twist,qpik,0.358483,-0.828805,1.018193,1.491902,0.358280,1.018958,0.363489,0.363489,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.1949,0.035334,0.021636,,0,0,0.400203,1.050104,0.114786,,,0.357741,,,1,0,0,0.23930,0.000000000e+00
30.3376,servo_twist,qpik,0.361993,-0.850843,1.014933,1.477063,0.361803,1.015747,0.369636,0.369636,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.1730,0.034516,0.021993,,0,0,0.399476,1.055352,0.101582,,,0.361624,,,1,0,0,0.24066,0.000000000e+00
30.4375,servo_twist,qpik,0.365486,-0.873813,1.010807,1.462495,0.365294,1.012186,0.375450,0.375450,-0.183750,-0.183750,68.0000,68.0000,1.000000,0.1644,0.035328,0.020859,,0,0,0.398411,1.060704,0.088957,,,0.365032,,,1,0,0,0.24221,0.000000000e+00
30.5520,servo_twist,qpik,0.369502,-0.899758,1.004468,1.447600,0.369313,1.005920,0.382456,0.382456,-0.181518,-0.181518,68.0000,68.0000,1.000000,0.1650,0.037204,0.020082,,0,0,0.403474,1.067387,0.076178,,,0.368828,,,1,0,0,0.24420,0.000000000e+00
30.6551,servo_twist,qpik,0.373305,-0.924298,0.997615,1.434513,0.373081,0.998712,0.388608,0.388608,-0.179443,-0.179443,68.0000,68.0000,1.000000,0.1690,0.038840,0.019539,,0,0,0.407542,1.074027,0.065320,,,0.372664,,,1,0,0,0.24585,0.000000000e+00
30.7520,servo_twist,qpik,0.376938,-0.948189,0.990512,1.422456,0.376742,0.991923,0.394341,0.394341,-0.177440,-0.177440,68.0000,68.0000,1.000000,0.1735,0.040296,0.019186,,0,0,0.410456,1.080548,0.055828,,,0.376727,,,1,0,0,0.24723,0.000000000e+00
30.8520,servo_twist,qpik,0.380923,-0.973404,0.979750,1.412069,0.380714,0.983493,0.400422,0.400422,-0.175440,-0.175440,68.0000,68.0000,1.000000,0.2820,0.054689,0.019082,,0,0,0.424213,1.088379,0.047863,,,0.380137,,,1,0,0,0.24881,0.000000000e+00
30.9520,servo_twist,qpik,0.385775,-1.000355,0.958745,1.406779,0.385500,0.967052,0.408232,0.408232,-0.173393,-0.173393,68.0000,68.0000,1.000000,0.8064,0.074308,0.019731,,0,0,0.466128,1.099693,0.043955,,,0.385196,,,1,0,0,0.25095,0.000000000e+00
31.0527,servo_twist,qpik,0.392239,-1.023907,0.930078,1.409171,0.391864,0.938830,0.418692,0.418692,-0.171342,-0.171342,68.0000,68.0000,1.000000,0.8238,0.084439,0.020913,,0,0,0.536763,1.116809,0.046092,,,0.390642,,,1,0,0,0.25349,0.000000000e+00
31.1520,servo_twist,qpik,0.399843,-1.040722,0.908402,1.412371,0.399433,0.915355,0.429316,0.429316,-0.169341,-0.169341,68.0000,68.0000,1.000000,0.8694,0.087532,0.020743,,0,0,0.539780,1.136377,0.051714,,,0.398926,,,1,0,0,0.25536,0.000000000e+00
31.2521,servo_twist,qpik,0.408261,-1.054091,0.892303,1.414870,0.407824,0.897832,0.439812,0.439812,-0.167328,-0.167328,68.0000,68.0000,1.000000,0.9187,0.088477,0.019876,,0,0,0.520506,1.157460,0.058835,,,0.407908,,,1,0,0,0.25683,0.000000000e+00
31.3524,servo_twist,qpik,0.416053,-1.069429,0.871624,1.418926,0.415632,0.879855,0.449703,0.449703,-0.166613,-0.166613,68.0000,68.0000,1.000000,0.8211,0.089171,0.016351,,0,0,0.529223,1.177474,0.067774,,,0.414400,,,1,0,0,0.25829,0.000000000e+00
31.4520,servo_twist,qpik,0.424809,-1.083897,0.852302,1.423215,0.424325,0.857707,0.460720,0.460720,-0.166513,-0.166513,68.0000,68.0000,1.000000,0.9366,0.089791,0.019884,,0,0,0.551352,1.200019,0.077245,,,0.423979,,,1,0,0,0.25988,0.000000000e+00
31.5527,servo_twist,qpik,0.433311,-1.096945,0.836499,1.426023,0.432950,0.843273,0.470823,0.470823,-0.166113,-0.166113,68.0000,68.0000,1.000000,0.9137,0.090211,0.018580,,0,0,0.533650,1.221504,0.086526,,,0.431155,,,1,0,0,0.26102,0.000000000e+00
31.6525,servo_twist,qpik,0.441896,-1.110318,0.819658,1.429176,0.441470,0.824773,0.481419,0.481419,-0.165913,-0.165913,68.0000,68.0000,1.000000,0.9302,0.090752,0.018763,,0,0,0.537669,1.243158,0.095942,,,0.440632,,,1,0,0,0.26232,0.000000000e+00
31.7520,servo_twist,qpik,0.450335,-1.123007,0.804851,1.431381,0.449946,0.810095,0.491288,0.491288,-0.165591,-0.165591,68.0000,68.0000,1.000000,0.8985,0.091160,0.017100,,0,0,0.535912,1.264431,0.105266,,,0.448275,,,1,0,0,0.26343,0.000000000e+00
31.8525,servo_twist,qpik,0.459381,-1.135362,0.790139,1.433705,0.458945,0.793008,0.502157,0.502157,-0.165391,-0.165391,68.0000,68.0000,1.000000,0.9760,0.091650,0.020271,,0,0,0.553533,1.287202,0.114053,,,0.457554,,,1,0,0,0.26463,0.000000000e+00
31.9524,servo_twist,qpik,0.468639,-1.145989,0.779186,1.434797,0.467944,0.782553,0.512382,0.512382,-0.165069,-0.165069,68.0000,68.0000,1.000000,0.9513,0.092007,0.020942,,0,0,0.525842,1.309863,0.121980,,,0.467102,,,1,0,0,0.26537,0.000000000e+00
32.0533,servo_twist,qpik,0.478811,-1.155277,0.771188,1.435366,0.478134,0.771959,0.523678,0.523678,-0.165069,-0.165069,68.0000,68.0000,1.000000,0.9385,0.092345,0.026479,,0,0,0.526473,1.334828,0.129175,,,0.477698,,,1,0,0,0.26615,0.000000000e+00
32.1523,servo_twist,qpik,0.489111,-1.163137,0.766524,1.435127,0.488583,0.766915,0.534469,0.534469,-0.165069,-0.165069,68.0000,68.0000,1.000000,0.9085,0.092549,0.032875,,0,0,0.509008,1.359372,0.136324,,,0.486314,,,1,0,0,0.26659,0.000000000e+00
32.2520,servo_twist,qpik,0.499923,-1.166792,0.771813,1.432090,0.499273,0.767962,0.544857,0.544857,-0.165069,-0.165069,68.0000,68.0000,1.000000,0.8893,0.092587,0.031216,,0,0,0.478015,1.384182,0.142995,,,0.498355,,,1,0,0,0.26660,0.000000000e+00
32.3530,servo_twist,qpik,0.509579,-1.173278,0.770630,1.430467,0.509223,0.771104,0.554351,0.554351,-0.164969,-0.164969,68.0000,68.0000,1.000000,0.9612,0.092613,0.026706,,0,0,0.489313,1.407002,0.150445,,,0.508789,,,1,0,0,0.26661,0.000000000e+00
32.4521,servo_twist,qpik,0.518678,-1.184184,0.758658,1.431555,0.518186,0.762430,0.564263,0.564263,-0.164869,-0.164869,68.0000,68.0000,1.000000,0.9792,0.092885,0.024155,,0,0,0.533025,1.429830,0.157481,,,0.516392,,,1,0,0,0.26741,0.000000000e+00
32.5520,servo_twist,qpik,0.528598,-1.193251,0.751569,1.431769,0.527878,0.753197,0.574753,0.574753,-0.164869,-0.164869,68.0000,68.0000,1.000000,0.9940,0.093190,0.027245,,0,1,0.537304,1.453944,0.165841,,,0.526926,,,1,0,0,0.26807,0.000000000e+00
32.6521,servo_twist,qpik,0.538845,-1.201815,0.744954,1.432213,0.538198,0.745168,0.585854,0.585854,-0.164769,-0.164769,68.0000,68.0000,1.000000,0.9371,0.093472,0.028088,,0,0,0.538306,1.479488,0.173706,,,0.537878,,,1,0,0,0.26865,0.000000000e+00
32.7525,servo_twist,qpik,0.548360,-1.210701,0.737663,1.432212,0.547888,0.738554,0.596067,0.596067,-0.164650,-0.164650,68.0000,68.0000,1.000000,0.9573,0.093706,0.025888,,0,0,0.535936,1.502971,0.180916,,,0.546049,,,1,0,0,0.26911,0.000000000e+00
32.8520,servo_twist,qpik,0.557806,-1.220269,0.728860,1.432635,0.557298,0.730734,0.606192,0.606192,-0.162621,-0.162621,68.0000,68.0000,1.000000,0.9719,0.093980,0.022080,,0,0,0.560364,1.526565,0.188957,,,0.556294,,,1,0,0,0.26960,0.000000000e+00
32.9533,servo_twist,qpik,0.567674,-1.229512,0.720855,1.432604,0.567343,0.721257,0.617090,0.617090,-0.160530,-0.160530,68.0000,68.0000,1.000000,0.9123,0.094282,0.024265,,0,0,0.556775,1.551296,0.195868,,,0.565936,,,1,0,0,0.27019,0.000000000e+00
33.0520,servo_twist,qpik,0.576844,-1.237264,0.715335,1.431058,0.576420,0.718133,0.626120,0.626120,-0.158517,-0.158517,68.0000,68.0000,1.000000,0.9994,0.094401,0.018207,,0,0,0.544127,1.573690,0.198558,,,0.576042,,,1,0,0,0.27051,0.000000000e+00
33.1520,servo_twist,qpik,0.586195,-1.246952,0.704486,1.430763,0.585637,0.707609,0.636190,0.636190,-0.156500,-0.156500,68.0000,68.0000,1.000000,0.9797,0.094712,0.016060,,0,0,0.569658,1.597500,0.199466,,,0.584802,,,1,0,0,0.27106,0.000000000e+00
33.2521,servo_twist,qpik,0.595071,-1.258537,0.688894,1.431474,0.594558,0.693227,0.646367,0.646367,-0.154472,-0.154472,68.0000,68.0000,1.000000,0.9415,0.095132,0.015531,,0,0,0.587394,1.620700,0.199927,,,0.593733,,,1,0,0,0.27179,0.000000000e+00
33.3520,servo_twist,qpik,0.604480,-1.268882,0.675721,1.431621,0.603990,0.678130,0.657292,0.657292,-0.152471,-0.152471,68.0000,68.0000,1.000000,0.9818,0.095532,0.017469,,0,0,0.600890,1.645068,0.199317,,,0.603511,,,1,0,0,0.27253,0.000000000e+00
33.4520,servo_twist,qpik,0.613916,-1.278057,0.665383,1.430788,0.613301,0.667536,0.667374,0.667374,-0.150442,-0.150442,68.0000,68.0000,1.000000,0.9956,0.095802,0.014232,,0,0,0.584678,1.669132,0.197485,,,0.612561,,,1,0,0,0.27301,0.000000000e+00
33.5522,servo_twist,qpik,0.623084,-1.287812,0.653413,1.430380,0.622668,0.655214,0.677773,0.677773,-0.148408,-0.148408,68.0000,68.0000,1.000000,0.9303,0.086445,0.015137,,0,0,0.601947,1.692927,0.195458,,,0.621017,,,1,0,0,0.27356,0.000000000e+00
33.6574,servo_twist,qpik,0.632092,-1.291037,0.653632,1.430582,0.631470,0.650851,0.686485,0.686485,-0.146259,-0.146259,68.0000,68.0000,1.000000,0.7643,0.016929,0.013665,,0,0,0.559016,1.714903,0.192303,,,0.630252,,,1,0,0,0.27352,0.000000000e+00
33.7574,servo_twist,qpik,0.641126,-1.285890,0.667920,1.433500,0.640630,0.659944,0.693311,0.693311,-0.144229,-0.144229,68.0000,68.0000,1.000000,0.7038,-0.000346,0.018392,,0,0,0.483422,1.736315,0.192242,,,0.639965,,,1,0,0,0.27248,0.000000000e+00
33.8574,servo_twist,qpik,0.647794,-1.273280,0.700191,1.430718,0.647468,0.687258,0.695849,0.695849,-0.142228,-0.142228,68.0000,68.0000,1.000000,0.5991,-0.000042,0.013309,,0,0,0.398265,1.749538,0.196299,,,0.647365,,,1,0,0,0.27050,0.000000000e+00
33.9574,servo_twist,qpik,0.653167,-1.255261,0.745504,1.422154,0.652905,0.730246,0.696457,0.696457,-0.140196,-0.140196,68.0000,68.0000,1.000000,0.6014,0.024585,0.010334,,0,0,0.371403,1.757911,0.202624,,,0.652089,,,1,0,0,0.26840,0.000000000e+00
34.0665,servo_twist,qpik,0.657779,-1.240214,0.778996,1.414631,0.657542,0.770545,0.698598,0.698598,-0.138047,-0.138047,68.0000,68.0000,1.000000,0.8388,0.055664,0.010968,,0,0,0.393793,1.765457,0.212156,,,0.657542,,,1,0,0,0.26684,0.000000000e+00
34.1639,servo_twist,qpik,0.662468,-1.232597,0.790135,1.411951,0.662232,0.789674,0.702547,0.702547,-0.136012,-0.136012,68.0000,68.0000,1.000000,0.9161,0.061542,0.012354,,0,0,0.453078,1.775648,0.216747,,,0.661602,,,1,0,0,0.26646,0.000000000e+00
34.2639,servo_twist,qpik,0.668063,-1.229815,0.787104,1.413812,0.667788,0.789936,0.709046,0.709046,-0.133987,-0.133987,68.0000,68.0000,1.000000,0.8592,0.062556,0.013852,,0,0,0.515748,1.789871,0.218067,,,0.667455,,,1,0,0,0.26664,0.000000000e+00
34.3648,servo_twist,qpik,0.674157,-1.229287,0.778345,1.418021,0.673878,0.782326,0.716413,0.716413,-0.131969,-0.131969,68.0000,68.0000,1.000000,0.8689,0.062853,0.015018,,0,0,0.557628,1.806008,0.220027,,,0.673882,,,1,0,0,0.26705,0.000000000e+00
34.4640,servo_twist,qpik,0.680382,-1.228212,0.770961,1.422002,0.680068,0.773617,0.723592,0.723592,-0.129958,-0.129958,68.0000,68.0000,1.000000,0.8873,0.063057,0.014893,,0,0,0.562448,1.822483,0.221892,,,0.679402,,,1,0,0,0.26746,0.000000000e+00
34.5644,servo_twist,qpik,0.686823,-1.226770,0.764204,1.425949,0.686451,0.766339,0.730659,0.730659,-0.127879,-0.127879,68.0000,68.0000,1.000000,0.9229,0.063202,0.015373,,0,0,0.561904,1.841662,0.223755,,,0.686150,,,1,0,0,0.26782,0.000000000e+00
34.6646,servo_twist,qpik,0.693221,-1.225043,0.758473,1.429583,0.692884,0.760492,0.737703,0.737703,-0.125855,-0.125855,68.0000,68.0000,1.000000,0.9022,0.063142,0.015306,,0,0,0.563032,1.875604,0.225495,,,0.692796,,,1,0,0,0.26807,0.000000000e+00
34.7642,servo_twist,qpik,0.699458,-1.223833,0.751598,1.433507,0.699145,0.753703,0.744763,0.744763,-0.123845,-0.123845,68.0000,68.0000,1.000000,0.8949,0.062918,0.014947,,0,0,0.567105,1.925087,0.227412,,,0.698594,,,1,0,0,0.26838,0.000000000e+00
34.8639,servo_twist,qpik,0.705750,-1.222712,0.744573,1.437469,0.705434,0.747018,0.751682,0.751682,-0.121844,-0.121844,68.0000,68.0000,1.000000,0.8927,0.062510,0.014542,,0,0,0.572137,1.991471,0.229318,,,0.704553,,,1,0,0,0.26870,0.000000000e+00
34.9644,servo_twist,qpik,0.712040,-1.221787,0.737045,1.441572,0.711730,0.739496,0.758755,0.758755,-0.119834,-0.119834,68.0000,68.0000,1.000000,0.8876,0.062026,0.014437,,0,0,0.577684,2.073932,0.231194,,,0.711457,,,1,0,0,0.26900,0.000000000e+00
35.0640,servo_twist,qpik,0.718249,-1.221048,0.729079,1.445786,0.717934,0.731415,0.765762,0.765762,-0.117830,-0.117830,68.0000,68.0000,1.000000,0.8763,0.061819,0.014398,,0,0,0.581990,2.172140,0.233036,,,0.717287,,,1,0,0,0.26935,0.000000000e+00
35.1639,servo_twist,qpik,0.724486,-1.220900,0.719502,1.450446,0.724159,0.722776,0.772800,0.772800,-0.115806,-0.115806,68.0000,68.0000,1.000000,0.8660,0.061490,0.014258,,0,0,0.593258,2.286962,0.234994,,,0.722990,,,1,0,0,0.26973,0.000000000e+00
35.2639,servo_twist,qpik,0.730641,-1.221043,0.709307,1.455233,0.730334,0.712182,0.779930,0.779930,-0.113805,-0.113805,68.0000,68.0000,1.000000,0.8597,0.060791,0.014270,,0,0,0.603953,2.416447,0.236836,,,0.729379,,,1,0,0,0.27016,0.000000000e+00
35.3639,servo_twist,qpik,0.736745,-1.221610,0.697998,1.460305,0.736422,0.701378,0.787008,0.787008,-0.111788,-0.111788,68.0000,68.0000,1.000000,0.8461,0.059961,0.014031,,0,0,0.611777,2.560818,0.238623,,,0.735773,,,1,0,0,0.27058,0.000000000e+00
35.4639,servo_twist,qpik,0.742765,-1.222636,0.685629,1.465621,0.742469,0.689440,0.794155,0.794155,-0.109787,-0.109787,68.0000,68.0000,1.000000,0.8412,0.059136,0.014170,,0,0,0.623889,2.718833,0.240320,,,0.742237,,,1,0,0,0.27105,0.000000000e+00
35.5661,servo_twist,qpik,0.748888,-1.224099,0.671909,1.471333,0.748586,0.675058,0.801531,0.801531,-0.107728,-0.107728,68.0000,68.0000,1.000000,0.8461,0.058367,0.014506,,0,0,0.634262,2.893517,0.241905,,,0.747479,,,1,0,0,0.27156,0.000000000e+00
35.6639,servo_twist,qpik,0.754646,-1.225984,0.657303,1.477238,0.754350,0.662003,0.808319,0.808319,-0.105698,-0.105698,68.0000,68.0000,1.000000,0.8190,0.057608,0.014120,,0,0,0.643639,3.076355,0.243428,,,0.753821,,,1,0,0,0.27205,0.000000000e+00
35.7639,servo_twist,qpik,0.760441,-1.228675,0.640711,1.483653,0.760144,0.646156,0.815553,0.815553,-0.103658,-0.103658,68.0000,68.0000,1.000000,0.8111,0.056930,0.000000,,1,0,0.661252,3.272797,0.244889,,,0.760166,,,1,0,0,0.27263,0.000000000e+00
35.8645,servo_twist,qpik,0.766154,-1.231897,0.622824,1.490401,0.765880,0.627289,0.822722,0.822722,-0.101614,-0.101614,68.0000,68.0000,1.000000,0.7858,0.056443,0.000000,,1,0,0.673246,3.480666,0.246161,,,0.765098,,,1,0,0,0.27332,0.000000000e+00
35.9639,servo_twist,qpik,0.770952,-1.238133,0.597973,1.498743,0.770717,0.606746,0.829040,0.829040,-0.099614,-0.099614,68.0000,68.0000,1.000000,0.6611,0.056238,0.000000,,1,0,0.714294,3.669773,0.248042,,,0.770560,,,1,0,0,0.27404,0.000000000e+00
36.0639,servo_twist,qpik,0.774635,-1.248862,0.562389,1.509449,0.774467,0.575418,0.835164,0.835164,-0.097594,-0.097594,68.0000,68.0000,1.000000,0.5284,0.056749,0.000000,,1,0,0.764944,3.822810,0.250626,,,0.773864,,,1,0,0,0.27520,0.000000000e+00
36.1639,servo_twist,qpik,0.776959,-1.264765,0.514878,1.522703,0.776835,0.530214,0.841041,0.841041,-0.095573,-0.095573,68.0000,68.0000,1.000000,0.3567,0.049420,0.000000,,1,0,0.783719,3.924644,0.253668,,,0.776712,,,1,0,0,0.27685,0.000000000e+00
36.2640,servo_twist,qpik,0.778315,-1.283185,0.464843,1.537936,0.778249,0.479861,0.845279,0.845279,-0.095573,-0.095573,68.0000,68.0000,1.000000,0.3908,0.006883,0.000000,,1,0,0.797111,3.986664,0.257314,,,0.778273,,,1,0,0,0.27839,0.000000000e+00
36.3653,servo_twist,qpik,0.779065,-1.299210,0.424358,1.551678,0.779027,0.437240,0.847774,0.847774,-0.095573,-0.095573,68.0000,68.0000,1.000000,0.3641,-0.004547,0.000000,,1,0,0.792962,4.022054,0.263067,,,0.778950,,,1,0,0,0.27929,0.000000000e+00
36.4639,servo_twist,qpik,0.779312,-1.311251,0.395325,1.563635,0.779300,0.402630,0.848558,0.848558,-0.095573,-0.095573,68.0000,68.0000,1.000000,0.0717,-0.061206,0.000000,,1,0,0.784570,4.036681,0.266806,,,0.779339,,,1,0,0,0.27982,0.000000000e+00
36.5639,servo_twist,qpik,0.778884,-1.318012,0.378941,1.576368,0.778988,0.383362,0.847006,0.847006,-0.095573,-0.095573,68.0000,68.0000,1.000000,0.2156,-0.089446,0.000000,,1,0,0.694990,4.021308,0.270823,,,0.779325,,,1,0,0,0.27931,0.000000000e+00
36.6640,servo_twist,qpik,0.778122,-1.321072,0.370532,1.587660,0.778209,0.371999,0.844015,0.844015,-0.095573,-0.095573,68.0000,68.0000,1.000000,0.3681,-0.089764,0.000000,,1,0,0.531997,3.991287,0.278396,,,0.778708,,,1,0,0,0.27837,0.000000000e+00
36.7639,servo_twist,qpik,0.776498,-1.318340,0.376788,1.594195,0.776610,0.371127,0.840107,0.840107,-0.095573,-0.095573,68.0000,68.0000,1.000000,0.4196,-0.089878,0.000000,,1,0,0.492811,3.924193,0.288468,,,0.776928,,,1,0,0,0.27733,0.000000000e+00
36.8639,servo_twist,qpik,0.773965,-1.308763,0.400132,1.594831,0.774111,0.391844,0.834356,0.834356,-0.095573,-0.095573,68.0000,68.0000,1.000000,0.4708,-0.089088,0.000000,,1,0,0.492735,3.819761,0.295943,,,0.774183,,,1,0,0,0.27585,0.000000000e+00
36.9639,servo_twist,qpik,0.770609,-1.295959,0.431848,1.592161,0.770779,0.419961,0.827828,0.827828,-0.095073,-0.095073,68.0000,68.0000,1.000000,0.5119,-0.088122,0.000000,,1,0,0.514909,3.684933,0.301640,,,0.771325,,,1,0,0,0.27405,0.000000000e+00
37.0639,servo_twist,qpik,0.766436,-1.278734,0.475026,1.585136,0.766647,0.459615,0.820016,0.820016,-0.093073,-0.093073,68.0000,68.0000,1.000000,0.5346,-0.086873,0.000000,,1,0,0.514348,3.523200,0.305367,,,0.766936,,,1,0,0,0.27211,0.000000000e+00
37.1639,servo_twist,qpik,0.761423,-1.258228,0.527113,1.574417,0.761675,0.509392,0.811046,0.811046,-0.091055,-0.091055,68.0000,68.0000,1.000000,0.5559,-0.085465,0.000000,,1,0,0.501720,3.338225,0.307453,,,0.761652,,,1,0,0,0.27011,0.000000000e+00
37.2639,servo_twist,qpik,0.755479,-1.235281,0.586196,1.560327,0.755779,0.567825,0.801356,0.801356,-0.089046,-0.089046,68.0000,68.0000,1.000000,0.5999,-0.084077,0.000000,,1,0,0.478849,3.132352,0.308046,,,0.756524,,,1,0,0,0.26785,0.000000000e+00
37.3640,servo_twist,qpik,0.748479,-1.217738,0.632570,1.548977,0.748825,0.621128,0.790061,0.790061,-0.087045,-0.087045,68.0000,68.0000,1.000000,0.6966,-0.083506,0.003991,,0,0,0.478879,2.910112,0.308994,,,0.749164,,,1,0,0,0.26587,0.000000000e+00
37.4651,servo_twist,qpik,0.740489,-1.206447,0.663654,1.541425,0.740889,0.655266,0.779593,0.779593,-0.085021,-0.085021,68.0000,68.0000,1.000000,0.7843,-0.084222,0.004858,,0,0,0.501606,2.682893,0.310379,,,0.742298,,,1,0,0,0.26439,0.000000000e+00
37.5639,servo_twist,qpik,0.732349,-1.198781,0.685547,1.536194,0.732762,0.680172,0.768907,0.768907,-0.083015,-0.083015,68.0000,68.0000,1.000000,0.8140,-0.085404,0.004906,,0,0,0.529876,2.478052,0.311440,,,0.733540,,,1,0,0,0.26328,0.000000000e+00
37.6639,servo_twist,qpik,0.723934,-1.192450,0.703929,1.531649,0.724344,0.698655,0.759072,0.759072,-0.081005,-0.081005,68.0000,68.0000,1.000000,0.8304,-0.086853,0.006313,,0,0,0.536225,2.296188,0.311403,,,0.724426,,,1,0,0,0.26234,0.000000000e+00
37.7639,servo_twist,qpik,0.715468,-1.186109,0.722107,1.526914,0.715902,0.716789,0.749091,0.749091,-0.079005,-0.079005,68.0000,68.0000,1.000000,0.8426,-0.088121,0.006362,,0,0,0.535993,2.144626,0.310073,,,0.715916,,,1,0,0,0.26151,0.000000000e+00
37.8639,servo_twist,qpik,0.706746,-1.181314,0.736280,1.523003,0.707184,0.732497,0.738933,0.738933,-0.077005,-0.077005,68.0000,68.0000,1.000000,0.8589,-0.089262,0.005787,,0,0,0.535630,2.019512,0.307963,,,0.708502,,,1,0,0,0.26063,0.000000000e+00
37.9639,servo_twist,qpik,0.697870,-1.177156,0.748660,1.519351,0.698310,0.745273,0.728893,0.728893,-0.075004,-0.075004,68.0000,68.0000,1.000000,0.8711,-0.090116,0.005600,,0,0,0.535712,1.924772,0.304899,,,0.698848,,,1,0,0,0.25996,0.000000000e+00
38.0639,servo_twist,qpik,0.688902,-1.173451,0.759700,1.515868,0.689350,0.756600,0.718500,0.718500,-0.073003,-0.073003,68.0000,68.0000,1.000000,0.8799,-0.090512,0.005101,,0,0,0.534032,1.862128,0.301026,,,0.689256,,,1,0,0,0.25935,0.000000000e+00
38.1640,servo_twist,qpik,0.679878,-1.169987,0.769940,1.512445,0.680357,0.767142,0.708385,0.708385,-0.070989,-0.070989,68.0000,68.0000,1.000000,0.8769,-0.090416,0.005171,,0,0,0.531897,1.830848,0.296465,,,0.680237,,,1,0,0,0.25875,0.000000000e+00
38.2639,servo_twist,qpik,0.670938,-1.166175,0.780848,1.508754,0.671393,0.778050,0.698126,0.698126,-0.068989,-0.068989,68.0000,68.0000,1.000000,0.8839,-0.090161,0.004672,,0,0,0.525745,1.807644,0.291259,,,0.671939,,,1,0,0,0.25814,0.000000000e+00
38.3639,servo_twist,qpik,0.661980,-1.162681,0.790849,1.505213,0.662485,0.787981,0.688092,0.688092,-0.068804,-0.068804,68.0000,68.0000,1.000000,0.8604,-0.089924,0.006578,,0,0,0.521130,1.784520,0.285745,,,0.663372,,,1,0,0,0.25757,0.000000000e+00
38.4646,servo_twist,qpik,0.652958,-1.158998,0.801253,1.501514,0.653412,0.798506,0.677699,0.677699,-0.070817,-0.070817,68.0000,68.0000,1.000000,0.8816,-0.089672,0.008593,,0,0,0.509049,1.761236,0.279956,,,0.655016,,,1,0,0,0.25696,0.000000000e+00
38.5639,servo_twist,qpik,0.644065,-1.155501,0.811152,1.497927,0.644514,0.808384,0.667690,0.667690,-0.072825,-0.072825,68.0000,68.0000,1.000000,0.8795,-0.089423,0.009571,,0,0,0.507409,1.738253,0.274154,,,0.644423,,,1,0,0,0.25636,0.000000000e+00
38.6639,servo_twist,qpik,0.635102,-1.151969,0.821132,1.494309,0.635534,0.818053,0.657568,0.657568,-0.074844,-0.074844,68.0000,68.0000,1.000000,0.8850,-0.089190,0.010008,,0,0,0.502430,1.715146,0.268372,,,0.637293,,,1,0,0,0.25579,0.000000000e+00
38.7639,servo_twist,qpik,0.626174,-1.148508,0.830934,1.490768,0.626621,0.828438,0.647376,0.647376,-0.076845,-0.076845,68.0000,68.0000,1.000000,0.8753,-0.088943,0.010189,,0,0,0.496542,1.692167,0.262845,,,0.626721,,,1,0,0,0.25517,0.000000000e+00
38.8800,servo_twist,qpik,0.615551,-1.145100,0.840861,1.487087,0.616141,0.839468,0.635500,0.635500,-0.079154,-0.079154,68.0000,68.0000,1.000000,0.8799,-0.088688,0.010338,,0,0,0.491485,1.665009,0.256904,,,0.616143,,,1,0,0,0.25447,0.000000000e+00
38.9798,servo_twist,qpik,0.606805,-1.141502,0.851092,1.483503,0.607289,0.849172,0.625386,0.625386,-0.081191,-0.081191,68.0000,68.0000,1.000000,0.8694,-0.088443,0.010348,,0,0,0.486300,1.642466,0.251782,,,0.607378,,,1,0,0,0.25389,0.000000000e+00
39.0798,servo_twist,qpik,0.597979,-1.137912,0.861401,1.479959,0.598443,0.859208,0.615350,0.615350,-0.083233,-0.083233,68.0000,68.0000,1.000000,0.8616,-0.088201,0.012468,,0,0,0.481050,1.619668,0.247050,,,0.599725,,,1,0,0,0.25323,0.000000000e+00
39.1798,servo_twist,qpik,0.589433,-1.133336,0.874190,1.475887,0.589867,0.871408,0.605359,0.605359,-0.085233,-0.085233,68.0000,68.0000,1.000000,0.8598,-0.087908,0.011601,,0,0,0.468950,1.597447,0.242542,,,0.590837,,,1,0,0,0.25241,0.000000000e+00
39.2798,servo_twist,qpik,0.580695,-1.129494,0.885369,1.472273,0.581095,0.882002,0.595262,0.595262,-0.087270,-0.087270,68.0000,68.0000,1.000000,0.8648,-0.087642,0.011243,,0,0,0.468536,1.574853,0.238649,,,0.582387,,,1,0,0,0.25171,0.000000000e+00
39.3798,servo_twist,qpik,0.571855,-1.126116,0.895547,1.469011,0.572259,0.892718,0.585058,0.585058,-0.089279,-0.089279,68.0000,68.0000,1.000000,0.8814,-0.087389,0.010395,,0,0,0.464418,1.552120,0.235372,,,0.573646,,,1,0,0,0.25100,0.000000000e+00
39.4804,servo_twist,qpik,0.562914,-1.122970,0.905320,1.465912,0.563360,0.903173,0.574850,0.574850,-0.091322,-0.091322,68.0000,68.0000,1.000000,0.8879,-0.087141,0.009835,,0,0,0.464670,1.529244,0.232569,,,0.563951,,,1,0,0,0.25029,0.000000000e+00
39.5798,servo_twist,qpik,0.554096,-1.120103,0.914468,1.463073,0.554563,0.911655,0.564916,0.564916,-0.093329,-0.093329,68.0000,68.0000,1.000000,0.8615,-0.086921,0.010096,,0,0,0.466155,1.506666,0.230164,,,0.554904,,,1,0,0,0.24973,0.000000000e+00
39.6798,servo_twist,qpik,0.545429,-1.116661,0.925082,1.459918,0.545866,0.921673,0.554911,0.554911,-0.095329,-0.095329,68.0000,68.0000,1.000000,0.8526,-0.086669,0.010562,,0,0,0.456591,1.484370,0.227951,,,0.547249,,,1,0,0,0.24904,0.000000000e+00
39.7798,servo_twist,qpik,0.536761,-1.113124,0.936048,1.456742,0.537226,0.932844,0.544775,0.544775,-0.097338,-0.097338,68.0000,68.0000,1.000000,0.8427,-0.086392,0.010641,,0,0,0.448931,1.462052,0.226074,,,0.537298,,,1,0,0,0.24827,0.000000000e+00
39.8798,servo_twist,qpik,0.528131,-1.109621,0.947028,1.453635,0.528561,0.943752,0.534630,0.534630,-0.099338,-0.099338,68.0000,68.0000,1.000000,0.8476,-0.086115,0.010707,,0,0,0.442068,1.439835,0.224531,,,0.529069,,,1,0,0,0.24752,0.000000000e+00
39.9798,servo_twist,qpik,0.519507,-1.106172,0.957987,1.450600,0.519941,0.955184,0.524470,0.524470,-0.101340,-0.101340,68.0000,68.0000,1.000000,0.8444,-0.085837,0.010691,,0,0,0.439356,1.417656,0.223310,,,0.520100,,,1,0,0,0.24671,0.000000000e+00
40.0798,servo_twist,qpik,0.510887,-1.102717,0.969072,1.447590,0.511331,0.965970,0.514443,0.514443,-0.103352,-0.103352,68.0000,68.0000,1.000000,0.8425,-0.085555,0.010991,,0,0,0.435726,1.395495,0.222306,,,0.512188,,,1,0,0,0.24595,0.000000000e+00
40.1798,servo_twist,qpik,0.502349,-1.099233,0.980305,1.444598,0.502786,0.977611,0.504261,0.504261,-0.105354,-0.105354,68.0000,68.0000,1.000000,0.8391,-0.085256,0.010812,,0,0,0.432046,1.373539,0.221554,,,0.503289,,,1,0,0,0.24512,0.000000000e+00
40.2798,servo_twist,qpik,0.493811,-1.095796,0.991497,1.441669,0.494232,0.988467,0.494168,0.494168,-0.107354,-0.107354,68.0000,68.0000,1.000000,0.8370,-0.084975,0.010815,,0,0,0.429305,1.351608,0.220973,,,0.494328,,,1,0,0,0.24435,0.000000000e+00
40.3798,servo_twist,qpik,0.485468,-1.091858,1.004023,1.438439,0.485889,0.999934,0.484186,0.484186,-0.109398,-0.109398,68.0000,68.0000,1.000000,0.8284,-0.084686,0.011527,,0,0,0.426784,1.330049,0.220431,,,0.487189,,,1,0,0,0.24349,0.000000000e+00
40.4798,servo_twist,qpik,0.476944,-1.088370,1.015497,1.435535,0.477370,1.012378,0.473868,0.473868,-0.111420,-0.111420,68.0000,68.0000,1.000000,0.8411,-0.084365,0.010379,,0,0,0.422786,1.308202,0.220179,,,0.477340,,,1,0,0,0.24263,0.000000000e+00
40.5798,servo_twist,qpik,0.468424,-1.085340,1.025897,1.432942,0.468850,1.023234,0.463646,0.463646,-0.113428,-0.113428,68.0000,68.0000,1.000000,0.8297,-0.084089,0.010400,,0,0,0.423674,1.286435,0.220190,,,0.469754,,,1,0,0,0.24179,0.000000000e+00
40.6798,servo_twist,qpik,0.460102,-1.081365,1.038666,1.429766,0.460512,1.035137,0.453593,0.453593,-0.115493,-0.115493,68.0000,68.0000,1.000000,0.8213,-0.083814,0.011205,,0,0,0.417987,1.264984,0.219967,,,0.461768,,,1,0,0,0.24090,0.000000000e+00
40.7798,servo_twist,qpik,0.451710,-1.077674,1.050421,1.426973,0.452126,1.047774,0.443365,0.443365,-0.117493,-0.117493,68.0000,68.0000,1.000000,0.8217,-0.083695,0.010848,,0,0,0.414513,1.243499,0.220059,,,0.452457,,,1,0,0,0.23996,0.000000000e+00
40.8798,servo_twist,qpik,0.443320,-1.073186,1.062672,1.424538,0.443738,1.058769,0.433355,0.433355,-0.119515,-0.119515,68.0000,68.0000,1.000000,0.8198,-0.084248,0.010918,,0,0,0.413582,1.222065,0.220814,,,0.444081,,,1,0,0,0.23909,0.000000000e+00
40.9798,servo_twist,qpik,0.434941,-1.066711,1.075460,1.423303,0.435361,1.072697,0.423020,0.423020,-0.121516,-0.121516,68.0000,68.0000,1.000000,0.8545,-0.084206,0.010654,,0,0,0.417246,1.200801,0.223648,,,0.435786,,,1,0,0,0.23796,0.000000000e+00
41.0798,servo_twist,qpik,0.426565,-1.056968,1.086267,1.425666,0.426975,1.084425,0.413344,0.413344,-0.123516,-0.123516,68.0000,68.0000,1.000000,0.9130,-0.079132,0.011126,,0,0,0.433609,1.180102,0.231725,,,0.427709,,,1,0,0,0.23689,0.000000000e+00
41.1798,servo_twist,qpik,0.418422,-1.044037,1.090839,1.433972,0.418853,1.091372,0.404412,0.404412,-0.125523,-0.125523,68.0000,68.0000,1.000000,0.5420,-0.065529,0.011741,,0,0,0.480419,1.161072,0.247748,,,0.418851,,,1,0,0,0.23605,0.000000000e+00
41.2798,servo_twist,qpik,0.411246,-1.028846,1.088302,1.447988,0.411597,1.090796,0.397107,0.397107,-0.127524,-0.127524,68.0000,68.0000,1.000000,0.3192,-0.051972,0.012785,,0,0,0.537704,1.145649,0.270709,,,0.412524,,,1,0,0,0.23556,0.000000000e+00
41.3810,servo_twist,qpik,0.405241,-1.010899,1.084244,1.464903,0.405546,1.087061,0.390947,0.390947,-0.129560,-0.129560,68.0000,68.0000,1.000000,0.1814,-0.040939,0.013243,,0,0,0.576741,1.133754,0.296940,,,0.406029,,,1,0,0,0.23515,0.000000000e+00
41.4798,servo_twist,qpik,0.400517,-0.990988,1.081253,1.482597,0.400751,1.082715,0.386604,0.386604,-0.131560,-0.131560,68.0000,68.0000,1.000000,0.1269,-0.035768,0.014054,,0,0,0.605753,1.125207,0.323640,,,0.400743,,,1,0,0,0.23453,0.000000000e+00
41.5800,servo_twist,qpik,0.396492,-0.968967,1.081130,1.500197,0.396716,1.081633,0.381910,0.381910,-0.133595,-0.133595,68.0000,68.0000,1.000000,0.1008,-0.031952,0.013652,,0,0,0.618753,1.118399,0.349637,,,0.397271,,,1,0,0,0.23366,0.000000000e+00
41.6798,servo_twist,qpik,0.393091,-0.947834,1.078033,1.518581,0.393260,1.080656,0.377302,0.377302,-0.135611,-0.135611,68.0000,68.0000,1.000000,0.0294,-0.016184,0.013064,,0,0,0.643996,1.113477,0.374537,,,0.393407,,,1,0,0,0.23278,0.000000000e+00
41.7870,servo_twist,qpik,0.390557,-0.930094,1.064110,1.539546,0.390665,1.069241,0.374924,0.374924,-0.137685,-0.137685,68.0000,68.0000,1.000000,0.0660,0.014061,0.014113,,0,0,0.681542,1.111819,0.397266,,,0.390714,,,1,0,0,0.23235,0.000000000e+00
41.8866,servo_twist,qpik,0.390028,-0.920835,1.037354,1.556641,0.390040,1.046290,0.376239,0.376239,-0.139716,-0.139716,68.0000,68.0000,1.000000,0.0138,0.032174,0.016172,,0,0,0.699159,1.115679,0.401875,,,0.389897,,,1,0,0,0.23295,0.000000000e+00
41.9880,servo_twist,qpik,0.391110,-0.912765,1.013464,1.570781,0.391057,1.018155,0.379965,0.379965,-0.141778,-0.141778,68.0000,68.0000,1.000000,0.0563,-0.013403,0.017686,,0,0,0.688615,1.123455,0.393785,,,0.390571,,,1,0,0,0.23395,0.000000000e+00
42.0870,servo_twist,qpik,0.391414,-0.898706,1.002721,1.586714,0.391379,1.004001,0.378902,0.378902,-0.143830,-0.143830,68.0000,68.0000,1.000000,0.2656,-0.069792,0.015132,,0,0,0.645777,1.129320,0.387689,,,0.391427,,,1,0,0,0.23266,0.000000000e+00
42.1881,servo_twist,qpik,0.390035,-0.884373,1.001890,1.599808,0.390200,1.000283,0.373912,0.373912,-0.145870,-0.145870,68.0000,68.0000,1.000000,0.4379,-0.079434,0.012092,,0,0,0.554983,1.130023,0.385662,,,0.390490,,,1,0,0,0.23058,0.000000000e+00
42.2882,servo_twist,qpik,0.386819,-0.867518,1.017066,1.605952,0.387034,1.008137,0.367065,0.367065,-0.147913,-0.147913,68.0000,68.0000,1.000000,0.5061,-0.080469,0.010797,,0,0,0.539158,1.123828,0.389837,,,0.387890,,,1,0,0,0.22817,0.000000000e+00
42.3931,servo_twist,qpik,0.381177,-0.847771,1.046082,1.605483,0.381464,1.036726,0.355806,0.355806,-0.150007,-0.150007,68.0000,68.0000,1.000000,0.6376,-0.079712,0.008673,,0,0,0.569445,1.109573,0.398933,,,0.382433,,,1,0,0,0.22520,0.000000000e+00
42.4931,servo_twist,qpik,0.374056,-0.831929,1.067414,1.605582,0.374451,1.062138,0.344356,0.344356,-0.152044,-0.152044,68.0000,68.0000,1.000000,0.8390,-0.075112,0.008543,,0,0,0.582383,1.092284,0.407033,,,0.375034,,,1,0,0,0.22307,0.000000000e+00
42.5936,servo_twist,qpik,0.366554,-0.818281,1.078078,1.610075,0.366959,1.075210,0.334061,0.334061,-0.154100,-0.154100,68.0000,68.0000,1.000000,0.9931,-0.071418,0.009516,,0,0,0.610468,1.075742,0.415633,,,0.367759,,,1,0,0,0.22166,0.000000000e+00
42.6923,servo_twist,qpik,0.359317,-0.804908,1.085655,1.615864,0.359675,1.082977,0.324413,0.324413,-0.156125,-0.156125,68.0000,68.0000,1.000000,0.9947,-0.070162,0.010176,,0,0,0.627879,1.060364,0.423393,,,0.359895,,,1,0,0,0.22028,0.000000000e+00
42.7936,servo_twist,qpik,0.352084,-0.790232,1.094267,1.621657,0.352475,1.091634,0.314652,0.314652,-0.158178,-0.158178,68.0000,68.0000,1.000000,0.9764,-0.068451,0.010414,,0,0,0.635702,1.045261,0.430553,,,0.352425,,,1,0,0,0.21894,0.000000000e+00
42.8967,servo_twist,qpik,0.344841,-0.774653,1.101721,1.628539,0.345215,1.099575,0.304948,0.304948,-0.160267,-0.160267,68.0000,68.0000,1.000000,0.9135,-0.066546,0.010243,,0,0,0.646444,1.030652,0.437718,,,0.346485,,,1,0,0,0.21747,0.000000000e+00
43.0016,servo_twist,qpik,0.337816,-0.757875,1.109797,1.635720,0.338225,1.107638,0.294895,0.294895,-0.162383,-0.162383,68.0000,68.0000,1.000000,0.9532,-0.065271,0.010400,,0,0,0.656353,1.016959,0.444586,,,0.339258,,,1,0,0,0.21588,0.000000000e+00
43.1013,servo_twist,qpik,0.331251,-0.741368,1.117495,1.642770,0.331589,1.115335,0.285531,0.285531,-0.164408,-0.164408,68.0000,68.0000,1.000000,0.9229,-0.064041,0.010217,,0,0,0.663631,1.004200,0.450984,,,0.332250,,,1,0,0,0.21430,0.000000000e+00
43.2034,servo_twist,qpik,0.324611,-0.724148,1.125210,1.650175,0.324953,1.123084,0.275756,0.275756,-0.166481,-0.166481,68.0000,68.0000,1.000000,0.9287,-0.063025,0.009788,,0,0,0.671917,0.991764,0.457096,,,0.325240,,,1,0,0,0.21280,0.000000000e+00
43.3077,servo_twist,qpik,0.317985,-0.706504,1.133016,1.657717,0.318310,1.131130,0.265951,0.265951,-0.168569,-0.168569,68.0000,68.0000,1.000000,0.9403,-0.062186,0.009654,,0,0,0.679259,0.979662,0.462912,,,0.318537,,,1,0,0,0.21107,0.000000000e+00
43.4227,servo_twist,qpik,0.310745,-0.687180,1.141380,1.666007,0.311125,1.140154,0.255192,0.255192,-0.170820,-0.170820,68.0000,68.0000,1.000000,0.9610,-0.061289,0.009578,,0,0,0.687559,0.966777,0.469016,,,0.311694,,,1,0,0,0.20910,0.000000000e+00
43.5335,servo_twist,qpik,0.303877,-0.668294,1.149020,1.674366,0.304204,1.148671,0.244328,0.244328,-0.173015,-0.173015,68.0000,68.0000,1.000000,0.9604,-0.060343,0.008744,,0,0,0.696450,0.954844,0.474786,,,0.305162,,,1,0,0,0.20713,0.000000000e+00
43.6421,servo_twist,qpik,0.297281,-0.649313,1.157582,1.682172,0.297610,1.156438,0.234085,0.234085,-0.175181,-0.175181,68.0000,68.0000,1.000000,0.8729,-0.062938,0.008748,,0,0,0.703752,0.943813,0.480182,,,0.297605,,,1,0,0,0.20519,0.000000000e+00
43.7497,servo_twist,qpik,0.290643,-0.631127,1.171405,1.685785,0.290956,1.167992,0.223764,0.223764,-0.177312,-0.177312,68.0000,68.0000,1.000000,0.6978,-0.069832,0.009209,,0,0,0.696777,0.931898,0.483243,,,0.291040,,,1,0,0,0.20340,0.000000000e+00
43.8646,servo_twist,qpik,0.282916,-0.614081,1.186973,1.686940,0.283299,1.186126,0.211641,0.211641,-0.179557,-0.179557,68.0000,68.0000,1.000000,0.7278,-0.071441,0.008062,,0,0,0.687486,0.917175,0.484207,,,0.283925,,,1,0,0,0.20118,0.000000000e+00
43.9709,servo_twist,qpik,0.275607,-0.599178,1.200566,1.687520,0.275997,1.198675,0.200953,0.200953,-0.181690,-0.181690,68.0000,68.0000,1.000000,0.7352,-0.071440,0.008816,,0,0,0.683969,0.903440,0.483909,,,0.276511,,,1,0,0,0.19960,0.000000000e+00
44.0739,servo_twist,qpik,0.268197,-0.585653,1.211857,1.688357,0.268547,1.209513,0.190495,0.190495,-0.183777,-0.183777,68.0000,68.0000,1.000000,0.7609,-0.071114,0.008453,,0,0,0.682911,0.889732,0.482964,,,0.268951,,,1,0,0,0.19818,0.000000000e+00
44.1804,servo_twist,qpik,0.260721,-0.571576,1.223566,1.689188,0.261190,1.220544,0.179910,0.179910,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.6945,-0.070628,0.009264,,0,0,0.684787,0.876315,0.481929,,,0.261784,,,1,0,0,0.19679,0.000000000e+00
44.2801,servo_twist,qpik,0.253692,-0.557732,1.235256,1.689924,0.254069,1.232168,0.169658,0.169658,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7505,-0.070174,0.006793,,0,0,0.688394,0.863934,0.481313,,,0.254517,,,1,0,0,0.19523,0.000000000e+00
44.3842,servo_twist,qpik,0.246347,-0.543838,1.246397,1.690973,0.246725,1.242919,0.159191,0.159191,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7403,-0.069708,0.004852,,0,0,0.690290,0.851402,0.480706,,,0.247527,,,1,0,0,0.19381,0.000000000e+00
44.4899,servo_twist,qpik,0.238872,-0.530092,1.256937,1.692288,0.239165,1.254159,0.148240,0.148240,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7550,-0.069226,0.003923,,0,0,0.693831,0.839120,0.480308,,,0.240003,,,1,0,0,0.19215,0.000000000e+00
44.5894,servo_twist,qpik,0.231895,-0.516576,1.267402,1.693626,0.232252,1.263653,0.138181,0.138181,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7308,-0.068755,0.003980,,0,0,0.697768,0.827829,0.480314,,,0.233251,,,1,0,0,0.19082,0.000000000e+00
44.6902,servo_twist,qpik,0.225102,-0.502374,1.278713,1.695018,0.225483,1.274841,0.127935,0.127935,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.6822,-0.068263,0.004698,,0,0,0.704565,0.817360,0.481058,,,0.226239,,,1,0,0,0.18923,0.000000000e+00
44.7904,servo_twist,qpik,0.218255,-0.488222,1.289928,1.696567,0.218637,1.286360,0.117718,0.117718,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7138,-0.067471,0.003825,,0,0,0.709932,0.807119,0.482523,,,0.219609,,,1,0,0,0.18760,0.000000000e+00
44.8874,servo_twist,qpik,0.211729,-0.474438,1.300655,1.698306,0.212106,1.296343,0.107873,0.107873,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7108,-0.065702,0.003403,,0,0,0.715569,0.797886,0.484392,,,0.212275,,,1,0,0,0.18630,0.000000000e+00
44.9867,servo_twist,qpik,0.205169,-0.460377,1.311475,1.700278,0.205506,1.307199,0.097690,0.097690,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.6957,-0.063707,0.002652,,0,0,0.722741,0.788995,0.486879,,,0.206539,,,1,0,0,0.18471,0.000000000e+00
45.0907,servo_twist,qpik,0.198437,-0.445532,1.323035,1.702444,0.198756,1.318910,0.086969,0.086969,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.6763,-0.061337,0.001867,,0,0,0.732128,0.780342,0.490220,,,0.199687,,,1,0,0,0.18313,0.000000000e+00
45.1919,servo_twist,qpik,0.192141,-0.430230,1.335260,1.704677,0.192455,1.331250,0.076622,0.076622,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.6514,-0.058786,0.001233,,0,0,0.743685,0.772836,0.494522,,,0.193061,,,1,0,0,0.18137,0.000000000e+00
45.2929,servo_twist,qpik,0.186044,-0.414718,1.347775,1.707043,0.186348,1.343153,0.066514,0.066514,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.6311,-0.056087,0.000603,,0,0,0.754776,0.766117,0.499666,,,0.186623,,,1,0,0,0.17981,0.000000000e+00
45.3948,servo_twist,qpik,0.180183,-0.398018,1.361510,1.709621,0.180491,1.357465,0.055935,0.055935,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.5910,-0.052699,-0.000037,,0,0,0.769060,0.760419,0.506094,,,0.180481,,,1,0,0,0.17791,0.000000000e+00
45.4937,servo_twist,qpik,0.174762,-0.380912,1.375766,1.712333,0.175070,1.371392,0.045713,0.045713,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.5654,-0.049023,-0.001056,,0,0,0.785567,0.755921,0.513595,,,0.175695,,,1,0,0,0.17612,0.000000000e+00
45.5935,servo_twist,qpik,0.169657,-0.367119,1.390356,1.718888,0.169941,1.385774,0.035720,0.035720,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.5490,-0.045272,-0.001590,,0,0,0.826089,0.752683,0.536958,,,0.170135,,,1,0,0,0.17425,0.000000000e+00
45.6987,servo_twist,qpik,0.164705,-0.359414,1.407127,1.730383,0.164935,1.401622,0.025038,0.025038,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.4673,-0.042634,-0.002982,,0,0,0.909293,0.750681,0.587605,,,0.165055,,,1,0,0,0.17263,0.000000000e+00
45.7987,servo_twist,qpik,0.160281,-0.355335,1.429068,1.741375,0.160511,1.422234,0.014636,0.014636,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.4262,-0.038284,-0.004082,,0,0,0.983344,0.749373,0.654686,,,0.161129,,,1,0,0,0.17099,0.000000000e+00
45.8987,servo_twist,qpik,0.156258,-0.353113,1.453516,1.749622,0.156461,1.446529,0.004733,0.004733,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.3931,-0.033234,-0.004961,,0,0,1.023925,0.748028,0.720193,,,0.156645,,,1,0,0,0.16946,0.000000000e+00
45.9987,servo_twist,qpik,0.152675,-0.351887,1.481313,1.754467,0.152850,1.472691,-0.004653,-0.004653,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.3563,-0.027646,-0.005705,,0,0,1.066122,0.745874,0.782612,,,0.153471,,,1,0,0,0.16809,0.000000000e+00
46.0987,servo_twist,qpik,0.149578,-0.351243,1.512361,1.755316,0.149729,1.501228,-0.013742,-0.013742,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.3406,-0.022293,-0.006468,,0,0,1.105585,0.742521,0.840670,,,0.150009,,,1,0,0,0.16665,0.000000000e+00
46.1987,servo_twist,qpik,0.147051,-0.350932,1.548627,1.746219,0.147178,1.536675,-0.022800,-0.022800,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.3556,-0.015869,-0.007518,,0,0,1.127333,0.733847,0.881640,,,0.147247,,,1,0,0,0.16537,0.000000000e+00
46.3037,servo_twist,qpik,0.145031,-0.350756,1.590170,1.730477,0.145135,1.578912,-0.031805,-0.031805,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.3134,-0.009558,-0.008450,,0,0,1.132826,0.723455,0.914625,,,0.145151,,,1,0,0,0.16397,0.000000000e+00
46.4106,servo_twist,qpik,0.143653,-0.350659,1.632625,1.712779,0.143726,1.624709,-0.041201,-0.041201,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.2157,-0.007053,-0.005395,,0,0,1.133806,0.715695,0.944815,,,0.143724,,,1,0,0,0.16255,0.000000000e+00
46.5142,servo_twist,qpik,0.142734,-0.350568,1.674200,1.694523,0.142782,1.665725,-0.049518,-0.049518,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.1605,-0.008349,-0.004803,,0,0,1.138519,0.711309,0.971947,,,0.142771,,,1,0,0,0.16120,0.000000000e+00
46.6195,servo_twist,qpik,0.141847,-0.350460,1.714832,1.674663,0.141893,1.707578,-0.058611,-0.058611,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.1735,-0.014198,-0.009231,,0,0,1.143177,0.709492,0.994457,,,0.141937,,,1,0,0,0.15986,0.000000000e+00
46.7275,servo_twist,qpik,0.140550,-0.350361,1.752908,1.653968,0.140618,1.747371,-0.068190,-0.068190,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.2246,-0.020769,-0.011691,,0,0,1.146309,0.708764,1.011141,,,0.140725,,,1,0,0,0.15857,0.000000000e+00
46.8381,servo_twist,qpik,0.138614,-0.350252,1.787365,1.635458,0.138709,1.783395,-0.077906,-0.077906,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.2533,-0.027845,-0.012476,,0,0,1.153200,0.708402,1.027521,,,0.138943,,,1,0,0,0.15734,0.000000000e+00
46.9450,servo_twist,qpik,0.136019,-0.350158,1.815278,1.622450,0.136155,1.813327,-0.087884,-0.087884,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.3375,-0.034350,-0.013218,,0,0,1.166607,0.709187,1.047106,,,0.136529,,,1,0,0,0.15628,0.000000000e+00
47.0918,servo_twist,qpik,0.131448,-0.350088,1.835466,1.608381,0.131613,1.841200,-0.099833,-0.099833,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.4879,-0.040553,-0.012762,,0,0,1.165495,0.706934,1.049400,,,0.131937,,,1,0,0,0.15537,0.000000000e+00
47.2011,servo_twist,qpik,0.127315,-0.350051,1.850539,1.594344,0.127523,1.854569,-0.107108,-0.107108,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.5693,-0.043174,-0.010390,,0,0,1.159550,0.704730,1.040121,,,0.128002,,,1,0,0,0.15496,0.000000000e+00
47.3027,servo_twist,qpik,0.123157,-0.350043,1.863388,1.580337,0.123400,1.864954,-0.113524,-0.113524,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.6109,-0.045036,-0.008787,,0,0,1.157491,0.702680,1.026774,,,0.123639,,,1,0,0,0.15463,0.000000000e+00
47.4074,servo_twist,qpik,0.118600,-0.350037,1.875456,1.565777,0.118824,1.875688,-0.120357,-0.120357,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.6518,-0.046828,-0.008326,,0,0,1.157527,0.700796,1.010647,,,0.119081,,,1,0,0,0.15428,0.000000000e+00
47.5089,servo_twist,qpik,0.113982,-0.350036,1.886760,1.551009,0.114234,1.885584,-0.126922,-0.126922,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.6737,-0.048383,-0.007931,,0,0,1.158803,0.699568,0.992803,,,0.114311,,,1,0,0,0.15393,0.000000000e+00
47.6074,servo_twist,qpik,0.109312,-0.350035,1.897167,1.536513,0.109554,1.895061,-0.133568,-0.133568,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.6961,-0.049900,-0.008080,,0,0,1.161386,0.698980,0.974183,,,0.110280,,,1,0,0,0.15360,0.000000000e+00
47.7215,servo_twist,qpik,0.103730,-0.350033,1.907948,1.520598,0.104005,1.906179,-0.141298,-0.141298,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7088,-0.051538,-0.007976,,0,0,1.164908,0.698802,0.953012,,,0.104313,,,1,0,0,0.15318,0.000000000e+00
47.8257,servo_twist,qpik,0.098453,-0.350030,1.917518,1.505551,0.098736,1.915621,-0.148664,-0.148664,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7283,-0.052945,-0.007870,,0,0,1.168946,0.698740,0.931988,,,0.098951,,,1,0,0,0.15278,0.000000000e+00
47.9257,servo_twist,qpik,0.093346,-0.350022,1.926493,1.490773,0.093622,1.924487,-0.155735,-0.155735,-0.185000,-0.185000,68.0000,68.0000,1.000000,0.7381,-0.053724,-0.008187,,0,0,1.172748,0.696304,0.910927,,,0.094596,,,1,0,0,0.15239,0.000000000e+00
48.0245,servo_twist,qpik,0.088082,-0.350017,1.934894,1.476249,0.088382,1.932429,-0.162844,-0.162844,-0.185900,-0.185900,68.0000,68.0000,1.000000,0.7489,-0.053329,-0.007618,,0,0,1.177429,0.691794,0.889746,,,0.088903,,,1,0,0,0.15201,0.000000000e+00
48.1245,servo_twist,qpik,0.082755,-0.350011,1.942792,1.461985,0.083032,1.940981,-0.170139,-0.170139,-0.187920,-0.187920,68.0000,68.0000,1.000000,0.7525,-0.051796,-0.007011,,0,0,1.181482,0.687171,0.868563,,,0.083433,,,1,0,0,0.15160,0.000000000e+00
48.2265,servo_twist,qpik,0.077379,-0.350002,1.950615,1.447392,0.077667,1.948294,-0.177193,-0.177193,-0.189963,-0.189963,68.0000,68.0000,1.000000,0.7431,-0.049701,-0.006154,,0,0,1.184806,0.685542,0.846810,,,0.077754,,,1,0,0,0.15120,0.000000000e+00
48.3299,servo_twist,qpik,0.072180,-0.349998,1.958523,1.432308,0.072431,1.956601,-0.184353,-0.184353,-0.192032,-0.192032,68.0000,68.0000,1.000000,0.7316,-0.047507,-0.005870,,0,0,1.183663,0.690756,0.823692,,,0.072326,,,1,0,0,0.15074,0.000000000e+00
48.4299,servo_twist,qpik,0.067292,-0.349987,1.966254,1.417247,0.067534,1.964159,-0.191089,-0.191089,-0.194052,-0.194052,68.0000,68.0000,1.000000,0.7185,-0.045281,-0.005742,,0,0,1.181022,0.705006,0.800634,,,0.068155,,,1,0,0,0.15031,0.000000000e+00
48.5299,servo_twist,qpik,0.062660,-0.349983,1.973976,1.401933,0.062896,1.972239,-0.197661,-0.197661,-0.196056,-0.196056,68.0000,68.0000,1.000000,0.7026,-0.043174,-0.005810,,0,0,1.178267,0.729828,0.777050,,,0.063116,,,1,0,0,0.14983,0.000000000e+00
48.6299,servo_twist,qpik,0.058196,-0.349979,1.981760,1.386184,0.058419,1.980355,-0.204102,-0.204102,-0.198071,-0.198071,68.0000,68.0000,1.000000,0.6884,-0.041146,-0.005839,,0,0,1.174241,0.766235,0.752549,,,0.058366,,,1,0,0,0.14935,0.000000000e+00
48.7299,servo_twist,qpik,0.053942,-0.349973,1.989729,1.369695,0.054172,1.988122,-0.210331,-0.210331,-0.200138,-0.200138,68.0000,68.0000,1.000000,0.6731,-0.039200,-0.005826,,0,0,1.169885,0.813996,0.726735,,,0.054743,,,1,0,0,0.14885,0.000000000e+00
48.8299,servo_twist,qpik,0.049912,-0.349970,1.997564,1.353122,0.050110,1.996220,-0.216549,-0.216549,-0.202170,-0.202170,68.0000,68.0000,1.000000,0.6574,-0.037525,-0.005979,,0,0,1.165440,0.871317,0.700551,,,0.050285,,,1,0,0,0.14831,0.000000000e+00
48.9302,servo_twist,qpik,0.046041,-0.349970,2.005330,1.336266,0.046241,2.003586,-0.222547,-0.222547,-0.204199,-0.204199,68.0000,68.0000,1.000000,0.6410,-0.035943,-0.006073,,0,0,1.159827,0.937399,0.673712,,,0.046199,,,1,0,0,0.14780,0.000000000e+00
49.0299,servo_twist,qpik,0.042392,-0.349970,2.012988,1.319163,0.042576,2.011806,-0.228505,-0.228505,-0.206223,-0.206223,68.0000,68.0000,1.000000,0.6278,-0.034617,-0.006232,,0,0,1.152262,1.010098,0.646112,,,0.043090,,,1,0,0,0.14720,0.000000000e+00
49.1299,servo_twist,qpik,0.038831,-0.349969,2.020455,1.301941,0.039004,2.019381,-0.234370,-0.234370,-0.208235,-0.208235,68.0000,68.0000,1.000000,0.6171,-0.033386,-0.006343,,0,0,1.140486,1.089606,0.618004,,,0.039203,,,1,0,0,0.14664,0.000000000e+00
49.2299,servo_twist,qpik,0.035416,-0.349969,2.027893,1.284188,0.035602,2.026764,-0.239996,-0.239996,-0.210285,-0.210285,68.0000,68.0000,1.000000,0.6043,-0.032200,-0.006265,,0,0,1.127459,1.174129,0.588635,,,0.035596,,,1,0,0,0.14606,0.000000000e+00
49.3300,servo_twist,qpik,0.032152,-0.349969,2.035096,1.266348,0.032326,2.034129,-0.245646,-0.245646,-0.212319,-0.212319,68.0000,68.0000,1.000000,0.5954,-0.031213,-0.006430,,0,0,1.102855,1.261938,0.558655,,,0.032692,,,1,0,0,0.14546,0.000000000e+00
49.4361,servo_twist,qpik,0.028749,-0.349969,2.042518,1.247223,0.028937,2.041285,-0.251265,-0.251265,-0.214481,-0.214481,68.0000,68.0000,1.000000,0.5856,-0.030069,0.000000,,1,0,1.080415,1.360400,0.525895,,,0.029285,,,1,0,0,0.14483,0.000000000e+00
49.5352,servo_twist,qpik,0.025759,-0.349969,2.049198,1.229278,0.025926,2.048545,-0.256645,-0.256645,-0.216494,-0.216494,68.0000,68.0000,1.000000,0.5745,-0.029283,0.000000,,1,0,1.057541,1.452810,0.494487,,,0.026118,,,1,0,0,0.14421,0.000000000e+00
49.6353,servo_twist,qpik,0.022822,-0.349969,2.055760,1.210943,0.022983,2.055335,-0.261858,-0.261858,-0.218537,-0.218537,68.0000,68.0000,1.000000,0.5667,-0.028480,0.000000,,1,0,1.037133,1.548575,0.461705,,,0.023579,,,1,0,0,0.14359,0.000000000e+00
49.7352,servo_twist,qpik,0.019931,-0.349969,2.061942,1.192850,0.020075,2.061426,-0.267049,-0.267049,-0.220549,-0.220549,68.0000,68.0000,1.000000,0.5598,-0.027730,0.000000,,1,0,1.016310,1.647098,0.428544,,,0.020378,,,1,0,0,0.14302,0.000000000e+00
49.8352,servo_twist,qpik,0.017110,-0.349969,2.067720,1.176060,0.017268,2.067482,-0.272182,-0.272182,-0.222580,-0.222580,68.0000,68.0000,1.000000,0.5469,-0.027630,0.000000,,1,0,0.994285,1.747159,0.396798,,,0.017411,,,1,0,0,0.14244,0.000000000e+00
49.9358,servo_twist,qpik,0.014407,-0.349969,2.072492,1.165714,0.014560,2.072177,-0.276773,-0.276773,-0.224615,-0.224615,68.0000,68.0000,1.000000,0.5989,-0.027312,0.000000,,1,0,0.956291,1.845405,0.377733,,,0.014612,,,1,0,0,0.14203,0.000000000e+00
50.0352,servo_twist,qpik,0.011737,-0.349969,2.076222,1.162462,0.011890,2.075999,-0.280821,-0.280821,-0.226681,-0.226681,68.0000,68.0000,1.000000,0.6975,-0.025938,0.000000,,1,0,0.901249,1.945220,0.373781,,,0.012305,,,1,0,0,0.14180,0.000000000e+00
50.1352,servo_twist,qpik,0.009035,-0.349969,2.079625,1.160290,0.009158,2.079176,-0.284731,-0.284731,-0.228687,-0.228687,68.0000,68.0000,1.000000,0.5944,-0.025081,0.000000,,1,0,0.862744,2.049703,0.372870,,,0.009364,,,1,0,0,0.14166,0.000000000e+00
50.2352,servo_twist,qpik,0.007279,-0.349969,2.084589,1.150916,0.007379,2.083714,-0.288001,-0.288001,-0.230688,-0.230688,68.0000,68.0000,1.000000,0.3788,-0.024147,0.000714,,1,0,0.845512,2.124784,0.357505,,,0.007396,,,1,0,0,0.14136,0.000000000e+00
50.3352,servo_twist,qpik,0.006358,-0.349970,2.090785,1.136845,0.006413,2.089875,-0.291149,-0.291149,-0.232690,-0.232690,68.0000,68.0000,1.000000,0.2401,-0.023994,0.003493,,1,0,0.826367,2.170501,0.332050,,,0.006560,,,1,0,0,0.14084,0.000000000e+00
50.4352,servo_twist,qpik,0.005832,-0.349970,2.097413,1.121130,0.005872,2.096751,-0.294097,-0.294097,-0.234701,-0.234701,68.0000,68.0000,1.000000,0.1535,-0.024185,0.002724,,1,0,0.784650,2.201937,0.302580,,,0.005918,,,1,0,0,0.14023,0.000000000e+00
50.5426,servo_twist,qpik,0.005479,-0.349971,2.104432,1.104030,0.005501,2.103750,-0.297037,-0.297037,-0.236808,-0.236808,68.0000,68.0000,1.000000,0.0973,-0.024480,0.000727,,1,0,0.747781,2.227771,0.269439,,,0.005518,,,1,0,0,0.13960,0.000000000e+00
50.6403,servo_twist,qpik,0.005294,-0.349971,2.111011,1.088819,0.005313,2.110557,-0.299790,-0.299790,-0.238847,-0.238847,68.0000,68.0000,1.000000,0.0653,-0.024566,0.000000,,1,0,0.716082,2.246717,0.238945,,,0.005358,,,1,0,0,0.13896,0.000000000e+00
50.7403,servo_twist,qpik,0.005200,-0.349970,2.115575,1.091205,0.005207,2.115391,-0.301408,-0.301408,-0.240848,-0.240848,68.0000,68.0000,1.000000,0.0152,-0.011011,0.000000,,1,0,0.684610,2.258300,0.248875,,,0.005235,,,1,0,0,0.13863,0.000000000e+00
50.8465,servo_twist,qpik,0.005164,-0.349970,2.117436,1.115428,0.005170,2.118219,-0.301123,-0.301123,-0.242976,-0.242976,68.0000,68.0000,1.000000,0.0012,0.005018,0.000000,,1,0,0.822348,2.262150,0.309525,,,0.005173,,,1,0,0,0.13881,0.000000000e+00
50.9505,servo_twist,qpik,0.005345,-0.353768,2.113062,1.146906,0.005332,2.116648,-0.298168,-0.298168,-0.245059,-0.245059,68.0000,68.0000,1.000000,0.3256,0.006395,0.000525,,1,0,0.928651,2.247378,0.375780,,,0.005173,,,1,0,0,0.13946,0.000000000e+00
51.0463,servo_twist,qpik,0.005553,-0.365496,2.111010,1.163658,0.005427,2.112826,-0.295163,-0.295163,-0.247083,-0.247083,68.0000,68.0000,1.000000,0.0331,-0.002137,0.001384,,1,0,0.973877,2.232670,0.414243,,,0.005174,,,1,0,0,0.13991,0.000000000e+00
51.1462,servo_twist,qpik,0.005253,-0.381315,2.119860,1.168233,0.005228,2.115444,-0.293931,-0.293931,-0.249085,-0.249085,68.0000,68.0000,1.000000,0.0001,-0.007088,0.001054,,1,0,0.987808,2.255252,0.433570,,,0.005174,,,1,0,0,0.13970,0.000000000e+00
51.2461,servo_twist,qpik,0.005161,-0.394853,2.143110,1.162672,0.005169,2.133264,-0.292928,-0.292928,-0.251108,-0.251108,68.0000,68.0000,1.000000,0.0000,-0.007404,0.001700,,1,0,0.950943,2.300278,0.444159,,,0.005174,,,1,0,0,0.13817,0.000000000e+00
51.3461,servo_twist,qpik,0.005169,-0.404385,2.177395,1.149448,0.005174,2.164243,-0.292640,-0.292640,-0.253116,-0.253116,68.0000,68.0000,1.000000,0.0000,-0.007747,0.001412,,1,0,0.968866,2.375061,0.452261,,,0.005174,,,1,0,0,0.13518,0.000000000e+00
51.4461,servo_twist,qpik,0.005170,-0.409934,2.223683,1.127543,0.005174,2.210006,-0.293490,-0.293490,-0.255130,-0.255130,68.0000,68.0000,1.000000,0.0000,-0.008517,0.000473,,1,0,0.990777,2.505411,0.456531,,,0.005174,,,1,0,0,0.13016,0.000000000e+00
51.5464,servo_twist,qpik,0.005170,-0.418464,2.252064,1.115639,0.005174,2.245890,-0.293972,-0.293972,-0.255538,-0.255538,68.0000,68.0000,1.000000,0.0000,-0.009228,0.000000,,1,0,0.978398,2.605714,0.455984,,,0.005174,,,1,0,0,0.12576,0.000000000e+00
51.6461,servo_twist,qpik,0.005170,-0.426773,2.262204,1.115006,0.005174,2.259974,-0.292357,-0.292357,-0.255538,-0.255538,68.0000,68.0000,1.000000,0.0000,-0.009202,0.000000,,1,0,0.909275,2.644259,0.453418,,,0.005174,,,1,0,0,0.12384,0.000000000e+00
51.7461,servo_twist,qpik,0.005170,-0.435498,2.266174,1.118442,0.005174,2.265368,-0.289932,-0.289932,-0.255538,-0.255538,68.0000,68.0000,1.000000,0.0000,-0.008872,0.001112,,1,0,0.831569,2.658800,0.450654,,,0.005174,,,1,0,0,0.12299,0.000000000e+00
51.8465,servo_twist,qpik,0.005169,-0.444387,2.267715,1.123403,0.005174,2.267532,-0.287352,-0.287352,-0.255538,-0.255538,68.0000,68.0000,1.000000,0.0000,-0.008418,0.001870,,1,0,0.771412,2.663526,0.447501,,,0.005174,,,1,0,0,0.12260,0.000000000e+00
51.9461,servo_twist,qpik,0.005170,-0.452955,2.268239,1.128837,0.005174,2.268265,-0.284524,-0.284524,-0.255538,-0.255538,68.0000,68.0000,1.000000,0.0000,-0.007843,0.002536,,1,0,0.744923,2.664103,0.443770,,,0.005174,,,1,0,0,0.12242,0.000000000e+00
52.0461,servo_twist,qpik,0.005170,-0.461153,2.268406,1.134352,0.005174,2.268631,-0.281405,-0.281405,-0.256542,-0.256542,68.0000,68.0000,1.000000,0.0000,-0.006888,0.004287,,1,0,0.727998,2.663216,0.439242,,,0.005174,,,1,0,0,0.12231,0.000000000e+00
52.1461,servo_twist,qpik,0.005170,-0.468925,2.268472,1.139766,0.005174,2.268684,-0.278313,-0.278313,-0.258554,-0.258554,68.0000,68.0000,1.000000,0.0000,-0.005693,0.005991,,1,0,0.719976,2.661886,0.433976,,,0.005174,,,1,0,0,0.12226,0.000000000e+00
52.2461,servo_twist,qpik,0.005170,-0.476207,2.268530,1.144960,0.005174,2.268631,-0.274782,-0.274782,-0.260560,-0.260560,68.0000,68.0000,1.000000,0.0000,-0.004433,0.007412,,1,0,0.714475,2.660519,0.427834,,,0.005174,,,1,0,0,0.12225,0.000000000e+00
52.3461,servo_twist,qpik,0.005170,-0.483003,2.268592,1.149911,0.005174,2.268684,-0.271712,-0.271712,-0.262563,-0.262563,68.0000,68.0000,1.000000,0.0000,-0.003363,0.007982,,1,0,0.710772,2.659202,0.420953,,,0.005174,,,1,0,0,0.12226,0.000000000e+00
52.4461,servo_twist,qpik,0.005170,-0.489317,2.268666,1.154607,0.005174,2.268579,-0.268245,-0.268245,-0.264564,-0.264564,68.0000,68.0000,1.000000,0.0000,-0.002309,0.008844,,1,0,0.707775,2.657924,0.413362,,,0.005174,,,1,0,0,0.12229,0.000000000e+00
52.5467,servo_twist,qpik,0.005169,-0.495249,2.268736,1.159091,0.005174,2.268701,-0.264809,-0.264809,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.001672,0.007563,,1,0,0.704587,2.656664,0.404807,,,0.005174,,,1,0,0,0.12232,0.000000000e+00
52.6461,servo_twist,qpik,0.005170,-0.500740,2.268809,1.163311,0.005174,2.268684,-0.261297,-0.261297,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.001699,0.003879,,1,0,0.702491,2.655393,0.395200,,,0.005174,,,1,0,0,0.12242,0.000000000e+00
52.7461,servo_twist,qpik,0.005170,-0.505648,2.268820,1.167183,0.005174,2.268771,-0.257742,-0.257742,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.001719,0.002229,,1,0,0.699829,2.653946,0.385158,,,0.005174,,,1,0,0,0.12251,0.000000000e+00
52.8495,servo_twist,qpik,0.005170,-0.510229,2.268820,1.170887,0.005174,2.268823,-0.253983,-0.253983,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.001945,0.002805,,1,0,0.698110,2.652417,0.373922,,,0.005174,,,1,0,0,0.12264,0.000000000e+00
52.9463,servo_twist,qpik,0.005170,-0.514168,2.268820,1.174144,0.005174,2.268910,-0.250495,-0.250495,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.002620,0.004977,,1,0,0.696573,2.650969,0.362552,,,0.005174,,,1,0,0,0.12281,0.000000000e+00
53.0461,servo_twist,qpik,0.005170,-0.517528,2.268820,1.177018,0.005174,2.268823,-0.246695,-0.246695,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.003445,0.006702,,1,0,0.695990,2.649571,0.350492,,,0.005174,,,1,0,0,0.12304,0.000000000e+00
53.1463,servo_twist,qpik,0.005170,-0.520382,2.268819,1.179549,0.005174,2.268893,-0.243204,-0.243204,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.004264,0.007546,,1,0,0.696079,2.648217,0.338148,,,0.005174,,,1,0,0,0.12326,0.000000000e+00
53.2497,servo_twist,qpik,0.005169,-0.522766,2.268819,1.181786,0.005174,2.268806,-0.239517,-0.239517,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.005170,0.008202,,1,0,0.697174,2.646869,0.324841,,,0.005174,,,1,0,0,0.12357,0.000000000e+00
53.3477,servo_twist,qpik,0.005170,-0.524471,2.268819,1.183524,0.005174,2.268754,-0.235848,-0.235848,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.006092,0.008733,,1,0,0.699063,2.645607,0.311618,,,0.005174,,,1,0,0,0.12387,0.000000000e+00
53.4541,servo_twist,qpik,0.005170,-0.525533,2.268823,1.184886,0.005174,2.268719,-0.232223,-0.232223,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.006999,0.008929,,1,0,0.702556,2.644398,0.297675,,,0.005174,,,1,0,0,0.12425,0.000000000e+00
53.5534,servo_twist,qpik,0.005170,-0.525856,2.268847,1.185708,0.005174,2.268649,-0.228879,-0.228879,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.007800,0.009110,,1,0,0.707368,2.643412,0.284430,,,0.005174,,,1,0,0,0.12463,0.000000000e+00
53.6511,servo_twist,qpik,0.005170,-0.525496,2.268879,1.186062,0.005174,2.268684,-0.225724,-0.225724,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.008473,0.009290,,1,0,0.713245,2.642586,0.271632,,,0.005174,,,1,0,0,0.12499,0.000000000e+00
53.7545,servo_twist,qpik,0.005170,-0.524344,2.268880,1.185965,0.005174,2.268806,-0.222827,-0.222827,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.009002,0.009230,,1,0,0.720724,2.641720,0.258334,,,0.005174,,,1,0,0,0.12543,0.000000000e+00
53.8564,servo_twist,qpik,0.005170,-0.522358,2.268879,1.185360,0.005174,2.268893,-0.219863,-0.219863,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.009422,0.009412,,1,0,0.730321,2.640976,0.245677,,,0.005174,,,1,0,0,0.12579,0.000000000e+00
53.9627,servo_twist,qpik,0.005170,-0.519352,2.268885,1.184202,0.005174,2.268596,-0.217103,-0.217103,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.009678,0.009471,,1,0,0.741949,2.640428,0.233229,,,0.005174,,,1,0,0,0.12618,0.000000000e+00
54.0626,servo_twist,qpik,0.005170,-0.515769,2.268893,1.182123,0.005174,2.268771,-0.214832,-0.214832,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.005514,0.009287,,1,0,0.755509,2.640341,0.221967,,,0.005174,,,1,0,0,0.12635,0.000000000e+00
54.1632,servo_twist,qpik,0.005170,-0.512904,2.268892,1.176458,0.005174,2.268841,-0.213240,-0.213240,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,-0.000152,0.008668,,1,0,0.752988,2.640437,0.208107,,,0.005174,,,1,0,0,0.12629,0.000000000e+00
54.2678,servo_twist,qpik,0.005174,-0.511107,2.268743,1.167173,0.005174,2.268823,-0.212113,-0.212113,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,0.000052,0.007112,,1,0,0.731284,2.639887,0.185897,,,0.005174,,,1,0,0,0.12596,0.000000000e+00
54.3756,servo_twist,qpik,0.005185,-0.509205,2.264606,1.160848,0.005174,2.267811,-0.211661,-0.211661,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,0.004885,0.005845,,1,0,0.742005,2.625516,0.166210,,,0.005174,,,1,0,0,0.12574,0.000000000e+00
54.4857,servo_twist,qpik,0.005469,-0.513226,2.245204,1.160722,0.005436,2.252278,-0.210090,-0.210090,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,0.010370,0.006252,,1,0,0.784789,2.546282,0.149692,,,0.005176,,,1,0,0,0.12751,0.000000000e+00
54.5864,servo_twist,qpik,0.005880,-0.517020,2.221592,1.165634,0.005785,2.229152,-0.208099,-0.208099,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,0.010928,0.006580,,1,0,0.826006,2.450124,0.143062,,,0.005513,,,1,0,0,0.13038,0.000000000e+00
54.6904,servo_twist,qpik,0.006615,-0.520591,2.195487,1.173740,0.006529,2.204054,-0.205897,-0.205897,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0000,0.011400,0.006746,,1,0,0.835622,2.345629,0.143198,,,0.006443,,,1,0,0,0.13356,0.000000000e+00
54.8054,servo_twist,qpik,0.007821,-0.524414,2.165397,1.185376,0.007772,2.172952,-0.203125,-0.203125,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.0024,0.011747,0.006696,,1,0,0.842041,2.221296,0.149411,,,0.007645,,,1,0,0,0.13753,0.000000000e+00
54.9092,servo_twist,qpik,0.009127,-0.529674,2.134358,1.198249,0.009057,2.143317,-0.198897,-0.198897,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.2003,0.021154,0.008325,,1,0,0.811016,2.102270,0.156564,,,0.008892,,,1,0,0,0.14125,0.000000000e+00
55.0133,servo_twist,qpik,0.011003,-0.540171,2.093250,1.215418,0.010906,2.106141,-0.189755,-0.189755,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.1632,0.031506,0.012788,,1,0,0.743276,1.955075,0.164420,,,0.010588,,,1,0,0,0.14654,0.000000000e+00
55.1117,servo_twist,qpik,0.013619,-0.552015,2.051737,1.228139,0.013481,2.065911,-0.177344,-0.177344,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.2083,0.035880,0.017423,,1,0,0.658610,1.796499,0.167875,,,0.013343,,,1,0,0,0.15235,0.000000000e+00
55.2113,servo_twist,qpik,0.016848,-0.562499,2.009017,1.239719,0.016675,2.025472,-0.163719,-0.163719,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.2357,0.040495,0.020432,,1,0,0.528996,1.630526,0.173080,,,0.016100,,,1,0,0,0.15799,0.000000000e+00
55.3173,servo_twist,qpik,0.020854,-0.572221,1.963153,1.250989,0.020659,1.977563,-0.147201,-0.147201,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.2546,0.040445,0.023732,,1,0,0.435888,1.452148,0.179979,,,0.020091,,,1,0,0,0.16448,0.000000000e+00
55.4192,servo_twist,qpik,0.024951,-0.580914,1.917497,1.261492,0.024739,1.933581,-0.131782,-0.131782,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.2603,0.040675,0.025011,,1,0,0.386595,1.288852,0.187605,,,0.024273,,,1,0,0,0.17024,0.000000000e+00
55.5299,servo_twist,qpik,0.029458,-0.589974,1.867992,1.272630,0.029230,1.880540,-0.113804,-0.113804,-0.265364,-0.265364,68.0000,68.0000,1.000000,0.2971,0.041512,0.027681,,1,0,0.379526,1.129823,0.195591,,,0.029169,,,1,0,1,0.17697,0.000000000e+00
55.6386,servo_twist,qpik,0.033963,-0.601599,1.828821,1.287623,0.033736,1.836278,-0.100053,-0.100053,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.4085,0.043199,0.025474,,0,0,0.495802,0.949153,0.196908,,,0.033685,,,1,0,0,0.18282,0.000000000e+00
55.7474,servo_twist,qpik,0.038576,-0.608903,1.799466,1.295827,0.038345,1.804810,-0.089088,-0.089088,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.4405,0.045466,0.023036,,0,0,0.522802,0.824994,0.191082,,,0.038291,,,1,0,0,0.18700,0.000000000e+00
55.8532,servo_twist,qpik,0.043319,-0.614819,1.772151,1.301125,0.043083,1.776222,-0.078648,-0.078648,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.4659,0.048120,0.022165,,0,0,0.530217,0.708637,0.179063,,,0.042998,,,1,0,0,0.19065,0.000000000e+00
55.9721,servo_twist,qpik,0.048893,-0.620165,1.743242,1.304974,0.048641,1.744561,-0.066538,-0.066538,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.4930,0.051553,0.022328,,0,0,0.526784,0.588781,0.160533,,,0.047938,,,1,0,0,0.19448,0.000000000e+00
56.0824,servo_twist,qpik,0.054414,-0.625135,1.716851,1.306398,0.054164,1.719568,-0.055997,-0.055997,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.5171,0.054673,0.021945,,0,0,0.518452,0.495178,0.138636,,,0.054198,,,1,0,0,0.19739,0.000000000e+00
56.1869,servo_twist,qpik,0.059972,-0.630434,1.692891,1.305492,0.059689,1.695465,-0.045849,-0.045849,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.5425,0.057768,0.021687,,0,0,0.508788,0.426434,0.113251,,,0.058807,,,1,0,0,0.20014,0.000000000e+00
56.2919,servo_twist,qpik,0.065849,-0.637840,1.670240,1.301486,0.065549,1.674731,-0.036128,-0.036128,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.5643,0.060540,0.021221,,0,0,0.501347,0.386750,0.084124,,,0.064668,,,1,0,0,0.20253,0.000000000e+00
56.3976,servo_twist,qpik,0.072057,-0.648415,1.649845,1.293301,0.071747,1.654171,-0.025299,-0.025299,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.6293,0.061669,0.021840,,0,0,0.491929,0.377390,0.051062,,,0.071116,,,1,0,0,0.20481,0.000000000e+00
56.5004,servo_twist,qpik,0.078293,-0.660722,1.633603,1.281410,0.077972,1.637346,-0.015039,-0.015039,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.7461,0.060143,0.021812,,0,0,0.478539,0.390971,0.015761,,,0.077196,,,1,0,0,0.20675,0.000000000e+00
56.5997,servo_twist,qpik,0.084309,-0.675072,1.621549,1.265850,0.084011,1.624954,-0.005520,-0.005520,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.9704,0.056359,0.021959,,0,0,0.466051,0.414261,0.021025,,,0.083454,,,1,0,0,0.20817,0.000000000e+00
56.6985,servo_twist,qpik,0.090002,-0.692164,1.615185,1.244549,0.089711,1.616873,0.003053,0.003053,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.4238,0.039914,0.021598,,0,0,0.456934,0.436832,0.062098,,,0.089770,,,1,0,0,0.20912,0.000000000e+00
56.8039,servo_twist,qpik,0.094859,-0.710782,1.619325,1.212384,0.094621,1.616210,0.009866,0.009866,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.0671,0.008982,0.020608,,0,0,0.441939,0.450477,0.119334,,,0.094042,,,1,0,0,0.20911,0.000000000e+00
56.9037,servo_twist,qpik,0.097234,-0.727542,1.631703,1.177343,0.097110,1.627729,0.011661,0.011661,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.0019,0.000155,0.018128,,0,0,0.440612,0.452193,0.177788,,,0.097059,,,1,0,0,0.20750,0.000000000e+00
57.0059,servo_twist,qpik,0.098099,-0.739739,1.641333,1.148619,0.098062,1.637974,0.012023,0.012023,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.0006,0.000143,0.017383,,0,0,0.457669,0.450937,0.229050,,,0.098091,,,1,0,0,0.20589,0.000000000e+00
57.1111,servo_twist,qpik,0.098368,-0.739751,1.649312,1.131335,0.098357,1.648167,0.011262,0.011262,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.0453,0.000006,0.016346,,0,0,0.459277,0.448503,0.269314,,,0.098360,,,1,0,0,0.20412,0.000000000e+00
57.2090,servo_twist,qpik,0.098409,-0.737452,1.650593,1.124697,0.098402,1.650994,0.010888,0.010888,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.1170,0.000001,0.016484,,0,0,0.475402,0.447664,0.292107,,,0.098396,,,1,0,0,0.20349,0.000000000e+00
57.3083,servo_twist,qpik,0.098397,-0.736181,1.650631,1.121257,0.098397,1.651029,0.010937,0.010937,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.0017,0.000000,0.009390,,0,0,0.071318,0.447284,0.305496,,,0.098397,,,1,0,1,0.20330,0.000000000e+00
57.4082,servo_twist,qpik,0.098397,-0.735221,1.650633,1.118646,0.098397,1.651081,0.010988,0.010988,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.0859,0.000000,0.004092,,0,0,0.002391,0.447037,0.315616,,,0.098397,,,1,0,1,0.20316,0.000000000e+00
57.5166,servo_twist,qpik,0.098397,-0.734347,1.650637,1.116250,0.098397,1.650837,0.011025,0.011025,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.3989,0.000000,0.001717,,0,0,0.000066,0.446824,0.324856,,,0.098397,,,1,0,1,0.20306,0.000000000e+00
57.6151,servo_twist,qpik,0.098397,-0.733653,1.650639,1.114337,0.098398,1.650680,0.011049,0.011049,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.4899,0.000000,0.000772,,0,0,0.000002,0.446657,0.332387,,,0.098398,,,1,0,1,0.20299,0.000000000e+00
57.7158,servo_twist,qpik,0.098398,-0.733043,1.650641,1.112648,0.098398,1.650628,0.011022,0.011022,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.4998,0.000000,0.000307,,0,0,0.000000,0.446517,0.339054,,,0.098398,,,1,0,1,0.20291,0.000000000e+00
57.8150,servo_twist,qpik,0.098398,-0.732523,1.650641,1.111205,0.098398,1.650628,0.011071,0.011071,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.5003,0.000000,0.000163,,0,0,0.000000,0.446400,0.344708,,,0.098398,,,1,0,1,0.20283,0.000000000e+00
57.9142,servo_twist,qpik,0.098398,-0.732076,1.650641,1.109963,0.098398,1.650645,0.011023,0.011023,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.5000,0.000000,0.000039,,0,0,0.000000,0.446303,0.349651,,,0.098398,,,1,0,1,0.20278,0.000000000e+00
58.0143,servo_twist,qpik,0.098398,-0.731692,1.650641,1.108889,0.098398,1.650645,0.011073,0.011073,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.5002,0.000000,0.000052,,0,0,0.000000,0.446221,0.353900,,,0.098398,,,1,0,1,0.20270,0.000000000e+00
58.1142,servo_twist,qpik,0.098398,-0.731359,1.650642,1.107959,0.098398,1.650575,0.011105,0.011105,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.5011,0.000000,0.000040,,0,0,0.000000,0.446151,0.357618,,,0.098398,,,1,0,1,0.20267,0.000000000e+00
58.2145,servo_twist,qpik,0.098398,-0.731066,1.650644,1.107134,0.098398,1.650628,0.011145,0.011145,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.5009,0.000000,0.000056,,0,0,0.000000,0.446091,0.360904,,,0.098398,,,1,0,1,0.20261,0.000000000e+00
58.3142,servo_twist,qpik,0.098398,-0.730818,1.650644,1.106437,0.098398,1.650610,0.011234,0.011234,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.5001,0.000000,0.000104,,0,0,0.000000,0.446041,0.363701,,,0.098398,,,1,0,1,0.20255,0.000000000e+00
58.4143,servo_twist,qpik,0.098398,-0.730601,1.650644,1.105826,0.098398,1.650575,0.011259,0.011259,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.0958,0.000000,0.000072,,0,0,0.000000,0.445998,0.366152,,,0.098398,,,1,0,1,0.20253,0.000000000e+00
58.5215,servo_twist,qpik,0.098398,-0.730402,1.650648,1.105257,0.098398,1.650645,0.011304,0.011304,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.3347,0.000000,0.000071,,0,0,0.000000,0.445958,0.368444,,,0.098398,,,1,0,1,0.20248,0.000000000e+00
58.6316,servo_twist,qpik,0.098398,-0.730224,1.650648,1.104755,0.098398,1.650610,0.011362,0.011362,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.5002,0.000000,0.000074,,0,0,0.000000,0.445924,0.370464,,,0.098398,,,1,0,1,0.20244,0.000000000e+00
58.7366,servo_twist,qpik,0.098398,-0.730076,1.650648,1.104336,0.098398,1.650610,0.011387,0.011387,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.4999,0.000000,0.000050,,0,0,0.000000,0.445895,0.372177,,,0.098398,,,1,0,1,0.20241,0.000000000e+00
58.8360,servo_twist,qpik,0.098399,-0.729953,1.650649,1.103988,0.098398,1.650610,0.011379,0.011379,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.9587,0.000000,0.000025,,0,0,0.000000,0.445872,0.373562,,,0.098398,,,1,0,1,0.20240,0.000000000e+00
58.9376,servo_twist,qpik,0.098398,-0.729841,1.650648,1.103673,0.098398,1.650628,0.011393,0.011393,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.5005,0.000000,0.000021,,0,0,0.000000,0.445851,0.374835,,,0.098398,,,1,0,1,0.20237,0.000000000e+00
59.0360,servo_twist,qpik,0.098399,-0.729748,1.650649,1.103410,0.098399,1.650628,0.011365,0.011365,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.5674,0.000000,-0.000016,,0,0,0.000000,0.445834,0.375899,,,0.098399,,,1,0,1,0.20237,0.000000000e+00
59.1363,servo_twist,qpik,0.098399,-0.729667,1.650649,1.103179,0.098399,1.650628,0.011382,0.011382,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.3182,0.000000,0.000008,,0,0,0.000000,0.445818,0.376823,,,0.098399,,,1,0,1,0.20235,0.000000000e+00
59.2360,servo_twist,qpik,0.098399,-0.729596,1.650649,1.102979,0.098399,1.650680,0.011346,0.011346,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.7216,0.000000,-0.000029,,0,0,0.000000,0.445805,0.377627,,,0.098398,,,1,0,1,0.20234,0.000000000e+00
59.3360,servo_twist,qpik,0.098399,-0.729535,1.650649,1.102805,0.098399,1.650645,0.011359,0.011359,-0.265364,-0.265364,68.0000,68.0000,0.000000,0.2739,0.000000,-0.000006,,0,0,0.000000,0.445794,0.378323,,,0.098399,,,1,0,1,0.20234,0.000000000e+00
0.0464,servo_twist,qpik,0.098399,-0.729443,1.650628,1.102054,0.098399,1.650628,0.011396,0.011396,-0.087031,-0.087031,76.2796,68.0000,0.123740,0.4992,0.000000,0.000000,,0,0,0.004455,0.011333,0.047089,,,0.098399,,,1,0,0,0.20231,0.000000000e+00
0.1449,servo_twist,qpik,0.098399,-0.729473,1.650628,1.102139,0.098399,1.650418,0.011694,0.011694,-0.087058,-0.087058,73.7773,68.0000,0.388564,0.5053,0.000000,0.000000,,0,0,0.092596,0.034239,0.147683,,,0.098399,,,1,0,0,0.20219,0.000000000e+00
0.2449,servo_twist,qpik,0.098399,-0.729597,1.650630,1.102487,0.098399,1.650436,0.011626,0.011626,-0.087058,-0.087058,71.2346,68.0000,0.388564,0.0190,0.000000,0.000000,,0,0,0.004781,0.033928,0.248808,,,0.098399,,,1,0,1,0.20223,0.000000000e+00
0.3462,servo_twist,qpik,0.098399,-0.729723,1.650630,1.102846,0.098399,1.650610,0.011556,0.011556,-0.087058,-0.087058,68.6467,68.0000,0.388564,0.0745,0.000000,0.000000,,0,0,0.000157,0.033604,0.351619,,,0.098399,,,1,0,1,0.20225,0.000000000e+00
0.4449,servo_twist,qpik,0.098399,-0.729832,1.650630,1.103157,0.098399,1.650628,0.011486,0.011486,-0.087058,-0.087058,68.0000,68.0000,0.388564,0.2791,0.000000,0.000000,,0,0,0.000005,0.033323,0.376356,,,0.098399,,,1,0,1,0.20229,0.000000000e+00
0.5476,servo_twist,qpik,0.098399,-0.729929,1.650630,1.103432,0.098399,1.650645,0.011417,0.011417,-0.087058,-0.087058,68.0000,68.0000,0.388564,0.2534,0.000000,0.000000,,0,0,0.000000,0.033074,0.375195,,,0.098399,,,1,0,1,0.20233,0.000000000e+00
0.6449,servo_twist,qpik,0.098399,-0.730011,1.650630,1.103663,0.098399,1.650575,0.011476,0.011476,-0.087058,-0.087058,68.0000,68.0000,0.388564,0.1802,0.000000,0.000000,,0,0,0.000000,0.032866,0.374239,,,0.098399,,,1,0,1,0.20234,0.000000000e+00
```
