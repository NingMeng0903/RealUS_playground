# 力控制律双版本审查稿

生成方式：机械摘录 git `e85c9ab957c78d500c3746159f1f47f4cabe3f0a`（A 版）与当前工作树（B 版），避免手抄走样。  
用途：第三方审查后决定 peirastic `track_hybrid` 用哪一套核。**本文不改控制代码。**

- A 版：`e85c9ab957c78d500c3746159f1f47f4cabe3f0a`（2026-08-04，「力控制器速度提升，调节参数，没有做SIN严重补偿」）
- B 版：当前 HEAD / 工作树，在 A 上叠加 surface force modulation、force-point 推进、Lee 结构双向能量流（yaml `bidirectional_flow.mode: observe`）

peirastic 力控接口先包住现有 `AdmittanceController`。A/B 都走同一入口，B 的额外层由配置门控。审查结论下来后只换 `ForceLaw` 实现或 yaml，不改六个模式。

---

## 0. 共同结构（两版都有）

笛卡尔力位混合在 **tool 系** 分解：

- 位置轴（默认 tool XY + 姿态）：`v_pos = v_ff + K_p e`，Z 向位置误差被清零，不和力轴抢。
- 力轴（tool Z）：`e_f = f_des_z - f_ext_z`（经 deadband）→ 导纳 / 主动前馈 `v_r` → DOB `u_dob` → 变阻尼 `ΔD_hf`。
- 合成：`v* = TFF(S) · v_pos + TFF(I-S) · v_force`，再交给内环纯速度 QPIK。

A 版把力轴做成「更快的欠力追逐 + 过力抽回」，并用 tool-XY 速度在折返处软化追逐，避免横向一顿。

B 版保留这条通路，再加：

1. `surface_force_modulation`：按表面法向/接触调制力参考。
2. force-point 推进：力参考点随运动走，而不是钉在初始 TCP。
3. `bidirectional_flow`（Lee 2024 proxy/real-port 工程化）：proxy 可双向，real 辅助路只许抽回；能量罐 `T0` 只给 press 开门；当前 yaml 是 `mode: observe`（算状态、不改速度）。

---

## 1. A 版控制律（e85c9ab）

### 1.1 力误差与死区

`e_f` 先过 `smooth_deadband`（`deadband_n=0.08`，`deadband_width_n=0.10`）。单 tick 接触跌落不会整段 sticky。

### 1.2 主动前馈 `v_r`（`ProactiveForceIntegrator`）

- 欠力（press）：`step = gain * drive * chase_scale`，`gain=0.24`
- 过力（retract）：`step = retract_gain * drive`，`gain=0.30`，不门控
- `gate_press_on_is=false`：press 不硬关；改用 `press_is_soft_floor=0.45` / `press_is_soft_stop=0.85` 对 `I_s` 软衰减
- 上升 slew：`press_slew_max_m_s2=0.35`（只限 press）
- 横向软化：`chase_scale = floor + (1-floor)*smoothstep(v_xy)`，仅在持续扫查后武装，纯力保持保持满增益

### 1.3 DOB

`u_dob += dt * ki * ki_scale * e_f`。过力 `ki_scale=1`；欠力乘 `chase_scale`。`I_s` 高则 freeze。

### 1.4 变阻尼 `ΔD_hf`

高频力误差进入 hold 后加 `ΔD`。手松 / `|e_f|` 大时用 `var_damping_hf_release_fast_s=0.04` 快卸，避免抽回发黏。

### 1.5 接触与再接触

DETACHED → RECONTACT 后 `recontact_vz_cap_m_s` 限压入速度。接触状态机见 `contact_state.py`。

### 1.6 A 版 yaml（`hybrid_motion`）

```yaml
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
    enter_confirm_s: 0.01
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
  # Low baseline MD for light feel + fast under/over-force chase.
  # Chatter: short-lived ΔD_hf(Is). Steady offset: force_dob. Not sticky Ke·D.
  admittance_mass_z: 1.0
  admittance_damping_z: 25.0
  max_vz_tool_m_s: 0.08
  desired_force_ramp_s: 0.8
  var_damping_enabled: true
  var_damping_omega_c_hz: 2.5
  var_damping_lambda: 0.951
  var_damping_f_max_n: 7.0
  # ΔD_hf amplitude (primary chatter dissipation); M bump is secondary.
  var_damping_d_u: 90.0
  var_damping_m_u: 1.5
  var_damping_m_max: 3.0
  var_damping_dc_alpha: 0.02
  var_damping_hf_attack_s: 0.02
  var_damping_hf_hold_s: 0.18
  var_damping_hf_release_s: 0.12
  var_damping_hf_release_fast_s: 0.04  # dump ΔD on hand-release / large |e_f|
  var_damping_hf_on: 0.30
  var_damping_hf_off: 0.15
  var_damping_hf_err_n: 0.8
  recontact_vz_cap_m_s: 0.008
  recontact_hold_s: 0.22
  force_dob:
    enabled: true
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
    # Observe Ke / impact burst only — do not hold high critical D in steady contact.
    drive_damping: false
    zeta: 0.9
    ke_initial: 80.0
    ke_min: 40.0
    ke_max: 2500.0
    ke_impact_initial: 1500.0
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
  proactive_feedforward: true
  proactive_retract_only: false
  # Faster under-force chase; stronger over-force escape; no Is gate on press.
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
```

---

## 2. B 版控制律（当前，双向能量流）

A 的死区 / `v_r` / DOB / `ΔD_hf` / lateral chase **都还在**。B 多三层：

### 2.1 Surface force modulation

按接触与表面估计缩放 `f_des`。开关在 yaml `surface_force_modulation`。

### 2.2 Force-point 推进

`_advance_force_point` / `_motion_twist_to_force_point`：力参考点随 `v*` 走，避免扫查时 Z 力环把 TCP 往旧点拽。

### 2.3 双向能量流（`bidirectional_flow.py`）

Lee et al. 2024 的 proxy/real-port **速度级**工程化，不是力矩定理：

- proxy 可双向；real 辅助路只加 retract
- 能量门只卡 press；retract 在门关时仍可通过
- 反馈过期或符号未验证 → press 门关
- `mode: observe`：算罐、α、辅助速度，**返回未调制的 proxy 速度**
- `mode: active`：才真正改 press/retract
- `mode: off`：整层旁路

当前 yaml：`mode: observe`，`sign_verified: false`，`feedback_delay_verified: false`，两道 `require_*` 为 true。审查未签核前不应改成 `active`。

### 2.4 B 版 yaml（`hybrid_motion`）

```yaml
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
  system_delay_s: 0.015
  contact_threshold_n: 0.8
  contact_use_fz_only: true
  physical_contact:
    enabled: true
    # Initial acquire uses filtered force only.  Replaying 162413 with
    # 0.85 N / 20 ms moves the false 3.49 s acquire to the stable load at
    # 4.14 s, while remaining reachable below the shipped 1 N target.
    enter_n: 0.85
    hard_enter_n: 1.5
    # The same log shows a ~0.65 N airborne residual.  Exit/rearm thresholds
    # therefore straddle that measured baseline instead of assuming <0.15 N.
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
  # Low baseline MD for light feel + fast under/over-force chase.
  # Chatter: short-lived ΔD_hf(Is). Steady offset: force_dob. Not sticky Ke·D.
  admittance_mass_z: 1.0
  admittance_damping_z: 25.0
  max_vz_tool_m_s: 0.08
  desired_force_ramp_s: 0.30
  var_damping_enabled: true
  var_damping_omega_c_hz: 2.5
  var_damping_lambda: 0.951
  var_damping_f_max_n: 7.0
  # ΔD_hf amplitude (primary chatter dissipation); M bump is secondary.
  var_damping_d_u: 60.0
  var_damping_m_u: 0.0
  var_damping_m_max: 3.0
  var_damping_dc_alpha: 0.02
  var_damping_hf_attack_s: 0.02
  var_damping_hf_hold_s: 0.18
  var_damping_hf_release_s: 0.12
  var_damping_hf_release_fast_s: 0.04  # dump ΔD on hand-release / large |e_f|
  var_damping_hf_on: 0.30
  var_damping_hf_off: 0.15
  var_damping_hf_err_n: 0.8
  recontact_vz_cap_m_s: 0.012
  recontact_hold_s: 0.12
  contact_episode_release_s: 0.30
  contact_episode_release_force_n: 0.75
  # Restored from e85c9ab.  Steady under-force offset rejection; 1bfe98b
  # disabled it as part of the anti-bounce sweep, and the force barrier below
  # now provides that brake instead.
  force_dob:
    enabled: true
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
    t_react_s: 0.050
    budget_min_n: 1.0
    budget_frac: 0.20
    f_keep_n: 0.5
    v_ref_m_s: 0.08
    v_min_retract_m_s: 0.002
    # Barrier keeps its force-error gating; this only stops it closing press
    # to exactly zero, which left the tool unable to recover a lost contact.
    v_min_press_m_s: 0.003
    # Free-space approach cap.  Impact ~ Ke*v*T_delay, so closing the gap at
    # the full 80 mm/s made ~8 N peaks on a 3 N target and the over-force
    # retract threw the tool off the surface.  In-contact response unchanged.
    v_seek_free_m_s: 0.030
    fdot_lpf_s: 0.040
    precontact_raw_trigger_n: 1.50 # short impact sleeve; never latches contact
    stiffness_cap_enabled: true
    ke_floor_n_m: 50.0
    mass_floor_kg: 0.05
  # Force-axis slew is press-positive and asymmetric.  A sign reversal into
  # retract gets the fastest allowance and is never tank/alpha gated.
  # 0.30 allowed press to rise only ~1.9 mm/s per 6.2 ms tick — 0.27 s to
  # reach the 80 mm/s cap, which is the "damped, not light" feel.  The force
  # barrier is the error-gated brake; this no longer has to be one.
  force_axis_slew_press_m_s2: 0.80
  force_axis_slew_retract_m_s2: 1.20
  force_axis_slew_reverse_m_s2: 2.00
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
    ke_impact_initial: 1500.0
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
  proactive_feedforward: true
  # Bidirectional press feedforward restored from e85c9ab: retract_only killed
  # the press-side v_r integration outright (measured v_r_z p95 = 0), which is
  # the single largest cause of slow under-force chase.  The force barrier
  # still caps press as the force error closes.
  proactive_retract_only: false
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
```

---

## 3. 差异与裁决点

| 层 | A | B | 开关 | 建议问审查 |
|---|---|---|---|---|
| 死区 / `v_r` / DOB / `ΔD_hf` / lateral chase | 有 | 有（同一套） | `hybrid_motion.*` | 增益是否维持 e85c9ab |
| `twist_sigma_floor` / `task_weight_min_frac` | 0.25 / 0.22（该 commit 改过） | 现 yaml 0.02 / 0.05（后来又改回去） | `inner.qp` | 力轴被 σ 刹车卡住时用哪组 |
| surface force modulation | 无 | 有 | `surface_force_modulation` | 手柄 tool-Z 要不要 |
| force-point 推进 | 无 | 有 | 代码路径，跟 hybrid episode | 扫查要、点力保持不要？ |
| 双向能量流 | 无 | 有，默认 observe | `bidirectional_flow.mode` | 先观察还是直接 active，还是整层删 |

**兼容核**：现有 `AdmittanceController` 同时含 A 通路和 B 门控层。peirastic `ForceLaw` 先适配它。  
- 要 A：关 surface modulation，`bidirectional_flow.mode=off`，hybrid 增益回到上表 A 列。  
- 要 B：保持 observe/active 与 surface 配置。  
- 要纯 A 代码：再换 `ForceLaw` 实现，模式层不动。

---

## 4. A 版完整源码（`e85c9ab957c78d500c3746159f1f47f4cabe3f0a`）

### `rm75_control/rm75_control/control/admittance_common/controller.py`

```python
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
from rm75_control.control.admittance_common.force_dob import (
    ForceDisturbanceObserver,
    ForceDobConfig,
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
    force_lateral_gain_floor: float = 0.35
    force_dob: ForceDobConfig = field(default_factory=ForceDobConfig)

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
                c.get("force_lateral_gain_floor", 0.35)
            ),
            force_dob=ForceDobConfig.from_dict(c),
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
        # Arm lateral chase softener only after real tool-XY scan motion.
        self._lat_soften_hold_s = 0.0
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
        self._lat_soften_hold_s = 0.0
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
    ) -> np.ndarray:
        # The hardware-proven admittance dynamics retain the nominal fixed dt.
        # Wall-clock dt is used only by contact/fast-force confirmation timers.
        del v_tcp_z_actual
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
        kp_rot = cfg.kp_pos[3:6] * cfg.track_axes[3:6]
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

        dt_eff = self.dt * self.time_scale
        if force_task_active:
            self._contact_time_s += dt_eff
        rising_edge = bool(contact_update.acquired)
        if bool(contact_update.reacquired) or rising_edge:
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
        damping = self._d_z_smooth
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
            # Implicit Euler: (M/dt + D) v+ = M/dt · v + D · v_r + drive
            denom = mass_z / dt_eff + max(damping, 0.0)
            velocity = (
                (mass_z / dt_eff) * self.v_force_z
                + max(damping, 0.0) * v_reference
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
```

### `rm75_control/rm75_control/control/admittance_common/proactive_force_ff.py`

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
    # When False, under-force press chase is never closed by Dimeas Iₛ
    # (over-force retract was already ungated). Chatter dissipation is left
    # to short-lived ΔD_hf in the passive admittance layer.
    gate_press_on_is: bool = True
    # Soft press attenuation vs Iₛ even when gate_press_on_is is False:
    # floor at Iₛ≥press_is_soft_stop (1=no soft atten). Stops single-tick
    # force dips from slamming v_r to the cap ("frame-drop" feel).
    press_is_soft_floor: float = 0.45
    press_is_soft_stop: float = 0.85
    # Max rising slew on press-side v_r [m/s²].
    press_slew_max_m_s2: float = 0.35
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
            gate_press_on_is=bool(
                p.get(
                    "gate_press_on_is",
                    p.get("proactive_gate_press_on_is", True),
                )
            ),
            press_is_soft_floor=float(
                p.get(
                    "press_is_soft_floor",
                    p.get("proactive_press_is_soft_floor", 0.45),
                )
            ),
            press_is_soft_stop=float(
                p.get(
                    "press_is_soft_stop",
                    p.get("proactive_press_is_soft_stop", 0.85),
                )
            ),
            press_slew_max_m_s2=float(
                p.get(
                    "press_slew_max_m_s2",
                    p.get("proactive_press_slew_max_m_s2", 0.35),
                )
            ),
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
        self.last_fast_retract_clear = False

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
        retract_fast_hold: bool = False,
        chase_scale: float = 1.0,
    ) -> float:
        cfg = self.cfg
        if not cfg.enabled:
            self.v_r = 0.0
            self.last_drive = 0.0
            self.last_instability_scale = 1.0
            self.last_reference_accel_m_s2 = 0.0
            self.last_reversal_reset = False
            self.last_fast_retract_clear = False
            return 0.0

        self.last_fast_retract_clear = False
        # The raw-force veto is a safety correction and must still remove a
        # stale retracting reference when the trajectory governor has frozen
        # its reference clock (dt_eff == 0).  It does not advance any
        # integrator state.
        if retract_fast_hold and self.v_r < 0.0:
            self.v_r = 0.0
            self.last_fast_retract_clear = True
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
        # The fast raw-force path is a one-way veto only.  It may remove an
        # already negative active reference when the raw force has fallen
        # ahead of the delayed 6 Hz control force, but it cannot command a
        # press and it never clears the passive admittance velocity.

        has_effective_error = in_contact and abs(eff) > 1e-12
        integrate = has_effective_error
        if integrate and cfg.retract_only and eff > 0.0:
            integrate = False
        if integrate and retract_fast_hold and eff < 0.0:
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
                # Slow tangential scan / turnaround: soften under-force chase
                # so force-axis motion does not feel like a lateral jerk.
                step = cfg.gain * drive * float(
                    np.clip(chase_scale, 0.0, 1.0)
                )
            if step > 0.0:
                if cfg.gate_press_on_is and cfg.press_is_gate > 1e-9:
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
                else:
                    # Soft floor: never fully kill press, but blunt noise dips.
                    soft_stop = max(float(cfg.press_is_soft_stop), 1e-9)
                    soft_floor = float(
                        np.clip(cfg.press_is_soft_floor, 0.0, 1.0)
                    )
                    if instability_index <= 0.0 or soft_floor >= 1.0 - 1e-9:
                        self.last_instability_scale = 1.0
                    elif instability_index >= soft_stop:
                        self.last_instability_scale = soft_floor
                    else:
                        u = float(instability_index / soft_stop)
                        blend = u * u * (3.0 - 2.0 * u)
                        self.last_instability_scale = float(
                            1.0 - blend * (1.0 - soft_floor)
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
            # Slew-limit rising press reference only (retract stays snappy).
            if step > 0.0 and cfg.press_slew_max_m_s2 > 0.0:
                max_step = float(cfg.press_slew_max_m_s2)
                step = min(step, max_step)
            self.last_reference_accel_m_s2 = float(step)
            self.v_r += dt_eff * step

        if cfg.v_r_max_m_s > 0.0:
            self.v_r = float(np.clip(self.v_r, -cfg.v_r_max_m_s, cfg.v_r_max_m_s))
        if v_z_cap > 0.0:
            self.v_r = float(np.clip(self.v_r, -v_z_cap, v_z_cap))
        return self.v_r
```

### `rm75_control/rm75_control/control/admittance_common/force_dob.py`

```python
"""Normal-axis force disturbance observer (DOSMAC-lite).

Models unmeasured contact disturbance (stiffness change, surface motion,
model error) as a scalar ``d`` on the tool-Z force equation and compensates
it with a leaky integrator on the deadbanded force error:

    u_dob ← u_dob + dt · (ki · e_f − u_dob / leak_s)
    M · v̇ + D · (v − v_r) = e_f + u_dob

Frozen while the Dimeas index is high so the observer does not wind up on
contact chatter.  Caps prevent fighting the passive admittance during impact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForceDobConfig:
    # Default off so unit tests keep the passive admittance baseline; YAML
    # enables this for hardware constant-force tracking.
    enabled: bool = False
    ki: float = 6.0
    leak_s: float = 0.45
    u_max_n: float = 1.5
    freeze_is: float = 0.45
    reset_on_reversal: bool = True

    @classmethod
    def from_dict(cls, parent: dict) -> ForceDobConfig:
        d = parent.get("force_dob", {})
        if not isinstance(d, dict):
            d = {}
        return cls(
            enabled=bool(d.get("enabled", False)),
            ki=float(d.get("ki", 6.0)),
            leak_s=float(d.get("leak_s", 0.45)),
            u_max_n=float(d.get("u_max_n", 1.5)),
            freeze_is=float(d.get("freeze_is", 0.45)),
            reset_on_reversal=bool(d.get("reset_on_reversal", True)),
        )


class ForceDisturbanceObserver:
    """Leaky PI-style disturbance estimate on the normal force error."""

    def __init__(self, cfg: ForceDobConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.u_dob = 0.0
        self.frozen = False
        self._last_eff = 0.0

    def update(
        self,
        eff: float,
        *,
        dt_eff: float,
        in_contact: bool,
        instability_index: float,
        chase_scale: float = 1.0,
    ) -> float:
        cfg = self.cfg
        if not cfg.enabled:
            self.u_dob = 0.0
            self.frozen = False
            self._last_eff = float(eff)
            return 0.0
        if not in_contact or dt_eff <= 0.0:
            if not in_contact and cfg.leak_s > 1e-6 and dt_eff > 0.0:
                self.u_dob -= (dt_eff / cfg.leak_s) * self.u_dob
            self.frozen = False
            self._last_eff = float(eff)
            return float(self.u_dob)

        # Do not let a press-side disturbance estimate fight an over-force
        # escape (or the reverse).
        if (
            cfg.reset_on_reversal
            and abs(float(eff)) > 1e-9
            and self._last_eff * float(eff) < 0.0
        ):
            self.u_dob = 0.0

        freeze = float(instability_index) >= float(cfg.freeze_is)
        self.frozen = freeze
        if not freeze:
            # Soften DOB integration on under-force when tangential speed is low
            # (scan turnaround); keep full ki for over-force escape.
            ki_scale = (
                1.0
                if float(eff) < 0.0
                else float(np.clip(chase_scale, 0.0, 1.0))
            )
            self.u_dob += dt_eff * float(cfg.ki) * ki_scale * float(eff)
        if cfg.leak_s > 1e-6:
            self.u_dob -= (dt_eff / cfg.leak_s) * self.u_dob
        if cfg.u_max_n > 0.0:
            self.u_dob = float(
                np.clip(self.u_dob, -cfg.u_max_n, cfg.u_max_n)
            )
        self._last_eff = float(eff)
        return float(self.u_dob)
```

### `rm75_control/rm75_control/control/admittance_common/force_barrier.py`

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
        v_z_cap_retract: float | None = None,
    ) -> tuple[float, float]:
        cfg = self.cfg
        v_hi = max(float(v_z_cap), 0.0)
        v_hi_retract = max(
            float(v_z_cap_retract) if v_z_cap_retract is not None else v_hi,
            0.0,
        )
        if not cfg.enabled:
            self.cap_press_z = v_hi
            self.cap_retract_z = v_hi_retract
            self.f_pred_z = float(f_z)
            return self.cap_press_z, self.cap_retract_z

        if not in_contact:
            seek = max(float(seek_vz_m_s), 0.0)
            if v_hi > 0.0:
                seek = min(seek, v_hi) if seek > 0.0 else v_hi
            del contact_enter_n
            self.cap_press_z = seek if seek > 0.0 else v_hi
            self.cap_retract_z = v_hi_retract
            self.f_pred_z = float(f_z)
            return self.cap_press_z, self.cap_retract_z

        if abs(float(f_des_z)) < 1e-6:
            self.cap_press_z = v_hi
            self.cap_retract_z = v_hi_retract
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
        if v_hi_retract > 0.0:
            cap_retract = min(cap_retract, v_hi_retract)

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

### `rm75_control/rm75_control/control/admittance_common/contact_state.py`

```python
"""Physical normal-contact tracking, separate from the force-task latch.

The force task must remain active while a moving surface temporarily leaves
the probe.  Environment-stiffness adaptation has a different requirement: it
must know when the probe is no longer carrying load so that a later impact can
re-arm stiff-first damping.

This tracker therefore never ends a task.  It only classifies the load-bearing
contact episode using:

* filtered force for a conservative, confirmed loss decision;
* compensated raw force for a low-latency re-acquisition decision;
* hysteresis and confirmation times so a short 4--12 Hz trough does not re-arm
  stiff-first every half-cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PhysicalContactConfig:
    enabled: bool = True
    enter_n: float = 0.80
    hard_enter_n: float = 1.50
    exit_n: float = 0.35
    enter_confirm_s: float = 0.010
    exit_confirm_s: float = 0.100

    @classmethod
    def from_dict(cls, raw: dict) -> PhysicalContactConfig:
        c = raw.get("hybrid_motion", raw.get("controller", raw))
        p = c.get("physical_contact", {})
        if not isinstance(p, dict):
            p = {}
        return cls(
            enabled=bool(p.get("enabled", True)),
            enter_n=float(
                p.get(
                    "enter_n",
                    c.get("physical_contact_enter_n", c.get("contact_threshold_n", 0.8)),
                )
            ),
            hard_enter_n=float(
                p.get(
                    "hard_enter_n",
                    c.get("physical_contact_hard_enter_n", 1.5),
                )
            ),
            exit_n=float(
                p.get(
                    "exit_n",
                    c.get("physical_contact_exit_n", 0.35),
                )
            ),
            enter_confirm_s=float(
                p.get(
                    "enter_confirm_s",
                    c.get("physical_contact_enter_confirm_s", 0.010),
                )
            ),
            exit_confirm_s=float(
                p.get(
                    "exit_confirm_s",
                    c.get("physical_contact_exit_confirm_s", 0.100),
                )
            ),
        )


@dataclass(frozen=True)
class PhysicalContactUpdate:
    present: bool
    state: str
    acquired: bool = False
    reacquired: bool = False
    lost: bool = False


class PhysicalContactTracker:
    """Four-state load-bearing contact tracker.

    ``CONTACT`` and ``SUSPECT_LOSS`` both count as physically present.  A
    confirmed ``LOST`` episode is required before the next force rise can emit
    ``reacquired=True``.
    """

    FREE = "free"
    CONTACT = "contact"
    SUSPECT_LOSS = "suspect_loss"
    LOST = "lost"

    def __init__(self, cfg: PhysicalContactConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.state = self.FREE
        self.low_timer_s = 0.0
        self.high_timer_s = 0.0
        self.ever_acquired = False
        self.filtered_force_n = 0.0
        self.raw_force_n = 0.0

    @property
    def present(self) -> bool:
        return self.state in (self.CONTACT, self.SUSPECT_LOSS)

    def force_state(self, present: bool) -> PhysicalContactUpdate:
        """Explicit-state compatibility path used by deterministic tests."""
        was_present = self.present
        had_contact = self.ever_acquired
        self.low_timer_s = 0.0
        self.high_timer_s = 0.0
        if present:
            self.state = self.CONTACT
            self.ever_acquired = True
            acquired = not was_present
            return PhysicalContactUpdate(
                present=True,
                state=self.state,
                acquired=acquired,
                reacquired=acquired and had_contact,
            )
        self.state = self.LOST if had_contact else self.FREE
        return PhysicalContactUpdate(
            present=False,
            state=self.state,
            lost=was_present,
        )

    def update(
        self,
        filtered_force_n: float,
        raw_force_n: float,
        *,
        dt_s: float,
    ) -> PhysicalContactUpdate:
        cfg = self.cfg
        dt = max(float(dt_s), 0.0)
        self.filtered_force_n = float(filtered_force_n)
        self.raw_force_n = float(raw_force_n)

        if not cfg.enabled:
            present = max(self.filtered_force_n, self.raw_force_n) >= cfg.enter_n
            return self.force_state(present)

        finite = np.isfinite(self.filtered_force_n) and np.isfinite(
            self.raw_force_n
        )
        if not finite:
            # Missing data must never manufacture a contact transition.
            self.low_timer_s = 0.0
            self.high_timer_s = 0.0
            return PhysicalContactUpdate(self.present, self.state)

        if self.present:
            self.high_timer_s = 0.0
            if self.filtered_force_n < cfg.exit_n:
                self.low_timer_s += dt
                self.state = self.SUSPECT_LOSS
                if self.low_timer_s + 1e-12 >= max(cfg.exit_confirm_s, 0.0):
                    self.state = self.LOST
                    self.low_timer_s = 0.0
                    return PhysicalContactUpdate(
                        present=False,
                        state=self.state,
                        lost=True,
                    )
            else:
                self.low_timer_s = 0.0
                self.state = self.CONTACT
            return PhysicalContactUpdate(self.present, self.state)

        self.low_timer_s = 0.0
        hard_hit = (
            cfg.hard_enter_n > 0.0
            and self.raw_force_n >= cfg.hard_enter_n
        )
        high = (
            self.raw_force_n >= cfg.enter_n
            or self.filtered_force_n >= cfg.enter_n
        )
        if hard_hit:
            self.high_timer_s = max(cfg.enter_confirm_s, 0.0)
        elif high:
            self.high_timer_s += dt
        else:
            self.high_timer_s = 0.0

        if self.high_timer_s + 1e-12 >= max(cfg.enter_confirm_s, 0.0):
            had_contact = self.ever_acquired
            self.ever_acquired = True
            self.state = self.CONTACT
            self.high_timer_s = 0.0
            return PhysicalContactUpdate(
                present=True,
                state=self.state,
                acquired=True,
                reacquired=had_contact,
            )

        self.state = self.LOST if self.ever_acquired else self.FREE
        return PhysicalContactUpdate(False, self.state)
```

## 5. B 版完整源码（当前工作树）

### `rm75_control/rm75_control/control/admittance_common/controller.py`

```python
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
    force_lateral_gain_floor: float = 0.35
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
                c.get("force_lateral_gain_floor", 0.35)
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
        # Force owns a Cartesian point along tool-Z; QPIK only tracks motion.
        self._force_point_base = np.zeros(3)
        self._force_point_inited = False
        self.force_point_z = 0.0
        self.last_pose_d_combined = np.zeros(6)
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
        self._force_point_base = np.zeros(3)
        self._force_point_inited = False
        self.force_point_z = 0.0
        self.last_pose_d_combined = np.zeros(6)
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
        """Legacy sleeve (tests). Runtime path uses the force point + PBAC."""
        v_pos_tool = np.zeros(6, dtype=float)
        v_pos_tool[:3] = r_mat.T @ np.asarray(v_pos_base[:3], dtype=float)
        v_pos_tool[3:6] = r_mat.T @ np.asarray(v_pos_base[3:6], dtype=float)
        v_cmd_tool = v_pos_tool.copy()
        v_cmd_tool[2] = float(v_force_tool[2])
        v_cmd_base = np.zeros(6, dtype=float)
        v_cmd_base[:3] = r_mat @ v_cmd_tool[:3]
        v_cmd_base[3:] = r_mat @ v_cmd_tool[3:6]
        return v_cmd_tool, v_cmd_base

    def _advance_force_point(
        self,
        pose_predicted: np.ndarray,
        desired_pose: np.ndarray,
        r_mat: np.ndarray,
        v_force_z: float,
        dt_s: float,
        *,
        reseeds: bool,
    ) -> None:
        n = np.asarray(r_mat[:, 2], dtype=float)
        p_now = np.asarray(pose_predicted[:3], dtype=float)
        if (not self._force_point_inited) or reseeds:
            self._force_point_base = p_now.copy()
            self._force_point_inited = True
        self._force_point_base = (
            self._force_point_base + n * float(v_force_z) * float(dt_s)
        )
        self.force_point_z = float(np.dot(n, self._force_point_base))
        pose_c = np.asarray(desired_pose, dtype=float).copy()
        p_scan = pose_c[:3]
        pose_c[:3] = p_scan - n * float(np.dot(n, p_scan)) + n * self.force_point_z
        self.last_pose_d_combined = pose_c

    def _motion_twist_to_force_point(
        self,
        pose_predicted: np.ndarray,
        desired_pose: np.ndarray,
        vel_ff: np.ndarray,
        r_mat: np.ndarray,
        v_force_z: float,
        use_pbac: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """PBAC to the fused scan+force point. QPIK sees only this motion twist."""
        cfg = self.cfg
        err_pose = pose_error(
            desired_pose,
            pose_predicted,
            cfg.euler_order,
        )
        if not use_pbac:
            err_pose[:] = 0.0
        err_tool = r_mat.T @ err_pose[:3]
        n = r_mat[:, 2]
        err_tool[2] = float(np.dot(n, self._force_point_base - pose_predicted[:3]))
        if cfg.pos_err_deadband_m > 0.0:
            for index in range(3):
                if abs(err_tool[index]) <= cfg.pos_err_deadband_m:
                    err_tool[index] = 0.0
        kp = cfg.kp_pos[:3] * cfg.track_axes[:3]
        v_corr_tool = kp * err_tool
        if cfg.pos_correction_max_m_s > 0.0:
            v_corr_tool = np.clip(
                v_corr_tool,
                -cfg.pos_correction_max_m_s,
                cfg.pos_correction_max_m_s,
            )
        err_rot_tool = r_mat.T @ err_pose[3:6]
        kp_rot = cfg.kp_pos[3:6] * cfg.track_axes[3:6]
        v_rot_tool = kp_rot * err_rot_tool
        vel_ff_tool = r_mat.T @ np.asarray(vel_ff[:3], dtype=float)
        vel_ff_tool[2] = float(v_force_z)
        w_ff_tool = r_mat.T @ np.asarray(vel_ff[3:6], dtype=float)
        v_cmd_tool = np.zeros(6, dtype=float)
        v_cmd_tool[:3] = vel_ff_tool + v_corr_tool
        v_cmd_tool[3:6] = w_ff_tool + v_rot_tool
        v_cmd_base = np.zeros(6, dtype=float)
        v_cmd_base[:3] = r_mat @ v_cmd_tool[:3]
        v_cmd_base[3:] = r_mat @ v_cmd_tool[3:6]
        path_task = np.concatenate((vel_ff_tool, w_ff_tool))
        feedback_task = np.concatenate((v_corr_tool, v_rot_tool))
        task_limit = np.asarray(cfg.max_velocity, dtype=float)
        self.last_path_twist = np.clip(path_task, -task_limit, task_limit)
        self.last_feedback_twist = np.clip(feedback_task, -task_limit, task_limit)
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
        self._advance_force_point(
            pose_predicted,
            desired_pose,
            r_mat,
            float(v_force_tool[2]),
            dt_eff,
            reseeds=rising_edge,
        )
        v_cmd_tool, v_cmd_base = self._motion_twist_to_force_point(
            pose_predicted,
            desired_pose,
            vel_ff,
            r_mat,
            float(v_force_tool[2]),
            use_pbac,
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
```

### `rm75_control/rm75_control/control/admittance_common/proactive_force_ff.py`

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
    # When False, under-force press chase is never closed by Dimeas Iₛ
    # (over-force retract was already ungated). Chatter dissipation is left
    # to short-lived ΔD_hf in the passive admittance layer.
    gate_press_on_is: bool = True
    # Soft press attenuation vs Iₛ even when gate_press_on_is is False:
    # floor at Iₛ≥press_is_soft_stop (1=no soft atten). Stops single-tick
    # force dips from slamming v_r to the cap ("frame-drop" feel).
    press_is_soft_floor: float = 0.45
    press_is_soft_stop: float = 0.85
    # Max rising slew on press-side v_r [m/s²].
    press_slew_max_m_s2: float = 0.35
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
            gate_press_on_is=bool(
                p.get(
                    "gate_press_on_is",
                    p.get("proactive_gate_press_on_is", True),
                )
            ),
            press_is_soft_floor=float(
                p.get(
                    "press_is_soft_floor",
                    p.get("proactive_press_is_soft_floor", 0.45),
                )
            ),
            press_is_soft_stop=float(
                p.get(
                    "press_is_soft_stop",
                    p.get("proactive_press_is_soft_stop", 0.85),
                )
            ),
            press_slew_max_m_s2=float(
                p.get(
                    "press_slew_max_m_s2",
                    p.get("proactive_press_slew_max_m_s2", 0.35),
                )
            ),
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
        self.last_fast_retract_clear = False

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
        retract_fast_hold: bool = False,
        chase_scale: float = 1.0,
    ) -> float:
        cfg = self.cfg
        if not cfg.enabled:
            self.v_r = 0.0
            self.last_drive = 0.0
            self.last_instability_scale = 1.0
            self.last_reference_accel_m_s2 = 0.0
            self.last_reversal_reset = False
            self.last_fast_retract_clear = False
            return 0.0

        self.last_fast_retract_clear = False
        # The raw-force veto is a safety correction and must still remove a
        # stale retracting reference when the trajectory governor has frozen
        # its reference clock (dt_eff == 0).  It does not advance any
        # integrator state.
        if retract_fast_hold and self.v_r < 0.0:
            self.v_r = 0.0
            self.last_fast_retract_clear = True
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
        # The fast raw-force path is a one-way veto only.  It may remove an
        # already negative active reference when the raw force has fallen
        # ahead of the delayed 6 Hz control force, but it cannot command a
        # press and it never clears the passive admittance velocity.

        has_effective_error = in_contact and abs(eff) > 1e-12
        integrate = has_effective_error
        if integrate and cfg.retract_only and eff > 0.0:
            integrate = False
        if integrate and retract_fast_hold and eff < 0.0:
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
                # Slow tangential scan / turnaround: soften under-force chase
                # so force-axis motion does not feel like a lateral jerk.
                step = cfg.gain * drive * float(
                    np.clip(chase_scale, 0.0, 1.0)
                )
            if step > 0.0:
                if cfg.gate_press_on_is and cfg.press_is_gate > 1e-9:
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
                else:
                    # Soft floor: never fully kill press, but blunt noise dips.
                    soft_stop = max(float(cfg.press_is_soft_stop), 1e-9)
                    soft_floor = float(
                        np.clip(cfg.press_is_soft_floor, 0.0, 1.0)
                    )
                    if instability_index <= 0.0 or soft_floor >= 1.0 - 1e-9:
                        self.last_instability_scale = 1.0
                    elif instability_index >= soft_stop:
                        self.last_instability_scale = soft_floor
                    else:
                        u = float(instability_index / soft_stop)
                        blend = u * u * (3.0 - 2.0 * u)
                        self.last_instability_scale = float(
                            1.0 - blend * (1.0 - soft_floor)
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
            # Slew-limit rising press reference only (retract stays snappy).
            if step > 0.0 and cfg.press_slew_max_m_s2 > 0.0:
                max_step = float(cfg.press_slew_max_m_s2)
                step = min(step, max_step)
            self.last_reference_accel_m_s2 = float(step)
            self.v_r += dt_eff * step

        if cfg.v_r_max_m_s > 0.0:
            self.v_r = float(np.clip(self.v_r, -cfg.v_r_max_m_s, cfg.v_r_max_m_s))
        if v_z_cap > 0.0:
            self.v_r = float(np.clip(self.v_r, -v_z_cap, v_z_cap))
        return self.v_r
```

### `rm75_control/rm75_control/control/admittance_common/force_dob.py`

```python
"""Normal-axis force disturbance observer (DOSMAC-lite).

Models unmeasured contact disturbance (stiffness change, surface motion,
model error) as a scalar ``d`` on the tool-Z force equation and compensates
it with a leaky integrator on the deadbanded force error:

    u_dob ← u_dob + dt · (ki · e_f − u_dob / leak_s)
    M · v̇ + D · (v − v_r) = e_f + u_dob

Frozen while the Dimeas index is high so the observer does not wind up on
contact chatter.  Caps prevent fighting the passive admittance during impact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForceDobConfig:
    # Default off so unit tests keep the passive admittance baseline; YAML
    # enables this for hardware constant-force tracking.
    enabled: bool = False
    ki: float = 6.0
    leak_s: float = 0.45
    u_max_n: float = 1.5
    freeze_is: float = 0.45
    reset_on_reversal: bool = True

    @classmethod
    def from_dict(cls, parent: dict) -> ForceDobConfig:
        d = parent.get("force_dob", {})
        if not isinstance(d, dict):
            d = {}
        return cls(
            enabled=bool(d.get("enabled", False)),
            ki=float(d.get("ki", 6.0)),
            leak_s=float(d.get("leak_s", 0.45)),
            u_max_n=float(d.get("u_max_n", 1.5)),
            freeze_is=float(d.get("freeze_is", 0.45)),
            reset_on_reversal=bool(d.get("reset_on_reversal", True)),
        )


class ForceDisturbanceObserver:
    """Leaky PI-style disturbance estimate on the normal force error."""

    def __init__(self, cfg: ForceDobConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.u_dob = 0.0
        self.frozen = False
        self._last_eff = 0.0

    def update(
        self,
        eff: float,
        *,
        dt_eff: float,
        in_contact: bool,
        instability_index: float,
        chase_scale: float = 1.0,
    ) -> float:
        cfg = self.cfg
        if not cfg.enabled:
            self.u_dob = 0.0
            self.frozen = False
            self._last_eff = float(eff)
            return 0.0
        if not in_contact or dt_eff <= 0.0:
            if not in_contact and cfg.leak_s > 1e-6 and dt_eff > 0.0:
                self.u_dob -= (dt_eff / cfg.leak_s) * self.u_dob
            self.frozen = False
            self._last_eff = float(eff)
            return float(self.u_dob)

        # Do not let a press-side disturbance estimate fight an over-force
        # escape (or the reverse).
        if (
            cfg.reset_on_reversal
            and abs(float(eff)) > 1e-9
            and self._last_eff * float(eff) < 0.0
        ):
            self.u_dob = 0.0

        freeze = float(instability_index) >= float(cfg.freeze_is)
        self.frozen = freeze
        if not freeze:
            # Soften DOB integration on under-force when tangential speed is low
            # (scan turnaround); keep full ki for over-force escape.
            ki_scale = (
                1.0
                if float(eff) < 0.0
                else float(np.clip(chase_scale, 0.0, 1.0))
            )
            self.u_dob += dt_eff * float(cfg.ki) * ki_scale * float(eff)
        if cfg.leak_s > 1e-6:
            self.u_dob -= (dt_eff / cfg.leak_s) * self.u_dob
        if cfg.u_max_n > 0.0:
            self.u_dob = float(
                np.clip(self.u_dob, -cfg.u_max_n, cfg.u_max_n)
            )
        self._last_eff = float(eff)
        return float(self.u_dob)
```

### `rm75_control/rm75_control/control/admittance_common/force_barrier.py`

```python
"""Force-space velocity damper for tool-Z press and retract motion.

The damper predicts near-future force from a filtered force derivative and
limits normal velocity before the delayed admittance loop can build a large
over-force transient.  It deliberately does not depend on the environment
stiffness estimate, which is least reliable at first impact.
"""

from __future__ import annotations

from dataclasses import dataclass

import math


@dataclass
class ForceBarrierConfig:
    enabled: bool = True
    t_react_s: float = 0.030
    budget_min_n: float = 1.0
    budget_frac: float = 0.20
    f_keep_n: float = 0.5
    v_ref_m_s: float = 0.05
    v_min_retract_m_s: float = 0.002
    # Floor under the in-contact press cap.  The prediction and stiffness
    # terms can both reach zero, and a controller that is not allowed to press
    # at all cannot recover from a detachment — hardware logs showed
    # cap_press_z pinned at 0 for the bottom 5% of contact ticks.  Applied
    # last, after the stiffness cap and the v_hi clamp.
    v_min_press_m_s: float = 0.003
    # Cap on the free-space approach.  Impact force goes roughly as
    # Ke * v * T_delay, so the speed used to close the last gap sets the first
    # peak.  Approaching at the full max_vz produced ~8 N peaks against a 3 N
    # target, and the resulting over-force retract threw the tool back off the
    # surface — press rail 22.7% of ticks, retract rail 23.1%, contact lost 30%
    # of the scan.  Only the free-space branch is capped; the in-contact
    # admittance response is untouched.  0 disables.
    v_seek_free_m_s: float = 0.030
    fdot_lpf_s: float = 0.040
    # Optional impact-energy/stiffness caps.  These use only controller-side
    # virtual quantities; no unmeasured physical damping is credited.
    stiffness_cap_enabled: bool = True
    ke_floor_n_m: float = 50.0
    mass_floor_kg: float = 0.05
    # Before the debounced physical-contact latch is established, a raw force
    # spike may request a short impact guard.  Keep this append-only in the
    # dataclass so positional construction of the older public fields remains
    # compatible.  Zero is the library-safe opt-out; the RM75 YAML opts in.
    precontact_raw_trigger_n: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict) -> "ForceBarrierConfig":
        root = raw if isinstance(raw, dict) else {}
        controller = root.get(
            "hybrid_motion", root.get("controller", root)
        )
        if not isinstance(controller, dict):
            controller = root
        barrier = controller.get(
            "force_barrier", root.get("force_barrier", {})
        )
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
            v_min_press_m_s=float(barrier.get("v_min_press_m_s", 0.003)),
            v_seek_free_m_s=float(barrier.get("v_seek_free_m_s", 0.030)),
            fdot_lpf_s=float(barrier.get("fdot_lpf_s", 0.040)),
            precontact_raw_trigger_n=float(
                barrier.get("precontact_raw_trigger_n", 0.0)
            ),
            stiffness_cap_enabled=bool(
                barrier.get("stiffness_cap_enabled", True)
            ),
            ke_floor_n_m=float(barrier.get("ke_floor_n_m", 50.0)),
            mass_floor_kg=float(barrier.get("mass_floor_kg", 0.05)),
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
        v_z_cap_retract: float | None = None,
        ke_est_n_m: float | None = None,
        mass_eq_kg: float | None = None,
        energy_available_j: float | None = None,
    ) -> tuple[float, float]:
        cfg = self.cfg
        v_hi = max(float(v_z_cap), 0.0)
        v_hi_retract = max(
            float(v_z_cap_retract) if v_z_cap_retract is not None else v_hi,
            0.0,
        )
        if not cfg.enabled:
            self.cap_press_z = v_hi
            self.cap_retract_z = v_hi_retract
            self.f_pred_z = float(f_z)
            return self.cap_press_z, self.cap_retract_z

        if not in_contact:
            seek = max(float(seek_vz_m_s), 0.0)
            if v_hi > 0.0:
                seek = min(seek, v_hi) if seek > 0.0 else v_hi
            # Free-space approach cap; see v_seek_free_m_s.  Take the smaller
            # of the two so a tighter external sleeve (recontact) still wins.
            free = max(float(cfg.v_seek_free_m_s), 0.0)
            if free > 0.0:
                seek = min(seek, free) if seek > 0.0 else free
            del contact_enter_n
            self.cap_press_z = seek if seek > 0.0 else v_hi
            self.cap_retract_z = v_hi_retract
            self.f_pred_z = float(f_z)
            return self.cap_press_z, self.cap_retract_z

        if abs(float(f_des_z)) < 1e-6:
            self.cap_press_z = v_hi
            self.cap_retract_z = v_hi_retract
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
        # A hard surface converts a small delayed penetration into a large
        # force rise.  Bound the approach kinetic energy by the remaining
        # force headroom and, when supplied, the verified tank balance:
        #
        #   v_force = DeltaF / sqrt(M_eq K_e)
        #   v_energy = sqrt(2 E_available / M_eq)
        #
        # Both are continuous in the positive headroom.  Missing estimates
        # leave the historical force-prediction cap unchanged.
        if cfg.stiffness_cap_enabled and ke_est_n_m is not None:
            ke = max(float(ke_est_n_m), float(cfg.ke_floor_n_m), 1e-9)
            mass = max(
                float(mass_eq_kg) if mass_eq_kg is not None else 1.0,
                float(cfg.mass_floor_kg),
                1e-9,
            )
            headroom = max((float(f_des_z) + budget) - f_pred, 0.0)
            cap_press = min(cap_press, headroom / math.sqrt(mass * ke))
            if energy_available_j is not None:
                energy = max(float(energy_available_j), 0.0)
                cap_press = min(cap_press, math.sqrt(2.0 * energy / mass))
        if v_hi > 0.0:
            cap_press = min(cap_press, v_hi)
        # Never close press completely while in contact; see v_min_press_m_s.
        # Bounded by v_hi so a small v_z_cap (recontact sleeve) still wins.
        v_min_press = max(float(cfg.v_min_press_m_s), 0.0)
        if v_hi > 0.0:
            v_min_press = min(v_min_press, v_hi)
        cap_press = max(cap_press, v_min_press)

        cap_retract = max(
            float(cfg.v_min_retract_m_s),
            (f_pred - float(cfg.f_keep_n)) / budget * v_ref,
        )
        if v_hi_retract > 0.0:
            cap_retract = min(cap_retract, v_hi_retract)

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

### `rm75_control/rm75_control/control/admittance_common/contact_state.py`

```python
"""Physical normal-contact tracking, separate from the force-task latch.

The force task must remain active while a moving surface temporarily leaves
the probe.  Environment-stiffness adaptation has a different requirement: it
must know when the probe is no longer carrying load so that a later impact can
re-arm stiff-first damping.

This tracker therefore never ends a task.  It only classifies the load-bearing
contact episode using:

* filtered force for a conservative, confirmed loss decision;
* compensated raw force for a low-latency re-acquisition decision;
* hysteresis and confirmation times so a short 4--12 Hz trough does not re-arm
  stiff-first every half-cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PhysicalContactConfig:
    enabled: bool = True
    enter_n: float = 0.80
    hard_enter_n: float = 1.50
    exit_n: float = 0.35
    enter_confirm_s: float = 0.010
    exit_confirm_s: float = 0.100

    @classmethod
    def from_dict(cls, raw: dict) -> PhysicalContactConfig:
        c = raw.get("hybrid_motion", raw.get("controller", raw))
        p = c.get("physical_contact", {})
        if not isinstance(p, dict):
            p = {}
        return cls(
            enabled=bool(p.get("enabled", True)),
            enter_n=float(
                p.get(
                    "enter_n",
                    c.get("physical_contact_enter_n", c.get("contact_threshold_n", 0.8)),
                )
            ),
            hard_enter_n=float(
                p.get(
                    "hard_enter_n",
                    c.get("physical_contact_hard_enter_n", 1.5),
                )
            ),
            exit_n=float(
                p.get(
                    "exit_n",
                    c.get("physical_contact_exit_n", 0.35),
                )
            ),
            enter_confirm_s=float(
                p.get(
                    "enter_confirm_s",
                    c.get("physical_contact_enter_confirm_s", 0.010),
                )
            ),
            exit_confirm_s=float(
                p.get(
                    "exit_confirm_s",
                    c.get("physical_contact_exit_confirm_s", 0.100),
                )
            ),
        )


@dataclass(frozen=True)
class PhysicalContactUpdate:
    present: bool
    state: str
    acquired: bool = False
    reacquired: bool = False
    lost: bool = False


class PhysicalContactTracker:
    """Four-state load-bearing contact tracker.

    ``CONTACT`` and ``SUSPECT_LOSS`` both count as physically present.  A
    confirmed ``LOST`` episode is required before the next force rise can emit
    ``reacquired=True``.
    """

    FREE = "free"
    CONTACT = "contact"
    SUSPECT_LOSS = "suspect_loss"
    LOST = "lost"

    def __init__(self, cfg: PhysicalContactConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.state = self.FREE
        self.low_timer_s = 0.0
        self.high_timer_s = 0.0
        self.ever_acquired = False
        self.filtered_force_n = 0.0
        self.raw_force_n = 0.0

    @property
    def present(self) -> bool:
        return self.state in (self.CONTACT, self.SUSPECT_LOSS)

    def force_state(self, present: bool) -> PhysicalContactUpdate:
        """Explicit-state compatibility path used by deterministic tests."""
        was_present = self.present
        had_contact = self.ever_acquired
        self.low_timer_s = 0.0
        self.high_timer_s = 0.0
        if present:
            self.state = self.CONTACT
            self.ever_acquired = True
            acquired = not was_present
            return PhysicalContactUpdate(
                present=True,
                state=self.state,
                acquired=acquired,
                reacquired=acquired and had_contact,
            )
        self.state = self.LOST if had_contact else self.FREE
        return PhysicalContactUpdate(
            present=False,
            state=self.state,
            lost=was_present,
        )

    def update(
        self,
        filtered_force_n: float,
        raw_force_n: float,
        *,
        dt_s: float,
    ) -> PhysicalContactUpdate:
        cfg = self.cfg
        dt = max(float(dt_s), 0.0)
        self.filtered_force_n = float(filtered_force_n)
        self.raw_force_n = float(raw_force_n)

        if not cfg.enabled:
            present = max(self.filtered_force_n, self.raw_force_n) >= cfg.enter_n
            return self.force_state(present)

        finite = np.isfinite(self.filtered_force_n) and np.isfinite(
            self.raw_force_n
        )
        if not finite:
            # Missing data must never manufacture a contact transition.
            self.low_timer_s = 0.0
            self.high_timer_s = 0.0
            return PhysicalContactUpdate(self.present, self.state)

        if self.present:
            self.high_timer_s = 0.0
            if self.filtered_force_n < cfg.exit_n:
                self.low_timer_s += dt
                self.state = self.SUSPECT_LOSS
                if self.low_timer_s + 1e-12 >= max(cfg.exit_confirm_s, 0.0):
                    self.state = self.LOST
                    self.low_timer_s = 0.0
                    return PhysicalContactUpdate(
                        present=False,
                        state=self.state,
                        lost=True,
                    )
            else:
                self.low_timer_s = 0.0
                self.state = self.CONTACT
            return PhysicalContactUpdate(self.present, self.state)

        self.low_timer_s = 0.0
        # Initial contact is confirmed on the filtered channel.  The raw
        # channel is intentionally reserved for *re*-acquisition after a
        # known flight: tool/gravity residuals produced isolated 1--2 N raw
        # spikes during the 162413 free-space approach and a single spike
        # used to latch the force episode for the rest of the scan.  Immediate
        # pre-contact impact limiting remains a separate force-barrier path.
        hard_hit = (
            self.ever_acquired
            and cfg.hard_enter_n > 0.0
            and self.raw_force_n >= cfg.hard_enter_n
        )
        if self.ever_acquired:
            high = (
                self.raw_force_n >= cfg.enter_n
                or self.filtered_force_n >= cfg.enter_n
            )
        else:
            high = self.filtered_force_n >= cfg.enter_n
        if hard_hit:
            self.high_timer_s = max(cfg.enter_confirm_s, 0.0)
        elif high:
            self.high_timer_s += dt
        else:
            self.high_timer_s = 0.0

        if self.high_timer_s + 1e-12 >= max(cfg.enter_confirm_s, 0.0):
            had_contact = self.ever_acquired
            self.ever_acquired = True
            self.state = self.CONTACT
            self.high_timer_s = 0.0
            return PhysicalContactUpdate(
                present=True,
                state=self.state,
                acquired=True,
                reacquired=had_contact,
            )

        self.state = self.LOST if self.ever_acquired else self.FREE
        return PhysicalContactUpdate(False, self.state)
```

### `rm75_control/rm75_control/control/admittance_common/bidirectional_flow.py`

```python
"""One-dimensional bidirectional energy-flow adaptation.

This module is an engineering adaptation of the proxy/real-port structure
described by Lee et al. (2024).  It is deliberately *not* a torque theorem:
the normal axis is a speed-level interface and the energy account only
credits damping which is explicitly identified as nominal.  Unidentified
physical friction, Dimeas damping, and actuator losses are never credited.

The implementation keeps the two important safety properties of the
structure useful to the RM75 controller:

* the proxy may be bidirectional, while the real auxiliary path is one-sided
  and can only add retract velocity;
* an energy gate is applied to positive (press) velocity only.  Retract
  velocity passes through when the gate is closed, and stale or unverified
  feedback closes the press gate.

``BidirectionalFlowController.update`` is intentionally small and scalar so
it can be used by simulation tests as well as the 200 Hz controller.  A
``step`` alias is provided for callers that use the usual discrete-controller
terminology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


@dataclass
class BidirectionalFlowConfig:
    """Configuration for the scalar normal-axis flow adapter.

    Upper-case gain/tank names are retained because they match the notation
    used in the design note (``K_d``, ``T_0``).  Lower-case aliases are
    accepted for YAML and Python callers as well.  ``Ki`` is intentionally
    zero by default; enabling integral mismatch feedback is an explicit
    tuning choice rather than an accidental source of energy.
    """

    # ``off`` preserves the legacy controller, ``observe`` computes all
    # states/telemetry but returns the unmodulated proxy speed, and ``active``
    # enables the retract-through/press gate.
    mode: str = "off"
    sign_verified: bool = False
    feedback_delay_verified: bool = False
    require_sign_verification: bool = True
    require_delay_verification: bool = True
    normal_sign: float = 1.0

    # Proxy/real-port mismatch feedback.
    # Defaults follow the speed-level design note: Dtrack is also the
    # mismatch damping and Kp gives a 0.10 s mismatch time constant.
    Kd: float | None = None
    Kp: float | None = None
    Ki: float = 0.0
    lambda_gain: float = 0.25
    Dtrack: float = 20.0
    track_correction_max_m_s: float = 0.020
    # Lee Sec. V-C: alpha must be zero in free space.  Below this |F| the
    # modulation is held off and the tank charges from proxy damping instead.
    # 0 restores the pure power test (paper-faithful, noise-sensitive).
    free_space_force_n: float = 0.5
    M_p: float = 1.0
    D_p: float = 0.0
    m_p: float | None = None
    d_p: float | None = None

    # Optional lower-case spelling used by configuration loaders.
    kd: float | None = None
    kp: float | None = None
    ki: float | None = None
    lambda_: float | None = None
    d_track: float | None = None

    # Positive press path and one-sided real auxiliary path.
    gamma_active: float = 1.0
    aux_tau_s: float = 0.05
    aux_max_retract_m_s: float = 0.05
    press_epsilon_m_s: float = 1.0e-6
    # Independent one-sided real auxiliary mass/impedance.  x_safe follows
    # the real port only while the gate is open; when alpha rises it freezes
    # and the implicit M_a/D_a update can only generate retract velocity.
    M_a: float = 0.01
    D_a: float = 0.20
    K_a: float = 5.0
    B_a: float = 0.0
    u_retract_n: float = 0.0
    # Deprecated speed-form spelling; converted to force with D_a.
    u_retract_m_s: float | None = None
    u_retract: float | None = None
    m_a: float | None = None
    d_a: float | None = None
    k_a: float | None = None

    # Energy tank.  The small values are intentional: this is the scalar
    # speed-level tank used by the normal axis, not a robot-wide battery.
    T0: float = 0.001
    Tmax: float = 0.004
    Tmin: float = 0.0001
    t0: float | None = None
    t_max: float | None = None
    t_min: float | None = None
    nominal_damping: float = 0.0
    # Constant unmodelled active-power allowance (watts).  Keep zero by
    # default; active effort*press is accounted separately below.
    mu_power_w: float = 0.0
    mu: float | None = None
    active_press_debit_n: float = 1.0
    positive_switching_cost_j: float = 0.0
    switch_epsilon_m_s: float = 1.0e-6

    # Lee-style modulation smoothing.  A rise (closing the gate) is quick;
    # reopening is deliberately slower to avoid press chatter.
    alpha_attack_s: float = 0.02
    alpha_release_s: float = 0.15

    # A missing actual velocity or an old feedback sample is fail-closed.
    max_feedback_age_s: float = 0.02
    feedback_timeout_s: float | None = None

    # Optional labels/telemetry metadata.
    engineering_adaptation_label: str = (
        "engineering adaptation; not a torque theorem"
    )

    def __post_init__(self) -> None:
        if self.kd is not None:
            self.Kd = float(self.kd)
        if self.kp is not None:
            self.Kp = float(self.kp)
        if self.ki is not None:
            self.Ki = float(self.ki)
        if self.lambda_ is not None:
            self.lambda_gain = float(self.lambda_)
        if self.d_track is not None:
            self.Dtrack = float(self.d_track)
        if self.m_p is not None:
            self.M_p = float(self.m_p)
        if self.d_p is not None:
            self.D_p = float(self.d_p)
        if self.m_a is not None:
            self.M_a = float(self.m_a)
        if self.d_a is not None:
            self.D_a = float(self.d_a)
        if self.k_a is not None:
            self.K_a = float(self.k_a)
        if self.u_retract is not None:
            self.u_retract_n = float(self.u_retract)
        if self.u_retract_m_s is not None:
            self.u_retract_n = float(self.u_retract_m_s) * max(
                float(self.D_a), 0.0
            )
        if self.mu is not None:
            # Backward-compatible ``mu`` spelling now denotes watts, not a
            # multiplier on active press effort.
            self.mu_power_w = float(self.mu)
        if self.t0 is not None:
            self.T0 = float(self.t0)
        if self.t_max is not None:
            self.Tmax = float(self.t_max)
        if self.t_min is not None:
            self.Tmin = float(self.t_min)

        mode = str(self.mode).strip().lower().replace("-", "_")
        if mode in {"disabled", "none", "legacy", "0"}:
            mode = "off"
        elif mode in {"monitor", "logging", "1"}:
            mode = "observe"
        elif mode in {"enabled", "on", "2"}:
            mode = "active"
        if mode not in {"off", "observe", "active"}:
            raise ValueError(
                "bidirectional flow mode must be one of off/observe/active"
            )
        self.mode = mode

        if self.Kd is None:
            self.Kd = self.Dtrack
        if self.Kp is None:
            self.Kp = self.Dtrack / 0.10
        self.Kd = max(_finite(self.Kd), 0.0)
        self.Kp = max(_finite(self.Kp), 0.0)
        self.Ki = max(_finite(self.Ki), 0.0)
        self.lambda_gain = max(_finite(self.lambda_gain), 0.0)
        self.Dtrack = max(_finite(self.Dtrack), 1.0e-9)
        self.M_p = max(_finite(self.M_p, 1.0), 1.0e-6)
        self.D_p = max(_finite(self.D_p, 0.0), 0.0)
        self.gamma_active = float(np.clip(_finite(self.gamma_active, 1.0), 0.0, 1.0))
        self.aux_tau_s = max(_finite(self.aux_tau_s, 0.05), 0.0)
        self.aux_max_retract_m_s = max(_finite(self.aux_max_retract_m_s, 0.05), 0.0)
        self.press_epsilon_m_s = max(_finite(self.press_epsilon_m_s, 1e-6), 0.0)
        self.M_a = max(_finite(self.M_a, 0.01), 1.0e-6)
        self.D_a = max(_finite(self.D_a, 0.20), 0.0)
        self.K_a = max(_finite(self.K_a, 5.0), 0.0)
        self.B_a = max(_finite(self.B_a, 0.0), 0.0)
        # Signed auxiliary effort in the press-positive frame; resulting
        # velocity is still clamped non-positive below.
        self.u_retract_n = _finite(self.u_retract_n, 0.0)
        self.switch_epsilon_m_s = max(_finite(self.switch_epsilon_m_s, 1e-6), 0.0)

        # Enforce the stated tank ordering even when a hand-edited YAML file
        # contains a malformed value.  T0 is clamped into the usable range.
        self.Tmin = max(_finite(self.Tmin, 0.0001), 0.0)
        self.Tmax = max(_finite(self.Tmax, 0.004), self.Tmin)
        self.T0 = float(np.clip(_finite(self.T0, 0.001), self.Tmin, self.Tmax))
        self.nominal_damping = max(_finite(self.nominal_damping), 0.0)
        self.mu_power_w = max(_finite(self.mu_power_w, 0.0), 0.0)
        self.active_press_debit_n = max(_finite(self.active_press_debit_n, 1.0), 0.0)
        self.positive_switching_cost_j = max(
            _finite(self.positive_switching_cost_j, 0.00005), 0.0
        )
        self.alpha_attack_s = max(_finite(self.alpha_attack_s, 0.02), 0.0)
        self.alpha_release_s = max(_finite(self.alpha_release_s, 0.15), 0.0)
        self.max_feedback_age_s = max(_finite(self.max_feedback_age_s, 0.02), 0.0)
        if self.feedback_timeout_s is not None:
            self.max_feedback_age_s = max(
                _finite(self.feedback_timeout_s, self.max_feedback_age_s), 0.0
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "BidirectionalFlowConfig":
        """Read the flow section from either a controller or root mapping."""

        if raw is None:
            return cls()
        root = dict(raw)
        c = root.get("hybrid_motion", root.get("controller", root))
        if not isinstance(c, Mapping):
            c = root
        section: Mapping[str, Any] = {}
        for name in (
            "bidirectional_flow",
            "bidirectional",
            "energy_flow",
            "normal_axis_flow",
            "befm",
        ):
            value = c.get(name, root.get(name))
            if isinstance(value, Mapping):
                section = value
                break
        if not section:
            section = c

        def value(*names: str, default: Any = None) -> Any:
            return _first(section, *names, default=_first(c, *names, default=default))

        mode = value("mode", "bidirectional_flow_mode", default="off")
        dtrack_value = _finite(value("Dtrack", "d_track", default=20.0), 20.0)
        kd_value = value("Kd", "kd", "mismatch_damping", default=None)
        kp_value = value("Kp", "kp", "mismatch_stiffness", default=None)
        sign_verified = value(
            "sign_verified",
            "normal_sign_verified",
            "force_sign_verified",
            "sign_verification",
            default=False,
        )
        delay_verified = value(
            "feedback_delay_verified",
            "delay_verified",
            "velocity_delay_verified",
            default=False,
        )
        # A mapping is a convenient explicit verification record.  Requiring
        # both fields prevents ``sign_verification: {configured: true}`` from
        # accidentally enabling an active press path.
        if isinstance(sign_verified, Mapping):
            sign_verified = bool(
                sign_verified.get("verified", sign_verified.get("ok", False))
                and sign_verified.get("positive_is_press", True)
            )
        return cls(
            mode=str(mode),
            sign_verified=bool(sign_verified),
            feedback_delay_verified=bool(delay_verified),
            require_sign_verification=bool(
                value("require_sign_verification", default=True)
            ),
            require_delay_verification=bool(
                value("require_delay_verification", default=True)
            ),
            normal_sign=_finite(value("normal_sign", "press_sign", default=1.0), 1.0),
            Kd=(None if kd_value is None else _finite(kd_value, dtrack_value)),
            Kp=(None if kp_value is None else _finite(kp_value, dtrack_value / 0.10)),
            Ki=_finite(value("Ki", "ki", "mismatch_integral", default=0.0), 0.0),
            lambda_gain=_finite(
                value("lambda_gain", "lambda", "lambda_", default=0.25), 0.25
            ),
            Dtrack=dtrack_value,
            M_p=_finite(value("M_p", "m_p", "proxy_mass", default=1.0), 1.0),
            D_p=_finite(value("D_p", "d_p", "proxy_damping", default=0.0), 0.0),
            track_correction_max_m_s=_finite(
                value("track_correction_max_m_s", "v_track_max_m_s", default=0.020),
                0.020,
            ),
            free_space_force_n=_finite(
                value("free_space_force_n", "air_force_n", default=0.5), 0.5
            ),
            gamma_active=_finite(value("gamma_active", "gamma", default=1.0), 1.0),
            aux_tau_s=_finite(value("aux_tau_s", "auxiliary_tau_s", default=0.05), 0.05),
            aux_max_retract_m_s=_finite(
                value("aux_max_retract_m_s", "v_aux_max_retract_m_s", default=0.05),
                0.05,
            ),
            M_a=_finite(value("M_a", "m_a", "aux_mass", default=0.01), 0.01),
            D_a=_finite(value("D_a", "d_a", "aux_damping", default=0.20), 0.20),
            K_a=_finite(value("K_a", "k_a", "aux_stiffness", default=5.0), 5.0),
            B_a=_finite(value("B_a", "b_a", "aux_velocity_damping", default=0.0), 0.0),
            u_retract_n=_finite(
                value("u_retract_n", "retract_effort_n", default=0.0), 0.0
            ),
            u_retract_m_s=(
                None
                if value("u_retract_m_s", "retract_through_m_s", default=None)
                is None
                else _finite(
                    value("u_retract_m_s", "retract_through_m_s", default=0.0),
                    0.0,
                )
            ),
            T0=_finite(value("T0", "t0", "tank_t0", default=0.001), 0.001),
            Tmax=_finite(value("Tmax", "t_max", "tank_tmax", default=0.004), 0.004),
            Tmin=_finite(value("Tmin", "t_min", "tank_tmin", default=0.0001), 0.0001),
            nominal_damping=_finite(
                value("nominal_damping", "D0", "d0", default=0.0), 0.0
            ),
            mu_power_w=_finite(
                value("mu_power_w", "mu", "tank_mu", default=0.0), 0.0
            ),
            active_press_debit_n=_finite(
                value("active_press_debit_n", "press_debit_n", default=1.0), 1.0
            ),
            positive_switching_cost_j=_finite(
                value(
                    "positive_switching_cost_j",
                    "switching_cost_j",
                    "switch_cost_j",
                    default=0.0,
                ),
                0.0,
            ),
            alpha_attack_s=_finite(value("alpha_attack_s", "attack_s", default=0.02), 0.02),
            alpha_release_s=_finite(
                value("alpha_release_s", "release_s", default=0.15), 0.15
            ),
            max_feedback_age_s=_finite(
                value("max_feedback_age_s", "feedback_timeout_s", "stale_after_s", default=0.02),
                0.02,
            ),
            engineering_adaptation_label=str(
                value(
                    "engineering_adaptation_label",
                    default="engineering adaptation; not a torque theorem",
                )
            ),
        )


@dataclass
class BidirectionalFlowTelemetry:
    """Snapshot returned by the most recent update.

    Scalar fields intentionally have stable names suitable for CSV logging.
    The controller mirrors these onto itself for existing loggers that use
    ``getattr(controller, name)``.
    """

    xp: float = 0.0
    vp: float = 0.0
    xa: float = 0.0
    va: float = 0.0
    fc: float = 0.0
    v_track: float = 0.0
    v_aux: float = 0.0
    retract_through: float = 0.0
    press: float = 0.0
    command: float = 0.0
    alpha: float = 1.0
    alpha_raw: float = 1.0
    alpha_case: str = "init"
    tank_energy: float = 0.001
    tank_power_credit: float = 0.0
    tank_power_debit: float = 0.0
    psi: float = 0.0
    tank_switch_cost: float = 0.0
    Pe: float = 0.0
    Pc: float = 0.0
    P_phys: float = 0.0
    P_mismatch: float = 0.0
    energy_phys_j: float = 0.0
    energy_mismatch_j: float = 0.0
    Sn: float = 0.001
    Sr_hat: float = 0.001
    alpha_delta_energy_j: float = 0.0
    modulation_debit_j: float = 0.0
    feedback_age_s: float = float("nan")
    feedback_stale: bool = True
    sign_verified: bool = False
    sign_fault: bool = False
    mode: str = "off"
    active: bool = False
    blocked_reason: str = ""
    feedback_delay_verified: bool = False
    gamma_effective: float = 0.0
    engineering_adaptation: str = "engineering adaptation; not a torque theorem"

    @property
    def velocity_mismatch(self) -> float:
        return float(self.vp - self.va)

    @property
    def position_mismatch(self) -> float:
        return float(self.xp - self.xa)


class BidirectionalFlowController:
    """Stateful scalar proxy/real-port controller.

    ``vp_cmd`` is the legacy force-admittance speed (positive means press).
    The real port can be supplied as a measured velocity and position.  If a
    position is unavailable, the last position is integrated from ``va``;
    missing velocity feedback is nevertheless considered stale and therefore
    cannot open the active press gate.
    """

    # Enough proxy-velocity history to reach back one staleness budget even at
    # the fastest tick rate this loop runs at.
    _VP_HISTORY_MAX = 64

    ENGINEERING_ADAPTATION = "engineering adaptation; not a torque theorem"

    def __init__(
        self,
        dt: float,
        config: BidirectionalFlowConfig | None = None,
    ) -> None:
        self.dt = max(_finite(dt, 0.005), 1.0e-6)
        self.cfg = config or BidirectionalFlowConfig()
        self.reset()

    def reset(self, *, x_actual: float | None = None) -> None:
        x0 = _finite(x_actual, 0.0) if x_actual is not None else 0.0
        self.xp = x0
        self.xa = x0
        self.va = 0.0
        self.vp = 0.0
        self.fc = 0.0
        self.v_track = 0.0
        self.v_aux = 0.0
        self.aux_anchor = x0
        self.x_aux = x0
        self.x_safe = x0
        self.retract_through = 0.0
        self.press = 0.0
        self.command = 0.0
        self.alpha = 0.0
        self.alpha_raw = 0.0
        self.alpha_case = "init"
        self.tank_energy = float(self.cfg.T0)
        self.tank_power_credit = 0.0
        self.tank_power_debit = 0.0
        self.psi = 0.0
        self.tank_switch_cost = 0.0
        self.Pe = 0.0
        self.Pc = 0.0
        self.P_phys = 0.0
        self.P_mismatch = 0.0
        self.energy_phys_j = 0.0
        self.energy_mismatch_j = 0.0
        self.Sn = self.tank_energy
        self.Sr_hat = self.tank_energy
        self.alpha_delta_energy_j = 0.0
        self.modulation_debit_j = 0.0
        self.feedback_age_s = float("nan")
        self.feedback_stale = True
        self._vp_history: list[float] = []
        self.mismatch_velocity_aligned = 0.0
        self.alpha_would_gate_m_s = 0.0
        self.sign_verified = bool(self.cfg.sign_verified)
        self.sign_fault = bool(
            self.cfg.require_sign_verification and not self.cfg.sign_verified
        )
        self.feedback_delay_verified = bool(self.cfg.feedback_delay_verified)
        self.active = False
        self.blocked_reason = ""
        self.gamma_effective = 0.0
        self.integral_position_error = 0.0
        self.proxy_mass_now = float(self.cfg.M_p)
        self.proxy_damping_now = float(self.cfg.D_p)
        self.nominal_damping_now = float(self.cfg.nominal_damping)
        self._prev_press_request = 0.0
        self._accounted_press_m_s = 0.0
        self._active_effort_budget_n = 0.0
        self._accounting_dt_s = self.dt
        self._initialized = False
        self.last_dt_actual = self.dt
        self.telemetry = BidirectionalFlowTelemetry(
            tank_energy=self.tank_energy,
            mode=self.cfg.mode,
            sign_verified=self.sign_verified,
            engineering_adaptation=self.ENGINEERING_ADAPTATION,
        )
        self._mirror_telemetry()

    def begin_episode(
        self,
        v_actual: float,
        *,
        tank_energy: float | None = None,
        energy_phys_j: float | None = None,
        energy_mismatch_j: float | None = None,
    ) -> None:
        """Clear episode transients without adding energy to the tank."""

        previous_tank = (
            float(self.tank_energy) if tank_energy is None else float(tank_energy)
        )
        previous_phys = (
            float(self.energy_phys_j)
            if energy_phys_j is None
            else float(energy_phys_j)
        )
        previous_mismatch = (
            float(self.energy_mismatch_j)
            if energy_mismatch_j is None
            else float(energy_mismatch_j)
        )
        if not np.isfinite(previous_tank):
            raise ValueError("tank energy must be finite at episode entry")
        if previous_tank < float(self.cfg.Tmin) - 1.0e-12:
            raise RuntimeError("tank energy is below Tmin at episode entry")
        seed = _finite(v_actual, 0.0)
        self.reset(x_actual=0.0)
        # A changed upper bound may remove available energy; no phase boundary
        # is allowed to raise the stored energy toward Tmin or T0.
        self.tank_energy = min(previous_tank, float(self.cfg.Tmax))
        self.energy_phys_j = previous_phys
        self.energy_mismatch_j = previous_mismatch
        self.xp = 0.0
        self.xa = 0.0
        self.aux_anchor = 0.0
        self.x_aux = 0.0
        self.x_safe = 0.0
        self.va = seed
        self.vp = seed
        self.v_track = seed
        self.command = seed
        self.v_aux = 0.0
        self.retract_through = min(seed, 0.0)
        self.press = max(seed, 0.0)
        self._prev_press_request = self.press
        # Re-arm conservatively. Stale feedback and the normal tank gate still
        # decide whether positive press is allowed on the first live tick.
        self.alpha = 1.0
        self.alpha_raw = 1.0
        self.alpha_case = "episode_entry"
        self.gamma_effective = 0.0
        self.Sn = self.tank_energy
        self.Sr_hat = self.tank_energy
        self.telemetry = BidirectionalFlowTelemetry(
            tank_energy=self.tank_energy,
            energy_phys_j=self.energy_phys_j,
            energy_mismatch_j=self.energy_mismatch_j,
            mode=self.cfg.mode,
            sign_verified=bool(self.cfg.sign_verified),
            feedback_stale=True,
            alpha=1.0,
            alpha_raw=1.0,
            alpha_case="episode_entry",
            engineering_adaptation=self.ENGINEERING_ADAPTATION,
        )
        self._mirror_telemetry()

    @property
    def mode(self) -> str:
        return self.cfg.mode

    @property
    def active_enabled(self) -> bool:
        return self.cfg.mode == "active" and (
            (bool(self.cfg.sign_verified) or not self.cfg.require_sign_verification)
            and (
                bool(self.cfg.feedback_delay_verified)
                or not self.cfg.require_delay_verification
            )
        )

    def _feedback_is_stale(
        self,
        *,
        v_actual: float | None,
        feedback_age_s: float | None,
        feedback_fresh: bool | float | None,
    ) -> bool:
        try:
            velocity_finite = v_actual is not None and np.isfinite(float(v_actual))
        except (TypeError, ValueError):
            velocity_finite = False
        if not velocity_finite:
            return True
        if feedback_fresh is not None:
            try:
                fresh = (
                    bool(feedback_fresh)
                    if isinstance(feedback_fresh, (bool, np.bool_))
                    else float(feedback_fresh) > 0.5
                )
            except (TypeError, ValueError):
                fresh = False
            if not fresh:
                return True
        if feedback_age_s is None:
            return False
        age = _finite(feedback_age_s, float("inf"))
        return (not np.isfinite(age)) or age > self.cfg.max_feedback_age_s

    def _vp_delayed(self, age_s: float, dt: float) -> float:
        """Proxy velocity resampled back to when ``va`` was measured."""
        hist = self._vp_history
        if not hist:
            return float(self.vp)
        age = _finite(age_s, 0.0)
        if not np.isfinite(age) or age <= 0.0 or dt <= 0.0:
            return float(hist[-1])
        # Cap at the staleness budget: beyond it the sample is rejected as
        # stale anyway, and reaching further back would fabricate a match.
        age = min(float(age), float(self.cfg.max_feedback_age_s))
        steps = int(round(age / dt))
        if steps <= 0:
            return float(hist[-1])
        idx = max(0, len(hist) - 1 - steps)
        return float(hist[idx])

    def _lee_alpha_raw(
        self,
        *,
        Pe: float,
        Pc: float,
        dt: float,
        stale: bool,
    ) -> tuple[float, str]:
        """Return the exact Lee ``P_e/P_c`` gate cases.

        Here ``Pe = (vp-va) F_g`` is the press-positive power at the real port
        and ``Pc = (vp-va) Fc`` is the mismatch-controller power.  The
        positive-power cases are intentionally asymmetric:

        ``0 < lambda Pc < Pe`` -> ``alpha=1``;
        ``0 < Pe < lambda Pc`` -> ``alpha=Pe/(lambda Pc)``;
        ``Pe <= 0`` or ``Pc <= 0`` -> ``alpha=0``.

        Tank-low and stale feedback are hard fail-closed overrides and return
        exactly ``alpha=1``.  ``Pc``/``Pe`` are *not* replaced by damping or
        an arbitrary press debit in this branch.
        """

        if stale:
            return 1.0, "stale"
        if self.tank_energy <= self.cfg.Tmin + 1.0e-12:
            return 1.0, "tank_low"
        if Pe <= 0.0 or Pc <= 0.0:
            return 0.0, "nonpositive"
        lam_pc = max(self.cfg.lambda_gain * Pc, 0.0)
        if lam_pc <= 0.0:
            return 0.0, "nonpositive"
        if lam_pc < Pe:
            return 1.0, "Pe"
        return float(np.clip(Pe / lam_pc, 0.0, 1.0)), "Pc"

    def _smooth_alpha(self, target: float, dt: float, *, hard: bool) -> float:
        target = float(np.clip(target, 0.0, 1.0))
        if hard:
            self.alpha = 1.0
            return self.alpha
        tau = self.cfg.alpha_attack_s if target > self.alpha else self.cfg.alpha_release_s
        if tau <= 1.0e-12:
            self.alpha = target
        else:
            self.alpha += float(np.clip(dt / tau, 0.0, 1.0)) * (target - self.alpha)
        self.alpha = float(np.clip(self.alpha, 0.0, 1.0))
        return self.alpha

    def _mirror_telemetry(self) -> None:
        t = self.telemetry
        t.xp = float(self.xp)
        t.vp = float(self.vp)
        t.xa = float(self.xa)
        t.va = float(self.va)
        t.fc = float(self.fc)
        t.v_track = float(self.v_track)
        t.v_aux = float(self.v_aux)
        t.retract_through = float(self.retract_through)
        t.press = float(self.press)
        t.command = float(self.command)
        t.alpha = float(self.alpha)
        t.alpha_raw = float(self.alpha_raw)
        t.alpha_case = str(self.alpha_case)
        t.tank_energy = float(self.tank_energy)
        t.tank_power_credit = float(self.tank_power_credit)
        t.tank_power_debit = float(self.tank_power_debit)
        t.psi = float(self.psi)
        t.tank_switch_cost = float(self.tank_switch_cost)
        t.Pe = float(self.Pe)
        t.Pc = float(self.Pc)
        t.P_phys = float(self.P_phys)
        t.P_mismatch = float(self.P_mismatch)
        t.energy_phys_j = float(self.energy_phys_j)
        t.energy_mismatch_j = float(self.energy_mismatch_j)
        t.Sn = float(self.Sn)
        t.Sr_hat = float(self.Sr_hat)
        t.alpha_delta_energy_j = float(self.alpha_delta_energy_j)
        t.modulation_debit_j = float(self.modulation_debit_j)
        t.feedback_age_s = float(self.feedback_age_s)
        t.feedback_stale = bool(self.feedback_stale)
        t.sign_verified = bool(self.sign_verified)
        t.sign_fault = bool(self.sign_fault)
        t.feedback_delay_verified = bool(self.feedback_delay_verified)
        t.mode = self.cfg.mode
        t.active = bool(self.active)
        t.blocked_reason = str(self.blocked_reason)
        t.gamma_effective = float(getattr(self, "gamma_effective", 0.0))
        t.engineering_adaptation = self.ENGINEERING_ADAPTATION

        # Upper-case and descriptive aliases are useful for existing loggers
        # and make the real-port/mismatch telemetry self-documenting.
        self.Fc = float(self.fc)
        self.Kp_error = float(self.cfg.Kp * (self.xp - self.xa))
        self.real_port_position = float(self.xa)
        self.real_port_velocity = float(self.va)
        self.mismatch_position = float(self.xp - self.xa)
        self.mismatch_velocity = float(self.vp - self.va)
        self.e = float(self.mismatch_position)
        self.edot = float(self.mismatch_velocity)
        self.x_aux = float(self.aux_anchor)
        self.x_safe = float(self.x_safe)
        self.alpha_gate = float(self.alpha)
        self.T = float(self.tank_energy)
        self.psi_tank = float(self.psi)
        self.tank_T = float(self.tank_energy)
        self.feedback_fresh = not bool(self.feedback_stale)
        self.retract_through_velocity = float(self.retract_through)
        self.press_velocity = float(self.press)
        self.v_cmd = float(self.command)
        self.Pphys = float(self.P_phys)
        self.Pmismatch = float(self.P_mismatch)
        self.E_phys = float(self.energy_phys_j)
        self.E_mismatch = float(self.energy_mismatch_j)
        self.cumulative_energy_phys_j = float(self.energy_phys_j)
        self.cumulative_energy_mismatch_j = float(self.energy_mismatch_j)

    def update(
        self,
        vp_cmd: float = 0.0,
        x_actual: float | None = None,
        v_actual: float | None = None,
        force: float = 0.0,
        dt_actual: float | None = None,
        *,
        feedback_age_s: float | None = None,
        feedback_fresh: bool | float | None = None,
        feedback_freshness: bool | float | None = None,
        nominal_damping: float | None = None,
        proxy_mass: float | None = None,
        proxy_damping: float | None = None,
        active_effort_n: float | None = None,
        **kwargs: Any,
    ) -> float:
        """Advance one scalar flow tick and return the normal command.

        Keyword aliases (``v_proxy``, ``v_p``, ``xa``, ``va``, ``sensor_age_s``)
        are accepted to ease integration with older loop code.  Unknown
        keywords are ignored intentionally; the controller is often called
        from a telemetry-rich loop with additional fields.
        """

        vp_cmd = _finite(
            kwargs.pop("v_proxy", kwargs.pop("v_p", kwargs.pop("vp", vp_cmd))),
            0.0,
        )
        proxy_position_input = kwargs.pop(
            "xp", kwargs.pop("proxy_position", None)
        )
        if x_actual is None:
            x_actual = kwargs.pop("xa", kwargs.pop("actual_position", None))
        if v_actual is None:
            v_actual = kwargs.pop("va", kwargs.pop("actual_velocity", None))
        if feedback_age_s is None:
            feedback_age_s = kwargs.pop("sensor_age_s", kwargs.pop("feedback_age", None))
        if feedback_freshness is not None and feedback_fresh is None:
            feedback_fresh = feedback_freshness
        if feedback_fresh is None:
            feedback_fresh = kwargs.pop("fresh", kwargs.pop("is_fresh", None))
        if force == 0.0:
            force = kwargs.pop(
                "F_g",
                kwargs.pop(
                    "f_g",
                    kwargs.pop("generalized_force", kwargs.pop("f_ext", force)),
                ),
            )
        force = _finite(force, 0.0)
        if dt_actual is None:
            dt_actual = kwargs.pop("dt", kwargs.pop("dt_s", None))
        if proxy_mass is None:
            proxy_mass = kwargs.pop("Mp", kwargs.pop("m_p", None))
        if proxy_damping is None:
            proxy_damping = kwargs.pop("Dp", kwargs.pop("d_p", None))
        if active_effort_n is None:
            active_effort_n = kwargs.pop(
                "active_effort", kwargs.pop("F_active", None)
            )

        dt = self.dt if dt_actual is None else _finite(dt_actual, self.dt)
        dt = float(np.clip(dt, 1.0e-6, 0.25))
        self.last_dt_actual = dt
        self.feedback_age_s = (
            float("nan") if feedback_age_s is None else _finite(feedback_age_s, float("inf"))
        )
        stale = self._feedback_is_stale(
            v_actual=v_actual,
            feedback_age_s=feedback_age_s,
            feedback_fresh=feedback_fresh,
        )
        self.feedback_stale = bool(stale)
        va = _finite(v_actual, self.va)
        if x_actual is None or not np.isfinite(float(x_actual)):
            xa = self.xa + va * dt
        else:
            xa = _finite(x_actual, self.xa)
        self.xa = xa
        self.va = va

        if not self._initialized:
            self.xp = (
                self.xa
                if proxy_position_input is None
                else _finite(proxy_position_input, self.xa)
            )
            self.aux_anchor = self.xa
            self.x_safe = self.xa
            self._initialized = True
        elif proxy_position_input is not None:
            self.xp = _finite(proxy_position_input, self.xp)

        # Force/mismatch feedback.  ``xp`` is the current proxy position; the
        # resulting ``vp`` is then integrated with wall-clock dt.  Solving the
        # one-step equation in closed form makes the -lambda*alpha*Fc update
        # genuinely implicit rather than an explicit force kick.
        dx = self.xp - self.xa
        self.integral_position_error += dx * dt
        # The proxy coupling uses the *previous* gate value.  This is the
        # causal one-tick form of the implicit Lee update; using an
        # unconditional lambda would inject a press correction while the
        # current gate is closed.
        alpha_prev = float(self.alpha)
        gain = self.cfg.lambda_gain * alpha_prev
        mp = max(
            _finite(
                self.cfg.M_p if proxy_mass is None else proxy_mass,
                self.cfg.M_p,
            ),
            1.0e-6,
        )
        dp = max(
            _finite(
                self.cfg.D_p if proxy_damping is None else proxy_damping,
                self.cfg.D_p,
            ),
            0.0,
        )
        A = mp / dt + dp
        self.proxy_mass_now = mp
        self.proxy_damping_now = dp
        # Reconstruct the nominal implicit-Euler RHS.  In particular, alpha=0
        # is exactly vp_cmd; only the gated mismatch coupling contributes when
        # alpha_prev is nonzero.
        denom = A + gain * self.cfg.Kd
        self.vp = (
            A * vp_cmd
            + gain
            * (
                self.cfg.Kd * self.va
                - self.cfg.Kp * dx
                - self.cfg.Ki * self.integral_position_error
            )
        ) / max(denom, 1.0e-9)
        self.fc = (
            self.cfg.Kd * (self.vp - self.va)
            + self.cfg.Kp * dx
            + self.cfg.Ki * self.integral_position_error
        )
        self.xp += self.vp * dt

        correction = float(
            np.clip(self.fc / self.cfg.Dtrack, -self.cfg.track_correction_max_m_s, self.cfg.track_correction_max_m_s)
        )
        self.v_track = self.vp + correction
        self.retract_through = min(self.v_track, 0.0)
        self.press = max(self.v_track, 0.0)

        # Independent one-sided real auxiliary.  While the press gate is
        # effectively open, x_safe follows the measured real port.  Once the
        # gate closes it freezes, and the implicit mass/damping update can only
        # produce retract velocity from K_a(x_safe-xa)-D_a*va+u_retract.
        if alpha_prev <= 1.0e-6:
            self.x_safe = self.xa
        aux_force = (
            self.cfg.K_a * (self.x_safe - self.xa)
            - self.cfg.B_a * self.va
            + self.cfg.u_retract_n
        )
        aux_den = self.cfg.M_a / dt + self.cfg.D_a
        self.v_aux = (
            (self.cfg.M_a / dt) * self.v_aux + aux_force
        ) / max(aux_den, 1.0e-9)
        self.v_aux = float(
            np.clip(self.v_aux, -self.cfg.aux_max_retract_m_s, 0.0)
        )
        self.aux_anchor += self.v_aux * dt

        nominal_d = (
            self.cfg.nominal_damping
            if nominal_damping is None
            else max(_finite(nominal_damping), 0.0)
        )
        self.nominal_damping_now = float(nominal_d)
        # Press-positive generalized force/port powers.  These are kept
        # separate from the tank's known nominal-damping credit below.
        #
        # ``edot`` is Lee's e_nr_dot and must compare the two ports at the same
        # instant.  ``va`` arrives one CANFD round trip late (15-20 ms here,
        # against a 20 ms staleness budget) while ``vp`` is current, so the raw
        # difference is dominated by transport lag rather than by energy
        # generation — alpha would then be measuring the link, not the contact.
        self._vp_history.append(float(self.vp))
        if len(self._vp_history) > self._VP_HISTORY_MAX:
            del self._vp_history[: -self._VP_HISTORY_MAX]
        vp_aligned = self._vp_delayed(self.feedback_age_s, dt)
        edot = vp_aligned - self.va
        self.mismatch_velocity_aligned = float(edot)
        self.P_phys = force * self.va
        self.P_mismatch = force * edot
        self.Pe = self.P_mismatch
        self.Pc = edot * self.fc
        self.energy_phys_j += self.P_phys * dt
        self.energy_mismatch_j += self.P_mismatch * dt

        # A discrete positive switch cost is optional bookkeeping; it must
        # not replace the per-tick alpha-flow debit required by the tank.
        switch_cost = 0.0
        if (
            self.press > self.cfg.press_epsilon_m_s
            and self._prev_press_request <= self.cfg.switch_epsilon_m_s
        ):
            switch_cost = self.cfg.positive_switching_cost_j
        raw_alpha, alpha_case = self._lee_alpha_raw(
            Pe=self.Pe,
            Pc=self.Pc,
            dt=dt,
            stale=stale,
        )
        # Lee Sec. V-C: "when the robot is moving in free space, alpha should
        # always be zero because there is no energy generation in the nominal
        # system."  Structurally Pe=0 without contact, but the paper's own
        # free-space run (Fig. 13) still saw alpha lifted by F/T noise at 4 kHz
        # with collocated sensing; this link is slower and delayed, so make it
        # explicit rather than hoping the power test stays clean.
        free_n = max(float(getattr(self.cfg, "free_space_force_n", 0.0)), 0.0)
        in_free_space = free_n > 0.0 and abs(float(force)) < free_n
        if in_free_space and not stale:
            raw_alpha, alpha_case = 0.0, "free_space"
        self.alpha_raw = float(raw_alpha)
        self.alpha_case = alpha_case
        # A drained tank must not force alpha=1 in free space: there is no
        # energy generation to gate, and alpha=1 there is exactly the
        # performance loss the paper warns about.  Stale feedback still is a
        # hard gate — an unknown port is not a safe port.
        hard_gate = stale or (
            self.tank_energy <= self.cfg.Tmin + 1.0e-12 and not in_free_space
        )
        alpha_before_smoothing = float(self.alpha)
        self._smooth_alpha(raw_alpha, dt, hard=hard_gate)

        sign_ok = bool(self.cfg.sign_verified) or not self.cfg.require_sign_verification
        self.sign_verified = bool(sign_ok)
        self.sign_fault = bool(self.cfg.require_sign_verification and not sign_ok)
        delay_ok = bool(self.cfg.feedback_delay_verified) or not self.cfg.require_delay_verification
        self.blocked_reason = ""
        if self.cfg.mode == "active" and not sign_ok:
            self.blocked_reason = "sign_unverified"
        elif self.cfg.mode == "active" and not delay_ok:
            self.blocked_reason = "feedback_delay_unverified"
        elif self.cfg.mode == "active" and stale:
            self.blocked_reason = "feedback_stale"

        active_effort_budget = max(
            _finite(active_effort_n, self.cfg.active_press_debit_n)
            if active_effort_n is not None
            else self.cfg.active_press_debit_n,
            0.0,
        )
        # Ki is disabled in the first release.  If a later configuration opts
        # in, its positive mismatch effort is an active term and must buy tank
        # energy instead of appearing as free proxy work.
        active_effort_budget += max(
            self.cfg.Ki * self.integral_position_error,
            0.0,
        )
        self._active_effort_budget_n = float(active_effort_budget)
        self._accounting_dt_s = float(dt)
        # Compute conservative storage/credit terms before selecting the
        # positive command so gamma is budget-limited, not retroactively
        # clipped after an overdraw.
        # Lee's S_n and Ŝ_r are the *same* scaled inertia (M_n = λM̂), so the
        # α-interpolated storage S = (1-α)S_n + αŜ_r is a single physical
        # quantity.  Using M_p=1.0 for one and M_a=0.05 for the other made
        # Ŝ_r - S_n a 20x scale artefact, so every α rise booked a positive
        # modulation debit and drained the tank one way.
        self.Sn = float(0.5 * self.proxy_mass_now * self.vp * self.vp)
        self.Sr_hat = float(
            max(
                0.5 * self.proxy_mass_now * self.va * self.va
                + 0.5
                * self.cfg.K_a
                * (self.x_safe - self.xa)
                * (self.x_safe - self.xa),
                0.0,
            )
        )
        credit_j = (
            (1.0 - self.alpha) * self.nominal_damping_now * self.vp * self.vp
            + self.alpha * self.cfg.D_a * self.va * self.va
        ) * dt
        if in_free_space:
            # Free-space motion is pure proxy damping dissipation, which is a
            # credit term in Lee eq. (32).  Without it the tank only ever sits
            # or falls and arrives at contact already empty.
            air_d = max(float(self.proxy_damping_now), float(self.cfg.D_p), 0.0)
            credit_j += air_d * self.vp * self.vp * dt
        delta_alpha = self.alpha - alpha_before_smoothing
        self.alpha_delta_energy_j = delta_alpha * (self.Sr_hat - self.Sn)
        self.modulation_debit_j = max(self.alpha_delta_energy_j, 0.0)
        fixed_debit_j = (
            (1.0 - self.alpha) * self.cfg.mu_power_w * dt
            + self.modulation_debit_j
            + (
                switch_cost
                if self.cfg.mode == "active"
                and sign_ok
                and delay_ok
                and not stale
                else 0.0
            )
        )
        # Shadow of what the gate would remove if it were driving.  In observe
        # this is the only way to judge alpha before handing it the command:
        # it must sit at zero in free space and rise only at force peaks.
        self.alpha_would_gate_m_s = float(
            max(self.press, 0.0)
            * (1.0 - (1.0 - self.alpha) * self.cfg.gamma_active)
        )
        if self.cfg.mode == "active" and sign_ok and delay_ok and not stale:
            # Retract-through is never alpha-gated.  Only the positive branch
            # is modulated by the tank and active gain.
            self.active = True
            requested_gain = (1.0 - self.alpha) * self.cfg.gamma_active
            if self.tank_energy <= self.cfg.Tmin + 1.0e-12:
                requested_gain = 0.0
            # Pre-limit positive velocity by the energy available this tick;
            # tank clipping below is then only a numerical guard, not a way
            # to hide an already-overdrawn command.
            available_j = max(
                self.tank_energy
                - self.cfg.Tmin
                + credit_j
                - fixed_debit_j,
                0.0,
            )
            cost_per_speed_j = active_effort_budget * dt
            if self.press > self.cfg.press_epsilon_m_s and cost_per_speed_j > 0.0:
                budget_press = available_j / cost_per_speed_j
                requested_gain = min(
                    requested_gain,
                    float(np.clip(budget_press / self.press, 0.0, 1.0)),
                )
            self.gamma_effective = float(np.clip(requested_gain, 0.0, 1.0))
            gated_press = self.gamma_effective * self.press
            self.command = self.v_aux + self.retract_through + gated_press
        elif self.cfg.mode == "active" and sign_ok and delay_ok:
            # Stale feedback/tank-low remains active for retract-through only.
            self.active = True
            self.gamma_effective = 0.0
            self.command = self.v_aux + self.retract_through
        elif self.cfg.mode == "active":
            # An unverified sign must never fall back to the positive legacy
            # command.  Fail closed to the one-sided retract path.
            self.active = True
            self.gamma_effective = 0.0
            self.command = self.v_aux + self.retract_through
        else:
            self.active = False
            # Observe computes the full state but must not alter the legacy
            # command.  Off does the same and keeps its telemetry harmless.
            self.command = vp_cmd

        # Tank bookkeeping is done after the command is known.  ``Sn`` and
        # ``Sr_hat`` are storage estimates, not aliases for tank fill:
        # proxy kinetic storage versus conservative real-port auxiliary
        # storage.  Only identified damping channels are credited.
        sent_press = max(self.command, 0.0) if self.active else 0.0
        self._accounted_press_m_s = float(sent_press)
        effort = active_effort_budget
        active_power_w = effort * sent_press
        active_press_debit_j = active_power_w * dt
        # fixed_debit_j already contains the conservative constant-power,
        # storage-interpolation, and optional switching terms computed before
        # command selection.
        debit_j = fixed_debit_j + active_press_debit_j
        self.tank_switch_cost = switch_cost if sent_press > 0.0 else 0.0
        self.tank_power_credit = credit_j / dt if dt > 0.0 else 0.0
        self.tank_power_debit = debit_j / dt if dt > 0.0 else 0.0
        self.psi = self.tank_power_credit - self.tank_power_debit
        self.tank_energy = float(
            np.clip(self.tank_energy + credit_j - debit_j, self.cfg.Tmin, self.cfg.Tmax)
        )
        self._prev_press_request = self.press
        self._mirror_telemetry()
        return float(self.command)

    def settle_applied_press(self, applied_press_m_s: float) -> float:
        """Charge any positive speed added after the flow controller.

        The outer force-axis slew can retain more press than ``command`` when
        the flow gate closes.  That extra real command must buy energy from the
        same tank.  Retraction and reductions never receive a refund.
        """

        requested = max(_finite(applied_press_m_s, 0.0), 0.0)
        if not self.active:
            return requested
        extra = max(requested - float(self._accounted_press_m_s), 0.0)
        effort = max(float(self._active_effort_budget_n), 0.0)
        dt = max(float(self._accounting_dt_s), 1.0e-9)
        if extra <= 0.0 or effort <= 0.0:
            return requested
        available = max(float(self.tank_energy) - float(self.cfg.Tmin), 0.0)
        allowed_extra = min(extra, available / (effort * dt))
        debit = allowed_extra * effort * dt
        self.tank_energy = max(float(self.cfg.Tmin), float(self.tank_energy) - debit)
        self.tank_power_debit += debit / dt
        self.psi = self.tank_power_credit - self.tank_power_debit
        self._accounted_press_m_s += allowed_extra
        self._mirror_telemetry()
        return min(requested, float(self._accounted_press_m_s))

    # Common aliases used by small simulation harnesses.
    step = update
    compute = update
    compute_velocity = update
    compute_velocity_command = update


# Descriptive aliases used by a few standalone simulation harnesses.
BidirectionalFlowCore = BidirectionalFlowController
BidirectionalEnergyFlowController = BidirectionalFlowController


__all__ = [
    "BidirectionalFlowConfig",
    "BidirectionalFlowController",
    "BidirectionalFlowCore",
    "BidirectionalEnergyFlowController",
    "BidirectionalFlowTelemetry",
]
```

