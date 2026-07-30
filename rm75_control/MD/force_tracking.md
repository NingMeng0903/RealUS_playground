# TCP-Z 恒力控制

当前只有一套可运行的力控制器：低固定阻尼 + 变惯量 + 刚度归一化前馈 +
力空间速度阻尼器。配置入口是 `configs/joint_admittance_8dof.yaml` 的
`hybrid_motion`；`damping_law: ke_critical` 可一键回到旧的 Ke 临界阻尼路径。

## 控制结构

- RealMan TCP 同步决定当前工具坐标系，力方向仍为 TCP-Z；
- TCP-X/Y 与姿态跟踪，TCP-Z 执行质量–阻尼导纳（隐式欧拉）；
- 滞回接触锁存：`|fz| > contact_threshold_n` 进入，低于
  `contact_release_n` 连续 `contact_release_ticks` 拍才释放；释放时清零
  力建立计时与 proactive 积分器，**每段接触重新 ramp**；
- 自由段寻面：`seek_vz_m_s` + `seek_force_sat_n`，下探速度与 `f_des` 解耦；
- `K̂e` 主要用于归一化前馈增益，**不再决定默认阻尼**；
- Dimeas `Iₛ` 只抬高虚拟惯量（`var_damping_d_u: 0`）；
- 力空间 Faverjon 速度阻尼器同时夹压入 / 回撤速度，不依赖 `K̂e`。

## 阻尼律（默认 `damping_law: trend`）

```text
e   = smooth_deadband(Fdes_eff - Fext)
ė   = LPF_20ms(de/dt)
b_d = b_base + α|e| + β·max(0, e·ė)     # clamp 到 damping_max_z
M   = M_base + m_u · Iₛ                 # clamp 到 m_max
v   = (M v + Δt (e_clamped + b_d v_r)) / (M + b_d Δt)   # 隐式欧拉
```

`ke_critical` 路径仍可用：`b_d = 2ζ√(M K̂e) + d_u·Iₛ`，便于 A/B。

## 主动前馈（默认 `proactive_gain_mode: ke_normalized`）

```text
欠力 / 压入:   ke = ke_floor_ff
过力 / 回撤:   ke = max(K̂e, ke_floor_ff)
v_r_target   = clip(e / (ke · τ_ff), ±v_r_max)
v_r         ← (1-α) v_r + α v_r_target          # τ_track
泄漏         = −v_r/leak_s − α_leak |v_r| v_r   # Li 幅值耦合
```

压入方向仍受 Dimeas `press_is_gate` 衰减；过力回撤不衰减。
`proactive_gain_mode: fixed` 回到旧的 setpoint 归一化增益。

## 力空间速度阻尼器

```text
budget = max(budget_min, budget_frac · |Fdes|)
f_pred = fz + ḟ · t_react
cap_press   = clip((Fdes + budget - f_pred)/budget · v_ref, 0, v_z_cap)
cap_retract = clip((f_pred - f_keep)/budget · v_ref, v_min, v_z_cap)
```

`Fdes ≈ 0`（手推导引）时不夹回撤，避免把轻阻尼手感夹死。
`t_react_s` 与 `force.causal_fc_hz=10` 对齐（约 30 ms）。

## 调参要点

| 手感问题 | 先动 |
|---|---|
| 越推越顶（desired=0） | `damping_base_z`、确认 `f_err_gate_floor_n≥3`、barrier 在 Fdes=0 旁路 |
| 下探随目标力变快 | `seek_vz_m_s` / `seek_force_sat_n` |
| 硬面弹跳 | `damping_beta_e_edot`、`force_barrier.v_ref_m_s`、`tau_ff_s`、`m_u` |
| 软面追不上 | `ke_floor_ff`、`tau_ff_s`↓、`damping_alpha_e`↓ |
| 换表面再接触过冲 | `contact_release_*`、`desired_force_ramp_s` |

质量、Dimeas 力尺度和阻尼上限不随运行时目标力改变。

## CSV 调试列

- `fz`、`f_des_z_eff`：实测力与建立后的目标力；
- `v_force_z`、`v_r_z`：导纳速度与主动参考；
- `force_reference_scale_n`、`force_reference_drive`、
  `force_reference_gate_scale`、`force_reference_accel_m_s2`、
  `force_reference_reversal_reset`：前馈内部量；
- `damping_z_eff`、`damping_trend_z`、`damping_ke_z`、`damping_dimeas_z`：
  实际阻尼 / 趋势阻尼 / 旧临界阻尼对照 / Dimeas 阻尼项；
- `cap_press_z`、`cap_retract_z`、`f_dot_z`：力空间速度阻尼器；
- `mass_z_eff`、`instability_idx`、`ke_est`；
- `contact_present`、`vz_achieved_tool`。

## 硬件验证清单

软 / 硬假体各跑 1 N、2 N、5 N，再过一次肋骨状边缘；保留全量 CSV，重点比对：

1. `damping_z_eff` 是否长期停在 `damping_base_z` 附近（手推 desired=0 不应随推力爬升）；
2. `mass_z_eff` 是否只在边缘 / 振荡时抬升；
3. `v_r_z` 在欠力追赶与过力回撤是否对称、硬面是否被 Ke 归一化压小；
4. `cap_press_z` / `cap_retract_z` 在过力时是否收紧；
5. `instability_idx` 与接触损失次数。
