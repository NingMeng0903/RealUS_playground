---
name: force control delay budget
overview: 删掉 force_dob 与 proactive_ff 这两个滞后网络（它们是 2–3 Hz 极限环的线性根因），修好把追力冻死的 F*=2 配置错位，然后只在 Ke 超过阈值时按延迟预算 D=Ke·4T/π 单边抬阻尼，并保留一个屈服地板 D_floor 防止软组织回退丢接触。
todos:
  - id: stage0-frf
    content: Stage 0a：接触内 FRF 扫频（1→8 Hz，恒位移，软垫/1420/2199 三个面）取 T_eff、Ke、探头惯量拐点。门：T_eff ∈ [30,60] ms
    status: pending
  - id: stage0-tissue-ke
    content: Stage 0b：志愿者腹部与甲状腺在 1 N 下的驱动点刚度 ΔF/Δx（行程 ≥ 0.3 mm，Stage-1 构型）。这是决定硬侧调度该不该武装的唯一一个数
    status: pending
  - id: stage0-mass
    content: Stage 0c：称传感器远端质量。> 0.8 kg 则 D_floor 从 12 起
    status: pending
  - id: stage1-delete-lags
    content: Stage 1：只关 force_dob 与 proactive_feedforward。门：fz 的 2–3.5 Hz 带内功率降 ≥ 10 dB 且峰值移到 3.2 Hz 以上，否则回滚
    status: pending
  - id: stage2-unfreeze
    content: Stage 2：force.yaml 的 desired_z_n 改 1.0 并让加载器对不一致报错；删两个慢闩、recontact_hold_s、recontact_settle_m_s；死区改 0.03/0.05；压入上限换成 Ke 调度的 ΔF_budget/(K̂e·T_eff)
    status: pending
  - id: stage3-schedule
    content: Stage 3：上位姿基的 TLS 刚度估计器（带指令一致性门）与 D = max(D_floor, K̂e·4T_eff/π) 单边调度；环振保护必须是双向的。门：三面 GM ≥ 4.5 dB，90% 上升 ≤ 300 ms
    status: pending
  - id: fix-mass-matrix
    content: 修 model.py 的 M + M.T - diag(M)（CRBA 本已对称，导致质量矩阵不定且 use_mass_weighted_reg 正在用它）
    status: pending
  - id: fix-latch-bugs
    content: 修 force_ok 比未斜坡 f_des、_first_contact_slow_latched 死锁、instability_index 归一化、adaptive_ke 的 displacement_source
    status: pending
isProject: false
---


# 力控方案：删滞后 + 单边延迟预算阻尼（带屈服地板）

## 结论：没有一篇论文可以直接用作变更律

六篇里唯一**结构**吻合的是 Keemink 2018（外环导纳 + 内环速度环，位置伺服黑盒）。但它给出的是约束而不是方案：不要滤力（相对阶 ≤1）、没有力矩口就无法被动降惯量、不要补偿传感器后质量。其余五篇在这个口上都是死路，且大部分已经在仓库里试过并留下了负面记录：

- De Stefano 2020 Sec. IV（`tdpa.py`，`apply:false`）：`Fc = Fe − α v` 灌进 `v=(Fd−F)/D` 时，只要 `α > D` 离散极点 `λ = a + bα > 1`，**D 直接反号**。yaml 里 "apply:true with alpha_max=400 inverts D (222808)" 就是这件事。
- Samuel 的 CDYOB（`cdyob.py`，`mode: off`）：原文 `t_d=3 ms / 1250 Hz / B_a=400–800`。按同样的 `ω_Q·t_d` 鲁棒性乘积缩放到 35 ms，Q 只能放在 **0.8–1.3 Hz**，碰不到 2–3 Hz。yaml 已记录实测把 `Ke*` 从 1059 压到 775。
- Secchi 2019 / Benzi 2021 的 tank：可以写成 QP 里一行 `T_k + dt·F^T J q̇ ≥ ε`，但把力从 0 追到 1 N 本身就是**抽取**能量（`0.5/Ke` = 软组织 6.3 mJ），tank 空了会同时堵住追力和回退；抬 `T_bar` 就是你禁止的假补能。
- Lee 2024 BEFM（`bidirectional_flow.py`，`observe`）：`τ_m` 是电机力矩，`S_r` 要 `M̂`。这台机器没有力矩口，`M̂` 是隐藏伺服的闭环而不是 URDF。tank 是 1 mJ = 1 N 下 1 mm，数值上是个开关。
- Benzi 的变阻尼其实不存在（实验用**恒定** D=0.05）；仓库里活着的 `var_damping_*` 是 Dimeas 2016 的 `I_s`，而且 `omega_c_hz: 2.5` 正好压在极限环上。

真正的答案是回路整形加一条延迟预算规则，不是论文拼装。

## 诊断（日志证实，不是推测）

`rm75_control/apps/logs/peirastic/` 里的接触段给出三件决定性事实：

- `proactive_ff` 是**滞后网络**（`leak_s 0.25` → 极点 0.64 Hz），`force_dob` 是第二个（0.40 s）。两者经 `rhs = eff + u_dob + D0*v_r` 并联注入后，回路相位穿越从约 2.8 Hz 掉到 **1.08 Hz**，在 Ke=80 留 **+4.5 dB**、Ke=2199 留 **+36.3 dB** 的过剩增益。**软组织上也是线性不稳定的。** 观测到的 2–3 Hz 是这个不稳定回路被 `v_r_max` 饱和、并被 `reset_on_reversal` 每秒倒空约 4 次之后的描述函数极限环。
- 追力一直是**冻结**的。`peirastic/configs/force.yaml` 第 6 行 `desired_z_n: 2.0`，而 `rm75_control/configs/joint_admittance_8dof.yaml` 是 1.0。first-contact 慢闩的释放条件是 `F ≥ 0.70·F*`（`controller.py:1539`）= 1.4 N，1 N 保持永远够不到。四个 8-29 日志里 `recontact_slow_latched` 在接触时间里占 80–100%，`u_dob_z` 恒为 0.000。同期压入上限是 `_v_delay_safe` 给的 **2.8 mm/s**（`1.80/(8000×0.080)`），不是大家一直在争的 8 mm/s。
- 硬件跑的 `damping_z_eff` 在弹跳战役期间是 **40**，现在 yaml 出货 25。25 在 8 月中跑过（`run_20260822_011052` 等）。也就是说：**被验证过的构型在振，被相信为好的构型从未被真正跑过。** `force.yaml` 里的所有 DOB/FF 增益都没有硬件支撑。

排除掉的机制：接触丢失时钟（2.36 Hz 振荡段 21.6 s 内 `physical_contact_loss_event` = 0）；Dimeas 调阻尼（`damping_dimeas_z` 在每个日志里恒为 0）；零空间耗散（`J q̇_N = 0` ⇒ `P_N ≡ 0`，无力矩口）；导轨作为力轴执行器（`|J_z,rail| = 0.00900`，在整个 `psi_envelope [40,110]` 上是 0.008995–0.008997，120 mm/s 导轨只给 1.08 mm/s 的 tool-Z）。

## 关键量化事实：删掉两个滞后后，D=25 在腹部组织上就够了

裸回路（M=1、探头 0.53 kg、tp=12 ms、实测 20 Hz 因果滤波）的增益裕度：

- Ke=80：T=35/45/55 ms 下 +9.6 / +9.2 / +9.0 dB
- Ke=300：+12.1 / +8.8 / +6.6 dB
- Ke=500：+4.3 / +2.1 / +0.4 dB
- Ke=800：−1.2 / −3.1 / −4.5 dB
- Ke=2199：−11.3 / −12.8 / −14.1 dB

所以在 80–300 N/m 上你**同时**拿到 `D=25`（1 N 推 39.9 mm/s）和不振。「柔顺」不需要靠调度换。只有 500 N/m 以上（甲状腺 / 绷紧腹壁 / 肋骨 / 你那两块 1420 与 2199 N/m 的治具）才真的需要动 D。

`|Z_robot| = |jωM + D|` 的换手点是 **333 N/m**：低于它调度让手感更软，高于它就是在抬阻尼。这就是必须先测量的东西。

## 最重要的一条否决：D 太小会在回退时丢接触

之前所有方案都只算了压入侧的稳定性，没有一个闭上**屈服侧**。在 Ke=80、T=45 ms、`D=6`（也就是 `D=clip(0.075·Ke, 4, 200)` 给出的值）下：

- 首次接触峰值冲到 **1.65 N**（目标 1 N）
- 从 1 N 回退时 `F_min ≈ −0.54 N`，即**丢接触**

要让 Ke=80 的回退不穿零，需要 **D ≥ 14.5**。所以 `D_min = 4`（或 6）违反需求 1，必须换成屈服地板 `D_floor ≈ 15–18`。

## 方案：Stage-1 + 带屈服地板的单边延迟预算阻尼

一条一阶导纳，无并联补偿器：

```
M v̇ + D(K̂e) v = deadband(F* − F)
D = clip( max(D_floor, K̂e · τ_f), D_floor, 200 )
τ_f = 4·T_eff/π            # T_eff 由 Stage 0 实测；57 ms → τ_f = 72.6 ms
D_floor = 15 … 18          # 由 Ke=80 回退不穿零定出，不是拍的
M = 1.0 kg                 # 不动；它给出稳定探头惯量支路的 8–14 Hz 滚降
```

`τ = D/Ke = τ_f` 是力环时间常数，所以**同一个数同时定住柔顺度和追力速度**，唯一与两者对立的量是延迟。这是选它而不是选 Smith 预测器或在线带阻抗调度的理由：`K̂e` 偏低 2 倍只留 +0.6 至 +1.4 dB 过剩增益（缓慢可见的增长），而不是现在这种 +36 dB。

保留仓库里已有的精确 ZOH（`a = exp(−D·dt/M)`，`b = (1−a)/D`）——D 到 200 时 `a=0.368`，正是它让硬支路在 200 Hz 上还安全。抗饱和从**实际**口速度 `n^T J q̇`（`vz_achieved_tool`）播种，而不是从发出的指令。

单边：`K̂e` 低于约 240–333 N/m 时 D 就停在 `D_floor`，调度只在硬侧起作用。

### 刚度估计（替换 `adaptive_ke`）

每个 episode，0.5 s 滑窗，用**实测位姿**投影到 tool-Z（不是 `displacement_source: admittance`，那个积分的是指令速度）与滤波后的力：

- 接受条件：行程 ≥ 0.3 mm，`|ΔF| ≥ 0.15 N`，接触存在，且指令一致性 `|Δx_meas − Σ u_sent·dt| ≤ 0.25|Δx_meas| + 0.1 mm`
- 这条一致性门是让**人手推动对估计器不可见**的机制：人推产生的是控制器没有指令的运动
- `Ke_win` 取 F–x 的全最小二乘斜率（2×2 协方差主特征向量），不是逐拍 `|ΔF/Δx|`
- episode 内单调上行 `K̂e ← max(K̂e, Ke_win)`；下行只走 5 s 衰减且需连续五个更低的接受窗（5 s = 0.032 Hz，比任何调度—振荡 hunt 回路低三个数量级）
- episode 起始 stiff-first 取 `Ke_max`，由第一个接受窗释放，典型 0.3 mm 压入内

### 环振棘轮：**不按原样实现**

原提案是单向的：`K̂e ← 1.5 K̂e`，最多四次（累计 5.06×），episode 内无回路。那正是你禁止的「把阻尼抬上去就不放」。要么删掉，要么给一条下行路径。它的功率符号判据也不干净：我实测 `fz/u_sent` 在 2.36 Hz 处已在 −132°，离符号边界很近，人手推与自激环在这里分不开。

## 分阶段执行，每级带门

- **Stage 0（只测量，不改代码）**
  - 0a 接触内 FRF——这个数从来没测过。`F*=1 N` 保持，在力轴速度上叠 1→8 Hz 对数扫频 30 s，幅度按**恒位移**排（治具 0.15 mm、组织 3 mm，使 ΔF ≤ 0.35 N）。在软垫、1420 治具、2199 治具各跑一次。由相位斜率取 `T_eff`，由 `|F/u|·ω` 取 Ke，由 `|F/u|` 停止按 1/ω 下降的位置取探头惯量拐点。**门：`T_eff ∈ [30,60] ms`。** 超过 60 ms 则所有 D 按 `T_eff/45ms` 缩放，硬侧不加末端轴就无法出货。
  - 0b **驱动点刚度**——决定硬侧调度到底该不该武装的唯一一个数。同一位姿、Stage-1 构型（补偿器关、D=25）、1 N 下测志愿者腹部**和**甲状腺的 `ΔF/Δx`，行程 ≥ 0.3 mm。**门：若两处都在 80–300 N/m，硬侧调度阈值放在 400–500，`D_floor` 就是常态；若任一处 ≥ 500 N/m，调度会落在换手点的硬侧，必须先接受手感损失或直接跳到末端轴。**
  - 0c 称传感器远端的质量。**门：> 0.8 kg 则 `D_floor` 取 12 起，因为 Ke=80 的裕度会从 +8.2 掉到 +2.4 dB，且约束穿越移到 8.2 Hz。**

- **Stage 1（一个下午，不会让机器更不安全）**
  只设 `force_dob.enabled: false` 和 `proactive_feedforward: false`，其余不动。这两块在慢闩期间本来就被 `chase_live` 关掉（`controller.py:2427-2447` 配 `860-871`），所以**不会让 2.8 mm/s 的上限提前解除**。**门：`fz` 的 2–3.5 Hz 带内功率下降 ≥ 10 dB 且峰值移到 3.2 Hz 以上。** 不满足就回滚，去插桩状态机——诊断错了。
  注意一个非严格性：`u_dob`/`v_r` 的回退侧也有 ±3.0 N，`overforce_escape` 现在是保留负值的。删掉后 cap 解除之后的过冲要看一眼；组织上 D=25 时峰值约 0.97 N，风险小，但要记录。

- **Stage 2（解冻）**
  `peirastic/configs/force.yaml` 的 `desired_z_n` 改 1.0，并让加载器在两份 yaml 不一致时**报错**而不是静默取一个。删掉两个慢闩（first-contact 与 recontact），删 `recontact_hold_s: 0.22`（上升沿武装，需要它的时候从不触发）与 `recontact_settle_m_s: 0.003`（压入时不可满足，是个死锁）。死区 0.08/0.10 → 0.03/0.05（实测静止力噪声 σ = 0.0151 N，1–3.5 Hz 带内 RMS 0.0064 N，0.08 是 5σ）。慢闩的安全功能由 Ke 调度的压入上限 `v_press ≤ ΔF_budget/(K̂e·T_eff)` 接手。仍然 D=25，**还没有调度**。**门：组织上无接触丢失事件，且 Ke≈80 的 90% 上升时间落在模型预测的区间内。** 明显更快说明被辨识的对象错了，回滚重做 0a。

- **Stage 3（单边调度）**
  上估计器、`D = max(D_floor, K̂e·τ_f)`（`τ_f` 用 Stage 0 的数）、以及**双向**的环振保护。**门：带调度重跑 0a 扫频，三个面上 GM ≥ 4.5 dB；三个面上 90% 上升 ≤ 300 ms；2 Hz 处 `|Z|` 与 D 相差 ≤ 1 dB。** 安静保持时每 episode 棘轮超过两次就回滚到 Stage 2。

- **Stage 4** 末端高带宽轴。

## 手感与追力契约（`|Z_robot| = |jωM+D|`，N·s/m，括号内是 1 N 推的 mm/s）

- 今天裸 D=25：0.3 Hz 25.1 (39.9)、1 Hz 25.8 (38.8)、2 Hz 28.0 (35.7)、3 Hz 31.3 (31.9)
- 今天三补偿器活着：2.3 (437) / 4.2 (241) / 7.9 (127) / 12.3 (81)——这是一个**在振**的系统的柔顺度，不是可用工作点
- 本方案 Ke=80，D=D_floor=18：约 18.0 (55.5) / 18.2 (55.0) / 19.4 (51.5) / 21.7 (46.1)
- 本方案 Ke=2000，D=150：150.0 (6.7) / 150.1 (6.7) / 150.5 (6.6) / 151.2 (6.6)

**必须让你看到的削弱：在肋骨、绷紧肌肉、或你那两块治具上，本方案比今天的标称硬 4–6 倍**（Ke=1420 是 −11.7 dB，Ke=2199 是 −15.4 dB）。理由是今天的标称在那些面上是 +7 至 +37 dB 不稳定，柔顺度是花不出去的。唯一同时买回两边的旋钮是降 `T_eff`，也就是末端轴。

追力（同一 plant 仿真，45 ms 延迟、12 ms、20 Hz、0.53 kg、死区、80 mm/s 上限），63% / 90% / 从 1 N 退到 0.1 N：

- Ke=80，D=18：约 190 / 250 / 250 ms（D=6 会给更快的数字，但代价是过冲 1.65 N 和回退穿零到 −0.54 N，不接受）
- Ke=300，D=22.5：98 / 124 / 134 ms
- Ke=2000，D=150：66 / 90 / 100 ms
- 今天 D=25、Ke=80：292 / 584 ms（而 8-29 的实际构型是被闩死的 2.8 mm/s，所以现场体感远比这更慢）

`4T_eff/π ≈ 70 ms` 的每 e 折是这个口的地板，也就是 90% 大约 200–290 ms，任何面上都一样。要更快只能上末端轴。

## 明确禁止（可以拿这份清单来卡我）

- 不复用的数：`system_delay_s: 0.055`（自己注释说是占位）；`force.fc_hz: 6.0` 在任何相位预算里（在线的是 20 Hz 一阶、2–3 Hz 处约 7.6 ms，`latest_wrench()` 零调用者）；`M_eff = 2.10 kg`；探头质量 1.10 kg（静态挂重，辨识出的 φ 质量是 0.529 kg）；`|J_z,rail| = 0.009` 当作可用的 Z 权限；`kin.mass_matrix` 当作 SPD（eigmin = −0.092）；3 Hz 群延迟散点 65/49/20 ms 当作 T0；`damping_z_eff = 25` 当作「机器一直在跑的值」（弹跳期日志是 40）
- 不做的断言：2–3 Hz 是接触丢失时钟（振荡段 21.6 s 内 0 次丢失）；Dimeas 在调阻尼（`damping_dimeas_z` 恒 0）；零空间能耗散接触功率；导轨能分到 `v_z` 的一个频段；抬 D 是手感方案；`force.yaml` 里任何 DOB/FF 增益有硬件支撑；末端轴只是把 `T_eff` 换个数（串联线圈是**另一个 plant**：两个质量、一个串联弹簧、第二个延迟，且腕部力传感器变成透过线圈测量）
- 不开的块：`tdpa.apply`、`safety_shield` 超出 `observe`、`cdyob` 任何模式、`bidirectional_flow` 超出 `observe`、`force_corridor`、`adaptive_ke.drive_damping`、`surface_force_modulation`，以及任何形式的能量罐

## 顺带要修的 bug（与控制决策无关，但都是活的）

- `joint_admittance_8dof/model.py:367`——`M + M.T - diag(M)` 作用在**本来就对称**的 Pinocchio CRBA 上，把非对角翻倍，矩阵变成不定（eigmin = −0.092）。因为 `use_mass_weighted_reg: true` 它是活的。直接返回 `M`。`joint_admittance/model.py:246` 同一个 bug
- `peirastic/configs/force.yaml:6` `desired_z_n: 2.0` 对 `configs/joint_admittance_8dof.yaml` 的 1.0——改 1.0 并让加载器在不一致时报错
- `controller.py:1538` `force_ok` 拿实测力和**未斜坡**的 `f_des[2]` 比，而回路追的是斜坡后的 `f_des_z_eff`，斜坡期间释放阈值不可达
- `controller.py:449` `_first_contact_slow_latched` 初值为 `True`，释放需 `|v_tcp| ≤ 3 mm/s`，压入时不可能成立——构造性死锁
- `controller.py:2319` `instability_index` 是 λ=0.951 的漏积分器（DC 增益 20.4，日志最大 6.3），而 `freeze_is 0.45`、`press_is_soft_stop 0.85`、`hf_on 0.30` 都把它当 [0,1] 用
- `adaptive_ke.py:259` `displacement_source: "admittance"` 积分的是指令速度，认证过的 2.8 mm/s 偏差会累进 Δx——改成 `"pose"`
- `fail_qdot_decay: 0.85` 被解析但从不施加（QP1 失败是 `q̇ = 0`）；`rail_allocator.reaction_s = 0.06` 名字像 m/s 实际是秒
