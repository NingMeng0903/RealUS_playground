# 2026-08-31 力轴结构对照：e85 / 63401843 / 现网 / Keemink 2018

先写结论，再对照源码。**本篇只分析，不改控制器、不改 yaml。**

三个 git 点：

- `e85c9ab957c78d500c3746159f1f47f4cabe3f0a`：力轴法在 `joint_admittance_8dof.yaml` + 951 行 `controller.py`。没有 peirastic `force.yaml`，没有 barrier/corridor/TDPA/R2。
- `6340184379a151aee5bf49ad3a77cde46e96c3f9`（「CDYOD 有点意思，硬有点不弹」）：peirastic 栈，`D=40`，chase 关，barrier+delay-safe，Z 跟位，`legacy.py` 有 R1。
- **现网（工作树）**：e85 力轴数嵌回 peirastic+QPIK；发出是 `u_sent`/`v_force_cmd_z`（R1 已修）；空中 R2；证书观察。

已测植物（`MD/todo_controller_logs/id_fit_20260829/identification.json`）：空中 chirp `T_0=35 ms`，`T_p=12 ms`。这是 `v_cmd→v`，**不是** Keemink 的 `Y_a=v/F_ext`。力环 FRF 标明未辨识。

---

## 0. 一句话

| 版本 | 为什么那样 | 不是因为 |
|---|---|---|
| e85 深得快、手感轻 | 确认接触后 `v≈e_f/D+v_r` 真发出，`D=25`，帽 ±80，力轴不跟位 | 后来的 R1 旁路（e85 没有 peirastic emit 旁路） |
| 634 少弹 | 稳态更钝 + 首触更慢 + Z 位 P 往回拉；导纳状态里已经被 delay-safe 卡住 | CDYOB 改命令（shadow 不 apply）；也不是「无源证过」 |
| 现网轻、有点弹 | 恢复了 e85 的发出，但 `ke_impact_initial` 从 e85 的 1500 变成了 0；Dimeas ΔD 在大误差/ramp 上是关的 | 没有按 Keemink 做成可证无源的低惯量渲染 |

好方案（仍不动代码）：确认后保持 e85 ±80；未确认/再贴用短袖套减 `ΔF`；颤起来加 `m` 不加稳态 `D`；**不要**把 Keemink Eq.(12) 的 `k_a` 当成 `e^{-sT_d}` 补偿。

---

## 1. 三个版本在 Keemink 框图里站哪

Keemink 的骨架是

```
F_ext → Y_v → v_d → C（内环速度）→ robot → v
人摸到的是 Y_a = v / F_ext，不是你写的 Y_v。
```

现网/e85/634 都是：**速度接口工业臂**。没有力矩口，做不了 Guideline 1 的 `C_ff = μ_ff s + β_ff`。内环 `C` 是 RealMan 伺服 + QPIK，外环加不了 `k_p,k_i`。

外环虚拟导纳（`Kc=0`）都是 Guideline 4 那种 mass+damping，不是纯惯量 `1/(m_v s)`：

\[
Y_v(s)=\frac{1}{Ms+D}.
\]

- e85 / 现网：`M=1`，`D=25` → `|Y_v|` 大，轻。
- 634：`M=1`，`D=40` → 同样 `e_f` 只有 25/40 的速度。

人碰到的 `Y_a` 还要乘上植物 `e^{-sT_0}/(T_p s+1)`、力滤波、饱和、QPIK。`T_0=35 ms` 在 8 Hz 上已经约 `-101°`。`Y_v=1/(j\omega+25)` 自己再滞后约 `64°`。接触共振上 `\operatorname{Re}\{Y_a\}<0` **本来就不会成立**。Keemink 的解析无源是单自由度、连续时间、一口、LTI；**不能**拿来证明 200 Hz / 35 ms / 6 轴无源。

七条对照现网（e85 外环数 + 现网栈）：

- **G1 前馈：** 无。`v_r`、DOB 不是 `μ_ff`。
- **G2 少滤力：** 部分。观测器 20 Hz、1 阶；导纳误差上不再加 LPF。
- **G3 探头惯量：** `use_inertia: false`。论文也说补 `m_ps` 会掉耦合裕度。
- **G4 虚拟阻尼：** 有，`D=25`。唯一认真对齐的一条。它**不能**从无源上再降低 `M`。
- **G5 `k_a` 改参考：** 无。`v_r` 不是 `(1+k_a s)Y_v^0`。Eq.(12) 最多 `<90°` 超前，抵消不了 8–14 Hz 上已经 `>90°` 的延迟。
- **G6 内环带宽：** 加不了。已测 `T_0` 把坏相位放在约 7–8 Hz。
- **G7 机器人刚度：** 探头–FT–垫非共位，没按 `γ,k_s` 建模。

---

## 2. 外环定律（源码）

### 2.1 e85（`e85c9ab`）隐式欧拉

文件：`rm75_control/.../admittance_common/controller.py`（当时 951 行）。头注释：

```
M0 · v̇ + (D0 + ΔD_hf) · (v − v_r) = e_f + u_DOB
```

积分（当时 `_admittance_z`）：

```python
# Implicit Euler: (M/dt + D) v+ = M/dt · v + D · v_r + drive
denom = mass_z / dt_eff + max(damping, 0.0)
velocity = (
    (mass_z / dt_eff) * self.v_force_z
    + max(damping, 0.0) * v_reference
    + drive
) / max(denom, 1e-6)
```

准稳态：`v = v_r + e_f/D`（再加 `u_dob`）。`D=25`、`e_f=2 N` → 单靠误差已是 **80 mm/s**。

yaml（`joint_admittance_8dof.yaml` hybrid_motion）：

- `track_axes[2]=0`，`kp_pos[2]=0`（XY 当时是 2，不是现网 10）
- `D=25`，`max_vz=0.08`，`max_velocity[2]=0.1`
- DOB `ki=8`，`v_r` 0.24/0.30 顶 60 mm/s
- `d_u=90`，`m_u=1.5`
- `ke_impact_initial: 1500` ← 上升沿可把 `b_d=2ζ√(M Ke)≈70`，短时 `D_eff≈70`
- 只有 `recontact` 8 mm/s × 0.22 s；没有 barrier / envelope / TDPA
- `system_delay_s: 0.015`（当时的占位，不是今天的 35 ms 辨识）

### 2.2 现网精确 ZOH

同一文件现在约 2470 行。定律：

```
v+ = a v + b (Fc + D0 v_r − Kc x̃),  a=e^{-D Ts/M}, b=(1-a)/D
```

```python
a_disc = math.exp(-damp * dt_eff / mass_z)
b_disc = (1.0 - a_disc) / damp
rhs = drive + max(damping_base, 0.0) * v_reference - kc * float(self.x_tilde_z)
velocity = a_disc * state + b_disc * rhs
```

`Ts=5 ms`、`D=25`、`M=1`：`a≈0.88`，和隐式欧拉同一量级。**轻不轻不取决于欧拉还是 ZOH**，取决于 `D`、`v_r`、发出帽。

`drive = e_f + u_dob`。`v_r` 在 [proactive_force_ff.py](rm75_control/rm75_control/control/admittance_common/proactive_force_ff.py)：漏积分 `e_f`，**不是** Li 2022 的人体观测器，也不是 Keemink `k_a`。

Dimeas ΔD（`_update_delta_d_hf`）：`target = d_u * I_s`，但 ramp 未完或 `|e_f|>0.8` 时 **target=0**。第一下弹正好在这个窗。`d_u=90` 不护首触。论文首选加 `m` 不加 `D`；现网/e85 yaml 是反的。

现网 `ke_impact_initial: 0`（三份 yaml）。上升沿 `b_d≈2*0.9*√80≈16 < D=25`，**加不上冲击 D**。这是相对 e85 的缺口，不是「按 Dimeas 改对了」。

### 2.3 63401843

yaml：`D=40`；`force_dob.enabled: false`；`proactive_feedforward: false`；`d_u=0`，`m_u=0`；`track_axes[2]=1`，`kp_pos[2]=5`；barrier 开；`cdyob.mode: shadow`；`recontact_hold_s=0.80`，帽 12 mm/s；`max_vz` 仍是 0.08。

准稳态：`v ≈ e_f/40`，没有 `v_r`、没有 DOB。同样 2 N 只有 **50 mm/s**，而且还经常到不了，因为 delay-safe / barrier 先把 `press_cap` 收掉。

---

## 3. 发出钳位（源码）

### 3.1 e85：导纳算完 ≈ 发出

```python
press_cap = self._press_vz_cap()   # 只在 recontact_timer 上收到 8 mm/s
lo = -v_z_cap                     # 回撤始终 max_vz
hi = press_cap
v_cmd_tool[2] = clip(..., lo, hi)
v_final = clip(v_out, ±max_velocity)   # Z 顶 0.10
# 力轴不做加速度 slew
return v_final
```

`_press_vz_cap`：

```python
cap = self._v_z_cap()
if self._recontact_timer_s > 0.0 and recontact_vz_cap > 0:
    cap = min(cap, 0.008)
```

确认接触后没有 first_touch、没有 barrier。`u_sent` 这个名字当时还不存在。机器人吃的就是 `v_final`。

### 3.2 634：状态里已经慢，后再叠一层；R1 只绕过后一层

导纳内同样 `clip(velocity, lo, hi)`，`hi = _press_vz_cap()`：

```python
# 634 _press_vz_cap
if self._use_delay_safe_press():      # first/recontact latch
    cap = min(cap, self._v_delay_safe())
elif not self.contact_present:
    cap = min(cap, self._v_air_seek())  # 20 mm/s
# 确认且 latch 释放：才是 max_vz
```

发出再：`barrier.clamp_velocity` → slew → shield（observe 不改）→ `v_final[2]=u_sent`。

peirastic 当时的 R1（`legacy.py`）：

```python
v_force_z = float(getattr(self.controller, "v_force_z", v[2]))
```

`v_force_z` 是**导纳状态**，会绕过 barrier/slew/shield 的最后一刀。但它**已经**被 `_admittance_z` 里的 `press_cap` 剪过。所以：

- 少弹的主因在导纳侧：`D=40`、无 chase、latch 上 delay-safe、位 P。
- R1 不是少弹的原因；它最多让「barrier 后再剪」失效。
- 现网已改成 `v_force_cmd_z`（钳后命令），不要再恢复 R1。

### 3.3 现网：e85 帽 + 后层观察

包络全 0 时（`_press_envelope_active()==False`）：

```python
# 确认接触：press = max_vz；空中：v_seek 20；recontact timer：8
# 回撤：始终 max_vz
```

后层：barrier `enabled: false` 时 `caps()` 原样返回 `v_hi`；走廊关；shield observe；TDPA `apply: false`。力轴 pose-P 在融合里清零：

```python
for index in range(3):
    if float(cfg.force_axes[index]) > 0.5:
        v_corr_tool[index] = 0.0
```

注释写明：634 的 `kp_z=5` 会把首触压深存成弹簧，latch 一开就往回抽。

空中 R2：`F*=0`，`v_force_tool[2]=_approach_governor`（20 mm/s）。e85 力任务 latch 后空中仍可能按 `e_f/D` 走 80。现网**故意**不抄这点（未知 `Ke` 不准 80 去撞）。

---

## 4. 为什么 63401843 弹跳好一点

按对弹跳的贡献排序（大 → 小）：

1. **`D=40` 且无 `v_r`/DOB。** 激励 `v` 小。`ΔF≈Ke v Td`：同样 `Ke=2000`、`Td=35 ms`，50 mm/s → 3.5 N，80 mm/s → 5.6 N。
2. **delay-safe 首触。** latch 期间 `v ≤ min(v_delay_safe, 12 mm/s, seek 20)`。12 mm/s 时 `ΔF≈0.84 N`。`recontact_hold=0.80 s` 把这扇窗拉得很长。
3. **位 P `kp_z=5`。** `v_+ = v_adm + 5(x*-x)`。压深 4 mm 就有 20 mm/s 往回拉。隐藏弹簧：稳、黏、不深。
4. **barrier 在导纳帽之外再按 `Kê Δx_pipe` 关下压。** 对 R1 发出可能无效，对 `v_force_z` 状态仍间接有效（状态已被 press_cap 剪）。
5. **CDYOB shadow 不改 `u_sent`。** `computes()` 还会冻 chase。commit 留言容易误读。Samuel 2024 自己的「轻」导纳仍是 `B_a=400–800`，`td=3 ms`，`ω_Q=15 Hz`。我们的延迟和 `D=25` 对不上那套数。

634 的 `max_vz` 已经是 80。少弹不是帽更小，是**很少用到这顶**。

---

## 5. 为什么 e85 深得快

「深」= 同样时间内压进垫子更多 = 平均下压速度大。

\[
v \approx \frac{e_f+u_{\mathrm{dob}}}{25}+v_r,\quad |v|\le 0.08.
\]

- 欠力 2 N：80 mm/s 量级，再加最多 60 mm/s 的 `v_r`。
- 确认后没有 10–25 的后层锁，所以 `Y_a` 在低频接近 `Y_v`（Keemink：clip 一紧，实现的 `|v/F|` 就塌，手感变沉）。
- 力轴不跟位：没有 `5·Δx` 把你从坑里拽出来。
- 回撤同样 80 + `retract_gain=0.30`：过冲后能抽回来，所以「快」但不一定「散」。
- 上升沿 `ke_impact=1500` 只在接触沿短时加 D，稳态仍是 25。现网把这个种子清零了，第一下比 e85 **更裸**。

e85 空中在力任务 latch 后也可以 80。现网空中 20。所以现网「接近 e85 手感」只该拿**确认接触之后**比，不要拿接近去比。

轻不需要 R1：e85 发出就是 `v_final`。现网发出是钳后的 `u_sent`，包络/barrier/走廊关掉后两者在确认接触上是同一档 ±80。

---

## 6. 和论文实现差在哪

### Keemink 2018

论文不给新算法，给 `Y_v→Y_a` 的分析。我们：

- 外环是 `1/(Ms+D)` + `v_r` + DOB + 饱和 + 6 轴 QPIK，不是 Figure 5 的单轴 PI。
- 没有测 `Y_a(jω)`。测的是运动 FOPDT。
- 没有 `k_a`，也不能把 Eq.(12) 当成 Smith / 延迟对消。
- 严格无源条件 (7)(8) 要求纯惯量时 `m_v ≳ m_r` 且 `K_i=0`。我们既不是纯惯量，也没有那套 `K_p,K_i`。套公式宣称无源是错的。

### Dimeas 2016

要杀的是力上 5.8–20 Hz 颤，首选加 `m_d`。e85/现网加的是 `d_u=90`，且大误差时关掉。第一下过冲不是他们要杀的持续颤。`I_s` 来不及，也不该用 `|e_f|` 触发。

### De Stefano 2020 TDPA

植物是自由惯量 HIL，不是 `D=25` 导纳。impedance PC：`F_c=F_e-αv`，若 `v=F_c/D` 则 `v=F_e/(D+α)`，**只加 D**。现网 `apply` 把 `F_c` 代进 `e_adm=F*-F_c`，得到 `M v̇+(D-α)v=e_f`，`α_max=400` 反号。观察可以，现接线 apply 不行。被动 ≠ 不弹。

### Lee 2024 BEFM / Secchi tank

口是 proxy–真机速度差或 tank 改 `v_d`。现网 `bidirectional_flow.sign_verified=false`。空罐会卡住。不要 apply。

### Li 2022

`ẋ_r=α û_h`。我们的 `v_r` 只是漏积分 `e_f`。注释里已经写了不是那篇。

---

## 7. 好方案（仍不改代码）

能合：

- **确认接触：** 保持 e85。`D=25`，`Kc=0`，力轴不跟位，`v_r`/DOB，±80，证书观察，不恢复 R1，空中仍 R2。
- **第一下：** 只借 634 的慢触思想——未确认/再贴 8–12 mm/s，窗约 **0.22 s**，不要 634 的 0.80 s。减的是 `ΔF`，不是稳态 D。
- **颤起来：** Dimeas (ii) 加 `m_u`，近设定点，不加 `d_u`，不打开大误差 ΔD。
- e85 的 `ke_impact_initial=1500` 是短时加 D。和 Dimeas「不要加 D」打架，和「少弹一下」同向。若以后要试，单独开实验，不要和合层绑死。

不能合：

- 稳态 `D=40` 或 `kp_z=5` 或确认后常开 barrier/first_touch/走廊。
- `cdyob.shadow|active`（冻 chase；Q 够不到接触共振）。
- `tdpa.apply` 走现接线；BEFM apply；空中 80。
- `Y_v←(1+k_a s)Y_v^0` 写成「抵消 35 ms」。

硬件上若确认后 `u_sent` 已经是 −80 仍大弹：问题在 `T_d`，不该再叠 clip。那是内环任务。

---

## 8. 公式自攻

1. `v=e_f/D+v_r` 忽略饱和、延迟、QPIK。只用来解释「为什么 e85 深、634 浅」。
2. `ΔF=Ke v Td` 假定延迟里速度不变。QPIK 还在爬时会更大。
3. `ζ=D/(2√(Ke M))` 假定纯弹簧、无 `v_r`。硬垫 `Ke~2 kN/m`、`D=25` 时 `ζ≈0.28`，弹一下是这条式子允许的。
4. Keemink `c_i≥0` 推出现网无源/非无源是滥用。
5. 634 少弹 ≠ CDYOB/TDPA 已在硬件上证过。
6. 现网「恢复 e85」没有把 `ke_impact_initial=1500` 带回来。比真 e85 更少一层冲击 D。

---

## 9. 关键源码索引

- 现网导纳 / 发出：`rm75_control/rm75_control/control/admittance_common/controller.py`（`_admittance_z`，`_press_vz_cap`，`_emit_retract_cap`，约 1998–2120 发出，1267–1282 力轴清 P）
- 现网 R1 已修：`peirastic/realman8dof/force/legacy.py` 59–64 行
- 634 R1：同文件在 `63401843`，`getattr(..., "v_force_z")`
- e85 发出：`e85c9ab` 同 controller，约 741–769 行，return `v_final`
- `v_r`：`proactive_force_ff.py`
- DOB：`force_dob.py`
- TDPA：`tdpa.py`（头注释：被动 ≠ 不弹；clamp α 论文没证）
- 力滤波：`observer.py`（20 Hz 1 阶；`use_inertia=False`）
- 植物数：`peirastic/configs/force.yaml` `safety_shield.plant.t0_s: 0.035`；辨识 JSON 如上
- 三份现网力轴 yaml：`peirastic/configs/force.yaml`、`controller.yaml`、`rm75_control/configs/joint_admittance_8dof.yaml`
