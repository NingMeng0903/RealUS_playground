# 导轨 VPC / 吸引子 / 限位松弛 — 第三方审查包

- 日期：2026-08-19
- 运行机：`camp@pei` / `rm75`
- 配置：`rm75_control/configs/joint_admittance_8dof.yaml`
- 本文件用途：把 2026-08-19 三次实机 log 的定量结论、因果链、以及**全部相关源码原文**放在一处，供第三方对照审查。本文**不改代码**。

## 0. 材料清单

### 0.1 原始 CSV（字节原样）

三份 WBC + 三份 FA24 体积合计约 **170 MB**。把它们内嵌进单个 `.md` 会使审查工具打不开。已用 `cp -a` 复制到：

`/media/camp/EXT_DRIVE/RealUS_playground/MD/todo_controller_logs/`

与 `rm75_control/apps/logs/` 下原件 **SHA-256 一致**：

| 文件 | 行×列 | 大小 | SHA-256 |
|---|---|---|---|
| `run_20260819_204333.csv` 手柄 WBC | 36710 × 376 | 101 MB | `a07ba9f902ca18f12b5a7b69d10a5d96e13f638435f4dfe5b9c5f6c1209a50eb` |
| `rail_20260819_204333.csv` 手柄 FA24 | 9201 × 47 | 3.2 MB | `f2f66dc36332c3d3c403d0615029bb9bbea2c3fb810dd60ab71c484d3f829bf4` |
| `run_20260819_204658.csv` 椭圆（中行程完整圈） | 7927 × 376 | 21 MB | `5c72da6337166577ee71cb31d3f6169d9c41f526766ee9aa39ca541859bf7258` |
| `rail_20260819_204658.csv` | 2050 × 47 | 721 KB | `a1aecafe2a2306f5dfb617a57485b73ead1384e20524ef92273666289013789e` |
| `run_20260819_204742.csv` 椭圆（贴 + 限位 / 联动崩溃） | 7733 × 376 | 21 MB | `541b2b499f6145266709ff086b5a46e81bfc80d273d7f90b000caa37d7c467bd` |
| `rail_20260819_204742.csv` | 123747 行（含表头），21:22 冻结 | 43.2 MB | `29356ce1ac002417f9749940041f0dc0219dc0ec98811caa61f55ac7d693b380` |

`rail_20260819_204742.csv` 在审查包冻结时 Window A 仍在后台追加（跨度已 >1600 s）。ellipse 本体只有 ~39 s。`follow` 占比极低，`t_write_ms` p50 = 0。审查椭圆伺服时请只看该文件前 ~40 s 或 WBC 时间窗 `[t0, t0+39]`。WBC 的 `run_20260819_204742.csv` 本身已结束，不受此影响。

### 0.2 三次实验对照

| # | 命令 | 日志 | 操作意图 | 时长 |
|---|---|---|---|---|
| A | `d_gamepad_vcmd.py` | `204333` | 手柄速度伺服，扫行程、两边死区、多段任务 | 200.1 s |
| B | `d_ellipse_track.py` | `204658` | X 10 cm / Y 30 cm / v≤4 cm/s / 40 s，从 live pose，**未贴墙** | 40.0 s |
| C | `d_ellipse_track.py` | `204742` | 同椭圆，从 q0≈0.69 出发，**贴 + 软限位测联动** | 39.0 s |

### 0.3 审查应打开的源码（全文附在本文附录 A）

- `configs/joint_admittance_8dof.yaml`
- `.../loop.py`（leave-band、`u_r`、press-stall nudge、`hold_d_star`）
- `.../tasks/rail_extension.py`（`e_mid`、`_in_plus_leave`、`escape_leave_m`）
- `.../tasks/rail_allocator.py`（Haviland 权重、mid-ranging PI、参考模型 / wall cap）
- `.../tasks/psi_retarget.py`（`d*` homotopy、`_clip_d_to_travel`、`nudge_d_star`）
- `.../solver/qp_builder.py`（任务松弛、rail box）
- `.../solver/constraint_mgr.py`（`wall_cap` / 速度盒）
- `.../hw/rail_servo.py`（软限位、FA24）
- `apps/joint_admittance_8dof/analyze_qpik_quality.py`（Phase 5 门槛）

---

## 1. 结论（先看这个）

1. **两边“死区”不是 5 mm / 780 mm。** 这次硬件真正顶住的是 **soft 30–755 mm**。导轨全程 `q_meas_0 ∈ [0.0251, 0.7550]`，**从未进入** hard box 的 5 mm 和 780 mm。再叠 `escape_leave_m=0.04`（+ 侧从 0.74 起禁止 `v_r_ref>0`）和 `limit_margin_m=0.15`（150 mm fade），操作手感觉的死区是 **数十毫米级**，不是 5 mm。
2. **手柄速度伺服的导轨分担已经合格，伺服跟踪只是中等。** `|v_cmd_vy|>20 mm/s` 时 `rail_motion_share` p50 = **0.709**，+Y/−Y 比 **1.03**。Y 轴相位残差 6.9 mm、相关 0.96。但 FA24 写入 **31.7 Hz**（门槛 40 Hz），`|Δrpm|` p95 = **72**（门槛 20），`e_track` p95 = **6.23 mm**。
3. **中行程椭圆（B）位置精度很好，滞后可忽略。** `track_err` p95 = **0.24 mm**，pose Y p95 = **0.20 mm**，axis Y tau = **0 ms**，`sign(v_r_ref)==sign(vy)` = **100%**，ψ 误差 p95 = **5.2°**，内环 ~198 Hz。
4. **贴墙椭圆（C）不是“松弛没开”，而是限位后错误地改写了吸引子，恢复时去追错误的 `d*`。** 时间线见 §4。`slack_norm` 在墙后 p50 = 0.107（76% tick > 1e-3），松弛在工作。崩溃顺序是：leave-band 把 `v_r_ref` 置 0 → 手臂用 J4 去跟 +Y → J4 90°→20° → `d*` 单 tick **+219 mm** 被 clip 到被拉长的 live 窗口 → 之后每 0.5 s +10 mm（`d_center_rate=0.02 m/s`）→ 离开 755 mm 后 J4 锁在 20°、ψ=21°（ref 仍 68°）、`u_alloc` 与 `u_mid` 反号、导轨乱飘。
5. **手柄后半段吸引子变差是同一类问题的慢版本。** 前 50 s `|ψ−ψ_ref|` p50 = 7.7°；后 50 s p50 = 27°、p95 = 72.7°。t≈60 s 顶在 30 mm 软下限时 share=0、ψ=116°。t≈100–160 s 松手窗口 `d*` 从 −0.185 走到 −0.376，ψ 到 151–159°，idle 导轨位移 p95 = **87 mm**、TCP p95 = **62 mm**。不是“吸引子完全没了”，是 **限位 + 松手 homotopy 把 `d*/ψ*` 带走后回不来侧面族**。

---

## 2. 限位栈（回答“死区为什么这么大”）

yaml 里同时存在三层行程，外加两条软件禁行带：

| 层 | 配置 | 数值 | 这次 log 是否真正撞上 |
|---|---|---|---|
| hard box | `qpik.hard_limits.rail.hard_*` | **5–780 mm** | 否。用户希望只留这一层 |
| soft wall | `soft_min_m / soft_max_m` | **30–755 mm** | **是。手柄 min=25.1 mm，三次实验 max 都卡在 755.0 mm** |
| d* 窗口收缩 | `psi_retarget.rail_margin_m` | 20 mm | 是。`_clip_d_to_travel` 用 `rail_lo+20mm … rail_hi−20mm` |
| plus leave-band | `rail_extension.escape_leave_m` | **40 mm** | **是。planned 椭圆在 q0≥0.74 且 v_r_ref>0 时强制 v_r_ref=0** |
| 软限位 fade | `limit_margin_m` | **150 mm** | 手柄 16% 的 tick 落在 30–170 mm 或 605–755 mm 带内 |
| 限位阻尼 | `limit_damper_band_rail_m` | 25 mm | 在 soft 内侧再削速度 |
| 伺服 approach | `hw.lw100.approach_m` | 40 mm | 工人侧单向刹车 |

`loop.py` 在参考模型之后（planned stroke）还有一次 leave-band：

```python
if self.rail_ext_task._in_plus_leave(y_r) and v_r_ref > 0.0:
    v_r_ref = 0.0
    self.rail_ref_model.reset(0.0)
```

`_in_plus_leave` 用的是 `rail_extension.soft_max_m`（默认 0.78）减 `escape_leave_m=0.04` → **0.74 m 起禁止 +v**。所以即使用户以为 780 mm 才是墙，**从 740 mm 开始 +Y 的导轨指令已经被软件清零**。椭圆 C 在 t=5.96 s、`q0=0.755`、`vy=+41 mm/s` 时 `v_r_ref` 已经是 0。

手柄占用：

| 带 | 手柄 204333 |
|---|---|
| < 5 mm hard | 0.0% |
| 5–30 mm（soft 以下） | 9.3%（最低 25.1 mm） |
| 30–170 mm（150 mm fade−） | 7.5% |
| 180–605 mm | 73.5% |
| 605–755 mm（fade+） | 8.6% |
| 755–780 mm | 1.1%（贴在 755.0，进不去） |
| > 780 mm | 0.0% |

**若审查同意操作手的规格（死区只留 0–5 mm 与 ≥780 mm），需要同时动 soft 30/755、leave 40 mm、fade 150 mm、rail_margin 20 mm，而不是只改 hard。**

---

## 3. 手柄速度伺服（A / 204333）

### 3.1 导轨–手臂分担（VPC 重建后的目标）

| 门槛 | 目标 | 实测 | 判据 |
|---|---|---|---|
| `rail_motion_share` p50 \|vy\|>20 mm/s | ≥ 0.60 | **0.709** n=16945 | PASS |
| +Y/−Y share 比 | ≤ 1.25 | **1.03** (0.697 / 0.716) | PASS |
| `sign(v_r_ref)==sign(vy)` \|vy\|>10 mm/s | ≥ 85% | **75.0–75.6%** | FAIL（几乎全是两端 leave-band 把 vref 置 0） |
| `track_err` p95 | ≤ 5 mm | **3.03 mm** | PASS（max 144 mm 出现在顶墙） |
| `\|ψ−ψ_ref\|` p95 | ≤ 15° | **74.8°** | FAIL |
| FA24 写入 | ≥ 40 Hz | **31.7 Hz** | FAIL |
| FA24 \|Δrpm\| p95 | ≤ 20 | **72** | FAIL |
| 内环 `deadline_slack>0` | ≥ 99% | **39.3%** | FAIL |
| `dt_actual` p50 | ≤ 5.2 ms | **5.01 ms**（p95 6.92，有效 183 Hz） | 中位达标，尾部未达标 |

速度环本身：`twist_achieved_vy − v_cmd_vy` 在 \|vy\|>10 mm/s 时 p50 = +1.1 mm/s，p95 = 57.9 mm/s。analyzer 给的 Y 相位 tau = **−10 ms**、残差 **6.88 mm**、corr(e,v)≈0。也就是说 **中行程速度跟随是贴得上的**；p95 速度误差来自顶墙时指令还在、导轨已被清零。

`u_alloc` p95 = 0.075 m/s，`u_mid` p95 = 0.119 m/s（打满 `u_mid_max=0.12`）。mid-ranging 在大 `e_mid` 时已经满权，这是设计如此，不是 bug。问题是 `e_mid` 的参考 `d*` 在后半段被带走（§5）。

### 3.2 FA24 / 形状误差

- worker `dt_wall` p50 = **23.0 ms（43 Hz）**，不是 59 Hz。
- `t_read` p50 = 7.5 ms，`t_write` p50 = **15.4 ms**（67% 的写落在 10–20 ms）。读+写经常 > 12 ms 预算，所以写入被摊薄到 32 Hz。
- `e_track` p95 = 6.23 mm，`e_shape` p95 = 5.64 mm。速度伺服在导轨层仍有数毫米形状债。
- 松手 idle：导轨位移 p95 **86.9 mm**、TCP **62.0 mm**、QP1 slack 占 idle tick 的 **51.7%**。这就是“任务后半段松手导轨自己走”。

### 3.3 吸引子随时间（10 s 窗，手柄）

| t (s) | q0 | J4° | ψ | ψ_ref | \|Δψ\| | d* | \|vy\| | share |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.204 | 92 | 61 | 68 | 9 | −0.185 | 0.042 | 0.47 |
| 40 | 0.607 | 105 | 69 | 68 | 1 | −0.188 | 0.100 | 0.91 |
| 60 | **0.030** | 117 | **116** | 68 | **48** | −0.188 | 0.100 | **0.00** |
| 100 | 0.351 | 55 | 71 | 68 | 5 | **−0.257** | 0 | — |
| 130 | 0.501 | 120 | **151** | 68 | **76** | −0.304 | 0 | 0.94 |
| 140 | 0.574 | 111 | **159** | **101** | 58 | −0.245 | 0 | 0.73 |
| 200 | 0.377 | 78 | 71 | 68 | 3 | −0.185 | 0 | — |

前 50 s 侧面族（ψ≈68°、J4≈100°、d*≈−0.185）是稳的。t=60 s 顶 soft_min 后 ψ 被拉到 116°。松手段 `hold_setpoint` 变 False，homotopy 从被污染的 `(d0, ψ0)` 往 `d_attr/ψ_attr` 走，但 `_maybe_retarget_psi` 在 live ψ 出 envelope 时把 `ψ_ref` 抬到 101°（envelope 上限 110°），**回不到 68° 侧面**。J4 一度到 55°（偏直）又到 126°（偏折），操作手描述的“被错误拉到快中间平面、不能恢复侧面”与 t=100–160 s 一致。

`family_ok` 全程 1.000：日志里的 family 位不反映 ψ 已经离开设计族。不要用这一列当吸引子健康标志。

---

## 4. 椭圆位置精度与滞后

### 4.1 B / 204658（完整圈，未贴墙）— 这是“好”的基线

| 量 | 值 |
|---|---|
| 导轨行程 | 0.372–0.706 m（334 mm，覆盖 300 mm Ypp × share 0.93） |
| pose X/Y/Z 误差 p95 | 0.11 / **0.20** / 0.09 mm |
| axis 相位 tau | **0 / 0 / 0 ms** |
| `track_err` p95 | **0.24 mm** |
| 速度 Y 相关 | 0.998 @ −2 tick |
| rail share p50 | **0.933**，±Y 比 1.00 |
| `sign` 一致率 | **100%** |
| \|ψ−ψ_ref\| p95 | **5.2°**，J4 中位 87° |
| 内环 | dt p50=5.00 ms，on-time 95%，tick_inner p50=3.57 ms，slack>0 **91.2%** |
| FA24 | 29.9 Hz，Δrpm p95=18.7（这次达标），e_track p95=2.41 mm |

**中行程椭圆的位置环和导轨联动是成功的。** 滞后不是问题。剩下的是 FA24 仍 ~30 Hz、以及 64.5 Hz 的 q_cmd 加速度谐波（与 worker 周期接近）。

### 4.2 C / 204742（从 0.69 m 出发贴 + 墙）— 崩溃时间线

1 s 窗中位数：

| t | q0 | J4° | ψ / ψ_ref | d* | slack | vy | v_r_ref | u_alloc | u_mid | eY mm | share | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0–4 | 0.68–0.71 | 86 | 68/68 | −0.185 | 0 | +0.02 | +0.02 | ~0 | +0.02 | 0.0 | 0.7–0.9 | 0 |
| 5 | 0.743 | 90 | 70/68 | −0.185 | 0 | +0.034 | +0.034 | 0.004 | 0.032 | 0.0 | 0.96 | 0.23 |
| 6 | **0.755** | 85 | 69/68 | −0.185 | 0 | +0.037 | **0** | 0.009 | 0.058 | 0.0 | **0** | **1** |
| 8 | 0.755 | 63 | 69/68 | −0.185 | 0 | +0.035 | 0 | 0.024 | 0.102 | 0.0 | 0 | 1 |
| 10 | 0.755 | **30** | 56/68 | −0.185 | 0.005 | +0.031 | 0 | 0.030 | 0.115 | −0.5 | 0 | 1 |
| 11 | 0.755 | **20** | **22/68** | −0.185 | **0.114** | +0.095 | 0 | 0.128 | 0.117 | **−7.6** | 0 | 1 |
| 13.75 | 0.755 | 20 | 21/68 | **−0.185 → +0.034** | 0.172 | +0.122 | 0 | 0.12 | ~0 | −20 | 0 | 1 |
| 17 | 0.728 | 20 | 21/68 | +0.094 | 0.145 | +0.065 | −0.029 | +0.15 | **−0.10** | −10 | 0.85 | 0 |
| 27 | 0.484 | 20 | 21/68 | +0.244 | 0.167 | +0.13 | ~0 | +0.20 | **−0.12** | −13 | 0.12 | 0 |
| 37 | 0.755 | 20 | 21/68 | +0.364 | 0.170 | +0.13 | 0 | +0.19 | −0.12 | −16 | 0 | 1 |

关键标记（相对 t0）：

- t=5.43 s：`q0>0.74`，leave-band 生效边缘
- t=5.96 s：`v_r_ref=0` 而 `vy=+41 mm/s`（软件禁 +v，不是电机到 780）
- t=10.01 s：J4 < 40°
- t=10.70 s：`|ψ−ψ_ref|>20°`
- t=10.98 s：`slack>0.05`（松弛开始真正买 Y）
- t=13.752 s：`d*` **单 tick +219.1 mm**（analyzer：`d_star step max 219.09 mm`，门槛 0.20 mm）
- 此后每 ~0.5 s `d* += 10 mm` = `d_center_rate 0.02 m/s`，一直加到 +0.37 m
- t=17.22 s：离开 0.74 带，导轨往回走，**J4 仍 20.1°、ψ 仍 21°**

analyzer 对 C：

- share p50 0.324 FAIL，sign 一致 40% FAIL
- `track_err` p95 **24.8 mm**，tool_y p95 **19.8 mm**
- axis Y tau **+35 ms**、残差 **10.8 mm**（这才是“滞后”；它出现在崩溃之后，不是椭圆几何本身）
- 墙内 7DOF IK 仅 42% 样本可行，tool_y p95 19.8 mm（墙内本应 < 3 mm）
- FA24 25.7 Hz，Δrpm p95 83.5（被 1669 s 空闲 log 污染，审查请切片）

### 4.3 “松弛 OK，但恢复后不追目标、手臂伸直、导轨乱飘”的机制

松弛 **没有关掉**。从 t=11 s 起 `slack_norm` 到 0.17，pose Y 误差被买到 20 mm 而不是硬顶爆。QP1 的语义是：rail 被 pin 在 0 时，用手臂 7 自由度去跟 6 维 twist，不够就加任务松弛。

恢复失败的三件事叠在一起：

1. **指令层仍把 +Y 的导轨分量清零或对打。** 离开 755 mm 后 `u_alloc ≈ +0.15`（还想跟 +Y），`u_mid ≈ −0.12`（`e_mid` 已被错误 `d*` 定义成“还要再伸”）。`v_r_ref` 在两者之间漂，share 在 0.05–0.90 间跳。这就是“rail 乱飘”。
2. **`d*` 在墙内被改写成 live 伸长，恢复时 homotopy 继续朝这个错误目标走。** `psi_retarget._clip_d_to_travel` / `_rate_limit_d`：当 `y_tcp` 随伸直手臂增大，可行窗 `d_lo = y_tcp − (rail_hi − margin)` 变成正数，把 −0.185 **一次性 clip** 成 +0.034。随后 `d_center_rate` 以 20 mm/s 跟着窗口走。审查请看 `psi_retarget.py` `_clip_d_to_travel` 与 `_rate_limit_d` 在 `y_lo > y_hi` 或 clip 时写入 `d_live` 的分支。
3. **ψ\* 仍是 68°，但 QP 没有把 J4 拉回去。** J4 停在 ~20°（branch_barrier `box_activate_rad=0.87`≈50° 的内侧地板附近）。QP1 优先 TCP/松弛，nullspace / 臂角任务挤不进。所以操作手看到的是“莫名其妙开始伸直手臂”，而不是沿原椭圆慢慢追回。

`press_stall` nudge（+10 mm / 0.5 s、`d_star_nudge_m=0.01`）与 `d_center_rate` 步长数值相同。C 的 +10 mm 台阶与 0.5 s 周期吻合；**不能单靠步长区分是 stall nudge 还是 homotopy**。但 **+219 mm 那一跳只能是 clip-to-window**，不是 10 mm nudge。

椭圆 **没有** `plan_scan_stroke` 的 `_planned` 位（CSV 无 planned 列；`d*` 在 planned 路径上本应冻结）。C 走的是 unplanned CARTESIAN_TRACK + live `vel_ff`。`hold_setpoint_from_vel_ff` 在 \|v_ff\|>5 mm/s 时应冻 `d*`；t=13.75 时 `v_cmd_vy=+122 mm/s` 但 `vel_ff_vy=−2 mm/s`（符号已翻、幅值低于阈值），**hold 解开，clip 发生**。这是恢复逻辑的触发条件，不是巧合。

---

## 5. 代码因果索引（附录有全文）

| 现象 | 代码位置 | 做什么 |
|---|---|---|
| 755 mm 就停 | yaml `soft_max_m: 0.755`；`RailReferenceModel` + `wall_cap`；servo `soft_max_m` | 软墙，不是 780 |
| 740 mm 起 +v 被清零 | `loop.py` planned leave-band；`rail_extension._in_plus_leave` / `escape_leave_m=0.04` | 禁行带比 soft 更宽 |
| 墙内手臂伸直 | QP1 `rail_task_vel=0` 后 7DOF 跟 Y；branch_barrier 没挡住 90°→20° | 宏轴停、微轴代偿 |
| `d*` +219 mm | `psi_retarget._clip_d_to_travel` / `_rate_limit_d` | 把 d* 夹到被拉长的 live 窗 |
| 之后每 0.5 s +10 mm | `d_center_rate_m_s: 0.02` 和/或 `nudge_d_star(0.01)` + `press_stall_s: 0.5` | 错误目标上继续走 |
| 恢复时 rail 对打 | `u_r = u_alloc + u_mid + u_escape`；`e_mid=(y_tcp−y_rail)−d*` | d* 错了，mid-ranging 满权反向 |
| 手柄后半 ψ 回不来 | `hold_setpoint` 松开 → `_advance_homotopy` + `_maybe_retarget_psi` envelope | 从污染的 (d0,ψ0) 再规划 |
| FA24 楼梯 | worker ~43 Hz、`t_write` p50 15 ms；deadband 已是 0 | 带宽不够 40 Hz 门槛 |

---

## 6. 审查时建议核对的问题（不在本包落地改）

1. soft 行程是否应改为 **5–780 mm**，只留 hard 几何与 `wall_cap` 做单向刹车，删掉 40 mm leave-band 和 150 mm fade？
2. 导轨被 pin 时，QP1 是否应 **禁止用 J4 去买 Y**（或把 arm-Y 权降到松弛之下），避免 90°→20°？
3. `_clip_d_to_travel` 在 rail 饱和时是否应 **拒绝改写 d\***，而不是把吸引子搬到 live 伸长？
4. `hold_setpoint` 是否应按 `v_cmd` / `u_alloc` 而不是已经翻号的 `vel_ff_vy` 来冻 d*？
5. 松手 homotopy 是否必须从 **yaml `(ψ_attr, d_attr)`** 重启，而不是从限位污染的 `(ψ0, d0)`？
6. FA24：`t_write` 15 ms 能否压到与 60 Hz worker 相容，否则 40 Hz 门槛达不到。

---

## 7. 附录 A — 相关源码原文

以下文件从审查时的 git 工作树 **逐字复制**。路径相对于 `rm75_control/`。

---

## 7.1 源码全文

### `configs/joint_admittance_8dof.yaml`

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
state_relay:
  enabled: false
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
    f_c_hz: 20.0
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
cartesian_track:
  k_task_lin: 10.0
  k_task_rot: 2.0
  max_pos_err_m: 0.05
  max_rot_err_rad: 0.35

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
    # 80 Hz is tight on p95.
    poll_hz: 60
    inter_frame_delay_s: 0.0005
    timeout_s: 0.06             # poll-budget; was 0.15 / class-default 1.0
    retries: 1
    deadband_mm: 0.5
    # FA23 overspeed trip: must sit ABOVE commanded peak (0.15 m/s ≈ 900 rpm
    # @ 10 mm/rev). Equal FA23=cmd caused Er-01 on scan overshoot (151334).
    max_speed_rpm: 1200
    # Soft CSP via FA24 (see apps/lw100_vel_pos_follow_demo.py).
    # Loaded PD scan BEST (400±40 mm): kp=14/kd=0.22 (was empty-load 18/0.22).
    # Rollback: vel_kp 14 / vel_kd 0.22.  Coupled path now drops P/D while
    # moving (L1 owns position).  POSITION scan/home still uses these.
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
    vel_ff_p_trim_m_s: 0.010
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

### `rm75_control/control/joint_admittance_8dof/loop.py`

```python
"""Joint-space inner loop: Cartesian twist -> absolute joint angles (rm_movej_canfd).

``JointIkController``: hardware-free WBC slack-QP IK + safety clamp (no send-path LPF).
``run_joint_admittance_phases``: on-robot orchestration closing on FK(q_meas).
"""

from __future__ import annotations

import copy
import csv
import gc
import inspect
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.admittance_common.async_state import arm_qdot_rad_s_from_snap
from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.ik_types import (
    saturate_error,
    sr_damping_lambda,
)
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
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig, QpIkController
from rm75_control.control.joint_admittance_8dof.solver import cpp_kernel
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
from rm75_control.control.joint_admittance_8dof.tasks.rail_allocator import (
    MidrangingController,
    RailAllocatorConfig,
    RailReferenceModel,
    RailStateObserver,
    allocate_rail,
    margin_weight_from_activation,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_goodness import (
    CachedRailGoodness,
    SigmaMinGoodness,
)
from rm75_control.control.joint_admittance_8dof.tasks.secondary_composer import (
    SecondaryComposer,
    max_limit_activation,
)
from rm75_control.control.joint_admittance_8dof.tasks.ird_adapter import (
    IrdConfig,
    try_load_ird,
)
from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import (
    PostureRetarget,
    PsiRetargetConfig,
    design_family_ok,
    fold_psi_to_positive,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_lock import (
    RailLockConfig,
    RailLockTask,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import (
    LockedStyle,
    RailMode,
)
from rm75_control.control.joint_admittance_8dof.filters import (
    smoothstep01,
)
from rm75_control.control.joint_admittance_8dof.saturation_latch import (
    SaturationConfig,
    secondary_scale_from_slack,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import (
    SafetyLimits,
    Watchdog,
    clamp_command_step,
    integration_period,
)

# Pure rotation used to skip hold_setpoint (only vff[:3] was checked), so
# homotopy q* chased live IK while the stick twisted J1 to −163°.
_HOLD_ROT_THR_RAD_S = 0.05


def hold_setpoint_from_vel_ff(
    vel_ff: np.ndarray | None,
    *,
    lin_thr_m_s: float,
    rot_thr_rad_s: float = _HOLD_ROT_THR_RAD_S,
) -> bool:
    """True if translation or rotation FF is commanding motion."""
    if vel_ff is None:
        return False
    vff = np.asarray(vel_ff, dtype=float).reshape(-1)
    if vff.size >= 3 and float(np.linalg.norm(vff[:3])) > float(lin_thr_m_s):
        return True
    if vff.size >= 6 and float(np.linalg.norm(vff[3:6])) > float(rot_thr_rad_s):
        return True
    return False


# ---------------------------------------------------------------------------
# Inner loop (hardware-free)
# ---------------------------------------------------------------------------
@dataclass
class CartesianTrackGains:
    """Outer-loop Cartesian P gains from yaml ``cartesian_track``."""

    k_task_lin: float = 10.0
    k_task_rot: float = 2.0
    max_pos_err_m: float = 0.05
    max_rot_err_rad: float = 0.35


@dataclass
class JointIkConfig:
    dt: float = 0.005
    control_frame: str = "tool"
    euler_order: str = "xyz"
    qp: QpConfig = field(default_factory=QpConfig)
    cartesian_track: CartesianTrackGains = field(default_factory=CartesianTrackGains)
    nullspace: NullspaceTaskConfig = field(default_factory=NullspaceTaskConfig)
    manipulability: ManipulabilityTaskConfig = field(default_factory=ManipulabilityTaskConfig)
    arm_angle: ArmAngleTaskConfig = field(default_factory=ArmAngleTaskConfig)
    psi_retarget: PsiRetargetConfig = field(default_factory=PsiRetargetConfig)
    ird: IrdConfig = field(default_factory=IrdConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    rail: RailLockConfig = field(default_factory=RailLockConfig)
    rail_extension: RailExtensionConfig = field(default_factory=RailExtensionConfig)
    rail_allocator: RailAllocatorConfig = field(default_factory=RailAllocatorConfig)
    v_scale: float = 0.5
    a_max_arm_rad_s2: float = 20.0
    a_max_rail_m_s2: float = 0.60
    position_margin_rad: float = 0.017
    position_margin_rail_m: float = 0.0
    control_cpu: int | None = None
    disable_cstates: bool = True
    resync_err_rad: float = 0.10
    resync_err_rail_m: float = 0.020
    feedback_timeout_s: float = 0.050
    # Consecutive rejected/stale feedback before the loop faults.  One
    # USB/GIL hitch must coast (FA24=0, arm holds TCP) instead of aborting.
    feedback_coast_s: float = 0.30
    rt_disable_gc: bool = True
    verbose_json: bool = False
    nullspace_d_null: float = 0.0
    nullspace_d_null_adaptive: float = 1.0
    nullspace_max_qdot_frac: float = 0.2
    saturation: SaturationConfig = field(default_factory=SaturationConfig)


@dataclass
class JointIkStep:
    q_send: np.ndarray
    qdot: np.ndarray
    twist_base: np.ndarray
    sigma_min: float
    manip: float
    slack_norm: float
    n_cbf_active: int
    follow_err_rad: float
    cart_err_mm: float = 0.0
    qdot_ff_norm: float = 0.0
    vel_clamped: bool = False
    acc_clamped: bool = False
    pos_clamped: bool = False
    tcp_jump_mm: float = 0.0
    rail_vel_pin: float = float("nan")
    rail_qdot_ff: float = float("nan")
    plan_drives_rail: bool = False
    arm_singularity_smooth: float = 1.0
    limit_activation: float = 0.0
    rail_ext_err_m: float = 0.0
    rail_ext_weight: float = 0.0
    rail_escape_active: bool = False
    psi_deg: float = float("nan")
    psi_ref_deg: float = float("nan")
    psi_retarget_score: float = float("nan")
    d_pref_m: float = float("nan")
    elbow_margin_rad: float = float("nan")
    wrist_open_rad: float = float("nan")
    family_ok: bool = True
    physical_saturated: bool = False
    rail_contrib_m_s: float = float("nan")
    arm_contrib_m_s: float = float("nan")
    rail_motion_share: float = float("nan")
    # Rail velocity the QP actually used for affine compensation, after the
    # command/measurement blend.  rail_exec_velocity_m_s is overwritten with
    # the raw worker estimate downstream, so the blended value needs its own.
    rail_exec_for_qp_m_s: float = float("nan")
    wln_scale_rail: float = float("nan")
    wln_scale_arm_max: float = float("nan")
    waste_ratio: float = float("nan")
    rail_ff_m: float = float("nan")
    rail_posture_err_m: float = float("nan")
    d_star_m: float = float("nan")
    psi_star_deg: float = float("nan")
    minmax_margin: float = float("nan")
    controller_mode: str = "qpik"
    qp_backend: str = ""
    qp_solver_status: str = "not_run"
    qp_solver_iterations: int = 0
    qp_solver_solve_ms: float = 0.0
    qp_solver_call_count: int = 0
    qp_solver_overrun: bool = False
    qp1_status: str = "not_run"
    qp2_status: str = "not_run"
    qp1_solve_ms: float = 0.0
    qp2_solve_ms: float = 0.0
    qp_assembly_ms: float = 0.0
    qp_fallback_ms: float = 0.0
    qpik_total_ms: float = 0.0
    qp2_fallback: bool = False
    # Coarse per-stage tick profile (ms).  The loop budgets 5.0 ms but
    # measured 6.16 ms mean with only 2.1% of ticks on time, and the log had
    # no way to attribute the overrun.
    tick_inner_ms: float = float("nan")
    tick_send_ms: float = float("nan")
    tick_log_ms: float = float("nan")
    qpik_alpha: float = 1.0
    qpik_beta: float = 1.0
    qpik_authority: float = 1.0
    qpik_equality_residual_max: float = float("nan")
    qpik_hard_residual_max: float = float("nan")
    qpik_anchor_valid: bool = True
    qpik_recovery_overflow: bool = False
    qpik_protected_nominal_overflow: np.ndarray = field(default_factory=lambda: np.zeros(4))
    qpik_recovery_caps: np.ndarray = field(default_factory=lambda: np.zeros(14))
    qpik_recovery_overflow_indices: tuple[int, ...] = ()
    qpik_working_slack: np.ndarray = field(default_factory=lambda: np.zeros(8))
    qpik_collision_slack: np.ndarray = field(default_factory=lambda: np.zeros(4))
    qpik_dexterity_slack: float = 0.0
    qpik_branch_slack: float = 0.0
    rail_macro_pref_v: float = 0.0
    rail_center_pref_v: float = 0.0
    arm_risk_pref_norm: float = 0.0
    arm_risk_pref: np.ndarray = field(default_factory=lambda: np.zeros(8))
    risk_direction_cosine: float = float("nan")
    path_velocity_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    feedback_xy_raw: np.ndarray = field(default_factory=lambda: np.zeros(2))
    feedback_xy_filtered: np.ndarray = field(default_factory=lambda: np.zeros(2))
    rail_xy_contribution: np.ndarray = field(default_factory=lambda: np.zeros(2))
    arm_xy_contribution: np.ndarray = field(default_factory=lambda: np.zeros(2))
    rail_task_projection: float = float("nan")
    rail_arm_cancel: float = float("nan")
    rail_decomposition_error: float = 0.0
    wrist_singularity: float = float("nan")
    hard_active_constraint_ids: tuple[str, ...] = ()
    protected_target: np.ndarray = field(default_factory=lambda: np.zeros(0))
    protected_achieved: np.ndarray = field(default_factory=lambda: np.zeros(0))
    protected_residual: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scan_target: np.ndarray = field(default_factory=lambda: np.zeros(2))
    scan_achieved: np.ndarray = field(default_factory=lambda: np.zeros(2))
    scan_residual: np.ndarray = field(default_factory=lambda: np.zeros(2))
    fallback_level: str = "none"
    fallback_reason: str = ""
    solver_fault_latched: bool = False
    health_state: str = "NORMAL"
    arm_health: float = float("nan")
    joint_margin_rad: float = float("nan")
    wrist_margin_rad: float = float("nan")
    accepted_reference_lag_s: float = 0.0
    pre_solve_feedback_age_s: float = float("nan")
    post_solve_feedback_age_s: float = float("nan")
    rail_sat: bool = False
    rail_exec_velocity_m_s: float = float("nan")
    rail_measured_velocity_m_s: float = float("nan")
    rail_commanded_velocity_m_s: float = float("nan")
    rail_commanded_acceleration_m_s2: float = float("nan")
    rail_feedback_age_s: float = float("nan")
    a_mirror_frac: float = float("nan")
    j_mirror_frac: float = float("nan")
    last_limit_saturated: bool = False
    keep_task_weight: bool = False
    pref_slack_scale: float = 1.0
    rail_task_vel: float = float("nan")
    v_escape: float = float("nan")
    v_reach: float = float("nan")
    v_ff_rail: float = float("nan")
    u_alloc: float = float("nan")
    u_posture: float = float("nan")
    u_mid: float = float("nan")
    v_r_ref: float = float("nan")
    comp_projected_frac: float = 0.0
    rail_coast_active: bool = False
    rail_feedback_reject_streak_s: float = 0.0
    wall_override: bool = False
    slack_zero_feasible: bool = False
    sigma_arm: float = float("nan")
    sns_scale: float = 1.0
    qdot_meas: np.ndarray = field(default_factory=lambda: np.full(8, np.nan))
    v_cmd: np.ndarray = field(default_factory=lambda: np.zeros(6))
    path_twist: np.ndarray = field(default_factory=lambda: np.zeros(6))
    feedback_twist: np.ndarray = field(default_factory=lambda: np.zeros(6))
    comfort_slack: np.ndarray = field(default_factory=lambda: np.zeros(7))
    cbf_min_dist: float = float("nan")
    cbf_pair: str = ""
    nullspace_norm: float = float("nan")
    nullspace_centering_norm: float = float("nan")
    nullspace_manip_norm: float = float("nan")
    nullspace_arm_angle_norm: float = float("nan")
    nullspace_damping_norm: float = float("nan")
    nullspace_rail_lock_norm: float = float("nan")
    post_qp_step_clamp_enabled: bool = True
    post_step_would_clamp: bool = False
    post_step_clamp_applied: bool = False
    dt_nom_s: float = float("nan")
    dt_int_s: float = float("nan")
    box_h1_s: float = float("nan")
    box_h2_s: float = float("nan")
    qdot_raw: np.ndarray = field(default_factory=lambda: np.full(8, np.nan))
    qdot_pre_commit: np.ndarray = field(default_factory=lambda: np.full(8, np.nan))
    qdot_committed: np.ndarray = field(default_factory=lambda: np.full(8, np.nan))
    qdot_prev_used: np.ndarray = field(default_factory=lambda: np.full(8, np.nan))
    qdot_prev2_used: np.ndarray = field(default_factory=lambda: np.full(8, np.nan))
    box_lo: np.ndarray = field(default_factory=lambda: np.full(8, np.nan))
    box_hi: np.ndarray = field(default_factory=lambda: np.full(8, np.nan))
    post_step_shadow_q: np.ndarray = field(default_factory=lambda: np.full(8, np.nan))
    arm_send_mono_ns: int = 0
    rail_target_publish_mono_ns: int = 0
    rail_fa24_write_mono_ns: int = 0
    rail_encoder_sample_mono_ns: int = 0


def scale_qdot_into_box(
    qdot: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    """Uniform task-scaling into [lo, hi]: preserve Cartesian direction.

    Per-joint ``np.clip`` would break the twist direction (Flacco SNS vs naive
    saturation).  A single ``s ∈ [0, 1]`` is applied to the whole vector.
    """
    qdot = np.asarray(qdot, dtype=float).reshape(-1).copy()
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    if qdot.shape != lo.shape or qdot.shape != hi.shape:
        return np.clip(qdot, lo, hi) if lo.shape == qdot.shape else qdot
    s = 1.0
    eps = 1.0e-12
    for i, v in enumerate(qdot):
        if v > hi[i] + eps:
            if v > eps:
                s = min(s, float(hi[i] / v))
            else:
                s = 0.0
        elif v < lo[i] - eps:
            if v < -eps:
                s = min(s, float(lo[i] / v))
            else:
                s = 0.0
    s = float(np.clip(s, 0.0, 1.0))
    return qdot * s


class JointIkController:
    """Reusable inner loop: (q_cmd, q_meas, twist) -> next joint command (rad)."""

    def __init__(self, kin: RobotKinematics, cfg: JointIkConfig | None = None) -> None:
        self.kin = kin
        self.cfg = cfg or JointIkConfig()
        if (
            not np.isfinite(float(self.cfg.feedback_timeout_s))
            or float(self.cfg.feedback_timeout_s) <= 0.0
        ):
            raise ValueError("feedback_timeout_s must be finite and > 0")
        self.cfg.qp.euler_order = self.cfg.euler_order
        self.cfg.qp.collision = self.cfg.collision
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
        if self.rail_ext_task is not None:
            self.rail_ext_task.cfg.soft_min_m = float(self.cfg.rail.soft_min_m)
            self.rail_ext_task.cfg.soft_max_m = float(self.cfg.rail.soft_max_m)
        self.posture_retarget = (
            PostureRetarget(kin, self.cfg.psi_retarget, euler_order=self.cfg.euler_order)
            if self.cfg.psi_retarget.enabled
            else None
        )
        self._rail_ext_active = True
        ird_cfg = self.cfg.ird if self.cfg.ird is not None else IrdConfig()
        self._ird = (
            try_load_ird(ird_cfg) if bool(getattr(ird_cfg, "enabled", False)) else None
        )
        if self.posture_retarget is not None:
            self.posture_retarget._ird = self._ird
        # IRD is one-shot d* at plan_scan_stroke only.  Autograd goodness
        # on this thread caused 127 ms hitches → servo FA24=0.
        self._rail_goodness = CachedRailGoodness(
            SigmaMinGoodness(kin), period_ticks=10
        )
        self._sigma_grad_rail_cached: float = 0.0
        a_max_vec = np.full(kin.nv, float(self.cfg.a_max_arm_rad_s2))
        a_max_vec[0] = float(self.cfg.a_max_rail_m_s2)
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
                float(self.kin.v_max[0]),
                float(self.cfg.rail.v_max_m_s),
            )
        hard_lo = float(getattr(self.cfg.rail, "hard_min_m", 0.005))
        hard_hi = float(getattr(self.cfg.rail, "hard_max_m", 0.78))
        if not (
            np.isfinite(hard_lo)
            and np.isfinite(hard_hi)
            and float(self.kin.q_lower[0]) <= hard_lo < hard_hi
            and hard_hi <= float(self.kin.q_upper[0])
        ):
            raise ValueError(
                "invalid rail hard limits: "
                f"[{hard_lo:.6f}, {hard_hi:.6f}]"
            )
        self.limits.q_lower[0] = max(float(self.limits.q_lower[0]), hard_lo)
        self.limits.q_upper[0] = min(float(self.limits.q_upper[0]), hard_hi)
        self.limits.rail_soft_min_m = float(self.cfg.rail.soft_min_m)
        self.limits.rail_soft_max_m = float(self.cfg.rail.soft_max_m)
        self.core = QpIkController(self.kin, self.limits, self.cfg.qp)
        self.core.set_q_star(self.centering_task.q_target)
        self.core.set_q_star_signs(self.centering_task.q_target)
        self.q_cmd = np.zeros(kin.nv, dtype=float)
        self._arm_task_suppressed = False
        self._centering_suppressed = False
        self._manipulability_active = False
        self._box_dt_last_t: float | None = None
        self._box_h1_last: float | None = None
        self._dq_prev: np.ndarray | None = None
        alloc_cfg = getattr(self.cfg, "rail_allocator", RailAllocatorConfig())
        self.rail_allocator_cfg = alloc_cfg
        v_rail = float(self.limits.v_max[0])
        self.rail_ref_model = RailReferenceModel(
            f_c_hz=float(alloc_cfg.f_c_hz),
            a_max=float(self.cfg.a_max_rail_m_s2),
            j_max=float(self.cfg.qp.j_max_rail_m_s3),
            v_max=v_rail,
            reaction_s=float(alloc_cfg.reaction_s),
            soft_min_m=float(self.cfg.rail.soft_min_m),
            soft_max_m=float(self.cfg.rail.soft_max_m),
        )
        self.rail_observer = RailStateObserver(
            pos_gain=float(alloc_cfg.observer_pos_gain),
            vel_gain=float(alloc_cfg.observer_vel_gain),
            vel_lpf_hz=float(alloc_cfg.observer_vel_lpf_hz),
            v_max=v_rail,
        )
        self.last_v_r_ref = 0.0
        self.last_u_alloc = 0.0
        self.last_u_posture = 0.0
        self.last_u_mid = 0.0
        self.last_comp_projected_frac = 0.0
        self._midrange_freeze = False
        self.midranging = MidrangingController(
            kp=float(alloc_cfg.kp_mid),
            ki=float(alloc_cfg.ki_mid),
            v_max=float(alloc_cfg.u_mid_max_m_s),
        )
        if float(getattr(self.cfg.rail_extension, "v_lpf_fc_hz", 0.0) or 0.0) <= 0.0:
            self.cfg.rail_extension.v_lpf_fc_hz = float(alloc_cfg.f_c_hz)
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
        self.last_slack_norm: float = 0.0
        self._sat_scale: float = 1.0
        self.last_sat_scale: float = 1.0
        self.last_arm_rho: float = float("nan")
        self._press_z_mark: float = float("nan")
        self._press_stall_s: float = 0.0
        self._d_star_nudge_cool_s: float = 0.0
        self._family_ok: bool = True
        self._rail_mode: RailMode = self.cfg.rail.mode
        self._locked_style: LockedStyle = self.cfg.rail.locked_style
        self._configured_rail_mode: RailMode = self.cfg.rail.mode
        self._plan_drives_rail: bool = False
        self._direct_joint_ptp: bool = False
        self._last_post_step: dict = {}
        self._apply_rail_mode_side_effects()

    @property
    def rail_mode(self) -> RailMode:
        return self._rail_mode

    def set_plan_drives_rail(self, enabled: bool) -> None:
        self._plan_drives_rail = bool(enabled)

    def set_direct_joint_ptp(self, enabled: bool) -> None:
        self._direct_joint_ptp = bool(enabled)

    @property
    def configured_rail_mode(self) -> RailMode:
        return self._configured_rail_mode

    @property
    def locked_style(self) -> LockedStyle:
        return self._locked_style

    @property
    def is_locked_hold(self) -> bool:
        return (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.HOLD
        )

    def set_arm_task_suppressed(self, suppressed: bool) -> None:
        self._arm_task_suppressed = bool(suppressed)

    def set_centering_suppressed(self, suppressed: bool) -> None:
        self._centering_suppressed = bool(suppressed)

    def set_manipulability_active(self, active: bool) -> None:
        self._manipulability_active = bool(active) and self.manipulability_task is not None

    def set_rail_extension_active(self, active: bool) -> None:
        self._rail_ext_active = bool(active)

    def set_rail_extension_mode(self, mode: str) -> None:
        if self.rail_ext_task is not None:
            self.rail_ext_task.set_mode(mode)  # type: ignore[arg-type]

    def set_rail_pose_target(self, y_rail_m: float | None) -> None:
        if self.rail_ext_task is not None:
            self.rail_ext_task.set_rail_pose_target(y_rail_m)

    def capture_rail_extension_ref(self) -> None:
        if self.rail_ext_task is not None:
            self.rail_ext_task.capture_reference(self.q_cmd)

    def _measure_box_periods(self, dt: float) -> tuple[float, float | None]:
        """Two most recent wall periods for the unequal-sample third-order box.

        Each period is clamped to ``[0.8, 2.0] × dt`` (pass the nominal
        period, not the jittering wall period) so one stalled tick cannot
        open the acceleration/jerk boxes.
        """
        now = time.monotonic()
        prev = self._box_dt_last_t
        prev_h1 = self._box_h1_last
        self._box_dt_last_t = now
        nominal = max(float(dt), 1.0e-6)
        if prev is None:
            self._box_h1_last = nominal
            return nominal, None
        measured = now - prev
        if not math.isfinite(measured) or measured <= 0.0:
            h1 = nominal
        else:
            h1 = float(np.clip(measured, 0.8 * nominal, 2.0 * nominal))
        self._box_h1_last = h1
        return h1, prev_h1

    def _commit_command_step(
        self,
        q_prev: np.ndarray,
        dt_int: float,
        dt_nom: float,
    ) -> tuple[np.ndarray, bool]:
        """Project ``q_cmd`` through the post-QP step box, or only shadow it.

        Always writes the actually sent ``dq/dt_int`` into ``core.qdot_prev``.
        Does not touch ``qdot_prev2`` / ``_qdot_prev_seen`` — ``step()`` shifts
        those at the start of the next solve.
        """
        q_desired = np.asarray(self.q_cmd, dtype=float).copy()
        q_shadow, _dq_shadow, would_clamp = clamp_command_step(
            q_prev,
            q_desired,
            self._dq_prev,
            self.limits.a_max,
            dt_int,
        )
        q_final = np.asarray(q_shadow, dtype=float)
        clamp_applied = bool(would_clamp)
        dq_final = q_final - np.asarray(q_prev, dtype=float)
        self.q_cmd = np.asarray(q_final, dtype=float).copy()
        self._dq_prev = np.asarray(dq_final, dtype=float).copy()
        period = max(float(dt_int), 1.0e-12)
        qdot_committed = np.asarray(dq_final, dtype=float) / period
        self.core.qdot_prev = qdot_committed.copy()
        self._last_post_step = {
            "would_clamp": bool(would_clamp),
            "clamp_applied": bool(clamp_applied),
            "shadow_q": np.asarray(q_shadow, dtype=float).copy(),
            "qdot_committed": qdot_committed.copy(),
        }
        return qdot_committed, bool(clamp_applied)

    def _attach_post_qp_ab(
        self,
        step: JointIkStep,
        *,
        dt_nom: float,
        dt_int: float,
        box_h1: float | None,
        box_h2: float | None,
        qdot_raw: np.ndarray,
        qdot_pre_commit: np.ndarray,
        qdot_committed: np.ndarray,
        qdot_prev_used: np.ndarray,
        qdot_prev2_used: np.ndarray,
    ) -> JointIkStep:
        """Stamp the A/B fields that the CSV / offline box check need."""
        tel = self._last_post_step or {}
        step.post_qp_step_clamp_enabled = True
        step.post_step_would_clamp = bool(tel.get("would_clamp", False))
        step.post_step_clamp_applied = bool(tel.get("clamp_applied", False))
        step.acc_clamped = bool(tel.get("clamp_applied", step.acc_clamped))
        step.dt_nom_s = float(dt_nom)
        step.dt_int_s = float(dt_int)
        step.box_h1_s = (
            float(box_h1) if box_h1 is not None and np.isfinite(box_h1) else float("nan")
        )
        step.box_h2_s = (
            float(box_h2) if box_h2 is not None and np.isfinite(box_h2) else float("nan")
        )
        step.qdot_raw = np.asarray(qdot_raw, dtype=float).copy()
        step.qdot_pre_commit = np.asarray(qdot_pre_commit, dtype=float).copy()
        step.qdot_committed = np.asarray(qdot_committed, dtype=float).copy()
        step.qdot_prev_used = np.asarray(qdot_prev_used, dtype=float).copy()
        step.qdot_prev2_used = np.asarray(qdot_prev2_used, dtype=float).copy()
        step.box_lo = np.asarray(self.core.last_lo_box, dtype=float).copy()
        step.box_hi = np.asarray(self.core.last_hi_box, dtype=float).copy()
        shadow = tel.get("shadow_q")
        step.post_step_shadow_q = (
            np.asarray(shadow, dtype=float).copy()
            if shadow is not None
            else np.full(self.kin.nv, np.nan)
        )
        return step

    def plan_scan_stroke(
        self,
        y_center_m: float,
        amplitude_m: float,
        q_rad: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """One-shot min-max (d*, ψ*) at scan start.  Raises if infeasible."""
        q = self.q_cmd if q_rad is None else np.asarray(q_rad, dtype=float)
        if self.posture_retarget is None:
            y_tcp = float(self.kin.fk_placement(q).translation[1])
            d_star = y_tcp - float(q[0])
            if self.rail_ext_task is not None:
                self.rail_ext_task.set_d_pref(d_star)
            return d_star, float("nan")
        d_star, psi_star = self.posture_retarget.plan_stroke(
            q,
            y_center_m=float(y_center_m),
            amplitude_m=float(amplitude_m),
            rail_lo=float(self.limits.q_lower[0]),
            rail_hi=float(self.limits.q_upper[0]),
        )
        if self.arm_task is not None:
            self.arm_task.set_reference(float(psi_star))
        if self.rail_ext_task is not None:
            self.rail_ext_task.set_d_pref(float(d_star))
        return float(d_star), float(psi_star)

    def _check_design_family(self, q_meas: np.ndarray) -> None:
        qn = np.asarray(self.centering_task._q_target_default, dtype=float)
        ok = design_family_ok(q_meas, qn)
        self._family_ok = bool(ok)
        if ok:
            return
        from rm75_control.kinematics.srs_ik import branch_from_q, psi_from_q

        q = np.asarray(q_meas, dtype=float).reshape(-1)
        psi_m = math.degrees(fold_psi_to_positive(psi_from_q(q)))
        psi_n = math.degrees(fold_psi_to_positive(psi_from_q(qn)))
        msg = (
            "DESIGN FAMILY MISMATCH: measured "
            f"ψ={psi_m:.1f}° branch={int(branch_from_q(q))} "
            f"J1={math.degrees(float(q[1])):+.1f}° vs design "
            f"ψ={psi_n:.1f}° branch={int(branch_from_q(qn))} "
            f"J1={math.degrees(float(qn[1])):+.1f}°"
        )
        print(f"[joint_ik] {msg}", flush=True)
        if bool(getattr(self.cfg.psi_retarget, "require_design_family", False)):
            raise ValueError(msg)

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
        """Homotopy q* while s<1; yaml nominal after s≈1 so centering cannot chase a yanked IK."""
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

    def reset(self, q0_rad: np.ndarray) -> None:
        self.q_cmd = np.asarray(q0_rad, dtype=float).copy()
        self.core.reset()
        if self.arm_task is not None:
            self.arm_task.reset(self.q_cmd)
        if self.manipulability_task is not None:
            self.manipulability_task.reset()
        self.rail_task.reset(self.q_cmd)
        if self.rail_ext_task is not None:
            self.rail_ext_task.reset(self.q_cmd)
        if self.posture_retarget is not None:
            self.posture_retarget.reset(self.q_cmd)
            if self.rail_ext_task is not None and np.isfinite(
                self.posture_retarget.d_star_m
            ):
                self.rail_ext_task.set_d_pref(float(self.posture_retarget.d_star_m))
            if self.arm_task is not None and self.posture_retarget._psi_cmd is not None:
                self.arm_task.set_reference(float(self.posture_retarget._psi_cmd))
        self._box_dt_last_t = None
        self._box_h1_last = None
        self._dq_prev = None
        self.rail_ref_model.reset(float(self.q_cmd[0]) * 0.0)
        self.rail_observer.reset(float(self.q_cmd[0]), 0.0)
        self.midranging.reset()
        self._midrange_freeze = False
        self.last_v_r_ref = 0.0
        self.last_u_alloc = 0.0
        self.last_u_posture = 0.0
        self.last_u_mid = 0.0
        self.last_comp_projected_frac = 0.0
        self._direct_joint_ptp = False
        self._plan_drives_rail = False
        self._press_z_mark = float("nan")
        self._press_stall_s = 0.0
        self._d_star_nudge_cool_s = 0.0
        self._last_post_step = {}
        self.last_slack_norm = 0.0
        self._sat_scale = 1.0
        self.last_sat_scale = 1.0
        self._apply_rail_mode_side_effects()
        self._latch_attractor_from_q(self.q_cmd)

    def begin_hybrid_episode(
        self,
        q_meas: np.ndarray,
        qdot_applied: np.ndarray | None = None,
    ) -> None:
        """Preserve velocity continuity and latch yaml branch signs.

        Homotopy ``q*`` starts at the measured pose.  Barrier signs stay on
        the yaml family so a planar start can still fold toward design J1.
        """
        applied = self.core.qdot_prev if qdot_applied is None else qdot_applied
        self.core.sync_applied(applied)
        self._latch_attractor_from_q(q_meas)

    def set_rail_mode(
        self,
        mode: RailMode | str,
        *,
        q_ref_m: float | None = None,
        locked_style: LockedStyle | str | None = None,
    ) -> None:
        if isinstance(mode, str):
            mode = RailMode(mode)
        self._rail_mode = mode
        if locked_style is not None:
            if isinstance(locked_style, str):
                locked_style = LockedStyle(locked_style)
            self._locked_style = locked_style
        if q_ref_m is not None:
            if (
                mode == RailMode.LOCKED
                and self._locked_style == LockedStyle.HOLD
                and abs(float(q_ref_m) - float(self.q_cmd[0])) > 1.0e-9
            ):
                raise ValueError(
                    "locked HOLD cannot move rail to a different reference; "
                    "use a continuous RAIL_ONLY/TCP_FIXED phase first"
                )
            self.rail_task.set_reference(q_ref_m)
        elif mode == RailMode.LOCKED and self._locked_style == LockedStyle.HOLD:
            self.rail_task.reset(self.q_cmd)
        self._apply_rail_mode_side_effects()

    def set_coupled(self) -> None:
        self.set_rail_mode(RailMode.COUPLED)

    def set_locked(
        self,
        style: LockedStyle | str = LockedStyle.HOLD,
        *,
        q_ref_m: float | None = None,
    ) -> None:
        self.set_rail_mode(RailMode.LOCKED, q_ref_m=q_ref_m, locked_style=style)

    def _apply_rail_mode_side_effects(self) -> None:
        self.rail_task.cfg.mode = self._rail_mode
        self.rail_task.cfg.locked_style = self._locked_style
        self.cfg.rail.mode = self._rail_mode
        self.cfg.rail.locked_style = self._locked_style

    def _pin_rail_if_locked_hold(self) -> None:
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
        manipulability_active: bool | float | None = None,
        centering_sigma_fade: bool = True,
        dt_s: float | None = None,
    ) -> np.ndarray:
        slack = float(self.last_slack_norm)
        target = secondary_scale_from_slack(slack, self.cfg.saturation)
        dt = float(self.cfg.dt if dt_s is None else dt_s)
        tau = float(getattr(self.cfg.saturation, "secondary_scale_tau_s", 0.10) or 0.0)
        if tau <= 1.0e-9 or dt <= 0.0:
            sat_scale = target
        else:
            alpha = min(1.0, dt / tau)
            sat_scale = float(self._sat_scale) + alpha * (target - float(self._sat_scale))
        self._sat_scale = float(sat_scale)
        self.last_sat_scale = float(sat_scale)
        qdot0 = self.secondary.compose(
            q,
            qdot_ff,
            self.core.qdot_prev,
            arm_suppressed=self._arm_task_suppressed,
            sigma_min=self.last_sigma_min,
            sigma_ref=self.cfg.qp.sr_damping.sigma_ref,
            centering_suppressed=self._centering_suppressed,
            centering_sigma_fade=centering_sigma_fade,
            manipulability_active=(
                self._manipulability_active
                if manipulability_active is None
                else manipulability_active
            ),
            soft_scale=sat_scale,
            dt_s=dt,
        )
        self.last_secondary_norm = float(np.linalg.norm(qdot0))
        return qdot0

    def _clip_qdot_to_box(
        self,
        q_prev: np.ndarray,
        qdot: np.ndarray,
        dt: float,
        q_meas: np.ndarray | None,
        resync_vec: np.ndarray,
        *,
        rail_locked: bool,
        rail_vel_pin: float | None,
        box_h1: float | None = None,
        box_h2: float | None = None,
        rail_lead_exempt: bool = False,
    ) -> np.ndarray:
        qdot = np.asarray(qdot, dtype=float).reshape(-1).copy()
        if qdot.shape != q_prev.shape or not np.all(np.isfinite(qdot)):
            qdot = np.zeros_like(q_prev)
        q_geom = q_meas if q_meas is not None else q_prev
        lo, hi = self.core.constraints.bounds(
            q_geom,
            dt,
            self.core.qdot_prev,
            q_meas=q_meas,
            q_cmd=q_prev,
            resync_err=resync_vec,
            rail_locked=rail_locked,
            rail_lock_vel_eps_m_s=self.cfg.rail.lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin,
            qdot_prev2=self.core.qdot_prev2,
            j_max=self.core._j_max,
            box_h1=box_h1,
            box_h2=box_h2,
            rail_lead_exempt=rail_lead_exempt,
        )
        return scale_qdot_into_box(qdot, lo, hi)

    def _make_step(
        self,
        *,
        qdot: np.ndarray,
        twist_base: np.ndarray,
        sigma_min: float,
        manip: float,
        slack_norm: float,
        n_cbf_active: int,
        follow_err: float,
        qdot_ff_norm: float,
        vel_clamped: bool = False,
        acc_clamped: bool = False,
        pos_clamped: bool = False,
        rail_vel_pin: float | None = None,
        rail_qdot_ff: float = float("nan"),
        plan_drives_rail: bool = False,
        rail_ext_err_m: float = 0.0,
        rail_ext_weight: float = 0.0,
        mode: str = "qpik",
        failed: bool = False,
        fallback_reason: str = "",
        rail_macro_pref_v: float = 0.0,
        rail_escape_active: bool = False,
        rail_contrib_m_s: float = float("nan"),
        arm_contrib_m_s: float = float("nan"),
        rail_motion_share: float = float("nan"),
        scan_target: np.ndarray | None = None,
        scan_achieved: np.ndarray | None = None,
        scan_residual: np.ndarray | None = None,
        physical_saturated: bool = False,
    ) -> JointIkStep:
        slack = float(slack_norm)
        self.last_slack_norm = slack if np.isfinite(slack) else 0.0
        alpha = 0.0 if failed else float(np.clip(1.0 - slack, 0.0, 1.0))
        qp_total_ms = float(getattr(self.core, "last_qp_total_ms", 0.0))
        qp2_fallback = bool(getattr(self.core, "last_qp2_fallback", False))
        return JointIkStep(
            q_send=self.q_cmd.copy(),
            qdot=np.asarray(qdot, dtype=float).copy(),
            twist_base=np.asarray(twist_base, dtype=float).copy(),
            sigma_min=float(sigma_min),
            manip=float(manip),
            slack_norm=slack,
            n_cbf_active=int(n_cbf_active),
            follow_err_rad=float(follow_err),
            qdot_ff_norm=float(qdot_ff_norm),
            vel_clamped=bool(vel_clamped),
            acc_clamped=bool(acc_clamped),
            pos_clamped=bool(pos_clamped),
            rail_vel_pin=(
                float(rail_vel_pin) if rail_vel_pin is not None else float("nan")
            ),
            rail_qdot_ff=float(rail_qdot_ff),
            plan_drives_rail=bool(plan_drives_rail),
            arm_singularity_smooth=self.secondary.last_arm_smooth,
            limit_activation=self.secondary.last_limit_activation,
            rail_ext_err_m=float(rail_ext_err_m),
            rail_ext_weight=float(rail_ext_weight),
            rail_escape_active=bool(rail_escape_active),
            psi_deg=(
                float(np.degrees(self.arm_task.arm_angle(self.q_cmd)))
                if self.arm_task is not None
                else float("nan")
            ),
            psi_ref_deg=(
                float(np.degrees(self.arm_task.psi_ref))
                if self.arm_task is not None and self.arm_task.psi_ref is not None
                else float("nan")
            ),
            psi_retarget_score=(
                float(self.posture_retarget.last_psi_score)
                if self.posture_retarget is not None
                else float("nan")
            ),
            d_pref_m=(
                float(self.rail_ext_task.d_pref_m)
                if self.rail_ext_task is not None and self.rail_ext_task.d_pref_m is not None
                else float("nan")
            ),
            elbow_margin_rad=(
                float(self.posture_retarget.last_elbow_margin_rad)
                if self.posture_retarget is not None
                else float("nan")
            ),
            wrist_open_rad=(
                float(self.posture_retarget.last_wrist_open_rad)
                if self.posture_retarget is not None
                else float("nan")
            ),
            family_ok=bool(self._family_ok),
            physical_saturated=bool(physical_saturated),
            rail_contrib_m_s=float(rail_contrib_m_s),
            arm_contrib_m_s=float(arm_contrib_m_s),
            rail_motion_share=float(rail_motion_share),
            wln_scale_rail=float(self.core.last_wln_scale[0]),
            wln_scale_arm_max=float(np.max(self.core.last_wln_scale[1:])),
            waste_ratio=(
                (abs(float(rail_contrib_m_s)) + abs(float(arm_contrib_m_s)))
                / max(abs(float(rail_contrib_m_s) + float(arm_contrib_m_s)), 1.0e-9)
                if np.isfinite(rail_contrib_m_s) and np.isfinite(arm_contrib_m_s)
                else float("nan")
            ),
            rail_ff_m=(
                float(getattr(self.rail_ext_task, "last_rail_ff_m", float("nan")))
                if self.rail_ext_task is not None
                else float("nan")
            ),
            rail_posture_err_m=(
                float(getattr(self.rail_ext_task, "last_track_err_m", float("nan")))
                if self.rail_ext_task is not None
                else float("nan")
            ),
            d_star_m=(
                float(self.posture_retarget.d_star_m)
                if self.posture_retarget is not None
                else float("nan")
            ),
            psi_star_deg=(
                float(np.degrees(self.posture_retarget.psi_star_rad))
                if self.posture_retarget is not None
                and np.isfinite(self.posture_retarget.psi_star_rad)
                else float("nan")
            ),
            minmax_margin=(
                float(self.posture_retarget.last_minmax_margin)
                if self.posture_retarget is not None
                else float("nan")
            ),
            controller_mode=mode,
            qp_backend=self.core.backend_name,
            qp_solver_status=self.core.last_status if mode == "qpik" else "not_run",
            qp_solver_call_count=int(self.core.solve_count) if mode == "qpik" else 0,
            qp_solver_solve_ms=qp_total_ms if mode == "qpik" else 0.0,
            qp_solver_overrun=bool(
                mode == "qpik"
                and qp_total_ms > float(getattr(self.cfg.qp, "max_solve_ms", 5.0))
            ),
            qp1_status=(
                str(getattr(self.core, "last_qp1_status", "not_run"))
                if mode == "qpik"
                else "not_run"
            ),
            qp2_status=(
                str(getattr(self.core, "last_qp2_status", "not_run"))
                if mode == "qpik"
                else "not_run"
            ),
            qp1_solve_ms=(
                float(getattr(self.core, "last_qp1_solve_ms", 0.0))
                if mode == "qpik"
                else 0.0
            ),
            qp2_solve_ms=(
                float(getattr(self.core, "last_qp2_solve_ms", 0.0))
                if mode == "qpik"
                else 0.0
            ),
            qp_assembly_ms=(
                max(
                    qp_total_ms
                    - float(getattr(self.core, "last_qp1_solve_ms", 0.0))
                    - float(getattr(self.core, "last_qp2_solve_ms", 0.0))
                    - float(getattr(self.core, "last_fallback_ms", 0.0)),
                    0.0,
                )
                if mode == "qpik"
                else 0.0
            ),
            qp_fallback_ms=(
                float(getattr(self.core, "last_fallback_ms", 0.0))
                if mode == "qpik"
                else 0.0
            ),
            qpik_total_ms=qp_total_ms if mode == "qpik" else 0.0,
            qp2_fallback=qp2_fallback if mode == "qpik" else False,
            qpik_alpha=alpha,
            qpik_beta=1.0,
            qpik_authority=1.0,
            qpik_hard_residual_max=0.0,
            qpik_dexterity_slack=float(getattr(self.core, "last_dexterity_slack", 0.0)),
            qpik_branch_slack=float(getattr(self.core, "last_branch_slack", 0.0)),
            rail_macro_pref_v=float(rail_macro_pref_v),
            rail_decomposition_error=0.0,
            scan_target=(
                np.asarray(scan_target, dtype=float).copy()
                if scan_target is not None
                else np.zeros(2)
            ),
            scan_achieved=(
                np.asarray(scan_achieved, dtype=float).copy()
                if scan_achieved is not None
                else np.zeros(2)
            ),
            scan_residual=(
                np.asarray(scan_residual, dtype=float).copy()
                if scan_residual is not None
                else np.zeros(2)
            ),
            fallback_level="stop" if failed else "none",
            fallback_reason=fallback_reason,
            solver_fault_latched=bool(mode == "qpik" and failed),
            arm_health=float(sigma_min),
            a_mirror_frac=float(getattr(self.core, "last_a_mirror_frac", float("nan"))),
            j_mirror_frac=float(getattr(self.core, "last_j_mirror_frac", float("nan"))),
        )

    def update(
        self,
        twist: np.ndarray,
        dt: float | None = None,
        q_meas: np.ndarray | None = None,
        qdot_ff: np.ndarray | None = None,
        *,
        vel_ff: np.ndarray | None = None,
        pose_d: np.ndarray | None = None,
        f_ext_z: float | None = None,
        f_des_z: float | None = None,
        contact_active: bool = False,
        task_rotation_base: np.ndarray | None = None,
        task_safety_rows: tuple = (),
        path_twist: np.ndarray | None = None,
        feedback_twist: np.ndarray | None = None,
        v_force_z: float | None = None,
        rail_exec_vel_m_s: float | None = None,
        rail_exec_smooth_m_s: float | None = None,
        dt_wall_s: float | None = None,
    ) -> JointIkStep:
        del f_ext_z, f_des_z, contact_active, task_safety_rows
        path_twist_arr = (
            np.asarray(path_twist, dtype=float).reshape(6)
            if path_twist is not None
            else np.zeros(6)
        )
        feedback_twist_arr = (
            np.asarray(feedback_twist, dtype=float).reshape(6)
            if feedback_twist is not None
            else np.zeros(6)
        )
        dt_nom = self.cfg.dt if dt is None else float(dt)
        if not np.isfinite(dt_nom) or dt_nom <= 0.0:
            raise ValueError("dt must be finite and > 0")
        if dt_wall_s is None:
            dt = dt_nom
            dt_rail = dt_nom
        else:
            # Integrate on a clipped wall period so a single overrun cannot
            # emit a 2x command step.  Force/proxy dynamics still see the
            # raw wall period via dt_actual_s in the outer loop.
            dt = integration_period(dt_nom, dt_wall_s)
            dt_rail = dt
        dt_int = float(dt)
        qdot_prev_used = np.asarray(self.core.qdot_prev, dtype=float).copy()
        qdot_prev2_used = np.asarray(self.core._qdot_prev_seen, dtype=float).copy()
        qdot_raw = np.full(self.kin.nv, np.nan)
        qdot_pre_commit = np.full(self.kin.nv, np.nan)
        box_h1: float | None = None
        box_h2: float | None = None
        q_prev = np.asarray(self.q_cmd, dtype=float).copy()
        if (
            self._rail_mode == RailMode.COUPLED
            and self.rail_observer._initialized
            and self.rail_observer._last_sample_t is not None
        ):
            q_prev[0] = float(self.rail_observer.q_hat)
            self.q_cmd[0] = float(q_prev[0])
        if q_meas is None:
            raise ValueError("q_meas is required for every Cartesian QPIK tick")
        q_state = np.asarray(q_meas, dtype=float).copy()
        if q_state.shape != (self.kin.nv,) or not np.isfinite(q_state).all():
            raise ValueError(f"q_meas must be a finite {(self.kin.nv,)} vector")
        follow_err = float(np.max(np.abs(q_prev - q_state)))
        twist_task = np.asarray(twist, dtype=float).reshape(-1)
        if twist_task.size != 6 or not np.isfinite(twist_task).all():
            raise ValueError("twist must be a finite 6-vector")
        if rail_exec_vel_m_s is not None and not np.isfinite(float(rail_exec_vel_m_s)):
            raise ValueError("rail_exec_vel_m_s must be finite when supplied")
        if rail_exec_smooth_m_s is not None and not np.isfinite(
            float(rail_exec_smooth_m_s)
        ):
            raise ValueError("rail_exec_smooth_m_s must be finite when supplied")
        # Hardware supplies the time-stamped worker estimate.  Offline callers
        # have no independent actuator, so the last applied rail command is
        # the least-surprising zero-order execution estimate.
        if rail_exec_vel_m_s is not None:
            rail_exec_for_qp = float(rail_exec_vel_m_s)
        else:
            rail_exec_for_qp = float(self.last_v_r_ref)

        if task_rotation_base is not None:
            rotation_base_task = np.asarray(task_rotation_base, dtype=float)
            twist_base = np.concatenate(
                (
                    rotation_base_task @ twist_task[:3],
                    rotation_base_task @ twist_task[3:],
                )
            )
        else:
            twist_base = self._twist_to_base(twist_task, q_state)

        need_mass = bool(self.cfg.qp.use_mass_weighted_reg)
        if bool(getattr(self.cfg.qp, "use_cpp_kernel", True)):
            J_pre, sigma_values_pre, mass_pre = cpp_kernel.kinematics_snapshot(
                self.kin, q_state, need_mass=need_mass
            )
        else:
            J_pre = self.kin.jacobian(q_state)
            sigma_values_pre = self.kin.singular_values(J_pre)
            mass_pre = self.kin.mass_matrix(q_state) if need_mass else None
        sigma_pre = float(sigma_values_pre.min())
        sigma_arm = float(cpp_kernel.singular_values(J_pre[:, 1:]).min())
        sigma_ref = float(self.cfg.qp.sr_damping.sigma_ref)

        locked_hold = self.is_locked_hold
        rail_only = (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.RAIL_ONLY
        )
        tcp_fixed = (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.TCP_FIXED
        )
        if qdot_ff is not None:
            v_lim_ff = np.asarray(self.limits.v_max, dtype=float)
            qdot_ff = np.clip(np.asarray(qdot_ff, dtype=float), -v_lim_ff, v_lim_ff)

        if self._direct_joint_ptp and qdot_ff is not None:
            qdot_cmd = np.asarray(qdot_ff, dtype=float).copy()
            if rail_only:
                qdot_cmd[1:] = 0.0
            q_next = q_prev + qdot_cmd * dt
            q_next[0] = float(q_prev[0]) + float(qdot_cmd[0]) * float(dt_rail)
            self.q_cmd = q_next
            if dt > 1e-9:
                applied = (self.q_cmd - q_prev) / dt
                if dt_rail > 1e-9:
                    applied[0] = (
                        float(self.q_cmd[0]) - float(q_prev[0])
                    ) / float(dt_rail)
            else:
                applied = qdot_cmd
            self.core.sync_applied(applied)
            qdot_raw = np.asarray(qdot_cmd, dtype=float).copy()
            qdot_pre_commit = (
                np.asarray(self.q_cmd, dtype=float) - q_prev
            ) / max(float(dt), 1.0e-12)
            applied, acc_clamped = self._commit_command_step(q_prev, dt, dt_nom)
            self.last_sigma_min = sigma_pre
            J = J_pre
            sigma = sigma_values_pre
            return self._attach_post_qp_ab(
                self._make_step(
                    qdot=applied,
                    twist_base=twist_base,
                    sigma_min=float(sigma.min()),
                    manip=float(np.prod(sigma)),
                    slack_norm=0.0,
                    n_cbf_active=0,
                    follow_err=follow_err,
                    qdot_ff_norm=float(np.linalg.norm(qdot_ff)),
                    rail_vel_pin=float(qdot_ff[0]),
                    rail_qdot_ff=float(qdot_ff[0]),
                    plan_drives_rail=True,
                    acc_clamped=acc_clamped,
                    mode="direct_joint_ptp",
                ),
                dt_nom=dt_nom,
                dt_int=dt_int,
                box_h1=box_h1,
                box_h2=box_h2,
                qdot_raw=qdot_raw,
                qdot_pre_commit=qdot_pre_commit,
                qdot_committed=applied,
                qdot_prev_used=qdot_prev_used,
                qdot_prev2_used=qdot_prev2_used,
            )

        plan_drives_rail = rail_only or tcp_fixed or bool(self._plan_drives_rail)
        qdot_ff_sec = qdot_ff
        rail_vel_pin: float | None = None
        rail_qdot_ff_val = float("nan")
        if qdot_ff is not None:
            qdot_ff_arr = np.asarray(qdot_ff, dtype=float)
            v_rail = float(qdot_ff_arr[0])
            rail_qdot_ff_val = v_rail
            qdot_ff_sec = qdot_ff_arr.copy()
            qdot_ff_sec[0] = 0.0
            if plan_drives_rail:
                rail_vel_pin = v_rail

        resync_vec = np.full(self.kin.nv, float(self.cfg.resync_err_rad))
        resync_vec[0] = float(self.cfg.resync_err_rail_m)

        pose_now = self.kin.fk_pose(q_prev)
        z_now = float(pose_now[2])
        y_tcp_d = None
        tool_y_err_m = 0.0
        if pose_d is not None:
            pose_d_arr = np.asarray(pose_d, dtype=float).reshape(-1)
            if pose_d_arr.size >= 2 and np.isfinite(pose_d_arr[1]):
                y_tcp_d = float(pose_d_arr[1])
                tool_y_err_m = y_tcp_d - float(pose_now[1])

        ext_cfg = self.rail_ext_task.cfg if self.rail_ext_task is not None else None
        v_force = (
            float(v_force_z)
            if v_force_z is not None and np.isfinite(float(v_force_z))
            else float("nan")
        )
        v_min = float(getattr(ext_cfg, "press_v_force_min_m_s", 0.02))
        dz_max = float(getattr(ext_cfg, "press_dz_max_m", 0.002))
        y_thr = float(getattr(ext_cfg, "press_y_err_m", 0.005))
        stall_need = float(getattr(ext_cfg, "press_stall_s", 0.5))
        v_z_demand = (
            v_force if np.isfinite(v_force) else float(twist_base[2])
        )
        demanding = bool(abs(v_z_demand) >= v_min)
        # Windowed stall: |Δz| over the timer, not per 5 ms tick (2 mm/tick
        # is 400 mm/s — every real press looked "stuck").
        if demanding:
            if not np.isfinite(self._press_z_mark):
                self._press_z_mark = z_now
            z_progress = abs(z_now - self._press_z_mark)
            if z_progress > dz_max:
                self._press_z_mark = z_now
                self._press_stall_s = 0.0
                z_stuck = False
            else:
                self._press_stall_s += float(dt)
                z_stuck = True
        else:
            self._press_z_mark = float("nan")
            self._press_stall_s = 0.0
            z_stuck = False
        press_stalled_timer = self._press_stall_s + 1.0e-12 >= stall_need
        has_travel = bool(
            self.rail_ext_task is not None
            and self.rail_ext_task._rail_has_open_travel(float(q_state[0]))
        )
        policy_leave = bool(
            self.rail_ext_task is not None
            and self.rail_ext_task._in_leave_band(
                float(q_state[0]), self.rail_ext_task._policy_escape_sign(float(q_state[0]))
            )
        )
        arm_starved = bool(abs(tool_y_err_m) >= y_thr)
        comfort_m = float(self.cfg.qp.joint_comfort.m_comfort_rad)
        j4_blocked = bool(
            (float(self.limits.q_upper[4]) - float(q_prev[4])) <= comfort_m
            or (float(q_prev[4]) - float(self.limits.q_lower[4])) <= comfort_m
        )
        allow_press_escape = bool(
            demanding
            and has_travel
            and (
                press_stalled_timer
                or (j4_blocked and not policy_leave)
                or (policy_leave and arm_starved)
            )
        )

        lin_thr = (
            float(self.rail_ext_task.cfg.v_ff_thr_m_s)
            if self.rail_ext_task is not None
            else 0.01
        )
        hold_d_star = hold_setpoint_from_vel_ff(vel_ff, lin_thr_m_s=lin_thr)

        if (
            self.posture_retarget is not None
            and self._rail_mode == RailMode.COUPLED
        ):
            psi_ref, d_pref = self.posture_retarget.step(
                q_prev,
                float(dt),
                rail_lo=float(self.limits.q_lower[0]),
                rail_hi=float(self.limits.q_upper[0]),
                hold_setpoint=hold_d_star,
            )
            if self.arm_task is not None:
                self.arm_task.set_reference(float(psi_ref))
            if self.rail_ext_task is not None:
                self.rail_ext_task.set_d_pref(float(d_pref))
            self._publish_homotopy_centering()

        if (
            press_stalled_timer
            and allow_press_escape
            and self.posture_retarget is not None
            and self.rail_ext_task is not None
            and self._d_star_nudge_cool_s <= 0.0
        ):
            y_des = y_tcp_d if y_tcp_d is not None else float(pose_now[1])
            lo, hi = self.rail_ext_task._soft_travel()
            away = self.rail_ext_task._preferred_escape_sign(float(q_prev[0]))
            delta = -away * float(self.rail_ext_task.cfg.d_star_nudge_m)
            d_new = self.posture_retarget.nudge_d_star(
                delta, y_des_m=y_des, rail_lo=lo, rail_hi=hi
            )
            if np.isfinite(d_new):
                self.rail_ext_task.set_d_pref(float(d_new))
            self._d_star_nudge_cool_s = stall_need
        else:
            self._d_star_nudge_cool_s = max(
                0.0, self._d_star_nudge_cool_s - float(dt)
            )

        rail_task_vel: float | None = None
        rail_task_weight = 0.0
        rail_ext_err = 0.0
        rail_escape_active = False
        manip_weight: float | bool = self._manipulability_active
        if (
            self.rail_ext_task is not None
            and self._rail_ext_active
            and self._rail_mode == RailMode.COUPLED
        ):
            sigma_now = float(sigma_arm)
            sig_scale = 1.0
            sigma_esc_ref = max(
                sigma_ref,
                float(self.cfg.manipulability.sigma_fade_ref),
            )
            # No artificial floor: deep singularity must raise escape authority.
            if sigma_esc_ref > 1e-9 and sigma_now < sigma_esc_ref:
                sig_scale = max(sigma_now / sigma_esc_ref, 0.0)
            _g, self._sigma_grad_rail_cached = self._rail_goodness.refresh(
                q_prev, g_hint=sigma_pre
            )
            del _g
            u_max = max_limit_activation(
                q_prev,
                self.centering_task.q_mid,
                self.centering_task.half,
                activation=self.centering_task.cfg.activation,
            )
            joint_margin_frac = float(np.clip(1.0 - u_max, 0.0, 1.0))
            stroke_planned = bool(
                self.posture_retarget is not None and self.posture_retarget.planned
            )
            homing_split = False
            if self.posture_retarget is not None and not stroke_planned:
                homing_split = float(self.posture_retarget.homotopy_s) < 1.0 - 1.0e-6
            elbow_floor = float(self.cfg.qp.branch_barrier.box_activate_rad)
            if elbow_floor <= 1.0e-9:
                elbow_floor = float(self.cfg.qp.branch_barrier.activate_rad)
            block_escape = abs(float(q_prev[4])) < elbow_floor
            unload_sign = 0.0
            if (
                has_travel
                and self.posture_retarget is not None
                and np.isfinite(float(self.posture_retarget.d_star_m))
            ):
                j4_c = float(self.posture_retarget.cfg.elbow_center_rad)
                if abs(float(q_prev[4])) > j4_c:
                    y_now = float(self.kin.fk_placement(q_prev).translation[1])
                    d_live = y_now - float(q_prev[0])
                    unload_sign = float(
                        np.sign(d_live - float(self.posture_retarget.d_star_m))
                    )
            v_ext, w_ext = self.rail_ext_task(
                q_prev,
                sigma_scale=sig_scale,
                sigma_grad_rail=self._sigma_grad_rail_cached,
                vel_ff=vel_ff,
                dt_s=float(dt),
                joint_margin_frac=joint_margin_frac,
                sigma_raw=sigma_now,
                y_tcp_d=y_tcp_d,
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
                else 0.0
            )
            freeze_mid = bool(self._midrange_freeze) or bool(
                self.rail_ref_model.last_wall_override
            )
            u_mid = self.midranging.step(e_mid, float(dt), freeze=freeze_mid)
            u_r = float(u_alloc) + float(u_mid) + float(u_escape)
            v_r_ref = self.rail_ref_model.step(
                u_r, float(dt), x_m=float(q_state[0])
            )
            # Leave-band is applied once, after the reference model, so a
            # planned stroke cannot drive into the plus stop (or the
            # policy-side pin).  Mid-ranging away from the wall is kept.
            stroke_planned = bool(
                self.posture_retarget is not None and self.posture_retarget.planned
            )
            if self.rail_ext_task is not None and stroke_planned:
                y_r = float(q_state[0])
                if self.rail_ext_task._in_plus_leave(y_r) and v_r_ref > 0.0:
                    v_r_ref = 0.0
                    self.rail_ref_model.reset(0.0)
                pol = float(self.rail_ext_task._policy_escape_sign(y_r))
                if (
                    self.rail_ext_task._in_leave_band(y_r, pol)
                    and v_r_ref * pol > 0.0
                ):
                    v_r_ref = 0.0
                    self.rail_ref_model.reset(0.0)
            if abs(float(v_r_ref)) < 1.0e-4:
                v_r_ref = 0.0
            rail_task_vel = float(v_r_ref)
            self.last_v_r_ref = float(v_r_ref)
            self.last_u_alloc = float(u_alloc)
            self.last_u_posture = float(u_escape)
            self.last_u_mid = float(u_mid)
        else:
            self.last_v_r_ref = (
                float(rail_task_vel) if rail_task_vel is not None else 0.0
            )
            self.last_u_alloc = 0.0
            self.last_u_posture = (
                float(self.rail_ext_task.last_v_escape)
                if self.rail_ext_task is not None
                else 0.0
            )
            self.last_u_mid = 0.0

        rail_reg_scale = 1.0
        if self.rail_ext_task is not None:
            rail_reg_scale = float(
                getattr(self.rail_ext_task, "last_d_star_reg_scale", 1.0) or 1.0
            )

        # Hard box is 5/780 mm.  Do not freeze q0 in a leave/fade band.
        rail_sat_now = bool(
            self.rail_ext_task is not None
            and bool(getattr(self.rail_ext_task, "last_limit_saturated", False))
        )
        rail_vel_pin_eff = rail_vel_pin
        rail_task_weight_eff = rail_task_weight
        keep_task_weight = False
        pref_slack_scale = 1.0

        box_h1, box_h2 = self._measure_box_periods(dt_nom)
        qdot_history_before_solve = np.asarray(self.core.qdot_prev, dtype=float).copy()
        r = self.core.step(
            q_prev,
            twist_base,
            dt,
            secondary_qdot=self._secondary(
                q_prev,
                qdot_ff_sec,
                manipulability_active=manip_weight,
                centering_sigma_fade=not (
                    self._rail_ext_active and self._rail_mode == RailMode.COUPLED
                ),
                dt_s=float(dt),
            ),
            q_meas=q_state,
            resync_err=resync_vec,
            rail_locked=locked_hold,
            rail_lock_reg_scale=self.cfg.rail.lock_reg_scale,
            rail_reg_scale=rail_reg_scale,
            rail_lock_vel_eps_m_s=self.cfg.rail.lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin_eff,
            zero_secondary_rail=not locked_hold,
            rail_task_vel_m_s=rail_task_vel,
            rail_task_weight=rail_task_weight_eff,
            box_dt=box_h1,
            box_h1=box_h1,
            box_h2=box_h2,
            keep_task_weight=keep_task_weight,
            pref_slack_scale=pref_slack_scale,
            rail_exec_vel_m_s=rail_exec_for_qp,
            jacobian=J_pre,
            sigma=sigma_values_pre,
            mass_matrix=mass_pre,
            kinematics_ready=True,
            rail_open_travel=bool(
                self._rail_mode == RailMode.COUPLED
                and has_travel
                and not locked_hold
            ),
            arm_qdot_pref=arm_qdot_pref,
        )

        qdot_out = np.asarray(r.qdot, dtype=float).copy()
        qdot_raw = qdot_out.copy()
        failed = bool(self.core.last_failed)
        fallback_reason = "qp_failed" if failed else ""
        if qdot_out.shape != q_prev.shape or not np.all(np.isfinite(qdot_out)):
            qdot_out = np.zeros_like(q_prev)
            failed = True
            fallback_reason = "final_qdot_nonfinite_or_bad_shape"
        if failed:
            # One infeasible / max-iter tick must not kill the session.
            # Brake with the certified previous command, same as 3d095f2.
            decay = float(getattr(self.cfg.qp, "fail_qdot_decay", 0.85))
            qdot_out = np.asarray(qdot_history_before_solve, dtype=float) * decay
            v_lim = np.asarray(self.limits.v_max, dtype=float)
            qdot_out = np.clip(qdot_out, -v_lim, v_lim)
            self.q_cmd = q_prev + qdot_out * float(dt)
            self.q_cmd[0] = float(q_prev[0]) + float(qdot_out[0]) * float(dt_rail)
            self.core.qdot_prev = qdot_out.copy()
            qdot_pre_commit = qdot_out.copy()
            qdot_out, acc_clamped = self._commit_command_step(
                q_prev, dt, dt_nom
            )
            return self._attach_post_qp_ab(
                self._make_step(
                    qdot=qdot_out,
                    twist_base=twist_base,
                    sigma_min=r.sigma_min,
                    manip=r.manip,
                    slack_norm=r.slack_norm,
                    n_cbf_active=r.n_cbf_active,
                    follow_err=follow_err,
                    qdot_ff_norm=(
                        float(np.linalg.norm(qdot_ff)) if qdot_ff is not None else 0.0
                    ),
                    rail_vel_pin=rail_vel_pin_eff,
                    rail_qdot_ff=rail_qdot_ff_val,
                    plan_drives_rail=bool(plan_drives_rail),
                    rail_ext_err_m=rail_ext_err,
                    rail_ext_weight=rail_task_weight,
                    failed=False,
                    acc_clamped=acc_clamped,
                    fallback_reason="qp1_decay",
                    rail_macro_pref_v=(
                        float(rail_task_vel) if rail_task_vel is not None else 0.0
                    ),
                    rail_escape_active=rail_escape_active,
                ),
                dt_nom=dt_nom,
                dt_int=dt_int,
                box_h1=box_h1,
                box_h2=box_h2,
                qdot_raw=qdot_raw,
                qdot_pre_commit=qdot_pre_commit,
                qdot_committed=qdot_out,
                qdot_prev_used=qdot_prev_used,
                qdot_prev2_used=qdot_prev2_used,
            )
        else:
            qdot_certified = qdot_out.copy()
            q_candidate = q_prev + qdot_out * dt
            margin = np.asarray(self.limits.position_margin, dtype=float)
            if np.any(q_candidate < self.limits.q_lower + margin - 1.0e-9) or np.any(
                q_candidate > self.limits.q_upper - margin + 1.0e-9
            ):
                qdot_out = self._clip_qdot_to_box(
                    q_prev, qdot_out, dt, q_state, resync_vec,
                    rail_locked=locked_hold, rail_vel_pin=rail_vel_pin_eff,
                    box_h1=box_h1, box_h2=box_h2,
                    rail_lead_exempt=(
                        abs(float(q_prev[0]) - float(q_state[0]))
                        > float(self.cfg.resync_err_rail_m)
                    ),
                )
                fallback_reason = fallback_reason or "projected_into_velocity_box"

        # Do NOT shape qdot_out[0] here to "match the rail servo bandwidth".
        # Whatever is written here becomes core.qdot_prev below, which is the
        # base of the QP acceleration box and the jerk box — a first-order
        # filter therefore multiplies those limits by its own alpha instead
        # of just smoothing.
        q_next = q_prev + qdot_out * dt
        self.q_cmd = q_next
        self.core.qdot_prev = qdot_out.copy()

        # When ``dt_wall_s`` is supplied, ``dt == dt_rail`` and this is a
        # no-op.  Kept so a caller that still passes only nominal ``dt``
        # does not leave the rail one period behind the servo.
        self.q_cmd[0] = float(self.q_cmd[0]) + float(qdot_out[0]) * (
            float(dt_rail) - float(dt)
        )

        # At the command floor, refuse to jog back into the switch.
        lo0 = float(self.limits.q_lower[0])
        hi0 = float(self.limits.q_upper[0])
        self.q_cmd[0] = float(np.clip(self.q_cmd[0], lo0, hi0))
        if self.q_cmd[0] <= lo0 + 1.0e-4 and self.core.qdot_prev[0] < 0.0:
            self.q_cmd[0] = lo0
            self.core.qdot_prev[0] = 0.0
        elif self.q_cmd[0] >= hi0 - 1.0e-4 and self.core.qdot_prev[0] > 0.0:
            self.q_cmd[0] = hi0
            self.core.qdot_prev[0] = 0.0
        if plan_drives_rail and qdot_ff is not None and dt_rail > 1e-9:
            v_rail = float(np.asarray(qdot_ff)[0])
            y = float(q_prev[0] + v_rail * float(dt_rail))
            y_lo = float(self.limits.q_lower[0])
            y_hi = float(self.limits.q_upper[0])
            self.q_cmd[0] = float(np.clip(y, y_lo, y_hi))
            self.core.qdot_prev[0] = (self.q_cmd[0] - q_prev[0]) / float(dt_rail)
            if rail_only:
                self.q_cmd[1:] = q_prev[1:]
                self.core.qdot_prev[1:] = 0.0
        else:
            self._pin_rail_if_locked_hold()
        qdot_out = self.core.qdot_prev.copy()
        if locked_hold and self.cfg.rail.lock_hard_pin:
            qdot_out[0] = 0.0
        elif plan_drives_rail and qdot_ff is not None:
            qdot_out[0] = float(np.asarray(qdot_ff)[0])
            if rail_only:
                qdot_out[1:] = 0.0

        final_hard_violation, final_task_lock_violation = (
            self.core.validate_final_qdot(qdot_out)
        )
        self.core.last_final_hard_violation = float(final_hard_violation)
        self.core.last_final_task_lock_violation = float(final_task_lock_violation)
        final_tol = max(
            10.0 * float(getattr(self.cfg.qp, "eps_abs", 1.0e-6)),
            1.0e-5,
        )
        if (
            not np.isfinite(final_hard_violation)
            or not np.isfinite(final_task_lock_violation)
            or final_hard_violation > final_tol
            or final_task_lock_violation > final_tol
        ):
            # A limiter/lead rewrite is not allowed to break QP1.  If this
            # tick already has a certified QP command, publish that instead
            # of stopping; stop only when no certified command exists.
            hard_qp, lock_qp = self.core.validate_final_qdot(qdot_certified)
            if (
                np.isfinite(hard_qp)
                and np.isfinite(lock_qp)
                and hard_qp <= final_tol
                and lock_qp <= final_tol
            ):
                fallback_reason = (
                    fallback_reason or "limiter_rewrite_rejected_keep_qp"
                )
                qdot_out = qdot_certified.copy()
                self.q_cmd = q_prev + qdot_out * dt
                self.core.qdot_prev = qdot_out.copy()
                self.core.last_final_hard_violation = float(hard_qp)
                self.core.last_final_task_lock_violation = float(lock_qp)
            else:
                failed = True
                fallback_reason = (
                    "final_publication_certificate_failed:"
                    f"hard={final_hard_violation:.3e},"
                    f"task_lock={final_task_lock_violation:.3e}"
                )
                self.q_cmd = q_prev.copy()
                self.core.qdot_prev = qdot_history_before_solve.copy()
                qdot_out = np.zeros_like(q_prev)
        self.last_sigma_min = r.sigma_min
        self.last_arm_rho = float(r.sigma_min)
        qdot_pre_commit = qdot_out.copy()
        qdot_out, acc_clamped = self._commit_command_step(q_prev, dt, dt_nom)
        # Decompose achieved linear velocity into rail vs arm along primary motion.
        J_fin = J_pre
        qdot_arr = np.asarray(qdot_out, dtype=float)
        twist_rail = J_fin[:, 0] * float(rail_exec_for_qp)
        twist_arm = J_fin[:, 1:] @ qdot_arr[1:]
        motion_dir = np.asarray(twist_base[:3], dtype=float)
        if vel_ff is not None:
            vff = np.asarray(vel_ff, dtype=float).reshape(-1)
            if vff.size >= 3 and float(np.linalg.norm(vff[:3])) > 1e-6:
                motion_dir = vff[:3].astype(float)
        n_dir = float(np.linalg.norm(motion_dir))
        if n_dir <= 1e-9:
            # Idle ticks have no commanded direction, which used to blank the
            # whole split for the entire release window — exactly where the
            # TCP leak lives.  The rail's own TCP axis is always defined and
            # is the axis the leak shows up on.
            rail_axis = np.asarray(J_fin[:3, 0], dtype=float)
            n_axis = float(np.linalg.norm(rail_axis))
            if n_axis > 1e-9:
                motion_dir = rail_axis
                n_dir = n_axis
        if n_dir > 1e-9:
            u = motion_dir / n_dir
            rail_contrib = float(np.dot(twist_rail[:3], u))
            arm_contrib = float(np.dot(twist_arm[:3], u))
            denom = abs(rail_contrib) + abs(arm_contrib)
            rail_share = (abs(rail_contrib) / denom) if denom > 1e-9 else float("nan")
        else:
            rail_contrib = float("nan")
            arm_contrib = float("nan")
            rail_share = float("nan")
        # Keep qpik_scan_* alive: primary linear motion des/achieved/residual (m).
        scan_t = np.array(
            [float(twist_base[0]), float(twist_base[1])], dtype=float
        )
        scan_a = np.array(
            [
                float(twist_rail[0] + twist_arm[0]),
                float(twist_rail[1] + twist_arm[1]),
            ],
            dtype=float,
        )
        q_now = np.asarray(self.q_cmd, dtype=float)
        near_arm_m = float(getattr(self.cfg.qp, "near_arm_margin_rad", 0.08))
        near_arm = bool(
            np.any(q_now[1:] < self.limits.q_lower[1:] + near_arm_m)
            or np.any(q_now[1:] > self.limits.q_upper[1:] - near_arm_m)
        )
        physical_saturated = bool(near_arm)
        step = self._make_step(
            qdot=qdot_out,
            twist_base=twist_base,
            sigma_min=r.sigma_min,
            manip=r.manip,
            slack_norm=r.slack_norm,
            n_cbf_active=r.n_cbf_active,
            follow_err=follow_err,
            qdot_ff_norm=float(np.linalg.norm(qdot_ff)) if qdot_ff is not None else 0.0,
            rail_vel_pin=rail_vel_pin_eff,
            rail_qdot_ff=rail_qdot_ff_val,
            plan_drives_rail=bool(plan_drives_rail),
            rail_ext_err_m=rail_ext_err,
            rail_ext_weight=rail_task_weight,
            failed=failed,
            acc_clamped=acc_clamped,
            fallback_reason=fallback_reason,
            rail_macro_pref_v=(
                float(rail_task_vel) if rail_task_vel is not None else 0.0
            ),
            rail_escape_active=rail_escape_active,
            rail_contrib_m_s=rail_contrib,
            arm_contrib_m_s=arm_contrib,
            rail_motion_share=rail_share,
            scan_target=scan_t,
            scan_achieved=scan_a,
            scan_residual=scan_t - scan_a,
            physical_saturated=physical_saturated,
        )
        actual_task_twist = twist_rail + twist_arm
        actual_task_residual = np.asarray(twist_base, dtype=float) - actual_task_twist
        step.protected_target = np.asarray(twist_base, dtype=float).copy()
        step.protected_achieved = np.asarray(actual_task_twist, dtype=float).copy()
        step.protected_residual = np.asarray(actual_task_residual, dtype=float).copy()
        step.qpik_working_slack = np.asarray(actual_task_residual, dtype=float).copy()
        step.qpik_equality_residual_max = float(np.max(np.abs(actual_task_residual)))
        step.qpik_hard_residual_max = float(
            getattr(self.core, "last_final_hard_violation", 0.0)
        )
        step.rail_xy_contribution = np.asarray(twist_rail[:2], dtype=float).copy()
        step.arm_xy_contribution = np.asarray(twist_arm[:2], dtype=float).copy()
        step.rail_exec_velocity_m_s = float(rail_exec_for_qp)
        step.rail_exec_for_qp_m_s = float(rail_exec_for_qp)
        if rail_exec_vel_m_s is not None:
            step.rail_measured_velocity_m_s = float(rail_exec_vel_m_s)
        if rail_exec_smooth_m_s is not None:
            step.rail_commanded_velocity_m_s = float(rail_exec_smooth_m_s)
        step.qp_solver_overrun = bool(getattr(self.core, "last_qp_overrun", False))
        step.qp1_status = str(getattr(self.core, "last_qp1_status", step.qp1_status))
        step.qp2_status = str(getattr(self.core, "last_qp2_status", step.qp2_status))
        step.qp2_fallback = bool(getattr(self.core, "last_qp2_fallback", False))
        step.rail_sat = bool(rail_sat_now)
        step.last_limit_saturated = bool(
            self.rail_ext_task is not None
            and self.rail_ext_task.last_limit_saturated
        )
        step.keep_task_weight = bool(keep_task_weight)
        step.pref_slack_scale = float(pref_slack_scale)
        step.rail_task_vel = (
            float(rail_task_vel) if rail_task_vel is not None else float("nan")
        )
        if self.rail_ext_task is not None:
            step.v_escape = float(self.rail_ext_task.last_v_escape)
            step.v_reach = float(self.rail_ext_task.last_v_reach)
            step.v_ff_rail = float(self.rail_ext_task.last_v_ff)
        step.u_alloc = float(self.last_u_alloc)
        step.u_posture = float(self.last_u_posture)
        step.u_mid = float(self.last_u_mid)
        step.v_r_ref = float(self.last_v_r_ref)
        step.comp_projected_frac = float(
            getattr(self.core, "last_comp_projected_frac", self.last_comp_projected_frac)
        )
        self.last_comp_projected_frac = float(step.comp_projected_frac)
        step.wall_override = bool(self.rail_ref_model.last_wall_override)
        step.slack_zero_feasible = bool(
            getattr(self.core, "last_zero_slack_feasible", False)
        )
        step.sigma_arm = float(sigma_arm)
        step.sns_scale = float(getattr(self.core, "last_sns_scale", 1.0))
        step.v_cmd = np.asarray(twist_base, dtype=float).reshape(6).copy()
        step.path_twist = np.asarray(path_twist_arr, dtype=float).reshape(6).copy()
        step.feedback_twist = np.asarray(
            feedback_twist_arr, dtype=float
        ).reshape(6).copy()
        comfort = getattr(self.core, "last_comfort_slack", None)
        if comfort is not None:
            step.comfort_slack = np.asarray(comfort, dtype=float).reshape(-1)[:7]
        step.cbf_min_dist = float(
            getattr(self.core, "last_cbf_min_dist", float("nan"))
        )
        step.cbf_pair = str(getattr(self.core, "last_cbf_pair", "") or "")
        step.nullspace_norm = float(self.last_secondary_norm)
        step.nullspace_centering_norm = float(self.secondary.last_centering_norm)
        step.nullspace_manip_norm = float(self.secondary.last_manip_norm)
        step.nullspace_arm_angle_norm = float(self.secondary.last_arm_angle_norm)
        step.nullspace_damping_norm = float(self.secondary.last_damping_norm)
        step.nullspace_rail_lock_norm = float(self.secondary.last_rail_lock_norm)
        return self._attach_post_qp_ab(
            step,
            dt_nom=dt_nom,
            dt_int=dt_int,
            box_h1=box_h1,
            box_h2=box_h2,
            qdot_raw=qdot_raw,
            qdot_pre_commit=qdot_pre_commit,
            qdot_committed=qdot_out,
            qdot_prev_used=qdot_prev_used,
            qdot_prev2_used=qdot_prev2_used,
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
        self.last_pose_d: np.ndarray | None = None
        self.last_path_twist = np.zeros(6)
        self.last_feedback_twist = np.zeros(6)
        self._reference_override = None

    def begin_hybrid_episode(
        self,
        applied_twist_base: np.ndarray,
        current_pose: np.ndarray,
    ) -> None:
        """Reset force-task transients and seed the output from applied motion."""

        seed = np.asarray(applied_twist_base, dtype=float).reshape(6).copy()
        if self.controller.cfg.control_frame == "tool":
            rotation = Rsc.from_euler(
                self.controller.cfg.euler_order,
                np.asarray(current_pose, dtype=float)[3:6],
                degrees=False,
            ).as_matrix()
            seed[:3] = rotation.T @ seed[:3]
            seed[3:] = rotation.T @ seed[3:]
        self.controller.begin_hybrid_episode(seed)
        self.last_path_twist.fill(0.0)
        self.last_feedback_twist.fill(0.0)

    def set_reference_override(self, reference) -> None:
        self._reference_override = reference

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        if hasattr(self.reference, "set_origin"):
            self.reference.set_origin(pose0, t_s=t_s)

    def sample(
        self,
        t_s: float,
        current_pose: np.ndarray,
        f_ext: np.ndarray,
        f_ext_raw: np.ndarray | None = None,
        dt_actual: float | None = None,
        sensor_age_s: float | None = None,
        feedback_age_s: float | None = None,
        feedback_fresh_tick: bool | None = None,
        feedback_velocity_valid: bool | None = None,
        v_tcp_z_actual: float | None = None,
    ) -> np.ndarray:
        ref = self._reference_override
        self._reference_override = None
        if ref is None:
            ref = self.reference.sample(t_s)
        self.last_pose_d = np.asarray(ref.pose_d, dtype=float).copy()
        self.last_vel_ff = np.asarray(ref.vel_ff, dtype=float).copy()
        # ``feedback_fresh_tick`` is a per-cycle telemetry edge, not a
        # validity gate: when one UDP frame is missed, retain the last valid
        # velocity and let ``feedback_age_s`` decide staleness.  Before the
        # first successful finite-difference estimate, pass no velocity so
        # BEFM remains fail-closed.
        velocity_valid = (
            bool(feedback_velocity_valid)
            if feedback_velocity_valid is not None
            else v_tcp_z_actual is not None
        )
        v_actual = v_tcp_z_actual if velocity_valid else None
        command = self.controller.compute_velocity_command(
            current_pose,
            ref.pose_d,
            ref.vel_ff,
            f_ext,
            self.desired_force,
            f_ext_raw=f_ext_raw,
            dt_actual=dt_actual,
            sensor_age_s=sensor_age_s,
            feedback_age_s=feedback_age_s,
            feedback_fresh=None,
            v_tcp_z_actual=v_actual,
        )
        pose_track = np.asarray(
            getattr(self.controller, "last_pose_d_combined", ref.pose_d),
            dtype=float,
        ).reshape(-1)
        if pose_track.size != 6 or not np.isfinite(pose_track).all():
            pose_track = np.asarray(ref.pose_d, dtype=float)
        tr_mm, tr_deg = pose_track_error_mm_deg(
            pose_track,
            current_pose,
            track_axes=self.controller.cfg.track_axes,
            euler_order=self.controller.cfg.euler_order,
        )
        self.last_err_mm = tr_mm
        self.last_track_rot_deg = tr_deg
        path = np.asarray(ref.vel_ff, dtype=float).reshape(6).copy()
        if self.controller.cfg.control_frame == "tool":
            rotation = Rsc.from_euler(
                self.controller.cfg.euler_order,
                current_pose[3:6],
                degrees=False,
            ).as_matrix()
            path[:3] = rotation.T @ path[:3]
            path[3:] = rotation.T @ path[3:]
        self.last_path_twist = np.asarray(
            self.controller.last_path_twist, dtype=float
        ).copy()
        self.last_feedback_twist = np.asarray(
            self.controller.last_feedback_twist, dtype=float
        ).copy()
        return command


@dataclass
class CartesianTrackConfig:
    """PD + feedforward Cartesian tracking (no force axis)."""

    k_task: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0, 10.0, 2.0, 2.0, 2.0]))
    max_pos_err_m: float = 0.05
    max_rot_err_rad: float = 0.35
    max_lin_vel_m_s: float = 0.4
    max_ang_vel_rad_s: float = 1.5
    euler_order: str = "xyz"
    # Must match JointIkConfig.control_frame (tool twist is rotated by R @ twist).
    control_frame: str = "tool"
    path_feedforward: bool = True


class CartesianTrackOuterLoop:
    """PD + feedforward Cartesian tracking against measured pose (no force)."""

    def __init__(self, reference, cfg: CartesianTrackConfig | None = None) -> None:
        self.reference = reference
        self.cfg = cfg or CartesianTrackConfig()
        self.last_err_mm: float = 0.0
        self.last_vel_ff: np.ndarray | None = None
        self.last_pose_d: np.ndarray | None = None
        self.last_path_twist = np.zeros(6)
        self.last_feedback_twist = np.zeros(6)
        self._reference_override = None

    def set_reference_override(self, reference) -> None:
        self._reference_override = reference

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        if hasattr(self.reference, "set_origin"):
            self.reference.set_origin(pose0, t_s=t_s)

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        del f_ext
        cfg = self.cfg
        ref = self._reference_override
        self._reference_override = None
        if ref is None:
            ref = self.reference.sample(t_s)
        self.last_pose_d = np.asarray(ref.pose_d, dtype=float).copy()
        self.last_vel_ff = np.asarray(ref.vel_ff, dtype=float).copy()
        err = pose_error(ref.pose_d, current_pose, cfg.euler_order)
        self.last_err_mm = float(np.linalg.norm(err[:3]) * 1000.0)
        err_sat = saturate_error(err, cfg.max_pos_err_m, cfg.max_rot_err_rad)
        v_ff = np.asarray(ref.vel_ff, dtype=float)
        path_base = v_ff.copy() if cfg.path_feedforward else np.zeros(6)
        feedback_base = cfg.k_task * err_sat

        def cap_twist(value: np.ndarray) -> np.ndarray:
            capped = np.asarray(value, dtype=float).copy()
            lin_norm = float(np.linalg.norm(capped[:3]))
            if cfg.max_lin_vel_m_s > 0.0 and lin_norm > cfg.max_lin_vel_m_s:
                capped[:3] *= cfg.max_lin_vel_m_s / lin_norm
            ang_norm = float(np.linalg.norm(capped[3:6]))
            if cfg.max_ang_vel_rad_s > 0.0 and ang_norm > cfg.max_ang_vel_rad_s:
                capped[3:6] *= cfg.max_ang_vel_rad_s / ang_norm
            return capped

        path_base = cap_twist(path_base)
        feedback_base = cap_twist(feedback_base)
        v = cap_twist(path_base + feedback_base)  # base-frame legacy output

        if cfg.control_frame == "tool":
            R = Rsc.from_euler(cfg.euler_order, current_pose[3:6], degrees=False).as_matrix()
            out = np.zeros(6, dtype=float)
            out[:3] = R.T @ v[:3]
            out[3:6] = R.T @ v[3:6]
            path = np.zeros(6)
            path[:3] = R.T @ path_base[:3]
            path[3:6] = R.T @ path_base[3:6]
            feedback = np.zeros(6)
            feedback[:3] = R.T @ feedback_base[:3]
            feedback[3:6] = R.T @ feedback_base[3:6]
            self.last_path_twist = path
            self.last_feedback_twist = feedback
            return out
        self.last_path_twist = path_base
        self.last_feedback_twist = feedback_base
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
        self.last_qdot_command: np.ndarray | None = None
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
        self.last_qdot_command = qdot_cmd.copy()
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


def _pin_control_cpu(cpu: int | None) -> bool:
    """Best-effort CPU affinity for the calling thread."""
    if cpu is None:
        return False
    try:
        os.sched_setaffinity(0, {int(cpu)})
        return True
    except (PermissionError, OSError, AttributeError, ValueError):
        return False


class _CStateGuard:
    """Hold ``/dev/cpu_dma_latency`` at 0 so the CPU stays out of deep C-states."""

    def __init__(self) -> None:
        self._fd = None

    def __enter__(self) -> "_CStateGuard":
        try:
            self._fd = open("/dev/cpu_dma_latency", "wb", buffering=0)
            self._fd.write(b"\x00\x00\x00\x00")
        except OSError:
            self._fd = None
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            try:
                self._fd.close()
            except OSError:
                pass
            self._fd = None

    @property
    def active(self) -> bool:
        return self._fd is not None


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


def reference_time_step(dt_elapsed_s: float, scale: float) -> float:
    """Advance ``t_ref`` by the time that actually passed, not ``dt_nom``."""
    elapsed = float(dt_elapsed_s)
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        return 0.0
    return elapsed * float(scale)


@dataclass
class Phase:
    """One leg of a multi-phase on-robot run (shared inner loop / watchdog).

    ``t_ref`` advances by ``dt_wall * governor_scale`` so the reference
    plays in real time. Set ``governor_err_max_mm=0`` to disable Cartesian
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
    arrival_plan_duration_s: float | None = None
    arrival_dwell_s: float = 0.0
    arrival_arm_speed_rad_s: float = 0.02
    arrival_rail_speed_m_s: float = 0.003
    governor_err_ok_mm: float = 5.0
    governor_err_max_mm: float = 25.0
    governor_scale_min: float = 0.25
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


@dataclass
class _ArrivalDwellGate:
    """Require plan completion, geometric arrival, and settled sent velocity."""

    plan_duration_s: float | None
    dwell_required_s: float
    arm_speed_rad_s: float
    rail_speed_m_s: float
    dwell_s: float = 0.0

    def update(
        self,
        *,
        geometric_arrival: bool,
        t_ref_s: float,
        qdot_applied: np.ndarray,
        dt_s: float,
        rail_settled: bool | None = None,
    ) -> bool:
        qdot = np.asarray(qdot_applied, dtype=float).reshape(-1)
        if qdot.size != 8 or not np.all(np.isfinite(qdot)):
            self.dwell_s = 0.0
            return False
        plan_complete = bool(
            self.plan_duration_s is None
            or float(t_ref_s) >= float(self.plan_duration_s) - 1.0e-12
        )
        rail_speed_ok = (
            abs(float(qdot[0])) <= max(float(self.rail_speed_m_s), 0.0)
            if rail_settled is None
            else bool(rail_settled)
        )
        speed_ok = bool(
            rail_speed_ok
            and np.max(np.abs(qdot[1:]), initial=0.0)
            <= max(float(self.arm_speed_rad_s), 0.0)
        )
        candidate = bool(geometric_arrival and plan_complete and speed_ok)
        if candidate:
            self.dwell_s += max(float(dt_s), 0.0)
        else:
            self.dwell_s = 0.0
        return bool(candidate and (
            self.dwell_s + 1.0e-12 >= max(float(self.dwell_required_s), 0.0)
        ))


class _TickLogger:
    """Async per-tick CSV telemetry (background writer; no sync flush in the RT loop)."""

    @staticmethod
    def _json_compact(value) -> str:
        """Encode structured telemetry as deterministic, strict JSON.

        CSV remains the transport for compatibility with existing replay
        tools.  Variable-length task rows/groups are kept in one compact JSON
        cell; non-finite floats become ``null`` instead of invalid JSON NaN.
        """

        def normalize(item):
            if isinstance(item, np.ndarray):
                return normalize(item.tolist())
            if isinstance(item, np.generic):
                return normalize(item.item())
            if isinstance(item, dict):
                return {
                    str(key): normalize(item[key])
                    for key in sorted(item, key=lambda key: str(key))
                }
            if isinstance(item, (tuple, list)):
                return [normalize(entry) for entry in item]
            if isinstance(item, (float, np.floating)):
                return float(item) if np.isfinite(item) else None
            if isinstance(item, (int, np.integer, bool, str)) or item is None:
                return item
            value = getattr(item, "value", None)
            if value is not None and value is not item:
                return normalize(value)
            return str(item)

        return json.dumps(
            normalize(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def _json_field(self, value) -> str:
        """Skip per-tick JSON dumps unless verbose telemetry is on."""

        if not self._verbose_json:
            return ""
        return self._json_compact(value)

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
           "dt_actual_s", "deadline_slack_s", "sensor_age_s", "feedback_age_s",
           "feedback_fresh_tick",
           "fx_raw_comp", "fy_raw_comp", "fz_raw_comp",
           "vz_achieved_tool", "contact_present",
           "force_pred_z", "force_dot_z", "cap_press_z", "cap_retract_z",
           "ke_update_gated", "ke_dx_m", "ke_df_n", "ke_update_count",
           "governor_scale", "governor_scale_raw", "sigma_min",
           "qdot_norm", "qdot_max_frac_vmax",
           "qdot_ff_norm", "tcp_jump_mm",
           "rail_target_sent_m", "rail_meas_m", "rail_cmd_meas_err_m",
           "rail_vel_pin", "plan_drives_rail", "rail_qdot_ff",
           # Motion-subspace accuracy (force axes excluded from "准" metrics).
           "pose_d_x", "pose_d_y", "pose_d_z", "pose_d_rx", "pose_d_ry", "pose_d_rz",
           "pose_meas_x", "pose_meas_y", "pose_meas_z",
           "pose_meas_rx", "pose_meas_ry", "pose_meas_rz",
           "motion_err_lin_x_mm", "motion_err_lin_y_mm", "motion_err_lin_z_mm",
           "motion_err_rot_x_deg", "motion_err_rot_y_deg", "motion_err_rot_z_deg",
           "motion_err_rms_mm", "motion_axis_peak_mm",
           "vel_ff_vx", "vel_ff_vy", "vel_ff_vz", "vel_ff_wx", "vel_ff_wy", "vel_ff_wz",
           "rail_contrib_m_s", "arm_contrib_m_s", "arm_y_qdot", "rail_motion_share",
           "rail_exec_for_qp",
           # Chan-Dubey reg multipliers: rail first, then the worst arm joint.
           "wln_scale_rail", "wln_scale_arm_max",
           "waste_ratio", "rail_ff_m", "rail_posture_err_m",
           "rail_escape_active",
           "psi_deg", "psi_ref_deg", "psi_retarget_score", "d_pref_m",
           "d_star_m", "psi_star_deg", "minmax_margin",
           "elbow_margin_rad", "wrist_open_rad", "family_ok",
           "tool_y_des_m", "tool_y_err_mm",
           "contact_phase", "v_air_cmd", "ke_hat", "dob_v", "barrier_cap_floor",
           # Append-only normal-axis BEFM/audit schema.
           "flow_x_p", "flow_v_p", "flow_v_aux", "flow_x_a", "flow_v_a",
           "flow_e", "flow_edot", "flow_F_c", "flow_v_track",
           "flow_P_e", "flow_P_c", "flow_alpha_target", "flow_alpha",
           "flow_alpha_case", "flow_T", "flow_psi", "flow_S_n",
           "flow_S_r_hat", "flow_P_phys", "flow_P_mismatch",
           "flow_E_phys", "flow_E_mismatch", "flow_gamma_active",
           # Observe-mode evidence: press (m/s) the gate would have removed.
           "flow_alpha_would_gate", "flow_edot_aligned",
           "flow_sign_fault", "flow_feedback_stale", "flow_blocked_reason",
           "contact_episode_rearm_event", "contact_episode_release_s",
           "surface_force_scale", "surface_force_alpha", "surface_xy_error_m",
           "force_barrier_contact_active",
           # Fixed single-shot QPIK telemetry.
           "qpik_backend", "qpik_solver_status", "qpik_solver_iterations",
           "qpik_solver_solve_ms", "qpik_solver_call_count",
           "qpik_solver_overrun",
           "qpik_qp1_status", "qpik_qp2_status",
           "qpik_qp1_solve_ms", "qpik_qp2_solve_ms",
           "qpik_assembly_ms", "qpik_fallback_ms",
           "qpik_total_ms", "qpik_qp2_fallback",
           "tick_inner_ms", "tick_send_ms", "tick_log_ms",
           "qpik_alpha", "qpik_beta", "qpik_authority",
           "qpik_equality_residual_max", "qpik_hard_residual_max",
           "qpik_anchor_valid", "qpik_recovery_overflow",
           "qpik_protected_nominal_overflow_json",
           "qpik_recovery_caps_json",
           "qpik_recovery_overflow_indices_json",
           "qpik_hard_active_constraint_ids_json",
           "qpik_protected_target_json", "qpik_protected_achieved_json",
           "qpik_protected_residual_json",
           "qpik_scan_target_json", "qpik_scan_achieved_json",
           "qpik_scan_residual_json", "qpik_working_slack_json",
           "qpik_collision_slack_json", "qpik_dexterity_slack",
           "qpik_branch_slack", "qpik_rail_macro_pref_v",
           "qpik_rail_center_pref_v",
           "qpik_rail_final_qdot", "qpik_arm_risk_pref_norm",
           "qpik_arm_risk_pref_json", "qpik_risk_direction_cosine",
           "qpik_path_velocity_xy_json",
           "qpik_feedback_xy_raw_json", "qpik_feedback_xy_filtered_json",
           "qpik_rail_xy_contribution_json", "qpik_arm_xy_contribution_json",
           "qpik_rail_task_projection", "qpik_rail_arm_cancel",
           "qpik_rail_decomposition_error",
           "qpik_arm_rho", "qpik_joint_margin_rad",
           "qpik_wrist_margin_rad", "qpik_wrist_singularity",
           "qpik_accepted_reference_lag_s",
           "qpik_pre_solve_feedback_age_s", "qpik_post_solve_feedback_age_s",
           "qpik_q_cmd_q_meas_norm", "qpik_fallback_level",
           "qpik_fallback_reason", "qpik_solver_fault_latched",
           "qpik_final_sent_qdot_json",
           "post_qp_step_clamp_enabled",
           "post_step_would_clamp",
           "post_step_clamp_applied",
           "dt_nom_s", "dt_int_s", "box_h1_s", "box_h2_s",
           "qpik_qdot_raw_json",
           "qpik_qdot_pre_commit_json",
           "qpik_qdot_committed_json",
           "qpik_qdot_prev_used_json",
           "qpik_qdot_prev2_used_json",
           "qpik_box_lo_json",
           "qpik_box_hi_json",
           "post_step_shadow_q_json",
           "q_cmd_json",
           "arm_send_mono_ns",
           "rail_target_publish_mono_ns",
           "rail_fa24_write_mono_ns",
           "rail_encoder_sample_mono_ns",
           "arm_qdot_target_wall_json",
           "rail_sat",
           "rail_exec_velocity_m_s", "rail_measured_velocity_m_s",
           "rail_commanded_velocity_m_s", "rail_commanded_acceleration_m_s2",
           "rail_feedback_age_s", "a_mirror_frac", "j_mirror_frac",
           "last_limit_saturated", "keep_task_weight",
           "pref_slack_scale", "rail_task_vel",
           "v_escape", "v_reach", "v_ff_rail",
           "u_alloc", "u_posture", "u_mid", "v_r_ref",
           "comp_projected_frac",
           "rail_coast_active", "rail_feedback_reject_streak_s",
           "wall_override", "slack_zero_feasible",
           "sigma_arm", "sns_scale",
           "qpik_nullspace_norm",
           "qpik_nullspace_centering_norm",
           "qpik_nullspace_manip_norm",
           "qpik_nullspace_arm_angle_norm",
           "qpik_nullspace_damping_norm",
           "qpik_nullspace_rail_lock_norm",
           "cbf_min_dist", "cbf_pair"]
        + [f"qdot_meas_{i}" for i in range(8)]
        + [f"v_cmd_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + [f"path_twist_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + [f"feedback_twist_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + [f"comfort_slack_j{i}" for i in range(1, 8)]
        + [
            "pad_connected",
            "pad_lx", "pad_ly", "pad_lt", "pad_rx", "pad_ry", "pad_rt",
            "pad_lb", "pad_rb",
            "pad_vx", "pad_vy", "pad_vz",
            "pad_wx", "pad_wy", "pad_wz",
        ]
        + [f"pad_vcmd_base_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
    )

    def __init__(self, path: str, *, verbose_json: bool = False) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._verbose_json = bool(verbose_json)
        self._prev_arm_send_ns = 0
        self._prev_q_send_arm: np.ndarray | None = None
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
                if callable(row):
                    row = row()
                w.writerow(row)
                n += 1
                if n % 200 == 0:
                    f.flush()

    @staticmethod
    def _fmt_pad_fields(outer) -> list[str]:
        """Stick + mapped v_cmd extras; empty when the outer loop is not a pad."""

        def _fmt_n(arr, n: int, prec: int = 6) -> list[str]:
            if arr is None:
                return [""] * n
            vals = np.asarray(arr, dtype=float).reshape(-1)
            out = []
            for i in range(n):
                if i >= vals.size or not np.isfinite(vals[i]):
                    out.append("")
                else:
                    out.append(f"{float(vals[i]):.{prec}f}")
            return out

        axes = getattr(outer, "last_pad_axes", None)
        buttons = getattr(outer, "last_pad_buttons", None)
        if axes is None and buttons is None:
            return [""] * 21
        connected = getattr(outer, "last_pad_connected", False)
        btn = (
            np.asarray(buttons, dtype=float).reshape(-1)
            if buttons is not None
            else np.zeros(8)
        )
        lb = 1 if (btn.size > 4 and float(btn[4]) > 0.5) else 0
        rb = 1 if (btn.size > 5 and float(btn[5]) > 0.5) else 0
        return (
            [str(int(bool(connected)))]
            + _fmt_n(axes, 6, 4)
            + [str(lb), str(rb)]
            + _fmt_n(getattr(outer, "last_v_world", None), 3, 6)
            + _fmt_n(getattr(outer, "last_w_tool", None), 3, 6)
            + _fmt_n(getattr(outer, "last_twist_base", None), 6, 6)
        )

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
        feedback_age_s: float = float("nan"),
        feedback_fresh_tick: bool = False,
        f_ext_raw: np.ndarray | None = None,
        twist_achieved_base: np.ndarray | None = None,
        v_tcp_z_actual: float = float("nan"),
        qdot_meas: np.ndarray | None = None,
        rail_target_sent_m: float | None = None,
        deadline_slack_s: float = float("nan"),
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
        force_barrier_contact_active = getattr(
            ctrl, "force_barrier_contact_active", False
        )
        contact_phase = getattr(ctrl, "contact_phase", "")
        v_air_cmd = getattr(ctrl, "v_air_cmd", float("nan"))
        ke_hat = getattr(ctrl, "ke_hat", getattr(ctrl, "ke_est", float("nan")))
        dob_v = getattr(ctrl, "dob_v", float("nan"))
        barrier_cap_floor = getattr(ctrl, "barrier_cap_floor", float("nan"))
        ke_tracker = getattr(ctrl, "_ke_estimator", None)
        ke_update_gated = getattr(ke_tracker, "update_gated", False)
        ke_dx_m = getattr(ke_tracker, "last_dx_m", float("nan"))
        ke_df_n = getattr(ke_tracker, "last_df_n", float("nan"))
        ke_update_count = getattr(ke_tracker, "update_count", 0)
        flow = getattr(ctrl, "bidirectional_flow", None)
        flow_xp = getattr(flow, "xp", float("nan"))
        flow_vp = getattr(flow, "vp", float("nan"))
        flow_v_aux = getattr(flow, "v_aux", float("nan"))
        flow_xa = getattr(flow, "xa", float("nan"))
        flow_va = getattr(flow, "va", float("nan"))
        flow_e = getattr(flow, "e", float("nan"))
        flow_edot = getattr(flow, "edot", float("nan"))
        flow_fc = getattr(flow, "fc", float("nan"))
        flow_v_track = getattr(flow, "v_track", float("nan"))
        flow_pe = getattr(flow, "Pe", float("nan"))
        flow_pc = getattr(flow, "Pc", float("nan"))
        flow_alpha_target = getattr(flow, "alpha_raw", float("nan"))
        flow_alpha = getattr(flow, "alpha", float("nan"))
        flow_alpha_case = getattr(flow, "alpha_case", "")
        flow_would_gate = getattr(flow, "alpha_would_gate_m_s", float("nan"))
        flow_edot_aligned = getattr(
            flow, "mismatch_velocity_aligned", float("nan")
        )
        flow_tank = getattr(flow, "tank_energy", float("nan"))
        flow_psi = getattr(flow, "psi", float("nan"))
        flow_sn = getattr(flow, "Sn", float("nan"))
        flow_sr = getattr(flow, "Sr_hat", float("nan"))
        flow_p_phys = getattr(flow, "P_phys", float("nan"))
        flow_p_mismatch = getattr(flow, "P_mismatch", float("nan"))
        flow_e_phys = getattr(flow, "energy_phys_j", float("nan"))
        flow_e_mismatch = getattr(flow, "energy_mismatch_j", float("nan"))
        flow_gamma = getattr(flow, "gamma_effective", float("nan"))
        flow_sign_fault = getattr(flow, "sign_fault", True)
        flow_stale = getattr(flow, "feedback_stale", True)
        flow_blocked = getattr(flow, "blocked_reason", "")
        episode_rearm = getattr(ctrl, "contact_episode_rearm_event", False)
        episode_release_s = getattr(
            ctrl, "contact_episode_release_s", float("nan")
        )
        surface_force_scale = getattr(ctrl, "surface_force_scale", float("nan"))
        surface_force_alpha = getattr(ctrl, "surface_force_alpha", float("nan"))
        surface_xy_error_m = getattr(ctrl, "surface_xy_error_m", float("nan"))
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
        if rail_target_sent_m is not None and np.isfinite(float(rail_target_sent_m)):
            rail_sent = float(rail_target_sent_m)
        else:
            rail_sent = float(step.q_send[0]) if step.q_send is not None else float("nan")
        try:
            q_cmd_q_meas_norm = float(
                np.linalg.norm(np.asarray(step.q_send, dtype=float) - qm)
            )
        except (TypeError, ValueError):
            q_cmd_q_meas_norm = float("nan")

        # Desired pose + motion-subspace errors (force axes zeroed / ignored).
        pose_d = getattr(outer, "last_pose_d", None)
        pose_d_arr = (
            np.asarray(pose_d, dtype=float).reshape(-1)
            if pose_d is not None
            else np.full(6, np.nan)
        )
        if pose_d_arr.size < 6:
            pose_d_arr = np.full(6, np.nan)
        pose_meas_arr = np.asarray(pose, dtype=float).reshape(-1)
        if pose_meas_arr.size < 6:
            pose_meas_arr = np.full(6, np.nan)
        vel_ff = getattr(outer, "last_vel_ff", None)
        vel_ff_arr = (
            np.asarray(vel_ff, dtype=float).reshape(-1)
            if vel_ff is not None
            else np.full(6, np.nan)
        )
        if vel_ff_arr.size < 6:
            vel_ff_arr = np.full(6, np.nan)
        motion_err_lin_mm = np.full(3, np.nan)
        motion_err_rot_deg = np.full(3, np.nan)
        motion_err_rms_mm = float("nan")
        motion_axis_peak_mm = float("nan")
        tool_y_des_m = float("nan")
        tool_y_err_mm = float("nan")
        euler_order = "xyz"
        track_axes = np.ones(6)
        ctrl_cfg = getattr(ctrl, "cfg", None) if ctrl is not None else None
        if ctrl_cfg is not None:
            euler_order = str(getattr(ctrl_cfg, "euler_order", "xyz"))
            ta = getattr(ctrl_cfg, "track_axes", None)
            if ta is not None:
                track_axes = np.asarray(ta, dtype=float).reshape(-1)
                if track_axes.size < 6:
                    track_axes = np.ones(6)
        if np.all(np.isfinite(pose_d_arr)) and np.all(np.isfinite(pose_meas_arr)):
            err_base = pose_error(pose_d_arr, pose_meas_arr, euler_order)
            r_cur = Rsc.from_euler(
                euler_order, pose_meas_arr[3:6], degrees=False
            ).as_matrix()
            err_tool = np.zeros(6, dtype=float)
            err_tool[:3] = r_cur.T @ err_base[:3]
            err_tool[3:6] = r_cur.T @ err_base[3:6]
            ta6 = np.asarray(track_axes, dtype=float)[:6]
            # Force-axis components excluded from accuracy (NaN in per-axis cols).
            for i in range(3):
                if ta6[i] > 0.5:
                    motion_err_lin_mm[i] = float(err_tool[i] * 1000.0)
                else:
                    motion_err_lin_mm[i] = float("nan")
            for i in range(3):
                if ta6[3 + i] > 0.5:
                    motion_err_rot_deg[i] = float(np.degrees(err_tool[3 + i]))
                else:
                    motion_err_rot_deg[i] = float("nan")
            lin_tracked = motion_err_lin_mm[np.isfinite(motion_err_lin_mm)]
            if lin_tracked.size:
                motion_err_rms_mm = float(np.sqrt(np.mean(np.square(lin_tracked))))
                motion_axis_peak_mm = float(np.max(np.abs(lin_tracked)))
            # SIN fixture alias: tool-Y from general des/meas (control frame).
            tool_y_des_m = float((r_cur.T @ pose_d_arr[:3])[1])
            if np.isfinite(motion_err_lin_mm[1]):
                tool_y_err_mm = float(motion_err_lin_mm[1])
            elif ta6[1] > 0.5:
                tool_y_meas = float((r_cur.T @ pose_meas_arr[:3])[1])
                tool_y_err_mm = float((tool_y_des_m - tool_y_meas) * 1000.0)

        def _fmt6(arr: np.ndarray | None) -> list[str]:
            if arr is None:
                return [""] * 6
            vals = np.asarray(arr, dtype=float).reshape(-1)
            if vals.size < 6:
                vals = np.pad(vals, (0, 6 - int(vals.size)), constant_values=np.nan)
            return [
                f"{float(v):.6f}" if np.isfinite(v) else ""
                for v in vals[:6]
            ]

        def _fmt8(arr: np.ndarray | None) -> list[str]:
            if arr is None:
                return [""] * 8
            vals = np.asarray(arr, dtype=float).reshape(-1)
            if vals.size < 8:
                vals = np.pad(vals, (0, 8 - int(vals.size)), constant_values=np.nan)
            return [
                f"{float(v):.6f}" if np.isfinite(v) else ""
                for v in vals[:8]
            ]

        def _fmt3(arr: np.ndarray, prec: int = 4) -> list[str]:
            return [
                f"{float(v):.{prec}f}" if np.isfinite(v) else ""
                for v in np.asarray(arr, dtype=float).reshape(-1)[:3]
            ]

        comfort = np.asarray(
            getattr(step, "comfort_slack", np.zeros(7)), dtype=float
        ).reshape(-1)
        if comfort.size < 7:
            comfort = np.pad(comfort, (0, 7 - int(comfort.size)))
        comfort = comfort[:7]
        pad_fields = self._fmt_pad_fields(outer)
        controller_mode = str(getattr(step, "controller_mode", "") or "none")
        qm = np.asarray(qm, dtype=float).copy()
        pose = np.asarray(pose, dtype=float).copy()
        f_ext = np.asarray(f_ext, dtype=float).copy()
        raw_comp = np.asarray(raw_comp, dtype=float).copy()
        twist_achieved = np.asarray(twist_achieved, dtype=float).copy()
        if qdot_meas is not None:
            qdot_meas = np.asarray(qdot_meas, dtype=float).copy()

        # Snapshot step so the writer thread cannot see the next tick mutate it.
        step = copy.copy(step)
        for _name, _val in vars(step).items():
            if isinstance(_val, np.ndarray):
                setattr(step, _name, np.array(_val, copy=True))
        arm_ns = int(getattr(step, "arm_send_mono_ns", 0) or 0)
        q_send_arr = np.asarray(step.q_send, dtype=float).reshape(-1)
        arm_qdot_wall = None
        if (
            self._prev_arm_send_ns > 0
            and arm_ns > self._prev_arm_send_ns
            and q_send_arr.size >= 8
            and self._prev_q_send_arm is not None
        ):
            dt_send = (arm_ns - self._prev_arm_send_ns) * 1.0e-9
            if dt_send > 0.0:
                arm_qdot_wall = (
                    q_send_arr[1:8] - self._prev_q_send_arm
                ) / dt_send
        if arm_ns > 0 and q_send_arr.size >= 8:
            self._prev_arm_send_ns = arm_ns
            self._prev_q_send_arm = q_send_arr[1:8].copy()
        # Format on the writer thread: f-strings of ~300 columns were ~0.4 ms
        # on the control thread even after the disk write was already queued.
        self._q.put(lambda: (
            [
                f"{t_wall:.4f}",
                label,
                controller_mode,
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
               f"{dt_actual_s:.6f}",
               (
                   f"{float(deadline_slack_s):.6f}"
                   if np.isfinite(deadline_slack_s)
                   else ""
               ),
               f"{sensor_age_s:.6f}",
               f"{feedback_age_s:.6f}", int(bool(feedback_fresh_tick)),
               f"{raw_comp[0]:.3f}", f"{raw_comp[1]:.3f}", f"{raw_comp[2]:.3f}",
               f"{v_tcp_z_actual:.6f}", int(bool(contact_present)),
               f"{force_pred_z:.4f}", f"{force_dot_z:.4f}",
               f"{cap_press_z:.6f}", f"{cap_retract_z:.6f}",
               int(bool(ke_update_gated)), f"{ke_dx_m:.8f}", f"{ke_df_n:.5f}",
               int(ke_update_count),
               f"{governor_scale:.4f}", f"{governor_scale_raw:.4f}",
               f"{step.sigma_min:.5f}",
               f"{qdot_norm:.5f}", f"{qdot_max_frac:.4f}",
               f"{step.qdot_ff_norm:.5f}", f"{step.tcp_jump_mm:.3f}",
               f"{rail_sent:.6f}",
               f"{rail_meas_m:.6f}" if np.isfinite(rail_meas_m) else "",
               (
                   f"{float(step.q_send[0]) - float(qm[0]):.6f}"
                   if (
                       step.q_send is not None
                       and np.isfinite(float(step.q_send[0]))
                       and np.isfinite(float(qm[0]))
                   )
                   else ""
               ),
               f"{step.rail_vel_pin:.6f}" if np.isfinite(step.rail_vel_pin) else "",
               int(bool(step.plan_drives_rail)),
               f"{step.rail_qdot_ff:.6f}" if np.isfinite(step.rail_qdot_ff) else "",
               *_fmt6(pose_d_arr),
               *_fmt6(pose_meas_arr),
               *_fmt3(motion_err_lin_mm, prec=3),
               *_fmt3(motion_err_rot_deg, prec=4),
               f"{motion_err_rms_mm:.3f}" if np.isfinite(motion_err_rms_mm) else "",
               (
                   f"{motion_axis_peak_mm:.3f}"
                   if np.isfinite(motion_axis_peak_mm)
                   else ""
               ),
               *_fmt6(vel_ff_arr),
               (
                   f"{step.rail_contrib_m_s:.6f}"
                   if np.isfinite(step.rail_contrib_m_s)
                   else ""
               ),
               (
                   f"{step.arm_contrib_m_s:.6f}"
                   if np.isfinite(step.arm_contrib_m_s)
                   else ""
               ),
               (
                   f"{step.arm_contrib_m_s:.6f}"
                   if np.isfinite(step.arm_contrib_m_s)
                   else ""
               ),
               (
                   f"{step.rail_motion_share:.4f}"
                   if np.isfinite(step.rail_motion_share)
                   else ""
               ),
               (
                   f"{step.rail_exec_for_qp_m_s:.6f}"
                   if np.isfinite(
                       getattr(step, "rail_exec_for_qp_m_s", float("nan"))
                   )
                   else ""
               ),
               (
                   f"{step.wln_scale_rail:.4f}"
                   if np.isfinite(getattr(step, "wln_scale_rail", float("nan")))
                   else ""
               ),
               (
                   f"{step.wln_scale_arm_max:.4f}"
                   if np.isfinite(getattr(step, "wln_scale_arm_max", float("nan")))
                   else ""
               ),
               (
                   f"{step.waste_ratio:.4f}"
                   if np.isfinite(getattr(step, "waste_ratio", float("nan")))
                   else ""
               ),
               (
                   f"{step.rail_ff_m:.6f}"
                   if np.isfinite(getattr(step, "rail_ff_m", float("nan")))
                   else ""
               ),
               (
                   f"{step.rail_posture_err_m:.6f}"
                   if np.isfinite(getattr(step, "rail_posture_err_m", float("nan")))
                   else ""
               ),
               int(bool(step.rail_escape_active)),
               f"{step.psi_deg:.4f}" if np.isfinite(step.psi_deg) else "",
               f"{step.psi_ref_deg:.4f}" if np.isfinite(step.psi_ref_deg) else "",
               (
                   f"{step.psi_retarget_score:.6f}"
                   if np.isfinite(step.psi_retarget_score)
                   else ""
               ),
               f"{step.d_pref_m:.6f}" if np.isfinite(step.d_pref_m) else "",
               (
                   f"{step.d_star_m:.6f}"
                   if np.isfinite(getattr(step, "d_star_m", float("nan")))
                   else ""
               ),
               (
                   f"{step.psi_star_deg:.4f}"
                   if np.isfinite(getattr(step, "psi_star_deg", float("nan")))
                   else ""
               ),
               (
                   f"{step.minmax_margin:.6f}"
                   if np.isfinite(getattr(step, "minmax_margin", float("nan")))
                   else ""
               ),
               (
                   f"{step.elbow_margin_rad:.6f}"
                   if np.isfinite(step.elbow_margin_rad)
                   else ""
               ),
               (
                   f"{step.wrist_open_rad:.6f}"
                   if np.isfinite(step.wrist_open_rad)
                   else ""
               ),
               "1" if bool(getattr(step, "family_ok", True)) else "0",
               f"{tool_y_des_m:.6f}" if np.isfinite(tool_y_des_m) else "",
               f"{tool_y_err_mm:.3f}" if np.isfinite(tool_y_err_mm) else "",
               str(contact_phase),
               f"{float(v_air_cmd):.6f}" if np.isfinite(v_air_cmd) else "",
               f"{float(ke_hat):.4f}" if np.isfinite(ke_hat) else "",
               f"{float(dob_v):.6f}" if np.isfinite(dob_v) else "",
               (
                   f"{float(barrier_cap_floor):.6f}"
                   if np.isfinite(barrier_cap_floor)
                   else ""
               ),
               f"{flow_xp:.8f}", f"{flow_vp:.8f}", f"{flow_v_aux:.8f}",
               f"{flow_xa:.8f}", f"{flow_va:.8f}", f"{flow_e:.8f}",
               f"{flow_edot:.8f}", f"{flow_fc:.8f}", f"{flow_v_track:.8f}",
               f"{flow_pe:.8f}", f"{flow_pc:.8f}",
               f"{flow_alpha_target:.8f}", f"{flow_alpha:.8f}",
               str(flow_alpha_case), f"{flow_tank:.9f}", f"{flow_psi:.8f}",
               f"{flow_sn:.9f}", f"{flow_sr:.9f}", f"{flow_p_phys:.8f}",
               f"{flow_p_mismatch:.8f}", f"{flow_e_phys:.9f}",
               f"{flow_e_mismatch:.9f}", f"{flow_gamma:.8f}",
               f"{float(flow_would_gate):.8f}",
               f"{float(flow_edot_aligned):.8f}",
               int(bool(flow_sign_fault)), int(bool(flow_stale)),
               str(flow_blocked), int(bool(episode_rearm)),
               f"{episode_release_s:.6f}", f"{surface_force_scale:.6f}",
               f"{surface_force_alpha:.6f}", f"{surface_xy_error_m:.8f}",
               int(bool(force_barrier_contact_active)),
               str(step.qp_backend), str(step.qp_solver_status),
               int(step.qp_solver_iterations),
               f"{step.qp_solver_solve_ms:.6f}",
               int(step.qp_solver_call_count),
               int(bool(step.qp_solver_overrun)),
               str(getattr(step, "qp1_status", "not_run")),
               str(getattr(step, "qp2_status", "not_run")),
               f"{float(getattr(step, 'qp1_solve_ms', 0.0)):.6f}",
               f"{float(getattr(step, 'qp2_solve_ms', 0.0)):.6f}",
               f"{float(getattr(step, 'qp_assembly_ms', 0.0)):.6f}",
               f"{float(getattr(step, 'qp_fallback_ms', 0.0)):.6f}",
               f"{float(getattr(step, 'qpik_total_ms', 0.0)):.6f}",
               int(bool(getattr(step, "qp2_fallback", False))),
               *(
                   f"{v:.4f}" if np.isfinite(v) else ""
                   for v in (
                       getattr(step, "tick_inner_ms", float("nan")),
                       getattr(step, "tick_send_ms", float("nan")),
                       getattr(step, "tick_log_ms", float("nan")),
                   )
               ),
               f"{step.qpik_alpha:.8f}", f"{step.qpik_beta:.8f}",
               f"{step.qpik_authority:.8f}",
               f"{step.qpik_equality_residual_max:.9e}",
               f"{step.qpik_hard_residual_max:.9e}",
               int(bool(step.qpik_anchor_valid)),
               int(bool(step.qpik_recovery_overflow)),
               self._json_field(step.qpik_protected_nominal_overflow),
               self._json_field(step.qpik_recovery_caps),
               self._json_field(step.qpik_recovery_overflow_indices),
               self._json_field(step.hard_active_constraint_ids),
               self._json_field(step.protected_target),
               self._json_field(step.protected_achieved),
               self._json_field(step.protected_residual),
               self._json_field(step.scan_target),
               self._json_field(step.scan_achieved),
               self._json_field(step.scan_residual),
               self._json_field(step.qpik_working_slack),
               self._json_field(step.qpik_collision_slack),
               f"{step.qpik_dexterity_slack:.9e}",
               f"{step.qpik_branch_slack:.9e}",
               f"{step.rail_macro_pref_v:.8f}",
               f"{step.rail_center_pref_v:.8f}", f"{step.qdot[0]:.8f}",
               f"{step.arm_risk_pref_norm:.8f}",
               self._json_field(step.arm_risk_pref),
               f"{step.risk_direction_cosine:.8f}",
               self._json_field(step.path_velocity_xy),
               self._json_field(step.feedback_xy_raw),
               self._json_field(step.feedback_xy_filtered),
               self._json_field(step.rail_xy_contribution),
               self._json_field(step.arm_xy_contribution),
               f"{step.rail_task_projection:.8f}",
               f"{step.rail_arm_cancel:.8f}",
               f"{step.rail_decomposition_error:.9e}",
               f"{step.arm_health:.8f}",
               f"{step.joint_margin_rad:.8f}", f"{step.wrist_margin_rad:.8f}",
               f"{step.wrist_singularity:.8f}",
               f"{step.accepted_reference_lag_s:.6f}",
               f"{step.pre_solve_feedback_age_s:.6f}",
               f"{step.post_solve_feedback_age_s:.6f}",
               f"{q_cmd_q_meas_norm:.8f}",
               str(step.fallback_level), str(step.fallback_reason),
               int(bool(step.solver_fault_latched)),
               self._json_compact(step.qdot),
               int(bool(getattr(step, "post_qp_step_clamp_enabled", True))),
               int(bool(getattr(step, "post_step_would_clamp", False))),
               int(bool(getattr(step, "post_step_clamp_applied", False))),
               (
                   f"{float(step.dt_nom_s):.9e}"
                   if np.isfinite(getattr(step, "dt_nom_s", float("nan")))
                   else ""
               ),
               (
                   f"{float(step.dt_int_s):.9e}"
                   if np.isfinite(getattr(step, "dt_int_s", float("nan")))
                   else ""
               ),
               (
                   f"{float(step.box_h1_s):.9e}"
                   if np.isfinite(getattr(step, "box_h1_s", float("nan")))
                   else ""
               ),
               (
                   f"{float(step.box_h2_s):.9e}"
                   if np.isfinite(getattr(step, "box_h2_s", float("nan")))
                   else ""
               ),
               self._json_field(getattr(step, "qdot_raw", None)),
               self._json_field(getattr(step, "qdot_pre_commit", None)),
               self._json_field(getattr(step, "qdot_committed", None)),
               self._json_field(getattr(step, "qdot_prev_used", None)),
               self._json_field(getattr(step, "qdot_prev2_used", None)),
               self._json_field(getattr(step, "box_lo", None)),
               self._json_field(getattr(step, "box_hi", None)),
               self._json_field(getattr(step, "post_step_shadow_q", None)),
               self._json_compact(step.q_send),
               (
                   str(int(arm_ns))
                   if arm_ns > 0
                   else ""
               ),
               (
                   str(int(step.rail_target_publish_mono_ns))
                   if int(getattr(step, "rail_target_publish_mono_ns", 0) or 0) > 0
                   else ""
               ),
               (
                   str(int(step.rail_fa24_write_mono_ns))
                   if int(getattr(step, "rail_fa24_write_mono_ns", 0) or 0) > 0
                   else ""
               ),
               (
                   str(int(step.rail_encoder_sample_mono_ns))
                   if int(getattr(step, "rail_encoder_sample_mono_ns", 0) or 0) > 0
                   else ""
               ),
               self._json_field(arm_qdot_wall),
               int(bool(step.rail_sat)),
               (
                   f"{float(step.rail_exec_velocity_m_s):.8f}"
                   if np.isfinite(step.rail_exec_velocity_m_s) else ""
               ),
               (
                   f"{float(step.rail_measured_velocity_m_s):.8f}"
                   if np.isfinite(step.rail_measured_velocity_m_s) else ""
               ),
               (
                   f"{float(step.rail_commanded_velocity_m_s):.8f}"
                   if np.isfinite(step.rail_commanded_velocity_m_s) else ""
               ),
               (
                   f"{float(step.rail_commanded_acceleration_m_s2):.8f}"
                   if np.isfinite(step.rail_commanded_acceleration_m_s2) else ""
               ),
               (
                   f"{float(step.rail_feedback_age_s):.6f}"
                   if np.isfinite(step.rail_feedback_age_s) else ""
               ),
               (
                   f"{float(step.a_mirror_frac):.6f}"
                   if np.isfinite(step.a_mirror_frac) else ""
               ),
               (
                   f"{float(step.j_mirror_frac):.6f}"
                   if np.isfinite(step.j_mirror_frac) else ""
               ),
               int(bool(step.last_limit_saturated)),
               int(bool(step.keep_task_weight)),
               f"{float(step.pref_slack_scale):.4f}",
               (
                   f"{float(step.rail_task_vel):.6f}"
                   if np.isfinite(step.rail_task_vel)
                   else ""
               ),
               f"{float(step.v_escape):.6f}" if np.isfinite(step.v_escape) else "",
               f"{float(step.v_reach):.6f}" if np.isfinite(step.v_reach) else "",
               f"{float(step.v_ff_rail):.6f}" if np.isfinite(step.v_ff_rail) else "",
               f"{float(step.u_alloc):.6f}" if np.isfinite(step.u_alloc) else "",
               f"{float(step.u_posture):.6f}" if np.isfinite(step.u_posture) else "",
               f"{float(getattr(step, 'u_mid', float('nan'))):.6f}" if np.isfinite(getattr(step, "u_mid", float("nan"))) else "",
               f"{float(step.v_r_ref):.6f}" if np.isfinite(step.v_r_ref) else "",
               f"{float(getattr(step, 'comp_projected_frac', 0.0)):.6f}",
               int(bool(getattr(step, "rail_coast_active", False))),
               (
                   f"{float(step.rail_feedback_reject_streak_s):.6f}"
                   if np.isfinite(getattr(step, "rail_feedback_reject_streak_s", float("nan")))
                   else ""
               ),
               "1" if bool(step.wall_override) else "0",
               "1" if bool(step.slack_zero_feasible) else "0",
               f"{float(step.sigma_arm):.5f}" if np.isfinite(step.sigma_arm) else "",
               f"{float(step.sns_scale):.4f}",
               (
                   f"{float(step.nullspace_norm):.6f}"
                   if np.isfinite(step.nullspace_norm)
                   else ""
               ),
               (
                   f"{float(getattr(step, 'nullspace_centering_norm', float('nan'))):.6f}"
                   if np.isfinite(getattr(step, "nullspace_centering_norm", float("nan")))
                   else ""
               ),
               (
                   f"{float(getattr(step, 'nullspace_manip_norm', float('nan'))):.6f}"
                   if np.isfinite(getattr(step, "nullspace_manip_norm", float("nan")))
                   else ""
               ),
               (
                   f"{float(getattr(step, 'nullspace_arm_angle_norm', float('nan'))):.6f}"
                   if np.isfinite(getattr(step, "nullspace_arm_angle_norm", float("nan")))
                   else ""
               ),
               (
                   f"{float(getattr(step, 'nullspace_damping_norm', float('nan'))):.6f}"
                   if np.isfinite(getattr(step, "nullspace_damping_norm", float("nan")))
                   else ""
               ),
               (
                   f"{float(getattr(step, 'nullspace_rail_lock_norm', float('nan'))):.6f}"
                   if np.isfinite(getattr(step, "nullspace_rail_lock_norm", float("nan")))
                   else ""
               ),
               (
                   f"{float(step.cbf_min_dist):.6f}"
                   if np.isfinite(step.cbf_min_dist)
                   else ""
               ),
               str(step.cbf_pair),
               *_fmt8(
                   qdot_meas
                   if qdot_meas is not None
                   else getattr(step, "qdot_meas", None)
               ),
               *_fmt6(getattr(step, "v_cmd", step.twist_base)),
               *_fmt6(getattr(step, "path_twist", None)),
               *_fmt6(getattr(step, "feedback_twist", None)),
               *(f"{float(v):.9e}" for v in comfort),
               *pad_fields]
        ))

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


def _rail_qdot_m_s(
    rail_bridge,
    q_new: np.ndarray,
    last_q: np.ndarray | None,
    dt_feedback: float | None,
) -> float | None:
    """Rail is not in the RealMan UDP frame; use FA24 speed, else Δq0/Δt."""
    meas = getattr(rail_bridge, "measured_speed_m_s", None)
    if meas is not None:
        try:
            v_rail = float(meas)
        except (TypeError, ValueError):
            v_rail = float("nan")
        if np.isfinite(v_rail):
            return v_rail
    if (
        last_q is not None
        and dt_feedback is not None
        and 0.001 <= float(dt_feedback) <= 0.050
    ):
        return float(q_new[0] - last_q[0]) / float(dt_feedback)
    return None


@dataclass
class RailExecutionEstimate:
    position_m: float
    velocity_m_s: float
    measured_velocity_m_s: float
    commanded_velocity_m_s: float
    commanded_acceleration_m_s2: float
    sample_mono_s: float
    age_s: float
    extrapolation_age_s: float
    command_mode: str
    rejected: bool = False
    reason: str = ""


def accumulate_feedback_coast(
    streak_s: float,
    dt_s: float,
    *,
    bad: bool,
    limit_s: float,
) -> tuple[float, bool, bool]:
    """Accumulate a rejected/stale streak.

    Returns ``(new_streak_s, coast_active, should_fault)``.  A single bad
    tick coasts; only a sustained streak past ``limit_s`` is a fault.
    """

    if not bad:
        return 0.0, False, False
    new = float(streak_s) + max(float(dt_s), 0.0)
    limit = max(float(limit_s), 0.0)
    return new, True, bool(new > limit)


def _rail_execution_velocity_estimate(
    rail_bridge,
    *,
    now_s: float | None = None,
    freshness_s: float,
    feedback=None,
    require_fresh: bool = True,
    observer: RailStateObserver | None = None,
    v_r_ref: float | None = None,
    dt_s: float | None = None,
) -> RailExecutionEstimate | None:
    """Bounded rail execution estimate for strict QPIK affine compensation.

    Between two FA24 samples the measured velocity is propagated with the
    worker's latest commanded acceleration over the true sample age, but
    never farther than two configured rail polls.  A sample older than
    ``freshness_s`` still coasts on that last reading (USB jitter must not
    kill the task).  Missing or non-finite feedback is a fault only when
    ``require_fresh`` is true.  Before the first sample the caller should
    pass ``require_fresh=False`` so a cold Modbus poll can finish; the QP
    then uses its existing command-velocity ZOH.
    """
    if rail_bridge is None or not getattr(rail_bridge, "enabled", False):
        return None
    now = time.monotonic() if now_s is None else float(now_s)
    try:
        if feedback is None:
            feedback = rail_bridge.execution_feedback
        position = float(feedback.position_m)
        sample_t = float(feedback.sample_mono_s)
        v_meas = float(feedback.v_meas_m_s)
        v_cmd = float(feedback.v_cmd_m_s)
        a_cmd = float(feedback.a_cmd_m_s2)
        feedback_valid = bool(getattr(feedback, "valid", True))
        mode_obj = feedback.command_mode
    except Exception as exc:
        if not require_fresh:
            return None
        return RailExecutionEstimate(
            position_m=float("nan"),
            velocity_m_s=0.0,
            measured_velocity_m_s=0.0,
            commanded_velocity_m_s=0.0,
            commanded_acceleration_m_s2=0.0,
            sample_mono_s=float("nan"),
            age_s=float("inf"),
            extrapolation_age_s=0.0,
            command_mode="",
            rejected=True,
            reason=f"unavailable:{exc}",
        )
    values = (now, position, sample_t, v_meas, v_cmd, a_cmd)
    if not all(np.isfinite(value) for value in values):
        if not require_fresh:
            return None
        return RailExecutionEstimate(
            position_m=position if np.isfinite(position) else float("nan"),
            velocity_m_s=0.0,
            measured_velocity_m_s=v_meas if np.isfinite(v_meas) else 0.0,
            commanded_velocity_m_s=v_cmd if np.isfinite(v_cmd) else 0.0,
            commanded_acceleration_m_s2=a_cmd if np.isfinite(a_cmd) else 0.0,
            sample_mono_s=sample_t if np.isfinite(sample_t) else float("nan"),
            age_s=float("inf"),
            extrapolation_age_s=0.0,
            command_mode=str(getattr(mode_obj, "value", mode_obj) or ""),
            rejected=True,
            reason="non_finite",
        )
    if not feedback_valid:
        if not require_fresh:
            return None
        return RailExecutionEstimate(
            position_m=position,
            velocity_m_s=0.0,
            measured_velocity_m_s=v_meas,
            commanded_velocity_m_s=0.0,
            commanded_acceleration_m_s2=0.0,
            sample_mono_s=sample_t,
            age_s=max(0.0, now - sample_t),
            extrapolation_age_s=0.0,
            command_mode=str(getattr(mode_obj, "value", mode_obj) or ""),
            rejected=True,
            reason="encoder_gate",
        )
    age = max(0.0, now - sample_t)
    # freshness_s is the call-site budget; stale age coasts instead of
    # raising.  One USB hiccup (age 83 ms vs 80 ms) must not stop QPIK.
    # Worker still hard-holds FA24 after 3 Modbus fails.
    _ = max(float(freshness_s), 0.0)
    cfg = getattr(rail_bridge, "config", None)
    poll_hz = max(float(getattr(cfg, "poll_hz", 50.0)), 1.0)
    extrap_age = min(age, 2.0 / poll_hz)
    if observer is not None:
        q_hat, v_est = observer.update(
            now_s=now,
            dt_s=float(dt_s) if dt_s is not None else max(age, 1.0e-3),
            v_r_ref=float(v_r_ref) if v_r_ref is not None else float(v_cmd),
            q_meas=position,
            sample_t=sample_t,
            v_meas=v_meas,
        )
        position = float(q_hat)
    else:
        v_est = v_meas + a_cmd * extrap_age
    v_cap = abs(float(getattr(cfg, "vel_max_m_s", float("inf"))))
    if np.isfinite(v_cap):
        v_est = float(np.clip(v_est, -v_cap, v_cap))
    mode = str(getattr(mode_obj, "value", mode_obj) or "")
    return RailExecutionEstimate(
        position_m=position,
        velocity_m_s=float(v_est),
        measured_velocity_m_s=v_meas,
        commanded_velocity_m_s=v_cmd,
        commanded_acceleration_m_s2=a_cmd,
        sample_mono_s=sample_t,
        age_s=age,
        extrapolation_age_s=extrap_age,
        command_mode=mode,
    )


def _qdot_meas_8dof(
    q_new: np.ndarray,
    last_q: np.ndarray | None,
    dt_feedback: float | None,
    snap,
    rail_bridge=None,
    rail_velocity_m_s: float | None = None,
) -> np.ndarray | None:
    """8-vector qdot: SDK arm speed + rail encoder/servo. Finite-diff only as fallback."""
    arm = arm_qdot_rad_s_from_snap(snap)
    if arm is not None:
        qdot = np.zeros(8, dtype=float)
        qdot[1:] = arm
        v_rail = (
            float(rail_velocity_m_s)
            if rail_velocity_m_s is not None
            and np.isfinite(float(rail_velocity_m_s))
            else _rail_qdot_m_s(rail_bridge, q_new, last_q, dt_feedback)
        )
        if v_rail is not None:
            qdot[0] = v_rail
        return qdot
    if last_q is None or dt_feedback is None:
        return None
    if not (0.001 <= float(dt_feedback) <= 0.050):
        return None
    qdot = wrap_joint_delta(last_q, q_new) / float(dt_feedback)
    if rail_velocity_m_s is not None and np.isfinite(float(rail_velocity_m_s)):
        qdot[0] = float(rail_velocity_m_s)
    return qdot


def _rail_m_for_init(rail_bridge, inner: JointIkController) -> float:
    """Seed WBC ``q_cmd[0]`` from encoder so the first set_target is near reality."""
    if rail_bridge is not None and rail_bridge.enabled:
        return float(rail_bridge.measured_m)
    return float(inner.q_cmd[0])


def _rail_m_for_feedback(rail_bridge, inner: JointIkController) -> float:
    """Return measured rail position; enabled-rail faults must stop 8D QPIK."""
    if rail_bridge is None or not getattr(rail_bridge, "enabled", False):
        return float(inner.q_cmd[0])
    try:
        meas = float(rail_bridge.measured_m)
    except Exception as exc:
        raise RuntimeError(f"rail feedback unavailable: {exc}") from exc
    sane = getattr(rail_bridge, "_encoder_sane", None)
    if callable(sane):
        if not sane(meas):
            raise RuntimeError(f"rail encoder value is invalid: {meas!r}")
    elif not (np.isfinite(meas)):
        raise RuntimeError(f"rail encoder value is non-finite: {meas!r}")
    return meas


def _rail_settled_for_arrival(
    rail_bridge,
    *,
    speed_limit_m_s: float,
    now_s: float,
    freshness_s: float,
) -> bool | None:
    """Return worker-aligned rail standstill, or None when no rail is active."""

    if rail_bridge is None or not getattr(rail_bridge, "enabled", False):
        return None
    try:
        sample = rail_bridge.servo_sample
        sample_time = float(sample.sample_mono_s)
        v_cmd = float(sample.v_cmd_m_s)
        v_meas = float(sample.v_meas_m_s)
    except Exception:
        return False
    if not all(np.isfinite(value) for value in (sample_time, v_cmd, v_meas)):
        return False
    if max(0.0, float(now_s) - sample_time) > max(float(freshness_s), 0.0):
        return False
    limit = max(float(speed_limit_m_s), 0.0)
    return bool(abs(v_cmd) <= limit and abs(v_meas) <= limit)


def _qpik_rail_v_ff_m_s(qdot0: float) -> float:
    """Servo ``v_ff`` is the QPIK rail velocity, never a pad/path bypass."""
    v = float(qdot0)
    if not math.isfinite(v):
        return 0.0
    return v


def _wall_clock_rail_target(
    q_send0: float,
    qdot0: float,
    dt_wall: float,
    dt_nom: float,
    *,
    soft_lo: float,
    soft_hi: float,
    meas_m: float | None = None,
    lead_max_m: float = 0.0,
) -> float:
    """Publish QPIK ``q_send[0]``; the rail already integrated on wall time.

    Soft limits and the ±``lead_max_m`` command-lead clamp stay here.
    Do not add ``qdot * (dt_wall - dt_nom)`` — that would double-count.
    Clamp against measured position even when idle so a parked command
    cannot wander tens of centimetres off the encoder.
    """
    del dt_wall, dt_nom, qdot0
    x = float(q_send0)
    lo = float(soft_lo)
    hi = float(soft_hi)
    if hi < lo:
        lo, hi = hi, lo
    x = max(lo, min(hi, x))
    lead = max(float(lead_max_m), 0.0)
    if lead > 0.0 and meas_m is not None and math.isfinite(float(meas_m)):
        meas = float(meas_m)
        x = max(meas - lead, min(meas + lead, x))
    return x


def _publish_rail_target_before_arm(
    rail_bridge,
    target_m: float,
    fault_stop,
    v_ff_m_s: float | None = None,
) -> tuple[bool, str]:
    """Require the rail to accept this 8D tick before publishing the arm half."""

    if rail_bridge is None or not getattr(rail_bridge, "enabled", False):
        return True, ""
    if not bool(getattr(rail_bridge, "calibrated", False)):
        reason = "rail_target_rejected:not_calibrated"
    elif bool(getattr(rail_bridge, "panicked", False)):
        detail = str(getattr(rail_bridge, "panic_reason", "") or "panic")
        reason = (
            f"rail_target_rejected:{detail}; "
            "restart Window A to re-arm (panic latches)"
        )
    elif not bool(getattr(rail_bridge, "armed", False)):
        reason = "rail_target_rejected:not_armed; restart Window A to re-arm"
    else:
        try:
            accepted = rail_bridge.set_target_m(
                float(target_m), v_ff_m_s=v_ff_m_s
            )
        except Exception as exc:
            reason = f"rail_target_exception:{type(exc).__name__}:{exc}"
        else:
            if accepted is True:
                return True, ""
            reason = "rail_target_rejected:bridge_declined"
    fault_stop(reason)
    return False, reason


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
    physical_saturated: bool = False,
) -> float:
    """Raw governor scale in [0, 1] (min of active bands); filter in GovernorFilter.

    Cartesian error only slows the clock when a joint/rail is physically
    saturated; otherwise tracking lag from a bad IK posture must not crawl
    the reference to ~4% speed.  A floor (``governor_scale_min``) still
    applies whenever a band is active.
    """
    scales: list[float] = []
    floor = float(getattr(phase, "governor_scale_min", 0.0) or 0.0)

    if phase.governor_joint_err_max_deg > 0.0 and joint_err_deg is not None:
        e0, e1 = phase.governor_joint_err_ok_deg, phase.governor_joint_err_max_deg
        if e1 > e0:
            scales.append(float(np.clip((e1 - joint_err_deg) / (e1 - e0), 0.0, 1.0)))
        else:
            scales.append(1.0)

    if phase.governor_err_max_mm > 0.0 and outer_err_mm is not None:
        if physical_saturated:
            e0, e1 = phase.governor_err_ok_mm, phase.governor_err_max_mm
            if e1 > e0:
                scales.append(float(np.clip((e1 - outer_err_mm) / (e1 - e0), 0.0, 1.0)))
        else:
            scales.append(1.0)

    if not scales:
        return 1.0
    return max(float(min(scales)), floor)


class GovernorFilter:
    """First-order LPF + freeze hysteresis on the governor scale."""

    def __init__(
        self,
        tau_s: float = 0.2,
        freeze_below: float = 0.02,
        release_above: float = 0.10,
        scale_min: float = 0.0,
    ) -> None:
        self.tau_s = float(tau_s)
        self.freeze_below = float(freeze_below)
        self.release_above = float(release_above)
        self.scale_min = float(scale_min)
        self.scale = 1.0
        self.frozen = False

    def update(self, raw: float, dt: float) -> float:
        floor = float(getattr(self, "scale_min", 0.0) or 0.0)
        raw = float(np.clip(raw, floor, 1.0))
        alpha = 1.0 if self.tau_s <= 0.0 else min(1.0, dt / self.tau_s)
        self.scale += alpha * (raw - self.scale)
        freeze_below = float(self.freeze_below)
        if floor >= freeze_below:
            self.frozen = False
            return float(np.clip(self.scale, floor, 1.0))
        if self.frozen:
            if raw >= self.release_above and self.scale >= self.release_above:
                self.frozen = False
        elif self.scale <= freeze_below:
            self.frozen = True
        if self.frozen:
            return 0.0
        return float(np.clip(self.scale, floor, 1.0))


def _send_joint_canfd_cmd(robot, q_deg, follow: bool, canfd_proxy=None) -> None:
    from rm75_control.motion.canfd import send_joint_canfd

    q = np.asarray(q_deg, dtype=float).reshape(-1)[:7]
    if canfd_proxy is not None:
        canfd_proxy.write(q, follow=follow)
        return
    if robot is None:
        raise RuntimeError("no robot handle and no CANFD proxy configured")
    send_joint_canfd(robot, list(q), follow=follow)


def _guard_qpik_step_before_send(step: JointIkStep, fault_stop) -> tuple[bool, str]:
    """Gate rail/CANFD publication.  A failed QP1 has no certified command."""
    if bool(step.solver_fault_latched) or str(step.fallback_level) == "stop":
        reason = f"qpik_fault:{step.fallback_level}:{step.fallback_reason}"
        fault_stop(reason)
        return False, reason
    return True, ""


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
    logger = (
        _TickLogger(
            log_csv,
            verbose_json=bool(getattr(inner.cfg, "verbose_json", False)),
        )
        if log_csv
        else None
    )
    cstate = (
        _CStateGuard()
        if realtime and bool(getattr(inner.cfg, "disable_cstates", True))
        else None
    )
    if cstate is not None:
        cstate.__enter__()
        if verbose and not cstate.active:
            print("  (/dev/cpu_dma_latency unavailable — C-states not held)", flush=True)
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

        if realtime:
            if not _set_realtime_priority():
                if verbose:
                    print("  (SCHED_FIFO unavailable - running at normal priority)", flush=True)
            if _pin_control_cpu(getattr(inner.cfg, "control_cpu", None)):
                if verbose:
                    print(
                        f"  control thread pinned to CPU {inner.cfg.control_cpu}",
                        flush=True,
                    )
            elif verbose and getattr(inner.cfg, "control_cpu", None) is not None:
                print("  (CPU affinity unavailable)", flush=True)

        gc_frozen = False
        if realtime and bool(getattr(inner.cfg, "rt_disable_gc", True)):
            gc.collect()
            gc.freeze()
            gc.disable()
            gc_frozen = True

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

        def _fault_stop(reason: str) -> None:
            """Stop both axes without publishing another trajectory target."""

            if verbose:
                print(f"  QPIK SAFETY STOP: {reason}", flush=True)
            if rail_bridge is not None and getattr(rail_bridge, "enabled", False):
                try:
                    rail_bridge.hold_current()
                except Exception:
                    try:
                        rail_bridge.kill_motion()
                    except Exception:
                        pass
            if robot is not None:
                try:
                    robot.rm_set_arm_slow_stop()
                except Exception:
                    pass

        try:
            pose_rm = _pose0_rm
            q_meas = q0_rad
            pose_pin = pose0
            jump_warn_t = 0.0
            phase_stopped = False
            stop_reason = ""
            rail_feedback_ready = False
            try:
                for phase_idx, phase in enumerate(phases):
                    if stop_check is not None and stop_check():
                        phase_stopped = True
                        if verbose:
                            extra = (
                                " before first tick (stop during pad/program init)"
                                if ticks == 0
                                else ""
                            )
                            print(f"  stopped by external request{extra}", flush=True)
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
                    if hasattr(phase.outer, "begin_hybrid_episode"):
                        applied_qdot = inner.core.qdot_prev
                        applied_twist = inner.kin.jacobian(q_meas) @ applied_qdot
                        inner.begin_hybrid_episode(q_meas, applied_qdot)
                        phase.outer.begin_hybrid_episode(applied_twist, pose_pin)

                    obs = phase.force_observer if phase.force_observer is not None else force_observer
                    phase_t0 = time.perf_counter()
                    next_tick = phase_t0
                    last_tick_time = phase_t0
                    t_ref = 0.0
                    gov_filter = GovernorFilter(
                        tau_s=phase.governor_tau_s,
                        freeze_below=phase.governor_freeze_below,
                        release_above=phase.governor_release_above,
                        scale_min=float(getattr(phase, "governor_scale_min", 0.25)),
                    )
                    scale = 1.0
                    phase_arrived = False
                    arrival_gate = _ArrivalDwellGate(
                        plan_duration_s=phase.arrival_plan_duration_s,
                        dwell_required_s=phase.arrival_dwell_s,
                        arm_speed_rad_s=phase.arrival_arm_speed_rad_s,
                        rail_speed_m_s=phase.arrival_rail_speed_m_s,
                    )
                    prev_pose_cmd = inner.kin.fk_pose(inner.q_cmd)
                    # TCP velocity from SDK joint_speed (rail from servo / Δq0).
                    last_feedback_seq = int(getattr(snap, "seq", 0))
                    last_feedback_t = float(getattr(snap, "t_s", 0.0))
                    last_feedback_q = np.asarray(q_meas, dtype=float).copy()
                    # ``feedback_age_s`` tracks the last sample from which a
                    # finite-difference TCP velocity was actually computed;
                    # sensor transport age is a separate diagnostic.
                    last_feedback_velocity_t = last_feedback_t
                    twist_achieved_base = np.zeros(6, dtype=float)
                    qdot_meas = None
                    v_tcp_z_actual = 0.0
                    feedback_velocity_valid = False
                    feedback_fresh_tick = False
                    first_tick = True
                    last_log_ms = float("nan")
                    rail_reject_streak_s = 0.0
                    sensor_stale_streak_s = 0.0
                    rail_coast_active = False
                    wd.arm()
                    while True:
                        if stop_check is not None and stop_check():
                            phase_stopped = True
                            break
                        if not wd.fired:
                            wd.beat()
                        now = time.perf_counter()
                        dt_raw = now - last_tick_time
                        last_tick_time = now
                        # The first phase tick occurs immediately after setup;
                        # subsequent wall periods are only sanity-clamped so
                        # >15 ms stalls remain visible to the force/proxy
                        # dynamics.  ``update()`` integrates arm and rail on
                        # this wall period when ``dt_wall_s`` is passed.
                        if first_tick:
                            dt_wall_actual = float(dt)
                            first_tick = False
                        else:
                            dt_wall_actual = float(
                                np.clip(
                                    dt_raw if np.isfinite(dt_raw) else dt,
                                    1.0e-4,
                                    0.10,
                                )
                            )
                        next_tick, late_ms = _resync_late_tick(next_tick, now, dt)
                        if late_ms > dt * 1000.0:
                            stutter_count += 1
                        max_jitter_ms = max(max_jitter_ms, late_ms)
                        t_wall = now - phase_t0
                        if (
                            phase.duration_s is not None
                            and phase.wait_until is None
                            and t_ref >= phase.duration_s
                        ):
                            break
                        if phase.max_duration_s is not None and t_wall >= phase.max_duration_s:
                            break
    
                        feedback_fresh_tick = False
                        snap = async_obs.read()
                        coast_limit_s = float(
                            getattr(inner.cfg, "feedback_coast_s", 0.30)
                        )
                        try:
                            rail_exec_estimate = _rail_execution_velocity_estimate(
                                rail_bridge,
                                now_s=time.monotonic(),
                                freshness_s=float(inner.cfg.feedback_timeout_s),
                                require_fresh=rail_feedback_ready,
                                observer=inner.rail_observer,
                                v_r_ref=float(inner.last_v_r_ref),
                                dt_s=float(dt_wall_actual),
                            )
                        except RuntimeError as exc:
                            rail_exec_estimate = RailExecutionEstimate(
                                position_m=float("nan"),
                                velocity_m_s=0.0,
                                measured_velocity_m_s=0.0,
                                commanded_velocity_m_s=0.0,
                                commanded_acceleration_m_s2=0.0,
                                sample_mono_s=float("nan"),
                                age_s=float("inf"),
                                extrapolation_age_s=0.0,
                                command_mode="",
                                rejected=True,
                                reason=str(exc),
                            )
                        rail_rejected = bool(
                            rail_exec_estimate is not None
                            and getattr(rail_exec_estimate, "rejected", False)
                        )
                        rail_reject_reason = (
                            str(getattr(rail_exec_estimate, "reason", "") or "")
                            if rail_rejected
                            else ""
                        )
                        (
                            rail_reject_streak_s,
                            rail_coast_active,
                            rail_coast_fault,
                        ) = accumulate_feedback_coast(
                            rail_reject_streak_s,
                            dt_wall_actual,
                            bad=rail_rejected,
                            limit_s=coast_limit_s,
                        )
                        if rail_coast_fault:
                            phase_stopped = True
                            stop_reason = (
                                "rail_feedback_fault:"
                                f"{rail_reject_reason or 'rejected'}"
                                f":streak={rail_reject_streak_s:.3f}s"
                            )
                            _fault_stop(stop_reason)
                            break
                        if rail_exec_estimate is not None and not rail_rejected:
                            rail_feedback_ready = True
                        inner._midrange_freeze = bool(rail_coast_active)
                        if rail_coast_active:
                            inner.last_v_r_ref = 0.0
                            try:
                                inner.rail_ref_model.reset(0.0)
                            except Exception:
                                pass
                            if rail_bridge is not None and getattr(
                                rail_bridge, "enabled", False
                            ):
                                try:
                                    rail_bridge.hold_current()
                                except Exception:
                                    pass
                        if snap.pose is not None:
                            pose_rm = snap.pose
                        if snap.q_deg is not None:
                            try:
                                if (
                                    rail_exec_estimate is not None
                                    and np.isfinite(float(rail_exec_estimate.position_m))
                                ):
                                    rail_measured_m = float(
                                        rail_exec_estimate.position_m
                                    )
                                else:
                                    rail_measured_m = float(q_meas[0])
                            except Exception:
                                rail_measured_m = float(q_meas[0])
                            q_new = _expand_q_meas(
                                deg2rad(snap.q_deg), rail_measured_m
                            )
                            snap_seq = int(getattr(snap, "seq", 0))
                            snap_t = float(getattr(snap, "t_s", 0.0))
                            if (
                                snap_seq != last_feedback_seq
                                and snap_t > last_feedback_t
                            ):
                                dt_feedback = snap_t - last_feedback_t
                                qdot_meas = _qdot_meas_8dof(
                                    q_new,
                                    last_feedback_q,
                                    dt_feedback,
                                    snap,
                                    rail_bridge,
                                    rail_velocity_m_s=(
                                        rail_exec_estimate.velocity_m_s
                                        if rail_exec_estimate is not None
                                        else None
                                    ),
                                )
                                if qdot_meas is not None:
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
                                    feedback_fresh_tick = True
                                    feedback_velocity_valid = True
                                    last_feedback_velocity_t = snap_t
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
                        feedback_age_s = (
                            max(0.0, time.monotonic() - last_feedback_velocity_t)
                            if last_feedback_velocity_t > 0.0
                            else float("inf")
                        )

                        sensor_stale_now = (
                            not np.isfinite(sensor_age_s)
                            or sensor_age_s > float(inner.cfg.feedback_timeout_s)
                        )
                        (
                            sensor_stale_streak_s,
                            sensor_coast_active,
                            sensor_coast_fault,
                        ) = accumulate_feedback_coast(
                            sensor_stale_streak_s,
                            dt_wall_actual,
                            bad=sensor_stale_now,
                            limit_s=coast_limit_s,
                        )
                        if sensor_coast_fault:
                            phase_stopped = True
                            stop_reason = (
                                "feedback_stale: "
                                f"age={sensor_age_s:.6f}s > "
                                f"{inner.cfg.feedback_timeout_s:.6f}s"
                                f":streak={sensor_stale_streak_s:.3f}s"
                            )
                            _fault_stop(stop_reason)
                            break
                        if sensor_coast_active:
                            if rail_bridge is not None and getattr(
                                rail_bridge, "enabled", False
                            ):
                                try:
                                    rail_bridge.hold_current()
                                except Exception:
                                    pass
                            ticks += 1
                            next_tick += dt
                            _wait_until(next_tick)
                            continue

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
                        sample_params = inspect.signature(phase.outer.sample).parameters
                        sample_kwargs: dict = {}
                        if "q_meas" in sample_params:
                            sample_kwargs["q_meas"] = q_meas
                        if "f_ext_raw" in sample_params and f_ext_raw is not None:
                            # Unfiltered wrench for Dimeas (LPF hides the band).
                            sample_kwargs["f_ext_raw"] = f_ext_raw
                        if "dt_actual" in sample_params:
                            sample_kwargs["dt_actual"] = dt_wall_actual
                        if "v_tcp_z_actual" in sample_params:
                            sample_kwargs["v_tcp_z_actual"] = v_tcp_z_actual
                        if "sensor_age_s" in sample_params:
                            sample_kwargs["sensor_age_s"] = sensor_age_s
                        if "feedback_age_s" in sample_params:
                            sample_kwargs["feedback_age_s"] = feedback_age_s
                        if "feedback_fresh_tick" in sample_params:
                            sample_kwargs["feedback_fresh_tick"] = feedback_fresh_tick
                        if "feedback_velocity_valid" in sample_params:
                            sample_kwargs["feedback_velocity_valid"] = (
                                feedback_velocity_valid
                            )
                        twist = np.asarray(
                            phase.outer.sample(t_ref, pose_pin, f_ext, **sample_kwargs),
                            dtype=float,
                        )
                        qdot_ff = (
                            phase.qdot_ff_provider(t_ref)
                            if phase.qdot_ff_provider is not None
                            else None
                        )
                        qdot_command = getattr(
                            phase.outer, "last_qdot_command", None
                        )
                        if qdot_command is not None:
                            qdot_ff = np.asarray(qdot_command, dtype=float).copy()
                        if qdot_ff is not None:
                            qdot_ff = np.asarray(qdot_ff, dtype=float)
                            if phase.scale_qdot_ff_with_governor:
                                qdot_ff = qdot_ff * scale
                        # Additive joint fb (not governor-scaled) closes nullspace q_err.
                        qdot_fb = getattr(phase.outer, "last_qdot_fb", None)
                        if qdot_fb is not None and qdot_command is None:
                            qdot_fb = np.asarray(qdot_fb, dtype=float)
                            qdot_ff = qdot_fb if qdot_ff is None else (qdot_ff + qdot_fb)
                        vel_ff_ref = getattr(phase.outer, "last_vel_ff", None)
                        pose_d_ref = getattr(phase.outer, "last_pose_d", None)
                        path_twist = getattr(phase.outer, "last_path_twist", None)
                        feedback_twist = getattr(
                            phase.outer, "last_feedback_twist", None
                        )
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
                        _t_inner0 = time.perf_counter()
                        step = inner.update(
                            twist,
                            control_dt,
                            q_meas=q_meas,
                            qdot_ff=qdot_ff,
                            vel_ff=vel_ff_ref,
                            pose_d=pose_d_ref,
                            f_ext_z=f_ext_z if math.isfinite(f_ext_z) else None,
                            f_des_z=f_des_z if math.isfinite(f_des_z) else None,
                            contact_active=bool(
                                getattr(ctrl, "contact_present", False)
                                if ctrl is not None
                                else False
                            ),
                            path_twist=path_twist,
                            feedback_twist=feedback_twist,
                            v_force_z=(
                                float(getattr(ctrl, "v_force_z", float("nan")))
                                if ctrl is not None
                                else None
                            ),
                            rail_exec_vel_m_s=(
                                rail_exec_estimate.velocity_m_s
                                if rail_exec_estimate is not None
                                and not rail_coast_active
                                else None
                            ),
                            rail_exec_smooth_m_s=(
                                rail_exec_estimate.commanded_velocity_m_s
                                if rail_exec_estimate is not None
                                and not rail_coast_active
                                else None
                            ),
                            dt_wall_s=dt_wall_actual,
                        )
                        step.rail_coast_active = bool(rail_coast_active)
                        step.rail_feedback_reject_streak_s = float(
                            rail_reject_streak_s
                        )
                        if rail_exec_estimate is not None:
                            step.rail_exec_velocity_m_s = float(
                                rail_exec_estimate.velocity_m_s
                            )
                            step.rail_measured_velocity_m_s = float(
                                rail_exec_estimate.measured_velocity_m_s
                            )
                            step.rail_commanded_velocity_m_s = float(
                                rail_exec_estimate.commanded_velocity_m_s
                            )
                            step.rail_commanded_acceleration_m_s2 = float(
                                rail_exec_estimate.commanded_acceleration_m_s2
                            )
                            step.rail_feedback_age_s = float(rail_exec_estimate.age_s)
                        step.tick_inner_ms = (
                            time.perf_counter() - _t_inner0
                        ) * 1000.0
                        step.pre_solve_feedback_age_s = sensor_age_s
                        # A hard-construction/final-validation fault is acted on before the
                        # rail target or CANFD joint command can be published.
                        sendable, qpik_stop_reason = _guard_qpik_step_before_send(
                            step, _fault_stop
                        )
                        if not sendable:
                            phase_stopped = True
                            stop_reason = qpik_stop_reason
                            if logger is not None:
                                rail_meas = float("nan")
                                if (
                                    rail_bridge is not None
                                    and rail_bridge.enabled
                                ):
                                    try:
                                        rail_meas = float(rail_bridge.measured_m)
                                    except Exception:
                                        rail_meas = float("nan")
                                step.tick_log_ms = last_log_ms
                                logger.write(
                                    now - total_t0,
                                    phase.label,
                                    t_ref,
                                    step,
                                    q_meas,
                                    pose_pin,
                                    f_ext,
                                    outer=phase.outer,
                                    rail_meas_m=rail_meas,
                                    dt_actual_s=dt_wall_actual,
                                    sensor_age_s=sensor_age_s,
                                    feedback_age_s=feedback_age_s,
                                    feedback_fresh_tick=feedback_fresh_tick,
                                    f_ext_raw=f_ext_raw,
                                    twist_achieved_base=twist_achieved_base,
                                    v_tcp_z_actual=v_tcp_z_actual,
                                    qdot_meas=qdot_meas,
                                )
                            break
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
                        publication_reason = ""
                        if stop_check is not None and stop_check():
                            publication_reason = "external_stop_before_send"
                        elif wd.fired:
                            publication_reason = "watchdog_fired_before_send"
                        else:
                            publish_snap = async_obs.read()
                            snap_time = float(getattr(publish_snap, "t_s", 0.0))
                            post_solve_sensor_age_s = (
                                max(0.0, time.monotonic() - snap_time)
                                if snap_time > 0.0
                                else float("inf")
                            )
                            step.post_solve_feedback_age_s = post_solve_sensor_age_s
                            post_stale = (
                                not np.isfinite(post_solve_sensor_age_s)
                                or post_solve_sensor_age_s
                                > float(inner.cfg.feedback_timeout_s)
                            )
                            if post_stale:
                                (
                                    sensor_stale_streak_s,
                                    _post_coast,
                                    post_fault,
                                ) = accumulate_feedback_coast(
                                    sensor_stale_streak_s,
                                    dt_wall_actual,
                                    bad=True,
                                    limit_s=coast_limit_s,
                                )
                                if post_fault:
                                    publication_reason = (
                                        "feedback_stale_before_send:"
                                        f"age={post_solve_sensor_age_s:.6f}s"
                                        f":streak={sensor_stale_streak_s:.3f}s"
                                    )
                            elif not wd.beat():
                                publication_reason = "watchdog_latched_before_send"
                        if publication_reason:
                            phase_stopped = True
                            stop_reason = publication_reason
                            _fault_stop(stop_reason)
                            break
                        _t_send0 = time.perf_counter()
                        qdot0_pub = _qpik_rail_v_ff_m_s(
                            float(np.asarray(step.qdot, dtype=float).reshape(-1)[0])
                        )
                        rail_meas_pub = float("nan")
                        if rail_bridge is not None and getattr(rail_bridge, "enabled", False):
                            try:
                                rail_meas_pub = float(rail_bridge.measured_m)
                            except Exception:
                                rail_meas_pub = float("nan")
                        rail_pub_m = _wall_clock_rail_target(
                            float(step.q_send[0]),
                            qdot0_pub,
                            dt_wall_actual,
                            float(inner.cfg.dt),
                            soft_lo=float(inner.limits.q_lower[0]),
                            soft_hi=float(inner.limits.q_upper[0]),
                            meas_m=rail_meas_pub,
                            lead_max_m=float(inner.cfg.resync_err_rail_m),
                        )
                        if rail_coast_active:
                            qdot0_pub = 0.0
                            step.v_r_ref = 0.0
                            if rail_bridge is not None and getattr(
                                rail_bridge, "enabled", False
                            ):
                                try:
                                    rail_bridge.hold_current()
                                except Exception:
                                    pass
                            rail_ok, rail_reason = True, ""
                        else:
                            step.rail_target_publish_mono_ns = time.monotonic_ns()
                            rail_ok, rail_reason = _publish_rail_target_before_arm(
                                rail_bridge,
                                float(rail_pub_m),
                                _fault_stop,
                                v_ff_m_s=qdot0_pub,
                            )
                        if not rail_ok:
                            phase_stopped = True
                            stop_reason = rail_reason
                            break
                        if rail_bridge is not None:
                            step.rail_fa24_write_mono_ns = int(
                                getattr(rail_bridge, "last_fa24_write_mono_ns", 0) or 0
                            )
                            step.rail_encoder_sample_mono_ns = int(
                                getattr(rail_bridge, "last_encoder_sample_mono_ns", 0)
                                or 0
                            )
                        try:
                            step.arm_send_mono_ns = time.monotonic_ns()
                            _send_joint_canfd_cmd(
                                robot,
                                rad2deg(arm_q_from_full(step.q_send)),
                                follow,
                                canfd_proxy,
                            )
                        except Exception as exc:
                            phase_stopped = True
                            stop_reason = (
                                "arm_send_fault:"
                                f"{type(exc).__name__}:{exc}"
                            )
                            _fault_stop(stop_reason)
                            break
                        step.tick_send_ms = (
                            time.perf_counter() - _t_send0
                        ) * 1000.0
    
                        joint_err_deg = getattr(
                            phase.outer, "last_joint_err_deg", None
                        )
                        if joint_err_deg is None:
                            joint_err_deg = _joint_plan_err_deg(
                                phase.outer, t_ref, q_meas
                            )
                        raw_scale = _reference_governor_scale(
                            phase,
                            outer_err_mm=outer_err_mm,
                            joint_err_deg=joint_err_deg,
                            physical_saturated=bool(step.physical_saturated),
                        )
                        scale = gov_filter.update(raw_scale, control_dt)
                        # Soft-start ramp: first ~0.3s cannot command near-vmax.
                        ramp_s = float(getattr(phase, "soft_start_ramp_s", 0.0) or 0.0)
                        if ramp_s > 1e-6 and step.controller_mode != "direct_joint_ptp":
                            scale *= float(np.clip(t_wall / ramp_s, 0.0, 1.0))
                        t_ref += reference_time_step(dt_wall_actual, scale)
                        step.accepted_reference_lag_s = max(0.0, t_wall - t_ref)
    
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
                            # The write cannot time itself into its own row, so
                            # carry the previous tick's cost; over a run the
                            # statistics are the same.
                            step.tick_log_ms = last_log_ms
                            _t_log0 = time.perf_counter()
                            deadline_slack_s = (next_tick + dt) - time.perf_counter()
                            logger.write(
                                now - total_t0, phase.label, t_ref, step, q_meas, pose_pin, f_ext,
                                outer=phase.outer,
                                governor_scale=scale,
                                governor_scale_raw=raw_scale,
                                v_max=inner.limits.v_max,
                                rail_meas_m=rail_meas,
                                rail_target_sent_m=rail_pub_m,
                                dt_actual_s=dt_wall_actual,
                                deadline_slack_s=deadline_slack_s,
                                sensor_age_s=sensor_age_s,
                                feedback_age_s=feedback_age_s,
                                feedback_fresh_tick=feedback_fresh_tick,
                                f_ext_raw=f_ext_raw,
                                twist_achieved_base=twist_achieved_base,
                                v_tcp_z_actual=v_tcp_z_actual,
                                qdot_meas=qdot_meas,
                            )
                            last_log_ms = (
                                time.perf_counter() - _t_log0
                            ) * 1000.0
                        if on_step is not None:
                            on_step(phase.label, t_ref, step, pose_pin, f_ext, t_wall)

                        if phase.wait_until is not None:
                            n_wait = len(inspect.signature(phase.wait_until).parameters)
                            if n_wait >= 2:
                                phase_arrived = bool(phase.wait_until(pose_pin, q_meas))
                            else:
                                phase_arrived = bool(phase.wait_until(pose_pin))
                            if arrival_gate.update(
                                geometric_arrival=phase_arrived,
                                t_ref_s=t_ref,
                                qdot_applied=step.qdot,
                                dt_s=control_dt,
                                rail_settled=_rail_settled_for_arrival(
                                    rail_bridge,
                                    speed_limit_m_s=phase.arrival_rail_speed_m_s,
                                    now_s=time.monotonic(),
                                    freshness_s=inner.cfg.feedback_timeout_s,
                                ),
                            ):
                                phase_arrived = True
                                break
                            phase_arrived = False
    
                        ticks += 1
                        next_tick += dt
                        _wait_until(next_tick)

                    if phase.on_exit is not None:
                        phase.on_exit()

                    if phase_stopped:
                        break

                    if phase.require_arrival and not phase_arrived:
                        stop_reason = f"arrival_timeout:{phase.label or phase_idx}"
                        phase_stopped = True
                        _fault_stop(stop_reason)
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
                            f"— safety stop",
                            flush=True,
                        )
                        break
            except KeyboardInterrupt:
                if verbose:
                    print("\nStopped.", flush=True)
        finally:
            inner.set_direct_joint_ptp(False)
            inner.set_plan_drives_rail(False)
            wd.stop()
            stalled = wd.fired
    finally:
        if own_bus:
            state_bus.stop()
        if logger is not None:
            logger.close()
        if cstate is not None:
            cstate.__exit__(None, None, None)
        if locals().get("gc_frozen"):
            gc.enable()
            gc.unfreeze()

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

### `rm75_control/control/joint_admittance_8dof/tasks/rail_extension.py`

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

### `rm75_control/control/joint_admittance_8dof/tasks/rail_allocator.py`

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
    # Reference-model cutoff.  τ = 1/(2π f_c).
    f_c_hz: float = 20.0
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
        f_c_hz: float = 20.0,
        a_max: float = 0.60,
        j_max: float = 60.0,
        v_max: float = 0.12,
        reaction_s: float = 0.06,
        soft_min_m: float = 0.015,
        soft_max_m: float = 0.77,
    ) -> None:
        self.f_c_hz = float(f_c_hz)
        self.a_max = float(a_max)
        self.j_max = float(j_max)
        self.v_max = float(v_max)
        self.reaction_s = float(reaction_s)
        self.soft_min_m = float(soft_min_m)
        self.soft_max_m = float(soft_max_m)
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
    ) -> float:
        dt = float(dt_s)
        if dt <= 1.0e-9:
            return float(self.state.v)
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
        da_max = float(self.j_max) * dt
        a = float(np.clip(a_raw, a_prev - da_max, a_prev + da_max))
        a = float(np.clip(a, -self.a_max, self.a_max))
        v = v_prev + a * dt
        v = float(np.clip(v, -self.v_max, self.v_max))
        self.last_wall_override = False
        if apply_wall:
            lo_cap, hi_cap = wall_cap(
                float(x_m),
                lo=self.soft_min_m,
                hi=self.soft_max_m,
                a_max=self.a_max,
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
    ) -> tuple[float, float]:
        if not self._initialized:
            self.reset(q_meas, float(v_meas) if v_meas is not None else 0.0)
            self._last_sample_t = float(sample_t)
            return float(self.q_hat), float(self.v_hat)
        dt = max(float(dt_s), 1.0e-6)
        v_pred = float(v_r_ref)
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
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.v_max = float(v_max)
        self.integ = 0.0

    def reset(self) -> None:
        self.integ = 0.0

    def step(self, err_m: float, dt_s: float, *, freeze: bool = False) -> float:
        err = float(err_m) if np.isfinite(err_m) else 0.0
        dt = max(float(dt_s), 0.0)
        if not freeze and dt > 0.0:
            self.integ += self.ki * err * dt
        raw = self.kp * err + self.integ
        sat = soft_saturate(raw, self.v_max)
        if not freeze and abs(raw) > self.v_max:
            self.integ -= self.ki * err * dt
        return float(sat)


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
    "lpf_tau_from_fc",
    "margin_weight_from_activation",
    "project_arm_compensation",
    "soft_saturate",
    "stopping_velocity",
    "wall_cap",
)
```

### `rm75_control/control/joint_admittance_8dof/tasks/psi_retarget.py`

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
current TCP (same branch), not the yaml photo at t=0.  ``d*`` is the
split that keeps IK J4 in the design band (center ~95°), then eases
toward ``d_attr``.  A live outer ``vel_ff`` freezes ``s`` via
``hold_setpoint``; an infeasible intermediate IK also freezes ``s``.
Local ψ search takes over only while the wrist is collapsed and the
elbow is still open (SEW is undefined near the J4 floor).
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
        self.last_psi_score = float("nan")
        self.last_dpref_score = float("nan")
        self.last_minmax_margin = float("nan")
        self.last_psi_family_degraded = False
        self._update_margins(q)

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
        self._maybe_retarget_psi(
            q,
            dt_s=dt,
            rail_lo=float(rail_lo),
            rail_hi=float(rail_hi),
        )
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
        has_travel = (d_hi - d_lo) > 0.01
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
    ) -> float:
        """Shift d* so rail_ff = y_des − d* stays inside the soft travel."""
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
        self._d_star = d_new
        self.d_star_m = d_new
        self._d0 = d_new
        return d_new

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

### `rm75_control/control/joint_admittance_8dof/solver/qp_builder.py`

```python
"""WBC velocity-IK core: strict two-level QP + CBF self-collision constraints.

Formulation (Escande et al. 2014 slack task + Faverjon velocity damper / Khazoom CBF):

    x = [qdot; w]  in R^{nv+6}

    QP1: min 0.5 wᵀ W_task w
         J_task qdot - w = v_cmd                   (protected equality)
         l_box <= qdot <= u_box, J_col qdot >= v_safe

    QP2: keep QP1's achieved J_task qdot as a hard equality and minimize
         regularization, posture and rail preferences.  Thus a rail box can
         only change the arm/rail allocation when the Cartesian task remains
         unchanged; it cannot buy task slack with a finite soft weight.

H is block-diagonal (no J^T J).  ProxQP warm-started each tick.

This layer consumes a *given* task twist ``v_cmd`` verbatim (Escande et al. 2014
Sec. III): the position-feedback loop that produces the twist lives exactly once
in the caller (outer loop / pose_ik), never here.  If ``rail_exec_vel_m_s`` is
provided, its measured TCP contribution is subtracted from the current task;
the rail command remains a next-sample secondary decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

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
from rm75_control.control.joint_admittance_8dof.solver import cpp_kernel
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.branch_barrier import (
    BranchBarrierBuilder,
    BranchBarrierConfig,
)
from rm75_control.control.joint_admittance_8dof.solver.joint_comfort import (
    JointComfortBuilder,
    JointComfortConfig,
)
from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import (
    CbfRows,
    CbfSlotTracker,
    build_cbf_rows,
)
from rm75_control.control.joint_admittance_8dof.filters import (
    first_order_lpf,
    first_order_lpf_vec,
)
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    VelocityBoxConstraints,
    build_wbc_inequalities,
    collapse_interval,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_allocator import (
    project_arm_compensation,
)
from rm75_control.control.joint_admittance_8dof.solver.sigma_setbased import (
    PrefInequalityRows,
    SigmaSetBasedConfig,
    SigmaSetBasedTracker,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits

N_TASK_SLACK = 6
N_PREF_SLACK = 9  # [sigma, branch, J1..J7 comfort]
MAX_PREF_ROWS = 16  # 1 sigma + 7 branch + 7 comfort
# Backward-compatible alias used by older call sites / tests.
N_SLACK = N_TASK_SLACK


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
    use_cpp_kernel: bool = True
    eps_abs: float = 1e-6
    max_iter: int = 200
    # Clamp applied in ProxQP backend so a yaml typo (e.g. 3000) cannot freeze
    # the 200 Hz loop for seconds near singularities / CBF.
    max_iter_cap: int = 400
    euler_order: str = "xyz"
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    # Chiaverini 1997 SR damping for nullspace projection.
    sr_damping: SrDampingConfig = field(default_factory=SrDampingConfig)
    # σ-adaptive primary-task weight (Chiaverini-style): as σ_min ↘, scale
    # W_task toward task_weight_min_frac so the slack absorbs infeasible
    # v_cmd instead of saturating qdot with near-zero TCP motion.  LPF on the
    # scale avoids the bang-bang chatter that motivated the (over-broad) Bug 1
    # removal — only the primary cost softens; rail_extension / reg stay put.
    task_weight_min_frac: float = 0.05
    task_weight_lpf_tau_s: float = 0.25
    # Chiaverini 1994 numerical filtering: only the degenerate left
    # singular directions of W^{1/2} J lose task weight.  Off falls
    # back to the isotropic (σ_min / σ_ref)² scale.
    aniso_task_damping: bool = True
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
    # Faverjon/Tournassoud joint-limit velocity damper band: allowed speed
    # toward a limit ramps to 0 across this zone before the margin.  Units are
    # PER JOINT: rad for the arm, metres for the prismatic rail.  The old
    # scalar band applied 0.15 "rad" = 0.15 m to the rail — the damper started
    # throttling rail velocity from |y| > 6.5 cm (60% of the ±0.25 m travel),
    # exactly where the rail is needed most to rescue arm singularities.
    limit_damper_band_rad: float = 0.15      # arm joints 1..7 (rad)
    limit_damper_band_rail_m: float = 0.01   # rail joint 0 (metres)
    # Rail stopping-envelope look-ahead.  0 uses the control period only.
    limit_damper_rail_reaction_s: float = 0.06
    warn_on_fail: bool = True
    # Deprecated compatibility setting.  Strict HQP fails closed on QP1
    # failure and never publishes a decayed previous command.
    fail_qdot_decay: float = 0.85
    # Hard wall-clock budget for one ProxQP attempt+retry (ms).  Exceeding
    # this skips the retry and returns fail — prevents GIL freezes of
    # multiple seconds near σ→0 that starve the rail Modbus loop (PANIC).
    max_solve_ms: float = 5.0
    # Below this σ_min, Cartesian twist (incl. force) is scaled down so
    # nullspace escape / rail recruitment can win over force-driven collapse.
    # Keep a tiny numeric floor; set-based σ + rail do the real "尽量不进".
    twist_sigma_floor: float = 0.02
    sigma_setbased: SigmaSetBasedConfig = field(default_factory=SigmaSetBasedConfig)
    branch_barrier: BranchBarrierConfig = field(default_factory=BranchBarrierConfig)
    joint_comfort: JointComfortConfig = field(default_factory=JointComfortConfig)
    # Arm joints this close to a stop count as physically saturated (rad).
    near_arm_margin_rad: float = 0.08
    # Soft velocity continuity: ½ w_s ‖q̇ − q̇_prev‖² added to the QP cost
    # (no extra decision variable).  0 disables.
    # May be a scalar or one value per joint.  A vector lets the rail use no
    # velocity-continuity preference while the arm keeps its tuned value.
    smoothness_weight: float | np.ndarray = 0.15
    # Third-order box on |a_k - a_{k-1}|.  The velocity and acceleration boxes
    # alone let the commanded acceleration flip sign every tick; this bounds
    # how fast it may turn.  0 disables either axis.
    j_max_arm_rad_s3: float = 300.0
    j_max_rail_m_s3: float = 3.0


class _ProxQpWbcBackend:
    def __init__(
        self,
        nv: int,
        max_cbf: int,
        cfg: QpConfig,
        *,
        n_eq: int = N_TASK_SLACK,
        allow_retry: bool = True,
    ) -> None:
        import proxsuite

        self._px = proxsuite
        self.nv = nv
        self.n_task_slack = N_TASK_SLACK
        self.n_pref_slack = N_PREF_SLACK
        self.n_slack = N_TASK_SLACK  # task equality slacks only
        self.n_var = nv + N_TASK_SLACK + N_PREF_SLACK
        self.n_eq = int(n_eq)
        self.n_in = nv + max_cbf + MAX_PREF_ROWS + N_PREF_SLACK
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
        # Strict HQP uses one solve per level.  Keep the legacy retry for the
        # old constructor/API, but never retry a strict level: a retry here
        # would silently turn the fixed two-solve budget into an SNS loop.
        self._allow_retry = bool(allow_retry)
        self.last_solve_ms = 0.0
        self.last_status = "not_run"

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
        *,
        warm_start_x: np.ndarray | None = None,
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
        if warm_start_x is not None:
            seed = np.asarray(warm_start_x, dtype=float).reshape(self.n_var)
            self.qp.settings.initial_guess = self._px.proxqp.InitialGuess.WARM_START
            self.qp.solve(seed, None, None)
        else:
            self.qp.solve()
        elapsed = _time.perf_counter() - t0
        self.last_solve_ms = elapsed * 1000.0
        self.last_status = str(self._status())

        if not self._solved() and self._allow_retry:
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
                self.last_solve_ms = (
                    _time.perf_counter() - t0
                ) * 1000.0
                self.last_status = str(self._status())

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
        self.last_status = "solved"
        return np.asarray(self.qp.results.x, dtype=float)


class _OsqpWbcBackend:
    """Fallback when ProxQP unavailable (no warm equality+ineq resize)."""

    def __init__(
        self,
        nv: int,
        max_cbf: int,
        cfg: QpConfig,
        *,
        allow_retry: bool = False,
    ) -> None:
        import osqp
        import scipy.sparse as sp

        self._osqp = osqp
        self._sp = sp
        self.nv = nv
        self.n_task_slack = N_TASK_SLACK
        self.n_pref_slack = N_PREF_SLACK
        self.n_slack = N_TASK_SLACK
        self.n_var = nv + N_TASK_SLACK + N_PREF_SLACK
        self.n_in = nv + max_cbf + MAX_PREF_ROWS + N_PREF_SLACK
        self.cfg = cfg
        self.prob = None
        self.last_solve_ms = 0.0
        self.last_status = "not_run"
        self._allow_retry = bool(allow_retry)

    def solve(self, H, g, A, b, C, lo, hi, *, warm_start_x=None):
        import time as _time

        sp = self._sp
        t0 = _time.perf_counter()
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
        if warm_start_x is not None:
            self.prob.warm_start(x=np.asarray(warm_start_x, dtype=float))
        res = self.prob.solve()
        self.last_solve_ms = (_time.perf_counter() - t0) * 1000.0
        if res.x is None or np.any(np.isnan(res.x)):
            self.last_status = "failed"
            return None
        self.last_status = "solved"
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
        task_weight = np.asarray(self.cfg.task_weight, dtype=float).reshape(-1)
        if (
            task_weight.size != N_TASK_SLACK
            or not np.all(np.isfinite(task_weight))
            or np.any(task_weight <= 0.0)
        ):
            raise ValueError(
                "task_weight must contain six finite, strictly positive values"
            )
        reg_weight = np.asarray(self.cfg.reg, dtype=float).reshape(-1)
        if reg_weight.size not in (1, int(kin.nv)):
            raise ValueError(
                f"reg must be scalar or contain {int(kin.nv)} values"
            )
        if not np.all(np.isfinite(reg_weight)) or np.any(reg_weight < 0.0):
            raise ValueError("reg must contain finite, non-negative values")
        # Per-joint damper band: arm in rad, prismatic rail (joint 0) in m.
        damper_band = np.full(kin.nv, float(self.cfg.limit_damper_band_rad))
        damper_band[0] = float(self.cfg.limit_damper_band_rail_m)
        self.constraints = VelocityBoxConstraints(
            limits,
            damper_band_rad=damper_band,
            rail_reaction_s=float(self.cfg.limit_damper_rail_reaction_s),
        )
        self.collision_cfg = self.cfg.collision
        self._max_cbf = max(1, int(self.collision_cfg.max_pairs))
        self.collision = collision
        if self.collision_cfg.enabled and self.collision is None:
            self.collision = CollisionModel(kin.model)
        self._cbf_slots = CbfSlotTracker(max_pairs=self._max_cbf)
        self.sigma_setbased = SigmaSetBasedTracker(self.cfg.sigma_setbased)
        self.branch_barrier = BranchBarrierBuilder(self.cfg.branch_barrier)
        self.joint_comfort = JointComfortBuilder(self.cfg.joint_comfort)
        self.qdot_prev = np.zeros(kin.nv, dtype=float)
        self.qdot_prev2 = np.zeros(kin.nv, dtype=float)
        self._qdot_prev_seen = np.zeros(kin.nv, dtype=float)
        j_max = np.full(kin.nv, float(self.cfg.j_max_arm_rad_s3), dtype=float)
        j_max[0] = float(self.cfg.j_max_rail_m_s3)
        self._j_max = j_max if np.all(j_max > 0.0) else None
        self._m_diag_lpf: np.ndarray | None = None
        self._task_scale_lpf: float = 1.0
        self._s_lpf: np.ndarray | None = None
        self._U_prev: np.ndarray | None = None
        self.last_task_weight_mat: np.ndarray = np.diag(
            np.asarray(self.cfg.task_weight, dtype=float)
        )
        self.last_s_sigma: np.ndarray = np.ones(N_TASK_SLACK, dtype=float)
        self.solve_count = 0
        self.last_status = "not_run"
        self.last_failed = False
        self.last_dexterity_slack = 0.0
        self.last_branch_slack = 0.0
        self.last_comfort_slack = np.zeros(7, dtype=float)
        self.last_sns_scale = 1.0
        self.last_cbf_min_dist = float("nan")
        self.last_cbf_pair = ""
        self.last_cbf_active_names: tuple[str, ...] = ()
        self.last_comp_projected_frac = 0.0
        self.last_wln_scale = np.ones(kin.nv, dtype=float)
        self._wln_scale_prev = np.ones(kin.nv, dtype=float)
        self.q_star: np.ndarray | None = None
        self.q_star_signs: np.ndarray | None = None
        self.backend = self._make_backend(kin.nv, slot="qp1")
        # Both levels have six fixed equality rows.  QP1 uses
        # ``J qdot - residual = target``; QP2 directly locks
        # ``J qdot = achieved_qp1``.  The direct form avoids a redundant
        # 12-row [task; residual==0] system that ProxQP could misclassify as
        # infeasible near rank loss even though the QP1 point was feasible.
        self._backend_qp2 = self._make_backend(kin.nv, n_eq=N_TASK_SLACK, slot="qp2")

        # Strict-HQP telemetry.  These are controller attributes rather than
        # IkStepResult fields for backwards compatibility with existing loop
        # and CSV consumers; callers that need them can read them immediately
        # after ``step``.
        self.last_qp1_status = "not_run"
        self.last_qp2_status = "not_run"
        self.last_qp1_solve_ms = 0.0
        self.last_qp2_solve_ms = 0.0
        self.last_qp_total_ms = 0.0
        self.last_fallback_ms = 0.0
        self.last_task_residual = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_task_residual_norm = 0.0
        self.last_qp1_residual = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_qp2_residual = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_qp1_residual_norm = 0.0
        self.last_qp2_residual_norm = 0.0
        self.last_task_target = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_task_achieved = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_rail_exec_contrib = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_rail_cmd_contrib = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_arm_contrib = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_qp2_fallback = False
        self.last_zero_slack_feasible = False
        self.last_hard_residual = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_qdot_qp1 = np.zeros(kin.nv, dtype=float)
        self.last_qp1_hard_violation = 0.0
        self.last_final_hard_violation = 0.0
        self.last_lo_box = np.full(kin.nv, -np.inf, dtype=float)
        self.last_hi_box = np.full(kin.nv, np.inf, dtype=float)
        self.last_qp2_seed_violation = 0.0
        self.last_qp2_seed_equality = 0.0
        # Final-publication certificate.  Retain the qdot-only hard set and
        # QP1 task value needed to verify the command that will actually be
        # sent.  Preference slack rows are deliberately excluded here.
        self.last_hard_cbf_jacobian = np.zeros((0, kin.nv), dtype=float)
        self.last_hard_cbf_lower = np.zeros(0, dtype=float)
        self.last_qp1_task_velocity = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_task_jacobian = np.zeros((N_TASK_SLACK, kin.nv), dtype=float)
        self.last_final_task_lock_violation = 0.0
        self.last_a_mirror_frac = float("nan")
        self.last_j_mirror_frac = float("nan")
        self.last_qp_overrun = False
        self._rail_exec_prev: float | None = None
        self._rail_a_prev: float | None = None

        w_reg = np.asarray(self.cfg.reg, dtype=float)
        if w_reg.ndim == 0 or w_reg.size == 1:
            w_reg = np.full(kin.nv, float(w_reg))
        self._w_reg = w_reg
        self._w_task = task_weight

    def _make_backend(
        self,
        nv: int,
        *,
        n_eq: int = N_TASK_SLACK,
        slot: str = "qp1",
    ):
        want = self.cfg.backend.lower()
        key = (want, int(nv), int(n_eq), int(self._max_cbf), str(slot))
        cache = getattr(self.kin, "_qp_backend_cache", None)
        if cache is None:
            cache = {}
            setattr(self.kin, "_qp_backend_cache", cache)
        cached = cache.get(key)
        if cached is not None:
            return cached
        backend = None
        if want == "proxqp":
            try:
                backend = _ProxQpWbcBackend(
                    nv,
                    self._max_cbf,
                    self.cfg,
                    n_eq=n_eq,
                    allow_retry=False,
                )
            except Exception:
                backend = None
        if backend is None and want in ("osqp", "proxqp"):
            try:
                backend = _OsqpWbcBackend(nv, self._max_cbf, self.cfg)
            except Exception as exc:
                raise RuntimeError(
                    "No QP backend available (install proxsuite or osqp)"
                ) from exc
        if backend is None:
            raise ValueError(f"unknown QP backend {self.cfg.backend!r}")
        cache[key] = backend
        return backend

    @property
    def backend_name(self) -> str:
        return type(self.backend).__name__.replace("_", "").replace("Backend", "").lower()

    def reset(self, q0_rad: np.ndarray | None = None) -> None:
        del q0_rad  # QP state is velocity history / LPF only
        self.qdot_prev = np.zeros(self.kin.nv, dtype=float)
        self.qdot_prev2 = np.zeros(self.kin.nv, dtype=float)
        self._qdot_prev_seen = np.zeros(self.kin.nv, dtype=float)
        self._m_diag_lpf = None
        self._task_scale_lpf = 1.0
        self._s_lpf = None
        self._U_prev = None
        self.last_task_weight_mat = np.diag(np.asarray(self.cfg.task_weight, dtype=float))
        self.last_s_sigma = np.ones(N_TASK_SLACK, dtype=float)
        self.solve_count = 0
        self.last_status = "not_run"
        self.last_failed = False
        self.last_dexterity_slack = 0.0
        self.last_branch_slack = 0.0
        self.last_comfort_slack = np.zeros(7, dtype=float)
        self.last_sns_scale = 1.0
        self.last_cbf_min_dist = float("nan")
        self.last_cbf_pair = ""
        self.last_cbf_active_names = ()
        self.last_wln_scale = np.ones(self.kin.nv, dtype=float)
        self._wln_scale_prev = np.ones(self.kin.nv, dtype=float)
        self.last_qp1_status = "not_run"
        self.last_qp2_status = "not_run"
        self.last_qp1_solve_ms = 0.0
        self.last_qp2_solve_ms = 0.0
        self.last_qp_total_ms = 0.0
        self.last_fallback_ms = 0.0
        self.last_task_residual = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_task_residual_norm = 0.0
        self.last_qp1_residual = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_qp2_residual = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_qp1_residual_norm = 0.0
        self.last_qp2_residual_norm = 0.0
        self.last_task_target = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_task_achieved = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_rail_exec_contrib = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_rail_cmd_contrib = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_arm_contrib = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_qp2_fallback = False
        self.last_hard_residual = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_qdot_qp1 = np.zeros(self.kin.nv, dtype=float)
        self.last_qp1_hard_violation = 0.0
        self.last_final_hard_violation = 0.0
        self.last_lo_box = np.full(self.kin.nv, -np.inf, dtype=float)
        self.last_hi_box = np.full(self.kin.nv, np.inf, dtype=float)
        self.last_qp2_seed_violation = 0.0
        self.last_qp2_seed_equality = 0.0
        self.last_hard_cbf_jacobian = np.zeros((0, self.kin.nv), dtype=float)
        self.last_hard_cbf_lower = np.zeros(0, dtype=float)
        self.last_qp1_task_velocity = np.zeros(N_TASK_SLACK, dtype=float)
        self.last_task_jacobian = np.zeros(
            (N_TASK_SLACK, self.kin.nv), dtype=float
        )
        self.last_final_task_lock_violation = 0.0
        self.last_a_mirror_frac = float("nan")
        self.last_j_mirror_frac = float("nan")
        self.last_qp_overrun = False
        self._rail_exec_prev = None
        self._rail_a_prev = None
        self.sigma_setbased.reset()
        self.branch_barrier.reset()
        self.joint_comfort.reset()

    def set_q_star(self, q_star: np.ndarray | None) -> None:
        """Homotopy / centering attractor (not necessarily yaml signs)."""
        if q_star is None:
            self.q_star = None
        else:
            self.q_star = np.asarray(q_star, dtype=float).reshape(-1).copy()

    def set_q_star_signs(self, q_star: np.ndarray | None) -> None:
        """Yaml-family signs for the near-zero branch barrier."""
        if q_star is None:
            self.q_star_signs = None
        else:
            self.q_star_signs = np.asarray(q_star, dtype=float).reshape(-1).copy()

    def sync_applied(self, qdot: np.ndarray) -> None:
        """Seed velocity history from an already-applied command."""
        self.qdot_prev = np.asarray(qdot, dtype=float).reshape(-1).copy()
        # An episode boundary is not a jerk event: start the third-order
        # history flat so the first tick is not boxed against a stale value.
        self.qdot_prev2 = self.qdot_prev.copy()
        self._qdot_prev_seen = self.qdot_prev.copy()

    def validate_final_qdot(self, qdot: np.ndarray) -> tuple[float, float]:
        """Certify a post-QP command against P0 and the QP1 task lock.

        Returns ``(hard_violation, task_lock_violation)`` as infinity norms.
        This is intentionally independent of QP2 preference slacks: only the
        velocity box, measured-rail CBF rows and the protected task value can
        make a hardware command unsafe or violate the hierarchy.
        """

        qdot_arr = np.asarray(qdot, dtype=float).reshape(-1)
        if qdot_arr.size != self.kin.nv or not np.all(np.isfinite(qdot_arr)):
            return float("inf"), float("inf")
        hard = max(
            float(np.max(np.maximum(self.last_lo_box - qdot_arr, 0.0), initial=0.0)),
            float(np.max(np.maximum(qdot_arr - self.last_hi_box, 0.0), initial=0.0)),
        )
        if self.last_hard_cbf_jacobian.size:
            cbf_value = self.last_hard_cbf_jacobian @ qdot_arr
            hard = max(
                hard,
                float(
                    np.max(
                        np.maximum(self.last_hard_cbf_lower - cbf_value, 0.0),
                        initial=0.0,
                    )
                ),
            )
        task_value = self.last_task_jacobian @ qdot_arr
        task_lock = float(
            np.max(np.abs(task_value - self.last_qp1_task_velocity), initial=0.0)
        )
        return hard, task_lock

    def _task_scale_sigma(self, sigma_min: float, dt: float) -> float:
        """LPF-smoothed W_task scale in [min_frac, 1] from σ_min."""
        sigma_ref = float(self.cfg.sr_damping.sigma_ref)
        raw = 1.0
        if sigma_ref > 1e-9 and sigma_min < sigma_ref:
            frac = float(sigma_min) / sigma_ref
            raw = max(frac * frac, float(self.cfg.task_weight_min_frac))
        tau = float(self.cfg.task_weight_lpf_tau_s)
        self._task_scale_lpf = first_order_lpf(
            self._task_scale_lpf, raw, dt, tau
        )
        return float(self._task_scale_lpf)

    def _task_weight_matrix(
        self,
        J: np.ndarray,
        dt: float,
        *,
        keep_task_weight: bool,
    ) -> np.ndarray:
        """Task slack Hessian block.  Aniso: only degenerate directions fade."""
        w = np.asarray(self._w_task, dtype=float).reshape(-1)
        ns = int(w.size)
        if keep_task_weight or not bool(getattr(self.cfg, "aniso_task_damping", True)):
            scale = 1.0 if keep_task_weight else self._task_scale_sigma(
                float(np.linalg.svd(J, compute_uv=False).min()), dt
            )
            mat = np.diag(w * scale)
            self.last_task_weight_mat = mat
            self.last_s_sigma = np.full(ns, scale, dtype=float)
            return mat
        w_sqrt = np.sqrt(np.maximum(w, 1.0e-12))
        jw = w_sqrt[:, None] * np.asarray(J, dtype=float)
        u, s_j, _vt = np.linalg.svd(jw, full_matrices=False)
        if u.shape[1] < ns:
            u_full = np.eye(ns, dtype=float)
            u_full[:, : u.shape[1]] = u
            u = u_full
            s_pad = np.zeros(ns, dtype=float)
            s_pad[: s_j.size] = s_j
            s_j = s_pad
        if self._U_prev is not None and self._U_prev.shape == u.shape:
            for i in range(u.shape[1]):
                if float(np.dot(u[:, i], self._U_prev[:, i])) < 0.0:
                    u[:, i] *= -1.0
        self._U_prev = u.copy()
        sigma_ref = float(self.cfg.sr_damping.sigma_ref)
        min_frac = float(self.cfg.task_weight_min_frac)
        s_raw = np.ones(ns, dtype=float)
        for i, si in enumerate(s_j[:ns]):
            if sigma_ref > 1.0e-9 and float(si) < sigma_ref:
                s_raw[i] = max((float(si) / sigma_ref) ** 2, min_frac)
        tau = float(self.cfg.task_weight_lpf_tau_s)
        if self._s_lpf is None or self._s_lpf.size != ns:
            self._s_lpf = s_raw.copy()
        elif tau > 1.0e-9 and dt > 1.0e-9:
            self._s_lpf = first_order_lpf_vec(self._s_lpf, s_raw, dt, tau)
        else:
            self._s_lpf = s_raw.copy()
        self.last_s_sigma = np.asarray(self._s_lpf, dtype=float).copy()
        usu = u @ np.diag(self.last_s_sigma) @ u.T
        mat = (w_sqrt[:, None] * usu) * w_sqrt[None, :]
        self.last_task_weight_mat = mat
        return mat

    def _update_mirror_telemetry(
        self,
        J: np.ndarray,
        *,
        rail_exec: float | None,
        h1: float,
    ) -> None:
        """Fraction of the arm a/j boxes spent cancelling measured rail motion."""
        self.last_a_mirror_frac = float("nan")
        self.last_j_mirror_frac = float("nan")
        if rail_exec is None or not np.isfinite(float(rail_exec)):
            self._rail_exec_prev = None
            self._rail_a_prev = None
            return
        period = float(h1)
        if not np.isfinite(period) or period <= 1.0e-9:
            self._rail_exec_prev = float(rail_exec)
            return
        a_rail = 0.0
        if self._rail_exec_prev is not None:
            a_rail = (float(rail_exec) - float(self._rail_exec_prev)) / period
        self._rail_exec_prev = float(rail_exec)
        j_rail = 0.0
        if self._rail_a_prev is not None:
            j_rail = (a_rail - float(self._rail_a_prev)) / period
        self._rail_a_prev = float(a_rail)
        ja = np.asarray(J[:, 1:], dtype=float)
        jr = np.asarray(J[:, 0], dtype=float)
        if ja.size == 0:
            return
        # Telemetry only.  Skip the 6×7 pinv when the rail is not accelerating.
        if abs(a_rail) < 1.0e-9 and abs(j_rail) < 1.0e-9:
            self.last_a_mirror_frac = 0.0
            self.last_j_mirror_frac = 0.0
            return
        try:
            qa_dir, *_ = np.linalg.lstsq(ja, jr, rcond=None)
        except np.linalg.LinAlgError:
            return
        a_max = self.constraints.lim.a_max
        if a_max is not None:
            a_arm = np.asarray(a_max, dtype=float).reshape(-1)[1:]
            if a_arm.size:
                qa = qa_dir * a_rail
                den = np.maximum(np.abs(a_arm), 1.0e-9)
                self.last_a_mirror_frac = float(np.max(np.abs(qa) / den))
        if self._j_max is not None:
            j_arm = np.asarray(self._j_max, dtype=float).reshape(-1)[1:]
            if j_arm.size:
                qj = qa_dir * j_rail
                den = np.maximum(np.abs(j_arm), 1.0e-9)
                self.last_j_mirror_frac = float(np.max(np.abs(qj) / den))

    def set_collision_enabled(self, enabled: bool) -> None:
        self.collision_cfg.enabled = bool(enabled)

    def _merge_pref_rows(
        self, *parts: PrefInequalityRows
    ) -> PrefInequalityRows:
        jac_list = [p.jacobian for p in parts if p.active and p.jacobian.size]
        if not jac_list:
            nv = self.kin.nv
            return PrefInequalityRows(
                jacobian=np.zeros((0, nv)),
                slack_col=np.zeros(0, dtype=int),
                lower=np.zeros(0),
                active=False,
            )
        jac = np.vstack(jac_list)
        scol = np.concatenate([p.slack_col for p in parts if p.active and p.jacobian.size])
        lo = np.concatenate([p.lower for p in parts if p.active and p.jacobian.size])
        if jac.shape[0] > MAX_PREF_ROWS:
            jac = jac[:MAX_PREF_ROWS]
            scol = scol[:MAX_PREF_ROWS]
            lo = lo[:MAX_PREF_ROWS]
        return PrefInequalityRows(
            jacobian=jac, slack_col=scol.astype(int), lower=lo, active=True
        )

    def _solve_qp(self, backend, H, g, A, b, C, lo, hi, *, warm_start_x=None):
        if bool(getattr(self.cfg, "use_cpp_kernel", True)) and cpp_kernel.available():
            packed = cpp_kernel.solve_dense_qp(
                H,
                g,
                A,
                b,
                C,
                lo,
                hi,
                warm_x=warm_start_x,
                max_iter=int(
                    min(max(int(self.cfg.max_iter), 1), int(self.cfg.max_iter_cap))
                ),
                eps_abs=float(self.cfg.eps_abs),
            )
            if packed is not None:
                x, ms, status = packed
                backend.last_solve_ms = float(ms)
                backend.last_status = str(status)
                return x
        return backend.solve(H, g, A, b, C, lo, hi, warm_start_x=warm_start_x)

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
        rail_reg_scale: float = 1.0,
        rail_lock_vel_eps_m_s: float = 0.0,
        rail_vel_pin_m_s: float | None = None,
        zero_secondary_rail: bool = False,
        rail_task_vel_m_s: float | None = None,
        rail_task_weight: float = 0.0,
        box_dt: float | None = None,
        box_h1: float | None = None,
        box_h2: float | None = None,
        keep_task_weight: bool = False,
        pref_slack_scale: float = 1.0,
        rail_exec_vel_m_s: float | None = None,
        jacobian: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        mass_matrix: np.ndarray | None = None,
        kinematics_ready: bool = False,
        rail_open_travel: bool = False,
        arm_qdot_pref: np.ndarray | None = None,
    ) -> IkStepResult:
        t_total = time.perf_counter()
        q_prev = np.asarray(q_prev, dtype=float).reshape(-1)
        nv = self.kin.nv
        if q_prev.size != nv:
            raise ValueError(f"q_prev must have {nv} joints, got {q_prev.size}")
        # ``qdot_prev`` is whatever the loop actually applied last tick (it may
        # rewrite it after clamping), so shift the third-order history here.
        self.qdot_prev2 = self._qdot_prev_seen
        self._qdot_prev_seen = np.asarray(self.qdot_prev, dtype=float).copy()
        v_cmd0 = np.asarray(twist_ref, dtype=float).reshape(N_TASK_SLACK)
        self.solve_count += 1
        self.last_qp2_fallback = False
        self.last_fallback_ms = 0.0

        # The measured state is authoritative for the kinematic snapshot.  A
        # precomputed snapshot may be supplied by the caller to avoid doing
        # FK/J/SVD/M twice in a 200 Hz loop.
        q_geom = (
            np.asarray(q_meas, dtype=float).reshape(-1)
            if q_meas is not None
            else q_prev
        )
        if q_geom.size != nv:
            raise ValueError(f"q_meas must have {nv} joints, got {q_geom.size}")
        J = (
            np.asarray(jacobian, dtype=float)
            if jacobian is not None
            else self.kin.jacobian(q_geom)
        )
        if J.shape != (N_TASK_SLACK, nv) or not np.all(np.isfinite(J)):
            raise ValueError(f"jacobian must have shape {(N_TASK_SLACK, nv)}")
        sigma_arr = (
            np.asarray(sigma, dtype=float).reshape(-1)
            if sigma is not None
            else self.kin.singular_values(J)
        )
        sigma_min = float(np.min(sigma_arr)) if sigma_arr.size else 0.0

        # When available, the rail feedback represents the motion that has
        # actually happened during this sample.  The rail command remains a
        # decision variable for the next sample, but is excluded from the
        # current task map so the arm solves the measured residual directly.
        rail_exec = None
        rail_exec_contrib = np.zeros(N_TASK_SLACK, dtype=float)
        J_task = np.asarray(J, dtype=float).copy()
        self.last_comp_projected_frac = 0.0
        if rail_exec_vel_m_s is not None and np.isfinite(float(rail_exec_vel_m_s)):
            rail_exec = float(rail_exec_vel_m_s)
            rail_exec_contrib = J[:, 0] * rail_exec
            J_task[:, 0] = 0.0
            delta_v_req = -rail_exec_contrib
            delta_v_cmp, frac = project_arm_compensation(
                J,
                delta_v_req,
                q_geom,
                self.constraints.lim.q_lower,
                self.constraints.lim.q_upper,
            )
            b_task = v_cmd0 + delta_v_cmp
            self.last_comp_projected_frac = float(frac)
        else:
            b_task = v_cmd0 - rail_exec_contrib
        # Public telemetry is expressed in the caller's original Cartesian
        # coordinates.  ``b_task`` is the internal arm-only target after
        # subtracting the measured rail contribution.
        self.last_task_target = v_cmd0.copy()
        self.last_rail_exec_contrib = rail_exec_contrib.copy()

        # Chiaverini SR projection is a secondary preference only.  It is
        # never present in QP1, so a posture preference cannot purchase task
        # slack there.
        proj_damping = sr_damping_lambda(sigma_min, self.cfg.sr_damping)
        M = (
            np.asarray(mass_matrix, dtype=float)
            if mass_matrix is not None
            else (
                self.kin.mass_matrix(q_geom)
                if self.cfg.use_mass_weighted_reg
                else None
            )
        )
        if M is not None and M.shape != (nv, nv):
            raise ValueError(f"mass_matrix must have shape {(nv, nv)}")
        qdot_nom = (
            (
                cpp_kernel.project_nullspace(
                    J_task,
                    secondary_qdot,
                    damping=proj_damping,
                    M=M,
                    use_dyn=False,
                )
                if bool(getattr(self.cfg, "use_cpp_kernel", True))
                else project_onto_task_nullspace(
                    J_task,
                    secondary_qdot,
                    damping=proj_damping,
                    sigma_min=sigma_min,
                    sr_cfg=self.cfg.sr_damping,
                    M=M,
                    use_dyn=False,
                )
            )
            if secondary_qdot is not None
            else np.zeros(nv, dtype=float)
        )
        if zero_secondary_rail and qdot_nom.size:
            qdot_nom[0] = 0.0
        if arm_qdot_pref is not None:
            pref = np.asarray(arm_qdot_pref, dtype=float).reshape(-1)
            n = min(pref.size, qdot_nom.size)
            qdot_nom[1:n] = pref[1:n]

        # Limit avoidance and the velocity box use the same measured geometry.
        w_reg = self._w_reg.copy()
        self.last_wln_scale = np.ones(self.kin.nv, dtype=float)
        if rail_locked and rail_lock_reg_scale > 1.0:
            w_reg[0] *= float(rail_lock_reg_scale)
        if (not rail_locked) and float(rail_reg_scale) > 1.0:
            w_reg[0] *= float(rail_reg_scale)
        w_task_mat = self._task_weight_matrix(
            J_task, dt, keep_task_weight=keep_task_weight
        )
        q_star_box = (
            self.q_star_signs
            if self.q_star_signs is not None
            else (self.q_star if self.q_star is not None else q_geom)
        )
        self.branch_barrier._update_dwell(q_geom, dt, q_star=q_star_box)
        rail_w_eff = float(rail_task_weight)
        pref_w = max(float(pref_slack_scale), 1.0e-6)
        n_task = N_TASK_SLACK
        n_pref = N_PREF_SLACK
        n_var = nv + n_task + n_pref

        # Shared hard constraints (P0) are built once and fed unchanged to
        # both levels.  Preference rows are added only to QP2 below.
        lo_box, hi_box = self.constraints.bounds(
            q_geom,
            dt,
            self.qdot_prev,
            q_meas=q_meas,
            q_cmd=q_prev,
            resync_err=resync_err,
            rail_locked=rail_locked,
            rail_lock_vel_eps_m_s=rail_lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin_m_s,
            qdot_prev2=self.qdot_prev2,
            j_max=self._j_max,
            box_dt=box_dt,
            box_h1=box_h1,
            box_h2=box_h2,
            rail_lead_exempt=(
                abs(float(q_prev[0]) - float(q_geom[0]))
                > float(np.asarray(resync_err, dtype=float).reshape(-1)[0])
                if np.size(np.asarray(resync_err))
                else False
            ),
        )
        lo_box, hi_box = self.branch_barrier.tighten_box(
            lo_box,
            hi_box,
            q_geom,
            q_star_box,
            self.constraints.lim.v_max,
            rail_open_travel=bool(rail_open_travel),
            q_lower=self.constraints.lim.q_lower,
            q_upper=self.constraints.lim.q_upper,
        )
        lo_box, hi_box = collapse_interval(
            lo_box,
            hi_box,
            qdot_prev=self.qdot_prev,
            a_max=self.constraints.lim.a_max,
            dt=dt,
        )
        self.last_lo_box = np.asarray(lo_box, dtype=float).copy()
        self.last_hi_box = np.asarray(hi_box, dtype=float).copy()
        if self.collision is not None and self.collision_cfg.enabled:
            cbf = build_cbf_rows(
                self.collision,
                self.kin,
                q_geom,
                self.collision_cfg,
                tracker=self._cbf_slots,
                kinematics_ready=bool(kinematics_ready),
            )
        else:
            cbf = CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))
            self._cbf_slots = CbfSlotTracker(max_pairs=self._max_cbf)
        if rail_exec is not None and cbf.jacobian.size:
            # CBF is a constraint on actual instantaneous motion just like the
            # protected TCP task.  Do not let a lagging rail command masquerade
            # as the rail velocity that is really changing collision distance.
            cbf_jac = np.asarray(cbf.jacobian, dtype=float).copy()
            cbf_lower = np.asarray(cbf.lower, dtype=float).copy()
            cbf_lower -= cbf_jac[:, 0] * rail_exec
            cbf_jac[:, 0] = 0.0
            cbf = CbfRows(
                jacobian=cbf_jac,
                lower=cbf_lower,
                slot_index=(
                    None
                    if cbf.slot_index is None
                    else np.asarray(cbf.slot_index, dtype=int).copy()
                ),
                names=tuple(cbf.names),
            )
        # Retain exactly the measured-rail affine CBF used by both QP levels
        # so the command publication path can certify any downstream rewrite.
        self.last_hard_cbf_jacobian = np.asarray(
            cbf.jacobian, dtype=float
        ).copy()
        self.last_hard_cbf_lower = np.asarray(cbf.lower, dtype=float).copy()
        self.last_task_jacobian = np.asarray(J_task, dtype=float).copy()
        self.last_cbf_min_dist = float("nan")
        self.last_cbf_pair = ""
        if self.collision is not None and self.collision_cfg.enabled:
            closest = self.collision.closest_pair()
            if closest is not None:
                self.last_cbf_min_dist = float(closest.distance)
                self.last_cbf_pair = f"{closest.name_a}:{closest.name_b}"
        _assemble = (
            cpp_kernel.build_wbc_inequalities
            if bool(getattr(self.cfg, "use_cpp_kernel", True))
            else build_wbc_inequalities
        )
        C_hard, lo, hi = _assemble(
            nv,
            n_task,
            lo_box,
            hi_box,
            cbf,
            self._max_cbf,
            n_pref_slack=n_pref,
            max_pref_rows=MAX_PREF_ROWS,
        )

        # QP1: only the protected task residual is optimized.  In particular,
        # qdot and preference-slack variables have exactly zero cost here:
        # even a tiny qdot regularizer would mathematically permit trading an
        # otherwise-zero Cartesian residual for less joint motion.  ProxQP's
        # own proximal terms handle the positive-semidefinite Hessian.
        if bool(getattr(self.cfg, "use_cpp_kernel", True)):
            H1, g1, A1 = cpp_kernel.setup_qp1(
                nv, n_task, n_pref, w_task_mat, J_task, use_native=True
            )
        else:
            H1 = np.zeros((n_var, n_var), dtype=float)
            H1[nv : nv + n_task, nv : nv + n_task] = w_task_mat
            g1 = np.zeros(n_var, dtype=float)
            A1 = np.zeros((n_task, n_var), dtype=float)
            A1[:, :nv] = J_task
            A1[:, nv : nv + n_task] = -np.eye(n_task)

        # ProxQP may return a point a few nanometres outside an inequality
        # while still satisfying ``eps_abs``.  QP2 then locks J*qdot from
        # that point and can incorrectly classify the hierarchy as primal
        # infeasible.  Solve QP1 against a conservatively inset hard set so
        # its achieved task is reproducibly feasible in QP2.  Exact pins
        # (lo==hi) are deliberately left untouched.
        feasibility_inset = max(2.0 * float(self.cfg.eps_abs), 1.0e-8)
        lo1 = np.asarray(lo, dtype=float).copy()
        hi1 = np.asarray(hi, dtype=float).copy()
        finite_lo = np.isfinite(lo1)
        finite_hi = np.isfinite(hi1)
        room = hi1 - lo1
        inset_both = finite_lo & finite_hi & (room > 2.0 * feasibility_inset)
        inset_lo_only = finite_lo & ~finite_hi
        inset_hi_only = ~finite_lo & finite_hi
        lo1[inset_both | inset_lo_only] += feasibility_inset
        hi1[inset_both | inset_hi_only] -= feasibility_inset

        x1 = self._solve_qp(
            self.backend,
            np.ascontiguousarray(H1),
            np.ascontiguousarray(g1),
            np.ascontiguousarray(A1),
            np.ascontiguousarray(b_task),
            np.ascontiguousarray(C_hard),
            np.ascontiguousarray(lo1),
            np.ascontiguousarray(hi1),
        )
        self.last_qp1_solve_ms = float(
            getattr(self.backend, "last_solve_ms", 0.0)
        )
        self.last_qp1_status = str(
            getattr(self.backend, "last_status", "failed" if x1 is None else "solved")
        )
        self.last_zero_slack_feasible = False
        if x1 is None:
            t_fallback = time.perf_counter()
            # Fail closed.  A scaled previous command is not certified against
            # this tick's acceleration/jerk/CBF set and must never leak out of
            # the low-level API as if it were a valid QP result.  Window A will
            # additionally invoke its rail+arm fault stop before publication.
            qdot = np.zeros_like(self.qdot_prev)
            residual = b_task - J_task @ qdot
            self.last_qp1_residual = residual.copy()
            self.last_qp1_residual_norm = float(np.linalg.norm(residual))
            self.last_qp2_residual = residual.copy()
            self.last_qp2_residual_norm = self.last_qp1_residual_norm
            self.last_qp2_status = "not_run"
            self.last_qp2_solve_ms = 0.0
            self.last_failed = True
            self.last_status = "failed"
            self.last_sns_scale = 1.0
            self.last_qp2_fallback = False
            dex_s = br_s = 0.0
            comfort = np.zeros(7, dtype=float)
            self.last_qdot_qp1 = np.asarray(qdot, dtype=float).copy()
            self.last_qp1_task_velocity = np.zeros(
                N_TASK_SLACK, dtype=float
            )
            self.last_qp1_hard_violation = float("nan")
            self.last_final_hard_violation = float("nan")
            self.last_final_task_lock_violation = float("nan")
            self.last_fallback_ms = (time.perf_counter() - t_fallback) * 1000.0
        else:
            qdot1 = np.asarray(x1[:nv], dtype=float).copy()
            if rail_exec is not None:
                # With measured-rail affine compensation the next rail command
                # is absent from both the protected task and the current CBF.
                # It is therefore a genuine QP1 null variable.  Prefer the
                # already-computed rail macro when one exists so a QP2
                # failure or limiter keep-QP1 still moves the carriage.
                # Otherwise brake to the hard-feasible standstill.
                if (
                    rail_task_vel_m_s is not None
                    and np.isfinite(float(rail_task_vel_m_s))
                    and not rail_locked
                    and rail_vel_pin_m_s is None
                ):
                    qdot1[0] = float(
                        np.clip(float(rail_task_vel_m_s), lo_box[0], hi_box[0])
                    )
                else:
                    qdot1[0] = float(np.clip(0.0, lo_box[0], hi_box[0]))
                x1 = np.asarray(x1, dtype=float).copy()
                x1[0] = qdot1[0]
            self.last_qdot_qp1 = qdot1.copy()
            hard_lo_violation = np.maximum(lo - C_hard @ x1, 0.0)
            hard_hi_violation = np.maximum(C_hard @ x1 - hi, 0.0)
            self.last_qp1_hard_violation = float(
                max(
                    np.max(hard_lo_violation, initial=0.0),
                    np.max(hard_hi_violation, initial=0.0),
                )
            )
            t1 = J_task @ qdot1
            self.last_qp1_task_velocity = np.asarray(t1, dtype=float).copy()
            residual1 = b_task - t1
            self.last_qp1_residual = residual1.copy()
            self.last_qp1_residual_norm = float(np.linalg.norm(residual1))

            # Build QP2's existing weighted secondary objective.  Its task
            # equality is augmented with w_task=0, locking QP1's achieved
            # task exactly while allowing all lower-priority preferences.
            if self.cfg.use_mass_weighted_reg and M is not None:
                m_diag = np.maximum(np.diag(M), self.cfg.mass_reg_floor)
                if self.cfg.mass_weight_exempt_rail:
                    m_diag[0] = 1.0
                tau = float(self.cfg.mass_reg_lpf_tau_s)
                if tau > 1.0e-9 and dt > 1.0e-9:
                    if self._m_diag_lpf is None:
                        self._m_diag_lpf = m_diag.copy()
                    else:
                        self._m_diag_lpf = first_order_lpf_vec(
                            self._m_diag_lpf, m_diag, dt, tau
                        )
                    m_diag = self._m_diag_lpf
                h_reg = w_reg * m_diag
            else:
                h_reg = w_reg
            slack_w = np.zeros(n_pref, dtype=float)
            slack_w[0] = float(self.cfg.sigma_setbased.slack_weight)
            slack_w[1] = (
                float(self.cfg.branch_barrier.slack_weight)
                * pref_w
                * float(self.branch_barrier.last_dwell_scale)
            )
            comfort_w = float(self.cfg.joint_comfort.slack_weight) * pref_w
            if n_pref > 2:
                slack_w[2:] = comfort_w
            rail_w_qp2 = 0.0
            rail_vel_qp2 = 0.0
            if (
                rail_task_vel_m_s is not None
                and rail_w_eff > 0.0
                and not rail_locked
                and rail_vel_pin_m_s is None
            ):
                rail_w_qp2 = float(rail_w_eff)
                rail_vel_qp2 = float(rail_task_vel_m_s)
            smooth_raw = np.asarray(
                getattr(self.cfg, "smoothness_weight", 0.0), dtype=float
            ).reshape(-1)
            if smooth_raw.size == 1:
                smooth = np.full(nv, float(smooth_raw[0]), dtype=float)
            elif smooth_raw.size == nv:
                smooth = smooth_raw.copy()
            else:
                raise ValueError(
                    f"smoothness_weight must be scalar or length {nv}, got {smooth_raw.size}"
                )
            smooth = np.maximum(smooth, 0.0)
            H2, g2 = cpp_kernel.setup_qp2_costs(
                nv,
                n_task,
                n_pref,
                h_reg,
                qdot_nom,
                slack_w,
                rail_w=rail_w_qp2,
                rail_vel=rail_vel_qp2,
                smooth=smooth,
                qdot_prev=self.qdot_prev,
                use_native=bool(getattr(self.cfg, "use_cpp_kernel", True)),
            )

            sigma_rows = self.sigma_setbased.build_row(self.kin, q_geom)
            q_star = (
                self.q_star_signs
                if self.q_star_signs is not None
                else (self.q_star if self.q_star is not None else q_geom)
            )
            # Soft branch rows never bound (max slack 2e-6).  Keep the hard
            # Faverjon damper in tighten_box.
            branch_rows = PrefInequalityRows(
                jacobian=np.zeros((0, nv)),
                slack_col=np.zeros(0, dtype=int),
                lower=np.zeros(0),
                active=False,
            )
            comfort_rows = self.joint_comfort.build_rows(
                q_geom, self.constraints.lim.q_lower, self.constraints.lim.q_upper
            )
            pref = self._merge_pref_rows(sigma_rows, branch_rows, comfort_rows)
            C2, lo2, hi2 = _assemble(
                nv,
                n_task,
                lo_box,
                hi_box,
                cbf,
                self._max_cbf,
                n_pref_slack=n_pref,
                max_pref_rows=MAX_PREF_ROWS,
                pref_jacobian=pref.jacobian,
                pref_slack_col=pref.slack_col,
                pref_lower=pref.lower,
            )
            A2 = np.zeros((n_task, n_var), dtype=float)
            A2[:, :nv] = J_task
            b2 = t1
            # Same-tick feasible hot start: qdot1 already satisfies all hard
            # constraints and exactly produces b2.  Fill only the one-sided
            # preference slacks needed by the added QP2 rows.  Seeding from
            # the previous tick here caused false PRIMAL_INFEASIBLE statuses
            # when the acceleration box moved between samples.
            x2_seed = np.zeros(n_var, dtype=float)
            x2_seed[:nv] = qdot1
            for k in range(n_pref):
                col = nv + n_task + k
                rows = C2[:, col] > 0.5
                finite_rows = rows & np.isfinite(lo2)
                if np.any(finite_rows):
                    base = C2[finite_rows, :nv] @ qdot1
                    need = float(np.max(lo2[finite_rows] - base, initial=0.0))
                    x2_seed[col] = max(need, 0.0) + feasibility_inset
            seed_c = C2 @ x2_seed
            self.last_qp2_seed_violation = float(
                max(
                    np.max(np.maximum(lo2 - seed_c, 0.0), initial=0.0),
                    np.max(np.maximum(seed_c - hi2, 0.0), initial=0.0),
                )
            )
            self.last_qp2_seed_equality = float(
                np.max(np.abs(A2 @ x2_seed - b2), initial=0.0)
            )
            qp2_exception_status = ""
            try:
                x2 = self._solve_qp(
                    self._backend_qp2,
                    np.ascontiguousarray(H2),
                    np.ascontiguousarray(g2),
                    np.ascontiguousarray(A2),
                    np.ascontiguousarray(b2),
                    np.ascontiguousarray(C2),
                    np.ascontiguousarray(lo2),
                    np.ascontiguousarray(hi2),
                    warm_start_x=np.ascontiguousarray(x2_seed),
                )
            except Exception as exc:
                # QP1 is already a valid protected solution.  A secondary
                # backend exception must not turn into a stale-velocity send.
                x2 = None
                qp2_exception_status = f"exception:{type(exc).__name__}"
            self.last_qp2_solve_ms = float(
                getattr(self._backend_qp2, "last_solve_ms", 0.0)
            )
            self.last_qp2_status = qp2_exception_status or str(
                getattr(
                    self._backend_qp2,
                    "last_status",
                    "failed" if x2 is None else "solved",
                )
            )
            if x2 is None:
                t_fallback = time.perf_counter()
                qdot = qdot1
                x = x1
                C_final, lo_final, hi_final = C_hard, lo, hi
                self.last_qp2_fallback = True
                self.last_fallback_ms = (
                    time.perf_counter() - t_fallback
                ) * 1000.0
            else:
                qdot = np.asarray(x2[:nv], dtype=float)
                x = x2
                C_final, lo_final, hi_final = C2, lo2, hi2
            final_c = C_final @ x
            self.last_final_hard_violation = float(
                max(
                    np.max(np.maximum(lo_final - final_c, 0.0), initial=0.0),
                    np.max(np.maximum(final_c - hi_final, 0.0), initial=0.0),
                )
            )
            self.last_final_task_lock_violation = float(
                np.max(
                    np.abs(J_task @ np.asarray(qdot, dtype=float) - t1),
                    initial=0.0,
                )
            )
            residual = b_task - J_task @ qdot
            self.last_qp2_residual = residual.copy()
            self.last_qp2_residual_norm = float(np.linalg.norm(residual))
            dex_s = float(max(0.0, x[nv + n_task]))
            br_s = float(max(0.0, x[nv + n_task + 1]))
            comfort = np.maximum(
                0.0, np.asarray(x[nv + n_task + 2 : nv + n_task + 9], dtype=float)
            )
            if comfort.size < 7:
                comfort = np.pad(comfort, (0, 7 - int(comfort.size)))
            comfort = comfort[:7]
            self.last_failed = False
            self.last_status = "solved"
            self.last_sns_scale = 1.0

        self.last_qp_total_ms = (time.perf_counter() - t_total) * 1000.0
        self.last_qp_overrun = bool(
            self.last_qp_total_ms > float(getattr(self.cfg, "max_solve_ms", 5.0))
        )
        # Preserve the legacy loop's ``core.backend.last_solve_ms`` telemetry,
        # but make it represent the complete two-level controller budget.
        self.backend.last_solve_ms = float(self.last_qp_total_ms)
        self._update_mirror_telemetry(
            J,
            rail_exec=rail_exec,
            h1=float(box_h1 if box_h1 is not None else (box_dt if box_dt is not None else dt)),
        )
        self.last_task_residual = np.asarray(residual, dtype=float).copy()
        # Legacy array retained for compatibility, but hard feasibility now
        # has its own scalar telemetry instead of aliasing Cartesian slack.
        self.last_hard_residual = np.full(
            N_TASK_SLACK, float(self.last_final_hard_violation), dtype=float
        )
        self.last_task_residual_norm = float(np.linalg.norm(residual))
        self.last_rail_cmd_contrib = J[:, 0] * float(qdot[0])
        self.last_arm_contrib = J[:, 1:] @ np.asarray(qdot[1:], dtype=float)
        # For a measured-rail tick, the actual contribution is measured rail
        # plus arm motion; without feedback the command is the best available
        # rail contribution and preserves legacy semantics.
        rail_actual = (
            rail_exec_contrib
            if rail_exec_vel_m_s is not None and np.isfinite(float(rail_exec_vel_m_s))
            else self.last_rail_cmd_contrib
        )
        self.last_task_achieved = rail_actual + self.last_arm_contrib
        if cbf.jacobian.size:
            cbf_value = np.asarray(cbf.jacobian, dtype=float) @ np.asarray(
                qdot, dtype=float
            )
            cbf_tol = max(2.0 * float(self.cfg.eps_abs), 1.0e-7)
            active_mask = np.abs(cbf_value - np.asarray(cbf.lower, dtype=float)) <= cbf_tol
            self.last_cbf_active_names = tuple(
                name
                for name, active in zip(tuple(cbf.names), active_mask)
                if bool(active)
            )
        else:
            self.last_cbf_active_names = ()
        self.last_dexterity_slack = dex_s
        self.last_branch_slack = br_s
        self.last_comfort_slack = np.asarray(comfort, dtype=float).reshape(7)
        self.sigma_setbased.last_slack = dex_s
        self.branch_barrier.last_slack = br_s
        # A failed QP1 has no certified command.  Preserve the applied-history
        # state until the outer safety stop/reset path explicitly synchronizes
        # it; do not seed a future jerk box from the diagnostic zero result.
        if not self.last_failed:
            self.qdot_prev = np.asarray(qdot, dtype=float).copy()
        q_next = q_prev + qdot * dt
        return IkStepResult(
            q_next=q_next,
            qdot=qdot,
            sigma_min=sigma_min,
            manip=self.kin.manipulability(J),
            slack_norm=float(self.last_task_residual_norm),
            n_cbf_active=int(cbf.jacobian.shape[0]),
            dexterity_slack=dex_s,
            branch_slack=br_s,
            sns_scale=1.0,
        )
```

### `rm75_control/control/joint_admittance_8dof/solver/constraint_mgr.py`

```python
"""Per-tick inequality constraints for the WBC QP inner loop.

Joint velocity box (velocity / position look-ahead / acceleration) plus optional
CBF self-collision rows stacked into ProxQP's l <= C x <= u form.
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import CbfRows
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


def collapse_interval(
    lo: np.ndarray,
    hi: np.ndarray,
    qdot_prev: np.ndarray | None = None,
    a_max: np.ndarray | None = None,
    dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse an empty velocity interval to a singleton feasible brake.

    When ``lo > hi``, set both bounds to one executable velocity: keep 0 if it
    lies strictly between the conflicting bounds, otherwise take the closest
    side that prefers braking toward the limit (matching command-lead
    behaviour).  Never raises.
    """
    lo = np.asarray(lo, dtype=float).copy()
    hi = np.asarray(hi, dtype=float).copy()
    crossed = lo > hi
    if not np.any(crossed):
        return lo, hi

    # hi < 0 < lo: the empty box straddles standstill — stop.
    keep_zero = crossed & (hi < 0.0) & (lo > 0.0)
    if qdot_prev is None:
        pick_lo = np.abs(lo) <= np.abs(hi)
        collapsed = np.where(pick_lo, lo, hi)
    else:
        prev = np.asarray(qdot_prev, dtype=float)
        # Moving positive: collapse onto lo (strongest brake of further +motion).
        # Moving negative: collapse onto hi.  Same rule as command_lead.
        collapsed = np.where(prev >= 0.0, lo, hi)
    collapsed = np.where(keep_zero, 0.0, collapsed)
    if (
        qdot_prev is not None
        and a_max is not None
        and dt is not None
        and float(dt) > 0.0
    ):
        prev = np.asarray(qdot_prev, dtype=float)
        a_step = np.asarray(a_max, dtype=float) * float(dt)
        collapsed = np.clip(collapsed, prev - a_step, prev + a_step)
    lo = np.where(crossed, collapsed, lo)
    hi = np.where(crossed, collapsed, hi)
    return lo, hi


def stopping_velocity(distance: np.ndarray, acceleration: np.ndarray, reaction_s: float) -> np.ndarray:
    """Maximum speed toward a limit while retaining delayed braking viability."""

    d = np.maximum(np.asarray(distance, dtype=float), 0.0)
    a = np.maximum(np.asarray(acceleration, dtype=float), 1.0e-9)
    reaction = np.maximum(np.asarray(reaction_s, dtype=float), 0.0)
    return np.sqrt(np.square(a * reaction) + 2.0 * a * d) - a * reaction


def wall_cap(
    x: float,
    *,
    lo: float,
    hi: float,
    a_max: float,
    reaction_s: float,
) -> tuple[float, float]:
    """One-sided speed limits toward each wall.  Never produces a restoring push."""

    v_out_lo = float(stopping_velocity(float(x) - float(lo), float(a_max), float(reaction_s)))
    v_out_hi = float(stopping_velocity(float(hi) - float(x), float(a_max), float(reaction_s)))
    return -v_out_lo, +v_out_hi


class VelocityBoxConstraints:
    def __init__(
        self,
        limits: SafetyLimits,
        *,
        damper_band_rad: float | np.ndarray = 0.15,
        rail_reaction_s: float = 0.06,
    ) -> None:
        self.lim = limits
        # Faverjon/Tournassoud velocity-damper influence zone before each
        # (margin-backed) joint limit; see bounds() below.  Scalar or per-joint
        # vector — units are per joint (rad for revolute, m for the prismatic
        # rail), so a scalar rad band must NOT be applied to the rail.
        self.damper_band_rad = np.asarray(damper_band_rad, dtype=float)
        # Extra look-ahead on the rail stopping envelope.  0 falls back to dt.
        self.rail_reaction_s = max(float(rail_reaction_s), 0.0)

    def bounds(
        self,
        q: np.ndarray,
        dt: float,
        qdot_prev: np.ndarray | None = None,
        *,
        q_meas: np.ndarray | None = None,
        q_cmd: np.ndarray | None = None,
        resync_err: float | np.ndarray = 0.0,
        rail_locked: bool = False,
        rail_lock_vel_eps_m_s: float = 0.0,
        rail_vel_pin_m_s: float | None = None,
        qdot_prev2: np.ndarray | None = None,
        j_max: np.ndarray | None = None,
        box_dt: float | None = None,
        box_h1: float | None = None,
        box_h2: float | None = None,
        rail_lead_exempt: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        lim = self.lim
        q = np.asarray(q, dtype=float)
        # ``dt`` is the nominal period the command is integrated with (the
        # CANFD stream assumes a fixed one).  ``box_h1`` / ``box_h2`` are the
        # two most recent wall periods; rate limits describe physical motion
        # so they belong on wall time.  ``box_dt`` remains a one-period
        # fallback for older callers.
        if box_h1 is not None:
            a_dt = float(box_h1)
        else:
            a_dt = float(dt if box_dt is None else box_dt)
        h2 = float(box_h2) if box_h2 is not None else float("nan")

        lo = -lim.v_max.copy()
        hi = lim.v_max.copy()

        m = lim.position_margin
        q_cmd_arr = None
        if q_cmd is not None:
            q_cmd_arr = np.asarray(q_cmd, dtype=float)
            if q_cmd_arr.shape != q.shape or not np.all(np.isfinite(q_cmd_arr)):
                raise ValueError("q_cmd must be finite and match q")
        # Rail damper / stop envelope use the state closer to the wall.
        # Command lead or servo overshoot of a few millimetres otherwise
        # eats a 10 mm band before qdot can fall.
        q_rail_hi = float(q[0])
        q_rail_lo = float(q[0])
        if q_cmd_arr is not None:
            q_rail_hi = max(q_rail_hi, float(q_cmd_arr[0]))
            q_rail_lo = min(q_rail_lo, float(q_cmd_arr[0]))

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
            # Rail linear taper uses the leading state so a few millimetres
            # of command lead / servo overshoot cannot skip the cone.
            m0 = float(np.broadcast_to(np.asarray(m, dtype=float).reshape(-1), q.shape)[0])
            b0 = float(np.broadcast_to(band, q.shape)[0])
            if b0 > 1e-9:
                d_hi[0] = float(
                    np.clip((float(lim.q_upper[0]) - m0 - q_rail_hi) / b0, 0.0, 1.0)
                )
                d_lo[0] = float(
                    np.clip((q_rail_lo - float(lim.q_lower[0]) - m0) / b0, 0.0, 1.0)
                )
                hi[0] = min(float(hi[0]), float(lim.v_max[0]) * float(d_hi[0]))
                lo[0] = max(float(lo[0]), -float(lim.v_max[0]) * float(d_lo[0]))

        m = np.broadcast_to(np.asarray(m, dtype=float), q.shape)
        a_max = None if lim.a_max is None else np.asarray(lim.a_max, dtype=float).copy()
        # Soft-limit braking envelope for the rail.  The linear damper above
        # is a leftover Faverjon cone; this is the actual stop-before-wall
        # bound.  Soft edges sit one damper-band inside the hard box.
        if a_max is not None and float(self.rail_reaction_s) > 0.0:
            b0 = float(np.broadcast_to(self.damper_band_rad, q.shape)[0])
            m0 = float(m[0])
            soft_lo = float(lim.q_lower[0]) + m0 + max(b0, 0.0)
            soft_hi = float(lim.q_upper[0]) - m0 - max(b0, 0.0)
            if soft_hi > soft_lo:
                lo_cap, hi_cap = wall_cap(
                    float(q[0]),
                    lo=soft_lo,
                    hi=soft_hi,
                    a_max=float(a_max[0]),
                    reaction_s=float(self.rail_reaction_s),
                )
                # Leading command state must also stop in time.
                lo_hi, hi_hi = wall_cap(
                    q_rail_hi,
                    lo=soft_lo,
                    hi=soft_hi,
                    a_max=float(a_max[0]),
                    reaction_s=float(self.rail_reaction_s),
                )
                lo_lo, hi_lo = wall_cap(
                    q_rail_lo,
                    lo=soft_lo,
                    hi=soft_hi,
                    a_max=float(a_max[0]),
                    reaction_s=float(self.rail_reaction_s),
                )
                hi[0] = min(float(hi[0]), hi_cap, hi_hi, hi_lo)
                lo[0] = max(float(lo[0]), lo_cap, lo_hi, lo_lo)

        p_lo = (lim.q_lower + m - q) / dt
        p_hi = (lim.q_upper - m - q) / dt
        # Rail hard box: past 5/780, one-tick look-ahead would require
        # returning by Δq/dt in a single period (reverse kick / chatter).
        # Kill into-wall only; leave stays open.
        rail_lo = float(lim.q_lower[0] + m[0])
        rail_hi = float(lim.q_upper[0] - m[0])
        if q[0] < rail_lo:
            p_lo[0] = min(float(p_lo[0]), 0.0)
        if q[0] > rail_hi:
            p_hi[0] = max(float(p_hi[0]), 0.0)
        lo = np.maximum(lo, p_lo)
        hi = np.minimum(hi, p_hi)
        lo, hi = collapse_interval(
            lo, hi, qdot_prev=qdot_prev, a_max=a_max, dt=dt
        )

        if a_max is not None and qdot_prev is not None:
            qdot_prev = np.asarray(qdot_prev, dtype=float)
            a = a_max * a_dt
            lo = np.maximum(lo, qdot_prev - a)
            hi = np.minimum(hi, qdot_prev + a)
            lo, hi = collapse_interval(
                lo, hi, qdot_prev=qdot_prev, a_max=a_max, dt=dt
            )

        # Third order.  Velocity and acceleration boxes still permit the
        # acceleration to flip sign every tick.  Bounding |a_k - a_{k-1}|
        # on unequal samples is
        #   qdot in qdot_prev + (h1/h2)(qdot_prev - qdot_prev2) +- j_max*h1^2
        # The equal-period form 2*qdot_prev - qdot_prev2 is recovered when
        # h1 == h2.  If h2 is unavailable (first tick / reset) the centre
        # stays at qdot_prev so only the acceleration box decides.
        if (
            j_max is not None
            and qdot_prev is not None
            and qdot_prev2 is not None
            and float(dt) > 0.0
        ):
            qdot_prev2 = np.asarray(qdot_prev2, dtype=float)
            if np.isfinite(h2) and h2 > 1.0e-9:
                centre = qdot_prev + (a_dt / h2) * (qdot_prev - qdot_prev2)
            else:
                centre = np.asarray(qdot_prev, dtype=float)
            span = np.asarray(j_max, dtype=float) * a_dt * a_dt
            lo = np.maximum(lo, centre - span)
            hi = np.minimum(hi, centre + span)
            lo, hi = collapse_interval(
                lo, hi, qdot_prev=qdot_prev, a_max=a_max, dt=dt
            )

        # Command lead is an anti-windup envelope, not a physical joint limit.
        # Start braking before |q_cmd-q_meas| reaches ``resync_err``.  If stale
        # tracking has already left too little distance for the acceleration
        # box, request the strongest acceleration-feasible braking velocity
        # instead of manufacturing an empty interval and stopping the robot.
        # ``resync_err`` is arm radians for joints 1..7 and metres for rail 0.
        if q_meas is not None:
            re = np.broadcast_to(
                np.asarray(resync_err, dtype=float), q.shape
            ).astype(float)
            active = re > 0.0
            if np.any(active):
                q_meas = np.asarray(q_meas, dtype=float)
                # Safety geometry is evaluated at measured q.  Command lead
                # is the one exception: compare the independently integrated
                # command state against the same measured snapshot.
                q_for_lead = q if q_cmd is None else np.asarray(q_cmd, dtype=float)
                lead = q_for_lead - q_meas
                # COUPLED rail velocity is authoritative; the 20 mm command
                # integrator lag is not a tracking error and must not freeze
                # the rail box.
                if rail_lead_exempt:
                    lead[0] = 0.0
                if a_max is None:
                    band = np.maximum(re * 0.5, 1.0e-6)
                    toward_hi = lim.v_max * np.clip((re - lead) / band, 0.0, 1.0)
                    toward_lo = -lim.v_max * np.clip((re + lead) / band, 0.0, 1.0)
                else:
                    reaction = np.full(q.shape, float(dt), dtype=float)
                    if float(self.rail_reaction_s) > 0.0:
                        reaction[0] = float(self.rail_reaction_s)
                    toward_hi = stopping_velocity(re - lead, a_max, reaction)
                    toward_lo = -stopping_velocity(re + lead, a_max, reaction)

                candidate_hi = np.minimum(hi, toward_hi)
                candidate_lo = np.maximum(lo, toward_lo)
                crossed = candidate_lo > candidate_hi
                # A positive lead must brake positive motion; a negative lead
                # must brake negative motion.  Collapse only the offending
                # side to the closest acceleration-feasible velocity.
                candidate_hi = np.where(
                    crossed & (lead >= 0.0), candidate_lo, candidate_hi
                )
                candidate_lo = np.where(
                    crossed & (lead < 0.0), candidate_hi, candidate_lo
                )
                hi = np.where(active, candidate_hi, hi)
                lo = np.where(active, candidate_lo, lo)
                lo, hi = collapse_interval(
                    lo, hi, qdot_prev=qdot_prev, a_max=a_max, dt=dt
                )

        if rail_vel_pin_m_s is not None:
            v = float(rail_vel_pin_m_s)
            if not np.isfinite(v):
                raise ValueError("rail_vel_pin_m_s must be finite")
            # Plan ownership is subordinate to the already assembled safety
            # box; it may pin the closest executable velocity, never replace
            # velocity/position/acceleration/command-lead bounds.
            v_safe = float(np.clip(v, lo[0], hi[0]))
            lo[0] = v_safe
            hi[0] = v_safe
        elif rail_locked:
            eps = max(float(rail_lock_vel_eps_m_s), 0.0)
            previous = 0.0 if qdot_prev is None else float(qdot_prev[0])
            rail_acceleration = (
                float(a_max[0]) if a_max is not None else float("inf")
            )
            if abs(previous) <= eps and float(lo[0]) <= 0.0 <= float(hi[0]):
                target = 0.0
            elif np.isfinite(rail_acceleration):
                target = np.sign(previous) * max(
                    abs(previous) - rail_acceleration * dt, 0.0
                )
                target = float(np.clip(target, lo[0], hi[0]))
            else:
                target = float(np.clip(0.0, lo[0], hi[0]))
            lo[0] = target
            hi[0] = target

        return lo, hi


def build_wbc_inequalities(
    nv: int,
    n_task_slack: int,
    lo_box: np.ndarray,
    hi_box: np.ndarray,
    cbf: CbfRows,
    max_cbf_rows: int,
    *,
    n_pref_slack: int = 0,
    max_pref_rows: int = 0,
    pref_jacobian: np.ndarray | None = None,
    pref_slack_col: np.ndarray | None = None,
    pref_lower: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack qdot box + CBF + optional preference inequalities + pref-slack >= 0.

    Decision vector: ``x = [qdot(nv); w_task(n_task_slack); s_pref(n_pref_slack)]``.
    Inactive CBF / pref slots are ``l=-inf, u=+inf``.
    """
    n_in = nv + max_cbf_rows + max_pref_rows + n_pref_slack
    n_var = nv + n_task_slack + n_pref_slack
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

    pref_base = nv + max_cbf_rows
    if (
        max_pref_rows > 0
        and pref_jacobian is not None
        and pref_lower is not None
        and pref_slack_col is not None
    ):
        n_pref = min(int(pref_jacobian.shape[0]), max_pref_rows)
        for k in range(n_pref):
            C[pref_base + k, :nv] = pref_jacobian[k]
            s_idx = int(pref_slack_col[k])
            if 0 <= s_idx < n_pref_slack:
                C[pref_base + k, nv + n_task_slack + s_idx] = 1.0
            l[pref_base + k] = float(pref_lower[k])

    # Pref slacks are one-sided: s >= 0.
    slack_base = pref_base + max_pref_rows
    for k in range(n_pref_slack):
        C[slack_base + k, nv + n_task_slack + k] = 1.0
        l[slack_base + k] = 0.0
    return C, l, u


__all__ = [
    "VelocityBoxConstraints",
    "build_wbc_inequalities",
    "collapse_interval",
    "stopping_velocity",
]
```

### `rm75_control/control/joint_admittance_8dof/solver/cbf_constraints.py`

```python
"""Control Barrier Function rows for self-collision avoidance (Faverjon / Khazoom)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from rm75_control.control.joint_admittance_8dof.collision_model import (
    CollisionConfig,
    CollisionModel,
    CollisionPairInfo,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


@dataclass
class CbfRows:
    jacobian: np.ndarray   # (n_rows, nv) — packed active or fixed slot layout
    lower: np.ndarray      # (n_rows,)  J_col qdot >= lower
    slot_index: np.ndarray | None = None  # (n_rows,) QP row offset within CBF block
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrameJacobian:
    """Base-aligned frame Jacobian together with its point of application."""

    jacobian: np.ndarray  # (6, nv), [v_origin; omega]
    origin: np.ndarray  # (3,), expressed in the base frame


@dataclass
class CbfSlotTracker:
    """Sticky pair→row slot assignment with enter/exit hysteresis.

    Keeps the same ProxQP inequality row for a given (geom_a, geom_b) across
    ticks so warm-start multipliers do not thrash when distance rank order
    changes.  A pair leaves its slot only after ``distance > d_activate + hyst``.
    """

    max_pairs: int
    hyst_m: float = 0.01
    _keys: list[tuple[int, int] | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._keys:
            self._keys = [None] * int(self.max_pairs)

    def update(
        self,
        pairs: list[CollisionPairInfo],
        d_activate: float,
    ) -> list[CollisionPairInfo | None]:
        """Return length-``max_pairs`` list of pair-or-None per sticky slot."""
        d_keep = float(d_activate) + float(self.hyst_m)
        by_key = {(int(p.geom_a), int(p.geom_b)): p for p in pairs}

        # Drop slots that left the keep band.
        for i, key in enumerate(self._keys):
            if key is None:
                continue
            p = by_key.get(key)
            if p is None or float(p.distance) > d_keep:
                self._keys[i] = None

        occupied = {k for k in self._keys if k is not None}

        # Prefer currently active pairs (distance <= d_activate) for free slots.
        candidates = sorted(
            (p for p in pairs if float(p.distance) <= float(d_activate)),
            key=lambda p: float(p.distance),
        )
        for p in candidates:
            key = (int(p.geom_a), int(p.geom_b))
            if key in occupied:
                continue
            try:
                free = self._keys.index(None)
            except ValueError:
                break
            self._keys[free] = key
            occupied.add(key)

        out: list[CollisionPairInfo | None] = []
        for key in self._keys:
            if key is None:
                out.append(None)
            else:
                out.append(by_key.get(key))  # may be None if momentarily missing
        return out


def _frame_linear_jacobians(
    model: pin.Model,
    data: pin.Data,
    geom_model: pin.GeometryModel,
    *,
    kinematics_ready: bool = False,
) -> dict[int, FrameJacobian]:
    if not kinematics_ready:
        pin.computeJointJacobians(model, data)
        pin.updateFramePlacements(model, data)
    out: dict[int, FrameJacobian] = {}
    for go in geom_model.geometryObjects:
        fid = int(go.parentFrame)
        if fid not in out:
            J6 = pin.getFrameJacobian(model, data, fid, pin.LOCAL_WORLD_ALIGNED)
            out[fid] = FrameJacobian(
                jacobian=np.asarray(J6, dtype=float).copy(),
                origin=np.asarray(data.oMf[fid].translation, dtype=float).copy(),
            )
    return out


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float
    )


def _point_linear_jacobian(
    frame_jac: FrameJacobian | np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    """Linear Jacobian at a collision witness point, not the frame origin."""

    if isinstance(frame_jac, FrameJacobian):
        J6 = np.asarray(frame_jac.jacobian, dtype=float)
        r = np.asarray(point, dtype=float).reshape(3) - frame_jac.origin
        # v_point = v_origin + omega x r = v_origin - skew(r) omega.
        return J6[:3, :] - _skew(r) @ J6[3:, :]
    J = np.asarray(frame_jac, dtype=float)
    if J.ndim != 2 or J.shape[0] < 3:
        raise ValueError("frame Jacobian must have at least three rows")
    return J[:3, :]


def cbf_v_safe(
    distance: float,
    cfg: CollisionConfig,
) -> float:
    """Closing-speed floor: J_col qdot >= v_safe.  Leave (positive) is free."""
    return float(-float(cfg.gamma) * (float(distance) - float(cfg.d_safe)))


def collision_jacobian(
    frame_jacs: dict[int, FrameJacobian | np.ndarray],
    geom_model: pin.GeometryModel,
    pair: CollisionPairInfo,
) -> np.ndarray:
    go_a = geom_model.geometryObjects[pair.geom_a]
    go_b = geom_model.geometryObjects[pair.geom_b]
    J_a = _point_linear_jacobian(
        frame_jacs[int(go_a.parentFrame)], pair.point_a
    )
    J_b = _point_linear_jacobian(
        frame_jacs[int(go_b.parentFrame)], pair.point_b
    )
    return pair.normal @ (J_a - J_b)


def build_cbf_rows(
    collision: CollisionModel,
    kin: RobotKinematics,
    q_rad: np.ndarray,
    cfg: CollisionConfig,
    *,
    tracker: CbfSlotTracker | None = None,
    kinematics_ready: bool = False,
) -> CbfRows:
    """Build CBF inequality rows J_col qdot >= v_safe with optional sticky slots."""
    nv = kin.nv
    if not cfg.enabled:
        return CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))

    # Caller has computed J(q_meas) immediately before CBF rows and
    # explicitly proves that fact with ``kinematics_ready``.  Direct callers
    # default to CollisionModel's self-contained kinematics path.
    snapshot_ready = bool(kinematics_ready)
    d_keep = float(cfg.d_activate) + (float(tracker.hyst_m) if tracker else 0.0)
    collision.update(
        q_rad,
        kinematic_data=kin.data if snapshot_ready else None,
        kinematics_ready=snapshot_ready,
        distance_threshold=d_keep,
    )
    jacobian_data = kin.data if snapshot_ready else collision._kin_data  # noqa: SLF001
    raw_pairs = collision.active_pairs(d_keep)

    if tracker is not None:
        slotted = tracker.update(raw_pairs, cfg.d_activate)
        frame_jacs = _frame_linear_jacobians(
            collision.model,
            jacobian_data,
            collision.geom_model,
            kinematics_ready=snapshot_ready,
        )
        rows = []
        lowers = []
        slots = []
        names = []
        for i, pair in enumerate(slotted):
            if pair is None:
                continue
            J_col = collision_jacobian(frame_jacs, collision.geom_model, pair)
            v_safe = cbf_v_safe(pair.distance, cfg)
            rows.append(J_col)
            lowers.append(v_safe)
            slots.append(i)
            names.append(f"self_collision:{pair.name_a}:{pair.name_b}")
        if not rows:
            return CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))
        return CbfRows(
            jacobian=np.vstack(rows),
            lower=np.asarray(lowers, dtype=float),
            slot_index=np.asarray(slots, dtype=int),
            names=tuple(names),
        )

    pairs = raw_pairs[: cfg.max_pairs]
    if not pairs:
        return CbfRows(jacobian=np.zeros((0, nv)), lower=np.zeros(0))

    frame_jacs = _frame_linear_jacobians(
        collision.model,
        jacobian_data,
        collision.geom_model,
        kinematics_ready=snapshot_ready,
    )
    rows = []
    lowers = []
    names = []
    for pair in pairs:
        J_col = collision_jacobian(frame_jacs, collision.geom_model, pair)
        v_safe = cbf_v_safe(pair.distance, cfg)
        rows.append(J_col)
        lowers.append(v_safe)
        names.append(f"self_collision:{pair.name_a}:{pair.name_b}")

    return CbfRows(
        jacobian=np.vstack(rows),
        lower=np.asarray(lowers, dtype=float),
        names=tuple(names),
    )
```

### `rm75_control/control/joint_admittance_8dof/collision_model.py`

```python
"""Pinocchio + HPP-FCL self-collision distance queries for CBF constraints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

DEFAULT_COLLISION_URDF = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robots"
    / "rm75_6f_8dof"
    / "RM75-6F-8dof.collision.urdf"
)
DEFAULT_PAIR_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robots"
    / "rm75_6f_8dof"
    / "collision_pairs.yaml"
)


@dataclass
class CollisionPairInfo:
    pair_index: int
    geom_a: int
    geom_b: int
    name_a: str
    name_b: str
    distance: float
    normal: np.ndarray          # unit vector from B toward A (base frame)
    point_a: np.ndarray
    point_b: np.ndarray


@dataclass(frozen=True)
class BoundingSphere:
    """Conservative sphere enclosing one collision geometry in local coords."""

    center: np.ndarray
    radius: float


@dataclass
class CollisionConfig:
    enabled: bool = True
    d_safe: float = 0.03
    d_activate: float = 0.08
    gamma: float = 5.0
    max_pairs: int = 8
    collision_urdf: Path = DEFAULT_COLLISION_URDF
    pair_config: Path = DEFAULT_PAIR_CONFIG


def _geom_name_map(geom_model: pin.GeometryModel) -> dict[str, int]:
    return {go.name: i for i, go in enumerate(geom_model.geometryObjects)}


def _disable_pairs(geom_model: pin.GeometryModel, disabled: list[list[str]]) -> None:
    name_to_id = _geom_name_map(geom_model)
    for pair in disabled:
        if len(pair) != 2:
            continue
        a, b = pair[0], pair[1]
        if a not in name_to_id or b not in name_to_id:
            continue
        cp = pin.CollisionPair(name_to_id[a], name_to_id[b])
        if geom_model.existCollisionPair(cp):
            geom_model.removeCollisionPair(cp)


class CollisionModel:
    """Self-collision geometry loaded from a collision-capable URDF."""

    def __init__(
        self,
        kin_model: pin.Model,
        *,
        collision_urdf: str | Path | None = None,
        pair_config: str | Path | None = None,
    ) -> None:
        self.collision_urdf = Path(collision_urdf or DEFAULT_COLLISION_URDF)
        if not self.collision_urdf.exists():
            raise FileNotFoundError(f"collision URDF not found: {self.collision_urdf}")
        mesh_dir = self.collision_urdf.parent
        self.model = kin_model
        self.geom_model = pin.buildGeomFromUrdf(
            self.model,
            str(self.collision_urdf),
            pin.COLLISION,
            package_dirs=[str(mesh_dir)],
        )
        self.geom_model.addAllCollisionPairs()
        pair_path = Path(pair_config or DEFAULT_PAIR_CONFIG)
        if pair_path.exists():
            raw = yaml.safe_load(pair_path.read_text()) or {}
            _disable_pairs(self.geom_model, raw.get("disabled_pairs", []))
        self.geom_data = self.geom_model.createData()
        self._kin_data = self.model.createData()
        self._q = np.zeros(self.model.nq, dtype=float)
        # The collision URDF currently contains triangle meshes.  Keeping a
        # local sphere for every geometry makes the broadphase independent of
        # HPP-FCL internals and, since it encloses every mesh vertex, strictly
        # conservative for triangle meshes as well.
        self._bounding_spheres = tuple(
            self._make_bounding_sphere(
                go.geometry,
                mesh_scale=getattr(go, "meshScale", np.ones(3)),
            )
            for go in self.geom_model.geometryObjects
        )
        n_geoms = len(self.geom_model.geometryObjects)
        n_pairs = len(self.geom_model.collisionPairs)
        self._sphere_centers = np.array(
            [s.center for s in self._bounding_spheres], dtype=float
        ).reshape(n_geoms, 3)
        self._sphere_radii = np.array(
            [s.radius for s in self._bounding_spheres], dtype=float
        )
        self._pair_first = np.array(
            [int(cp.first) for cp in self.geom_model.collisionPairs], dtype=np.intp
        )
        self._pair_second = np.array(
            [int(cp.second) for cp in self.geom_model.collisionPairs], dtype=np.intp
        )
        self._geom_translation = np.zeros((n_geoms, 3), dtype=float)
        self._geom_rotation = np.zeros((n_geoms, 3, 3), dtype=float)
        self._world_centers = np.zeros((n_geoms, 3), dtype=float)
        self._exact_pair_indices: tuple[int, ...] = ()
        self._last_lower_bounds = np.full(n_pairs, np.inf, dtype=float)
        self._last_distance_threshold: float | None = None
        self._last_skipped_pair_indices: tuple[int, ...] = ()

    @staticmethod
    def _make_bounding_sphere(
        geometry: object,
        *,
        mesh_scale: np.ndarray,
    ) -> BoundingSphere:
        """Build a conservative local sphere from the mesh vertices.

        ``coal`` exposes ``vertices`` as a method in some versions and as an
        array in others.  For an unsupported primitive, use an infinite
        radius; that preserves the old full narrow-phase behaviour rather
        than risking a false negative in collision checking.
        """

        try:
            scale = np.asarray(mesh_scale, dtype=float).reshape(-1)
            # Pinocchio/HPP-FCL versions differ on whether meshScale has
            # already been baked into ``vertices``.  A non-unit scale is
            # therefore ambiguous; disable broadphase for that geometry
            # instead of risking a sphere that is too small.
            if (
                scale.size < 3
                or not np.all(np.isfinite(scale[:3]))
                or not np.allclose(scale[:3], np.ones(3), atol=1.0e-12, rtol=0.0)
            ):
                raise ValueError("non-unit mesh scale is not safely inferable")
            vertices = getattr(geometry, "vertices")
            if callable(vertices):
                vertices = vertices()
            vertices = np.asarray(vertices, dtype=float)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
                raise ValueError("geometry has no Nx3 vertices")
            if not np.all(np.isfinite(vertices)):
                raise ValueError("geometry vertices are non-finite")
            center = np.mean(vertices, axis=0)
            radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
            if not np.isfinite(radius):
                raise ValueError("geometry radius is non-finite")
            return BoundingSphere(center=center, radius=radius)
        except Exception:
            return BoundingSphere(
                center=np.zeros(3, dtype=float), radius=float("inf")
            )

    @property
    def bounding_spheres(self) -> tuple[BoundingSphere, ...]:
        """Precomputed local-space spheres, exposed for diagnostics/tests."""

        return self._bounding_spheres

    @property
    def exact_pair_indices(self) -> tuple[int, ...]:
        """Pair indices whose narrow-phase distance was evaluated this tick."""

        return self._exact_pair_indices

    @property
    def broadphase_lower_bounds(self) -> np.ndarray:
        """Latest sphere lower bound for every collision pair."""

        return self._last_lower_bounds.copy()

    @property
    def skipped_pair_indices(self) -> tuple[int, ...]:
        """Pair indices skipped by the latest broadphase update."""

        return self._last_skipped_pair_indices

    @property
    def distance_query_count(self) -> int:
        """Number of exact narrow-phase pair queries in the latest update."""

        return len(self._exact_pair_indices)

    def _refresh_world_centers(self) -> None:
        n_geoms = self._geom_translation.shape[0]
        oMg = self.geom_data.oMg
        for i in range(n_geoms):
            T = oMg[i]
            self._geom_translation[i] = T.translation
            self._geom_rotation[i] = T.rotation
        np.einsum(
            "nij,nj->ni",
            self._geom_rotation,
            self._sphere_centers,
            out=self._world_centers,
        )
        self._world_centers += self._geom_translation

    def _pair_lower_bounds(self) -> np.ndarray:
        """Sphere-sphere separation for every collision pair.

        The sphere-sphere separation is a lower bound on the mesh distance.
        Do not clamp to zero: a negative value is still a valid conservative
        lower bound for overlapping spheres.  Infinite-radius spheres (unsafe
        mesh scale) force ``-inf`` so those pairs always take the narrow phase.
        """

        self._refresh_world_centers()
        ga = self._pair_first
        gb = self._pair_second
        delta = self._world_centers[ga] - self._world_centers[gb]
        dist = np.linalg.norm(delta, axis=1)
        dist -= self._sphere_radii[ga]
        dist -= self._sphere_radii[gb]
        bad = ~(
            np.isfinite(self._sphere_radii[ga]) & np.isfinite(self._sphere_radii[gb])
        )
        if np.any(bad):
            dist = dist.copy()
            dist[bad] = -np.inf
        return dist

    def _pair_lower_bound(self, pair_index: int) -> float:
        """Scalar wrapper kept for tests; uses the vectorized path."""

        return float(self._pair_lower_bounds()[int(pair_index)])

    def update(
        self,
        q_rad: np.ndarray,
        *,
        kinematic_data: pin.Data | None = None,
        kinematics_ready: bool = False,
        distance_threshold: float | None = None,
    ) -> None:
        """Update witness distances, optionally reusing this tick's FK data.

        ``RobotKinematics.jacobian`` has already computed joint Jacobians and
        frame placements for the immutable measured-state snapshot used by
        QPIK.  Reusing that data avoids a second forward-kinematics pass while
        preserving the exact same collision geometry and distance queries.
        Standalone callers retain the original self-contained behaviour.
        """

        self._q = np.asarray(q_rad, dtype=float)
        data = self._kin_data if kinematic_data is None else kinematic_data
        if not kinematics_ready:
            pin.forwardKinematics(self.model, data, self._q)
        pin.updateGeometryPlacements(
            self.model, data, self.geom_model, self.geom_data
        )
        # Placements are already current; the five-argument overload would
        # recompute them a second time.  A missing threshold preserves the
        # standalone/full narrow-phase API.  CBF callers provide the current
        # activation+hysteresis band and use the conservative sphere test.
        n_pairs = len(self.geom_model.collisionPairs)
        self._exact_pair_indices = ()
        self._last_distance_threshold = (
            None if distance_threshold is None else float(distance_threshold)
        )
        self._last_lower_bounds = np.full(n_pairs, np.inf, dtype=float)
        self._last_skipped_pair_indices = ()
        if distance_threshold is None:
            pin.computeDistances(self.geom_model, self.geom_data)
            self._exact_pair_indices = tuple(range(n_pairs))
            return

        threshold = float(distance_threshold)
        if not np.isfinite(threshold):
            # An infinite threshold is equivalent to the old full query and
            # avoids treating NaNs as an opportunity to skip safety checks.
            pin.computeDistances(self.geom_model, self.geom_data)
            self._exact_pair_indices = tuple(range(n_pairs))
            return

        self._last_lower_bounds = self._pair_lower_bounds()

        # Every pair whose true distance is <= threshold has a sphere lower
        # bound <= threshold, so this set cannot omit an active CBF pair.
        selected = np.flatnonzero(self._last_lower_bounds <= threshold).tolist()
        # Keep closest-pair telemetry meaningful even when every sphere is
        # outside the activation band.  The minimum lower-bound pair is the
        # only extra narrow-phase query needed for that telemetry.
        if n_pairs and not selected:
            selected = [int(np.argmin(self._last_lower_bounds))]
        selected = tuple(sorted(set(int(i) for i in selected)))
        for i in selected:
            pin.computeDistance(self.geom_model, self.geom_data, int(i))
        self._exact_pair_indices = selected
        selected_set = set(selected)
        self._last_skipped_pair_indices = tuple(
            i for i in range(n_pairs) if i not in selected_set
        )

    def pair_info(self, pair_index: int) -> CollisionPairInfo | None:
        if int(pair_index) not in self._exact_pair_indices:
            return None
        dr = self.geom_data.distanceResults[pair_index]
        d = float(dr.min_distance)
        if not np.isfinite(d):
            return None
        pa = np.asarray(dr.getNearestPoint1(), dtype=float)
        pb = np.asarray(dr.getNearestPoint2(), dtype=float)
        cp = self.geom_model.collisionPairs[pair_index]
        ga, gb = int(cp.first), int(cp.second)
        na = pa - pb
        n_norm = float(np.linalg.norm(na))
        if n_norm < 1e-9:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = na / n_norm
        go_a = self.geom_model.geometryObjects[ga]
        go_b = self.geom_model.geometryObjects[gb]
        return CollisionPairInfo(
            pair_index=pair_index,
            geom_a=ga,
            geom_b=gb,
            name_a=go_a.name,
            name_b=go_b.name,
            distance=d,
            normal=normal,
            point_a=pa,
            point_b=pb,
        )

    def all_pairs(self) -> list[CollisionPairInfo]:
        out: list[CollisionPairInfo] = []
        for i in self._exact_pair_indices:
            info = self.pair_info(i)
            if info is not None:
                out.append(info)
        return out

    def active_pairs(self, d_activate: float) -> list[CollisionPairInfo]:
        # Reading witness points/normals allocates several arrays per pair.
        # First filter on HPP-FCL's scalar distance result, then materialise
        # full information only for pairs that can enter/leave a CBF slot.
        threshold = float(d_activate)
        indices = [
            i
            for i in self._exact_pair_indices
            for result in (self.geom_data.distanceResults[i],)
            if np.isfinite(float(result.min_distance))
            and float(result.min_distance) < threshold
        ]
        pairs = [self.pair_info(i) for i in indices]
        pairs = [p for p in pairs if p is not None]
        pairs.sort(key=lambda p: p.distance)
        return pairs

    def min_distance(self) -> float:
        distances = [
            float(self.geom_data.distanceResults[i].min_distance)
            for i in self._exact_pair_indices
            for result in (self.geom_data.distanceResults[i],)
            if np.isfinite(float(result.min_distance))
        ]
        if not distances:
            return float("inf")
        return min(distances)

    def closest_pair(self) -> CollisionPairInfo | None:
        """Nearest pair after ``update``; not the CBF slot occupancy count."""
        best_i = -1
        best_d = float("inf")
        for i in self._exact_pair_indices:
            result = self.geom_data.distanceResults[i]
            d = float(result.min_distance)
            if np.isfinite(d) and d < best_d:
                best_d = d
                best_i = i
        if best_i < 0:
            return None
        return self.pair_info(best_i)
```

### `rm75_control/control/joint_admittance_8dof/config.py`

```python
"""YAML loader for the 8-DOF slack-QP inner loop (Escande WBC + rail extension)."""

from __future__ import annotations

import math

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.ik_types import SrDampingConfig
from rm75_control.control.joint_admittance_8dof.loop import (
    CartesianTrackGains,
    JointIkConfig,
)
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import (
    QpConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import ArmAngleTaskConfig
from rm75_control.control.joint_admittance_8dof.tasks.manipulability_task import (
    ManipulabilityTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_allocator import (
    RailAllocatorConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_lock import RailLockConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode
from rm75_control.control.joint_admittance_8dof.tasks.ird_adapter import IrdConfig
from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import PsiRetargetConfig
from rm75_control.control.joint_admittance_8dof.saturation_latch import SaturationConfig


def _mapping(value, *, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _reject_unknown(section: dict, allowed: set[str], *, name: str) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {name} configuration keys: " + ", ".join(unknown))


def _finite_float(value, *, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return out


def _arr(value, default) -> np.ndarray:
    return np.asarray(value if value is not None else default, dtype=float)


def _finite_array(value, *, name: str, ndim: int | None = None) -> np.ndarray:
    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {out.ndim}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _resolve_rail_mode(r: dict) -> tuple[RailMode, LockedStyle]:
    mode_str = str(r.get("mode", "coupled")).lower()
    raw_style = r.get("locked_style", "hold")
    if mode_str == "coupled":
        return RailMode.COUPLED, LockedStyle.HOLD
    if mode_str == "locked":
        style = LockedStyle(str(raw_style).lower()) if raw_style else LockedStyle.HOLD
        return RailMode.LOCKED, style
    raise ValueError(f"unknown rail.mode: {r.get('mode')!r}")


_RETIRED_QPIK = {
    "protected_task",
    "scalable_tasks",
    "task_profile",
    "compatibility",
    "reference_governor",
    "accepted_reference_governor",
    "governor",
    "psi_lift",
}
_RETIRED_SOLVER = {
    "max_rows",
    "max_constraint_rows",
    "max_p0_rows",
    "max_scalable_groups",
    "max_groups",
    "protected_tolerance",
    "regularization",
    "previous_velocity_weight",
    "scalable_weight",
    "posture_weight",
    "posture_regularization",
    "margin_weight",
    "margin_weight_gain",
    "psi_weight",
    "psi_k",
    "psi_lift_weight_scale",
    "psi_err_boost_rad",
    "psi_err_weight_scale",
    "comfort_k_g",
    "comfort_qdot_max",
    "row_scale_floor",
    "qp1",
    "qp2",
    "qp3",
    "retry",
    "regularization_retry",
    "fallback_qp",
    "p0_fallback",
    "health_to_alpha",
    "sigma_escape_enter",
    "sigma_escape_exit",
    "rail_escape_v_min_m_s",
    "rail_escape_v_max_m_s",
}
_LEFTOVER_28VAR = {
    "dexterity",
    "working_set",
    "whole_body",
    "health",
    "indices",
    "task_velocity_scales",
}


def _reject_retired_qpik(qpik: dict) -> None:
    solver = _mapping(qpik.get("solver"), name="qpik.solver")
    retired = sorted((set(qpik) & _RETIRED_QPIK) | (set(solver) & _RETIRED_SOLVER))
    if retired:
        raise ValueError(
            "retired multi-level QPIK configuration keys: " + ", ".join(retired)
        )
    leftover = sorted(set(qpik) & _LEFTOVER_28VAR)
    if leftover:
        raise ValueError(
            "retired 28-var QPIK keys (use inner.qp / inner.nullspace / "
            "inner.rail_extension): " + ", ".join(leftover)
        )
    if "solver" in qpik:
        raise ValueError("qpik.solver is retired; use inner.qp")


def _parse_collision(raw: dict, *, name: str) -> CollisionConfig:
    section = _mapping(raw, name=name)
    _reject_unknown(
        section,
        {"enabled", "d_safe", "d_activate", "gamma", "max_pairs"},
        name=name,
    )
    collision = CollisionConfig(
        enabled=bool(section.get("enabled", True)),
        d_safe=_finite_float(section.get("d_safe", 0.01), name=f"{name}.d_safe"),
        d_activate=_finite_float(
            section.get("d_activate", 0.04), name=f"{name}.d_activate"
        ),
        gamma=_finite_float(section.get("gamma", 5.0), name=f"{name}.gamma"),
        max_pairs=int(section.get("max_pairs", 8)),
    )
    if not 0.0 <= collision.d_safe < collision.d_activate:
        raise ValueError("collision distances must satisfy 0 <= d_safe < d_activate")
    if collision.gamma <= 0.0 or collision.max_pairs <= 0:
        raise ValueError("collision gamma/max_pairs must be positive")
    return collision


def _parse_qp(inner: dict, collision: CollisionConfig, euler_order: str) -> QpConfig:
    c = _mapping(inner.get("qp"), name="inner.qp")
    _reject_unknown(
        c,
        {
            "task_weight", "reg", "backend", "eps_abs", "max_iter", "max_iter_cap",
            "max_solve_ms", "fail_qdot_decay", "twist_sigma_floor", "warn_on_fail",
            "sr_damping", "task_weight_min_frac", "task_weight_lpf_tau_s",
            "aniso_task_damping",
            "use_mass_weighted_reg", "mass_reg_floor", "mass_weight_exempt_rail",
            "mass_reg_lpf_tau_s",
            "limit_damper_band_rad", "limit_damper_band_rail_m",
            "sigma_setbased", "branch_barrier", "joint_comfort",
            "smoothness_weight", "near_arm_margin_rad",
            "j_max_arm_rad_s3", "j_max_rail_m_s3",
            "use_cpp_kernel",
        },
        name="inner.qp",
    )
    backend = str(c.get("backend", "proxqp")).lower()
    if backend not in {"proxqp", "osqp", "scipy"}:
        raise ValueError(
            "inner.qp.backend must be 'proxqp', 'osqp', or 'scipy' "
            f"(got {backend!r})"
        )
    if backend == "scipy":
        # Slack QP has no scipy path; ProxQP falls back to OSQP at runtime.
        backend = "proxqp"
    sr = _mapping(c.get("sr_damping"), name="inner.qp.sr_damping")
    _reject_unknown(
        sr, {"lam0", "sigma_ref", "sigma_floor"}, name="inner.qp.sr_damping"
    )
    from rm75_control.control.joint_admittance_8dof.solver.branch_barrier import (
        BranchBarrierConfig,
    )
    from rm75_control.control.joint_admittance_8dof.solver.joint_comfort import (
        JointComfortConfig,
    )
    from rm75_control.control.joint_admittance_8dof.solver.sigma_setbased import (
        SigmaSetBasedConfig,
    )

    ss = _mapping(c.get("sigma_setbased"), name="inner.qp.sigma_setbased")
    _reject_unknown(
        ss,
        {
            "enabled", "activate", "safe", "exit", "gamma", "slack_weight",
            "grad_eps", "grad_period_ticks",
        },
        name="inner.qp.sigma_setbased",
    )
    bb = _mapping(c.get("branch_barrier"), name="inner.qp.branch_barrier")
    _reject_unknown(
        bb,
        {
            "enabled", "activate_rad", "box_activate_rad", "eps_rad", "gamma",
            "slack_weight", "target_eps_rad", "dwell_free_s", "dwell_ramp_s",
            "dwell_scale_max", "j4_limit_eps_rad", "j4_limit_activate_rad",
            "j1_overfold_abs_rad", "j1_overfold_activate_rad", "j1_overfold_eps_rad",
        },
        name="inner.qp.branch_barrier",
    )
    jc = _mapping(c.get("joint_comfort"), name="inner.qp.joint_comfort")
    _reject_unknown(
        jc,
        {
            "enabled", "m_comfort_deg", "activate_deg", "gamma", "slack_weight",
        },
        name="inner.qp.joint_comfort",
    )
    smooth_raw = c.get("smoothness_weight", 0.15)
    if isinstance(smooth_raw, (list, tuple, np.ndarray)):
        smoothness_weight = _finite_array(
            smooth_raw,
            name="inner.qp.smoothness_weight",
            ndim=1,
        )
        if smoothness_weight.size != 8:
            raise ValueError(
                "inner.qp.smoothness_weight must be scalar or length 8"
            )
    else:
        smoothness_weight = _finite_float(
            smooth_raw, name="inner.qp.smoothness_weight"
        )
    return QpConfig(
        task_weight=_arr(c.get("task_weight"), [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
        reg=_arr(
            c.get("reg"),
            [1.0e-3, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2, 5.0e-3, 5.0e-3, 5.0e-3],
        ),
        backend=backend,
        use_cpp_kernel=bool(c.get("use_cpp_kernel", True)),
        eps_abs=_finite_float(c.get("eps_abs", 1.0e-6), name="inner.qp.eps_abs"),
        max_iter=int(c.get("max_iter", 400)),
        max_iter_cap=int(c.get("max_iter_cap", 400)),
        euler_order=euler_order,
        collision=collision,
        sr_damping=SrDampingConfig(
            lam0=_finite_float(sr.get("lam0", 0.05), name="sr_damping.lam0"),
            sigma_ref=_finite_float(
                sr.get("sigma_ref", 0.08), name="sr_damping.sigma_ref"
            ),
            sigma_floor=_finite_float(
                sr.get("sigma_floor", 1e-6), name="sr_damping.sigma_floor"
            ),
        ),
        task_weight_min_frac=_finite_float(
            c.get("task_weight_min_frac", 0.05), name="inner.qp.task_weight_min_frac"
        ),
        task_weight_lpf_tau_s=_finite_float(
            c.get("task_weight_lpf_tau_s", 0.25),
            name="inner.qp.task_weight_lpf_tau_s",
        ),
        aniso_task_damping=bool(c.get("aniso_task_damping", True)),
        use_mass_weighted_reg=bool(c.get("use_mass_weighted_reg", True)),
        mass_reg_floor=_finite_float(
            c.get("mass_reg_floor", 0.05), name="inner.qp.mass_reg_floor"
        ),
        mass_weight_exempt_rail=bool(c.get("mass_weight_exempt_rail", True)),
        mass_reg_lpf_tau_s=_finite_float(
            c.get("mass_reg_lpf_tau_s", 0.2), name="inner.qp.mass_reg_lpf_tau_s"
        ),
        limit_damper_band_rad=_finite_float(
            c.get("limit_damper_band_rad", 0.15),
            name="inner.qp.limit_damper_band_rad",
        ),
        limit_damper_band_rail_m=_finite_float(
            c.get("limit_damper_band_rail_m", 0.01),
            name="inner.qp.limit_damper_band_rail_m",
        ),
        warn_on_fail=bool(c.get("warn_on_fail", False)),
        fail_qdot_decay=_finite_float(
            c.get("fail_qdot_decay", 0.85), name="inner.qp.fail_qdot_decay"
        ),
        max_solve_ms=_finite_float(
            c.get("max_solve_ms", 5.0), name="inner.qp.max_solve_ms"
        ),
        twist_sigma_floor=_finite_float(
            c.get("twist_sigma_floor", 0.02), name="inner.qp.twist_sigma_floor"
        ),
        sigma_setbased=SigmaSetBasedConfig(
            enabled=bool(ss.get("enabled", True)),
            activate=_finite_float(
                ss.get("activate", 0.14), name="sigma_setbased.activate"
            ),
            safe=_finite_float(ss.get("safe", 0.06), name="sigma_setbased.safe"),
            exit=_finite_float(ss.get("exit", 0.18), name="sigma_setbased.exit"),
            gamma=_finite_float(ss.get("gamma", 8.0), name="sigma_setbased.gamma"),
            slack_weight=_finite_float(
                ss.get("slack_weight", 200.0), name="sigma_setbased.slack_weight"
            ),
            grad_eps=_finite_float(
                ss.get("grad_eps", 1.0e-4), name="sigma_setbased.grad_eps"
            ),
            grad_period_ticks=max(1, int(ss.get("grad_period_ticks", 10))),
        ),
        branch_barrier=BranchBarrierConfig(
            enabled=bool(bb.get("enabled", True)),
            activate_rad=_finite_float(
                bb.get("activate_rad", 0.52), name="branch_barrier.activate_rad"
            ),
            box_activate_rad=_finite_float(
                bb.get("box_activate_rad", 0.87),
                name="branch_barrier.box_activate_rad",
            ),
            eps_rad=_finite_float(
                bb.get("eps_rad", 0.35), name="branch_barrier.eps_rad"
            ),
            j4_limit_eps_rad=_finite_float(
                bb.get("j4_limit_eps_rad", 5.0 * math.pi / 180.0),
                name="branch_barrier.j4_limit_eps_rad",
            ),
            j4_limit_activate_rad=_finite_float(
                bb.get("j4_limit_activate_rad", 25.0 * math.pi / 180.0),
                name="branch_barrier.j4_limit_activate_rad",
            ),
            j1_overfold_abs_rad=_finite_float(
                bb.get("j1_overfold_abs_rad", 140.0 * math.pi / 180.0),
                name="branch_barrier.j1_overfold_abs_rad",
            ),
            j1_overfold_activate_rad=_finite_float(
                bb.get("j1_overfold_activate_rad", 25.0 * math.pi / 180.0),
                name="branch_barrier.j1_overfold_activate_rad",
            ),
            j1_overfold_eps_rad=_finite_float(
                bb.get("j1_overfold_eps_rad", 0.0),
                name="branch_barrier.j1_overfold_eps_rad",
            ),
            gamma=_finite_float(bb.get("gamma", 6.0), name="branch_barrier.gamma"),
            slack_weight=_finite_float(
                bb.get("slack_weight", 80.0), name="branch_barrier.slack_weight"
            ),
            target_eps_rad=_finite_float(
                bb.get("target_eps_rad", 1.0e-3),
                name="branch_barrier.target_eps_rad",
            ),
            dwell_free_s=_finite_float(
                bb.get("dwell_free_s", 0.3), name="branch_barrier.dwell_free_s"
            ),
            dwell_ramp_s=_finite_float(
                bb.get("dwell_ramp_s", 1.0), name="branch_barrier.dwell_ramp_s"
            ),
            dwell_scale_max=_finite_float(
                bb.get("dwell_scale_max", 5.0),
                name="branch_barrier.dwell_scale_max",
            ),
        ),
        joint_comfort=JointComfortConfig(
            enabled=bool(jc.get("enabled", True)),
            m_comfort_rad=math.radians(
                _finite_float(
                    jc.get("m_comfort_deg", 15.0),
                    name="joint_comfort.m_comfort_deg",
                )
            ),
            activate_rad=math.radians(
                _finite_float(
                    jc.get("activate_deg", 25.0),
                    name="joint_comfort.activate_deg",
                )
            ),
            gamma=_finite_float(jc.get("gamma", 6.0), name="joint_comfort.gamma"),
            slack_weight=_finite_float(
                jc.get("slack_weight", 80.0), name="joint_comfort.slack_weight"
            ),
        ),
        near_arm_margin_rad=_finite_float(
            c.get("near_arm_margin_rad", 0.08),
            name="inner.qp.near_arm_margin_rad",
        ),
        smoothness_weight=smoothness_weight,
        j_max_arm_rad_s3=_finite_float(
            c.get("j_max_arm_rad_s3", 300.0), name="inner.qp.j_max_arm_rad_s3"
        ),
        j_max_rail_m_s3=_finite_float(
            c.get("j_max_rail_m_s3", 3.0), name="inner.qp.j_max_rail_m_s3"
        ),
    )


def _parse_saturation(raw) -> SaturationConfig:
    s = _mapping(raw, name="inner.saturation")
    _reject_unknown(
        s,
        {
            "slack_enter", "slack_exit",
            "secondary_scale", "secondary_scale_tau_s",
        },
        name="inner.saturation",
    )
    enter = _finite_float(s.get("slack_enter", 0.15), name="saturation.slack_enter")
    exit_ = _finite_float(s.get("slack_exit", 0.03), name="saturation.slack_exit")
    if exit_ > enter:
        raise ValueError("inner.saturation.slack_exit must be <= slack_enter")
    return SaturationConfig(
        slack_enter=enter,
        slack_exit=exit_,
        secondary_scale=_finite_float(
            s.get("secondary_scale", 0.15), name="saturation.secondary_scale"
        ),
        secondary_scale_tau_s=_finite_float(
            s.get("secondary_scale_tau_s", 0.10),
            name="saturation.secondary_scale_tau_s",
        ),
    )


def _parse_nullspace(inner: dict) -> tuple[NullspaceTaskConfig, ManipulabilityTaskConfig]:
    n = _mapping(inner.get("nullspace"), name="inner.nullspace")
    _reject_unknown(
        n,
        {
            "k_center", "k_limit", "activation", "weights", "q_nominal_deg",
            "manipulability",
        },
        name="inner.nullspace",
    )
    q_nominal_deg = n.get("q_nominal_deg")
    m = _mapping(n.get("manipulability"), name="inner.nullspace.manipulability")
    _reject_unknown(
        m, {"k_mu", "eps_rad", "sigma_fade_ref", "grad_period_ticks", "qdot_tau_s"},
        name="inner.nullspace.manipulability",
    )
    nullspace = NullspaceTaskConfig(
        k_center=_finite_float(n.get("k_center", 1.0), name="nullspace.k_center"),
        k_limit=_finite_float(n.get("k_limit", 2.0), name="nullspace.k_limit"),
        activation=_finite_float(
            n.get("activation", 0.85), name="nullspace.activation"
        ),
        weights=(
            _finite_array(n["weights"], name="nullspace.weights", ndim=1)
            if n.get("weights") is not None
            else None
        ),
        q_nominal_rad=(
            np.radians(_finite_array(q_nominal_deg, name="nullspace.q_nominal_deg", ndim=1))
            if q_nominal_deg is not None
            else None
        ),
    )
    manipulability = ManipulabilityTaskConfig(
        k_mu=_finite_float(m.get("k_mu", 0.8), name="manipulability.k_mu"),
        eps_rad=_finite_float(m.get("eps_rad", 1e-4), name="manipulability.eps_rad"),
        sigma_fade_ref=_finite_float(
            m.get("sigma_fade_ref", 0.12), name="manipulability.sigma_fade_ref"
        ),
        grad_period_ticks=max(1, int(m.get("grad_period_ticks", 10))),
        qdot_tau_s=_finite_float(
            m.get("qdot_tau_s", 0.05), name="manipulability.qdot_tau_s"
        ),
    )
    return nullspace, manipulability


def _parse_arm_angle(inner: dict) -> ArmAngleTaskConfig:
    a = _mapping(inner.get("arm_angle"), name="inner.arm_angle")
    _reject_unknown(
        a,
        {
            "enabled", "k_psi", "psi_ref_deg", "fd_eps_rad", "safe_denom_eps",
            "obs_decay_gain", "obs_smooth_floor", "max_qdot_frac",
            "psi_home_deg", "max_psi_swing_deg",
        },
        name="inner.arm_angle",
    )
    psi_ref_deg = a.get("psi_ref_deg")
    psi_home_deg = a.get("psi_home_deg")
    return ArmAngleTaskConfig(
        enabled=bool(a.get("enabled", False)),
        k_psi=_finite_float(a.get("k_psi", 1.0), name="arm_angle.k_psi"),
        obs_smooth_floor=_finite_float(
            a.get("obs_smooth_floor", 0.3), name="arm_angle.obs_smooth_floor"
        ),
        psi_ref_rad=(
            math.radians(_finite_float(psi_ref_deg, name="arm_angle.psi_ref_deg"))
            if psi_ref_deg is not None
            else None
        ),
        psi_home_rad=(
            math.radians(_finite_float(psi_home_deg, name="arm_angle.psi_home_deg"))
            if psi_home_deg is not None
            else None
        ),
    )


def _parse_psi_retarget(inner: dict) -> PsiRetargetConfig:
    p = _mapping(inner.get("psi_retarget"), name="inner.psi_retarget")
    _reject_unknown(
        p,
        {
            "enabled", "n_y", "n_d", "n_psi", "w_sigma", "w_wrist",
            "margin_floor_deg", "psi_rate_deg_s", "rail_margin_m",
            "wrist_min_deg", "d_center_rate_m_s",
            "psi_cmd_lead_deg",
            "psi_replan_period_s", "psi_search_half_span_deg", "psi_search_n",
            "psi_wrist_ok_deg", "psi_envelope_deg",
            "psi_attr_deg", "d_attr_m", "psi_return_dwell_s",
            "require_design_family",
        },
        name="inner.psi_retarget",
    )
    env = p.get("psi_envelope_deg", [40.0, 110.0])
    if isinstance(env, (list, tuple)) and len(env) == 2:
        env_lo, env_hi = float(env[0]), float(env[1])
    else:
        env_lo, env_hi = 40.0, 110.0
    return PsiRetargetConfig(
        enabled=bool(p.get("enabled", True)),
        n_y=int(p.get("n_y", 9)),
        n_d=int(p.get("n_d", 8)),
        n_psi=int(p.get("n_psi", 9)),
        w_sigma=_finite_float(p.get("w_sigma", 0.5), name="psi_retarget.w_sigma"),
        w_wrist=_finite_float(p.get("w_wrist", 0.5), name="psi_retarget.w_wrist"),
        margin_floor_rad=math.radians(
            _finite_float(
                p.get("margin_floor_deg", 15.0), name="psi_retarget.margin_floor_deg"
            )
        ),
        psi_rate_rad_s=math.radians(
            _finite_float(
                p.get("psi_rate_deg_s", 25.0), name="psi_retarget.psi_rate_deg_s"
            )
        ),
        d_center_rate_m_s=_finite_float(
            p.get("d_center_rate_m_s", 0.02), name="psi_retarget.d_center_rate_m_s"
        ),
        psi_cmd_lead_rad=math.radians(
            _finite_float(
                p.get("psi_cmd_lead_deg", 18.0),
                name="psi_retarget.psi_cmd_lead_deg",
            )
        ),
        psi_attr_rad=math.radians(
            _finite_float(p.get("psi_attr_deg", 68.0), name="psi_retarget.psi_attr_deg")
        ),
        d_attr_m=_finite_float(
            p.get("d_attr_m", -0.185), name="psi_retarget.d_attr_m"
        ),
        psi_return_dwell_s=_finite_float(
            p.get("psi_return_dwell_s", 1.0), name="psi_retarget.psi_return_dwell_s"
        ),
        require_design_family=bool(p.get("require_design_family", False)),
        psi_replan_period_s=_finite_float(
            p.get("psi_replan_period_s", 0.1),
            name="psi_retarget.psi_replan_period_s",
        ),
        psi_search_half_span_rad=math.radians(
            _finite_float(
                p.get("psi_search_half_span_deg", 45.0),
                name="psi_retarget.psi_search_half_span_deg",
            )
        ),
        psi_search_n=int(p.get("psi_search_n", 9)),
        psi_wrist_ok_rad=math.radians(
            _finite_float(
                p.get("psi_wrist_ok_deg", 40.0),
                name="psi_retarget.psi_wrist_ok_deg",
            )
        ),
        psi_envelope_lo_rad=math.radians(env_lo),
        psi_envelope_hi_rad=math.radians(env_hi),
        rail_margin_m=_finite_float(
            p.get("rail_margin_m", 0.02), name="psi_retarget.rail_margin_m"
        ),
        wrist_min_rad=math.radians(
            _finite_float(
                p.get("wrist_min_deg", 30.0), name="psi_retarget.wrist_min_deg"
            )
        ),
    )


def _parse_ird(inner: dict) -> IrdConfig:
    r = _mapping(inner.get("ird"), name="inner.ird")
    _reject_unknown(
        r,
        {
            "enabled", "checkpoint", "robot_spec", "device", "allow_stale",
        },
        name="inner.ird",
    )
    defaults = IrdConfig()
    return IrdConfig(
        enabled=bool(r.get("enabled", False)),
        checkpoint=str(r.get("checkpoint", defaults.checkpoint)),
        robot_spec=str(r.get("robot_spec", defaults.robot_spec)),
        device=str(r.get("device", "cpu")),
        allow_stale=bool(r.get("allow_stale", True)),
    )


def _parse_rail_extension(inner: dict) -> RailExtensionConfig:
    r = _mapping(inner.get("rail_extension"), name="inner.rail_extension")
    _reject_unknown(
        r,
        {
            "enabled", "k_ext", "k_ff", "v_ff_thr_m_s", "v_ff_span_m_s",
            "e0_m", "e1_m", "w_max", "v_max_m_s", "limit_margin_m",
            "pin_margin_m", "escape_leave_m",
            "k_sigma_boost", "k_esc", "w_sigma_floor",
            "k_pose", "pose_e0_m", "pose_e1_m", "pose_w_max",
            "sigma_guard_enter", "sigma_guard_exit", "v_guard_max_m_s",
            "v_lpf_tau_s", "v_lpf_fc_hz", "v_lpf_tau_escape_s",
            "sigma_escape_enter", "sigma_escape_exit",
            "margin_escape_enter", "margin_escape_exit", "sigma_drop_rate",
            "escape_enter_dwell_s",
            "k_escape_boost", "escape_grad_floor",
            "k_margin_boost", "w_ext_cap",
            "soft_min_m", "soft_max_m", "d_band_m",
            "healthy_sigma_mute",
            "d_star_err0_m", "d_star_err1_m", "d_star_w_mult", "d_star_reg_mult",
            "press_v_force_min_m_s", "press_dz_max_m", "press_y_err_m",
            "press_stall_s", "d_star_nudge_m", "open_travel_min_m",
            "escape_sign_policy",
        },
        name="inner.rail_extension",
    )
    return RailExtensionConfig(
        enabled=bool(r.get("enabled", True)),
        k_ext=_finite_float(r.get("k_ext", 1.0), name="rail_extension.k_ext"),
        k_ff=_finite_float(r.get("k_ff", 1.0), name="rail_extension.k_ff"),
        v_ff_thr_m_s=_finite_float(
            r.get("v_ff_thr_m_s", 0.01), name="rail_extension.v_ff_thr_m_s"
        ),
        v_ff_span_m_s=_finite_float(
            r.get("v_ff_span_m_s", 0.03), name="rail_extension.v_ff_span_m_s"
        ),
        e0_m=_finite_float(r.get("e0_m", 0.05), name="rail_extension.e0_m"),
        e1_m=_finite_float(r.get("e1_m", 0.15), name="rail_extension.e1_m"),
        w_max=_finite_float(r.get("w_max", 1.5), name="rail_extension.w_max"),
        v_max_m_s=_finite_float(
            r.get("v_max_m_s", 0.08), name="rail_extension.v_max_m_s"
        ),
        limit_margin_m=_finite_float(
            r.get("limit_margin_m", 0.15), name="rail_extension.limit_margin_m"
        ),
        pin_margin_m=_finite_float(
            r.get("pin_margin_m", 0.008), name="rail_extension.pin_margin_m"
        ),
        escape_leave_m=_finite_float(
            r.get("escape_leave_m", 0.04), name="rail_extension.escape_leave_m"
        ),
        k_sigma_boost=_finite_float(
            r.get("k_sigma_boost", 2.0), name="rail_extension.k_sigma_boost"
        ),
        k_esc=_finite_float(r.get("k_esc", 0.5), name="rail_extension.k_esc"),
        w_sigma_floor=_finite_float(
            r.get("w_sigma_floor", 1.0), name="rail_extension.w_sigma_floor"
        ),
        k_pose=_finite_float(r.get("k_pose", 2.0), name="rail_extension.k_pose"),
        pose_e0_m=_finite_float(
            r.get("pose_e0_m", 0.005), name="rail_extension.pose_e0_m"
        ),
        pose_e1_m=_finite_float(
            r.get("pose_e1_m", 0.04), name="rail_extension.pose_e1_m"
        ),
        pose_w_max=_finite_float(
            r.get("pose_w_max", 4.0), name="rail_extension.pose_w_max"
        ),
        sigma_guard_enter=_finite_float(
            r.get("sigma_guard_enter", 0.45), name="rail_extension.sigma_guard_enter"
        ),
        sigma_guard_exit=_finite_float(
            r.get("sigma_guard_exit", 0.70), name="rail_extension.sigma_guard_exit"
        ),
        v_guard_max_m_s=_finite_float(
            r.get("v_guard_max_m_s", 0.04), name="rail_extension.v_guard_max_m_s"
        ),
        v_lpf_tau_s=_finite_float(
            r.get("v_lpf_tau_s", 0.05), name="rail_extension.v_lpf_tau_s"
        ),
        v_lpf_fc_hz=_finite_float(
            r.get("v_lpf_fc_hz", 0.0), name="rail_extension.v_lpf_fc_hz"
        ),
        v_lpf_tau_escape_s=_finite_float(
            r.get("v_lpf_tau_escape_s", 0.04),
            name="rail_extension.v_lpf_tau_escape_s",
        ),
        sigma_escape_enter=_finite_float(
            r.get("sigma_escape_enter", 0.55),
            name="rail_extension.sigma_escape_enter",
        ),
        sigma_escape_exit=_finite_float(
            r.get("sigma_escape_exit", 0.80),
            name="rail_extension.sigma_escape_exit",
        ),
        margin_escape_enter=_finite_float(
            r.get("margin_escape_enter", 0.12),
            name="rail_extension.margin_escape_enter",
        ),
        margin_escape_exit=_finite_float(
            r.get("margin_escape_exit", 0.25),
            name="rail_extension.margin_escape_exit",
        ),
        sigma_drop_rate=_finite_float(
            r.get("sigma_drop_rate", 0.0), name="rail_extension.sigma_drop_rate"
        ),
        escape_enter_dwell_s=_finite_float(
            r.get("escape_enter_dwell_s", 0.05),
            name="rail_extension.escape_enter_dwell_s",
        ),
        k_escape_boost=_finite_float(
            r.get("k_escape_boost", 1.2), name="rail_extension.k_escape_boost"
        ),
        escape_grad_floor=_finite_float(
            r.get("escape_grad_floor", 0.0), name="rail_extension.escape_grad_floor"
        ),
        k_margin_boost=_finite_float(
            r.get("k_margin_boost", 4.0), name="rail_extension.k_margin_boost"
        ),
        w_ext_cap=_finite_float(
            r.get("w_ext_cap", 24.0), name="rail_extension.w_ext_cap"
        ),
        soft_min_m=_finite_float(
            r.get("soft_min_m", 0.025), name="rail_extension.soft_min_m"
        ),
        soft_max_m=_finite_float(
            r.get("soft_max_m", 0.78), name="rail_extension.soft_max_m"
        ),
        healthy_sigma_mute=_finite_float(
            r.get("healthy_sigma_mute", 0.08),
            name="rail_extension.healthy_sigma_mute",
        ),
        d_band_m=_finite_float(
            r.get("d_band_m", 0.005), name="rail_extension.d_band_m"
        ),
        d_star_err0_m=_finite_float(
            r.get("d_star_err0_m", 0.01), name="rail_extension.d_star_err0_m"
        ),
        d_star_err1_m=_finite_float(
            r.get("d_star_err1_m", 0.04), name="rail_extension.d_star_err1_m"
        ),
        d_star_w_mult=_finite_float(
            r.get("d_star_w_mult", 6.0), name="rail_extension.d_star_w_mult"
        ),
        d_star_reg_mult=_finite_float(
            r.get("d_star_reg_mult", 20.0), name="rail_extension.d_star_reg_mult"
        ),
        press_v_force_min_m_s=_finite_float(
            r.get("press_v_force_min_m_s", 0.02),
            name="rail_extension.press_v_force_min_m_s",
        ),
        press_dz_max_m=_finite_float(
            r.get("press_dz_max_m", 0.002), name="rail_extension.press_dz_max_m"
        ),
        press_y_err_m=_finite_float(
            r.get("press_y_err_m", 0.005), name="rail_extension.press_y_err_m"
        ),
        press_stall_s=_finite_float(
            r.get("press_stall_s", 0.5), name="rail_extension.press_stall_s"
        ),
        d_star_nudge_m=_finite_float(
            r.get("d_star_nudge_m", 0.01), name="rail_extension.d_star_nudge_m"
        ),
        open_travel_min_m=_finite_float(
            r.get("open_travel_min_m", 0.01),
            name="rail_extension.open_travel_min_m",
        ),
        escape_sign_policy=str(r.get("escape_sign_policy", "auto")).strip().lower(),
    )


def _parse_rail_allocator(inner: dict) -> RailAllocatorConfig:
    r = _mapping(inner.get("rail_allocator"), name="inner.rail_allocator")
    _reject_unknown(
        r,
        {
            "v0_m_s", "w0_rad_s", "k_margin",
            "kp_mid", "ki_mid", "u_mid_max_m_s", "k_err_rail", "e_ref_m",
            "f_c_hz", "reaction_s",
            "observer_pos_gain", "observer_vel_gain",
            "observer_vel_lpf_hz",
        },
        name="inner.rail_allocator",
    )
    return RailAllocatorConfig(
        v0_m_s=_finite_float(r.get("v0_m_s", 0.05), name="rail_allocator.v0_m_s"),
        w0_rad_s=_finite_float(
            r.get("w0_rad_s", 0.30), name="rail_allocator.w0_rad_s"
        ),
        k_margin=_finite_float(
            r.get("k_margin", 4.0), name="rail_allocator.k_margin"
        ),
        kp_mid=_finite_float(r.get("kp_mid", 1.2), name="rail_allocator.kp_mid"),
        ki_mid=_finite_float(r.get("ki_mid", 0.80), name="rail_allocator.ki_mid"),
        u_mid_max_m_s=_finite_float(
            r.get("u_mid_max_m_s", 0.12), name="rail_allocator.u_mid_max_m_s"
        ),
        k_err_rail=_finite_float(
            r.get("k_err_rail", 4.0), name="rail_allocator.k_err_rail"
        ),
        e_ref_m=_finite_float(
            r.get("e_ref_m", 0.08), name="rail_allocator.e_ref_m"
        ),
        f_c_hz=_finite_float(r.get("f_c_hz", 20.0), name="rail_allocator.f_c_hz"),
        reaction_s=_finite_float(
            r.get("reaction_s", 0.06), name="rail_allocator.reaction_s"
        ),
        observer_pos_gain=_finite_float(
            r.get("observer_pos_gain", 0.35), name="rail_allocator.observer_pos_gain"
        ),
        observer_vel_gain=_finite_float(
            r.get("observer_vel_gain", 2.0), name="rail_allocator.observer_vel_gain"
        ),
        observer_vel_lpf_hz=_finite_float(
            r.get("observer_vel_lpf_hz", 8.0),
            name="rail_allocator.observer_vel_lpf_hz",
        ),
    )


def _parse_rail(rail_raw: dict, hw_lw: dict) -> RailLockConfig:
    _reject_unknown(
        rail_raw,
        {
            "mode", "locked_style", "q_ref_m", "lock_gain", "lock_reg_scale",
            "lock_vel_eps_m_s", "lock_hard_pin", "v_max_m_s", "travel_m",
            "soft_min_m", "soft_max_m", "hard_min_m", "hard_max_m",
        },
        name="rail",
    )
    rail_mode, locked_style = _resolve_rail_mode(rail_raw)
    soft_min = _finite_float(
        rail_raw.get("soft_min_m", hw_lw.get("soft_min_m", 0.015)),
        name="rail.soft_min_m",
    )
    soft_max = _finite_float(
        rail_raw.get("soft_max_m", hw_lw.get("soft_max_m", 0.77)),
        name="rail.soft_max_m",
    )
    hard_min = _finite_float(
        rail_raw.get("hard_min_m", hw_lw.get("hard_min_m", 0.005)),
        name="rail.hard_min_m",
    )
    hard_max = _finite_float(
        rail_raw.get("hard_max_m", hw_lw.get("hard_max_m", 0.78)),
        name="rail.hard_max_m",
    )
    travel = _finite_float(rail_raw.get("travel_m", 0.80), name="rail.travel_m")
    if not 0.0 <= hard_min <= soft_min < soft_max <= hard_max <= travel:
        raise ValueError(
            "rail limits must satisfy "
            "0 <= hard_min <= soft_min < soft_max <= hard_max <= travel"
        )
    if rail_raw and hw_lw and ("soft_min_m" in hw_lw or "soft_max_m" in hw_lw):
        hw_min = _finite_float(hw_lw.get("soft_min_m", soft_min), name="hw rail soft_min")
        hw_max = _finite_float(hw_lw.get("soft_max_m", soft_max), name="hw rail soft_max")
        if abs(hw_min - soft_min) > 1.0e-6 or abs(hw_max - soft_max) > 1.0e-6:
            raise ValueError("rail soft-limit mismatch between QPIK and hardware")
    if rail_raw and hw_lw and ("hard_min_m" in hw_lw or "hard_max_m" in hw_lw):
        hw_hmin = _finite_float(hw_lw.get("hard_min_m", hard_min), name="hw rail hard_min")
        hw_hmax = _finite_float(hw_lw.get("hard_max_m", hard_max), name="hw rail hard_max")
        if abs(hw_hmin - hard_min) > 1.0e-6 or abs(hw_hmax - hard_max) > 1.0e-6:
            raise ValueError("rail hard-limit mismatch between QPIK and hardware")
    return RailLockConfig(
        mode=rail_mode,
        locked_style=locked_style,
        q_ref_m=(
            None
            if rail_raw.get("q_ref_m") is None
            else _finite_float(rail_raw["q_ref_m"], name="rail.q_ref_m")
        ),
        lock_gain=_finite_float(rail_raw.get("lock_gain", 200.0), name="rail.lock_gain"),
        lock_reg_scale=_finite_float(
            rail_raw.get("lock_reg_scale", 100.0), name="rail.lock_reg_scale"
        ),
        lock_vel_eps_m_s=_finite_float(
            rail_raw.get("lock_vel_eps_m_s", 0.0), name="rail.lock_vel_eps_m_s"
        ),
        lock_hard_pin=bool(rail_raw.get("lock_hard_pin", True)),
        v_max_m_s=(
            None
            if rail_raw.get("v_max_m_s") is None
            else _finite_float(rail_raw["v_max_m_s"], name="rail.v_max_m_s")
        ),
        travel_m=travel,
        soft_min_m=soft_min,
        soft_max_m=soft_max,
        hard_min_m=hard_min,
        hard_max_m=hard_max,
    )


def _parse_cartesian_track(raw: dict) -> CartesianTrackGains:
    section = _mapping(raw.get("cartesian_track"), name="cartesian_track")
    _reject_unknown(
        section,
        {"k_task_lin", "k_task_rot", "max_pos_err_m", "max_rot_err_rad"},
        name="cartesian_track",
    )
    defaults = CartesianTrackGains()
    gains = CartesianTrackGains(
        k_task_lin=_finite_float(
            section.get("k_task_lin", defaults.k_task_lin),
            name="cartesian_track.k_task_lin",
        ),
        k_task_rot=_finite_float(
            section.get("k_task_rot", defaults.k_task_rot),
            name="cartesian_track.k_task_rot",
        ),
        max_pos_err_m=_finite_float(
            section.get("max_pos_err_m", defaults.max_pos_err_m),
            name="cartesian_track.max_pos_err_m",
        ),
        max_rot_err_rad=_finite_float(
            section.get("max_rot_err_rad", defaults.max_rot_err_rad),
            name="cartesian_track.max_rot_err_rad",
        ),
    )
    if gains.k_task_lin < 0.0 or gains.k_task_rot < 0.0:
        raise ValueError("cartesian_track gains must be non-negative")
    if gains.max_pos_err_m <= 0.0 or gains.max_rot_err_rad <= 0.0:
        raise ValueError("cartesian_track error limits must be positive")
    return gains


def build_joint_ik_config(raw: dict) -> JointIkConfig:
    """Build JointIkConfig from inner.qp + qpik.hard_limits."""

    if not isinstance(raw, dict):
        raise ValueError("controller config root must be a mapping")
    timing = _mapping(raw.get("timing"), name="timing")
    _reject_unknown(
        timing,
        {
            "dt_ms", "feedback_timeout_ms", "feedback_coast_ms",
            "rt_disable_gc", "verbose_json", "control_cpu", "disable_cstates",
        },
        name="timing",
    )
    inner = _mapping(raw.get("inner"), name="inner")
    qpik = _mapping(raw.get("qpik"), name="qpik")
    _reject_retired_qpik(qpik)
    _reject_unknown(qpik, {"hard_limits"}, name="qpik")

    hard = _mapping(qpik.get("hard_limits"), name="qpik.hard_limits")
    _reject_unknown(
        hard,
        {
            "v_scale", "a_max_arm_rad_s2", "a_max_rail_m_s2",
            "position_margin_deg", "position_margin_rail_mm",
            "command_lead_arm_deg", "command_lead_rail_mm",
            "velocity_damper", "collision", "rail",
        },
        name="qpik.hard_limits",
    )
    retired_inner = sorted(
        set(inner)
        & {
            "a_max_rail_escape_m_s2",
            "rail_escape_v_min_m_s",
            "rail_escape_v_max_m_s",
            "sigma_escape_enter",
            "sigma_escape_exit",
        }
    )
    if retired_inner:
        raise ValueError(
            "retired QPIK configuration keys in inner: " + ", ".join(retired_inner)
        )
    _reject_unknown(
        inner,
        {
            "control_frame", "euler_order", "sync_tcp_from_robot",
            "v_scale", "a_max_arm", "a_max_arm_rad_s2", "a_max_rail_m_s2",
            "position_margin_deg", "position_margin_rail_mm",
            "resync_err_deg", "resync_err_rail_mm",
            "qp", "collision", "nullspace", "arm_angle", "rail_extension", "rail",
            "rail_allocator",
            "psi_retarget", "ird",
            "nullspace_d_null", "nullspace_d_null_adaptive", "nullspace_max_qdot_frac",
            "saturation",
        },
        name="inner",
    )

    euler_order = str(
        _mapping(raw.get("frames"), name="frames").get(
            "euler_order", inner.get("euler_order", "xyz")
        )
    )
    collision = _parse_collision(
        hard.get("collision", inner.get("collision")),
        name="collision",
    )
    qp = _parse_qp(inner, collision, euler_order)
    damper = _mapping(hard.get("velocity_damper"), name="qpik.hard_limits.velocity_damper")
    if damper:
        _reject_unknown(
            damper, {"arm_band_rad", "rail_band_m"},
            name="qpik.hard_limits.velocity_damper",
        )
        if "arm_band_rad" in damper:
            qp.limit_damper_band_rad = _finite_float(
                damper["arm_band_rad"], name="velocity_damper.arm_band_rad"
            )
        if "rail_band_m" in damper:
            qp.limit_damper_band_rail_m = _finite_float(
                damper["rail_band_m"], name="velocity_damper.rail_band_m"
            )
    if qp.limit_damper_band_rad < 0.0 or qp.limit_damper_band_rail_m < 0.0:
        raise ValueError("velocity damper bands must be non-negative")

    nullspace, manipulability = _parse_nullspace(inner)
    arm_angle = _parse_arm_angle(inner)
    psi_retarget = _parse_psi_retarget(inner)
    ird = _parse_ird(inner)
    rail_extension = _parse_rail_extension(inner)
    rail_allocator = _parse_rail_allocator(inner)
    qp.limit_damper_rail_reaction_s = float(rail_allocator.reaction_s)
    cartesian_track = _parse_cartesian_track(raw)

    hw_lw = _mapping(
        _mapping(raw.get("hw"), name="hw").get("lw100"), name="hw.lw100"
    )
    rail_raw = _mapping(
        hard.get("rail", inner.get("rail")), name="rail"
    )
    rail = _parse_rail(rail_raw, hw_lw)
    if "hard_min_m" in rail_raw or "hard_max_m" in rail_raw:
        band = float(qp.limit_damper_band_rail_m)
        lo_gap = float(rail.soft_min_m) - float(rail.hard_min_m)
        hi_gap = float(rail.hard_max_m) - float(rail.soft_max_m)
        if abs(lo_gap - band) > 1.0e-6 or abs(hi_gap - band) > 1.0e-6:
            raise ValueError(
                "rail damper band must equal the hard–soft gap "
                f"(band={band:.6f}, lo_gap={lo_gap:.6f}, hi_gap={hi_gap:.6f})"
            )
    rail_extension.soft_min_m = float(rail.soft_min_m)
    rail_extension.soft_max_m = float(rail.soft_max_m)
    policy = str(rail_extension.escape_sign_policy).strip().lower()
    if policy in {"minus", "-", "neg", "negative"}:
        rail_extension.escape_sign_policy = "minus"
    elif policy in {"plus", "+", "pos", "positive"}:
        rail_extension.escape_sign_policy = "plus"
    elif policy in {"auto", "open", "grad", "gradient"}:
        rail_extension.escape_sign_policy = "auto"
    else:
        raise ValueError(
            "rail_extension.escape_sign_policy must be 'auto', 'minus', or 'plus', "
            f"got {rail_extension.escape_sign_policy!r}"
        )

    def hard_value(name: str, legacy_name: str, default):
        return hard.get(name, inner.get(legacy_name, default))

    cfg = JointIkConfig(
        dt=_finite_float(timing.get("dt_ms", 5.0), name="timing.dt_ms") / 1000.0,
        feedback_timeout_s=_finite_float(
            timing.get("feedback_timeout_ms", 50.0),
            name="timing.feedback_timeout_ms",
        )
        / 1000.0,
        feedback_coast_s=_finite_float(
            timing.get("feedback_coast_ms", 300.0),
            name="timing.feedback_coast_ms",
        )
        / 1000.0,
        rt_disable_gc=bool(timing.get("rt_disable_gc", True)),
        verbose_json=bool(timing.get("verbose_json", False)),
        control_cpu=(
            int(timing["control_cpu"])
            if timing.get("control_cpu") is not None
            else None
        ),
        disable_cstates=bool(timing.get("disable_cstates", True)),
        control_frame=str(inner.get("control_frame", "tool")),
        euler_order=euler_order,
        qp=qp,
        nullspace=nullspace,
        manipulability=manipulability,
        arm_angle=arm_angle,
        psi_retarget=psi_retarget,
        ird=ird,
        collision=collision,
        rail=rail,
        rail_extension=rail_extension,
        rail_allocator=rail_allocator,
        cartesian_track=cartesian_track,
        v_scale=_finite_float(hard_value("v_scale", "v_scale", 0.5), name="v_scale"),
        a_max_arm_rad_s2=_finite_float(
            hard_value("a_max_arm_rad_s2", "a_max_arm", 20.0), name="a_max_arm_rad_s2"
        ),
        a_max_rail_m_s2=_finite_float(
            hard_value("a_max_rail_m_s2", "a_max_rail_m_s2", 0.60),
            name="a_max_rail_m_s2",
        ),
        position_margin_rad=math.radians(
            _finite_float(
                hard_value("position_margin_deg", "position_margin_deg", 0.3),
                name="position_margin_deg",
            )
        ),
        position_margin_rail_m=_finite_float(
            hard_value("position_margin_rail_mm", "position_margin_rail_mm", 0.0),
            name="position_margin_rail_mm",
        )
        / 1000.0,
        resync_err_rad=math.radians(
            _finite_float(
                hard_value("command_lead_arm_deg", "resync_err_deg", 6.0),
                name="command_lead_arm_deg",
            )
        ),
        resync_err_rail_m=_finite_float(
            hard_value("command_lead_rail_mm", "resync_err_rail_mm", 20.0),
            name="command_lead_rail_mm",
        )
        / 1000.0,
        nullspace_d_null=_finite_float(
            inner.get("nullspace_d_null", 0.5), name="inner.nullspace_d_null"
        ),
        nullspace_d_null_adaptive=_finite_float(
            inner.get("nullspace_d_null_adaptive", 1.0),
            name="inner.nullspace_d_null_adaptive",
        ),
        nullspace_max_qdot_frac=_finite_float(
            inner.get("nullspace_max_qdot_frac", 0.2),
            name="inner.nullspace_max_qdot_frac",
        ),
        saturation=_parse_saturation(inner.get("saturation")),
    )
    assert_design_attractor_consistent(cfg)
    return cfg


def assert_design_attractor_consistent(cfg: JointIkConfig, kin=None) -> None:
    """Refuse a yaml whose two nullspace attractors point at different families."""
    qn = getattr(cfg.nullspace, "q_nominal_rad", None)
    if qn is None:
        return
    qn = np.asarray(qn, dtype=float).reshape(-1)
    if qn.size != 8:
        raise ValueError(
            f"nullspace.q_nominal_deg must be length 8, got {qn.size}"
        )
    from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
    from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import (
        d_from_q,
        fold_psi_to_positive,
    )
    from rm75_control.kinematics.srs_ik import Q_LOWER, Q_UPPER, psi_from_q

    psi_attr = float(cfg.psi_retarget.psi_attr_rad)
    d_attr = float(cfg.psi_retarget.d_attr_m)
    lo = float(cfg.psi_retarget.psi_envelope_lo_rad)
    hi = float(cfg.psi_retarget.psi_envelope_hi_rad)
    if psi_attr < lo - 1.0e-9 or psi_attr > hi + 1.0e-9:
        raise ValueError(
            "psi_retarget.psi_attr_deg must lie inside psi_envelope_deg "
            f"({math.degrees(psi_attr):.2f} not in "
            f"[{math.degrees(lo):.2f}, {math.degrees(hi):.2f}])"
        )
    psi_q = fold_psi_to_positive(psi_from_q(qn))
    psi_err = abs(psi_q - fold_psi_to_positive(psi_attr))
    if psi_err > math.radians(1.0) + 1.0e-9:
        raise ValueError(
            "q_nominal ψ disagrees with psi_attr: "
            f"ψ(q_nominal)={math.degrees(psi_q):.2f}° "
            f"psi_attr={math.degrees(psi_attr):.2f}° "
            f"(|Δ|={math.degrees(psi_err):.2f}° > 1°)"
        )
    if kin is None:
        kin = RobotKinematics()
    d_q = d_from_q(kin, qn)
    if abs(d_q - d_attr) > 0.005 + 1.0e-9:
        raise ValueError(
            "q_nominal d disagrees with d_attr: "
            f"d(q_nominal)={d_q:.4f} m d_attr={d_attr:.4f} m "
            f"(|Δ|={abs(d_q - d_attr) * 1000.0:.1f} mm > 5 mm)"
        )
    q_arm = qn[1:]
    margin = float(np.min(np.minimum(q_arm - Q_LOWER, Q_UPPER - q_arm)))
    need = float(cfg.qp.joint_comfort.activate_rad)
    if margin + 1.0e-9 < need:
        raise ValueError(
            "q_nominal worst-joint margin is inside the comfort wall: "
            f"margin={math.degrees(margin):.2f}° "
            f"joint_comfort.activate={math.degrees(need):.2f}°"
        )


__all__ = ["assert_design_attractor_consistent", "build_joint_ik_config"]
```

### `rm75_control/control/joint_admittance_8dof/hw/rail_servo.py`

```python
"""LW100 rail servo bridge: PC soft position loop → FA24 continuous velocity.

Controller path (virtual-rail WBC structure; motor replaces sim rail):
  * WBC streams ``q_cmd[0]`` (metres) via ``set_target_m`` each control tick,
    optionally with ``v_ff_m_s`` so the worker does not differentiate a
    nominal-dt position stream (5 ms integrate / ~6.5 ms wall → 25% slow).
  * Soft CSP: stream-aware online ``(x_ref,v_ref)`` from ``set_target_m`` +
    ``v = v_ref + kp*(x_ref−x) + kd*(v_ref−v_enc)`` → FA24.
    ``v_enc`` is a bounded encoder-position difference (the 0x1000 speed
    register lags ~150 ms and plugged the carriage on every gamepad stop).
    Position is closed on the shaped reference, never ``x_goal`` (command
    lead / later KMP OTG stay outside).  Same law for QPIK coupled-velocity
    and a position+FF stream (KMP/DMP ``p_cmd``, ``p_dot``).  Host ``a_max``
    is capped to FA40 so PD cannot chop FA24 against the 200 ms drive ramp.
  * Standstill hysteresis freezes FA24 after a tight settle (enter band) and
    only re-engages if disturbed past the wider exit band or ``v_ref≠0``.
  * Encoder → SHM / Genesis twin only. Encoder is **never** fed into the WBC.
  * Exit: FA24=0, SON held by default (``release_son_on_exit: false``) so a
    controller restart does not edge-enable and wipe the multi-turn monitor.

Pr P1 + CTRG continuous follow is not used (stuttery point-to-point).
"""

from __future__ import annotations

import csv
import math
import queue
import threading
import time
from collections import deque
from collections.abc import Sequence
from statistics import median
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

# Shared idle / park / stream-stationary threshold (m/s).  Do not scatter
# 1 mm/s literals — they used to gate catch-up and park independently.
RAIL_IDLE_EPS_M_S = 1.0e-3
# Consecutive agreeing samples needed to re-anchor after a rejected leap.
RESTITCH_REANCHOR_POLLS = 3
RESTITCH_MARGIN_SCALE = 0.5


def encoder_jump_limit_m(
    v_max_m_s: float,
    gap_s: float,
    jump_margin_m: float,
    *,
    restitch: bool = False,
    restitch_margin_scale: float = RESTITCH_MARGIN_SCALE,
) -> float:
    """Time-aware encoder jump limit.  Restitch only tightens the margin.

    A GIL stall of a few hundred milliseconds can move the carriage by
    ``v_max * gap``.  Collapsing the limit to a fixed millimetre margin
    after ``_link_restitch`` rejects that real motion and never recovers.
    """

    margin = max(float(jump_margin_m), 0.0)
    if restitch:
        margin *= max(float(restitch_margin_scale), 0.0)
    return max(float(v_max_m_s), 0.0) * max(float(gap_s), 0.0) + margin


def samples_agree_for_reanchor(
    latest_m: float,
    previous_m: float,
    *,
    v_max_m_s: float,
    dt_s: float,
    agree_floor_m: float = 0.001,
) -> bool:
    """True when two restitch candidates differ by at most ``v_max·dt``."""

    if not (math.isfinite(float(latest_m)) and math.isfinite(float(previous_m))):
        return False
    lim = max(float(agree_floor_m), max(float(v_max_m_s), 0.0) * max(float(dt_s), 0.0))
    return abs(float(latest_m) - float(previous_m)) <= lim

from rm75_control.hw.lw100.drive import (
    LW100Drive,
    LW100DriveConfig,
    di_limits_pressed_from_mask,
)
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuError
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    wall_cap,
)
from rm75_control.hw.lw100.rail_calibration import (
    COMMS_FAIL_MSG,
    FRAME_UNKNOWN_MSG,
    MISSING_CAL_MSG,
    POWER_CYCLE_CAL_MSG,
    CalValidationError,
    default_calibration_path,
    invalidate_calibration,
    load_calibration,
    save_calibration,
    sync_calibration_frame,
    validate_on_drive,
)


def live_host_accel_m_s2(
    *,
    vel_max_m_s: float,
    accel_ms: float,
    configured_m_s2: float,
    match_drive: bool = True,
) -> float:
    """Cap host ``a_max`` so PD cannot outrun FA40 (loaded-scan 30→20→24 chop)."""
    configured = max(float(configured_m_s2), 1.0e-3)
    if not match_drive:
        return configured
    accel_s = max(float(accel_ms) * 1.0e-3, 0.05)
    a_drive = max(float(vel_max_m_s), 1.0e-6) / accel_s
    return min(configured, max(0.08, 0.85 * a_drive))


def next_poll_deadline(next_t: float, now: float, period: float) -> float:
    """Absolute schedule: overruns skip catch-up instead of accumulating debt."""
    return max(float(next_t) + float(period), float(now))


@dataclass
class RailServoConfig:
    enabled: bool = False
    host: str = "192.168.0.7"
    port: int = 8234
    slave_id: int = 1
    lead_mm: float = 10.0
    # calibrated_file | current | fixed
    zero_mode: str = "calibrated_file"
    counts0: int = 0
    calibration_path: str = ""
    require_calibration: bool = True
    home_di: str = "di4"
    plus_di: str = "di3"
    di_nc: bool = True
    di_debounce_n: int = 3
    soft_min_m: float = 0.015
    soft_max_m: float = 0.77
    hard_min_m: float = 0.005
    hard_max_m: float = 0.78
    post_home_m: float = 0.025
    limit_poll_every: int = 5  # worker: check DI every N polls when calibrated
    # +1 / -1: maps host rail_y (+Y) ↔ motor RPM and encoder metres together.
    sign: float = 1.0
    enable_settle_s: float = 0.2
    # Cold-start arming: worker must prove Modbus read+FA24=0 healthy before follow.
    arm_good_reads: int = 25  # consecutive healthy polls (~0.5 s @ 50 Hz)
    arm_settle_s: float = 0.5  # hold FA24=0 after good reads before ARMED
    arm_max_span_mm: float = 2.0  # encoder jitter allowed during arm window
    arm_timeout_s: float = 8.0
    poll_hz: float = 50.0
    deadband_mm: float = 0.5
    # FA23 + software FA24 clamp (r/min). 1800 @ 10 mm/rev = 0.30 m/s.
    max_speed_rpm: int = 1800
    busy_speed_rpm: int = 1
    # Encoder outside [-margin, travel+margin] → panic (FA24=0, follow off).
    fault_margin_m: float = 0.05
    # Soft position loop (rail metres) — empty-load 2 min FA24 demo / scan.
    vel_kp: float = 14.0  # 1/s (loaded first-pass value)
    vel_kd: float = 0.22  # dimensionless gain on velocity error
    vel_max_m_s: float = 0.30
    vel_amax_m_s2: float = 0.8  # softer slew vs Er-01 / host overshoot
    # Coupled-mode bounded catch-up of x_ref toward x_goal while moving.
    # Pure integration ratcheted 15.9 mm of e_shape over 84 s of gamepad.
    catch_v_max_m_s: float = 0.02
    catch_k: float = 5.0
    # Catch-up may not exceed this fraction of |v_goal|.  0.3 keeps it a
    # correction term so a 1.4 mm/s turn cannot kick 7x via catch_v_max.
    catch_frac: float = 0.3
    # Encoder-noise hysteresis for same-sign brake detection (m/s).
    decel_request_margin_m_s: float = 0.005
    # Live v_ff: position is a slow trim so PD cannot outrun FA40.
    vel_ff_p_trim_m_s: float = 0.010
    match_drive_accel: bool = True
    # Skip FA24 writes smaller than this (r/min) while moving.  12 ≈ 2 mm/s.
    fa24_rpm_deadband: int = 0
    vel_deadband_mm: float = 0.05
    # Standstill hysteresis: enter hold tightly, wake only if disturbed.
    # Tracking deadband stays tight; this freezes FA24 after settle so the
    # motor does not hum while fighting a sub-deadband residual forever.
    standstill_enter_mm: float = 0.05
    standstill_exit_mm: float = 0.25
    standstill_dwell_s: float = 0.08
    # Soft-end braking band (m).  Envelope is one-sided and anchors at soft
    # limits; this is a speed-limit margin, not a travel cut.
    approach_m: float = 0.040
    # Measurement + comms + accept.  Do not include FA41 (already in a_max).
    wall_reaction_s: float = 0.06
    vel_kd_max_m_s: float = 0.005
    # FA24 nonzero without a fresh encoder this long → hard kill.
    latch_watch_s: float = 0.12
    target_timeout_s: float = 0.10  # stale age before the stream is "old"
    # Extra coast after target_timeout before FA24=0.  A 127 ms QPIK hitch
    # must not hard-brake the carriage; only a true end-of-stream should.
    target_stale_coast_s: float = 0.35
    # Soft lag hold (FA24=0 this tick); does NOT DISARM.
    encoder_freeze_s: float = 1.0
    encoder_freeze_min_v_m_s: float = 0.02
    encoder_freeze_min_move_mm: float = 0.15
    # End-of-stream settle: close residual before releasing follow.
    settle_tol_mm: float = 0.05
    settle_v_m_s: float = 0.006
    settle_timeout_s: float = 1.5
    # Stall-safe speed: worst-case latched FA24 overshoot ≤ |err|.
    max_stall_s: float = 0.06
    stall_v_floor_m_s: float = 0.004
    # Run-time encoder jump: soft-reject above v_max·gap + margin; only a
    # hard leap (or repeated soft jumps) wipes calibration / DISARMs.
    jump_margin_mm: float = 3.0
    jump_hard_mm: float = 50.0
    jump_soft_streak_panic: int = 2
    accel_ms: int = 150  # FA40 — drive accel stays above host 0.8 m/s² limit
    decel_ms: int = 150  # FA41
    scurve_ms: int = 30  # FA42
    travel_m: float = 0.80
    timeout_s: float = 0.06
    retries: int = 1
    inter_frame_delay_s: float = 0.0005
    home_on_exit: bool = False
    # False (default): stop() leaves SON on (FA24=0 hold) so the next controller
    # start skips enable-edge wipe and keeps the absolute encoder frame.
    release_son_on_exit: bool = False
    home_speed_rpm: int = 900
    home_approach_mm: float = 40.0
    home_timeout_s: float = 60.0
    verbose: bool = False
    # Per-poll soft-loop CSV (debug). None = off. Window A -v / task params can set.
    log_csv: str | None = None

    def stream_dead_s(self) -> float:
        """Age after which a live follow stream is treated as ended.

        ``target_timeout_s`` marks the stream old; ``target_stale_coast_s``
        is extra coast so a 127 ms QPIK hitch does not hard-brake FA24.
        """
        timeout = max(float(self.target_timeout_s), 0.02)
        coast = max(0.0, float(self.target_stale_coast_s))
        return timeout + coast

    def live_host_accel_m_s2(self) -> float:
        """Host slew cap that cannot outrun FA40 on a live follow stream."""
        return live_host_accel_m_s2(
            vel_max_m_s=float(self.vel_max_m_s),
            accel_ms=float(self.accel_ms),
            configured_m_s2=float(self.vel_amax_m_s2),
            match_drive=bool(self.match_drive_accel),
        )


@dataclass(frozen=True)
class RailServoSample:
    """One time-aligned worker sample for diagnostics and acceptance tests."""

    sample_mono_s: float = float("nan")
    target_rx_mono_s: float = float("nan")
    motion_seq: int = 0
    x_goal_m: float = float("nan")
    x_ref_m: float = float("nan")
    x_meas_m: float = float("nan")
    v_goal_est_m_s: float = 0.0
    v_ref_m_s: float = 0.0
    a_ref_m_s2: float = 0.0
    v_meas_m_s: float = 0.0
    v_des_m_s: float = 0.0
    v_cmd_m_s: float = 0.0
    a_cmd_m_s2: float = 0.0
    x_goal_eval_m: float = float("nan")
    rpm_cmd: int = 0
    follow: bool = False
    armed: bool = False
    panic: bool = False
    poll_ok: bool = True
    mb_fail_n: int = 0
    freeze_flag: bool = False
    hold_count: int = 0
    hold_reason: str = ""
    command_mode: str = "position"
    feedback_valid: bool = False


class RailCommandMode(str, Enum):
    """Execution semantics for a rail command.

    ``COUPLED_VELOCITY`` is the 8-DOF QPIK stream: velocity is authoritative
    and position is only a travel/lead guard.  ``POSITION`` retains the old
    soft-CSP behaviour and settles at the requested position.
    """

    COUPLED_VELOCITY = "coupled_velocity"
    POSITION = "position"

    @classmethod
    def coerce(cls, value: "RailCommandMode | str | None") -> "RailCommandMode":
        if isinstance(value, cls):
            return value
        text = "" if value is None else str(value).strip().lower()
        aliases = {
            "coupled": cls.COUPLED_VELOCITY,
            "velocity": cls.COUPLED_VELOCITY,
            "coupled_velocity": cls.COUPLED_VELOCITY,
            "coupled-velocity": cls.COUPLED_VELOCITY,
            "position": cls.POSITION,
            "pos": cls.POSITION,
        }
        try:
            return aliases[text]
        except KeyError as exc:
            raise ValueError(
                f"unknown rail command mode {value!r}; expected "
                "'coupled_velocity' or 'position'"
            ) from exc


@dataclass(frozen=True)
class RailCommand:
    """Immutable command snapshot accepted by :class:`RailServoBridge`."""

    target_m: float
    v_ff_m_s: float
    mode: RailCommandMode
    rx_mono_s: float
    motion_seq: int

    @property
    def command_mode(self) -> RailCommandMode:
        return self.mode


@dataclass(frozen=True)
class RailExecutionFeedback:
    """Time-stamped rail execution feedback for QPIK.

    ``sample_age_s`` is measured when the snapshot is created.  A caller can
    use :meth:`is_fresh` with its own budget; no mutable bridge state is
    exposed through this object.
    """

    position_m: float = float("nan")
    v_meas_m_s: float = 0.0
    v_cmd_m_s: float = 0.0
    a_cmd_m_s2: float = 0.0
    sample_mono_s: float = float("nan")
    sample_age_s: float = float("inf")
    motion_seq: int = 0
    valid: bool = False
    command_mode: RailCommandMode = RailCommandMode.POSITION
    follow: bool = False
    armed: bool = False
    panic: bool = False

    @property
    def x_meas_m(self) -> float:
        return float(self.position_m)

    @property
    def freshness_s(self) -> float:
        return float(self.sample_age_s)

    @property
    def fresh(self) -> bool:
        return bool(
            bool(self.valid)
            and math.isfinite(float(self.sample_mono_s))
            and math.isfinite(float(self.sample_age_s))
            and float(self.sample_age_s) >= 0.0
        )

    def is_fresh(self, max_age_s: float) -> bool:
        budget = max(float(max_age_s), 0.0)
        return bool(self.fresh and float(self.sample_age_s) <= budget)


def parse_rail_servo_config(raw: dict) -> RailServoConfig:
    """Build ``RailServoConfig`` from joint admittance YAML (``hw.lw100``)."""
    hw = raw.get("hw", {}).get("lw100", {}) or {}
    rail = raw.get("inner", {}).get("rail", {}) or {}
    travel_m = float(rail.get("travel_m", 0.80))
    lead_mm = float(hw.get("lead_mm", 10.0))
    v_max = float(rail.get("v_max_m_s", 0.30))
    default_rpm = max(60, int(round(v_max * 1000.0 / max(lead_mm, 1e-6) * 60.0)))
    zero_mode = str(hw.get("zero_mode", "calibrated_file")).strip().lower()
    if zero_mode not in ("current", "fixed", "calibrated_file"):
        zero_mode = "calibrated_file"
    log_csv = hw.get("log_csv", None)
    log_csv_s = str(log_csv).strip() if log_csv else None
    cal_path = str(hw.get("calibration_path", "") or "").strip()
    # Canonical travel is qpik.hard_limits.rail (same as build_joint_ik_config).
    # inner.rail / hw.lw100 are fallbacks; any two that are set must match.
    qpik_rail = (
        (raw.get("qpik") or {}).get("hard_limits", {}) or {}
    )
    qpik_rail = qpik_rail.get("rail") or {}
    hw_soft_min = float(hw.get("soft_min_m", 0.015))
    hw_soft_max = float(hw.get("soft_max_m", 0.77))
    hw_hard_min = float(hw.get("hard_min_m", 0.005))
    hw_hard_max = float(hw.get("hard_max_m", 0.78))
    if "soft_min_m" in qpik_rail:
        soft_min = float(qpik_rail["soft_min_m"])
        soft_max = float(qpik_rail.get("soft_max_m", hw_soft_max))
    elif "soft_min_m" in rail or "soft_max_m" in rail:
        soft_min = float(rail.get("soft_min_m", hw_soft_min))
        soft_max = float(rail.get("soft_max_m", hw_soft_max))
    else:
        soft_min = hw_soft_min
        soft_max = hw_soft_max
    if "hard_min_m" in qpik_rail:
        hard_min = float(qpik_rail["hard_min_m"])
        hard_max = float(qpik_rail.get("hard_max_m", hw_hard_max))
    elif "hard_min_m" in rail or "hard_max_m" in rail:
        hard_min = float(rail.get("hard_min_m", hw_hard_min))
        hard_max = float(rail.get("hard_max_m", hw_hard_max))
    else:
        hard_min = hw_hard_min
        hard_max = hw_hard_max
    sources = []
    if "soft_min_m" in qpik_rail:
        sources.append(("qpik.hard_limits.rail", float(qpik_rail["soft_min_m"]),
                        float(qpik_rail.get("soft_max_m", soft_max))))
    if "soft_min_m" in rail or "soft_max_m" in rail:
        sources.append(("inner.rail", float(rail.get("soft_min_m", soft_min)),
                        float(rail.get("soft_max_m", soft_max))))
    if "soft_min_m" in hw or "soft_max_m" in hw:
        sources.append(("hw.lw100", hw_soft_min, hw_soft_max))
    for name, lo, hi in sources[1:]:
        if abs(lo - sources[0][1]) > 1.0e-6 or abs(hi - sources[0][2]) > 1.0e-6:
            raise ValueError(
                "rail soft-limit mismatch: "
                f"{sources[0][0]} [{sources[0][1]:.6f}, {sources[0][2]:.6f}] vs "
                f"{name} [{lo:.6f}, {hi:.6f}]"
            )
    hard_sources = []
    if "hard_min_m" in qpik_rail:
        hard_sources.append(
            (
                "qpik.hard_limits.rail",
                float(qpik_rail["hard_min_m"]),
                float(qpik_rail.get("hard_max_m", hard_max)),
            )
        )
    if "hard_min_m" in rail or "hard_max_m" in rail:
        hard_sources.append(
            (
                "inner.rail",
                float(rail.get("hard_min_m", hard_min)),
                float(rail.get("hard_max_m", hard_max)),
            )
        )
    if "hard_min_m" in hw or "hard_max_m" in hw:
        hard_sources.append(("hw.lw100", hw_hard_min, hw_hard_max))
    for name, lo, hi in hard_sources[1:]:
        if abs(lo - hard_sources[0][1]) > 1.0e-6 or abs(hi - hard_sources[0][2]) > 1.0e-6:
            raise ValueError(
                "rail hard-limit mismatch: "
                f"{hard_sources[0][0]} [{hard_sources[0][1]:.6f}, {hard_sources[0][2]:.6f}] vs "
                f"{name} [{lo:.6f}, {hi:.6f}]"
            )
    if not (0.0 <= hard_min <= soft_min < soft_max <= hard_max <= travel_m):
        raise ValueError(
            "invalid rail limits: expected "
            "0 <= hard_min <= soft_min < soft_max <= hard_max <= travel_m "
            f"({travel_m:.6f}), got soft=[{soft_min:.6f}, {soft_max:.6f}] "
            f"hard=[{hard_min:.6f}, {hard_max:.6f}]"
        )
    qpik_limits = (raw.get("qpik") or {}).get("hard_limits", {}) or {}
    hw_vel_max = float(hw.get("vel_max_m_s", v_max))
    qp_v_max = qpik_rail.get("v_max_m_s")
    if qp_v_max is not None:
        box_v = float(qp_v_max) * float(qpik_limits.get("v_scale", 1.0))
        vel_max_m_s = min(hw_vel_max, box_v)
    else:
        vel_max_m_s = hw_vel_max
    hw_a_max = float(hw.get("vel_amax_m_s2", 0.8))
    qp_a_max = qpik_limits.get("a_max_rail_m_s2")
    if qp_a_max is not None:
        vel_amax_m_s2 = min(hw_a_max, float(qp_a_max))
    else:
        vel_amax_m_s2 = hw_a_max
    standstill_enter_mm = max(float(hw.get("standstill_enter_mm", 0.05)), 0.01)
    standstill_exit_mm = max(
        float(hw.get("standstill_exit_mm", standstill_enter_mm * 5.0)),
        standstill_enter_mm,
    )
    standstill_dwell_s = max(float(hw.get("standstill_dwell_s", 0.08)), 0.0)
    return RailServoConfig(
        enabled=bool(hw.get("enabled", False)),
        host=str(hw.get("host", "192.168.0.7")),
        port=int(hw.get("port", 8234)),
        slave_id=int(hw.get("slave", hw.get("slave_id", 1))),
        lead_mm=lead_mm,
        zero_mode=zero_mode,
        counts0=int(hw.get("counts0", 0)),
        calibration_path=cal_path,
        require_calibration=bool(hw.get("require_calibration", True)),
        home_di=str(hw.get("home_di", "di3")),
        plus_di=str(hw.get("plus_di", "di4")),
        di_nc=bool(hw.get("di_nc", True)),
        di_debounce_n=int(hw.get("di_debounce_n", 3)),
        soft_min_m=soft_min,
        soft_max_m=soft_max,
        hard_min_m=hard_min,
        hard_max_m=hard_max,
        post_home_m=float(hw.get("post_home_m", soft_min)),
        limit_poll_every=max(1, int(hw.get("limit_poll_every", 5))),
        sign=float(hw.get("sign", 1.0)),
        enable_settle_s=float(hw.get("enable_settle_s", 0.2)),
        arm_good_reads=int(hw.get("arm_good_reads", 25)),
        arm_settle_s=float(hw.get("arm_settle_s", 0.5)),
        arm_max_span_mm=float(hw.get("arm_max_span_mm", 2.0)),
        arm_timeout_s=float(hw.get("arm_timeout_s", 8.0)),
        poll_hz=float(hw.get("poll_hz", 50.0)),
        deadband_mm=float(hw.get("deadband_mm", 0.5)),
        max_speed_rpm=int(hw.get("max_speed_rpm", default_rpm)),
        busy_speed_rpm=int(hw.get("busy_speed_rpm", 1)),
        fault_margin_m=float(hw.get("fault_margin_m", 0.05)),
        vel_kp=float(hw.get("vel_kp", 14.0)),
        vel_kd=float(hw.get("vel_kd", 0.22)),
        vel_max_m_s=vel_max_m_s,
        vel_amax_m_s2=vel_amax_m_s2,
        catch_v_max_m_s=float(hw.get("catch_v_max_m_s", 0.02)),
        catch_k=float(hw.get("catch_k", 5.0)),
        catch_frac=float(hw.get("catch_frac", 0.3)),
        decel_request_margin_m_s=float(hw.get("decel_request_margin_m_s", 0.005)),
        vel_ff_p_trim_m_s=float(hw.get("vel_ff_p_trim_m_s", 0.010)),
        match_drive_accel=bool(hw.get("match_drive_accel", True)),
        fa24_rpm_deadband=max(0, int(hw.get("fa24_rpm_deadband", 0))),
        vel_deadband_mm=float(hw.get("vel_deadband_mm", 0.05)),
        standstill_enter_mm=standstill_enter_mm,
        standstill_exit_mm=standstill_exit_mm,
        standstill_dwell_s=standstill_dwell_s,
        approach_m=float(hw.get("approach_m", 0.040)),
        wall_reaction_s=float(
            ((raw.get("inner") or {}).get("rail_allocator") or {}).get(
                "reaction_s", 0.06
            )
        ),
        vel_kd_max_m_s=float(hw.get("vel_kd_max_m_s", 0.005)),
        latch_watch_s=float(hw.get("latch_watch_s", 0.12)),
        target_timeout_s=float(hw.get("target_timeout_s", 0.10)),
        target_stale_coast_s=float(hw.get("target_stale_coast_s", 0.35)),
        encoder_freeze_s=float(hw.get("encoder_freeze_s", 1.0)),
        encoder_freeze_min_v_m_s=float(hw.get("encoder_freeze_min_v_m_s", 0.02)),
        encoder_freeze_min_move_mm=float(hw.get("encoder_freeze_min_move_mm", 0.15)),
        settle_tol_mm=float(hw.get("settle_tol_mm", 0.05)),
        settle_v_m_s=float(hw.get("settle_v_m_s", 0.006)),
        settle_timeout_s=float(hw.get("settle_timeout_s", 1.5)),
        max_stall_s=float(hw.get("max_stall_s", 0.06)),
        stall_v_floor_m_s=float(hw.get("stall_v_floor_m_s", 0.004)),
        jump_margin_mm=float(hw.get("jump_margin_mm", 3.0)),
        jump_hard_mm=float(hw.get("jump_hard_mm", 50.0)),
        jump_soft_streak_panic=int(hw.get("jump_soft_streak_panic", 2)),
        accel_ms=int(hw.get("accel_ms", 150)),
        decel_ms=int(hw.get("decel_ms", 150)),
        scurve_ms=int(hw.get("scurve_ms", 30)),
        travel_m=travel_m,
        timeout_s=float(hw.get("timeout_s", 0.06)),
        retries=int(hw.get("retries", 1)),
        inter_frame_delay_s=float(hw.get("inter_frame_delay_s", 0.0005)),
        home_on_exit=bool(hw.get("home_on_exit", False)),
        release_son_on_exit=bool(hw.get("release_son_on_exit", False)),
        home_speed_rpm=int(hw.get("home_speed_rpm", default_rpm)),
        home_approach_mm=float(hw.get("home_approach_mm", 40.0)),
        home_timeout_s=float(hw.get("home_timeout_s", 60.0)),
        verbose=bool(hw.get("verbose", False)),
        log_csv=log_csv_s or None,
    )


class _RailCsvLogger:
    """Per-poll rail soft-loop CSV (queued; never blocks the 50 Hz worker)."""

    _HEADER = (
        "t_wall_s,event,target_m,commanded_m,measured_m,"
        "v_ff,v_des,v_cmd,rpm,follow,armed,panic,poll_ok,"
        "dt_wall_ms,last_rpm_cmd,mb_fail_n,freeze_flag,arm_good,"
        "sample_mono_s,target_rx_mono_s,target_age_ms,motion_seq,feedback_valid,"
        "x_goal_m,x_ref_m,x_meas_m,v_goal_est_m_s,v_ref_m_s,a_ref_m_s2,"
        "v_reg_m_s,v_enc_m_s,v_enc_source,v_des_m_s,v_cmd_m_s,a_cmd_m_s2,x_goal_eval_m,"
        "rpm_cmd,e_track_mm,e_shape_mm,"
        "hold_count,hold_reason,command_mode,"
        "t_read_ms,t_write_ms,n_modbus,"
        "fa24_write_mono_ns,encoder_sample_mono_ns"
    ).split(",")

    def __init__(self, path: str) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        self._worker = threading.Thread(
            target=self._run, name="lw100-rail-csv", daemon=True
        )
        self._worker.start()

    def _run(self) -> None:
        with open(self.path, "w", newline="") as f:
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
                if n % 100 == 0:
                    f.flush()

    def write(
        self,
        *,
        event: str = "",
        target_m: float = float("nan"),
        commanded_m: float = float("nan"),
        measured_m: float = float("nan"),
        v_ff: float = float("nan"),
        v_des: float = float("nan"),
        v_cmd: float = float("nan"),
        rpm: float = float("nan"),
        follow: bool = False,
        armed: bool = False,
        panic: bool = False,
        poll_ok: bool = True,
        dt_wall_ms: float = float("nan"),
        last_rpm_cmd: int = 0,
        mb_fail_n: int = 0,
        freeze_flag: bool = False,
        arm_good: int = 0,
        sample_mono_s: float = float("nan"),
        target_rx_mono_s: float = float("nan"),
        motion_seq: int = 0,
        feedback_valid: bool = False,
        x_goal_m: float = float("nan"),
        x_ref_m: float = float("nan"),
        x_meas_m: float = float("nan"),
        v_goal_est_m_s: float = float("nan"),
        v_ref_m_s: float = float("nan"),
        a_ref_m_s2: float = float("nan"),
        v_reg_m_s: float = float("nan"),
        v_enc_m_s: float = float("nan"),
        v_enc_source: str = "",
        v_des_m_s: float = float("nan"),
        v_cmd_m_s: float = float("nan"),
        a_cmd_m_s2: float = float("nan"),
        x_goal_eval_m: float = float("nan"),
        rpm_cmd: int = 0,
        hold_count: int = 0,
        hold_reason: str = "",
        command_mode: str = "position",
        t_read_ms: float = float("nan"),
        t_write_ms: float = float("nan"),
        n_modbus: int = 0,
        fa24_write_mono_ns: int = 0,
        encoder_sample_mono_ns: int = 0,
    ) -> None:
        t_wall = time.monotonic() - self._t0

        def _f(v: float) -> str:
            return f"{v:.6f}" if math.isfinite(v) else ""

        target_age_ms = (
            (sample_mono_s - target_rx_mono_s) * 1000.0
            if math.isfinite(sample_mono_s)
            and math.isfinite(target_rx_mono_s)
            and target_rx_mono_s > 0.0
            else float("nan")
        )
        e_track_mm = (
            (x_ref_m - x_meas_m) * 1000.0
            if math.isfinite(x_ref_m) and math.isfinite(x_meas_m)
            else float("nan")
        )
        e_shape_mm = (
            (x_goal_eval_m - x_ref_m) * 1000.0
            if math.isfinite(x_goal_eval_m) and math.isfinite(x_ref_m)
            else (
                (x_goal_m - x_ref_m) * 1000.0
                if math.isfinite(x_goal_m) and math.isfinite(x_ref_m)
                else float("nan")
            )
        )

        self._q.put(
            [
                f"{t_wall:.4f}",
                str(event),
                _f(target_m),
                _f(commanded_m),
                _f(measured_m),
                _f(v_ff),
                _f(v_des),
                _f(v_cmd),
                _f(rpm),
                int(bool(follow)),
                int(bool(armed)),
                int(bool(panic)),
                int(bool(poll_ok)),
                _f(dt_wall_ms),
                int(last_rpm_cmd),
                int(mb_fail_n),
                int(bool(freeze_flag)),
                int(arm_good),
                _f(sample_mono_s),
                _f(target_rx_mono_s),
                _f(target_age_ms),
                int(motion_seq),
                int(bool(feedback_valid)),
                _f(x_goal_m),
                _f(x_ref_m),
                _f(x_meas_m),
                _f(v_goal_est_m_s),
                _f(v_ref_m_s),
                _f(a_ref_m_s2),
                _f(v_reg_m_s),
                _f(v_enc_m_s),
                str(v_enc_source),
                _f(v_des_m_s),
                _f(v_cmd_m_s),
                _f(a_cmd_m_s2),
                _f(x_goal_eval_m),
                int(rpm_cmd),
                _f(e_track_mm),
                _f(e_shape_mm),
                int(hold_count),
                str(hold_reason),
                str(command_mode),
                _f(t_read_ms),
                _f(t_write_ms),
                int(n_modbus),
                str(int(fa24_write_mono_ns)) if int(fa24_write_mono_ns) > 0 else "",
                (
                    str(int(encoder_sample_mono_ns))
                    if int(encoder_sample_mono_ns) > 0
                    else ""
                ),
            ]
        )

    def close(self) -> None:
        self._q.put(None)
        self._stop.set()
        self._worker.join(timeout=5.0)

class RailServoBridge:
    """LW100 tracker: WBC target → FA24 velocity; encoder → twin only."""

    def __init__(self, config: RailServoConfig) -> None:
        self.config = config
        self.enabled = bool(config.enabled)
        self._target_m = float("nan")
        self._target_v_ff_m_s = float("nan")
        self._command_mode = RailCommandMode.POSITION
        self._command_seq = 0
        self._commanded_m = float("nan")
        self._measured_m = float("nan")
        self._measured_speed_rpm = 0  # drive monitor 0x1000 (drive frame)
        self._measured_seq = 0  # bumps on every successful encoder/speed poll
        self._measured_mono_s = float("nan")
        self._servo_sample = RailServoSample()
        self._lock = threading.Lock()
        self._drive: LW100Drive | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._follow_enabled = False
        self._armed = False
        self._calibrated = False
        self._arm_req = threading.Event()  # set → worker restarts arming
        self._speed_cap_rpm: int | None = None
        self._panic = False
        self._panic_reason = ""
        self._wall_override_count = 0
        self._wall_override_last = False
        self._abort = threading.Event()
        self._last_target_rx_mono = 0.0
        self._target_history: deque[tuple[float, float]] = deque(maxlen=64)
        self._last_enc_ok_mono = 0.0
        self._last_fa24_write_mono_ns = 0
        self._last_encoder_sample_mono_ns = 0
        self._last_reject_unarmed_log = 0.0
        self._last_hold_log = 0.0
        self._last_hold_reason = ""
        self._last_hold_mono = 0.0
        self._hold_count = 0
        # Task-end / explicit hold: FA24=0 is not a position lock.  The
        # worker re-writes zero and re-anchors if the encoder still walks.
        self._hold_active = False
        self._hold_anchor_m = float("nan")
        self._hold_origin_m = float("nan")
        self._last_hold_zero_mono = 0.0
        self._last_hold_drift_log_mono = 0.0
        self._safety_thread: threading.Thread | None = None
        self._latch_kill_req = threading.Event()
        self._csv: _RailCsvLogger | None = None
        self._limit_poll_i = 0
        self._calibration_path: Path | None = None
        # False after a rejected mid-run leap — skip stop() cal rewrite only.
        # The taught zero JSON is never erased from the live worker.
        self._frame_continuous = True
        # True after TCP was torn (emergency_zero): next samples are restitch,
        # so leaps vs last-sane are rejected without touching calibration.
        self._link_restitch = False
        self._restitch_x_m = float("nan")
        self._restitch_v_m_s = float("nan")
        self._restitch_mono = 0.0
        if config.log_csv:
            self.enable_log_csv(str(config.log_csv))

    @property
    def log_csv_path(self) -> str | None:
        return None if self._csv is None else self._csv.path

    def enable_log_csv(self, path: str | None) -> str | None:
        """Start (or replace) the per-poll rail CSV logger. Returns path or None."""
        if not path:
            return self.log_csv_path
        path_s = str(path).strip()
        if not path_s:
            return self.log_csv_path
        if self._csv is not None and self._csv.path == path_s:
            return path_s
        if self._csv is not None:
            try:
                self._csv.close()
            except Exception:
                pass
            self._csv = None
        self._csv = _RailCsvLogger(path_s)
        self.config.log_csv = path_s
        print(f"lw100 rail: debug CSV → {path_s}", flush=True)
        return path_s

    def _log_event(self, event: str, **kwargs) -> None:
        if self._csv is None:
            return
        try:
            with self._lock:
                kwargs.setdefault("target_m", float(self._target_m))
                kwargs.setdefault("commanded_m", float(self._commanded_m))
                kwargs.setdefault("measured_m", float(self._measured_m))
                kwargs.setdefault("follow", bool(self._follow_enabled))
                kwargs.setdefault("armed", bool(self._armed))
                kwargs.setdefault("panic", bool(self._panic))
                kwargs.setdefault(
                    "command_mode", RailCommandMode(self._command_mode).value
                )
            self._csv.write(event=event, **kwargs)
        except Exception:
            pass

    def _encode_rail_m(self, drive_m: float) -> float:
        """Drive encoder metres → host ``rail_y`` (applies ``sign``)."""
        return float(self.config.sign) * float(drive_m)

    def _encode_speed_rpm(self, drive_rpm: int) -> int:
        """Drive monitor rpm → host rail direction (same ``sign`` as position)."""
        return int(round(float(self.config.sign) * float(drive_rpm)))

    def _publish_motion(
        self,
        host_m: float,
        host_speed_rpm: int,
        *,
        sample_mono_s: float | None = None,
    ) -> None:
        with self._lock:
            self._measured_m = float(host_m)
            self._measured_speed_rpm = int(host_speed_rpm)
            self._measured_seq = int(self._measured_seq) + 1
            self._measured_mono_s = (
                time.monotonic() if sample_mono_s is None else float(sample_mono_s)
            )

    @property
    def measured_m(self) -> float:
        with self._lock:
            return float(self._measured_m)

    @property
    def last_fa24_write_mono_ns(self) -> int:
        with self._lock:
            return int(self._last_fa24_write_mono_ns)

    @property
    def last_encoder_sample_mono_ns(self) -> int:
        with self._lock:
            return int(self._last_encoder_sample_mono_ns)

    @property
    def measured_speed_rpm(self) -> int:
        """Last drive-monitor speed (0x1000), host-signed (``sign`` applied)."""
        with self._lock:
            return int(self._measured_speed_rpm)

    @property
    def measured_speed_m_s(self) -> float:
        """Last drive-monitor speed converted to host-frame m/s."""
        with self._lock:
            rpm = float(self._measured_speed_rpm)
        return self._rpm_to_mps(rpm)

    @property
    def servo_sample(self) -> RailServoSample:
        """Latest worker-aligned goal/reference/feedback/control sample."""
        with self._lock:
            return self._servo_sample

    @property
    def commanded_m(self) -> float:
        with self._lock:
            return float(self._commanded_m)

    @property
    def panicked(self) -> bool:
        with self._lock:
            return bool(self._panic)

    @property
    def panic_reason(self) -> str:
        with self._lock:
            return str(self._panic_reason or "")

    @property
    def armed(self) -> bool:
        """True after cold-start Modbus+encoder health gate; follow allowed only then."""
        with self._lock:
            return bool(self._armed)

    @property
    def calibrated(self) -> bool:
        """True after a valid software zero is loaded (or debug current/fixed)."""
        with self._lock:
            return bool(self._calibrated)

    def _soft_lo_hi(self) -> tuple[float, float]:
        """Command snap box is the hard travel."""
        lo = float(self.config.hard_min_m)
        hi = float(self.config.hard_max_m)
        if hi <= lo:
            return 0.005, min(0.78, float(self.config.travel_m))
        return lo, hi

    def _envelope_lo_hi(self) -> tuple[float, float]:
        """Braking-envelope anchors: host soft limits."""
        lo = float(self.config.soft_min_m)
        hi = float(self.config.soft_max_m)
        hard_lo, hard_hi = self._soft_lo_hi()
        if hi <= lo:
            return hard_lo, hard_hi
        return max(lo, hard_lo), min(hi, hard_hi)

    def set_velocity_gains(
        self,
        *,
        kp: float | None = None,
        kd: float | None = None,
    ) -> tuple[float, float]:
        if kp is not None:
            self.config.vel_kp = float(kp)
        if kd is not None:
            self.config.vel_kd = float(kd)
        return float(self.config.vel_kp), float(self.config.vel_kd)

    def begin_tracking_session(self) -> None:
        """Discard stale SHM/target state before a new QPIK COUPLED session.

        Prevents inheriting a multi-second ``target_age`` / standstill hold
        from the previous Window-C task.
        """
        with self._lock:
            meas = float(self._measured_m)
            if not (math.isfinite(meas) and self._encoder_sane(meas)):
                meas = float(self._target_m) if math.isfinite(self._target_m) else 0.0
            now = time.monotonic()
            self._target_m = meas
            self._target_v_ff_m_s = float("nan")
            self._commanded_m = meas
            self._follow_enabled = False
            self._last_target_rx_mono = 0.0
            self._target_history.clear()
            self._target_history.append((now, meas))
            self._hold_count = 0
            self._hold_active = False
            self._hold_anchor_m = float("nan")
            self._hold_origin_m = float("nan")
        self._log_event(
            "session_begin",
            measured_m=meas,
            target_m=meas,
            commanded_m=meas,
            follow=False,
        )

    @property
    def target_v_ff_m_s(self) -> float:
        """Last QPIK rail velocity handed to the worker, or NaN if unused."""
        with self._lock:
            return float(self._target_v_ff_m_s)

    @property
    def command_mode(self) -> RailCommandMode:
        with self._lock:
            return RailCommandMode(self._command_mode)

    @property
    def command(self) -> RailCommand:
        with self._lock:
            return RailCommand(
                target_m=float(self._target_m),
                v_ff_m_s=float(self._target_v_ff_m_s),
                mode=RailCommandMode(self._command_mode),
                rx_mono_s=float(self._last_target_rx_mono),
                motion_seq=int(self._command_seq),
            )

    @property
    def execution_feedback(self) -> RailExecutionFeedback:
        """Latest rail execution sample for the QPIK kinematic snapshot."""
        now = time.monotonic()
        with self._lock:
            sample = self._servo_sample
            sample_t = float(sample.sample_mono_s)
            age = (
                max(0.0, now - sample_t)
                if math.isfinite(sample_t)
                else float("inf")
            )
            mode = RailCommandMode.coerce(sample.command_mode)
            return RailExecutionFeedback(
                position_m=float(sample.x_meas_m),
                v_meas_m_s=float(sample.v_meas_m_s),
                v_cmd_m_s=float(sample.v_cmd_m_s),
                a_cmd_m_s2=float(sample.a_cmd_m_s2),
                sample_mono_s=sample_t,
                sample_age_s=age,
                motion_seq=int(sample.motion_seq),
                valid=bool(sample.feedback_valid),
                command_mode=mode,
                follow=bool(sample.follow),
                armed=bool(sample.armed),
                panic=bool(sample.panic),
            )

    def set_target_m(
        self,
        target_m: float,
        v_ff_m_s: float | None = None,
        *,
        mode: RailCommandMode | str | None = None,
    ) -> bool:
        """Accept a rail goal and report whether it entered the follow buffer.

        ``v_ff_m_s`` is the QPIK rail velocity for this tick.  When finite the
        worker uses it as the authoritative velocity in
        :attr:`RailCommandMode.COUPLED_VELOCITY`; the target position is then
        only a travel/lead guard.
        """
        raw_v = float("nan") if v_ff_m_s is None else float(v_ff_m_s)
        if mode is None:
            command_mode = (
                RailCommandMode.COUPLED_VELOCITY
                if math.isfinite(raw_v)
                else RailCommandMode.POSITION
            )
        else:
            command_mode = RailCommandMode.coerce(mode)
        with self._lock:
            armed = bool(self._armed)
            calibrated = bool(self._calibrated)
            panic = bool(self._panic)
        if not calibrated:
            now = time.monotonic()
            if now - self._last_reject_unarmed_log >= 1.0:
                self._last_reject_unarmed_log = now
                print(
                    "lw100 rail: NOT CALIBRATED — ignore set_target "
                    "(run apps/lw100_rail_home_limit.py)",
                    flush=True,
                )
                self._log_event("reject_uncalibrated", target_m=float(target_m))
            return False
        if not armed:
            now = time.monotonic()
            if now - self._last_reject_unarmed_log >= 1.0:
                self._last_reject_unarmed_log = now
                print(
                    "lw100 rail: NOT READY — ignore set_target until ARMED "
                    "(Modbus/encoder warm-up)",
                    flush=True,
                )
                self._log_event("reject_unarmed", target_m=float(target_m))
            return False
        raw = float(target_m)
        soft_lo, soft_hi = self._soft_lo_hi()
        if not math.isfinite(raw):
            print(f"lw100 rail: reject non-finite target {raw}", flush=True)
            self._log_event("reject_nonfinite", target_m=raw)
            return False
        if raw < soft_lo - 0.005 or raw > soft_hi + 0.005:
            print(
                f"lw100 rail: reject target {raw * 1000:.1f} mm "
                f"(hard=[{soft_lo * 1000:.0f}, {soft_hi * 1000:.0f}] mm)",
                flush=True,
            )
            self._log_event("reject_oob", target_m=raw)
            return False
        snapped = max(soft_lo, min(soft_hi, raw))
        rx_mono = time.monotonic()
        with self._lock:
            # PANIC latches until explicit rearm (limit DI / encoder fault).
            # Do not auto-clear here — that let WBC resume while the arm kept moving.
            if panic or self._panic:
                return False
            self._target_m = snapped
            self._command_mode = command_mode
            self._command_seq = int(self._command_seq) + 1
            if command_mode is RailCommandMode.POSITION:
                self._target_v_ff_m_s = float("nan")
            else:
                self._target_v_ff_m_s = raw_v if math.isfinite(raw_v) else 0.0
            self._last_target_rx_mono = rx_mono
            self._target_history.append((rx_mono, snapped))
            self._follow_enabled = True
            self._hold_active = False
            self._hold_anchor_m = float("nan")
            self._hold_origin_m = float("nan")
        return True

    def hold_current(self) -> None:
        """Stop following; FA24=0. Keep last sane target (do not adopt insane encoder)."""
        with self._lock:
            meas = float(self._measured_m)
            if self._encoder_sane(meas):
                self._target_m = meas
                self._target_v_ff_m_s = float("nan")
                self._command_mode = RailCommandMode.POSITION
                self._command_seq = int(self._command_seq) + 1
                self._commanded_m = meas
                self._target_history.clear()
                self._target_history.append((time.monotonic(), meas))
                self._hold_anchor_m = meas
                self._hold_origin_m = meas
            else:
                self._hold_anchor_m = float("nan")
                self._hold_origin_m = float("nan")
            self._follow_enabled = False
            self._hold_active = True
            self._last_hold_zero_mono = 0.0
            self._last_hold_drift_log_mono = 0.0
        self.kill_motion()

    def hold_or_settle_after_task(self) -> bool:
        """Task-end: always snap-hold (FA24=0). Never re-open follow."""
        if not self.enabled or self._drive is None:
            return True
        with self._lock:
            meas = float(self._measured_m)
            target = float(self._target_m)
        if math.isfinite(meas) and math.isfinite(target) and self._encoder_sane(meas):
            err_mm = abs(target - meas) * 1000.0
            print(
                f"lw100 rail: task end hold (residual={err_mm:.2f} mm)",
                flush=True,
            )
        else:
            print("lw100 rail: task end hold (encoder/target invalid)", flush=True)
        self.hold_current()
        return True

    def _hold_watchdog(self, measured: float, now_s: float) -> None:
        """While follow is down, keep FA24=0 if the encoder walks.

        Velocity mode has no position lock.  Host skip-if-unchanged plus a
        forged ``_last_rpm_cmd=0`` is how 125211 crept at ~1 r/min after C
        exited.  Re-write zero every second; re-anchor at 2 mm.  A 5 mm
        walk only logs — do not PANIC/DISARM the whole controller.
        """
        if not self._hold_active or self._drive is None:
            return
        if now_s - float(self._last_hold_zero_mono) >= 1.0:
            try:
                self._drive.set_velocity_rpm(0, force=True)
            except Exception:
                pass
            self._last_hold_zero_mono = float(now_s)
        if not (math.isfinite(measured) and self._encoder_sane(measured)):
            return
        origin = float(self._hold_origin_m)
        anchor = float(self._hold_anchor_m)
        if math.isfinite(origin) and abs(measured - origin) > 0.005:
            if now_s - float(self._last_hold_drift_log_mono) >= 1.0:
                print(
                    f"lw100 rail: hold drift "
                    f"{abs(measured - origin) * 1000:.1f} mm "
                    f"(FA24 rewrite, stay ARMED)",
                    flush=True,
                )
                self._last_hold_drift_log_mono = float(now_s)
        if math.isfinite(anchor) and abs(measured - anchor) > 0.002:
            try:
                self._drive.set_velocity_rpm(0, force=True)
            except Exception:
                pass
            self._last_hold_zero_mono = float(now_s)
            with self._lock:
                self._hold_anchor_m = float(measured)

    def settle_and_hold(
        self,
        *,
        tol_mm: float | None = None,
        timeout_s: float | None = None,
    ) -> bool:
        """Close residual to last target (±tol), then freeze (FA24=0).

        Used when residual is large after a task. Returns True if settled
        within tolerance. Always ends in ``hold_current``.
        """
        if not self.enabled or self._drive is None:
            return True
        tol_m = max(float(self.config.settle_tol_mm if tol_mm is None else tol_mm), 0.01) * 1e-3
        timeout = max(float(self.config.settle_timeout_s if timeout_s is None else timeout_s), 0.1)
        deadline = time.monotonic() + timeout
        crawled = False
        with self._lock:
            can_settle = not (
                self._panic or not self._armed or not self._calibrated
            )
            # Keep last WBC target; refresh rx so worker does not drop follow.
            target = float(self._target_m)
            meas0 = float(self._measured_m)
            if can_settle:
                self._follow_enabled = True
                now = time.monotonic()
                self._last_target_rx_mono = now
                self._target_history.append((now, target))
        if not can_settle:
            self.hold_current()
            return False
        if math.isfinite(target) and math.isfinite(meas0) and abs(target - meas0) > tol_m:
            crawled = True
        while time.monotonic() < deadline:
            if self._abort.is_set() or self._stop.is_set():
                break
            with self._lock:
                if self._panic:
                    break
                meas = float(self._measured_m)
                target = float(self._target_m)
                self._follow_enabled = True
                now = time.monotonic()
                self._last_target_rx_mono = now
                self._target_history.append((now, target))
            if self._encoder_sane(meas) and abs(target - meas) <= tol_m:
                err_mm = abs(target - meas) * 1000.0
                print(
                    f"lw100 rail: settled residual={err_mm:.2f} mm "
                    f"(crawled={int(crawled)}); hold",
                    flush=True,
                )
                self.hold_current()
                return True
            time.sleep(0.02)
        with self._lock:
            meas = float(self._measured_m)
            target = float(self._target_m)
        err_mm = abs(target - meas) * 1000.0 if math.isfinite(target) and math.isfinite(meas) else float("nan")
        print(
            f"lw100 rail: settle timeout — residual={err_mm:.2f} mm "
            f"(tol={tol_m * 1000:.2f} mm, crawled={int(crawled)}); freezing",
            flush=True,
        )
        self.hold_current()
        return bool(math.isfinite(err_mm) and err_mm <= tol_m * 1000.0)

    def request_rearm(self) -> None:
        """Drop armed/panic/abort and ask the worker to re-prove Modbus health."""
        with self._lock:
            self._armed = False
            self._follow_enabled = False
            self._panic = False
            self._panic_reason = ""
        # Clear estop latch so a prior Ctrl+C/limit kill cannot block re-arm forever.
        self._abort.clear()
        self._arm_req.set()

    def limits_pressed(self) -> tuple[bool, bool]:
        """Live ``(di3, di4)`` pressed, or ``(False, False)`` if unreadable."""
        drive = self._drive
        if drive is None:
            return False, False
        try:
            return drive.read_limit_pressed(
                nc=bool(self.config.di_nc),
                debounce_n=max(1, min(3, int(self.config.di_debounce_n))),
                settle_s=0.01,
            )
        except Exception:
            return False, False

    def wait_limits_clear(self, *, timeout_s: float = 8.0) -> bool:
        """Block until both limit DIs are released (manual recovery after a trip)."""
        deadline = time.monotonic() + max(0.5, float(timeout_s))
        logged = False
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return False
            di3_p, di4_p = self.limits_pressed()
            if not di3_p and not di4_p:
                if logged:
                    print("lw100 rail: limits clear — continuing arming", flush=True)
                return True
            if not logged:
                which = "+".join(
                    [n for n, p in (("DI3", di3_p), ("DI4", di4_p)) if p]
                )
                print(
                    f"lw100 rail: waiting for limit release ({which}) — "
                    f"nudge carriage off the switch, then arming resumes",
                    flush=True,
                )
                logged = True
            time.sleep(0.1)
        di3_p, di4_p = self.limits_pressed()
        which = "+".join([n for n, p in (("DI3", di3_p), ("DI4", di4_p)) if p]) or "?"
        print(
            f"lw100 rail: limits still pressed ({which}) after "
            f"{float(timeout_s):.1f}s — refuse arming",
            flush=True,
        )
        return False

    def wait_until_armed(self, timeout_s: float | None = None) -> bool:
        """Block until worker marks ARMED, or timeout. Returns True if armed."""
        timeout = float(
            self.config.arm_timeout_s if timeout_s is None else timeout_s
        )
        deadline = time.monotonic() + max(0.5, timeout)
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return False
            # Abort may be cleared by request_rearm; do not treat a stale abort
            # as permanent failure once re-arm was requested.
            if self.armed:
                return True
            time.sleep(0.05)
        return bool(self.armed)

    def ensure_armed(self, *, timeout_s: float | None = None, rearm: bool = False) -> bool:
        """Guarantee rail is ARMED before any motion command / task START.

        After a limit DI panic: wait for the switch to clear, then re-arm.
        If already armed and ``rearm`` is False, returns immediately.
        """
        if not self.enabled:
            return True
        if not self.calibrated:
            print(MISSING_CAL_MSG, flush=True)
            return False
        timeout = float(self.config.arm_timeout_s if timeout_s is None else timeout_s)
        need = bool(rearm or self.panicked or not self.armed)
        if need:
            # Manual recovery after hard-limit trip: must be off the switch first.
            if not self.wait_limits_clear(timeout_s=min(timeout, 15.0)):
                return False
            self.request_rearm()
            print("lw100 rail: warming (Modbus read + FA24=0)…", flush=True)
        ok = self.wait_until_armed(timeout_s=timeout)
        if not ok:
            di3_p, di4_p = self.limits_pressed()
            extra = ""
            if di3_p or di4_p:
                which = "+".join(
                    [n for n, p in (("DI3", di3_p), ("DI4", di4_p)) if p]
                )
                extra = f" (limit still active: {which})"
            print(
                f"lw100 rail: NOT READY after {timeout:.1f}s "
                f"— refuse motion{extra}",
                flush=True,
            )
        return ok

    def _resolve_calibration_path(self) -> Path:
        raw = str(self.config.calibration_path or "").strip()
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                # Prefer rm75_control package root (parent of configs/).
                here = Path(__file__).resolve()
                pkg = here.parents[4]  # …/hw/rail_servo.py → rm75_control/
                p = (pkg / p).resolve()
            return p
        here = Path(__file__).resolve()
        pkg = here.parents[4]
        return default_calibration_path(pkg)

    def _apply_zero_at_start(self, drive: LW100Drive) -> str:
        """Load software zero. Sets ``_calibrated``. Raises if required cal missing."""
        mode = str(self.config.zero_mode).strip().lower()
        if mode == "fixed":
            counts0 = int(self.config.counts0)
            drive.set_rail_zero(counts0)
            with self._lock:
                self._calibrated = True
            return f"fixed counts0={counts0}"
        if mode == "current":
            if bool(self.config.require_calibration):
                raise RuntimeError(
                    "zero_mode=current is a debug bypass; set require_calibration: false "
                    "or use calibrated_file after apps/lw100_rail_home_limit.py"
                )
            counts0 = int(drive.set_rail_zero())
            with self._lock:
                self._calibrated = True
            return f"current-as-zero counts0={counts0}"

        # calibrated_file (default)
        path = self._resolve_calibration_path()
        self._calibration_path = path
        cal = load_calibration(path)
        if cal is None:
            with self._lock:
                self._calibrated = False
            print(MISSING_CAL_MSG, flush=True)
            raise CalValidationError("no valid calibration file", power_cycle=False)
        # Pose gate is the hard travel 5/780.  yaml soft 25/760 is only the
        # full-speed edge; older cal files store 25/780 as travel and must
        # still start.  780 is reachable.
        cal_gate = replace(
            cal,
            soft_min_m=float(self.config.hard_min_m),
            soft_max_m=float(self.config.hard_max_m),
        )
        ok, reason, host_m, power_cycle, comms_fail = validate_on_drive(
            drive,
            cal_gate,
            sign=float(self.config.sign),
            di_nc=bool(self.config.di_nc),
            home_di=str(self.config.home_di),
            plus_di=str(self.config.plus_di),
        )
        if not ok:
            with self._lock:
                self._calibrated = False
            if comms_fail:
                print(COMMS_FAIL_MSG, flush=True)
                print(f"lw100 rail: {reason}", flush=True)
                # Surface as Modbus so start() reconnect loop can retry.
                raise ModbusRtuError(reason)
            print(POWER_CYCLE_CAL_MSG if power_cycle else MISSING_CAL_MSG, flush=True)
            print(f"lw100 rail: {reason}", flush=True)
            raise CalValidationError(reason, power_cycle=power_cycle)
        cal.last_raw_counts = cal_gate.last_raw_counts
        try:
            save_calibration(path, cal)
        except OSError:
            pass
        with self._lock:
            self._calibrated = True
        return (
            f"calibrated_file counts0={cal.raw_counts0} "
            f"raw={cal.last_raw_counts} "
            f"host={host_m * 1000:.1f} mm"
        )

    def _invalidate_cal_after_frame_loss(self, reason: str) -> None:
        """Cold-start only: mark the taught zero unusable when bring-up fails.

        Mid-run leaps / Modbus stalls must not call this — they HOLD and keep
        the home zero file.  Clears the in-memory latch so WARN is not spammed.
        """
        with self._lock:
            already = not bool(self._calibrated) and not bool(self._frame_continuous)
            self._calibrated = False
            self._frame_continuous = False
        if already:
            return
        path = self._calibration_path or self._resolve_calibration_path()
        try:
            invalidate_calibration(path)
        except Exception:
            pass
        print(
            f"lw100 rail: WARN {reason} — calibration invalidated; "
            f"re-run apps/lw100_rail_home_limit.py --force before next start",
            flush=True,
        )

    def _resync_cal_frame_after_wipe(self, delta_bias: int, *, reason: str) -> None:
        """Trusted wipe (valid pre-read): keep live pose, re-pair JSON to new raw frame.

        Refuse to write if the live pose / raw looks corrupt (seen: FC-13/14 write
        leaving monitor at ~-62e6 → host kilometres). Only an untrusted jump should
        call ``_invalidate_cal_after_frame_loss``.
        """
        path = self._calibration_path or self._resolve_calibration_path()
        if path is None or self._drive is None:
            return
        try:
            raw_now = int(self._drive._read_encoder_counts_raw(retries=3))
            host_m = float(self._encode_rail_m(self._drive.read_rail_m_fast()))
        except Exception as exc:  # noqa: BLE001
            print(
                f"lw100 rail: WARN {reason} — post-wipe read failed ({exc}); "
                f"skip cal resync (Δbias={delta_bias})",
                flush=True,
            )
            return
        # Raw beyond ~1.2× travel is not a real rail pose (corrupt monitor).
        max_raw = int(
            abs(float(self.config.travel_m))
            / max(float(self.config.lead_mm) * 1e-3, 1e-9)
            * 131_072
            * 1.2
        )
        if abs(raw_now) > max_raw or not self._encoder_sane(host_m):
            print(
                f"lw100 rail: WARN {reason} — refuse cal resync "
                f"(raw={raw_now}, host={host_m * 1000:.1f} mm corrupt); "
                f"live bias kept, re-home before next cold start",
                flush=True,
            )
            self._frame_continuous = False
            return
        try:
            synced = sync_calibration_frame(
                path, self._drive, require_continuity=False
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"lw100 rail: WARN {reason} — cal resync failed ({exc}); "
                f"Δbias={delta_bias}",
                flush=True,
            )
            return
        self._frame_continuous = True
        if synced is not None:
            print(
                f"lw100 rail: encoder wipe during session (Δbias={delta_bias}) — "
                f"cal frame resynced counts0={synced.raw_counts0} "
                f"raw={synced.last_raw_counts} ({reason})",
                flush=True,
            )
        else:
            print(
                f"lw100 rail: WARN {reason} — cal resync returned None "
                f"(Δbias={delta_bias}); live pose still uses bias",
                flush=True,
            )

    def kill_motion(self) -> None:
        """Best-effort FA24=0. Prefer ``estop()`` from signal handlers (non-blocking)."""
        drive = self._drive
        if drive is None:
            return
        try:
            drive.kill_velocity_hard(attempts=2, disable_on_fail=False)
        except Exception:
            pass

    def estop(self) -> None:
        """Signal-safe stop: flags + drop TCP (unblocks Modbus). No Modbus write.

        Must not block in a signal handler: never wait on ``_lock`` (worker may
        hold it in ``recv``).  Flags + socket close are enough to stop FA24.
        """
        self._abort.set()
        self._stop.set()
        self._latch_kill_req.set()
        got = False
        try:
            got = bool(self._lock.acquire(blocking=False))
            if got:
                self._follow_enabled = False
                self._armed = False
        except Exception:
            pass
        finally:
            if got:
                try:
                    self._lock.release()
                except Exception:
                    pass
        drive = self._drive
        if drive is not None:
            # Do not forge ``_last_rpm_cmd=0`` — the worker skips the
            # Modbus write when the latch already says zero.
            try:
                drive._client.close()
            except Exception:
                pass

    def _encoder_sane(self, measured_m: float | None = None) -> bool:
        meas = float(self.measured_m if measured_m is None else measured_m)
        travel = float(self.config.travel_m)
        margin = max(float(self.config.fault_margin_m), 0.0)
        return math.isfinite(meas) and (-margin <= meas <= travel + margin)

    def _trip_panic(self, measured: float, reason: str) -> None:
        with self._lock:
            already = self._panic
            self._panic = True
            self._panic_reason = str(reason)
            self._follow_enabled = False
            self._armed = False
        # Avoid blocking Modbus from panic path when link may be dead.
        if not already:
            try:
                drive = self._drive
                if drive is not None and drive._client._sock is not None:
                    drive.kill_velocity_hard(attempts=1, disable_on_fail=False)
            except Exception:
                pass
            print(
                f"lw100 rail: PANIC — {reason} "
                f"(meas={measured * 1000:.1f} mm). FA24=0, DISARMED, task must stop.",
                flush=True,
            )
            last_rpm = 0
            try:
                last_rpm = int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
            except Exception:
                pass
            self._log_event(
                "PANIC",
                measured_m=float(measured),
                last_rpm_cmd=last_rpm,
                panic=True,
                armed=False,
                follow=False,
            )

    def _hold_velocity(self, measured: float, reason: str) -> None:
        """Soft fault: FA24=0 this tick, stay ARMED so follow resumes next good poll.

        Host-side hunting / brief Modbus lag must not permanently kill the rail —
        the drive itself is fine; only refuse to keep streaming velocity.
        """
        now = time.monotonic()
        with self._lock:
            self._last_hold_reason = str(reason)
            self._last_hold_mono = now
            self._hold_count += 1
        try:
            drive = self._drive
            if drive is not None and drive._client._sock is not None:
                drive.kill_velocity_hard(attempts=1, disable_on_fail=False)
        except Exception:
            pass
        if now - getattr(self, "_last_hold_log", 0.0) >= 1.0:
            self._last_hold_log = now
            print(
                f"lw100 rail: HOLD — {reason} "
                f"(meas={measured * 1000:.1f} mm; stay ARMED)",
                flush=True,
            )
            self._log_event(
                "HOLD",
                measured_m=float(measured),
                armed=True,
                follow=True,
                hold_count=self._hold_count,
                hold_reason=str(reason),
            )

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        drive_cfg = LW100DriveConfig(
            host=self.config.host,
            port=self.config.port,
            slave_id=self.config.slave_id,
            timeout_s=self.config.timeout_s,
            # Hot path: exactly 1 attempt.  Inflating retries (old max(2,…))
            # stacked timeouts into multi-second freezes with FA24 latched.
            retries=max(1, int(self.config.retries)),
            inter_frame_delay_s=self.config.inter_frame_delay_s,
            lead_mm=self.config.lead_mm,
            enable_settle_s=self.config.enable_settle_s,
            verbose=self.config.verbose,
        )
        self._drive = LW100Drive(drive_cfg)
        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                self._drive.connect()
                self._drive._client.recover()
                # Validate/apply software zero BEFORE velocity session. FA-60 /
                # FA61 / SON may wipe the multi-turn monitor; bias bookkeeping
                # + cal-file resync keep the zero continuous.
                zero_note = self._apply_zero_at_start(self._drive)
                self._frame_continuous = True
                self._link_restitch = False
                bias_before = int(self._drive._counts_bias)
                self._drive.start_velocity_session(
                    accel_ms=self.config.accel_ms,
                    decel_ms=self.config.decel_ms,
                    scurve_ms=self.config.scurve_ms,
                    max_speed_rpm=self.config.max_speed_rpm,
                )
                if not self._drive.frame_trusted:
                    print(FRAME_UNKNOWN_MSG, flush=True)
                    raise CalValidationError(
                        "encoder frame unknown after velocity session",
                        frame_unknown=True,
                    )
                bias_after = int(self._drive._counts_bias)
                if bias_after != bias_before:
                    delta = bias_after - bias_before
                    if self._drive.frame_trusted:
                        # Pre-read valid = our FA61/SON wipe; pose still exact.
                        self._resync_cal_frame_after_wipe(
                            delta,
                            reason="session start wipe",
                        )
                    else:
                        self._invalidate_cal_after_frame_loss(
                            f"encoder wiped during session start "
                            f"(Δbias={delta}, frame untrusted)"
                        )
                try:
                    self._drive.ensure_fa20_ignore()
                except Exception as exc:
                    print(f"lw100 rail: WARN FA-20={exc}", flush=True)
                try:
                    inner = self._drive.read_velocity_loop_params()
                    print(
                        "lw100 rail: drive velocity loop "
                        + " ".join(f"{name}={value}" for name, value in inner.items()),
                        flush=True,
                    )
                    self._log_event(
                        "DRIVE_VELOCITY_LOOP "
                        + " ".join(f"{name}={value}" for name, value in inner.items())
                    )
                except ModbusRtuError as exc:
                    print(f"lw100 rail: WARN read FA5/6/7/8 failed ({exc})", flush=True)
                last_err = None
                break
            except ModbusRtuError as exc:
                last_err = exc
                print(
                    f"lw100 rail: start attempt {attempt}/3 failed ({exc}); "
                    "reconnecting…",
                    flush=True,
                )
                try:
                    self._drive._client.reconnect()
                except Exception:
                    try:
                        self._drive.close()
                    except Exception:
                        pass
                    self._drive = LW100Drive(drive_cfg)
                time.sleep(0.2)
            except (CalValidationError, RuntimeError):
                # Missing/invalid calibration — do not retry as Modbus.
                try:
                    if self._drive is not None:
                        self._drive.set_velocity_rpm(0, force=True)
                        self._drive.disable()
                        self._drive.close()
                except Exception:
                    pass
                self._drive = None
                raise
        if last_err is not None:
            raise ModbusRtuError(f"lw100 rail: start failed: {last_err}") from last_err

        # Pre-check encoder before worker; follow stays off until ARMED.
        samples: list[float] = []
        for _ in range(8):
            samples.append(self._encode_rail_m(self._drive.read_rail_m_fast()))
            time.sleep(0.02)
        measured = float(samples[-1])
        if not self._encoder_sane(measured):
            self._drive.set_velocity_rpm(0, force=True)
            with self._lock:
                self._calibrated = False
            # Poisoned resync / corrupt monitor — do not keep a bad JSON.
            self._invalidate_cal_after_frame_loss(
                f"encoder out of range at start (meas={measured * 1000:.1f} mm)"
            )
            print(MISSING_CAL_MSG, flush=True)
            raise RuntimeError(
                f"lw100 rail: encoder out of range at start "
                f"(meas={measured * 1000:.1f} mm, travel={self.config.travel_m * 1000:.0f} mm) "
                f"— re-run apps/lw100_rail_home_limit.py"
            )
        span = max(samples) - min(samples)
        if span > 0.005:
            print(
                f"lw100 rail: WARN encoder unsettled at start "
                f"(span={span * 1000:.1f} mm); will re-check during arming",
                flush=True,
            )

        try:
            rpm0, _ = self._drive.read_motion_fast()
            self._publish_motion(measured, self._encode_speed_rpm(rpm0))
        except Exception:
            self._publish_motion(measured, 0)
        try:
            raw = int(self._drive._read_encoder_counts_raw(retries=1))
        except Exception:
            raw = -1
        with self._lock:
            self._commanded_m = measured
            self._target_m = measured
            self._target_history.clear()
            self._target_history.append((time.monotonic(), measured))
            self._follow_enabled = False
            self._armed = False
            self._panic = False
            self._panic_reason = ""
            self._speed_cap_rpm = None
            self._last_target_rx_mono = 0.0
            self._servo_sample = RailServoSample(
                sample_mono_s=float(self._measured_mono_s),
                target_rx_mono_s=0.0,
                motion_seq=int(self._measured_seq),
                x_goal_m=measured,
                x_ref_m=measured,
                x_meas_m=measured,
            )
            self._last_hold_reason = ""
            self._last_hold_mono = 0.0
            self._hold_count = 0
        self._stop.clear()
        self._abort.clear()
        self._arm_req.set()  # worker begins arming immediately
        self._last_enc_ok_mono = time.monotonic()
        self._thread = threading.Thread(
            target=self._worker_velocity, name="lw100-rail", daemon=True
        )
        self._safety_thread = threading.Thread(
            target=self._latch_safety_watchdog, name="lw100-rail-safety", daemon=True
        )
        self._thread.start()
        self._safety_thread.start()
        hard_lo, hard_hi = self._soft_lo_hi()
        print(
            f"lw100 rail: connecting hold @ {measured:+.4f} m ({zero_note}, "
            f"raw={raw} bias={self._drive._counts_bias}, "
            f"hard=[{hard_lo * 1000:.0f}, {hard_hi * 1000:.0f}] mm "
            f"travel={self.config.travel_m:.2f} m, "
            f"velocity-follow (kp={self.config.vel_kp}, kd={self.config.vel_kd}, "
            f"v_max={self.config.vel_max_m_s:.2f} m/s, "
            f"a_max={self.config.vel_amax_m_s2:.2f} m/s², "
            f"poll={self.config.poll_hz:.0f}Hz, "
            f"FA23={self.config.max_speed_rpm}, FA40/41={self.config.accel_ms}ms), "
            f"home_on_exit={self.config.home_on_exit}) — warming…",
            flush=True,
        )
        if not self.ensure_armed(timeout_s=self.config.arm_timeout_s, rearm=False):
            self.stop(home=False)
            raise RuntimeError(
                "lw100 rail: cold-start arming failed — refuse to accept motion"
            )

    def go_home(self, *, timeout_s: float | None = None) -> bool:
        """Command rail to ``post_home_m`` (soft park, not mechanical zero)."""
        if not self.enabled or self._drive is None:
            return True
        if self._thread is None or not self._thread.is_alive():
            return abs(self.measured_m - float(self.config.post_home_m)) * 1000.0 <= float(
                self.config.deadband_mm
            )

        if not self.ensure_armed(timeout_s=self.config.arm_timeout_s):
            print("lw100 rail: SKIP home — rail NOT READY", flush=True)
            self.kill_motion()
            return False

        meas0 = self.measured_m
        if not self._encoder_sane(meas0):
            print(
                f"lw100 rail: SKIP home — encoder out of range "
                f"(meas={meas0 * 1000:.1f} mm)",
                flush=True,
            )
            self.kill_motion()
            return False

        target = float(self.config.post_home_m)
        soft_lo, soft_hi = self._soft_lo_hi()
        target = max(soft_lo, min(soft_hi, target))
        timeout = float(self.config.home_timeout_s if timeout_s is None else timeout_s)
        with self._lock:
            self._panic = False
            self._speed_cap_rpm = int(self.config.home_speed_rpm)
        self._abort.clear()
        self.set_target_m(target)
        print(
            f"lw100 rail: park to {target * 1000:.0f} mm (timeout={timeout:.0f}s, "
            f"cruise≤{self.config.home_speed_rpm} r/min)…",
            flush=True,
        )
        deadband_m = float(self.config.deadband_mm) * 1e-3
        deadline = time.monotonic() + max(0.5, timeout)
        ok = False
        last_log = 0.0
        while time.monotonic() < deadline:
            if self._abort.is_set() or self._stop.is_set():
                self.kill_motion()
                with self._lock:
                    self._follow_enabled = False
                    self._speed_cap_rpm = None
                print("lw100 rail: home ABORTED", flush=True)
                return False
            meas = self.measured_m
            if not self._encoder_sane(meas):
                self._trip_panic(meas, "encoder left travel band during home")
                with self._lock:
                    self._speed_cap_rpm = None
                return False
            cmd = self.commanded_m
            try:
                busy = self._drive.is_busy(speed_threshold_rpm=self.config.busy_speed_rpm)
            except ModbusRtuError:
                busy = True
            if abs(meas - target) <= deadband_m and not busy:
                ok = True
                break
            if abs(cmd - target) <= deadband_m and abs(meas - target) <= 5.0 * deadband_m and not busy:
                ok = True
                break
            now = time.monotonic()
            if now - last_log >= 2.0:
                last_log = now
                print(
                    f"lw100 rail: park… meas={meas * 1000:.1f} mm cmd={cmd * 1000:.1f} mm "
                    f"busy={busy}",
                    flush=True,
                )
            time.sleep(0.05)
        self.hold_current()
        with self._lock:
            self._speed_cap_rpm = None
        print(
            f"lw100 rail: park {'OK' if ok else 'TIMEOUT'} @ {self.measured_m:+.4f} m "
            f"(cmd={self.commanded_m:+.4f} m)",
            flush=True,
        )
        return ok

    def stop(self, *, home: bool | None = None) -> None:
        """Stop worker quickly; optional home only if encoder in-band and link up."""
        do_home = self.config.home_on_exit if home is None else bool(home)
        if do_home and self._drive is not None and self._thread is not None:
            if self.panicked:
                print("lw100 rail: SKIP home on exit — rail is panicked", flush=True)
            elif self._encoder_sane():
                try:
                    self.go_home()
                except Exception as exc:
                    print(f"lw100 rail: WARN home on exit failed: {exc}", flush=True)
            else:
                print(
                    f"lw100 rail: SKIP home on exit — encoder out of range "
                    f"(meas={self.measured_m * 1000:.1f} mm); disabling only",
                    flush=True,
                )

        self._abort.set()
        self._stop.set()
        with self._lock:
            self._follow_enabled = False
            self._armed = False

        # Stop motion, then join the worker.  Prefer an in-band FA24=0; only
        # tear TCP when the stream is wedged.  Never rewrite the zero file
        # after a TCP tear — that sample is not a calibration event.
        drive = self._drive
        tore_link = False
        if drive is not None:
            try:
                drive.kill_velocity_hard(attempts=2, disable_on_fail=False)
            except Exception:
                pass
            if int(getattr(drive, "_last_rpm_cmd", 0) or 0) != 0:
                try:
                    drive.emergency_zero_fa24()
                except Exception:
                    pass
                tore_link = True
            try:
                drive._client.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=0.6)
            self._thread = None
        if self._safety_thread is not None:
            self._safety_thread.join(timeout=0.3)
            self._safety_thread = None
        if self._drive is not None:
            try:
                self._drive._client.connect()
                try:
                    self._drive.set_velocity_rpm(0, force=True)
                except Exception:
                    pass

                can_snapshot = (
                    self._calibration_path is not None
                    and self.calibrated
                    and self._drive.frame_trusted
                    and self._frame_continuous
                    and not tore_link
                    and not bool(self._panic)
                    and not bool(self._link_restitch)
                )
                if can_snapshot:
                    try:
                        synced = sync_calibration_frame(
                            self._calibration_path,
                            self._drive,
                            require_continuity=True,
                        )
                    except Exception:
                        synced = None
                    if synced is None:
                        print(
                            "lw100 rail: WARN stop() calibration snapshot skipped "
                            "(read/continuity); existing zero file retained",
                            flush=True,
                        )
                elif tore_link or bool(self._link_restitch) or bool(self._panic):
                    print(
                        "lw100 rail: stop() keeps existing zero file "
                        "(no mid-run calibration rewrite)",
                        flush=True,
                    )

                # Hold SON by default (FA24=0) so the next start does not
                # edge-enable and wipe multi-turn.
                if bool(self.config.release_son_on_exit):
                    self._drive.disable()
                else:
                    # Keep velocity session flag consistent with live SON.
                    self._drive._disable_on_exit = False  # noqa: SLF001
                    print(
                        "lw100 rail: SON held (FA24=0) — start controller again "
                        "without power-cycling; use release_son_on_exit to drop SON",
                        flush=True,
                    )
            except Exception:
                pass
            try:
                self._drive.close()
            except Exception:
                pass
            self._drive = None
        if self._csv is not None:
            try:
                self._log_event("STOP")
                self._csv.close()
            except Exception:
                pass
            self._csv = None

    def _latch_safety_watchdog(self) -> None:
        """If we are commanding velocity but encoder feed is dark, stop motion.

        Policy (intentionally simple):
        - Commanding + no feedback → FA24=0 and HOLD (stay ARMED).
        - Never DISARM / never touch the zero file from this path.
        - TCP tear marks ``_link_restitch`` so the next encoder samples are
          accepted only if continuous with the last sane host pose.
        """
        dark_s = max(float(self.config.latch_watch_s), 0.0)
        while not self._stop.wait(0.05):
            drive = self._drive
            if drive is None:
                continue
            last_rpm = int(getattr(drive, "_last_rpm_cmd", 0) or 0)
            if abs(last_rpm) <= 0:
                continue
            age = time.monotonic() - float(self._last_enc_ok_mono)
            if age <= dark_s:
                continue
            with self._lock:
                self._restitch_x_m = float(self._measured_m)
                self._restitch_v_m_s = self._rpm_to_mps(
                    float(self._measured_speed_rpm)
                )
                self._restitch_mono = time.monotonic()
            self._latch_kill_req.set()
            self._link_restitch = True
            try:
                drive.emergency_zero_fa24()
            except Exception:
                pass

    def _mps_to_rpm(self, v_m_s: float) -> float:
        lead = max(float(self.config.lead_mm), 1e-6)
        return float(v_m_s) * 1000.0 / lead * 60.0

    def _rpm_to_mps(self, rpm: float) -> float:
        lead = max(float(self.config.lead_mm), 1e-6)
        return float(rpm) / 60.0 * lead * 1e-3

    @staticmethod
    def _encoder_velocity(
        samples: Sequence[tuple[float, float]],
        *,
        poll_hz: float,
        fallback_m_s: float,
        period_s: float | None = None,
        hold_m_s: float = float("nan"),
        hold_budget: int = 0,
    ) -> tuple[float, str]:
        """Least-squares slope of the accepted encoder samples.

        Returns ``(velocity_m_s, source)`` with source in ``lsq`` / ``hold``
        / ``reg``.  The window is sized from ``period_s`` (the worker's
        measured poll period) rather than the nominal ``poll_hz``: run
        225941 polled at 56 Hz against a nominal 60, so a fixed
        ``3 / poll_hz`` window rejected 11.2% of the ticks and hard-switched
        the D term back to the 157 ms-lagged drive register.  Slope over the
        whole window also averages down the encoder quantisation and the
        Modbus timestamp jitter that a two-point difference amplifies.

        Repeated positions give 0 (not a spike).  When no window qualifies
        the previous value is held for ``hold_budget`` ticks before the
        register value is used, so a single dropped poll is not a step into
        the derivative.
        """
        period = float(period_s) if period_s is not None else float("nan")
        if not (math.isfinite(period) and period > 1.0e-6):
            period = 1.0 / max(float(poll_hz), 1.0)
        lo = 0.5 * period
        hi = 5.0 * period

        def _degraded() -> tuple[float, str]:
            if int(hold_budget) > 0 and math.isfinite(float(hold_m_s)):
                return float(hold_m_s), "hold"
            return float(fallback_m_s), "reg"

        if len(samples) < 2:
            return _degraded()
        t_new, x_new = float(samples[-1][0]), float(samples[-1][1])
        if not (math.isfinite(t_new) and math.isfinite(x_new)):
            return _degraded()
        window: list[tuple[float, float]] = []
        for t_s, x_s in reversed(samples):
            t_f, x_f = float(t_s), float(x_s)
            if not (math.isfinite(t_f) and math.isfinite(x_f)):
                break
            age = t_new - t_f
            if age < 0.0 or age > hi:
                break
            window.append((t_f, x_f))
        if len(window) < 2:
            return _degraded()
        span = t_new - window[-1][0]
        if span < lo:
            return _degraded()
        n = float(len(window))
        t_bar = sum(p[0] for p in window) / n
        x_bar = sum(p[1] for p in window) / n
        s_tt = sum((p[0] - t_bar) ** 2 for p in window)
        if s_tt <= 1.0e-12:
            return _degraded()
        s_tx = sum((p[0] - t_bar) * (p[1] - x_bar) for p in window)
        return s_tx / s_tt, "lsq"

    @staticmethod
    def _motion_from_candidates(
        *candidates: float,
        zero_eps: float = RAIL_IDLE_EPS_M_S,
    ) -> float:
        """First finite candidate whose magnitude exceeds ``zero_eps``."""
        eps = max(float(zero_eps), 0.0)
        for candidate in candidates:
            value = float(candidate)
            if math.isfinite(value) and abs(value) >= eps:
                return value
        return 0.0

    @staticmethod
    def _is_decel_request(
        v_goal: float,
        v_motion: float,
        *,
        zero_eps: float = RAIL_IDLE_EPS_M_S,
        margin: float = 0.005,
    ) -> bool:
        """True when the goal is a same-sign slowdown (including stop).

        An opposite-sign ``v_goal`` is an explicit reverse and is not a
        brake request.  ``margin`` covers encoder-difference noise so a
        cruise tick with |v_goal| slightly below |v_enc| still counts as
        a brake (root cause B).
        """
        vg = float(v_goal)
        vm = float(v_motion)
        eps = max(float(zero_eps), 0.0)
        if not (math.isfinite(vg) and math.isfinite(vm)):
            return False
        if abs(vm) < eps:
            return False
        if vg * vm < 0.0:
            return False
        return abs(vg) <= abs(vm) + max(float(margin), 0.0)

    @staticmethod
    def _estimate_goal_motion(
        samples: Sequence[tuple[float, float]],
        *,
        now_s: float,
        max_age_s: float,
        window_s: float = 0.10,
        stationary_span_m: float = 0.00002,
    ) -> tuple[float, float, bool]:
        """Return time-aligned position, local velocity, and stationary state."""
        if not samples:
            return float("nan"), 0.0, False
        cutoff = float(now_s) - max(float(window_s), 1.0e-3)
        recent = [(float(t), float(x)) for t, x in samples if float(t) >= cutoff]
        if len(recent) < 3:
            return float(samples[-1][1]), 0.0, False
        stationary_limit = max(float(stationary_span_m), 0.0)
        tail = recent[-3:]
        stationary = all(
            abs(tail[i][1] - tail[i - 1][1]) <= stationary_limit
            for i in range(1, len(tail))
        )
        fit = tail if stationary else recent
        t_mean = sum(t for t, _ in fit) / len(fit)
        x_mean = sum(x for _, x in fit) / len(fit)
        denom = sum((t - t_mean) ** 2 for t, _ in fit)
        velocity = (
            0.0
            if denom <= 1.0e-12
            else sum((t - t_mean) * (x - x_mean) for t, x in fit) / denom
        )
        goal = tail[-1][1]
        dx0 = tail[1][1] - tail[0][1]
        dx1 = tail[2][1] - tail[1][1]
        if (
            not stationary
            and abs(dx0) > stationary_limit
            and abs(dx1) > stationary_limit
            and dx0 * dx1 > 0.0
        ):
            age_s = min(
                max(0.0, float(now_s) - tail[-1][0]),
                max(float(max_age_s), 0.0),
            )
            goal += velocity * age_s
        return goal, velocity, stationary

    @staticmethod
    def _resolve_stream_goal(
        samples: Sequence[tuple[float, float]],
        *,
        now_s: float,
        max_age_s: float,
        target_m: float,
        last_rx_s: float,
        v_ff_m_s: float,
    ) -> tuple[float, float, bool]:
        """Prefer QPIK ``v_ff``; fall back to differentiating the position stream."""
        if math.isfinite(v_ff_m_s):
            age_s = 0.0
            if last_rx_s > 0.0 and math.isfinite(now_s):
                age_s = min(
                    max(0.0, float(now_s) - float(last_rx_s)),
                    max(float(max_age_s), 0.0),
                )
            goal = float(target_m) + float(v_ff_m_s) * age_s
            return goal, float(v_ff_m_s), abs(float(v_ff_m_s)) < RAIL_IDLE_EPS_M_S
        return RailServoBridge._estimate_goal_motion(
            samples,
            now_s=now_s,
            max_age_s=max_age_s,
        )

    @staticmethod
    def _step_reference(
        x_ref: float,
        v_ref: float,
        x_goal: float,
        v_goal: float,
        *,
        stationary: bool,
        dt: float,
        v_max: float,
        a_max: float,
    ) -> tuple[float, float, float]:
        """One bounded tracking step for streamed and static position goals."""
        dt = max(float(dt), 1.0e-4)
        v_max = max(float(v_max), 1.0e-6)
        a_max = max(float(a_max), 1.0e-6)
        v_goal = max(-v_max, min(v_max, float(v_goal)))
        err = float(x_goal) - float(x_ref)
        catch_speed = min(
            a_max / v_max * abs(err),
            math.sqrt(2.0 * a_max * abs(err)),
        )
        v_catch = math.copysign(catch_speed, err) if abs(err) > 1.0e-12 else 0.0
        v_des = max(-v_max, min(v_max, v_goal + v_catch))
        if v_goal * v_des < 0.0:
            v_des = 0.0
        dv_max = a_max * dt
        v_new = max(v_ref - dv_max, min(v_ref + dv_max, v_des))
        v_new = max(-v_max, min(v_max, v_new))
        x_new = float(x_ref) + v_new * dt
        a_new = (v_new - float(v_ref)) / dt
        if (
            stationary
            and abs(float(x_goal) - x_new) <= 0.00002
            and abs(v_new) < RAIL_IDLE_EPS_M_S
        ):
            return float(x_goal), 0.0, 0.0
        return x_new, v_new, a_new

    @staticmethod
    def _step_velocity_reference(
        x_ref: float,
        v_ref: float,
        v_goal: float,
        *,
        dt: float,
        v_max: float,
        a_max: float,
        x_goal: float | None = None,
        catch_v_max: float = 0.0,
        k_catch: float = 0.0,
        catch_frac: float = 0.3,
        x_min: float | None = None,
        x_max: float | None = None,
    ) -> tuple[float, float, float]:
        """Advance a velocity-authoritative reference with bounded catch-up.

        Catch-up is a correction on top of ``v_goal``.  Its cap is
        ``min(catch_v_max, catch_frac*|v_goal|)``, so a parked or near-zero
        goal cannot fire a 7x kick.  Parked ticks still re-anchor ``x_ref``.
        """
        dt = max(float(dt), 1.0e-4)
        v_max = max(float(v_max), 1.0e-6)
        a_max = max(float(a_max), 1.0e-6)
        v_goal = max(-v_max, min(v_max, float(v_goal)))
        v_catch = 0.0
        if x_goal is not None and math.isfinite(float(x_goal)):
            err = float(x_goal) - float(x_ref)
            cap = min(
                max(float(catch_v_max), 0.0),
                max(float(catch_frac), 0.0) * abs(v_goal),
            )
            gain = max(float(k_catch), 0.0)
            v_catch = max(-cap, min(cap, gain * err))
        v_target = max(-v_max, min(v_max, v_goal + v_catch))
        dv_max = a_max * dt
        v_new = max(float(v_ref) - dv_max, min(float(v_ref) + dv_max, v_target))
        v_new = max(-v_max, min(v_max, v_new))
        x_new = float(x_ref) + v_new * dt
        if x_min is not None and x_new < float(x_min):
            x_new = float(x_min)
            if v_new < 0.0:
                v_new = 0.0
        if x_max is not None and x_new > float(x_max):
            x_new = float(x_max)
            if v_new > 0.0:
                v_new = 0.0
        a_new = (v_new - float(v_ref)) / dt
        return x_new, v_new, a_new

    @staticmethod
    def _parked_reanchor(
        x_ref: float,
        v_ref: float,
        a_ref: float,
        *,
        measured: float,
        v_goal: float,
        v_meas: float,
        zero_eps: float = RAIL_IDLE_EPS_M_S,
    ) -> tuple[float, float, float, bool]:
        """Wipe P-term debt when the coupled stream is standing still.

        Orthogonal to standstill hysteresis (FA24 hold): this only snaps
        ``x_ref`` to ``measured`` so ``v_p = kp*(x_ref−x_meas)`` is zero
        on the release tick.  It does not move the carriage.
        """
        parked = (
            abs(float(v_goal)) < float(zero_eps)
            and abs(float(v_meas)) < float(zero_eps)
            and abs(float(v_ref)) < float(zero_eps)
        )
        if parked:
            return float(measured), 0.0, 0.0, True
        return float(x_ref), float(v_ref), float(a_ref), False

    @staticmethod
    def _clamp_zero_target_brake(
        v_des: float,
        *,
        v_goal: float,
        v_ref: float,
        v_meas: float,
        v_prev_cmd: float,
        zero_eps: float = RAIL_IDLE_EPS_M_S,
        margin: float = 0.005,
    ) -> float:
        """Do not turn a deceleration / stop into an active reversal.

        Direction comes from actual motion (``v_meas`` first — encoder
        difference after the 157 ms register lag fix), then ``v_ref``, then
        the previous command.  Engages for the whole same-sign slowdown
        (``v_goal * v_motion >= 0`` and ``|v_goal| <= |v_motion|``), not
        only when ``v_goal≈0``.  An opposite-sign ``v_goal`` is a real
        reverse and is left alone.
        """
        desired = float(v_des)
        v_motion = RailServoBridge._motion_from_candidates(
            v_meas, v_ref, v_prev_cmd, zero_eps=zero_eps
        )
        if abs(v_motion) < max(float(zero_eps), 0.0):
            if abs(float(v_goal)) < max(float(zero_eps), 0.0):
                return 0.0
            return desired
        if not RailServoBridge._is_decel_request(
            v_goal, v_motion, zero_eps=zero_eps, margin=margin
        ):
            return desired
        if v_motion > 0.0:
            return max(desired, 0.0)
        return min(desired, 0.0)

    @staticmethod
    def _standstill_hold_update(
        *,
        held: bool,
        enter_since_s: float | None,
        now_s: float,
        err_m: float,
        v_ref_m_s: float,
        v_cmd_m_s: float,
        v_meas_m_s: float,
        enter_m: float,
        exit_m: float,
        dwell_s: float,
        motion_wake_m_s: float = RAIL_IDLE_EPS_M_S,
    ) -> tuple[bool, float | None]:
        """Hysteresis standstill latch for FA24 freeze.

        Enter when |err|<=enter for ``dwell_s`` with near-zero motion; release
        only when |err|>exit or a non-trivial velocity reference appears.
        Tracking accuracy is the enter band; exit is a disturbance wake gate.
        """

        enter_m = max(float(enter_m), 0.0)
        exit_m = max(float(exit_m), enter_m)
        dwell_s = max(float(dwell_s), 0.0)
        wake = max(float(motion_wake_m_s), 0.0)
        err_abs = abs(float(err_m))
        motion_cmd = abs(float(v_ref_m_s)) >= wake
        if motion_cmd:
            return False, None
        if held:
            if err_abs > exit_m:
                return False, None
            return True, None
        quiet = (
            abs(float(v_cmd_m_s)) < wake
            and abs(float(v_meas_m_s)) < wake
            and err_abs <= enter_m
        )
        if not quiet:
            return False, None
        if enter_since_s is None:
            return False, float(now_s)
        if dwell_s <= 0.0 or (float(now_s) - float(enter_since_s)) >= dwell_s:
            return True, None
        return False, float(enter_since_s)

    def _worker_velocity(self) -> None:
        """Continuous soft-CSP → FA24: stream-aware reference + P–V law."""
        assert self._drive is not None
        period = 1.0 / max(float(self.config.poll_hz), 1.0)
        deadband_m = max(float(self.config.vel_deadband_mm), 0.01) * 1e-3
        # Gains and reference limits are re-read each tick so scan/approach
        # overrides take effect without restarting the worker.
        sign = float(self.config.sign)
        travel = float(self.config.travel_m)
        margin = max(float(self.config.fault_margin_m), 0.0)
        # Soft-end taper only when *goal* is near that end (homing), not mid-scan.
        approach_m = max(float(self.config.approach_m), 0.0)
        stream_dead_s = float(self.config.stream_dead_s())
        freeze_s = max(float(self.config.encoder_freeze_s), 0.1)
        freeze_vmin = max(float(self.config.encoder_freeze_min_v_m_s), 0.005)
        freeze_dx = max(float(self.config.encoder_freeze_min_move_mm), 0.1) * 1e-3
        settle_tol_m = max(float(self.config.settle_tol_mm), 0.01) * 1e-3
        settle_v = max(float(self.config.settle_v_m_s), 0.001)
        settle_timeout = max(float(self.config.settle_timeout_s), 0.1)
        max_stall_s = max(float(self.config.max_stall_s), 0.02)
        stall_v_floor = max(float(self.config.stall_v_floor_m_s), 0.001)
        jump_margin_m = max(float(self.config.jump_margin_mm), 0.5) * 1e-3
        jump_hard_m = max(float(self.config.jump_hard_mm), 10.0) * 1e-3
        jump_soft_streak_panic = max(1, int(self.config.jump_soft_streak_panic))
        prev_t = time.monotonic()
        last_modbus_warn = 0.0
        prev_v_cmd = 0.0
        x_ref = float(self.measured_m) if math.isfinite(self.measured_m) else 0.0
        v_ref = 0.0
        a_ref = 0.0
        ref_inited = False
        loop_n = 0
        loop_t0 = time.monotonic()
        freeze_anchor_x = float(self.measured_m)
        freeze_anchor_t = time.monotonic()
        moving_without_fb = False
        mb_fail_n = 0
        slow_poll_n = 0
        jump_soft_streak = 0
        idle_jump_n = 0
        idle_jump_m = float("nan")
        last_status_t = time.monotonic()
        last_enc_ok_t = time.monotonic()
        last_accepted_enc_t = last_enc_ok_t
        verbose = bool(self.config.verbose)
        # Cap PD/slew dt so a stalled poll cannot blow kd·de or fake a freeze.
        dt_cap = max(3.0 * period, 0.05)
        # If FA24 is nonzero but we have not read encoder this long → hard kill.
        latch_watch_s = max(float(self.config.latch_watch_s), 0.0)
        # Cold-start / re-arm: consecutive healthy polls with FA24=0.
        arm_need = max(5, int(self.config.arm_good_reads))
        arm_settle_s = max(0.0, float(self.config.arm_settle_s))
        arm_max_span_m = max(0.0005, float(self.config.arm_max_span_mm) * 1e-3)
        arm_good = 0
        arm_samples: list[float] = []
        arm_settle_deadline: float | None = None
        arm_log_t = 0.0
        settling = False
        settle_deadline: float | None = None
        standstill_held = False
        standstill_enter_since: float | None = None
        last_bias = int(getattr(self._drive, "_counts_bias", 0) or 0)
        next_t = time.monotonic()
        di_streak = 0
        enc_history: deque[tuple[float, float]] = deque(maxlen=8)
        # Window the encoder slope by what the worker actually achieves, not
        # by config: 225941 asked for 60 Hz and got 56.
        poll_period_history: deque[float] = deque(maxlen=16)
        v_enc_hold = float("nan")
        enc_hold_left = 0
        enc_hold_max = 2

        while not self._stop.is_set():
            if self._arm_req.is_set():
                self._arm_req.clear()
                with self._lock:
                    self._armed = False
                    self._follow_enabled = False
                    self._target_history.clear()
                arm_good = 0
                arm_samples.clear()
                arm_settle_deadline = None
                prev_v_cmd = 0.0
                v_ref = 0.0
                a_ref = 0.0
                ref_inited = False
                standstill_held = False
                standstill_enter_since = None
                last_accepted_enc_t = time.monotonic()
                enc_history.clear()
                poll_period_history.clear()
                v_enc_hold = float("nan")
                enc_hold_left = 0
                try:
                    self._drive.set_velocity_rpm(0, force=True)
                except Exception:
                    pass
                print("lw100 rail: arming… (FA24=0, proving Modbus)", flush=True)

            t0 = time.monotonic()
            dt_wall = max(t0 - prev_t, 1e-4)
            prev_t = t0
            dt = min(dt_wall, dt_cap)
            poll_ok = dt_wall <= dt_cap
            if poll_ok:
                poll_period_history.append(float(dt_wall))
            enc_period_s = (
                float(median(poll_period_history))
                if len(poll_period_history) >= 4
                else None
            )
            v_max = max(float(self.config.vel_max_m_s), 1.0e-4)
            a_max = max(float(self.config.vel_amax_m_s2), 1.0e-3)
            follow = False
            panic = False
            measured = float(self.measured_m)
            target = measured
            x_goal = target
            x_goal_eval = target
            last_rx = 0.0
            target_history: tuple[tuple[float, float], ...] = ()
            v_goal_est = 0.0
            target_v_ff = float("nan")
            command_mode = RailCommandMode.POSITION
            goal_stationary = False
            v_reg = self._rpm_to_mps(float(self.measured_speed_rpm))
            v_enc = v_reg
            v_meas = v_enc
            v_enc_source = "reg"
            v_des = 0.0
            v_cmd = 0.0
            a_cmd = 0.0
            hard_hold_this_tick = False
            encoder_accepted = True
            try:
                # Safety flag from latch watchdog (no concurrent Modbus there).
                if self._latch_kill_req.is_set():
                    self._latch_kill_req.clear()
                    self._hold_velocity(measured, "FA24 latched without encoder (safety flag)")
                    prev_v_cmd = 0.0
                    v_cmd = 0.0
                    hard_hold_this_tick = True

                # Latched-FA24 watchdog in-worker (same thread as Modbus).
                last_rpm = int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
                if abs(last_rpm) > 0 and (t0 - last_enc_ok_t) > latch_watch_s:
                    self._hold_velocity(
                        measured,
                        f"FA24 latched ({last_rpm} r/min) without encoder "
                        f"for {t0 - last_enc_ok_t:.2f}s",
                    )
                    prev_v_cmd = 0.0
                    v_cmd = 0.0
                    hard_hold_this_tick = True

                t_read0 = time.monotonic()
                drive_rpm, drive_m, di_mask = self._drive.read_motion_and_di_fast()
                t_read1 = time.monotonic()
                t_read_ms = (t_read1 - t_read0) * 1000.0
                n_modbus = 1
                # Stamp the middle of the Modbus read, not its end: the read
                # takes 8 ms median with a long tail, and timing the samples
                # off the tail put that jitter straight into the slope.
                motion_sample_mono = 0.5 * (t_read0 + t_read1)
                measured = self._encode_rail_m(drive_m)
                speed_rpm_host = self._encode_speed_rpm(drive_rpm)
                # Any successful Modbus read proves the encoder feed is not
                # dark.  Jump acceptance uses last_accepted_enc_t instead.
                last_enc_ok_t = motion_sample_mono
                self._last_enc_ok_mono = motion_sample_mono
                encoder_sample_ns = (
                    int(round(float(motion_sample_mono) * 1.0e9))
                    if math.isfinite(float(motion_sample_mono))
                    else 0
                )
                if encoder_sample_ns > 0:
                    with self._lock:
                        self._last_encoder_sample_mono_ns = encoder_sample_ns
                mb_fail_n = 0
                # Snapshot command state under lock; only stamp encoder if sane.
                with self._lock:
                    target = float(self._target_m)
                    target_v_ff = float(self._target_v_ff_m_s)
                    command_mode = RailCommandMode(self._command_mode)
                    follow = bool(self._follow_enabled)
                    panic = bool(self._panic)
                    speed_cap = self._speed_cap_rpm
                    last_rx = float(self._last_target_rx_mono)
                    target_history = tuple(self._target_history)
                    armed = bool(self._armed)
                    calibrated = bool(self._calibrated)
                    last_sane = float(self._measured_m)

                if not self._encoder_sane(measured):
                    # Out-of-band reading: stop streaming, keep session/cal.
                    measured = last_sane
                    self._hold_velocity(
                        measured, "invalid encoder sample (rejected; cal kept)"
                    )
                    hard_hold_this_tick = True
                    encoder_accepted = False
                    self._frame_continuous = False
                else:
                    # Continuity gate on host pose.  Impossible leaps are
                    # rejected; the taught zero file is never rewritten here —
                    # cold start / home owns calibration validity.
                    if (
                        math.isfinite(last_sane)
                        and self._encoder_sane(last_sane)
                        and calibrated
                    ):
                        gap_s = max(
                            float(dt_wall),
                            max(0.0, float(motion_sample_mono) - last_accepted_enc_t),
                        )
                        jump_lim = encoder_jump_limit_m(
                            v_max,
                            gap_s,
                            jump_margin_m,
                            restitch=bool(self._link_restitch),
                        )
                        jump = abs(measured - last_sane)
                        if jump > jump_lim:
                            raw_jump = float(measured)
                            same_idle = samples_agree_for_reanchor(
                                raw_jump,
                                idle_jump_m,
                                v_max_m_s=v_max,
                                dt_s=float(dt_wall),
                            )
                            idle_jump_n = idle_jump_n + 1 if same_idle else 1
                            idle_jump_m = raw_jump
                            fa24_zero = abs(int(last_rpm)) <= 0
                            v_quiet = abs(float(v_ref)) < RAIL_IDLE_EPS_M_S
                            # Live follow used to block re-anchor forever after
                            # a restitch reject.  FA24=0 + agreeing samples is
                            # enough; leftover v_ref after emergency_zero is
                            # ignored while restitch is still latched.
                            can_reanchor = fa24_zero and (
                                v_quiet or bool(self._link_restitch)
                            )
                            if can_reanchor and idle_jump_n >= RESTITCH_REANCHOR_POLLS:
                                jump_soft_streak = 0
                                idle_jump_n = 0
                                idle_jump_m = float("nan")
                                self._link_restitch = False
                                last_accepted_enc_t = motion_sample_mono
                                self._publish_motion(
                                    raw_jump,
                                    speed_rpm_host,
                                    sample_mono_s=motion_sample_mono,
                                )
                            else:
                                measured = last_sane
                                jump_soft_streak = jump_soft_streak + 1
                                self._hold_velocity(
                                    measured,
                                    f"encoder jump rejected {jump * 1000:+.1f} mm "
                                    f"(lim={jump_lim * 1000:.1f} mm; cal kept)",
                                )
                                hard_hold_this_tick = True
                                encoder_accepted = False
                                v_ref = 0.0
                                if jump >= jump_hard_m or jump_soft_streak >= jump_soft_streak_panic:
                                    # Pose stream untrusted for this session; do not
                                    # DISARM or erase the home zero.
                                    self._frame_continuous = False
                                    jump_soft_streak = 0
                        else:
                            jump_soft_streak = 0
                            idle_jump_n = 0
                            idle_jump_m = float("nan")
                            if self._link_restitch:
                                self._link_restitch = False
                            last_accepted_enc_t = motion_sample_mono
                            self._publish_motion(
                                measured,
                                speed_rpm_host,
                                sample_mono_s=motion_sample_mono,
                            )
                    else:
                        jump_soft_streak = 0
                        last_accepted_enc_t = motion_sample_mono
                        self._publish_motion(
                            measured,
                            speed_rpm_host,
                            sample_mono_s=motion_sample_mono,
                        )

                v_reg = self._rpm_to_mps(float(speed_rpm_host))
                if (
                    encoder_accepted
                    and math.isfinite(measured)
                    and math.isfinite(motion_sample_mono)
                ):
                    enc_history.append(
                        (float(motion_sample_mono), float(measured))
                    )
                v_enc, v_enc_source = self._encoder_velocity(
                    enc_history,
                    poll_hz=float(self.config.poll_hz),
                    fallback_m_s=v_reg,
                    period_s=enc_period_s,
                    hold_m_s=v_enc_hold,
                    hold_budget=enc_hold_left,
                )
                if v_enc_source == "lsq":
                    v_enc_hold = float(v_enc)
                    enc_hold_left = enc_hold_max
                elif v_enc_source == "hold":
                    enc_hold_left = max(0, enc_hold_left - 1)
                else:
                    v_enc_hold = float("nan")
                    enc_hold_left = 0
                v_meas = v_enc

                # Mid-session bias change = FA-60/SON wipe (trusted → resync).
                # Untrusted mid-run: HOLD and keep the taught zero (no wipe).
                try:
                    bias_now = int(getattr(self._drive, "_counts_bias", 0) or 0)
                except Exception:
                    bias_now = last_bias
                if bias_now != last_bias and calibrated and not panic:
                    delta = bias_now - last_bias
                    if getattr(self._drive, "frame_trusted", False):
                        self._resync_cal_frame_after_wipe(
                            delta,
                            reason=f"mid-run bias {last_bias}→{bias_now}",
                        )
                    else:
                        self._hold_velocity(
                            measured,
                            f"encoder bias changed mid-run "
                            f"({last_bias}→{bias_now}, frame untrusted; cal kept)",
                        )
                        hard_hold_this_tick = True
                        self._frame_continuous = False
                    last_bias = bias_now

                # DI comes from the same 16-reg read.  Debounce in software
                # (3 consecutive polls) so we never spend a second Modbus trip.
                if calibrated and not panic:
                    di3_p, di4_p = di_limits_pressed_from_mask(
                        di_mask, nc=bool(self.config.di_nc)
                    )
                    if di3_p or di4_p:
                        di_streak += 1
                    else:
                        di_streak = 0
                    if di_streak >= max(1, int(self.config.di_debounce_n)):
                        which = []
                        if di3_p:
                            which.append("DI3")
                        if di4_p:
                            which.append("DI4")
                        self._trip_panic(
                            measured,
                            f"limit DI hit in run ({'+'.join(which)})",
                        )
                        panic = True
                        follow = False
                        armed = False
                        di_streak = 0
                else:
                    di_streak = 0

                # Over-budget poll: do NOT zero FA24 on a single slow cycle —
                # that made mid-travel "tugs" (meas velocity → 0 while target
                # kept moving). Coast with the previous command; only hard-hold
                # after several consecutive over-budget polls.
                if poll_ok:
                    slow_poll_n = 0
                else:
                    slow_poll_n += 1
                    if slow_poll_n >= 3 and abs(
                        int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
                    ) > 0:
                        self._hold_velocity(
                            measured,
                            f"poll over-budget ×{slow_poll_n} "
                            f"dt_wall={dt_wall * 1000:.0f}ms",
                        )
                        prev_v_cmd = 0.0
                        v_cmd = 0.0
                        hard_hold_this_tick = True

                if not math.isfinite(target):
                    self._hold_velocity(measured, "invalid target")
                    prev_v_cmd = 0.0
                    v_cmd = 0.0
                    follow = False

                # --- Arming gate: no follow until Modbus path is proven hot ---
                if not armed and not panic and not self._abort.is_set():
                    self._drive.set_velocity_rpm(0, force=False)
                    prev_v_cmd = 0.0
                    if poll_ok and self._encoder_sane(measured):
                        arm_good += 1
                        arm_samples.append(measured)
                        if len(arm_samples) > arm_need:
                            arm_samples = arm_samples[-arm_need:]
                    else:
                        arm_good = 0
                        arm_samples.clear()
                        arm_settle_deadline = None
                    if arm_good >= arm_need and len(arm_samples) >= arm_need:
                        span = max(arm_samples) - min(arm_samples)
                        if span > arm_max_span_m:
                            if t0 - arm_log_t >= 1.0:
                                arm_log_t = t0
                                print(
                                    f"lw100 rail: arming — encoder span "
                                    f"{span * 1000:.1f} mm > "
                                    f"{arm_max_span_m * 1000:.1f} mm; reset",
                                    flush=True,
                                )
                            arm_good = 0
                            arm_samples.clear()
                            arm_settle_deadline = None
                        elif arm_settle_deadline is None:
                            arm_settle_deadline = t0 + arm_settle_s
                            print(
                                f"lw100 rail: arming — {arm_need} good polls "
                                f"@ {measured * 1000:.1f} mm; settle "
                                f"{arm_settle_s:.2f}s…",
                                flush=True,
                            )
                        elif t0 >= arm_settle_deadline:
                            with self._lock:
                                self._armed = True
                                self._target_m = measured
                                self._commanded_m = measured
                                self._target_history.clear()
                                self._target_history.append((t0, measured))
                                self._follow_enabled = False
                                self._panic = False
                            print(
                                f"lw100 rail: ARMED @ {measured:+.4f} m "
                                f"(FA24=0, Modbus OK, follow gated)",
                                flush=True,
                            )
                            self._log_event(
                                "ARMED",
                                measured_m=measured,
                                target_m=measured,
                                commanded_m=measured,
                                armed=True,
                                follow=False,
                                panic=False,
                                poll_ok=poll_ok,
                                dt_wall_ms=dt_wall * 1000.0,
                                arm_good=arm_need,
                            )
                            arm_good = 0
                            arm_samples.clear()
                            arm_settle_deadline = None
                    elif t0 - arm_log_t >= 2.0:
                        arm_log_t = t0
                        print(
                            f"lw100 rail: NOT READY — arming "
                            f"{arm_good}/{arm_need} good polls "
                            f"meas={measured * 1000:.1f} mm"
                            f"{'' if poll_ok else ' SLOW'}",
                            flush=True,
                        )
                    # Hold zero; skip soft loop until ARMED.
                    now = time.monotonic()
                    next_t = next_poll_deadline(next_t, now, period)
                    if self._stop.wait(max(0.0, next_t - now)):
                        break
                    continue

                if follow and last_rx > 0.0 and (t0 - last_rx) > stream_dead_s:
                    # A velocity stream has no terminal position contract.
                    # Ramp to zero and re-anchor at the measured position
                    # after the encoder sample is valid.  Never reopen the
                    # old integrated target for a position catch-up.
                    if command_mode is RailCommandMode.COUPLED_VELOCITY:
                        target_v_ff = 0.0
                        v_goal_est = 0.0
                        goal_stationary = True
                        settling = False
                        settle_deadline = None
                        motion_active = (
                            abs(v_ref) >= RAIL_IDLE_EPS_M_S
                            or abs(prev_v_cmd) >= RAIL_IDLE_EPS_M_S
                            or abs(self._rpm_to_mps(float(speed_rpm_host))) >= RAIL_IDLE_EPS_M_S
                        )
                        if (
                            not motion_active
                            and abs(self._rpm_to_mps(float(speed_rpm_host))) < RAIL_IDLE_EPS_M_S
                            and self._encoder_sane(measured)
                        ):
                            x_ref = measured
                            v_ref = 0.0
                            a_ref = 0.0
                            follow = False
                            with self._lock:
                                self._target_m = measured
                                self._follow_enabled = False
                    else:
                        err_abs = abs(target - measured) if math.isfinite(target) else 0.0
                        motion_active = (
                            abs(v_ref) >= RAIL_IDLE_EPS_M_S
                            or abs(prev_v_cmd) >= RAIL_IDLE_EPS_M_S
                            or abs(self._rpm_to_mps(float(speed_rpm_host))) >= RAIL_IDLE_EPS_M_S
                        )
                        if (err_abs > settle_tol_m or motion_active) and not panic and armed:
                            if not settling:
                                settling = True
                                settle_deadline = t0 + settle_timeout
                                print(
                                    f"lw100 rail: target stream ended — settling "
                                    f"residual={err_abs * 1000:.2f} mm "
                                    f"(tol={settle_tol_m * 1000:.2f} mm)",
                                    flush=True,
                                )
                            elif settle_deadline is not None and t0 >= settle_deadline:
                                settling = False
                                settle_deadline = None
                                follow = False
                                with self._lock:
                                    self._follow_enabled = False
                                print(
                                    f"lw100 rail: settle timeout → FA24=0 "
                                    f"(residual={err_abs * 1000:.2f} mm)",
                                    flush=True,
                                )
                        else:
                            settling = False
                            settle_deadline = None
                            follow = False
                            with self._lock:
                                self._follow_enabled = False
                            if err_abs > 1e-6:
                                print(
                                    f"lw100 rail: target timeout → FA24=0 "
                                    f"(residual={err_abs * 1000:.2f} mm)",
                                    flush=True,
                                )
                            else:
                                print("lw100 rail: target timeout → FA24=0", flush=True)
                elif follow and last_rx > 0.0:
                    # Fresh targets — exit settle substate.
                    settling = False
                    settle_deadline = None

                if settling and follow and not panic and armed:
                    err_abs = abs(target - measured)
                    motion_settled = (
                        abs(v_ref) < RAIL_IDLE_EPS_M_S
                        and abs(prev_v_cmd) < RAIL_IDLE_EPS_M_S
                        and abs(self._rpm_to_mps(float(speed_rpm_host))) < RAIL_IDLE_EPS_M_S
                    )
                    if err_abs <= settle_tol_m and motion_settled:
                        settling = False
                        settle_deadline = None
                        follow = False
                        with self._lock:
                            self._follow_enabled = False
                        print(
                            f"lw100 rail: settled @ "
                            f"{measured * 1000:.2f} mm (err={err_abs * 1000:.2f} mm)",
                            flush=True,
                        )

                if (
                    panic
                    or self._abort.is_set()
                    or not follow
                    or not armed
                    or hard_hold_this_tick
                ):
                    v_cmd = 0.0
                    v_des = 0.0
                    v_goal_est = 0.0
                    v_ref = 0.0
                    a_ref = 0.0
                    ref_inited = False
                    freeze_anchor_x = measured
                    freeze_anchor_t = t0
                    moving_without_fb = False
                    settling = False
                    settle_deadline = None
                    standstill_held = False
                    standstill_enter_since = None
                    if (
                        not follow
                        and not panic
                        and not self._abort.is_set()
                        and bool(self._hold_active)
                    ):
                        self._hold_watchdog(measured, t0)
                else:
                    # --- Stream-aware soft CSP: arbitrary x_goal → (x_ref, v_ref) ---
                    if not ref_inited or not math.isfinite(x_ref):
                        x_ref = measured
                        v_ref = 0.0
                        a_ref = 0.0
                        ref_inited = True
                    x_goal = float(target)
                    soft_lo, soft_hi = self._soft_lo_hi()
                    velocity_coupled = (
                        command_mode is RailCommandMode.COUPLED_VELOCITY
                    )
                    if velocity_coupled:
                        v_goal_est = (
                            target_v_ff if math.isfinite(target_v_ff) else 0.0
                        )
                        v_goal_est = max(-v_max, min(v_max, v_goal_est))
                        x_goal_eval = max(soft_lo, min(soft_hi, x_goal))
                        v_ff_live = bool(follow) and not settling
                        a_ref_max = (
                            float(self.config.live_host_accel_m_s2())
                            if v_ff_live
                            else a_max
                        )
                        x_ref, v_ref, a_ref, parked = self._parked_reanchor(
                            x_ref,
                            v_ref,
                            a_ref,
                            measured=measured,
                            v_goal=v_goal_est,
                            v_meas=v_meas,
                        )
                        if not parked:
                            x_ref, v_ref, a_ref = self._step_velocity_reference(
                                x_ref,
                                v_ref,
                                v_goal_est,
                                dt=dt,
                                v_max=v_max,
                                a_max=a_ref_max,
                                x_goal=x_goal_eval,
                                catch_v_max=float(self.config.catch_v_max_m_s),
                                k_catch=0.0,
                                catch_frac=float(self.config.catch_frac),
                                x_min=soft_lo,
                                x_max=soft_hi,
                            )
                    else:
                        stream_v_ff = (
                            target_v_ff
                            if math.isfinite(target_v_ff)
                            and bool(follow)
                            and not settling
                            else float("nan")
                        )
                        x_goal_eval, v_goal_est, goal_stationary = (
                            self._resolve_stream_goal(
                                target_history,
                                now_s=motion_sample_mono,
                                max_age_s=min(2.0 * period, 0.05),
                                target_m=x_goal,
                                last_rx_s=last_rx,
                                v_ff_m_s=stream_v_ff,
                            )
                        )
                        v_goal_est = max(-v_max, min(v_max, v_goal_est))
                        x_goal_eval = max(soft_lo, min(soft_hi, x_goal_eval))
                        if settling:
                            v_goal_est = 0.0
                            goal_stationary = True
                            x_goal_eval = x_goal
                        v_ff_live = math.isfinite(stream_v_ff)
                        a_ref_max = (
                            float(self.config.live_host_accel_m_s2())
                            if v_ff_live
                            else a_max
                        )
                        x_ref, v_ref, a_ref = self._step_reference(
                            x_ref,
                            v_ref,
                            x_goal_eval,
                            v_goal_est,
                            stationary=goal_stationary,
                            dt=dt,
                            v_max=v_max,
                            a_max=a_ref_max,
                        )

                    kp = float(self.config.vel_kp)
                    kd = float(self.config.vel_kd)
                    err_x = x_ref - measured
                    err_v = v_ref - v_meas
                    # Position+FF on the shaped reference (papers: ẋd + Kp(xd−x)
                    # + Kd(ẋd−ẋ)).  xd is x_ref, never x_goal — command lead
                    # and later KMP OTG stay outside this loop.  Pure velocity
                    # (v_p=0) integrates drift; that was the 3 mm tool-Y.
                    # v_meas is encoder-difference, not the lagged 0x1000
                    # register (157 ms stale → plugging brake on every stop).
                    v_p = kp * err_x
                    if settling:
                        v_p_allow = abs(err_x) / max_stall_s
                    else:
                        v_p_allow = max(abs(err_x) / max_stall_s, stall_v_floor)
                    if velocity_coupled:
                        trim = max(float(self.config.vel_ff_p_trim_m_s), 0.0)
                        if trim > 0.0:
                            v_p_allow = min(v_p_allow, trim)
                    v_p = max(-v_p_allow, min(v_p_allow, v_p))
                    if velocity_coupled and abs(v_ref) > 1.0e-3:
                        # Motion: L1 owns position.  Standstill latch keeps P.
                        v_p = 0.0
                    brake_margin = float(self.config.decel_request_margin_m_s)
                    v_d = kd * err_v
                    if velocity_coupled:
                        v_d = 0.0
                    else:
                        d_cap = max(float(self.config.vel_kd_max_m_s), 0.0)
                        if d_cap > 0.0:
                            v_d = max(-d_cap, min(d_cap, v_d))
                    v_raw = v_ref + v_p + v_d
                    if velocity_coupled:
                        v_raw = self._clamp_zero_target_brake(
                            v_raw,
                            v_goal=v_goal_est,
                            v_ref=v_ref,
                            v_meas=v_meas,
                            v_prev_cmd=prev_v_cmd,
                            margin=brake_margin,
                        )

                    # Standstill when the shaped reference has stopped
                    # (|v_ref| < 1 mm/s), including live follow.  Do not
                    # veto on follow — that left P hunting FA24 at idle.
                    # Latch on e_track (x_ref−x_meas), never x_goal (20 mm lead).
                    # Instant deadband only for a truly stopped ref so a
                    # tiny nonzero v_ref still moves.
                    target_stale = bool(
                        last_rx <= 0.0 or (t0 - last_rx) > stream_dead_s
                    )
                    follow_live = bool(follow) and not settling and not target_stale
                    v_ref_stopped = abs(v_ref) < RAIL_IDLE_EPS_M_S

                    enter_m = max(float(self.config.standstill_enter_mm), 0.01) * 1e-3
                    exit_m = max(float(self.config.standstill_exit_mm), 0.01) * 1e-3
                    dwell_s = max(float(self.config.standstill_dwell_s), 0.0)
                    was_held = standstill_held
                    if not v_ref_stopped:
                        standstill_held = False
                        standstill_enter_since = None
                    else:
                        standstill_held, standstill_enter_since = (
                            self._standstill_hold_update(
                                held=standstill_held,
                                enter_since_s=standstill_enter_since,
                                now_s=t0,
                                err_m=err_x,
                                v_ref_m_s=v_ref,
                                v_cmd_m_s=prev_v_cmd,
                                v_meas_m_s=v_meas,
                                enter_m=enter_m,
                                exit_m=exit_m,
                                dwell_s=dwell_s,
                            )
                        )
                        if standstill_held:
                            v_raw = 0.0
                            v_ref = 0.0
                            a_ref = 0.0
                            x_ref = measured
                            if verbose and not was_held:
                                print(
                                    f"lw100 rail: standstill latch "
                                    f"|e_track|={abs(err_x) * 1000:.2f} mm → FA24=0",
                                    flush=True,
                                )
                        elif verbose and was_held:
                            print(
                                f"lw100 rail: standstill wake "
                                f"|e_track|={abs(err_x) * 1000:.2f} mm",
                                flush=True,
                            )

                    v_des = max(-v_max, min(v_max, v_raw))
                    if settling:
                        v_des = max(-settle_v, min(settle_v, v_des))
                    if measured <= 0.0 and v_des < 0.0:
                        v_des = 0.0
                    if measured >= travel and v_des > 0.0:
                        v_des = 0.0

                    if speed_cap is not None:
                        rpm_per_mps = max(abs(self._mps_to_rpm(1.0)), 1e-6)
                        cruise_m_s = abs(float(speed_cap)) / rpm_per_mps
                        home_band = max(float(self.config.home_approach_mm), 1.0) * 1e-3
                        if abs(err_x) >= home_band:
                            lim = cruise_m_s
                        else:
                            lim = cruise_m_s * (abs(err_x) / home_band)
                        v_des = max(-lim, min(lim, v_des))

                    dv_max = a_ref_max * dt
                    v_cmd = max(prev_v_cmd - dv_max, min(prev_v_cmd + dv_max, v_des))
                    env_lo, env_hi = self._envelope_lo_hi()
                    lo_cap, hi_cap = wall_cap(
                        measured,
                        lo=env_lo,
                        hi=env_hi,
                        a_max=float(a_ref_max),
                        reaction_s=float(self.config.wall_reaction_s),
                    )
                    v_env = max(lo_cap, min(hi_cap, v_cmd))
                    if abs(v_env - v_cmd) > 1.0e-9:
                        self._wall_override_count = (
                            int(getattr(self, "_wall_override_count", 0)) + 1
                        )
                        self._wall_override_last = True
                    else:
                        self._wall_override_last = False
                    v_cmd = v_env
                    a_cmd = (v_cmd - prev_v_cmd) / max(dt, 1.0e-6)
                    if standstill_held:
                        # Instant freeze — do not coast down through stiction hum.
                        v_des = 0.0
                        v_cmd = 0.0
                        a_cmd = 0.0
                    elif not follow_live:
                        if (
                            abs(v_ref) < RAIL_IDLE_EPS_M_S
                            and abs(err_x) <= max(deadband_m, settle_tol_m)
                            and abs(v_cmd) <= 1.0e-6
                        ):
                            v_cmd = 0.0
                            a_cmd = 0.0

                    # Single/double slow poll: coast. ≥3 → hard zero.
                    if not poll_ok:
                        if slow_poll_n >= 3:
                            v_cmd = 0.0
                            prev_v_cmd = 0.0
                            a_cmd = 0.0
                        else:
                            v_cmd = prev_v_cmd
                            a_cmd = 0.0
                        freeze_anchor_t = t0
                    elif abs(v_cmd) >= freeze_vmin:
                        # Freeze only if drive RPM≈0 AND host Δx is stuck.
                        drive_moving = abs(speed_rpm_host) >= 3
                        if drive_moving or abs(measured - freeze_anchor_x) >= freeze_dx:
                            freeze_anchor_x = measured
                            freeze_anchor_t = t0
                            moving_without_fb = False
                        elif (t0 - freeze_anchor_t) >= freeze_s:
                            moving_without_fb = True
                            self._hold_velocity(
                                measured,
                                f"encoder lag while cmd={v_cmd:+.3f} m/s "
                                f"(Δx<{freeze_dx * 1000:.1f}mm, drive_rpm="
                                f"{speed_rpm_host} for {freeze_s:.2f}s)",
                            )
                            v_cmd = 0.0
                            prev_v_cmd = 0.0
                            a_cmd = 0.0
                            v_ref = 0.0
                            a_ref = 0.0
                            ref_inited = False
                            freeze_anchor_t = t0
                    else:
                        freeze_anchor_x = measured
                        freeze_anchor_t = t0
                        moving_without_fb = False

                    # Open-loop travel guard: zero cmd near ends, do not DISARM.
                    x_pred = measured + v_cmd * dt
                    if x_pred < -margin or x_pred > travel + margin:
                        self._hold_velocity(
                            measured,
                            f"predicted rail near end x_pred={x_pred * 1000:.1f} mm",
                        )
                        v_cmd = 0.0
                        prev_v_cmd = 0.0
                        a_cmd = 0.0
                        v_ref = 0.0
                        a_ref = 0.0
                        ref_inited = False

                rpm = sign * self._mps_to_rpm(v_cmd)
                rpm_deadband = (
                    int(self.config.fa24_rpm_deadband)
                    if bool(follow) and not settling and not panic
                    else 0
                )
                last_rpm_before = int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
                t_write0 = time.monotonic()
                fa24_write_ns = time.monotonic_ns()
                rpm_cmd = self._drive.set_velocity_rpm(rpm, deadband=rpm_deadband)
                t_write_ms = (time.monotonic() - t_write0) * 1000.0
                wrote = (
                    t_write_ms > 0.5
                    or int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
                    != last_rpm_before
                )
                if wrote:
                    n_modbus += 1
                    with self._lock:
                        self._last_fa24_write_mono_ns = int(fa24_write_ns)
                else:
                    t_write_ms = 0.0
                modbus_ms = float(t_read_ms) + float(t_write_ms)
                if modbus_ms > 12.0:
                    now_warn = time.monotonic()
                    if now_warn - last_modbus_warn >= 1.0:
                        last_modbus_warn = now_warn
                        print(
                            f"lw100 rail: Modbus {modbus_ms:.1f} ms "
                            f"(read {t_read_ms:.1f} + write {t_write_ms:.1f}) "
                            "exceeds 12 ms budget",
                            flush=True,
                        )
                prev_v_cmd = sign * self._rpm_to_mps(float(rpm_cmd))
                control_mono = time.monotonic()
                sample_mono = motion_sample_mono
                sample_x_ref = x_ref if ref_inited else measured
                with self._lock:
                    self._commanded_m = sample_x_ref
                    motion_seq = int(self._measured_seq)
                    hold_count = int(self._hold_count)
                    hold_reason = (
                        str(self._last_hold_reason)
                        if control_mono - self._last_hold_mono <= max(2.0 * period, 0.05)
                        else ""
                    )
                    prev_sample_t = float(self._servo_sample.sample_mono_s)
                    if not encoder_accepted and math.isfinite(prev_sample_t):
                        sample_mono = prev_sample_t
                    self._servo_sample = RailServoSample(
                        sample_mono_s=sample_mono,
                        target_rx_mono_s=last_rx,
                        motion_seq=motion_seq,
                        x_goal_m=float(target),
                        x_goal_eval_m=x_goal_eval,
                        x_ref_m=sample_x_ref,
                        x_meas_m=measured,
                        v_goal_est_m_s=v_goal_est,
                        v_ref_m_s=v_ref if ref_inited else 0.0,
                        a_ref_m_s2=a_ref if ref_inited else 0.0,
                        v_meas_m_s=v_meas,
                        v_des_m_s=v_des,
                        v_cmd_m_s=v_cmd,
                        a_cmd_m_s2=a_cmd,
                        rpm_cmd=int(rpm_cmd),
                        follow=follow,
                        armed=armed,
                        panic=panic,
                        poll_ok=poll_ok,
                        mb_fail_n=mb_fail_n,
                        freeze_flag=moving_without_fb,
                        hold_count=hold_count,
                        hold_reason=hold_reason,
                        command_mode=command_mode.value,
                        feedback_valid=bool(encoder_accepted),
                    )
                if self._csv is not None:
                    last_rpm = int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
                    self._csv.write(
                        event="",
                        target_m=target,
                        commanded_m=x_ref if ref_inited else target,
                        measured_m=measured,
                        v_ff=v_ref,
                        v_des=v_des,
                        v_cmd=v_cmd,
                        rpm=rpm,
                        follow=follow,
                        armed=armed,
                        panic=panic,
                        poll_ok=poll_ok,
                        dt_wall_ms=dt_wall * 1000.0,
                        last_rpm_cmd=last_rpm,
                        mb_fail_n=mb_fail_n,
                        freeze_flag=moving_without_fb,
                        arm_good=arm_good,
                        sample_mono_s=sample_mono,
                        target_rx_mono_s=last_rx,
                        motion_seq=motion_seq,
                        x_goal_m=target,
                        x_goal_eval_m=x_goal_eval,
                        x_ref_m=sample_x_ref,
                        x_meas_m=measured,
                        v_goal_est_m_s=v_goal_est,
                        v_ref_m_s=v_ref if ref_inited else 0.0,
                        a_ref_m_s2=a_ref if ref_inited else 0.0,
                        v_reg_m_s=v_reg,
                        v_enc_m_s=v_enc,
                        v_enc_source=v_enc_source,
                        v_des_m_s=v_des,
                        v_cmd_m_s=v_cmd,
                        a_cmd_m_s2=a_cmd,
                        rpm_cmd=rpm_cmd,
                        hold_count=hold_count,
                        hold_reason=hold_reason,
                        command_mode=command_mode.value,
                        feedback_valid=bool(encoder_accepted),
                        t_read_ms=t_read_ms,
                        t_write_ms=t_write_ms,
                        n_modbus=n_modbus,
                        fa24_write_mono_ns=int(self._last_fa24_write_mono_ns),
                        encoder_sample_mono_ns=int(self._last_encoder_sample_mono_ns),
                    )
                # Rare SP-slot reassert (avoid extra Modbus during tracking).
                if loop_n > 0 and loop_n % max(1, int(self.config.poll_hz * 30)) == 0:
                    try:
                        self._drive.ensure_velocity_slot_safe()
                    except ModbusRtuError:
                        pass

                loop_n += 1
                if t0 - last_status_t >= 5.0:
                    last_status_t = t0
                    hz = loop_n / max(t0 - loop_t0, 1e-6)
                    print(
                        f"lw100 rail: loop {hz:.0f} Hz "
                        f"tgt={target * 1000:.1f} meas={measured * 1000:.1f} mm "
                        f"follow={follow}{' PANIC' if panic else ''}"
                        f"{' FREEZE?' if moving_without_fb else ''}"
                        f"{'' if poll_ok else ' SLOW'}",
                        flush=True,
                    )
                    if verbose and follow and abs(rpm) > 1.0:
                        print(
                            f"lw100 rail: v_follow v={v_cmd:+.3f} m/s → {rpm:+.0f} r/min",
                            flush=True,
                        )
                    loop_n = 0
                    loop_t0 = t0
            except ModbusRtuError as exc:
                if self._stop.is_set() or self._abort.is_set():
                    break
                mb_fail_n += 1
                arm_good = 0
                arm_samples.clear()
                arm_settle_deadline = None
                latched = int(getattr(self._drive, "_last_rpm_cmd", 0) or 0)
                # One or two short USR-TCP232 misses coast on the latched FA24.
                # Three consecutive failures hard-hold below; the independent
                # 120 ms encoder-age watchdog remains the absolute safety bound.
                if mb_fail_n in (1, 2, 3, 10) or mb_fail_n % 50 == 0:
                    print(
                        f"lw100 rail: modbus error ({mb_fail_n}x): {exc}",
                        flush=True,
                    )
                # Consecutive poll failures → zero FA24, stay ARMED (resume on next OK).
                if mb_fail_n >= 3:
                    prev_v_cmd = 0.0
                    v_ref = 0.0
                    a_ref = 0.0
                    ref_inited = False
                    self._hold_velocity(
                        self.measured_m,
                        f"modbus poll failed {mb_fail_n}x"
                        + (f" with latched FA24={latched} r/min" if abs(latched) > 0 else ""),
                    )
                # Skip / hold: short yield only (never 0.25–0.5 s reconnect sleep).
                if self._stop.wait(0.02 if mb_fail_n < 5 else 0.05):
                    break
                continue
            except Exception as exc:
                if self._stop.is_set() or self._abort.is_set():
                    break
                prev_v_cmd = 0.0
                v_ref = 0.0
                a_ref = 0.0
                ref_inited = False
                # Socket already closed during teardown — exit quietly.
                if "NoneType" in str(exc) or "not connected" in str(exc):
                    break
                print(f"lw100 rail: worker error: {exc}", flush=True)
                if self._stop.wait(0.05):
                    break
                continue

            now = time.monotonic()
            next_t = next_poll_deadline(next_t, now, period)
            if self._stop.wait(max(0.0, next_t - now)):
                break

        # Teardown: socket may already be closed by estop/stop — never block.
        try:
            if self._drive is not None and self._drive._client._sock is not None:
                self._drive.kill_velocity_hard(attempts=1, disable_on_fail=False)
        except Exception:
            pass
```

### `apps/joint_admittance_8dof/analyze_qpik_quality.py`

```python
#!/usr/bin/env python3
"""Score a sin_tool_y debug CSV against phase-2 QPIK + force gates.

Usage (after a hardware run)::

    python apps/joint_admittance_8dof/analyze_qpik_quality.py \\
        apps/logs/sin_tool_y/run_YYYYMMDD_HHMMSS.csv

First fixture: 30 cm peak-to-peak.  Promote to 60 cm only after all gates pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


GATES = {
    "waste_ratio": 1.15,
    "rail_min_m": 0.02,
    "rail_max_m": 0.78,
    "j4_j7_margin_deg": 10.0,
    "arm_acc_max": 8.0,
    "contact_loss_frac": 0.02,
    "fz_p99_n": 4.0,
    "track_err_p95_mm": 1.0,
    "j4_limit_deg": 135.0,
    "j6_limit_deg": 128.0,
    "j7_limit_deg": 360.0,
    # Jitter budget, measured from q_cmd on a UNIFORM step (never from wall
    # time, and never from differentiated pose feedback whose 0.1 mm
    # quantisation aliases to ~20 mm/s per tick).  Run 230940 on a uniform
    # step: reversals 6.5-14.3/s, jerk RMS 94-130, so 20/s and 200 are just
    # above the current machine and will catch a real regression.
    "accel_reversals_per_s": 20.0,
    "jerk_rms": 200.0,
    "accel_saturation_frac": 0.05,
    # rm_movej_canfd consumes at a fixed cadence; an irregular producer is
    # felt as roughness.  Measured 6.16 ms mean against a 5.0 ms budget with
    # only 2.1% of ticks on time, so this starts as a tracked failure.
    "dt_nominal_s": 0.005,
    "dt_on_time_frac": 0.80,
    # Command-step ripple is what rm_movej_canfd actually consumes.
    "step_ripple_p999": 0.50,
    "step_ripple_max": 1.00,
    # 58 Hz measurement-geometry limit cycle: 40-80 Hz power on q_cmd accel.
    "q_cmd_accel_hf_frac": 0.15,
    "rail_write_dt_corr": 0.30,
    "deadline_slack_pos_frac": 0.99,
    # t_ref used to advance by dt_nom while wall ran 6.64 ms: lag grew to 14 s.
    "accepted_reference_lag_p95_s": 0.10,
    "rail_period_nominal_s": 1.0 / 60.0,
    "rail_period_on_time_frac": 0.80,
    "rail_target_age_p95_ms": 50.0,
}

# If 5 ms misses the slack gate, step back up; do not skip rungs.
PERIOD_LADDER_MS = (7.0, 6.0, 5.0)


def next_period_ms(
    current_ms: float,
    slack_pos_frac: float,
    *,
    threshold: float = 0.99,
) -> float:
    """Return the next lower period only when deadline slack already passes."""
    current = float(current_ms)
    if not np.isfinite(slack_pos_frac) or float(slack_pos_frac) < float(threshold):
        return current
    lower = [p for p in PERIOD_LADDER_MS if p < current - 1.0e-9]
    return float(lower[0]) if lower else current


def raise_period_ms(current_ms: float) -> float:
    """Next slower rung if 5 ms cannot hold the slack gate."""
    current = float(current_ms)
    higher = [p for p in PERIOD_LADDER_MS if p > current + 1.0e-9]
    return float(higher[0]) if higher else current


_GATES_CONT = {
    "j6_open_frac": 0.05,
    "j4_near_limit_frac": 0.05,
    "j2_near_limit_frac": 0.05,
    "j2_limit_deg": 130.0,
    "tick_inner_max_ms": 20.0,
    # Rail-at-wall is not workspace-sat.  If 7DOF IK exists at locked q0,
    # track_err in the band must stay at the scan gate (not rail_share).
    "track_err_at_limit_mm": 3.0,
    "rail_limit_band_m": 0.06,
    # Carriage servo (rail_servo CSV, sibling directory).
    "rail_servo_accel_reversals_per_s": 3.0,
    "rail_servo_track_err_p95_mm": 2.0,
    # After v_goal→0, encoder-diff reverse peak / entry speed.
    # Hardware before the brake fix: 0.50–0.60.
    "rail_stop_reverse_frac": 0.15,
    # q_cmd[0]−q_meas[0] stuck on the 20 mm resync window is a fault.
    "rail_resync_err_m": 0.018,
    "rail_resync_bind_frac": 0.005,
    # v_enc falling back to the 157 ms-lagged drive register puts a step in
    # the D term.  Run 225941 fell back on 11.2% of ticks and cost the
    # gamepad 1.43 → 3.23 mm of e_track.
    "rail_v_enc_register_frac": 0.02,
    # Rail travel after the operator lets go: the posture preference used to
    # dump its accumulated debt for ~1 s (25–73 mm).
    "idle_rail_travel_mm": 8.0,
    # TCP must hold station while that happens.  After wall-dt integration
    # the planned rail/arm cancel is already ~0; idle pose_d is latched.
    "idle_tcp_drift_mm": 1.0,
    # QP1 buys the rail motion it cannot cancel with Cartesian slack, which
    # is how the rail slide reaches the TCP.  Run 225941: 31.3% of idle ticks.
    "idle_task_slack_frac": 0.05,
    # |rail_posture_err| while driving (preferred-extension residual, not
    # TCP tracking).  The old shared 80 mm/s budget starved reach whenever
    # FF asked for its legal 120, and the error grew at 39 mm/s until
    # release (p95 94 mm).
    "drive_rail_posture_err_p95_m": 0.030,
    # Coupled-mode open-loop drift of x_goal − x_ref.  Run 002843 ratcheted
    # to 15.93 mm because _step_velocity_reference never used x_goal.
    "rail_eshape_p95_mm": 2.0,
    # QP box: rail.v_max_m_s 0.15 × v_scale 0.8, a_max_rail_m_s2 0.60.
    "rail_v_box_m_s": 0.12,
    "rail_a_box_m_s2": 0.60,
    "rail_v_box_frac": 0.01,
    "rail_a_box_frac": 0.01,
    # rail_task_vel empty while |v_ff_rail| is live: QP1 pins the rail to 0.
    "rail_task_dropout_frac": 0.01,
    "rail_task_dropout_ff_m_s": 1.0e-4,
    # Live (v_des − v_ref)·sign(v_goal) < −5 mm/s is the cruise P-term leak.
    "rail_p_term_leak_m_s": 0.005,
    "rail_p_term_leak_frac": 0.001,
    # |v_goal| < 5 mm/s: catch-up must not fire a 7x turn kick.
    "rail_turn_v_goal_m_s": 0.005,
    "rail_turn_overspeed_p99": 2.0,
    # d* per-tick step (d_center_rate 20 mm/s × dt × 2).
    "d_star_rate_m_s": 0.02,
    "d_star_step_margin": 2.0,
    # Δq_meas / ∫(qdot_sent · dt_wall) after wall-dt integration.
    "joint_exec_ratio_lo": 0.90,
    "joint_exec_ratio_hi": 1.10,
    "joint_exec_min_integral": 0.01,
    # VPC mid-ranging rebuild (Phase 5).
    "rail_share_p50": 0.60,
    "rail_share_vy_m_s": 0.020,
    "rail_share_pm_ratio": 1.25,
    "psi_err_p95_deg": 15.0,
    "rail_sign_agree_frac": 0.85,
    "rail_sign_vy_m_s": 0.010,
    "vpc_track_err_p95_mm": 5.0,
    "fa24_write_hz": 40.0,
    "fa24_drpm_p95": 20.0,
}
GATES.update(_GATES_CONT)


def best_axis_time_shift(
    ref: np.ndarray,
    meas: np.ndarray,
    dt: float,
    *,
    max_lag_s: float = 0.5,
) -> tuple[float, float]:
    """Best lag ``tau`` such that ``meas(t) ≈ ref(t - tau)``.

    Positive ``tau`` means the measurement lags the reference.  Negative
    means it leads.  Returns ``(tau_s, residual_rms)``.
    """
    ref_a = np.asarray(ref, dtype=float).reshape(-1)
    meas_a = np.asarray(meas, dtype=float).reshape(-1)
    n = int(min(ref_a.size, meas_a.size))
    if n < 20 or not np.isfinite(dt) or dt <= 0.0:
        return float("nan"), float("nan")
    ref_a = ref_a[:n]
    meas_a = meas_a[:n]
    max_k = min(int(round(float(max_lag_s) / float(dt))), n // 4)
    best_tau = 0.0
    best_rms = float("inf")
    found = False
    for k in range(-max_k, max_k + 1):
        if k >= 0:
            r = ref_a[: n - k]
            m = meas_a[k:]
        else:
            r = ref_a[-k:]
            m = meas_a[: n + k]
        valid = np.isfinite(r) & np.isfinite(m)
        if int(valid.sum()) < 20:
            continue
        resid = m[valid] - r[valid]
        rms = float(np.sqrt(np.mean(resid * resid)))
        if rms < best_rms:
            best_rms = rms
            best_tau = float(k) * float(dt)
            found = True
    if not found:
        return float("nan"), float("nan")
    return best_tau, best_rms


def err_vel_correlation(err: np.ndarray, vel: np.ndarray) -> float:
    """Pearson correlation of tracking error vs desired velocity."""
    e = np.asarray(err, dtype=float).reshape(-1)
    v = np.asarray(vel, dtype=float).reshape(-1)
    n = int(min(e.size, v.size))
    if n < 20:
        return float("nan")
    mask = np.isfinite(e[:n]) & np.isfinite(v[:n])
    if int(mask.sum()) < 20:
        return float("nan")
    ee = e[:n][mask]
    vv = v[:n][mask]
    if float(np.std(ee)) < 1.0e-12 or float(np.std(vv)) < 1.0e-12:
        return float("nan")
    return float(np.corrcoef(ee, vv)[0, 1])


def _latency_histogram(values: np.ndarray, edges_ms: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)) -> str:
    """Compact one-line histogram for Modbus read/write tails."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "hist n=0"
    bounds = (0.0, *tuple(float(e) for e in edges_ms), float("inf"))
    counts = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if math.isinf(hi):
            n = int(np.count_nonzero(finite >= lo))
            label = f">={lo:.0f}"
        else:
            n = int(np.count_nonzero((finite >= lo) & (finite < hi)))
            label = f"{lo:.0f}-{hi:.0f}"
        counts.append(f"{label}:{100.0 * n / finite.size:.0f}%")
    return "hist " + " ".join(counts)


def _parse_json_vec(raw) -> np.ndarray:
    if raw in ("", None):
        return np.empty(0, dtype=float)
    try:
        data = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return np.empty(0, dtype=float)
    if not isinstance(data, list):
        return np.empty(0, dtype=float)
    return np.asarray(
        [float(v) if v is not None else np.nan for v in data],
        dtype=float,
    )


def hysteresis_flip_count(series: np.ndarray, deadband: float) -> int:
    """Count sign changes that leave ``±deadband`` (ignore chatter around 0)."""
    state = 0
    flips = 0
    band = abs(float(deadband))
    for v in np.asarray(series, dtype=float).reshape(-1):
        if not np.isfinite(v):
            continue
        if state >= 0 and v < -band:
            if state != 0:
                flips += 1
            state = -1
        elif state <= 0 and v > band:
            if state != 0:
                flips += 1
            state = 1
    return int(flips)


def uneven_accel_from_qdot(qdot: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Non-uniform acceleration: ``a_k = 2 Δqdot / (Δt_k + Δt_{k-1})``."""
    q = np.asarray(qdot, dtype=float)
    t = np.asarray(t_s, dtype=float)
    n = int(min(q.shape[0], t.size))
    if n < 3:
        return np.empty(0, dtype=float)
    dt = np.diff(t[:n])
    dq = np.diff(q[:n], axis=0)
    acc = np.full((n - 1,) + q.shape[1:], np.nan, dtype=float)
    for k in range(1, n - 1):
        denom = float(dt[k] + dt[k - 1])
        if denom <= 0.0 or not np.isfinite(denom):
            continue
        acc[k] = 2.0 * dq[k] / denom
    return acc


def lomb_scargle_power(
    t_s: np.ndarray,
    y: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """Classical Lomb–Scargle periodogram for uneven samples."""
    t = np.asarray(t_s, dtype=float).reshape(-1)
    z = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(t) & np.isfinite(z)
    t = t[mask]
    z = z[mask]
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    power = np.full(freqs.shape, np.nan, dtype=float)
    if t.size < 8 or freqs.size == 0:
        return power
    z = z - float(z.mean())
    two_t = 2.0 * t
    for i, f in enumerate(freqs):
        if not np.isfinite(f) or f <= 0.0:
            continue
        w = 2.0 * math.pi * float(f)
        tan_2wt = math.atan2(float(np.sum(np.sin(w * two_t))), float(np.sum(np.cos(w * two_t))))
        tau = 0.5 * tan_2wt / w
        arg = w * (t - tau)
        c = np.cos(arg)
        s = np.sin(arg)
        cc = float(np.dot(c, c))
        ss = float(np.dot(s, s))
        if cc <= 1.0e-18 or ss <= 1.0e-18:
            continue
        power[i] = 0.5 * ((float(np.dot(z, c)) ** 2) / cc + (float(np.dot(z, s)) ** 2) / ss)
    return power


def band_power(t_s: np.ndarray, y: np.ndarray, f_lo: float, f_hi: float) -> float:
    freqs = np.linspace(float(f_lo), float(f_hi), 21)
    p = lomb_scargle_power(t_s, y, freqs)
    finite = p[np.isfinite(p)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _ab_run_invalid_reasons(rows: list[dict], *, psi_ref_deg: float = 68.0) -> list[str]:
    """Whole-run vetoes for the post-QP clamp A/B (any hit voids the run)."""
    reasons: list[str] = []
    if not rows:
        return ["empty"]
    for row in rows:
        if str(row.get("qpik_qp1_status", "") or "") != "solved":
            reasons.append("qp1_status")
            break
        if str(row.get("qpik_qp2_status", "") or "") != "solved":
            reasons.append("qp2_status")
            break
        if str(row.get("qpik_qp2_fallback", "0") or "0") not in {"0", "false", "False"}:
            reasons.append("qp2_fallback")
            break
        if str(row.get("qpik_solver_fault_latched", "0") or "0") not in {"0", "false", "False"}:
            reasons.append("solver_fault_latched")
            break
        if str(row.get("qpik_fallback_reason", "") or "") != "":
            reasons.append("fallback_reason")
            break
    psi = _col(rows, "psi_ref_deg")
    if psi.size and np.isfinite(psi).any():
        med = float(np.nanmedian(psi))
        if abs(med - float(psi_ref_deg)) > 0.2:
            reasons.append(f"psi_ref={med:.3f}")
    pre_raw = 0.0
    n_cmp = 0
    for row in rows:
        raw = _parse_json_vec(row.get("qpik_qdot_raw_json", ""))
        pre = _parse_json_vec(row.get("qpik_qdot_pre_commit_json", ""))
        if raw.size == 0 or pre.size == 0 or raw.size != pre.size:
            continue
        n_cmp += 1
        pre_raw = max(pre_raw, float(np.nanmax(np.abs(pre - raw))))
    if n_cmp == 0:
        reasons.append("missing_qdot_layers")
    elif pre_raw >= 1.0e-8:
        reasons.append(f"precommit_raw_inf={pre_raw:.3e}")
    return reasons


def _ab_band_and_flips(rows: list[dict]) -> tuple[float, float]:
    """55–65 Hz Lomb–Scargle power and hysteresis flip rate from send clock."""
    t_ns = _col(rows, "arm_send_mono_ns")
    if not np.isfinite(t_ns).any():
        return float("nan"), float("nan")
    t = t_ns * 1.0e-9
    qdots = []
    times = []
    for row, ti in zip(rows, t):
        if not np.isfinite(ti):
            continue
        vec = _parse_json_vec(row.get("arm_qdot_target_wall_json", ""))
        if vec.size == 0:
            q = _parse_json_vec(row.get("q_cmd_json", ""))
            if q.size >= 8 and times:
                dt = ti - times[-1]
                if dt > 0.0:
                    prev = _parse_json_vec(rows[len(times) - 1].get("q_cmd_json", ""))
                    if prev.size >= 8:
                        vec = (q[1:8] - prev[1:8]) / dt
        if vec.size >= 7:
            qdots.append(vec[:7])
            times.append(ti)
    if len(qdots) < 16:
        return float("nan"), float("nan")
    qdot = np.vstack(qdots)
    t_a = np.asarray(times, dtype=float)
    acc = uneven_accel_from_qdot(qdot, t_a)
    if acc.shape[0] < 8:
        return float("nan"), float("nan")
    t_acc = t_a[1:]
    powers = []
    flips = 0
    dur = float(t_a[-1] - t_a[0])
    for j in range(acc.shape[1]):
        powers.append(band_power(t_acc, acc[:, j], 55.0, 65.0))
        flips += hysteresis_flip_count(acc[:, j], 0.05 * 3.0)
    finite = [p for p in powers if np.isfinite(p)]
    p_band = float(np.mean(finite)) if finite else float("nan")
    flip_rate = float(flips) / dur if dur > 0.0 else float("nan")
    return p_band, flip_rate


def evaluate_post_qp_ab(
    true1_rows: list[dict],
    false_rows: list[dict],
    true2_rows: list[dict],
) -> dict:
    """Score True→False→True post-QP clamp A/B with the locked criteria."""
    report = {
        "true1_invalid": _ab_run_invalid_reasons(true1_rows),
        "false_invalid": _ab_run_invalid_reasons(false_rows),
        "true2_invalid": _ab_run_invalid_reasons(true2_rows),
        "verdict": "invalid",
    }
    if report["true1_invalid"] or report["false_invalid"] or report["true2_invalid"]:
        return report
    p1, f1 = _ab_band_and_flips(true1_rows)
    pf, ff = _ab_band_and_flips(false_rows)
    p2, f2 = _ab_band_and_flips(true2_rows)
    report.update(
        {
            "P_true1": p1,
            "P_false": pf,
            "P_true2": p2,
            "flips_true1": f1,
            "flips_false": ff,
            "flips_true2": f2,
        }
    )
    if not all(np.isfinite(v) for v in (p1, pf, p2, f1, ff, f2)):
        report["verdict"] = "invalid"
        report["reason"] = "spectrum_or_flips_nan"
        return report
    if p1 > 0.0 and p2 > 0.0 and abs(p1 - p2) / math.sqrt(p1 * p2) > 0.20:
        report["R_P"] = float("nan")
        report["verdict"] = "no_conclusion"
        report["reason"] = "true_drift>20%"
        return report
    if f1 > 0.0 and f2 > 0.0 and abs(f1 - f2) / math.sqrt(f1 * f2) > 0.20:
        report["R_P"] = float("nan")
        report["verdict"] = "no_conclusion"
        report["reason"] = "true_flip_drift>20%"
        return report
    geom = math.sqrt(p1 * p2)
    r_p = pf / geom if geom > 0.0 else float("nan")
    flip_geom = math.sqrt(f1 * f2)
    flip_drop = 1.0 - (ff / flip_geom) if flip_geom > 0.0 else float("nan")
    report["R_P"] = r_p
    report["flip_drop"] = flip_drop
    if np.isfinite(r_p) and r_p <= 0.5 and np.isfinite(flip_drop) and flip_drop >= 0.30:
        report["verdict"] = "amplifier"
    elif (
        abs(pf - geom) / geom <= 0.20
        and abs(ff - flip_geom) / flip_geom <= 0.20
    ):
        report["verdict"] = "unchanged"
    else:
        report["verdict"] = "inconclusive"
    return report


def _cmd_accel_spectrum(
    qi: np.ndarray, fs: float
) -> tuple[float, float] | None:
    """Peak frequency and 40-80 Hz power fraction of commanded acceleration."""
    q = np.asarray(qi, dtype=float)
    if q.size < 512 or not math.isfinite(float(fs)) or float(fs) <= 0.0:
        return None
    acc = np.diff(np.diff(q)) * (float(fs) ** 2)
    acc = acc[np.isfinite(acc)]
    if acc.size < 512:
        return None
    acc = acc - float(acc.mean())
    n = 1 << int(math.floor(math.log2(acc.size)))
    window = np.hanning(n)
    power = np.abs(np.fft.rfft(acc[:n] * window)) ** 2
    freq = np.fft.rfftfreq(n, 1.0 / float(fs))
    band = freq >= 2.0
    denom = float(power[band].sum()) if np.any(band) else 0.0
    if denom <= 0.0:
        return None
    peak = float(freq[band][int(np.argmax(power[band]))])
    high = float(power[(freq >= 40.0) & (freq < 80.0)].sum() / denom)
    return peak, high


def _col(rows: list[dict], name: str) -> np.ndarray:
    out = np.empty(len(rows))
    for i, row in enumerate(rows):
        raw = row.get(name, "")
        try:
            out[i] = float(raw) if raw not in ("", None) else np.nan
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def _col_any(rows: list[dict], *names: str) -> np.ndarray:
    """First column that has any finite values (new name, then legacy)."""
    empty = np.empty(0)
    for name in names:
        vals = _col(rows, name)
        if vals.size and np.isfinite(vals).any():
            return vals
        if vals.size:
            empty = vals
    return empty


def _encoder_diff_from_position(
    t: np.ndarray,
    x: np.ndarray,
    *,
    poll_hz: float = 60.0,
    span_ticks: int = 2,
) -> np.ndarray:
    """Bounded position difference matching the rail-servo ``v_enc`` path."""
    t_a = np.asarray(t, dtype=float)
    x_a = np.asarray(x, dtype=float)
    n = int(min(t_a.size, x_a.size))
    out = np.full(n, np.nan, dtype=float)
    period = 1.0 / max(float(poll_hz), 1.0)
    lo = 0.5 * period
    hi = 3.0 * period
    back = max(int(span_ticks), 1)
    for i in range(back, n):
        dt = t_a[i] - t_a[i - back]
        if lo <= dt <= hi and np.isfinite(x_a[i]) and np.isfinite(x_a[i - back]):
            out[i] = (x_a[i] - x_a[i - back]) / dt
    return out


def rail_stop_reverse_frac(
    t: np.ndarray,
    v_goal: np.ndarray,
    v_enc: np.ndarray,
    *,
    entry_m_s: float = 0.015,
    zero_m_s: float = 0.005,
    window_s: float = 0.40,
    entry_window_s: float = 0.20,
) -> float:
    """Worst reverse-peak / entry-speed after ``v_goal`` falls to ~0.

    Returns NaN when no stop event is found.  A plugging-brake stop that
    reverses at 50–60% of entry speed scores ~0.5–0.6.

    Entry speed is the fastest ``v_enc`` in ``entry_window_s`` before the
    stop, and stops entering below ``entry_m_s`` are skipped.  Reading a
    single sample at the backtrack index instead let one near-zero
    quantisation sample divide the ratio into 2632175%.
    """
    t_a = np.asarray(t, dtype=float)
    vg = np.asarray(v_goal, dtype=float)
    ve = np.asarray(v_enc, dtype=float)
    n = int(min(t_a.size, vg.size, ve.size))
    if n < 8:
        return float("nan")
    t_a = t_a[:n]
    vg = vg[:n]
    ve = ve[:n]
    worst = float("nan")
    i = 1
    while i < n:
        if not (np.isfinite(vg[i]) and np.isfinite(vg[i - 1])):
            i += 1
            continue
        if abs(float(vg[i])) > float(zero_m_s):
            i += 1
            continue
        if abs(float(vg[i - 1])) <= float(zero_m_s):
            i += 1
            continue
        j = i - 1
        while j >= 0 and (not np.isfinite(vg[j]) or abs(float(vg[j])) < float(entry_m_s)):
            j -= 1
        if j < 0:
            i += 1
            continue
        t_stop = float(t_a[i])
        entry_mask = (
            np.isfinite(t_a)
            & np.isfinite(ve)
            & (t_a >= t_stop - float(entry_window_s))
            & (t_a <= t_stop)
        )
        v_entry = 0.0
        if np.any(entry_mask):
            entry_vals = ve[entry_mask]
            v_entry = float(entry_vals[int(np.argmax(np.abs(entry_vals)))])
        if abs(v_entry) < float(entry_m_s) and np.isfinite(vg[j]):
            if abs(float(vg[j])) > abs(v_entry):
                v_entry = float(vg[j])
        if abs(v_entry) < float(entry_m_s):
            i += 1
            continue
        # Only score while the goal is still asking for a stop.  A gamepad
        # goal that dips through zero and drives the other way is a new
        # command, not the brake reversing itself.
        end = i
        while (
            end + 1 < n
            and float(t_a[end + 1]) <= t_stop + float(window_s)
            and (
                not np.isfinite(vg[end + 1])
                or abs(float(vg[end + 1])) <= float(zero_m_s)
            )
        ):
            end += 1
        mask = np.zeros(n, dtype=bool)
        mask[i : end + 1] = True
        mask &= np.isfinite(t_a) & np.isfinite(ve)
        if not np.any(mask):
            i += 1
            continue
        sign = 1.0 if v_entry >= 0.0 else -1.0
        peak = float(np.max(-sign * ve[mask]))
        frac = peak / abs(v_entry)
        if not np.isfinite(worst) or frac > worst:
            worst = frac
        while i < n and float(t_a[i]) <= t_stop + float(window_s):
            i += 1
    return worst


def _vpc_midrange_checks(
    rows: list[dict],
    results: list[tuple[str, bool, str]],
) -> None:
    """Phase 5 VPC mid-ranging gates from the QPIK CSV."""
    share = _col(rows, "rail_motion_share")
    vy = _col(rows, "v_cmd_vy")
    if not np.isfinite(vy).any():
        vy = _col(rows, "path_twist_vy")
    moving = np.isfinite(share) & np.isfinite(vy) & (np.abs(vy) > GATES["rail_share_vy_m_s"])
    if int(moving.sum()) >= 20:
        p50 = float(np.nanmedian(share[moving]))
        results.append(
            (
                "rail_motion_share p50 ≥ 0.60 (|v_cmd_vy| > 20 mm/s)",
                p50 >= GATES["rail_share_p50"],
                f"p50 {p50:.3f}  n={int(moving.sum())}",
            )
        )
        plus = moving & (vy > GATES["rail_share_vy_m_s"])
        minus = moving & (vy < -GATES["rail_share_vy_m_s"])
        if int(plus.sum()) >= 10 and int(minus.sum()) >= 10:
            r_plus = float(np.nanmedian(np.abs(share[plus])))
            r_minus = float(np.nanmedian(np.abs(share[minus])))
            denom = max(min(r_plus, r_minus), 1.0e-6)
            ratio = max(r_plus, r_minus) / denom
            results.append(
                (
                    "+Y/−Y rail share ratio ≤ 1.25",
                    ratio <= GATES["rail_share_pm_ratio"],
                    f"+ {r_plus:.3f}  − {r_minus:.3f}  ratio {ratio:.2f}",
                )
            )
    psi = _col(rows, "psi_deg")
    psi_ref = _col(rows, "psi_ref_deg")
    if np.isfinite(psi).any() and np.isfinite(psi_ref).any():
        dpsi = np.abs(((psi - psi_ref + 180.0) % 360.0) - 180.0)
        finite = dpsi[np.isfinite(dpsi)]
        if finite.size:
            p95 = float(np.nanpercentile(finite, 95))
            results.append(
                (
                    "|ψ − ψ_ref| p95 ≤ 15°",
                    p95 <= GATES["psi_err_p95_deg"],
                    f"p95 {p95:.1f}°",
                )
            )
    vref = _col(rows, "v_r_ref")
    live_y = np.isfinite(vref) & np.isfinite(vy) & (np.abs(vy) > GATES["rail_sign_vy_m_s"])
    if int(live_y.sum()) >= 20:
        agree = float(np.mean(np.sign(vref[live_y]) == np.sign(vy[live_y])))
        results.append(
            (
                "sign(v_r_ref)==sign(v_cmd_vy) ≥ 85% (|vy|>10 mm/s)",
                agree >= GATES["rail_sign_agree_frac"],
                f"{100.0 * agree:.1f}%",
            )
        )
    err = _col(rows, "track_err_mm")
    if not np.isfinite(err).any():
        err = _col(rows, "motion_err_rms_mm")
    finite_err = np.abs(err[np.isfinite(err)])
    if finite_err.size:
        p95 = float(np.nanpercentile(finite_err, 95))
        results.append(
            (
                "track_err p95 ≤ 5 mm",
                p95 <= GATES["vpc_track_err_p95_mm"],
                f"{p95:.2f} mm",
            )
        )


def _rail_servo_vpc_checks(
    rows: list[dict],
    results: list[tuple[str, bool, str]],
) -> None:
    """FA24 write-rate / step-size gates from the rail_servo CSV."""
    rpm = _col(rows, "rpm_cmd")
    t = _col(rows, "t_wall_s")
    tw = _col(rows, "t_write_ms")
    writes = np.isfinite(tw) & (tw > 0.05)
    t_w = t[writes] if t.size == writes.size else np.array([])
    if t_w.size > 2:
        span = float(t_w[-1] - t_w[0])
        hz = (float(t_w.size) - 1.0) / max(span, 1.0e-6)
        results.append(
            (
                "FA24 write ≥ 40 Hz (active window)",
                hz >= GATES["fa24_write_hz"],
                f"{hz:.1f} Hz  n={int(t_w.size)}",
            )
        )
    finite_rpm = rpm[np.isfinite(rpm)]
    if finite_rpm.size > 2:
        drpm = np.abs(np.diff(finite_rpm))
        drpm = drpm[drpm > 0.5]
        if drpm.size:
            p95 = float(np.percentile(drpm, 95))
            results.append(
                (
                    "FA24 |Δrpm| p95 ≤ 20",
                    p95 <= GATES["fa24_drpm_p95"],
                    f"p95 {p95:.1f} rpm",
                )
            )


def _rail_servo_checks(
    scan_path: Path,
    results: list[tuple[str, bool, str]],
    info: list[tuple[str, str]],
) -> None:
    """Score the carriage servo from its sibling log.

    The rail is a separate Modbus servo with its own shaper, so the QPIK CSV
    cannot see whether it actually tracked.  Its measured position in the QPIK
    log is a zero-order hold (stale on ~79% of ticks) and differentiating that
    only yields the sampling artefact, not real motion.
    """
    stamp = scan_path.stem.replace("run_", "")
    servo = scan_path.parent.parent / "rail_servo" / f"rail_{stamp}.csv"
    if not servo.exists():
        info.append(("rail servo log", f"not found ({servo.name})"))
        return
    with servo.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 50:
        info.append(("rail servo log", f"only {len(rows)} rows"))
        return
    _rail_servo_vpc_checks(rows, results)

    t = _col(rows, "t_wall_s")
    span = float(t[-1] - t[0]) if t.size > 1 else 0.0
    dtw = _col(rows, "dt_wall_ms")
    dtw = dtw[np.isfinite(dtw)]
    if dtw.size:
        nom_ms = 1000.0 * GATES["rail_period_nominal_s"]
        on_time = float(
            np.mean((dtw > 0.9 * nom_ms) & (dtw < 1.1 * nom_ms))
        )
        info.append(
            (
                "rail servo loop",
                f"med {np.median(dtw):.1f} ms ({1000.0 / max(np.median(dtw), 1e-6):.0f} Hz)"
                f"  p95 {np.percentile(dtw, 95):.1f} ms  max {dtw.max():.1f} ms",
            )
        )
        results.append(
            (
                f"rail period on-time > 80% of {nom_ms:.1f} ms",
                on_time > GATES["rail_period_on_time_frac"],
                f"{100.0 * on_time:.1f}% within ±10% of {nom_ms:.1f} ms",
            )
        )
        for name in ("t_read_ms", "t_write_ms"):
            lat = _col(rows, name)
            lat = lat[np.isfinite(lat)]
            if lat.size:
                info.append(
                    (
                        f"rail {name}",
                        f"p50 {np.median(lat):.2f}  p95 {np.percentile(lat, 95):.2f}  "
                        f"max {lat.max():.1f} ms  {_latency_histogram(lat)}",
                    )
                )
        tw = _col(rows, "t_write_ms")
        if dtw.size and tw.size == _col(rows, "dt_wall_ms").size:
            dt_all = _col(rows, "dt_wall_ms")
            if tw.size > 8:
                x = tw[:-1]
                y = dt_all[1:]
                mask = np.isfinite(x) & np.isfinite(y) & (x > 0.5)
                if int(mask.sum()) >= 8 and float(np.std(x[mask])) > 1.0e-6:
                    corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
                    results.append(
                        (
                            "rail t_write vs next dt_wall |corr| < 0.30",
                            abs(corr) < GATES["rail_write_dt_corr"],
                            f"corr {corr:.2f}  n={int(mask.sum())}",
                        )
                    )
    age = _col(rows, "target_age_ms")
    follow = _col(rows, "follow")
    if np.isfinite(age).any():
        live_age = np.isfinite(age)
        if np.isfinite(follow).any():
            live_age &= follow > 0.5
        if np.any(live_age):
            p95_age = float(np.nanpercentile(age[live_age], 95))
            results.append(
                (
                    "rail target_age p95 < 50 ms (live follow)",
                    p95_age < GATES["rail_target_age_p95_ms"],
                    f"p95 {p95_age:.1f} ms",
                )
            )

    follow = _col(rows, "follow")
    age = _col(rows, "target_age_ms")
    live = np.ones(len(rows), dtype=bool)
    if np.isfinite(follow).any():
        live &= follow > 0.5
    if np.isfinite(age).any():
        med_age = float(np.nanmedian(age[np.isfinite(age)]))
        fresh_lim = max(50.0, 2.0 * med_age) if np.isfinite(med_age) else 50.0
        live &= np.isfinite(age) & (age <= fresh_lim)
    t_live = t[live]
    live_span = float(t_live[-1] - t_live[0]) if t_live.size > 1 else 0.0
    a_cmd = _col(rows, "a_cmd_m_s2")
    a_live = a_cmd[live & np.isfinite(a_cmd)]
    if a_live.size > 2 and live_span > 0.0:
        big = a_live[np.abs(a_live) > 0.05]
        rev = 0.0
        if big.size > 1:
            rev = float(
                np.count_nonzero(np.sign(big[1:]) != np.sign(big[:-1])) / live_span
            )
        results.append(
            (
                "rail servo accel reversals < 3/s (live follow)",
                rev < GATES["rail_servo_accel_reversals_per_s"],
                f"{rev:.1f}/s  |a| p95 {np.percentile(np.abs(a_live), 95):.2f} m/s²"
                f"  live {int(np.count_nonzero(live))}/{len(rows)}",
            )
        )
    elif a_cmd[np.isfinite(a_cmd)].size > 2 and span > 0.0:
        info.append(
            (
                "rail servo accel reversals",
                "no live follow=1 / fresh target_age window; skipped idle dilution",
            )
        )
        info.append(
            (
                "mid-scan jerk",
                "compare this a_cmd rate to apps/lw100_isolated_sine_track.py; "
                "command jerk RMS is L0 only (honor d* is not a mid-jerk gate)",
            )
        )

    e_track = _col(rows, "e_track_mm")
    e_track = e_track[np.isfinite(e_track)]
    if e_track.size:
        p95 = float(np.percentile(np.abs(e_track), 95))
        results.append(
            (
                "rail servo |e_track| p95 < 2 mm",
                p95 < GATES["rail_servo_track_err_p95_mm"],
                f"{p95:.2f} mm  max {np.abs(e_track).max():.2f} mm",
            )
        )
    e_shape = _col(rows, "e_shape_mm")
    e_shape_live = e_shape[live & np.isfinite(e_shape)]
    if e_shape_live.size:
        p95_shape = float(np.percentile(np.abs(e_shape_live), 95))
        results.append(
            (
                "rail |e_shape| p95 < 2 mm (coupled reference drift)",
                p95_shape < GATES["rail_eshape_p95_mm"],
                f"{p95_shape:.2f} mm  max {np.abs(e_shape_live).max():.2f} mm",
            )
        )
    v_enc_box = _col(rows, "v_enc_m_s")
    v_box_live = v_enc_box[live & np.isfinite(v_enc_box)]
    if v_box_live.size:
        v_over = float(np.mean(np.abs(v_box_live) > GATES["rail_v_box_m_s"]))
        results.append(
            (
                "rail |v_enc| over QP box < 1%",
                v_over < GATES["rail_v_box_frac"],
                f"{100.0 * v_over:.1f}%  max {1000.0 * np.max(np.abs(v_box_live)):.1f} mm/s",
            )
        )
    a_box_live = a_cmd[live & np.isfinite(a_cmd)]
    if a_box_live.size:
        a_over = float(np.mean(np.abs(a_box_live) > GATES["rail_a_box_m_s2"]))
        results.append(
            (
                "rail |a_cmd| over QP box < 1%",
                a_over < GATES["rail_a_box_frac"],
                f"{100.0 * a_over:.1f}%  max {np.max(np.abs(a_box_live)):.2f} m/s²",
            )
        )

    age = _col(rows, "target_age_ms")
    age = age[np.isfinite(age)]
    if age.size:
        info.append(
            (
                "rail target age",
                f"med {np.median(age):.2f} ms  p95 {np.percentile(age, 95):.2f} ms"
                f"  max {age.max():.0f} ms",
            )
        )

    v_enc = _col(rows, "v_enc_m_s")
    if not np.isfinite(v_enc).any():
        v_enc = _encoder_diff_from_position(t, _col(rows, "x_meas_m"))
    sources = [str(r.get("v_enc_source", "") or "") for r in rows]
    live_sources = [s for s, keep in zip(sources, live) if keep and s]
    if live_sources:
        n_src = len(live_sources)
        reg = sum(1 for s in live_sources if s == "reg") / n_src
        hold = sum(1 for s in live_sources if s == "hold") / n_src
        results.append(
            (
                "rail v_enc register fallback < 2% (live follow)",
                reg < GATES["rail_v_enc_register_frac"],
                f"reg {100.0 * reg:.1f}%  hold {100.0 * hold:.1f}%  n={n_src}",
            )
        )
    v_goal = _col(rows, "v_goal_est_m_s")
    rev_frac = rail_stop_reverse_frac(t, v_goal, v_enc)
    if np.isfinite(rev_frac):
        results.append(
            (
                "rail stop reverse < 15% of entry",
                rev_frac < GATES["rail_stop_reverse_frac"],
                f"{100.0 * rev_frac:.1f}% of entry",
            )
        )
    else:
        info.append(("rail stop reverse", "no v_goal→0 event"))

    v_des = _col(rows, "v_des_m_s")
    v_ref = _col(rows, "v_ref_m_s")
    leak_n = 0
    leak_hits = 0
    turn_ratios: list[float] = []
    for keep, vg, vd, vr in zip(live, v_goal, v_des, v_ref):
        if not keep:
            continue
        if not (np.isfinite(vg) and np.isfinite(vd) and np.isfinite(vr)):
            continue
        leak_n += 1
        if abs(vg) > 1.0e-12 and (vd - vr) * float(np.sign(vg)) < -GATES[
            "rail_p_term_leak_m_s"
        ]:
            leak_hits += 1
        if 1.0e-6 < abs(vg) < GATES["rail_turn_v_goal_m_s"]:
            turn_ratios.append(abs(vd) / abs(vg))
    if leak_n:
        frac = leak_hits / leak_n
        results.append(
            (
                "rail P-term leak < 0.1% of live ticks",
                frac < GATES["rail_p_term_leak_frac"],
                f"{100.0 * frac:.2f}%  ({leak_hits}/{leak_n})",
            )
        )
    if turn_ratios:
        p99 = float(np.percentile(turn_ratios, 99))
        results.append(
            (
                "rail turn overspeed p99 < 2 (|v_goal|<5 mm/s)",
                p99 < GATES["rail_turn_overspeed_p99"],
                f"p99 {p99:.2f}  n={len(turn_ratios)}",
            )
        )


def _finite6(row: dict, keys: tuple[str, ...]) -> np.ndarray | None:
    vals = []
    for key in keys:
        raw = row.get(key, "")
        try:
            val = float(raw) if raw not in ("", None) else float("nan")
        except (TypeError, ValueError):
            return None
        if not np.isfinite(val):
            return None
        vals.append(val)
    return np.asarray(vals, dtype=float)


def _q8_from_row(row: dict) -> np.ndarray | None:
    q = _finite6(row, tuple(f"q_cmd_{i}" for i in range(8)))
    if q is not None:
        return q
    return _finite6(row, tuple(f"q_meas_{i}" for i in range(8)))


def _q8_meas_from_row(row: dict) -> np.ndarray | None:
    return _finite6(row, tuple(f"q_meas_{i}" for i in range(8)))


def _qdot_sent_from_row(row: dict) -> np.ndarray | None:
    raw = row.get("qpik_final_sent_qdot_json", "")
    if raw in ("", None):
        return None
    try:
        vals = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    arr = np.asarray(vals, dtype=float).reshape(-1)
    if arr.size != 8 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _d_star_step_check(
    d_star: np.ndarray,
    dt_wall: np.ndarray,
    results: list[tuple[str, bool, str]],
    info: list[tuple[str, str]],
) -> None:
    if d_star.size < 3 or not np.isfinite(d_star).any():
        return
    dd = np.abs(np.diff(d_star))
    dd = dd[np.isfinite(dd)]
    if not dd.size:
        return
    dt_med = float(np.median(dt_wall)) if dt_wall.size else GATES["dt_nominal_s"]
    limit = GATES["d_star_rate_m_s"] * dt_med * GATES["d_star_step_margin"]
    peak = float(np.max(dd))
    results.append(
        (
            "d_star step max < 2 × d_center_rate × dt",
            peak < limit,
            f"{1000.0 * peak:.2f} mm  limit {1000.0 * limit:.2f} mm",
        )
    )


def _joint_exec_ratio_check(
    rows: list[dict],
    t: np.ndarray,
    results: list[tuple[str, bool, str]],
    info: list[tuple[str, str]],
) -> None:
    qdots: list[np.ndarray] = []
    qmeas: list[np.ndarray] = []
    times: list[float] = []
    for row, ti in zip(rows, t):
        qd = _qdot_sent_from_row(row)
        qm = _q8_meas_from_row(row)
        if qd is None or qm is None or not np.isfinite(ti):
            continue
        qdots.append(qd)
        qmeas.append(qm)
        times.append(float(ti))
    if len(times) < 20:
        if not any(r.get("qpik_final_sent_qdot_json") for r in rows[:8]):
            info.append(("joint exec ratio", "no qpik_final_sent_qdot_json"))
        return
    qdots_a = np.asarray(qdots, dtype=float)
    qmeas_a = np.asarray(qmeas, dtype=float)
    times_a = np.asarray(times, dtype=float)
    dt = np.diff(times_a)
    good = np.isfinite(dt) & (dt > 0.0) & (dt < 0.10)
    if int(np.count_nonzero(good)) < 10:
        info.append(("joint exec ratio", "too few finite wall periods"))
        return
    integ = np.sum(qdots_a[:-1][good] * dt[good, None], axis=0)
    # Match the integrated interval: q[0] → q[last good dt].
    idx = np.nonzero(good)[0]
    dq = qmeas_a[idx[-1] + 1] - qmeas_a[idx[0]]
    parts: list[str] = []
    ok = True
    scored = 0
    for j in range(8):
        if abs(float(integ[j])) < GATES["joint_exec_min_integral"]:
            continue
        ratio = float(dq[j] / integ[j])
        scored += 1
        parts.append(f"j{j} {ratio:.3f}")
        if not (GATES["joint_exec_ratio_lo"] <= ratio <= GATES["joint_exec_ratio_hi"]):
            ok = False
    if not scored:
        info.append(("joint exec ratio", "no joint with |∫qdot dt_wall| ≥ 0.01"))
        return
    results.append(
        (
            "joint exec ratio 0.9–1.1 (Δq_meas / ∫qdot·dt_wall)",
            ok,
            "  ".join(parts),
        )
    )


def _ik_exists_7dof(
    pose_d: np.ndarray,
    y_rail: float,
    *,
    q_hint: np.ndarray | None = None,
    kin=None,
) -> bool:
    """True if a URDF-box 7DOF IK exists at locked ``y_rail``."""
    from rm75_control.kinematics.srs_ik import (
        branch_from_q,
        d_wt_from_kin,
        flange_tcp_from_kin,
        psi_from_q,
        shoulder_y_from_q_rail,
        srs_ik,
    )

    kwargs: dict = {
        "y_rail": float(shoulder_y_from_q_rail(y_rail)),
        "check_limits": True,
    }
    if kin is not None:
        try:
            r_off, t_off = flange_tcp_from_kin(kin)
            kwargs["R_flange_tcp"] = r_off
            kwargs["t_flange_tcp"] = t_off
            kwargs["d_wt"] = d_wt_from_kin(kin)
        except Exception:
            pass
    hints: list[tuple[float, int]] = []
    if q_hint is not None:
        try:
            hints.append((float(psi_from_q(q_hint)), int(branch_from_q(q_hint))))
        except Exception:
            pass
    if not hints:
        hints.append((0.0, 0))
    seen: set[tuple[int, int]] = set()
    extras = (0.0, 0.5, -0.5, 1.0, -1.0, 1.57, -1.57)
    for psi0, branch0 in hints:
        for dpsi in extras:
            psi = float(psi0 + dpsi)
            for branch in range(8) if dpsi == 0.0 else (branch0,):
                key = (int(branch), int(round(psi * 1000.0)))
                if key in seen:
                    continue
                seen.add(key)
                if srs_ik(pose_d, psi, int(branch), **kwargs) is not None:
                    return True
    return False


def _rail_handoff_checks(
    rows: list[dict],
    rail: np.ndarray,
    err: np.ndarray,
    results: list[tuple[str, bool, str]],
    info: list[tuple[str, str]],
) -> None:
    """At the rail wall, IK-feasible ticks must keep tool-Y error < 3 mm."""
    band = GATES["rail_limit_band_m"]
    at_limit = np.isfinite(rail) & (
        (rail < GATES["rail_min_m"] + band) | (rail > GATES["rail_max_m"] - band)
    )
    n_limit = int(np.count_nonzero(at_limit))
    if n_limit < 50:
        info.append(("rail wall handoff", "rail never entered the band"))
        return

    idxs = np.flatnonzero(at_limit)
    step = max(1, idxs.size // 12)
    sample = idxs[::step][:12]
    kin = None
    try:
        from rm75_control.control.joint_admittance_8dof.model import RobotKinematics

        kin = RobotKinematics()
    except Exception:
        kin = None

    feasible = 0
    checked = 0
    for i in sample:
        pose_d = _finite6(
            rows[int(i)],
            ("pose_d_x", "pose_d_y", "pose_d_z", "pose_d_rx", "pose_d_ry", "pose_d_rz"),
        )
        q = _q8_from_row(rows[int(i)])
        if pose_d is None:
            continue
        checked += 1
        y_rail = float(rail[int(i)])
        if _ik_exists_7dof(pose_d, y_rail, q_hint=q, kin=kin):
            feasible += 1

    if checked == 0:
        info.append(
            (
                "rail wall 7DOF IK",
                f"{n_limit} ticks in band but no pose_d columns; skip IK gate",
            )
        )
        return

    frac = feasible / max(checked, 1)
    info.append(
        (
            "rail wall 7DOF IK",
            f"{feasible}/{checked} sampled ticks feasible at locked q0 "
            f"({n_limit} ticks in band)",
        )
    )
    if feasible == 0:
        info.append(
            (
                "rail wall track_err",
                "no 7DOF IK in the band (workspace hole); slack allowed",
            )
        )
        return

    band_err = err[at_limit]
    band_err = band_err[np.isfinite(band_err)]
    e95 = (
        float(np.nanpercentile(np.abs(band_err), 95))
        if band_err.size
        else float("nan")
    )
    results.append(
        (
            "IK-feasible rail wall: tool_y_err p95 < 3 mm",
            bool(np.isfinite(e95) and e95 < GATES["track_err_at_limit_mm"]),
            f"{e95:.2f} mm  (IK {frac:.0%} of samples)",
        )
    )


def _idle_hold_checks(
    rows: list[dict],
    results: list[tuple[str, bool, str]],
    info: list[tuple[str, str]],
    *,
    min_idle_s: float = 0.4,
) -> None:
    """The rail and the TCP must both stand still once the operator lets go.

    Score TCP hold from ``pose_meas`` latched at idle start, not
    ``tool_y_err_mm``.  Idle ``pose_d`` is now latched, but the physical
    drift gate still uses measured pose.
    """
    t = _col(rows, "t_wall_s")
    if t.size < 8:
        return
    req = np.zeros(t.size, dtype=float)
    have_request = False
    for axis in ("vx", "vy", "vz", "wx", "wy", "wz"):
        vals = _col(rows, f"twist_requested_{axis}")
        if vals.size == t.size and np.isfinite(vals).any():
            have_request = True
            req = np.maximum(req, np.abs(np.nan_to_num(vals, nan=0.0)))
    if not have_request:
        info.append(("idle hold", "no twist_requested_* columns"))
        return
    idle = req < 1.0e-6
    if not np.any(idle):
        info.append(("idle hold", "no twist_requested=0 window"))
        return

    rail = _col(rows, "rail_meas_m")
    slack = _col(rows, "slack_norm")
    pose = np.stack(
        [_col(rows, f"pose_meas_{a}") for a in ("x", "y", "z")], axis=1
    )
    travels: list[float] = []
    drifts: list[float] = []
    slack_hits = 0
    slack_n = 0
    i = 0
    while i < t.size:
        if not idle[i]:
            i += 1
            continue
        j = i
        while j + 1 < t.size and idle[j + 1]:
            j += 1
        if float(t[j] - t[i]) >= float(min_idle_s):
            seg_rail = rail[i : j + 1]
            seg_rail = seg_rail[np.isfinite(seg_rail)]
            if seg_rail.size > 1:
                travels.append(
                    float(np.max(np.abs(seg_rail - seg_rail[0]))) * 1000.0
                )
            seg_pose = pose[i : j + 1]
            good = np.all(np.isfinite(seg_pose), axis=1)
            if int(np.count_nonzero(good)) > 1:
                anchored = seg_pose[good]
                drifts.append(
                    float(
                        np.max(np.linalg.norm(anchored - anchored[0], axis=1))
                    )
                    * 1000.0
                )
            seg_slack = slack[i : j + 1]
            seg_slack = seg_slack[np.isfinite(seg_slack)]
            slack_n += int(seg_slack.size)
            slack_hits += int(np.count_nonzero(seg_slack > 1.0e-6))
        i = j + 1

    if not travels:
        info.append(("idle hold", "no idle window longer than 0.4 s"))
        return

    travel_p95 = float(np.percentile(travels, 95))
    results.append(
        (
            "idle rail travel p95 < 8 mm",
            travel_p95 < GATES["idle_rail_travel_mm"],
            f"{travel_p95:.1f} mm  max {max(travels):.1f} mm  n={len(travels)}",
        )
    )
    if drifts:
        drift_p95 = float(np.percentile(drifts, 95))
        results.append(
            (
                "idle TCP drift p95 < 1 mm (pose_meas latched)",
                drift_p95 < GATES["idle_tcp_drift_mm"],
                f"{drift_p95:.1f} mm  max {max(drifts):.1f} mm  n={len(drifts)}",
            )
        )
    if slack_n:
        frac = slack_hits / slack_n
        results.append(
            (
                "idle QP1 task slack < 5% of ticks",
                frac < GATES["idle_task_slack_frac"],
                f"{100.0 * frac:.1f}%  ({slack_hits}/{slack_n} ticks)",
            )
        )


def _posture_debt_check(
    rows: list[dict],
    results: list[tuple[str, bool, str]],
    info: list[tuple[str, str]],
    *,
    drive_ff_m_s: float = 0.07,
) -> None:
    """While driving hard, the rail must stay near its preferred extension.

    A shared velocity budget that cannot hold FF *and* reach starves reach
    for the whole stroke and dumps the accumulated error on release, so
    this is the gate that sees the conflict before the slide happens.
    """
    ff = _col(rows, "rail_qdot_ff")
    err = _col_any(rows, "rail_posture_err_m", "rail_track_err_m")
    n = int(min(ff.size, err.size))
    if n < 8:
        return
    driving = np.isfinite(ff[:n]) & np.isfinite(err[:n]) & (
        np.abs(ff[:n]) >= float(drive_ff_m_s)
    )
    if int(np.count_nonzero(driving)) < 20:
        info.append(("rail posture debt", "no sustained hard-drive window"))
        return
    p95 = float(np.percentile(np.abs(err[:n][driving]), 95))
    results.append(
        (
            "driving |rail_posture_err| p95 < 30 mm",
            p95 < GATES["drive_rail_posture_err_p95_m"],
            f"{1000.0 * p95:.1f} mm  n={int(np.count_nonzero(driving))}",
        )
    )


def _rail_task_dropout_check(
    rows: list[dict],
    results: list[tuple[str, bool, str]],
    info: list[tuple[str, str]],
) -> None:
    """w_ext=0 must not pin the rail to 0 while feedforward is still live."""
    tv = _col(rows, "rail_task_vel")
    ff = _col(rows, "v_ff_rail")
    t = _col(rows, "t_wall_s")
    n = int(min(tv.size, ff.size))
    if n < 8:
        return
    dead = ~np.isfinite(tv[:n])
    live_ff = np.isfinite(ff[:n]) & (np.abs(ff[:n]) > GATES["rail_task_dropout_ff_m_s"])
    hit = dead & live_ff
    frac = float(np.mean(hit)) if n else 0.0
    longest_s = 0.0
    if t.size >= n and np.isfinite(t[:n]).any():
        i = 0
        while i < n:
            if not hit[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and hit[j + 1]:
                j += 1
            if np.isfinite(t[i]) and np.isfinite(t[j]):
                longest_s = max(longest_s, float(t[j] - t[i]))
            i = j + 1
    results.append(
        (
            "rail task dropout < 1% while |v_ff| live",
            frac < GATES["rail_task_dropout_frac"],
            f"{100.0 * frac:.1f}%  longest {1000.0 * longest_s:.0f} ms  n={int(np.count_nonzero(hit))}",
        )
    )


def _posture_followup(
    rows: list[dict],
    info: list[tuple[str, str]],
) -> None:
    """J5 / J4 / J6 parks from existing columns; do not retune q_nominal here.

    Hardware CSVs parked J5 at −15° (nominal +40° never won), J4 at the
    comfort stop (~120° = 135°−15°), and J6 closed.  Pose-task roll lock
    starves centering; buying comfort/branch slack is cheaper than opening
    the elbow/wrist.  Wall handoff now *raises* those slack costs.
    """
    j4 = np.degrees(_col(rows, "q_meas_4"))
    j5 = np.degrees(_col(rows, "q_meas_5"))
    j6 = np.degrees(_col(rows, "q_meas_6"))
    if not np.isfinite(j4).any():
        j4 = np.degrees(_col(rows, "q_cmd_4"))
    if not np.isfinite(j5).any():
        j5 = np.degrees(_col(rows, "q_cmd_5"))
    if not np.isfinite(j6).any():
        j6 = np.degrees(_col(rows, "q_cmd_6"))
    if np.isfinite(j5).any():
        info.append(
            (
                "J5 vs nominal +40°",
                f"median {float(np.nanmedian(j5)):.1f}°  "
                f"(pose-task roll lock beats centering; do not retune q_nominal "
                f"until feedback_twist / nullspace_norm are logged)",
            )
        )
    if np.isfinite(j4).any():
        info.append(
            (
                "J4 comfort park",
                f"max {float(np.nanmax(j4)):.1f}°  "
                f"(120° = 135° limit − 15° comfort; wall now raises pref slack)",
            )
        )
    if np.isfinite(j6).any():
        closed = float(np.nanmean(np.abs(j6) < 15.0))
        info.append(
            (
                "J6 close",
                f"{100.0 * closed:.1f}% |J6|<15°  min {float(np.nanmin(np.abs(j6))):.1f}°",
            )
        )
    ns = _col(rows, "qpik_nullspace_norm")
    fb = _col(rows, "feedback_twist_wz")
    if np.isfinite(ns).any() or np.isfinite(fb).any():
        info.append(
            (
                "posture nullspace / feedback",
                (
                    f"nullspace p50 {float(np.nanmedian(ns)):.4f}  "
                    if np.isfinite(ns).any()
                    else ""
                )
                + (
                    f"fb_wz p95 {float(np.nanpercentile(np.abs(fb[np.isfinite(fb)]), 95)):.4f}"
                    if np.isfinite(fb).any()
                    else "feedback_twist not logged"
                ),
            )
        )


def _tick_profile(rows: list[dict], info: list[tuple[str, str]]) -> None:
    """Attribute the per-tick budget so the period overrun is not a guess."""
    stages = [
        ("qpik_solver_solve_ms", "QP solve"),
        ("tick_inner_ms", "inner.update (incl. QP)"),
        ("tick_send_ms", "rail publish + CANFD send"),
        ("tick_log_ms", "CSV write"),
    ]
    shown = False
    for key, label in stages:
        a = _col(rows, key)
        a = a[np.isfinite(a)]
        if not a.size:
            continue
        shown = True
        info.append(
            (
                f"tick stage: {label}",
                f"med {np.median(a):.3f} ms  p95 {np.percentile(a, 95):.3f} ms"
                f"  max {a.max():.2f} ms",
            )
        )
    if not shown:
        info.append(
            ("tick stage profile", "not logged (older CSV, re-run to populate)")
        )


def analyze(path: Path) -> int:
    with path.open(newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    scan_rows = [r for r in all_rows if r.get("phase") == "scan"]
    if scan_rows:
        rows = scan_rows
        phase_used = "scan"
    else:
        rows = all_rows
        labels = sorted({str(r.get("phase") or "") for r in rows})
        phase_used = ",".join(labels) if labels else "(none)"
    if not rows:
        print("no rows", file=sys.stderr)
        return 2

    t = _col(rows, "t_wall_s")
    rail = _col(rows, "q_meas_0")
    if not np.isfinite(rail).any():
        rail = _col(rows, "rail_meas_m")
    j4 = _col(rows, "q_meas_4")
    j7 = _col(rows, "q_meas_7")
    pose_y = _col(rows, "pose_meas_y")
    waste = _col(rows, "waste_ratio")
    contact = _col(rows, "contact_present")
    cap = _col(rows, "cap_press_z")
    fz = _col(rows, "fz")
    if not np.isfinite(fz).any():
        fz = _col(rows, "fz_raw_comp")
    phase = np.array([str(r.get("contact_phase", "")) for r in rows])
    vz = _col(rows, "vz_achieved_tool")
    err = _col(rows, "tool_y_err_mm")
    if not np.isfinite(err).any():
        err = _col(rows, "motion_err_rms_mm")
    if not np.isfinite(err).any():
        err = _col(rows, "track_err_mm")
    motion_rms = _col(rows, "motion_err_rms_mm")
    if not np.isfinite(motion_rms).any():
        motion_rms = _col(rows, "track_err_mm")
    d_star = _col(rows, "d_star_m")

    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    info.append(("phase filter", phase_used))
    _vpc_midrange_checks(rows, results)

    if np.isfinite(waste).any():
        w = float(np.nanmedian(waste[np.isfinite(waste)]))
    else:
        rail_c = _col(rows, "rail_contrib_m_s")
        arm_c = _col(rows, "arm_contrib_m_s")
        net = np.abs(rail_c + arm_c)
        tot = np.abs(rail_c) + np.abs(arm_c)
        wr = tot / np.maximum(net, 1e-9)
        wr = wr[np.isfinite(wr) & (net > 1e-4)]
        w = float(np.nanmedian(wr)) if wr.size else float("nan")
    results.append(
        (
            "waste ratio < 1.15",
            bool(np.isfinite(w) and w < GATES["waste_ratio"]),
            f"{w:.3f}",
        )
    )

    rmin, rmax = float(np.nanmin(rail)), float(np.nanmax(rail))
    rail_ok = rmin >= GATES["rail_min_m"] - 1e-3 and rmax <= GATES["rail_max_m"] + 1e-3
    tcp_ptp = float(np.nanmax(pose_y) - np.nanmin(pose_y)) if np.isfinite(pose_y).any() else float("nan")
    d_abs = float(np.nanmedian(np.abs(d_star))) if np.isfinite(d_star).any() else 0.0
    rail_ptp = rmax - rmin
    span_ok = (not np.isfinite(tcp_ptp)) or rail_ptp <= tcp_ptp + 2.0 * d_abs + 0.02
    results.append(
        (
            f"rail in [{GATES['rail_min_m']:.3f}, {GATES['rail_max_m']:.2f}] "
            "and stroke ≤ TCP+2|d*|",
            rail_ok and span_ok,
            f"rail [{rmin:.3f}, {rmax:.3f}] ptp={rail_ptp:.3f} tcp={tcp_ptp:.3f}",
        )
    )

    j4_m = np.degrees(np.minimum(np.abs(j4 - (-2.356)), np.abs(2.356 - j4)))
    j7_m = np.degrees(np.minimum(np.abs(j7 - (-6.28)), np.abs(6.28 - j7)))
    j4_min = float(np.nanmin(j4_m)) if np.isfinite(j4_m).any() else float("nan")
    j7_min = float(np.nanmin(j7_m)) if np.isfinite(j7_m).any() else float("nan")
    results.append(
        (
            "J4 and J7 margin > 10°",
            j4_min > GATES["j4_j7_margin_deg"] and j7_min > GATES["j4_j7_margin_deg"],
            f"J4 min {j4_min:.1f}°  J7 min {j7_min:.1f}°",
        )
    )

    # Loop period is a first-class metric: the commanded trajectory is
    # consumed by rm_movej_canfd at a fixed cadence, so an irregular producer
    # shows up as motion roughness no joint-space metric can see.
    dt_wall = np.diff(t)
    dt_wall = dt_wall[np.isfinite(dt_wall) & (dt_wall > 0.0)]
    dt_step = float(np.median(dt_wall)) if dt_wall.size else 0.005
    if dt_wall.size:
        on_time = float(
            np.mean(
                (dt_wall > 0.9 * GATES["dt_nominal_s"])
                & (dt_wall < 1.1 * GATES["dt_nominal_s"])
            )
        )
        results.append(
            (
                "loop period on-time > 80%",
                on_time > GATES["dt_on_time_frac"],
                f"{100.0 * on_time:.1f}% within ±10% of "
                f"{1000.0 * GATES['dt_nominal_s']:.1f} ms; "
                f"med {1000.0 * dt_step:.2f} ms "
                f"p95 {1000.0 * np.percentile(dt_wall, 95):.2f} ms "
                f"-> {1.0 / max(np.mean(dt_wall), 1e-9):.0f} Hz effective",
            )
        )

    acc_ok = True
    acc_max = 0.0
    rev_worst = 0.0
    jerk_worst = 0.0
    sat_worst = 0.0
    a_box = 3.0  # qpik.hard_limits.a_max_arm_rad_s2
    span_s = float(t[-1] - t[0]) if t.size > 1 else 0.0
    for i in range(1, 8):
        qi = _col(rows, f"q_cmd_{i}")
        if not np.isfinite(qi).any():
            qi = _col(rows, f"q_meas_{i}")
        # Differentiate on a UNIFORM step, never on wall time.  Dividing by a
        # jittering dt injects the scheduler's 21% period noise into the
        # second difference: on run 230940 that inflated the reversal rate
        # from 6.5-14.3/s to 29-48/s and the jerk RMS from ~110 to ~370, and
        # sent three rounds of tuning after a metric artefact.  The consumer
        # replays these samples at a fixed cadence, so the uniform-step
        # derivative is also the physically meaningful one.
        vi = np.diff(qi) / dt_step
        ai = np.diff(vi) / dt_step
        amax = float(np.nanmax(np.abs(ai))) if ai.size else 0.0
        acc_max = max(acc_max, amax)
        acc_ok = acc_ok and amax < GATES["arm_acc_max"]
        af = ai[np.isfinite(ai)]
        if af.size > 2 and span_s > 0.0:
            # Sign reversals of commanded acceleration: the direct signature
            # of QP / secondary-task chatter (the reference itself is smooth).
            big = af[np.abs(af) > 0.5]
            if big.size > 1:
                flips = int(np.count_nonzero(np.sign(big[1:]) != np.sign(big[:-1])))
                rev_worst = max(rev_worst, flips / span_s)
            sat_worst = max(sat_worst, float(np.mean(np.abs(af) > 0.97 * a_box)))
            ji = np.diff(af) / dt_step
            jerk_worst = max(jerk_worst, float(np.sqrt(np.mean(ji * ji))))
    results.append(("arm |a| max < 8 rad/s²", acc_ok, f"{acc_max:.2f} rad/s²"))
    ripple_p999 = 0.0
    ripple_max = 0.0
    for i in range(1, 8):
        qi = _col(rows, f"q_cmd_{i}")
        if not np.isfinite(qi).any():
            continue
        dqi = np.diff(qi)
        med = float(np.nanmedian(np.abs(dqi)))
        if med < 1.0e-6:
            continue
        moving = np.abs(dqi) > 0.5 * med
        if int(np.count_nonzero(moving)) < 8:
            continue
        rip = np.abs(np.diff(np.abs(dqi[moving]))) / med
        if rip.size:
            ripple_p999 = max(ripple_p999, float(np.nanpercentile(rip, 99.9)))
            ripple_max = max(ripple_max, float(np.nanmax(rip)))
    results.append(
        (
            "command-step ripple p99.9 < 0.50 and max < 1.00",
            ripple_p999 < GATES["step_ripple_p999"]
            and ripple_max < GATES["step_ripple_max"],
            f"p99.9 {ripple_p999:.2f}  max {ripple_max:.2f}",
        )
    )
    hf_worst = 0.0
    peak_hz = float("nan")
    for i in range(1, 8):
        qi = _col(rows, f"q_cmd_{i}")
        if not np.isfinite(qi).any():
            continue
        spec = _cmd_accel_spectrum(qi, 1.0 / dt_step)
        if spec is None:
            continue
        pk, hf = spec
        if hf >= hf_worst:
            hf_worst = hf
            peak_hz = pk
    if math.isfinite(peak_hz):
        results.append(
            (
                "q_cmd accel 40-80 Hz power < 0.15",
                hf_worst < GATES["q_cmd_accel_hf_frac"],
                f"peak {peak_hz:.1f} Hz  40-80 frac {hf_worst:.2f}",
            )
        )
    slack = _col(rows, "deadline_slack_s")
    if np.isfinite(slack).any():
        pos_frac = float(np.mean(slack[np.isfinite(slack)] > 0.0))
        results.append(
            (
                "deadline slack > 0 on ≥99% of ticks",
                pos_frac >= GATES["deadline_slack_pos_frac"],
                f"{100.0 * pos_frac:.1f}% positive  "
                f"med {1000.0 * np.nanmedian(slack):.2f} ms "
                f"p5 {1000.0 * np.nanpercentile(slack, 5):.2f} ms",
            )
        )
        current_ms = 1000.0 * GATES["dt_nominal_s"]
        if pos_frac < GATES["deadline_slack_pos_frac"]:
            up = raise_period_ms(current_ms)
            info.append(
                (
                    "period ladder",
                    f"slack missed; raise dt_ms to {up:.1f} "
                    f"(now {current_ms:.1f})",
                )
            )
        else:
            info.append(
                (
                    "period ladder",
                    f"slack passed at dt_ms={current_ms:.1f}",
                )
            )
    lag = _col(rows, "qpik_accepted_reference_lag_s")
    if not np.isfinite(lag).any():
        lag = _col(rows, "accepted_reference_lag_s")
    if np.isfinite(lag).any():
        p95_lag = float(np.nanpercentile(lag[np.isfinite(lag)], 95))
        results.append(
            (
                "accepted reference lag p95 < 0.1 s",
                p95_lag < GATES["accepted_reference_lag_p95_s"],
                f"p95 {p95_lag:.3f} s  max {float(np.nanmax(lag)):.3f} s",
            )
        )
    for name in ("rt_fifo_ok", "cpu_pinned", "cstate_ok"):
        col = _col(rows, name)
        if col.size and np.isfinite(col).any():
            frac = float(np.mean(col[np.isfinite(col)] > 0.5))
            info.append((name, f"{100.0 * frac:.0f}% of ticks"))
    results.append(
        (
            "accel sign reversals < 20/s and jerk RMS < 200 (uniform step)",
            rev_worst < GATES["accel_reversals_per_s"]
            and jerk_worst < GATES["jerk_rms"],
            f"worst {rev_worst:.1f}/s  jerk_rms {jerk_worst:.0f} rad/s³",
        )
    )
    results.append(
        (
            "accel box saturation < 5%",
            sat_worst < GATES["accel_saturation_frac"],
            f"worst {100.0 * sat_worst:.1f}% of ticks at |a|>{0.97 * a_box:.1f}",
        )
    )

    j6 = _col(rows, "q_meas_6")
    if not np.isfinite(j6).any():
        j6 = _col(rows, "q_cmd_6")
    j6_deg = np.degrees(j6)
    j6_open = (
        float(np.nanmean(np.abs(j6_deg) < 5.0)) if np.isfinite(j6_deg).any() else float("nan")
    )
    results.append(
        (
            "|J6| < 5° (wrist singularity) frac < 5%",
            bool(np.isfinite(j6_open) and j6_open < GATES["j6_open_frac"]),
            f"{100.0 * j6_open:.1f}%  min |J6| {np.nanmin(np.abs(j6_deg)):.1f}°",
        )
    )
    j4_deg = np.degrees(j4)
    j4_near = (
        float(np.nanmean(np.abs(GATES["j4_limit_deg"] - np.abs(j4_deg)) < 5.0))
        if np.isfinite(j4_deg).any()
        else float("nan")
    )
    results.append(
        (
            "J4 within 5° of limit frac < 5%",
            bool(np.isfinite(j4_near) and j4_near < GATES["j4_near_limit_frac"]),
            f"{100.0 * j4_near:.1f}%",
        )
    )
    j2 = _col(rows, "q_meas_2")
    if not np.isfinite(j2).any():
        j2 = _col(rows, "q_cmd_2")
    j2_deg = np.degrees(j2)
    j2_near = (
        float(np.nanmean(np.abs(GATES["j2_limit_deg"] - np.abs(j2_deg)) < 5.0))
        if np.isfinite(j2_deg).any()
        else float("nan")
    )
    results.append(
        (
            "J2 within 5° of limit frac < 5%",
            bool(np.isfinite(j2_near) and j2_near < GATES["j2_near_limit_frac"]),
            f"{100.0 * j2_near:.1f}%",
        )
    )
    inner_ms = _col(rows, "tick_inner_ms")
    if np.isfinite(inner_ms).any():
        inner_max = float(np.nanmax(inner_ms))
        results.append(
            (
                "tick_inner max < 20 ms (no plan_stroke hitch)",
                inner_max < GATES["tick_inner_max_ms"],
                f"{inner_max:.1f} ms",
            )
        )

    if np.isfinite(contact).any():
        loss = float(np.nanmean(contact < 0.5))
    else:
        loss = float("nan")
    results.append(
        (
            "contact-loss frac < 2%",
            bool(np.isfinite(loss) and loss < GATES["contact_loss_frac"]),
            f"{100.0 * loss:.2f}%",
        )
    )

    fz_f = fz[np.isfinite(fz)]
    p99 = float(np.nanpercentile(np.abs(fz_f), 99)) if fz_f.size else float("nan")
    results.append(
        ("|fz| p99 < 4 N", bool(np.isfinite(p99) and p99 < GATES["fz_p99_n"]), f"{p99:.2f} N")
    )

    # Air descent speed and cap_press==0 were phase-2 force gates.  The
    # 55e261d force stack has no fixed air seek and deliberately lets the
    # barrier close press, so both are reported but not judged.
    air = phase == "air"
    if not air.any() and np.isfinite(contact).any():
        air = contact < 0.5
    air_vz = vz[air & np.isfinite(vz)]
    descent = float(np.nanmedian(air_vz)) if air_vz.size else float("nan")
    info.append(
        (
            "air descent (median)",
            f"{1000.0 * descent:.1f} mm/s" if np.isfinite(descent) else "n/a",
        )
    )

    in_c = contact >= 0.5 if np.isfinite(contact).any() else np.ones(len(rows), dtype=bool)
    cap_c = cap[in_c & np.isfinite(cap)]
    zero_frac = float(np.mean(cap_c <= 1e-9)) if cap_c.size else float("nan")
    info.append(
        (
            "cap_press==0 during contact",
            f"{100.0 * zero_frac:.2f}%" if np.isfinite(zero_frac) else "n/a",
        )
    )

    # Rail-at-wall ≠ workspace-sat.  Share dropping only means the carriage
    # stopped; the arm must still hold XY if 7DOF IK exists at locked q0.
    _rail_handoff_checks(rows, rail, err, results, info)
    _posture_followup(rows, info)
    sat = _col(rows, "rail_sat")
    if np.isfinite(sat).any():
        info.append(
            (
                "rail_sat",
                f"{100.0 * float(np.nanmean(sat > 0.5)):.1f}% of scan ticks",
            )
        )

    band = GATES["rail_limit_band_m"]
    at_limit = np.isfinite(rail) & (
        (rail < GATES["rail_min_m"] + band) | (rail > GATES["rail_max_m"] - band)
    )
    esc = _col(rows, "rail_escape_active")
    if np.isfinite(esc).any() and int(at_limit.sum()) >= 50:
        esc_at_limit = float(np.nanmean(esc[at_limit] > 0.5))
        results.append(
            (
                "sigma-escape off inside the rail limit band",
                esc_at_limit <= 1.0e-9,
                f"{100.0 * esc_at_limit:.1f}% of ticks",
            )
        )

    _rail_servo_checks(path, results, info)
    _idle_hold_checks(rows, results, info)
    _posture_debt_check(rows, results, info)
    _rail_task_dropout_check(rows, results, info)
    _d_star_step_check(d_star, dt_wall, results, info)
    _joint_exec_ratio_check(rows, t, results, info)
    _tick_profile(rows, info)

    div = _col(rows, "rail_cmd_meas_err_m")
    if not np.isfinite(div).any():
        div = _col(rows, "q_cmd_0") - _col(rows, "q_meas_0")
    div_abs = np.abs(div)
    div_ok = div_abs[np.isfinite(div_abs)]
    if div_ok.size:
        bind = float(np.mean(div_ok >= GATES["rail_resync_err_m"]))
        p95_div = float(np.percentile(div_ok, 95))
        results.append(
            (
                "rail |q_cmd-q_meas| lead clamp duty < 0.5%",
                bind < GATES["rail_resync_bind_frac"],
                f"bind {100.0 * bind:.1f}%  p95 {1000.0 * p95_div:.1f} mm",
            )
        )

    dt_med = float(np.median(dt_wall)) if dt_wall.size else GATES["dt_nominal_s"]
    for axis, d_name, m_name, v_name in (
        ("X", "pose_d_x", "pose_meas_x", "vel_ff_vx"),
        ("Y", "pose_d_y", "pose_meas_y", "vel_ff_vy"),
        ("Z", "pose_d_z", "pose_meas_z", "vel_ff_vz"),
    ):
        ref = _col(rows, d_name)
        meas = _col(rows, m_name)
        if not (np.isfinite(ref).any() and np.isfinite(meas).any()):
            continue
        tau, resid = best_axis_time_shift(ref, meas, dt_med)
        axis_err = (ref - meas) * 1000.0
        corr = err_vel_correlation(axis_err, _col(rows, v_name))
        info.append(
            (
                f"axis {axis} phase",
                (
                    f"tau {1000.0 * tau:.0f} ms  "
                    if np.isfinite(tau)
                    else "tau n/a  "
                )
                + (
                    f"resid {1000.0 * resid:.2f} mm  "
                    if np.isfinite(resid)
                    else "resid n/a  "
                )
                + (
                    f"corr(e,v) {corr:.3f}"
                    if np.isfinite(corr)
                    else "corr(e,v) n/a"
                ),
            )
        )

    e95 = (
        float(np.nanpercentile(np.abs(err[np.isfinite(err)]), 95))
        if np.isfinite(err).any()
        else float("nan")
    )
    results.append(
        (
            "tool_y_err p95 < 1 mm",
            bool(np.isfinite(e95) and e95 < GATES["track_err_p95_mm"]),
            f"{e95:.2f} mm",
        )
    )
    rms95 = (
        float(np.nanpercentile(np.abs(motion_rms[np.isfinite(motion_rms)]), 95))
        if np.isfinite(motion_rms).any()
        else float("nan")
    )
    if np.isfinite(rms95):
        info.append(("motion_err_rms p95 (force-Z included)", f"{rms95:.2f} mm"))

    failed = 0
    print(f"rows: {len(rows)}  phase={phase_used}  file: {path}")
    for name, detail in info:
        print(f"  [INFO] {name}: {detail}")
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{mark}] {name}: {detail}")
    return 1 if failed else 0


def _load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, nargs="?")
    ap.add_argument(
        "--ab-clamp",
        nargs=3,
        metavar=("TRUE1", "FALSE", "TRUE2"),
        help="Score post-QP clamp A/B from three full CSVs (True→False→True).",
    )
    args = ap.parse_args()
    if args.ab_clamp is not None:
        report = evaluate_post_qp_ab(
            _load_csv_rows(Path(args.ab_clamp[0])),
            _load_csv_rows(Path(args.ab_clamp[1])),
            _load_csv_rows(Path(args.ab_clamp[2])),
        )
        def _clean(value):
            if isinstance(value, dict):
                return {k: _clean(v) for k, v in value.items()}
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        print(json.dumps(_clean(report), indent=2, ensure_ascii=True))
        return 0 if report.get("verdict") in {"amplifier", "unchanged"} else 1
    if args.csv is None:
        ap.error("csv is required unless --ab-clamp is given")
    return analyze(args.csv)


if __name__ == "__main__":
    raise SystemExit(main())
```

### `apps/joint_admittance_8dof/d_gamepad_vcmd.py`

```python
#!/usr/bin/env python3
"""Send Xbox stick velocity into the 8-DOF QPIK inner loop (no force / no scan).

Window A must already be running. This submits a ``gamepad_vcmd`` task; A reads
the pad and feeds ``v_cmd`` to ``JointIkController.update``. All existing QP
limits, CBF, rail pin/escape, and nullspace stay on.

  source env.sh
  # terminal A
  python apps/joint_admittance_8dof/run_joint_admittance.py \\
      --config configs/joint_admittance_8dof.yaml -v
  # terminal C
  python apps/joint_admittance_8dof/d_gamepad_vcmd.py --config configs/joint_admittance_8dof.yaml

  Left stick: world XY (left = +Y, up = +X)
  LB / LT:    world +Z / −Z
  Right stick + RB/RT: TCP-frame rotation
"""

from __future__ import annotations

import argparse
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.admittance_common.phase_ipc import (
    PhaseCommandClient,
    PhaseStatus,
    SinToolYTaskParams,
)
from rm75_control.control.admittance_common.state_bus import RobotStateBus
from rm75_control.control.admittance_common.state_relay import (
    RelayStateBus,
    parse_state_relay_config,
    relay_shm_has_publisher,
)
from rm75_control.control.joint_admittance_8dof.api import compute_move_plan
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.gamepad_vcmd_program import (
    build_gamepad_vcmd_program,
    close_built_pad,
)
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    deg2rad,
    full_q_from_arm,
)
from rm75_control.control.joint_admittance_8dof.sin_tool_y_program import (
    execute_sin_tool_y_program,
    resolve_scan_target_at_d,
)
from rm75_control.control.joint_admittance_8dof.teleop.gamepad_twist import MAPPING_HELP
from rm75_control.control.joint_admittance_8dof.teleop.xbox_pad import XboxPad
from rm75_control.core.session import RobotSession
from rm75_control.force.compensation.tool_pose import maybe_sync_kin_tcp_from_config
from rm75_control.kinematics.srs_ik import psi_from_q


class _AttachSession:
    config: dict
    ip: str
    robot: object = None

    def __init__(self, config: dict, ip: str) -> None:
        self.config = config
        self.ip = ip

    def move_joints(self, *args, **kwargs) -> None:
        raise RuntimeError("move_j is unavailable in attach mode (window A owns TCP)")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _poll_attach_status(phase_client: PhaseCommandClient, cmd_seq: int) -> PhaseStatus:
    def _on_sig(_signum, _frame) -> None:
        try:
            phase_client.stop()
        except Exception:
            pass
        raise KeyboardInterrupt

    prev_int = signal.signal(signal.SIGINT, _on_sig)
    prev_term = signal.signal(signal.SIGTERM, _on_sig)
    try:
        while True:
            st = phase_client.read_status()
            if st is not None and st["status_seq"] == cmd_seq:
                status = st["status"]
                if status in (PhaseStatus.DONE, PhaseStatus.ERROR, PhaseStatus.STOPPED):
                    return status
            time.sleep(0.05)
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, default=Path("configs/joint_admittance_8dof.yaml"))
    ap.add_argument("--slot", type=str, default="d")
    ap.add_argument(
        "--goto-d",
        action="store_true",
        help="MoveJ to taught slot D before teleop (default: start from the live pose).",
    )
    ap.add_argument("--move-duration", type=float, default=None)
    ap.add_argument("--move-duration-margin", type=float, default=0.80)
    ap.add_argument("--move-duration-min", type=float, default=2.5)
    ap.add_argument("--move-duration-max", type=float, default=20.0)
    ap.add_argument(
        "--move-kp",
        type=float,
        default=None,
        help="Override cartesian_track.k_task_lin (default: yaml).",
    )
    ap.add_argument("--move-mode", choices=("cartesian", "joint"), default="joint")
    ap.add_argument("--trans-m-s", type=float, default=0.10, help="Full-stick world translation (m/s).")
    ap.add_argument("--rot-rad-s", type=float, default=0.60, help="Full-stick TCP rotation (rad/s).")
    ap.add_argument("--deadzone", type=float, default=0.18)
    ap.add_argument(
        "--trigger-deadzone",
        type=float,
        default=0.08,
        help="Rest deadzone on LT/RT after mapping to [0, 1].",
    )
    ap.add_argument(
        "--trans-a-max",
        type=float,
        default=0.8,
        help="Stick translation slew limit (m/s^2).",
    )
    ap.add_argument(
        "--rot-a-max",
        type=float,
        default=4.0,
        help="Stick rotation slew limit (rad/s^2).",
    )
    ap.add_argument(
        "--hold-v-max",
        type=float,
        default=0.03,
        help="Idle hold linear cap (m/s).",
    )
    ap.add_argument(
        "--hold-w-max",
        type=float,
        default=0.20,
        help="Idle hold angular cap (rad/s).",
    )
    ap.add_argument(
        "--device-index",
        type=int,
        default=-1,
        help="Force pygame joystick index. Default −1 = USB/wired over Bluetooth.",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Teleop wall time (s). 0 = until Ctrl+C / window-C stop.",
    )
    ap.add_argument("--cartesian-max-lin-vel", type=float, default=None)
    ap.add_argument("--log-csv", type=str, default=None)
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--print-axes",
        action="store_true",
        help="Dump raw pad axes and exit (no robot).",
    )
    ap.add_argument(
        "--no-attach-state",
        action="store_true",
        help="Own robot TCP/UDP locally (debug only; do not run with window A).",
    )
    args = ap.parse_args()

    if args.print_axes:
        pad = XboxPad(
            device_index=int(args.device_index),
            auto_select=int(args.device_index) < 0,
            allow_missing=True,
        )
        print(MAPPING_HELP, flush=True)
        print(f"connected={pad.connected} {getattr(pad, 'describe', lambda: '')()}", flush=True)
        try:
            t_end = time.monotonic() + 8.0
            while time.monotonic() < t_end:
                state = pad.read()
                print(
                    f"axes={np.round(state.axes, 3).tolist()} "
                    f"buttons={state.buttons.astype(int).tolist()}",
                    flush=True,
                )
                time.sleep(0.2)
        finally:
            pad.close()
        return 0

    raw = load_yaml(args.config)
    startup = raw.get("startup", {})
    relay_cfg = parse_state_relay_config(raw)
    kin = RobotKinematics()
    inner_cfg = build_joint_ik_config(raw)

    ts = time.strftime("%Y%m%d_%H%M%S")
    if not args.log_csv:
        log_dir = Path(__file__).resolve().parents[1] / "logs" / "gamepad_vcmd"
        log_dir.mkdir(parents=True, exist_ok=True)
        args.log_csv = str(log_dir / f"run_{ts}.csv")
    if not getattr(args, "rail_log_csv", None):
        rail_dir = Path(__file__).resolve().parents[1] / "logs" / "rail_servo"
        rail_dir.mkdir(parents=True, exist_ok=True)
        args.rail_log_csv = str(rail_dir / f"rail_{ts}.csv")
    print(f"gamepad WBC log: {args.log_csv}", flush=True)
    print(f"gamepad rail log: {args.rail_log_csv}", flush=True)

    if args.dry_run:
        params = SinToolYTaskParams(
            config_path=str(args.config.resolve()),
            slot=str(args.slot),
            task_kind="gamepad_vcmd",
            scan_duration=float(args.duration),
            gamepad_trans_m_s=float(args.trans_m_s),
            gamepad_rot_rad_s=float(args.rot_rad_s),
            gamepad_deadzone=float(args.deadzone),
            gamepad_device_index=int(args.device_index),
            gamepad_trigger_deadzone=float(args.trigger_deadzone),
            gamepad_trans_a_max_m_s2=float(args.trans_a_max),
            gamepad_rot_a_max_rad_s2=float(args.rot_a_max),
            gamepad_hold_v_max_m_s=float(args.hold_v_max),
            gamepad_hold_w_max_rad_s=float(args.hold_w_max),
            q0_rad=[0.0] * 8,
            q_target_rad=[0.0] * 8,
            tcp_offset_pose=[0.0] * 6,
        )
        from rm75_control.control.joint_admittance_8dof.teleop.xbox_pad import FakePad

        built = build_gamepad_vcmd_program(params, raw=raw, pad=FakePad())
        close_built_pad(built)
        print("dry-run: gamepad_vcmd program built OK", flush=True)
        return 0

    robot_cfg = raw.get("robot", {})
    max_lin = float(args.cartesian_max_lin_vel) if args.cartesian_max_lin_vel is not None else 0.4
    sigma_ref = float(inner_cfg.qp.sr_damping.sigma_ref)
    local_bus: RobotStateBus | None = None
    state_bus = None
    phase_client: PhaseCommandClient | None = None
    attach_mode = not args.no_attach_state
    shm_name = str(relay_cfg.name or "rm75_state")

    if attach_mode:
        attach_bus = RelayStateBus(shm_name)
        try:
            attach_bus.wait_first_pose(timeout_s=30.0)
        except TimeoutError as exc:
            raise RuntimeError(
                f"no live relay on shm {shm_name!r} — start window A first"
            ) from exc
        state_bus = attach_bus
        phase_client = PhaseCommandClient()
        phase_client.wait_for_hub(timeout_s=30.0)
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
        snap0 = state_bus.read()
        if snap0.q_deg is None:
            raise RuntimeError("no joint feedback")
        rail_start_m = float(getattr(state_bus, "last_rail_m", 0.0) or 0.0)
        q0_rad = full_q_from_arm(deg2rad(snap0.q_deg), rail_start_m)
        q_target_rad = q0_rad.copy()
        pose_d = kin.fk_pose(q0_rad)
        plan_duration_s = 0.0
        plan_move_mode = str(args.move_mode)
        plan_gov = 0.0
        psi_tgt = None

        if args.goto_d:
            scan_target = resolve_scan_target_at_d(
                args.slot,
                kin,
                euler_order=inner_cfg.euler_order,
                rail_m=rail_start_m,
                q_seed_rad=q0_rad,
                require_path=(str(args.move_mode) != "joint"),
            )
            pose_d = scan_target.pose_d
            q_target_rad = np.asarray(scan_target.q_target_rad, dtype=float)
            psi_tgt = float(psi_from_q(q_target_rad))
            plan = compute_move_plan(
                kin,
                q0_rad,
                q_target_rad,
                pose_d,
                v_scale=inner_cfg.v_scale,
                duration_s=args.move_duration,
                move_mode=str(args.move_mode),
                peak_joint_v_frac=float(args.move_duration_margin),
                max_lin_vel_m_s=max_lin,
                duration_min_s=float(args.move_duration_min),
                duration_max_s=float(args.move_duration_max),
                approach_dz_m=0.22,
                sigma_ref=sigma_ref,
                euler_order=inner_cfg.euler_order,
            )
            plan_duration_s = float(plan.duration_s)
            plan_move_mode = str(plan.move_mode)
            plan_gov = float(plan.gov_joint_max_deg)

        task_params = SinToolYTaskParams(
            config_path=str(args.config.resolve()),
            slot=str(args.slot),
            task_kind="gamepad_vcmd",
            move_kp=args.move_kp,
            scan_duration=float(args.duration),
            log_csv=args.log_csv,
            rail_log_csv=getattr(args, "rail_log_csv", None),
            cartesian_max_lin_vel=args.cartesian_max_lin_vel,
            q0_rad=np.asarray(q0_rad, dtype=float).reshape(-1).tolist(),
            q_target_rad=np.asarray(q_target_rad, dtype=float).reshape(-1).tolist(),
            pose_d=np.asarray(pose_d, dtype=float).reshape(6).tolist(),
            plan_duration_s=plan_duration_s,
            plan_move_mode=plan_move_mode,
            plan_gov_joint_max_deg=plan_gov,
            psi_tgt=psi_tgt,
            tcp_offset_pose=(
                np.asarray(kin.tcp_offset_pose, dtype=float).reshape(6).tolist()
                if kin.tcp_offset_pose is not None
                else []
            ),
            gamepad_trans_m_s=float(args.trans_m_s),
            gamepad_rot_rad_s=float(args.rot_rad_s),
            gamepad_deadzone=float(args.deadzone),
            gamepad_device_index=int(args.device_index),
            gamepad_trigger_deadzone=float(args.trigger_deadzone),
            gamepad_trans_a_max_m_s2=float(args.trans_a_max),
            gamepad_rot_a_max_rad_s2=float(args.rot_a_max),
            gamepad_hold_v_max_m_s=float(args.hold_v_max),
            gamepad_hold_w_max_rad_s=float(args.hold_w_max),
        )

        try:
            if attach_mode:
                assert phase_client is not None
                cmd_seq = phase_client.start(task_params)
                final = _poll_attach_status(phase_client, cmd_seq)
                if final == PhaseStatus.ERROR:
                    st = phase_client.read_status()
                    raise RuntimeError(
                        f"window A task failed: {st['msg'] if st else 'unknown'}"
                    )
            else:
                built = build_gamepad_vcmd_program(task_params, raw=raw)
                try:
                    execute_sin_tool_y_program(
                        sess,
                        state_bus,
                        task_params,
                        raw=raw,
                        built=built,
                        verbose=bool(args.verbose) or bool(startup.get("verbose", False)),
                    )
                finally:
                    close_built_pad(built)
        except KeyboardInterrupt:
            if attach_mode and phase_client is not None:
                phase_client.stop()
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

### `apps/joint_admittance_8dof/d_ellipse_track.py`

```python
#!/usr/bin/env python3
"""Cartesian TRACKING ellipse on the 8-DOF QPIK backend (no force).

Same-frequency tool-XY ellipse through the live TCP (no jump). Window A must
already be running the current dispatcher (restart A once after this lands).

  source env.sh
  # terminal A  (restart if it was started before ellipse_track existed)
  python apps/joint_admittance_8dof/run_joint_admittance.py \\
      --config configs/joint_admittance_8dof.yaml -v
  # terminal C
  python apps/joint_admittance_8dof/d_ellipse_track.py \\
      --config configs/joint_admittance_8dof.yaml

Defaults are conservative: X 4 cm pp, Y 8 cm pp, 3 cm/s, 40 s, from the live
pose.  Add ``--goto-d`` to MoveJ to taught slot D first.
"""

from __future__ import annotations

import argparse
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import yaml

from rm75_control.control.admittance_common.phase_ipc import (
    PhaseCommandClient,
    PhaseStatus,
    SinToolYTaskParams,
)
from rm75_control.control.admittance_common.state_bus import RobotStateBus
from rm75_control.control.admittance_common.state_relay import (
    RelayStateBus,
    parse_state_relay_config,
    relay_shm_has_publisher,
)
from rm75_control.control.joint_admittance_8dof.api import compute_move_plan
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.ellipse_track_program import (
    build_ellipse_track_program,
)
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    deg2rad,
    full_q_from_arm,
)
from rm75_control.control.joint_admittance_8dof.sin_tool_y_program import (
    execute_sin_tool_y_program,
    resolve_scan_target_at_d,
)
from rm75_control.core.session import RobotSession
from rm75_control.force.compensation.tool_pose import maybe_sync_kin_tcp_from_config
from rm75_control.kinematics.srs_ik import psi_from_q


class _AttachSession:
    config: dict
    ip: str
    robot: object = None

    def __init__(self, config: dict, ip: str) -> None:
        self.config = config
        self.ip = ip

    def move_joints(self, *args, **kwargs) -> None:
        raise RuntimeError("move_j is unavailable in attach mode (window A owns TCP)")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _poll_attach_status(phase_client: PhaseCommandClient, cmd_seq: int) -> PhaseStatus:
    def _on_sig(_signum, _frame) -> None:
        try:
            phase_client.stop()
        except Exception:
            pass
        raise KeyboardInterrupt

    prev_int = signal.signal(signal.SIGINT, _on_sig)
    prev_term = signal.signal(signal.SIGTERM, _on_sig)
    try:
        while True:
            st = phase_client.read_status()
            if st is not None and st["status_seq"] == cmd_seq:
                status = st["status"]
                if status in (PhaseStatus.DONE, PhaseStatus.ERROR, PhaseStatus.STOPPED):
                    return status
            time.sleep(0.05)
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, default=Path("configs/joint_admittance_8dof.yaml"))
    ap.add_argument("--slot", type=str, default="d")
    ap.add_argument(
        "--goto-d",
        action="store_true",
        help="MoveJ to taught slot D before the ellipse (default: live pose).",
    )
    ap.add_argument("--move-duration", type=float, default=None)
    ap.add_argument("--move-duration-margin", type=float, default=0.80)
    ap.add_argument("--move-duration-min", type=float, default=2.5)
    ap.add_argument("--move-duration-max", type=float, default=120.0)
    ap.add_argument(
        "--move-kp",
        type=float,
        default=None,
        help="Override cartesian_track.k_task_lin (default: yaml).",
    )
    ap.add_argument("--move-mode", choices=("cartesian", "joint"), default="joint")
    ap.add_argument("--x-pp-cm", type=float, default=10.0, help="Tool-X peak-to-peak (cm).")
    ap.add_argument("--y-pp-cm", type=float, default=30.0, help="Tool-Y peak-to-peak (cm).")
    ap.add_argument("--max-vel-cm-s", type=float, default=4.0)
    ap.add_argument("--period-s", type=float, default=None)
    ap.add_argument("--scan-duration", type=float, default=40.0)
    ap.add_argument("--cartesian-max-lin-vel", type=float, default=None)
    ap.add_argument("--log-csv", type=str, default=None)
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--no-attach-state",
        action="store_true",
        help="Own robot TCP/UDP locally (debug only; do not run with window A).",
    )
    args = ap.parse_args()

    raw = load_yaml(args.config)
    startup = raw.get("startup", {})
    relay_cfg = parse_state_relay_config(raw)
    kin = RobotKinematics()
    inner_cfg = build_joint_ik_config(raw)

    if args.dry_run:
        params = SinToolYTaskParams(
            config_path=str(args.config.resolve()),
            slot=str(args.slot),
            task_kind="ellipse_track",
            x_pp_cm=float(args.x_pp_cm),
            y_pp_cm=float(args.y_pp_cm),
            max_vel_cm_s=float(args.max_vel_cm_s),
            period_s=args.period_s,
            scan_duration=float(args.scan_duration),
            move_kp=args.move_kp,
            q0_rad=[0.0] * 8,
            q_target_rad=[0.0] * 8,
            tcp_offset_pose=[0.0] * 6,
        )
        built = build_ellipse_track_program(params, raw=raw)
        ref = built.reference
        print(
            f"dry-run: ellipse_track OK  "
            f"ax={ref.amplitude_x_m * 100:.1f}cm  ay={ref.amplitude_y_m * 100:.1f}cm  "
            f"T={ref.period_s:.2f}s  phases={[p.label for p in built.phases]}",
            flush=True,
        )
        return 0

    ts = time.strftime("%Y%m%d_%H%M%S")
    if not args.log_csv:
        log_dir = Path(__file__).resolve().parents[1] / "logs" / "ellipse_track"
        log_dir.mkdir(parents=True, exist_ok=True)
        args.log_csv = str(log_dir / f"run_{ts}.csv")
    if not getattr(args, "rail_log_csv", None):
        rail_dir = Path(__file__).resolve().parents[1] / "logs" / "rail_servo"
        rail_dir.mkdir(parents=True, exist_ok=True)
        args.rail_log_csv = str(rail_dir / f"rail_{ts}.csv")
    print(f"ellipse WBC log: {args.log_csv}", flush=True)
    print(f"ellipse rail log: {args.rail_log_csv}", flush=True)

    robot_cfg = raw.get("robot", {})
    max_lin = float(args.cartesian_max_lin_vel) if args.cartesian_max_lin_vel is not None else 0.4
    sigma_ref = float(inner_cfg.qp.sr_damping.sigma_ref)
    local_bus: RobotStateBus | None = None
    state_bus = None
    phase_client: PhaseCommandClient | None = None
    attach_mode = not args.no_attach_state
    shm_name = str(relay_cfg.name or "rm75_state")

    if attach_mode:
        attach_bus = RelayStateBus(shm_name)
        try:
            attach_bus.wait_first_pose(timeout_s=30.0)
        except TimeoutError as exc:
            raise RuntimeError(
                f"no live relay on shm {shm_name!r} — start window A first"
            ) from exc
        state_bus = attach_bus
        phase_client = PhaseCommandClient()
        phase_client.wait_for_hub(timeout_s=30.0)
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

        snap0 = state_bus.read()
        if snap0.q_deg is None:
            raise RuntimeError("no joint feedback")
        rail_start_m = float(getattr(state_bus, "last_rail_m", 0.0) or 0.0)
        q0_rad = full_q_from_arm(deg2rad(snap0.q_deg), rail_start_m)
        q_target_rad = q0_rad.copy()
        pose_d = kin.fk_pose(q0_rad)
        plan_duration_s = 0.0
        plan_move_mode = str(args.move_mode)
        plan_gov = 0.0
        psi_tgt = None

        if args.goto_d:
            scan_target = resolve_scan_target_at_d(
                args.slot,
                kin,
                euler_order=inner_cfg.euler_order,
                rail_m=rail_start_m,
                q_seed_rad=q0_rad,
                require_path=(str(args.move_mode) != "joint"),
            )
            pose_d = scan_target.pose_d
            q_target_rad = np.asarray(scan_target.q_target_rad, dtype=float)
            psi_tgt = float(psi_from_q(q_target_rad))
            plan = compute_move_plan(
                kin,
                q0_rad,
                q_target_rad,
                pose_d,
                v_scale=inner_cfg.v_scale,
                duration_s=args.move_duration,
                move_mode=str(args.move_mode),
                peak_joint_v_frac=float(args.move_duration_margin),
                max_lin_vel_m_s=max_lin,
                duration_min_s=float(args.move_duration_min),
                duration_max_s=float(args.move_duration_max),
                approach_dz_m=0.22,
                sigma_ref=sigma_ref,
                euler_order=inner_cfg.euler_order,
            )
            plan_duration_s = float(plan.duration_s)
            plan_move_mode = str(plan.move_mode)
            plan_gov = float(plan.gov_joint_max_deg)

        task_params = SinToolYTaskParams(
            config_path=str(args.config.resolve()),
            slot=str(args.slot),
            task_kind="ellipse_track",
            enable_force=False,
            move_kp=args.move_kp,
            x_pp_cm=float(args.x_pp_cm),
            y_pp_cm=float(args.y_pp_cm),
            max_vel_cm_s=float(args.max_vel_cm_s),
            period_s=args.period_s,
            scan_duration=float(args.scan_duration),
            log_csv=args.log_csv,
            rail_log_csv=getattr(args, "rail_log_csv", None),
            cartesian_max_lin_vel=args.cartesian_max_lin_vel,
            q0_rad=np.asarray(q0_rad, dtype=float).reshape(-1).tolist(),
            q_target_rad=np.asarray(q_target_rad, dtype=float).reshape(-1).tolist(),
            pose_d=np.asarray(pose_d, dtype=float).reshape(6).tolist(),
            plan_duration_s=plan_duration_s,
            plan_move_mode=plan_move_mode,
            plan_gov_joint_max_deg=plan_gov,
            psi_tgt=psi_tgt,
            tcp_offset_pose=(
                np.asarray(kin.tcp_offset_pose, dtype=float).reshape(6).tolist()
                if kin.tcp_offset_pose is not None
                else []
            ),
        )
        print(
            f"rm75 ellipse: CARTESIAN_TRACK  "
            f"X {args.x_pp_cm:.1f} cm pp  Y {args.y_pp_cm:.1f} cm pp  "
            f"v≤{args.max_vel_cm_s:.1f} cm/s  {args.scan_duration:.0f}s  "
            f"{'goto D then ' if args.goto_d else ''}from live pose",
            flush=True,
        )

        try:
            if attach_mode:
                assert phase_client is not None
                cmd_seq = phase_client.start(task_params)
                final = _poll_attach_status(phase_client, cmd_seq)
                if final == PhaseStatus.ERROR:
                    st = phase_client.read_status()
                    raise RuntimeError(
                        f"window A task failed: {st['msg'] if st else 'unknown'}"
                    )
            else:
                built = build_ellipse_track_program(task_params, raw=raw)
                execute_sin_tool_y_program(
                    sess,
                    state_bus,
                    task_params,
                    raw=raw,
                    built=built,
                    verbose=bool(args.verbose) or bool(startup.get("verbose", False)),
                )
        except KeyboardInterrupt:
            if attach_mode and phase_client is not None:
                phase_client.stop()
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

### `rm75_control/control/joint_admittance_8dof/ellipse_track_program.py`

```python
"""Build a no-force Cartesian TRACKING ellipse program (window A / standalone)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rm75_control.control.admittance_common.phase_ipc import SinToolYTaskParams
from rm75_control.control.joint_admittance_8dof.api import (
    CompileContext,
    GovernorSpec,
    SecondaryPolicy,
    compile_phases,
    phase_cartesian_track,
)
from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
from rm75_control.control.joint_admittance_8dof.loop import JointIkController
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    shared_robot_kinematics,
)
from rm75_control.control.joint_admittance_8dof.reference import EllipseToolXYReference
from rm75_control.control.joint_admittance_8dof.wbc_arm import WbcArm
from rm75_control.force.compensation.tool_pose import maybe_sync_kin_tcp_from_config


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class BuiltEllipseTrackProgram:
    phases: list
    compiled: list
    inner: JointIkController
    kin: RobotKinematics
    force_observer: Any
    reference: EllipseToolXYReference


def build_ellipse_track_program(
    params: SinToolYTaskParams,
    *,
    raw: dict | None = None,
) -> BuiltEllipseTrackProgram:
    raw = raw if raw is not None else load_yaml(params.config_path)
    kin = shared_robot_kinematics()
    maybe_sync_kin_tcp_from_config(
        kin,
        raw,
        attach_mode=True,
        tcp_offset_pose=params.tcp_offset_pose if params.tcp_offset_pose else None,
    )
    inner_cfg = build_joint_ik_config(raw)
    inner = JointIkController(kin, inner_cfg)
    ctx = CompileContext(
        kin=kin,
        inner=inner,
        euler_order=inner_cfg.euler_order,
        control_frame=inner_cfg.control_frame,
        v_scale=inner_cfg.v_scale,
    )

    specs = []
    q_target = np.asarray(params.q_target_rad, dtype=float).reshape(-1)
    q0 = np.asarray(params.q0_rad, dtype=float).reshape(-1)
    if float(params.plan_duration_s) > 1.0e-9 and q_target.size == q0.size and q0.size > 0:
        move_mode = str(params.plan_move_mode)
        if move_mode == "joint":
            specs.append(
                WbcArm.make_movej_phase(
                    kin,
                    q0,
                    q_target,
                    duration_s=float(params.plan_duration_s),
                    label=f"movej->{params.slot}",
                    move_kp=SinToolYTaskParams.optional_move_kp(params.move_kp),
                    gov_joint_max_deg=float(params.plan_gov_joint_max_deg),
                )
            )
        else:
            pose_d = np.asarray(kin.fk_pose(q_target), dtype=float).reshape(6)
            specs.append(
                WbcArm.make_movel_phase(
                    kin,
                    q0,
                    pose_d,
                    q_target,
                    duration_s=float(params.plan_duration_s),
                    label=f"movel->{params.slot}",
                    move_kp=SinToolYTaskParams.optional_move_kp(params.move_kp),
                    max_lin_vel_m_s=(
                        float(params.cartesian_max_lin_vel)
                        if params.cartesian_max_lin_vel is not None
                        else 0.4
                    ),
                    gov_joint_max_deg=float(params.plan_gov_joint_max_deg),
                    euler_order=inner_cfg.euler_order,
                )
            )

    ax_m = float(getattr(params, "x_pp_cm", 0.0) or 0.0) * 0.01 / 2.0
    ay_m = float(params.y_pp_cm) * 0.01 / 2.0
    max_vel_m_s = float(params.max_vel_cm_s) * 0.01
    track_ref = EllipseToolXYReference(
        ax_m,
        ay_m,
        period_s=params.period_s,
        max_vel_m_s=None if params.period_s is not None else max_vel_m_s,
        soft_start=True,
        ramp_s=2.0,
        euler_order=inner_cfg.euler_order,
    )
    track_lin = (
        float(params.cartesian_max_lin_vel)
        if params.cartesian_max_lin_vel is not None
        else max(0.15, 3.0 * max_vel_m_s)
    )
    duration = float(params.scan_duration)
    specs.append(
        phase_cartesian_track(
            track_ref,
            label="ellipse_track",
            duration_s=None if duration <= 0.0 else duration,
            move_kp=SinToolYTaskParams.optional_move_kp(params.move_kp),
            max_lin_vel_m_s=track_lin,
            secondary=SecondaryPolicy(preset="track", qdot_ff="off"),
            governor=GovernorSpec(err_ok_mm=10.0, err_max_mm=40.0),
        )
    )

    compiled = compile_phases(specs, ctx)
    return BuiltEllipseTrackProgram(
        phases=[item.phase for item in compiled],
        compiled=compiled,
        inner=inner,
        kin=kin,
        force_observer=None,
        reference=track_ref,
    )
```

## 7.2 CSV 身份核验（表头 + 首 3 行 + 末 2 行）

完整字节见 `MD/todo_controller_logs/`。下面用于确认审查者打开的是同一份文件。

### `run_20260819_204333.csv`  （36711 lines including header, 105443034 bytes）

```csv
t_wall_s,phase,controller_mode,t_ref_s,q_cmd_0,q_cmd_1,q_cmd_2,q_cmd_3,q_cmd_4,q_cmd_5,q_cmd_6,q_cmd_7,q_meas_0,q_meas_1,q_meas_2,q_meas_3,q_meas_4,q_meas_5,q_meas_6,q_meas_7,pose_x,pose_y,pose_z,pose_rx,pose_ry,pose_rz,twist_vx,twist_vy,twist_vz,twist_wx,twist_wy,twist_wz,twist_requested_vx,twist_requested_vy,twist_requested_vz,twist_requested_wx,twist_requested_wy,twist_requested_wz,twist_achieved_vx,twist_achieved_vy,twist_achieved_vz,twist_achieved_wx,twist_achieved_wy,twist_achieved_wz,track_err_mm,follow_err_deg,slack_norm,n_cbf,vel_clamped,acc_clamped,pos_clamped,fx,fy,fz,instability_idx,instability_idx_raw,instability_idx_active,damping_z_eff,damping_ke_z,damping_dimeas_z,v_force_z,ke_est,f_des_z_eff,v_r_z,force_reference_scale_n,force_reference_drive,force_reference_gate_scale,force_reference_accel_m_s2,force_reference_reversal_reset,force_reference_fast_clear,force_fast_z,retract_guard_armed,retract_fast_hold,retract_fast_stop_count,retract_fast_rearm_count,force_task_latched,physical_contact_state,physical_contact_acquire_event,physical_contact_loss_event,physical_contact_reacquire_event,physical_contact_low_timer_s,physical_contact_high_timer_s,mass_z_eff,takeover,dt_actual_s,deadline_slack_s,sensor_age_s,feedback_age_s,feedback_fresh_tick,fx_raw_comp,fy_raw_comp,fz_raw_comp,vz_achieved_tool,contact_present,force_pred_z,force_dot_z,cap_press_z,cap_retract_z,ke_update_gated,ke_dx_m,ke_df_n,ke_update_count,governor_scale,governor_scale_raw,sigma_min,qdot_norm,qdot_max_frac_vmax,qdot_ff_norm,tcp_jump_mm,rail_target_sent_m,rail_meas_m,rail_cmd_meas_err_m,rail_vel_pin,plan_drives_rail,rail_qdot_ff,pose_d_x,pose_d_y,pose_d_z,pose_d_rx,pose_d_ry,pose_d_rz,pose_meas_x,pose_meas_y,pose_meas_z,pose_meas_rx,pose_meas_ry,pose_meas_rz,motion_err_lin_x_mm,motion_err_lin_y_mm,motion_err_lin_z_mm,motion_err_rot_x_deg,motion_err_rot_y_deg,motion_err_rot_z_deg,motion_err_rms_mm,motion_axis_peak_mm,vel_ff_vx,vel_ff_vy,vel_ff_vz,vel_ff_wx,vel_ff_wy,vel_ff_wz,rail_contrib_m_s,arm_contrib_m_s,arm_y_qdot,rail_motion_share,rail_exec_for_qp,wln_scale_rail,wln_scale_arm_max,waste_ratio,rail_ff_m,rail_posture_err_m,rail_escape_active,psi_deg,psi_ref_deg,psi_retarget_score,d_pref_m,d_star_m,psi_star_deg,minmax_margin,elbow_margin_rad,wrist_open_rad,family_ok,tool_y_des_m,tool_y_err_mm,contact_phase,v_air_cmd,ke_hat,dob_v,barrier_cap_floor,flow_x_p,flow_v_p,flow_v_aux,flow_x_a,flow_v_a,flow_e,flow_edot,flow_F_c,flow_v_track,flow_P_e,flow_P_c,flow_alpha_target,flow_alpha,flow_alpha_case,flow_T,flow_psi,flow_S_n,flow_S_r_hat,flow_P_phys,flow_P_mismatch,flow_E_phys,flow_E_mismatch,flow_gamma_active,flow_alpha_would_gate,flow_edot_aligned,flow_sign_fault,flow_feedback_stale,flow_blocked_reason,contact_episode_rearm_event,contact_episode_release_s,surface_force_scale,surface_force_alpha,surface_xy_error_m,force_barrier_contact_active,qpik_backend,qpik_solver_status,qpik_solver_iterations,qpik_solver_solve_ms,qpik_solver_call_count,qpik_solver_overrun,qpik_qp1_status,qpik_qp2_status,qpik_qp1_solve_ms,qpik_qp2_solve_ms,qpik_assembly_ms,qpik_fallback_ms,qpik_total_ms,qpik_qp2_fallback,tick_inner_ms,tick_send_ms,tick_log_ms,qpik_alpha,qpik_beta,qpik_authority,qpik_equality_residual_max,qpik_hard_residual_max,qpik_anchor_valid,qpik_recovery_overflow,qpik_protected_nominal_overflow_json,qpik_recovery_caps_json,qpik_recovery_overflow_indices_json,qpik_hard_active_constraint_ids_json,qpik_protected_target_json,qpik_protected_achieved_json,qpik_protected_residual_json,qpik_scan_target_json,qpik_scan_achieved_json,qpik_scan_residual_json,qpik_working_slack_json,qpik_collision_slack_json,qpik_dexterity_slack,qpik_branch_slack,qpik_rail_macro_pref_v,qpik_rail_center_pref_v,qpik_rail_final_qdot,qpik_arm_risk_pref_norm,qpik_arm_risk_pref_json,qpik_risk_direction_cosine,qpik_path_velocity_xy_json,qpik_feedback_xy_raw_json,qpik_feedback_xy_filtered_json,qpik_rail_xy_contribution_json,qpik_arm_xy_contribution_json,qpik_rail_task_projection,qpik_rail_arm_cancel,qpik_rail_decomposition_error,qpik_arm_rho,qpik_joint_margin_rad,qpik_wrist_margin_rad,qpik_wrist_singularity,qpik_accepted_reference_lag_s,qpik_pre_solve_feedback_age_s,qpik_post_solve_feedback_age_s,qpik_q_cmd_q_meas_norm,qpik_fallback_level,qpik_fallback_reason,qpik_solver_fault_latched,qpik_final_sent_qdot_json,post_qp_step_clamp_enabled,post_step_would_clamp,post_step_clamp_applied,dt_nom_s,dt_int_s,box_h1_s,box_h2_s,qpik_qdot_raw_json,qpik_qdot_pre_commit_json,qpik_qdot_committed_json,qpik_qdot_prev_used_json,qpik_qdot_prev2_used_json,qpik_box_lo_json,qpik_box_hi_json,post_step_shadow_q_json,q_cmd_json,arm_send_mono_ns,rail_target_publish_mono_ns,rail_fa24_write_mono_ns,rail_encoder_sample_mono_ns,arm_qdot_target_wall_json,rail_sat,rail_exec_velocity_m_s,rail_measured_velocity_m_s,rail_commanded_velocity_m_s,rail_commanded_acceleration_m_s2,rail_feedback_age_s,a_mirror_frac,j_mirror_frac,last_limit_saturated,keep_task_weight,pref_slack_scale,rail_task_vel,v_escape,v_reach,v_ff_rail,u_alloc,u_posture,u_mid,v_r_ref,comp_projected_frac,rail_coast_active,rail_feedback_reject_streak_s,wall_override,slack_zero_feasible,sigma_arm,sns_scale,qpik_nullspace_norm,qpik_nullspace_centering_norm,qpik_nullspace_manip_norm,qpik_nullspace_arm_angle_norm,qpik_nullspace_damping_norm,qpik_nullspace_rail_lock_norm,cbf_min_dist,cbf_pair,qdot_meas_0,qdot_meas_1,qdot_meas_2,qdot_meas_3,qdot_meas_4,qdot_meas_5,qdot_meas_6,qdot_meas_7,v_cmd_vx,v_cmd_vy,v_cmd_vz,v_cmd_wx,v_cmd_wy,v_cmd_wz,path_twist_vx,path_twist_vy,path_twist_vz,path_twist_wx,path_twist_wy,path_twist_wz,feedback_twist_vx,feedback_twist_vy,feedback_twist_vz,feedback_twist_wx,feedback_twist_wy,feedback_twist_wz,comfort_slack_j1,comfort_slack_j2,comfort_slack_j3,comfort_slack_j4,comfort_slack_j5,comfort_slack_j6,comfort_slack_j7,pad_connected,pad_lx,pad_ly,pad_lt,pad_rx,pad_ry,pad_rt,pad_lb,pad_rb,pad_vx,pad_vy,pad_vz,pad_wx,pad_wy,pad_wz,pad_vcmd_base_vx,pad_vcmd_base_vy,pad_vcmd_base_vz,pad_vcmd_base_wx,pad_vcmd_base_wy,pad_vcmd_base_wz
0.0020,gamepad_vcmd,qpik,0.0050,0.025120,-0.333375,-1.279867,0.731939,1.481366,0.499775,1.780655,2.236727,0.025120,-0.333410,-1.279833,0.731886,1.481383,0.499810,1.780655,2.236727,0.045556,-0.167724,0.274787,3.141146,0.000026,3.141156,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.000,0.0030,0.00000,0,0,0,0,0.000,0.000,0.000,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,nan,0,0,0,0,0,,0,0,0,nan,nan,nan,0,0.005000,-0.002381,0.001177,0.001178,0,nan,nan,nan,0.000000,0,nan,nan,nan,nan,0,nan,nan,0,1.0000,1.0000,0.16441,0.00000,0.0000,0.00000,0.000,0.025120,0.025120,0.000000,,0,0.000000,0.045556,-0.167724,0.274787,3.141146,0.000026,3.141156,0.045556,-0.167724,0.274787,3.141146,0.000026,3.141156,0.000,0.000,0.000,0.0000,0.0000,0.0000,0.000,0.000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000000,-0.000000,0.5056,0.000000,1.0000,1.0000,83.1928,0.025119,-0.000001,0,58.4392,58.5640,,-0.192843,-0.192843,68.0000,,0.874634,1.780655,1,-0.167581,0.000,,,,,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,1,1,,0,nan,nan,nan,nan,0,proxqpwbc,solved,0,3.089051,1,0,solved,solved,0.048693,0.052194,2.988164,0.000000,3.089051,0,5.9951,0.3698,,0.99999999,1.00000000,1.00000000,4.984054139e-09,6.419864101e-07,1,0,,,,,,,,,,,,,1.626879536e-18,1.626879537e-18,0.00000000,0.00000000,0.00000000,0.00000000,,nan,,,,,,nan,nan,0.000000000e+00,0.16440991,nan,nan,nan,0.000000,0.001177,0.003316,0.00008188,none,,0,"[0.0,6.419864084783455e-07,4.330321878853738e-07,6.503666938328934e-07,1.4998100539287407e-07,-6.852635681120489e-07,-4.155905397595916e-07,6.566658328210906e-07]",1,0,0,5.000000000e-03,5.000000000e-03,5.000000000e-03,,,,,,,,,,"[0.02512046813964844,-0.33337533428402766,-1.279867413985354,0.7319387364543855,1.4813656470852323,0.49977503187700006,1.7806547501957328,2.2367266848713734]",787454399422709,787454399411867,,787454380145254,,0,0.00000004,0.00000105,0.00000000,0.00000000,0.012365,0.000000,0.000000,0,0,1.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000002,0.000000,0.000000,0,0.000000,1,0,0.12203,1.0000,0.072417,1.123650,0.000000,0.003973,0.000000,0.000000,0.041441,link_6_0:link_8_0,,,,,,,,,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.626879537e-18,1.626879537e-18,1.626879537e-18,1.626879537e-18,1.626879537e-18,1.626879537e-18,1.626879537e-18,1,-0.0163,0.0117,-1.0000,-0.0139,0.0287,-1.0000,0,0,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
0.0096,gamepad_vcmd,qpik,0.0125,0.025120,-0.333375,-1.279867,0.731939,1.481366,0.499775,1.780655,2.236727,0.025120,-0.333393,-1.279867,0.731886,1.481366,0.499827,1.780672,2.236709,0.045545,-0.167719,0.274785,3.141173,0.000049,3.141200,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.000,0.0030,0.00000,0,0,0,0,0.000,0.000,0.000,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,nan,0,0,0,0,0,,0,0,0,nan,nan,nan,0,0.007541,-0.004857,0.000719,0.000719,1,nan,nan,nan,0.000000,0,nan,nan,nan,nan,0,nan,nan,0,1.0000,1.0000,0.16441,0.00000,0.0000,0.00000,0.000,0.025120,0.025120,0.000000,,0,0.000000,0.045556,-0.167724,0.274787,3.141146,0.000026,3.141156,0.045545,-0.167719,0.274785,3.141173,0.000049,3.141200,-0.010,-0.005,-0.002,-0.0015,0.0013,0.0026,0.007,0.010,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000000,-0.000000,0.4218,0.000000,1.0000,1.0000,6.3965,0.025117,-0.000003,0,58.4392,58.7203,,-0.192841,-0.192841,68.0000,,0.874634,1.780655,1,-0.167590,-0.005,,,,,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,1,1,,0,nan,nan,nan,nan,0,proxqpwbc,solved,0,2.506198,2,0,solved,solved,0.025093,0.036893,2.444212,0.000000,2.506198,0,6.3556,0.2442,0.2194,0.99999999,1.00000000,1.00000000,1.130469493e-08,1.692445936e-07,1,0,,,,,,,,,,,,,2.000153046e-18,2.000153046e-18,0.00000000,0.00000000,0.00000000,0.00000000,,nan,,,,,,nan,nan,0.000000000e+00,0.16441044,nan,nan,nan,0.000000,0.000719,0.003387,0.00007998,none,,0,"[0.0,1.6924459167455552e-07,1.8259207479331963e-08,-2.59752575004768e-08,1.5732251057443136e-07,-1.396675397558056e-07,-5.880128384205818e-08,-2.1090684754199174e-08]",1,0,0,5.000000000e-03,6.250000000e-03,8.116021054e-03,5.000000000e-03,,,,,,,,,"[0.02512046813964844,-0.33337533322624896,-1.279867413871234,0.7319387362920401,1.481365648068498,0.49977503100407794,1.7806547498282248,2.2367266847395566]",787454407009594,787454406995287,,787454395948251,,0,0.00000003,0.00000105,0.00000000,0.00000000,0.019911,0.000002,0.000002,0,0,1.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000004,0.000000,0.000000,0,0.000000,1,0,0.12203,1.0000,0.074704,1.123678,0.000000,0.008946,0.000001,0.000000,0.041441,link_6_0:link_8_0,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,2.000153046e-18,2.000153046e-18,2.000153046e-18,2.000153046e-18,2.000153046e-18,2.000153046e-18,2.000153046e-18,1,-0.0163,0.0117,-1.0000,-0.0139,0.0287,-1.0000,0,0,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
0.0170,gamepad_vcmd,qpik,0.0200,0.025120,-0.333375,-1.279867,0.731939,1.481366,0.499775,1.780655,2.236727,0.025120,-0.333393,-1.279815,0.731886,1.481366,0.499775,1.780707,2.236709,0.045548,-0.167737,0.274787,3.141187,0.000002,3.141146,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,-0.00000,0.00000,0.00000,0.00000,0.00000,0.000,0.0030,0.00000,0,0,0,0,0.000,0.000,0.000,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,nan,0,0,0,0,0,,0,0,0,nan,nan,nan,0,0.007480,-0.001786,0.000656,0.000657,1,nan,nan,nan,-0.000000,0,nan,nan,nan,nan,0,nan,nan,0,1.0000,1.0000,0.16441,0.00000,0.0000,0.00000,0.000,0.025120,0.025120,0.000000,,0,0.000000,0.045556,-0.167724,0.274787,3.141146,0.000026,3.141156,0.045548,-0.167737,0.274787,3.141187,0.000002,3.141146,-0.008,0.013,-0.001,-0.0024,-0.0014,-0.0006,0.009,0.013,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000000,-0.000000,-0.000000,0.8739,-0.000000,1.0000,1.0000,1.0000,0.025115,-0.000005,0,58.4392,58.8765,,-0.192839,-0.192839,68.0000,,0.874634,1.780655,1,-0.167592,0.013,,,,,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,1,1,,0,nan,nan,nan,nan,0,proxqpwbc,solved,0,2.569756,3,0,solved,solved,0.024824,0.036323,2.508609,0.000000,2.569756,0,5.8600,0.3728,0.2321,0.99999967,1.00000000,1.00000000,3.316155032e-07,1.535686847e-07,1,0,,,,,,,,,,,,,2.000153601e-18,2.000153601e-18,0.00000000,0.00000000,0.00000000,0.00000000,,nan,,,,,,nan,nan,0.000000000e+00,0.16440803,nan,nan,nan,0.000000,0.000656,0.003446,0.00009395,none,,0,"[0.0,1.5356868665605816e-07,4.475460002595355e-09,-4.857451330053664e-08,1.5756640436848102e-07,-1.214090339374252e-07,-4.678671672309065e-08,-4.35039737567422e-08]",1,0,0,5.000000000e-03,6.250000000e-03,6.209118990e-03,8.116021054e-03,,,,,,,,,"[0.025120441436767577,-0.33337533226644467,-1.2798674138432624,0.7319387359884494,1.481365649053288,0.4997750302452715,1.7806547495358078,2.2367266844676568]",787454413869294,787454413858638,,787454395948251,,0,-0.00000029,-0.00000279,-0.00000000,0.00000000,0.011588,0.000065,0.000102,0,0,1.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000006,0.000000,0.000000,0,0.000000,1,0,0.12203,1.0000,0.077388,1.123709,0.000000,0.013920,0.000000,0.000000,0.041441,link_6_0:link_8_0,-0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,2.000153601e-18,2.000153601e-18,2.000153601e-18,2.000153601e-18,2.000153601e-18,2.000153601e-18,2.000153601e-18,1,-0.0163,0.0117,-1.0000,-0.0139,0.0287,-1.0000,0,0,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
...
200.1293,gamepad_vcmd,qpik,200.1323,0.376245,-1.822745,-1.319341,1.014970,1.357706,1.569774,1.505289,1.374004,0.376281,-1.822822,-1.320132,1.013932,1.357587,1.571285,1.505957,1.373749,0.396870,0.205770,0.188936,-3.134649,-0.202589,-3.049855,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00045,-0.00284,-0.00105,-0.01078,-0.00207,-0.00665,0.000,0.0627,0.00015,1,0,0,0,0.000,0.000,0.000,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,nan,0,0,0,0,0,,0,0,0,nan,nan,nan,0,0.006092,-0.000750,0.003617,0.003617,1,nan,nan,nan,0.001008,0,nan,nan,nan,nan,0,nan,nan,0,1.0000,1.0000,0.17240,0.13414,0.0399,0.00000,0.004,0.376245,0.376269,-0.000037,,0,0.000000,0.396730,0.205821,0.188911,-3.134105,-0.203488,-3.050131,0.396870,0.205770,0.188936,-3.134649,-0.202589,-3.049855,0.127,0.064,0.051,0.0280,0.0516,0.0151,0.087,0.127,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.006654,0.006662,0.006662,0.4997,-0.006654,1.0000,1.0000,1674.0032,0.390821,0.014540,0,71.1056,68.0000,1.417952,-0.185000,-0.185000,68.0000,,0.998133,1.505683,1,0.166744,0.064,,,,,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,1,1,,0,nan,nan,nan,nan,0,proxqpwbc,solved,0,2.083268,36709,0,solved,solved,0.029173,0.025597,2.028498,0.000000,2.083268,0,5.0415,0.0969,0.1918,0.99985061,1.00000000,1.00000000,1.378700335e-04,0.000000000e+00,1,0,,,,,,,,,,,,,2.000094285e-12,2.000094285e-12,-0.00607121,0.00000000,-0.00599201,0.00000000,,nan,,,,,,nan,nan,0.000000000e+00,0.17239772,nan,nan,nan,0.000000,0.003617,0.003057,0.00212586,none,,0,"[-0.005992008860637657,0.031264402016260344,0.041312009144817106,0.06935039367268031,-0.02646821213405735,-0.06821555657811579,-0.06470694440659489,0.03037778611864143]",1,0,0,5.000000000e-03,6.092198077e-03,5.809445982e-03,6.537655951e-03,,,,,,,,,"[0.37624457414443224,-1.8227452300070348,-1.3193411719188664,1.0149700308142349,1.3577056338461122,1.5697743845905179,1.5052888448254687,1.374003576098767]",787654525468401,787654525460652,787654498526043,787654516459594,,0,-0.00665422,-0.00796695,-0.00647288,0.01610275,0.026086,0.039092,0.082387,0,0,1.0000,-0.006071,0.000000,0.000000,0.000000,0.000000,0.000000,-0.005944,-0.006071,0.000000,0,0.000000,0,0,0.11199,1.0000,0.307991,0.298109,0.005760,0.105445,0.068070,0.000000,0.039259,link_6_0:link_8_0,-0.006654,0.031416,0.035605,0.067021,-0.025133,-0.075398,-0.067021,0.031416,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,2.000094285e-12,2.000094285e-12,2.000094285e-12,2.000094285e-12,2.000094285e-12,2.000094285e-12,2.000094285e-12,1,-0.0066,0.0117,-1.0000,-0.0239,0.0010,-1.0000,0,0,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
200.1353,gamepad_vcmd,qpik,200.1383,0.376210,-1.822563,-1.319097,1.015376,1.357549,1.569373,1.504912,1.374182,0.376245,-1.822578,-1.319888,1.014368,1.357447,1.570796,1.505591,1.373836,0.396847,0.205772,0.188922,-3.134541,-0.202645,-3.049797,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,-0.00036,-0.00162,-0.00222,-0.00403,0.00490,-0.00477,0.000,0.0586,0.00013,1,0,0,0,0.000,0.000,0.000,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,nan,0,0,0,0,0,,0,0,0,nan,nan,nan,0,0.005943,-0.001708,0.003522,0.003523,1,nan,nan,nan,0.002286,0,nan,nan,nan,nan,0,nan,nan,0,1.0000,1.0000,0.17241,0.13238,0.0393,0.00000,0.004,0.376210,0.376269,-0.000035,,0,0.000000,0.396730,0.205821,0.188911,-3.134105,-0.203488,-3.050131,0.396847,0.205772,0.188922,-3.134541,-0.202645,-3.049797,0.108,0.060,0.032,0.0211,0.0484,0.0184,0.074,0.108,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.006520,0.006524,0.006524,0.4998,-0.006520,1.0000,1.0000,3104.9562,0.390821,0.014576,0,71.1341,68.0000,1.417952,-0.185000,-0.185000,68.0000,,0.998294,1.505289,1,0.166691,0.060,,,,,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,1,1,,0,nan,nan,nan,nan,0,proxqpwbc,solved,0,2.053309,36710,0,solved,solved,0.028915,0.021249,2.003145,0.000000,2.053309,0,5.0287,0.1596,0.1809,0.99987067,1.00000000,1.00000000,7.975700441e-05,0.000000000e+00,1,0,,,,,,,,,,,,,2.000095073e-12,2.000095073e-12,-0.00597901,0.00000000,-0.00590114,0.00000000,,nan,,,,,,nan,nan,0.000000000e+00,0.17240970,nan,nan,nan,0.000000,0.003522,0.002764,0.00206422,none,,0,"[-0.005901141876132165,0.030725171962242535,0.041128022472366244,0.06828807322537005,-0.026354534608985514,-0.06758619336081259,-0.0634844511907281,0.029973287635769195]",1,0,0,5.000000000e-03,5.942506017e-03,5.932910019e-03,5.809445982e-03,,,,,,,,,"[0.37620993286627524,-1.8225626454877708,-1.3190967683978503,1.0153758331002771,1.3575490218656188,1.569372753229794,1.5049115880922714,1.3741816925408967]",787654531361198,787654531353343,787654498526043,787654516459594,,0,-0.00652013,-0.00796695,-0.00647288,0.01610275,0.032028,0.034511,0.007726,0,0,1.0000,-0.005979,0.000000,0.000000,0.000000,0.000000,0.000000,-0.005856,-0.005979,0.000000,0,0.000000,0,0,0.11198,1.0000,0.308101,0.298036,0.005778,0.106456,0.067069,0.000000,0.039258,link_6_0:link_8_0,-0.006520,0.029322,0.037699,0.071209,-0.025133,-0.071209,-0.062832,0.031416,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,2.000095073e-12,2.000095073e-12,2.000095073e-12,2.000095073e-12,2.000095073e-12,2.000095073e-12,2.000095073e-12,1,-0.0066,0.0117,-1.0000,-0.0239,0.0010,-1.0000,0,0,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
```

### `rail_20260819_204333.csv`  （9202 lines including header, 3302108 bytes）

```csv
t_wall_s,event,target_m,commanded_m,measured_m,v_ff,v_des,v_cmd,rpm,follow,armed,panic,poll_ok,dt_wall_ms,last_rpm_cmd,mb_fail_n,freeze_flag,arm_good,sample_mono_s,target_rx_mono_s,target_age_ms,motion_seq,feedback_valid,x_goal_m,x_ref_m,x_meas_m,v_goal_est_m_s,v_ref_m_s,a_ref_m_s2,v_reg_m_s,v_enc_m_s,v_enc_source,v_des_m_s,v_cmd_m_s,a_cmd_m_s2,x_goal_eval_m,rpm_cmd,e_track_mm,e_shape_mm,hold_count,hold_reason,command_mode,t_read_ms,t_write_ms,n_modbus,fa24_write_mono_ns,encoder_sample_mono_ns
0.0003,session_begin,0.025120,0.025120,0.025120,,,,,0,1,0,1,,0,0,0,0,,,,0,0,,,,,,,,,,,,,,0,,,0,,position,,,0,,
0.0119,,0.025120,0.025120,0.025120,0.000000,0.000000,-0.000000,0.000000,1,1,0,1,19.561770,0,0,0,0,787454.395948,787454.399418,-3.469759,189,1,0.025120,0.025120,0.025120,0.000000,0.000000,0.000000,0.000000,-0.000003,lsq,0.000000,-0.000000,0.000000,0.025120,0,0.000000,0.000076,0,,coupled_velocity,11.959242,0.000000,1,,787454395948251
0.0241,,0.025120,0.025120,0.025120,0.000000,0.000000,-0.000000,0.000000,1,1,0,1,17.195733,0,0,0,0,787454.410642,787454.413865,-3.223118,190,1,0.025120,0.025120,0.025120,0.000000,0.000000,0.000000,0.000000,-0.000002,lsq,0.000000,-0.000000,0.000000,0.025120,0,0.000000,-0.000027,0,,coupled_velocity,6.949827,0.000000,1,,787454410642106
...
203.6501,,0.376269,0.376269,0.375514,0.000000,0.000000,0.000000,-0.000000,0,1,0,1,21.086853,0,0,0,0,787658.030622,787654.531358,3499.263671,9386,1,0.376269,0.375514,0.375514,0.000000,0.000000,0.000000,0.000000,-0.000001,lsq,0.000000,0.000000,0.000000,0.375515,0,0.000000,0.000076,9,,position,18.893961,0.000000,1,787654519573044,787658030621502
203.6563,,0.376269,0.376269,0.375515,0.000000,0.000000,0.000000,-0.000000,0,1,0,1,19.071313,0,0,0,0,787658.043307,787654.531358,3511.949446,9387,1,0.376269,0.375515,0.375515,0.000000,0.000000,0.000000,0.000000,-0.000001,lsq,0.000000,0.000000,0.000000,0.375514,0,0.000000,-0.000076,9,,position,6.138051,0.000000,1,787654519573044,787658043307276
```

### `run_20260819_204658.csv`  （7928 lines including header, 21736085 bytes）

```csv
t_wall_s,phase,controller_mode,t_ref_s,q_cmd_0,q_cmd_1,q_cmd_2,q_cmd_3,q_cmd_4,q_cmd_5,q_cmd_6,q_cmd_7,q_meas_0,q_meas_1,q_meas_2,q_meas_3,q_meas_4,q_meas_5,q_meas_6,q_meas_7,pose_x,pose_y,pose_z,pose_rx,pose_ry,pose_rz,twist_vx,twist_vy,twist_vz,twist_wx,twist_wy,twist_wz,twist_requested_vx,twist_requested_vy,twist_requested_vz,twist_requested_wx,twist_requested_wy,twist_requested_wz,twist_achieved_vx,twist_achieved_vy,twist_achieved_vz,twist_achieved_wx,twist_achieved_wy,twist_achieved_wz,track_err_mm,follow_err_deg,slack_norm,n_cbf,vel_clamped,acc_clamped,pos_clamped,fx,fy,fz,instability_idx,instability_idx_raw,instability_idx_active,damping_z_eff,damping_ke_z,damping_dimeas_z,v_force_z,ke_est,f_des_z_eff,v_r_z,force_reference_scale_n,force_reference_drive,force_reference_gate_scale,force_reference_accel_m_s2,force_reference_reversal_reset,force_reference_fast_clear,force_fast_z,retract_guard_armed,retract_fast_hold,retract_fast_stop_count,retract_fast_rearm_count,force_task_latched,physical_contact_state,physical_contact_acquire_event,physical_contact_loss_event,physical_contact_reacquire_event,physical_contact_low_timer_s,physical_contact_high_timer_s,mass_z_eff,takeover,dt_actual_s,deadline_slack_s,sensor_age_s,feedback_age_s,feedback_fresh_tick,fx_raw_comp,fy_raw_comp,fz_raw_comp,vz_achieved_tool,contact_present,force_pred_z,force_dot_z,cap_press_z,cap_retract_z,ke_update_gated,ke_dx_m,ke_df_n,ke_update_count,governor_scale,governor_scale_raw,sigma_min,qdot_norm,qdot_max_frac_vmax,qdot_ff_norm,tcp_jump_mm,rail_target_sent_m,rail_meas_m,rail_cmd_meas_err_m,rail_vel_pin,plan_drives_rail,rail_qdot_ff,pose_d_x,pose_d_y,pose_d_z,pose_d_rx,pose_d_ry,pose_d_rz,pose_meas_x,pose_meas_y,pose_meas_z,pose_meas_rx,pose_meas_ry,pose_meas_rz,motion_err_lin_x_mm,motion_err_lin_y_mm,motion_err_lin_z_mm,motion_err_rot_x_deg,motion_err_rot_y_deg,motion_err_rot_z_deg,motion_err_rms_mm,motion_axis_peak_mm,vel_ff_vx,vel_ff_vy,vel_ff_vz,vel_ff_wx,vel_ff_wy,vel_ff_wz,rail_contrib_m_s,arm_contrib_m_s,arm_y_qdot,rail_motion_share,rail_exec_for_qp,wln_scale_rail,wln_scale_arm_max,waste_ratio,rail_ff_m,rail_posture_err_m,rail_escape_active,psi_deg,psi_ref_deg,psi_retarget_score,d_pref_m,d_star_m,psi_star_deg,minmax_margin,elbow_margin_rad,wrist_open_rad,family_ok,tool_y_des_m,tool_y_err_mm,contact_phase,v_air_cmd,ke_hat,dob_v,barrier_cap_floor,flow_x_p,flow_v_p,flow_v_aux,flow_x_a,flow_v_a,flow_e,flow_edot,flow_F_c,flow_v_track,flow_P_e,flow_P_c,flow_alpha_target,flow_alpha,flow_alpha_case,flow_T,flow_psi,flow_S_n,flow_S_r_hat,flow_P_phys,flow_P_mismatch,flow_E_phys,flow_E_mismatch,flow_gamma_active,flow_alpha_would_gate,flow_edot_aligned,flow_sign_fault,flow_feedback_stale,flow_blocked_reason,contact_episode_rearm_event,contact_episode_release_s,surface_force_scale,surface_force_alpha,surface_xy_error_m,force_barrier_contact_active,qpik_backend,qpik_solver_status,qpik_solver_iterations,qpik_solver_solve_ms,qpik_solver_call_count,qpik_solver_overrun,qpik_qp1_status,qpik_qp2_status,qpik_qp1_solve_ms,qpik_qp2_solve_ms,qpik_assembly_ms,qpik_fallback_ms,qpik_total_ms,qpik_qp2_fallback,tick_inner_ms,tick_send_ms,tick_log_ms,qpik_alpha,qpik_beta,qpik_authority,qpik_equality_residual_max,qpik_hard_residual_max,qpik_anchor_valid,qpik_recovery_overflow,qpik_protected_nominal_overflow_json,qpik_recovery_caps_json,qpik_recovery_overflow_indices_json,qpik_hard_active_constraint_ids_json,qpik_protected_target_json,qpik_protected_achieved_json,qpik_protected_residual_json,qpik_scan_target_json,qpik_scan_achieved_json,qpik_scan_residual_json,qpik_working_slack_json,qpik_collision_slack_json,qpik_dexterity_slack,qpik_branch_slack,qpik_rail_macro_pref_v,qpik_rail_center_pref_v,qpik_rail_final_qdot,qpik_arm_risk_pref_norm,qpik_arm_risk_pref_json,qpik_risk_direction_cosine,qpik_path_velocity_xy_json,qpik_feedback_xy_raw_json,qpik_feedback_xy_filtered_json,qpik_rail_xy_contribution_json,qpik_arm_xy_contribution_json,qpik_rail_task_projection,qpik_rail_arm_cancel,qpik_rail_decomposition_error,qpik_arm_rho,qpik_joint_margin_rad,qpik_wrist_margin_rad,qpik_wrist_singularity,qpik_accepted_reference_lag_s,qpik_pre_solve_feedback_age_s,qpik_post_solve_feedback_age_s,qpik_q_cmd_q_meas_norm,qpik_fallback_level,qpik_fallback_reason,qpik_solver_fault_latched,qpik_final_sent_qdot_json,post_qp_step_clamp_enabled,post_step_would_clamp,post_step_clamp_applied,dt_nom_s,dt_int_s,box_h1_s,box_h2_s,qpik_qdot_raw_json,qpik_qdot_pre_commit_json,qpik_qdot_committed_json,qpik_qdot_prev_used_json,qpik_qdot_prev2_used_json,qpik_box_lo_json,qpik_box_hi_json,post_step_shadow_q_json,q_cmd_json,arm_send_mono_ns,rail_target_publish_mono_ns,rail_fa24_write_mono_ns,rail_encoder_sample_mono_ns,arm_qdot_target_wall_json,rail_sat,rail_exec_velocity_m_s,rail_measured_velocity_m_s,rail_commanded_velocity_m_s,rail_commanded_acceleration_m_s2,rail_feedback_age_s,a_mirror_frac,j_mirror_frac,last_limit_saturated,keep_task_weight,pref_slack_scale,rail_task_vel,v_escape,v_reach,v_ff_rail,u_alloc,u_posture,u_mid,v_r_ref,comp_projected_frac,rail_coast_active,rail_feedback_reject_streak_s,wall_override,slack_zero_feasible,sigma_arm,sns_scale,qpik_nullspace_norm,qpik_nullspace_centering_norm,qpik_nullspace_manip_norm,qpik_nullspace_arm_angle_norm,qpik_nullspace_damping_norm,qpik_nullspace_rail_lock_norm,cbf_min_dist,cbf_pair,qdot_meas_0,qdot_meas_1,qdot_meas_2,qdot_meas_3,qdot_meas_4,qdot_meas_5,qdot_meas_6,qdot_meas_7,v_cmd_vx,v_cmd_vy,v_cmd_vz,v_cmd_wx,v_cmd_wy,v_cmd_wz,path_twist_vx,path_twist_vy,path_twist_vz,path_twist_wx,path_twist_wy,path_twist_wz,feedback_twist_vx,feedback_twist_vy,feedback_twist_vz,feedback_twist_wx,feedback_twist_wy,feedback_twist_wz,comfort_slack_j1,comfort_slack_j2,comfort_slack_j3,comfort_slack_j4,comfort_slack_j5,comfort_slack_j6,comfort_slack_j7,pad_connected,pad_lx,pad_ly,pad_lt,pad_rx,pad_ry,pad_rt,pad_lb,pad_rb,pad_vx,pad_vy,pad_vz,pad_wx,pad_wy,pad_wz,pad_vcmd_base_vx,pad_vcmd_base_vy,pad_vcmd_base_vz,pad_vcmd_base_wx,pad_vcmd_base_wy,pad_vcmd_base_wz
0.0009,ellipse_track,qpik,0.0050,0.375515,-1.822488,-1.318971,1.015583,1.357462,1.569074,1.504686,1.374257,0.375515,-1.822508,-1.318980,1.015590,1.357500,1.569068,1.504683,1.374220,0.396886,0.204727,0.188907,-3.134521,-0.202927,-3.050847,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.000,0.0000,0.00663,1,0,0,0,0.000,0.000,0.000,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,nan,0,0,0,0,0,,0,0,0,nan,nan,nan,0,0.005000,-0.002586,0.001849,0.001849,0,nan,nan,nan,0.000000,0,nan,nan,nan,nan,0,nan,nan,0,1.0000,1.0000,0.17233,0.01157,0.0030,0.00000,0.019,0.375515,0.375515,0.000000,,0,0.000000,0.396886,0.204727,0.188907,-3.134521,-0.202927,-3.050847,0.396886,0.204727,0.188907,-3.134521,-0.202927,-3.050847,0.000,0.000,0.000,0.0000,0.0000,0.0000,0.000,0.000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000000,0.003498,0.003498,0.0000,-0.000000,1.0000,1.0000,1.0000,0.375615,0.000100,0,71.1486,71.0239,,-0.170887,-0.170887,68.0000,,0.998500,1.504683,1,0.166016,0.000,,,,,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,1,1,,0,nan,nan,nan,nan,0,proxqpwbc,solved,0,2.465767,1,0,solved,solved,0.042174,0.022675,2.400918,0.000000,2.465767,0,6.8952,0.1197,,0.99336878,1.00000000,1.00000000,4.509797151e-03,0.000000000e+00,1,0,,,,,,,,,,,,,2.000137089e-12,2.000137089e-12,0.00000000,0.00000000,0.00000000,0.00000000,,nan,,,,,,nan,nan,0.000000000e+00,0.17232977,nan,nan,nan,0.000000,0.001849,0.000689,0.00005787,none,,0,"[0.0,0.0038795714149664207,0.0018485683370261796,-0.0013324710761875025,-0.0074979931966812075,0.0010603600355985066,0.0006061977476878866,0.007483983921563464]",1,0,0,5.000000000e-03,5.000000000e-03,5.000000000e-03,,,,,,,,,,"[0.3755145263671875,-1.822488262527751,-1.3189709549743875,1.0155829617065295,1.3574621372247773,1.5690737696798311,1.5046862505643634,1.3742573045402675]",787658072486164,787658072475490,787654519573044,787658065598868,,0,-0.00000005,-0.00000125,0.00000000,0.00000000,0.021781,0.000000,0.000000,0,0,1.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000120,0.000000,0.000000,0,0.000000,0,0,0.11197,1.0000,0.011975,0.000317,0.010264,0.004291,0.000000,0.000000,0.039257,link_6_0:link_8_0,,,,,,,,,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,2.000137089e-12,2.000137089e-12,2.000137089e-12,2.000137089e-12,2.000137089e-12,2.000137089e-12,2.000137089e-12,,,,,,,,,,,,,,,,,,,,,
0.0087,ellipse_track,qpik,0.0128,0.375515,-1.822493,-1.318966,1.015581,1.357457,1.569073,1.504689,1.374256,0.375515,-1.822508,-1.319033,1.015607,1.357517,1.569138,1.504736,1.374202,0.396872,0.204744,0.188879,-3.134542,-0.202998,-3.050794,0.00013,-0.00017,0.00029,-0.00003,-0.00014,-0.00010,0.00013,-0.00017,0.00029,-0.00003,-0.00014,-0.00010,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.036,0.0037,0.00000,1,0,0,0,0.000,0.000,0.000,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,nan,0,0,0,0,0,,0,0,0,nan,nan,nan,0,0.007793,-0.004187,0.001267,0.001267,1,nan,nan,nan,-0.000000,0,nan,nan,nan,nan,0,nan,nan,0,1.0000,1.0000,0.17233,0.00139,0.0003,0.00000,0.002,0.375515,0.375515,0.000000,,0,-0.000000,0.396886,0.204727,0.188907,-3.134521,-0.202927,-3.050847,0.396872,0.204744,0.188879,-3.134542,-0.202998,-3.050794,-0.006,-0.018,-0.030,0.0006,-0.0040,0.0030,0.021,0.030,-0.000000,-0.000000,0.000000,0.000000,0.000000,0.000000,-0.000000,0.000359,0.000359,0.0002,0.000000,1.0000,1.0000,1.0004,0.375740,0.000225,0,71.1486,70.8676,,-0.171012,-0.171012,68.0000,,0.998538,1.504686,1,0.166000,-0.018,,,,,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,1,1,,0,nan,nan,nan,nan,0,proxqpwbc,solved,0,2.121737,2,0,solved,solved,0.034141,0.021801,2.065795,0.000000,2.121737,0,5.7473,0.1016,0.2372,0.99999998,1.00000000,1.00000000,1.707494211e-08,0.000000000e+00,1,0,,,,,,,,,,,,,2.000135550e-12,2.000135550e-12,0.00000000,0.00000000,0.00000000,0.00000000,,nan,,,,,,nan,nan,0.000000000e+00,0.17233415,nan,nan,nan,0.000000,0.001267,0.003221,0.00013499,none,,0,"[0.0,-0.000732735939479312,0.0007266038157283106,-0.0003021764159782947,-0.0007510481781736189,-0.00018107162880198757,0.00036730863278933157,-0.0001810751355790785]",1,0,0,5.000000000e-03,6.250000000e-03,8.270422928e-03,5.000000000e-03,,,,,,,,,"[0.37551455307006837,-1.8224928421273727,-1.3189664137005392,1.0155810731039296,1.3574574431736637,1.5690726379821511,1.5046885462433184,1.3742561728206701]",787658079106962,787658079096835,787654519573044,787658065598868,,0,0.00000015,0.00000067,0.00000000,0.00000000,0.007290,0.000037,0.000045,0,0,1.0000,0.000000,0.000000,0.000000,-0.000000,0.000005,0.000000,0.000271,0.000000,0.000000,0,0.000000,0,0,0.11197,1.0000,0.013987,0.000741,0.009700,0.009630,0.005787,0.000000,0.039257,link_6_0:link_8_0,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000134,-0.000169,0.000287,-0.000027,-0.000144,-0.000098,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000058,-0.000183,-0.000304,0.000020,-0.000140,0.000105,2.000135550e-12,2.000135550e-12,2.000135550e-12,2.000135550e-12,2.000135550e-12,2.000135550e-12,2.000135550e-12,,,,,,,,,,,,,,,,,,,,,
0.0153,ellipse_track,qpik,0.0193,0.375515,-1.822493,-1.318968,1.015580,1.357457,1.569074,1.504689,1.374257,0.375515,-1.822525,-1.318945,1.015607,1.357517,1.569086,1.504701,1.374202,0.396884,0.204726,0.188910,-3.134577,-0.202945,-3.050838,0.00001,0.00002,-0.00003,-0.00011,-0.00005,0.00000,0.00001,0.00002,-0.00003,-0.00011,-0.00005,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.004,0.0034,0.00000,1,0,0,0,0.000,0.000,0.000,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,nan,0,0,0,0,0,,0,0,0,nan,nan,nan,0,0.006529,-0.006433,0.003697,0.003697,1,nan,nan,nan,-0.000000,0,nan,nan,nan,nan,0,nan,nan,0,1.0000,1.0000,0.17232,0.00035,0.0001,0.00000,0.000,0.375515,0.375515,0.000000,,0,-0.000000,0.396886,0.204727,0.188907,-3.134521,-0.202927,-3.050847,0.396884,0.204726,0.188910,-3.134577,-0.202945,-3.050838,-0.002,0.002,0.003,0.0031,-0.0010,0.0005,0.002,0.003,-0.000000,-0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000036,0.000036,0.0014,0.000000,1.0000,1.0000,1.0000,0.375865,0.000350,0,71.1484,70.7114,,-0.171137,-0.171137,68.0000,,0.998543,1.504689,1,0.166028,0.002,,,,,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,1,1,,0,nan,nan,nan,nan,0,proxqpwbc,solved,0,2.175616,3,0,solved,solved,0.038778,0.022822,2.114016,0.000000,2.175616,0,6.3797,0.2464,0.1765,1.00000000,1.00000000,1.00000000,2.314207488e-10,0.000000000e+00,1,0,,,,,,,,,,,,,2.000133297e-12,2.000133297e-12,0.00000000,0.00000000,0.00000000,0.00000000,,nan,,,,,,nan,nan,0.000000000e+00,0.17232340,nan,nan,nan,0.000000,0.003697,0.003163,0.00009529,none,,0,"[0.0,-4.339785274254382e-05,-0.00023769574912790858,-0.00015399386516179447,-5.263879319983289e-05,0.00017021964147545532,6.19926566258755e-05,7.664693388420574e-05]",1,0,0,5.000000000e-03,6.250000000e-03,7.030996028e-03,8.270422928e-03,,,,,,,,,"[0.37551455307006837,-1.8224931133639524,-1.3189678992989713,1.0155801106422724,1.3574571141812062,1.5690737018549104,1.5046889336974223,1.374256651864007]",787658086206111,787658086195463,787654519573044,787658077445522,,0,0.00000011,0.00000067,0.00000000,0.00000000,0.013818,0.000008,0.000064,0,0,1.0000,0.000000,0.000000,0.000000,-0.000000,0.000002,0.000000,0.000423,0.000000,0.000000,0,0.000000,0,0,0.11197,1.0000,0.019714,0.001138,0.009212,0.014986,0.000693,0.000000,0.039257,link_6_0:link_8_0,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000013,0.000016,-0.000030,-0.000105,-0.000046,0.000004,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000020,0.000015,0.000026,0.000107,-0.000036,0.000019,2.000133297e-12,2.000133297e-12,2.000133297e-12,2.000133297e-12,2.000133297e-12,2.000133297e-12,2.000133297e-12,,,,,,,,,,,,,,,,,,,,,
...
39.9923,ellipse_track,qpik,39.9963,0.691282,-1.816488,-1.321965,1.014042,1.510790,1.578280,1.505531,1.228027,0.691368,-1.815526,-1.322314,1.014211,1.511542,1.578388,1.505067,1.227158,0.391313,0.491327,0.182621,-3.134767,-0.203005,-3.050382,0.01283,-0.01682,-0.00199,-0.00046,-0.00020,-0.00083,0.01283,-0.01682,-0.00199,-0.00046,-0.00020,-0.00083,0.01133,-0.01828,-0.00420,0.00035,0.00969,-0.00436,0.179,0.0460,0.00000,1,0,0,0,0.000,0.000,0.000,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,nan,0,0,0,0,0,,0,0,0,nan,nan,nan,0,0.005000,0.001128,0.003713,0.003713,1,nan,nan,nan,0.002304,0,nan,nan,nan,nan,0,nan,nan,0,1.0000,1.0000,0.16585,0.07942,0.1139,0.00000,0.108,0.691282,0.691650,-0.000085,,0,-0.015055,0.391330,0.491150,0.182642,-3.134521,-0.202927,-3.050847,0.391313,0.491327,0.182621,-3.134767,-0.203005,-3.050382,0.003,-0.178,-0.019,0.0087,-0.0043,0.0261,0.103,0.178,0.012652,-0.015055,-0.002195,0.000000,0.000000,0.000000,0.012752,0.008467,0.008467,0.6010,-0.016761,1.0000,1.0000,1.0000,0.676150,-0.015217,0,72.9575,68.0000,,-0.185000,-0.185000,68.0000,,0.844967,1.505555,1,0.451635,-0.178,,,,,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,1,1,,0,nan,nan,nan,nan,0,proxqpwbc,solved,0,1.970767,7926,0,solved,solved,0.023265,0.024804,1.922698,0.000000,1.970767,0,3.2678,0.1716,0.2208,1.00000000,1.00000000,1.00000000,8.176757801e-11,0.000000000e+00,1,0,,,,,,,,,,,,,2.002546336e-12,2.002546336e-12,-0.01709801,0.00000000,-0.01709164,0.00000000,,nan,,,,,,nan,nan,0.000000000e+00,0.16585407,nan,nan,nan,0.000000,0.003713,0.007220,0.00162142,none,,0,"[-0.017091642140926882,-0.04835251591503345,-0.007428395933670771,0.01436502552580434,-0.04862584311928955,0.02924497096022295,-0.004864715356419772,0.013156161343435795]",1,0,0,5.000000000e-03,5.000047036e-03,4.810284008e-03,4.977677017e-03,,,,,,,,,"[0.6912821075138487,-1.8164875210581513,-1.3219645768149462,1.0140417717494863,1.5107898102440722,1.5782804256353096,1.5055307276918022,1.2280270324994054]",787698060097291,787698060089258,787698020509439,787698041621671,,0,-0.01676061,-0.01575577,-0.01657695,-0.01014181,0.040059,0.022749,0.000424,0,0,1.0000,-0.017098,0.000000,0.000000,-0.015055,-0.002313,0.000000,-0.014859,-0.017098,0.000000,0,0.000000,0,0,0.12186,1.0000,0.330225,0.278797,0.000000,0.167394,0.042818,0.000000,0.040231,link_6_0:link_8_0,-0.016761,-0.050265,-0.006283,0.018850,-0.046077,0.025133,0.000000,0.012566,0.012826,-0.016821,-0.001987,-0.000465,-0.000198,-0.000831,-0.011441,-0.016146,-0.000003,0.000000,0.000000,0.000000,0.000030,-0.001776,-0.000194,0.000303,-0.000149,0.000912,2.002546336e-12,2.002546336e-12,2.002546336e-12,2.002546336e-12,2.002546336e-12,2.002546336e-12,2.002546336e-12,,,,,,,,,,,,,,,,,,,,,
39.9973,ellipse_track,qpik,40.0013,0.691196,-1.816726,-1.322003,1.014113,1.510545,1.578426,1.505506,1.228098,0.691282,-1.815841,-1.322331,1.014298,1.511228,1.578511,1.505067,1.227228,0.391386,0.491226,0.182610,-3.134737,-0.203041,-3.050413,0.01272,-0.01661,-0.00199,-0.00040,-0.00027,-0.00078,0.01272,-0.01661,-0.00199,-0.00040,-0.00027,-0.00078,0.01220,-0.01627,-0.00212,-0.00157,0.00284,0.00087,0.152,0.0458,0.00000,1,0,0,0,0.000,0.000,0.000,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,nan,0,0,0,0,0,,0,0,0,nan,nan,nan,0,0.005000,0.000466,0.001454,0.001454,1,nan,nan,nan,0.000042,0,nan,nan,nan,nan,0,nan,nan,0,1.0000,1.0000,0.16584,0.07938,0.1141,0.00000,0.106,0.691196,0.691650,-0.000086,,0,-0.015099,0.391394,0.491075,0.182631,-3.134521,-0.202927,-3.050847,0.391386,0.491226,0.182610,-3.134737,-0.203041,-3.050413,0.011,-0.151,-0.018,0.0074,-0.0064,0.0244,0.088,0.151,0.012650,-0.015099,-0.002194,0.000000,0.000000,0.000000,0.012820,0.008169,0.008169,0.6108,-0.016828,1.0000,1.0000,1.0000,0.676075,-0.015207,0,72.9564,68.0000,,-0.185000,-0.185000,68.0000,,0.845210,1.505531,1,0.451560,-0.151,,,,,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,1,1,,0,nan,nan,nan,nan,0,proxqpwbc,solved,0,2.107683,7927,0,solved,solved,0.022961,0.025100,2.059622,0.000000,2.107683,0,3.7039,0.3186,0.1745,1.00000000,1.00000000,1.00000000,6.917563397e-11,0.000000000e+00,1,0,,,,,,,,,,,,,2.002563691e-12,2.002563690e-12,-0.01712691,0.00000000,-0.01712051,0.00000000,,nan,,,,,,nan,nan,0.000000000e+00,0.16584023,nan,nan,nan,0.000000,0.001454,0.005482,0.00153514,none,,0,"[-0.017120509799239396,-0.047688686919530335,-0.007683231705677576,0.01416344348349072,-0.04898730390458175,0.029147599085327824,-0.0048808310365525825,0.014187705820998873]",1,0,0,5.000000000e-03,5.000217934e-03,5.125686992e-03,4.810284008e-03,,,,,,,,,"[0.6911964664569762,-1.8167259748857418,-1.322002994647913,1.0141125920536016,1.5105448630485434,1.5784261699829931,1.5055063224729197,1.2280979741204958]",787698065615067,787698065606070,787698045576173,787698041621671,,0,-0.01682838,-0.01575577,-0.01657695,-0.01014181,0.045060,0.019807,0.005736,0,0,1.0000,-0.017127,0.000000,0.000000,-0.015099,-0.002285,0.000000,-0.014887,-0.017127,0.000000,0,0.000000,0,0,0.12184,1.0000,0.330406,0.278809,0.000000,0.167364,0.043056,0.000000,0.040230,link_6_0:link_8_0,-0.016828,-0.048171,-0.006283,0.014661,-0.052360,0.027227,-0.002094,0.014661,0.012721,-0.016606,-0.001986,-0.000401,-0.000265,-0.000782,-0.011434,-0.016189,-0.000003,0.000000,0.000000,0.000000,0.000107,-0.001508,-0.000180,0.000257,-0.000222,0.000853,2.002563690e-12,2.002563690e-12,2.002563690e-12,2.002563690e-12,2.002563690e-12,2.002563690e-12,2.002563690e-12,,,,,,,,,,,,,,,,,,,,,
```

### `rail_20260819_204658.csv`  （2051 lines including header, 737289 bytes）

```csv
t_wall_s,event,target_m,commanded_m,measured_m,v_ff,v_des,v_cmd,rpm,follow,armed,panic,poll_ok,dt_wall_ms,last_rpm_cmd,mb_fail_n,freeze_flag,arm_good,sample_mono_s,target_rx_mono_s,target_age_ms,motion_seq,feedback_valid,x_goal_m,x_ref_m,x_meas_m,v_goal_est_m_s,v_ref_m_s,a_ref_m_s2,v_reg_m_s,v_enc_m_s,v_enc_source,v_des_m_s,v_cmd_m_s,a_cmd_m_s2,x_goal_eval_m,rpm_cmd,e_track_mm,e_shape_mm,hold_count,hold_reason,command_mode,t_read_ms,t_write_ms,n_modbus,fa24_write_mono_ns,encoder_sample_mono_ns
0.0009,session_begin,0.375515,0.375515,0.375515,,,,,0,1,0,1,,0,0,0,0,,,,0,0,,,,,,,,,,,,,,0,,,0,,position,,,0,,
0.0058,,0.375515,0.375515,0.375515,0.000000,0.000000,0.000000,-0.000000,0,1,0,1,21.965329,0,0,0,0,787658.065599,0.000000,,9388,1,0.375515,0.375515,0.375515,0.000000,0.000000,0.000000,0.000000,0.000001,lsq,0.000000,0.000000,0.000000,0.375515,0,0.000000,-0.000076,0,,position,6.771289,0.000000,1,787654519573044,787658065598868
0.0178,,0.375515,0.375515,0.375515,0.000000,0.000000,0.000000,-0.000000,1,1,0,1,11.730139,0,0,0,0,787658.077446,787658.079103,-1.657521,9389,1,0.375515,0.375515,0.375515,0.000000,0.000000,0.000000,0.000000,0.000001,lsq,0.000000,0.000000,0.000000,0.375515,0,0.000000,0.000027,0,,coupled_velocity,7.017813,0.000000,1,787654519573044,787658077445522
...
43.6184,,0.691650,0.691650,0.690736,0.000000,0.000000,0.000000,-0.000000,0,1,0,1,21.456610,0,0,0,0,787701.673617,787698.065611,3608.005796,11435,1,0.691650,0.690736,0.690736,0.000000,0.000000,0.000000,0.000000,-0.000004,lsq,0.000000,0.000000,0.000000,0.690736,0,0.000000,-0.000076,0,,position,15.762201,0.000000,1,787698045576173,787701673616956
43.6244,,0.691650,0.691650,0.690736,0.000000,0.000000,0.000000,-0.000000,0,1,0,1,15.918160,0,0,0,0,787701.684599,787698.065611,3618.987493,11436,1,0.691650,0.690736,0.690736,0.000000,0.000000,0.000000,0.000000,-0.000002,lsq,0.000000,0.000000,0.000000,0.690736,0,0.000000,0.000000,0,,position,5.914696,0.000000,1,787698045576173,787701684598654
```

### `run_20260819_204742.csv`  （7734 lines including header, 21147599 bytes）

```csv
t_wall_s,phase,controller_mode,t_ref_s,q_cmd_0,q_cmd_1,q_cmd_2,q_cmd_3,q_cmd_4,q_cmd_5,q_cmd_6,q_cmd_7,q_meas_0,q_meas_1,q_meas_2,q_meas_3,q_meas_4,q_meas_5,q_meas_6,q_meas_7,pose_x,pose_y,pose_z,pose_rx,pose_ry,pose_rz,twist_vx,twist_vy,twist_vz,twist_wx,twist_wy,twist_wz,twist_requested_vx,twist_requested_vy,twist_requested_vz,twist_requested_wx,twist_requested_wy,twist_requested_wz,twist_achieved_vx,twist_achieved_vy,twist_achieved_vz,twist_achieved_wx,twist_achieved_wy,twist_achieved_wz,track_err_mm,follow_err_deg,slack_norm,n_cbf,vel_clamped,acc_clamped,pos_clamped,fx,fy,fz,instability_idx,instability_idx_raw,instability_idx_active,damping_z_eff,damping_ke_z,damping_dimeas_z,v_force_z,ke_est,f_des_z_eff,v_r_z,force_reference_scale_n,force_reference_drive,force_reference_gate_scale,force_reference_accel_m_s2,force_reference_reversal_reset,force_reference_fast_clear,force_fast_z,retract_guard_armed,retract_fast_hold,retract_fast_stop_count,retract_fast_rearm_count,force_task_latched,physical_contact_state,physical_contact_acquire_event,physical_contact_loss_event,physical_contact_reacquire_event,physical_contact_low_timer_s,physical_contact_high_timer_s,mass_z_eff,takeover,dt_actual_s,deadline_slack_s,sensor_age_s,feedback_age_s,feedback_fresh_tick,fx_raw_comp,fy_raw_comp,fz_raw_comp,vz_achieved_tool,contact_present,force_pred_z,force_dot_z,cap_press_z,cap_retract_z,ke_update_gated,ke_dx_m,ke_df_n,ke_update_count,governor_scale,governor_scale_raw,sigma_min,qdot_norm,qdot_max_frac_vmax,qdot_ff_norm,tcp_jump_mm,rail_target_sent_m,rail_meas_m,rail_cmd_meas_err_m,rail_vel_pin,plan_drives_rail,rail_qdot_ff,pose_d_x,pose_d_y,pose_d_z,pose_d_rx,pose_d_ry,pose_d_rz,pose_meas_x,pose_meas_y,pose_meas_z,pose_meas_rx,pose_meas_ry,pose_meas_rz,motion_err_lin_x_mm,motion_err_lin_y_mm,motion_err_lin_z_mm,motion_err_rot_x_deg,motion_err_rot_y_deg,motion_err_rot_z_deg,motion_err_rms_mm,motion_axis_peak_mm,vel_ff_vx,vel_ff_vy,vel_ff_vz,vel_ff_wx,vel_ff_wy,vel_ff_wz,rail_contrib_m_s,arm_contrib_m_s,arm_y_qdot,rail_motion_share,rail_exec_for_qp,wln_scale_rail,wln_scale_arm_max,waste_ratio,rail_ff_m,rail_posture_err_m,rail_escape_active,psi_deg,psi_ref_deg,psi_retarget_score,d_pref_m,d_star_m,psi_star_deg,minmax_margin,elbow_margin_rad,wrist_open_rad,family_ok,tool_y_des_m,tool_y_err_mm,contact_phase,v_air_cmd,ke_hat,dob_v,barrier_cap_floor,flow_x_p,flow_v_p,flow_v_aux,flow_x_a,flow_v_a,flow_e,flow_edot,flow_F_c,flow_v_track,flow_P_e,flow_P_c,flow_alpha_target,flow_alpha,flow_alpha_case,flow_T,flow_psi,flow_S_n,flow_S_r_hat,flow_P_phys,flow_P_mismatch,flow_E_phys,flow_E_mismatch,flow_gamma_active,flow_alpha_would_gate,flow_edot_aligned,flow_sign_fault,flow_feedback_stale,flow_blocked_reason,contact_episode_rearm_event,contact_episode_release_s,surface_force_scale,surface_force_alpha,surface_xy_error_m,force_barrier_contact_active,qpik_backend,qpik_solver_status,qpik_solver_iterations,qpik_solver_solve_ms,qpik_solver_call_count,qpik_solver_overrun,qpik_qp1_status,qpik_qp2_status,qpik_qp1_solve_ms,qpik_qp2_solve_ms,qpik_assembly_ms,qpik_fallback_ms,qpik_total_ms,qpik_qp2_fallback,tick_inner_ms,tick_send_ms,tick_log_ms,qpik_alpha,qpik_beta,qpik_authority,qpik_equality_residual_max,qpik_hard_residual_max,qpik_anchor_valid,qpik_recovery_overflow,qpik_protected_nominal_overflow_json,qpik_recovery_caps_json,qpik_recovery_overflow_indices_json,qpik_hard_active_constraint_ids_json,qpik_protected_target_json,qpik_protected_achieved_json,qpik_protected_residual_json,qpik_scan_target_json,qpik_scan_achieved_json,qpik_scan_residual_json,qpik_working_slack_json,qpik_collision_slack_json,qpik_dexterity_slack,qpik_branch_slack,qpik_rail_macro_pref_v,qpik_rail_center_pref_v,qpik_rail_final_qdot,qpik_arm_risk_pref_norm,qpik_arm_risk_pref_json,qpik_risk_direction_cosine,qpik_path_velocity_xy_json,qpik_feedback_xy_raw_json,qpik_feedback_xy_filtered_json,qpik_rail_xy_contribution_json,qpik_arm_xy_contribution_json,qpik_rail_task_projection,qpik_rail_arm_cancel,qpik_rail_decomposition_error,qpik_arm_rho,qpik_joint_margin_rad,qpik_wrist_margin_rad,qpik_wrist_singularity,qpik_accepted_reference_lag_s,qpik_pre_solve_feedback_age_s,qpik_post_solve_feedback_age_s,qpik_q_cmd_q_meas_norm,qpik_fallback_level,qpik_fallback_reason,qpik_solver_fault_latched,qpik_final_sent_qdot_json,post_qp_step_clamp_enabled,post_step_would_clamp,post_step_clamp_applied,dt_nom_s,dt_int_s,box_h1_s,box_h2_s,qpik_qdot_raw_json,qpik_qdot_pre_commit_json,qpik_qdot_committed_json,qpik_qdot_prev_used_json,qpik_qdot_prev2_used_json,qpik_box_lo_json,qpik_box_hi_json,post_step_shadow_q_json,q_cmd_json,arm_send_mono_ns,rail_target_publish_mono_ns,rail_fa24_write_mono_ns,rail_encoder_sample_mono_ns,arm_qdot_target_wall_json,rail_sat,rail_exec_velocity_m_s,rail_measured_velocity_m_s,rail_commanded_velocity_m_s,rail_commanded_acceleration_m_s2,rail_feedback_age_s,a_mirror_frac,j_mirror_frac,last_limit_saturated,keep_task_weight,pref_slack_scale,rail_task_vel,v_escape,v_reach,v_ff_rail,u_alloc,u_posture,u_mid,v_r_ref,comp_projected_frac,rail_coast_active,rail_feedback_reject_streak_s,wall_override,slack_zero_feasible,sigma_arm,sns_scale,qpik_nullspace_norm,qpik_nullspace_centering_norm,qpik_nullspace_manip_norm,qpik_nullspace_arm_angle_norm,qpik_nullspace_damping_norm,qpik_nullspace_rail_lock_norm,cbf_min_dist,cbf_pair,qdot_meas_0,qdot_meas_1,qdot_meas_2,qdot_meas_3,qdot_meas_4,qdot_meas_5,qdot_meas_6,qdot_meas_7,v_cmd_vx,v_cmd_vy,v_cmd_vz,v_cmd_wx,v_cmd_wy,v_cmd_wz,path_twist_vx,path_twist_vy,path_twist_vz,path_twist_wx,path_twist_wy,path_twist_wz,feedback_twist_vx,feedback_twist_vy,feedback_twist_vz,feedback_twist_wx,feedback_twist_wy,feedback_twist_wz,comfort_slack_j1,comfort_slack_j2,comfort_slack_j3,comfort_slack_j4,comfort_slack_j5,comfort_slack_j6,comfort_slack_j7,pad_connected,pad_lx,pad_ly,pad_lt,pad_rx,pad_ry,pad_rt,pad_lb,pad_rb,pad_vx,pad_vy,pad_vz,pad_wx,pad_wy,pad_wz,pad_vcmd_base_vx,pad_vcmd_base_vy,pad_vcmd_base_vz,pad_vcmd_base_wx,pad_vcmd_base_wy,pad_vcmd_base_wz
0.0012,ellipse_track,qpik,0.0050,0.690736,-1.816678,-1.322052,1.014123,1.510530,1.578441,1.505504,1.228031,0.690736,-1.816678,-1.322052,1.014124,1.510530,1.578441,1.505504,1.228031,0.391502,0.490519,0.182710,-3.135117,-0.203213,-3.051213,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.000,0.0000,0.00000,0,0,0,0,0.000,0.000,0.000,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,nan,0,0,0,0,0,,0,0,0,nan,nan,nan,0,0.005000,-0.000506,0.003708,0.003709,0,nan,nan,nan,0.000000,0,nan,nan,nan,nan,0,nan,nan,0,1.0000,1.0000,0.16577,0.00004,0.0000,0.00000,0.000,0.690736,0.690736,0.000000,,0,0.000000,0.391502,0.490519,0.182710,-3.135117,-0.203213,-3.051213,0.391502,0.490519,0.182710,-3.135117,-0.203213,-3.051213,0.000,0.000,0.000,0.0000,0.0000,0.0000,0.000,0.000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000000,0.000000,0.000000,0.6235,-0.000000,1.0000,1.0000,4.0475,0.690636,-0.000100,0,72.9548,72.8296,,-0.200118,-0.200118,68.0000,,0.845470,1.505504,1,0.451445,0.000,,,,,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,1,1,,0,nan,nan,nan,nan,0,proxqpwbc,solved,0,2.049714,1,0,solved,solved,0.073081,0.022992,1.953641,0.000000,2.049714,0,4.9778,0.1154,,0.99999968,1.00000000,1.00000000,2.357282407e-07,0.000000000e+00,1,0,,,,,,,,,,,,,2.000192508e-12,2.000192508e-12,0.00000000,0.00000000,0.00000000,0.00000000,,nan,,,,,,nan,nan,0.000000000e+00,0.16576804,nan,nan,nan,0.000000,0.003708,0.003278,0.00000018,none,,0,"[0.0,-8.460946387955914e-06,-1.38148241735081e-05,-1.8577219940141276e-05,4.820931209792434e-07,1.7478609359855568e-05,1.95085959386887e-05,-2.0150090485060446e-07]",1,0,0,5.000000000e-03,5.000000000e-03,5.000000000e-03,,,,,,,,,,"[0.690736312866211,-1.8166783093794834,-1.3220520900495454,1.0141234609957221,1.510530059001305,1.5784410180970159,1.5055037038924035,1.2280311150534902]",787701702626587,787701702616005,787698045576173,787701684598654,,0,-0.00000006,-0.00000159,0.00000000,0.00000000,0.012706,0.000000,0.000000,0,0,1.0000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000120,0.000000,0.000000,0,0.000000,0,0,0.12180,1.0000,0.004485,0.012349,0.000000,0.004225,0.000000,0.000000,0.040230,link_6_0:link_8_0,,,,,,,,,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,2.000192508e-12,2.000192508e-12,2.000192508e-12,2.000192508e-12,2.000192508e-12,2.000192508e-12,2.000192508e-12,,,,,,,,,,,,,,,,,,,,,
0.0068,ellipse_track,qpik,0.0107,0.690736,-1.816679,-1.322052,1.014123,1.510531,1.578441,1.505504,1.228030,0.690736,-1.816678,-1.322052,1.014124,1.510513,1.578458,1.505521,1.228014,0.391500,0.490528,0.182709,-3.135115,-0.203222,-3.051172,0.00002,-0.00009,0.00001,0.00001,-0.00002,-0.00008,0.00002,-0.00009,0.00001,0.00001,-0.00002,-0.00008,0.00000,-0.00000,0.00000,0.00000,0.00000,0.00000,0.010,0.0010,0.00000,0,0,0,0,0.000,0.000,0.000,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,nan,0,0,0,0,0,,0,0,0,nan,nan,nan,0,0.005660,-0.002203,0.003799,0.003800,1,nan,nan,nan,0.000000,0,nan,nan,nan,nan,0,nan,nan,0,1.0000,1.0000,0.16577,0.00039,0.0001,0.00000,0.001,0.690736,0.690736,0.000000,,0,-0.000000,0.391502,0.490519,0.182710,-3.135117,-0.203213,-3.051213,0.391500,0.490528,0.182709,-3.135115,-0.203222,-3.051172,-0.001,-0.010,-0.001,-0.0006,-0.0005,0.0023,0.006,0.010,-0.000000,-0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000097,0.000097,0.0005,-0.000000,1.0000,1.0000,1.0000,0.690523,-0.000213,0,72.9548,72.6881,,-0.200004,-0.200004,68.0000,,0.845470,1.505504,1,0.451427,-0.010,,,,,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,1,1,,0,nan,nan,nan,nan,0,proxqpwbc,solved,0,2.251512,2,0,solved,solved,0.037833,0.030235,2.183444,0.000000,2.251512,0,5.6226,0.1279,0.1844,1.00000000,1.00000000,1.00000000,2.208385208e-11,0.000000000e+00,1,0,,,,,,,,,,,,,2.000194680e-12,2.000194680e-12,0.00000000,0.00000000,0.00000000,0.00000000,,nan,,,,,,nan,nan,0.000000000e+00,0.16576869,nan,nan,nan,0.000000,0.003799,0.000724,0.00003433,none,,0,"[0.0,-0.00016745381481937373,2.3136237241183607e-05,-4.8054396435043245e-05,0.00019440586878763627,5.6673534863486005e-05,5.261446766748607e-05,-0.0002746494322057069]",1,0,0,5.000000000e-03,5.660100956e-03,6.768218009e-03,5.000000000e-03,,,,,,,,,"[0.690736312866211,-1.8166792571849808,-1.3220519590961068,1.014123189002987,1.510531159358149,1.5784413388749448,1.5055040016956023,1.2280295605099762]",787701709311052,787701709301154,787698045576173,787701702679926,,0,-0.00000005,-0.00000159,0.00000000,0.00000000,0.018373,0.000003,0.000005,0,0,1.0000,0.000000,0.000000,0.000000,-0.000000,-0.000007,0.000000,-0.000257,0.000000,0.000000,0,0.000000,0,0,0.12179,1.0000,0.009538,0.012361,0.000000,0.009003,0.000019,0.000000,0.040230,link_6_0:link_8_0,-0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000022,-0.000094,0.000008,0.000007,-0.000019,-0.000082,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000012,-0.000096,-0.000010,-0.000021,-0.000019,0.000080,2.000194680e-12,2.000194680e-12,2.000194680e-12,2.000194680e-12,2.000194680e-12,2.000194680e-12,2.000194680e-12,,,,,,,,,,,,,,,,,,,,,
0.0135,ellipse_track,qpik,0.0173,0.690736,-1.816680,-1.322055,1.014124,1.510536,1.578444,1.505503,1.228026,0.690736,-1.816661,-1.322000,1.014124,1.510513,1.578441,1.505504,1.228014,0.391500,0.490532,0.182726,-3.135147,-0.203195,-3.051161,0.00002,-0.00013,-0.00016,-0.00006,0.00003,-0.00009,0.00002,-0.00013,-0.00016,-0.00006,0.00003,-0.00009,-0.00000,0.00003,0.00000,0.00129,0.00013,-0.00164,0.021,0.0030,0.00000,0,0,0,0,0.000,0.000,0.000,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,nan,0,0,0,0,0,,0,0,0,nan,nan,nan,0,0.006671,-0.002751,0.001227,0.001228,1,nan,nan,nan,-0.000001,0,nan,nan,nan,nan,0,nan,nan,0,1.0000,1.0000,0.16577,0.00110,0.0003,0.00000,0.001,0.690736,0.690736,0.000000,,0,-0.000000,0.391502,0.490519,0.182710,-3.135117,-0.203213,-3.051213,0.391500,0.490532,0.182726,-3.135147,-0.203195,-3.051161,-0.004,-0.013,0.016,0.0011,0.0010,0.0029,0.012,0.016,-0.000000,-0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000207,0.000207,0.0000,-0.000000,1.0000,1.0000,1.0000,0.690398,-0.000338,0,72.9548,72.5319,,-0.199879,-0.199879,68.0000,,0.845469,1.505504,1,0.451431,-0.013,,,,,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,1,1,,0,nan,nan,nan,nan,0,proxqpwbc,solved,0,2.118415,3,0,solved,solved,0.032362,0.025965,2.060088,0.000000,2.118415,0,4.6171,0.1025,0.1698,1.00000000,1.00000000,1.00000000,4.730107337e-11,0.000000000e+00,1,0,,,,,,,,,,,,,2.000198144e-12,2.000198144e-12,0.00000000,0.00000000,0.00000000,0.00000000,,nan,,,,,,nan,nan,0.000000000e+00,0.16576566,nan,nan,nan,0.000000,0.001227,0.000382,0.00006387,none,,0,"[0.0,-0.00011933611460790416,-0.0004256470144881064,0.00012272533467694302,0.0007356969018346149,0.00034839093331129334,-0.0001161780248537525,-0.0005680047618028539]",1,0,0,5.000000000e-03,6.250000000e-03,5.743670976e-03,6.768218009e-03,,,,,,,,,"[0.690736312866211,-1.816680003035697,-1.3220546193899474,1.0141239560363287,1.5105357574637854,1.578443516318278,1.505503275582947,1.228026010480215]",787701714885566,787701714876929,787698045576173,787701702679926,,0,-0.00000001,0.00000045,0.00000000,0.00000000,0.006971,0.000010,0.000012,0,0,1.0000,0.000000,0.000000,0.000000,-0.000000,-0.000016,0.000000,-0.000409,0.000000,0.000000,0,0.000000,0,0,0.12179,1.0000,0.015165,0.012383,0.000000,0.014278,0.000209,0.000000,0.040230,link_6_0:link_8_0,-0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.002094,0.000020,-0.000131,-0.000160,-0.000060,0.000030,-0.000091,0.000000,0.000000,-0.000000,0.000000,0.000000,0.000000,-0.000040,-0.000131,0.000155,0.000038,0.000036,0.000100,2.000198144e-12,2.000198144e-12,2.000198144e-12,2.000198144e-12,2.000198144e-12,2.000198144e-12,2.000198144e-12,,,,,,,,,,,,,,,,,,,,,
...
38.9914,ellipse_track,qpik,38.9952,0.754987,-1.985034,-1.369464,0.349896,0.349953,2.233056,1.599300,2.353501,0.754985,-1.984439,-1.369839,0.349816,0.349886,2.233236,1.599245,2.354798,0.362601,0.766445,0.170357,3.120085,-0.177598,-3.026608,0.07644,0.11949,0.04877,-0.05992,0.04532,-0.03861,0.07644,0.11949,0.04877,-0.05992,0.04532,-0.03861,0.01306,-0.00662,-0.00040,-0.00272,-0.00347,-0.00226,25.126,0.0682,0.17136,1,0,0,0,0.000,0.000,0.000,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,nan,0,0,0,0,0,,0,0,0,nan,nan,nan,0,0.005000,0.000244,0.004190,0.004190,1,nan,nan,nan,-0.001941,0,nan,nan,nan,nan,0,nan,nan,0,1.0000,1.0000,0.24977,0.04562,0.0128,0.00000,0.071,0.754987,0.754973,0.000001,,0,-0.005891,0.373295,0.787492,0.178957,-3.135117,-0.203213,-3.051213,0.362601,0.766445,0.170357,3.120085,-0.177598,-3.026608,-11.314,19.909,-10.342,1.3363,1.4571,1.3953,14.506,19.909,0.012768,-0.005891,-0.002464,0.000000,0.000000,0.000000,-0.000093,0.014198,0.014198,0.0065,0.000225,1.0000,1.0000,1.0132,0.413398,-0.341587,0,20.9417,68.0000,,0.374094,0.374094,68.0000,,2.006047,1.599390,1,0.744833,19.909,,,,,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,1,1,,0,nan,nan,nan,nan,0,proxqpwbc,solved,0,2.053500,7732,0,solved,solved,0.041785,0.033164,1.978551,0.000000,2.053500,0,4.0092,0.1741,0.1718,0.82863953,1.00000000,1.00000000,1.249650362e-01,1.817916675e-10,1,0,,,,,,,,,,,,,2.276082841e-01,3.061857679e-13,0.00024206,0.00000000,0.00024205,0.00000000,,nan,,,,,,nan,nan,0.000000000e+00,0.24976790,nan,nan,nan,0.000000,0.004190,0.003354,0.00149155,none,,0,"[0.0002420513224743459,-0.03214931919597319,-0.005262753263757247,-1.8178791805212313e-10,1.9617587110332124e-06,0.0153983762329446,-0.017865608069023153,-0.021520491483606463]",1,0,0,5.000000000e-03,5.000000000e-03,5.087825004e-03,4.902414046e-03,,,,,,,,,"[0.7549866377440314,-1.985034307167984,-1.3694636783687892,0.34989552837615345,0.3499526151222068,2.2330556589088792,1.5993002950448791,2.3535007100936536]",787740692052017,787740692043276,787740334738630,787740672783050,,0,0.00022531,0.00000281,0.00000000,0.00000000,0.014712,0.013253,0.024450,0,0,1.0000,0.000242,0.000000,0.000000,-0.005891,0.193717,0.000000,-0.119774,0.000242,0.000000,0,0.000000,1,0,0.03155,1.0000,0.197732,1.320118,0.000000,0.734328,0.040021,0.000000,0.041186,link_6_0:link_8_0,0.000225,-0.031416,-0.002094,0.000000,0.000000,0.010472,-0.014661,-0.023038,0.076439,0.119494,0.048766,-0.059917,0.045325,-0.038611,-0.012254,-0.007322,0.000147,0.000000,0.000000,0.000000,-0.067542,0.118853,-0.061741,0.046646,0.050864,0.048705,1.523053390e-12,1.523053390e-12,1.523053390e-12,1.523053390e-12,1.523053390e-12,1.523053390e-12,1.523053390e-12,,,,,,,,,,,,,,,,,,,,,
38.9964,ellipse_track,qpik,39.0002,0.754984,-1.985193,-1.369490,0.349896,0.349953,2.233132,1.599212,2.353394,0.754982,-1.984649,-1.369857,0.349869,0.349904,2.233306,1.599088,2.354659,0.362690,0.766414,0.170356,3.120027,-0.177572,-3.026594,0.07635,0.11956,0.04874,-0.06004,0.04536,-0.03862,0.07635,0.11956,0.04874,-0.06004,0.04536,-0.03862,0.01246,-0.00669,0.00019,-0.00644,-0.00221,0.00434,25.112,0.0663,0.17143,1,0,0,0,0.000,0.000,0.000,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0,0,nan,0,0,0,0,0,,0,0,0,nan,nan,nan,0,0.005000,0.000689,0.003962,0.003963,1,nan,nan,nan,-0.002415,0,nan,nan,nan,nan,0,nan,nan,0,1.0000,1.0000,0.24974,0.04510,0.0126,0.00000,0.071,0.754984,0.754973,0.000001,,0,-0.005939,0.373359,0.787463,0.178945,-3.135117,-0.203213,-3.051213,0.362690,0.766414,0.170356,3.120027,-0.177572,-3.026594,-11.291,19.913,-10.325,1.3395,1.4585,1.3962,14.499,19.913,0.012769,-0.005939,-0.002463,0.000000,0.000000,0.000000,-0.000081,0.014050,0.014050,0.0057,0.000195,1.0000,1.0000,1.0116,0.413369,-0.341614,0,20.9416,68.0000,,0.374094,0.374094,68.0000,,2.006047,1.599300,1,0.744804,19.913,,,,,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,1,1,,0,nan,nan,nan,nan,0,proxqpwbc,solved,0,2.045887,7733,0,solved,solved,0.042888,0.033946,1.969053,0.000000,2.045887,0,3.5013,0.1071,0.1807,0.82857180,1.00000000,1.00000000,1.250012901e-01,1.943151613e-10,1,0,,,,,,,,,,,,,2.275953161e-01,2.985425351e-13,0.00029514,0.00000000,0.00029512,0.00000000,,nan,,,,,,nan,nan,0.000000000e+00,0.24973596,nan,nan,nan,0.000000,0.003962,0.003322,0.00144133,none,,0,"[0.0002951244161170462,-0.03176781252074692,-0.005346945368735726,-1.9431010255527336e-10,1.957155812110261e-06,0.01518747058379294,-0.017704968577272707,-0.0212556221367658]",1,0,0,5.000000000e-03,5.000029108e-03,5.326585029e-03,5.087825004e-03,,,,,,,,,"[0.754983694839827,-1.9851931471553006,-1.3694904132512744,0.3498955283751819,0.34995262490804285,2.2331315967038825,1.599211769686628,2.3533944313642507]",787740696673612,787740696665117,787740334738630,787740689221286,,0,0.00019524,-0.00000135,0.00000000,0.00000000,0.003274,0.039257,0.098576,0,0,1.0000,0.000295,0.000000,0.000000,-0.005939,0.193782,0.000000,-0.119774,0.000295,0.000000,0,0.000000,1,0,0.03155,1.0000,0.197724,1.320036,0.000000,0.734330,0.039725,0.000000,0.041186,link_6_0:link_8_0,0.000195,-0.031416,0.000000,0.000000,0.000000,0.012566,-0.012566,-0.031416,0.076348,0.119562,0.048744,-0.060038,0.045362,-0.038617,-0.012250,-0.007370,0.000145,0.000000,0.000000,0.000000,-0.067445,0.118943,-0.061674,0.046759,0.050913,0.048735,1.484836609e-12,1.484836609e-12,1.484836609e-12,1.484836609e-12,1.484836609e-12,1.484836609e-12,1.484836609e-12,,,,,,,,,,,,,,,,,,,,,
```

### `rail_20260819_204742.csv`  （123747 lines including header, 43224796 bytes；21:22 冻结）

```csv
t_wall_s,event,target_m,commanded_m,measured_m,v_ff,v_des,v_cmd,rpm,follow,armed,panic,poll_ok,dt_wall_ms,last_rpm_cmd,mb_fail_n,freeze_flag,arm_good,sample_mono_s,target_rx_mono_s,target_age_ms,motion_seq,feedback_valid,x_goal_m,x_ref_m,x_meas_m,v_goal_est_m_s,v_ref_m_s,a_ref_m_s2,v_reg_m_s,v_enc_m_s,v_enc_source,v_des_m_s,v_cmd_m_s,a_cmd_m_s2,x_goal_eval_m,rpm_cmd,e_track_mm,e_shape_mm,hold_count,hold_reason,command_mode,t_read_ms,t_write_ms,n_modbus,fa24_write_mono_ns,encoder_sample_mono_ns
0.0003,session_begin,0.690736,0.690736,0.690736,,,,,0,1,0,1,,0,0,0,0,,,,0,0,,,,,,,,,,,,,,0,,,0,,position,,,0,,
0.0103,,0.690736,0.690736,0.690736,0.000000,0.000000,0.000000,-0.000000,1,1,0,1,17.651090,0,0,0,0,787701.702680,787701.702623,0.057255,11437,1,0.690736,0.690736,0.690736,0.000000,0.000000,0.000000,0.000000,0.000000,lsq,0.000000,0.000000,0.000000,0.690736,0,0.000000,0.000000,0,,coupled_velocity,6.766025,0.000000,1,787698045576173,787701702679926
0.0279,,0.690736,0.690736,0.690736,0.000000,0.000000,0.000000,-0.000000,1,1,0,1,18.289945,0,0,0,0,787701.720595,787701.720528,0.066507,11438,1,0.690736,0.690736,0.690736,-0.000000,0.000000,0.000000,0.000000,0.000001,lsq,0.000000,0.000000,0.000000,0.690736,0,0.000000,-0.000001,0,,coupled_velocity,6.017280,0.000000,1,787698045576173,787701720594696
...
1832.2236,,0.754973,0.754973,0.754982,0.000000,0.000000,0.000000,-0.000000,0,1,0,1,16.744502,0,0,0,0,789533.916619,787740.696670,1793219.949048,120903,1,0.754973,0.754982,0.754982,0.000000,0.000000,0.000000,0.000000,-0.000004,lsq,0.000000,0.000000,0.000000,0.754983,0,0.000000,0.000992,0,,position,5.339221,0.000000,1,787740334738630,789533916619112
1832.2406,,0.754973,0.754973,0.754983,0.000000,0.000000,0.000000,-0.000000,0,1,0,1,16.694219,0,0,0,0,789533.933496,787740.696670,1793236.825743,120904,1,0.754973,0.754983,0.754983,0.000000,0.000000,0.000000,0.000000,0.000006,lsq,0.000000,0.000000,0.000000,0.754982,0,0.000000,-0.000992,0,,position,5.686710,0.000000,1,787740334738630,789533933495807
```

