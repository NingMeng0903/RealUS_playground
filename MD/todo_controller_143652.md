# run_20260825_143652：escape 修好了，但扭转后 TCP 仍大偏、联动变差

快照日期：2026-08-25 14:36。生产路径 `inner.backend=native`，`wbc_rt pid=158528`。这是 **escape 自锁修复之后** 的硬件复测。

上一份 122744 分析已挪到 `MD/todo_controller_122744.md`。本文件只讲 143652（以及同午的 143248）。

| 文件 | 内容 |
|---|---|
| `MD/todo_controller.md` | 本分析 + 要害源码 |
| `MD/todo_controller_122744.md` | 上一轮（σ-escape 自锁）全文 |
| `MD/todo_controller_logs/run_20260825_143652.csv` | 本 run 原始 log 全文（23 MB / 8501 拍 / 454 列） |
| `MD/todo_controller_logs/run_20260825_143248.csv` | 同午前一次 93 s log（J1 已经打到 −140°） |
| `MD/todo_controller_logs/run_20260825_143652_t0-10_keycols.csv` | t=0–10 s 要害列，每拍 |
| `MD/todo_controller_logs/run_20260825_143652_stickup_t65-95_keycols.csv` | t=6.5–9.5 松杆后 TCP 仍在飞 |
| `MD/todo_controller_logs/run_20260825_143652_yawdrift_t30-40_keycols.csv` | t=30–40 无平移命令仍偏 TCP |
| `MD/todo_controller_logs/code_143652/` | 当时控制器源码副本 |

不要把 `v_cmd_*` 当成「手柄没动」。native 路径这列**永远是 0**。手柄在 `twist_requested_*`。

---

## 0. 结论（先看这个）

三件事叠在一起，不是同一条 bug。

### 0.1 「vcmd 都没了」——日志列，不是手柄死了

`v_cmd_vx…wz` 全程精确 0，和 122744、143248 **完全一样**。native `JointIkStep` 构造时没填 `v_cmd`，dataclass 默认 `zeros(6)`；CSV 后段 `getattr(step, "v_cmd", step.twist_base)` 命中默认零向量，不会回退到 `twist_base`。

本 run 手柄是活的：

| 量 | 值 |
|---|---|
| `twist_requested` 线速度 p95 / max | 74 / 86 mm/s |
| `twist_requested` 角速度 p95 / max | 0.40 / 0.61 rad/s（≈23 / 35 °/s） |
| 纯平移拍 | **9.3%** |
| 显式旋转拍 `‖ω‖>0.03` | **69.0%** |
| 空闲 | 21.7% |

主轴是 **tool-Z 偏航 `wz`**（p50=0.11，p95=0.40，max=0.61）。你说的「扭转角度」和数据一致。122744 才是「大部分在给平移」（78.7%）。

`pad_*` 全 NaN：Window C 的摇杆原始轴没有写进这份 inner CSV。不影响 `twist_requested`。

### 0.2 「TCP 大偏移」——前 3.5 s 是你推的 +X；松杆后的飞，不是 escape

净位移 **+295 mm X、+40 mm Y、+88 mm Z**。拆开：

1. **t=1.0–3.3 s**：左摇杆 +X ≈ 86 mm/s，TCP X 0.046→0.239 m（**+193 mm**）。这是命令，不是漂。
2. **t=4.0–6.5 s**：平移松开，右摇杆满偏航 `wz≈−0.60`。TCP XYZ 几乎钉住（好），但 **J1 −55°→−104°、J4 96°→33°、σ 0.13→0.05**。偏航跟踪本身很好（req −0.60 / ach −0.59）。
3. **t=6.5–9.5 s：摇杆已经是 0**，TCP 仍以线速度 p95 **88 mm/s**、max **112 mm/s** 在飞，X 再偏 **+41 mm**。`u_escape_raw=0`，`escape_active=0`。驱动是 QP2 对中 / homotopy / `u_post`（p95 57 mm/s），J4 顶在 20°、`j4_design_slack≈3.3`。
4. **t=10–39 s**：线速度命令 p95=0，但 `wz` 一直约 **0.12 rad/s（7 °/s）** 直到 t=39。J1 从 −83° 收到 **−139.6°**。TCP 再净移 **+78 / +79 / +93 mm**。`u_task≈0`，`u_post` 2–4 cm/s 在拖轨。

所以：不是 122744 那种 ±80 mm/s bang-bang escape。Escape **修好了**。松杆后的 TCP 大偏是 **零空间对中 + d\* posture 在奇异附近抢主任务**。自身 TCP 速度又让 `quiescent` 永远进不去（本 run 0 拍），停不下来。

### 0.3 「联动变差」——你从「沿轨平移」改成了「原地扭转」；rail share 必然掉

| run | 纯平移 | 旋转 | 平移时 `rail_motion_share` p50 | 平移时 `req_vy` p95 | escape |
|---|---|---|---|---|---|
| 122744（修前） | 78.7% | 10.4% | **0.63** | **105 mm/s（沿轨 Y）** | 8.7%，绑死 σ |
| 143248（同午前一段） | 14.4% | 61.3% | 0.003 | 119 mm/s | **0** |
| 143652（本 run） | 9.3% | **69.0%** | **0.00** | **2 mm/s** | **0** |

122744 的「联动好」很大一块是：你在推 **Y（轨道方向）**，rail 理应承担。本 run 平移几乎全是 **+X**，轨道是 Y 轴，share 掉到 0 是几何，不是 escape 修坏了分配器。

扭转时轨道仍在动（t=4–6 rail 0.41→0.48 m），那是构型折叠带着滑台，看起来像「手在转、轨在自己走」——联动变差的体感主要来自这里。

纯平移段姿态泄漏：本 run `‖ω_ach‖` p95 **2.6 °/s**，122744 是 **5.7 °/s**。各向异性阻尼没有把平移时的角泄漏搞得更差；变差的是 **扭转工况 + J1 收到 −140°**。

### 0.4 Escape 验收（这次过了）

- 全程 `contact_present=0`、`overforce_escape=0`
- `sigma_arm<0.08`：**802 拍**（t=5.485–9.490，min **0.030**）
- `escape_active`：**0 拍**，`u_escape_raw`：**0**
- 122744 里这两件事 100% 重合；现在低 σ 不再开轨

`healthy_sigma_mute` 不再驱动轨道。奇异恢复只剩 QP2 σ preference + d\* owner——而 **d\* owner 在 J4 顶死时会拖着 TCP 走**，这就是下一轮该管的，不是把 escape 加回去。

---

## 1. 这次 log

| 项 | 值 |
|---|---|
| 源文件 | `rm75_control/apps/logs/peirastic/run_20260825_143652.csv` |
| 行数 / 列数 | 8501 拍，454 列，约 23 MB |
| 时长 / 周期 | t_ref = 0.005 … 41.375 s，中位 dt = 5.0 ms |
| 后端 / 相位 / 模式 | `qpik_backend=native`，`phase=servo_twist`，`controller_mode=qpik` |
| 启动 | Window A `python -m peirastic.apps.run_controller --log-csv`；rail hold @ 0.588 m 后实际从 **0.421 m** 开始 |
| 工具 | `tcp sync: cached tool='gripper2' xyz(mm)=[0.0, -15.2, 121.4] rpy(deg)=[1.0, 49.9, -90.2]`（法兰→TCP 约 12 cm） |
| 接触 / 过力 | 全程 0 |
| `command_stale` | 全程 0 |

同午 `run_20260825_143248.csv`（93 s）已经是 61% 旋转、J1 到 −140°、escape=0。本 run 是同一操作习惯的短复测，不是新故障模式。

---

## 2. 怎么读这几列

| 列 | 含义 |
|---|---|
| `twist_requested_*` | 真正进 QP 的 TCP twist（native `v_cmd_received`）。**用这个看手柄** |
| `v_cmd_*` | native CSV **恒 0**，不要用 |
| `twist_achieved_*` | `J(q) q̇` 实际 TCP |
| `pose_*` | 当前 TCP。`SERVO_TWIST` 下 `pose_d` 每拍跟着当前 pose 走，所以 `pose−pose_d=0`、`track_err_mm=0`，**不能**用来判断「有没有跟丢目标」 |
| `q_meas_0` / `rail_meas_m` | 轨道，米 |
| `q_meas_1` | J1。yaml 照片 −89.5°；本 run 收到 **−139.6°** |
| `q_meas_4` | J4。70–115° 设计带；约 20° 硬附近 |
| `sigma_arm` | 手臂最小奇异值。修后不再开 escape |
| `escape_active` / `u_escape_raw` | mixer 软逃离。本 run 全 0 |
| `u_task_raw` / `u_post_*` / `u_feasible` | TCP 分给轨的份额 / d\* posture / 合成。松杆后 task≈0，posture 仍可非零 |
| `qpik_nullspace_centering_norm` | QP2 对中。松杆后 p95≈2，这是 TCP 乱飞的主嫌疑 |
| `homotopy_s` | 0→1 把构型拉向照片。本 run 1 s 内就到 1，t=7 一度掉回 0 |
| `j4_design_slack` | 0=带内；>3=带外且 QP 已放弃 |
| `quiescent` | 本 run 0 拍。TCP 自己在动，quiet-hold 进不去 |

---

## 3. 全程统计

| 量 | 值 |
|---|---|
| 轨道 | 0.248 … 0.483 m（没撞 122744 那种 0.03/0.76 墙） |
| TCP 包络 | X 跨度 **309 mm**，Y 86 mm，Z 98 mm |
| TCP 路径长度 | **1115 mm**（41 s 里画了一米多） |
| 其中 `‖v_req‖<0.02` 时的路径 | **824 mm**（松杆/无平移时仍在走） |
| J1 | −19.1° → **−139.6°**（min −139.6°） |
| J4 | 20.1° … 116.5°；落在 70–115° 仅 **56.6%**（122744 是 74.9%） |
| J5 | 28.6° → **177.6°** |
| J7 | 128.2° → 20.0° |
| `sigma_arm` | min 0.030，p50 0.119，max 0.143 |
| `escape_active=1` | **0** |
| `quiescent` | **0** |
| \|e_d\| | p50 1.5 mm，p95 20 mm，max 60 mm（比 122744 的 175 mm p95 好，因为没 escape 狂奔） |
| `rail_commanded_velocity` | p95 38 mm/s，max 60 mm/s（不再是 80 mm/s 方波） |
| homotopy | 0.013 → 1.0（t≈1 s）；t=7 掉到 0；t=10–38 再次钉在 1.0；文件结束约 0.17 |
| ψ / ψ\* | 58.6° → 65.1°；ψ\* 钉 68° |

---

## 4. 时间线（要害）

### 4.1 t=0–3.5 s：你在推 +X，构型已经被 homotopy 拉开

| t (s) | pose xyz (m) | rail | J1 | J4 | req v xyz | req w | hom | σ |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.046, 0.229, 0.275 | 0.421 | −19.1 | 84.9 | 0 | 0 | 0.013 | 0.122 |
| 1.00 | 0.052, 0.228, 0.274 | 0.415 | −20.3 | 84.8 | **+X 52 mm/s** | 0 | **1.00** | 0.122 |
| 2.00 | 0.126, 0.227, 0.274 | 0.412 | −25.8 | 93.7 | +X 86 | 0 | 1.00 | 0.129 |
| 3.00 | 0.213, 0.224, 0.274 | 0.410 | −41.3 | 97.3 | +X 86 | 0 | 1.00 | 0.130 |
| 3.50 | 0.239, 0.223, 0.274 | 0.409 | −49.4 | 96.7 | 0 | 0 | 1.00 | 0.129 |

Homotopy 在 **第一秒** 就到 1。你还在走直线，J1 已经从 −19° 折向照片。轨几乎不动（命令是 X 不是 Y）——这就是「联动没了」的第一印象：以前推 Y 轨会跟，现在推 X 轨不跟。

### 4.2 t=4.0–6.5 s：满偏航，TCP 钉住，膀子被拧塌

`wz_req` p50 = **−0.60**，`wz_ach` p50 = **−0.59**（跟踪好）。yaw 期间 TCP 线速度 p95 仅 24 mm/s。

同时：J1 −55°→−104°，J4 96°→33°，rail 0.409→0.481，σ 在 **t=5.485** 跌破 0.08。Escape 仍为 0。

工具长度 121 mm × 0.60 rad/s ≈ 73 mm/s 切向；QP 把 TCP 几乎钉在原地，切向只能进关节——所以构型塌、轨被带着走。

### 4.3 t=6.5–9.5 s：摇杆已零，TCP 仍以 ~9 cm/s 飞

`‖ω_req‖` max = 0.0001。`‖v_ach‖` p50/p95/max = **34 / 88 / 112 mm/s**。

- J4 到 **20°**，`j4_design_slack` p50=3.25、max=3.59
- `qpik_nullspace_centering_norm` p95=**1.96**
- `u_post` p95=57 mm/s，`u_task` p95=22 mm/s
- homotopy 1.0 → **0.0**（t≈7.0）→ 0.22
- `secondary_suppressed` 仅 14%；`quiescent=0`
- σ min **0.030**，全程无 escape

这是你说的「vcmd 没了 TCP 还大偏」的数据对应：命令确实没了，TCP 被二次任务推着走。

### 4.4 t=10–39 s：持续小偏航，J1 收到 −140°

`req_v` p95=0，`req_w` p50/p95 ≈ 0.116 / 0.124 rad/s。不是「完全松杆」，是拧完之后摇杆没有回中到死区以下。

t=32–38 TCP 再次明显爬：

| t | X | Y | Z | rail | J1 | J4 | `u_post` |
|---|---|---|---|---|---|---|---|
| 30 | 0.261 | 0.184 | 0.267 | 0.249 | −77 | 115 | +3 mm/s |
| 33 | 0.269→0.294 | 0.200→0.219 | 0.270→0.285 | 0.253→0.272 | −107→−114 | 103→96 | +24 |
| 35 | 0.320→0.352 | 0.232→0.239 | 0.301→0.321 | 0.294→0.327 | −123→−134 | 88→82 | +38 |
| 38 | 0.336→0.341 | 0.264→0.269 | 0.351→0.362 | 0.340→0.352 | −139→−140 | 62 | +12 |
| 39–41 | 钉住 | | | 0.352 | −139.6 | 61.8 | 0 |

t=39 以后 `req_w` 也到 0，TCP 才停。说明 **不是积分器发散**，是只要还有一丁点偏航命令（或二次任务），奇异构型就会继续被拉。

---

## 5. 和 122744 对比：什么好了、什么没好

| | 122744 | 143652 |
|---|---|---|
| 低 σ 开 escape | 是，满幅 ±80 mm/s，过中点翻转 | **否** |
| 松杆后轨狂奔 | 是 | **否**（轨 p95 38 mm/s） |
| J4 带内比例 | 74.9% | 56.6%（扭转把肘拧出带） |
| J1 | −23°→−93°（照片附近） | −19°→**−140°**（过折） |
| 操作 | 79% 沿轨平移 | 69% 偏航 |
| 平移时角泄漏 p95 | 5.7 °/s | 2.6 °/s（aniso 没有明显搞坏平移） |
| 松杆后 TCP 自己走 | escape 拖着走 | **对中 / d\*** 拖着走 |

各向异性任务阻尼的验收只能写成：native 现在真的在用 `aniso_task_damping`，退化方向先掉权。它 **不保证** 扭转时 TCP 不漂；本 run 也没保证。

---

## 6. 「v_cmd 恒 0」的代码原因

CSV 同时写两遍 `twist_base` 当 `twist_*` 和 `twist_requested_*`，后面才写 `step.v_cmd`：

```3773:3775:rm75_control/rm75_control/control/joint_admittance_8dof/loop.py
            + [f"{v:.5f}" for v in step.twist_base]
            + [f"{v:.5f}" for v in step.twist_base]
            + [f"{v:.5f}" for v in twist_achieved]
```

```4219:4219:rm75_control/rm75_control/control/joint_admittance_8dof/loop.py
               *_fmt6(getattr(step, "v_cmd", step.twist_base)),
```

`JointIkStep.v_cmd` 默认全 0。native client 只设了 `twist_base=v_recv`，**没设 `v_cmd=`**：

```389:392:rm75_control/rm75_control/control/joint_admittance_8dof/wbc_rt/client.py
        step = JointIkStep(
            q_send=q_cmd,
            qdot=qdot,
            twist_base=v_recv,
```

所以 native 所有历史 CSV 的 `v_cmd_*` 都是假零。这不是这次修 escape 引入的，也不是 Window C 没开（有 `twist_requested` 就说明 TwistBus 有数）。

手柄进程是 `python -m peirastic.apps.gamepad`（Window C）。只开 Window A 时 `twist_requested` 应为 0；本 run 不是那种情况。

---

## 7. Escape 修复（本 run 已生效）

Native 只在 `allow_press` 时写 `u_escape`；本 run 无接触，永远不授权：

```929:946:rm75_control/native/wbc_rt/src/inner.cpp
    if (allow_press) {
      if (!escape_active_ || std::abs(escape_sign_) < 1.0e-12) {
        escape_sign_ = policy_escape_sign(cfg_.escape_sign_policy, y, soft.first, soft.second, 0.0);
      }
      last_v_escape_ = clip(0.25 * cfg_.k_esc * escape_sign_, -cfg_.v_max_ext, cfg_.v_max_ext);
      escape_active_ = std::abs(last_v_escape_) > 1e-12;
    } else {
      escape_active_ = false;
      last_v_escape_ = 0.0;
      escape_sign_ = 0.0;
    }
```

Python fallback 同样只吃 `press_escape_allowed`，低 σ / backoff 不能自行授权。全文在 `MD/todo_controller_logs/code_143652/`。

---

## 8. 各向异性阻尼（本 run 在用，不是 TCP 漂移主因）

QP1 现在走 `TaskWeightState::step(J_task, …)`。σ 掉到 0.03 时退化方向的任务权被压到 `min_frac=0.05`。扭转进奇异时 **主任务变软、对中变相对硬**，正好放大第 4.3 节的松杆漂移。这是副作用，不是「aniso 算错」。

完整实现：`MD/todo_controller_logs/code_143652/task_weight.hpp`。

---

## 9. 下一轮不该做什么 / 该做什么

**不要** 把 σ-escape 加回去。本 run 已经证明低 σ 不再开轨。

**不要** 指望静止自动把 J4 拉回 96°。计划里写过：松杆后停轨，不回构型。本 run 连「停」都没做到，因为二次任务还在动 TCP。

该开的下一轮（按体感优先级）：

1. **奇异附近二次任务限幅**：`centering_norm` 在 `sigma_arm<0.08` 或 `j4_design_slack>1` 时压到接近 0；否则松杆后 9 cm/s 的 TCP 会一直自己走。
2. **J1 过折保护**：−140° 远过照片 −89.5°。branch / comfort 没挡住。
3. **日志**：native `JointIkStep.v_cmd = v_recv`，避免再被 `v_cmd_*` 骗。
4. **硬件复测操作**：对照 122744 再做一段 **沿轨 Y 平移**（不要先拧满偏航），才能公平说「联动变差」是不是还在。本 run 几乎没给 `vy`。
5. 若仍要压扭转时的角残差：零角速度保护 / 角向 slack 上限——计划写明必须另开一轮，本 run 不能当 aniso 失败。

---

## 10. 附录：源码副本位置

当前（复测时）源码已复制：

```
MD/todo_controller_logs/code_143652/
  inner.cpp
  task_weight.hpp
  rail_command.hpp
  rail_extension.py
  rail_command.py
  loop.py
  qp_builder.py
  client.py
  servo.py
  run_controller.py
  gamepad.py
```

`SERVO_TWIST` 每拍把 `pose_d` 设成当前 TCP（所以 track_err 恒 0）：

```53:63:peirastic/realman8dof/modes/servo.py
    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        del t_s, f_ext
        twist = _as_twist(self.source)
        self.last_path_twist = twist.copy()
        self.last_feedback_twist[:] = 0.0
        self.last_vel_ff = twist.copy()
        self.last_err_mm = 0.0
        self.last_pose_d = np.asarray(current_pose, dtype=float).reshape(6).copy()
        return twist
```

要害 CSV 片段（每拍，要害列）在 `MD/todo_controller_logs/run_20260825_143652_*_keycols.csv`。原始 454 列全文是旁边那份 23 MB `run_20260825_143652.csv`。
