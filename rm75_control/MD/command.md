# 常用命令

仓库根目录（所有命令均在此执行）：

```bash
cd /media/camp/EXT_DRIVE/rm75_control
source env.sh
```

---

## 1. 末端 TCP 重量 / 力补偿辨识

辨识重力补偿参数 φ（含等效质量 `m` kg、质心 `mc`、传感器偏置等）。  
配置：`configs/force_compensation/force_id.yaml`  
位姿槽位：`configs/force_compensation/poses.yaml`（A/B/C/D）  
输出：`data/force_compensation/logs/force_id_phi.json`

### 前置

1. 示教器 / Web UI 将**当前工具坐标系**设为 **Arm_Tip**（与 `poses.yaml` 中 `pose_tool_frame` 一致）。
2. 末端勿碰外物；六维力传感器工作正常。
3. 若需更新某位姿，先保存再标定：

```bash
python apps/force_compensation/force_calibrate.py --save-pose a   # 或 b / c / d
```

### 完整标定（采集 A→B→C→D→A + 拟合 φ）

```bash
python apps/force_compensation/force_calibrate.py
```

### 仅预览采集流程（不连真机写 npz）

```bash
python apps/force_compensation/force_calibrate.py --dry-run
```

### 已有 npz，只做拟合

```bash
python apps/force_compensation/force_calibrate.py --identify-only
```

完成后终端会打印 `m = … kg` 等；控制器里 `phi_source: phi_recommended` 会读上述 json。

---

## 2. force_monitor（实时补偿力监视）

拖动机械臂（自由空间），对比 **原始力** 与 **补偿后 F_ext**。  
依赖：先完成 §1 标定，存在 `data/force_compensation/logs/force_id_phi.json`。

```bash
python apps/force_compensation/force_monitor.py
```

可选：

```bash
python apps/force_compensation/force_monitor.py \
  --phi data/force_compensation/logs/force_id_phi.json \
  --phi-source phi_recommended

python apps/force_compensation/force_monitor.py --10p-only
```

关闭 matplotlib 窗口或 `Ctrl+C` 退出。

---

## 3. 三窗口：控制器 + Viewer + 任务编排

**同一台机器、三个终端，推荐日常用法。**

### 架构（当前 main 分支）

| 窗口 | 进程 | 连真机？ | 职责 |
|------|------|----------|------|
| **A** | `run_joint_admittance.py` | **是**（唯一 TCP + UDP） | 常驻；发布 `rm75_state`；**200 Hz WBC 在 A 内**；直发 CANFD |
| **B** | `run_with_twin.py` | 否 | Genesis 数字孪生，只读 `rm75_state` |
| **C** | `d_sin_tool_y.py` | 否（默认 attach） | 本地 IK / 相位规划；经 phase IPC 提交任务；监控进度 |

**进程间通信（POSIX SHM，同机）：**

| 段名 | 方向 | 用途 |
|------|------|------|
| `rm75_state` | A → B/C | 关节 / 位姿 / 力 / 滑轨（200 Hz） |
| `rm75_phase_ctl` | A ↔ C | 任务 START / STOP、运行状态 |
| `rm75_phase_payload` | C → A | 任务参数 JSON（IK 结果、scan 参数等） |

> **要点：** C **不跑** 200 Hz 控制环，也不经 SHM 转发 CANFD（旧版 attach 卡顿根因）。WBC 路径与 ebd313a 单进程一致：A 内 UDP 反馈 → QP → `send_joint_canfd`。

### 启动顺序

1. **先开 A**
2. B、C 任意顺序；B 可先开，等 A 上线后自动 `running`
3. C 可反复启动；A 保持 hot-wait，任务结束后再提交下一项
4. **仅重启 C/B 时不必重启 A**；若 C 连不上 A，先 Ctrl+C 重启 A 再试

### 窗口 A — 控制器 daemon

```bash
cd /media/camp/EXT_DRIVE/rm75_control
source env.sh

python apps/joint_admittance_8dof/run_joint_admittance.py \
  --config configs/joint_admittance_8dof.yaml
```

**正常输出（精简）：**

```
rm75 controller: running (shm 'rm75_state' @ 200 Hz)
rm75 controller: hot-wait
rm75 controller: running task #1
rm75 controller: task #1 done (8.6s, 1709 ticks)
rm75 controller: hot-wait
```

| 选项 | 说明 |
|------|------|
| `--no-state-relay` | 不发布 SHM（仅调试） |
| `--verbose` / `-v` | 打印 WBC 相位细节、力补偿 tool 提示等 |
| `--hold` | A 本地 idle hold（**勿与 C 同时**） |

### 窗口 B — Genesis Viewer

```bash
cd /media/camp/EXT_DRIVE/rm75_control
source env.sh

python apps/joint_admittance_8dof/run_with_twin.py
```

**正常输出：**

```
rm75 twin: waiting for 'rm75_state' …
rm75 twin: running
```

A 重启后会多一行 `rm75 twin: reconnected to controller`。  
**不连机器人**；只订阅 A 的 SHM。`Ctrl+C` 退出即可。

### 窗口 C — 任务编排（attach A）

```bash
cd /media/camp/EXT_DRIVE/rm75_control
source env.sh

# 示例：到 D 后力控 Y 扫描（需 §1 力补偿标定）
python apps/joint_admittance_8dof/d_sin_tool_y.py \
  --config configs/joint_admittance_8dof.yaml \
  --enable-force --desired-z 1.0 \
  --scan-duration 30
```

```bash
# 示例：到 D → hold 5s → tcp_fixed 滑轨 +Y 15cm（无扫描）
python apps/joint_admittance_8dof/d_sin_tool_y.py \
  --config configs/joint_admittance_8dof.yaml \
  --scan-duration 0 \
  --hold-at-d-s 5 \
  --rail-move-cm 15 \
  --rail-move-mode tcp_fixed \
  --rail-move-dir +y
```

**正常输出（默认精简）：**

```
rm75 task: connecting to window A …
rm75 task: connected
D dz=220mm tool=Pin-tcp z=0.405
scan: Y 16cmpp Fz=1.0N 30s
rm75 task: submitted task #1
rm75 task: move->d
rm75 task: scan
rm75 task: done
```

| 参数 | 含义 |
|------|------|
| `--scan-duration 0` | 不做 sin 扫描 |
| `--hold-at-d-s 5` | 到 D 后 hold 5 s（滑轨锁定） |
| `--rail-move-cm 15` | 滑轨移动 15 cm |
| `--rail-move-mode rail_only` | 臂不动，TCP 随滑轨 |
| `--rail-move-mode tcp_fixed` | 滑轨动，TCP 尽量固定 |
| `--rail-move-dir +y` / `-y` | 滑轨方向 |
| `--enable-force` | 扫描阶段力控（需 §1 标定） |
| `--desired-z 1.0` | 目标 Fz (N) |
| `--log-interval 2` | attach 模式下相位切换日志间隔（0=关闭） |
| `--verbose` / `-v` | IK 细节、move 时长估算等 |
| `--no-attach-state` | **单进程模式**：C 自带 TCP+WBC（勿与 A 同开） |

滑轨峰值速度：`configs/joint_admittance_8dof.yaml` → `inner.rail.v_max_m_s`（默认 5 cm/s）。

**attach 默认规则：**

- 默认 attach 窗口 A（读 `rm75_state`），C **不连** TCP
- C **不应**出现 `current c api version`（出现说明 C 在连 TCP，与 A 冲突）
- C 中 `Ctrl+C` → 向 A 发 `stop_req`，A 打印 `task #N stopped` 后回到 hot-wait
- 任务编号 `#1, #2, #3…` 在 A/C 一致；仅 START 占序号，STOP 不占

### 单进程调试（勿与三窗口同开）

```bash
python apps/joint_admittance_8dof/d_sin_tool_y.py \
  --config configs/joint_admittance_8dof.yaml \
  --no-attach-state \
  --enable-force --desired-z 1.0 --scan-duration 30
```

Viewer 请单独开窗口 B（`run_with_twin.py`），不要和 C 绑在一起。

### 辅助

```bash
# 查看 SHM relay 是否在发
python -m rm75_control.tools.state_echo --subscribe rm75_state

# FK 对齐检查（读 relay，可选直连真机对比）
python apps/joint_admittance_8dof/check_fk_once.py --subscribe rm75_state
```

### 故障排查

| 现象 | 处理 |
|------|------|
| C 卡在 `connecting to window A` | 确认 A 已开；不行则重启 A |
| C 出现 `current c api version` | C 误连 TCP；关掉 C，勿加 `--no-attach-state` |
| 臂不动 | 仅 A 可连 TCP；确认 C 为 attach 模式 |
| B viewer 不动 | 等 A 发布 `rm75_state`；或重启 B |
| 扫描卡顿 | 确认 C 为 attach（WBC 在 A），勿用旧版 CANFD SHM 转发 |
| 退出后 C 再连不上 A | 重启 A（旧客户端可能破坏 SHM）；现已修复 tracker 误 unlink |

---

## 4. Genesis 首次安装（Viewer 报缺依赖时）

```bash
source env.sh
bash rm75_control/control/joint_admittance_8dof/viewer/install_torch.sh
pip install -r rm75_control/control/joint_admittance_8dof/viewer/requirements.txt
```

更多细节见 `rm75_control/control/joint_admittance_8dof/viewer/README.md`（参数化模型见 `param_model/README.md`，配置 `config/slider_rail.yaml`）。

## 5. Genesis 离线 Viewer（不连机器人 / 不调 SHM）

```bash
cd /media/camp/EXT_DRIVE/rm75_control
source env.sh

# 默认加载 config/slider_rail.yaml，一般无需 --spec
python -m rm75_control.control.joint_admittance_8dof.viewer.demo --show-viewer

# 显式指定配置（路径在 config/，不在 genesis/）
python -m rm75_control.control.joint_admittance_8dof.viewer.demo --show-viewer \
  --spec rm75_control/control/joint_admittance_8dof/config/slider_rail.yaml
```

改几何/标定：编辑 `rm75_control/control/joint_admittance_8dof/config/slider_rail.yaml`。
