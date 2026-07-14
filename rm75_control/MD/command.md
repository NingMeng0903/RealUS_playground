# 常用命令

仓库根目录（控制器 / 任务窗口在此执行）：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/rm75_control
source env.sh
```

**两个 Python 环境：**

| 用途 | 环境 | 激活 |
|------|------|------|
| 窗口 A / C（控制器、任务、连真机） | `envs/rm75` | `source env.sh` |
| 窗口 B（Genesis twin / viewer） | `envs/genesis` | `source env_viewer.sh` |

全链路（相机 → SMPL-X → 解剖 → twin overlay）见上级目录 `RealUS_playground/MD/COMMAND.md`。

---

## 0. 控制器启动流程（三窗口 + 可选感知）

日常真机控制推荐 **三个终端**，**同一台机器**。感知服务（5598 / 5601）仅在需要人体 overlay 时另开。

### 0.1 前置（首次或换工具后）

1. **力补偿标定**（扫描力控必做）：完成本文 §1，生成 `data/force_compensation/logs/force_id_phi.json`。
2. **示教器 / Web UI**：当前工具坐标系设为 **Arm_Tip**（与 `configs/force_compensation/poses.yaml` 一致）。
3. **网络**：`configs/joint_admittance_8dof.yaml` 中 `robot.ip` / `port` 与真机一致（默认 `192.168.1.18:8080`）。
4. **首次 rm75 环境**：`pip install -r requirements.txt`（pinocchio、proxsuite 等）。

### 0.2 架构

| 窗口 | 进程 | 环境 | 连真机？ | 职责 |
|------|------|------|----------|------|
| **A** | `run_joint_admittance.py` | `env.sh` | **是**（唯一 TCP + UDP） | 常驻；200 Hz WBC；发布 `rm75_state`；直发 CANFD |
| **B** | `run_with_twin.py` | `env_viewer.sh` | 否 | Genesis 数字孪生，只读 `rm75_state`；可选 5598/5601 人体 overlay |
| **C** | `d_sin_tool_y.py` | `env.sh` | 否（attach A） | 本地 IK / 相位规划；经 phase IPC 提交任务 |

**SHM（同机 POSIX 共享内存）：**

| 段名 | 方向 | 用途 |
|------|------|------|
| `rm75_state` | A → B/C | 关节 / 位姿 / 力 / 滑轨（200 Hz） |
| `rm75_phase_ctl` | A ↔ C | 任务 START / STOP、运行状态 |
| `rm75_phase_payload` | C → A | 任务参数 JSON |

> C **不跑** 200 Hz 控制环，也不经 SHM 转发 CANFD。WBC 只在 A 内：UDP 反馈 → QP → `send_joint_canfd`。

### 0.3 启动顺序

```
[可选感知] Cam → 8(SMPL-X) → 9(解剖)
        ↓
    先开 A（控制器）
        ↓
    B、C 任意顺序（B 可先开，等 A 上线后自动 running）
```

1. **必须先开窗口 A**；B/C 可后开。
2. 仅重启 C/B 时**不必**重启 A；C 连不上时先 Ctrl+C 重启 A。
3. 需要 twin 上橙色人 / 解剖 overlay 时，在 A 之前或并行启动感知（端口见下表）。

| 端口 | 用途 |
|------|------|
| `tcp://127.0.0.1:17356` | 相机 ZMQ（SMPL-X 输入） |
| `tcp://127.0.0.1:5598` | 橙色 SMPL-X mesh |
| `tcp://127.0.0.1:5601` | 解剖 asset upsert |

感知命令详见 `../MD/COMMAND.md` Phase 1（`run_realsense_camera_publisher.py` → `run_smplx_capture.py` → `run_anatomy_retarget.py`）。

### 0.4 窗口 A — 控制器（必开）

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/rm75_control
source env.sh

python apps/joint_admittance_8dof/run_joint_admittance.py \
  --config configs/joint_admittance_8dof.yaml
```

**正常输出：**

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
| `--verbose` / `-v` | WBC 相位细节、力补偿 tool 提示 |
| `--hold` | A 本地 idle hold（**勿与 C 同时**） |
| `--dry-run` | 不连真机，检查配置 |

保持 A 常驻；`Ctrl+C` 会断开 TCP/UDP 并停止 SHM。

### 0.5 窗口 B — Genesis 数字孪生

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/rm75_control
source env_viewer.sh

# 仅机械臂镜像（无人体）
python apps/joint_admittance_8dof/run_with_twin.py

# 机械臂 + 橙色 SMPL-X + 解剖（需感知已发布 5598/5601）
python apps/joint_admittance_8dof/run_with_twin.py \
  --track-subscribe tcp://127.0.0.1:5598 \
  --track-mesh-alpha 120 \
  --anatomy-subscribe tcp://127.0.0.1:5601 \
  --canonical-human-source fitted
```

**正常输出：**

```
rm75 twin: waiting for 'rm75_state' …
rm75 twin: running
rm75 twin: human overlay track=tcp://127.0.0.1:5598 canonical=fitted
```

| 选项 | 说明 |
|------|------|
| `--backend cpu` | CUDA 不可用时降级（慢，见 §0.7） |
| `--headless` | 无 Genesis 窗口，仅后台同步 |
| `--no-anatomy` | 关闭 5601 解剖绘制 |
| `--track-mesh-alpha 0-255` | 橙色皮肤透明度（默认 55） |

A 重启后 B 会打印 `rm75 twin: reconnected to controller`。B **不连机器人**。

### 0.6 窗口 C — 任务编排（attach A）

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/rm75_control
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

**正常输出：**

```
rm75 task: connecting to window A …
rm75 task: connected
rm75 task: submitted task #1
rm75 task: done
```

- 默认 **attach 窗口 A**，C **不应**出现 `current c api version`（出现说明 C 误连 TCP）。
- C 中 `Ctrl+C` → A 打印 `task #N stopped` 后回到 hot-wait。
- 单进程调试加 `--no-attach-state`（**勿与 A 同开**）。

### 0.7 故障排查（启动阶段）

| 现象 | 处理 |
|------|------|
| C 卡在 `connecting to window A` | 确认 A 已开；不行则重启 A |
| C 出现 `current c api version` | C 误连 TCP；关掉 C，勿加 `--no-attach-state` |
| B viewer 不动 | 等 A 发布 `rm75_state`；或重启 B |
| `Backend gs.cuda not available` | GPU 驱动异常；**重启电脑**后重试；临时加 `--backend cpu` |
| `CUDA unknown error` / `cuInit 999` | 内核日志 `Xid 154 Node Reboot Required` → 必须重启整机 |
| twin 无橙色人 / 解剖 | 确认 5598/5601 有发布；先跑 Phase 1 窗口 8、9 |
| 扫描卡顿 | 确认 C 为 attach 模式（WBC 在 A） |

**辅助：**

```bash
python -m rm75_control.tools.state_echo --subscribe rm75_state
python apps/joint_admittance_8dof/check_fk_once.py --subscribe rm75_state
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

## 3. 任务参数速查（窗口 C）

启动流程见 **§0**。以下为 `d_sin_tool_y.py` 常用参数。

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

### 单进程调试（勿与三窗口同开）

```bash
python apps/joint_admittance_8dof/d_sin_tool_y.py \
  --config configs/joint_admittance_8dof.yaml \
  --no-attach-state \
  --enable-force --desired-z 1.0 --scan-duration 30
```

Viewer 请单独开窗口 B（`run_with_twin.py`），不要和 C 绑在一起。

---

## 4. Genesis 首次安装（Viewer 报缺依赖时）

```bash
source env_viewer.sh
bash rm75_control/control/joint_admittance_8dof/viewer/install_torch.sh
pip install -r rm75_control/control/joint_admittance_8dof/viewer/requirements.txt
```

更多细节见 `rm75_control/control/joint_admittance_8dof/viewer/README.md`（参数化模型见 `param_model/README.md`，配置 `config/slider_rail.yaml`）。

## 5. Genesis 离线 Viewer（不连机器人 / 不调 SHM）

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/rm75_control
source env_viewer.sh

# 默认加载 config/slider_rail.yaml，一般无需 --spec
python -m rm75_control.control.joint_admittance_8dof.viewer.demo --show-viewer

# 显式指定配置（路径在 config/，不在 genesis/）
python -m rm75_control.control.joint_admittance_8dof.viewer.demo --show-viewer \
  --spec rm75_control/control/joint_admittance_8dof/config/slider_rail.yaml
```

改几何/标定：编辑 `rm75_control/control/joint_admittance_8dof/config/slider_rail.yaml`。
