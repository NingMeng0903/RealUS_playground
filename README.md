# RealUS_playground

超声/机器人相关功能的**工作区根目录**。每个子目录是一个**独立功能包**（自带 `env.sh`、`requirements`、配置与数据），共用外部 conda 环境，**包与包之间不符号链接**。

## 当前包

| 目录 | 环境 | 功能 |
|------|------|------|
| [`rm75_control/`](rm75_control/) | `envs/rm75` / viewer:`envs/genesis` | RM75 8-DOF 导纳、Genesis twin（可挂 SMPL/解剖 overlay） |
| [`ird_playground/`](ird_playground/) | `envs/rm75` + PyTorch | 通用 Neural IRD 点场 + 查询侧区域 A；离散能力图/IRD GT |
| [`camera_calibration/`](camera_calibration/) | `envs/camera_calib` | 4 相机 + 床内外参标定 UI、`genesis_bundle.yaml` |
| [`perception/`](perception/) | `envs/genesis` | 真相机 ZMQ、DWPose+EasyMocap、解剖 retarget 入口 |
| [`src/`](src/) | `envs/genesis` | 从 Among_US 迁移的 `genesis_ue_sync` / `bridge` / `common` |
| [`configs/`](configs/) | — | RealUS scene / tracking / anatomy |
| [`ref_code_library/`](ref_code_library/) | — | EasyMocap、DWPose ONNX、SMPL-X 权重 |

全链路启动：[`MD/COMMAND.md`](MD/COMMAND.md)。根目录 `source env.sh`。

## 规划中的包（按需新建同级目录）

| 建议目录名 | 职责 |
|------------|------|
| `servo_executor/` | 发布给真机执行的伺服层（轨迹/力控接口，调用 `rm75_control` 或独立 CANFD） |
| `shared/`（可选） | 跨包公共类型：坐标系、时间戳、消息格式（仅当两包强耦合时再抽） |

Neural IRD 全期清单见 [`MD/todo.md`](MD/todo.md)。

## 划分原则

1. **一包一职责**：标定、控制、感知、执行分层，避免混在一个 `src/` 里。
2. **数据跟包走**：标定结果在 `camera_calibration/calibration_results/`；机器人在 `rm75_control/data/`。
3. **接口用文件/协议**：例如 `genesis_bundle.yaml` 给感知/仿真消费，不跨包 `import`。
4. **环境分离**：控制用 `rm75`（pinocchio + proxqp）；标定用 `camera_calib`（PyQt5 + RealSense）；避免一个 env 装全部。
5. **启动从包根目录**：`source <pkg>/env.sh` 再 `python scripts/...` 或 `python apps/...`。

## 快速启动

### 机器人控制 + Genesis demo

```bash
# 控制器 — rm75 环境
cd rm75_control && source env.sh
python apps/joint_admittance_8dof/run_joint_admittance.py --config configs/joint_admittance_8dof.yaml

# Viewer — Among_US genesis 环境（不要 source env.sh）
cd rm75_control && source env_viewer.sh
python -m rm75_control.control.joint_admittance_8dof.viewer.demo --show-viewer
```

详见 [`MD/RUN_8DOF.md`](MD/RUN_8DOF.md)。Git 操作备忘见 [`MD/GITHUB.md`](MD/GITHUB.md)。

### 相机标定 UI（或查看已有结果）

```bash
cd camera_calibration && source env.sh
python scripts/run_ui.py
```

标定产物：`camera_calibration/calibration_results/genesis_bundle.yaml`（4 相机 + 床世界系）。

## 包间数据流（目标架构）

```
camera_calibration/calibration_results/genesis_bundle.yaml
        │
        ├─► rm75_control viewer (demo / twin)  自动加载床+相机
        ├─► perception/ (RealSense ZMQ + EasyMocap)
        └─► genesis_ue_sync SceneInit / UE bridge
```

全链路命令见 [`MD/COMMAND.md`](MD/COMMAND.md)。

标定结果路径（viewer 自动发现，按优先级）：

1. 环境变量 `CAMERA_CALIB_BUNDLE`
2. 同级目录 `../camera_calibration/calibration_results/genesis_bundle.yaml`
3. `rm75_control/data/calibration/genesis_bundle.yaml`（可选本地副本）

## 包间数据流（控制）

```
rm75_control  ◄──── 任务/关节目标 ─────  servo_executor（未来）
```
