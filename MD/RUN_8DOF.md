# joint_admittance_8dof — RealUS_playground 独立副本

**两个环境分工：**

| 用途 | 环境 | 激活 |
|------|------|------|
| 控制器 / 任务 / 连机器人 | `envs/rm75` | `source env.sh` |
| **Genesis viewer / twin** | `envs/genesis`（Among_US） | `source env_viewer.sh` |

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/rm75_control
```

## Genesis demo viewer（GPU，Among_US genesis 环境）

```bash
source env_viewer.sh
python -m rm75_control.control.joint_admittance_8dof.viewer.demo --show-viewer
```

不要在 `rm75` 环境里装 torch/genesis；`genesis` 环境已具备。

## 窗口 A — 控制器（连机器人，rm75 环境）

```bash
source env.sh
pip install -r requirements.txt   # 首次：pinocchio/proxsuite 等

python apps/joint_admittance_8dof/run_joint_admittance.py \
  --config configs/joint_admittance_8dof.yaml
```

## 窗口 B — 数字孪生（genesis 环境）

```bash
source env_viewer.sh
python apps/joint_admittance_8dof/run_with_twin.py
```

## 窗口 C — 任务编排（rm75 环境）

```bash
source env.sh
python apps/joint_admittance_8dof/d_sin_tool_y.py \
  --config configs/joint_admittance_8dof.yaml
```

## 配置与资产

| 路径 | 说明 |
|------|------|
| `configs/joint_admittance_8dof.yaml` | 主控制器配置 |
| `rm75_control/control/joint_admittance_8dof/config/slider_rail.yaml` | Genesis 参数化滑轨 |
| `../camera_calibration/calibration_results/genesis_bundle.yaml` | 相机+床（viewer 自动加载） |

详细命令见 `rm75_control/MD/command.md`。
