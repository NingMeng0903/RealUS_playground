# Genesis viewer — parametric slider/rail + RM75 arm (8 DOF)

## 环境（重要）

| 组件 | 环境 |
|------|------|
| **本 viewer / twin** | `envs/genesis`（Among_US）→ `source env_viewer.sh` |
| 控制器 / WBC / 真机 | `envs/rm75` → `source env.sh` |

不要在 `rm75` 里安装 `torch` / `genesis-world`。

## Run (offline demo)

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/rm75_control
source env_viewer.sh
python -m rm75_control.control.joint_admittance_8dof.viewer.demo --show-viewer
```

Default loads `config/slider_rail.yaml` + sibling `camera_calibration/calibration_results/genesis_bundle.yaml`.

Disable calib scene: `--no-calib-scene`. Demo **requires CUDA GPU** (no `--backend cpu`).

## Digital twin

```bash
source env_viewer.sh
python apps/joint_admittance_8dof/run_with_twin.py
```

Controller (separate terminal, `source env.sh`):

```bash
python apps/joint_admittance_8dof/run_joint_admittance.py --config configs/joint_admittance_8dof.yaml
```
