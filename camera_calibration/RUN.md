# 在 RealUS_playground 工作区中使用

本包从 `Among_US/camera_calibration` 拷贝，路径自包含（`PROJECT_ROOT` = 本目录）。

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/camera_calibration
source env.sh
pip install -r requirements.txt   # 首次

python scripts/run_ui.py
```

当前标定结果在 `calibration_results/`（4 相机内外参 + 床 `world_meta.yaml` + `genesis_bundle.yaml`）。

**Viewer 自动加载：** 同级 `rm75_control` 的 demo / twin 启动时会读 `calibration_results/genesis_bundle.yaml`（相机位姿+FOV、地面、淡蓝床面）。保持 RealUS_playground 目录结构即可，无需再拷贝一份到 rm75_control。

导出/验证：

```bash
python scripts/export_genesis_calibration.py
python scripts/verify_calibration.py
```

会话续标（可选）：从 Among_US 拷贝 `data/stage1_extrinsics/last` 与 `data/stage2_world/last` 到本目录 `data/`（约 140MB）。
