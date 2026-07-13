# perception

真相机 ingress + DWPose/EasyMocap SMPL-X 拟合入口。

## Apps

| Script | Role |
|--------|------|
| `apps/run_realsense_camera_publisher.py` | N 路 RealSense → ZMQ `amongus_camera_frame_v1` |
| `apps/run_camera_preview.py` | 单条横条 OpenCV 预览（相机名从 bundle 读；默认非多窗） |
| `apps/run_genesis_perception_viewer.py` | Phase 1 Genesis viewer：5598 + 5601 + 血管 overlay |
| `apps/run_capture_remote_gui.py` | （实验）远程一键 Capture + SMPL-X 按钮 |
| `apps/run_smplx_capture.py` | Window 8：多视角拟合 → 5598 + canonical staging |
| `apps/run_anatomy_retarget.py` | Window 9：解剖 retarget + 可选血管/骨骼导出 |
| `apps/run_capture_trigger_service.py` | （可选）ZMQ 触发 Window 8 |
| `apps/fire_capture_trigger.py` | 发送一次 capture trigger |

启动说明见 [`../MD/COMMAND.md`](../MD/COMMAND.md) **Phase 1** 章节。

依赖：`source ../env.sh`（`envs/genesis`）。RealSense 需要 `pyrealsense2`：

```bash
pip install -r perception/requirements.txt
```

Cam 发布器用 `$REALUS_CAMERA_PY`（genesis 无 pyrealsense2 时自动用 `envs/camera_calib`）。
