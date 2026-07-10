# perception

真相机 ingress + DWPose/EasyMocap SMPL-X 拟合入口。

## Apps

| Script | Role |
|--------|------|
| `apps/run_realsense_camera_publisher.py` | N 路 RealSense → ZMQ `amongus_camera_frame_v1` |
| `apps/run_smplx_capture.py` | Terminal 8：多视角拟合 → 5598 + canonical staging |
| `apps/run_anatomy_retarget.py` | Terminal 9：解剖 retarget + 可选血管/骨骼导出 |

启动说明见 [`../MD/COMMAND.md`](../MD/COMMAND.md)。

依赖：`source ../env.sh`（`envs/genesis`）。RealSense 需要 `pyrealsense2`。
