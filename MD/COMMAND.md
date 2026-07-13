# RealUS launch commands

## Phase 1 — 真相机 → Genesis（当前；UE 见下文 Phase 2）

**范围：** 常开 N 路相机 → 触发 SMPL-X → 解剖/血管 retarget → **仅在 Genesis 可视化**。不含 UE / SceneInit / bake / 5599。

### 端口（Phase 1）

| 端口 | 用途 |
|------|------|
| `tcp://127.0.0.1:17356` | 真相机 ZMQ（见下双 topic） |
| `amongus_camera_preview_v1` | 低分辨率预览（OpenCV / 标定 UI，可丢帧） |
| `amongus_camera_frame_v1` | 全分辨率 capture（标定 Space / Window 8 SMPL-X） |
| `tcp://127.0.0.1:5598` | 橙色 SMPL-X mesh（`amongus_multiview_track_v1`） |
| `tcp://127.0.0.1:5601` | Anatomy asset upsert |
| `tcp://127.0.0.1:17357` | （可选）capture trigger |

### 启动顺序 Phase 1

`Cam → Preview(可选) → G → 8(触发) → 9`

#### Prelude

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground && source env.sh
```

#### Cam — N 路 RealSense（常开）

```bash
$REALUS_CAMERA_PY perception/apps/run_realsense_camera_publisher.py \
  --bundle "$CAMERA_CALIB_BUNDLE" \
  --cameras-yaml "$REALUS_CAMERAS_YAML" \
  --pub-bind tcp://127.0.0.1:17356 \
  --undistort
```

首次缺 `pyrealsense2` 时：`pip install -r perception/requirements.txt`（或 `$REALUS_CAMERA_PY` 会自动回退到 `envs/camera_calib`）。

相机数 = `genesis_bundle.yaml` 全部 alias（扩展相机只改标定，不写死 4）。

Publisher 并行采集，时间戳：`source_time_ns` = RealSense global time（硬件），`sim_time_ns` 同值；Window 8 / 标定 capture 按 **hardware timestamp** 对齐（非 per-camera frame_index）。

#### 标定 PyQt（Stage 1/2，与 Cam 共用 publisher）

先启动 Cam publisher，再开标定 UI（ZMQ 预览 + capture 与 SMPL-X 同链路）：

```bash
cd camera_calibration && source env.sh
python scripts/run_ui.py --zmq-connect tcp://127.0.0.1:17356
```

- 预览：`amongus_camera_preview_v1`（低分辨率，只显示最新帧，不卡）
- Space 采集：`amongus_camera_frame_v1`（全分辨率，按 `source_time_ns` 选最紧同步组）
- Stage 0 内参仍用本地 USB（单相机），Stage 1/2 建议 ZMQ 模式

本地 USB 模式（无 publisher）仍可用：`python scripts/run_ui.py`（已优化：隐藏 tab 暂停检测、preview decimate×2、device timestamp 同步）。

#### Preview — 单条横条预览（可选，与 Cam 并行）

```bash
$PY perception/apps/run_camera_preview.py \
  --bundle "$CAMERA_CALIB_BUNDLE" \
  --connect tcp://127.0.0.1:17356
```

默认 **一个窗口** 横排 cam1..camN（约 1920×270）；只显示每路 **最新帧**（丢弃 ZMQ 队列里的旧帧，避免数秒滞后）。需要多窗口时才加 `--separate-windows`。

`run_camera_preview.py` 从 bundle 读相机名；也可 `--cameras cam1 cam2` 指定子集。

#### G — Genesis viewer（两种模式）

**view（Phase 1 默认）** — 不连真机、不读 SHM；显示床+滑轨机械臂（静态 demo 姿态）+ 5598/5601 人：

```bash
cd rm75_control && source env_viewer.sh && cd "$REALUS_PROJECT_ROOT"
$PY perception/apps/run_genesis_perception_viewer.py
```

**twin** — 窗口 A 真机运行时，Genesis 镜像 `rm75_state`（机械臂随真机动）+ 可选人体 overlay：

```bash
cd rm75_control && source env_viewer.sh && cd "$REALUS_PROJECT_ROOT"
python apps/joint_admittance_8dof/run_with_twin.py \
  --track-subscribe tcp://127.0.0.1:5598 \
  --anatomy-subscribe tcp://127.0.0.1:5601 \
  --canonical-human-source fitted
```

view 模式默认 **显示机械臂+滑轨**；仅床+相机时用 `--no-robot`。床体高度 **完全来自** `genesis_bundle.yaml` 的 `bed.height_m`（或 `support_surface.top_z_m`）：Genesis 网格从 z=0 铺到该标定顶面，无固定厚度常数。

#### 8 — 触发 SMPL-X 拟合（躺下后执行，可多次）

**实验用远程按钮（临时）：**

```bash
$PY perception/apps/run_capture_remote_gui.py
```

弹窗点 **CAPTURE AND GENERATE SMPLX** = Window 8 全流程 + Genesis 投射；结果在 `smplx_outputs/<timestamp>/moment_0000/`（原图、DWPose、融合点、SMPL-X 重投影 PNG）。

**命令行等价：**

```bash
source env.sh
$PY perception/apps/run_smplx_capture.py \
  --config configs/tracking/realus_dwpose_easymocap.yaml \
  --connect tcp://127.0.0.1:17356 \
  --output-root smplx_outputs \
  --write-debug-images \
  --publish-kind smplx_mesh \
  --publish-genesis
```

**验收：** `smplx_outputs/<run>/moment_0000/smplx_result.npz`、`beta_calibration/betas.npy`、`moment.json` quality gate；G 窗口见橙色 mesh。

同一被试换姿势：Cam/G 不关，重复 Window 8；可加 `--betas-path smplx_outputs/<run>/beta_calibration/betas.npy` 跳过 shape 重标定。

#### 9 — 解剖 retarget + 血管导出

```bash
source env.sh
$PY perception/apps/run_anatomy_retarget.py \
  --publish-genesis \
  --export-vessels
```

**验收：** `outputs/anatomy_retarget/latest_asset/anatomy_rigged.npz`、`limb_vessel_planning/`；G 窗口见半透明解剖 + 血管 overlay。

#### （可选）ZMQ 触发 Window 8

服务（常开）：

```bash
$PY perception/apps/run_capture_trigger_service.py --bind tcp://127.0.0.1:17357
```

触发一次 capture：

```bash
$PY perception/apps/fire_capture_trigger.py --connect tcp://127.0.0.1:17357
```

---

## Phase 2 — 全链路 Genesis + UE（延后）

真相机 SMPL-X → Genesis twin → UE（同体型 bake）。**Phase 1 验收通过后再跑本节。**

环境：共用 `/media/camp/EXT_DRIVE/envs/genesis`；UE/Blender 用硬盘外置安装。

## 0. Prelude（每个窗口先跑）

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
# PY / PYTHONPATH / SESSION_DIR / CAMERA_CALIB_BUNDLE 已设置
```

可选清理：

```bash
pkill -f "run_realsense_camera_publisher|run_smplx_capture|run_anatomy_retarget|run_genesis_perception_viewer|run_capture_trigger|run_scene_init|run_canonical_zmq|run_ue_scene_session|run_with_twin" 2>/dev/null || true
```

## 端口表

| 端口 / 端点 | 用途 |
|-------------|------|
| `tcp://127.0.0.1:17356` | 真相机 JPEG 帧（`amongus_camera_frame_v1`） |
| `tcp://127.0.0.1:5598` | 拟合橙色 SMPL-X mesh（`amongus_multiview_track_v1`） |
| `tcp://127.0.0.1:5599` | Canonical 场景（机器人 + **拟合人**）→ UE |
| `tcp://127.0.0.1:5588` | SceneInit |
| `tcp://127.0.0.1:5601` | Anatomy asset upsert |
| UE UDP `5601` | Canonical bridge → UE（与 ZMQ 5601 不同通道，见桥实现） |

## 启动顺序

`Cam → 1 → 4 → 5 → PIE → 6 → B(/A) → 8 → bake → (可选重跑 5) → 9`

### Cam — N 路 RealSense publisher

```bash
source env.sh
# 需要 pyrealsense2（可用 camera_calib env 的 python，或 genesis 已装）
$PY perception/apps/run_realsense_camera_publisher.py \
  --bundle "$CAMERA_CALIB_BUNDLE" \
  --cameras-yaml "$REALUS_CAMERAS_YAML" \
  --pub-bind tcp://127.0.0.1:17356 \
  --undistort
```

相机数 = `genesis_bundle.yaml` 里全部 alias（当前 cam1–cam4；加相机只改标定 yaml）。

### 窗口 1 — UE session watcher

```bash
source env.sh
$PY -m projects.genesis_ue_sync.cli.render.unreal.run_ue_scene_session \
  --session-dir "$SESSION_DIR" \
  --watcher-only \
  --clear-pending-commands
```

### 窗口 4 — SceneInit publisher

```bash
source env.sh
$PY -m projects.genesis_ue_sync.cli.render.unreal.run_scene_init_publisher \
  --scene-spec configs/scenes/realus_bed_rail_scene.yaml \
  --bind tcp://127.0.0.1:5588 \
  --repeat-s 2.0 \
  --robot-model rm75_6f
```

### 窗口 5 — SceneInit → UE prepare（一次）

首次可用固定 BEDLAM body 烟测；**同体型交付**需先跑窗口 8 + bake，再重跑本窗口。

```bash
source env.sh
$PY -m projects.genesis_ue_sync.cli.render.unreal.run_scene_init_zmq_ue_bridge \
  --connect tcp://127.0.0.1:5588 \
  --session-dir "$SESSION_DIR" \
  --scene-apply-mode prepare \
  --exit-after-first
```

然后在 UE 中 **PIE**。

### 窗口 6 — Canonical → UE 动态桥

```bash
source env.sh
$PY -m projects.genesis_ue_sync.cli.render.unreal.run_canonical_zmq_ue_bridge \
  --canonical-connect tcp://127.0.0.1:5599 \
  --session-dir "$SESSION_DIR"
```

### 窗口 A — 机器人导纳（可选，现有）

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/rm75_control
source env.sh
python apps/joint_admittance_8dof/run_joint_admittance.py --config configs/joint_admittance_8dof.yaml
```

### 窗口 B — Genesis twin + 橙色人 + canonical 拟合人

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/rm75_control
source env_viewer.sh
python apps/joint_admittance_8dof/run_with_twin.py \
  --track-subscribe tcp://127.0.0.1:5598 \
  --canonical-human-source fitted \
  --anatomy-subscribe tcp://127.0.0.1:5601
```

### 窗口 8 — EasyMocap SMPL-X 拟合（单帧；可重复跑做动态更新）

```bash
source env.sh
$PY perception/apps/run_smplx_capture.py \
  --config configs/tracking/realus_dwpose_easymocap.yaml \
  --connect tcp://127.0.0.1:17356 \
  --publish-kind smplx_mesh \
  --publish-genesis
```

产物：`smplx_outputs/<run>/moment_0000/smplx_result.npz`、`beta_calibration/betas.npy`，以及 `outputs/anatomy_retarget/latest_canonical/`。

**不要用** `--publish-kind smpl_pose`（72D，对 SMPL-X 会报错）。

### 同体型 UE bake（主路径，必须）

```bash
source env.sh
# 把 <run> 换成最新 smplx_outputs 目录名
SHAPES=smplx_outputs/<run>/beta_calibration/betas.npy
# 若无 betas.npy，可用 smplx_result.npz（含 shapes）
$PY scripts/bake_subject_ue_body.py --shapes "$SHAPES" --gender male
$PY scripts/point_scene_to_subject_bake.py \
  --scene configs/scenes/realus_bed_rail_scene.yaml \
  --npz outputs/ue_bake/subject_shape_tpose.npz
```

然后**重跑窗口 5 prepare**，使 UE `GEN_visible_human` 使用同一份 EasyMocap 10-D shapes 烤出的 T-pose（Blender 缺维 pad 0）。固定 `it_4375` 仅烟测，不算同体型交付。

### 窗口 9 — 解剖 retarget + 腿血管中线 / 大腿骨骼点云

```bash
source env.sh
$PY perception/apps/run_anatomy_retarget.py \
  --publish-genesis \
  --export-vessels
```

不做体表最近投影。血管中线与骨骼 markers 在 `outputs/anatomy_retarget/limb_vessel_planning/`。

## 体型对齐检查

1. Genesis 橙色 mesh 与 UE 贴图人：骨盆高度、腿长、肩宽、整体 bbox。  
2. `outputs/ue_bake/subject_shape_meta.json` 中 `shapes10` 必须等于 EasyMocap `betas.npy`。  
3. 坐标：Genesis RH·m·Z-up → UE LH·cm（桥内 `diag(1,-1,1)`）。  
4. 姿态：twin `--canonical-human-source fitted` 经 `pose_adapter`（Rh+87→body）驱动 UE 骨。

## 场景 / 标定再生

床、相机、导轨位姿变更后：

```bash
source env.sh
$PY scripts/build_realus_scene_from_bundle.py \
  --bundle "$CAMERA_CALIB_BUNDLE" \
  --slider-rail rm75_control/rm75_control/control/joint_admittance_8dof/config/slider_rail.yaml
```

## 关闭顺序

`9 → 8 → B/A → 6 → 4 → Cam → 1`（5 通常已退出）
