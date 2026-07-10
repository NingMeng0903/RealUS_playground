# RealUS full pipeline launch commands

真相机 SMPL-X → Genesis twin → UE（同体型 bake）全链路。  
环境：共用 `/media/camp/EXT_DRIVE/envs/genesis`；UE/Blender 用硬盘外置安装。

## 0. Prelude（每个窗口先跑）

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
# PY / PYTHONPATH / SESSION_DIR / CAMERA_CALIB_BUNDLE 已设置
```

可选清理：

```bash
pkill -f "run_realsense_camera_publisher|run_smplx_capture|run_anatomy_retarget|run_scene_init|run_canonical_zmq|run_ue_scene_session|run_with_twin" 2>/dev/null || true
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

产物：`outputs/offline_capture/<run>/moment_0000/smplx_result.npz`、`beta_calibration/betas.npy`，以及 `outputs/anatomy_retarget/latest_canonical/`。

**不要用** `--publish-kind smpl_pose`（72D，对 SMPL-X 会报错）。

### 同体型 UE bake（主路径，必须）

```bash
source env.sh
# 把 <run> 换成最新 offline_capture 目录名
SHAPES=outputs/offline_capture/<run>/moment_0000/beta_calibration/betas.npy
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
