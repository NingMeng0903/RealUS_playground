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
  --width 1920 --height 1080 --fps 30
```

首次缺 `pyrealsense2` 时：`pip install -r perception/requirements.txt`（或 `$REALUS_CAMERA_PY` 会自动回退到 `envs/camera_calib`）。

相机数 = `genesis_bundle.yaml` 全部 alias（扩展相机只改标定，不写死 4）。

直播不要加 `--undistort`（四路 1080p remap 会掉到 ~20 fps）。Xbox Y / Window 8 只对抓到的 8–12 组去畸变。

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

**view（Phase 1 默认）** — 不连真机、不读 SHM；显示床+滑轨+**静态 demo 机械臂** + 5598/5601 人体 overlay：

```bash
cd rm75_control && source env_viewer.sh && cd "$REALUS_PROJECT_ROOT"
$PY perception/apps/run_genesis_perception_viewer.py --anatomy-alpha 0.3
```

**twin** — 窗口 A 真机运行时，Genesis 镜像 `rm75_state`（机械臂随真机动）。默认订橙色 SMPL-X（5598）和腕部点云（17358）；发布器没开就空转：

```bash
cd rm75_control && source env_viewer.sh && cd "$REALUS_PROJECT_ROOT"
python apps/joint_admittance_8dof/run_with_twin.py
```

只要机械臂：`--no-track-subscribe --no-orbbec-cloud`。解剖 / fitted 人体再加 `--canonical-human-source fitted`。

床体高度 **完全来自** `genesis_bundle.yaml` 的 `bed.height_m`（或 `support_surface.top_z_m`）：Genesis 网格从 z=0 铺到该标定顶面，无固定厚度常数。

#### 8 — 触发 SMPL-X 拟合（躺下后执行，可多次）

**推荐：Xbox Y（Window C 已开时）**

```bash
source env.sh
$PY -m peirastic.apps.gamepad
```

手柄 **Y** = Window 8 全流程：多相机抓拍 → DWPose / EasyMocap → 预览 PNG → 5598 橙色 mesh 进 Genesis。终端会打 `[CAPTURE] Y — …`。L3 / R3 仍是力混合和急停。关掉绑定：`--no-capture-y`。

**实验用远程按钮（临时）：**

```bash
$PY perception/apps/run_capture_remote_gui.py
```

弹窗点 **CAPTURE AND GENERATE SMPLX** 与 Xbox Y 同一条任务；结果在 `smplx_outputs/<timestamp>/moment_0000/`（原图、DWPose、融合点、SMPL-X 重投影 PNG）。

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

质量检查始终写入 `quality_report.json`，但人工运行默认不会因诊断失败而阻止更新或发布。
CI/自动验收需要阻断失败产物时显式增加 `--enforce-quality-gate`。

**验收：** `outputs/anatomy_retarget/latest_asset/anatomy_rigged.npz`、`limb_vessel_planning/`；G 窗口见半透明解剖 + 血管 overlay。

Stage 1 单独交付时，用采集时固定的 beta 只做一次离线调和与 target bind：

```bash
$PY perception/apps/run_anatomy_retarget.py \
  --run <tpose_or_shape_capture_run> \
  --stage1-harmonic-only \
  --enforce-quality-gate
```

产物中的 `anatomy_rigged.npz` 保存 Blender source rig、source weights、SMPL-X target bind 和 driver coupling；`stage1_runtime_contract.json` 必须通过。它不保存任何特定动作的 pose cache。新动作只输入 SMPL-X 55-joint pose 到运行时 LBS，不重新运行 Blender、不重新求调和场。

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

## Phase 2 — Genesis + UE 当前链路

Phase 1 保持不变。Phase 2 使用 UE 四机虚拟相机、Genesis 运行时和严格
benchmark recorder；虚拟相机配置固定为 cam1..cam4。每个终端都从仓库根目录
开始，并先加载同一份环境：

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
~~~

按窗口关闭程序，或使用各自的 session command。虚拟 benchmark 不需要真相机，也不需要
controller；只有要接真实机器人反馈时才启动最后的 controller role。

应用代码和当前场景、URDF、标定从 RealUS_playground 加载。大型 BEDLAM UE
项目、AMASS 数据和 Genesis 库复用现有资产路径；`env.sh` 中的
`REALUS_BEDLAM_UNREAL_ROOT`、`REALUS_AMASS_NPZ` 可以覆盖这些位置。

### 端口和发布者

| 端口 / 端点 | 用途 |
|-------------|------|
| tcp://127.0.0.1:17355 | UE capture plugin 的 TCP 输入；mux 从这里接收 UE RGB |
| tcp://127.0.0.1:17366 | UE 四机 JPEG PUB（mux 输出，preview、sequence、Terminal 8 使用） |
| tcp://127.0.0.1:17356 | 真 RealSense 四机 JPEG PUB（物理相机 path 使用） |
| tcp://127.0.0.1:17367 | UE strict benchmark capture ACK（recorder → watcher） |
| tcp://127.0.0.1:5588 | SceneInit publisher → UE prepare |
| tcp://127.0.0.1:5598 | DWPose/EasyMocap 拟合 SMPL-X track |
| tcp://127.0.0.1:5599 | 唯一 Genesis canonical publisher → UE bridge |
| tcp://127.0.0.1:5601 | Anatomy asset upsert |
| udp://127.0.0.1:5601 | canonical bridge → UE 人体/机器人状态；与上面的 TCP 端点独立 |

scripts/realus_fullflow.sh 已把虚拟输出固定为 17366、真实相机固定为 17356，
插件 TCP 固定为 17355，strict ACK 默认固定为 17367。5599 只能由一个 Genesis
runtime 占用：AMASS GT 和 fitted 人体不能同时各自启动 canonical publisher。

### 一次性 setup / plugin

setup 重建 bundle 对应的 scene spec；plugin 编译并安装 UE capture plugin。两者各
执行一次：

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh setup
~~~

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh plugin
~~~

### 九个窗口：UE 四机 → Genesis → UE

先按 1–4 开窗口，再在 5 中执行 T-pose prepare；prepare 成功退出后，用同一
窗口的 play 命令让 UE 启动 PIE，随后开 6、7。8 用于 RGB 重建；9 为可选解剖
资产扩展，录制动作捕捉 benchmark 不需要它。

**窗口 1 — UE watcher**

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh 1
~~~

**窗口 2 — UE TCP 17355 → PUB 17366 mux**

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh 2
~~~

**窗口 3 — 四机预览**

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh 3
~~~

**窗口 4 — 8-DOF SceneInit publisher**

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh 4
~~~

**窗口 5 — T-pose prepare，然后启动 PIE**

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh 5
# prepare 收到严格成功 ACK、命令退出后，在 UE editor 已就绪时执行：
bash scripts/realus_fullflow.sh play
~~~

**窗口 6 — canonical → UE bridge**

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh 6
~~~

**窗口 7 — Genesis simulation / AMASS GT**

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh 7 --human-source amass
~~~

需要 fitted 人体时，停止上面的 AMASS runtime，改为唯一的 fitted runtime：

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh 7 --human-source fitted
~~~

同时只保留一个 5599 publisher；human-source fitted 读取 5598 的拟合 track。
不要让 AMASS GT runtime 和 fitted runtime 同时发布 5599。

**窗口 8 — UE RGB burst → DWPose / EasyMocap SMPL-X**

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh 8
~~~

窗口 8 使用 UE source profile 和零畸变 metadata 内参；输出 SMPL-X moment 并可
发布到 5598。它不需要真相机。连续拟合和严格录制离线拟合见下文。

**窗口 9 — anatomy retarget / vessel export**

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh 9
~~~

### 真实 C++ controller（单独可选）

没有真实机器人时不要启动 controller。要让 Genesis 读取 controller 写入的
rm75_state，先在独立终端启动当前 native controller，再让唯一的 runtime 使用
SHM 状态：

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh controller
~~~

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh 7 \
  --robot-state-source shm \
  --human-source fitted
~~~

human-source amass 与 human-source fitted 二选一；不要再开第二个 runtime
占用 5599。

### 连续拟合和录制目录逐组离线拟合

连续 live sequence 从 17366 逐组取四机输入，保留原有 duration / interval 选项。
下面的 duration-s 0 表示持续运行，直到手动停止或达到 max-moments：

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh sequence \
  --duration-s 0 \
  --interval-s 0.5 \
  --publish-genesis
~~~

离线模式直接读取 recorder 目录的 manifest.jsonl 和每组四机 JPEG/metadata，按
manifest 顺序逐组拟合；它不通过实时 PUB 重放再跳帧：

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh sequence \
  --recording-dir "$SESSION_DIR/benchmark_recording_01" \
  --output-root "$SESSION_DIR/smplx_sequences" \
  --publish-genesis
~~~

recorded sequence 默认使用前 15 个同步组（30 fps 下的 0.5 秒）拟合一次体型，
随后冻结 beta；静态初始化失败时停止，不会逐帧重新估计 beta。可用
--betas-path <betas.npy> 显式指定体型。每个结果保留 recorder 的 frame、loop、
sim-time 和 capture-group 关联。若 fit 时不发布，可在结束后按源时间循环重发：

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
$PY perception/apps/publish_saved_smplx.py \
  --sequence "$SESSION_DIR/smplx_sequences/<run>" \
  --loop
~~~

sequence 的 --publish-genesis 与 publish_saved_smplx.py --sequence 都发布 5598，
同一时刻只运行一个。5598 的 fitted track 再由唯一的 Genesis runtime 以
human-source fitted 转成 5599 canonical。

### Benchmark GT、严格录制和视频

gt-log 是有限的 headless Genesis AMASS 日志，脚本默认写入新的
$SESSION_DIR/benchmark_gt.jsonl，达到 frame limit 后退出。若默认文件已经存在，
换一个新的 REALUS_GT_LOG，不要覆盖旧日志：

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
export REALUS_GT_LOG="$SESSION_DIR/benchmark_gt.jsonl"
bash scripts/realus_fullflow.sh gt-log
~~~

录制前暂停窗口 6 和 7，避免 bridge/runtime 的实时预览状态覆盖 recorder 请求的
canonical 帧；保留 UE watcher、PIE 和四机 mux。然后以同一份 GT JSONL 请求 UE
逐组 capture：

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
export REALUS_GT_LOG="$SESSION_DIR/benchmark_gt.jsonl"
export RECORDING_DIR="$SESSION_DIR/benchmark_recording_01"
bash scripts/realus_fullflow.sh record \
  --canonical-jsonl "$REALUS_GT_LOG" \
  --output-root "$RECORDING_DIR" \
  --session-id "$AMONGUS_SESSION_ID"_benchmark_01 \
  --ack-connect tcp://127.0.0.1:17367
~~~

每个成功组包含 frames/<capture_group_id>/ 下的四机 JPEG、metadata、canonical
state 和 gt.json。gt.json 是 UE 实际应用的 skeleton/bones（actual UE-applied
poseable skeletal mesh），用于 benchmark GT；canonical 中的 betas 只作为来源记录。

从同一个录制目录导出对齐视频：

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
export RECORDING_DIR="$SESSION_DIR/benchmark_recording_01"
bash scripts/realus_fullflow.sh video \
  --recording-dir "$RECORDING_DIR" \
  --output-dir "$RECORDING_DIR/videos" \
  --fps 30
~~~

同一目录可直接交给上面的 sequence --recording-dir 做逐组离线 DWPose /
EasyMocap fit；结果再用 publish_saved_smplx.py --sequence ... --loop 重发到 5598。

静态 RGB 初始化、四机躺卧录像、GPU EasyMocap/MV4DHMR/HybrIK 测速、GT 精度
评估和 SMPL-X 网格叠图的完整命令见 [MOCAP_BENCHMARK.md](MOCAP_BENCHMARK.md)。
strict 录制完成后运行 `bash scripts/realus_fullflow.sh resume-preview` 恢复连续图传。

### 体型与 GT 的边界

UE rendered 形体的权威来源是现有 BEDLAM 纹理 skeletal mesh。SMPL-X betas 仅
表示拟合结果和动作来源；它们不会通过某个命令自动 bake 成新的 UE rendered 体型，
也不能把 fit betas 宣称为 UE mesh shape。UE benchmark 的形体与姿态验收以实际加载
的 BEDLAM mesh 和 frames/*/gt.json 的 actual UE bones 为准。

### 当前 benchmark 质量/速率（2026-10-03）

当前主 clip 是
outputs/realus_sync/tests/diagnostics_cameras/lying_open_arms5_neutral16_v2：
五机 1920×1080、30 fps，15 组静态初始化加 75 组运动，源动作为 CMU
114/114_11_poses.npz 的 frames 1000:4:1296。所有 UE/scene/prepare/runtime 窗口在 source env.sh 后都应 export 上面的五机场景，
相机预览用 `bash scripts/realus_fullflow.sh 3 --camera-names cam1 cam2 cam3 cam4 cam5`。
角色 7 的现有播放窗口保持
frame-start 1000、frame-step 4、frame-limit 75、loop：

```bash
export REALUS_SCENE_SPEC="$REPO/configs/scenes/realus_bed_rail_5cam_scene.yaml"
bash scripts/realus_fullflow.sh 7 \
  --frame-start 1000 --frame-step 4 --frame-limit 75 --loop \
  --human-shape-npz /media/camp/EXT_DRIVE/Among_US/dataset/raw/humans/bedlam2/motions/motions_npz_training/it_4375_M_2400.npz \
  --human-body-model /media/camp/EXT_DRIVE/Among_US/ref_code_library/bedlam_clothing/support_data/body_models/smplx_locked_head/SMPLX_NEUTRAL.npz
```

质量优先重建显式使用 permanent 五机配置：
configs/scenes/realus_bed_rail_5cam_scene.yaml、
configs/tracking/realus_ue_5cam_calibration.yaml 和
configs/tracking/realus_ue_5cam_dwpose_easymocap.yaml。使用 neutral10
识别端的 fresh 静态初始化，HybrIK-X refine60、triangulation-workers 3，
不使用 VPoser、SAM2 或 learned beta prior：

```bash
bash scripts/realus_fullflow.sh sequence \
  --config configs/tracking/realus_ue_5cam_dwpose_easymocap.yaml \
  --recording-dir outputs/realus_sync/tests/diagnostics_cameras/lying_open_arms5_neutral16_v2 \
  --output-root outputs/realus_sync/tests/pose \
  --run-name lying_open_arms5_hybrik60_reg0005_cuda_v1 \
  --tracking-backend hybrik --gender neutral --fit-model smplx \
  --hybrik-device cuda --hybrik-refine-steps 60 \
  --triangulation-workers 3 --triangulation-pair-hypotheses 1 \
  --dwpose-temporal-bbox-fallback \
  --warmup-moments 15 --sample-stride 1 --max-moments 75 \
  --disable-face-iteration --no-write-debug-images
```

该 fresh run 已覆盖 75/75 帧、0 reject：body22 MPJPE 44.1030 mm，
hands30 MPJPE 11.3938 mm，骨盆误差 64.9355 mm；平均算法耗时 461.953 ms，
P95 536.806 ms，离线计算吞吐 2.1647 Hz。P95 超过 500 ms，不能据此宣称
稳定 2 Hz live FPS；没有 UE 皮肤顶点 GT，也不报告 surface PVE。正式结果见
outputs/realus_sync/tests/diagnostics_cameras/lying_open_arms5_hybrik60_reg0005_evaluation.json。
五机 75 帧已生成 375 张真实网格叠图与 75 张拼图；清单为
outputs/realus_sync/tests/diagnostics_cameras/overlays/lying_open_arms5_hybrik60_reg0005/overlay_summary.json。
五路叠图视频和 montage.mp4 在同目录 videos/，详细录制、测速与叠图导出命令见 [MOCAP_BENCHMARK.md](MOCAP_BENCHMARK.md)。

CPU DLT/重复解缓存从旧约 470 ms 降到约 211 ms，完整输出 exact parity；
fast-LBS backend parity 也已单独核对。learned beta prior 在独立 60 帧检查中
没有优于 zero-reg .0005，因此生产命令不加入该先验。渲染身份为 BEDLAM2
locked-head neutral16，识别端拟合 neutral10；root/pelvis 使用实测对齐修正。
MidHip 缺失时只在左右两个实际髋点都通过可信度和跨度检查时恢复 hip-pair 根，
否则拒绝该帧，不合成单侧根点。

### 物理相机 path（不与虚拟 mux 同时占用同一输入）

物理 RealSense publisher 使用 17356；没有真机时不要运行这两个 role：

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh camera
~~~

~~~bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
bash scripts/realus_fullflow.sh capture-real
~~~

### 当前验证状态

UE capture plugin 已实际编译安装；已严格录制 165 个四机同步组、导出并完整
解码四个 MP4。实际相机 K、世界坐标和 UE 55 骨骼 GT 已核对。RGB 拟合结果
经 CUDA Genesis → ZMQ canonical → UE UDP → 四机图传闭环通过，165D 姿态与
beta 完整保留。全流程仿真不连接真机；八自由度 SHM 反馈用隔离模拟状态验证，
尚未连接实体机器人验收。

可选 anatomy 扩展能启动并读取实际 RGB 拟合 beta，但当前模板的 surface
correspondence、长骨端点和血管/神经 containment 未通过已有质量门，因此不
更新 latest、不发布。该失败与四机视频和 SMPL-X 动作同步独立，不能写成解剖
资产已验收。具体报告在 `outputs/realus_sync/tests/anatomy/actual_rgb_init/`。
