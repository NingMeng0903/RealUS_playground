# Multi-Camera Extrinsic Calibration

A brand-agnostic multi-camera calibration toolbox targeting RealSense first but
built around a generic `CameraDevice` abstraction so new drivers (industrial
cameras, etc.) can be added with a single file.

Three calibration stages:

- **Stage 0 — Intrinsics (optional).** Per-camera pinhole + distortion via
  chessboard. If skipped, the factory intrinsics from each RealSense's EEPROM
  are used.
- **Stage 1 — Relative extrinsics.** Multiple cameras look at a moving
  AprilTag board; joint bundle adjustment recovers each camera's SE(3) pose
  relative to the reference camera.
- **Stage 2 — World alignment.** Three phases define the world frame:
  1. **Robot (hand-eye)** — EE 4×4 AprilTag board on the gripper. Shared-memory
     `rm75_state` supplies `T_railbase_tcp`. Bundle adjustment recovers
     `T_ref_railbase` and `T_tcp_board`. World +X is the URDF rail axis
     (`R[:,1]`); +Z is `base_link` +Z orthogonalized against the rail so the
     rail is strictly horizontal. The floor is the plane through `base_link`
     minus `base_link_height_above_floor_m` (274 mm, tape-measured).
  2. **Bed plane** — large board on the bed; parallel-plane height `z_bed`
     (normal locked to the robot-derived floor normal).
  3. **Bed corners** — four captures, one large-board placement per physical
     bed corner (**any rotation** is fine); outer-corner tags **151 / 1 / 162 / 12**
     are fused across ≥3 cameras; a **minimum-area rectangle** gives bed size
     and `bed_rotation_deg`. World origin is the bed-center projected onto the
     floor. XY stay rail-aligned (`align_xy_to_bed: false`); the bed may be skewed.

  Re-calibrating the robot phase **cascades** and clears bed + corners data.

## Quick start

```bash
# One-time env setup (outside the workspace):
conda create -y -p /media/camp/EXT_DRIVE/envs/camera_calib python=3.10
conda activate /media/camp/EXT_DRIVE/envs/camera_calib
pip install -r camera_calibration/requirements.txt

# Bind serials to stable cam1..camN aliases (permanent per serial):
python camera_calibration/scripts/discover_cameras.py

# Print the physical target used by every capture step:
python camera_calibration/scripts/generate_board_pdf.py \
    --out camera_calibration/calibration_results/board.pdf

# Launch the calibration UI:
python camera_calibration/scripts/run_ui.py

# After Stage 2 corners export, merge into one Genesis handoff file:
python camera_calibration/scripts/export_genesis_calibration.py
# → calibration_results/genesis_bundle.yaml
```

The launcher scripts export `PYTHONNOUSERSITE=1` before importing anything, so
the user-site NumPy 1.x cv2 / older PyQt5 in `~/.local` cannot pollute the env.

## Repository layout

```
camera_calibration/
├── configs/
│   ├── cameras.yaml            # serial -> alias, permanent binding
│   ├── board.yaml              # AprilTag grid geometry
│   ├── world.yaml              # Stage 2 robot/bed/corners thresholds + corner tag IDs
│   ├── board_ee.yaml           # 4×4 EE AprilTag board (robot phase)
│   ├── robot.yaml              # 274 mm floor anchor, SHM name, stillness gates
│   └── app.yaml                # stream + sync + detector + BA parameters
├── calibration_results/        # canonical outputs — always overwritten in place
│   ├── intrinsics.yaml
│   ├── extrinsics_rel.yaml
│   ├── extrinsics_world.yaml
│   ├── world_meta.yaml         # bed size, heights, origin (Stage 2)
│   └── genesis_bundle.yaml     # unified Genesis handoff (auto on corners export)
├── data/                       # per-run session dumps (gitignored)
├── scripts/
│   ├── discover_cameras.py     # enumerate + bind serials to aliases
│   ├── run_ui.py               # PyQt5 UI entrypoint
│   ├── generate_board_pdf.py   # printable calibration target
│   └── verify_calibration.py   # depth cloud fusion for visual QA
└── src/multicam_calib/
    ├── devices/                # CameraDevice abstract + realsense impl + registry
    ├── board/                  # AprilTag board geometry + detector wrapper
    ├── calib/                  # PnP + pose graph + bundle adjustment + world align
    ├── recording/              # per-session capture, threaded streams, sync gating
    ├── io/                     # yaml (de)serialisation for configs and results
    └── ui/                     # PyQt5 main window, live view grid, stage panels
```

## `configs/cameras.yaml` — serial ↔ alias binding

`discover_cameras.py` writes this file the first time. Every serial is bound
permanently to a `camN` alias by discovery order (smallest free N). Once the
binding exists, **the same physical camera keeps the same alias forever** even
if you unplug it, change USB port, or hot-plug it into a hub — because we look
up by serial, never by port.

You may rename `alias: cam1` to `alias: cam_front_left` (or anything) after
the initial capture; downstream code reads whatever is in the yaml.

New camera brands need a new driver in `src/multicam_calib/devices/`. Each
driver subclasses `CameraDevice`, implements `open/close/read/factory_intrinsics`,
and calls `registry.register(name, ...)` at import time. Then add the driver
name to your entry in `cameras.yaml`.

## Adjusting the board

Edit `configs/board.yaml` if you print a different target. The generator
supports any rows/cols but currently only the `tag36h11` family (OpenCV's
built-in AprilTag dictionary). Adding another family means either adding an
external renderer or bundling that family's images.

The board frame convention used by the algorithm:

- Origin at the board's geometric center.
- +X: along the columns (row 0 col 0 → row 0 col cols-1).
- +Y: perpendicular to X in the board plane, pointing "up" (from row rows-1
  toward row 0).
- +Z: out of the front face; right-handed.

## Algorithms and references

The choices below are motivated by peer-reviewed methods; nothing is
home-brewed.

- **Full-corner planar PnP per view.** Every visible tag on the board
  contributes all four of its sub-pixel-refined corners to a single
  `cv2.solvePnP` call using `SOLVEPNP_SQPNP` — the current state of the art
  for planar targets, immune to the two-fold ambiguity that trips up
  `SOLVEPNP_ITERATIVE`.
  Reference: G. Terzakis, M. Lourakis, *"A Consistently Fast and Globally
  Optimal Solution to the Perspective-n-Point Problem"*, ECCV 2020.

- **Multi-camera relative-pose initialisation.** For every frame where camera
  `i` and camera `j` both see the board, we get an estimate of
  `T_i_j = T_i_board · T_j_board^{-1}`. These pairwise SE(3) samples are then
  averaged via chordal L2 mean rotation + arithmetic mean translation.
  Reference: A. Chatterjee, V. M. Govindu, *"Robust Relative Rotation
  Averaging"*, IEEE PAMI 2018.

- **Joint bundle adjustment.** All non-reference camera SE(3)s and all
  per-frame board SE(3)s are optimised simultaneously by
  `scipy.optimize.least_squares` with a Cauchy loss (robust to outlier
  detections). The residual is the 2-D reprojection error of every observed
  corner. This is the same formulation used by Kalibr for its multi-camera
  extrinsic calibration.
  References:
  - B. Triggs et al., *"Bundle Adjustment — A Modern Synthesis"*, Vision
    Algorithms 1999.
  - T. Svoboda, D. Martinec, T. Pajdla, *"A Convenient Multi-Camera
    Self-Calibration for Virtual Environments"*, Presence 2005.
  - P. Furgale, J. Rehder, R. Siegwart, *"Unified Temporal and Spatial
    Calibration for Multi-Sensor Systems"*, IROS 2013.

- **AprilTag detection.** `pupil_apriltags` (a maintained Python wrapper for
  AprilTag 3) with `refine_edges=1` for sub-pixel corners.
  Reference: M. Krogius, A. Haggenmiller, E. Olson, *"Flexible Layouts for
  Fiducial Tags"*, IROS 2019.

- **Single-camera intrinsics (Stage 0).** OpenCV's
  `findChessboardCornersSB` (Duda & Frese, BMVC 2018) followed by
  `calibrateCamera` (Zhang, PAMI 2000).

The historic `easy_handeye-master` project referenced during design lives at
`../easy_handeye-master`. Only its rqt sample-panel interaction (capture /
delete / clear / run buttons + sample list) informed the UI layout; its
`cv2.calibrateHandEye`-based algorithms solve a different problem (AX=XB
hand-eye) and are not used here.

## Coordinate outputs

- `extrinsics_rel.yaml`: dictionary `alias -> T_ref_cam (4×4)`. The reference
  is the first alias in `cameras.yaml` at the time Stage 1 ran (usually
  `cam1`).
- `extrinsics_world.yaml`: dictionary `alias -> T_world_cam (4×4)`. World +Z is
  the robot-derived floor normal; +X is the rail axis; origin is bed center
  projected to floor (`origin_mode`). With `align_xy_to_bed: false` (default)
  the bed may carry a nonzero `bed_rotation_deg`.
- `robot_world.yaml`: `T_world_railbase`, `T_world_baselink_at_rail0`
  (`T_world_railbase @ translate([0,-0.4,0])`), `T_tcp_board`, rail direction,
  and hand-eye diagnostics. Export a twin overlay with
  `scripts/export_twin_rail_calib.py`.
- `world_meta.yaml`: bed envelope (`bed_size_m`, `bed_rotation_deg`,
  `bed_outer_rect_xy`, `bed_center_world`, `bed_center_on_floor` — all in the
  **final** world frame, origin at bed center on floor), plane residuals, and
  `bed_xy_skew_deg_pre_align` (bed yaw vs world +X).
- `genesis_bundle.yaml`: **single Genesis handoff file** — merges intrinsics,
  world extrinsics, bed geometry, and quality metrics under fixed top-level
  sections (`metadata`, `world_frame`, `bed`, `cameras`). The `cameras` block
  matches `genesis_ue_sync` `load_calibration_bundle()` expectations.

Every result yaml includes a `metadata` block with the total reprojection
RMSE, per-camera RMSE, and how many frames contributed.

## Stage 1 workflow (UI)

**Session storage:** same `working/` + `last/` scheme as Stage 2, under
`data/stage1_extrinsics/`:

- `working/` — current captures (cleared on every UI restart; panel starts empty)
- `last/` — previous completed capture session (optional **Load last session**;
  overwritten with `working/` when **Run calibration** succeeds)

Capture ≥8 frames of the board seen by as many cameras as possible, use
**Delete selected** to drop individual bad frames (e.g. a frame with an
outlier board pose) without recapturing everything, then **Run calibration**.

## Stage 2 workflow (UI)

Prerequisite: Stage 1 complete with acceptable RMSE (`extrinsics_rel.yaml`).

**Session storage:** only two folders under `data/stage2_world/`:

- `working/` — current captures (cleared on every UI restart; panel starts empty)
- `last/` — previous completed calibration (optional **Load last session**; overwritten when corners export succeeds)

1. Open **Stage 2: World Alignment**. The panel starts empty — capture directly.
2. Optional: click **Load last session** to restore the previous `last/` data.
3. Select **Ground plane**, capture ≥3 positions, **Run: Ground plane**.
4. Select **Bed plane**, capture ≥3 positions, **Run: Bed plane**.
5. Select **Bed corners**, capture 4 valid positions, **Run: Bed corners** → writes
   `extrinsics_world.yaml` and archives `working/` → `last/`.

## Sync considerations

For static calibration (board stationary at capture time) we do not need
hardware synchronisation. Each camera streams in its own background thread and
the UI captures the **latest** frame from every camera when you press "Space";
if the host-timestamp spread exceeds `configs/app.yaml → sync.max_spread_ms`
(default 50 ms) the sample is rejected.

For dynamic scenarios that need hardware-level sync, RealSense D435/D435i
support a genlock cable via `Multi-cam sync` pin — that is out of scope for
this calibration tool but the `realsense` driver here does not preclude it.
