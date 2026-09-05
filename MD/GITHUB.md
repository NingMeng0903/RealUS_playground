# RealUS_playground GitHub Notes

Repository root: `/media/camp/EXT_DRIVE/RealUS_playground`  
Remote: `https://github.com/NingMeng0903/RealUS_playground.git`

## Active layout

- `rm75_control/`: RM75 8-DOF joint admittance WBC, Genesis viewer/twin, parametric slider rail
- `camera_calibration/`: 4-camera + bed extrinsics/intrinsics UI, `genesis_bundle.yaml` export
- `MD/`: operator notes (`RUN_8DOF.md`, this file)
- `scripts/`: workspace-level helpers (optional)

Each package is **self-contained** (`env.sh`, configs, data). No symlinks between packages.

## First push scope

Commit:

- `rm75_control/` (code, configs, small `data/`, URDF/meshes)
- `camera_calibration/` (`src/`, `scripts/`, `configs/`, `calibration_results/*.yaml`)
- `MD/`
- `README.md`

Do not commit:

- `camera_calibration/data/` (raw capture sessions, ~140MB+)
- `rm75_control/data/force_compensation/logs/*.npz`
- `rm75_control/**/.genesis_urdf_cache/`
- `**/.cuda_shim/`
- large media: `*.mp4`, `*.zip`, `*.pdf` (except small operator PDFs if needed)
- checkpoints: `*.pt`, `*.pth`, `*.npz`, `*.pkl`
- `.env`, secrets, local editor caches

## Git bootstrap

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
git init
git add .
git status
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:NingMeng0903/RealUS_playground.git
# HTTPS (needs interactive login): https://github.com/NingMeng0903/RealUS_playground.git
git push -u origin main
```
cd /media/camp/EXT_DRIVE/RealUS_playground
git status
git add .
git commit -m "这是很重要的一个版本,记录了单独轴力控制和hover模式，好好分析，可以掉多个grok.首先，单独轴在向上推的时候手感特别不柔顺，阻尼感特别强，弹跳在硬表面好了很多，还有一点，但是在软表面也有一点点，怎么回事，怎么兼顾，或者证明为什么向上的柔顺手感在我icra2027的辨识下不能成立。怎么上退手感尽量保持柔顺不超力。但是至少这个控制器还有改进的余地啊，那至少严格无源，不弹跳。还有激素havor,平移加阻尼有点大，确实不耦合了但是阻尼要求特别大，平移加扭动，或春扭动的扭曲都不小，而且扭的时候会感觉在来回晃动，不过松手是稳定的，我想6dof手感柔顺！！！。还有icra2027的的10——tn是什么，你用了吗，看起来确实能补偿大速度滞后？"
git push origin main


## Subsequent pushes

```bash
980
```

If the remote is non-empty:

```bash
git pull origin main --rebase
git push -u origin main
```

## Current command entry points

### Genesis demo viewer (GPU, Among_US genesis env)

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/rm75_control
source env_viewer.sh
python -m rm75_control.control.joint_admittance_8dof.viewer.demo --show-viewer
```

Auto-loads `../camera_calibration/calibration_results/genesis_bundle.yaml` (4 cameras + bed + ground Z=0).

### Robot controller (rm75 env)

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/rm75_control
source env.sh
python apps/joint_admittance_8dof/run_joint_admittance.py \
  --config configs/joint_admittance_8dof.yaml
```

### Digital twin (genesis env)

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/rm75_control
source env_viewer.sh
python apps/joint_admittance_8dof/run_with_twin.py
```

### Camera calibration UI

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/camera_calibration
source env.sh
python scripts/run_ui.py
```

Export / verify calibration:

```bash
python scripts/export_genesis_calibration.py
python scripts/verify_calibration.py
```

## Package data flow

```
camera_calibration/calibration_results/genesis_bundle.yaml
        │
        └─► rm75_control viewer (demo / twin)  auto-discovered
              • 4 cameras: extrinsics + intrinsics FOV
              • ground Z=0 (calibration world frame)
              • bed: opaque light-blue box (origin = bed center on floor)
```

Robot placement: `rm75_control/.../config/slider_rail.yaml` → `world_calib` only.

## Notes

- `README.md` is the primary GitHub landing page.
- Viewer uses **Among_US `genesis` env** via `rm75_control/env_viewer.sh`, not `rm75` env.
- `bed.center_on_floor` in bundle is pre-origin metadata; viewer places bed at world `(0,0)`.
- Use Git LFS only for unavoidable large meshes; keep raw capture sessions out of Git.
