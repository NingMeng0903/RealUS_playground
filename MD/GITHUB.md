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

## Subsequent pushes

```bash
git add .
git commit -m "备份"
git push -u origin main
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
