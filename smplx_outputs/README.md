# Experiment capture output (remote GUI / Window 8)

Each run: `smplx_outputs/<YYYYMMDD_HHMMSS>/moment_0000/`

| Subfolder | Content |
|-----------|---------|
| `images_raw/` | 4-camera original PNG at capture instant |
| `skeleton_2d/` | DWPose 2D per camera |
| `skeleton_fused/` | Red/gray/green fused 2D+3D per camera |
| `skeleton_3d_repro/` | Triangulated 3D reprojected on image |
| `overlays/` | SMPL-X mesh reprojection per camera |
| `panels/` | Combined review panels |
| `smplx_result.npz` | Fit result |
| `moment.json` | Quality gate + timing |

Launch remote button:

```bash
source env.sh
$PY perception/apps/run_capture_remote_gui.py
```

Requires Cam publisher + Genesis viewer G running.
