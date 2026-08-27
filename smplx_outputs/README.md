# Experiment capture output (remote GUI / Window 8)

Each run: `smplx_outputs/<YYYYMMDD_HHMMSS>/moment_0000/`

| Subfolder | Content |
|-----------|---------|
| `images_raw/` | 4-camera original PNG at capture instant |
| `skeleton_2d/` | DWPose 2D per camera |
| `skeleton_fused/` | Red/gray 2D plus green fused Body25 + both 21-joint hands |
| `skeleton_3d_repro/` | Triangulated Body25 + both hands reprojected on image |
| `overlays/` | SMPL-X mesh reprojection per camera |
| `panels/` | Combined review panels |
| `smplx_result.npz` | Fit result |
| `moment.json` | Quality gate + timing |

Trigger (same job):

```bash
source env.sh
# Xbox Y on Window C (genesis: TensorRT/CUDA DWPose)
$PY -m peirastic.apps.gamepad
# or the remote button
$PY perception/apps/run_capture_remote_gui.py
```

Requires Cam publisher + Genesis viewer G running.
