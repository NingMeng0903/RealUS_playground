# ird_playground

Differentiable signed inverse-reachability field for the RM75 rail robot and
the physical `probe45` TCP.

## Current Contract

- Robot assets are defined by `configs/robot_probe45.yaml`.
- The rail coordinate is the raw URDF value `rail_y in [0.0, 0.8]` m.
- RM4D features are evaluated in the physical `joint_1` axis frame, not about
  the `rail_base` origin.
- Every GT set and checkpoint records kinematic URDF, collision URDF,
  collision-pair and collision-mesh SHA256 values. Old synthetic-probe
  artifacts are rejected by default.
- The scalar field is a signed reachability-margin proxy, not a metric SDF or
  a collision certificate. Final paths still require exact full-pose IK and
  collision checks.

The intrinsic quotient is 5-D, but the production network uses a smooth,
redundant 9-D flange embedding. It removes only common J1 yaw and retains
probe roll, including the calibrated probe45 offset. J1's `+/-177.96 deg`
limit makes this a task-audited approximate symmetry; final joint validation
always applies the true limit and no seam fallback is used.

## Continuous Dynamic Guidance

`ird_playground.optimization.DifferentiableTrajectoryEnergy` exposes a batched,
solver-independent trajectory energy over cubic task-spline controls
`[theta, tip_x, tip_y, roll, rail]`. The frozen IRD score and the dynamic
obstacle SDF are separate channels; obstacle updates never rewrite raw IRD.
The forward graph stays continuous, while closed-form SRS whole-path DP and
exact robot checks remain outside the guidance graph as final certification.

The GPU-only moving-obstacle demos are:

```bash
python experiments/moving_obstacle_u_band_demo.py --device cuda
python experiments/diffusion_guidance_foundation_demo.py --device cuda
python experiments/finalize_moving_obstacle_artifacts.py \
  --out-dir data/reports/moving_obstacle_u_band --clean
```

Their artifacts are written to `data/reports/moving_obstacle_u_band/`. Each
video frame re-queries a local U-band centered on the current TCP with the
current rail axis; the obstacle moves independently and supplies a red
ellipsoid-SDF halo. Only the first and last spline controls are fixed to the
medical reference. Every interior control remains differentiable, and every
optimizer step recomputes pose, orientation, rail-axis IRD, angular aggregation
and obstacle distance from the current controls. The reference supplies the
medical projection rule, not a cached heatmap or active-set mask.
The current manifest-linked GPU run passes 30 warm end-to-end trials with
P50/P95 `4.288/4.564 s`, minimum raw IRD `6.393`, and exact dense TCP-to-
ellipsoid margin `6.189 mm`. It uses a `3 mm` hard certificate, `5 mm`
planning margin, and a shorter `2 mm` transition. Its obstacle score rises
from zero to the IRD safety score across that transition, so the repulsive
gradient is strong near the edge and hands back to raw IRD outside the compact band.
The no-learning foundation recovers
`37/48` noisy proposals and reports every failed basin cell.
Production field balanced accuracy is still `90.38%`, so
these demos prove the planning/guidance architecture, not the `>=95%` field
release target.

## Rebuild

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source rm75_control/env.sh
export PYTHONPATH="$PWD/ird_playground:$PYTHONPATH"

# 1. Collision-checked full-pose labels.
python -m ird_playground.cli.build_gpu_pose_gt \
  --config configs/gpu_pose_gt_production.yaml

# 2. Signed position and orientation boundary stencils.
python -m ird_playground.cli.build_gpu_boundary_stencil \
  --config configs/gpu_boundary_stencil_production.yaml

# 3. Independent workspace negatives/positives for far-field calibration.
python -m ird_playground.cli.build_uniform_pose_gt \
  --config configs/gpu_uniform_pose_production.yaml

# 4. Transform to the J1-axis RM4D embedding.
python -m ird_playground.cli.build_canonical_gt \
  --config configs/rm4d_signed_production.yaml

# 5. Train, independently calibrate, then open the final test split once.
python -m ird_playground.cli.train_signed \
  --config configs/rm4d_signed_production.yaml
python -m ird_playground.cli.calibrate_conformal \
  --config configs/rm4d_signed_production.yaml
python -m ird_playground.cli.eval_signed \
  --config configs/rm4d_signed_production.yaml
```

Start with `gpu_pose_gt_smoke.yaml` and
`gpu_boundary_stencil_smoke.yaml` before launching the production jobs.

## Partial Tasks

`ird_playground.region.TrajectoryTaskOperator` accepts batched TCP trajectories
and separates three reductions that must not be conflated:

1. Registration/control uncertainty: robust soft-min or lower-tail CVaR.
2. Approximate task angle: selectable soft-max, robust soft-min, expectation,
   or CVaR.
3. Whole trajectory: soft-min, CVaR, or exact minimum.

The result retains per-waypoint margins, per-angle margins, all sampled
scenario scores, coverage, recoverable angle/psi choices and weights. This
is the interface intended for SMPL-X surface-path optimization; it avoids
hiding failure locations behind one trajectory scalar.

## 8-DOF trajectory planning

`ird_playground.optimization` implements an offline local SQP over
`q=[rail,j1..j7]` and explicit soft task-pose offsets. IRD ranks task/rail warm
starts; Pinocchio FK/Jacobians, ProxSuite, HPP-FCL witness-point gradients and
independent world-frame constraints determine the joint lift. Results are
fail-closed and include `T_tcp_ref`, `q_ref`, `qdot_ff`, rail reference,
contact normals, KKT residual and every hard-validation margin.

Timing starts from TCP arc length at a maximum of `0.02 m/s`; Ruckig may slow it to
meet joint/rail velocity and acceleration limits. The result feeds the existing
QP-IK/force-position controller and does not add torque dynamics to its loop.

## Mount compare figures (RM + IRD × 3)

Paper-style **reachability map** and **global IRD** for three TCP mounts
(probe45 / vertical 220 mm / horizontal probe):

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source ird_playground/env.sh
export PYTHONPATH="$PWD/ird_playground:$PYTHONPATH"
export PYVISTA_OFF_SCREEN=true

# Build missing capability maps first (from rm75_control), e.g.:
#   source rm75_control/env.sh
#   python scripts/build_coll_map.py --config configs/reachability/rm75_6f_3cm_15deg_coll_probe45.yaml
#   python scripts/build_coll_map.py --config configs/reachability/rm75_6f_3cm_15deg_coll_tcp220.yaml

python -m ird_playground.cli.viz_mount_compare \
  --config configs/mount_compare.yaml \
  --skip-missing
```

Outputs six PNGs under `data/reports/mount_compare/`:
`{probe45,tcp220,horizontal}_{reachability,ird}.png`.
