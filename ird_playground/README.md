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

The calibrated probe TCP is not aligned with J7: its origin is 15.23 mm from
the J7 axis and TCP +Z is about 49.9 degrees from it. Therefore the 5-D field
can guide position plus acoustic-axis tasks, but it cannot represent
longitudinal/transverse probe roll by itself. Use roll as an explicit task
variable and validate it with full-pose IK/collision, or train a roll-aware
residual/full-pose field.

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

# 5. Train and evaluate. Validation is grouped by source_pose_id.
python -m ird_playground.cli.train_signed \
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
scenario scores, coverage, the worst waypoint and the best angle index. This
is the interface intended for SMPL-X surface-path optimization; it avoids
hiding failure locations behind one trajectory scalar.
