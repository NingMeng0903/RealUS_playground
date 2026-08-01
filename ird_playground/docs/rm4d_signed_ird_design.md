# RM75 Signed IRD Operator

## Scope

This is the first-stage reachability operator only. It does not optimize a
patient trajectory, select a continuous IK branch, or replace final IK and
collision validation.

The operator answers a narrower question efficiently:

> For a medically valid TCP pose and a rail-induced base pose, what smooth
> direction increases collision-checked arm reachability, including under a
> local position/orientation uncertainty region?

## Why The Previous Model Was Removed

The removed model learned an independent full-pose classifier, margin head,
and quality head from 9-D rotation-6D features with a 3-D hash grid. Held-out
boundary direction accuracy was poor even after increasing the dataset. This
was a representation problem, not a GPU sampling problem:

- a classifier is not a distance field and can have flat or misleading
  gradients outside the feasible set;
- full `SE(3)` wastes two nearly symmetric dimensions for RM75;
- a sparse high-resolution hash grid memorized sampled boundary locations;
- the old NumPy Region A broke the autograd chain.

The same gradient limitation is explicitly discussed by Murooka et al.: a
classification decision function can become nearly constant outside the
reachable set, while a signed distance function has a useful closest-boundary
gradient.

## Representation

RM75 has J1 limits of approximately `+/-178 deg` and J7 limits of
approximately `+/-360 deg`. A 2,000-configuration audit with 16 J7 values per
configuration found numerically zero TCP position/axis change and only 0.05%
self-collision-label variation. This supports the RM4D symmetry assumptions for
the primary guidance field.

Instead of RM4D's pole-sensitive `atan2` coordinates, the implementation uses
a smooth redundant embedding of the intrinsic 4-D quotient space:

```text
[p_z,
 approach_z,
 ||p_xy||,
 p_xy dot approach_xy,
 p_xy cross approach_xy]
```

These features are invariant to common base yaw and TCP axial roll. Probe-roll
collision differences remain a rare approximation error and must be rejected
by final multi-seed IK plus collision validation.

## Training Target

The model has one output: signed reachability clearance. Positive is reachable
and negative is unreachable. There are no classifier, margin, or quality heads.

Training uses:

- balanced global sign supervision;
- exact collision-checked `+/-1/3/6/10 mm` and `+/-1/3/5 deg` stencil values;
- explicit zero-valued bisected boundary centers;
- boundary normal alignment.

An Eikonal term was tested and disabled. Workspace-normalized coordinates mix
meters and orientation invariants, so a unit gradient norm was not physically
well-defined and dominated the useful signed-value loss. Direction and scale
are instead supervised directly by collision-checked local stencils.

## Region A

Region A is a query operation over the point field, not another learned model.
It uses one fixed joint 5-D Sobol sequence:

```text
[local tangent, local binormal, local normal, cone radius, cone azimuth]
```

The default conservative position box is `b=4 mm, t=3 mm, n=2 mm` in the
medical TCP frame `[b,t,n]`; the orientation cone half-angle is `3 deg`;
`K=64`. A normalized soft minimum preserves the value of
a constant field. Pose perturbation, base/rail transforms, canonical mapping,
field evaluation, and aggregation all remain in one Torch graph.

## Production Evidence

Source data:

- 500,000 collision-checked poses sampled around collision-free FK centers;
- 500,000 independent uniform full-workspace poses, including 20% horizontal-
  probe-like orientations;
- 642,074 local boundary stencil poses;
- 91,738 held-together boundary groups;
- 1,733,608 canonical rows in total: 1,000,000 global rows and 733,608
  boundary/zero-boundary rows.

Held-out acceptance metrics:

| Metric | Result |
| --- | ---: |
| Balanced accuracy | 97.64% |
| Reachable recall | 97.64% |
| Unreachable specificity | 97.65% |
| Position direction agreement | 99.96% |
| Rotation direction agreement | 99.97% |
| Position strict `+/-1 mm` straddle | 53.58% |
| Position wide straddle | 97.14% |
| Rotation strict `+/-1 deg` straddle | 93.79% |
| Rotation wide straddle | 99.07% |
| Position bracketed crossing MAE / P95 | 0.453 / 0.928 mm |
| Rotation bracketed crossing MAE / P95 | 0.191 / 0.601 deg |
| Region A direction agreement, position / rotation | 99.80% / 100% |
| Rail AD-vs-FD median relative error | 0.0098% |
| TCP-x AD-vs-FD median relative error | 0.0101% |
| Independent positive `q_best` collision failures (10,000 audit) | 0 |
| Independent positive pose-tolerance failures (10,000 audit) | 0 |
| J7 roll collision-label variation (512 x 12 audit) | 0% |

The original 0.1 mm aspiration was not achieved. The demonstrated held-out
position boundary error is sub-millimeter when bracketed, with P95 below 1 mm.
Strict `+/-1 mm` zero straddling is 53.6%. This metric asks whether predictions
at both ends of a 2 mm bracket have opposite signs; it is not absolute position
accuracy. The field is accepted as an optimization-direction operator, not as
a sub-millimeter feasibility certificate.

### Scale and independent visualization

A fixed 20-epoch scale ablation showed that reducing the original data to
one-half or one-quarter degraded both balanced accuracy and boundary coverage.
The 500,000-row original scale reached 93.45% balanced accuracy and 94.16%
position wide-straddle coverage. Adding 500,000 independently uniform poses
was more valuable than duplicating the original FK-neighborhood distribution:
the current model reaches 97.64% balanced accuracy and 97.14% position
wide-straddle coverage. Therefore, the current 1.73M-row dataset is the minimum
recommended production split. Expanding toward 2M global poses and roughly
200k boundary groups is reasonable only if a new held-out ablation shows a
material gain; sample count alone does not fix coverage bias.

The horizontal-probe IRD visualization is independent of the training rows. A
`31^3` full-pose grid is solved with multi-seed GPU IK and robot-plus-probe
self-collision, then the neural field is evaluated at the same positions. On
29,791 points the neural result has 99.17% accuracy, 98.88% balanced accuracy,
0.81% false-positive rate, and 1.43% false-negative rate. The gradient slice
plots autograd of neural clearance with respect to base position; black is the
neural zero boundary and green dashed is the collision-checked GT boundary.
This validates neural IRD for smooth guidance. Hard feasibility still requires
final IK and collision checking.

## Reproduction

```bash
source env.sh
python -m ird_playground.cli.build_gpu_pose_gt --config configs/gpu_pose_gt_production.yaml
python -m ird_playground.cli.build_gpu_boundary_stencil --config configs/gpu_boundary_stencil_production.yaml
python -m ird_playground.cli.build_uniform_pose_gt --config configs/gpu_uniform_pose_production.yaml
python -m ird_playground.cli.build_canonical_gt --config configs/rm4d_signed_production.yaml
python -m ird_playground.cli.train_signed --config configs/rm4d_signed_production.yaml
python -m ird_playground.cli.eval_signed --config configs/rm4d_signed_production.yaml
python -m ird_playground.cli.viz_signed_ird
```

## References

- F. Zacharias, C. Borst, S. Wolf, and G. Hirzinger, "The capability map: A
  tool to analyze robot arm workspaces," International Journal of Humanoid
  Robotics, 2013.
- N. Vahrenkamp, T. Asfour, and R. Dillmann, "Robot placement based on
  reachability inversion," ICRA 2013.
- M. Rudorfer, "RM4D: A Combined Reachability and Inverse Reachability Map for
  Common 6-/7-axis Robot Arms by Dimensionality Reduction to 4D," ICRA 2025,
  [arXiv:2410.06968](https://arxiv.org/abs/2410.06968).
- M. Murooka et al., "Learning Differentiable Reachability Maps for
  Optimization-based Humanoid Motion Generation," Humanoids 2025,
  [arXiv:2508.11275](https://arxiv.org/abs/2508.11275).
- S. L. Chiu, "Task Compatibility of Manipulator Postures," International
  Journal of Robotics Research, 1988.
- M. Koptev, N. Figueroa, and A. Billard, "Neural Joint Space Implicit Signed
  Distance Functions for Reactive Robot Manipulator Control," RA-L 2023,
  DOI `10.1109/LRA.2022.3227860`.
