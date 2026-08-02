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

The task-audited symmetry is common J1 yaw. Because J1 is limited to about
`+/-177.96 deg`, it is an approximation rather than an unconditional group
action; the unused two-degree seam is outside the scan distribution. There is
no seam-coverage gate or fallback. Final SQP/IK applies the true joint limits.

Quotienting one pose dimension leaves the intrinsic 5-D space
`SE(3)/Yaw_J1`. The production representation is a smooth redundant 9-D flange
embedding. It avoids azimuth branch cuts, remains well behaved near the J1
axis, and retains flange/probe roll. Nine dimensions are an engineering choice,
not a claim that every valid implementation must use exactly nine coordinates.

## Training Target

The model has one output: signed reachability clearance. Positive is reachable
and negative is unreachable. There are no classifier, margin, or quality heads.

Training uses:

- balanced global sign supervision;
- exact collision-checked `+/-1/3/6/10 mm` and `+/-1/3/5 deg` stencil values;
- explicit zero-valued bisected boundary centers;
- boundary normal alignment.

Empirical boundary direction and slope are supervised independently. Generic
Eikonal is disabled: a target norm of one in normalized redundant 9-D
coordinates is dimensionally incorrect. It may only be enabled after a tested
quotient-Jacobian metric derives the physical target; disabling it never
disables empirical stencil slope supervision.

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

The current immutable baseline contains 1,755,822 canonical rows and 83,984
boundary groups. Its held-out report is
`data/reports/eval_rm4d_signed.json`; these numbers are a baseline for the new
paired-sampler ablation, not a completed high-precision claim.

| Metric | Result |
| --- | ---: |
| Balanced accuracy | 90.38% |
| Reachable recall | 91.94% |
| Unreachable specificity | 88.82% |
| Position direction agreement | 99.80% |
| Rotation direction agreement | 99.58% |
| Position strict straddle | 23.91% |
| Position wide straddle | 90.79% |
| Rotation strict straddle | 10.28% |
| Rotation wide straddle | 70.44% |
| Position bracketed crossing P95 | 0.953 mm |
| Rotation bracketed crossing P95 | 0.0955 deg |
| Rail AD-vs-FD median relative error | 0.0139% |
| TCP-x AD-vs-FD median relative error | 0.0074% |
| Independent positive `q_best` collision failures (10,000 audit) | 0 |
| Independent positive pose-tolerance failures (10,000 audit) | 0 |
| J7 roll collision-label variation (512 x 12 audit) | 0% |

Direction and AD correctness are already strong. The weak metrics are zero
placement, strict straddling and near-axis coverage; crossing error only counts
already bracketed pairs and must not be presented as whole-boundary accuracy.
The field remains guidance rather than a feasibility certificate.

### Scale and independent visualization

Training now uses five source-disjoint roles, complete 8/9-row boundary groups,
on-manifold SE(3) zero poses, train-only normalization, independent zero-bias
calibration and an unreachable-only false-accept threshold. The online network
stays `192x5`; any larger teacher is distilled back to this student.

Final-test metrics are opened only after calibration and are hash-linked to the
dataset and checkpoint. Legacy checkpoints/calibration files require an
explicit stale-audit flag and cannot silently enter production evaluation.

## 8-DOF local trajectory optimisation

The planning variable is `q_i=[rail_i,j1_i,...,j7_i]`; rail has no duplicate
copy. An explicit six-dimensional task offset represents soft nominal pose and
rotation inside a hard medical envelope. IRD ranks rail/task warm starts, then
Pinocchio and ProxSuite lift them to a continuous joint path. HPP-FCL
witness-point gradients, world-frame patient/bed constraints, true limits and
adaptive segment checks determine validity. The lowest-cost local solution is
returned only when every hard validator and the scaled KKT threshold pass.

TCP arc length initially sets time at a maximum of `0.02 m/s`. Ruckig may only slow
the path for rail/joint limits. The output supplies pose and joint references,
feedforward velocities, rail and contact normal to the existing QP-IK and
force-position controller; torque dynamics are outside v1.

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
