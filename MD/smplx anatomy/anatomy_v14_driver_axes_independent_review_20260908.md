# V14 collar + driver-axes independent review — 2026-09-08

This review covers the combined `v14_collar_driver_axes` exports for both
captures. The numerical audit is read-only: no fit or tuning was performed.
The Genesis comparisons put `before_motion_repair` on the left and
`candidate` on the right.

Machine-readable audits:

- [213328 arm audit](../outputs/anatomy_retarget/v14_driver_axes_arm_audit_213328_20260908_001/report.json)
- [213712 arm audit](../outputs/anatomy_retarget/v14_driver_axes_arm_audit_213712_20260908_001/report.json)
- [Genesis manifest](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/manifest.json)

The two input packages record `rest_refit_performed=false`,
`rotation_transport=driver_axes`, `inherits_beta_vertex_basis=false`,
unchanged source faces and weights, and bit-exact target rest geometry. The
response is the baked collar/local-rotation and joint-13 pivot basis. The
audit remains evidence-only with `anatomical_passed=false` and
`publishable=false`.

## Genesis images opened

I opened the candidate contact sheet and the comparison images for whole AP,
left collar AP, left elbow lateral, pelvis AP, left hip oblique, left and
right knee lateral, and left and right ankle oblique in all six cells.

| cell | contact sheet | upper/collar | pelvis/hip | knees | ankles |
|---|---|---|---|---|---|
| 213328 T | [sheet](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_tpose/candidate/contact_sheet.png) | [elbow](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_tpose/comparison/left_elbow_lateral.png), [collar](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_tpose/comparison/left_collar_ap.png) | [pelvis](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_tpose/comparison/pelvis_ap.png), [hip](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_tpose/comparison/left_hip_oblique.png) | [L](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_tpose/comparison/left_knee_lateral.png), [R](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_tpose/comparison/right_knee_lateral.png) | [L](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_tpose/comparison/left_ankle_oblique.png), [R](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_tpose/comparison/right_ankle_oblique.png) |
| 213328 self | [sheet](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213328/candidate/contact_sheet.png) | [elbow](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213328/comparison/left_elbow_lateral.png), [collar](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213328/comparison/left_collar_ap.png) | [pelvis](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213328/comparison/pelvis_ap.png), [hip](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213328/comparison/left_hip_oblique.png) | [L](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213328/comparison/left_knee_lateral.png), [R](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213328/comparison/right_knee_lateral.png) | [L](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213328/comparison/left_ankle_oblique.png), [R](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213328/comparison/right_ankle_oblique.png) |
| 213328 swap | [sheet](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213712/candidate/contact_sheet.png) | [elbow](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213712/comparison/left_elbow_lateral.png), [collar](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213712/comparison/left_collar_ap.png) | [pelvis](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213712/comparison/pelvis_ap.png), [hip](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213712/comparison/left_hip_oblique.png) | [L](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213712/comparison/left_knee_lateral.png), [R](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213712/comparison/right_knee_lateral.png) | [L](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213712/comparison/left_ankle_oblique.png), [R](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213712/comparison/right_ankle_oblique.png) |
| 213712 T | [sheet](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_tpose/candidate/contact_sheet.png) | [elbow](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_tpose/comparison/left_elbow_lateral.png), [collar](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_tpose/comparison/left_collar_ap.png) | [pelvis](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_tpose/comparison/pelvis_ap.png), [hip](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_tpose/comparison/left_hip_oblique.png) | [L](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_tpose/comparison/left_knee_lateral.png), [R](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_tpose/comparison/right_knee_lateral.png) | [L](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_tpose/comparison/left_ankle_oblique.png), [R](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_tpose/comparison/right_ankle_oblique.png) |
| 213712 self | [sheet](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213712/candidate/contact_sheet.png) | [elbow](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213712/comparison/left_elbow_lateral.png), [collar](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213712/comparison/left_collar_ap.png) | [pelvis](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213712/comparison/pelvis_ap.png), [hip](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213712/comparison/left_hip_oblique.png) | [L](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213712/comparison/left_knee_lateral.png), [R](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213712/comparison/right_knee_lateral.png) | [L](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213712/comparison/left_ankle_oblique.png), [R](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213712/comparison/right_ankle_oblique.png) |
| 213712 swap | [sheet](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213328/candidate/contact_sheet.png) | [elbow](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213328/comparison/left_elbow_lateral.png), [collar](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213328/comparison/left_collar_ap.png) | [pelvis](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213328/comparison/pelvis_ap.png), [hip](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213328/comparison/left_hip_oblique.png) | [L](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213328/comparison/left_knee_lateral.png), [R](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213328/comparison/right_knee_lateral.png) | [L](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213328/comparison/left_ankle_oblique.png), [R](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213328/comparison/right_ankle_oblique.png) |

## Numerical gates

The complete connected left-arm skin population has these candidate maxima.
The values use the audit's 1 mm reporting threshold for the count column.

| subject/state | max skin outside | p95 outside | vertices >1 mm |
|---|---:|---:|---:|
| 213328 T | 0.000 mm | 0.000 mm | 0 |
| 213328 self | 11.625 mm | 8.119 mm | 4,682 |
| 213328 swap | 31.629 mm | 24.309 mm | 9,417 |
| 213712 T | 0.000 mm | 0.000 mm | 0 |
| 213712 self | 28.899 mm | 20.664 mm | 8,371 |
| 213712 swap | 10.648 mm | 6.613 mm | 4,397 |

Complete VTK triangle-pair checks (pair count / audit result) are:

| subject/state | Humerus–Radius | Humerus–Ulna | Radius–Ulna |
|---|---:|---:|---:|
| 213328 T | 19 / fail-contact-review | 0 / pass | 83 / fail-penetration |
| 213328 self | 0 / pass | 0 / pass | 83 / fail-penetration |
| 213328 swap | 84 / fail-penetration | 72 / fail-penetration | 83 / fail-penetration |
| 213712 T | 28 / fail-contact-review | 0 / pass | 83 / fail-penetration |
| 213712 self | 85 / fail-penetration | 66 / fail-penetration | 83 / fail-penetration |
| 213712 swap | 12 / fail-contact-review | 0 / pass | 83 / fail-penetration |

The persistent T-pose Radius–Ulna crossings have sampled lower-bound depths
up to approximately 2.20 mm. They are an intrinsic rest-bone placement
failure, not an image-only observation. The active-pose rows add humerus–
forearm crossings and large skin-boundary excursions.

## Cap shape provenance

The candidate-versus-frozen-operator-template cap comparison applies the
authorized **0.1 mm** shape gate. All six caps pass for each subject; the
largest candidate residual is 0.006758 mm (213712 left humerus), below the
gate. This is the total authored-142 cap check because the compiled
provenance is `shape_reference_kind=frozen_operator_template`.

The raw subject-materialization diagnostic is separate. For 213328 its
raw-versus-template maximum reaches 0.3034 mm, so that raw diagnostic fails
the 0.1 mm threshold before the candidate map is considered. For 213712 the
raw diagnostic remains below the threshold. Both package reports explicitly
state `inherits_beta_vertex_basis=false`; the raw 213328 discrepancy is not
silently counted as a candidate cap pass.

## Visual findings

In both T-pose render pairs, the pelvis, femoral heads, knee surfaces, ankle
blocks, and elbow caps appear in the same gross locations on the two sides of
the comparison. The femoral heads look seated in the acetabular region, and
the distal tibia/foot chains remain visually connected. The elbow images show
the distal humerus and proximal forearm close together, with vessels running
along the joint. The narrow gaps and occlusions cannot certify surface
clearance; the T-pose triangle results above show why the rendered appearance
alone is insufficient.

For 213328 self and swap, the whole-body and local images preserve the broad
pose, while the detailed elbow and forearm views retain crowded vessel/bone
corridors. The numerical skin errors are large even where no gross
dislocation is obvious at whole-body scale. The 213712 self and swap images
show the same limitation: hip, knee, and ankle chains look connected in the
render, but the active-pose triangle and skin checks still fail. The collar
and driver-axes basis therefore preserves a plausible gross layout without
meeting the rest-bone or multi-pose clearance constraints.

The Genesis images are visual evidence only. No image in this document is
used to claim a bone–bone or tube–bone intersection pass.

