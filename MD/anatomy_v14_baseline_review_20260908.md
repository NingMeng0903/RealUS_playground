# V14 frozen V7 baseline: Genesis 3D review

Date: 2026-09-08

This review uses only the new Genesis render at
`outputs/anatomy_retarget/v14_baseline_genesis_20260908_001`.  Every comparison
has `raw142` on the left and `candidate` (frozen V7) on the right.  The
renderer used the existing Genesis platform, opaque internal anatomy
materials (`alpha=1`), a 10 mm scale bar, and local joint cameras with a
`±160 mm` depth slab.  The local slab can clip flat ends at the image border;
such an end is not evidence of a broken bone.

The candidate is the existing V7 subject bundle replayed by `PoseMapV1` using
the recorded right-multiply composition.  It is not V10, V11, or V12e:

| item | frozen identity |
|---|---|
| source | 142 `ResidentPoseEvaluatorV8` source runtime |
| candidate | V7 `pose_whole_chain_vertices(PoseMapV1)` |
| pose composition | `G_prime = G_source @ inv(B_source) @ B_target` (`right_multiply_bind_v6`) |
| V7 rest method | `full_main_chain_right_multiply_pose_v7_femur_axial` |
| operator digest | `17f5d4e0bc328e85aef0d6dc6eba0e3fa8ca1ddd0a79f751ae259e129d00972b` |
| oracle SHA | `60bf4c3f7803b62b2113fe2715e9b53b35d16caf35410ce4bd1f9b9c47e8dd3d` |

The six input geometry cells and their render manifest are in
[v14_baselines_20260908_001](../outputs/anatomy_retarget/v14_baselines_20260908_001)
and
[v14_baseline_genesis_20260908_001/manifest.json](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/manifest.json).
The corresponding frozen-domain signed surface measurements are in
[articular_surface_audit_v13/report.json](../outputs/anatomy_retarget/v14_baselines_20260908_001/articular_surface_audit_v13/report.json).

## Images actually inspected

The following are the six required local views for each subject and pose.  The
links point to the side-by-side Genesis comparison (`raw142 | candidate`).

### Subject 213328, T-pose

[left elbow](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213328_tpose/comparison/left_elbow_lateral.png) ·
[right elbow](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213328_tpose/comparison/right_elbow_lateral.png) ·
[left knee](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213328_tpose/comparison/left_knee_lateral.png) ·
[right knee](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213328_tpose/comparison/right_knee_lateral.png) ·
[left hip](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213328_tpose/comparison/left_hip_oblique.png) ·
[right hip](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213328_tpose/comparison/right_hip_oblique.png)

The T-pose has no gross whole-limb break.  The candidate left knee looks
slightly more closely seated on the medial view, while the lateral contour is
not uniformly improved.  The left and right elbow views remain materially
different from the raw source in the proximal forearm, although the right
side is visually close.  Both hip views retain the head-and-cup arrangement;
the vessels and partial occlusion prevent a visual contact claim.

### Subject 213328, own captured pose

[left elbow](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213328_pose_213328/comparison/left_elbow_lateral.png) ·
[right elbow](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213328_pose_213328/comparison/right_elbow_lateral.png) ·
[left knee](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213328_pose_213328/comparison/left_knee_lateral.png) ·
[right knee](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213328_pose_213328/comparison/right_knee_lateral.png) ·
[left hip](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213328_pose_213328/comparison/left_hip_oblique.png) ·
[right hip](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213328_pose_213328/comparison/right_hip_oblique.png)

The candidate left elbow visibly loses the raw humerus-to-radius alignment:
the proximal forearm begins away from the distal humerus and the red vessel
passes through the changed joint neighborhood.  The right elbow is much less
changed.  The knees retain their gross articulation but still show different
condyle/plateau placement and soft-tissue routing.  The right hip is visually
closer to the raw head-and-cup arrangement; the left hip has a more apparent
offset and remains occluded enough that this is only a comparison observation.

### Subject 213712, T-pose

[left elbow](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213712_tpose/comparison/left_elbow_lateral.png) ·
[right elbow](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213712_tpose/comparison/right_elbow_lateral.png) ·
[left knee](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213712_tpose/comparison/left_knee_lateral.png) ·
[right knee](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213712_tpose/comparison/right_knee_lateral.png) ·
[left hip](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213712_tpose/comparison/left_hip_oblique.png) ·
[right hip](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213712_tpose/comparison/right_hip_oblique.png)

The same T-pose pattern is present for the second beta shape.  The candidate
left medial knee view is slightly tighter, but the elbow change is not
uniformly beneficial and the hip images do not establish concentric seating.
Right-sided elbow and knee views stay close to the raw source.

### Subject 213712, own captured pose

[left elbow](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213712_pose_213712/comparison/left_elbow_lateral.png) ·
[right elbow](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213712_pose_213712/comparison/right_elbow_lateral.png) ·
[left knee](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213712_pose_213712/comparison/left_knee_lateral.png) ·
[right knee](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213712_pose_213712/comparison/right_knee_lateral.png) ·
[left hip](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213712_pose_213712/comparison/left_hip_oblique.png) ·
[right hip](../outputs/anatomy_retarget/v14_baseline_genesis_20260908_001/subject_213712_pose_213712/comparison/right_hip_oblique.png)

The candidate left elbow has a clear large gap/offset in the radius view.
The right elbow appears closer than the left, showing that the problem is
side-specific rather than a single global arm scale.  Both knee views keep a
continuous lower limb but do not show a stable articular surface relationship
through the pose.  The hips preserve gross placement while the head/cup
surfaces remain partly hidden by the vessels and cannot be accepted from
projection alone.

## What the V7 baseline improves

V7 provides a useful structural baseline.  The full anatomy remains in one
consistent 235-controller result, the pose replay is deterministic, and the
Genesis views show no whole-body topology disappearance.  In the frozen
surface measurements, the left medial-knee unsigned `p05` distance improves
in both T-poses: `0.565 -> 0.212 mm` for 213328 and `0.712 -> 0.218 mm` for
213712.  The 213328 own-pose right-hip unsigned `p05` also improves from
`0.528 -> 0.196 mm`.

Those are local comparison improvements.  They do not establish that every
surface is seated, because the query domains include points away from the
actual contact patch and the signed query can report overlap at the same
time.

## What remains wrong

The decisive failure is the left elbow in the two own poses.  The frozen
surface audit gives the following humerus-to-radius values; `closest` and
`p05` are unsigned point-to-triangle distances:

| cell | raw142 closest / p05 (mm) | V7 closest / p05 (mm) | visual conclusion |
|---|---:|---:|---|
| 213328 T-pose, left elbow | 0.077 / 5.376 | 0.470 / 5.081 | mixed local change; no clear contact proof |
| 213328 own pose, left elbow | 0.147 / 3.818 | **6.610 / 11.363** | clear candidate gap/offset |
| 213712 T-pose, left elbow | 0.077 / 5.351 | 0.472 / 5.057 | mixed local change; no clear contact proof |
| 213712 own pose, left elbow | 0.097 / 0.588 | **9.091 / 18.001** | clear candidate gap/offset |

The ulna is not sufficient to hide this failure: it can remain close while the
radius is displaced, which is why the fit must constrain the radius and ulna
as one forearm assembly around the same frozen elbow center.

The hip queries also remain locally overlapped.  For example, the candidate
left-hip signed minima are `-4.074 mm` and `-1.958 mm` for 213328 T-pose and
own pose, and `-3.992 mm` and `-4.012 mm` for 213712.  Negative means sampled
points are inside the watertight opposing mesh under the audit convention; it
is not a direct measurement of the visible joint gap.  The knee signed minima
are similarly negative, reaching roughly `-13 mm` in the own-pose medial
queries.  Therefore the candidate does not meet an anatomical joint gate.

The visual and numeric evidence agree on the practical conclusion: V7 is a
valid historical comparison baseline with preserved linkage machinery, but
its rest fit cannot be used as the final left-arm geometry.  Copying its
forearm correction into a new version would reproduce the own-pose radius gap.

## Handoff for the next left-arm fit

Use V7 only as the frozen comparison.  The next candidate should fit the left
radius and ulna together as one rigid forearm assembly about the frozen elbow
center, preserve their authored thickness and joint-end shapes, and carry the
same correction to the linked vessels and nerves.  Re-render the four critical
cells (both subjects, T-pose and own pose) with the same cameras, then rerun
the signed surface audit for both humerus-to-radius and humerus-to-ulna.  A
candidate that improves one elbow view while worsening either own-pose
radius row should be rejected.

This document is evidence-only.  It does not declare V7, raw142, or any
candidate an anatomical pass.
