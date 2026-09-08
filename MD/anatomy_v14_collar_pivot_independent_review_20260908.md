# V14 collar-pivot independent Genesis review — 2026-09-08

This is a read-only visual review of the new Genesis render set
`v14_collar_pivot_genesis_20260908_001`. Each comparison image shows the
`original_driver` on the left and the collar-pivot `candidate` on the right.
The review covers both captures (`213328`, `213712`), each T-pose, own pose,
and held-out sitting pose. I opened the collar AP, collar superior, shoulder
oblique, and wrist AP views for all six cells, plus their candidate contact
sheets.

The renderer manifest records `camera_source=SMPL-X joints and skin only`,
internal material alpha 1.0, local-view clipping of ±160 mm, and
`publishable=false`. A render `pass` in that manifest is an operational image
check; it is not a bone or vessel intersection result.

## Files reviewed

| capture/state | collar AP | collar superior | shoulder oblique | wrist AP |
|---|---|---|---|---|
| 213328 T-pose | [AP](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213328_tpose/comparison/left_collar_ap.png) | [superior](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213328_tpose/comparison/left_collar_superior.png) | [oblique](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213328_tpose/comparison/left_shoulder_oblique.png) | [wrist](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213328_tpose/comparison/left_wrist_ap.png) |
| 213328 own pose | [AP](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213328_pose_213328/comparison/left_collar_ap.png) | [superior](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213328_pose_213328/comparison/left_collar_superior.png) | [oblique](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213328_pose_213328/comparison/left_shoulder_oblique.png) | [wrist](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213328_pose_213328/comparison/left_wrist_ap.png) |
| 213328 held-out sitting | [AP](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213328_heldout_sitting/comparison/left_collar_ap.png) | [superior](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213328_heldout_sitting/comparison/left_collar_superior.png) | [oblique](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213328_heldout_sitting/comparison/left_shoulder_oblique.png) | [wrist](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213328_heldout_sitting/comparison/left_wrist_ap.png) |
| 213712 T-pose | [AP](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213712_tpose/comparison/left_collar_ap.png) | [superior](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213712_tpose/comparison/left_collar_superior.png) | [oblique](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213712_tpose/comparison/left_shoulder_oblique.png) | [wrist](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213712_tpose/comparison/left_wrist_ap.png) |
| 213712 own pose | [AP](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213712_pose_213712/comparison/left_collar_ap.png) | [superior](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213712_pose_213712/comparison/left_collar_superior.png) | [oblique](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213712_pose_213712/comparison/left_shoulder_oblique.png) | [wrist](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213712_pose_213712/comparison/left_wrist_ap.png) |
| 213712 held-out sitting | [AP](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213712_heldout_sitting/comparison/left_collar_ap.png) | [superior](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213712_heldout_sitting/comparison/left_collar_superior.png) | [oblique](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213712_heldout_sitting/comparison/left_shoulder_oblique.png) | [wrist](../outputs/anatomy_retarget/v14_collar_pivot_genesis_20260908_001/subject_213712_heldout_sitting/comparison/left_wrist_ap.png) |

## Visual findings

In both T-pose cells, the medial clavicle remains visually continuous with
the manubrial/sternal region in AP and superior views. The candidate and
original have the same broad clavicle-to-acromion line and scapular position.
The red arterial bundle and yellow nerve bundle stay routed beneath/behind
the clavicle and into the shoulder. They are very close to the bone surfaces
at the sternoclavicular and subclavian regions, with occasional occlusion in
the rendered depth ordering, so the images cannot determine whether a tube
surface is intersecting the bone.

In the 213328 own pose, the candidate keeps the collar attachment while the
arm rotates down beside the thorax. The AP and oblique views show the
proximal humerus and clavicle in a continuous layout; the superior view shows
the vessels following the shoulder turn. The 213328 sitting pose preserves
the same medial attachment under thorax flexion. In the superior view the
large red vessel bundle tracks along the underside of the clavicle/scapular
edge and appears crowded against the bone, which requires the depth and
triangle audit to resolve.

The 213712 own pose raises both arms. The candidate still shows no gross
disconnection at the medial clavicle in AP or superior view, and the oblique
view keeps the proximal vessel bundle traveling with the shoulder. In the
213712 sitting view the collar/sternum relationship remains continuous while
the thorax rotates. The subclavian red vessels and yellow nerves again occupy
a narrow corridor along the clavicle and first-rib region; visual separation
is insufficient to call this clear.

The wrist AP images are useful as a scope check. T-pose bone and vessel
layers are close to the original. In both active poses, the forearm vessels
remain visibly close to or over the radius/ulna, and the candidate does not
make the known distal-arm pose discrepancy disappear. This review therefore
does not extend the collar observation into a wrist or forearm acceptance
claim.

## Interpretation and limits

The collar-pivot change removes no visible gross sternoclavicular gap in the
six reviewed states, but it also does not establish bone–bone or tube–bone
non-intersection. The nearby red/yellow bundles are crowded enough that the
corresponding signed-distance and triangle-pair measurements remain required.
The virtual marker displacement is outside this review and was not used as a
collar attachment criterion.

The reported S16 source kinematic error of approximately 0.14–0.18 mm is a
source-motion diagnostic only. Separate package audits still show roughly
8–43 mm left-arm pose errors over the tested package states. The render set
is consequently evidence for visual collar continuity, with
`anatomical_passed=false` and `publishable=false`; no image here is treated
as an intersection pass.

