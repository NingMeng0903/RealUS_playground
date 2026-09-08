# V14 Genesis image review (bounded independent audit)

Date: 2026-09-08. This review is read-only. I opened the rendered Genesis RGB
images from the three requested output trees and compared `raw142` with the
rendered `candidate` where both variants exist. I did not change the renderer,
fit, runtime, bind, weights, or geometry.

The render manifests mark the images as operationally rendered, while the
artifacts remain `publishable=false` and `anatomical_passed=false`:

* [`v14_original_shape_genesis_20260908_001/manifest.json`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/manifest.json)
* [`v14_arm_pose_genesis_20260908_002/manifest.json`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/manifest.json)
* [`v14_continuous_sitting_213328_20260908_002/report.json`](../outputs/anatomy_retarget/v14_continuous_sitting_213328_20260908_002/report.json)

## T-pose: original-shape Genesis render

I opened `whole_ap`, `left_elbow_ap`, `left_elbow_lateral`, and
`left_wrist_ap` for both variants of both capture bodies:

| body | candidate RGB opened | raw142 RGB opened |
|---|---|---|
| 213328 | [`whole_ap`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/subject_213328_tpose/candidate/rgb/whole_ap.png), [`left_elbow_ap`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/subject_213328_tpose/candidate/rgb/left_elbow_ap.png), [`left_elbow_lateral`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/subject_213328_tpose/candidate/rgb/left_elbow_lateral.png), [`left_wrist_ap`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/subject_213328_tpose/candidate/rgb/left_wrist_ap.png) | [`whole_ap`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/subject_213328_tpose/raw142/rgb/whole_ap.png), [`left_elbow_ap`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/subject_213328_tpose/raw142/rgb/left_elbow_ap.png), [`left_elbow_lateral`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/subject_213328_tpose/raw142/rgb/left_elbow_lateral.png), [`left_wrist_ap`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/subject_213328_tpose/raw142/rgb/left_wrist_ap.png) |
| 213712 | [`whole_ap`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/subject_213712_tpose/candidate/rgb/whole_ap.png), [`left_elbow_ap`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/subject_213712_tpose/candidate/rgb/left_elbow_ap.png), [`left_elbow_lateral`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/subject_213712_tpose/candidate/rgb/left_elbow_lateral.png), [`left_wrist_ap`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/subject_213712_tpose/candidate/rgb/left_wrist_ap.png) | [`whole_ap`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/subject_213712_tpose/raw142/rgb/whole_ap.png), [`left_elbow_ap`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/subject_213712_tpose/raw142/rgb/left_elbow_ap.png), [`left_elbow_lateral`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/subject_213712_tpose/raw142/rgb/left_elbow_lateral.png), [`left_wrist_ap`](../outputs/anatomy_retarget/v14_original_shape_genesis_20260908_001/subject_213712_tpose/raw142/rgb/left_wrist_ap.png) |

The whole-body images have a plausible upright T-pose and no gross pelvic or
knee displacement at this scale. The two candidate bodies also look nearly
the same in the whole view, so these images do not provide visible evidence of
subject-specific shape adaptation.

The elbow close-ups show the candidate forearm station moved relative to
`raw142`, but the distal humerus and the radius/ulna caps still form a crowded
junction without a clean, readable articular gap. Thick red vessel branches
cross the joint in front of and behind the white bone surfaces. The vessels
make it impossible to read a continuous perivascular clearance around the
joint from RGB alone, and several branches visibly cross one another at the
elbow.

The wrist close-ups contain the expected carpal row and hand bones, but red
vessels sweep across the radius/carpals and bunch at the wrist-side junction.
The candidate changes the forearm station while retaining this crowding. The
whole view therefore looks acceptable as a skeleton silhouette, while the
local elbow and wrist images still fail a visual anatomical review.

The updated independent numerical audit ([`report.json`](../outputs/anatomy_retarget/v14_original_shape_audit_20260908_003/report.json)) agrees with this visual reading for
the left candidate T-pose: the full forearm remains inside the skin, but the
complete triangle audit still reports 19 Humerus--Radius contacts and 83
Radius--Ulna contacts. The frozen-template cap gate is a separate result and
does pass for the candidate (Humerus maximum residual 0.0066 mm); it does not
clear these joint-surface contacts.

## Continuous sitting frames

I opened the contact sheet and the three RGB views (`whole_ap`,
`left_elbow_lateral`, and `left_wrist_ap`) for saved frames 00000, 00018, and
00035:

| frame | opened RGB files | visible finding |
|---|---|---|
| 00000 | [`whole_ap`](../outputs/anatomy_retarget/v14_continuous_sitting_213328_20260908_002/genesis/00000/rgb/whole_ap.png), [`left_elbow_lateral`](../outputs/anatomy_retarget/v14_continuous_sitting_213328_20260908_002/genesis/00000/rgb/left_elbow_lateral.png), [`left_wrist_ap`](../outputs/anatomy_retarget/v14_continuous_sitting_213328_20260908_002/genesis/00000/rgb/left_wrist_ap.png), [`contact_sheet`](../outputs/anatomy_retarget/v14_continuous_sitting_213328_20260908_002/genesis/00000/contact_sheet.png) | Upright whole pose is broadly plausible, but the elbow vessel bundle crosses the bone junction and the wrist branches lie across the carpal row. |
| 00018 | [`whole_ap`](../outputs/anatomy_retarget/v14_continuous_sitting_213328_20260908_002/genesis/00018/rgb/whole_ap.png), [`left_elbow_lateral`](../outputs/anatomy_retarget/v14_continuous_sitting_213328_20260908_002/genesis/00018/rgb/left_elbow_lateral.png), [`left_wrist_ap`](../outputs/anatomy_retarget/v14_continuous_sitting_213328_20260908_002/genesis/00018/rgb/left_wrist_ap.png), [`contact_sheet`](../outputs/anatomy_retarget/v14_continuous_sitting_213328_20260908_002/genesis/00018/contact_sheet.png) | The asymmetric seated/leg pose is visible globally. At the elbow, the rotated forearm and humerus are partly hidden by a thick red bundle; at the wrist, branches remain crowded over the carpals. |
| 00035 | [`whole_ap`](../outputs/anatomy_retarget/v14_continuous_sitting_213328_20260908_002/genesis/00035/rgb/whole_ap.png), [`left_elbow_lateral`](../outputs/anatomy_retarget/v14_continuous_sitting_213328_20260908_002/genesis/00035/rgb/left_elbow_lateral.png), [`left_wrist_ap`](../outputs/anatomy_retarget/v14_continuous_sitting_213328_20260908_002/genesis/00035/rgb/left_wrist_ap.png), [`contact_sheet`](../outputs/anatomy_retarget/v14_continuous_sitting_213328_20260908_002/genesis/00035/contact_sheet.png) | The legs cross in the whole view. Long red trunks run across the hip/thigh and the wrist view shows vessels crossing the hand and distal forearm; the local bone/soft-tissue clearance is not preserved visibly. |

These are candidate-only motion renders. The continuous report quantifies the
same failure: maximum signed outside distances for all sampled bones/vessels
are 29.96/23.95 mm at 00000, 42.43/37.40 mm at 00018, and 30.41/24.05 mm at
00035 (bones/vessels respectively). The saved MP4 is therefore an operational
playback artifact, not an anatomical pass.

## Pose-fit own, swap, and 120-degree elbow

I opened the candidate and raw142 contact sheets for all three requested pose
cells, then opened elbow AP/lateral and wrist AP close-ups for both variants.
Candidate whole views were also opened for all three cells:

| pose cell | candidate files opened | raw142 files opened |
|---|---|---|
| own `subject_213328_pose_213328` | [`contact_sheet`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213328/candidate/contact_sheet.png), [`whole_ap`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213328/candidate/rgb/whole_ap.png), [`left_elbow_ap`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213328/candidate/rgb/left_elbow_ap.png), [`left_elbow_lateral`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213328/candidate/rgb/left_elbow_lateral.png), [`left_wrist_ap`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213328/candidate/rgb/left_wrist_ap.png) | [`contact_sheet`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213328/raw142/contact_sheet.png), [`left_elbow_ap`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213328/raw142/rgb/left_elbow_ap.png), [`left_elbow_lateral`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213328/raw142/rgb/left_elbow_lateral.png), [`left_wrist_ap`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213328/raw142/rgb/left_wrist_ap.png) |
| swap `subject_213328_pose_213712` | [`contact_sheet`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213712/candidate/contact_sheet.png), [`whole_ap`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213712/candidate/rgb/whole_ap.png), [`left_elbow_ap`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213712/candidate/rgb/left_elbow_ap.png), [`left_elbow_lateral`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213712/candidate/rgb/left_elbow_lateral.png), [`left_wrist_ap`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213712/candidate/rgb/left_wrist_ap.png) | [`contact_sheet`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213712/raw142/contact_sheet.png), [`left_elbow_ap`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213712/raw142/rgb/left_elbow_ap.png), [`left_elbow_lateral`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213712/raw142/rgb/left_elbow_lateral.png), [`left_wrist_ap`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_pose_213712/raw142/rgb/left_wrist_ap.png) |
| `subject_213328_elbow_L_120` | [`contact_sheet`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_elbow_L_120/candidate/contact_sheet.png), [`whole_ap`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_elbow_L_120/candidate/rgb/whole_ap.png), [`left_elbow_ap`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_elbow_L_120/candidate/rgb/left_elbow_ap.png), [`left_elbow_lateral`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_elbow_L_120/candidate/rgb/left_elbow_lateral.png), [`left_wrist_ap`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_elbow_L_120/candidate/rgb/left_wrist_ap.png) | [`contact_sheet`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_elbow_L_120/raw142/contact_sheet.png), [`left_elbow_ap`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_elbow_L_120/raw142/rgb/left_elbow_ap.png), [`left_elbow_lateral`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_elbow_L_120/raw142/rgb/left_elbow_lateral.png), [`left_wrist_ap`](../outputs/anatomy_retarget/v14_arm_pose_genesis_20260908_002/subject_213328_elbow_L_120/raw142/rgb/left_wrist_ap.png) |

In the own pose, the whole-body leg configuration is plausible, but the elbow
close-ups still show red vessel crossings over the joint and the candidate's
forearm station does not create a clean cap-to-cap relation. The wrist branch
layout remains crowded along the radius and across the hand.

In the swapped pose both arms are elevated. The global pose is recognizable,
but the AP and lateral elbow views show the vascular bundle wrapping across the
flexed joint and obscure the bone junction. The wrist vessels follow the long
forearm direction yet cross and bunch over the carpal bones, so the linkage is
not spatially convincing under this pose.

At 120 degrees, the intended left elbow bend is visible in the whole image.
The close-ups still show the red bundle crossing the flexure and an offset or
overlapped forearm cap; the candidate and raw142 differences do not remove the
problem. The wrist close-up has a continuous-looking forearm column, but its
vessels still pass across the carpal surfaces.

The pose-fit report supplies the numerical reason these images cannot be
accepted. After correction, the candidate maximum skin outside distance is
1.18 mm arm / 5.91 mm hand for the own pose, 5.24 / 8.21 mm for the swap, and
3.34 / 10.24 mm at 120 degrees. The directed joint checks also retain
humerus-to-radius penetration of 1.02 mm (own), 2.21 mm (swap), and 3.85 mm
(120 degrees), with the 120-degree radius-to-Scaphoid check at 1.61 mm. The
visual and metric evidence agree that the baked correction does not yet
preserve the elbow/wrist relationship across poses.

## Bounded conclusion

The Genesis images show a usable global skeleton pose and confirm that the
candidate is being driven through the intended poses, but they do not show a
stable anatomical mapping. The recurring failures are local elbow cap
relation, forearm/wrist station, and vessel paths crossing or crowding bone and
carpal surfaces. The original-shape candidate's frozen-template cap shape gate
can pass while these joint and vessel failures remain; that gate must therefore
stay separate from the joint-surface and motion acceptance decision.
