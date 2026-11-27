# V14 full-arm graph-roll candidate review — 2026-09-08

This is an independent audit and Genesis image review of the frozen
`v14_full_arm_graph_roll_fit_213328_20260908_001` candidate. The four audited
cells are the subject's T-pose, the 213328 capture pose, the 213712 capture
pose applied to subject 213328, and the left-elbow 120-degree pose. The audit
uses the current `audit_consistent_arm_v14.py`, including its 0.1 mm posed-cap
hard gate. No fitting, tuning, runtime edit, or renderer change was performed
here.

- [independent audit report](../outputs/anatomy_retarget/v14_full_arm_graph_roll_audit_20260908_001/report.json)
- [fit output report](../outputs/anatomy_retarget/v14_full_arm_graph_roll_fit_213328_20260908_001/report.json)
- [compiled manifest](../outputs/anatomy_retarget/v14_full_arm_graph_roll_fit_213328_20260908_001/compiled/manifest.json)
- [Genesis render manifest](../outputs/anatomy_retarget/v14_full_arm_graph_roll_genesis_20260908_001/manifest.json)
- audit report SHA-256: `fe1c44f9f31c32b42c559650e9d02a06572b2580c85c2c594e6641366f79282a`
- fit report SHA-256: `94901094fe8dfcc39a5dd848cd1a4e35ba91c58145cb7de3fcaeb37806a67efa`
- compiled manifest SHA-256: `928e27044104f2901c3b20573d5ddfb3e427c8cb2003f00b0ba33b1356eaae55`
- Genesis manifest SHA-256: `073c111b5d0f84e3e9c5382efdf2afdcc6db69fac99a8d866da17d7e719b024f`
- audited operator runtime digest: `17f5d4e0bc328e85aef0d6dc6eba0e3fa8ca1ddd0a79f751ae259e129d00972b`

The input fit report says the optimizer stopped at the maximum number of
function evaluations (`optimizer_success=false`,
`independent_validation_complete=false`). The compiled package records
`composition=complete_parent_local_v14`, `rotation_transport=driver_axes`,
`translation_transport=motion_reference_axes`,
`shape_reference_kind=frozen_operator_template`, and unchanged authored
weights. Both the package and this audit retain `anatomical_passed=false` and
`publishable=false`.

## Aggregate skin boundary result

The complete connected left-arm population was measured, not only the fit
sample. The candidate's maximum outside distance is the requested failed
candidate result: T 9.011 mm, own 12.009 mm, swap 11.763 mm, and elbow120
7.393 mm.

| cell | raw142 max outside | candidate max outside | raw142 vertices >1 mm | candidate vertices >1 mm |
|---|---:|---:|---:|---:|
| T-pose | 18.233 mm | **9.011 mm** | 72 | 370 |
| own `pose_213328` | 19.022 mm | **12.009 mm** | 364 | 3,877 |
| swap `pose_213712` | 16.642 mm | **11.763 mm** | 113 | 3,597 |
| `elbow_L_120` | 16.564 mm | **7.393 mm** | 88 | 1,147 |

The maximum distance decreases against raw142 in these four rows, but the
number of arm vertices outside 1 mm increases in every row. This candidate is
therefore a tradeoff. The audit does not describe the underlying matrix or
transport result as an across-the-board improvement.

## Hard cap and local bone checks

The authenticated T-pose comparison against the frozen operator template
passes the 0.1 mm total authored-142 cap gate for all six humerus/radius/ulna
caps. That is a rest-shape result and does not certify skin containment or
joint clearance.

For posed caps, the audit sets `pose_rigidity_gate_applied=true` and compares
each variant with its own T-pose. The 0.1 mm gate passes for all caps in
`elbow_L_120`. In own `pose_213328`, both radius caps fail: left max residual
0.327 mm and right 0.130 mm. In swap `pose_213712`, both radius caps fail:
left 0.162 mm and right 0.237 mm. Humerus and ulna caps pass those rows. The
four posed-radius failures are included in the audit's 68 candidate flags.

Complete validation-domain signed queries still find penetration in all four
poses and in both directions. The worst left-arm candidate minima are:

| cell | worst signed query | minimum signed distance |
|---|---|---:|
| T-pose | `ulna_to_humerus` | -3.714 mm |
| own `pose_213328` | `radius_to_humerus` | -5.600 mm |
| swap `pose_213712` | `ulna_to_humerus` | -8.114 mm |
| `elbow_L_120` | `ulna_to_humerus` | -9.244 mm |

The complete left Humerus/Radius/Ulna triangle checks also fail in every
cell. Contact counts are:

| cell | Humerus--Radius | Humerus--Ulna | Radius--Ulna |
|---|---:|---:|---:|
| T-pose | 51 | 154 | 83 |
| own `pose_213328` | 43 | 138 | 83 |
| swap `pose_213712` | 146 | 135 | 83 |
| `elbow_L_120` | 98 | 142 | 83 |

The unchanged 83 Radius--Ulna contacts across all four candidate poses are a
persistent forearm-pair issue. Several other local measures move in opposite
directions: for example, T-pose Humerus--Ulna contacts rise from 147 raw142
to 154 candidate, swap Humerus--Radius rises from 144 to 146, and elbow120
Humerus--Radius rises from 86 to 98. The signed minima likewise worsen for
some query directions even where aggregate skin maximum improves. These are
why the candidate remains failed.

The audit completed 4 cells with no evaluation errors. Its 68 flags comprise
4 full-arm skin-outside flags, 4 posed-radius cap failures, 48 validation
domain penetration flags, and 12 triangle-pair contact/depth flags. The
technical audit completion is not an anatomical pass.

## Genesis image review

I opened the actual Genesis contact sheets for all four cells. Each sheet has
the whole-body, left-shoulder-oblique, left-elbow-AP, left-elbow-lateral, and
left-wrist-AP views. I inspected whole-body, elbow, and wrist views for T,
both captured poses, and elbow120, with the raw142 sheet beside each candidate
sheet.

| cell | candidate Genesis sheet | raw142 comparison sheet | visual finding |
|---|---|---|---|
| T-pose | [candidate](../outputs/anatomy_retarget/v14_full_arm_graph_roll_genesis_20260908_001/subject_213328_tpose/candidate/contact_sheet.png) | [raw142](../outputs/anatomy_retarget/v14_full_arm_graph_roll_genesis_20260908_001/subject_213328_tpose/raw142/contact_sheet.png) | The whole silhouette is continuous. The elbow views show the extended humerus/forearm chain, but bright red vessels cross the joint and lie directly over the white bone surfaces. The wrist view has a dense vessel bundle over the carpal/forearm region; the 2-D view cannot decide clearance. |
| own `pose_213328` | [candidate](../outputs/anatomy_retarget/v14_full_arm_graph_roll_genesis_20260908_001/subject_213328_pose_213328/candidate/contact_sheet.png) | [raw142](../outputs/anatomy_retarget/v14_full_arm_graph_roll_genesis_20260908_001/subject_213328_pose_213328/raw142/contact_sheet.png) | The whole body preserves the captured asymmetric lower-body pose. The elbow AP/lateral views keep the bent arm connected, while vessels wrap over the joint and the forearm. The wrist view shows the carpal chain but multiple red branches sit on top of the bone silhouettes. |
| swap `pose_213712` | [candidate](../outputs/anatomy_retarget/v14_full_arm_graph_roll_genesis_20260908_001/subject_213328_pose_213712/candidate/contact_sheet.png) | [raw142](../outputs/anatomy_retarget/v14_full_arm_graph_roll_genesis_20260908_001/subject_213328_pose_213712/raw142/contact_sheet.png) | The whole view shows the arms raised into the swapped capture pose. In the local elbow views, the forearm is visually attached but a small pale bone segment and the red bundle crowd the joint, making separation ambiguous. The wrist view has several thick vessels traversing the hand/forearm bones. |
| `elbow_L_120` | [candidate](../outputs/anatomy_retarget/v14_full_arm_graph_roll_genesis_20260908_001/subject_213328_elbow_L_120/candidate/contact_sheet.png) | [raw142](../outputs/anatomy_retarget/v14_full_arm_graph_roll_genesis_20260908_001/subject_213328_elbow_L_120/raw142/contact_sheet.png) | The whole view shows the intended left-elbow flexion. The AP/lateral elbow views keep the humerus and forearm in a readable bent chain, but several red branches cross directly through the joint projection. The wrist view remains continuous with a thick vessel bundle laid across the forearm and carpal bones. |

The Genesis manifest reports valid rendered depth for these views and uses the
existing renderer; that technical render status does not establish
non-intersection. The images support continuity of the displayed motion, while
the signed and triangle audits establish unresolved local penetration and the
posed radius-cap failures.

This graph-roll candidate improves the aggregate maximum outside distance in
the four inspected cells but fails the complete-arm boundary, local signed
bone, triangle-contact, and posed-cap gates. It is a useful failed comparison
candidate for the next fit, not a validated retarget package.
