# V14 neutral-first restfit review — 2026-09-08

This is the independent validation pass for
`outputs/anatomy_retarget/v14_arm_restfit_20260908_004`. It uses the reusable
[`audit_consistent_arm_v14.py`](../src/projects/genesis_ue_sync/anatomy_retarget/cli/audit_consistent_arm_v14.py)
with the original frozen operator and calibration. It does not alter the fit,
runtime, or Genesis renderer.

Machine-readable output:
[`v14_arm_restfit_audit_20260908_006/report.json`](../outputs/anatomy_retarget/v14_arm_restfit_audit_20260908_006/report.json).

The audit covers `tpose` and own capture pose `pose_213328`. All measurements
use the original calibration `validation` domains. The independent VTK checks
query complete Humerus_L--Radius_L, Humerus_L--Ulna_L, and Radius_L--Ulna_L surfaces. The audit
finished operationally, but this candidate is not an anatomical pass.

## Decision

The neutral-first restfit is a useful diagnostic candidate, but it is rejected
for the requested one-target/all-pose behavior:

- T-pose full connected-arm maximum outside is **1.137 mm** with one vertex,
  just above the 1 mm target. The Humerus, Radius, and Ulna meshes themselves
  are skin-contained in this cell; the failing vertex is
  `_4th_Proximal_Phalanges_Hand_L`.
- In own pose, the maximum outside becomes **45.519 mm** with **9,611**
  vertices. Radius_L reaches 18.772 mm and Ulna_L 27.966 mm; the hand branch
  reaches 45.519 mm. This is a large pose regression before any baked pose
  correction.
- T validation queries remove most humerus-vs-forearm sampled penetration, but
  the frozen Radius-vs-Ulna overlap remains (`−1.185` to `−2.208` mm). The
  complete Humerus--Radius surfaces still have seven crossing triangle pairs
  in T and six in the own pose; the complete Radius--Ulna pair has 83 contacts
  in both cells.

This is therefore a **neutral-stage observation**, not evidence that the map
works outside T-pose. The fit report itself says
`anatomical_passed=false`, `publishable=false`, and
`independent_validation_complete=false`.

## Identity and scope

| item | value |
|---|---|
| candidate geometry | `outputs/anatomy_retarget/v14_arm_restfit_20260908_004` |
| audit output | `outputs/anatomy_retarget/v14_arm_restfit_audit_20260908_006` |
| operator runtime digest | `17f5d4e0bc328e85aef0d6dc6eba0e3fa8ca1ddd0a79f751ae259e129d00972b` |
| calibration content digest | `c040ed0df9f7f37cb1235895458346540b0228d62e509026327ed16600bdaca5` |
| calibration fixed-domain digest | `d86806ecdb2aee09725f291c1add90fcede3dcfbf66cef7bf1cf2e70488297f7` |
| validation partition | `validation` only; no fit IDs |
| arm population | 30 connected left-arm bone meshes, 11,415 vertices |
| mesh-pair method | VTK OBB triangle contacts plus bidirectional vertex/face-centroid signed depths |

The two audited input cells have SHA-256 values `93d0b5b17b32b581f54f337473887af7e6d35feff2b5235d9567b10d4583b600`
(T) and `66ad5494b1fc5d18a85f4d667bb691b50236ac9e1bc6d4574a2e3972a81f233b`
(own pose). The audit script SHA-256 is
`e477f4f92a776048214c12f7b81079df2ac941721cc62248f961834a2b9b7b00`.

## Skin boundary: full arm, raw versus candidate

| pose | geometry | max outside | outside vertices >1 mm | main candidate regions |
|---|---|---:|---:|---|
| T | raw142 | 18.233 mm | 72 | Humerus_L 18.233 mm |
| T | candidate | **1.137 mm** | **1** | _4th_Proximal_Phalanges_Hand_L 1.137 mm |
| own `213328` | raw142 | 19.022 mm | 364 | Humerus_L 19.022 mm |
| own `213328` | candidate | **45.519 mm** | **9,611** | _1st_Distal_Phalanges_Hand_L 45.519; Humerus_L 2.396; Radius_L 18.772; Ulna_L 27.966 |

The neutral-first fit relocates the T-pose error out of the primary three bone
meshes, but the same map sends the connected arm far outside the skin in the
captured pose. This shows why a T-only optimization result cannot be used as a
runtime acceptance result.

## Validation-domain signed penetration

Negative values mean the validation-domain sample is inside the complete target
bone. The count is the number below −0.1 mm.

| pose | query → target | candidate minimum | count |
|---|---|---:|---:|
| T | humerus → radius | +0.643 mm | 0 |
| T | humerus → ulna | +0.401 mm | 0 |
| T | radius → humerus | +0.258 mm | 0 |
| T | radius → ulna | **−1.185 mm** | 3 |
| T | ulna → humerus | +0.559 mm | 0 |
| T | ulna → radius | **−2.208 mm** | 5 |
| own `213328` | humerus → radius | +2.492 mm | 0 |
| own `213328` | humerus → ulna | +4.191 mm | 0 |
| own `213328` | radius → humerus | +0.335 mm | 0 |
| own `213328` | radius → ulna | **−1.185 mm** | 3 |
| own `213328` | ulna → humerus | +1.876 mm | 0 |
| own `213328` | ulna → radius | **−2.208 mm** | 5 |

The positive humerus/forearm values in the own pose indicate that this
candidate has moved those primary bones apart relative to the validation
station; they do not prove a healthy joint gap. The remaining Radius--Ulna
negative values are unchanged local forearm overlap and remain a failure.

## Complete mesh triangle checks

Every checked bone mesh is watertight, consistently wound, and has valid
signed volume. A crossing is retained as unresolved even when sampled depth is
below 0.5 mm.

| pose | geometry | Humerus--Radius | Humerus--Ulna | Radius--Ulna | result |
|---|---|---:|---:|---:|---|
| T | raw142 | 52 | 147 | 83 | penetration exceeds tolerance |
| T | candidate | **7** | **0** | **83** | Humerus--Radius crossing requires review; Radius--Ulna overlap |
| own `213328` | raw142 | 46 | 145 | 83 | penetration exceeds tolerance |
| own `213328` | candidate | **6** | **0** | **83** | Humerus--Radius crossing requires review; Radius--Ulna overlap |

The validation-domain R-U overlap above is retained as complete-mesh evidence
and is not hidden by the H-R/H-U pair result. Raw crossings document a
pre-existing local mesh issue; the candidate's zero H-U contacts do not erase
that baseline or establish physiological articulation.

## T-pose cap shape

Rigid Procrustes comparison against subject-materialized raw142 preserves the
selected cap shapes: the incremental rest-shape gate is 0.1 mm RMS, 0.1 mm
maximum residual, and 0.1 mm pairwise absolute p95. Pose-rigidity measurements
below are kept separate. This gate measures only the candidate map's change
after `materialize_subject(beta)` has produced the raw142 subject geometry.

| cap | RMS | max residual | pairwise p95 | gate |
|---|---:|---:|---:|---|
| left humerus | 0.000775 mm | 0.007209 mm | 0.000021 mm | pass |
| left radius | 0.000022 mm | 0.000144 mm | 0.000027 mm | pass |
| left ulna | 0.000009 mm | 0.000017 mm | 0.000017 mm | pass |

The six right-arm controls also pass at approximately 0.000001–0.000003 mm
maximum residual. Cap preservation is thus not the cause of the own-pose
failure; the problem is the pose-dependent station/branch relationship.

## Frozen-template shape limitation

The audit separately compares the two T-pose geometries with the frozen
`operator.template_asset.vertices_rest`; this reference comparison is
ungated. For subject 213328, the maximum pairwise-distance deltas are:

| cap | materialized raw142 vs frozen template | candidate vs frozen template | gate |
|---|---:|---:|---|
| left humerus | **0.418408 mm** | **0.418422 mm** | not applied |
| left radius | **0.443122 mm** | **0.443281 mm** | not applied |
| left ulna | **0.367494 mm** | **0.367487 mm** | not applied |

The materialized raw142 geometry already exceeds the user's 0.1 mm cap
target on this reference before candidate transport. Therefore the incremental
passes in the preceding table do not establish a total authored-142 cap
error ≤0.1 mm. The JSON
`tpose_cap_shape_vs_frozen_template` section records these ungated metrics and
sets `total_authored_142_gate_demonstrated=false`; the values are a limitation
to the current result, not an anatomical pass. Upstream operator preflight
measured 213712 as exact zero for these three caps and synthetic PC1+1.5 as
2.375436/2.384439/2.045453 mm (humerus/radius/ulna), which further shows why
the materialization effect must be reported separately.

## Posed cap rigidity against each variant's own T rest

For each non-T cell, the cap is compared with the T rest from the same variant
(`raw142` posed → raw142 T, `candidate` posed → candidate T). The rigid angle
and translation describe motion in the stored world frame; the residual is the
shape deformation after that rigid motion. This is informative pose evidence
and has no rest-shape gate.

| pose | variant | cap | rigid angle | rigid translation norm | deformation RMS / max |
|---|---|---|---:|---:|---:|
| own `213328` | raw142 | humerus | 36.140° | 677.684 mm | 0.000 / 0.000 mm |
| own `213328` | raw142 | radius | 38.864° | 731.734 mm | **0.069 / 0.307 mm** |
| own `213328` | raw142 | ulna | 38.711° | 731.941 mm | 0.000 / 0.000 mm |
| own `213328` | candidate | humerus | 38.084° | 675.276 mm | 0.000 / 0.000 mm |
| own `213328` | candidate | radius | 42.026° | 730.536 mm | **0.069 / 0.307 mm** |
| own `213328` | candidate | ulna | 41.884° | 730.706 mm | 0.000 / 0.000 mm |

The radius cap's nonzero residual confirms that blended forearm influences can
deform a posed cap even when the T-pose cap comparison is rigid. The large
translation norms are world-frame transforms over the body lever arm, not
joint-station offsets; all raw and candidate rows, including right-arm
controls, remain available in the audit JSON.

## Genesis visual review

I opened the new Genesis RGB renders directly and compared them with the
matching raw142 views. The T candidate elbow cap remains visibly full-sized,
while its local station and vessel bundle are shifted relative to raw142. The
T wrist AP views are close in the neutral pose, consistent with the 1.137 mm
near-pass. In the own pose, the candidate elbow foreground and vascular route
are visibly offset from raw142, and the wrist/hand branch no longer keeps the
same relation to the skin. The elbow-120 views show the same terminal-chain
problem under deeper flexion.

Representative views opened:

- T elbow: [candidate AP](../outputs/anatomy_retarget/v14_arm_restfit_genesis_20260908_004/subject_213328_tpose/candidate/rgb/left_elbow_ap.png), [candidate lateral](../outputs/anatomy_retarget/v14_arm_restfit_genesis_20260908_004/subject_213328_tpose/candidate/rgb/left_elbow_lateral.png), [raw lateral](../outputs/anatomy_retarget/v14_arm_restfit_genesis_20260908_004/subject_213328_tpose/raw142/rgb/left_elbow_lateral.png).
- T wrist: [candidate AP](../outputs/anatomy_retarget/v14_arm_restfit_genesis_20260908_004/subject_213328_tpose/candidate/rgb/left_wrist_ap.png), [raw AP](../outputs/anatomy_retarget/v14_arm_restfit_genesis_20260908_004/subject_213328_tpose/raw142/rgb/left_wrist_ap.png).
- own elbow: [candidate lateral](../outputs/anatomy_retarget/v14_arm_restfit_genesis_20260908_004/subject_213328_pose_213328/candidate/rgb/left_elbow_lateral.png), [raw lateral](../outputs/anatomy_retarget/v14_arm_restfit_genesis_20260908_004/subject_213328_pose_213328/raw142/rgb/left_elbow_lateral.png).
- own wrist: [candidate AP](../outputs/anatomy_retarget/v14_arm_restfit_genesis_20260908_004/subject_213328_pose_213328/candidate/rgb/left_wrist_ap.png), [raw AP](../outputs/anatomy_retarget/v14_arm_restfit_genesis_20260908_004/subject_213328_pose_213328/raw142/rgb/left_wrist_ap.png).
- deeper flexion: [candidate elbow-120 lateral](../outputs/anatomy_retarget/v14_arm_restfit_genesis_20260908_004/subject_213328_elbow_L_120/candidate/rgb/left_elbow_lateral.png), [raw elbow-120 lateral](../outputs/anatomy_retarget/v14_arm_restfit_genesis_20260908_004/subject_213328_elbow_L_120/raw142/rgb/left_elbow_lateral.png), [candidate wrist AP](../outputs/anatomy_retarget/v14_arm_restfit_genesis_20260908_004/subject_213328_elbow_L_120/candidate/rgb/left_wrist_ap.png).

The visual review supports the numerical conclusion while preserving the
proper limitation: images show placement and continuity, not a proof of
triangle-level non-intersection.
