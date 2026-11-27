# V14 lower-chain surface audit — 2026-09-08

This is a read-only audit of the latest combined driver-axes exports for
subjects 213328 and 213712. It evaluates the six requested cells (each
subject's T-pose, own captured pose, and the other capture's pose) using the
exported `candidate_vertices`. No fitting, parameter tuning, rebinding,
runtime edit, Blender call, or geometry copy was performed.

- [audit report](../outputs/anatomy_retarget/v14_lower_chain_audit_20260908_001/report.json)
- output size: 404 KiB (JSON only)
- report SHA-256: `e8b0c56b72a4e525af6725d45d07c351d9dbde23faf5d3d4cdbb0e3623e115a1`
- operator runtime digest: `17f5d4e0bc328e85aef0d6dc6eba0e3fa8ca1ddd0a79f751ae259e129d00972b`
- surface validator: `surface_validation_v14.audit_bone_pair`
- triangle depth tolerance: 0.5 mm

The input cells are the NPZs under
`outputs/anatomy_retarget/v14_collar_driver_axes_213328_20260908_001` and
`..._213712_20260908_001`. The report stores every input and candidate array
hash, the frozen operator hash, the compiled manifest hash, and the script
hash. It confirms identical frozen face topology across all six cells.

## Mesh naming and pair scope

The frozen asset contains the following relevant authored bone meshes:

| mesh | vertices | triangles | controller |
|---|---:|---:|---|
| `Ilium_L` / `Ilium_R` | 949 each | 1,898 each | `Hip_bone` |
| `Sacrum` | 4,698 | 9,424 | `Hip_bone` |
| `Femur_L` / `Femur_R` | 915 each | 1,826 each | `Femur_Rot_L/R` |
| `Tibia_L` / `Tibia_R` | 489 each | 974 each | `Tibia_Bone_L/R` |
| `Talus_L` / `Talus_R` | 163 each | 322 each | `Ankle_Rot_L/R` |

There is no authored `Pelvis_L/R`, `Ischium_L/R`, or `Pubis_L/R` mesh in
this asset. `Ilium_L--Femur_L` and `Ilium_R--Femur_R` are consequently the
required side-specific hipbone pairs and cover the available femoral-head
adjacency. `Sacrum--Femur_L/R` is included as an explicitly labelled
supplementary central-pelvis context pair; it is not substituted for either
Ilium pair.

For each side, the required chain is therefore:

1. `Ilium_side--Femur_side` (hip),
2. `Femur_side--Tibia_side` (knee), and
3. `Tibia_side--Talus_side` (ankle).

The two supplementary Sacrum pairs make 48 pair checks in total: six cells
times eight pairs. Every pair calls the complete VTK all-contact triangle
query and, when both meshes are valid for winding-number signs, both
directions of vertex plus face-centroid signed samples. The reported maximum
depth is explicitly a sampled lower bound, not a guarantee about unsampled
volume.

## Six-cell candidate result

The table gives `triangle contacts / maximum of the two directional sampled
penetration lower bounds`, in millimetres. Each JSON row retains both
directions separately, including sample counts, minimum absolute distances,
and counts beyond the 0.5 mm tolerance.

| subject | cell | hip L | hip R | knee L | knee R | ankle L | ankle R |
|---|---|---:|---:|---:|---:|---:|---:|
| 213328 | T-pose | 83 / 4.068 | 92 / 4.354 | 0 / 0.000 | 0 / 0.000 | 9 / 0.835 | 33 / 2.537 |
| 213328 | own (`pose_213328`) | 84 / 3.601 | 86 / 4.156 | 109 / 17.058 | 0 / 0.000 | 36 / 1.332 | 98 / 4.776 |
| 213328 | swap (`pose_213712`) | 89 / 3.530 | 81 / 5.041 | 27 / 2.000 | 53 / 4.594 | 50 / 6.558 | 89 / 7.191 |
| 213712 | T-pose | 83 / 4.068 | 92 / 4.354 | 0 / 0.000 | 0 / 0.000 | 9 / 0.835 | 33 / 2.537 |
| 213712 | own (`pose_213712`) | 89 / 3.537 | 81 / 5.048 | 27 / 1.993 | 53 / 4.615 | 52 / 6.520 | 89 / 7.294 |
| 213712 | swap (`pose_213328`) | 86 / 3.595 | 86 / 4.155 | 109 / 16.940 | 0 / 0.000 | 36 / 1.372 | 100 / 4.783 |

All 12 required hip rows and all 12 required ankle rows fail the conservative
surface check. Six of the 12 required knee rows also fail; the remaining six
have no detected contact or sampled depth above tolerance. Thus 30 of the 36
required rows are flagged. This is the reason the report's
`anatomical_passed=false` and `publishable=false` fields remain false.

The supplementary central-pelvis rows are closed, have zero triangle contacts,
and have zero sampled penetration in both directions in all six cells. Their
large minimum absolute distances (about 65.9–68.6 mm) show that they are
context checks, not a substitute for the side-specific acetabular region.

## Closure and winding

Every inspected mesh in every cell reports `watertight=true`,
`winding_consistent=true`, and `signed_distance_valid=true` in the validator
output. This means the bidirectional signed samples are available for this
asset. It does not mean the paired surfaces are anatomically correct: VTK
still finds triangle contacts in the hip, many knee, and every ankle failure
row, and the signed samples find millimetre-scale penetration.

The technical `surface_validation_passed` field is retained in each JSON pair
for reproducibility. It is never promoted to an anatomical pass in this
document or in the top-level report.

## Independent Genesis image review: 213328 kicking-lower

I also opened the actual Genesis renders for the frozen lower-chain motion at
frames 00000, 00018, and 00035. These images are visual evidence only and are
not used to override the triangle or signed audits.

- [Genesis frame 00000 contact sheet](../outputs/anatomy_retarget/v14_driver_axes_kicking_lower_genesis_213328_20260908_001/genesis/00000/contact_sheet.png)
- [Genesis frame 00018 contact sheet](../outputs/anatomy_retarget/v14_driver_axes_kicking_lower_genesis_213328_20260908_001/genesis/00018/contact_sheet.png)
- [Genesis frame 00035 contact sheet](../outputs/anatomy_retarget/v14_driver_axes_kicking_lower_genesis_213328_20260908_001/genesis/00035/contact_sheet.png)
- [wide frame 00000](../outputs/anatomy_retarget/v14_driver_axes_kicking_lower_genesis_213328_20260908_001/video_frames/00000.png), [wide frame 00018](../outputs/anatomy_retarget/v14_driver_axes_kicking_lower_genesis_213328_20260908_001/video_frames/00018.png), [wide frame 00035](../outputs/anatomy_retarget/v14_driver_axes_kicking_lower_genesis_213328_20260908_001/video_frames/00035.png)

| frame | source frame | numeric skin maxima | visual observation |
|---:|---:|---|---|
| 00000 | 120 | all bones 34.064 mm; left arm 6.733 mm; vessels 35.481 mm | Whole body is near upright. The oblique hip view shows the femoral head near the ilium rim and a red branch crossing the pelvic region. Knee and ankle silhouettes look connected, while vessels run immediately alongside the surfaces; the crop cannot establish clearance. |
| 00018 | 300 | all bones 41.691 mm; left arm 6.986 mm; vessels 43.534 mm | The whole body is in the kicking transition. The hip remains visually seated, but red branches crowd the iliac/femoral region. The lateral knee view shows the bent condyle/plateau arrangement and nearby vessel paths; the oblique ankle view is continuous but depth ordering is unresolved. |
| 00035 | 470 | all bones 31.922 mm; left arm 5.868 mm; vessels 24.151 mm | The body is returning toward upright. Hip and knee remain visually connected, and the ankle/foot chain is readable. Red vessels lie over or directly beside the lower bones, so the render cannot distinguish intended adjacency from penetration. |

The corresponding motion report has 36/36 evaluated frames, zero errors, and
records `anatomical_passed=false` and `publishable=false`. Recomputed directly
from this kicking-lower report, the 36-frame ranges are all-bone maximum
outside 25.509–47.841 mm (maximum at frame 00011), left-arm maximum
3.680–20.155 mm (maximum at frame 00014), and vessel maximum 21.343–48.278 mm
(maximum at frame 00011). These values belong to the kicking-lower sequence;
the left-arm metric is reported only for identity and is independent of this
lower-chain audit.
The images demonstrate that the sequence renders continuously; they do not
repair or waive the lower-chain contact flags.

This bounded result identifies the specific next geometry problem: the lower
chain meshes are individually closed, but their pose-dependent placements leave
side hip pairs and ankle pairs in triangle contact with multi-millimetre signed
penetration, with additional pose-specific knee failures. A later fit/runtime
change must be re-audited on these same six cells and must preserve the
separation between closure, contact, sampled depth, and anatomical acceptance.
