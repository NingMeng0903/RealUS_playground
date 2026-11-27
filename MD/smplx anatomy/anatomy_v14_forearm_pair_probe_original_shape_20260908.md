# V14 original-shape Radius–Ulna rest-pair probe — 2026-09-08

This is a bounded static feasibility experiment. It starts from
`v14_original_shape_restfit_213328_20260908_002`, whose upstream report marks
`shape_source=original142`, and applies independent rigid offsets to
`Radius_L` and `Ulna_L`. It does not modify the retarget runtime, bind,
weights, vessels, or the production fit.

Machine-readable result:
[`v14_forearm_pair_probe_20260908_002/report.json`](../outputs/anatomy_retarget/v14_forearm_pair_probe_20260908_002/report.json)

Renderable before/after geometry:
[`subject_213328_tpose_forearm_pair_probe.npz`](../outputs/anatomy_retarget/v14_forearm_pair_probe_20260908_002/subject_213328_tpose_forearm_pair_probe.npz)

## Source-reference change

The earlier probe used the neutral restfit candidate from
`v14_arm_restfit_20260908_005`. This follow-up uses the separate
`v14_original_shape_restfit_213328_20260908_002` T-pose input after the
upstream original-shape reference repair. The independent offsets therefore
start from a different placement and must be read as a separate feasibility
measurement, not as a version ranking.

The probe's exact source is the input file recorded in the JSON. Its
`shape_source=original142` upstream report is kept as provenance; the probe
itself only consumes the exported geometry and frozen operator/calibration.

## Objective and bounds

The SLSQP objective used only frozen `fit` domains for bone queries, with a
positive target clearance of **0.25 mm** and a nearest signed-sample gap target
of **≤3 mm**. It also evaluated every one of the 893 Radius/Ulna vertices
against the supplied skin. The six directed bone queries were:

| query domain | target |
|---|---|
| `elbow/left/humerus.fit` | `Radius_L` |
| `elbow/left/radius.fit` | `Humerus_L` |
| `elbow/left/humerus.fit` | `Ulna_L` |
| `elbow/left/ulna.fit` | `Humerus_L` |
| `elbow/left/radius.fit` | `Ulna_L` |
| `elbow/left/ulna.fit` | `Radius_L` |

Each bone had an independent rigid rotation about its proximal fit-domain
centroid and translation, constrained to 3 degrees and 5 mm norm. The wrist
and Scaphoid geometry stayed fixed. Validation and complete-triangle reports
retain the original **0.5 mm** penetration threshold; that acceptance value is
separate from the 0.25 mm fit target.

## Fit result

| bone | rotation | translation norm | fit bound |
|---|---:|---:|---|
| `Radius_L` | 0.8209° | 1.0644 mm | within 3° / 5 mm |
| `Ulna_L` | 0.6055° | 1.7846 mm | within 3° / 5 mm |

The exact-zero input score is `58.3356`; the chosen feasible trial score is
`2.883e-7`. All six directed fit queries meet the 0.25 mm signed target and
the 3 mm nearest-gap target after the probe. The 893-vertex forearm skin
population remains fully inside the skin, with maximum outside distance 0 mm.

## Independent validation

The original 0.5 mm reporting threshold gives the following directed
Radius–Ulna results:

| state | radius → ulna minimum / count below 0.5 mm | ulna → radius minimum / count below 0.5 mm |
|---|---:|---:|
| zero transform | −1.180 mm / 2 | −2.198 mm / 5 |
| chosen offsets | −1.017 mm / 1 | −0.730 mm / 2 |

The fit-domain clearance result therefore does not transfer to the frozen
validation samples. This is sample overfitting evidence, not an acceptance
pass.

## Complete triangle crossings and coverage

The complete surface audit reports:

| pair | zero transform | chosen offsets | location after chosen offsets |
|---|---:|---:|---|
| Humerus–Radius | 19 | 0 | no remaining crossing |
| Humerus–Ulna | 0 | 0 | no crossing |
| Radius–Ulna | 83 | **55** | 23 proximal, 32 distal, 0 midshaft |

The remaining Radius–Ulna crossings span normalized forearm axial positions
`0.024–1.023`. The union of the two Radius/Ulna elbow fit-query axial ranges
is only `−0.061–0.183`; only **41.8%** of the remaining crossing centroids
fall inside that axial range. The distal crossings (32/55) lie outside the
fit-query range, with representative crossing triangles such as local
`Radius_L` triangle 395 against `Ulna_L` triangle 954 at normalized axial
position 1.004 and approximately 210.7 mm nearest distance from the fit
query points. The proximal crossings are inside the axial range, but point
queries still cannot certify complete triangle non-intersection there.

This identifies the graph gap: the current elbow fit caps constrain a proximal
sampled station, while the complete Radius–Ulna surfaces still overlap at the
distal wrist-side end and at some proximal triangles. A production solution
would need the forearm pair's full bone-surface relationship, plus its distal
wrist/carpal attachment, represented in the bone graph. Any resulting rigid
or non-rigid field would then have to propagate consistently to the linked
vessels and downstream wrist/hand bones. This probe intentionally does none of
that, so its geometry must not be promoted to runtime.

## Wrist diagnostics

Radius/Ulna-to-Scaphoid queries remain diagnostics only. The validation minimum
signed values change from `+1.952/+11.233 mm` (radius/ulna) to
`+1.936/+12.329 mm`. Both diagnostics have `gap_gate_applied=false`; the
Ulna–Scaphoid entry is explicitly marked as a TFCC-side diagnostic with no
direct-contact requirement. The probe does not force an Ulna–Scaphoid gap.

## Scope and identity

The two moved meshes remain rigid: pairwise shape deltas are below
`2.3e-16 m`. Every other vertex is unchanged, including all vessel vertices.
The JSON records `probe_only=true`, `production_candidate=false`,
`anatomical_passed=false`, and `publishable=false`.

| item | value |
|---|---|
| input | `outputs/anatomy_retarget/v14_original_shape_restfit_213328_20260908_002/subject_213328_tpose.npz` |
| input SHA-256 | `537b0e091c55b78a01d92a8563809a20d345546539d336b5556cc112dbe04f7b` |
| operator digest | `17f5d4e0bc328e85aef0d6dc6eba0e3fa8ca1ddd0a79f751ae259e129d00972b` |
| calibration content digest | `c040ed0df9f7f37cb1235895458346540b0228d62e509026327ed16600bdaca5` |
| calibration fixed-domain digest | `d86806ecdb2aee09725f291c1add90fcede3dcfbf66cef7bf1cf2e70488297f7` |
| probe script SHA-256 | `ca3bb04e81e410b8eb14dde3b297559616b218ce6a0f2ef540189452130ef96d` |
| output NPZ SHA-256 | `2d5b601982405bb1a9721f80f425000560fbc8771354b4d8daa69f500d4a1ac7` |
