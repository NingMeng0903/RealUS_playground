# V14 independent Radius–Ulna rest-pair feasibility probe — 2026-09-08

This is a static ablation of the V14 neutral rest geometry. It starts from
the `tpose` candidate in
`outputs/anatomy_retarget/v14_arm_restfit_20260908_005` and applies separate
rigid transforms to `Radius_L` and `Ulna_L`. It does not modify a retarget
map, bind, weights, vessels, runtime, or any production candidate.

The machine-readable result is
[`v14_forearm_pair_probe_20260908_001/report.json`](../outputs/anatomy_retarget/v14_forearm_pair_probe_20260908_001/report.json).
The renderable before/after geometry is
[`subject_213328_tpose_forearm_pair_probe.npz`](../outputs/anatomy_retarget/v14_forearm_pair_probe_20260908_001/subject_213328_tpose_forearm_pair_probe.npz).

## Probe contract

The SLSQP objective queried only the frozen `fit` domains. It included all six
directions of the three elbow surface pairs and every vertex of both forearm
meshes against the supplied T-pose SMPL-X skin:

| objective term | query | target |
|---|---|---|
| humerus → radius | `elbow/left/humerus.fit` | complete `Radius_L` |
| radius → humerus | `elbow/left/radius.fit` | complete `Humerus_L` |
| humerus → ulna | `elbow/left/humerus.fit` | complete `Ulna_L` |
| ulna → humerus | `elbow/left/ulna.fit` | complete `Humerus_L` |
| radius → ulna | `elbow/left/radius.fit` | complete `Ulna_L` |
| ulna → radius | `elbow/left/ulna.fit` | complete `Radius_L` |
| skin | all 366 `Radius_L` + 527 `Ulna_L` vertices | T-pose SMPL-X skin |

Each bone has a 12-parameter block in normalized coordinates. The selected
transform rotates about that bone's proximal `elbow/left/<bone>.fit`
centroid and allows at most 3 degrees rotation and 5 mm translation norm. The
wrist is held fixed. Radius/Ulna-to-Scaphoid values are diagnostics only; the
probe has no wrist objective, and it does not require an Ulna–Scaphoid gap.

## Result

The optimizer found a feasible independent placement with a much lower fit
objective, using only small offsets:

| bone | rotation | translation norm | bound result |
|---|---:|---:|---|
| `Radius_L` | 0.5196° | 0.8660 mm | within 3° / 5 mm |
| `Ulna_L` | 0.5193° | 0.8660 mm | within 3° / 5 mm |

The score fell from `201.719` at the exact zero transform to `1.20e-7`.
Every forearm vertex stayed inside the skin in both before and after states
(893 vertices; maximum outside distance 0 mm).

## Bone overlap evidence

Signed values below use the independent 0.5 mm validation threshold. Negative
values indicate a sample inside the target bone.

| state | radius → ulna minimum / count | ulna → radius minimum / count | full Radius–Ulna triangles |
|---|---:|---:|---:|
| zero transform on rest5 candidate | −1.185 mm / 2 | −2.208 mm / 5 | 83 |
| chosen independent offsets | −0.660 mm / 1 | −0.511 mm / 1 | 43 |

The Humerus–Radius complete triangle contacts fall from 13 to 0 and
Humerus–Ulna remains at 0. The Radius–Ulna pair still has 43 complete-surface
contacts and one validation sample deeper than 0.5 mm in each direction. The
bounded independent rigid offsets therefore improve the intrinsic overlap but
are **not sufficient** to establish full Radius–Ulna non-overlap.

The validation-domain humerus/forearm directions are all outside the 0.5 mm
penetration threshold after the probe (`humerus→radius` 1.023 mm,
`radius→humerus` 0.727 mm, `humerus→ulna` 0.799 mm, `ulna→humerus` 0.563 mm).
Those local improvements do not override the remaining complete-pair failure.

## Wrist diagnostic

The radius-to-Scaphoid minimum signed value changes from 1.933 mm to 1.445 mm.
The Ulna-to-Scaphoid diagnostic changes from 11.211 mm to 12.203 mm. Both
queries are explicitly marked `gap_gate_applied=false`; the Ulna-side value is
annotated as a TFCC-side diagnostic with no direct-contact requirement. The
probe therefore does not turn the anatomical wrist separation into a false
failure or a forced contact target.

## Shape and scope checks

Each moved mesh remains rigid: maximum pairwise shape delta is below
`2.3e-16 m`, with determinant 1.0. All non-forearm vertices are unchanged
exactly; vessel displacement is 0 m. The NPZ is a diagnostic before/after
geometry export and is not a runtime artifact. The JSON keeps
`production_candidate=false`, `anatomical_passed=false`, and
`publishable=false`.

This single static experiment tests bounded sufficiency of separate forearm
placement. It does not establish that independent placement is necessary, and
it cannot be used as an ordinary linkage-preserving retarget because the
probe intentionally moves bone meshes without transporting the associated
vessels or updating bind/runtime data.

## Identity

| item | value |
|---|---|
| input | `outputs/anatomy_retarget/v14_arm_restfit_20260908_005/subject_213328_tpose.npz` |
| input SHA-256 | `e8d4239a532629d92fa020090da9909500442dc5b8d6c116a0b50af5f93e25e1` |
| operator digest | `17f5d4e0bc328e85aef0d6dc6eba0e3fa8ca1ddd0a79f751ae259e129d00972b` |
| calibration content digest | `c040ed0df9f7f37cb1235895458346540b0228d62e509026327ed16600bdaca5` |
| calibration fixed-domain digest | `d86806ecdb2aee09725f291c1add90fcede3dcfbf66cef7bf1cf2e70488297f7` |
| probe script SHA-256 | `f33d947c8bf2948f60e84e204d803043ecf81efa743b07c359f2ac532bec4ac9` |

