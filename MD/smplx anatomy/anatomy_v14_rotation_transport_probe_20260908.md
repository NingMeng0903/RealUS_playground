# V14 local-rotation transport probe — 2026-09-08

This is a bounded mathematical ablation. It loads the compiled
`v14_original_shape_restfit_213328_20260908_002` subject and the nine existing
pose NPZs, then evaluates the current response and a rotation-only local
transport variant. It does not alter runtime, bind, weights, vessels, or any
fit result.

Machine-readable output:
[`v14_rotation_transport_probe_20260908_001/report.json`](../outputs/anatomy_retarget/v14_rotation_transport_probe_20260908_001/report.json)

Genesis-compatible NPZs:
[`v14_rotation_transport_probe_20260908_001/`](../outputs/anatomy_retarget/v14_rotation_transport_probe_20260908_001/)

The previous trace/acos report is retained as
[`v14_rotation_transport_probe_20260908_001_rawtrace/`](../outputs/anatomy_retarget/v14_rotation_transport_probe_20260908_001_rawtrace/)
for provenance; the `_001` directory is the corrected report.

## Composition tested

For each of the 235 controllers, the current local response is

`D_i = inv(L0bind_i) @ L0pose_i`.

The primary ablation retains the existing translation map and changes only
the local rotation:

`Rrootmap = Rtarget_bind_root @ Rreference_bind_root.T`

`Q_i = Rreference_bind_i.T @ Rrootmap.T @ Rtarget_bind_i`

`Dprime_i.R = Q_i.T @ D_i.R @ Q_i`.

All positions are still produced by 235-controller parent FK with the target
bind translations. A separately stored diagnostic also applies `Q_i.T` to
the existing mapped translation, but it is not the primary variant.

The `q_controller` array and primary geometry use the raw bind-derived `Q`.
Angular errors in this report are evaluated as
`Rotation.from_matrix(project_so3(actual) @ project_so3(expected).T).magnitude()`;
the matrix error is the pre-projection maximum absolute element difference.

## Mathematical consistency result

The expected primary rotation is
`Rrootmap @ RsourceG_i @ Q_i`. Across all nine poses, the corrected metric
gives a maximum rotation error of **3.51e-6 degrees** (the T-pose cell; the
largest non-neutral value is **3.06e-9 degrees**) and a maximum pre-projection
matrix difference of **3.064e-5**. The raw bind-derived Q has maximum
orthogonality error **1.188e-6**, determinant range **0.99999897–1.00000056**,
and differs from its proper-SO(3) projection by at most **5.95e-7** in any
matrix element. Thus the former approximately 0.414°
`rotation_conjugated_vs_expected` value was numerical trace/acos drift, not a
composition failure.

| pose | current→conjugated max angle | current→conjugated translation | full left-arm skin max outside, current → conjugated |
|---|---:|---:|---:|
| T-pose | 0.000° | 0.000 mm | 0.000 → 0.000 mm |
| own 213328 | 5.334° | 38.340 mm | 44.825 → 14.265 mm |
| own 213712 | 3.278° | 6.327 mm | 34.832 → 31.757 mm |
| elbow 30° | 0.098° | 0.517 mm | 1.625 → 1.369 mm |
| elbow 60° | 0.628° | 2.478 mm | 12.924 → 11.836 mm |
| elbow 90° | 1.113° | 2.978 mm | 23.790 → 23.051 mm |
| elbow 120° | 1.517° | 3.119 mm | 33.015 → 32.540 mm |
| held-out sitting | 5.179° | 22.395 mm | 27.079 → 30.643 mm |
| held-out kicking | 2.771° | 4.536 mm | 30.697 → 29.444 mm |

The conjugated rotation changes the existing compiled response materially in
the 213328 own pose and improves its full-arm outside maximum, while the
213712 own pose remains poor. Sitting becomes worse (27.079 to 30.643 mm),
so this bounded result does not select the variant as a production rule.
The translation-conjugated diagnostic has nearly the same geometry in this
asset; its values remain separately labeled in every NPZ/report cell.

## Scope

This probe answers the frame-consistency question only. It is marked
`probe_only=true`, `fit_or_tuning_performed=false`,
`anatomical_passed=false`, and `publishable=false`. The reported skin and
elbow metrics are evidence for comparing the two transports and do not certify
joint anatomy or tube–bone clearance.

