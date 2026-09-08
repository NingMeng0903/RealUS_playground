# V14 collar response and pivot diagnostic

This review isolates source controller **129, `Clavicle_Rot_L`**.  The
original 142 response is a `segment_root` driver with `a=9`, `b=13` and the
explicit position frame `(9, 13, 16)`.  Its frame uses the positions of those
SMPL-X joints and the reference axis from `joint_delta[9]`; it therefore does
not consume the local rotation state of SMPL-X collar joint 13.  Controller
131, `Shoulder_Rotate_L`, independently keeps its desired SMPL-X 16--18 world
orientation.  This is the omitted response being measured here.

The fixed-anchor candidate changes only controller 129's response declaration
to `joint_local`, `a=b=13`, with padded frame `(13, 13, -1)`.  Its coupling is
rebaked once at the neutral pose.  The source target bind, source hierarchy,
the 14-slot per-vertex weights, vertices and topology stay unchanged.  The effective asset
uses the existing full parent-local FK: 129 receives the joint-13 rotation,
130 (`bind_follow`) inherits it, and 131 plus the hand chain retain their
original desired world orientations.  The 129 origin remains on its source
parent-107/local-translation anchor; it is not set to the detached
`Delta13 @ B129` world translation.

The implementation is in
[`collar_response_v14.py`](../src/projects/genesis_ue_sync/anatomy_retarget/collar_response_v14.py).
`make_collar_response_asset_v14` and `apply_collar_response_v14` create the
effective copy.  `collar_response_runtime_arrays_v14` exposes detached
array-only driver fields and the baked coupling, while
`restore_collar_response_asset_v14` restores the original 129 declaration for
comparisons.  `source_bone_posed_global_collar_v14` evaluates the corrected
235-bone FK and `source_bone_local_correction_v14` returns the complete
parent-local correction array for an offline response bake.  Coupling rows for
the other 234 controllers are copied byte-for-byte; only row 129 is replaced.
`make_collar_pivot_response_asset_v14` composes that driver response with the
measured J13 target-pivot rebind and coherently stores all three target bind
matrices.

## Rest geometry and the fixed-anchor result

On the compiled 213328 V14 source, the fitted target bind and neutral SMPL-X
joint locations are:

| relation | distance |
| --- | ---: |
| `B129` to SMPL-X `J9` | 0.322154 mm |
| `B129` to SMPL-X `J13` | 86.610909 mm |
| `B130` to `J13` | 70.788533 mm |
| `B131` to `J16` | 0.156320 mm |

The first and last values show that the fitted controller origins were placed
near the old source shoulder and the target shoulder respectively.  The
86.6-mm middle value means that a joint-13 rotation response around a fixed
`B129` origin is a response correction, not a proof that the 129 origin is the
physical sternoclavicular socket.  The source pack has no mesh whose name or
metadata explicitly identifies a sternoclavicular capsule.  `Clavicle_L` is
used only as a geometric proxy in the audit.

The candidate removes the omitted 129 orientation response to numerical
precision while leaving the 129 origin unchanged:

| input | original 129 orientation error vs `Delta13 @ B129` | fixed-anchor candidate | 131 origin error before → after |
| --- | ---: | ---: | ---: |
| T-pose | 0.000001° | 0.000000° | 0.156314 → 0.156317 mm |
| 213328 own pose | 2.397245° | 0.000000° | 6.201895 → 3.727614 mm |
| 213712 own pose | 0.020621° | 0.000000° | 0.157323 → 0.167839 mm |
| held-out sitting | 21.245505° | 0.000000° | 48.667557 → 30.006741 mm |
| held-out kicking | 17.918425° | 0.000000° | 46.761657 → 26.601123 mm |

The fixed candidate keeps 131--135 world rotations within about
`2e-14°` of their original desired values.  It does move the downstream
world translations with the changed 130 inherited collar orientation: the
largest all-bone translation changes are 9.43 mm in the 213328 own pose,
74.21 mm in sitting and 69.69 mm in kicking.  That translation is a known
attachment risk for any tissue still authored against the old collar pivot.

The hand signed-distance audit against each captured skin surface remains a
failure, although it improves the large held-out drift:

| input | hand maximum outside before → after |
| --- | ---: |
| 213328 own pose | 7.138 → 5.250 mm |
| 213712 own pose | 0.796 → 0.806 mm |
| held-out sitting | 30.569 → 28.539 mm |
| held-out kicking | 32.728 → 22.311 mm |

These numbers are evidence for a bounded driver correction.  They do not pass
the requested anatomical tolerances.

## Collar-only support probe

The candidate was also evaluated on isolated SMPL-X joint-13 rotations of
`±30°`, with every other joint zero.  The 129 origin stays fixed in all six
probes and its orientation follows joint 13, but the shoulder-origin error is
not uniformly improved:

| axis/sign | original 131-to-J16 | fixed-anchor candidate |
| --- | ---: | ---: |
| 0 / −30° | 17.943 mm | 39.223 mm |
| 0 / +30° | 3.920 mm | 39.074 mm |
| 1 / −30° | 15.113 mm | 21.968 mm |
| 1 / +30° | 13.587 mm | 22.151 mm |
| 2 / −30° | 78.651 mm | 44.638 mm |
| 2 / +30° | 77.923 mm | 44.876 mm |

Thus the clean fixed-anchor response is useful for diagnosing the missing
local-collar input and improves the two held-out actions, but it is not a
universal production response over a ±30° collar domain.  The direct
`Delta13 @ B129` construction is diagnostic only: in the held-out actions it
differs from the parent-local anchored 129 matrix by about 30.14 mm and
26.72 mm in full-matrix norm, and would detach the source-parent attachment.

## Second diagnostic: rebind 129 to J13

Because the actual proximal `Clavicle_L` patch is around J13 while `B129` is
near J9, a separate in-memory diagnostic moved only the **target** bind origin
of 129 to neutral SMPL-X J13.  It recomputed every target parent-local bind
matrix and `target_inverse_bind`, kept B130's global bind unchanged, rebuilt
the 129 coupling, and left the source bind, mesh, topology and weights alone.
This is not part of `collar_response_v14.py` or the production runtime.

The rebind makes the 129 origin exactly J13 at rest, so its distance to J9
becomes 86.417191 mm.  It is a much better geometric hypothesis for the
proximal patch and almost eliminates the 131 origin drift in all tested
poses:

| input | 131-to-J16 after J13 rebind |
| --- | ---: |
| T-pose | 0.156322 mm |
| 213328 own pose | 0.155728 mm |
| 213712 own pose | 0.156315 mm |
| held-out sitting | 0.181675 mm |
| held-out kicking | 0.138683 mm |

Its reconstructed hand maximum outside the captured skin is 0.620 mm in T,
4.643 mm in the 213328 own pose, 0.796 mm in the 213712 pose, 1.030 mm in
sitting and 1.841 mm in kicking.  For a 77-vertex patch selected by proximity
to J13 (rest centroid 11.606 mm from J13 and about 91.1 mm from J9), the
fixed-anchor candidate's centroid displacement from the captured raw patch is
3.797 mm in the 213328 own pose, 0.023 mm in the 213712 pose, 32.017 mm in
sitting and 28.207 mm in kicking.  The J13 rebind reduces those values to
0.485, 0.004, 4.161 and 3.634 mm respectively; its patch centroid stays
11.43--11.61 mm from posed J13.  Relative to the fixed-anchor candidate, the
rebind patch moves 3.603, 0.019, 30.075 and 26.704 mm in those four poses.
Those movements must be reviewed against the real sternoclavicular capsule
and adjacent vessels before accepting this pivot.  If that attachment is
actually owned by parent 107/J9, the rebind is a clear 86-mm detachment even
though the shoulder and hand metrics improve.

## Verification and status

`tests/test_collar_response_v14.py` contains 16 tests against the real
compiled 235-controller source: immutable-copy checks, exact coupling-row
scope, array-view detachment, apply/restore, neutral bind recovery, six
collar probes, five captured/held-out poses and fail-closed contract checks.

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -q tests/test_collar_response_v14.py
16 passed
```

The full fixed-anchor audit is in
[`v14_collar_response_20260908_001/metrics.json`](../outputs/anatomy_retarget/v14_collar_response_20260908_001/metrics.json)
with compact controller arrays beside it.  The target-bind rebind diagnostic
is recorded in
[`v14_collar_response_rebind_j13_diagnostic_20260908_001/metrics.json`](../outputs/anatomy_retarget/v14_collar_response_rebind_j13_diagnostic_20260908_001/metrics.json).
Both audits are explicitly analysis-only and non-publishable.  The J13 target
bind rebase is now exposed as
`make_collar_pivot_response_asset_v14`; root's authorized compile and Genesis
review can apply it while checking the real capsule and proximal vessel
attachment.
