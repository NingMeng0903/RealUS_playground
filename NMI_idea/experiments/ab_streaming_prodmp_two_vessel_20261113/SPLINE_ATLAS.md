# Smooth material atlas regression

The frozen reference is the tetrahedral chart at 876e0c0. The alternative
uses the same angular stations, radial layers, graph-harmonic nodal depths,
and skin attachment stencils. It changes the interpolation of the volume and
the surface pose, carrying the same probe calibration and arm controller
settings. The legacy binding carries the original numeric vessel addresses.
The registered binding takes the old physical reference registrations,
expresses them once in the new spline, and freezes the resulting addresses
across every shape and pose. The trace records the old numeric addresses as
provenance as well as the new ones. The migration hash covers the reference
physical registrations and both address tables.

## Frozen closure test

Repeat all seven cohorts and all three controller arms in DYNAMIC_RESIDUAL.md.
For each spline run the simulator still generates observations using the
tetrahedral truth, scene tape and the common baseline initial robot state.
The candidate uses the spline for its public nominal vessels, probe
position, attached surface frame and planning. The same sampled IK and
execution checks apply. Count every run and compare the actual FK trajectory,
including unsuccessful runs. No candidate-specific retuning is allowed.

The primary comparison is the OU controller on the three 32-case, 15-degree
cohorts. The spline must complete at least as many valid scans, the paired
95% lower limit must exceed -5 percentage points, and the paired 95% cost
intervals must lie within 2 mm, 2 degrees and 0.10 s. Resample the original
disturbance IDs as clusters, carrying all linked shape instances together.
In the complete seven-cohort suite, no arm may have a lower total success
count or any newly accepted wrong identity. Report every cohort separately
and the approximate timing scope of the simulator. This regression gate
does not certify real-world noninferiority or exact equality of success
probabilities. Supporting cohorts include reused disturbance IDs.

The geometric gate is separate. The curved chart must be one-to-one and
locally regular on the used domain. Numeric point inversion, original knot
positions, graph-depth values, and derivative continuity are checked
separately from the closure results.

## Construction and certificate

Use the existing material lattice $(h,\theta,r)$, including its fixed
periodic angular seam. Let $S_{\beta,q}(h,\theta)$ be the tensor-product
interpolating cubic spline through the transported skin vertices, and let
$C_{\beta,q}(h)$ be the cubic spline through the transported core. The new
physical tube is

$$
F_{\beta,q}(h,\theta,r)=C_{\beta,q}(h)+r[S_{\beta,q}(h,\theta)-C_{\beta,q}(h)] .
$$

This reproduces the radial lattice to the precision of its stored vertices.
It changes the geometry between lattice knots, including between vertices of
an old planar skin triangle. The skin source face and barycentric stencils at
the lattice knots still come from the original SMPL-X mesh. The spline probe
frame uses the actual spline surface tangent and inward normal, and preserves
the probe rotation expressed in the corresponding reference frame. The
registered rotation is still the original interpolated calibration. Smooth
atlas coordinates and smooth surface directions do not imply that the
calibration's piecewise Slerp has continuous angular velocity.

Depth is reconstructed once from the positive gaps between consecutive
radial layers of the reference graph-harmonic field. Periodic interpolation
of the logarithms of the gaps, followed by positive normalization, keeps
their sum at one and reproduces the original nodal depths. The per-ray
Hermite interpolation uses the same knot slopes on neighboring intervals.
For adjacent positive secant magnitudes $a,b$, use

$$
m(a,b)=\frac{3ab}{a+b+\sqrt{(a-b)^2+ab}} .
$$

It reproduces a linear profile when $a=b$ and satisfies
$0<m<1.5\min(a,b)$. Hence the three Bernstein controls for the magnitude of
the derivative on an interval of secant magnitude $s$ are
$m_\mathrm{left}$, $3s-m_\mathrm{left}-m_\mathrm{right}$, and
$m_\mathrm{right}$, all strictly positive. Endpoint slopes use their
adjacent secant. The curve goes monotonically from depth 1 at the inner
shell to depth 0 at the skin. Smooth dependence of slopes on the gaps gives
joint C1 continuity in $(h,\theta,r)$. One common depth field and a unique
radial inverse let the numeric address $(\theta,h,d)$ identify the same
material sample for every admitted shape. This addresses model coordinates.
True vessel correspondence across actual people requires observations.

### Global one-to-one check

Embed $(r,\theta)$ as $(u,v)=(r\cos\theta,r\sin\theta)$ in the full solid
disk, so the enlarged parameter cylinder is convex. Extend the tube to its
axis using $F(h,0,0)=C(h)$. Away from the axis the $u,v$ Jacobian columns
are independent of $r$, and the $h$ column is affine in $r$. Choose a fixed
invertible output preconditioner and coordinate scale for the whole tube.
The same preconditioner handles both radial endpoints. The checker bounds
the symmetric part of the resulting Jacobian below by $mI$, with $m>0$, at
both $r=0$ (directional limiting values) and $r=1$, and hence at every
intermediate radius.

For two parameters $a,b$ in the scaled cylinder and their preconditioned
images $G(a),G(b)$, integrate the Jacobian along their line segment. This
gives $(G(b)-G(a))\cdot(b-a)\ge m\|b-a\|^2$. A segment crossing the axis
can be split at the crossing, and a segment contained in the axis uses
the core derivative. This proves global injection on the cylinder and
thus on the material shell. Positive symmetric part also rules out a
singular Jacobian away from the axis. The material region starts at
$r=0.04$ and excludes the radial coordinate singularity at the axis.

ASTRA MAX implemented the independent enclosure in spline_certificate.py.
Bernstein bounds enclose each actual cubic patch on its full interval,
using outward-rounded interval arithmetic and interval trigonometry.
Sylvester minors certify the candidate positive lower bound. The sampled
Jacobians are only used to propose the preconditioner and to reject early
when it is clearly insufficient. The proof applies to the continuously
glued mathematical spline. The implementation reports the floating-point
seam and derivative residuals and refuses failed or inconclusive
certificates. The audit also tests the prior folded configuration and a
constructed two-to-one tube with positive sampled local determinants.

## Reproduce

Use the same runtime and input assets as DYNAMIC_RESIDUAL.md, with NumPy,
SciPy and mpmath available. The default --chart tet is the reference.
Use either legacy or registered as the binding below. The --first-case
option skips the original tape generator to a given ID for sharding. The
runner merges and checks contiguous case slices without dropping cases.

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
    python run_spline_suite.py --among-us "$among" --repair "$repair" --spline-binding "$binding" --out "$out"
    python audit_spline_atlas.py --among-us "$among" --repair "$repair" --spline-binding "$binding" --out "$out/GEOMETRY.json"
    python compare_spline.py --baseline verification/DYNAMIC --spline "$out" --out "$out/COMPARISON.json"
    python verify_atlas_scan.py --among-us "$among" --repair "$repair" --runs "$out/reference" --out "$out/reference/KINEMATIC_REPLAY.json"

For the FK and observation replay, run the last command for all seven cohort
directories. The original forecast analysis can also be run on the spline's
public histories. In that analysis, motion tapes express physical offsets
relative to the fixed simulator world, whereas the spline is permitted a
small shift in its initial nominal prior. Its forecasting accuracy includes
that mismatch.

## Legacy binding result

The first frozen closure trial reused the old numeric vessel labels
directly. The five charts passed the continuous-domain geometry check.
The geometry comparison found a maximum nominal vessel shift of 0.1194 mm.
The full test rejected this binding under the preset all-arm gate. The
static-map controller fell from 151/172 to 147/172. All four losses were
reuses of the same disturbance ID, and all became false completions. At the
decisive frame, the original absolute lateral errors were 2.975 to 2.996 mm
and the spline errors were 3.030 to 3.054 mm against the fixed 3 mm cutoff.
The IID arm exchanged one win for one loss and remained at 144/172. The OU
arm matched all 172 outcomes, including all 80/96 successes in the primary
comparison. There were zero wrong identities or execution failures in both
implementations.

The registered binding addresses the measured nominal-prior shift by
preserving the original physical reference registration. Its address
conversion uses only the archived 130 nominal reference points. The
simulator truth, dynamic disturbance, controller weights, entry state
and 3 mm acceptance limit all retain the frozen settings. The repeat of
the same benchmark is a software regression check, since the first trial
prompted the calibration correction.

## Registered binding result

All five benchmark geometries passed the interval check on the whole
cylinder. The smallest certified normalized symmetric-Jacobian bound
was 0.08126. The prior folded case at 90 degrees and the deliberately
overlapping two-turn tube were rejected. The two-turn example retained
positive sampled local determinants, so determinant signs alone would
miss its global collision. All 22,400 inverse queries were
checked across the five charts, including the seam and both shells.
Maximum physical round-trip error was 2.0 nm in the model. The original
graph depths at lattice nodes agree to $2.3\times10^{-16}$.
The rebind reconstructs the 130 registered reference points to 1.8 nm
and reduces the maximum prior difference across tested configurations
from 0.1194 mm to 0.0147 mm.

| Arm | Tet and registered, primary 96 | Tet and registered, full 172 | Paired success disagreements, full |
| --- | ---: | ---: | ---: |
| Static map | 78 / 78 | 151 / 151 | 0 |
| IID residual | 73 / 73 | 144 / 144 | 0 |
| OU residual | 80 / 80 | 153 / 153 | 0 |

The registered binding passed the preset full regression gate. Both
implementations had zero accepted wrong identities and zero run failures.
Their false-completion counts were also identical, at 6, 12 and 3.
For primary OU, the mean actual TCP travel difference was -0.00444 mm
with a paired 95% interval of [-0.02750, 0.01171] mm. Mean rotation
difference was +0.00895 degrees, with a paired 95% interval of
[0.00803, 0.00981] degrees. All scans ran for 4 simulated seconds with 41
acquisitions. The replay recomputed 21,156 image packets and 103,716 FK
samples. The source comparison asserts the paths' common initial FK and
all disturbance tapes exactly.

A zero-discordance bootstrap describes this fixed replay. With 32 distinct
primary disturbance IDs, the supplementary exact one-sided 95% upper
bound for the probability that a new independent tape produces any
lost OU success across the three tested shapes is still 8.94%. The
geometry certificate holds for the tested shape and pose instances and
the specified mathematical interpolant. The closed loops use a fixed
body shape and knee pose for each run, synthetic centroid observations,
and ideal sampled robot kinematics with zero simulated planning latency.

The original OU 100 ms scalar forecast RMSE changed from about 0.921 to
0.923 mm when replaying the spline arm's public history. In a separate
single-process pose generation microbenchmark at the reference geometry,
one query used median CPU time 0.40 ms (tet) and 0.69 ms (spline).
A 51-pose candidate batch used 6.87 ms (tet) and 0.46 ms (spline).
These numbers exclude setup, IK and planning, and do not establish
end-to-end real-time performance.

Use --chart spline to enable the registered binding. Pass
--spline-binding legacy to reproduce the first trial. Archived data
include the [legacy comparison](verification/SPLINE/legacy/COMPARISON.json),
[registered comparison](verification/SPLINE/registered/COMPARISON.json)
and [geometry audit](verification/SPLINE/registered/GEOMETRY.json).
The compressed summaries can be expanded with xz -dk before invoking
verify_atlas_scan.py. The stored digests refer to the replay inputs
listed in each report.
