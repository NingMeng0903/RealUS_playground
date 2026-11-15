# Identifying a knee-angle vessel response in streaming ProDMP MPC

## Protocol

The fixed material chart transports the nominal leg and centrelines. The
synthetic world adds shared lateral motion to each vertex of the two vessel
curves before intersecting them with the actual FK probe plane.

$$
d_t(s)=w_{\mathrm{lat}}+z_t+k(q_t-15^\circ)E(s).
$$

Here $q_t$ is the programmed body angle. $E(s)$ is one through 5 mm
from the knee-side end of the existing 31.41 mm corridor, falls by a cosine
taper to zero at 17 mm, and stays zero beyond it. The end was selected by
the nearest reference artery sample to the SMPL-X knee joint. The
existing per-case persistent offset, calibration discrepancy, OU motion,
dropout and centroid noise remain. The primary cases draw a continuous
signed slope with magnitude uniform on 0.12–0.24 mm/degree. The fitted
model uses a 17-cell slope lattice from −0.32 to 0.32 mm/degree and
normal cell masses with a 0.15 mm/degree prior scale. The two signs and
millimetre-scale excursions test the mechanism. They are simulation
settings rather than measurements of vascular displacement in people.

Record a common reference pass at 15 degrees. Add two more visits at
15 degrees for the fixed-angle control. For the multiple-angle history
use the same reference pass and revisit the same material stations at
5 and 25 degrees. The two estimators condition on the same public
packets and actual FK history for each visit. The two three-visit
histories share respective sensor/OU tapes and have the same exposure
schedule. Report realized calibration costs separately. Only the
same-person joint persistent posterior carries across visits. Each new
visit starts an independent OU and visibility tape. At the reference
angle the likelihood is flat in the slope at fixed offset. An
independent initial slope prior stays unchanged under reference-only
acquisition. A reference visit can update the slope indirectly after
multiple-angle acquisitions correlate the slope and persistent offset.

Evaluate 24 response cases and 12 independent null cases with $k=0$ under the same
protocol. Each history leads to paired executions of the original
$w+z$ MPC and the joint $w+g+z$ MPC. Both start at the same robot
state and receive the same programmed angle trajectory and random
tapes. The held-out angle is a smooth flexion/extension excursion of
eight degrees around 15 degrees, returning to 15 degrees after 2.4 or
2.8 seconds. Excursion sign and duration are counterbalanced by case ID.
The forecasts are conditional on this public programmed input. A
trajectory candidate uses the supplied body angle at each robot sample.
The physical world uses transported tetrahedral centrelines and the
planner uses the registered spline atlas. Between five-degree anchors,
interpolate corresponding positions and rotations. Check midpoint
position and orientation against newly evaluated SMPL-X atlases before
running cases. These comparisons assess the interpolation at sampled
poses.

Both arms use the existing two-segment score, FK and joint constraints,
trained ProDMP, anonymous two-role likelihood, acceptance threshold and
in-segment feedback. The simulator generates packets at each arm's
realized probe pose. The new arm marginalizes slope and visibility in
the joint prediction. The slope is unchanged by state propagation.

## Outcomes

Report the signed-slope posterior error, interval width and nominal 90%
coverage across cases after each acquisition history. The primary
forecast diagnostic uses the 0.5 s lateral forecast on a shared held-out
packet/FK history for each case, at future stations where
$|(q-15^\circ)E(s)|$ is at least one degree. Also report 0.1 and 1 s,
all stations and origins with no arterial return. Score forecast errors,
interval coverage and width against the effective map offset, which
includes the simulator's calibration discrepancy.

On the paired executions count all attempts, correct frames, correct
completion, wrong identities, false acceptance, false completion and
execution failures. Report association quality on the common public
history as well. A frame counts as correct only if the selected
identity and physical centering are correct. Compute paired 95%
intervals by resampling case IDs. Check actual TCP distance, rotation
and time against the previous ±2 mm, ±2 degree and ±0.10 s equivalence
margins. A positive completion claim requires a positive paired interval,
compatible costs and the observed identity outcomes. Treat the null cohort
as a check for false slope and degradation. A mechanism check with
synthetic centroids does not establish a human vascular response or
transfer outside the observed angle range.

## Results

All 180 acquisition visits and all 216 held-out executions completed
the 41-frame schedule. The repeated fixed-angle and multiple-angle
histories each supplied 2,952 frames in the response cohort. Mean
calibration TCP travel per visit was 24.44 and 23.98 mm, respectively.
These costs cover scans from their registered initial conditions.
The reported mechanical costs omit positioning between visits and
simulation compute time.

| Acquisition history | Slope RMSE (mm/degree) | Mean slope posterior SD (mm/degree) | Slope interval coverage |
|---|---:|---:|---:|
| One 15-degree pass | 0.193 | 0.146 | 24/24 |
| Three 15-degree passes | 0.193 | 0.146 | 24/24 |
| One pass each at 15, 5 and 25 degrees | 0.071 | 0.071 | 21/24 |

The first two histories leave the slope prior unchanged. The paired
multiple-angle minus repeated-reference slope RMSE was −0.122 mm/degree
with a 95% bootstrap interval of [−0.141, −0.103]. Ten of 24
multiple-angle intervals excluded zero in the correct direction.
The latent effective lateral intervals cover the uncertainty in the
offset and OU state at the scheduled material stations. Their coverage
concerns that offset under the supplied angle program. Image-domain
measurement noise and physical centering have separate outcome measures.

After multiple-angle acquisition, the prespecified 0.5 s forecast in
the active knee region was 1.462 versus 1.383 mm RMSE for the original
and slope-aware models. The paired change was −0.078 mm with a 95%
interval of [−0.167, +0.017]. The 90% interval coverage was 84.7% and
88.4%, with mean widths of 4.37 and 4.46 mm. The 0.1 s diagnostic
in the same region was 0.964 versus 0.883 mm RMSE, paired change
[−0.121, −0.039] mm. Within the slope-aware model, varied-angle
acquisition reduced the primary interval width from 4.61 to 4.46 mm
relative to repeated reference acquisition. Its primary RMSE change
had an interval of [−0.150, +0.019] mm.

| Multiple-angle acquisition | Original OU MPC | Slope-aware OU MPC |
|---|---:|---:|
| Response cases with correct completion | 20/24 | 20/24 |
| Response frames with correct identity and physical centering | 729/984 | 733/984 |
| Response false acceptances | 14 | 18 |
| Wrong accepted identities in response cases | 0 | 0 |
| Null cases with correct completion | 11/12 | 11/12 |
| Null false acceptances | 4 | 4 |

All paired completion outcomes agree case by case. The response cohort
correct-frame difference was +0.41 percentage points, with a paired
interval of [−0.61, +1.42]. The mean TCP travel difference was +0.078 mm,
with a paired interval of [−0.149, +0.303] mm. The realized
travel, rotation and duration comparisons passed the stated equivalence
margins in each history. Both controllers made in-segment ProDMP revisions
throughout the benchmark.

Across the same exposed packets in the multiple-angle response replay,
the controllers accepted the same 743 slots and their probabilities
differed by at most 1.02e−14. The extra false acceptances in the executed
paths were physical centering errors with the right identity. Two of
the four incomplete response cases had a required region with no
artery return on either realized path. The other two returned an
artery without accepting a centred image in the missed region. In the
null cohort one of 12 slope intervals excluded zero before the held-out
replay. The null result and the absence of observed identity errors
are conditional on this scan and simulator.

ASTRA MAX independently recomputed the slope, forecast and closed-loop
results. SOL checked the exact two-role update, the original scan
regression, and replayed the pilot and one frozen case per cohort.
The independent replay checked 6,633 FK samples and 1,353 exposure
packets. It verified the executed ProDMP suffixes and ran a hidden-tape
causality check on the pilot. The structured injection uses the
same known spatial envelope that the estimator uses. The results
support identifying the local slope and a short-horizon prediction
benefit under that model. The primary forecast RMSE advantage,
recognition benefit and completion advantage remain unresolved.

[Aggregated statistics](verification/KNEE_RESPONSE/ANALYSIS.json) ·
[Response cases](verification/KNEE_RESPONSE/response/CASES.json.xz) ·
[Null cases](verification/KNEE_RESPONSE/null/CASES.json.xz) ·
[Independent checks](verification/KNEE_RESPONSE/independent/README.md)

## Reproduce

From this experiment directory, use the same Among_US, repair assets and
Python environment as the registered spline study. The output directory
can be outside the checkout.

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
    python run.py knee-response --among-us "$among" --repair "$repair" --cohort response --cases 24 --out "$out/response"
    python run.py knee-response --among-us "$among" --repair "$repair" --cohort null --cases 12 --out "$out/null"
    python analyze_knee_response.py --runs "$out/response" "$out/null" --out "$out/merged"

Use `--first-case` with a smaller `--cases` to shard either cohort. Give
all shard directories to the analysis script. It checks the frozen case
IDs and shared metadata before writing the merged archives. `CASES.json.xz`
contains the individual outcomes, public packets, assignment probabilities
and forecast intervals. `TRACES.npz` contains realized FK, joint and
ProDMP samples, body-angle tapes and the evaluator's disturbance tapes.
