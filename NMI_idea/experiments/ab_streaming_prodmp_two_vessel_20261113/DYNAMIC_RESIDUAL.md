# Dynamic vessel residual in the streaming leg scan

The existing right-leg chart, one artery and one vein, two-step MPC, servo-trained ProDMP, and sampled RM75 kinematics form the test bed. Each arm starts from the same transported material chart and robot state. The leg surface stays fixed within a run. A shared lateral displacement moves the two vessel centrelines relative to the skin. This test isolates unmeasured local motion in the internal map.

## State and controller

Write the effective displacement relative to the chart as `w + (z, 0)`, in physical millimetres on the two instantiated offset axes. The prior for the persistent offset `w` is the existing grid. The new scalar lateral state uses a zero-mean Ornstein–Uhlenbeck process with time constant 0.8 s and stationary standard deviation 1.5 mm. At interval Δ the retention is `a = exp(-Δ/0.8)` and innovation variance is `1.5²(1-a²)`. This benchmark has no measured motion-response input. The effective offset used by the measurement includes the existing case calibration discrepancy. The evaluator subtracts that discrepancy for physical vessel placement.

The filter retains the joint posterior of `w`, `z` and the two visibility states. It predicts once per 100 ms frame and then evaluates the likelihood of the anonymous packet at actual FK, marginalizing assignments. Natural empty packets are observations. The OU filter integrates the transition over source and destination cells of a ±6 mm lattice with 0.5 mm spacing and tail cells. The IID filter uses the same stationary marginal and resets the residual independently at each later exposure. The static map uses the original constant offset and visibility filter. The planner evaluates the full predicted mixture at each future exposure it scores. Within a segment, the image residual is computed after the update, and the goal correction uses the end-of-segment forecast. The remaining ProDMP is reconditioned from its realized lateral position and velocity. [Implementation](residual.py) · [Scan loop](atlas_scan.py)

## Paired protocol

- Primary cohorts use the reference β, Δβ₀/Δβ₁ = (−1, +1), and Δβ₀/Δβ₁ = (+1, −1) at 15° knee, with 32 OU disturbances per shape. Two supporting cohorts use Δβ₂ = +0.5 and 25° knee respectively, with 16 OU disturbances each. They reuse the first 16 disturbance IDs.
- A separate 32-case cohort uses the reference geometry and one unmodelled signed transient per case, with a 0.3 s rise and an exponential recovery. It tests model mismatch. The transient results are reported separately.
- Every case has the same static offset, calibration bias, motion tape, Markov visibility and noise in the three arms. Each arm acquires its own packet at its realized FK pose. Static-map, IID and OU share the candidate and fallback rules, realized start, feedback schedule, ProDMP weights, identity thresholds and execution checks. Each has at most 4 s and 41 acquisitions. The 100 ms acquisition schedule assumes simulated planning latency of zero.
- Completion requires a new, correctly associated, physically centred arterial return in each of five longitudinal bins, execution to the end, and zero wrong identity declarations. Report correct completion, wrong and false completion, execution rejection, physical tracking error, actual TCP travel, turn, time and acquisition count. Every attempted case stays in the denominator. A loss-recovery diagnostic uses the first run of at least three arterial dropout frames from the sampled visibility chain starting at or after 0.5 s and leaving five frames after the loss ends. Count a recovery when one of those five frames contains an accepted, physically centred correct arterial observation.
- On the static-map arm's public trajectory and packets, replay all three estimators. Score effective lateral offset forecasts at 0, 0.1, 0.3, 0.5 and 1 s against the same evaluator tape. Report RMSE, mean error, 90% interval coverage and width, plus results for prediction origins whose current packet lacks a true artery return. The evaluation uses the full observed prefix available at the origin and no future image or FK. Run a 0.25 mm grid replay on the reference cohort and report the effect of changing resolution.
- The primary outcome comparison is OU versus IID on the three 15° cohorts. OU versus static is the comparison with the historical controller. Report pointwise paired 95% intervals and discordant success counts. Cluster aggregate primary results by shared disturbance ID across shapes. For the same-cost comparison, assess mean differences in realized TCP travel, turn, and duration against ±2 mm, ±2° and ±0.10 s using the paired intervals. A positive correctness difference is supported when its paired interval excludes zero. The additional cohorts and individual forecast horizons are diagnostics. Zero observed wrong identities does not establish a small upper risk bound at this sample size.
- Reproduce the prior static atlas and legacy section benchmark separately. A supporting 12-case reference-geometry cohort runs all three filters with zero true dynamic motion. All configurations use synthetic centroid packets and ideal sampled kinematics. Patient motion, actual B-mode segmentation, contact force and hardware timing require separate validation.

## Results

| Correct completion | Static map | IID residual | OU residual |
| --- | ---: | ---: | ---: |
| Reference β | 26/32 | 26/32 | 27/32 |
| Δβ₀/Δβ₁ = (−1, +1) | 28/32 | 27/32 | 29/32 |
| Δβ₀/Δβ₁ = (+1, −1) | 24/32 | 20/32 | 24/32 |
| Primary aggregate | **78/96** | **73/96** | **80/96** |
| Δβ₂ = +0.5 | 16/16 | 15/16 | 16/16 |
| Knee 25° | 16/16 | 15/16 | 16/16 |
| Nonperiodic transient | 29/32 | 29/32 | 29/32 |
| Static world | 12/12 | 12/12 | 12/12 |

Primary paired differences in completion were +2.08 percentage points versus static (95% interval [−3.13, 9.38], 3 wins and 1 loss) and +7.29 points versus IID ([−1.04, 17.71], 8 wins and 1 loss). The IID comparison had five nonzero disturbance clusters and an exact two-sided cluster sign-flip sensitivity result of 0.25. Primary false completions were 5, 9 and 2. Eligible losses recovered in 41/45, 41/45 and 44/45 shape-case runs. All 516 executions ran for 4 s with 41 acquisitions and zero accepted wrong identities. ASTRA MAX checked the temporal model, comparison design and paired interpretation. SOL independently reproduced packets, outcomes and costs for all 21,156 exposures and replayed the joint posterior. [Paired results and forecasts](verification/DYNAMIC/ANALYSIS.json) · [Independent replay](verification/DYNAMIC/independent/README.md)

In the main cohorts, the OU controller had mean physical lateral error of 1.408 mm on acquired frames with a real crossing, compared with 1.470 mm for static and 1.564 mm for IID. It travelled 0.340 and 0.554 mm farther respectively. The paired cost intervals satisfied the prespecified equivalence margins. The run archive retains all 103,716 kinematic samples and recomputed FK, configured self-clearance and joint-rate checks. [Kinematic replay](verification/DYNAMIC/reference/KINEMATIC_REPLAY.json)

On the same packet histories, 100 ms effective offset RMSE was **1.435, 1.390 and 0.921 mm**. At origins lacking an arterial return it was 1.723, 1.625 and 1.132 mm. At 500 ms it was 1.619, 1.577 and 1.406 mm. Halving the reference lattice spacing changed RMSE by at most 0.0022 mm over the reported horizons and subsets. The transient test showed lower short-horizon prediction error and tied completion. With zero true motion, the static model predicted more accurately. The test supports short-horizon prediction of total effective displacement and a small tracking improvement. The completion advantage remains uncertain. [Numerical results](verification/DYNAMIC/ANALYSIS.json)

## Reproduction

Run from the repository root with the geometry and RM75 dependencies listed in the [cross-shape study](CROSS_BETA.md). Set `among` to the local Among_US checkout and `repair` to the frozen repair directory from that study. The commands below use the beta values saved in the archived reports. They emit the paired arms by default for moving worlds.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exp=NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113
archive="$exp/verification/DYNAMIC"
out=/tmp/leg-dynamic
scan() {
  python "$exp/run.py" atlas --among-us "$among" --repair "$repair" "$@"
}
scan --motion ou --cases 32 --out "$out/reference"
scan --motion ou --cases 32 --delta-beta -1 1 --out "$out/minus-plus"
scan --motion ou --cases 32 --delta-beta 1 -1 --out "$out/plus-minus"
read -r -a full_beta < <(python -c 'import json,lzma,sys; print(*json.load(lzma.open(sys.argv[1], "rt"))["geometry"]["beta"])' "$archive/beta2/SUMMARY.json.xz")
scan --motion ou --cases 16 --beta "${full_beta[@]}" --out "$out/beta2"
scan --motion ou --cases 16 --knee-deg 25 --out "$out/pose25"
scan --motion transient --cases 32 --out "$out/transient"
scan --motion static --cases 12 --arm mpc_stream --arm mpc_iid --arm mpc_ou --out "$out/static-world"
python "$exp/analyze_dynamic.py" --runs "$out" --out "$out/ANALYSIS.json"
for cohort in reference minus-plus plus-minus beta2 pose25 transient static-world; do
  python "$exp/verify_atlas_scan.py" --among-us "$among" --repair "$repair" \
    --runs "$out/$cohort" --out "$out/$cohort/KINEMATIC_REPLAY.json"
done
python "$exp/render_dynamic_video.py" --summary "$out/reference/SUMMARY.json" \
  --traces "$out/reference/TRACES.npz" --case 0 --arm mpc_ou \
  --skin-outline "$archive/LEG_OUTLINE.npz" --out "$out/LEG_SCAN.mp4"
```

The archive contains the simulator histories and derived kinematic data. To run the analysis and replay on the archive, unpack each `SUMMARY.json.xz` with `xz -dk` alongside its `TRACES.npz`, or into matching cohort directories in a scratch folder. The example clip uses the first disturbance ID in the reference cohort. Its optional leg outline is the projected convex hull of the transported skin chart in the first probe frame. [Video](verification/DYNAMIC/LEG_SCAN.mp4)
