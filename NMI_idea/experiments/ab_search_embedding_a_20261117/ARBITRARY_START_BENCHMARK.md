# Starting or handing off at different positions on the leg

## Tasks and disturbances

Use the same fixed-knee atlas, vessel hypotheses, feasible action menu, ProDMP fit and feedback controller as the [window encoding experiment](WINDOW_ENCODING_BENCHMARK.md). Sample a continuous longitudinal start uniformly from the interior of each of the six task intervals on the reference leg. Sample a transverse displacement within ±6 mm of the nominal probe coordinate and apply the shared chart bounds. Freeze the reference knee angle. An arbitrary start here means a sampled position within that atlas and motion domain.

| Task | Shared prefix | Required coverage |
| --- | --- | --- |
| Start anywhere and scan to the end | One stationary second at the selected pose | One actual target image in the starting interval and in each downstream interval, plus reaching the endpoint |
| Start anywhere and cover the whole leg | The exact same cold-start pose and scout | One actual target image in all six intervals, plus reaching the endpoint |
| Handoff during scanning | One, two or three complete MPC segments from the original start, with the evidence ledger and actual motion state carried forward | All six intervals and the endpoint |

The two cold-start tasks use the same scene seeds and disturbances. The only protocol difference is the first required interval. The whole-leg task can return to upstream intervals. The regional task can also use upstream views as identity evidence. The final score for the regional task uses the required intervals. New wrong confirmations are counted throughout execution. The in-progress cohort requires a genuine target observation before the handoff or before the programmed loss, and an actual handoff position past the first interval.

Both stress conditions multiply the centroid noise standard deviations by 1.75, yielding 1.26 mm laterally and 0.91 mm in depth for a detected vessel. Retain the existing occasional packet outages, false candidates and opening shadow. Draw the persistent lateral and depth registration offsets within ±3 mm and ±1.5 mm respectively as in the preceding experiment. For the second condition add a contiguous loss of 4–7 ultrasound packets across the recovery decision, with 2–4 frames still lost after the handoff. The outage is fixed in time for each case and replayed identically across policies. Policies receive the observed packets and actual controller state. Their inputs exclude the programmed outage duration, latent identity and true registration offset.

## Paired experiment

All arms share the actual scout or tracking prefix, allowed motion candidates, observation updates, and within-segment feedback. They change the first recovery action only. The same MPC tail closes every recorded trial. The maximum execution limits remain 67 mm and 8 s. Keep the previous calibration and Bayesian observation model as the shared inference setup under the increased noise. Count final wrong identity, new wrong identity, reacquisition, endpoint attainment, physically imaged intervals, travel, turn and simulated duration. Simulated duration excludes CPU time.

Compare the full strong MPC, the previous frozen calibrated B, and that B retrained for the wider start distribution. The retrained selector uses the same features and learner, with one additional scalar for the first required interval. It predicts the same six paired closed-loop outcomes. Give each mechanically admissible candidate six shared label disturbances and optimize the predicted completion first, using the existing utility to break ties. The training cases contain both cold-start tasks and the in-progress task. Deduplicate the two identical task definitions for training starts in interval zero.

Use 96 independent cold-start scenes and 48 in-progress scenes for training. Freeze the weights, then run 144 fresh cold-start scenes and 72 in-progress scenes, with eight shared future disturbance tapes per scene. Suffix and whole-leg cold-start results are reported separately because they reuse the same cases. The cold-start task has balanced bins, families and stress conditions. The in-progress task has balanced prefix durations, families and stress conditions. Confidence intervals resample scenes.

## Conditions for a valid start

The controller can condition its local candidate motions on any supported start and use the task definition to compute coverage. Task completion also requires a reachable observation history that distinguishes the target identity. If two hypotheses have the same likelihood for every observation available along the executed path, their posterior odds stay at the initial odds. In that case a correct starting pose and a centered image can still leave the target identity ambiguous. Inspect this directly by matching the projected vessel tracks under the paired identity hypotheses and comparing unlabeled positions and visibility behind and around the simulated identifying cue. The audit is independent of action selection.

## Results

The 144 training scenes supplied 224 task states, 4,655 candidate records and 27,930 full continuation labels. Across the independent evaluation scenes, cold starts range from 2.186 to 31.114 mm. In-progress handoffs span 9.8 to 25.4 mm.

| Task and stress | Strong MPC | Frozen B | Retrained B | Retrained B minus MPC, points [paired 95% CI] |
| --- | ---: | ---: | ---: | ---: |
| Start anywhere and scan to end, noise | 243/576 | 263/576 | **357/576** | +19.79 [+11.28, +28.99] |
| Start anywhere and scan to end, noise and loss | 238/576 | 279/576 | **353/576** | +19.97 [+10.24, +30.03] |
| Start anywhere and cover all six intervals, noise | 198/576 | 177/576 | **303/576** | +18.23 [+9.72, +26.91] |
| Start anywhere and cover all six intervals, noise and loss | 187/576 | 181/576 | **302/576** | +19.97 [+11.28, +28.99] |
| In-progress, noise | 225/288 | 219/288 | **252/288** | +9.38 [+2.08, +18.06] |
| In-progress, noise and loss | 194/288 | 199/288 | **255/288** | +21.18 [+6.94, +36.11] |

In the in-progress loss condition, every arm obtains another genuine target image after the dropout in 288/288 runs. The retrained controller obtains physical target images in all required intervals in 280/288 runs, compared with 229/288 for MPC. The completed and correctly identified results are 255 and 194 respectively. Final wrong identity is 0 versus 8. Interim wrong identity is 25 versus 21.

| Task, both stress conditions | Final wrong, MPC / frozen / retrained | New interim wrong, MPC / frozen / retrained | Mean travel in mm, MPC / frozen / retrained | Mean turn in degrees, MPC / frozen / retrained | Mean time in s, MPC / frozen / retrained |
| --- | ---: | ---: | ---: | ---: | ---: |
| Start anywhere and scan to end | 47 / 46 / 67 | 116 / 148 / 187 | 18.81 / 22.73 / 26.07 | 2.28 / 2.72 / 3.46 | 6.632 / 6.678 / 6.342 |
| Start anywhere and cover all six intervals | 58 / 59 / 65 | 237 / 235 / 286 | 31.41 / 32.49 / 38.75 | 4.86 / 4.44 / 6.58 | 7.417 / 7.560 / 7.336 |
| In-progress | 16 / 22 / 9 | 40 / 65 / 56 | 36.87 / 42.35 / 45.75 | 6.61 / 7.56 / 8.71 | 5.627 / 6.281 / 6.448 |

Retraining improves completion under the shared hard caps, with more realized travel and rotation in all three tasks. In-progress mean execution also increases by 0.821 s [0.576, 1.061]. Cold starts show more new interim wrong confirmations. The final error rates and resource costs therefore qualify the completion improvement. The paired overall completion changes against MPC are +19.88 [+13.28, +26.82], +19.10 [+12.93, +25.43] and +15.28 [+6.77, +24.13] percentage points in the order above. Retraining also improves over the frozen B by +14.58, +21.44 and +15.45 points. All units depend on the fixed-knee simulation.

Cold starts exhibit a strong longitudinal dependence. Starting in the first interval, retrained B completes 149/192 versus 160/192 for the frozen B and 142/192 for MPC. For the regional task started in the last interval, retrained B completes 4/192 and the other two complete 0/192. The all-interval task from that last interval has 0/192 completions in all arms under the tested budget and action menu.

The identity audit finds matching unordered vessel centroids to within (7.2\times10^{-15}\) mm and identical visibility in the connection task beyond 22 mm. Its cue region instead has centroid differences reaching 12.65 mm between those hypotheses. In the unbranched task the late shift is at most 0.047 mm, compared with 4.20 mm around its cue and the stressed depth measurement noise of 0.91 mm per detected observation. For the regional connection task from 22 mm or later, frozen B physically captures all required intervals and reaches the endpoint in 192/192 runs, with 0 correct completions. Retrained B completes 31/192, and every correct completion includes travel back upstream of 18.4 mm. This supports the distinction between getting a view of a vessel and finding evidence sufficient to identify the requested vessel. The task's available landmarks and the resources to reach them remain prerequisites for reliable use from a new start.

The result archives were checked for split isolation, task scope, paired start and disturbance definitions, artifact hashes and physical execution limits. A late cold-start case and a separate in-progress loss case reproduce the recorded selections, predictions and outcomes. [Summary](results/arbitrary_start_v1/summary.json) · [Paired outcomes and training archives](results/arbitrary_start_v1/)

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=results/arbitrary_start_v1
python -B benchmark_arbitrary_start.py prepare --out "$out" --jobs 24
python -B benchmark_arbitrary_start.py train --out "$out" --jobs 24
python -B benchmark_arbitrary_start.py test --out "$out" --jobs 24
```
