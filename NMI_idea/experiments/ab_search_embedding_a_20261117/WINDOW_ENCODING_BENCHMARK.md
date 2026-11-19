# Relative ultrasound window features for startup recovery

## Paired comparison

Keep the reference leg at the original fixed knee angle. Draw fresh startup scenes from the existing connection-ambiguity and unbranched worlds. Place the stationary scout within the first required interval, 0–2 mm downstream of the original start, at a known transverse offset of 1–6 mm on either side of the nominal scout location. Add one persistent relative registration offset per simulated scene, uniformly distributed within ±3 mm laterally and ±1.5 mm in depth. Keep the existing anatomy, shadow and frame-noise distributions. All policies receive the same measured starting pose, eleven scout frames, posterior and shared feasible actions. Balance family, scout side and starting observation across the splits. Each scout sees either no genuine vessel or only the competing vessel. Use distinct seeds for motion fitting, the new calibration, training, development and test scenes.

Train the same B regressors on the same first-action continuation labels using three feature sets.

| Learner | Candidate features |
| --- | --- |
| Current | Existing coarse surface address and window features, vessel posterior, nominal observability and execution features |
| Margin | Current features plus signed target offsets and margins relative to the candidate window |
| Calibrated | Margin features plus target-detection probabilities marginalized over residual registration uncertainty |

The extra features use the same five first-segment acquisition instants as the existing observability features. At each instant take the posterior-weighted signed lateral displacement and the signed distance to the closest window edge along each of the lateral and depth axes. A negative edge margin indicates a predicted projection beyond that edge. Preserve the full discrete vessel belief in all three learner inputs. Append the extra features and their differences from the incumbent's features. The calibrated learner also receives the sum of its marginal probabilities for sampled views in the first pending interval.

## Calibration and freeze

Use 96 independent reference fixtures with known target correspondence to estimate the lateral and depth residual means and variances from detected centroids at a well-observed station. Subtract the known centroid-averaging variance from the cross-fixture variance. Integrate the resulting diagonal Gaussian residual using three-point quadrature per axis. For each sampled view, evaluate the existing acoustic model at each offset, include centroid truncation and packet outage, and average over both the residual samples and current discrete vessel posterior. A separate 64-scene calibration split chooses one shared opening-shadow width multiplier by frame-level Brier score. Inference uses the frozen calibration and public posterior. The simulated registration offset, identity and current draw stay in the evaluator.

Generate eight paired full continuation traces per feasible candidate on each of 192 training scenes and 64 development scenes. Fit every learner on the same training labels. Rank actions by predicted correct completion, with the same original utility for ties and no relative forecast gates. Freeze the models, calibration, shared motion library, and the fixed opening selected per family on development. Include the original fixed `cem_4` opening and strong MPC at the same initial decision. If a fixed request has no admissible realization, it falls back to the common MPC.

## Held-out evaluation

Use 128 new scenes with eight independent paired disturbance tapes per scene. From the chosen first motion, execute the same ProDMP or shared analytic fallback, within-segment ultrasound feedback, candidate retention, delayed identity confirmation, and full updating and replanning MPC tail. Use the same 67 mm and 8 s hard execution limits. Correct completion requires a correct acquired target image for each of the six scan intervals and reaching the endpoint. Record both final and new interim wrong identities, first-interval reacquisition, travel, rotation, scan time, acquisitions and decision CPU time. Compute paired intervals by resampling scenes. Simulated scan time excludes decision computation.

Audit nominal and calibrated target-detection probabilities on a separate disturbance tape per held-out scene at the nominal candidate view poses. The audit measures the effect on probability prediction independently of end-to-end policy outcomes. The benchmark concerns fixed-knee simulation with synthetic observations and a simple common calibration model.

## Results

The shared training set contains 4,029 candidate records and 32,232 full continuation runs. The development split contains 1,344 candidate records and 10,752 runs. The fitted input widths are 414, 444 and 456. On development, the fixed actions are `back_right` for connection ambiguity and `cem_2` for unbranched tracking. They complete 473/512, while a per-scene action selected with hindsight over the tested disturbances completes 494/512. The hindsight comparison quantifies action-level headroom on development.

The reference fixtures estimate residual standard deviations of 1.614 mm laterally and 0.903 mm in depth. The independent visibility calibration selects a shadow width multiplier of 1.6 at the upper end of the frozen grid. It is a shared predictive parameter for this simulator. In the held-out audit of 13,435 planned views, target detection occurs in 21.31% of sampled observations. Mean prediction changes from 39.04% to 21.76%, and Brier score improves from 0.17266 to 0.12493. The equal-scene-weight paired Brier change is −0.04777 with a 95% interval of [−0.06063, −0.03508]. The sampled views share scenes and disturbance tapes.

| First selector | Connection ambiguity | Unbranched | Correct completion | Final wrong | New interim wrong | Travel | Turn | Scan time | Decision CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Strong MPC | 420/512 | 363/512 | 783/1024 | 31 | 121 | 37.59 mm | 6.56° | 6.368 s | 151 ms |
| Fixed `cem_4` | 438/512 | 465/512 | 903/1024 | 38 | 102 | 35.56 mm | 5.70° | 6.251 s | — |
| Fixed per family from development | 475/512 | 475/512 | 950/1024 | 30 | 114 | 37.02 mm | 5.97° | 6.340 s | — |
| Current B encoding | 460/512 | 483/512 | 943/1024 | 37 | 122 | 37.21 mm | 6.14° | 6.294 s | 212 ms |
| B with margins | 455/512 | 480/512 | 935/1024 | 39 | 126 | 37.41 mm | 6.13° | 6.330 s | 214 ms |
| B with margins and calibrated visibility | 461/512 | 489/512 | 950/1024 | 35 | 116 | 37.63 mm | 6.15° | 6.312 s | 383 ms |

Margins alone change completion by −0.78 percentage points relative to the current encoding (paired 95% interval [−2.73, +0.88]). Adding calibrated visibility changes completion by +1.46 points relative to margins alone [−0.49, +3.81] and +0.68 points relative to the current encoding [−0.59, +2.15]. Against the current encoding the calibrated learner changes the first choice in 64 of 128 scenes, with 22 additional completions and 15 lost completions across paired runs. The calibrated learner captures the first interval in 1013/1024 runs, compared with 1019/1024 for the current encoding.

The calibrated learner improves on direct MPC by 16.31 points [10.94, 22.17], with +0.04 mm mean travel and −0.057 s mean simulated time. Its final wrong-identity difference against direct MPC is +0.39 points [−2.05, +3.13]. It ties the development-selected fixed strategy in overall completion, with 28 wins and 28 losses, 5 more final wrong identities, and 0.61 mm more mean travel. Completion gains over the current B remain uncertain. The calibration substantially improves frame-level detection forecasts and adds about 171 ms to mean decision CPU time in this implementation. This ablation scores the combined registration marginalization and shadow-width calibration as one input change.

The audit checks split disjointness, frozen input and output hashes, and hard execution limits for all archived runs. Two sampled scenes replay with identical decisions, predictions, and paired outcomes. [Summary](results/window_encoding_v1/summary.json) · [Frozen calibration](results/window_encoding_v1/calibration.json) · [Full trace and label archives](results/window_encoding_v1/)

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=results/window_encoding_v1
python -B benchmark_window_encoding.py prepare --out "$out" --jobs 24
python -B benchmark_window_encoding.py collect --out "$out" --jobs 24
python -B benchmark_window_encoding.py train --out "$out"
python -B benchmark_window_encoding.py test --out "$out" --jobs 24
```
