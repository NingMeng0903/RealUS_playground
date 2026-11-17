# Visible target recovery with material addresses

The primary endpoint is reaching the scan end with one physically acquired detection of the correct target retained in each of the six intervals. Measure the ±3 mm center criterion and absolute lateral error separately on the retained correct images. A visible candidate can be off-center. The cases use synthetic centroid packets on the fixed 15° reference leg and the original sampled chart kinematics.

## Evidence and action scoring

Record every detected candidate in its scan interval, including frames acquired with identity confidence below 0.95. Retain the original acquisition time, station, and hypothesis-conditional target association. Re-evaluate each candidate after every Bayesian update using

\[
q_{f,k,t}=\sum_z b_t(z)\Pr(k\text{ is the target}\mid z,Y_f,T_f).
\]

Each interval retains the highest-confidence candidate when its value is at least 0.95, with ties going to the earliest acquisition and slot. A later posterior can replace or revoke the selected image. Historical candidates are scored against the new belief and each packet's likelihood enters the filter once. The evaluator independently checks latent identities and the lateral error. A retained image may come from the common initial scan when identification occurs during the continuation.

Use the existing motion options and feedback in both policies. In the MPC forecast, value a usable detection according to visibility under each hypothesis. Score the capture opportunity of a bin relative to the best candidate it already contains under that hypothesis. The shared query prioritizes intervals that still lack acquisition evidence. Identity threshold crossing still determines official coverage and stopping.

The learned selector uses material addresses for the current probe and sampled points along each candidate path. The address uses the transported surface labels \(h,\sin\theta,\cos\theta\) at \(d=0\) from the existing `ScanTemplate`. Represent the lateral limits of the known FOV with their corresponding nominal skin addresses. The depth limits remain in calibrated probe millimetres before feature normalization. This is the footprint representation for the fixed reference chart, which uses the same local, linearized transverse relation as the existing pose binding. The full 24-hypothesis vessel posterior and the model's candidate visibility predictions are already available. The same calibrated observation model generates predictions and observes the realized motion in both arms.

## Paired comparison

Reuse the same ProDMP fit and six previously archived CEM requests for both policies. Strong MPC and B share the two-block initial scan, valid action set, detector, posterior and actual segment feedback every 100 ms. B can change the first recovery choice and hands off to the same MPC for the rest of the scan.

Fit the existing learned B regressors on the new completion and risk labels. Use 96 new training cases per family, with eight posterior rollouts and four simulator outcomes per action. Select between the existing posterior-only and mixed training labels on 32 further cases per family, using eight posterior and six evaluator tapes. Freeze the model, library, fit, atlas binding and label archives before the main test.

Test 96 further cases per family with four paired future disturbance tapes per case. Pair both policies on the identical initial scene and complete disturbance tape. Report correct completion, any final wrong identity, and any new wrong identity confirmation after the common scan. The latter includes contemporaneous target declarations and newly confirmed historical detections. Report travel, turn, simulated execution time, acquisition count, and first-decision CPU time separately. Use scenario-level paired intervals with the existing 5 percentage-point improvement, 2 percentage-point error and ±2 mm, ±2°, ±0.10 s, ±1 frame margins. A successful run needs real detections in the execution being scored. The simulator holds time during policy computation.

## Held-out results

| Task | MPC correct completion | B correct completion | Final wrong, MPC / B | New wrong confirmation, MPC / B |
| --- | ---: | ---: | ---: | ---: |
| Connection ambiguity | 384/384 | 384/384 | 0 / 0 | 0 / 0 |
| Unbranched | 384/384 | 384/384 | 0 / 0 | 0 / 0 |
| Pooled | 768/768 | 768/768 | 0 / 0 | 0 / 0 |

The new completion criterion saturates on the tested distribution. Both policies reached the end and retained six correctly identified detections on every test tape. The learned B selected another first action in 35 of the 192 cases. Model selection chose the posterior-only labels. The classification endpoints are for synthetic detections under matched simulation and observation models.

| Task | Travel, MPC / B | Turn, MPC / B | Simulated duration, MPC / B | Acquisitions, MPC / B |
| --- | ---: | ---: | ---: | ---: |
| Connection ambiguity | 31.36 / 31.41 mm | 5.876 / 5.876° | 4.021 / 4.021 s | 41.21 / 41.21 |
| Unbranched | 32.30 / 32.21 mm | 5.875 / 5.875° | 4.031 / 4.010 s | 41.31 / 41.10 |
| Pooled | 31.83 / 31.81 mm | 5.876 / 5.876° | 4.026 / 4.016 s | 41.26 / 41.16 |

Both cohorts and the pooled analysis passed all symmetric execution cost margins. The paired pooled travel difference was −0.020 mm [−0.120, 0.054] and the simulated time difference was −0.010 s [−0.026, 0.000]. Mean first-decision CPU time was 21.6 ms for strong MPC and 64.4 ms for B.

| Centering of correctly retained detections | Within ±3 mm, MPC / B | Mean absolute lateral error, MPC / B | Runs with all six within ±3 mm, MPC / B |
| --- | ---: | ---: | ---: |
| Connection ambiguity | 33.55% / 33.29% | 4.22 / 4.28 mm | 16 / 12 |
| Unbranched | 21.27% / 21.53% | 5.17 / 5.11 mm | 0 / 0 |
| Pooled | 27.41% / 27.41% | 4.70 / 4.69 mm | 16 / 12 |

Every unbranched case on the archived first disturbance tape had a retained image whose identity was confirmed after capture. The audit recomputed the posterior, frame identity scores, retention decisions, actual motion costs and terminal metrics. All four tapes were then replayed from the frozen decisions and common seeds. The 908 distinct trajectories reproduced all 1,536 policy outcomes, including the cases where both policies selected the same motion. The checks also verified the frozen input and model digests and the disjoint train, development and test seed sets. [Summary](results/visible_atlas_v1/test_summary.json) · [Paired outcomes](results/visible_atlas_v1/test.jsonl.gz) · [Trace audit](results/visible_atlas_v1/trace_audit.json)

## Reproduction

The 65 surface knots in [MATERIAL_SURFACE.json](MATERIAL_SURFACE.json) are derived from the same right-leg reference `ScanTemplate` used in the [cross-shape benchmark](../ab_streaming_prodmp_two_vessel_20261113/CROSS_BETA.md). They record `s_mm`, the two natural spline labels at the knots, and the projected transverse scale in metres per radian. The export includes a digest of the surface binding and of the source atlas and section table. The benchmark needs only the archived labels, section table and motion coefficients. Individual pose changes and new body shapes require their own movements and observation geometry.

Run in this experiment directory.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=results/visible_atlas_v1
python -m unittest -v test_qualified_evidence
python run_learned_recovery.py --out "$out" prepare \
  --discovery results/learned_b_cem_v1/discovery --evidence-mode visible
python run_learned_recovery.py --out "$out" collect --split train \
  --cases 96 --samples 8 --replicas 4 --offset 52000 --jobs 24
python run_learned_recovery.py --out "$out" collect --split dev \
  --cases 32 --samples 8 --replicas 6 --offset 52000 --jobs 24
python run_learned_recovery.py --out "$out" train
python run_learned_recovery.py --out "$out" test \
  --cases 96 --samples 0 --replicas 4 --offset 52000 --jobs 24
python audit_adaptive_recovery.py "$out/test.jsonl.gz"
```
