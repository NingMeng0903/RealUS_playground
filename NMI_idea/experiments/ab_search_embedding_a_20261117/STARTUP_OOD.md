# Recovery from a missed or partially observed startup

The robot first holds the initial contact pose for one second. The resulting 11 anonymous synthetic packets are shared between methods. The zero-vessel cohort contains no genuine detection of either vessel across that prefix. The one-vessel cohort contains detections of exactly one physical vessel across the prefix. Clutter detections may appear in both cohorts. These roles are used to form the test cohorts and evaluate performance. The planner sees anonymous observations, the Bayesian posterior, robot state, and remaining budget.

## Geometry and split

The previous benchmark had static lateral offsets of −4, 0, and +4 mm and middle-scan shadow centres of 12.4 and 15.4 mm. This experiment adds a spatial startup shadow. Its nominal centre is at a scan station of 3 mm, with longitudinal and lateral widths of 3.6 and 3.8 mm. One latent shadow mode attenuates both vessels, and the other attenuates the deeper physical vessel. Moving away laterally changes detection probability. The rule uses depth and geometry for both possible identities.

The common public filter uses lateral offsets of −9, −3, 0, +3, and +9 mm and its nominal shadow. Development scenes use truth offsets of −3, 0, and +3 mm. Holdout scenes use the ±9 mm offset modes with independent continuous lateral jitter in ±0.6 mm, depth jitter in ±0.55 mm, lateral shadow shift in ±0.7 mm and shadow widths amplified uniformly by factors between 1.25 and 1.55. The latter continuous perturbations are held constant within each scene and excluded from the planner's exact state. Holdout variation is outside the motion-demonstration distribution. The shared public filter has broad offset support for all cohorts.

Screen 160 development and 224 holdout seeds per task family. In each split, retain the earliest seeds in each zero-/one-vessel and connection/unbranched group, with quotas of 12 development and 16 holdout scenes per cell. Screen only the opening packets. Freeze the cohorts before discovery or before testing controller outcomes. The holdout has eight fresh paired disturbances per scene, beginning after the common opening.

The auxiliary entirely empty-scout test screens another 1,024 seeds per family and retains the first 16 per family with all 11 packets empty, including clutter. It uses the same frozen demonstrations, weights, and action libraries. Identical public scouts in the same task family receive the same first choice. All other test conditions stay in force.

## Ablations and fitting

On development roots, compare the original CEM and a frozen-proposal random search. Use 36 proposals per scene per source with three shared scoring rollouts per distinct admissible candidate. Both can use the simulator's development latent state for offline scoring. Retain the leading admissible candidate after at least four correct completions on six additional disturbances and zero final wrong identities. Use equally many qualified nominal trajectories to fit the two ProDMP variants, with the same displacement and waypoint features. Freeze equal-size candidate lists of up to six requests.

On each holdout root, use the same public posterior with eight shared draws to choose a nominal first request from each menu. The three menus are stock, stock plus random proposals, and stock plus CEM proposals. The selector keeps the previous time and error gates. Compare old and CEM-fit ProDMP for the same chosen request, with the same analytic fallback and online feedback. Where the request is a standard legacy skill, both run that skill. Also compare the random-trained fit against the CEM-trained fit on the very same CEM-menu requests. Count mechanically admissible realizations of all six CEM-menu requests in the held-out scout states.

The memory factor executes the first request and all subsequent sensing with either candidate retention or the acquisition-time 0.95 gate. Both modes update on each common packet and can reconsider what they stored later. The nominal public first choice is held fixed within each menu across renderers and memory modes. The shared MPC continuation uses each mode's actual observations and ledger and may replan differently. The direct MPC arm can react at the same scout checkpoint. All arms use the absolute simulated budgets of 67 mm travel and 8 s elapsed time, including the common scout. Primary completion requires reaching the end and delivering correctly identified images from all six required intervals. Centring within ±3 mm is a supplementary metric. The simulation pauses during planning.

## Results

The main held-out study contains 64 independent scenes and 512 paired disturbance trials. The auxiliary empty-scout study contains 32 separate scenes and 256 paired trials. In the main one-vessel cohort, 24 of the 32 scouts detect only the distractor. Confidence intervals resample scenes and keep all eight disturbance repeats together.

| Controller at the scout exit | Zero vessels | One vessel | Main total | Entirely empty scout |
| --- | ---: | ---: | ---: | ---: |
| Direct full MPC with retention | 254/256 | 224/256 | 478/512 | 241/256 |
| Public rollout, stock library, retention | 253/256 | 213/256 | 466/512 | 255/256 |
| Public rollout, random proposals, CEM-trained ProDMP, retention | 254/256 | 228/256 | 482/512 | 255/256 |
| Public rollout, CEM proposals, old ProDMP, retention | 254/256 | 220/256 | 474/512 | 253/256 |
| Public rollout, CEM proposals, CEM-trained ProDMP, capture-time gate | 225/256 | 150/256 | 375/512 | 256/256 |
| Public rollout, CEM proposals, CEM-trained ProDMP, retention | 255/256 | 229/256 | 484/512 | 255/256 |
| Same CEM requests, analytic trajectory, retention | 255/256 | 229/256 | 484/512 | 255/256 |

The stock row uses the old renderer and its completion counts are identical with the CEM-trained renderer. The fixed-request analytic replay was added after the factorial run to audit the renderer. It reused each saved nominal request and disturbance tape.

- **Candidate discovery.** With selector, renderer and retention fixed, adding CEM proposals increases main-cohort completion by 18/512, or 3.52 percentage points (95% paired CI 0.39 to 7.42). First-interval target reacquisition rises from 470/512 to 486/512. Random proposals receive the same 1,728 proposal attempts and six library slots. CEM exceeds that control by 2/512, or 0.39 points (95% CI −2.34 to 3.13). On completely empty scouts all three libraries complete 255/256 with public rollout, and reacquire a target image in the first interval on 256/256.
- **Trajectory learning.** CEM and random search yield 34 and 35 qualified nominal demonstrations. Both fits use 34 examples and the same displacement and interior-waypoint features. Their coefficient matrices agree to within \(1.14\times10^{-13}\) and their chosen-request completions agree on every main and blank trial. With the CEM library fixed, all ten extra main completions of the fitted version relative to the old renderer occur when the fitted renderer switches to the analytic fallback. The six scenes where the fitted version directly replaces the old fallback have no completion difference. Executing the same archived requests analytically reproduces every completion and both identity-error indicators across the main and blank cohorts.
- **Evidence retention.** In the full stack, keeping candidates for later confirmation adds 109/512 main completions, or 21.29 points (95% CI 12.70 to 30.47). Mean scan time drops from 6.742 to 5.969 s. Final wrong-identity runs change from 0 to 1 and new interim wrong-identity runs from 4 to 8. In the entirely empty-scout cohort both modes are near ceiling and the retention arm scans 0.824 s faster on average.
- **Against direct MPC.** The full stack adds 6/512 main completions, or 1.17 points (95% CI −0.78 to 4.30). Both finish with one wrong-identity run and eight runs with a new interim wrong identity. Mean simulated travel, turn and time are 37.09 mm, 6.69°, and 5.969 s for the full stack and 37.57 mm, 6.58°, and 6.094 s for direct MPC. In the entirely empty-scout study the stock and CEM public rollouts both add 14/256 completions over direct MPC. That isolates a benefit of choosing the first probe with rollout in that cohort.

This supports retaining uncertain evidence and adding a source of executable probe candidates. The incremental advantage of CEM over random search with the same proposal count and the task advantage of the learned ProDMP shape remain unestablished. The stress conditions cover fixed-pose synthetic startup shadow and geometry shifts, with broad public prior support for the held-out nominal offset modes. Each menu uses the same public rollout selector. The learned B ranker and its embedding are outside this ablation. Reported simulated scan times exclude decision time.

[Main summary](results/startup_ood_v1/holdout_summary.json) · [Empty-scout summary](results/startup_ood_v1/blank_summary.json) · [Analytic replay](results/startup_ood_v1/nominal_request_summary.json) · [Paired records and fitted weights](results/startup_ood_v1/)

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=results/startup_ood_v1
python -B verify_startup_ood.py screen --out "$out" --jobs 20
python -B verify_startup_ood.py discover --out "$out" --jobs 24
python -B verify_startup_ood.py verify --out "$out" --jobs 24
python -B verify_startup_ood.py screen_blank --out "$out" --jobs 8
python -B verify_startup_ood.py verify_blank --out "$out" --jobs 8
python -B verify_startup_ood.py verify_nominal --out "$out" --jobs 20
```
