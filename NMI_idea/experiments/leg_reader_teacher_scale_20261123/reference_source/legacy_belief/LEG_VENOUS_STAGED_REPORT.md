# Do two probe decisions improve leg-vein scanning?

## Controllers and training

Every variant starts at the same measured initial-scan exit, with the same frozen vein posterior, detector interruption model, ProDMP fits and expanded MPC tail. The probe motions last one second, use live 10 Hz observations and preserve the existing position, velocity and path-length checks. The shared Cartesian travel limit is 67 mm. The MPC tail can re-evaluate its full menu throughout the run.

The candidate catalog has five probe requests. The three search requests revisit a previous useful region or move laterally to another window. The two verification requests cover a nearby longitudinal region. `short` moves forward by about 3.7 mm with posterior guidance, and `grid_02` requests a 4 mm backward movement at the current lateral position. The stage-specific design allows search requests in the entry decision and verification requests in the subsequent decision. The shared-menu design offers all five at both decisions. The single-probe controller can choose from all five, then passes the scan to MPC. Each decision has a zero-duration skip option.

From every training checkpoint, each feasible first motion is actually executed in the simulator. Each candidate continuation is executed from that measured exit with the same time-indexed perturbation seed, and followed by the common MPC. The program records complete outcomes, including false matches and failures. At each measured exit the final probe model learns the utility increment relative to immediate handoff. The first probe model learns the increment over directly calling the frozen final probe model. Its labels use the continuation chosen from each actually observed exit. This follows the backward policy-fitting order of [Policy Search by Dynamic Programming](https://proceedings.neurips.cc/paper/2003/hash/3837a451cd0abc5ce4069304c5442c87-Abstract.html).

The action regressors use only public state, posterior and observation descriptors and public candidate geometry. The target is correct completion minus 0.55 times final identity error. A probe is selected only when its predicted increment exceeds the threshold set on the independent calibration roots. The fixed comparator chooses the best complete sequence on calibration for each public task family. Direct MPC, the previously released PSDP policy, search-only and verification-only policies, and both newly trained two-decision designs are evaluated on the same trials.

On the stage-specific policy, a same-first-action control disables the second probe. A second control keeps the actual first exit pose, speed, clock, path cost and legal candidate geometries, then replaces the second scorer's observation history and posterior with those at the original decision. The executor and final MPC still receive the real observations. This tests the extra contribution of fresh evidence to the second selection under an identical physical exit.

## Data and statistical comparison

Training uses 240 independently seeded cases, 180 under joint anatomical and detector stress and 60 under the reference condition. Calibration uses 48 new roots in each condition. The held-out data use 80 further roots and two noise tapes per root per condition, for 160 paired trials per condition. Stress includes patient-specific vein curves, joining branches or projected crossings, missed observations, false contours, and merged detections. Completion requires correct retained images in all required regions and a new correct image after the common decision. Final error counts scans with at least one incorrect retained image, including incomplete attempts. Differences use a paired 5,000-draw bootstrap grouped by anatomical seed.

## Results

| Controller | Stress correct /160 | Stress final wrong /160 | Stress travel (mm) | Stress time (s) | Reference correct /160 | Reference final wrong /160 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Expanded MPC | 69 | 76 | 48.64 | 8.46 | 137 | 0 |
| Previously released PSDP | **86** | **70** | 51.80 | 7.53 | **156** | 0 |
| One probe, union menu | 75 | 72 | 48.85 | 7.45 | 149 | 0 |
| Entry-style search only | 71 | 76 | 48.13 | 7.79 | 145 | 0 |
| Verification only from the same initial scan | 67 | 78 | 48.39 | 8.08 | 141 | 0 |
| Best calibrated fixed sequence | 75 | 75 | 51.91 | 8.41 | 149 | 0 |
| Two decisions, shared menu | 61 | 83 | 49.34 | 7.84 | 153 | 0 |
| Two decisions with search and verification menus | 74 | 79 | 49.87 | 7.88 | 153 | 0 |
| Same search decision followed immediately by MPC | 68 | 79 | 47.93 | 7.64 | 151 | 0 |
| Same physical exit, stale information for verification selection | 68 | 84 | 49.94 | 7.71 | 154 | 0 |

The stage-specific sequence raises stressed completion by 8.13 percentage points over the newly fitted shared-menu sequence, with a paired 95% interval of +1.25 to +15.00. The shared-menu sequence acquires true observations across all six regions in 148 runs versus 142 for the single-probe controller, while the final error count is 83 versus 72. Correct completion favors the single-probe controller. The stage-specific sequence is within one stressed completion of either the one-probe or the calibrated fixed comparator. The previously released PSDP, with its original broader action library and prior training, remains the highest-completion controller on these holdouts.

The stage-specific entry motion is used in 119 stressed scans and both probes are executed in 64. With the same verification policy, adding the entry layer increases the total from 67 to 74, a paired change of +4.38 percentage points (95% interval −3.75 to +12.50). With exactly the same entry action and exit in each case, enabling the second decision improves 68 to 74 (19 wins and 13 losses, with a 95% interval of −2.50 to +10.63 points). It adds about 1.94 mm and 0.24 s on average and retains the same total of 79 final errors.

With the first motion and physical exit matched, access to fresh evidence changes the second request on 55 of 160 stressed scans. It chooses 95 additional probes rather than the 118 chosen with stale evidence, produces 74 instead of 68 correct completions, and reduces final errors from 84 to 79. The paired completion change is +3.75 points (95% interval −1.25 to +8.75). The reference completion counts are 153 with fresh evidence and 154 with stale evidence. The image-ablation test shares the candidate geometries and low-level feedback, so it measures the increment from fresh evidence in the second scoring rule.

In the prespecified diagnostic subgroup of 135 stressed scans with the target missing at the shared decision frame, expanded MPC, released PSDP, single-probe and the new stage-specific sequence complete 60, 74, 62 and 60 cases, respectively. The stage-specific search acts at entry in 98 of these scans and the first probe actually images the target in 58.

Exhaustive inspection of the 48 calibration stress episodes finds at least one completing path on 38 cases with one probe, 43 with the shared two-probe menu, and 40 with the stage-specific menu. These retrospective feasibility counts select successful paths using each actual simulator trajectory. In three of the five additional feasible cases, the restrictions exclude all successful two-probe sequences. The excluded orders include revisiting the region before a lateral change of window and a short backward motion before a lateral change. This illustrates a concrete effect of restricting the second menu to longitudinal verification.

The evaluation archive contains 3,200 traces over the 320 paired trials. All traces satisfy the shared travel and trajectory splice checks. The training data contain 8,546 recorded branch outcome entries, and calibration contains 3,395, including reused continuations. Policy inputs are generated from the public state and motion predictions. An additional check changed the simulator truth fields in the observed record while preserving all public inputs and obtained identical scoring features.

[Frozen parameters and calibration audit](results/leg_venous_staged/frozen.json) · [Per-case holdout](results/leg_venous_staged/holdout.jsonl.gz) · [Paired statistics](results/leg_venous_staged/summary.json)

## Reproduction

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python active_observation_benchmark/leg_venous_staged_data.py --workers 24
python active_observation_benchmark/leg_venous_staged.py --workers 22
```

The `results/leg_venous_staged` directory contains the frozen policy, seed plan, training and calibration trees, per-case holdout and paired summary.
