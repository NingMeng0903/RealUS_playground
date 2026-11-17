# Training data for identity recovery after loss and revisit

## Question

The [earlier search comparison](SEARCH_EFFICIENCY.md) found the same robust first-round trajectories for structured random proposals and CEM, starting from stationary scouts with no target detection. The [first-interval augmentation](FIRST_INTERVAL_DATA_BENCHMARK.md) improved completion mostly outside the forced-loss subset. Its burst checkpoints were inside the loss itself. Here the decision starts after observations have returned. We test whether the existing menu contains actions with different identity and completion outcomes at those checkpoints, and whether adding these situations to B's training improves its decisions.

## Case generation

Use the fixed-knee atlas simulator, two vessel families, registration perturbations and 1.75× centroid noise of the [arbitrary-start benchmark](ARBITRARY_START_BENCHMARK.md). A real common MPC prefix builds each pose, velocity, posterior and image ledger.

- **After loss.** Advance the prefix for 2, 3 or 4 blocks. Inject 4–7 consecutive lost packets, then permit observations for 1–3 frame positions before and including the decision. Require a public target association of at least 0.95 before the gap and a nonempty packet at the decision after the gap.
- **Revisit.** Advance the prefix for 1, 2 or 3 blocks. Run one common feasible motion from the existing fitted menu into an upstream interval that contains an earlier image. Prefer an uncovered interval when one is reachable. Use MPC to select this motion within the return subset. Require a packet at the new decision in the revisited interval.
- **Identity ambiguity.** A current candidate has a maximum target association from 0.20 up to the existing confirmation threshold of 0.95, or resumed observations have led to revocation of a previously active certificate below that threshold. For revisit, the revocation has to concern the revisited interval. Clear controls have a confirmed current candidate and no such conflict. The state selection uses these public tests. True detections and identities are recorded separately by the simulator evaluator.

Freeze independent seeds by split. Balance vessel family, scenario and prefix length in the 12 cells. Select 6 states per cell for each of the two augmentation sets. The targeted set requires identity ambiguity. The unconditioned set accepts each eligible prefix regardless of the ambiguity classification. Use another 8 ambiguous and 4 clear states per cell for the test. Add 64 fresh first-interval and 40 fresh later cold starts as regression checks. Later starts test both a partial scan and an all-interval scan on the same underlying scenes.

## Labels and evaluation

For every feasible menu action, run one intervention and the original updating and replanning MPC tail on six paired disturbances. Use the same simulated world, prefix, held-fixed observation history and disturbance for the incumbent. Save all six outcome vectors per action, including failures. Fit on their mean deltas for correct completion, wrong final identity, any new wrong association or certification after intervention, TCP travel, turn and simulated seconds.

To check action differences independently, nominate the actions with the highest labelled completion and highest labelled post-checkpoint identity error. Replay these and the incumbent on 16 distinct common tapes, kept outside the fit. For a descriptive action contrast, the best nominee must complete at least two more trials than the risky nominee with at least two fewer new identity-error trials and no more final errors. Improvement opportunities against MPC use at least two more completed trials and no more final errors. The nominees use retrospective simulation labels. This audit measures candidate differences under the frozen menu and budgets.

Fit the same B model twice, retaining all 320 preceding states and adding either the targeted or the unconditioned labels. Keep 457 inputs, calibration, motion representations, streaming feedback, evidence ledger, selection ordering and the MPC tail fixed. Use the common 67 mm / 8 s simulator limits. The main evaluation compares strong MPC, previous B and the two supplemental models on the independent ambiguous test, with eight paired disturbances per scene. Report completion, new and final wrong identity, target-image recovery, decision CPU time and realized execution costs, plus guard cohorts. Compute paired intervals by resampling scenes. CPU time does not advance the simulation clock.

## What the frozen action menu can do

The targeted set contributes 1,410 feasible candidate rows and 8,460 closed-loop training labels. The equal-count unconditioned set contributes 1,451 rows and 8,706 labels. It contains seven ambiguous, 45 clear and 20 weak states under the same public classification. The targeted set has 36 states per context, with seven contradicted certificates after loss. Every revisit state in the targeted set qualifies by the current association score. Sixty-five of the 72 targeted states show at least two candidates at the checkpoint. The evaluator observes the true target in 69 of these 72 current packets.

| Training checkpoint cohort | MPC completes on audit tapes | Best labelled action completes | Improvement opportunities vs MPC | Completion and identity contrast against risky action |
| --- | ---: | ---: | ---: | ---: |
| Targeted after loss | 214/576 | 365/576 | 11/36 scenes | 3/36 scenes |
| Targeted revisit | 501/576 | 510/576 | 2/36 scenes | 0/36 scenes |
| Unconditioned after loss | 347/576 | 410/576 | 4/36 scenes | 1/36 scenes |
| Unconditioned revisit | 466/576 | 485/576 | 1/36 scenes | 2/36 scenes |

After loss in the targeted set, switching retrospectively to the best labelled action changes final wrong from 19 to 18, new interim wrong from 47 to 75, and mean travel from 37.69 to 40.83 mm. Candidate selection for this audit uses known simulated outcomes on six tapes and replays the selected action on new tapes of the same scene. It estimates opportunities in the menu. A deployable selector must predict the useful action from public observations. The strict contrast count compares the top and risky nominees and is a descriptive count under the stated repeat criterion.

In training scene `65000007`, the loss ends before the checkpoint at 17.60 mm. The two current candidate associations are 0.612 and 0.207. Continuing forward, returning toward the right window, and returning through the center all acquire a true target somewhere after the checkpoint on all 16 tapes. They obtain a fresh correct confirmation in the required query interval on 0/16, 16/16 and 13/16 and end with 2, 0 and 1 wrong identities, respectively. The right and center returns have 6 and 8 new interim errors. The paired trajectories show that both the return and the window affect whether the system acquires and uses the required evidence.

## Fresh paired results

| Held-out cohort | Strong MPC | Previous B | B with unconditioned data | B with targeted data | Targeted data vs previous B, points [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identity ambiguity, pooled | 461/768 | 533/768 | **558/768** | 549/768 | +2.08 [−0.52, +5.34] |
| After loss | 168/384 | 243/384 | 246/384 | **249/384** | +1.56 [−0.78, +5.47] |
| Revisit | 293/384 | 290/384 | **312/384** | 300/384 | +2.60 [−1.30, +7.81] |
| Clear identity at the checkpoint | 312/384 | 304/384 | 295/384 | **318/384** | +3.65 [+0.52, +7.81] |
| First-interval cold starts | 399/512 | 469/512 | 472/512 | **483/512** | +2.73 [−0.39, +6.64] |
| Later cold starts, scan to end | 117/320 | **199/320** | **199/320** | 178/320 | −6.56 [−13.12, −1.25] |
| Later cold starts, cover all intervals | 90/320 | **168/320** | 167/320 | 160/320 | −2.50 [−6.88, 0.00] |

Across the 96 ambiguous scenes, unconditioned augmentation improves completion by 3.26 points [+0.13, +7.03] over previous B. The targeted effect against the unconditioned supplement is −1.17 [−3.91, +0.78] points. The targeted model improves over strong MPC by 11.46 [+4.04, +19.14] points in the pooled cohort. By family the targeted counts are 250/384 versus MPC's 177/384 for connection and 299/384 versus 284/384 for unbranched anatomy. The corresponding intervals are [+8.85, +30.21] and [−6.25, +14.58] points.

| Ambiguous cohort, 768 paired runs per controller | Final wrong | Any new interim wrong | Mean travel | Mean turn | Mean simulated duration | Mean decision CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Strong MPC | 19 | 61 | 41.36 mm | 7.77° | 6.81 s | 85 ms |
| Previous B | 24 | 114 | 50.79 mm | 9.77° | 7.26 s | 291 ms |
| B with unconditioned data | 26 | 99 | 48.73 mm | 9.30° | 7.12 s | 290 ms |
| B with targeted data | 24 | 104 | 50.52 mm | 9.61° | 7.23 s | 290 ms |

All four controllers reacquire a physical target image after the checkpoint on every ambiguous trial. Fresh correct confirmation in the query interval increases from 481/768 for MPC to 604/768 for previous B and 607/768 for either supplemented model. In the later partial-scan guard, the targeted model loses 23 completions and gains two compared with previous B. Every one of the 23 losses reaches the endpoint and physically images every required interval. Nineteen finish with unresolved certificates and four retain a wrong identity. This guard isolates a recognition and evidence-use regression despite acquisition of the target.

## How to select the next cases

On the 36 targeted loss training checkpoints, the best labelled choices complete 365/576 in the independent within-scene replay. Previous B already completes 364/576. Only two scenes have a nominated action with at least two more completions and no more final wrong identities than previous B. The same criterion yields 11 scenes when the reference is MPC. Screening on observable uncertainty concentrates the MPC's weaknesses. Measuring the incumbent learner on the same tapes identifies which cases still offer useful improvement for training.

Prioritize real loss-and-return histories with stable differences in full continuation versus previous B. Preserve the actions with identity errors and the outcomes with no useful intervention, and maintain coverage of later partial tasks. Ordinary revisit coverage provides a useful equal-count control. Its advantage in this test warrants keeping it when collecting new examples. The tested targeted supplement has a later-start regression and no established pooled advantage over equal-count unconditioned augmentation. The comparisons use the existing absolute budgets and show the realized execution costs. This is a fixed-knee simulation with two anatomical families.

Split and artifact checksums, the forced-loss and revisit histories, feature dimension, physical limits and concordance for shared actions were checked. Five independently reconstructed test cases reproduce the selected actions and paired outcomes. [Label and audit summary](results/identity_recovery_data_v1/label_summary.json) · [Previous B replay](results/identity_recovery_data_v1/previous_b_label_audit.json) · [Held-out results](results/identity_recovery_data_v1/summary.json) · [Frozen weights and raw data](results/identity_recovery_data_v1/)

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=results/identity_recovery_data_v1
python -B benchmark_identity_recovery_data.py prepare --out "$out" --jobs 24
python -B benchmark_identity_recovery_data.py collect --out "$out" --jobs 24
python -B benchmark_identity_recovery_data.py train --out "$out" --jobs 24
python -B benchmark_identity_recovery_data.py test --out "$out" --jobs 24
python -B benchmark_identity_recovery_data.py audit_previous --out "$out" --jobs 24
```
