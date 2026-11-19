# First-interval recovery and targeted training data

## Audit of the preceding result

Rebuild the 24 first-interval scenes used in the [arbitrary-start benchmark](ARBITRARY_START_BENCHMARK.md). Replay the same selected actions with the same eight tapes and full MPC continuation. Check the archived outcomes and travel. Record the earliest motion's realized goal, all physically imaged intervals, the final retained evidence and where identity remains unresolved. Both cold-start task descriptions require all six intervals in this group, so the audit counts each scene once. Repeat the two previously worst discordant scenes with 32 independent paired tapes for diagnosis. These scenes and their repeats stay outside training and the fresh test.

The original multi-start training set contains 224 task states and 16 first-interval cold starts. The earlier window-encoding learner was trained on 192 cold starts inside the first interval, under its earlier noise and scout selection. The two datasets differ in observation regime as well as concentration on the first interval.

## Targeted data addition

Add 96 independently sampled first-interval cases. Balance the connection and unbranched families, noise with and without a consecutive burst, both longitudinal halves of the first interval, and both sides of the nominal probe center. Sample by those public geometric conditions and prescribed disturbance regime before running the closed loop. For every admissible motion, generate six paired full execution labels with the original score against the common MPC. Retain the 224 previous task states and fit a single model on the combined 320. Use the same 457 features, calibrated probabilities, architecture, score ordering, motion budget and execution path as the preceding multi-start model. The training change is the added cases and their paired labels.

Use a new frozen test split of 128 first-interval starts, 80 later starts and 48 in-progress states. The 128 first-interval starts balance the same strata and are counted once. The two task scopes for later starts share all scenes and disturbance tapes. As in the preceding benchmark, each scene has eight paired future tapes. Test strong MPC, the original window-encoded frozen learner, the preceding multi-start model, and the single augmented model. Keep the same motion, per-frame updates, evidence ledger, and shared MPC after the chosen action. Compare actual completion, wrong identity and execution costs in each group, including the noise and burst subsets of the first interval.

## What caused the observed first-interval gap

The preceding multi-start B completed 149/192 versus 160/192 for the frozen B, with nine gains and 20 losses on paired trials. Its paired difference was −5.73 percentage points [−16.15, +2.60]. Both policies reached the endpoint on all 192 trials. Among the 20 runs completed by frozen B alone, three lost a required physical image. The other 17 acquired physical target observations in every interval. Fifteen ended with unresolved evidence in at least one interval and two retained a wrong identity. The connection family accounts for ten of the net eleven lost completions. Nine of the net eleven losses were in the noise-only condition.

Scene `62006002` starts at 5.43 mm. Frozen B selects `back_right`, which returns to 2.00 mm and changes the lateral window. Multi-start B selects `cem_3`, which advances to 9.16 mm. In an illustrative paired tape, both paths acquire three target images in the first interval and at least one target image in every required interval. After the advancing opening, four intervals remain unconfirmed at termination. The two actions complete 31/32 and 16/32 on new paired disturbances. In the other large-discordance scene, `62000000`, the original completion gap of 7/8 versus 2/8 shrinks to 21/32 versus 19/32 on new disturbances. These cases illustrate action ordering errors and finite-tape variability.

In the original first-interval cohort, multi-start B averages 36.13 mm travel versus 36.67 mm for frozen B, and 6.302 versus 6.349 s. The pooled increase in motion comes from harder later states. For regional scans started at 22 mm or beyond, the mean excursion upstream of the start grows from 0.65 to 3.21 mm. For the all-interval task at those starts, it grows from 11.80 to 16.08 mm. The selector prioritizes expected correct completion, with cost entering the existing tie score. The same absolute limits allow this trade-off in movement.

## Fresh paired results after adding data

The added cases contribute 2,016 candidate rows and 12,096 execution labels. The first-interval representation rises from 16/224 to 112/320 training states.

| Held-out group | Strong MPC | Frozen B | Multi-start B | Multi-start B plus first-interval data | Added-data effect on multi-start B, points [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| First interval | 825/1024 | 899/1024 | 899/1024 | **940/1024** | +4.00 [+0.88, +7.13] |
| Later starts, scan to end | 225/640 | 257/640 | 369/640 | 382/640 | +2.03 [−0.78, +5.47] |
| Later starts, cover all six intervals | 155/640 | 112/640 | 334/640 | 336/640 | +0.31 [−2.19, +3.29] |
| Scan already in progress | 273/384 | 286/384 | 316/384 | 316/384 | 0.00 [−5.21, +5.47] |
| First interval, noise alone | 404/512 | 442/512 | 441/512 | **475/512** | +6.64 [+2.53, +11.13] |
| First interval, noise and burst | 421/512 | 457/512 | 458/512 | 465/512 | +1.37 [−2.93, +5.47] |

The added data improve first-interval completion against both previous learners. The effect against the frozen learner is +4.00 points [+0.39, +7.71]. The noise-only subset accounts for 34 of the 41 additional first-interval completions. In that group, final wrong identities decrease from 10 to 6. With noise and burst they increase from 13 to 17. Over the entire first-interval cohort, final wrong is 23 in both models, interim wrong changes from 242 to 235, and mean travel increases by 0.71 mm [+0.06, +1.42]. Mean scan time is 6.456 versus 6.443 s.

Later groups retain their point completion rates or improve. Paired intervals allow both a modest benefit and a loss. In the in-progress group, the augmented model yields five more final wrong identities (3 to 8) and uses an additional 0.95 mm and 0.14 s on average. The validation thus supports targeted first-interval data to improve completion. It leaves the burst subset, in-progress identity risk and actual travel as criteria for testing any next training update.

The split and artifact hashes, common feasible menus, task scopes and execution limits were checked. Replays of an independent first-interval loss scene and an in-progress scene reproduce the decisions and all paired outcomes. [Audit](results/first_interval_data_v1/first_interval_audit.json) · [Fresh test summary](results/first_interval_data_v1/summary.json) · [Weights, rollouts and frozen splits](results/first_interval_data_v1/)

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=results/first_interval_data_v1
python -B benchmark_first_interval_data.py audit --out "$out" --jobs 8
python -B benchmark_first_interval_data.py prepare --out "$out" --jobs 24
python -B benchmark_first_interval_data.py train --out "$out" --jobs 24
python -B benchmark_first_interval_data.py test --out "$out" --jobs 24
```
