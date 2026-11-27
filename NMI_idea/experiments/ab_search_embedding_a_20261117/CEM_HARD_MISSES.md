# CEM on missed required acquisitions

The test asks whether a CEM proposal can obtain a target image from an interval where the original MPC acquired zero target detections, and then finish the inspection. Earlier twice-width shadow runs had 20 such missed tapes, all with a successful ordinary revisit. This experiment changes both existing acoustic shadow widths to three times their reference values for every arm. It uses the same fixed-knee geometry, sensor model, required intervals, frame feedback, candidate retention, delayed identity confirmation, and absolute motion budget.

## Selection and endpoints

Screen 192 development and 256 holdout seeds per family, using four independent screening tapes per seed. Select the first 24 development and 32 holdout seeds per family with at least one required interval containing zero raw target detections under direct stock MPC. Record all screening outcomes. Freeze this selection before evaluating CEM candidates. Run eight additional disturbance tapes per selected scene. Report the entire screened hard cohort and, within it, the paired tapes on which direct MPC still genuinely missed an interval. The screening labels and evaluation identities are available to the simulator and offline analysis.

The primary outcomes are acquiring a target detection in every interval missed by the direct MPC on that tape, and finishing the required scan with the correct identities. Score incorrect confirmations, TCP travel, accumulated turn, and execution duration. Report scene-paired 95% bootstrap intervals. An affirmative CEM completion result against a comparator requires at least five percentage points of paired gain and a positive lower bound. Report added time and mistakes separately. The +2 mm and +2° relative travel and turn gates were omitted. The shared planner retains the previous +0.10 s and two percentage-point error forecast margins and the simulator's 67 mm and 8 s absolute limits. Fixed revisit is evaluated with the shared absolute limits. Durations use the simulation clock.

## Proposal sources and controls

Run the existing single-segment CEM discovery search on development cases with known simulated latent geometry. Three initial distributions, two iterations, and six draws give 36 proposals per scene for both algorithms. Feasible distinct candidates receive three shared rollout tapes. The random arm keeps the initial distributions fixed. Stock, random and CEM use the same rendering, evidence retention, objective and ordinary MPC continuation. The privileged baseline selects among all feasible stock first actions under the same forecasts. Each algorithm's added proposal can replace that choice if it has higher predicted utility under the common forecast constraints. Evaluate the choices on independent development tapes and nominate equal-size libraries of up to six distinct relative requests using development outcomes.

For the held-out test, compare direct stock MPC, an unconditional ordinary revisit followed by MPC, and the same public posterior selector with stock, stock plus random requests, or stock plus CEM requests. Give the time- and error-matched selector 12 shared posterior draws per action. The public predictions and action choices use live simulated observations, the public state and the model posterior. Candidate lists are frozen from development. Use the same original ProDMP and analytic candidate renderer in all arms. This comparison estimates the effect of the source of first-action proposals. The CEM-fitted ProDMP and the learning B scorer have their own existing component comparisons.

## Results

Screening found genuine acquisition misses in 110/192 development and 143/256 holdout connection scenes, and in 30/192 development and 34/256 holdout unbranched scenes. The frozen development set has 48 scenes and the frozen holdout has 64 scenes. On the eight fresh holdout tapes per selected seed, the direct MPC missed at least one raw target acquisition in 511/512 tapes. The following table uses those 511 misses.

| First-action policy | Original gaps all acquired | Correct completion | TCP mm / turn ° / scan s | New interim / final wrong identity |
| --- | ---: | ---: | ---: | ---: |
| Direct MPC | 0/511 | 0/511 | 34.37 / 5.887 / 8.000 | 0 / 0 |
| Fixed ordinary revisit | 465/511 | 437/511 | 55.29 / 10.274 / 7.172 | 5 / 0 |
| Public selector, stock menu | 502/511 | 491/511 | 54.56 / 10.623 / 6.988 | 4 / 0 |
| Public selector, random additions | 505/511 | 500/511 | 50.82 / 9.746 / 6.496 | 7 / 0 |
| Public selector, CEM additions | 504/511 | 498/511 | 50.75 / 9.795 / 6.467 | 7 / 0 |

With CEM additions, the connection cases complete 252/256 and the unbranched cases complete 246/255. Random additions complete 254/256 and 246/255 in those groups. Across the full frozen hard cohort the correct-completion counts are 1, 438, 492, 501 and 499 out of 512 in the order above. An extra request was used in 54/64 CEM cases and 56/64 random cases. Their menus each contained six extra requests, with three requests common to both menus.

| Paired effect of CEM additions | Correct completion change, pp [95% CI] | Original gaps acquired change, pp [95% CI] | Scan duration change, s [95% CI] |
| --- | ---: | ---: | ---: |
| Versus fixed revisit | +12.00 [7.03, 17.38] | +7.65 [3.54, 12.39] | −0.705 [−0.852, −0.559] |
| Versus public selector on stock | +1.37 [−0.20, 3.13] | +0.39 [−0.59, 1.56] | −0.521 [−0.641, −0.408] |
| Versus public selector on random additions | −0.39 [−1.37, 0.39] | −0.20 [−0.98, 0.39] | −0.029 [−0.090, 0.027] |

Compared with the stock selector, CEM won 12 paired completions and lost five. It reduced the mean executed travel by 3.81 mm [3.18, 4.44] and produced three more interim error tapes. The estimated error increase was +0.59 points [−0.20, 1.37]. Compared with random additions, it won two completions and lost four. This scene-paired gain falls short of the five-point success criterion against both public selectors.

Development search also gave similar recovery from both proposal sources. From 1,728 proposals per source, CEM scored 1,588 feasible distinct paths and random scored 1,526. On the 376 genuine misses in the independent development tapes, the privileged stock choice recaptured all missing intervals in 354 cases and completed 350. Random additions recaptured 359 and completed 354. CEM additions recaptured 359 and completed 353. These privileged outcomes use simulator knowledge in scoring and help explain the performance on held-out public information.

These paired experiments support active reacquisition and choosing a suitable revisit. At this proposal budget, added candidate diversity improves measured scan time relative to the stock menu. CEM and random search have similar recovery and completion on the tested distribution. The simulated pressure setting uses a threefold width of the existing acoustic shadow model. The estimates apply to the screened synthetic scenes and the common single-segment intervention controller.

[Screening](results/cem_hard_misses_v1/screening_summary.json) · [Development search](results/cem_hard_misses_v1/development_search_summary.json) · [Holdout summary](results/cem_hard_misses_v1/holdout_summary.json) · [Paired holdout records](results/cem_hard_misses_v1/holdout_cases.jsonl.gz)

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=results/cem_hard_misses_v1
python -B verify_cem_on_misses.py screen --out "$out" --jobs 16
python -B verify_cem_on_misses.py discover --out "$out" --jobs 16
python -B verify_cem_on_misses.py verify --out "$out" --jobs 16
```
