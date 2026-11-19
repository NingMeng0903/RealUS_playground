# Probe-generation pilot results

The direct geometric teacher increased the yield of promising training clips under the same count of full closed-loop evaluations. Neither directed teacher increased held-out completion over structured random nominations or over the same online selector with its stock library. The existing learned B had the highest held-out completion in this pilot. Its overall interval against the best frozen fixed recheck still included zero, and it made more intermediate identity mistakes than MPC.

## Discovery and qualification

| Nomination rule | Verified nominations | Positive nominations | Scenes with a positive nomination |
|---|---:|---:|---:|
| Structured random | 192 | 16 | 9/48 |
| Direct geometry | 192 | 28 | 8/48 |
| Precommitted follow-up | 192 | 25 | 7/48 |
| Observation-dependent follow-up | 192 | 25 | 7/48 |

A positive nomination had higher complete-scan count and no increase in terminal wrong-identity count on four paired verification tapes. The sources proposed from 1,093 feasible novel motions in total. The run reused duplicate evaluations when sources proposed the same path, for 1,776 physical simulations including the discovery MPC reference. Each method was credited with the same four nominations and their paired full replays. The discovery sift is inexpensive for the direct teacher and concentrated in a few scenes. Its selection count should not be treated as a count of independently validated skills.

The information score changed the first nomination in two of 48 scenes. One change was a numerical tie at approximately 2e-41. The other came from an eight-draw estimate of 0.0579 whose exact within-score perfect-information ceiling was 0.00717. Both proposals gave zero correct completions across the four tapes in each of these two scenes. Astra independently evaluated two 512-world banks for the substantive change and found both reversed the initial ranking. Across the discovery set, 33/48 scenes had negligible estimated information for every novel request. The precommit and evidence arms froze the same four requests in the same order.

All four-request libraries were frozen before independent qualification. Best fixed used the single relative request `grid_00` in the connection task and `wait` in the continuous parallel task. Its full MPC suffix continued to update from new packets. No fixed action needed the infeasibility fallback on holdout.

## Held-out synthetic closed loop

There were 64 independent scenes, each replayed on eight common future tapes. Correct completion included retention of new correctly identified imagery after intervention in every counted success. A transient error is a trial that made at least one new identity mistake. Costs are measured executed means.

| Controller | Completed /512 | Terminal wrong | Transient error | Distance mm | Turn deg | Scan s |
|---|---:|---:|---:|---:|---:|---:|
| MPC, union menu | 390 | 12 | 62 | 40.01 | 7.41 | 5.77 |
| MPC, stock menu | 387 | 17 | 57 | 40.55 | 7.50 | 5.79 |
| Best frozen fixed recheck | 437 | 9 | 76 | 44.49 | 8.33 | 6.37 |
| Existing learned B | 470 | 9 | 95 | 46.60 | 8.93 | 6.34 |
| Public rollout, stock menu | 448 | 12 | 70 | 41.42 | 7.84 | 5.64 |
| Public rollout, random library | 448 | 12 | 72 | 41.50 | 7.82 | 5.65 |
| Public rollout, direct library | 446 | 7 | 68 | 41.10 | 7.78 | 5.63 |
| Public rollout, precommit library | 443 | 13 | 73 | 41.13 | 7.79 | 5.64 |
| Public rollout, evidence library | 443 | 13 | 73 | 41.13 | 7.79 | 5.64 |

- Evidence library versus union-menu MPC was **+10.35 percentage points**, paired scene bootstrap 95% interval [3.91, 17.58]. Public rollout with the stock menu was **+11.33** [5.08, 18.36]. The stock selector is the relevant control for the effect of new demonstrations.
- Against that stock selector, the direct library was **−0.39** [−1.76, 0.98] and the evidence library **−0.98** [−2.34, 0.39]. Direct had five fewer terminal wrong trials and 0.31 mm lower mean travel, with completion inconclusive. Against best fixed, direct and evidence were **+1.76** [−5.08, 8.40] and **+1.17** [−5.86, 7.82]. Evidence and precommit chose the same action in every scene and had exactly matching outcomes on all tapes.
- Existing B versus union MPC was **+15.63** [7.42, 24.41]. Against best fixed it was **+6.45** [−0.20, 13.28]. It gained 33 completions in connection and tied best fixed at 243/256 in the no-fork task. It used 6.59 mm and 0.57 s more on average than union MPC and produced 33 more transient-error trials. The terminal wrong count was lower by three. Its frozen model used only the stock menu and was not trained on these new demonstrations.
- In-block scanning remained at the same simulated 10 Hz. The evidence library's public search took a mean 27.63 s CPU per decision, excluding the common menu setup, and the frozen B model took approximately 0.065 s for its feature and prediction step. The simulated clock does not run during planning. The public-search result does not establish a live timing budget.

The controller with the evidence library acquired a target-derived detection in the requested interval during the first block on 370 tapes. Existing B acquired 293 such immediate detections and finished 27 more correct scans. Selecting an early high-yield view alone is an incomplete training target.

The public rollout forecast a mean completion probability of 99.98% for the selected evidence-library actions, compared with observed completion of 86.52%. This compares the internal nominal forecast against the held-out stressed simulator. Selection over only eight sampled worlds and mismatch between the nominal sensor and the perturbed execution may both contribute.

## Independent checks and scope

The Astra 6.2 Max review checked the no-information limit, the loss of action differences at library freezing, and the finite-sample ranking instability. The Sol 6.2 Max audit checked archive hashes, balanced disjoint seeds, 4,608 arm-outcome records, exact agreement when arms realized the same action, and the physical budgets. It independently reran two held-out cases and recovered all nine arm outcomes on every tape. Maximum reported handoff error was 1.78e-15. Maximum executed distance was 66.98 mm and scan duration was 8.0 s.

Interpretation is limited to a short first block, a small existing ProDMP request family, fixed knee pose, and two simulated anatomy families with delay-free centroid observations. The result supports retaining the current policy and using direct geometry to sift potential training clips if needed. It does not support adopting either new library for higher completion. The current learned B remains a useful control, with completion, intermediate errors, and cost reported together. The experiment did not fit new ProDMP weights, compare CEM in the new action grid, or validate live ultrasound.

A subsequent mechanism experiment should first characterize repeatable image evidence from alternate feasible views at matched exits and match the simulated error model to detector outputs. A normalized finite conditional observation table can be used to verify the information estimate before policy or skill fitting. Advance an action family only where predeclared public conditions predict a change in feasible continuation and independent replay confirms that effect.

## Reproduction

From this experiment directory run

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u run.py all --workers 20
```

The exact inputs and seeded splits are in [the protocol](PROTOCOL.md) and [the staged results](results/teacher_generation_v1). The raw files include discovery clips, independent qualification, and every held-out result. The planned torso setup is in [Torso transfer](TORSO_TRANSFER.md).
