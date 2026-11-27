# Equal-dose training of second-stage B

The three newly fitted B policies improved pooled held-out correct completion over full MPC and the development-selected fixed recheck. Under the same training dose, the geometry and evidence teachers produced completion close to the random control. The results do not establish a directed-teacher training advantage.

## Shared ProDMP and data

All arms used the original requests plus eight archived requests selected from the previous generator study. The eight relative requests were `grid_00` through `grid_04`, `grid_10`, `grid_11` and `grid_14`. The common fitted ProDMP generated their trajectories from the current state. Every first-block selection of one of these eight IDs used the fitted renderer. The online MPC suffix retained the shared expanded menu at recovery decisions.

Each source added 512 distinct-within-state nominations and the same 128 MPC anchor samples to the 8,122 historical action rows. Each resulting model used 8,762 rows and the original six-output gradient boosting trainer with the original visibility calibration. The rebuilt historical data reproduced the frozen old B's predictions exactly. The experiment compared these three actually fitted models with that frozen old model.

| Nomination source | Non-incumbent samples | Corresponding paired outcomes | Completion below 0.5 | Final misidentification above 0.5 | Intermediate misidentification above 0.5 |
|---|---:|---:|---:|---:|---:|
| Random | 512 | 4,096 | 988 | 61 | 403 |
| Geometry | 512 | 4,096 | 906 | 72 | 405 |
| Evidence | 512 | 4,096 | 900 | 65 | 414 |

The thresholds in the table summarize simulated labels with fractional uncertainty from historical witnesses. The fit used all outcomes at their original values. The shared evaluation cache executed 7,416 predictive continuations. The geometry and evidence teachers nominated the same four actions in 87 of 128 training states and shared 442 of 512 non-incumbent nominations. Their overlap with random was 263 and 257, respectively. The newly added motions formed the common training nomination pool at 125 states. At three states the pool was the full feasible menu.

## Held-out closed loops

Each task family had 64 fresh scenes with eight paired future disturbances per scene. The connection reference chose `back_center` and the continuous no-fork reference chose `grid_02` on the separate qualification set. Both used the common closed-loop suffix. There were no infeasibility fallbacks for either fixed request on holdout.

Completion required a correct scan with a newly captured, retained correct image. Identity errors are counts of trials, including those in which a new mistake was later corrected. Distance, turn and time are measured executed means in simulation.

| Policy | Task | Complete /512 | Final wrong | Intermediate wrong | Distance mm | Turn deg | Scan s |
|---|---|---:|---:|---:|---:|---:|---:|
| Full MPC | Connection | 389 | 11 | 40 | 40.14 | 7.44 | 5.90 |
| Best fixed recheck | Connection | 425 | 16 | 49 | 47.02 | 9.52 | 6.64 |
| Frozen old B | Connection | 452 | 17 | 43 | 45.23 | 8.77 | 6.31 |
| Random-data B | Connection | 460 | 16 | 43 | 45.47 | 8.69 | 6.35 |
| Geometry-data B | Connection | 459 | 15 | 38 | 45.64 | 8.80 | 6.43 |
| Evidence-data B | Connection | 458 | 14 | 40 | 44.82 | 8.53 | 6.32 |
| Full MPC | No fork | 360 | 14 | 76 | 38.66 | 7.01 | 5.85 |
| Best fixed recheck | No fork | 409 | 18 | 96 | 44.79 | 8.96 | 6.50 |
| Frozen old B | No fork | 464 | 18 | 108 | 47.19 | 9.53 | 6.38 |
| Random-data B | No fork | 467 | 19 | 106 | 46.92 | 9.48 | 6.34 |
| Geometry-data B | No fork | 467 | 20 | 105 | 46.82 | 9.39 | 6.30 |
| Evidence-data B | No fork | 471 | 17 | 103 | 47.21 | 9.42 | 6.33 |

| Pooled policy | Complete /1,024 | Final wrong | Intermediate wrong | Distance mm | Turn deg | Scan s |
|---|---:|---:|---:|---:|---:|---:|
| Full MPC | 749 | 25 | 116 | 39.40 | 7.22 | 5.88 |
| Best fixed recheck | 834 | 34 | 145 | 45.90 | 9.24 | 6.57 |
| Frozen old B | 916 | 35 | 151 | 46.21 | 9.15 | 6.34 |
| Random-data B | 927 | 35 | 149 | 46.19 | 9.09 | 6.35 |
| Geometry-data B | 926 | 35 | 143 | 46.23 | 9.09 | 6.37 |
| Evidence-data B | 929 | 31 | 143 | 46.02 | 8.98 | 6.32 |

- Relative to equal-dose random, geometry changed completion by **−0.10 percentage points** with paired 95% interval **[−1.86, +1.46]**. Evidence changed completion by **+0.20** **[−2.64, +2.54]**. Relative to frozen old B, evidence changed completion by **+1.27** **[−0.78, +3.32]**.
- Evidence B improved over full MPC by **+17.58** **[+11.42, +23.93]** percentage points. It traveled **6.61 mm** farther, used **0.45 s** more and recorded 27 more intermediate-error trials and six more final-error trials. Against best fixed in the pooled analysis it improved by **+9.28** **[+4.39, +14.45]** percentage points at similar mean travel **+0.11 mm**, and took **0.25 s** less. Its connection-only completion interval against best fixed included zero **[−0.20, +13.48]**.
- The frozen B chose a newly added grid request at 26 of 128 handoffs. Random, geometry and evidence B did so at 46, 44 and 44 handoffs, respectively. Full MPC chose one at 28. The new library was used in actual simulation execution, and the comparison keeps it shared.

These are statistically paired synthetic results across two task families, with fixed knee angle and noisy segmentation packets. The study provides evidence for learned selection within this setup and leaves the advantage of directed nomination over random augmentation unresolved. It does not isolate the contribution of the expanded action library itself. Planning CPU is separate from the simulated scan time. The four B implementations averaged 442–443 ms of first-decision CPU with calibrated features, and MPC averaged 125 ms.

## Independent check and reproduction

Sol 6.2 Max independently verified the source hashes, equal label counts, the two fixed request choices, weights for all 18 regressors and the target deltas reconstructed from raw paired outcomes. A natural missing-packet training case independently reproduced the complete label row. The holdout audit verified 128 disjoint scenes, 6,144 arm outcomes with 3,952 distinct executions, exact agreement for arms sharing an action, and all distance, time and handoff constraints. Independent replays of two holdout cases that used new grid requests reproduced all six arm decisions and their outcomes.

From this experiment directory run

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u run.py all --workers 24 --out results/replica
```

The [protocol](PROTOCOL.md) defines the experiment. [The results directory](results/teacher_b_training_v1) contains the frozen manifests, all training and qualification rows, fitted weights, holdout outcomes and scene-level bootstrap summary. The tested adapter SHA-256 is `286541828599c2382d09242507558d581c427ee1ecc991230374db09375d7989`.
