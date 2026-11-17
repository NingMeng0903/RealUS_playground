# Closed-loop result for the learned B and CEM-fitted ProDMP

The full learned policy completed **317/768 scans, compared with 292/768 for strong MPC**. The paired gain of **3.26 percentage points** had a scene-bootstrap 95% interval of **[0.39, 6.38]**. It fell below the five-point minimum effect in the [protocol](LEARNED_B_PROTOCOL.md). All 25 net additional completions were in the local connection task. The unbranched task agreed with the MPC's completion decision on every replay.

| Controller | Local connection | Unbranched | Pooled | Pooled difference against MPC, pp [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| Strong MPC with the shared library | 178/384 | 114/384 | 292/768 | — |
| Learned B with CEM labels and fitted ProDMP | 203/384 | 114/384 | 317/768 | +3.26 [0.39, 6.38] |
| Learned B without CEM candidate labels | 206/384 | 114/384 | 320/768 | +3.65 [0.91, 6.51] |
| Public online rollout with fitted ProDMP | 233/384 | 132/384 | 365/768 | +9.51 [5.99, 13.28] |

The complete learned B won 44 paired replays and lost 19. It improved local connection completion by 6.51 points [0.78, 12.76] and fresh query confirmation there by 10.68 points [3.91, 17.71]. It changed the first action in 115 of 192 scenarios. It used a CEM request in 46 scenarios and the fitted renderer in 80 scenarios, compared with 29 and 63 for direct MPC. Both arms can use the legacy ProDMP skills and common admissible fallback. The first action executes with feedback every 100 ms and hands back to the same full MPC, including its normal motion renderer.

| Pooled outcome | Strong MPC | Complete learned B | Learned B minus MPC [95% CI] |
| --- | ---: | ---: | ---: |
| Final wrong identity | 1/768 | 0/768 | −0.13 pp [−0.39, 0.00] |
| New centered mistake | 1/768 | 0/768 | −0.13 pp [−0.39, 0.00] |
| Executed chart travel | 39.81 mm | 38.85 mm | −0.96 mm [−1.55, −0.37] |
| Executed turn | 7.39° | 7.18° | −0.21° [−0.34, −0.09] |
| Simulated scan time | 6.767 s | 6.581 s | −0.186 s [−0.268, −0.109] |
| First decision CPU time | 36.9 ms | 75.3 ms | +38.5 ms [36.3, 40.6] |
| Scan time plus first decision CPU time | 6.804 s | 6.656 s | −0.148 s [−0.230, −0.070] |

The upper-margin execution cost checks passed. Symmetric equivalence failed for time and acquisition count because the reductions exceeded the lower bounds. Zero of 192 paired scenarios had any added error under the complete learned method, giving a 1.90% one-sided 97.5% upper bound for this scenario-level outcome. The online rollout averaged 29.1 CPU seconds for the first decision. Its simulated scan time excludes computation, and the simulation does not evolve the plant during planning.

## Contribution of CEM labels and training

CEM discovery used 96 scenarios from the two task families. It produced 29 verified trajectories for the new ProDMP fit and six relative requests for the shared candidate library. The separate training set had 288 scenarios and 5,905 feasible action rows, including 1,642 CEM candidate rows. It recorded 59,050 posterior rollouts and 23,620 evaluator rollouts, preserving failures and execution cost. In the 80 development scenarios, the complete learned model selected the 65/35 mixture of posterior and evaluator labels. The label ablation selected posterior labels. Both were frozen before the 192 held-out scenarios.

The learned model with CEM candidate labels gave **−0.39 percentage points** against the learned model trained without those labels [−2.86, +2.08]. These policies share the ProDMP renderer and request library at test time. The comparison measures the effect of including those labeled examples under the same development-selection procedure, with different training table sizes. It leaves the total effect of supplying the CEM library and fitted renderer to both policies unmeasured.

The online and learned models chose the same action on 57 of 192 scenarios. With the same options and feedback, the online teacher retained a substantial completion advantage over the student. In the unbranched cases it won 18 replays over MPC, while the student reproduced every MPC completion outcome. This establishes selection headroom in that simulated task family and identifies a limit of the trained scorer.

## Records and verification

The [frozen-model manifest](results/learned_b_cem_v1/frozen_models.json) records the sample split and hashes. [Full metrics and paired intervals](results/learned_b_cem_v1/test_summary.json) and [individual replays](results/learned_b_cem_v1/test.jsonl.gz) preserve the observed evidence, selected actions, executed motion and evaluator labels. The [trace audit](results/learned_b_cem_v1/trace_audit.json) checks the recorded kinematics, costs, associations, and completion decisions for 768 held-out traces. Model and data archives are in the same directory. Reproduction used Python 3.10, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2 and joblib 1.5.2.

The result is for fixed-knee chart simulation with synthetic centroid packets and a matched observation model. It measures complete closed-loop simulated scans under common feedback and motion checks.

The [per-case teacher audit](TEACHER_CHOICE_AUDIT.md) and the [verified-teacher training comparison](VERIFIED_TEACHER_RESULTS.md) test which of these online gains can be transferred by direct supervision on independent paired outcomes.
