# Predicting the value of executable continuations

This follow-up uses the shared [full-episode benchmark](../ab_full_b_rollout_20261113/README.md) and its checked motion bank. It first measures which continuations can improve complete scan outcomes when suffix disturbances are resampled. It then trains B to predict the paired correct-completion difference for each legal continuation. The primary contextual B makes active decisions in the new cohort. It does not demonstrate the prespecified matched-cost improvement over replanning MPC.

[Protocol](PROTOCOL.md) · [Freeze](FREEZE.json) · [Independent reviews](INDEPENDENT_REVIEW.md) · [Archive hashes](DATA_MANIFEST.json)

## Closed-loop comparison

Both policies start with the same synthetic anchor image, latent case and physical-clock/channel disturbance tape. The two prespecified decision points are after the anchor and after two shared MPC commands. B selects from every first command with a checked full-route witness. The expanded library also permits a checked second command if the first leaves public scan progress unchanged. Once its commands are exhausted, or the first command advances progress in a two-command continuation, control returns to the exact same MPC. This MPC updates its posterior and joint probability mass for prior claims and replans on each actual packet. Witness paths show mechanical reachability under ideal acquisitions. Success requires fresh correct identities at both stations and actual completion of the checked scan. A false claim anywhere in the episode counts as wrong. Private return identities enter the evaluator.

The shared budgets are 65 mm TCP distance, 90° cumulative rotation, 250 mm rail and six acquisitions. Measured differences are separately tested for equivalence at ±2 mm, ±2° and ±0.10 s of sampled motion. The simulated edges all take 1.5 s. The per-case cost screen on twelve ranking tapes therefore requires equal total acquisitions on those tapes to meet the time margin. The cohort-average analysis allows shorter and longer episodes to offset.

## Opportunity under independent disturbances

Each library was tested on 96 independent cases with 24 suffix tapes. The label-assisted selector ranks candidate means on half the tapes and evaluates those choices on the other half, then swaps the halves. Figures below are changes in correct completion in percentage points. The cohort solver uses the ranking halves to allocate one continuation per case within the three mean cost margins. Its intervals are descriptive across cases conditional on the two fitted allocations.

| Task and common prefix | Single-command, per-case matched | Expanded, per-case matched | Expanded, cohort cost allocation | Expanded cohort motion equivalent |
| --- | ---: | ---: | ---: | --- |
| Daughter, anchor | 0.00 | −0.04 | +2.82 | No |
| Daughter, two commands | 0.00 | −0.04 | +0.95 | No |
| Single trunk, anchor | 0.00 | +0.09 | +2.47 | No |
| Single trunk, two commands | 0.00 | 0.00 | −0.26 | Yes |

The per-case screen columns pass realized all-case equivalence. The expanded library gives a same-tape hindsight ceiling of +28.13 points for daughter at the anchor. Its repeat-validated cost-screened estimate is −0.04 points. The ceiling uses future outcomes, while the split-repeat procedure can still be affected by finite ranking noise. The diagnostics establish the performance of these selectors and the tested library. They do not identify a population oracle. The study stopped expanding the candidate library after the two-command test.

[Single-command opportunity and per-route census](data/oracle/opportunity.json) · [Expanded opportunity and per-route census](data/extended_oracle/opportunity.json)

## Direct completion-difference predictor

The supervised response is the paired average of \(Y_c-Y_{\mathrm{MPC}}\) through task termination, with separate responses for wrong identity and costs. A shared ridge model uses compressed public posterior statistics, task progress, observed packets, remaining resources and checked candidate geometry. The 192 training cases and 96 development cases are disjoint from both opportunity cohorts and the prospective test. Direct MPC has predicted difference zero and wins ties. The primary selector maximizes predicted correct completion minus fixed signed motion prices. Prices were fit using the public development forecasts and half of the inherited cost margins as the dual fitting target. The deterministic argmax may differ from a fractional optimum. Its fitted cost means are recorded with the prices. No adjustable success or uncertainty threshold is used. The per-state cost screen, hard-cap-only score and train-selected fixed continuation are diagnostic controls.

In development, the contextual predictor's mean squared paired-difference error improves over the task/checkpoint action mean by about 0.6% and 1.7% in daughter at the two checkpoints. It is worse than the action mean in both trunk conditions. The cohort selector's largest development gain was +1.69 points and all four cohort comparisons failed measured cost equivalence.

[Frozen predictor and prices](model/predictor.json) · [Development forecast audit](model/forecast_audit.json)

## Prospective paired benchmark

There are 512 fresh independent latent cases, reused across the two tasks and checkpoints, and one common tape per case. Every assigned case remains in its denominator. Each outcome cell shows correct completions / any wrong confirmations. Motion equivalence refers to the primary B comparison.

| Task and common prefix | MPC | Primary cohort-priced B | Per-state cost B | Fixed continuation | Primary motion equivalent |
| --- | ---: | ---: | ---: | ---: | --- |
| Daughter, anchor | 131 / 68 | 132 / 72 | 132 / 67 | 136 / 68 | No |
| Daughter, two commands | 131 / 68 | 141 / 67 | 130 / 69 | 142 / 69 | No |
| Single trunk, anchor | 425 / 33 | 431 / 38 | 428 / 35 | 425 / 33 | No |
| Single trunk, two commands | 425 / 33 | 427 / 32 | 426 / 33 | 425 / 33 | Yes |

The primary selector changes the sampled route in 318, 270, 313 and 95 cases in table order. It selects two-command candidates in 307, 209, 318 and 155. A public claim after the first command produces immediate handoff in 25, 45, 35 and 92, respectively. It uses a same-pose realization change as its first command in 11, 123, 34 and 15 cases. One executed pair in daughter after two commands uses opposing tilts at the same site. The actual intervention evidence is for this continuation library, with limited use of opposing-view complementarity.

The largest primary gain is +1.95 points in daughter after two commands, with a paired 95% interval of +0.20 to +3.71. Sampled duration misses the strict two-sided equivalence margin on the faster side. The harmful wrong-identity discordance upper bound is 2.53%, above the 2% criterion despite one fewer total wrong confirmation. In the only motion-equivalent primary cell, trunk after two commands, the gain is +0.39 points with a paired interval of 0 to +0.98. All four per-state B conditions are motion-equivalent, with gains of +1, −1, +3 and +1 correct cases. None meets the prespecified +5-point success criterion.

The fixed control after two daughter commands is the most informative route finding. The rule is available in 95/512 cases. It goes to the center at the final site, uses its checked +12° view if unresolved, then returns to the shared MPC. It wins 12 pairs and loses one, giving +2.15 points with a paired interval of +0.78 to +3.52. All 13 discordant cases start at the center of the already accepted previous site. Direct MPC initially spends a capture at that same site. The fixed rule initially advances to the final site. Its TCP distance and rotation intervals lie inside their equivalence margins. The simultaneous sampled-duration interval is [−0.132, −0.015] s, so it satisfies the no-worse cost criterion and falls outside strict two-sided time equivalence. Any wrong confirmation rises by one net case, with two harmful and one favorable discordance and a 1.40% upper bound on harmful discordance. This secondary comparison points to a task-progress scheduling improvement. It does not isolate a benefit of contextual prediction and has no supporting effect in the trunk task.

The paired tables include the absolute cost means, simultaneous intervals, route counts, wrong-return subtypes and an additional CPU ledger. CPU includes candidate checks, feature generation and prediction for B. The acoustic clock advances with sampled motion only.

[Paired endpoint and cost tables](data/confirmation/paired.json) · [Per-case public inputs, forecasts, executed trajectories and evaluator labels](data/confirmation/cases.jsonl.xz)

## Reproduce

Run from this directory with Python 3.10, NumPy, SciPy and Numba. The prior study's checked bank and pinned `acoustics.py`, `control.py` and `run.py` are verified by hash at startup. The raw JSONL columns are documented in each cohort's settings. `combined_s` includes motion and measured CPU and is machine-dependent. The confirmed policy is deterministic from the frozen model and public history. No train or diagnostic labels are needed to run it.

```bash
repo_root="$(git rev-parse --show-toplevel)"
replay_dir="$(mktemp -d)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for cohort in oracle extended_oracle train development confirmation; do
  mkdir -p "$replay_dir/$cohort"
  xz -dc "data/$cohort/cases.jsonl.xz" > "$replay_dir/$cohort/cases.jsonl"
done
python opportunity.py "$replay_dir/oracle"
python opportunity.py "$replay_dir/extended_oracle"
python predictor.py "$replay_dir/train" "$replay_dir/development" --out "$replay_dir/refit"
python validate.py --out "$replay_dir/confirmation" --report-only
diff -u data/confirmation/paired.json "$replay_dir/confirmation/paired.json"
python validate.py --repo "$repo_root" --out "$replay_dir/rerun" --freeze FREEZE.json --workers 16
```

To regenerate candidate records, run `continuations.py` with `--repo`, `--out`, and the corresponding split, count, repeat count and library in each `settings.json`. The two cohort allocations can have multiple integer solutions. Chosen actions for the two ranking halves are archived for direct audit. This evidence comes from one nominal anatomical chart, synthetic centroid packets, and sampled checked kinematics. Physical contact, B-mode images, and object motion during planning remain outside the test.
