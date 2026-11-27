# Startup recovery without relative forecast gates

## Comparison

Freeze the trained B regressors and the shared ProDMP candidate library from [the labeled startup study](STARTUP_B_WITH_RECOVERY_LIBRARY.md). The original learner filters candidates by predicted relative scan time and identity-error increments. Its relative travel and rotation caps were already disabled. Compare two additional selectors.

- **B without relative gates.** Rank every mechanically admissible candidate by the original predicted utility.
- **B prioritizing completion.** Rank the same candidates by predicted correct completion, breaking ties with the original utility and then the MPC choice. No relative forecast gates are applied.

Both use the same public features and fitted model. These changes affect first-action selection. The shared MPC continuation, delayed identity confirmation threshold, measurement model, mechanics, and absolute limits of 67 mm TCP travel and 8 s simulated scan time continue to apply. Report realized errors, travel, rotation and scan time.

## Evaluation

Replay the original 128 scenes using the same scout, candidate menu, and eight disturbance tapes. Check the frozen predictions and original action choices. Reuse archived outcomes for identical first requests and run a new closed loop on each newly selected request and tape. This cohort isolates the change after examining the previous comparison.

Confirm on 128 fresh scenes from the same synthetic fixed-pose stress distribution. Use new scene and disturbance seeds, with the same balanced design across task family and opening visibility. The target is missing during the shared scout in all cases, with half of the scouts detecting only the distractor. Keep the original MPC, gated B, and development-selected fixed action as references. Neither the weights nor the selection rules are tuned using these fresh trials.

For a finite-tape performance ceiling, replay every feasible first action in each scene where the fixed probe has any completion failure. Use the same eight disturbance tapes and full continuation. On each such scene report the best completion count attainable by a single first action chosen with hindsight over all eight tapes. Where the fixed probe already completes all eight, all eight counts are attained. This diagnostic uses evaluator outcomes and can be optimistic about new disturbances.

## Results

| Selector | Original scenes, correct completion | Fresh scenes, correct completion | Fresh final / interim wrong identity | Fresh mean travel | Fresh mean turn | Fresh simulated time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Strong MPC | 922/1024 | 906/1024 | 1 / 12 | 37.47 mm | 6.66° | 6.203 s |
| Original gated B | 1005/1024 | 1003/1024 | 1 / 11 | 36.52 mm | 6.28° | 6.065 s |
| B without relative gates | 1015/1024 | 1017/1024 | 1 / 9 | 36.17 mm | 6.06° | 6.080 s |
| B prioritizing completion | 1013/1024 | 1021/1024 | 1 / 10 | 36.78 mm | 5.99° | 6.219 s |
| Fixed first request followed by MPC | 1022/1024 | 1024/1024 | 0 / 11 | 35.92 mm | 5.94° | 6.053 s |

With the original utility, removing forecast gates adds 10 completions on the original scenes and 14 on the fresh scenes. On the fresh scenes this is **+1.37 percentage points** over gated B (scene-paired 95% CI +0.20 to +3.03). The average time difference is +0.015 s and travel decreases by 0.35 mm. Prioritizing completion adds 18 fresh completions over gated B, or **+1.76 points** (95% CI +0.49 to +3.52). Its observed mean scan time is 0.153 s longer than gated B. The completion-first result falls below the ungated utility selector on the original cohort. This limits the conclusion about removing the cost preferences from the score.

All recorded executions respect the physical and absolute scan limits. The largest actual travel among the two ungated policies on the fresh scenes is 60.60 mm and the longest scan is 8.0 s. Scan durations exclude inference time. The two cohorts are fixed-pose simulation draws with synthetic detections from the same stress distribution.

The observed single-first-action ceiling is 1024/1024 in both cohorts. For the original fixed probe's two affected scenes, the complete replays find an admissible choice that completes all eight tapes from each scene. The fresh fixed probe already attains the ceiling with one fixed choice. At this level, the current benchmark has too few fixed-policy failures to establish a completion benefit from adaptive first-action selection.

[Original-scene paired results](results/startup_b_without_gates_v1/replay_summary.json) · [Fresh paired results](results/startup_b_without_gates_v1/fresh_summary.json) · [Paired records and per-scene ceiling replays](results/startup_b_without_gates_v1/)

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=results/startup_b_without_gates_v1
python -B evaluate_startup_without_gates.py replay --out "$out" --jobs 24
python -B evaluate_startup_without_gates.py fresh --out "$out" --jobs 24
```
