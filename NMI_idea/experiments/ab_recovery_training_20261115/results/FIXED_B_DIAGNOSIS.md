# Why the older learned B did not improve fixed-knee completion

The original test pairs each policy on the same held-out scene and disturbance tape. All 192 scenes are evaluated at the reference knee angle of 15 degrees. The held-out vessel geometry and observation conditions still differ from the nominal training distribution. Correct completion requires correct new images in six intervals and arrival at the endpoint, without a centered wrong acceptance.

In the original test, the learned B finishes 136 scans and strong MPC finishes 143. The paired difference is −3.65 percentage points with a 95% interval of [−8.33, +1.04]. B changes the first choice in 94 cases, yielding eight exclusive wins and fifteen exclusive losses. Of those fifteen losses, twelve miss required coverage despite reaching the endpoint without a centered wrong acceptance. Two fail to arrive and one has a centered wrong acceptance. Incorrect candidate ranking does not always appear as incorrect identity.

Across the 94 modified choices, the scorer predicts 80.0% completion for its selected action and 75.2% for the MPC action. Observed completion is 61.7% and 69.1%. The selected action has higher predicted completion in 86 cases. A post-hoc reranking with the saved completion probability alone replays 133/192 completions, compared with 136/192 for the original learned rule and 143/192 for MPC. The model's success prediction remains an issue after removing the distance, local evidence and mistake terms from its action score.

## Repeated outcomes within the original action library

Training uses 1,410 action examples from 235 first recovery states. Each action at a state has one observed future disturbance tape. The six examples from a state share the patient geometry, public history and tape, so their outcomes do not represent six independent states. The original development check shows why labeling the first realized best action is unreliable. Its +7.58 point advantage on the selection tape becomes −1.52 points on an independent tape.

The supplement keeps the original 66 triggered development scenes, their public checkpoints, all six candidates, segment execution and MPC continuation. On eight new simulated future tapes per scene, it ranks the candidates by estimated correct completion, with mistake, local evidence and distance tie rules. A second selector enforces the existing upper resource margins and no increase in predicted wrong acceptance relative to MPC on the search tapes. Both selectors are privileged development references since the simulator keeps the actual hidden scene fixed. Final paired estimates use eight further, disjoint tapes per scene.

| First action selection | Correct completions on eight new tapes per scene | Difference to MPC, paired by scene |
| --- | ---: | ---: |
| Strong MPC | 411/528 | — |
| Archived best on one tape | 391/528 | −3.79 pp [−8.14, +0.95] |
| Eight-tape privileged selection | 404/528 | −1.33 pp [−3.98, +1.52] |
| Eight-tape selection with predicted resource limits | 407/528 | −0.76 pp [−2.46, +0.76] |

The resource-limited reference saves 1.49 mm on average in these replays. The interval reflects repeatability over the 66 development scenes, rather than 528 independent anatomies. This finite search through the original action menu finds little evidence of a repeatable completion gain from changing the first action. It leaves open improvements with other acoustic cues, actions or decision times, as well as finer estimates over disturbances. The completion-only re-ranking uses the original test seeds and diagnoses the saved scorer. It is not a fresh benchmark result for a retuned controller.

For this version, the useful distinction is between learning a more accurate ranking and furnishing repeatable opportunities that the ranking can exploit. Evaluate candidates with several paired rollouts at similar executed cost before labeling a training action as better. Spatial shifts in the held-out cohort and the documented severe undercoverage of the shared belief make the transfer test harder, even at a fixed knee angle.

[Summary](fixed_b_diagnosis/summary.json) · [Fixed score replay](fixed_b_diagnosis/fixed_score_replay.jsonl.gz) · [Development repeated outcomes](fixed_b_diagnosis/development_headroom_replay.jsonl.gz)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B diagnose_fixed_b.py \
  --out results/fixed_b_diagnosis --jobs 12
```
