# Learning the recovery choice with the new ProDMP library

## Frozen comparison

Use the six requests and the ProDMP fit from the 61 matched recovery demonstrations in [the budget-matched search](SEARCH_EFFICIENCY.md). The CEM and structured random search versions of those files are identical. The paths are replayed in the reference fixed-knee simulator with continuous geometry jitter and expanded opening shadow, as in the prior startup study. Both arms share the same data association, delayed identity confirmation, within-segment feedback, motion checker, and absolute execution budgets of 67 mm and 8 s. The reference MPC sees the same feasible menu of stock and newly discovered requests as the learner.

Both arms start at the same one-second scout. Select new scenes from the recorded initial packets only. The target is missed across all 11 packets. Half the scenes in each split contain a detection of only the competing vessel, and half have no genuine detection. The connection and unbranched families are balanced within each split. There are 192 training scenes, 64 development scenes, and 128 reserved deployment scenes. None is reused from the motion fit. The first action comes from the assigned selector. After that action the same MPC continues from its actual measured state in every arm, with observation updates and replanning.

## Labels and learner

At each training and development scout exit, enumerate every mechanically admissible menu candidate. Evaluate the same six new post-scout disturbance tapes for each candidate and the MPC reference. The trajectory is generated with the shared ProDMP fit or its common analytic fallback and evaluated through task termination. Geometry and disturbance realizations are paired across candidates. Candidate outcomes can differ because both successor poses and observations can differ.

Each row gives six targets relative to the complete closed-loop MPC result from the same scout. The targets are correctly identified completion, final wrong identity, new intermediate identity error, TCP travel, rotation and simulated time. Correct completion requires correctly identified detections from all six task intervals and reaching the scan endpoint. Centring within 3 mm is a secondary measure. Simulated time excludes decision computation.

Use the existing regressor and public input representation. Features include the material addresses of the probe and candidate imaging windows, depth limits, the current vessel belief and association evidence, and the candidate relative to the incumbent. True geometry is available when simulating and scoring training outcomes. For inference the regressor receives only the public chart, controller state and posteriors. Train on the six paired outcome differences without selecting a new model family, fitting new ProDMPs or tuning the existing price weights and forecast caps. The development report diagnoses the selected requests and compares them to the optimistic choices available when all its evaluator labels are known.

## Deployment validation

The reserved test applies both policies to eight fresh paired disturbances per scene. Report completion, identity errors, costs, chosen requests and CPU decision time. Compute intervals with scene-level paired resampling. Report the previous 5-point completion and 2-point error criteria as reference, alongside actual changes in every resource. Retain positive and negative outcomes, and keep the evaluation set out of refitting or threshold selection.

## Results

The training archive contains **3,840 candidate records from 192 scenes** with 23,040 actual post-scout rollouts. Each candidate has six disturbances and six measured outcomes per disturbance. The separate development archive contains 1,280 records from 64 scenes and 7,680 rollouts. The trained models use 414 inputs per candidate and retain the same ProDMP weights from the 61 source demonstrations.

| First recovery selector | Correct completion | Final wrong identity | New interim wrong identity | Mean TCP travel | Mean turn | Mean simulated scan time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Strong MPC with the shared menu | 922/1024 | 2 | 11 | 37.47 mm | 6.73° | 6.138 s |
| Learned B with the shared menu | 1005/1024 | 0 | 11 | 36.85 mm | 6.36° | 6.040 s |
| Development-selected fixed first request followed by MPC | 1022/1024 | 0 | 9 | 35.76 mm | 5.90° | 6.023 s |

Learned B improves completion over strong MPC by **8.11 percentage points** with a scene-paired 95% interval of **+4.98 to +11.52**. There are 85 B-only completions and two MPC-only completions. The connection family changes from 498/512 to 504/512 and the unbranched family from 424/512 to 501/512. Gains concentrate in the unbranched cohort whose scout observes only the competing vessel, where completion changes from 172/256 to 247/256. There is little room to improve in the cohort with no genuine detection at the scout, where completion changes from 505/512 to 508/512.

B changes the first choice in 77 of 128 scenes and uses a new-library request in 57 scenes, compared with five for MPC. Target reacquisition in the first required interval increases from 931/1024 to 1008/1024. In 78 of the 85 B-only completions, MPC missed that interval and B obtained its target image. Mean decision CPU time is 144 ms for MPC and 202 ms for B. The simulation pauses during this computation.

The fixed request is an additional diagnostic added after the main comparison. Its action name is chosen per known task family using mean closed-loop utility on the 64 development scenes alone. Both families independently choose the same request from the new library. It advances 2.60 mm, moves 6.53 mm laterally, and uses a 1.10 mm interior lateral deflection. Its frozen choice is replayed on the main held-out scenes with their original tapes and the same full MPC continuation. Learned B trails it by **1.66 percentage points** with a scene-paired 95% interval of **−3.52 to −0.29**. The fixed request already reacquires the early target in 1022/1024 executions.

These results establish a benefit of the learned first recovery choice over the incumbent MPC in this startup distribution. The fixed diagnostic shows that one effective recovery request suffices to attain the observed benchmark gains and leaves any adaptive advantage over that prefix unestablished. This comparison uses a fixed knee angle, a single reference chart, and synthetic observations. It evaluates the learned selector as a whole under shared sensing and motion generation.

[Label counts](results/startup_b_new_library_v1/labels.json) · [Frozen learner](results/startup_b_new_library_v1/frozen_models.json) · [Main paired results](results/startup_b_new_library_v1/test_summary.json) · [Fixed-request diagnostic](results/startup_b_new_library_v1/fixed_check_summary.json) · [Records and weights](results/startup_b_new_library_v1/)

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=results/startup_b_new_library_v1
python -B train_startup_b.py prepare --out "$out" --jobs 24
python -B train_startup_b.py collect --out "$out" --jobs 24
python -B train_startup_b.py train --out "$out"
python -B train_startup_b.py test --out "$out" --jobs 20
python -B train_startup_b.py fixed_check --out "$out" --jobs 20
```
