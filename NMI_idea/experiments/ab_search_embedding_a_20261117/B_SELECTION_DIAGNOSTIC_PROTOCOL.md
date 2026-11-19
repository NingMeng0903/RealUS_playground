# Diagnose recovery ranking and eligibility under the frozen executor

The original learning study and the verified-teacher study leave a gap between the public rollout scores and the learned first action. This diagnostic freezes the CEM action library, fitted ProDMP, actual frame-by-frame feedback and shared MPC continuation from the initial learning study. It reuses the original frozen mixed-label student and changes its ranking and eligibility independently.

## New development cases

Use 32 new independent seeds in each of the connection and unbranched cases. For each public prefix, form the smallest set of hypotheses, in descending posterior order, with at least 0.9999 of the total mass. Renormalize that set and average twelve disturbance tapes per retained hypothesis. The same disturbance tapes are shared across hypotheses and actions. Both the weight and the future simulated belief are public. Retaining the rest of the probability mass could change any modeled mean bounded between 0 and 1 by at most the omitted mass. This bound concerns hypothesis truncation. Finite disturbance sampling and the simulated likelihood remain separate sources of uncertainty.

The weighted teacher uses the existing cost and risk tolerance and original scalar objective. Freeze one bank for choice. Use a second independent weighted bank to reassess the chosen interventions and up to two other high-ranked eligible actions. Record how often the teacher choice stays eligible and best among those checked. For the four crossed policies, fixed MPC, and any additional chosen action, acquire the complete feedback segment and common MPC suffix on eight independent evaluator tapes. For each scene the real simulator latent and evaluator tapes are common across the arms.

| Arm | Predicted eligibility | Utility ranking |
| --- | --- | --- |
| Original student | Original student | Original student |
| Teacher eligibility and student ranking | Weighted teacher | Original student |
| Student eligibility and teacher ranking | Original student | Weighted teacher |
| Teacher | Weighted teacher | Weighted teacher |

Report both crossed contrasts against the original student. Eligibility includes forecast time, path length, turn and both mistake rates. The ranking includes the same costs and success score as the original. The ranking and eligibility can interact. A positive contrast after replacing both gives neither marginal component an automatic attribution. These are interventions on predicted decision components and evaluator outcomes. The teacher gate itself is an estimate.

## Matched supervision and input checks

After completing the crossing, keep the 64 diagnostic cases for controlled evaluation. Train on the earlier 288 training states, keeping the same trees, motion labels and decision rule. Compare all four combinations of including or excluding CEM candidate labels and posterior-only or 65/35 mixed labels. The variants have different counts of labeled candidate actions. All variants have the complete CEM library at execution. Reuse old replays if an arm chooses an already executed action on one diagnostic tape. Execute a new paired replay for each newly selected action.

For the representation check, append each public certificate's presence, age, relative station, centered-and-correct posterior probability, and full arrays of conditional association and centering. Compare this encoder with the frozen mixed-label encoder under their own gates and under the identical teacher gate. Each uses the same underlying mixed labels. This development comparison tests whether the extra input improves choice under the current hypothesis and training model.

For a training-state sanity check, sample 32 states per family from the archived training set with a fixed RNG prior to checking performance. Compare the original frozen student with the best option under its exact 65/35 training target. Re-evaluate both on a separate weighted posterior bank, and replay the two choices and MPC on eight new evaluator tapes. Report regret on the fitted labels and the utility and completion gap on independent noise. Evaluate novel features and fixed label recipes independently of this training-state replay.

Use case-level paired intervals, retaining all replays in each resampled case. These are development diagnostics under the matched synthetic fixed-pose benchmark. A candidate deployment change would require a separate model freeze and untouched confirmation cases.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B diagnose_b_selection.py --source results/learned_b_cem_v1 \
  --out results/b_selection_diagnostic_v1 --cases 32 --samples 12 --replicas 8 --jobs 24
python -B check_b_training.py --source results/learned_b_cem_v1 \
  --out results/b_selection_diagnostic_v1 --train-cases 32 --jobs 24
```
