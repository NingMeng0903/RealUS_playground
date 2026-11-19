# Learning independently verified online recovery choices

The initial scan, feasible first-action library, CEM-fitted ProDMP, feedback every 100 ms and MPC continuation are unchanged from the [learned-B comparison](LEARNED_B_PROTOCOL.md). The experiment changes how the first decision is supervised.

## Training data

For the 288 previous training states, retrieve the teacher choice from the stored ten-sample public posterior rollouts. On 128 new unbranched training states use the same twelve-sample public teacher as the benchmark. The new states include 48 unselected prefixes and forty prefixes in each of two public strata. One stratum has exactly one confirmed interval and an unfilled interval behind the probe. The other has its first unfilled interval more than one interval ahead of the probe. Scout based on the public prefix and select before checking teacher outcomes. The previous test cases informed these strata through the [audit](TEACHER_CHOICE_AUDIT.md). Their seeds, raw outcomes and labels are excluded from the new training and model selection.

At each root keep the MPC choice, the public teacher's preferred alternative when different, and the highest-scoring remaining alternatives on either side of the teacher's resource check. Pair all these choices on twelve new evaluator disturbance tapes and run the actual feedback segment and common MPC through scan termination. A verified choice has a positive realized completion difference, no added mistake on any paired tape and a mean realized cost difference within the existing margins. Use repeat outcomes from the entire task to supply positives and harder negatives.

Fit a choice classifier that imitates the best verified action or the incumbent. Fit a second model that predicts a signed advantage with positive targets restricted to verified actions. Its zero-valued incumbent remains available. Both use the previous public position and belief features with the public coverage mask and candidate/ incumbent action codes. They learn from realized paired outcomes. The prior separately predicted duration threshold is absent from their online decision. The simulator still checks admissibility, and the evaluation still checks realized resources.

## Development and held-out comparisons

Select on 96 newly seeded development cases per family with eight paired repeats. Minimize realized mean margin violation, including first-decision CPU time in the duration comparison. Among variants with equal violation, select higher correct completion. Hash the resulting weights, training and development data, library and ProDMP before running the held-out test.

Test 96 independent cases per family with four paired repeats. The five arms are direct strong MPC, frozen prior learned B, verified-teacher B, full twelve-sample online B and online B without extra CEM proposals at the first decision. That last arm retains the actual MPC incumbent even when it is a CEM proposal. All other permissions and the full MPC tail are shared. The online ablation uses the same bank and full set of computed forecasts so the comparison isolates additional selection options. Its timing column reflects the reused full search.

Report correct completion with the previous five-percentage-point threshold and scene-clustered 95% intervals. Report final wrong and new centered mistakes, the realized TCP path, rotation, samples and simulated scan duration, plus measured first-decision CPU and its sum with simulated scan duration. Use the prior margins and one-sided scenario-level error sensitivity. Both task families have independent held-out prefixes. The simulated chart and acoustic model define the scope of the result.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B distill_recovery.py --source results/learned_b_cem_v1 --out results/verified_teacher_b_v1 \
  collect --random-cases 48 --stratum-cases 40 --scout-cases 2500 --replicas 12 --jobs 24
python -B distill_recovery.py --source results/learned_b_cem_v1 --out results/verified_teacher_b_v1 \
  train --cases 96 --replicas 8 --jobs 24
python -B distill_recovery.py --source results/learned_b_cem_v1 --out results/verified_teacher_b_v1 \
  test --cases 96 --replicas 4 --jobs 24
```
