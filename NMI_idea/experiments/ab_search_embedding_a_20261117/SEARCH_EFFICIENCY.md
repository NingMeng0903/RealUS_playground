# Equal-rollout search for startup recovery demonstrations

## Paired discovery

Use the fixed-knee, streaming ultrasound simulator defined in [STARTUP_OOD.md](STARTUP_OOD.md). Discovery and fit evaluation each have 64 fresh independent scenes. Each set has 16 per cell of connection ambiguity or unbranched anatomy, crossed with no genuine detection or detections of only the competing vessel during the first 11 packets. Select the earliest eligible seeds based solely on the stationary scout. All scenes use the prior study's lateral offsets near ±9 mm and wider startup shadows. The two sets are independent draws from the same stress distribution. Fit evaluation seeds stay out of demonstration search and library construction.

Both offline search methods start from the same scout, three proposal anchors, initial Gaussian scales, and motion constraints. Each round scores four distinct feasible requests per anchor. A request consists of the longitudinal endpoint, lateral endpoint, and one interpolated lateral deflection. Invalid and duplicate proposals are recorded in the proposal count and do not consume a simulation. For CEM, the best two by simulated utility per anchor update that anchor's Gaussian. For random search, the initial distributions stay fixed. The common first round must match exactly.

Every scored request uses three shared disturbance tapes. Replay its first segment with the same streaming belief update, candidate retention and feedback, then finish with the same MPC. The ranking uses the existing utility for correct completion, final identity error, travel and time. On each round independently replay the top two overall choices on six fresh shared tapes each. Qualification requires at least four correct completions and zero final wrong identities. Confirmation outcomes do not feed back into the search distributions.

The two methods each run 36 scoring and 12 confirmation rollouts per round, for **192 actual post-scout closed loops per scene** over four rounds. Record the number of charged calls when the first candidate qualifies. Keep running to the full budget to measure the same search curves. Lock that first candidate and independently audit it again on 12 fresh tapes. A successful audit requires at least eight correct completions and zero final wrong identities. Audits are reported separately from the 192-call search budget. A separate common bank of eight direct MPC runs identifies a descriptive difficult subset, defined as at most six correct completions. This subset does not affect scene selection or training.

The primary comparison is the number of scenes with a verified recovery at the same number of paid simulations. Report the stricter independent audit counts, cost to first verified with failures censored at 192, and the threshold counts at 48, 96, 144 and 192. Bootstrap paired differences at the scene level. The task requires correctly identified images for all six intervals and reaching the endpoint. The inherited physical caps are 67 mm of travel and 8 s simulated time. These times exclude planning computation.

After the main run, audit the seven other nominated finalists on the scenes where both methods found a candidate that failed the independent audit. Give each method 84 extra closed loops per eligible scene using the same 12-tape audit bank. Require the original six-tape qualification and the independent audit to pass. This diagnostic checks whether continuing the search would supply a more reliable candidate. Fitting and holdout both use the frozen main-run outputs.

## ProDMP fitting and holdout

Use the first independently audited demonstration from each method for the same set of scenes where both methods pass. Fit the same lateral ProDMP mapping from displacement and interior deflection to basis weights with the existing nominal trajectory fitter. Executed trajectories use shared streaming ultrasound feedback. The open-loop ProDMP fit learns the proposed lateral reference prior to feedback.

Each search method contributes up to six distinct independently audited requests to its holdout library, with library sizes matched. Rank those requests by their audit utility on discovery scenes. In each new holdout scene choose a request using the existing public posterior rollout over stock motions plus one library. This planner receives simulated geometry and observations sampled from its public posterior. The evaluator alone sees the actual holdout scene.

Compare the candidate libraries under the same analytic trajectory generator. Within each selected request compare the analytic trajectory, the random-demo fit, and the CEM-demo fit. If a fitted trajectory fails the shared motion checks, the analytic reference is the common fallback. Record direct applications, fallbacks, geometry differences and paired terminal metrics. Use eight disturbance tapes per holdout scene shared between all methods. Direct MPC and the public rollout over the stock library provide references. All configurations share the candidate ledger with delayed confirmation.

## Results

Each method used **12,288 actual search rollouts** across 64 starts, plus the separate audit of the first nominated recovery. Both found the same first qualifying request in every scene at the same cost. In 63 scenes the leading candidate passed at 42 calls and in one scene the second passed at 48 calls. The paired mean was 42.09 calls.

| Cumulative search calls per start | Fixed-distribution random search | CEM |
| ---: | ---: | ---: |
| 48 | 64/64 | 64/64 |
| 96 | 64/64 | 64/64 |
| 144 | 64/64 | 64/64 |
| 192 | 64/64 | 64/64 |
| First qualifying request also passes the 12-tape audit | 61/64 | 61/64 |
| First qualifying request also passes on the MPC-difficult subset | 14/15 | 14/15 |

The pair was concordant on each independent audit. In the three scenes where the first request failed its audit, both methods already had a robust second nominee in the common first batch. The additional audit used 252 rollouts per method across those three scenes. Each of the second nominees met the six-tape criterion and independently completed 11/12, 12/12 and 12/12 trials respectively with zero final wrong identities. Later CEM updates were unnecessary for these first validated demonstrations. Mean raw proposal attempts over the full budget were 50.47 for fixed-distribution search and 49.39 for CEM. The paid closed-loop counts were equal.

The 61 jointly audited demonstrations and the six library requests were identical for both sources. The two ProDMP coefficient files were byte-identical. In the holdout, the two libraries selected the same first request in every scene. Fitted trajectories directly executed the chosen probe in 16 scenes, including 13 additional-library requests and three preexisting free-form requests. The other 48 used the ordinary stock skills. No fitted request used the analytic fallback. The maximum lateral reference difference between the fitted and analytic realization was 0.113 mm.

| Holdout controller | Correct completion | Final wrong identity | New interim wrong identity | Mean travel | Mean simulated time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct MPC | 441/512 | 1 | 12 | 37.21 mm | 6.127 s |
| Public rollout over stock candidates | 443/512 | 1 | 7 | 36.39 mm | 6.102 s |
| Random-library rollout and random-fit ProDMP | 438/512 | 1 | 8 | 35.94 mm | 6.137 s |
| CEM-library rollout and CEM-fit ProDMP | 438/512 | 1 | 8 | 35.94 mm | 6.137 s |
| Either library with the same request executed analytically | 438/512 | 1 | 8 | 35.93 mm | 6.137 s |

For each selected request the learned and analytic versions agree on every correct-completion and identity-error outcome. Both expanded stacks have a paired completion difference versus direct MPC of −0.59 percentage points (95% CI −3.13 to +1.76). The efficient discovery comparison reached its ceiling in the common first round. It supports adequate first-round coverage by the fixed proposal distributions for this simulator and demonstrates continuity and replay fidelity for the fitted lateral references. It offers no evidence of an additional completion benefit from later CEM adaptation or from these ProDMP fits.

[Discovery summary](results/rollout_matched_search_v1/discovery_summary.json) · [Follow-up audit](results/rollout_matched_search_v1/remaining_finalists_summary.json) · [Holdout comparison](results/rollout_matched_search_v1/holdout_summary.json) · [Paired records and weights](results/rollout_matched_search_v1/)

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=results/rollout_matched_search_v1
python -B benchmark_search_efficiency.py screen --out "$out" --jobs 20
python -B benchmark_search_efficiency.py discover --out "$out" --jobs 24
python -B benchmark_search_efficiency.py holdout --out "$out" --jobs 20
python -B benchmark_search_efficiency.py audit_remaining --out "$out" --jobs 3
```
