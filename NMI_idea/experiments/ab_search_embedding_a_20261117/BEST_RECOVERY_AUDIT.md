# Recovery trajectory quality after using the full search budget

The previous [equal-rollout search](SEARCH_EFFICIENCY.md) compared the cost to find the first qualified demonstration. All first demonstrations came from rounds shared by the two methods. This follow-up fixes the original searches and ranks the qualified finalists from all four rounds. Pick the finalist with the best mean task utility on its archived six confirmation tapes, breaking ties in favour of earlier candidates. Both methods have the same 192 charged search rollouts. The controller receives the simulated observations at each frame. The offline search and evaluator have access to the true scene and use it to score correctly identified completion.

## Fresh-disturbance replay

Replay on the 64 original discovery scenes, each with 16 paired tapes disjoint from search, confirmation and the earlier audit. This measures the quality of offline requests in their source scenes. Generalization of a learned request selector or a fitted motion library would require separate holdout scenes. The shared portion of each trial includes the initial 11-frame scout, visual feedback, observation updates, revisable candidate retention, and the full MPC continuation. Simulated scan time excludes planning.

| Offline request or controller | Correct completion | Final wrong identity | TCP travel | Simulated scan time |
| --- | ---: | ---: | ---: | ---: |
| Direct MPC | 887/1024 | 5 | 38.63 mm | 6.271 s |
| Shared first qualified request | 967/1024 | 2 | 36.72 mm | 6.405 s |
| Best qualified fixed-distribution random request | 996/1024 | 1 | 35.97 mm | 6.172 s |
| Best qualified CEM request | 995/1024 | 2 | 35.93 mm | 6.314 s |

The best requests differ in 57 of 64 scenes. Fifty random requests and 52 CEM requests come from after the shared first round. The paired CEM-minus-random completion difference is −0.10 percentage points (scene bootstrap 95% CI −1.95 to +1.86). The difference in travel is −0.036 mm (95% CI −0.523 to +0.452), and the difference in simulated scan time is +0.143 s (95% CI +0.044 to +0.244). Best-of-budget selection yields higher point estimates than the shared first candidate. The random-search increase is 2.83 points (95% CI −0.20 to +6.05). Refining the sampling distribution provides no observed completion advantage at this budget.

## What the sampling distribution can express

The current smoothed, multi-start CEM updates a three-parameter distribution over longitudinal progress, lateral displacement, and one lateral interior deflection. Sampling at three anchors precedes the updates. Retiming the scan and choosing independent inspection angles are outside that parameterization. Starting at rest, the derivative of the nominal lateral reference with respect to its normalized phase is

\[
x'(t)=t^2(1-t)^2\,[30\Delta x+192b(1-2t)].
\]

It has at most one isolated interior zero. Adding adaptive iterations can relocate that turn. Generating several complementary observations requires motion degrees of freedom that express them. The iCEM paper used temporally correlated action sequences and reused nominated trajectories. Popović and colleagues optimized continuous information-gathering paths from a grid initialization using CMA-ES. [iCEM](https://proceedings.mlr.press/v155/pinneri21a.html) · [Popović et al.](https://doi.org/10.1007/s10514-020-09903-2)

Chen and colleagues' *Tuning movement for sensing in an uncertain world* generated active motions by matching the distribution of time spent along a trajectory to a predicted information density while penalizing movement effort. They used expected entropy reduction. The earlier robot study by Miller and colleagues built an information density from Fisher information. Multimodal information distributions can generate visits to several useful locations, and the scene's posterior drives the distribution. [Chen et al., eLife](https://doi.org/10.7554/eLife.52371) · [Miller et al., IEEE T-RO](https://doi.org/10.1109/TRO.2015.2500441)

## Fisher geometry and a complementary-view example

In the fixed-orientation simulator, a correctly associated image centroid changes as a translation in the local scan chart. The conditional-Gaussian approximation with constant measurement covariance \(R\) and visibility \(p_t\) gives \(F_{\tau}=\sum_t p_tR^{-1}\). Its local design ranking is the visibility ranking. This was checked on all 768 nominal first-round motions using the scout belief, modeled emission, packet outages, and the Gaussian in-field probability at each frame. This approximation omits information carried by censoring and by changes in the association distribution. The project's earlier tilted-view study also illustrates why the difference matters. With matched 36-degree travel, the fitted Fisher and visibility rules had nominal RMSE 3.6173 and 3.5559 mm, respectively. A separate task-weighted Fisher study reduced simulated oblique-response prediction RMSE from 0.3092 to 0.2072 mm using 24 measurements. The metric there was continuous response prediction. [Fixed-travel Fisher report](../ab_fisher_prodmp_20261112/FISHER_PRODMP_REPORT.txt) · [Response and continuation report](../ab_fisher_continuation_20261112/FISHER_CONTINUATION_REPORT.txt)

A controlled two-view model shows when information-aware generation can beat independent random sampling. Let the two equally likely target identities give signed response \(H\in\{-1,+1\}\), and let the measured feature be

\[
y_i=c+u_i H+\epsilon_i,\qquad
u_i\in\{-1,+1\},\quad
c\sim\mathcal N(0,9),\quad
\epsilon_i\overset{\mathrm{iid}}{\sim}\mathcal N(0,1).
\]

Each view starts and returns at neutral, with equal time, total travel and number of frames for every two-view program. With the same sign both times, the optimal identity classifier is correct with probability \(\Phi(1/\sqrt{9.5})=62.72\%\). With opposite signs, the difference cancels the unknown \(c\), and the probability is \(\Phi(\sqrt{2})=92.14\%\). Choosing each sign independently and uniformly gives 77.43%. In the corresponding continuous approximation to the signed response, the marginal information after integrating out \(c\) is \(2/19\) for matching signs and \(2\) for opposing signs. The advantage assumes that the two actual views have different responses to the identity and a common nuisance. It does not establish that the existing fixed-angle simulation has this mechanism.

[Replay summary](results/best_recovery_audit_v1/summary.json) · [Paired raw outcomes](results/best_recovery_audit_v1/audit.jsonl.gz)

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B audit_best_recovery.py --out results/best_recovery_audit_v1 --jobs 20
```
