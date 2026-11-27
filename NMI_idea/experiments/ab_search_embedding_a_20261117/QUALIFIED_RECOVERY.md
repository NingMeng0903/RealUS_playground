# Close the evidence, acquisition and training loop

## Shared evidence rules

From the first frame, strong MPC, CEM discovery, virtual and real teacher rollouts, and learned B use the same ledger. An image can enter the ledger when its observed centering and association pass the existing gates. For each retained image \(f\), define

\[
Q_t(f)=\sum_z b_t(z)P(\text{correct association for }f\mid z,Y_f)\,
\mathbf 1\{|x_{\mathrm{target}}(z,s_f,q_f)-x_f|\le 3\,\mathrm{mm}\}.
\]

A newly accepted image replaces the current image if its joint confidence is higher. An image is revoked when its marginal identity confidence falls below 0.95. An interval counts as covered when its retained image has joint confidence of at least 0.95 under the current posterior. Coverage is recomputed with each observed packet, including after an update that leaves the retained image in place. The scan ends successfully when it reaches the endpoint and all six intervals are covered with truly correct centered images. The evaluator checks the true identities and centering separately. The 0.95 joint cutoff reuses the existing identity cutoff as a simulation design choice. The controller retains one image per interval and the archive preserves the full packet and replacement history.

The same updated coverage controls replanning, stopping, and virtual completion. Closed-loop execution maintains ProDMP references and reconditions the segment using real packets every 0.10 s. The correction uses association, geometry under public hypotheses, and the current belief. Simulator labels are used offline for CEM, training outcomes, and evaluation. Legacy mode retains historical reproducibility. The [prior fixed-action audit](TEACHER_COMPLETION_AUDIT.md) evaluated a replacement rule without requiring joint confidence for coverage.

## New development runs

Strong MPC and the public teacher independently selected first actions from the shared menu after a common initial scan. Their full continuations updated and replanned after receiving new observations. The teacher used twelve paired posterior draws per first action and the original predicted error and resource limits. The 32 new cases per family use four separate evaluator disturbances each.

| Correctly completed | Connection | Unbranched | All |
| --- | ---: | ---: | ---: |
| Strong MPC | 105/128 | 52/128 | 157/256 |
| Public teacher | 116/128 | 52/128 | 168/256 |

The teacher's pooled improvement was 4.30 percentage points with a paired 95% interval of [1.17, 8.59]. In unbranched scans it lacked a centered eligible target observation in 220 interval instances. Of those, 216 were outside the true centering tolerance at every 50 Hz movement sample in the interval. The other four lacked an eligible target detection at acquisition. The two earliest intervals account for 136 of those 216 placement misses. The forensic counts are interval instances over repeated runs. The [paired development summary](results/qualified_recovery_v1/teacher_rerank/summary.json), [records](results/qualified_recovery_v1/teacher_rerank/cases.jsonl.gz), and [trace audit](results/qualified_recovery_v1/teacher_rerank/trace_audit.json) include actual paths and testable public evidence transitions.

## CEM on repeated acquisition failures

The search operated offline in the 18 unbranched development cases where the teacher lacked eligible target images in a common interval on at least two tapes. It targeted the most frequently missed interval, which was the first interval in 17 of these cases. The search optimized first-segment displacement and lateral bow. A second search also adjusted within-segment longitudinal timing with an added \(64\eta t^3(1-t)^3\) term. The term preserves the endpoint positions and velocities. Both searches used the common motion checks, fitted lateral primitive, real segment feedback, and MPC continuation. Each used five search tapes and eight independent verification tapes per case. The same-endpoint zero-timing control isolates the timing contribution within the chosen path.

In 16 of the 17 first-interval cases, the recovery decision began at station 17.6 mm with forward velocity. The first interval ends at station 7 mm. Even the combined 6 mm backward endpoint request and 2.4 mm within-segment timing bound leave the segment short of that interval. Recovery requires backtracking in the shared MPC continuation. The initial scan and the insertion point impose this limit on the tested search.

Under the original relative margins, the direct MPC, reoptimized teacher, and both targeted searches completed 0/144 checks on this failure-enriched development subset. Spatial search found 467 feasible draws and timing search found 397. None of the draws with a positive completion difference on the search bank passed the original relative margins there. No new request was admitted to the frozen library. Only one timing-search choice had a matched feasible zero-timing replay within the screened set, which limits the timing-specific conclusion.

For the 13 cases with a positive feasible candidate on the search bank, a separate replay verified the best candidate with the relative margins relaxed and the hard physical bounds preserved. It completed 103/104 checks against 0/104 on the same MPC tapes, with mean differences of +20.42 mm travel and +6.51° turning. There was one new centered mistake and no final wrong identity. This result measures the cost of recovering selected failures with privileged offline geometry. The [search outcomes](results/qualified_recovery_v1/targeted_cem/summary.json), [candidate archive](results/qualified_recovery_v1/targeted_cem/cases.jsonl.gz), and [higher-cost replay](results/qualified_recovery_v1/targeted_cem/higher_cost_summary.json) keep the exploratory and constrained results separate.

## Frozen learning and tests

The six previous motion requests and the earlier ProDMP lateral fit provide the shared executable geometry. Recompute all training and development outcomes under the new evidence rules. The training set uses 96 seeds per family, eight paired posterior rollouts and four evaluator repeats per feasible action. The model selection set uses 32 new seeds per family, eight posterior rollouts and six evaluator repeats. Freeze the shared library, fit, data and selected models before the test. Keep the original student target and cost limits to measure the effect of the corrected data and execution. The label ablation omits CEM proposals during training and still evaluates on the full menu.

The held-out test uses 96 further seeds per family and four paired evaluator repeats. Each B choice is followed by the same closed-loop MPC. Report correct six-interval completion, final wrong identity, new centered mistakes, simulated costs, and measured first-decision CPU time. Use a paired scene bootstrap and the prior success criterion of at least +5 percentage points with the paired 95% interval above zero. The cost margins are ±2 mm, ±2°, ±0.10 s and ±1 acquired packet. Report cost equivalence and the upper-margin check. The error margins are +2 percentage points. The test uses fixed-knee synthetic centroid packets and a finite shape model. Scan execution uses chart kinematics. Computational latency is measured separately from simulated time.

### Held-out results

| Policy | Connection | Unbranched | All correct | Final wrong | New centered mistakes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strong MPC | 304/384 | 128/384 | 432/768 | 1/768 | 5/768 |
| Learned B | 352/384 | 125/384 | 477/768 | 1/768 | 5/768 |
| Learned B without CEM labels | 353/384 | 128/384 | 481/768 | 1/768 | 5/768 |
| Online teacher | 372/384 | 129/384 | 501/768 | 1/768 | 4/768 |

The primary learned policy gained 5.86 percentage points overall, with a paired 95% interval of [2.73, 9.24] and 57 wins to 12 losses. The connection gain was 12.50 points [6.51, 18.49]. The unbranched contrast was −0.78 points [−2.86, 0.52]. The online teacher gained 17.71 points in the connection group and 0.26 points in the unbranched group. These results support a gain on connection ambiguity. Recovering early nonbranching misses at similar cost remains open.

| Mean change relative to MPC | TCP travel | Turning | Simulated scan time | Frames | First-decision CPU time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learned B | −0.91 mm | −0.14° | −0.210 s | −2.10 | 80 ms |
| Learned B without CEM labels | −0.75 mm | −0.12° | −0.190 s | −1.90 | 80 ms |
| Online teacher | −1.94 mm | −0.31° | −0.314 s | −3.14 | 31.21 s |

MPC used 41 ms mean first-decision CPU time. Learned B passed the pooled completion, error and every upper-cost margin. Its travel and turning also passed symmetric equivalence. Its duration and frame count fell below the lower equivalence margins because it finished earlier. The scene-level 97.5% upper bound for any added mistake across the four replay tapes was 1.90% in the pooled cohort. The simulations use stage execution time and do not advance during computation.

Development selected posterior-only target mixtures for both learned models. The full model used 3,975 candidate rows. The contrast from adding CEM labels was −0.52 points [−2.34, 1.30]. Both learned variants could use CEM motions at execution.

The trace audit independently recomputed frame-by-frame coverage, motion costs and terminal outcomes on the four policies' first tape per test case. It also checked handoffs and sampled motion limits. The archive retains all realized outcomes from the other tapes. The train, development and test seed sets were disjoint. All frozen input and model hashes were verified. The [frozen configuration](results/qualified_recovery_v1/frozen/frozen_models.json), [held-out summary](results/qualified_recovery_v1/frozen/test_summary.json), [test traces and outcomes](results/qualified_recovery_v1/frozen/test.jsonl.gz), and [trace audit](results/qualified_recovery_v1/frozen/trace_audit.json) include the evidence for these results.

## Reproduction

Run in this experiment directory with the archived CEM discovery inputs.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
study=results/qualified_recovery_v1
discovery=results/learned_b_cem_v1/discovery
python -m unittest -v test_qualified_evidence
python run_learned_recovery.py --out "$study" prepare --discovery "$discovery"
python run_qualified_recovery.py --source "$study" --out "$study/teacher_rerank" \
  --cases 32 --samples 12 --replicas 4 --jobs 24
python targeted_cem_recovery.py --source "$study" --out "$study/targeted_cem" \
  --audit "$study/teacher_rerank/cases.jsonl.gz"
python targeted_cem_recovery.py --source "$study" --out "$study/targeted_cem" \
  --audit "$study/targeted_cem/cases.jsonl.gz" --higher-cost-replay --jobs 6
python run_learned_recovery.py --out "$study/frozen" prepare --discovery "$discovery" \
  --targeted "$study/targeted_cem/summary.json"
python run_learned_recovery.py --out "$study/frozen" collect --split train \
  --cases 96 --samples 8 --replicas 4 --offset 40000 --jobs 24
python run_learned_recovery.py --out "$study/frozen" collect --split dev \
  --cases 32 --samples 8 --replicas 6 --offset 40000 --jobs 24
python run_learned_recovery.py --out "$study/frozen" train
python run_learned_recovery.py --out "$study/frozen" test \
  --cases 96 --samples 12 --replicas 4 --offset 40000 --jobs 24
python audit_adaptive_recovery.py "$study/teacher_rerank/cases.jsonl.gz" "$study/frozen/test.jsonl.gz"
```
