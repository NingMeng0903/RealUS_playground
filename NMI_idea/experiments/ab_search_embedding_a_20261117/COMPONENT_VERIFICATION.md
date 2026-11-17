# Component verification for visible-target recovery

The archived visible-target test achieved 768/768 correct completions for both the strong MPC and B. These paired experiments measure the contribution of candidate retention, CEM proposal discovery, ProDMP rendering, and first-action selection separately.

## Common protocol

Correct completion requires reaching the endpoint and retaining a valid observation of the requested vessel in all six intervals, with identity confirmed at posterior probability 0.95. The ±3 mm centering result is recorded separately. Every closed-loop comparison shares an initial scan and, within each scenario, the same latent geometry and disturbance tape. The realized observations depend on each motion. All continuations use the existing MPC with actual synthetic observations and feedback every 0.1 s. Ground truth scores the outcomes and is also available during the explicitly privileged offline discovery and failure diagnostics. The online selectors receive the public posterior.

The evidence test reuses the archived fixed-knee-angle reference trajectories. The discovery, rendering, and selection tests use separate seeded scenes on the same chart, with both existing acoustic shadow widths doubled from 2.6 to 5.2 mm and from 4.5 to 9 mm. This single sensitivity setting was chosen to check for missed observations after the reference ceiling. It uses synthetic centroid packets and a matched prediction model. It leaves the task criterion, physical budgets, and FOV in place.

Report simulated TCP travel, accumulated rotation, execution time, image count, and decision CPU time separately. The simulated clock holds during planning. The 95% intervals resample paired scenario means with 10,000 bootstrap draws. Four or eight disturbance repeats within each scenario measure variability on that scene. The completion check requires a point gain of at least 5 percentage points and a positive interval lower bound. An error increase must have an upper bound of at most 2 points. Cost equivalence uses ±2 mm, ±2°, ±0.10 s, and ±1 acquired frame. The forecast constraint uses the same one-sided travel, rotation, time, and error margins. We also report cases that meet those upper bounds while using fewer resources.

## Candidate retention with delayed confirmation

Replay all 768 archived strong-MPC trajectories for 192 scenes with four disturbance tapes each. Both arms read the same packets in order and update the filter once per packet. One arm stores candidates only when their acquisition-time probability reaches 0.95. The other stores the conditional identity probabilities for all detections. Both re-evaluate their stored candidates after each new observation and use the same current identity threshold. The evaluation freezes the scan trajectory and stops at the same endpoint.

| Task | Keep candidates confirmed at capture | Keep candidates for later confirmation |
| --- | ---: | ---: |
| Connection ambiguity | 384/384 | 384/384 |
| Unbranched adjacent vessels | 0/384 | 384/384 |
| Pooled | 384/768 | 768/768 |

Final wrong identity was zero in both arms. Both arms registered interim wrong confirmation in four tapes from the same shared initial scan. The retained-candidate update took 0.293 ms median and 0.754 ms at the 95th percentile per synthetic packet during CPU replay. This comparison measures the use of previously acquired observations under the archived motion, where acquisition can continue while identity is pending. [Summary](results/component_verification_v1/evidence_summary.json) · [Scenario records](results/component_verification_v1/evidence_cases.jsonl.gz)

## CEM proposal discovery

Use 48 development scenes, three shared initial proposal distributions, six draws per round, and two rounds. CEM refits the sampling distribution between rounds. The random comparison keeps the initial distribution. Both propose 1,728 paths across the development set. Feasible, distinct paths get three paired search tapes with privileged knowledge of scene geometry. The winning executable request is then checked on eight independent disturbance tapes. A verified proposal must add at least two correct completions over the stock MPC opening, satisfy the relative execution caps on those tapes, and maintain or reduce both error outcomes.

| Development count | Random | CEM |
| --- | ---: | ---: |
| Distinct admissible paths scored | 1,510 | 1,558 |
| Paths within the forecast resource and error caps | 205 | 275 |
| Admitted paths with positive forecast completion gain | 10 | 10 |
| Scenes with a verified positive winning request | 0/48 | 0/48 |

For the online comparison, add the six best distinct development requests per source to the stock menu. The selector uses 16 posterior draws per action, the analytic renderer, and the same closed-loop continuation across sources. On 128 further scenes and four independent tapes each, stock-only completed 467/512. Adding random requests or CEM requests each completed 465/512, with identical completion outcomes on every tape between those two menus. Each added menu incurred one new interim mistake and zero final wrong identities. For CEM versus stock the paired change was −0.39 points, 95% interval [−1.17, 0.00], with equivalent execution costs. CEM and random differed by about 0.10 mm in mean executed travel in favor of CEM. At the same proposal budget, the extra CEM candidates improved the count of admissible paths without increasing correct completion. The same cap on proposals allows the number of simulated feasible paths to differ. [Summary](results/component_verification_v1/discovery_summary.json) · [Discovery records](results/component_verification_v1/discovery_cases.jsonl.gz) · [Online outcomes](results/component_verification_v1/library_cases.jsonl.gz)

## ProDMP execution at a shared request

In 96 fresh scenes, rotate through the six archived requests and test the first admissible analytic request for each public initial state. All scenes supplied such a request. All renderers start with the same state and end at the same longitudinal and transverse position and velocity with the same duration. Feedback, MPC continuation, and evaluator are shared. Both ProDMP versions can use the same analytic fallback. The old ProDMP first attempts the archived side-scan shape and then its zero-residual version.

| Outcome | Analytic curve | Old ProDMP | CEM-fit ProDMP |
| --- | ---: | ---: | ---: |
| Requests realized by that renderer before analytic fallback | 96/96 | 85/96 | 96/96 |
| Correct completions with common fallback | 359/384 | 359/384 | 359/384 |
| New interim wrong / final wrong | 2 / 0 | 2 / 0 | 2 / 0 |

The old ProDMP's initial side-scan fit passed 82 times and its zero-residual option passed three more. The fitted version required eleven fewer analytic fallbacks. Its completion outcomes matched the analytic curve on every tape. Relative to the old ProDMP it gained one completion and lost one. Both versions met the execution equivalence margins against the analytic curve. The trajectory representations and training differed between the two ProDMP versions, so the comparison estimates their renderer-level difference as used. [Summary](results/component_verification_v1/render_summary.json) · [Scenario records](results/component_verification_v1/render_cases.jsonl.gz)

## B selection under a common action menu

The strong MPC, the fixed probe, the posterior rollout teacher, and the frozen learned B share the original candidate menu including the six archived CEM requests, the fitted ProDMP and analytic fallback, and the same continuation. The fixed opening is advance, selected on the 48 development scenes by the smallest normalized cost and error cap violation and then completion. The fixed revisit had higher development completion in the connection group, 94/96 against 76/96 for MPC, with an extra 5.60 mm and 0.354 s on average. The teacher evaluated each candidate on 16 public posterior draws with no extra motion permissions. The learned B reused the reference fit trained under the original visibility setting.

| Selector | Connection ambiguity | Unbranched | Pooled | New interim wrong / final wrong | TCP mm / turn ° / scan s / frames | Decision CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Strong MPC | 216/256 | 256/256 | 472/512 | 1 / 0 | 34.98 / 6.128 / 4.738 / 48.38 | 24.0 ms |
| Fixed advance | 192/256 | 244/256 | 436/512 | 1 / 0 | 33.77 / 5.926 / 4.723 / 48.23 | — |
| Posterior rollout teacher | 216/256 | 256/256 | 472/512 | 2 / 0 | 34.26 / 6.133 / 4.576 / 46.76 | 23.95 s |
| Frozen learned B | 212/256 | 248/256 | 460/512 | 1 / 0 | 34.42 / 6.073 / 4.719 / 48.19 | 61.6 ms |

The table uses 128 holdout scenes with four tapes each. The teacher and MPC completed exactly the same tapes despite changing the opening in 31 scenes. The teacher reduced simulated travel by 0.72 mm and duration by 0.162 s on average. The planning CPU is accounted for separately. The learned B met execution equivalence and its completion change was −2.34 points with a 95% interval of [−5.47, 0.00]. In development, the strong MPC, high-budget teacher and privileged forecast selector each completed 172/192 under the forecast caps. The lack of a demonstrated teacher gain in completion leaves no matching training target for this test, so B was kept frozen.

Enumerate all legal first actions for each of the MPC's 40 failed held-out tapes while holding the continuation and disturbance fixed. All 40 reached the endpoint. Twenty had at least one interval with zero raw target detections. Every tape had a recovery under the absolute task budget. All 315 successful alternatives exceeded both the +2 mm travel and +2° rotation margins. This bounds the opportunity to improve completion with a different single opening in the tested menu and cost range. [Selection summary](results/component_verification_v1/selection_summary.json) · [Holdout outcomes](results/component_verification_v1/selection_holdout.jsonl.gz) · [Failure enumeration](results/component_verification_v1/misses_summary.json) · [Recovery costs](results/component_verification_v1/misses_cases.jsonl.gz)

## Recorded ultrasound

Audit the available aligned recordings by reading metadata and decoding one midpoint JPEG per file. All 204 files had strictly increasing frame timestamps, paired ultrasound and TCP samples, and a decodable sample. Across 123,486 images, the median interval was 32.10 ms and the median absolute TCP alignment timestamp error was 1.57 ms. The inspected datasets and neighboring sidecar filenames contained zero independent vessel identity labels. Identity accuracy, correction delay on actual B-mode, and closed-loop hardware performance remain unmeasured. The stored audit contains aggregate quantities and schema field names. [Recorded-input audit](results/component_verification_v1/recorded_input_audit.json)

## Reproduction

Run from this directory with the archived `results/visible_atlas_v1` inputs. The resulting case files contain paired outcomes. The scenario seeds and tape namespaces are specified in the validation runner.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=results/component_verification_v1
python -m unittest -v test_qualified_evidence
python verify_recovery_components.py evidence --out "$out" --jobs 6
python verify_recovery_components.py selection --out "$out" --cases 64 --samples 16 --jobs 16
python verify_recovery_components.py discovery --out "$out" --cases 64 --samples 16 --jobs 14
python verify_recovery_components.py render --out "$out" --cases 48 --jobs 6
python verify_recovery_components.py ceiling --out "$out" --jobs 12
```

The metadata audit can be repeated with an aligned ultrasound/TCP directory and an environment with `h5py` and `Pillow`.

```sh
python verify_recovery_components.py recordings --out "$out" --recordings /path/to/aligned-recordings
```
