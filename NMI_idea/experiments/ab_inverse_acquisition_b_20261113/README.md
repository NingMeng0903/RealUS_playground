# Model-generated task-preserving acquisition: an active negative B test

The new inverse-view mechanism was implemented and exercised. It did not demonstrate the prespecified correct-completion improvement over full replanning MPC at matched measured cost. Independent ASTRA and SOL reviews stopped the tilt predictor before training or opening a prospective efficacy cohort. A geometry-only offset preflight also did not justify another spatial bank.

[Protocol](PROTOCOL.md) · [Protocol freeze](PROTOCOL_FREEZE.json) · [Theory](theory/BOUNDED_DESIGN.md) · [ASTRA verdict](theory/OUTCOME_VERDICT.md) · [Independent SOL review](independent/INDEPENDENT_REVIEW.md)

## What changed from the failed B studies

Earlier studies already tested conditional observation programs, full MPC handoff, persistent joint hypotheses, complementary tilts, equivalent dIRD realizations, direct supervised continuation values and all-return association. This study changes the acquisition itself. A finite search of continuous tilt parameters adds ±3, ±6 and ±9 degree views within the old ±12 degree range. At the common anchor, the public posterior proposes up to two new angles per required station: one for expected correct-minus-wrong identity acceptance and one for expected information about subsequent identity decisions. Proposals replace needed captures, without a compulsory extra probe.

The generated menu is available to direct MPC and every intervention. All use the same observation kernel, claim rule, posterior and prior-claim correctness updates, graph, constraints and termination. MPC continues to update and replan after every real packet. The tested MPC is the inherited two-capture receding optimizer with a remaining-task value approximation; it is not an optimal POMDP solver and is not a fixed hand-coded route.

The generator uses the probabilistic model and numerical optimization over a finite angle grid. It does not use RL. Its information objective is a proposal surrogate, not a learned probability of final task completion. Generic decision-directed sensing, dual control and rollout are prior foundations. The proposed contribution was their specific coupling to task-preserving feasible acquisitions, contingent on new empirical evidence. This study did not establish that contribution.

The benchmark continues the [two-station full-episode study](../ab_full_b_rollout_20261113/README.md). Its endpoint and initialization differ from the original 76/128 moving-Y test; those rates should not be combined. ASTRA MAX designed/reviewed theory, SOL MAX independently audited execution and statistics, and a Luna worker built the mechanical extension. The primary agent implemented the observation extension, trial runner and temporal integration. These are separately tasked model-agent audits, not external human peer review.

## Execution and observation validity

The checked bank contains 90 configurations and 2,196 sampled edges, including 60 new configurations and 2,028 new edges. The old 30 configurations and 168 edges are preserved, including JSON types. New configurations use the original dIRD model, seeded IK and endpoint checks. Every added edge has 76 samples over 1.5 seconds, checked against joint limits, configured self-collision and speed limits. Independent SOL replay rechecked all endpoint FK/dIRD and all edge rate/limit traces; 173 spread paths received independent FK/collision/cost reconstruction. These checks do not establish maintained tissue contact, body collision avoidance or native controller execution.

All arms use the legacy fine 7×7 nearest-return packet and multiplicity flag. The observation model exactly reproduces the legacy angles. Novel angles rotate the same latent lumen geometry. False-return visibility and effective clipped positions are smoothly interpolated between the old endpoint regimes, without using a new random seed per new view ID. This is a declared synthetic model extension, not measured ultrasound. Independent audits checked 2,016 legacy packet/detection/CDF combinations, 4,032 new projections and exact view-ID alias invariance.

Diagonal advancement while tilting is added for both old and new angles. Comparisons therefore distinguish original graph, coarse diagonal graph, inverse-angle menu, claim-only menu and fixed matched-cardinality angle menu. The latter controls prevent attributing every benefit of more motion choices to information-aware B.

## New paired opportunity experiment

There are **32 independent new latent cases**, each with 24 suffix disturbance tapes, evaluated on daughter and genuinely nonbranch trunk tasks. The same cases across tasks and repeated tapes do not increase the independent sample count. The new seeds were checked against 1,685 distinct seeds in 10,314 previous records; none overlaps. Every assigned case remains in the denominator.

Each route starts from the same anchor observation and robot state. A correct outcome requires fresh correctly identified target acquisitions at both ordered stations and actual completion of the checked continuation. Any wrong identity claim is recorded and makes correct completion false. Hard caps are 65 mm TCP travel, 90 degrees turn, 250 mm rail travel, nine sampled seconds and six acquisitions.

First, an evaluator-only selector ranks feasible task-preserving first commands on 12 tapes and tests the selected command on the other 12; the halves swap. Its tail is exactly the shared inverse-menu MPC. The selection sees outcome labels from other tapes, so it is **not deployed B**, not evidence that an observable predictor exists, and not a population oracle upper bound. Its purpose is to test whether this library offers reproducible improvement worth learning.

| Diagnostic over full inverse-menu MPC | Correct-completion change | Paired 95% interval | Any-wrong change | Strict cost equivalence | No-worse costs within margins |
| --- | ---: | ---: | ---: | --- | --- |
| Cost-screened continuation, daughter | +0.91 pp | −0.26 to +2.34 pp | −0.13 pp | No | Yes |
| Cost-screened continuation, nonbranch trunk | +0.39 pp | −0.39 to +1.30 pp | −0.13 pp | Yes | Yes |
| Hard-cap-only continuation, daughter | +4.04 pp | +0.78 to +7.81 pp | −0.65 pp | No | No |

The daughter cost-screened selection gives 196 versus 189 correct suffix episodes, and 71 versus 72 wrong episodes, over 768 tapes. Trunk gives 664 versus 661 correct and 41 versus 42 wrong. These are repeated outcomes from 32 cases, not 768 independent subjects. Both cost-screened improvements span zero, fall below the frozen three-point screen and do not beat the simple fixed progress control. Daughter's strict-equivalence failure is on the lower side of the turn interval, [−2.081,+0.203] degrees. It meets the separate upper-only no-worse test.

The hard-cap daughter gain requires +2.90 degrees mean turn. Its simultaneous upper excess costs are +2.76 mm, +5.60 degrees and +0.219 s, all beyond the respective +2 mm/+2 degree/+0.10 s margins. Equal hard caps therefore do not make this a matched-execution-cost improvement.

The generator alone also lacks a positive result. Full MPC with the inverse menu changes correct completion by −0.65 pp in daughter and −0.39 pp in trunk versus coarse diagonal MPC. Against a claim-only menu, its daughter mean gain is exactly zero. Adding coarse diagonal edges over the original graph gives an uncertain +2.47 pp daughter difference accompanied by +7.56 degrees mean turn. Detailed rates, cost means, simultaneous intervals, fresh packets and complete traces are retained in [the full summary](data/screen/summary.json) and [raw records](data/screen/cases.jsonl.xz).

This is an **active negative result**. Inverse-menu MPC uses new tilts in 590/768 daughter and 490/768 trunk suffix episodes, and every latent case uses a new tilt on at least one tape. Its menu differs from claim-only in 30/32 daughter and all 32 trunk cases. New decisions and nonzero predicted information did not yield the required terminal task gain. No advantage gate was adjusted.

## Independent checks and limits

SOL independently reconstructed 64 common anchors, 768 unique tapes reused across tasks, 768 proposal witnesses, 285 forced-continuation witnesses, all 16,056 arm trajectories and 76,328 captured packets. All 128 split-half choices and 16 paired contrasts reproduce. Spread controller replay rechecked 2,868 live MPC commands and 668 final public-state hashes. All 64 sampled original-control runs match the pinned compact-graph implementation.

The selector uses private-case repeat labels only for diagnostic ranking; the actual rollout controller never receives them. A future learned predictor would have to use public evidence alone and be tested on new cases. No such tilt predictor was trained, and namespaces reserved for its training, development and 256-case efficacy confirmation were not opened.

CPU is recorded, including inverse-menu construction charges. Computation does not move the acoustic clock in this spatial experiment. The combined motion-plus-CPU ledger is not a real-time robot validation. Diagnostic menu controls receive the same menu CPU allocation by convention rather than independent optimized-implementation timing. The test uses synthetic returns and one nominal posed chart; it is not a measured B-mode, cardiac or respiratory validation.

The builder's import-resolution wrapper was repaired after the screen. The numerical bank and online controller/world/analysis files did not change. [The exact screen-time builder](motion/motion_at_screen.py), its matching recorded hash, [the exact patch](motion/motion_postrun.diff) and [change log](motion/POST_RUN_CHANGELOG.md) preserve the distinction. Original settings were not rewritten.

## Distinct pivot preflights

ASTRA checked existing projected skin offsets using only the 256-particle public prior, with a single canonical false structure to avoid view-ID artifacts. Small translations cancel with the nominal-relative observation bins except at the aperture. No-aperture bin differences are below 4.3e−15. Offsets +8/+16 mm give little separation; +24 mm is not surface-projection feasible at daughter station 4 or trunk stations 6/7, and loses trunk target observations. At the one supported final daughter station, selected-target probability increases only 0.15–0.29 points. This does not justify building an offset bank. [Preflight data](theory/APERTURE_PREFLIGHT.json) · [Reproduction](theory/aperture_preflight.py).

A separate causal, prior-only temporal preflight checks delayed capture during existing temporary loss/lag. It uses original movement plus stationary holds of 0/0.5/1 second. There are 128 model-generated histories per task, with checkpoints after one and two public MPC captures. All histories remain in each denominator; only needed-site commands are scored. Depth one can overlap the additional dropout window; depth two's earliest next capture at 4.5 seconds is beyond its latest end at 4.45 seconds and tests lag/identity timing.

| Task/checkpoint | Active histories / 128 | Contextual nominal value above best fixed hold | Mean added hold over all histories |
| --- | ---: | ---: | ---: |
| Daughter / one capture | 108 | 0.002529 | 0.594 s |
| Daughter / two captures | 116 | 0.004225 | 0.445 s |
| Nonbranch / one capture | 26 | 0.000389 | 0.090 s |
| Nonbranch / two captures | 111 | 0.000344 | 0.363 s |

These are nominal utility changes, **not completion-rate gains**. Added holds are commanded waiting, not measured net final-episode time differences. No full suffix outcomes or time-aware-MPC comparison were run. The small forecast advantage did not justify opening another efficacy cohort or training a timing predictor. Both reviewers recommend stopping this specific same-command/hold pilot; the result does not rule out all temporal policies. [Summary](temporal_preflight/SUMMARY.json) · [ASTRA verdict](theory/TEMPORAL_VERDICT.md) · [SOL review](independent/TEMPORAL_REVIEW.md).

SOL independently rebuilt all 512 causal public states and 1,083 eligible hold scores. A correction separates unused finer-time innovation draws from the restored legacy-time draws; all 512 records and numeric summaries are byte/value identical before and after it. Both source versions and a parity record are retained. Planning CPU does not advance this preflight's sensor clock, so timing deployment still requires explicit latency treatment before validation.

## Reproduce

Use Python 3.10 with NumPy 2.2.6, SciPy 1.15.3 and Numba 0.65.0 for outcome replay; regenerating motion additionally needs the repository's original robot/dIRD environment. From this study directory, extract the current and inherited banks, then run the command below with `--motion` and `--base` pointing to those extracted JSON files.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench.py \
  --repo ../../.. --motion /path/to/BANK.json --base /path/to/BANK_MAIN.json \
  --out /path/to/new-replay --split screen --n 32 --repeats 24 --model-size 256 --workers 12
```

The bank is in `motion/BANK.json.xz`; the inherited bank is in `../ab_full_b_rollout_20261113/motion/BANK_MAIN.json.xz`. To recompute archived statistics, extract `data/screen/cases.jsonl.xz` into a separate replay directory and run `bench.py --repo ../../.. --out /path/to/replay --report-only`. CPU measurements are machine-dependent; chosen trajectories and non-CPU outcomes reproduce under the fixed code/model/tapes. The independent audit scripts document their own command-line arguments. `DATA_MANIFEST.json` hashes the released files.
