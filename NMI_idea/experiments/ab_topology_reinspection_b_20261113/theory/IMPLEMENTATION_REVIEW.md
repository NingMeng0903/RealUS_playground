# Implementation review: finite-family discovery and fresh offshoot inspection

Reviewed source on 2026-11-13: `world.py`, `core.py`, `suffix.py`, `prescan.py`, `learner.py`, `train.py`, `run_topology.py`, and inherited `return_sets.py`. This review does not open the primary held-out results and does not modify execution code. Line references describe the source at review time. Development counts below were supplied by the execution owner.

## Current formulation and what is supported

The implementation creates continuous, case-specific centerline geometry and a persistent ghost, then projects three possible emitters (offshoot/second vessel, trunk, ghost) at a fixed public bank of physically checked poses. The hidden family can be Y, single, parallel or crossing. The inference bank and evaluated worlds use disjoint draws. The common prescan uses five actual acquisitions with two-step replanning; all compared policies begin from the resulting common posterior, physical exit and clock. New positive declarations require two subsequent observed frames at prescribed adjacent sites. Candidate identities are marginalized under an unordered-multiset observation kernel, and private source identity is used for final scoring.

These are meaningful improvements over a single fixed-Y world. The task now includes uncertainty about whether the offshoot exists, where it begins and which side it occupies, with clutter and wrong-frame identity capable of causing false completion. The inherited exact multiset likelihood also permits a clean all-return versus nearest-return observation ablation.

The honest task name is **finite-family offshoot discovery followed by two-plane fresh inspection**, under a synthetic centroid sensor and a checked discrete motion bank. The following broader interpretations are not established by the code:

- **Rendered lumen connectivity:** `world.py:275–305` projects centerline points and uses an offshoot-presence cutoff. It does not intersect a connected tube volume with an imaging plane. A connected Y and a disconnected crossing are discriminated through the chosen point-geometry/visibility families. The sensor does not independently observe a joining lumen.
- **Actual U-Net observations:** no image segmentation network is executed. There are no rendered mask areas, eccentricities, merged masks or learned uncertainty scores. Inherited `return_sets.py` explicitly preserves repeated equal-bin returns as multiplicities. The comment claiming that quantization “merges indistinguishable returns” should distinguish identity ambiguity from mask merging.
- **General graph recovery:** the declaration is one of five task classes: no offshoot, or one of two fork intervals on either side. Single, parallel and crossing all share the correct no-offshoot declaration. Success does not require distinguishing all four graph families or recovering an arbitrary edge set.
- **A specified daughter among alternatives:** there is one side offshoot plus the continuing trunk. No separate public desired-daughter instruction is supplied. The current task is to inspect the discovered side offshoot. Choosing among two externally specified daughter identities would require an explicit public goal and corresponding evaluator.
- **Continuous coverage:** the endpoint credits two consecutive fresh observations at the prescribed near/far sites. It does not measure maximum along-vessel coverage gap or image acquisition along the connecting motion. Report two-plane inspection rather than full daughter coverage.
- **An initially unseen territory:** the common prescan already visits sites 7 through 3. The suffix is fresh evidence collection in that same region. This supports fresh inspection after uncertain prescan, not first exploration beyond the prescan field.

At review time the stress split widens the same random parameter families. A genuinely withheld sinusoidal branch/ghost shape and shifted continuous position, as now proposed, is a stronger family stress test. It still does not independently validate acoustic realism. Retain it as a separate shift cohort if the inference bank deliberately lacks that family.

## Correct implementation elements

The joint two-frame identity calculation in `core.py:terminal_evidence` is structurally appropriate for the current conditional-independence model. For a fixed hypothesis, the product of the two per-frame conditional target probabilities is valid because the frame randomness is independent across their clocks conditional on the persistent geometry/dropout parameters. The final belief can then reweight this old evidence. Summing over the shared hypothesis before deciding avoids the incorrect product of two unconditional marginal confidences.

The prefix is excluded from new inspection evidence: `Episode.__init__` starts `last` and `pairs` empty. Thus an old successful prescan cannot alone produce positive fresh-inspection success. Real motion costs and remaining caps are updated on every suffix command, and the action bank is shared. The primary `evaluate()` loop carries recurrent hidden state correctly.

Potential shaping is already policy-invariant for the declared undiscounted episodic reward. `Episode.step` returns `Phi(next)-Phi(current)-cost`, while `_finish` returns `true_terminal_utility-Phi(current)`. Therefore every complete episode satisfies

`sum shaped rewards = true_terminal_utility - sum movement/acquisition costs - Phi(initial)`.

The initial potential is fixed across policies for a paired episode. This means extra generic shaping is not a principled remedy for the zero-positive PPO pilot. The arbitrary constants inside the potential may affect training and approximate MPC decisions, but they do not change the exact complete-episode optimization target through this telescoping reward. This is standard potential-based shaping; see [Ng, Harada and Russell](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).

## Changes/checks needed before the frozen comparison

### Align the fork estimator with its actual loss

The present code chooses a weighted posterior mean fork location, then tests probability within ±5 mm of that mean. A posterior mean is optimal for squared position loss; it is not generally optimal for a tolerance hit/miss endpoint. This can discard an otherwise valid positive declaration when the conditional location posterior is multimodal.

For class `l` and frame-slot pair `(i,j)`, define

`w_h = b_h * 1[label_h=l] * eta_old(i|h) * eta_new(j|h)`.

The exact modeled probability of correct completion with declared fork position `c` is

`p(l,i,j,c) = sum_h w_h * 1[abs(fork_h-c) <= 5 mm]`.

Choose the length-10 mm interval containing the greatest total `w_h`, and place the reported center anywhere that covers that interval. A sorted-particle sliding window computes this directly. Then maximize over class and candidate slots. With the current wrong-completion penalty 1.5, the exact modeled stop value is

`L_stop = max(0, max_lijc [2.5*p(l,i,j,c)-1.5])`,

also including the no-offshoot decision. This is a Bayes decision correction for the existing endpoint, not a threshold adjustment. Apply it to **all** policies and freeze the change before primary evaluation. Keep the posterior mean as a separate position-summary statistic if desired.

### Align development selection with training/planning utility

At review time, `train.py:416,608` selects the checkpoint by

`success - 1.0*wrong - .001*TCP - .001*turn - .01*seconds - .001*captures`.

The environment/planners instead optimize

`success - 1.5*wrong - .001*TCP - .0005*turn - .0001*rail - .012*suffix_captures`.

Use the latter for declared utility-based development selection. Common prefix costs can be included or excluded consistently because they are policy-independent on each development cohort. Log positive completion and wrong absence separately; a pooled utility alone can hide a policy that serves only the easy negative class.

### Recurrent helper and clock interpretation

`train.neural_actions()` reinitializes the GRU with the prescan at every call; it does not retain intervening suffix history. The main `evaluate()` loop is correct. Do not use that helper for sequential DAgger collection without maintaining hidden state or replaying the complete public history.

The sensor clock advances 1.60 seconds per acquisition regardless of measured solve time. The comment assigns 0.10 seconds to planning, but the code does not enforce that allocation. Batched amortized solve time is not single-robot decision latency. Report this as a fixed-clock simulation unless a warmed single-case timing audit shows every policy fits the common allocation; otherwise charge delays or cap computations equally. Planning-time equality should not be inferred from the shared 1.60-second constant.

## Constructive learning recommendation

Use public-model demonstrations as a warm start, then online PPO, with pure cloning and same-data no-memory controls. This directly addresses the observed early-exit exploration collapse. Do not frame the first failed 64k-transition PPO run as evidence that RL cannot solve the task.

The most useful small addition to ordinary behavior cloning is **one or two rounds of dataset aggregation on student-visited states**. Query the public MPC/stronger teacher after the student deviates, record that state's full public sequence and legal mask, and imitate the teacher there. If the teacher exposes action values, cost-sensitive targets are better than treating a harmless adjacent view disagreement the same as a wrong early stop. Keep a pure initial-cloning arm and an aggregation-only arm if separating the online PPO contribution is important.

This is standard DAgger/AggreVaTe-style training, not a new RL algorithm. [DAgger](https://proceedings.mlr.press/v15/ross11a.html) addresses the state-distribution shift caused by the learner's own actions; [AggreVaTe](https://arxiv.org/abs/1406.5979) uses action cost-to-go information. Their guarantees require their learning/reduction assumptions and do not automatically apply to an approximate sampled MPC teacher or finite neural optimization here.

For the current bounded experiment, the minimum reliable sequence is:

1. Freeze one common public teacher dataset, containing all training cases and complete sequences. Count teacher queries and simulator interactions separately.
2. Train recurrent and no-GRU clones on exactly those sequences; check teacher-action agreement separately on move, abstain, absence and positive-declaration states.
3. Verify that both clones actually reach fresh-pair states and make positive declarations on development cases before spending the full PPO budget.
4. Fine-tune each with the same online interaction budget and original telescoping reward. Retain pure clones as controls.
5. Freeze multiple training seeds and select checkpoints with the aligned declared utility; report whether PPO adds anything beyond cloning.

Full particle posterior, class/family masses and current terminal-success probabilities are already in the policy input. Thus a recurrent advantage would not show that memory reconstructs anatomy from raw images. It would reflect useful history beyond this compressed task-state summary, approximation/optimization differences, or both. An exact Markov task state also includes old frame-pair evidence; the current scalar probability summary does not prove sufficiency under future posterior reweighting.

Do not replace terminal realized success with posterior expected success and claim unchanged real-world optimization: that substitution is unbiased only under a correct/calibrated model. It is a useful matched-model or low-variance teacher target, but can exploit inference miscalibration under the withheld-shape test.

## A tractable task-directed envelope for candidate selection

The current long control rolls out a greedy public tail to termination and ranks only ten first actions using a short heuristic. It is a legitimate sampled rollout control, but it is not an optimal long-horizon belief solver and has no automatic improvement guarantee over the two-step MPC. In particular, the MPC's first action is not forced into that top-ten set, and its continuation is not the rollout base policy.

A more grounded two-stage computation is possible without changing particles, poses, sensor or endpoint:

**Stage 1:** bound which physical first actions can still be valuable for completing fresh inspection. Use a cheap fully informed task relaxation to obtain an upper envelope.

**Stage 2:** spend observation-conditioned public-policy rollout/search effort on the remaining candidates, using the exact joint tolerance/identity terminal score. Distill those decisions for fast execution if useful.

These are computational stages, not a fixed confidence gate that irreversibly separates “discover” from “inspect.” Future acquisition, inference and inspection remain jointly valued. Diagnostic views should be considered whenever they may change the best complete feasible continuation.

### Oracle dynamic program

For each inference particle `h`, give an oracle the correct graph class, fork location and candidate identity, but **not future detection outcomes**. Preserve target-detection probability `d_h(t,a)` from the model. Use the same checked physical edges and acquisition clock. Relax cumulative movement caps initially; a relaxation creates an upper bound rather than a feasible policy.

For a Y particle, the task state needs only `(t, node, v, z)`:

- `v` records whether the immediately preceding acquisition was at the required near site and contained the true offshoot;
- `z` records whether a true near/far pair has already been completed.

Permit the oracle to retain its best completed pair forever. This is at least as permissive as the current implementation, which overwrites a group's pair with a later candidate pair. With `z=1`, the oracle may stop correctly for value 1, so this state need not be expanded. At the final clock and `z=0`, its best value is 0 by abstention. A non-Y particle may stop correctly for value 1 immediately.

For a scan from node `n` to `a`, let `D` be a Bernoulli variable with success probability `d_h(t+1,a)`. The next flags are

`z' = z OR (v AND site(a)=far(h) AND D)`

`v' = 1[site(a)=near(h)] * D`.

With actual scalar reward cost `c(n,a)`, the relaxed recurrence is

`V_h(t,n,v,0) = max(0, max_a {-c(n,a) + E_D V_h(t+1,a,v',z')})`.

Maximize only over edges present in the checked bank. The recurrence has `O(H*T*N^2)` work and two unfinished progress states per node, roughly a few million transitions for the current bank/particle sizes. It gives the oracle topology and geometry for free, so every admissible public-history policy is dominated **under the declared particle model**.

For a forced first action `a`, apply its cost and transition before the recurrence and average over `b_h`. This produces `U(a)`. At the prescan exit, `v=z=0` because fresh evidence is empty, so the initialization is exact and especially simple.

At later public histories, the old retained pair is truly valid under particle `h` with probability

`q_pair(h) = [sum_i eta_old(i|h)] * [sum_j eta_new(j|h)]`.

The immediately preceding near frame is valid with probability `q_near(h)=sum_i eta_last(i|h)`. The slot sums represent an oracle that sees true identity, rather than the public rule that must select one slot. Under the current emitter model these two events use distinct acquisition clocks when both are relevant; their conditional independence can be checked from the pair bookkeeping. Average the oracle state values over these latent flags. If that audit is not completed, initializing any observed candidate pair as valid is a looser but safe upper relaxation; do not call it exact.

### Tighten the envelope for all three motion caps

The cap-free oracle can be loose. A small nonnegative Lagrange-multiplier bank tightens it while keeping the same small dynamic program. Let `r` be the three remaining resources and `d(n,a)` the edge-resource vector. For `lambda >= 0`, replace each move cost inside the oracle with

`c_lambda(n,a) = c(n,a) + lambda dot d(n,a)`.

For any feasible complete path, `lambda dot (r - total_resources) >= 0`. Thus the forced-first-action value

`U_lambda(a) = lambda dot r - c(n,a) - lambda dot d(n,a) + E_{h,D,old_flags} V_h^lambda(after a)`

upper-bounds its feasible public-policy value. The minimum over a fixed finite multiplier bank remains an upper bound:

`U(a) = min_lambda U_lambda(a)`.

Use the same multiplier bank for every policy; including zero recovers the cap-free envelope. The bound remains conservative when it gives exact fork position, association and persistent-pair retention to the oracle. It is not a prediction that the public policy can achieve that value.

### Lower bound and safe pruning

The exact current public `best_stop` value is a lower bound `L`. A complete feasible **public** continuation provides another lower bound if its expectation is evaluated exactly. A Monte Carlo estimate alone is not a certified lower bound; use a one-sided confidence lower bound if making a probabilistic pruning guarantee.

Discard a first move only when `U(a) < L`. Every discarded move is then incapable of improving on the available public policy under the model. No advantage gate is fitted to outcomes. This can replace heuristic top-ten root filtering and expose long-horizon diagnostic moves that a one-step score would omit.

With only 12 rollout samples, distribution-free confidence bounds will usually be wide. In that regime, use the envelope to allocate computation and report the result as approximate search; do not advertise certified near-optimality. Exact-stop pruning remains valid even when other ranking scores are sampled.

An informative final diagnostic is the interval between a feasible public-policy value and this oracle envelope. A small gap says there is little room for B in the current sensor/action/task model. A large gap does not establish attainable B benefit: it can reflect unresolved topology, cap relaxation, irreversible pair bookkeeping, or weak search. Report which relaxation dominates on a few development cases.

The oracle DP, Lagrangian relaxation, policy rollout and distillation are standard tools. The task-specific element is the explicit coupling of graph/identity uncertainty to a fresh, physically feasible inspection certificate. A useful contribution could be an efficient, audited implementation of that coupling with evidence of benefit; the present formulas alone do not establish theoretical novelty.

## Development interpretation and next decision

The supplied 96-case development counts are:

| Policy | All correct completion | Positive Y completion |
|---|---:|---:|
| Two-step live MPC | 58/96 | 14/48 |
| Complete greedy-tail rollout | 62/96 | 16/48 |
| Fisher-augmented greedy | 57/96 | 14/48 |
| Nearest-return MPC | 46/96 | 0/48 |
| Greedy belief policy | 56/96 | 13/48 |

These counts support a strong reason to retain the all-return ablation: the nearest packet loses the evidence needed for fresh positive inspection in this development cohort. They do **not** isolate a novel B effect, and Fisher is currently a bonus to greedy, not a separate matched-depth MPC objective. Rollout improves only two positive completions over MPC; paired discordances, wrong outcomes and realized resource differences are required to interpret the four-case pooled difference. Development counts are not confirmation evidence.

The corrected Bayes terminal decoder and aligned checkpoint objective should precede the frozen primary comparison. A modest BC-initialized PPO experiment with cloning and no-memory controls is justified as an exploration repair. If it only matches its public teacher, the result is policy distillation, potentially useful for runtime. If it improves completion at higher physical cost, report that tradeoff. If neither long public search nor the oracle-envelope gap reveals useful opportunity, increasing training again is not the next justified step.

The most defensible claim from a positive primary result would remain: within this finite-family centroid simulator and checked motion bank, a frozen policy improves correctly identified two-plane fresh offshoot inspection at the measured resources after a shared uncertain prescan. Claims about general anatomical graph discovery, real U-Net failure modes, full branch coverage, continuous contact safety or clinical performance require additional evidence.
