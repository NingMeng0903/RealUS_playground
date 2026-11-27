# Executable-skill stress test: pre-result design and audit contract

Prepared 2026-11-11. This is a mechanism experiment, not validation of real ultrasound, a trained embodied model, SMPL-X registration, native reachability, or continuous probe motion. Design and independent-review ownership: Astra; implementation is owned by the root agent and its workers.

## What the prior result establishes

The prior native conservative bound selected A in 45/45 comparisons. The soft graph test scored complete programs but executed only their first primitive: soft_exit achieved 186/200 recoverable completions, 12 censored episodes, and 297 returns, against A_task 198/200, zero censored, and four returns. Executing the previously selected suffix repaired completeness to 198/200, but still required 1,084 acquisitions and 56 returns versus A_task's 808 acquisitions and four returns. Thus suffix fidelity is a necessary repair; existing evidence does not establish superiority over A.

## Primary question and non-negotiable invariants

Does predicting a diagnostic skill's actual exit, and using its public observation to choose an executable remaining task, change a decision and improve task acquisition/rework performance relative to nominal-exit prediction and strong task-directed comparators?

All arms receive the same legal actions, public packet, full posterior, acceptance test, stored reference, valid coverage, route finder, and option-progress/release mechanism. A missing or clutter packet is evidence, not a reset of reference, coverage, task progress, or physical pose. The selected feedback program must be the executed program, including legal continuation, unless a recorded shared release rule applies. A release compares remaining suffixes from the same current state; it cannot compare a fresh long program with a one-step remainder.

A false public acceptance is an absorbing wrong stop. Rejected packets remain in the posterior. Administrative checkpoints do not define failure or remove access. All graph edges remain reversible. If all reachable trials eventually succeed, that is expected: the scientific endpoint is acquisition burden/returns/rework, not an invented terminal-success gap.

## Minimal runnable world

Reuse the separately validated eight-state public world kernel from the preceding study without importing a policy: hidden h in {L,R}, permanent in {false,true}, present in {false,true}. Retain its likelihoods, transition probabilities, shared initial prior and initial true lost condition. This avoids confounding a controller repair with a new favorable observation model.

Graph: C--EL--TL and C--ER--TR, with optional EL--ER. Every move acquires one packet at its destination. E/T allow in-place peek and scan. Only valid fresh T packets can complete bin1. Start at EL, pending TL, with bin0/reference already valid; mirror the start if computationally inexpensive. Primary study preserves bin0 on every move. It must not silently turn geometric travel into deletion of acquired evidence.

Structural fixtures are the previous predefined target side x public shadow side x narrow/wide eight recoverable cases, two no-shadow easy controls, and two permanent-negative controls. Do not select fixture/seeds after seeing policy wins. A structural oracle can use hidden truth only to verify a legal positive-probability completion path, or certify permanent absence. The oracle is not passed to the policy.

Contact-invalidating geometry, if desired, is a separate supplementary fixture with an explicit public rule and reacquisition action. Explain what physical validity is lost and what stored evidence survives. Never treat an arbitrary one-way graph or deadline as an unavoidable physical loss. Do not mix such a fixture into the primary aggregate after inspecting results.

## A short executable feedback skill

At a decision state consider every legal first primitive a. Acquire its packet y, update the full public posterior, and stop immediately if that packet qualifies. Otherwise choose a terminal goal k(y) in {TL,TR}; walk a legal shortest route from the actual exit to that terminal, accepting any qualifying fresh T packet along the route, then obtain at most one extra scan at that terminal. Arrival is the first terminal attempt; an eligible first primitive already at that terminal counts as its first attempt. Do not silently add a third scan. No label such as probe versus scan can veto an otherwise valid fresh packet.

The routing response map is part of the selected program and uses only y, public exit, and posterior. At an exhausted unsuccessful program the task continues with a new program; exhaustion is not task failure. For the cleanest proof test use the same retained-program executive for all arms, releasing only on public stop, exhaustion, or illegal continuation. A supplementary adaptive-release test may permit a competing program to dominate the exact remaining suffix in success per acquisition without increasing false-stop probability, with tol=1e-12 and ties retaining the current program. Its policy and results must be separate from the retained-program proof.

## Score and exact computation

For every candidate compute C=P(correct public stop), W=P(wrong public stop), U=P(program exhausts unfinished), and N=E(executed acquisitions until stop or exhaustion). They sum C+W+U=1. A defensible efficiency score for this new frozen study is Q=C/N; report W alongside Q. This is a finite-skill opportunity score, not an all-task completion probability or an execution deadline. If the implementer chooses the earlier raw-C score instead, freeze and disclose that choice before outcome sampling; do not choose the scoring rule from observed B wins.

Exact enumerate the first packet and the short route's public emissions, preserving the shared h/permanent state and the per-packet present transition. Full posterior and acceptance must update at every packet; correct/wrong mass is integrated using the model's hidden target contribution only inside evaluation. Do not independently multiply marginal scan-success rates or ignore a first false stop.

Cache exact continuation statistics for each (belief after y, actual/nominal route start, goal, already-used terminal attempts). With five possible first packets, there are only 2^5=32 binary goal-response maps per primitive. Aggregate the C/W/N statistics for every map and choose the best whole-program ratio. Independently maximizing C_y/N_y for each y need not maximize aggregate C/N and is not an exact adaptive-program optimum.

Where exact enumeration is expensive, the implementation may use a predeclared 64-draw screen and 512-draw paired refinement for close candidates, followed by exact valuation of every reported activated witness and the selected rival. Candidate estimates use common random numbers; actual world noise is a separate family. Log any unresolved decision interval, not just its point estimate.

## Comparators and attribution

* A_existing: retain the old full-belief task-completion/fallback controller as an external diagnostic reference. If its executive is altered, rename it and do not imply numerical equivalence to the prior 198/200 result.
* A_direct: exact best direct terminal route with the same acceptance and terminal-attempt accounting. It has full posterior and legal actual-state routing.
* A_fixed_goal: exact best over the same first primitives and terminal goals, with the terminal goal selected before the first packet. It has identical probe access and filters every packet. This is stronger than comparing B only to a MAP or reset-on-miss policy.
* B_actual: choose the best executable packet-conditioned goal map using the actual post-primitive state for prediction.
* B_nominal: identical observation integration and response-map capacity, but use nominal C only inside continuation prediction. Execution always follows a legal route from the actual state; record predicted and executed routes separately. This mismatch is the intended exit-model ablation, never a claim that an illegal nominal route was executed.
* B_MAP: optionally collapse present/lost only for candidate valuation; keep actual filtering and acceptance full. This tests selection uncertainty separately from memory retention.

The exact arm list must be frozen before sampling. The essential attribution is B_actual versus B_nominal. B_actual versus A_fixed_goal also changes feedback-routing capacity, so do not attribute that entire contrast to actual-exit conditioning. If a no-lookahead control is mathematically identical to A_direct, identical outputs are a useful implementation check, not independent corroboration.

## Predeclared evidence gates

1. Legality/fidelity: no illegal motion; every executed primitive appears in the selected program's legal continuation, or a shared reasoned release record. Coverage/reference survive rejected packets. Paired arms share world likelihood and random-number family.
2. Activation: at least one reachable public history of positive probability changes the chosen program under actual versus nominal exit prediction. Save both candidate value tables, predicted exits, response maps, and actual executed continuation. If no such history exists, this experiment did not activate the proposed mechanism.
3. Correctness: on permanent negatives there can be no true target completion; false stops are errors, never recovery. An oracle proves positive-probability recovery for each hard positive fixture. Easy controls must not be presented as proof of B's necessity.
4. Outcome: report paired C/W/unresolved at every common administrative checkpoint and continue recoverable unresolved episodes. Report acquisitions, returns, switches, and option releases, with paired differences and uncertainty. Do not compare mean completed-only acquisition counts when completion sets differ; use paired commonly completed episodes plus all-episode restricted counts and censoring bounds.
5. Claim threshold: claim a synthetic mechanism benefit only if actual-exit decisions are activated, continuation is faithful, paired acquisition/rework improves against strong A and nominal prediction, and error/completion does not materially worsen. A tradeoff, tie, unactivated mechanism, or loss is an explicit negative result. No synthetic result upgrades a native, learned-model, SMPL-X, or smooth-control claim.

## Independent review queue

Pending implementation: inspect hidden-data boundaries, exact C/W/N recursion, terminal-attempt accounting, actual-exit versus nominal-only prediction, response-map optimization, commitment/release equivalence, common random coupling, complete paired records, checkpoint continuation, and source hashes. Review the complete frozen cohort rather than showcase traces only. Add audit findings and final conclusion below once the implementation exists.

## Completed primary audit, study_721621

The frozen primary used 24 mirrored structural cases, 32 replicates each, seven arms, and checkpoints 8/16/32/64. The no-lookahead control is exactly A_task. A_legacy retains its historical one-step executive and is an external comparator. The primary score is the frozen C/N ratio and all six matched arms retain their selected program until stop or exhaustion.

One pre-run issue was found and repaired: the nominal-exit ablation initially discarded a prefix terminal acquisition when resetting only its predicted route origin to C. It could therefore score three terminal attempts while actual execution used two. The repaired implementation carries that attempt counter independently of the hypothesized route origin; the source manifest captures the repair.

Independent complete-record checks passed:

* 5,376 expected episode records with no missing or duplicate arm/case/replicate keys.
* Source hashes match the frozen manifest.
* All 75,988 recorded transitions, packets, hidden target associations, full public posteriors, acceptance decisions, and terminal outcomes replay from the declared paired random streams.
* All 34,624 saved programs and their actual continuations are legal and faithful, including every stored suffix after missing packets and every checkpoint cursor.
* A_task and B_no_lookahead are identical on all 768 paired episodes, apart from the arm label.
* Independent normalized scalar public-history enumeration for 160 selected programs spanning all six matched arms agrees with stored C/W/N to maximum absolute error 2.665e-15. This includes nominal prediction and MAP selection with full public acceptance.
* Hidden side/permanence is used only by the world/evaluator, not the planner. The acquisition checkpoint is not available to planning. No permanent-negative trial contains a true target completion.

Machine-readable audit notes: ASTRA_RAW_AUDIT.json, ASTRA_PAIRED_AUDIT.json, and ASTRA_EXACT_VALUE_AUDIT.json. The complete raw input is study_721621/EPISODES.jsonl.

For the 640 recoverable pairs per arm, all outcomes resolved by the administrative checkpoint:

| Arm | Correct | Wrong stop | Acquisitions | Returns | Retreats |
|---|---:|---:|---:|---:|---:|
| A_task | 631 | 9 | 3059 | 70 | 181 |
| A_fixed_goal | 630 | 10 | 3049 | 62 | 177 |
| B_actual | 632 | 8 | 2819 | 81 | 155 |
| B_nominal | 632 | 8 | 3351 | 164 | 254 |
| B_no_lookahead | 631 | 9 | 3059 | 70 | 181 |
| B_MAP | 633 | 7 | 2883 | 87 | 169 |
| A_legacy | 630 | 10 | 2792 | 30 | 139 |

B_actual's paired mean acquisition difference is -0.375 versus A_task, -0.359 versus A_fixed_goal, -0.831 versus B_nominal, and +0.042 versus A_legacy. Resampling the 20 enumerated recoverable structural-case means gives descriptive 95% intervals [-0.917,+0.166], [-0.891,+0.180], [-1.409,-0.313], and [-0.605,+0.681], respectively. These are structural-case sensitivity intervals, not clinical/population generalization. Common-correct-only comparisons preserve the same substantive result; versus legacy there are still nine additional acquisitions and 50 additional returns. Outcome discordances are too small to establish superiority in correctness (B_actual versus A_task 4 wins/3 losses, fixed-goal and legacy 4/2, nominal 3/3, MAP 0/1).

Primary conclusion: faithful executable continuation and exact valuation are demonstrated; actual-exit conditioning improves this synthetic controller over its nominal-exit ablation. Broad superiority over a genuinely strong A, lower returns than A, and a soft-over-MAP advantage are not established. Permanent negatives remain censored and must not be pooled into acquisition-efficiency wins, because earlier false stops can artificially shorten such runs. Stored bin0/reference preservation is a toy invariant. The physical exit is deterministic action.dest; uncertain anatomical registration, physical contact validity, native reachability, and continuous smoothing remain separate claims.

## Supplementary shared-release diagnostic: frozen before its outcomes

This is a separate contrast under a new paired noise seed; do not pool it with study_721621. Run retained and shared-release versions on identical new-seed episode streams for every arm. Preserve the primary source files/hashes by implementing the supplementary executive separately.

After at least one real suffix primitive is executed, a new public posterior may justify switching the remaining terminal route. Do not override the first packet's newly selected response map immediately using the same packet. Compare the exact remaining saved suffix with the alternative terminal route from the identical current pose/full posterior. Carry per-goal terminal acquisition counters across every release, counting any T packet (including prefix and arrival). Exclude a goal that already has two terminal acquisitions within that program. No new probe, fresh terminal-attempt budget, or posterior/mode reset is allowed during release.

Release iff Q_alternative > Q_remaining + 1e-12 and W_alternative <= W_remaining + 1e-12; ties retain the program. Reset attempt counters only when the program genuinely exhausts and a new program starts. Log the triggering public packet, counters, old/new routes, and both exact C/W/N/Q values. All six matched arms share this release policy and full posterior. A_legacy remains external and unchanged. A_fixed_goal is fixed at its initial choice but shares later evidence-dependent rerouting, which strengthens the comparator.

The supplementary question is whether evidence-dependent release improves faithful retained execution without recreating the previous reset-and-thrash problem. A mixed or negative outcome is retained. This does not upgrade the finite one-observation feedback-skill proof into an unrestricted embodied controller.

## Completed supplementary audit, release_721622

The supplementary study used 16 paired replicates per mirrored fixture, with retained and released executives on the same seed-721622 streams. Primary source hashes remained unchanged. All 2,688 records in each executive variant and all 75,970 events passed independent world/public-posterior/continuation replay. All 14,133 eligible release-gate decisions were recomputed, including non-releases; all 284 actual releases were additionally checked by independent scalar public-history enumeration of the old and proposed suffix. Terminal counts never exceeded two per goal per program. A_task/no-lookahead identity and unchanged external legacy behavior were verified. Audit data: ASTRA_RELEASE_AUDIT.json.

For B_actual's 320 recoverable pairs, four releases left correctness unchanged at 317 correct / three wrong, but added seven acquisitions and one return (1,341 to 1,348 acquisitions; 28 to 29 returns). A_fixed_goal's two releases added eight acquisitions with no outcome/return change. A_task's releases saved 56 acquisitions but added 12 returns and one wrong stop. Nominal's releases saved 30 acquisitions and 14 returns but added one wrong stop. No recoverable episode remained censored. This diagnostic does not support claiming that shared evidence-dependent release improves B_actual; local exact suffix dominance is not a guarantee of better realized whole-task outcomes.

Initial program scores in this supplementary executive still describe retained execution. Subsequent releases are explicitly recorded replanning overrides. Thus these scores are not an exact predicted value for the entire future released controller. The primary matched-program fidelity result remains separate.

## Independent witness and oracle checks

The recoverable-only witness file contains 167 distinct recorded B_actual public decision states. It reports 151 actual-versus-nominal executed-program differences and 115 first-action differences. Of 990 selected feedback programs on recoverable trials, 580 have more than one supported non-stopping goal branch. These are mechanism-activation counts, not independent population samples.

A concrete actual-exit witness is case 01-ER, replicate zero, at the initial public state. Both models select move_ER_to_TR. After a missing packet (public probability 0.655704), the actual model retains scan_TR; nominal-origin prediction instead chooses the opposite terminal route. Nominal predicted C/N is 0.211207, but evaluating its saved program at the real exit gives 0.147450; B_actual's faithful value is 0.201686. Both realized episodes finish correctly at acquisition six, so the example establishes a decision and value discrepancy, not a realized per-episode speed win.

A concrete evidence-preserving recovery history is case 08-ER, replicate zero, frame five. Its observed prefix is peek_ER:4, move_ER_to_TR:missing, scan_TR:missing, peek_TR:missing, scan_TR:missing; the public history has positive probability 0.003197619. The initial E packet is actually clutter yet produces public target probability 0.983579; because E does not cover bin1 it is not accepted as completion. At the later common public state, B_actual chooses move_TR_to_ER with a packet-conditioned goal map (C/N 0.199418), versus fixed-goal A's peek_TR program (0.171393). The realized continuation is move_TR_to_ER:missing, move_ER_to_EL:4, move_EL_to_TL:missing, scan_TL:4. The last packet correctly completes with public target probability 0.997043. The missed terminal arrival retains its saved final scan and all accumulated evidence. This is an inspected conditional trace, not a separate sample or proof of average superiority.

Independent structural-oracle calculation finds a legal direct target path with positive correct-completion probability in all 20 recoverable mirrored fixtures; the minimum conditional probability from the declared true lost state is 0.767546. Across 256 permanent-state/action checks, transition followed by target-emission mass is exactly zero. Consequently, permanent-negative correct-completion upper bounds are exactly zero, even if a generic censoring summary prints the vacuous interval [0, unresolved/n]. Their unresolved status records a non-stopping policy, not a possible future correct completion. Oracle details: ASTRA_ORACLE_AUDIT.json.

The separate native witness records two selected consecutive native/QP segments with complete exit-to-next-entry state continuity and a fresh deterministic replay. Its downstream set of 14 proposal actions is explicitly predicted_unknown/native_checked=false; those proposals are not 14 independently certified executable alternatives. This remains a selected two-transition native witness, separate from the graph study and from real ultrasound/robot validation.

## Final joint conclusion

The experiment now evaluates and executes a real finite feedback skill, preserves evidence and saved continuation through missing/clutter, activates actual-exit-conditioned decisions, and checks the full paired cohort without task deadlines. Actual-exit conditioning improves acquisition burden and returns relative to nominal-exit prediction in this synthetic setting. It has not established broad superiority over the strongest A, a soft-uncertainty advantage over MAP, or a benefit from additional shared release. Continuous probe-to-scan control, calibrated uncertain SMPL-X anatomy, actual contact/coverage invalidation, and native task-level superiority remain unvalidated by this experiment. These limits are results of the predefined checks, not reasons to change the cohort or objective after seeing outcomes.
