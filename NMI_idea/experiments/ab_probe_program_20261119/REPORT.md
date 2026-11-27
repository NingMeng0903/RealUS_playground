# Paired evaluation of B training data and two-probe control

This study uses the `b7a2121e76c896148b4bdb68f97651436e049497`
upstream execution stack. Policies run the same post-atlas probability
update and candidate retention, causal in-segment ultrasound feedback,
trajectory generator and real-observation-replanning MPC. Execution
starts from one common measured checkpoint for each simulated case.
The PROTOCOL was frozen before the holdouts. The older evidence-B
checkpoint from `3f2db45` is also replayed on both new splits.

## What was compared

There are 128 independent main-test scenes and 48 independent drift-test
scenes, each with eight time-indexed perturbation tapes. The two task
families are connection ambiguity and non-fork identity recovery between
adjacent vessels. Each is tested under ordinary segmentation perturbation
and under a forced loss burst. With actual feedback, different motions
produce different image packets despite sharing the random tapes.

The data experiment adds 640 records to the same 8,122 historical
records for each one-probe student. Random, evidence, and outcome
contrast change the four nominated alternatives per root. Every root
also supplies the MPC anchor. Contrast screening found a good/bad
candidate pair in 96 of 128 starts. When evaluated with the independent
label tapes, 78 pairs preserved the ordering, 12 tied, and 6 reversed.
Contrast/evidence nominations overlapped on 425 of 512 requests.
Matched labeled dose does not equate offline compute. Outcome
contrast made 3,996 additional closed-loop screening replays.

The program experiment trains the same regression architecture at a
second decision point, using 192 actually reached simulation histories
and 1,701 extra records. It has an explicit choice to return control to
the same MPC. The sole difference between the two-probe program and the
child-trained control is the availability of second-stage outcome
labels for choosing the first command. Conditional control and the
outcome-contrast one-probe control have the same 640 root examples;
conditional control also uses the successor data. Repeated B tests the
effect of another decision with the same one-probe learner.
The precommitted and shuffled second decisions keep the same first
program command. They test dependence on a combination of actual
measurements, reachable exit, and remaining resources.

The fixed single- and two-probe choices come from 262 sequences
compared on 32 separate development scenes. The selected recipes are
`back_right` and `revisit → grid_14` for connection, and `grid_02`
and `cem_4 → grid_02` for the non-fork task. The fixed recipes
fall back to the same MPC when a requested move is unavailable.

## Decision model

Let $H_1$ be the history at the *measured*, feedback-driven end of a
first command $a_0$. The child's action value uses
`current probe → common MPC`, with immediate handoff as an action.
With the child frozen, program labels average
`first probe → child's H1-dependent choice → common MPC`
over public-posterior simulations. All paired quantities retain
completion, final identity errors, new identity mistakes, TCP travel,
turn, and elapsed time. The regression target is the existing
corrected-completion score. The acceptance metric also verifies that
a newly captured, correct image remains in the delivered witness set.
The strict centered score is reported separately.

In an exact two-world example, let a missed observation occur with
probability 0.20 for a target on the left and 0.95 for a target on the
right. With an equal prior and only one subsequent look, posterior
conditioning can choose left after a hit and right after a miss.
That achieves 0.875 rather than 0.5 for a fixed look. Equal observation
likelihoods reduce both to 0.5. The check establishes the mechanism in
the example, separate from the measured outcome of the learned program.

## Execution audit

The prior paired labels were recovered byte-for-value for the replayed
outcome vectors. On eight checkpoints, immediate handoff recovered the
same fully replanning MPC result as the previous execution protocol.
Replacing the oracle identity fields in the logs while leaving public
histories unchanged left the learned second-stage features and decisions
unchanged in all eight checks. The two-probe permission is bounded by
the same absolute frame allowance regardless of early handoff. The
truth used for synthetic imaging and evaluation never chooses a control
request. A separate SOL Max review checked the dose, public input,
matching and accounting before execution.

## Held-out execution

| Policy | Main correct / 1,024 | Drift correct / 384 | Main final / interim wrong | Main TCP mm | Main turn ° | Main scan s | Median main selector CPU ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full MPC | 793 | 285 | 40 / 119 | 38.29 | 7.01 | 5.56 | 129.2 |
| Best fixed single | 880 | 307 | 35 / 174 | 46.05 | 9.14 | 6.48 | — |
| Best fixed sequence | 861 | 286 | 39 / 154 | 45.87 | 8.41 | 6.57 | — |
| Original frozen B | 900 | 328 | 48 / 176 | 45.39 | 8.92 | 6.26 | 506.8 |
| Released evidence B, reused without training | 912 | 329 | 30 / 154 | 44.90 | 8.81 | 6.25 | — |
| One probe, random nominations | 889 | 328 | 35 / 161 | 45.46 | 8.83 | 6.32 | 506.3 |
| One probe, evidence nominations | 897 | 330 | 41 / 165 | 45.86 | 9.02 | 6.33 | 506.2 |
| One probe, outcome-contrast nominations | 923 | 345 | 39 / 161 | 45.80 | 8.96 | 6.21 | 506.2 |
| Same one-probe B at both decisions | 884 | 318 | 36 / 174 | 49.25 | 9.80 | 6.73 | 696.9 |
| First contrast B + trained child | 921 | 338 | 36 / 171 | 48.37 | 9.52 | 6.58 | 696.7 |
| Two-probe program B | 908 | 321 | 39 / 181 | 47.72 | 9.38 | 6.56 | 682.8 |
| Same first, precommitted second | 934 | 337 | 38 / 166 | 47.06 | 9.30 | 6.44 | 7559.1 |
| Same first, shuffled second | 905 | 322 | 40 / 177 | 47.62 | 9.34 | 6.55 | — |
| One-probe online rollout | 875 | 295 | 27 / 153 | 41.45 | 7.88 | 5.96 | 14785.7 |
| Two-probe online rollout | 885 | 311 | 32 / 167 | 43.72 | 8.32 | 6.17 | 35113.9 |

Main scenes, correct completion with new retained image per task.

| Policy | Connection | Non-fork |
|---|---:|---:|
| Full MPC | 403/512 | 390/512 |
| Best fixed single | 444/512 | 436/512 |
| Best fixed sequence | 428/512 | 433/512 |
| Original frozen B | 428/512 | 472/512 |
| One probe, random nominations | 419/512 | 470/512 |
| One probe, evidence nominations | 430/512 | 467/512 |
| One probe, outcome-contrast nominations | 449/512 | 474/512 |
| Same one-probe B at both decisions | 427/512 | 457/512 |
| First contrast B + trained child | 452/512 | 469/512 |
| Two-probe program B | 439/512 | 469/512 |
| Same first, precommitted second | 461/512 | 473/512 |
| Same first, shuffled second | 437/512 | 468/512 |
| One-probe online rollout | 440/512 | 435/512 |
| Two-probe online rollout | 438/512 | 447/512 |

Independent drift scenes, correct completion with new retained image per task.

| Policy | Connection | Non-fork |
|---|---:|---:|
| Full MPC | 139/192 | 146/192 |
| Best fixed single | 150/192 | 157/192 |
| Best fixed sequence | 134/192 | 152/192 |
| Original frozen B | 148/192 | 180/192 |
| One probe, random nominations | 148/192 | 180/192 |
| One probe, evidence nominations | 150/192 | 180/192 |
| One probe, outcome-contrast nominations | 163/192 | 182/192 |
| Same one-probe B at both decisions | 146/192 | 172/192 |
| First contrast B + trained child | 162/192 | 176/192 |
| Two-probe program B | 146/192 | 175/192 |
| Same first, precommitted second | 154/192 | 183/192 |
| Same first, shuffled second | 147/192 | 175/192 |
| One-probe online rollout | 141/192 | 154/192 |
| Two-probe online rollout | 154/192 | 157/192 |

Selected paired differences in percentage points with case-clustered 95% intervals.

| Contrast | Main | Drift |
|---|---:|---:|
| One probe, evidence nominations vs One probe, random nominations | +0.78 [-0.59, +2.64] | +0.52 [-5.47, +7.03] |
| One probe, outcome-contrast nominations vs One probe, random nominations | +3.32 [+0.78, +6.25] | +4.43 [+0.00, +10.16] |
| One probe, outcome-contrast nominations vs One probe, evidence nominations | +2.54 [+0.20, +5.27] | +3.91 [-0.26, +9.38] |
| One probe, outcome-contrast nominations vs Original frozen B | +2.25 [+0.20, +4.79] | +4.43 [+1.04, +8.59] |
| One probe, outcome-contrast nominations vs Full MPC | +12.70 [+7.42, +18.26] | +15.62 [+5.99, +26.04] |
| One probe, outcome-contrast nominations vs Best fixed single | +4.20 [+0.20, +8.69] | +9.90 [+2.86, +17.97] |
| Same one-probe B at both decisions vs One probe, outcome-contrast nominations | -3.81 [-7.03, -0.78] | -7.03 [-13.28, -1.82] |
| First contrast B + trained child vs Same one-probe B at both decisions | +3.61 [+0.98, +6.45] | +5.21 [+1.56, +9.90] |
| Two-probe program B vs First contrast B + trained child | -1.27 [-3.61, +0.78] | -4.43 [-10.42, +0.26] |
| Two-probe program B vs One probe, outcome-contrast nominations | -1.46 [-4.20, +1.07] | -6.25 [-12.76, -1.04] |
| Two-probe program B vs Full MPC | +11.23 [+5.76, +16.99] | +9.38 [+1.04, +18.23] |
| Two-probe program B vs Best fixed single | +2.73 [-1.66, +7.52] | +3.65 [-4.95, +12.24] |
| Two-probe program B vs Best fixed sequence | +4.59 [+0.39, +9.28] | +9.11 [+1.56, +17.71] |
| Two-probe program B vs Same first, precommitted second | -2.54 [-5.18, -0.29] | -4.17 [-9.11, -0.26] |
| Two-probe program B vs Same first, shuffled second | +0.29 [-0.78, +1.46] | -0.26 [-1.56, +1.04] |
| Two-probe online rollout vs One-probe online rollout | +0.98 [-0.29, +2.54] | +4.17 [+0.78, +8.33] |

## Interpretation

**The data-only comparison supports outcome-contrast supervision on the
main set.** It added 34 successes over equal-dose random selection with
nearly unchanged actual motion and the same 161 interim mistakes. It
logged four more final misidentifications, with a paired interval
spanning zero. The extra successes over random selection concentrated in
connection ambiguity (449 vs 419). The non-fork difference was 474 vs 470.
The drift point estimate over random stayed positive, but its interval
included zero. Against the previously released evidence B, the new
outcome-contrast B gained 11/1,024 in the main set (95% interval −1.46 to
+4.01 percentage points) and 16/384 with drift (−0.26 to +9.64).
These comparisons do not establish a reliable lead over that checkpoint.
The main-set improvement over MPC came with +7.51 mm, +1.95° and +0.65 s
on average and 42 more runs with an interim mistaken confirmation.

**The two-probe program learned to branch and return control.** In the
main set it used 0, 1 and 2 explicit probes in 152, 423 and 449 runs.
It chose handoff despite another feasible command in 407 runs after
the first probe. For 37 roots there was a pair of tapes with exactly
matching first-probe exit position, velocity, time, and remaining
travel within 1e-7 in native units. At 18 of those roots the requested
next action changed. The test confirms actual history-dependent
decisions even at matched exits. Those decisions did not improve
completion compared with the same-first precommitted choice, which
finished 934 vs 908 in the main set and 337 vs 321 with drift.
The program had 16 matched wins and 42 losses against precommitment
in the main set. On the 247 trials where the program took a second
probe and precommitment returned to MPC, it won 3 and lost 15.
Shuffling the second action across same-root tapes gave comparable
outcomes. The shuffled test preserves scene-level information and
returns to MPC on an unavailable request. There were seven such
unexecuted second commands in the same-first fixed control and one
in the shuffle. For the separate fixed sequence arm, a programmed
second request is also logged as unavailable when its preceding fixed
command was unavailable and the whole sequence was skipped.

**Success over the full MPC does not establish a conditional-program
advantage.** Program B also spent 9.43 mm and 1.00 s more on average
with 62 additional main-set interim mistakes. It did not separate
reliably from the strongest tested fixed single probe. Training the
child with actual successor histories helped over repeating the
one-probe model by +3.61 points on the main split and +5.21 points on
drift. Relabeling the first choice for the joint program did not
improve over that trained child control. The matched online rollout
showed a modest two-probe gain with drift, at substantially higher
decision cost and lower absolute success than the learned one-probe
outcome-contrast policy. These controls favor the latter for this
simulation and bound the claim about two-segment scheduling.

All execution stayed within 67 mm and 8 s, with maximum numerical
handoff error 1.78e-15. There is no claim of execution-cost equivalence
versus MPC. Selector CPU is separate from scan time and the simulator
does not advance in response to computation delay. The time-varying
motion cohort uses a prescribed smooth displacement and the same
estimator. This evidence comes from synthetic observation packets,
shared trajectory families and a leg model. It establishes neither
real-ultrasound robustness nor generalization to anatomy with different
imaging physics. New tilt or free-form motion skills were outside the
action space.

## Reproduction artifacts

- [Frozen design and commands](PROTOCOL.md)
- [Frozen hashes and seed partitions](results/main/frozen.json)
- [Results, paired intervals, risk and execution metrics](results/main/summary.json)
- [Earlier evidence B on the same tapes](results/main/archived_summary.json)
- [Matched-exit example cases and action switches](results/main/branch_diagnostics.json)
- [Main case outcomes](results/main/test.jsonl.gz) and [drift outcomes](results/main/drift.jsonl.gz)

To replay the released evidence B and the matched-exit analysis after the
core run, execute the following from this experiment directory.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python archived_baseline.py record
python archived_baseline.py summary
python branch_diagnostics.py
```
