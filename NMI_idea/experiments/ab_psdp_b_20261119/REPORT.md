# PSDP-style B with separate continuation supervision

The policy was trained backward. Four first-motion roll-ins on each of 96
scenes produced 384 actual reached histories in the streaming simulator.
The frozen successor policy used 3,384 legal-action labels, each estimated
with eight posterior replays. It was frozen before the new 128 roots were
labeled. Each of 1,152 first requests was executed once per tape, followed
by paired direct-MPC and adaptive-child continuations, from the same
measured exit and with the same absolute resource deadline. Of those
paired root labels, 376 had a positive and 94 a negative estimated change
in correct completion when followed by the frozen child. The training
nomination stage used 4,016 additional screening replays.

At execution, the frozen 923 direct head and the separately trained
conditional increment offer zero, one, and up to two probes. Conditional
continuations feed the newly received observations to the frozen child,
which selects a feasible motion or releases to complete MPC. The
existing MPC ran with streaming feedback, evidence updates and replanning
in every arm. All published models were replayed on the same new streams.
The finite collection of candidates, two-probe cap, simulated segmentation
and inherited error/cost ranking define the scope of the result.

## Audit before holdout

The independent SOL Max implementation review identified a normalization
issue for the MPC first motion. The known direct-MPC increment of that
motion is exactly zero. The root policy now sets it to zero before
adding the separately predicted adaptive increment. As a result, an MPC
first motion followed by a child decision can be chosen based on its
adaptive increment, with immediate full MPC still an explicit separate
candidate. The equal-dose Q1/Q2 controls use the same continuation menu.
Both fixes preceded the held-out evaluation. The frozen child weight
digest was reproduced from the same data after updating the runtime code
and remained identical to the digest bound to the original root labels.

The root label audit compared all 128 sampled MPC anchors, and 3,120 paired
labels where the child chose immediate MPC. Both matched the actual direct
continuation. Eight real simulation-prefix cases also passed full MPC
equivalence, zero-increment single-policy recovery, and the oracle-metadata
feature and action invariance checks. The two control policies with the
PSDP first motion and the legacy precommitment each preserve the recorded
first-probe physical exit and public history tape by tape.

See [design and commands](PROTOCOL.md), [frozen provenance](results/main/frozen.json),
and [root target design](results/main/target_design.json).

## New paired test results

128 main and 48 drift scenes each supply eight paired disturbance tapes.
Completion requires correct task completion and a newly acquired correct
witness retained in the final output. Misidentification counts are the
number of trials with final or interim wrong identity. Scan time is
simulated physical execution time. See the summary for drift costs,
task and loss strata, and paired intervals for all metrics.

| Policy | Main completion / final wrong / interim wrong | Drift completion / final wrong / interim wrong | Main travel mm | Main turn degrees | Main scan s | Selector CPU p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| Full MPC | 718 / 48 / 115 | 283 / 19 / 44 | 37.72 | 6.72 | 5.76 | 53.7 |
| Frozen fixed single probe | 827 / 24 / 184 | 320 / 19 / 66 | 45.84 | 9.07 | 6.64 | — |
| Single B from random nominations | 891 / 44 / 185 | 339 / 14 / 53 | 45.94 | 9.04 | 6.44 | 498.0 |
| Single B from evidence-teacher nominations | 897 / 28 / 169 | 330 / 15 / 54 | 45.88 | 8.97 | 6.40 | 497.7 |
| Previously published evidence B | 914 / 28 / 164 | 324 / 17 / 54 | 45.15 | 8.91 | 6.30 | 497.6 |
| Frozen 923 outcome-contrast single B | 916 / 42 / 174 | 323 / 14 / 51 | 46.01 | 9.10 | 6.31 | 497.7 |
| Published 908 two-probe B | 919 / 33 / 177 | 326 / 15 / 58 | 47.96 | 9.48 | 6.63 | 623.2 |
| Legacy first + precommitted second | 940 / 30 / 167 | 332 / 13 / 57 | 47.18 | 9.34 | 6.53 | 8782.4 |
| Frozen 923 first + new child | 934 / 42 / 170 | 332 / 12 / 56 | 48.92 | 9.67 | 6.77 | 630.6 |
| Matched direct labels, one probe | 907 / 33 / 172 | 322 / 17 / 57 | 45.53 | 8.78 | 6.37 | 498.3 |
| Matched direct labels + new child | 927 / 36 / 172 | 331 / 13 / 62 | 49.41 | 9.43 | 6.81 | 678.4 |
| Matched adaptive labels + new child | 931 / 45 / 169 | 318 / 15 / 54 | 48.89 | 9.42 | 6.88 | 663.9 |
| Matched mixed labels + new child | 930 / 39 / 171 | 328 / 12 / 58 | 50.21 | 9.78 | 6.89 | 680.3 |
| Optional-continuation B with mixed residual labels | 928 / 28 / 165 | 337 / 13 / 55 | 49.29 | 9.67 | 6.83 | 613.4 |
| Full PSDP-style B | 944 / 31 / 162 | 327 / 13 / 54 | 49.61 | 9.60 | 6.90 | 684.6 |
| PSDP first + direct MPC | 854 / 28 / 160 | 301 / 22 / 53 | 44.52 | 8.49 | 6.42 | 510.2 |
| PSDP first + precommitted second | 960 / 20 / 164 | 320 / 18 / 59 | 48.11 | 9.48 | 6.79 | 8736.9 |

Correct completion differences in percentage points with stratified paired scene-level 95% intervals.

| Policy compared against reference | Main | Drift |
|---|---:|---:|
| Full PSDP-style B vs Full MPC | +22.07 [+15.33, +28.91] | +11.46 [+2.34, +21.61] |
| Full PSDP-style B vs Single B from random nominations | +5.18 [+1.66, +9.08] | -3.12 [-7.29, +0.52] |
| Full PSDP-style B vs Single B from evidence-teacher nominations | +4.59 [+1.17, +8.30] | -0.78 [-5.21, +3.12] |
| Full PSDP-style B vs Frozen 923 outcome-contrast single B | +2.73 [-0.10, +5.76] | +1.04 [-3.39, +6.25] |
| Full PSDP-style B vs Published 908 two-probe B | +2.44 [-0.39, +5.47] | +0.26 [-4.17, +5.47] |
| Full PSDP-style B vs PSDP first + direct MPC | +8.79 [+4.49, +13.38] | +6.77 [+0.52, +14.58] |
| Full PSDP-style B vs PSDP first + precommitted second | -1.56 [-3.22, +0.00] | +1.82 [-1.56, +6.25] |
| Frozen 923 first + new child vs Frozen 923 outcome-contrast single B | +1.76 [+0.39, +3.22] | +2.34 [+0.52, +4.69] |
| Frozen 923 outcome-contrast single B vs Single B from random nominations | +2.44 [-0.10, +5.57] | -4.17 [-8.85, -0.52] |
| Full PSDP-style B vs Optional-continuation B with mixed residual labels | +1.56 [-0.88, +4.30] | -2.60 [-5.99, +0.52] |
| Matched adaptive labels + new child vs Matched direct labels + new child | +0.39 [-2.25, +2.93] | -3.39 [-7.55, -0.26] |
| Matched adaptive labels + new child vs Matched mixed labels + new child | +0.10 [-2.15, +1.95] | -2.60 [-7.03, +1.04] |

Task-specific correct completions. Each task has 512 main and 192 drift runs.

| Policy | Main connection | Main non-fork | Drift connection | Drift non-fork |
|---|---:|---:|---:|---:|
| Full MPC | 383 | 335 | 126 | 157 |
| Single B from random nominations | 418 | 473 | 160 | 179 |
| Single B from evidence-teacher nominations | 429 | 468 | 152 | 178 |
| Frozen 923 outcome-contrast single B | 438 | 478 | 146 | 177 |
| Published 908 two-probe B | 445 | 474 | 149 | 177 |
| Frozen 923 first + new child | 458 | 476 | 150 | 182 |
| Full PSDP-style B | 469 | 475 | 154 | 173 |
| PSDP first + precommitted second | 483 | 477 | 148 | 172 |

## Reading the comparisons

The full PSDP-style policy beat full MPC in both simulated splits. Its
main gain over the frozen single B was 28/1,024 and its drift gain was
4/384. Both paired intervals include zero. On the main cases it used
3.59 mm and 0.59 s more than the single B. The final wrong-identity
counts were 31 versus 42 and the interim counts 162 versus 174.
The paired final-error difference was −1.07 percentage points
[−3.22, +0.68]. Nearly all of the main completion difference came from
connection ambiguity (469 versus 438). The non-fork group was
475 versus 478. Relative to full MPC the main benefit used
11.89 mm and 1.14 s more, with 47 more trials carrying an interim
misidentification. The common hard execution bounds held.

The strongest evidence involving the new successor policy is from
keeping the frozen 923 first decision and comparing direct handoff
with calling the new child. Completion increased from 916 to 934 in
the main set and from 323 to 332 in drift. Both paired intervals are
positive. The extra execution averaged 2.90 mm and 0.47 s in the main
set. Main final errors were 42 in both variants.

Using the learned child helped complete the exact first plan selected
by PSDP. Forcing those same first plans to hand off immediately dropped
completion to 854 in the main set and 301 in drift. This control changes
both access to the second probe and how it is selected. The critical
comparison for conditioning is the same-first precommitted second
probe, which completed 960 main scans and 320 drift scans. On the main
split there were 11 more final wrong runs under PSDP than precommitment
(+1.07 percentage points [+0.20, +2.15]) and 1.50 mm more travel. On
drift the small completion difference reversed direction with a paired
interval spanning zero. Median selector CPU was about 0.69 s in the
main PSDP arm and 8.74 s in its precommitted comparison. Simulation
time is paused during selector computation.

Within the equal-row, equal-child comparison, consistent Q2 achieved
931 main completions, compared with 927 for Q1-first and 930 for mixed
labels. On drift it achieved 318 compared with 331 and 328. The
consistent-Q2 versus Q1-first drift difference was negative with an
interval below zero. The clean residual optional policy improved the
main point estimate over masked residual supervision, with an interval
crossing zero, and had a lower drift point estimate. The controls do
not establish label mixing as the dominant cause of the older
retreat in performance.

Among single B policies, outcome-contrast data gave the best main
point estimate (916 versus 891 with random data and 897 with evidence
nominations). The main paired intervals include zero in this rerun.
In the drift set, random data finished 339, evidence data 330,
and outcome-contrast 323. This reversal limits a claim of uniform
superiority for the nomination method or the full PSDP-trained B.

In the main set the new PSDP selected zero, one and two executed probes
8, 419 and 597 times, respectively. It delegated to child on 896 tapes
and that child chose handoff with another feasible move on 299 tapes.
The second decision varied among perturbation tapes at 74 of the
128 scene roots. Neither matched precommitment fell back for a
non-handoff request. All runs used at most 66.999 mm and 8 s, with
position and velocity handoff discrepancy at most 1.78e-15 in native
units. The benchmark uses two classes of synthetic vessel geometry and
streamed detections. Physical or cross-organ validation remains separate.

## Archived evidence and reproduction

- [Full grouped statistics and paired intervals](results/main/summary.json)
- [Main paired execution records](results/main/test.jsonl.gz)
- [Drift paired execution records](results/main/drift.jsonl.gz)
- [Child-only reached states and labels](results/main/successors.jsonl.gz)
- [Root continuations and decision traces](results/main/pairs.jsonl.gz)
- [Child and its freeze](results/main/child_frozen.json) and [root replay binding](results/main/pairs_frozen.json)
- [Original pre-evaluation freeze](results/main/pre_evaluation_frozen.json) and [release freeze](results/main/frozen.json)

The only source digest changed between the pre-evaluation and release
freezes is the runner. Its final summary command now imports the
experiment-local summary by path. Models, source dependencies, case
manifests, rollout archives, and executed policy/benchmark code stayed
identical. The main and drift streams were complete before this
reporting import correction. SOL Max independently audited the label
contract and recomputed both held-out comparisons from the raw records.
