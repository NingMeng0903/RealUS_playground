# Prior temporal preflight: no-go for an expanded timing cohort

Reviewed `temporal_preflight.py` and `temporal_preflight/SUMMARY.json` after completion on 2026-11-13. The final source digest is `4cdbf3a9a948250313f98b0a4cc4ff40bcb22b15e9033939313eddb130db33f5`. An unused fine-grid tape-seeding correction followed the initial review; the execution owner verified that all 512 records and numerical summaries stayed identical. This is a design judgment from a prior-model preflight, not a claim of measured B efficacy or a proof of zero temporal headroom.

## Mechanism and scope

Timing was a credible distinct question because the existing world already contains dynamic lag and transient return losses. Those processes are shared across views at a given physical clock; spatial pose refinement does not itself escape a common temporary absence. No new acoustic physics was added.

The preflight retains the original checked motion bank and considers appending a 0, 0.5, or 1.0 second stationary hold to the current MPC's next acquisition only when its command reaches the currently needed site. It does not change the spatial command, run a future policy, retime every action, or compare to MPC equipped with timing choices. Half-second likelihood tables reproduce the original 1.5-second-grid probabilities exactly. Joint resource checks retain both the six-acquisition and nine-second caps.

There are 128 public-model histories per task. A hidden model atom generates each history, but the hold score uses only the resulting public posterior, joint prior-claim correctness mass, resources, progress, and model tables. Histories outside the intervention class remain in the denominator with zero difference and no extra hold. In particular, the active counts are 108/128 and 116/128 for daughter at depths one/two, and 26/128 and 111/128 for trunk. The 102 inactive trunk depth-one histories have a next MPC command outside the required-site acquisition class; this is not a claim that they are all complete.

The depth-one checkpoint can still encounter the additional loss process. That process ends no later than 4.45 seconds. After two original 1.5-second captures, the earliest next acquisition is at 4.5 seconds, so the depth-two check primarily addresses lag/identity timing rather than avoiding that additional loss burst. This scope distinction was corrected before temporal outcomes were inspected.

## Results

The nominal score is a one-capture expected joint claim utility with the shared approximate suffix potential and a joint-cap witness. Its numerical increments are shown below in percentage-point units to make their scale legible; they are **not** changes in completed-scan probability.

| Task / common MPC captures | Active histories / 128 | Contextual nominal increment over no hold | Increment over best fixed hold | Mean selected extra hold |
| --- | ---: | ---: | ---: | ---: |
| Daughter / 1 | 108 | 0.388 | 0.253 | 0.594 s |
| Daughter / 2 | 116 | 0.559 | 0.423 | 0.445 s |
| Trunk / 1 | 26 | 0.0389 | 0.0389 | 0.090 s |
| Trunk / 2 | 111 | 0.0344 | 0.0344 | 0.363 s |

No eligible hold was rejected for loss of every ideal-acceptance completion witness in these histories. That does not certify equal stochastic completion opportunity: an added hold consumes time that could otherwise support retries. The nominal potential is approximate and does not execute those future retries.

`contextual_claim_minus_best_fixed` in the JSON independently maximizes the signed-claim score per history. It is a different selector from the nominal-value selector used for the added-hold counts. Do not present that field as the signed-claim performance of the reported nominal policy. The absolute mean nominal values also include zero placeholders for ineligible histories; use their paired differences, not those absolute means as task-success estimates.

## Verdict

Do not launch a full timing-headroom cohort or train a timing B from this preflight. The model already sees only a small contextual nominal-value increment, especially in trunk, while choosing material delays. This is insufficient positive evidence to justify the next experiment after the active-negative spatial screen and the aperture no-go. It is not a theorem excluding an untested temporal policy, and it does not measure the complete-task effect of the holds.

If timing is revisited after new evidence, it must use complete paired suffixes, give identical timing actions to the shared MPC, and include planning latency in the physical clock or enforce a measured common allocation. The current preflight logs its omission of CPU from the acoustic clock; that omission prevents a deployable timing claim.

The next B redesign should change a validated measurement mechanism that can distinguish consequential anatomical alternatives. Simply increasing the learner, changing an admission threshold, or extending a grid of observationally similar actions does not address the evidence collected here.
