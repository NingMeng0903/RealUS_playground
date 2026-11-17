# Paired diagnosis of the learned decision gap

The crossed comparison uses 64 new development cases with eight paired evaluator tapes per case. Every arm renders the same action library and shares segment feedback and the MPC continuation. The weighted teacher used a mean of 2.20 retained hypotheses per case, twelve tapes for each and at most 0.000097 omitted posterior mass. On an independent posterior bank, 59/64 teacher choices passed the predicted constraint check and 52/64 stayed best among the checked actions.

| Eligibility and ranking | Connection | Unbranched | Pooled | Difference vs frozen student [95% CI], pp |
| --- | ---: | ---: | ---: | ---: |
| Direct MPC | 90/256 | 69/256 | 159/512 | — |
| Student eligibility and student ranking | 118/256 | 69/256 | 187/512 | — |
| Teacher eligibility and student ranking | 121/256 | 72/256 | 193/512 | +1.17 [−0.39, 2.93] |
| Student eligibility and teacher ranking | 131/256 | 69/256 | 200/512 | +2.54 [0.39, 5.08] |
| Teacher eligibility and teacher ranking | 134/256 | 72/256 | 206/512 | +3.71 [0.39, 7.42] |

Replacing only the ranking with the teacher's added 13 net completions. All 13 were in connection cases. This replacement kept exactly the student's eligibility rule, saved 0.61 mm mean travel [−1.09, −0.22] and incurred the same mistake counts. The connection contrast was +5.08 points [0.78, 10.16]. Replacing eligibility alone added six completions, three per family, with +0.50 mm mean travel [0.10, 0.95] and the same mistake counts. Replacing both components added 19 net completions. The marginal contrasts depend on the component kept fixed.

In unbranched cases the teacher ranking under the student's eligibility achieved the same 69 completions. Teacher eligibility exposed a feasible choice to the student's ranking and it reached 72. The added simulated duration for this hybrid was +0.031 s [0.000, 0.086], measured relative to the student. The full teacher also completed 72, with additional simulated duration +0.055 s [0.008, 0.117]. The unbranched success interval for either +3 comparison is [0.00, 2.73] points. The data locate the candidate vetoes and provide a small diagnostic contrast.

The student vetoed 12 teacher choices from the weighted selection bank, including eight due to time. In the unbranched set it vetoed eight, seven due to time, with multiple violated caps possible. The three unbranched cases with more teacher completions all involved a vetoed teacher choice. For 18310024, the student predicted +0.337 s for `forward_left` over MPC. The eight paired executions averaged a time difference of zero and produced one more correct completion. Two other winning teacher choices actually added +0.25 s and +0.50 s relative to the student. The teacher gate is a forecast and an error-analysis reference.

## Fit on training states and feature ablation

On 64 sampled original training states, the frozen student agreed with the best choice under its original 65/35 mixed target on 24 states. The average regret under that target was 0.0212 utility units. An independent weighted rollout bank reduced the utility gap to 0.0110. On eight new realized tapes at each state, the student's selected actions completed 217/512 while the original training-target selections completed 226/512, a difference of +1.76 points [0.00, 4.10]. The gap is present even at familiar states. The difference between the fitted label and independently estimated gap also illustrates the uncertainty of treating a sampled training optimum as a certain expert label.

Adding each public witness's conditional association and centering vectors, time, station and posterior joint centering/identity support held the original student at **187/512**, with 21 changed first choices. Under the identical teacher eligibility set, it changed **193/512 to 198/512**, +0.98 points [−0.59, 2.93]. It added zero unbranched completions in both comparisons. The witness variables were genuinely supplied at training and decision time with the same mixed labels, hyperparameters and motion options.

For context, the prior learned-B holdout had 18 unbranched teacher-only winning tapes. Three had a certificate retraction in either arm and those three also had retractions in both arms. There were zero repaired incorrect certificates from before the intervention on the 18 winning tapes. Obtaining a new correctly centered witness later in the scan explains the earlier traced gains more directly than repairing an old wrong certificate. The conditional feature ablation tests the current matched, fixed-pose family, where old certificate changes are uncommon.

## CEM label control with matched targets

| Target recipe | Excluding CEM labels | Including CEM labels | Difference [95% CI], pp |
| --- | ---: | ---: | ---: |
| Posterior only | 185/512 | 186/512 | +0.20 [−1.37, 1.76] |
| The same 65/35 mixed target | 188/512 | 187/512 | −0.20 [−2.54, 1.95] |

Both controls retain the entire CEM motion library. The candidate row counts are 4,263 and 5,905. Each matched target comparison gives a negligible completion difference. The verified-teacher study separately observed a small gain from offering extra CEM choices to the online selector. That isolates the action library more directly than the training-label contrast.

## Scope and records

The ranking errors in connection and the eligibility errors in unbranched scans call for different training targets. Independent paired completion ordering within the reachable candidate set can supervise the first. The second needs a calibrated relative resource prediction for the shared scan after the intervention. Additional witness features alone yielded no clear improvement here.

See the [protocol](B_SELECTION_DIAGNOSTIC_PROTOCOL.md), [crossed comparison](results/b_selection_diagnostic_v1/cross_summary.json), [per-case crossed records](results/b_selection_diagnostic_v1/cross_cases.jsonl.gz), [label and input checks](results/b_selection_diagnostic_v1/training_checks.json), and [trace audit](results/b_selection_diagnostic_v1/trace_audit.json). This release gives development evidence from closed-loop simulation and fresh-noise checks on archived training states.

The [teacher completion audit](TEACHER_COMPLETION_AUDIT.md) separately checks the absolute success ceiling and the shared image-retention rule.
