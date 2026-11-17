# Per-case audit of the learned recovery gap

SOL MAX independently recomputed these results from the original [held-out archive](results/learned_b_cem_v1/test.jsonl.gz). It did not use the new training or held-out cases. All comparisons retain the original common library, online belief update, segment feedback and MPC continuation.

The online teacher completed 365/768 and the complete learned B completed 317/768. The direct paired difference was +6.25 percentage points [3.65, 9.11]. The teacher won 54 individual replays and lost six. Local connection accounted for 30 net completions and unbranched scans for 18.

## Unbranched cases

Both learned policies chose the MPC action in every one of the 11 cases in which the teacher improved completion. The complete model rejected ten of the teacher choices because its predicted duration difference exceeded +0.10 s. Each rejection was due to duration alone. Before applying that check, those ten choices had positive utility and nine ranked in the top three. The remaining teacher choice was eligible and ranked second to MPC.

| Held-out case | Public station and coverage mask | MPC to teacher | Added completions | Learned predicted time delta (s) | Executed mean time delta (s) | Executed mean travel delta (mm) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 14310000 | 4.50 / 000011 | advance → short | 2 | +0.388 | 0.000 | −1.66 |
| 14310009 | 4.50 / 000011 | advance → short | 1 | +0.386 | +0.500 | −0.19 |
| 14310019 | 4.50 / 000011 | advance → short | 2 | +0.375 | 0.000 | −1.75 |
| 14310033 | 4.50 / 000011 | advance → short | 2 | +0.388 | +0.500 | −0.30 |
| 14310040 | 4.50 / 000011 | advance → short | 1 | +0.381 | −0.250 | −3.19 |
| 14310050 | 11.83 / 000010 | revisit → cem_1 | 3 | +0.094 | 0.000 | +2.11 |
| 14310058 | 4.50 / 000011 | advance → short | 1 | +0.387 | 0.000 | −1.72 |
| 14310059 | 11.83 / 000010 | revisit → cem_1 | 2 | +0.114 | 0.000 | +2.72 |
| 14310076 | 4.50 / 000011 | advance → short | 1 | +0.377 | 0.000 | −1.84 |
| 14310089 | 4.50 / 000011 | advance → short | 2 | +0.378 | +0.500 | −0.26 |
| 14310093 | 11.83 / 000010 | revisit → cem_1 | 1 | +0.108 | −0.250 | −1.92 |

Observed scene-mean costs stayed within all original per-action margins on six of these cases, accounting for eight extra completions. The other cases included +0.50 s scan extensions and +2.11 or +2.72 mm travel increases. This is a retrospective analysis of the winning cases.

Every additional completion had exactly one more genuinely correct centered interval witness. Target reappearance and fresh confirmation in the queried interval agreed on these replays. In the `short` cases, the first segment stopped at station 8.2 mm instead of 12.3 mm. That changed the timing and position of the next observations. In 14310033 MPC stored an off-center witness for interval 2 at frame 31, while the teacher obtained a centered witness at frame 39. In the three `cem_1` cases, both policies confirmed interval 0 on the detailed replay. The CEM-derived reverse stopped at 6.89 mm rather than 5.83 mm and gave the MPC tail a usable view of interval 2.

The original unbranched training archive has only 23 `advance` cases at station 4.50 mm with coverage mask 000011. Its development archive has three. Neither archive has a `revisit` case at station 11.83 mm with mask 000010. Of 144 unbranched training states, five have a top-choice teacher action with an observed positive completion difference, zero added errors and compliant mean costs. Three more have such an action among the top three. The 40 unbranched development states had zero such alternatives even when considering all candidates.

## Connection cases and CEM labels

The complete learned model missed 36 replays won by the online teacher. In ten cases the teacher choice was eligible and ranked lower because its predicted completion advantage was smaller. Those cases account for 20 of the 36 missed wins. Eight cost-rejected cases account for the other 16. None of the winning actions in either task was blocked by the predicted identity-risk check.

The model trained with CEM candidate labels won 18 replays and lost 21 against the label ablation across 19 connection cases. The net effect was −0.39 percentage points [−2.86, +2.08]. When both learned models chose ordinary actions, the split was 2 gains and 11 losses. When at least one chose a CEM proposal, it was 16 gains and 10 losses. Two ordinary `left` choices account for seven losses. Both learned arms had the same CEM proposals and fitted renderer at test time. Development selected a 65/35 simulated and evaluator label mixture for one and simulated-posterior labels for the other. This comparison measures those two learned scorers.

An example of a near-winning hard negative is case 14210095. The public rollout forecast a +0.583 completion benefit for `back_center` and no increase in mistakes. On independent paired tapes the action lost two completions and added a mistake. This motivates training from independent executions of the nominated choice, the incumbent and close alternatives.
