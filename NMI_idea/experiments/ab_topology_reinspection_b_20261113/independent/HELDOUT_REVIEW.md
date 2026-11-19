# Independent held-out verdict

The test demonstrates synthetic branch discovery followed by new, identity-checked acquisition after a common two-step MPC prescan. It finds no reliable completion improvement over all-return replanning MPC at the declared cost margins. The primary recurrent policy and its controls rarely attempted the necessary consecutive station pair. The witness-guided and full-rollout planners were active, with uncertain nominal gains that did not hold under the shifted shape and interference condition.

## Independent reconstruction

I verified all six source, bank, protocol and checkpoint manifests against the runtime settings. I independently rerendered 41,638 suffix packets for 11,264 arm-case records, including the primary nominal and held-shape arms and the learned training-seed sensitivity. The two cohorts contain 512 independent anatomies. The extra learned policies reuse those cases and tapes. For every record I reconstructed the shared prescan, actual checked edge sequence, arrival clocks, public Bayesian updates, accepted frame slots, physical costs and endpoint. Every suffix observation was supported by the public model. The posterior interval scores and the joint mass of each selected claim were recomputed by interval enumeration. All reported seed-7 paired completion intervals, four-metric cost intervals, cost margin flags and exact harmful-affirmative discordance bounds reproduced.

| Controller | Nominal correct / 256 | Nominal Y / 128 | Nominal negative / 128 | Held-shape correct / 256 | Held-shape Y / 128 | Held-shape negative / 128 |
|---|---:|---:|---:|---:|---:|---:|
| All-return live MPC | 165 | 51 | 114 | 147 | 38 | 109 |
| Longer belief rollout | 176 | 56 | 120 | 141 | 34 | 107 |
| Checked route witnesses | 174 | 58 | 116 | 140 | 31 | 109 |
| One-step with common tail | 172 | 58 | 114 | 147 | 38 | 109 |
| Fisher-assisted one-step | 170 | 56 | 114 | 149 | 39 | 110 |
| Same-packet nearest suffix | 119 | 1 | 118 | 107 | 2 | 105 |
| Prescan-only | 111 | 0 | 111 | 96 | 0 | 96 |
| MPC-distilled recurrent PPO | 115 | 0 | 115 | 102 | 0 | 102 |
| MPC-distilled MLP PPO | 113 | 0 | 113 | 95 | 0 | 95 |
| Discrete CQL | 116 | 2 | 114 | 108 | 0 | 108 |
| Recurrent behavior cloning | 120 | 5 | 115 | 111 | 3 | 108 |
| MLP behavior cloning | 120 | 0 | 120 | 109 | 0 | 109 |

On nominal Y cases, the paired witness advantage over MPC is +5.47 percentage points [−0.78,+12.50], from 13 wins and six losses. It costs +4.45° mean turn, with simultaneous interval [+1.38,+7.61]°, and the one-sided 97.5% upper probability of a new erroneous affirmative claim is 6.70%. The paired rollout advantage is +3.91 [−4.69,+12.50], from 18 wins and 13 losses, with +11.29° mean turn [+7.12,+15.52]° and a 12.93% harm upper bound. On held-shape Y cases witness changes completion by −5.47 [−11.72,0.00] and rollout by −3.13 [−10.16,+4.69] points. No witness or rollout comparison passes the joint upper-cost test or strict equivalence in either cohort.

The seed-7 recurrent policy never executed the qualifying ordered pair in either cohort. Across the three frozen seeds, recurrent PPO and MLP PPO each complete zero positive cases in both cohorts. Two of the other recurrent runs execute a small number of correct ordered moves without completion. CQL completes 2/0/1 nominal cases and zero held-shape cases at seeds 7/29/61. Pure recurrent clones complete 5/1/3 nominal and 3/1/1 held-shape cases. The independent XOR check passed at every recurrent seed. This documents working memory infrastructure and failure of the tested learned policies on fresh inspection.

## Discovery and evidence

The five paid prescan captures span 49 distinct view sequences in the nominal cohort and 53 in held shape. The nominal prescan posterior alone correctly classifies Y/no-Y in 224/256 cases and the exact task label in 204/256. Taking the best positive fork/side label among true Y cases gives 95/128. Held shape scores 205/256, 173/256 and 87/128. Four-family MAP gives 213/256 and 188/256. Four-family MAP is an evaluator readout from the public posterior, separate from the terminal no-target declaration. These readouts do not count prescan images as fresh evidence.

The following breaks down the Y endpoint along the actual executed routes. Ordered travel refers to consecutive required stations after prescan. Both returns refers to an actual consecutive pair with the true daughter present in both packets. Choosing private slots to salvage the actual declared pair would change the nominal MPC outcome by zero cases. Granting its actual selected correct identity pair the true fork coordinate would raise completion from 51 to 59. Neither is a deployed intervention.

| Controller | Nominal ordered / both / complete | Held-shape ordered / both / complete |
|---|---:|---:|
| All-return MPC | 98 / 67 / 51 | 84 / 55 / 38 |
| Longer belief rollout | 97 / 76 / 56 | 84 / 51 / 34 |
| Checked route witnesses | 99 / 75 / 58 | 85 / 51 / 31 |
| Fisher-assisted one-step | 101 / 80 / 56 | 85 / 62 / 39 |
| Nearest suffix | 16 / 1 / 1 | 19 / 4 / 2 |

Of the 77 nominal MPC Y failures, 30 lack the required consecutive moves and 31 lack both returns after those moves. The remaining 16 comprise six abstentions, one false absence despite both returns, one wrong branch label and eight fork estimates outside tolerance. Of 90 held-shape MPC Y failures, 44 lack the required consecutive moves, 29 lack both returns, 14 abstain despite both and three miss the fork tolerance. On the nearest control's same realized paths, retaining full packets would make both daughter returns available in 11 nominal and 13 held-shape cases. Compression retains one and four. This isolates lost evidence on those paths while leaving the larger route-selection difference visible.

Under the common checked motion and resource caps, all 128 Y cases in each cohort have a mechanically legal ordered pair. With access to the hidden identity and entire future tape, 127/128 nominal and 126/128 held-shape cases have some feasible path containing both target returns. This is a fixed-tape feasibility ceiling. It establishes available motion and acquisition support while leaving the online inference and choice problem intact.

Reports `CONFIRM_AUDIT.json` and `SHAPE_AUDIT.json` contain all numerical contrasts and per-family counts. `CONFIRM_SEED29_AUDIT.json`, `CONFIRM_SEED61_AUDIT.json`, `SHAPE_SEED29_AUDIT.json` and `SHAPE_SEED61_AUDIT.json` contain the learned-policy sensitivity checks. The files are independently derived from the paired arm records using `audit_trials.py`.
