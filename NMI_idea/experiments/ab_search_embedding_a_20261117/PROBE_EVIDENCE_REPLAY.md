# Evidence from the first recovery segment

The original paired comparisons establish what the selected action does to final completion. This additional replay asks how much of that difference uses evidence acquired during that first segment. Both previously published ordinary cohorts contribute 56 scenes with three nuisance tapes apiece. The two compared arms select one segment and hand back to the same live MPC. The first experiment uses `public_first_only`. The execution-constrained experiment uses `public_budgeted`.

The script reproduces the chosen first path and verifies the original success, identity error, travel and duration. It then replays that exact 50 Hz first motion, pays the same time and travel, and changes one input to estimation. One replay withholds incoming centroid packets and certificates from that segment. The other allows the frames to be processed while keeping the posterior weights of the two structural hypotheses fixed until the handoff. This second replay still updates the probabilities for offsets, shadow lanes and layouts within each hypothesis and can accept a new image. After the first segment all arms again use the normal filter, evidence bookkeeping, and MPC.

| Cohort | Normal MPC → B | Without first-segment packets | With the first-segment structure weights fixed |
| --- | ---: | ---: | ---: |
| Primary pooled, 168 trials | 67 → 106 | 74 → 105 | 65 → 102 |
| Execution-constrained pooled, 168 trials | 62 → 75 | 63 → 69 | 58 → 62 |
| Execution-constrained connection, 84 trials | 46 → 57 | 46 → 51 | 42 → 44 |
| Execution-constrained unbranched, 84 trials | 16 → 18 | 17 → 18 | 16 → 18 |

The connection advantage in the execution-constrained study falls by six completions when first-segment packets are withheld. The case-bootstrap interaction is +7.14 percentage points for normal first-segment observations, with a descriptive 95% interval of [+1.19, +15.48]. Among the 12 connection scenes in that study with both structural hypotheses having initial posterior mass above .05, the probability of the true structure exceeded .95 at the handoff in 31/36 B replays and 18/36 MPC replays. All 28 constrained unbranched scenes already had probability above .95 on the true structure at the beginning of B. That cohort primarily tests recovery and coverage once identity is clear.

The primary study retains most of its first-action benefit in both altered-evidence replays. The change attributed to withholding early packets is +4.76 percentage points, with descriptive 95% interval [−2.98, +12.50]. Seek separate outcomes for target reappearance, identity resolution, and final completion. It is possible to choose a better motion without acquiring a discriminating observation on that motion.

These are exploratory interventions on the published seeds. The fixed first trajectory itself was generated with streaming feedback, so it can already encode information through its geometry and exit state. Fixing that trajectory controls initial motion when studying the effect of the first packets on evidence and subsequent MPC. The remaining gain can include the earlier feedback correction, geometric repositioning, and new frames collected later. The follow-up scan after the fixed segment can differ in cost. Neither altered replay proves total mediation or a new cost-matched performance result.

[Summary](results/adaptive_evidence_audit_v2/summary.json) · [Per-case replay records](results/adaptive_evidence_audit_v2/cases.jsonl.gz)

```sh
OPENBLAS_NUM_THREADS=1 python -B replay_probe_evidence.py \
  --primary results/adaptive_paired_v2/cases.jsonl.gz \
  --limited results/adaptive_budgeted_v2/cases.jsonl.gz \
  --out results/adaptive_evidence_audit_v2 --jobs 12
```
