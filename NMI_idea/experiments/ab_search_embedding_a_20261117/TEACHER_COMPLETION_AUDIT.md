# Why the teacher's absolute completion was low

Replayed the 64 development cases from the [selection comparison](B_SELECTION_DIAGNOSTIC_RESULTS.md), each on the same eight paired evaluator disturbances. Reproduction verified the original metrics for both MPC and the public weighted teacher. The original teacher chooses the first recovery segment and then invokes the common MPC. Completion requires the endpoint and an accepted correctly identified and centered image in each of six intervals.

## Teacher failure chain

| First blocking condition | Connection, 256 replays | Unbranched, 256 replays | Pooled, 512 replays |
| --- | ---: | ---: | ---: |
| Endpoint not reached | 0 | 0 | 0 |
| A required interval lacked an eligible centered target observation | 21 | 130 | 151 |
| Observations existed in every interval, with some missing correct associations | 4 | 30 | 34 |
| A correct centered image had been accepted in every interval but the ledger retained an inferior one | 97 | 24 | 121 |
| Correctly completed | 134 | 72 | 206 |

The diagnostic checks physical centering at acquisition, presence of the target in the packet and the measured-centering condition. In 128 of the 130 unbranched missing-observation failures, at least one interval had no recorded acquisition within the true lateral centering tolerance. In all 121 ledger failures an active correct-identity but off-center witness remained after a better associated image for that interval arrived. The original replacement logic revoked on low identity confidence. It kept a high-confidence off-center witness until that identity condition failed.

## Shared online evidence update

The experimental controller retains a newly associated image if its probability of correct identity and centering exceeds the stored image's. It computes this probability using the current Bayesian belief and each image's conditional association and centering vectors. It acts when the new frame arrives and keeps the same identity threshold. A public terminal reselection on the frozen paths first established that these images can be selected from the recorded evidence.

| Method | Original | Updated only after the initial scan | Updated from the first frame | Full update, connection | Full update, unbranched |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strong MPC | 159/512 | 207/512 | 271/512 | 175/256 | 96/256 |
| Public teacher | 206/512 | 247/512 | 327/512 | 231/256 | 96/256 |

Both policies used the same updated ledger and the archived first-action choice. The MPC tail still received the live measurements and replanned. The initial trajectories and post-prefix belief matched the original executions. On every replay, TCP travel, turning and simulated scan duration matched for each policy before and after the change. On each of the archived detailed tapes the executed position and velocity samples also matched. Wrong-identity counts agreed. The teacher's scan completion gain from the shared change was +23.63 pp [14.45, 33.59] and MPC's was +21.88 pp [12.89, 31.64]. The new causal ledger reproduced the public terminal reselection on the same tests. Enabling the update during the initial scan matters because correct centered images can already be missed by the old retention rule before B starts.

The contrast with both ledger updates enabled is +10.94 pp [3.71, 18.95] for the teacher's previously chosen first actions. It is concentrated in the connection cases. The study isolates evidence retention on the original development cases and preserves the original action selections. A fresh test of a planner optimized for the new ledger is a separate comparison.

## What the old first-action menu could achieve

Execute every available first action under the original evaluator and normal MPC tail. A privileged selector has access to the hidden geometry and calibrates on eight separate disturbances per action, then runs on the same evaluator tapes as the original teacher. The hindsight rows select independently per evaluator tape and provide finite-menu ceilings under the stated restrictions.

| Original ledger and MPC tail | Connection | Unbranched | Pooled |
| --- | ---: | ---: | ---: |
| Public teacher | 134/256 | 72/256 | 206/512 |
| Privileged choice within the hard scan budget | 134/256 | 170/256 | 304/512 |
| Privileged choice screened on its calibration mean costs relative to MPC | 139/256 | 71/256 | 210/512 |
| Hindsight over every legal first action within the hard scan budget | 150/256 | 175/256 | 325/512 |
| Hindsight under stricter per-replay MPC cost margins | 141/256 | 72/256 | 213/512 |
| Hindsight with correct centered raw images and endpoint | 256/256 | 232/256 | 488/512 |

In unbranched scans the privileged unscreened selector's extra completions came with +9.18 mm mean travel and +2.68 degrees relative to MPC. The screened selector stayed near the original teacher in completion and executed slightly less travel than MPC. The hindsight comparison fixes the old certificate rule, the first-action menu and the downstream MPC. It does not bound a redesigned multi-step teacher or a new evidence rule. Per-replay feasibility is stricter than the usual cohort-average margins. The raw-image row is a geometric and sensing diagnostic with evaluator truth.

## Records

The [ceiling summary](results/teacher_ceiling_v1/ceiling_summary.json) and [candidate replays](results/teacher_ceiling_v1/ceiling_cases.jsonl.gz) contain all candidate outcomes. The [online comparison](results/teacher_ceiling_v1/full_online_update_summary.json), [execution records](results/teacher_ceiling_v1/full_online_update_cases.jsonl.gz) and [trace audit](results/teacher_ceiling_v1/online_update_audit.json) preserve the shared-rule validation. All data come from fixed-knee synthetic packets and chart-based execution.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B audit_teacher_ceiling.py --source results/learned_b_cem_v1 \
  --cases results/b_selection_diagnostic_v1/cross_cases.jsonl.gz --out results/teacher_ceiling_v1 \
  --calibration 8 --jobs 24
python -B replay_centered_witnesses.py --source results/learned_b_cem_v1 \
  --cases results/b_selection_diagnostic_v1/cross_cases.jsonl.gz --out results/teacher_ceiling_v1 --jobs 24
python -B replay_centered_witnesses.py --source results/learned_b_cem_v1 \
  --cases results/b_selection_diagnostic_v1/cross_cases.jsonl.gz --out results/teacher_ceiling_v1 \
  --update-prefix --jobs 24
```
