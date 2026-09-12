# T7 2026-09-12 failure audit

Dataset: `/media/camp/PEI_T7/icra 2027_contact/uncalibrated`.

58 attempts across 36 trials: 36 completed, 22 failed. Session metadata reports 21 `Stale or missing force feedback.` errors and 1 `Scan task ended without confirmed completion: servo_twist`. This message is not a reliable root-cause label. Of the 14 failures retaining controller QP logs, 8 stopped on outer-QP solver rejection and 6 on source interval rejection after an expired-publication retry. Eight other failures lack QP evidence.

| Subject | Complete | Failed | Force stale | Other |
|---|---:|---:|---:|---:|
| yangming | 12 | 5 | 5 | 0 |
| yuhan_L | 6 | 13 | 12 | 1 |
| yuhan_R | 6 | 1 | 1 | 0 |
| zhongyao2 | 12 | 3 | 3 | 0 |

## Every failed attempt

| Subject | Attempt | Evidence / controller cause |
|---|---|---|
| yangming | RH_Per_C_DtP/001 | tracking_started_but_attempt_directory_missing; directory absent |
| yangming | RH_Per_C_DtP/002 | prepare_only; directory absent |
| yangming | RH_Per_C_DtP/003 | tracking_started_but_attempt_directory_missing; directory absent |
| yangming | RH_Per_S_PtD/001 | prepare_only; failure.json only |
| yangming | RH_Per_S_PtD/002 | prepare_only; failure.json only |
| yuhan_L | RH_Per_L_DtP/001 | source_interval_exceeds_maximum |
| yuhan_L | RH_Per_S_DtP/001 | solver_status_or_residual |
| yuhan_L | RH_Per_S_DtP/002 | source_interval_exceeds_maximum |
| yuhan_L | RH_Per_S_DtP/003 | source_interval_exceeds_maximum |
| yuhan_L | RH_Per_L_PtD/001 | source_interval_exceeds_maximum |
| yuhan_L | RH_Per_L_PtD/002 | prepare_only; failure.json only |
| yuhan_L | RH_Per_L_PtD/003 | source_interval_exceeds_maximum |
| yuhan_L | RH_Per_S_PtD/001 | solver_status_or_residual |
| yuhan_L | RH_Per_S_PtD/002 | prepare_only; failure.json only |
| yuhan_L | RH_Per_S_PtD/003 | source_interval_exceeds_maximum |
| yuhan_L | RH_Per_S_PtD/004 | solver_status_or_residual |
| yuhan_L | RH_Per_S_PtD/005 | solver_status_or_residual |
| yuhan_L | RH_Per_S_PtD/006 | solver_status_or_residual |
| yuhan_R | RH_Per_L_PtD/001 | prepare_only; failure.json only |
| zhongyao2 | RH_Per_S_DtP/001 | solver_status_or_residual |
| zhongyao2 | RH_Per_S_DtP/002 | solver_status_or_residual |
| zhongyao2 | LH_Per_L_PtD/001 | solver_status_or_residual |

## Solver failures

All 8 `qp_prepare_rejected` records report `QPSolverOutput.PROXQP_MAX_ITER_REACHED` with `reason=solver_status_or_residual`. Four are long solves (33.34–37.45 ms; 18,045–19,513 iterations), four report 2.79–4.41 ms and 207–770 iterations. The logs alone do not prove infeasibility: solver/model replay is needed to distinguish conditioning, constraints, or a solver failure.

Every rejected solve has `image_valid=true`; force at rejection is 3.973–4.336 N, below soft 4.5 N and hard 6 N. Every last accepted tank balance is 0.10116–0.14980 J, comfortably above 0.050 J reserve. None of these stops is labeled image loss, overforce, or depleted tank. One rejection has differential sign 0, so the failure is not confined to a nonzero confidence imbalance.

| Subject / attempt | Reject line | Solve ms | Iterations | Force N | Previous tank J | Quality L/R |
|---|---:|---:|---:|---:|---:|---|
| yuhan_L / RH_Per_S_DtP/001 | 25569 | 37.449 | 19303 | 4.271 | 0.14980 | 0.802 / 0.900 |
| yuhan_L / RH_Per_S_PtD/001 | 29409 | 34.929 | 18707 | 4.193 | 0.10116 | 0.897 / 0.806 |
| yuhan_L / RH_Per_S_PtD/004 | 17737 | 3.988 | 440 | 4.066 | 0.10465 | 0.631 / 0.909 |
| yuhan_L / RH_Per_S_PtD/005 | 15310 | 4.410 | 770 | 3.973 | 0.10391 | 0.747 / 0.890 |
| yuhan_L / RH_Per_S_PtD/006 | 16057 | 3.419 | 246 | 4.336 | 0.10536 | 0.743 / 0.878 |
| zhongyao2 / RH_Per_S_DtP/001 | 36967 | 33.340 | 18045 | 4.135 | 0.12997 | 0.989 / 0.975 |
| zhongyao2 / RH_Per_S_DtP/002 | 13602 | 35.476 | 19513 | 4.286 | 0.11192 | 0.855 / 0.977 |
| zhongyao2 / LH_Per_L_PtD/001 | 19048 | 2.794 | 207 | 4.127 | 0.10126 | 0.892 / 0.939 |

Example: `yuhan_L/attempts/RH_Per_S_DtP/001/contact_qp.jsonl` line 25569 rejects control 3090 at monotonic 18418.512958 s (37.449 ms solve). Controller stop at line 25576 records `contact_qp_exception:RuntimeError:outer QP: solver_failed: solver_status_or_residual` at 18418.639198 s. `failure.json` instead reports stale/missing force. Raw TCP continues to 18418.577604 s, whereas the controller-published force stream ends at source 18418.465407 s. This is consistent with force output ceasing after controller failure, not the force sensor disappearing first.

## Source/retry failures

All 6 `source interval exceeds declared maximum` stops occur in `yuhan_L`. Each terminal sequence is `publication_review_rejected:wrench_source_expired` → `publication_fresh_retry` (`definitely_not_sent=true`, previous command retained within lease) → interval exception. The 15 ms source interval check remains capable of rejecting the next sample even though publication retry is enabled.

| Attempt | Expired review source age ms | Retained lease remaining ms | Final rejected source s | Next raw TCP source s | Raw seq difference |
|---|---:|---:|---:|---:|---:|
| RH_Per_L_DtP/001 | 16.091 | 38.605 | 18291.378266 | 18291.396187 | 2 |
| RH_Per_S_DtP/002 | 26.154 | 24.720 | 18457.912449 | 18457.940351 | 1 |
| RH_Per_S_DtP/003 | 15.661 | 39.167 | 18476.091932 | 18476.102326 | 1 |
| RH_Per_L_PtD/001 | 16.453 | 36.803 | 18542.115242 | 18542.133230 | 1 |
| RH_Per_L_PtD/003 | 15.919 | 38.529 | 18586.179633 | 18586.187536 | 1 |
| RH_Per_S_PtD/003 | 16.030 | 38.531 | 18750.847717 | 18750.861720 | 2 |

Raw TCP is sampled by a separate recorder and may omit source frames; its observed gaps do not prove UDP sensor gaps. For example `S_DtP/003` and `L_PtD/003` retain consecutive raw TCP rows around the failure with intervals below 15 ms, while the controller still throws the filter/source interval exception. Control history can skip available sources during computation/retry. Conversely, `S_DtP/002` has consecutive raw TCP sequence numbers separated by 27.902 ms at the terminal gap. These records support a mixed timing problem; they cannot identify CPU scheduling vs driver publication vs acquisition latency without the driver timing logs.

## Recording and interpretation limits

- All 14 failed attempts with raw recordings have H5 `complete=true`, empty recorder `error`, and zero nonfinite values in raw TCP wrench and force arrays. Here complete means the recording closed cleanly, not the scan succeeded.
- All 14 have TCP rows after the final causal preparation/review, while the force stream ends beforehand; continuing TCP is evidence against complete feedback transport loss in these cases.
- Across recorded failed attempts the highest accepted contact force is 4.825 N. No logged stop is a hard-force supervisor stop.
- Every terminal energy tail latches `logical_epoch_expired_actual_tail_unknown` after the exception / stopping delay, while balances remain positive. This is a consequence of losing the command epoch assurance; it is not evidence that the energy tank ran empty.
- Six missing-evidence failures are prepare-only. Two tracking failures in yangming (`RH_Per_C_DtP/001` and `/003`) have missing attempt directories. A third missing directory is prepare-only `RH_Per_C_DtP/002`. Do not assign logged causes to these missing cases.
- Session subject IDs / log paths retain old numeric source directories (`/media/camp/yameng/.../001`, `/002`) despite the renamed T7 subject folders. Resolve attempts by the current session-relative directory, as this audit does.

Machine-readable details, all rare events, last 12 control records, raw timing statistics, and local failure timelines: [summary.json](summary.json). Reproduce with `/media/camp/EXT_DRIVE/envs/genesis/bin/python analysis_artifacts/contact_qp_t7_20260912/failures/analyze.py`.
