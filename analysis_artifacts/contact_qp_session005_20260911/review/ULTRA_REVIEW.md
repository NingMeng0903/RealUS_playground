# Session 005 independent ULTRA review — 2026-09-11

**Disposition: no remaining blocking finding in the reviewed software changes. One concrete image-validity defect was found, reproduced independently, fixed by the HIGH implementation worker, and retested.** This is a code/model review with offline tests, not approval of physical robot/tissue performance or image improvement.

The reviewer changed only files in this review directory. No controller, native worker, arm, rail or image device was started. Existing force-calibration/rail changes outside this task were not reverted or re-reviewed. Exact production files at the end of review are recorded in `audited_sha256.json`.

## Finding resolved during review

`ContactQpOuter._image_observation` originally classified a frame as `ok` from registration and timestamps alone. A fresh registered frame with an invalid left or right window could bypass the initial required-image gate. After an outage it could print `visual resumed`, emit `image_feedback_recovered` and clear the outage while `ContactObservation.fresh()` and the QP correctly considered that frame unusable.

Independent evidence is retained in `invalid_window_before_fix.txt`: **3 failed, 4 passed**. The failures separately cover invalid left startup, invalid right startup, and false recovery after a stale frame. This was a meaningful outer-state/telemetry defect; it did not make the QP execute the invalid frame's visual rows.

The final fix checks precisely `REQUIRED_WINDOWS=(0,2)` and classifies `invalid_required_windows`. Before a valid contact image, required invalid feedback rejects. After that established image, `pause_visual` treats invalid lateral windows as unavailable, preserves the outage, passes no observation to the QP, and resumes only on valid compatible feedback. Center-only invalidity remains diagnostic under the existing two-lateral-window contract. The independent tests now pass.

## Energy model and transaction audit

The production implementation matches its expressly supplied, two-port **command model**:

```text
W = -raw_control_tcp
A = max(0, -W @ V_nominal_before_current_QP)
P = W @ V_final_model
S_used = min(A, max(0, -P))
Delta E = (P + S_used) Delta t
E - E_initial = cumulative_port_work + cumulative_source_used - cumulative_capacity_discard
```

- Nominal twist, wrench and rotation are copied to immutable snapshots. The energy certificate identity is checked at reservation; a replaced certificate cannot increase the allowance. A later snapshot cannot change the earlier active epoch's source.
- The QP and final model review use `P + A + beta*available/hold >= 0`. Final admission recomputes current spendable balance and reserves `max(0,-P-A)*hold`; the old epoch's remaining obligation remains reserved while a new proposal is pending. The final payload model includes the Python rail correction and full six-axis wrench power.
- Settlement uses only the successfully committed frozen pair. Review-to-commit delay earns no new-pair work; the previous pair continues during that delay. An epoch's expiry is set at review and is not extended at commit. Work stops at that mathematical expiry, and the budget then latches against further commands.
- Unused authorization cannot increase the tank. At zero final velocity, both actual port work and source use are zero. Positive final model port work refills up to capacity; discarded capacity is logged. Finite arithmetic and strict admission protect the ledger from nonfinite credit and reserve underflow.
- An aborted, definitely unsent candidate removes only its new reservation. An unknown/partial or already-started publication retains its liability and latches; it receives no candidate-source credit. Measured-port diagnostics cannot recharge the spendable command account. Retry does not reset the tank or create a new source epoch.

The model is mathematically coherent under its declared frozen command-port assumptions. Its source is an explicit nominal task authorization, **not physical harvesting, an external-port passivity theorem, Lee's storage interconnection, or an independent pure-visual account**. Nominal state contains previously executed visual motion and is taken before final IK. This limitation is accurately described in the session document and literature addendum. Consequently, motion supported by the evolving nominal command can continue without draining this tank; that is the declared source model, not evidence of finite-energy autonomous physical passivity.

The `.10/.15/.05 J` initial/capacity/reserve values provide `.05 J` spendable within the new model. The recorded `.02634 J` excess drawdown cited by the analysis supports a trial margin on those recordings, not a physical safety-energy limit or a guarantee for future paths. This reviewer checked the account and model interpretation, not independently every raw-record integration or every supplied PDF formula.

## Visual direction, fusion and unavailable feedback

The retained task uses the shared total angular velocity. `TorqueTilt.commit_applied` stores the accepted total omega; repeatedly adding a fresh visual increment to the next nominal would therefore accumulate it. The archived increment experiment and the real TorqueTilt/QP multi-step regressions capture that mechanism. The optional increment mode is absent from the final production configuration API. Attribution diagnostics distinguish nominal, total and QP delta without changing the objective.

With registered `image_x_sign=-1`, image right maps to negative tool x. For tool-y rotation, endpoint into-contact velocity is `vz-x*wy`. Therefore right-low confidence requests positive wy, and left-low confidence requests negative wy. The geometry and mirrored tests are consistent with this algebra. This verifies the declared mapping and internal sign, not an experimentally identified image-quality gradient. A nominal torque action already meeting the total request need not receive an additional increment.

The selected `.03` deadband is a bounded engineering change from `.10`; reported stable-frame statistics support its trial rationale. The 22% shallow ROI and 4%–96% width selection are unchanged. Deep/narrow darkness may remain outside the statistic, and acoustic shadowing need not represent correctable contact loss. Continuous visual permission and nonexhausted energy do not prove that rotation will remove all visible black regions.

After the first valid registered contact image, stale/missing/invalid-required-window feedback is removed from the QP under `pause_visual`; no image-age-only shutdown or timeout count is introduced. Recovery is automatic on a valid frame. The receiver snapshot is read before the decision timestamp, preventing an asynchronously arriving frame during nominal computation from being mislabeled as future. Initial invalid feedback, registration mismatch and genuinely future time still reject. Existing receiver sequence/source history rejects delayed duplicates and retired-source frames. Previously committed torque-admittance state remains legitimate nominal history; removing the visual task does not erase that dynamic state or guarantee immediate zero rotation.

## Unsent force-age retry and stopping boundaries

Final-review `wrench_source_expired` and pre-send `dispatch_source_expired` are the only eligible reasons. Other dispatch invariants are checked separately. The actual runner aborts the new rail reservation, outer nominal/energy proposal and inner/native pending proposal before considering retry; both send sites remain downstream of these branches.

The old successfully committed command, its original 50 ms logical lease, accepted command history and reference time remain in force. No send, commit, arrival processing or watchdog heartbeat occurs on retry/wait branches. A strictly newer physical force timestamp is required before recomputation. Repeating the same snapshot waits before `prepare_source` or any new force/outer/inner proposal. Native abort sets `IN_ABORT_PREV`, clears pending commit and prevents the rejected native proposal from being implicitly committed on the next request.

The common loop checks external stop, watchdog and fault epoch before every retry wait. `_resync_late_tick` resets an overdue schedule when it is more than one period late; a missed send can cause one immediate next iteration, but does not cause an unbounded catch-up burst. Independent tests execute the actual guard blocks and check schedule resynchronization.

No prior committed lease, old-lease expiry, other budget fault, source-epoch change/reversed provenance, actual long source interval, partial/unknown send and rail faults still stop through the existing paths. Retry is bounded by lease/watchdog time rather than an arbitrary retry count. It does not relax the 15 ms source limit or manufacture source freshness. A genuinely stale newer source or a source interval exceeding the declared maximum can still fail during ingress; this is explicitly outside the narrow unsent-age recovery guarantee.

## Supervisor and offline verification

The external `scan_robot.py` completion branch now distinguishes `done_seq` acknowledgment from successful completion. `err_code==0` plus `Status.DONE` or successful `icra_wait` matches the daemon's completed-task lifecycle; `status()` already rejects stopped/error states and unrelated command sequence changes. Unsuccessful termination is raised before `path_done` or final TCP arrival waiting, preserving the available message. No endpoint tolerance or timeout is changed. The actual external file matches the reviewed staged file with SHA256 `475651e872615123a75624d9bfe37d876ec01dddbe23843c41ca3a149d597b0b`. IPC message truncation remains an existing diagnostic limit.

Final independent command used the existing `rm75_control/env.sh` environment with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `PYTHONNOUSERSITE=1`, single-threaded BLAS and the repository/rm75/src Python paths. **77 tests passed in 3.16 s**, covering:

- Supplied task-power ledger and random six-axis independent work arithmetic.
- Real TorqueTilt/QP multi-step total-velocity recursion, mirrored confidence direction and energy admission.
- Active transient-image handling, both no-send retry branches, native abort flags and strictly-newer-source waiting.
- The staged supervisor completion branch.
- Seven independent reviewer tests for invalid startup/recovery, actual runner stop/watchdog/fault guards and overdue-schedule behavior.

Full output is in `final_targeted_tests.txt`. The root separately reports **513 full-suite tests passed in 253.42 s**, followed after the invalid-window fix by **80 affected tests passed**, plus configuration validation, shell syntax and scoped diff checks. Those broader runs are root-owned evidence, not reruns by this reviewer.

These tests establish the stated software behavior and internal model arithmetic. They do not measure controller timing distributions under load, vendor inner-loop tracking, stopping-tail work, tissue force peaks, physical harvesting or causal image-quality improvement.
