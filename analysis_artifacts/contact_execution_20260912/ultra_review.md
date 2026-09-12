# Independent ULTRA implementation review

Date: 2026-09-12. Reviewer: `/root/execution_ultra_review`.

**Verdict:** No unresolved correctness blocker was found in the final inspected implementation. This verdict covers the offline code and tests described below. The coordinator's final full-suite run is separate from this review. No robot, rail, camera, or running hardware controller was operated by this reviewer.

## Scope and invariants

Reviewed the outer active transaction, reference/quality state, deficit objective, bounded numerical solver, source-gap admission, compensated observer and variable-step filters, final rocking envelope, native/Python QPIK integration, native request/commit protocol, force/status publication ordering, command-budget interaction, and rejection/replay recording. Production files were read only; this report is the reviewer's sole authored repository artifact.

The final implementation preserves these inspected invariants:

- A failed or waiting proposal does not commit nominal command state, quality slew state, final angular history, or reference advancement. Resumption excludes the rejected computation interval from reference advancement. Real measurements already consumed remain consumed once.
- Held source identities do not advance either measurement filter. A recovered history gap requires the immutable original committed lease, matching source epoch and both components' own previous-source timestamps. Current send age remains strictly below 15 ms; an original lease is never extended to admit recovery.
- The long-gap LP update uses the exact first-order-hold solution of the existing pole and actual endpoint measurements. The normal interval path retains its deployed bilinear update. The independent fresh raw/filtered 6 N stop remains present.
- The outer numerical attempts retain the same matrices, physical rows, objective, and tolerances: cached primary 30 outer / 10 inner iterations, followed at most once by a fresh Ruiz-preconditioned 200 / 10 attempt with rho 1e-3. Deadline and finite/residual acceptance checks fence results. Online refusal does not run offline LP feasibility diagnostics.
- Rocking tiers relax jerk, acceleration, then task angular rate. Original joint/collision rows remain enforced. Final payload energy reservation still uses the existing complete six-dimensional command model.
- Rejection and elapsed holding settle the existing frozen command pair once. Diagnostic measured-port work does not refill the logical budget. Partial or uncertain publication remains a latched failure rather than a retry opportunity.

## Findings fixed during review

1. **False jerk feasibility after a mechanical override.** Clipping both jerk-acceleration endpoints could turn an empty intersection into a valid singleton. The raw jerk interval is now retained and an actual empty intersection requires an explicit tier change.
2. **Stale numerical failure metadata.** Per-proposal attempt/deferred diagnostics are reset at solve entry, preventing a new expired proposal from inheriting the previous proposal's numerical attempts.
3. **Reference catch-up following rejection.** Reference resumption now accounts for the pause endpoint rather than advancing through a failed solve's wall time. Its zero-advance boundary keeps a valid candidate interval without committing reference time.
4. **Source expiry between observer ingress and QP preparation.** This definitely-unsent scheduling miss now uses typed deferral under the new policy. The accepted source's nominal measurement update is consumed once; the raw 6 N, retract, and uncertified-brake checks retain priority.
5. **Successful dispatch markers blocking the next ingress retry.** A successful commit now retires pending identity, dispatch/review markers, and rejection reason. A stale next ingress can retry under the original live lease, and its abort log no longer names an already-published command as the rejected proposal.
6. **Missing rocking facts and mismatched tolerances.** Active final review rejects disabled/missing/unknown policy facts. It uses the existing inner rocking certification tolerance, rather than the stricter unrelated outer-QP tolerance. Outer-QP and energy tolerances are unchanged.
7. **Provisional Python tier relaxation leaking through a rejected rewrite.** Restoring the certified QP command also restores the prior tier/bounds before subsequent publication checks.
8. **Fault telemetry on the device-stop thread.** A precreated notification worker receives fault reasons through a nonblocking queue. Status publication cannot block entry into the original stop calls, while the status can still be published if a device stop itself blocks.
9. **Aborted native late reply paired with a fresh outer proposal.** A completed aborted reply is discarded; a fresh request carries the fresh frame/twist/envelope and `IN_ABORT_PREV`. No old reply is accepted or committed. The original 50 ms request-age fence is checked first for both completed and unfinished requests; abort does not renew it.
10. **Failure diagnostics losing machine-readable structure/numeric bounds.** Mapping-backed diagnostics remain structured. A separate versioned replay payload retains array shape/dtype and distinguishes positive infinity, negative infinity, and NaN; ordinary display diagnostics retain their null convention.

## Verification

All reviewer-run tests used `source rm75_control/env.sh`, disabled pytest plugin autoload/user-site packages, and fixed OpenBLAS/OMP thread counts to one.

- Final focused run: **67 passed** across `test_contact_qp_execution_runtime.py`, `test_contact_qp_execution_policy.py`, `test_contact_qp_bounded_solver.py`, and `test_wbc_rt_aborted_reply.py`.
- Earlier independent core run: **82 passed** across two-clock, variable-timebase, bounded-solver, and lease-gap tests.
- Earlier independent compatibility run: **45 passed** across execution-policy, active command-budget, and continuous-visual tests. These suites overlap; their counts are not additive.
- Independently reproduced the stale dispatch-marker retry failure before its fix and checked its new real-outer regression afterward. Reviewed the mechanical worker's real native late-reply test, including fresh sequence/frame/envelope, `IN_ABORT_PREV`, and unchanged committed qdot history.
- `git diff --check` passed at final inspection. Inspected the new numeric replay encoder/decoder and replay/test scripts. The exact-record replay explicitly does not replay a live deadline.

## Remaining limits

There is **no unresolved implementation defect identified by this review**, but these limits must remain explicit:

- The rocking row is constructed before transport. Actual publication latency can change acceleration and jerk calculated from consecutive completed sends. The final publication audit records the actual interval, acceleration, jerk, actual tier, and `timing_limited`; it does not retroactively constrain a completed send. Therefore the implementation does **not** establish a hard bound on actual publication-time jerk under arbitrary scheduling or transport delay. Numerical certification tolerance also applies.
- Iteration caps and at most four mechanical envelope tiers bound attempted numerical work; they are not a hard wall-clock scheduling guarantee. Expired candidates are fenced, and the old request/command lease and original watchdog rules remain relevant.
- FOH recovery is an endpoint interpolation model. It cannot reconstruct an unobserved force peak inside a source gap. The retained 6 N guard acts on actual received measurements.
- T7 historical replay uses explicitly documented reconstruction assumptions for values absent from the original logs. New versioned rejected-record payloads support exact numeric reconstruction; they do not make those historical inputs exact or replay physical closed-loop behavior.
- The logical command budget, publication success, and these offline tests establish neither physical passivity nor successful acoustic repair. No new physical stiffness/center estimate is introduced by this change.

## Final diagnostic-only follow-up

Reviewed the subsequent `contact_active.py` additions for outer angular/rate/acceleration row annotations, energy-margin and repair-shortfall reasons, selected native rocking-boundary annotations, `control_sample.rotation_base_tcp`, and `final_rocking_review.axis_base`. The tier-to-boundary mapping is consistent: tier 1 checks all three pairs, tier 2 checks rate/acceleration, tier 3 checks rate, and tier 4 checks none. The values and array shapes used by these additions are supplied by the validated solver/smoothing objects; no new exception path was found for those valid internal values.

These additions update diagnostic reason sets and record fields only. They do not change solver inputs, selected tiers, limits, reference/command commits, or energy reservation. Existing freshness/deadline checks still follow final-review logging before any dispatch. Independent follow-up verification: **47 passed** across execution-runtime, execution-policy, and continuous-visual tests. The review verdict is unchanged.

Also reviewed the final `publication.progress_limited_by` expression: its validated scalar/reference inputs introduce no identified exception risk, and it computes a record label without changing accepted alpha or controller state.
