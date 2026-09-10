# Active runtime integration (software only)

**V4 update:** the former live-port settlement limitation described below has been superseded. A measured-only W/arm/raw-rail interval adapter now feeds the same ledger and charges both net output work and D. See [ENERGY_D_ALIGNMENT_MEDIUM_V1.md](ENERGY_D_ALIGNMENT_MEDIUM_V1.md) and [FINAL_HANDOFF_V4.md](FINAL_HANDOFF_V4.md). Physical certification remains unverified; this document preserves the earlier integration record.

The active adapter now wraps the shared original ICRA 4 N TFF candidate/commit core and sends its three-variable outer-QP result through the unchanged native IK. `wrap_study_phase` selects baseline, shadow, or active. The real YAML remains uncalibrated and defaults to baseline; no hardware was connected or started for this work.

## Runtime boundaries

- A fresh source timestamp advances observer measurement state once; held samples still allow each control-cycle command candidate. The original observer coefficients are restored when leaving active at a stopped phase boundary.
- The actual control interval remains in original nominal control timers and logs. The bounded reference grant and next-command hold model are separate and cannot recover long wall-clock gaps.
- Python applies the final rail-target correction, reserves the original rail transaction, then checks the final Cartesian payload model and optional single-tank command budget immediately before arm publication. Native IK, its mathematical cost/constraints, protocol and rail worker were not changed.
- Only successful arm send plus rail transaction commit commits the nominal candidate and reference. Partial/unknown publication records the device facts, aborts the uncommitted outer candidate and freezes reference. Existing commands are not declared physically rolled back.
- Active source/prepare/publication exceptions invoke the original stop chain before cleanup or logging. Native timeout/coast and unavailable rail transactions also stop this mode. Necessary stopping is never constrained by the outer visual QP or its budget.
- The physical energy assurance stays unverified. A commanded/predicted velocity is not actual motion; unaligned measured ports do not settle energy. Original IK task residuals are recorded, not represented as a certified exact outer-task execution.
- Active completion uses accepted-reference exhaustion plus measured endpoint distance, with `4T + 12 s` timeout. The old daemon reference-duration hot swap is disabled only for active.

## Deployment

Before `hfpc(contact_qp=...)` sends anything, the client requires a capability companion written by the running updated Python service. The companion is tied to the current control/header inode and a fresh service timestamp. Old running Python services without the handler fail explicitly instead of silently executing TFF. Updating disk files alone does not update an existing process. The default CLI validation remains offline.

Contact-study construction runs after the existing stop boundary, outside the live 200 Hz callback. Log draining occurs after coordinated stopping, not before it. The new companion does not alter the existing shared-memory or native ABI.

## Validation scope

`test_contact_qp_active.py` exercises the real shared nominal core, candidate abort/commit, expired review, partial publication, exceptions after nominal preparation, and extracted production runner publication/exception blocks with detached device fakes. Existing baseline/shadow/source-clock/proxy/daemon/API tests are rerun separately. These are software correctness checks, not new real-patient acoustic efficacy or whole-loop real-time certification. No new acoustic parameter tuning or synthetic efficacy claim was made.

The current production measured-twist tracker does not supply `port_time_aligned` or `port_calibration_version`. Thus the live adapter currently reports **monitor_unavailable** for physical port settlement. `energy_constraint_enabled` defaults to false. Explicitly enabling it provides command-model admission only: unsettled reservations remain occupied and may exhaust the budget and stop the mode. This implementation must not be described as an already connected measured-energy feedback loop.
