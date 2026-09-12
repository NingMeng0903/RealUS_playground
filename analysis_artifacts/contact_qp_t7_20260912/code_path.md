# T7 current-code audit, 2026-09-12

Input: `/media/camp/PEI_T7/icra 2027_contact/uncalibrated`. This is a read-only investigation of recorded runs. No production controller setting, robot state, or original recording was changed.

## Why one surface error can hide different failures

`ICRA_YM/script/scan_robot.py:LiveSamples.fresh` checks the compensated force sample's original monotonic timestamp against a 100 ms limit. It raises `Stale or missing force feedback.` for both an absent row and a row beyond that age. The message does not identify the sensor or solver as the root cause.

`ICRA_YM/script/record.py:Collector.poll_shm` reads raw robot feedback and compensated force from separate shared-memory publishers. It checks source timestamps and shared-clock provenance. Raw TCP records include raw sensor wrench; they are not the same stream as the compensated TCP force used by the scan supervisor.

`peirastic/realman8dof/daemon.py:_on_step` relays compensated force through `_publish_force_sample`. A failure before the next successful control step can therefore leave the supervisor with an old compensated sample even while the raw TCP stream continues. `joint_admittance_8dof/loop.py` calls the fault stop before logging and proposal cleanup, so the later JSON exception timestamp is not a direct timestamp of the initial fault detection.

## Three distinct temporal checks

1. Image age: 300 ms against the effective image timestamp. With `pause_visual`, a previously established valid visual stream may pause and recover without stopping the fresh-force task. Invalid startup/registration and future timestamps are separate checks. Paused visual feedback is an actual temporary removal of visual input, not a claim that all motion is frozen.
2. Force-source age: 15 ms at source admission and before publication. A definitely-unsent expired proposal can be discarded and recomputed with a newer sample while the previous committed command retains its original bounded lease. No new lease or energy is created by this retry.
3. Fresh force-source spacing: independently capped at 15 ms by `SourceClock.observe`, with a corresponding observer interval check. Thus a fresh new observation can still be rejected because its timestamp is more than 15 ms after the previous accepted observation. Existing retry logic does not bypass this check.

The supervisor's 100 ms limit is downstream of these checks. Raising that limit alone cannot repair an outer-QP solver failure or the source-spacing exception.

## Visual direction and fusion

Recorded profile: flat 50 mm aperture, centered TCP, `image_x_sign=-1`; left/right quality windows use the shallow top 22% and exclude the outermost 4% of image width. The policy compares right-minus-left confidence after a 0.03 deadband. With the declared geometry, right weakness requests positive tool-y angular velocity, left weakness negative. This is internally consistent geometric reasoning, not an experimentally measured image-quality derivative.

The QP targets total angular velocity. The nominal torque controller retains previously accepted total angular velocity; adding a fresh incremental request on top every tick would accumulate past the intended request. A zero QP-minus-nominal correction may mean the nominal motion already satisfies the visual request. Conversely, a nonzero total rotation alone is not evidence that vision caused it. The audit must match confidence, request, QP candidate, and successful publication for the same control ID.

Visual rows are soft task requests. Common loading, aperture budget, measured force scheduling, mechanical limits, and the energy constraint can prevent full request achievement. At or above 4.5 N the configured visual force gate is zero. Confidence in a shallow averaged ROI does not establish the origin of all deep or narrow dark regions.

## Recorded-code provenance

All 49 `study_start` records share the same hashes for their nine recorded source/config entries. Each hash matches its current local counterpart: contact mode wrapper, recording wrapper, QP, port constraint, nominal force/torque/transaction modules, force configuration, and admittance controller. This verifies these recorded entries only; the logger does not hash every file involved in the control process.

## Scope of energy evidence

The active ledger uses `task_power_source=nominal_command`, `settlement_port=logical_final_model`, and explicitly logs physical assurance as unverified. Frozen accepted command-model port work plus the actually used nominal-task power allowance determines tank work; unused allowance is not banked. Recorded ledger closure and fluctuations demonstrate that accounting operates. They do not establish a measured physical passivity theorem or that energy limiting was exercised if reserve never became active.
