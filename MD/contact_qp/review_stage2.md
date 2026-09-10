# Stage 2 independent review — 2026-09-10

**Decision: PASS for the stage-2 implementation gate.** All concrete review blockers are closed in the sources hashed below. Reviewed scope includes `SIM_MODEL_V1.md`, `plant.py`, `history.py`, `geometry.window_rows`, `qp.py`, `coverage.py`, `evaluation.py`, the detached experiment, the shared production nominal/position helpers, and their tests. This permits the stage-3 execution/fault-injection work; it is not statistical behavior acceptance or a hardware/real-time/passivity certificate.

Findings reported to the implementation owner:

1. **Closed at model level:** path-dependent surface heights changed spring energy without a matching tangential reaction in the physical wrench. The plant now includes `sum(pressure * surface_slope)` in its scan-direction reaction and separately reports prescribed surface-motion power. Independent finite-difference stored-energy tests cover left-gap, curvature, and moving-surface cases without damping/friction; the owner reports those tests passing.
2. **Closed at model level:** `step` previously omitted world TCP x translation during tool-z motion at nonzero rocking angle. It now integrates both world translation components from the tool twist, evaluates the surface at the moving/rotated face points, includes the corresponding transverse contact reaction/moment, and rejects unsupported x/z rotations. An independent pose-difference regression covers the correction.
3. **Closed in current source:** mechanical `window_coupling` alone classified a mechanically coupled acoustic shadow as valid acquisition. Independent acoustic availability now shares the renderer's physical dropout conditions, with a shadow regression. The frozen mechanical-fraction acceptance metric remains unchanged. Delivered registered-image coverage uses the additional acoustic truth and required-window measurement validity; completion is assigned from that acquisition coverage.
4. **Closed for the fixed simulator mapping:** `MotionHistory` stores velocities already projected through geometry/window rows fixed at construction. Its new `reconfigure` operation installs new rows and clears samples while preserving the measurement high-water mark. The experiment rejects unrecognized registration, window and calibration versions. The later runtime adapter must install changed geometry/window rows and clear history rather than merely resetting gamma.
5. **Closed in current source:** the initial experiment used a fixed feedforward path, omitting the production position feedback. It now uses `CartesianTrackOuterLoop`, the production TFF force/position selection, and a reference sampled at committed `t_ref`. Both path feedforward and position feedback enter the alpha-scaled path basis. Internal force-controller telemetry is not substituted for the final TFF command.
6. **Closed in current source:** nominal state/reference progress was committed before publication, and successful endpoint exit skipped the last send. The detached queue now accepts each candidate before its nominal and reference commit. Tests distinguish nominal, QP, sent and measured vectors, and verify no reference catch-up during alpha-zero or aborted exposure.
7. **Closed in current source:** coverage formerly credited only the first forward visit, and image coverage was credited before receipt. Valid forward intervals now form a union, so a valid revisit repairs an earlier gap. Images are credited only after receipt and source/version acceptance at their stored measured effective-time path; missing image intervals are bounded. Continuous mechanical/acoustic truth remains separately reported.
8. **Closed in current source:** matched rocking could violate final acceleration bounds after clipping. A final mechanical recheck now rejects such candidates. Delayed measured motion is checked independently, and completed as well as aborted runs retain stopping exposure and delivered-image drain.
9. **Closed:** an explicitly invalid measured-motion sample was silently omitted by `MotionHistory.add`, allowing interpolation between valid samples through an unavailable interval. The reproduced 5 ms invalid-span case now returns unavailable. Invalid samples clear interpolation history while retaining measurement ID/time fencing, and the new regression passes.
10. **Closed:** stopping-tail checks omitted measured acceleration. Stopping now decelerates commands within the same acceleration limits and checks measured acceleration, velocity and angle on every tail step. The completed ordinary step is also checked before tail entry, and residual measured motion above the declared stopping threshold aborts the result.
11. **Closed in current source:** design statistics inferred seeds from surviving records, silently dropping a wholly missing design seed. They now require the prescribed design seeds, with a regression. Acceptance already required its independent prescribed seeds.

Positive checks: scalar motion is registered from measured velocity, not proposed/sent commands; response consistency is explicitly a sign policy rather than a learned Jacobian or causal certificate. Receipt/effective-image times, motion bracketing, gap duration, large planar travel, duplicate observations, and version transitions are checked. Window rows now use the configured window centroids instead of unrelated nominal thirds.

The QP audit found no remaining concrete blocker in normalized objective units, full-geometry endpoint constraints, force-priority signs, hard mechanical precedence, zero-path behavior, feasible omega-interval diagnostics, or exported final-twist/subspace/progress certificates. Visual rows remain soft policies and do not become certificate claims. Two design-seed numerical stalls were captured in regressions. The final solver uses the declared physical variable/row normalization without ProxQP's additional Ruiz equilibration; increasing iteration limits alone had not reliably removed the second dual stall. Tolerances remain unchanged. Initial and cached-workspace regressions pass, and the affected complete shadow/seed 0/full_consistency replay now finishes with gaps, no solver failure and no measured mechanical violation.

Independent reviewer execution of plant/history, geometry and QP tests completed **39 passed in 32.95 s**, including the 100,000-tick exact transparent-path test. A second independent run of corrected plant/history, evaluation and experiment regressions completed **39 passed in 1.30 s**. After the final normalization correction, the reviewer reran **24 QP tests passed**, with the long test deselected; the owner's final `qp_solver_tests_normalization.log` records **25 passed in 44.72 s**, including 100,000 ticks, zero maximum twist error and zero downstream drift.

`smoke_v2` retains all 21 design-seed runs: 20 completed with gaps and the second numerical stall was retained as a failed run. `shadow_seed0_solver_normalization_regression.json` records its corrected complete replay: reference 3 s, measured 60 mm, mechanical coverage 100%, acoustic/image coverage 0%, and `complete_with_gaps`. Shadow force error remains worse than baseline in the smoke results; no efficacy acceptance is inferred from this implementation gate. Omega-interval diagnostics distinguish mechanical, force-priority and complete limits and show the added aperture restriction where active.

Final isolated-QP performance (`qp_solver_performance_normalization.json`) uses 4,000 repairs each with 0/96 external rows: p99 **1.027/0.765 ms**, maxima **1.899/1.066 ms**, zero failures and zero 5 ms overruns. Image work and interval diagnostics are excluded. The composed shadow replay separately reports p99 **4.442 ms**, maximum **14.732 ms**, and **two** calls exceeding 5 ms under shared host load. These measured software timings do not establish hard real-time operation.

The model document now describes general slope reactions/torque, external surface power, consistent tool/world kinematics and independently seeded scenario families. The source manifest has been expanded to cover the peirastic/rm75 Python dependencies, native source/header/build files, configuration and relevant library versions.

Final reviewed SHA-256 values:

```text
72a63d7291c6c44cdbdf4bf9400a58833715277a11e1be973893733801778fb2  peirastic/contact_qp/qp.py
ba1a02b90665308dfdbf7ef4b6bec1237951a3491398c3fd1c2804978eca71f2  peirastic/contact_qp/plant.py
718e2a6ac3e7bf087e7ce79add961d51bb6b03923ca9168abe65e482e3db8ee4  peirastic/contact_qp/history.py
8c0311f6320401cb0e19e790252bf732a53ee3cfb4b89a6daa7e3a44800181f7  peirastic/contact_qp/geometry.py
8318cf1ea4c5f940ae817ede06836067c7500864cab0be55d5f38dc1a6a85d79  peirastic/contact_qp/coverage.py
8668101ae99b2b1d0fc0cdd74268a8f46c6aa63141ea684dc4bdfb210617f12c  peirastic/contact_qp/evaluation.py
ade17c8dee92b2c03b1d6118a378d8208e05c22b993ea5cd534481b9faae0252  peirastic/apps/contact_qp_experiment.py
255ed6b844c0d7ed6c4fda22a32fba513480c9b939bbc86e46c01c7682a955a8  peirastic/realman8dof/force/contact_nominal.py
810089efc37775bb2bae7257487c66955a91bcf1353525cb3dd7fe8e33fe1148  peirastic/tests/test_contact_qp_solver.py
939bb18f114ab0dfe56adc72d5406455b2b839a81909dce24e6a149189a0cedf  peirastic/tests/test_contact_qp_plant_history.py
aaa797cd068e97e6399cd7b34663dc483293c96a6f71f387057809232b0da9a5  peirastic/tests/test_contact_qp_experiment.py
ed9f2fb6388908bf5bcdc175d74a3740f50764297d5968dd6b6a90f3431b4953  peirastic/tests/test_contact_qp_evaluation.py
4aaf88e4fa1d5fc5667c449b2bf1c278036761e32c1b46508b87384fb34cd3ef  MD/contact_qp/SIM_MODEL_V1.md
```

No held-out acceptance data or hardware was used. Synthetic dynamics, acoustic assumptions, and policy efficacy remain separate from real geometry and physical-port certification.
