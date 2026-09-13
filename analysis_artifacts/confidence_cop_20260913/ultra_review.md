# Independent ULTRA implementation review — 2026-09-13

Result: no remaining blocking issue found in the reviewed implementation after the angular-history correction below. This is a source and offline regression review; it is not a hardware release or evidence of improved ultrasound images.

Reviewed the actual confidence task, three-stage allocator, shared QP rows, active-mode proposal/publication transaction, torque-drive bypass, loading/scan budget source, feature contract, profile validation and capability gate. Existing default profile selection remains unchanged. The prior native inner-controller and QP-builder edits were not modified by this review.

## Blocking finding resolved

The new visual dynamics initially followed the successfully published final angular velocity, while the outer QP's angular slew row still followed its previous outer candidate. Those two histories can differ legitimately after the inner controller. The mismatch could make the new aperture and angular acceleration rows mutually infeasible.

A numerical reproduction with a 5 ms interval, 2 rad/s² angular acceleration bound, 0.20 rad/s previous outer velocity, 0.10 rad/s final velocity/next visual target, and ample energy deferred at the progress stage. Replacing angular history with 0.10 rad/s made the same problem feasible.

The runtime now projects the stored final angular vector into the current tool frame and uses its tool-y component for the new mode's outer angular slew history. Translational history remains in outer-command coordinates. This matches the visual dynamic state without feeding the rail-compensated final normal velocity into the loading controller. The real runtime regression `test_outer_angular_slew_reconciles_final_history_without_rail_feedback` verifies the case and passes.

## Verified design properties

- Each new image identity updates a held angular target. Reusing a frame evolves the held-target M/D response; it does not repeatedly add a frame-sized correction. Invalid, paused or unavailable evidence removes the image target through the existing dynamic and mechanical limits.
- Image mode bypasses the torque/Coulomb angular drive. Force gates the image request and retains the existing contact/force stop and mechanical limits. The confidence centroid is an image coordinate with an explicit response scale, not an identified physical angle or image Jacobian.
- The allocator fixes the feasible quality-progress optimum, then angular optimum, then normal-pairing optimum. Earlier decisions are eliminated algebraically from subsequent problems. There is no CoP/confidence blending weight.
- CoP is computed as `-My/Fz` in the restricted centered, tool-aligned, compressive +Z geometry, with finite/load/face validity checks. Pairing uses the full ongoing angular velocity: `vz = loading_requested + cp * omega`. Thus already-present angular motion does not erase its geometric normal preference.
- Accepted loading strips the full CoP component from the outer candidate and is bounded by the signed requested loading interval. The force controller receives this loading and accepted scan state; the angular task receives the final published angular vector. Abort does not commit proposed command state.
- Loading and scan authorization vectors exclude tool-y rotation and its paired normal motion. Their positive requested outward powers are accounted separately, checked for finite overflow, and summed. Full final six-axis model power is still reserved and settled under the existing logical epoch contract. Unknown/partial publication and expiration behavior remain enforced.
- New mode and centroid versions are explicit opt-in configuration/capability requirements. No tissue-parameter identification is introduced.

## Accepted limits that must remain explicit

The progress-first order can require angular deviation from even a zero image target when hard constraints couple progress and rotation. A zero target is not a hard zero-rotation command.

CoP pairing is an outer allocation preference. The final inner-controller model can differ; candidate/final pairing residuals are logged. The instantaneous pressure-weighted geometric argument does not establish constant force, improved acoustic coupling or a tissue stiffness center. CoP uses compensated filtered wrench, while energy uses the separately defined compensated raw wrench port.

The task source is requested loading/scan authorization, not measured or achieved task work. When achieved progress or loading is smaller, its authorized surplus can fund other final model motion. Unused authorization is discarded rather than added to the tank. Therefore absence of direct visual funding does not imply every visual movement must deplete the tank.

The budget certifies only the configured supplied command model. It does not certify physical passivity, actual robot work, inter-sample force peaks or device stopping tails. Offline replay with recorded images/wrenches cannot establish closed-loop image improvement or physical safety.

## Independent regression evidence

`146 passed, 1 deselected in 4.15 s` using the rm75 environment with pytest plugin autoload disabled, single-threaded BLAS, Python bytecode disabled and pytest cache disabled. Covered:

- `test_contact_qp_confidence_fusion.py`
- `test_contact_qp_priority_allocation.py`
- `test_contact_qp_centroid_features.py`
- `test_contact_qp_command_budget.py`
- `test_contact_qp_command_budget_active.py`
- `test_contact_qp_nominal.py`
- `test_contact_qp_runtime_config.py`

The sole deselected case was the 100,000-tick nominal equivalence test already running in the root's complete regression batch. An earlier duplicate review run was interrupted during that long case. No completion or result is claimed for it here. The 49-recording replay is owned and reported separately.

An initial test invocation used the Genesis analysis environment, which lacks ProxSuite (and initially loaded incompatible unrelated pytest plugins). The review corrected the environment to rm75 and disabled plugin autoload. Those import/environment errors are not reported as algorithm failures.

## Additional replay evidence review

Independently read `extract.py`, `replay.py` and `report.py`. Their replay keeps recorded pose, wrench, path, alpha preference and image evidence fixed, uses recorded nominal Z as a loading proxy, and treats admitted candidates as ideal successful inner publications. It does not exercise the complete new normal-force feedback loop or simulate the changed measured trajectory. The explicit counterfactual segment continuation is an offline experiment, not runtime recovery.

Found and reported these replay/report corrections to their owner:

- `source_t_s + source.age_s` reconstructs source-ingress time, preceding outer sampling; it is not the exact proposal clock. For reviewed proposals, `exported_task_valid_until_s - certificate_horizon_s` recovers the sample clock because the active runtime supplies no external mechanical rows that could shorten expiry. Rejected-review `created_time_s` is direct evidence. Missing-review fallback must be marked approximate. Two independently checked scans showed median ingress-to-sample-start proxies of 0.390 and 0.344 ms, with maxima of 2.034 and 1.771 ms. No recorded-used frame arrived between those clocks in their combined 10,135 controls.
- Counterfactual restart must persistently clear stored previous velocity/rotation, not only the local previous variable; otherwise a deferred first proposal can restore pre-restart translational history on the next row.
- Replay provenance must include the replay implementation/semantics, not only imported controller files, before reusing cached results.
- Report wording must distinguish exclusion of direct new angular/CoP source vectors from possible funding through the historical normal proxy and requested-authorization surplus. Per-row `dt` sums are sampled-time weights, not directly measured image hold durations.

Verified the owner's clock reconstruction, persistent restart-state reset, script-hash reuse validation and report wording corrections after implementation. Also verified the report-only guard: aggregation selects the current replay script hash and expected schema before requiring 49 results, preventing mixed replay semantics from passing a file-count check.

An independent cache-integrity pass covered all 49 manifests, 241,127 control rows and 31,646 cached frame rows available at read time. It found no duplicate control IDs, frame/publisher collisions, duplicate cached frames, cache-key mismatches or manifest/control-count mismatches.

At the source check, all 24 frozen contact-QP files matched their snapshot manifest and current live counterparts, all available extraction manifests used the current feature-source hash, and completed result controller hashes were consistent. Full corrected replay completion and numerical results remain the replay owner's responsibility; they are not included in the 146-test implementation result above.

## Exact-priority fast-path supplement

Reviewed the final optimization in `priority_allocation.py` and its `qp.py` integration. When the point `alpha = alpha_preferred`, `omega = visual_target`, `vz = loading_requested + cp * omega` is feasible, all three nonnegative squared objectives attain their global minimum of zero simultaneously. Returning that point is therefore the same lexicographic optimization law. A zero path still fixes the unobservable alpha to zero. Cases with energy epigraph auxiliary variables retain the numerical solver.

The final helper checks finite coordinates, normalized constraints and raw Cartesian hard rows, using `min(solver_tolerance, feasibility_tolerance)`. It additionally requires direct full-energy admissibility with zero power and velocity tolerance. Any failure uses the original bounded numerical fallback. The original initial/final deadline fences, final constraint verification, certificate construction and publication budget admission remain in place.

The review caught and resolved a diagnostic regression: setting `transparent=True` for this shortcut incorrectly changed a modified visual/CoP command from REPAIR to NOMINAL. The final code keeps the prior status and uses a separate `exact_priority_fast_path` flag with `exact_priority_optimum` solver status. It also incorporates the review's tolerance-order guard and zero-tolerance energy eligibility, avoiding a shortcut result that would fail the stricter downstream admission when numerical fallback remains available.

Independent checks after optimization:

- 141 tests passed in 1.65 s across priority allocation, confidence runtime, centroid, command-budget unit/runtime and configuration.
- After the final tolerance guards, the focused priority/runtime suite passed 51 tests in 0.76 s, including reversed tolerance ordering, a tiny negative energy margin, exact-versus-forced-numerical status equivalence and both deadline fences.
- Forty randomized feasible problems with raw mechanical rows, full energy constraints, both angular policies and zero/nonzero paths matched the forced numerical path's status and admitted every exact candidate under zero energy tolerance. The largest absolute coordinate difference in the six-component twist was 1.866e-7; the largest alpha difference was 6.293e-6. The numerical path is approximate, so bitwise equality or a universal 1e-8 difference is not established.

Final reviewed SHA-256: `priority_allocation.py` = `9bcee00a5476553396757fb7f28179e5eff73583a4c0d572d8f55c25010c9767`; `qp.py` = `8bbddf12b3358908c719979b0ee7245159f85a86d7e139c05397e80bc64f93c4`.

No remaining blocking issue was found in this optimization. The original three-QP replay snapshot remains a separate reference. Its comparison against a full replay using the optimized source, including candidate, energy and timing differences, is reported separately; this review does not establish a hardware-rate performance certificate.
