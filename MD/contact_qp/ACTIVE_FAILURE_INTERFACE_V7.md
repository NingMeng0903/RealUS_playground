# Active publication interface correction v7

Scope: original 4 N target, force-priority band, mechanical limits, IK/native/rail implementation and 10 ms review horizon are unchanged. No device connection or service restart was performed.

## Observed failures

- Attempt 001: stale force reported during preparation, before any contact-QP recording. There is no failed-step evidence to attribute this preparation failure to the active solver. Similar preparation failures also occurred with baseline.
- Attempt 002: one successful publication; next QP reported relative baseline/mechanical conflict. Failed QP input was not logged, so an exact counterfactual reconstruction is unavailable.
- Attempt 003: 946 successful publications, then final review rejected candidate 947 with energy disabled. The existing review can reject nonfinite payload/time or its 10 ms deadline; chronology supports expiry, but the old event did not record the exact rejection reason. The rejection still stops the original path. This patch does not claim to resolve the underlying timing outlier.
- Attempt 004: 3357 successful publications, then motion subspace conflict. Last successful outer command Y was 4.60409697 mm/s, whereas the final rail-compensated payload model Y was 9.67398624 mm/s. The adapter incorrectly reused this model as the next outer command-slew history. These are different velocity definitions. The failed next input was not logged.

Publication counts mean successful software sends, not confirmed physical completion. Reference time reached 0.6208108 s in attempt 003 and 11.4120069 s in attempt 004; neither completed the scan.

The later scan-UI `Stale or missing force feedback` message does not identify the initial failure in attempts 003/004. The controller-derived force stream ended at source times 1115531.137527842 / 1115561.479474658; TCP samples continued to 1115531.236930148 / 1115561.584198335. QP rejection was logged between those times. This is consistent with force publication ceasing when active control stopped, rather than evidence that the force sensor first failed. The untimestamped console FA24/Modbus errors cannot all be assigned a causal order from these records and remain a separate unresolved execution/communication issue. Exact event excerpts and raw-stream ending times are in [active001_failure_evidence_v7.json](active001_failure_evidence_v7.json).

## Minimal corrections

The successful outer QP command now supplies only the outer command-slew history, rotated from its old TCP orientation into the current TCP components. The same accepted outer command is committed to the original nominal command integrators; feeding the final payload model there would reintroduce a different normal slew origin. The separate final payload model remains available to reference-progress accounting, energy review and logs. Actual measurement feedback is unchanged. Outer slew is a command policy, not a claim about measured physical acceleration; original inner mechanical protection remains independent.

QP acceleration rows now use explicit `acceleration_dt_s` (defaulting to the existing `dt_s` for old callers). Active supplies actual control elapsed time, matching the original nominal law. Future command hold and reference grants remain bounded at 5 ms; certificate horizon remains 10 ms. Previously a 6 ms nominal normal slew could be tested against a 5 ms box while force-priority simultaneously forbade reducing that nominal value.

Failed preparations now log nominal/path/previous velocities, force, actual/hold times, angle and solver diagnostics. Failed final reviews log their specific reason and creation/expiry/review times. No failed review is silently bypassed.

## Evidence and limits

`test_contact_qp_active_history.py` covers: a causal counterexample using the actual 004 final successful velocities, independent actual-dt versus hold-dt conflict, active commit/abort and frame transformation, unchanged expiry rejection with diagnostics, and two real active adapters over four identical measured cycles with different normal payload-model residuals. Their committed nominal states and subsequent QP requests remain identical; rejection does not commit. A separate .001 outer / .002 model / .006 next nominal counterexample confirms why both nominal and outer command histories must use the accepted outer command. It does not pretend the unrecorded next input was recovered.

rm75 environment, existing project/PIN/cmeel paths:

```
python -m pytest -q peirastic/tests/test_contact_qp_active_history.py peirastic/tests/test_contact_qp_active.py peirastic/tests/test_contact_qp_solver.py -k 'not 100000'
```

Final result after the nominal-history correction: 43 passed, 1 deselected in 1.42 s. The deselected 100,000-cycle test had passed in the earlier 42-test run; the final targeted run focuses on the changed publication/solver interfaces.

Recorded ingress-to-control-log times (observer + nominal + QP, not isolated solver timings), P50/P95/P99/max in milliseconds:

| Attempt | Ingress → outer log | Outer log → final review |
|---|---|---|
| 003 | 2.425 / 2.904 / 3.351 / 5.064 | 1.807 / 2.603 / 4.490 / 6.083 |
| 004 | 2.373 / 2.860 / 3.179 / 4.848 | 1.839 / 2.636 / 4.363 / 6.293 |

Rejected 003 sample 947 took about 2.305 ms ingress-to-outer-log and 8.402 ms outer-log-to-abort. This does not establish a slow QP; the old log cannot separate IK/postprocessing/OS scheduling. No timing threshold was increased to hide it. Real active performance and timing after this correction still require actual execution evidence.

## Final regression and review

- GPT-6 medium implementation; GPT-6 ultra independent scoped interface PASS, with 43 targeted tests independently passing. See [review_active_interface_v7.md](review_active_interface_v7.md).
- Final source: `python -m pytest -q peirastic/tests -k 'not 100000'`: 584 passed, 1 native-startup failure, 3 previously exercised long tests deselected, 22.06 s. The original result is retained in `peirastic_regression_active_v7.xml`.
- The single native-startup test failed again in the root sandbox. The sandbox also denied `ptrace`; the independent worker environment passed. The exact test was then run with tool-approved execution outside the root sandbox: **1 passed in 1.38 s**, with no source or test change (`native_handoff_host_recheck_active_v7.xml`). This closes the environment-dependent test failure, not the separate real active timing problem. The original failing recheck remains in `native_handoff_recheck_active_v7.xml`.
- Original force/contact/observer/Ke/guard/TDPA eight-suite regression: **46 passed in 11.79 s**, `force_regression_active_v7.xml`.
- Active configuration offline validation passed (`hardware_connected: false`). Source-scoped whitespace validation passed. No robot, rail or controller service was started or restarted by the agent.

Commands used the existing rm75 Python, project/rm75/cmeel import paths and single-thread BLAS. The isolated native test was `python -m pytest -q peirastic/tests/test_cartesian_ptp_handoff.py::test_replanned_standoff_can_enter_native_hold_with_same_constraints`.
