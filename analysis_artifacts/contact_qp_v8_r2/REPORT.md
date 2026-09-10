# Frozen synthetic v7/v8 comparison

**Observed r2 outcome:** all 20 new-seed units completed, with both shadow runs per policy retaining acoustic gaps. The two old-seed v8 shadow regressions also completed with gaps, resolving their initial angle-limit aborts. In new-seed shadow, v8 versus v7 mean force RMSE fell by 0.04181 N and angular travel by 0.00767 rad, while peak force rose by 0.03798 N. Acoustic bad path remained 60 mm. This demonstrates bounded behavior for these cases, not acoustic repair efficacy. Actual v8 shadow travel was about 0.0547 rad including delayed motion and stopping tails, slightly above the 3° permission threshold; that threshold is not a physical hard travel guarantee.

Twenty prescribed units: seeds 801/802, five scenarios, two policies. The unchanged shared nominal controller targets 4 N. Both policies use the identical v3 FeatureExtractor at 145×100, cmin 0.8; path 60 mm at 5 mm/s, 5 ms controls. Finite-area mechanics, actuator delay, image delay, and stop tails are retained. See protocol.json and compare.py.

**Scope:** r2 introduces bounded repair attempts after the retained initial shadow failures. Seeds 801/802 are new fixed runs, but this is design validation after a controller revision, not independent efficacy acceptance. The original left/right gap runs at 5 mm/s kept image quality above cmin under the shared nominal controller: they were already-coupled conditions and did not test acoustic improvement when repair was needed. Persistent shadow remains physically unrepairable; the plant, speed, features, and thresholds are unchanged.

This is a small exploratory simulation, not acceptance evidence, a real-image causal result, or physical certification. No hardware or controller service was connected. Closed-loop energy admission is **off** because the reused run_case has no energy input. energy_sweep.json separately checks instantaneous admission with the existing single full-port constraint and known synthetic full-six-dimensional wrenches; it does not settle or simulate a second ledger.

Force RMSE and peak include the recorded scan and stopping tail; >4.5 N is sampled measured-force exposure at 5 ms. Angular travel is integrated measured rocking, including the tail. Acoustic bad path is latent plant truth, not an image contact label. Image quality is the scan mean of the extracted left/center/right values, including held observations. Completion with gaps is distinct from clean completion. Deadline counts are the existing measured controller block >5 ms; image extraction runs outside that timer. Concurrent offline Python timing is not a real-time certification.

| Scenario / seed | Policy | Status / reason | Force RMSE N | Peak N | >4.5 N s | Angular rad | Acoustic bad mm | Quality L/C/R | p99 ms | >5 ms |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| delayed_execution / 801 | differential_repair_v8 | complete | 0.1285 | 4.1847 | 0.0000 | 0.0445 | 0.0000 | 0.997/0.996/0.996 | 2.1845 | 0 |
| delayed_execution / 801 | legacy_v7 | complete | 0.1285 | 4.1847 | 0.0000 | 0.0445 | 0.0000 | 0.997/0.996/0.996 | 2.2320 | 0 |
| delayed_execution / 802 | differential_repair_v8 | complete | 0.1294 | 4.1860 | 0.0000 | 0.0395 | 0.0000 | 0.996/0.996/0.996 | 2.0338 | 0 |
| delayed_execution / 802 | legacy_v7 | complete | 0.1294 | 4.1860 | 0.0000 | 0.0395 | 0.0000 | 0.996/0.996/0.996 | 2.4096 | 0 |
| healthy / 801 | differential_repair_v8 | complete | 0.0621 | 3.9538 | 0.0000 | 0.0000 | 0.0000 | 0.996/0.996/0.996 | 2.1828 | 1 |
| healthy / 801 | legacy_v7 | complete | 0.0621 | 3.9538 | 0.0000 | 0.0000 | 0.0000 | 0.996/0.996/0.996 | 1.9926 | 1 |
| healthy / 802 | differential_repair_v8 | complete | 0.0612 | 3.9547 | 0.0000 | 0.0000 | 0.0000 | 0.997/0.996/0.996 | 2.0999 | 0 |
| healthy / 802 | legacy_v7 | complete | 0.0612 | 3.9547 | 0.0000 | 0.0000 | 0.0000 | 0.997/0.996/0.996 | 2.3265 | 0 |
| left_gap / 801 | differential_repair_v8 | complete | 0.1304 | 4.1853 | 0.0000 | 0.0405 | 0.0000 | 0.996/0.997/0.996 | 2.1750 | 0 |
| left_gap / 801 | legacy_v7 | complete | 0.1304 | 4.1853 | 0.0000 | 0.0405 | 0.0000 | 0.996/0.997/0.996 | 2.2265 | 0 |
| left_gap / 802 | differential_repair_v8 | complete | 0.1315 | 4.1868 | 0.0000 | 0.0405 | 0.0000 | 0.996/0.996/0.996 | 2.1007 | 0 |
| left_gap / 802 | legacy_v7 | complete | 0.1315 | 4.1868 | 0.0000 | 0.0405 | 0.0000 | 0.996/0.996/0.996 | 2.2152 | 0 |
| right_gap / 801 | differential_repair_v8 | complete | 0.1322 | 4.1881 | 0.0000 | 0.0315 | 0.0000 | 0.997/0.996/0.997 | 2.4898 | 0 |
| right_gap / 801 | legacy_v7 | complete | 0.1322 | 4.1881 | 0.0000 | 0.0315 | 0.0000 | 0.997/0.996/0.997 | 2.2678 | 0 |
| right_gap / 802 | differential_repair_v8 | complete | 0.1316 | 4.1871 | 0.0000 | 0.0333 | 0.0000 | 0.997/0.997/0.997 | 2.2224 | 0 |
| right_gap / 802 | legacy_v7 | complete | 0.1316 | 4.1871 | 0.0000 | 0.0333 | 0.0000 | 0.997/0.997/0.997 | 2.0302 | 0 |
| shadow / 801 | differential_repair_v8 | complete_with_gaps | 0.0771 | 4.1822 | 0.0000 | 0.0548 | 60.0000 | 0.272/0.801/0.994 | 2.6412 | 4 |
| shadow / 801 | legacy_v7 | complete_with_gaps | 0.1198 | 4.1448 | 0.0000 | 0.0668 | 60.0000 | 0.272/0.801/0.994 | 2.7542 | 2 |
| shadow / 802 | differential_repair_v8 | complete_with_gaps | 0.0778 | 4.1824 | 0.0000 | 0.0547 | 60.0000 | 0.312/0.813/0.993 | 2.5055 | 0 |
| shadow / 802 | legacy_v7 | complete_with_gaps | 0.1187 | 4.1438 | 0.0000 | 0.0581 | 60.0000 | 0.312/0.813/0.993 | 2.4620 | 0 |

Paired changes below are v8 minus v7, averaged over the two fixed seeds. They are descriptive; two seeds do not establish generalization.

| Scenario | Δ RMSE N | Δ peak N | Δ >4.5 N s | Δ angular rad | Δ acoustic bad mm |
|---|---:|---:|---:|---:|---:|
| delayed_execution | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 |
| healthy | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 |
| left_gap | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 |
| right_gap | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 |
| shadow | -0.04181 | +0.03798 | +0.00000 | -0.00767 | +0.00000 |

Completed records: 20/20. Status counts: `{"legacy_v7": {"complete": 8, "complete_with_gaps": 2}, "differential_repair_v8": {"complete": 8, "complete_with_gaps": 2}}`.

Source files changed during execution: `["peirastic/realman8dof/modes/contact_active.py"]`.

The sole change was the separately reviewed real execution adapter's `repair_execution_enabled` contact-present guard. This adapter is not invoked by the reused detached `run_case`; QP, repair episode, repair policy, nominal controller, features and plant hashes did not change. `frozen_source.zip` preserves the main run's actual pre-change snapshot and does not claim to archive the final adapter revision. The subsequent old-seed regression archive includes the updated adapter and its full source list remained unchanged through those runs.

Separate energy sweep: 144/144 admitted; 0 admission violations. Minimum work margin -4.86e-14 J (numerical tolerance 1e-9 J). Ample-budget maximum deviation from the energy-off optimum 8.8e-11. Zero-budget alpha spans 0.0000–0.9055; net full-port cancellation can permit nonzero motion. These are static snapshots, not closed-loop energy histories.

Old-seed shadow regression (701/702) below is a revisit of the initial failures, not independent acceptance.

| Seed | Policy | Status | Force RMSE N | Peak N | Angular rad | Acoustic bad mm |
|---|---|---|---:|---:|---:|---:|
| 701 | differential_repair_v8 | complete_with_gaps | 0.0778 | 4.1820 | 0.0550 | 60.0000 |
| 701 | legacy_v7 | complete_with_gaps | 0.1189 | 4.1445 | 0.0587 | 60.0000 |
| 702 | differential_repair_v8 | complete_with_gaps | 0.0788 | 4.1836 | 0.0546 | 60.0000 |
| 702 | legacy_v7 | complete_with_gaps | 0.1188 | 4.1470 | 0.0510 | 60.0000 |

Regression source files changed during execution: `[]`.
