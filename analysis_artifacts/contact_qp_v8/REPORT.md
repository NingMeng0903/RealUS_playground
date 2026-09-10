# Frozen synthetic v7/v8 comparison

Twenty prescribed units: seeds 701/702, five scenarios, two policies. The unchanged shared nominal controller targets 4 N. Both policies use the identical v3 FeatureExtractor at 145×100, cmin 0.8; path 60 mm at 5 mm/s, 5 ms controls. Finite-area mechanics, actuator delay, image delay, and stop tails are retained. See protocol.json and compare.py.

**Observed limitation:** both persistent-shadow v8 runs abort on the measured 0.35 rad angle limit, whereas v7 finishes with acoustic gaps. The fixed shadow cannot be repaired by pressing or rocking; v8 accumulates greater actual rocking and force error without reducing latent acoustic bad path. Delayed actuation carries measured angle beyond the command-side bound. The other four scenarios maintain image quality above cmin under the shared nominal controller and do not distinguish the policies. These frozen results do not demonstrate a closed-loop v8 benefit. No controller or experimental parameter was tuned after observing this result.

This is a small exploratory simulation, not acceptance evidence, a real-image causal result, or physical certification. No hardware or controller service was connected. Closed-loop energy admission is **off** because the reused run_case has no energy input. energy_sweep.json separately checks instantaneous admission with the existing single full-port constraint and known synthetic full-six-dimensional wrenches; it does not settle or simulate a second ledger.

Force RMSE and peak include the recorded scan and stopping tail; >4.5 N is sampled measured-force exposure at 5 ms. Angular travel is integrated measured rocking, including the tail. Acoustic bad path is latent plant truth, not an image contact label. Image quality is the scan mean of the extracted left/center/right values, including held observations. Completion with gaps is distinct from clean completion. Deadline counts are the existing measured controller block >5 ms; image extraction runs outside that timer. Concurrent offline Python timing is not a real-time certification.

| Scenario / seed | Policy | Status / reason | Force RMSE N | Peak N | >4.5 N s | Angular rad | Acoustic bad mm | Quality L/C/R | p99 ms | >5 ms |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| delayed_execution / 701 | differential_repair_v8 | complete | 0.1301 | 4.1865 | 0.0000 | 0.0394 | 0.0000 | 0.996/0.997/0.996 | 2.3436 | 0 |
| delayed_execution / 701 | legacy_v7 | complete | 0.1301 | 4.1865 | 0.0000 | 0.0394 | 0.0000 | 0.996/0.997/0.996 | 2.1988 | 0 |
| delayed_execution / 702 | differential_repair_v8 | complete | 0.1306 | 4.1875 | 0.0000 | 0.0306 | 0.0000 | 0.996/0.996/0.997 | 2.2893 | 0 |
| delayed_execution / 702 | legacy_v7 | complete | 0.1306 | 4.1875 | 0.0000 | 0.0306 | 0.0000 | 0.996/0.996/0.997 | 2.2305 | 0 |
| healthy / 701 | differential_repair_v8 | complete | 0.0620 | 3.9539 | 0.0000 | 0.0000 | 0.0000 | 0.996/0.996/0.996 | 2.2518 | 1 |
| healthy / 701 | legacy_v7 | complete | 0.0620 | 3.9539 | 0.0000 | 0.0000 | 0.0000 | 0.996/0.996/0.996 | 2.1998 | 1 |
| healthy / 702 | differential_repair_v8 | complete | 0.0623 | 3.9536 | 0.0000 | 0.0000 | 0.0000 | 0.996/0.996/0.996 | 2.1240 | 0 |
| healthy / 702 | legacy_v7 | complete | 0.0623 | 3.9536 | 0.0000 | 0.0000 | 0.0000 | 0.996/0.996/0.996 | 2.4353 | 0 |
| left_gap / 701 | differential_repair_v8 | complete | 0.1323 | 4.1883 | 0.0000 | 0.0358 | 0.0000 | 0.996/0.996/0.996 | 2.0902 | 0 |
| left_gap / 701 | legacy_v7 | complete | 0.1323 | 4.1883 | 0.0000 | 0.0358 | 0.0000 | 0.996/0.996/0.996 | 2.1447 | 0 |
| left_gap / 702 | differential_repair_v8 | complete | 0.1316 | 4.1872 | 0.0000 | 0.0344 | 0.0000 | 0.997/0.997/0.997 | 2.1797 | 0 |
| left_gap / 702 | legacy_v7 | complete | 0.1316 | 4.1872 | 0.0000 | 0.0344 | 0.0000 | 0.997/0.997/0.997 | 2.7322 | 0 |
| right_gap / 701 | differential_repair_v8 | complete | 0.1304 | 4.1854 | 0.0000 | 0.0410 | 0.0000 | 0.997/0.996/0.997 | 2.3053 | 0 |
| right_gap / 701 | legacy_v7 | complete | 0.1304 | 4.1854 | 0.0000 | 0.0410 | 0.0000 | 0.997/0.996/0.997 | 2.2858 | 0 |
| right_gap / 702 | differential_repair_v8 | complete | 0.1316 | 4.1868 | 0.0000 | 0.0440 | 0.0000 | 0.996/0.996/0.996 | 2.2261 | 0 |
| right_gap / 702 | legacy_v7 | complete | 0.1316 | 4.1868 | 0.0000 | 0.0440 | 0.0000 | 0.996/0.996/0.996 | 2.1689 | 0 |
| shadow / 701 | differential_repair_v8 | aborted / actual_mechanical_limit | 0.2043 | 4.2706 | 0.0000 | 0.3502 | 60.0000 | 0.314/0.813/0.880 | 2.6248 | 0 |
| shadow / 701 | legacy_v7 | complete_with_gaps | 0.1189 | 4.1445 | 0.0000 | 0.0587 | 60.0000 | 0.301/0.810/0.993 | 2.5825 | 0 |
| shadow / 702 | differential_repair_v8 | aborted / actual_mechanical_limit | 0.2090 | 4.2717 | 0.0000 | 0.3502 | 60.0000 | 0.327/0.817/0.864 | 3.1626 | 0 |
| shadow / 702 | legacy_v7 | complete_with_gaps | 0.1188 | 4.1470 | 0.0000 | 0.0510 | 60.0000 | 0.317/0.815/0.993 | 2.6094 | 0 |

Paired changes below are v8 minus v7, averaged over the two fixed seeds. They are descriptive; two seeds do not establish generalization.

| Scenario | Δ RMSE N | Δ peak N | Δ >4.5 N s | Δ angular rad | Δ acoustic bad mm |
|---|---:|---:|---:|---:|---:|
| delayed_execution | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 |
| healthy | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 |
| left_gap | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 |
| right_gap | +0.00000 | +0.00000 | +0.00000 | +0.00000 | +0.00000 |
| shadow | +0.08776 | +0.12538 | +0.00000 | +0.29534 | +0.00000 |

Completed records: 20/20. Status counts: `{"legacy_v7": {"aborted": 0, "complete": 8, "complete_with_gaps": 2}, "differential_repair_v8": {"aborted": 2, "complete": 8, "complete_with_gaps": 0}}`.

Source files changed during execution: `[]`.

Separate energy sweep: 144/144 admitted; 0 admission violations. Minimum work margin -4.86e-14 J (numerical tolerance 1e-9 J). Ample-budget maximum deviation from the energy-off optimum 8.8e-11. Zero-budget alpha spans 0.0000–0.9055; net full-port cancellation can permit nonzero motion. These are static snapshots, not closed-loop energy histories.
