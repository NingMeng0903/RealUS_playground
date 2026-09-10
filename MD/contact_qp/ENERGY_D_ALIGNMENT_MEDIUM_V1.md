# Single-tank dissipation and measured interval connection

Implemented without native/IK/rail-worker changes or hardware startup:

- `PortInterval.dissipation_work_j` defaults to zero, rejects negative/nonfinite/bool input. `EnergyLedger.settle` atomically debits **net port output work + dissipation work**, logs both terms and total, then releases only corresponding measured reservations. Arithmetic failure preserves balance and reservations. Existing D=0 behavior remains.
- `RuntimeEnergy.bind_dissipation` freezes the exact `energy.constraint.contact_map` / `damping` used by the QP. A different model is rejected. `observe_interval` consumes a complete measured interval directly; it does not reinterpret interval-average velocity as a new instantaneous sample. For monitoring, D is estimated as `dt * sum(d * (Pc @ Vmean)^2)`. This is explicitly not an upper bound for unresolved out-and-back motion. Conditional physical certification would require an actual full-interval dissipation bound, for example `dt * sum(d * rbar^2)` with validated `|r(t)| <= rbar` throughout the interval.
- `MeasuredPortAligner` stores bounded histories of original fresh UDP timestamps, same-packet measured arm positions, compensated unfiltered TCP wrench, and original raw rail encoder samples. Rail samples must bracket both interval endpoints; no extrapolation. Rail knots split the W/arm interval into subintervals. Each uses interpolated measured positions, `J(qmid) * delta_q/dt`, and consistent TCP/body-axis wrench endpoints. Source identity, interpolation weights, physical interval endpoints, estimation method and uncertified status are retained.
- `ContactQpOuter.observe_measured_port` receives this raw measured data before command generation, independently of command publication/commit. It settles the same ledger, including D, even in pure monitoring mode. The loop's older SDK/rail-predictor diagnostic twist is not used for this settlement. Missing data/epoch changes/timeouts do not refund outstanding motion exposure. Stopping tails remain unsettled unless corresponding actual measurements are received.

`physical_w_checked` remains an actual deployment binding requirement, not an inferred flag. Numerical interpolation validity is not a verified physical error bound. This closes the monitor data path; it does not claim physical passivity certification. `energy_constraint_enabled: false` still means no energy constraint modifies the command, while a configured energy ledger can monitor and settle measured intervals.

Validation command (rm75 Python environment, repository and rm75_control on PYTHONPATH, cmeel library paths as in earlier reports):

```sh
python -m pytest -q peirastic/tests/test_contact_qp_port_alignment.py peirastic/tests/test_contact_qp_energy.py peirastic/tests/test_contact_qp_runtime_energy.py peirastic/tests/test_contact_qp_active.py
```

Result: **66 passed**. Includes zero-port-work/nonzero-D depletion, D=0 recovery regression, total-charge overflow preserving reservations, actual active measurement-method to ledger connection without commands, asynchronous rail bracketing/splitting, no extrapolation, no duplicate settlement, source/rail gaps and retained missing-data liabilities. No new acoustic simulation or physical scan was run.
