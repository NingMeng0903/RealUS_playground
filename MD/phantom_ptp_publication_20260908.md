# Phantom PTP to HOLD failure, 2026-09-08

The `20260908_054226_368190` scan reached the standoff within 0.1 mm. The
script then found Window A already in ESTOP at the pre-contact guard, before
issuing force-control configuration or HFPC. The running controller had no
CSV enabled. Native input/output SHM was read as bytes, without attaching a
client or issuing commands; its measured FK matches the reported standoff.

## Evidence

The retained [input](phantom_publication_20260908/in.json) and
[output](phantom_publication_20260908/out.json) show:

- QP1 `primal_infeasible`, QP2 `not_run`, native solve 2.331835 ms.
- J4 measured 2.3539254769 rad (134.8700°).
- Runtime upper limit 2.356 rad minus the configured 0.0052359878 rad margin
  gives 2.3507640122 rad (134.6889°).
- `(effective_upper - measured_J4) / 0.005 = -0.6322929399 rad/s`, exactly
  equal to the retained J4 `box_lo` and `box_hi`.
- Previous J4 command velocity was −0.003159 rad/s; the collapsed J4 box is
  an abrupt required retreat. Four collision CBF rows were active. Their
  individual coefficients were not retained, so the exact conflicting pair
  cannot be inferred from this snapshot alone.

`publication_infeasible` is an umbrella label for rejected native outputs;
it does not distinguish QP1 infeasibility from final publication validation.
`box_excess_max` here compares the fallback zero command with the velocity
box; it does not show that the original box endpoints crossed. This is
distinct from `native_timeout`, and provides no evidence of SHM corruption.

## Change

The old SRS selection accepted any physical-limit-valid goal. Replaying the
saved TCP goal and retained posture attractor selected J4 = 134.8289°, already
outside the online position margin. Cartesian PTP now passes the runtime
lower/upper limits minus position margins into SRS enumeration and filters
each candidate before ranking. Explicit Cartesian `q_target` overrides are
checked as well. Shared kinematics limits and online safety settings are
unchanged.

The [replacement IK](phantom_publication_20260908/replanned.json) reaches the
same TCP with rail 550 mm and J4 131.0518°, leaving 3.6370° to the effective
upper bound. The scanner default `--ptp-v` is now 0.10, previously 0.30.

Rejected steps now print QP1/QP2 status, solve time, CBF count and any
collapsed joint velocity bounds after the stop requests have been issued.
The formatter performs no I/O, and diagnostic errors cannot interrupt the
stop path. Fault categories and reset behavior remain unchanged.

## Verification

The captured command/measurement pair reproduces native failure in a private
UUID SHM test process. The replacement endpoint completes 100 native HOLD
ticks with certified hard residual at most 1e-5 and no latched solver fault.
Tests use ideal feedback and cores 6/8, separate from the running controller.

101 samples of the new joint PTP interpolation were checked against the
configured self-collision geometry. The smallest sampled pair distance was
23.697 mm (`link_2_0` / `link_4_0`), above the configured 10 mm threshold;
this is a sampled check, not a continuous-path clearance certificate.
See [sample results](phantom_publication_20260908/ptp_collision_samples.json).

API, SRS IK, candidate-bound, scanner workflow, recorded handoff, publication
guard and fault-diagnostic regressions passed (164 distinct tests). No
hardware motion, restart, reset, or contact was issued.

Restart Window A to load the planner change, using `--log-csv` for a complete
record. Existing ESTOP remains subject to the normal recovery procedure.
