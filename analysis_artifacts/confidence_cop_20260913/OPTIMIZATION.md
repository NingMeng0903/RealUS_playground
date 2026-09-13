# Analytic fast path versus the frozen numerical reference

Both calculations use all 49 saved scans, the same exact proposal clocks and frozen inputs. The original source snapshot/results remain unchanged. The optimized calculation has a separate snapshot and output tree. Candidate equality is measured, not assumed: the numerical QP reference has solver tolerances, whereas jointly feasible exact targets have zero lexicographic error.

Velocity, measured-angle, aperture and consecutive moving-command slew constraints are independently rechecked. The first moving command’s prior approach-state slew is not reconstructed by this comparison; the actual replay QP still checked it. Alpha is reconstructed from the fixed non-Z/non-Y path projection. Energy admissibility and model admission are checked by each replay’s actual budget; these calculations retain the zero-compute-latency publication and frozen-loading limitations described in REPORT.md.

- confidence_angular_v1: maximum twist-component difference 1.57776097e-09; omega difference 1.57776097e-09 rad/s; alpha difference 9.42955428e-08; tank-trace difference 4.16342238e-11 J; final-tank difference 3.98474725e-11 J. Acceptance differences: 0. Mean measured compute 2.398→1.805 ms (1.33×). Maximum observed optimized compute 2.813 ms.
  Independent maximum limit excesses: velocity 0, slew 0, angle 0, aperture 0; all within the 1e-8 numerical validation tolerance.
- confidence_cop_v1: maximum twist-component difference 1.57776097e-09; omega difference 1.57776097e-09 rad/s; alpha difference 9.42955428e-08; tank-trace difference 2.19747415e-11 J; final-tank difference 2.19747415e-11 J. Acceptance differences: 0. Mean measured compute 2.396→1.807 ms (1.33×). Maximum observed optimized compute 11.427 ms.
  Independent maximum limit excesses: velocity 1.19e-13, slew 0, angle 0, aperture 0; all within the 1e-8 numerical validation tolerance.

The timing comparison is observed under concurrent replay scheduling. Python/bookkeeping is included; native IK, transport, and complete controller-cycle latency are absent. The smaller mean does not imply a smaller observed worst case or fewer live deadline failures. Optimized CoP has an observed 11.427 ms outlier; its timing report retains conservative deadline-risk bounds rather than inventing a hardware stopping rate. This is not a worst-case realtime guarantee.
