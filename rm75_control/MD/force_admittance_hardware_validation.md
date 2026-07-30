# Hardware validation — force admittance redesign

Offline regressions cover transparency, approach decoupling, and stiffness
transition envelopes.  Hardware confirmation is still required before calling
the redesign done on the robot.

## Matrix

| Surface | Setpoints | Extra |
|---|---|---|
| Soft phantom | 1 N, 2 N, 5 N | hand-push with desired=0 |
| Hard phantom | 1 N, 2 N, 5 N | same |
| Rib-like edge | 2 N scan across edge | watch bounce / contact losses |

## Capture

Use the production 8-DoF loop CSV.  Required columns already present:

- `damping_z_eff`, `damping_trend_z`, `mass_z_eff`, `v_r_z`
- `instability_idx`, `ke_est`, `v_force_z`
- `cap_press_z`, `cap_retract_z`, `f_dot_z`
- `fz`, `f_des_z_eff`, `contact_present`

Keep the full CSV for each run under a dated directory, e.g.
`logs/force_redesign_YYYYMMDD/`.

## Pass criteria (operator)

1. desired=0 hand push on soft tissue feels light; pushing harder must not
   feel progressively heavier (`damping_z_eff` stays near `damping_base_z`).
2. Free-space approach speed looks the same at 1/2/5 N (~15 mm/s seek).
3. Soft and hard holds at 1/2/5 N settle without sustained bounce.
4. Edge crossing: brief `mass_z_eff` / `instability_idx` rise is OK; no
   repeated contact loss cascade.
5. Over-force hand push retracts without slamming off the surface
   (`cap_retract_z` should tighten, not open to the full vz cap and hold).
