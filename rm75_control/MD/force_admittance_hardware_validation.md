# Hardware validation — force-axis paper alignment

Offline regressions cover Faverjon free-space bias immunity, approach
decoupling, Ke-normalized FF, and fixed damping.  Hardware confirmation is
still required before calling the redesign done on the robot.

## Matrix

| Surface | Setpoints | Extra |
|---|---|---|
| Soft phantom | 1 N, 2 N, 5 N | hand-push with desired=0 |
| Hard phantom | 1 N, 2 N, 5 N | same |
| Rib-like edge | 2 N scan across edge | watch bounce / contact losses |

## Capture

Use the production 8-DoF loop CSV.  Required columns already present:

- `damping_z_eff`, `mass_z_eff`, `v_r_z`
- `instability_idx`, `ke_est`, `v_force_z`
- `cap_press_z`, `cap_retract_z`
- `fz`, `f_des_z_eff`, `contact_present`

Keep the full CSV for each run under a dated directory, e.g.
`logs/force_redesign_YYYYMMDD/`.

## Pass criteria (operator)

1. desired=0 hand push on soft tissue feels light; pushing harder must not
   feel progressively heavier (`damping_z_eff` stays near `damping_base_z`).
2. Free-space approach speed looks the same at 1/2/5 N (~12 mm/s seek).
3. Soft and hard holds at 1/2/5 N settle without sustained bounce.
4. Edge crossing: brief `mass_z_eff` / `instability_idx` rise is OK; no
   repeated contact loss cascade.
5. Over-force hand push retracts without slamming off the surface
   (`cap_retract_z` should tighten, not open to the full vz cap and hold).

## Paper-alignment acceptance (cross-run)

From ≥5 identical `sin_tool_y` (or move→D) free-space approach runs:

| Metric | Target |
|---|---|
| Descent rate cross-run sd | **< 1 mm/s** (was ~5 vs ~43 mm/s bistable) |
| First-contact force peak | **< f_des + 1 N** |
| Single-tick `D_eff` / `M_eff` jump | much smaller than pre-fix (no 15→200 / 1→5 spikes) |
| Free-air `instability_idx` (Iₛ) | **median < 0.1** (paper operating region ~0.03–0.14) |

Notes:

- Contact latch is **change-based** (`contact_delta_n` vs free-space baseline).
  A constant ~0.45 N sensor bias must **not** latch or brake descent.
- Steady-state force may still sit ~0.45 N above setpoint until φ is
  recalibrated; that is intentional and out of scope for this pass.
- `var_damping_f_max_n: 20` is a first guess for sensor/task scale; retune
  from the free-air Iₛ distribution if needed.
