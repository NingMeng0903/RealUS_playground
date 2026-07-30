# Hardware validation — force-axis jerk / push-resistance fix

Offline regressions cover continuous seek→vz press ramp, Ke-symmetric FF,
Iₛ noise floor, trend damping without β·ė, and measured-`dt` integration.
Hardware confirmation is still required before calling this pass done.

## Matrix

| Surface | Setpoints | Extra |
|---|---|---|
| Soft phantom | 1 N, 2 N, 5 N | hand-push with desired=0; fast upward over-force |
| Hard phantom | 1 N, 2 N, 5 N | same |
| Rib-like edge | 2 N scan across edge | watch bounce / contact losses |

## Capture

Use the production 8-DoF loop CSV.  Required columns already present:

- `damping_z_eff`, `damping_trend_z`, `mass_z_eff`, `v_r_z`
- `instability_idx`, `ke_est`, `v_force_z`
- `force_reference_scale_n`, `force_reference_reversal_reset`
- `cap_press_z`, `cap_retract_z`, `dt_actual_s`
- `fz`, `f_des_z_eff`, `contact_present`

Keep the full CSV for each run under a dated directory, e.g.
`logs/force_jerk_fix_YYYYMMDD/`.

## Pass criteria (operator)

1. desired=0 hand push on soft tissue feels light; pushing harder must not
   feel progressively heavier (`damping_z_eff` stays near `damping_base_z`).
2. Free-space approach speed looks the same at 1/2/5 N (~12 mm/s seek).
3. Soft and hard holds at 1/2/5 N settle without sustained bounce.
4. Edge crossing: brief `mass_z_eff` / `instability_idx` rise is OK; no
   repeated contact loss cascade.
5. **Fast upward over-force** retracts without feeling "heavy" — operator
   should not fight a sudden damping wall.
6. Contact latch should not produce a visible velocity snap; press ceiling
   opens over ~0.15 s (`seek_release_s`).

## CSV acceptance (this fix)

From a representative `sin_tool_y` / hybrid@D run with intentional fast
upward pushes and bone/skeleton contact:

| Metric | Target |
|---|---|
| Fast-push `damping_z_eff` p95 (`eff < −4 N`) | **≤ 60** |
| Quiet-contact `mass_z_eff` median | **≈ 1.0–1.5** (base mass; Iₛ floor 0.28) |
| Bone-bounce windows (`fz_rms>1`) | `instability_idx` **not stuck at 0**; `mass_z_eff` rises |
| Free-air / quiet `instability_idx` median | **≈ 0** after floor |
| Single-tick `\|Δ force_reference_scale_n\|` | **< 5** |
| `force_reference_reversal_reset` | **always 0** |
| Latch → +0.15 s `cap_press_z` | **monotone rise**, no 12→100 mm/s step |
| `\|Δv_force_z\| / dt` p95 | **< 0.3 m/s²** |
| Deep σ (`sigma_min < 0.05`) | `rail_ext_w > 0`, measurable `Δrail`, no multi-second `dt` freeze |

Notes:

- Contact latch is **change-based** (`contact_delta_n` vs free-space baseline).
  A constant ~0.45 N sensor bias must **not** latch or brake descent.
- Steady-state force may still sit ~0.45 N above setpoint until φ is
  recalibrated; that is intentional and out of scope for this pass.
- Controller now integrates with measured `dt_actual` (clamped 2–15 ms).
  If hand feel feels slightly slower than before, prefer lowering
  `tau_track_s` (e.g. 0.08 → 0.06) rather than raising damping.
- Damping follows Dimeas 2016 technique (ii): inflate **mass** with Iₛ,
  keep damping near `b_base + α|e|` (no β·ė). Bone bounce: raise
  `damping_base_z` (Keemink G4) and keep `is_floor` low enough that
  brief ringing is not zeroed.
- Singularity path follows 4d15c1d: earlier twist attenuator (enter at
  `2·σ_ref`), harder ProxQP-fail decay, stronger `k_esc` / `w_sigma_floor`.
  `hybrid@D` keeps rail **COUPLED + reach** (not HOLD-locked).
