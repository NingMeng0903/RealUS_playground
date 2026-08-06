# Fast + accurate + no-bounce force stack

Target: light free-space feel (`D0=25`) with stiff-surface bounce killed by a
stiffness-scheduled two-sided velocity barrier + delay-aware contact damping.

## Stack

| Knob | Value | Role |
|---|---|---|
| `admittance_damping_z` | 25 | Fast / light yield (free space) |
| `force_barrier.enabled` | true | Predictive press/retract caps via `K̂b·T_dead` |
| `force_barrier.t_dead_s` | 0.040 | Measured cmd→TCP delay |
| `force_barrier.ke_seek_default` | 300 | Pre-contact keep 80 mm/s seek |
| `force_barrier.f_keep_n` | 0.3 | Retract → 0 as f→f_keep (no lift-off) |
| `delay_damping_mode` | impact_only | Steady contact stays at D0=25 (no sticky D) |
| `delay_damping_kappa` | 0.8 | Impact-burst magnitude only |
| `force_barrier.yield_overforce_n` | 1.5 | Hand yield opens retract (mild ḟ only) |
| `force_barrier.ke_f_err_gate_n` | 1.5 | Freeze K̂b on hand over-force |
| `var_damping_m_u` | 0 | No mass inflation (constant-force scan) |
| `var_damping_omega_c_hz` | 3.5 | Dimeas paper band |
| `var_damping_f_max_n` | 30 | Dimeas paper scale |
| `adaptive_ke.drive_damping` | false | Log Ke; barrier owns contact D |
| `proactive_feedforward` | true | Accurate chase via `v_r` (physical contact only) |
| `wrist_relax_enabled` | true | Attenuate tool-wz near q6≈0 |

## Hardware checklist (skin / abdomen, 2 N)

Compare against `run_20260804_152810`:

1. `fz` p99 ≤ 4 N (was 18.5)
2. `|fz| < 0.35 N` fraction < 3% (was 20.8%)
3. 2–5 Hz force-band energy < 15% (was 48–61%)
4. Contact loss events ≈ 0 (was 22)
5. Steady-contact `damping_z_eff` ≈ 25 (not 200+); free-space also ≈ 25
6. Hand push: light yield (`cap_retract` opens when over-force & mild ḟ); no low-freq shake
7. Soft tissue: press not choked (`cap_press_z` near `v_z_cap` when `ke_barrier` low)
8. CSV: `ke_barrier` median ≪ 3000 in steady hand press; `cap_*` finite

## Do not “fix” with

- Raising fixed D back to 50+
- Turning `drive_damping` on without checking `damping_z_eff`
- Re-enabling `var_damping_m_u` mass inflation as the primary bounce fix
