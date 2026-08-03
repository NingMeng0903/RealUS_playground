# Fast + accurate + no-bounce force stack (924c32d feel)

Restore target: pure-control `924c32d` hand feel, with Dimeas `M(I_s)` for impact
and `adaptive_ke.drive_damping: false` so contact D stays at `D_free` (not ~70).

## Stack

| Knob | Value | Role |
|---|---|---|
| `admittance_damping_z` | 25 | Fast / light yield |
| `proactive_feedforward` | true | Accurate chase via `v_r` |
| `press_is_gate` 0.20→0.60 | on | Freeze press `v_r` when bouncing |
| retract gate | never | Escape over-force |
| `var_damping_m_u` | 4 | Anti-bounce inertia |
| `var_damping_d_u` | 2 | Small residual only |
| `adaptive_ke.drive_damping` | false | Log Ke, do not sticky-D |
| `force_barrier` | false | Do not choke chase |
| contact | enter-only | Keep latch across bounce flight |
| rail `sign` | -1 | HW direction (unchanged) |

## Hardware checklist (skin / abdomen, 2 N)

1. Steady median `damping_z_eff` ≪ 50 (target mostly 25–40)
2. Hand press-up yields lighter than D=50 sticky build
3. Impact: `mass_z_eff` rises with `I_s`; less bounce than low-M alone
4. Fast retract: contact loss not much worse than prior good runs
5. CSV: `v_r_z` signed on under/over force; `proactive` active

## Do not “fix” with

- Raising fixed D back to 50+
- Re-enabling ForceBarrier this pass
- Turning `drive_damping` on without checking `damping_z_eff`
