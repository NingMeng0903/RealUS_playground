# Force-compliance acceptance and measurement

Hardware: RM75, velocity interface (`rm_movej_canfd`, 5 ms). Plant centre
from 2026-09-03 chirp: Z `T0=28 ms`, `Tp=14 ms`. Loop delay `Td ≈ 40 ms`.

This file is the checklist for the full-axis compliance plan. Unit tests
cover the law; the rows below need the arm.

## 1. Identification still missing on hardware

Do these before claiming tissue / hand numbers. Use `--log-csv`.

| ID | Protocol | What to record |
| --- | --- | --- |
| 07 contact Gv | ICRA `07_contact_gv` on soft pad and fixture | Contact-plant `T0`, `Tp`, `K` vs air |
| 08 Ke | ICRA `08_ke` on tissue / pad / fixture | Settled `Ke` (N/m), first-touch peak |
| Hand stiffness | Hover, operator holds the probe and resists a slow 5–10 mm lateral / ±0.1 rad twist | Rough `k_h` (N/m) and `K_θ` (N·m/rad) |

Write the medians into the next yaml pass. Until then the law uses
`D = max(22, K̂e/14)` online.

## 2. Hover (all force axes, `F*=M*=0`)

Law: Kikuuwe 0.32 N / 0.025 N·m, `M=1.5`, `D=25`, `I=0.04`, `D_ω=0.65`,
ω about `r̂`.

| Check | Pass |
| --- | --- |
| 60 s free space, no touch | `\|v\|_p95 < 1 mm/s`, no crawl |
| 0.3 N leftover (ID residual) | still; no walk |
| 1 N lateral push | translation follows, `\|ω\| < 0.02 rad/s` (no yank) |
| 0.1 N·m couple | rotation about `r̂`, not TCP |
| Release | stop inside `3τ` (`τ ≈ 60 ms`) |
| F/M spectrum | no peak above 2 Hz while held still |

## 3. Tool-Z force track

Air 30 mm/s → touch brake 8 mm/s → confirm (`≥0.15 s` + `K̂e`) then
`e_f/D(K̂e)`. `κ=0`. Press and retract share the same linear gain.

| Check | Pass |
| --- | --- |
| Soft pad `F* 0→3 N` | overshoot `<20%`, no oscillation above 2 Hz |
| Soft pad `3→1 N` | retract time same order as the press |
| Fixture 1420 / 2200 N/m first touch | peak `<1 N` above settle, no second impact |
| Unknown / unconfirmed Ke | press stays at 8 mm/s |
| Tank | contact balance does not sit on `ε`; PO active energy bounded |
| Hand feel vs Ke-schedule-only | passivity layer must not feel obviously duller |

## 4. Commands

```text
python -m peirastic.apps.run_controller
python -m peirastic.DEMO.hover
python -m peirastic.apps.identify_air --log-csv
```

Hybrid / pad force modes load `peirastic/configs/force.yaml`.
Observer LPF is `force.causal_fc_hz: 45` in the machine yaml.
