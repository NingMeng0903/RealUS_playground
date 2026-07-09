#!/usr/bin/env python3
"""Regenerate MD/debug.md — verbatim joint-admittance + WBC/CBF source mirror."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "MD" / "debug.md"

SECTIONS: list[tuple[str, Path]] = [
    # joint_admittance inner loop (WBC slack-QP + CBF)
    ("rm75_control/control/joint_admittance/__init__.py", REPO / "rm75_control/control/joint_admittance/__init__.py"),
    ("rm75_control/control/joint_admittance/config.py", REPO / "rm75_control/control/joint_admittance/config.py"),
    ("rm75_control/control/joint_admittance/model.py", REPO / "rm75_control/control/joint_admittance/model.py"),
    ("rm75_control/control/joint_admittance/ik_types.py", REPO / "rm75_control/control/joint_admittance/ik_types.py"),
    ("rm75_control/control/joint_admittance/collision_model.py", REPO / "rm75_control/control/joint_admittance/collision_model.py"),
    ("rm75_control/control/joint_admittance/pose_ik.py", REPO / "rm75_control/control/joint_admittance/pose_ik.py"),
    ("rm75_control/control/joint_admittance/reference.py", REPO / "rm75_control/control/joint_admittance/reference.py"),
    ("rm75_control/control/joint_admittance/validation.py", REPO / "rm75_control/control/joint_admittance/validation.py"),
    ("rm75_control/control/joint_admittance/loop.py", REPO / "rm75_control/control/joint_admittance/loop.py"),
    ("rm75_control/control/joint_admittance/tasks/__init__.py", REPO / "rm75_control/control/joint_admittance/tasks/__init__.py"),
    ("rm75_control/control/joint_admittance/tasks/nullspace_task.py", REPO / "rm75_control/control/joint_admittance/tasks/nullspace_task.py"),
    ("rm75_control/control/joint_admittance/tasks/arm_angle.py", REPO / "rm75_control/control/joint_admittance/tasks/arm_angle.py"),
    ("rm75_control/control/joint_admittance/tasks/manipulability_task.py", REPO / "rm75_control/control/joint_admittance/tasks/manipulability_task.py"),
    ("rm75_control/control/joint_admittance/tasks/secondary_composer.py", REPO / "rm75_control/control/joint_admittance/tasks/secondary_composer.py"),
    ("rm75_control/control/joint_admittance/solver/__init__.py", REPO / "rm75_control/control/joint_admittance/solver/__init__.py"),
    ("rm75_control/control/joint_admittance/solver/constraint_mgr.py", REPO / "rm75_control/control/joint_admittance/solver/constraint_mgr.py"),
    ("rm75_control/control/joint_admittance/solver/cbf_constraints.py", REPO / "rm75_control/control/joint_admittance/solver/cbf_constraints.py"),
    ("rm75_control/control/joint_admittance/solver/qp_builder.py", REPO / "rm75_control/control/joint_admittance/solver/qp_builder.py"),
    ("rm75_control/control/joint_admittance/utils/__init__.py", REPO / "rm75_control/control/joint_admittance/utils/__init__.py"),
    ("rm75_control/control/joint_admittance/utils/safety.py", REPO / "rm75_control/control/joint_admittance/utils/safety.py"),
    # outer loop + force observer
    ("rm75_control/control/hybrid_motion/__init__.py", REPO / "rm75_control/control/hybrid_motion/__init__.py"),
    ("rm75_control/control/hybrid_motion/controller.py", REPO / "rm75_control/control/hybrid_motion/controller.py"),
    ("rm75_control/control/hybrid_motion/adaptive_ke.py", REPO / "rm75_control/control/hybrid_motion/adaptive_ke.py"),
    ("rm75_control/control/hybrid_motion/observer.py", REPO / "rm75_control/control/hybrid_motion/observer.py"),
    ("rm75_control/control/hybrid_motion/reference.py", REPO / "rm75_control/control/hybrid_motion/reference.py"),
    ("rm75_control/control/hybrid_motion/async_state.py", REPO / "rm75_control/control/hybrid_motion/async_state.py"),
    ("rm75_control/control/hybrid_motion/loop.py", REPO / "rm75_control/control/hybrid_motion/loop.py"),
    ("rm75_control/core/session.py", REPO / "rm75_control/core/session.py"),
    # tool / pose D (gripper vs Arm_Tip)
    ("rm75_control/force/compensation/tool_pose.py", REPO / "rm75_control/force/compensation/tool_pose.py"),
    ("rm75_control/motion/canfd.py", REPO / "rm75_control/motion/canfd.py"),
    # configs + entry points
    ("configs/joint_admittance.yaml", REPO / "configs/joint_admittance.yaml"),
    ("configs/force_compensation/poses.yaml", REPO / "configs/force_compensation/poses.yaml"),
    ("tmp/Velocity_Admittance/demo/config/human_soft_scan.yaml", REPO / "tmp/Velocity_Admittance/demo/config/human_soft_scan.yaml"),
    ("apps/joint_admittance/run_joint_admittance.py", REPO / "apps/joint_admittance/run_joint_admittance.py"),
    ("apps/joint_admittance/d_sin_tool_y.py", REPO / "apps/joint_admittance/d_sin_tool_y.py"),
    ("apps/joint_admittance/debug_pose_ik.py", REPO / "apps/joint_admittance/debug_pose_ik.py"),
    ("apps/joint_admittance/analyze_move_csv.py", REPO / "apps/joint_admittance/analyze_move_csv.py"),
    ("apps/joint_admittance/analyze_scan_force_csv.py", REPO / "apps/joint_admittance/analyze_scan_force_csv.py"),
    ("apps/joint_admittance/sim_move_offline.py", REPO / "apps/joint_admittance/sim_move_offline.py"),
    ("tests/test_joint_ik_offline.py", REPO / "tests/test_joint_ik_offline.py"),
    ("tests/test_governor.py", REPO / "tests/test_governor.py"),
    ("tests/test_limit_damper.py", REPO / "tests/test_limit_damper.py"),
    ("tests/test_task_weight_scale.py", REPO / "tests/test_task_weight_scale.py"),
    ("tests/test_sr_damping.py", REPO / "tests/test_sr_damping.py"),
    ("tests/test_arm_angle_singularity.py", REPO / "tests/test_arm_angle_singularity.py"),
    ("tests/test_nullspace_dyn.py", REPO / "tests/test_nullspace_dyn.py"),
    ("tests/test_secondary_composer.py", REPO / "tests/test_secondary_composer.py"),
    ("tests/test_adaptive_ke.py", REPO / "tests/test_adaptive_ke.py"),
    ("tests/test_realtime_state.py", REPO / "tests/test_realtime_state.py"),
    ("tests/test_move_guards.py", REPO / "tests/test_move_guards.py"),
    ("tests/test_manipulability_task.py", REPO / "tests/test_manipulability_task.py"),
    ("tests/test_force_yaml_3n.py", REPO / "tests/test_force_yaml_3n.py"),
    ("tests/test_var_damping_adaptation.py", REPO / "tests/test_var_damping_adaptation.py"),
    # assets (kinematics + collision)
    ("rm75_control/assets/robots/rm75_6f/RM75-6F.urdf", REPO / "rm75_control/assets/robots/rm75_6f/RM75-6F.urdf"),
    ("rm75_control/assets/robots/rm75_6f/RM75-6F.collision.urdf", REPO / "rm75_control/assets/robots/rm75_6f/RM75-6F.collision.urdf"),
    ("rm75_control/assets/robots/rm75_6f/collision_pairs.yaml", REPO / "rm75_control/assets/robots/rm75_6f/collision_pairs.yaml"),
    ("scripts/blender_simplify_collision_meshes.py", REPO / "scripts/blender_simplify_collision_meshes.py"),
    ("scripts/run_collision_mesh_simplify.sh", REPO / "scripts/run_collision_mesh_simplify.sh"),
]


def lang(path: Path) -> str:
    if path.suffix in (".yaml", ".yml"):
        return "yaml"
    if path.suffix == ".sh":
        return "bash"
    if path.suffix == ".urdf":
        return "xml"
    return "py"


def embed(rel: str, path: Path) -> str:
    body = path.read_text(encoding="utf-8")
    return f"## FILE: `{rel}`\n\n```{lang(path)}\n{body}```\n"


def diagnostic_header() -> str:
    """Force-control regression: root cause + fix, from real telemetry spectral
    evidence (2026-07 revision, supersedes the earlier config-diff-only pass).
    Documentation only — the actual fixes are applied in the source files below.
    """
    return _DIAGNOSTIC_MD_V3 + _DIAGNOSTIC_MD_V2 + _DIAGNOSTIC_MD


_DIAGNOSTIC_MD_V3 = """\
## Force regression, round 3 — WBC inner-loop slack blocks retract at scan extremes

**Status: root cause identified, not yet fixed.** After round-2 var_damping
fixes, a fresh hardware run logged to `/tmp/scan_16cm.csv` (182 s, 36530 scan
ticks, 14 MB, 2026-07-03 21:28) still shows force overshoot and bounce. FFT
summary (`analyze_scan_force_csv.py`):

```text
fz: mean=3.10 std=2.75 min=-7.14 max=14.32
v_force_z: mean=-0.0217 std=0.0918  (saturated ±0.20 m/s on over/under-force)
|fz|<0.5N: 8.3%   fz>5N: 15.4%   in±0.5N band: 29.7%
fz 7-11Hz band power: 0.53%   (round-1 was ~2.4% — HF jitter improved)
CHECK hf_v_force_z<5%: PASS
CHECK airborne_press_mean<0.02: FAIL   (airborne mean v_force_z=0.057 m/s, 99.3% positive)
```

The remaining complaints (slow retract when fz>3N, Fz spikes to 11–14N, elastic
surface bounce, -7N negative excursion) are **not** primarily admittance-layer
jitter. They come from the **inner WBC QP slack** eating the Z retract command
when the arm enters a near-singular / self-collision posture at the 16 cm scan
extremes.

### 1. Mechanism: admittance commands retract, WBC cannot execute it

At two scan-Y extremes the inner loop activates CBF self-collision constraints
(`n_cbf=3`) and σ_min drops to 0.021–0.033 while `slack_norm` climbs to 0.09:

```text
Episode A  t=144.65–152.52 s  pose_y=[-0.068,-0.041]  max_fz=14.32 N
Episode B  t=171.69–180.48 s  pose_y=[-0.069,-0.034]  max_fz=13.27 N
```

During episode A (elastic / high-stiffness contact):

```text
t=147.0  fz=10.0  v_force_z=-0.20 (admittance: full retract)  twist_vz=+0.060 (actual: still pressing!)
         slack=0.084  n_cbf=3  sigma_min=0.030  governor=0.81
t=147.2  fz=14.3  v_force_z=-0.20  twist_vz=+0.062  slack=0.086
```

99% of `fz>5N` ticks have `v_force_z<0` (correct retract command) but
`twist_vz>0` (actual TCP still moving into the surface) — the outer admittance
loop is doing the right thing; the inner `QpIkController` equality
`J_tcp qdot - w = v_cmd` absorbs the infeasible Z component into slack `w`
because CBF rows `J_col qdot >= v_safe` and low σ_min task-weight scaling
prevent a feasible retract.

Episode B ends with a -7.1N negative Fz bounce (t≈179–182 s): after ~8 s of
slack-limited retract the arm finally escapes the constrained posture; the
stored elastic energy releases as a rebound. `instability_idx` stays <0.06
throughout — var_damping is not the bottleneck here.

### 2. Why round-2 var_damping did not fix this

Round-2 correctly switched to Dimeas technique (ii) mass adaptation and
rescaled `f_max_n`. Measured on this same CSV: `instability_idx` mean=0.019,
max=0.213, p99=0.171 — the detector now has authority, but the dominant failure
mode is **kinematic infeasibility** (CBF + singularity), not HF contact
oscillation. Increasing virtual mass cannot overcome a QP that literally cannot
produce negative tool-Z velocity at that joint configuration.

### 3. Candidate fix directions (deferred — needs planning)

1. **Scan workspace**: reduce Y amplitude or shift ψ_ref so scan extremes stay
   away from σ_min<0.05 / CBF-active configurations.
2. **Manipulability task during scan**: currently scan uses centering; move uses
   manipulability ascent — consider enabling σ ascent in scan at Y extremes.
3. **CBF / collision tuning**: review `d_activate=0.04`, `d_safe=0.01`,
   `collision_pairs.yaml` disabled pairs — may be over-conservative at this pose.
4. **Force safety net**: if `fz > f_max` and `v_force_z < 0`, temporarily boost
   Z task weight or bypass non-Z CBF rows (safety trade-off).
5. **Governor interaction**: governor drops to 0.61–0.78 during slack episodes,
   further slowing retract — secondary effect.

### 4. Telemetry file location

```text
Path:     /tmp/scan_16cm.csv
Size:     14 582 528 bytes (~14 MB)
Lines:    37 086 (header + 37 085 data rows)
Duration: ~182 s scan phase (KeyboardInterrupt at t≈181.5 s wall)
Columns:  t_wall_s, phase, fz, v_force_z, instability_idx, damping_z_eff,
          ke_est, slack_norm, n_cbf, sigma_min, governor_scale, twist_vz, ...
Analyze:  python apps/joint_admittance/analyze_scan_force_csv.py /tmp/scan_16cm.csv
```

Copy off the robot PC if needed:
`scp camp@192.168.1.80:/tmp/scan_16cm.csv .`

---

"""

_DIAGNOSTIC_MD_V2 = """\
## Force regression, round 2 — var_damping adaptation law was Dimeas' worst technique

**Status: fixed in this working tree.** Round 1 (below) fixed the Kdf
derivative-kick and re-enabled `var_damping`. A real hardware run after that
fix (`/tmp/scan_16cm.csv`, 16cm sweep, 10624 scan ticks) still showed two
complaints: bouncing on release, and vibration while pressing into a
muscle-like (compliant, viscoelastic) surface. Re-analyzing that CSV against
`instability_idx`/`damping_z_eff`/`ke_est` (not just the FFT summary) found a
second, independent bug: `var_damping` was correctly detecting the contact
oscillation, but reacting to it with the adaptation law Dimeas & Aspragathos
2016 explicitly measured to be the worst one, and with a force-scale constant
copied from a different experimental setup.

### 1. Bug: var_damping used technique (i) — the one the cited paper says not to use

`var_damping_d_u: float = 60.0` / `var_damping_m_u: float = 0.0` (defaults,
unmodified by round-1's yaml) is exactly Dimeas Eq. (7), technique (i):
damping-only adaptation, constant inertia. The same paper's Table 2 measured
this against technique (ii) (inertia-only) and technique (iii) (both, fixed
ratio) at high environment stiffness:

```text
Technique (i)  const md, var cd : D=1.77J  Is=8.90   <- damping-only (what we had)
Technique (ii) const cd, var md : D=0.28J  Is=0.32   <- inertia-only
Technique (iii) var cd, md      : D=0.62J  Is=0.37
```

Their own conclusion: *"a sole increase in the damping gain cd... required
energy... is increased by 534% compared to technique (ii)... the virtual
damping has the most dominant effect on the cooperation and therefore, it
should not be increased when the robot turns unstable."* This is a direct,
literature-contradicted misconfiguration, not a tuning preference.

### 2. Bug: var_damping_f_max_n=30N was never rescaled to this task's force regime

`var_damping_f_max_n: float = 30.0` matches Dimeas' virtual-wall experiment
task-force scale (their double-wall / stiffness-apparatus setup). This task's
contact setpoint is 1–3N over a 0–10N range. Measured on `/tmp/scan_16cm.csv`
(entire run, `instability_idx` column): **max = 0.076**, even though the same
CSV contains an unambiguous ~8Hz, ~4N-amplitude bounce (see §3). `Irms =
rms(f_ac)/f_max` was chronically under-driven by the mismatched denominator,
so the adaptation law — whichever technique — could never engage with any
real authority.

### 3. Data-confirmed bounce mechanism (ties both bugs together)

`/tmp/scan_16cm.csv`, t≈54.45–54.9s, continuous press-phase (not a lift-off
ripple):

```text
t=54.446 fz= 2.18  v_force_z=-0.1601  inst=0.0648  damp=25.9
t=54.461 fz=-0.37  v_force_z=-0.0757  inst=0.0654  damp=25.9
t=54.476 fz=-1.11  v_force_z=-0.0033  inst=0.0662  damp=26.0   <- true loss of contact
t=54.496 fz=-0.10  v_force_z= 0.0491  inst=0.0670  damp=26.0
```

`fz` swings from +2.9N through zero to **-1.1N** (genuine contact loss) and
back, period ≈120ms (~8Hz) — matching the round-1 spectral peak and Dimeas
Fig. 4's own prediction that increasing environment stiffness produces a
"resonant frequency of approximately up to 10Hz". Root mechanism: with
`admittance_damping_z_release=10.0` (round-1 fix, light retract damping),
retreat velocity builds to -0.16 m/s while `fz` is still above the 3N
setpoint (`eff<0`); once `fz` crosses back under 3N (`eff` flips sign to
"press"), the correction has ~30ms to decelerate that velocity before the
probe coasts through the zero-force boundary. `instability_idx` (0.065–0.075)
correlates with this bounce (`corr=0.71` between a >5Hz high-pass envelope of
`fz` and `instability_idx` during press-phase segments, `corr=0.60` with the
extra damping actually applied) — confirming the Dimeas detector fires
correctly here; only the response law and its scale were wrong.

Architecturally, this also explains why the round-1 fix (`var_damping_d_u`)
could never have fixed this bounce even with correct scaling:

```582:592:rm75_control/control/hybrid_motion/controller.py
def _effective_damping_z(self, d_base: float, eff: float) -> float:
    ...
    if eff < -1e-9 and d_release is not None:
        return min(d, float(d_release))
```

Whenever `eff<0` (retract — exactly the velocity-buildup half of the bounce),
this clamps `d` down to `admittance_damping_z_release` regardless of
`var_damping_d_u * instability_idx`. A mass-based correction (`var_damping_m_u`,
which feeds `self._m_z_now` directly and is never touched by
`_effective_damping_z`) is the only channel that can act during that phase —
a second, independent argument for Dimeas technique (ii)/(iii) beyond the
effort numbers in §1.

### 4. Fix set applied in this revision

1. `var_damping_m_u: 0.0 → 4.0` (primary channel, Dimeas technique ii/iii).
2. `var_damping_d_u: 60.0(default) → 2.0` (small residual only, not the driver).
3. `var_damping_f_max_n: 30.0(default) → 7.0` (rescaled to this task's 1–3N
   setpoint / 0–10N contact range instead of Dimeas' virtual-wall task force).
4. No structural code change: mass increases already flow through the existing
   `self._m_z_now` pipeline into both the tool-Z admittance ODE inertia and
   (via `adaptive_bd = 2ζ√(m·K̂e)`) the K̂_e-based damping — and bypass the
   press/release `_effective_damping_z` clamp entirely (§3).
5. New regression tests (`tests/test_var_damping_adaptation.py`): assert the
   shipped yaml keeps `var_damping_m_u` as the dominant channel and
   `var_damping_f_max_n` rescaled (guards against silently reverting to
   technique (i)); and two controller-level tests that a sustained 9Hz
   oscillation raises `instability_index` and the effective mass while the
   damping contribution stays smaller, including specifically while
   `eff<0` (retract, where the damping channel is clamped away).

### 5. Explicitly deferred / not changed this pass

- `admittance_damping_z_release=10.0` left unchanged — re-evaluate only if a
  fresh hardware run (below) still shows `fz` crossing negative after the
  var_damping fixes above; the intent is to let the (now correctly scaled)
  adaptive mass do this job rather than hand-tuning a second asymmetric
  constant.
- Li 2022's proactive feedforward (Eq. 23) is direction-symmetric in the
  original paper; this codebase keeps it retract-only
  (`alpha_proactive_press: 0.0`). Not touched this pass — revisit only if the
  var_damping fix above does not fully resolve the press-phase muscle
  vibration complaint.
- Faverjon & Tournassoud 1987 reviewed and confirmed unrelated to this
  mechanism — its velocity damper is already correctly used for joint-limit
  avoidance in the inner QP (`limit_damper_band_rad`), a different subsystem
  from the tool-Z force admittance discussed here.

### Verification

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q` — 60 tests
  passing (57 prior + 3 new in `test_var_damping_adaptation.py`).
- **Still needed**: a real robot run of `d_sin_tool_y.py` (16cm sweep) against
  a real muscle/soft target with this fix set, analyzed the same way as §3
  (`apps/joint_admittance/analyze_scan_force_csv.py`), to confirm
  `instability_idx` now reaches a meaningful level during a genuine bounce
  (previously capped at 0.076) and that `fz` no longer swings negative.

---

"""

_DIAGNOSTIC_MD = """\
## Force regression vs a686e07f — RESOLVED (spectral-evidence revision)

**Status: fixed in this working tree.** This revision replaces the earlier
static "config diff vs a686e07f" diagnosis, which correctly found the changed
parameters but mis-ranked the root cause (it blamed `adaptive_ke`'s K̂_e
mis-learning as the primary driver). Real telemetry FFT analysis below shows a
different, more fundamental mechanism was the actual #1 contributor, and the
fix set below addresses both.

### 0. What changed since the first diagnosis

The first pass compared `configs/joint_admittance.yaml` against `a686e07f`
statically and stopped there. This pass ran an FFT over two independent real
scan recordings (`/tmp/scan2.csv`, `/tmp/scan.csv`) and computed actual filter
frequency responses for `observer.py` and the Dimeas high-frequency detector
in `controller.py`. Conclusion: **the `admittance_force_derivative_gain_z`
(Kdf) term was differentiating a real, reproducible ~9 Hz contact oscillation
that the observer's 6 Hz low-pass only attenuates to ~40%, turning a modest
force ripple into a large velocity-command jitter ("derivative kick")** —
this is a bigger and more direct contributor to "抖动特别厉害" than the
`adaptive_ke` lateral-scan mis-learning, which is real but secondary.

### 1. Hard evidence: a reproducible ~9 Hz contact oscillation

FFT of `/tmp/scan2.csv` (30 s scan, 777 ticks @ 200 Hz) and `/tmp/scan.csv`
(independent, longer run, 4025 ticks):

```text
scan2.csv fz top:        9.009Hz(mag 420) 8.752Hz(mag 399) 0.257Hz(mag 317, = sin scan freq)
scan2.csv fx top:        8.752Hz(mag 142) 9.009Hz(mag 133) 8.494Hz(mag 98)
scan2.csv v_force_z top: 9.009Hz(mag 12.7) 8.752Hz(mag 12.1)   <- same peaks in the *commanded velocity*, not just raw force
scan2.csv pose_z top:    0.257Hz(mag 1.18) 9.009Hz(mag 0.24)   <- 9Hz nearly invisible in physical position
scan.csv  fz top (separate day): 9.391Hz(mag 426) 9.242Hz(mag 384)   <- same band, reproducible
```

Reading:

- **Reproducible**: two independent runs (different day / parameters) both
  show a strong peak in **8.5–9.5 Hz**.
- **AM sidebands**: the `scan2.csv` peaks are spaced by exactly 0.257 Hz (the
  sin-scan frequency) — classic amplitude modulation of a ~9 Hz carrier by the
  slow scan geometry. The carrier itself is independent of scan frequency, so
  this is separate dynamics being *modulated* by contact depth, not a scan
  artifact by itself.
- **Force-channel strong, position-channel weak**: this is a high-frequency
  vibration/resonance (sensor/tool/gripper assembly) picked up by the force
  channel and then **amplified by the control loop into an actual velocity
  command** — that amplification, not a visibly large arm motion, is what
  "抖动特别厉害" feels like.
- **Not CBF/QP clamping**: `n_cbf` and `vel_clamped` are 0 throughout, ruling
  out a velocity-box limit-cycle explanation.

### 2. Two concrete amplification mechanisms

**2.1 Observer's 6 Hz cutoff leaks into Kdf's differentiator.** Actual gain of
`observer.py`'s `causal_fc_hz=6.0` 2nd-order Butterworth LPF at 200 Hz:

```text
3.5Hz: |H|=0.947   6.0Hz: |H|=0.707 (cutoff)   8.5Hz: |H|=0.444   9.0Hz: |H|=0.403 (-7.9dB)   9.5Hz: |H|=0.367
```

9 Hz is only attenuated to ~40%; a meaningful fraction of the oscillation
reaches `f_ext_z` unfiltered. `_admittance_z` then computes
`f_dot_z = (f_ext_z - f_ext_z_prev)/dt` and applies `kdf * f_dot_z`. At 200 Hz,
differentiating a 9 Hz signal multiplies its amplitude by `2π·9 ≈ 56 rad/s` —
this is the textbook "derivative kick" (Eppinger & Seering, *Understanding
Bandwidth Limitations in Robot Force Control*, ASME/ICRA 1987: most mid/high
frequency contact instability comes from differentiating unmodeled
sensor/end-effector dynamics, not from the nominal 2nd-order admittance model
itself). **`a686e07f` (good feel) never configured Kdf (effectively 0.0); the
working tree had it at 0.035** — the single largest jitter amplifier found.

**2.2 The channel built to suppress exactly this was switched off.**
`_update_instability_index` (Dimeas & Aspragathos 2016) high-passes the force
signal at 3.5 Hz to detect high-frequency energy and add damping. Its actual
response:

```text
1Hz: |H|=0.081   3.5Hz (cutoff)=0.707   6Hz: |H|=0.947   9Hz: |H|=0.989
```

9 Hz passes almost 100% through this filter — it is purpose-built to catch
exactly this kind of oscillation and respond with `d += d_u·Is`. It was
`var_damping_enabled: false` in the working tree; `a686e07f` had it **on**.

### 3. Paper cross-check

| Source | Relevance |
|---|---|
| Li, Ge & Yang, *Int. J. Control* 2012 (user-provided) | Joint-torque-level iterative-learning impedance control of the robot's *own* dynamics uncertainty (M/C/G bounds) — does **not** cover online environment-stiffness estimation. `adaptive_ke.py` previously cited this family of work for ΔF/Δx learning; that citation did not match the algorithm and has been corrected (see file). |
| Li & Ge, *IEEE/ASME TMech* 2014 (user-provided) | Joint-torque-level adaptive impedance + NN human motion-intention estimation for pHRI — a different problem (estimating a human's target position), not environment stiffness. Same citation mismatch, now corrected. |
| Eppinger & Seering, *Understanding Bandwidth Limitations in Robot Force Control*, ASME/ICRA 1987 | Explains the exact 9 Hz mechanism: differentiating unmodeled sensor/structure dynamics (not badly-tuned nominal admittance parameters) is the classic root cause of mid/high-frequency contact instability. |
| Kronander & Billard, *IEEE T-RO* 2016, "Stability Considerations for Variable Impedance Control" | Standard (time-invariant) stability analysis breaks down when impedance parameters vary with time; needs an explicit bound relating `dK/dt` to damping. `adaptive_ke.py`'s `bd_slew_max`/`ke_slew_max` are empirical, not derived from this criterion — flagged as a theory gap for a future iteration, not blocking this fix. |
| 2023–2025 variable-impedance soft-tissue scanning surveys (new, ultrasound/palpation, structurally similar to this project's constant-force scan): Persson et al. 2023 (arXiv 2309.14893), "Optimization-Based Variable Impedance Control..." IEEE TIM 2024, "Force Tracking Control Method for Robotic Ultrasound Scanning..." MDPI Actuators 2024 | Common pattern in modern systems: (1) an energy-tank / passivity guarantee rather than relying solely on fixed/adaptive damping formulas; (2) an explicit clamp/QP-constraint/energy-budget between stiffness estimate and control law — raw ΔF/Δx or force derivative is never fed straight to a velocity command; (3) the "soft→rigid transition" (muscle→bone here) is a named benchmark scenario, handled by keeping the *estimator* fast but constraining the *output* (damping/stiffness command), not by disabling estimation during motion. This project has **no energy tank / passivity constraint** yet — a more thorough fix than scan-gating, recorded as future work (§7) rather than attempted in this pass. |

### 4. Fix set applied in this revision

1. **Kdf → 0.0** (`configs/joint_admittance.yaml`): removes the derivative-kick
   amplification path (§2.1), aligned with `a686e07f`.
2. **`var_damping_enabled: true`** with `var_damping_lambda: 0.995`: restores
   the Dimeas HF-damping channel purpose-built to suppress this class of
   oscillation (§2.2), aligned with `a686e07f`.
3. **`proactive_feedforward: true`**, `admittance_mass_z: 1.0`,
   `admittance_damping_z: 25.0`, `max_vz_tool_m_s: 0.20`: restore the
   remaining `a686e07f`-verified baseline values.
4. **`adaptive_ke` gating generalized and kept ON**: the K̂_e estimator still
   learns environment stiffness (needed for the muscle↔bone "feel" goal), but
   the gate that used to look only at the tool-Y feedforward component of a
   scripted sin-scan (`v_scan_tool_y`, `gate_scan_velocity`) is replaced with
   `gate_lateral_velocity` / `lateral_vel_gate_m_s`, driven by the **magnitude**
   of the tool-XY-plane speed of the actual commanded velocity (feedforward +
   PBAC correction). This is orientation-agnostic: a tool-X sweep, a diagonal
   sweep, or a manually dragged probe are all gated the same way a tool-Y sin
   scan is — not a mode-specific special case (see `adaptive_ke.py` docstring
   and the `test_gate_lateral_velocity_is_direction_agnostic` regression test).
5. **Corrected paper citations** in `adaptive_ke.py` — see §3.

### 5. Code bug / redundancy audit

| # | Location | Issue | Disposition |
|---|---|---|---|
| 1 | `controller.py` `_admittance_z` | `Kdf * f_dot_z` differentiates a signal only attenuated to ~40% at 9Hz by the 6Hz observer LPF — primary jitter amplifier found this pass | **Fixed**: `admittance_force_derivative_gain_z: 0.0` (§4.1) |
| 2 | `configs/joint_admittance.yaml` `var_damping_enabled: false` | Disabled the Dimeas HF-suppression channel purpose-built for this (§2.2) | **Fixed**: re-enabled, `var_damping_lambda: 0.995` |
| 3 | `adaptive_ke.py` docstring + `controller.py` comments | Cited "Yanan Li & Ge, IEEE T-CST 2014" for ΔF/Δx stiffness learning; neither of the user-provided Li & Ge papers (2012 IJC iterative-learning impedance / 2014 TMech motion-intention) actually discusses online environment-stiffness estimation | **Fixed**: docstring corrected to a generic "critical-damping impedance adaptation" framing (Duan et al. 2018 family), misattribution removed |
| 4 | `controller.py` lines ~416 & ~588 (pre-fix) | `m_z = max(admittance_mass_z + var_damping_m_u * instability_index, 1e-3)` computed once in `compute_velocity_command` and again identically in `_admittance_z` | **Fixed**: computed once, cached as `self._m_z_now`, reused in `_admittance_z` — pure refactor, no behavior change |
| 5 | `controller.py` `_admittance_z` | `admittance_damping_z_press`/`release` (asymmetric press/release damping) are bypassed entirely when `adaptive_ke.enabled=True`, not combined with it | **Not fixed** (both values are `None`/unconfigured in the current yaml, so no behavioral difference today) — recorded for whoever wants to combine asymmetric damping with adaptive K̂_e later |
| 6 | `controller.py` `_admittance_z` | When both `adaptive_ke` and `var_damping` are enabled, their damping contributions add (`d = adaptive_bd; d += var_damping_d_u * instability_index`); this is plausible (they respond to different phenomena — quasi-static stiffness vs. transient HF oscillation) but has no dedicated test for combined-trigger over-damping | **Not fixed this pass** — recorded as a bench-test checkpoint: if restoring `var_damping` makes the feel "sluggish/over-damped" rather than "stable", check this interaction first |
| 7 | `adaptive_ke.py` `_should_update_ke` (pre-fix) | Gate signal was `controller.py`'s hardcoded tool-Y component of the *feedforward* velocity — only valid for `d_sin_tool_y.py`'s scripted sin-Y scan, not manual drags or other sweep directions | **Fixed**: generalized to `v_lateral_m_s`, the tool-XY-plane speed magnitude of the actual commanded velocity (§4.4) |
| 8 | Global | No energy-tank / passivity constraint on `M/K̂_e/b_d` — all heuristic formulas + empirical slew limits (`bd_slew_max`, `ke_slew_max`), not derived from a stability criterion like Kronander & Billard 2016 | **Not fixed this pass** (large change: needs a new energy-accounting state machine) — recorded as the next-iteration direction (§7) |

### 6. Explicitly deferred to a future iteration

- **No energy tank / passivity constraint** (#8 above) — larger change,
  deferred until the lower-risk fix set here (remove Kdf amplification +
  restore the already-implemented HF-damping channel) is verified on hardware.
- **`observer.py`'s `causal_fc_hz=6.0` left unchanged** — tightening it would
  reduce 9Hz leakage further but adds phase lag that could hurt 3N force
  tracking response; only worth revisiting if the Kdf/var_damping fix above
  doesn't sufficiently reduce the residual 9Hz peak on new telemetry.
- **Bug #5 (asymmetric damping vs adaptive_ke) and #6 (combined-damping
  interaction untested)** — not triggered by the current yaml, left as bench
  observations.
- Move/IK code, and `tmp/Velocity_Admittance/demo/config/human_soft_scan.yaml`
  (reference-only) — unchanged, confirmed unrelated to this regression.

### Verification

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q` — full suite
  passing after this fix set (56 tests), including a new
  `test_gate_lateral_velocity_is_direction_agnostic` regression test in
  `tests/test_adaptive_ke.py` that simulates a pure geometric-coupling Fz
  ripple driven by tool-Y / tool-X / diagonal sweeps at equal tangential speed
  and confirms `gate_lateral_velocity=True` keeps K̂_e near `ke_initial` for
  all three directions, while `False` lets all three drift.
- **Still needed**: a real robot run of `d_sin_tool_y.py --enable-force
  --log-csv` with this fix set, FFT'd the same way as §1, to confirm the ~9Hz
  peak drops.

---

"""


def stack_header(n_files: int) -> str:
    return (
        "## Stack\n\n"
        "- **Outer loop**: task-frame admittance (`hybrid_motion`) — PBAC position "
        "feedback on measured FK, variable damping (Dimeas 2016), proactive retract "
        "(Li 2022).\n"
        "- **Inner loop**: WBC slack-QP IK (Escande 2014, ProxQP) + optional CBF "
        "self-collision (Khazoom 2022).\n"
        "- **Nullspace redundancy** (7-DOF RM75 S-R-S):\n"
        "  - `tasks/arm_angle.py` — swivel angle ψ(q) gradient task (Shimizu 2008)\n"
        "  - `tasks/nullspace_task.py` — Liegeois joint centering (Siciliano 1990)\n"
        "  - `tasks/manipulability_task.py` — Yoshikawa ∇μ ascent during joint move (P1)\n"
        "  - `ik_types.project_onto_task_nullspace` — SR-damped Liegeois projection "
        "(Chiaverini 1997) keeps secondary qdot alive near singularities\n"
        "- **Move outer loops** (`loop.py`) — same inner `JointIkController` for all:\n"
        "  - `CartesianTrackOuterLoop` — FK(q_ref) PD + scaled vel_ff; "
        "`robust_qdot_ff` closed-loop nullspace anchor in `d_sin_tool_y.py`\n"
        "  - `JointTrackOuterLoop` — $v=J(q_{meas})(\\dot q_{plan}+k\\Delta q)$; "
        "still a 6D equality for the QP → 1-DOF nullspace alive (centering/CBF)\n"
        "  - Phase cascade: move phase → `AdmittanceOuterLoop` scan phase; "
        "inner loop never restarts; force admittance unaffected by move mode\n"
        "  - Governor: `GovernorFilter` LPF + hysteresis; `qdot_ff` and cartesian "
        "`vel_ff` scaled by governor `scale`; admittance `time_scale` freezes "
        "`v_force_z`\n"
        "- **QP reg**: uniform Euclidean `reg` (`use_mass_weighted_reg: false`) — "
        "min joint motion, not kinetic energy; `use_dyn_nullspace: false`\n"
        "- **Secondary tasks** (`secondary_composer.py`): soft tasks fade at "
        "low σ; `qdot_ff` added after fade; magnitude cap per joint\n"
        "- **Joint limits**: Faverjon velocity damper (`constraint_mgr.py`); "
        "resync anti-windup as smooth lead damper (not box collapse)\n"
        "- **Near-singularity**: Chiaverini SR + $W_{task}\\propto(\\sigma/\\sigma_{ref})^2$; "
        "ProxQP cold-start (`NO_INITIAL_GUESS`) after solve failure\n"
        "- **Feedback**: UDP realtime push (`async_state.py` / `rm_set_realtime_push` "
        "cycle=1, ~200 Hz); no TCP `rm_get_current_arm_state` polling in the control loop.\n"
        "- **Closed on encoders** (CLIK): FK/Jacobian/twist rotation use `q_meas`; "
        "reference-clock governor freezes phase time when tracking error is large.\n"
        "- **Move guards**: `move_arrived` (pose + joint), `require_arrival`, "
        "`robust_joint_qdot_ff`, `scale_qdot_ff_with_governor=False`, "
        "move-phase `centering_suppressed` + `manipulability_active`.\n"
        "- **Command-lead anti-windup**: extra QP velocity bound "
        "(`resync_err_deg`), never a position teleport.\n"
        "- **No send-path low-pass** — SafetyLimiter + QP velocity box only.\n"
        "- **Interface**: `rm_movej_canfd` mode 0 (joint position stream).\n\n"
        f"---\n\n"
        f"## Source files ({n_files} verbatim)\n\n"
    )


def main() -> None:
    missing = [rel for rel, p in SECTIONS if not p.is_file()]
    if missing:
        raise SystemExit("missing files:\n  " + "\n  ".join(missing))

    header = (
        "# Joint-Position WBC + Nullspace Control — Full Source Dump\n\n"
        f"Generated from workspace `{REPO}`. {len(SECTIONS)} files, verbatim "
        "(previous contents replaced on each run).\n\n"
        "Regenerate: `python scripts/gen_debug_ja.py`\n\n"
        + diagnostic_header()
        + stack_header(len(SECTIONS))
    )
    parts = [header]
    for rel, path in SECTIONS:
        parts.append(embed(rel, path))
    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(SECTIONS)} files)")


if __name__ == "__main__":
    main()
