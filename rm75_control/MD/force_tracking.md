# Active TCP-Z force control

The force controller is `control/admittance_common/controller.py`.
`peirastic` reloads `peirastic/configs/force.yaml` for each force-mode request,
then applies that request's payload overrides. Machine, QPIK and rail settings
come from `peirastic/configs/controller.yaml`.

The standalone 8-DOF and 7-DOF scan scripts use, respectively,
`configs/joint_admittance_8dof.yaml` and `configs/joint_admittance.yaml`.
These configurations enable different shared features.

## Retained execution paths

- Tool-Z uses the existing mass/damping force admittance with exact ZOH
  integration, force deadband and desired-force ramp.
- Motion tracking, force-point integration, normal-direction limits and
  emitted-command anti-windup retain their existing order.
- Physical contact detection, first-touch/recontact limits, adaptive stiffness
  estimation and the configured stiffness schedule remain active.
- The 7-DOF configuration also uses proactive force feedforward, variable
  inertia/damping, fast-retract guarding and the force-space velocity damper.
- In 8-DOF configurations the damper's command clamp is disabled, but its
  approach speed, force prediction and recontact parameters still feed the
  running controller. Its configuration is therefore retained.
- Peirastic keeps its enabled torque-tilt/TFF path and the FCE law requested
  explicitly by `DEMO/hover.py`.
- QPIK, rail allocation, IRD, collision constraints, native execution and the
  Python backend used by dry runs and offline tools remain available.

TDPA, bidirectional-flow and safety-shield observations remain in the
configured diagnostic paths, including their dynamic telemetry. These
observations are not a patient-port passivity certificate.

## Removed inactive mechanisms

The following mechanisms were disabled in all current shipped force
configurations and were not enabled by the normal app/DEMO overrides:

- force disturbance observer (`force_dob`);
- old active-term energy tank (`energy_tank`);
- combined-dynamics observer (`cdyob`);
- two-sided force-corridor clamp (`force_corridor`);
- tangential-error desired-force modulation (`surface_force_modulation`).

Their implementations, controller branches and inactive CSV columns are
removed. `PressEnvelopeConfig` remains in `admittance_common/press_envelope.py`
with its original parsing defaults. Plant identification remains available;
the CDYOB-specific replay command is removed.

Old disabled blocks are accepted when reading existing configurations.
Requests to activate a removed mechanism are rejected during configuration
or force-payload loading.

## Offline regression

`tests/test_active_force_regression.py` compares emitted velocity, motion
components and active state/telemetry against traces captured before removal.
It covers both shipped controller configurations and representative force
payload overrides. The fixture must only be regenerated for an intentional
change to controller behavior.

Run from `rm75_control` after sourcing `env.sh`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests ../peirastic/tests -q
```

Offline equivalence checks do not measure robot timing, contact-force peaks
or image quality on hardware.
