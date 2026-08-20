"""Energy-aware leaky force-error reference for the tool-Z ``v_r`` slot.

This is an engineering complement to the 2nd-order admittance loop:

    M · v̇ + D · (v − v_r) = F_err

It is **not** the human-input observer or Eq. (23)/(35) controller from
Li et al. (2022): it has no human dynamics model or observer-error dynamics.
It keeps the hardware-tested 0.3 s short-memory structure and a
setpoint-normalized drive.  The two signs have the same small-error gain, but
their safety treatment follows contact power:

* ``eff > 0`` presses farther into the surface and can inject contact energy,
  so Dimeas attenuates this branch as high-frequency instability rises;
* ``eff < 0`` releases an over-force contact, so Dimeas must not suppress the
  escape direction.  Its drive is still bounded, and the virtual
  mass/critical damping remain active in the passive admittance layer.

Bidirectional integration (``retract_only=False``) gives the "error-large →
proactive chase" hand feel on both press and retract.  Its guards are:

* leaky decay toward zero (``leak_s``);
* |v_r| ≤ ``v_r_max_m_s`` (< unified tool-Z cap — leaves headroom for D·v);
* only energy-injecting press fades as Dimeas Iₛ → ``press_is_gate``;
* bounded normalized drive on both signs;
* same-contact error reversal projects away an old, opposing ``v_r``;
* Åström anti-windup at both the reference and force-velocity caps;
* the caller clears either sign on contact re-acquire.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProactiveFfConfig:
    enabled: bool = True
    retract_only: bool = False
    # Small-error normalized gains [m/s²].  They default equal; the
    # directional difference comes from the press-only energy gate and the
    # over-force branch not being closed by the instability gate.
    gain: float = 0.10
    retract_gain: float = 0.10
    leak_s: float = 0.3         # leak time constant [s]
    v_r_max_m_s: float = 0.06
    # Energy-injecting press stays fully available below ``gate_start``, then
    # fades linearly to zero at ``press_is_gate``.  Retraction is an
    # over-force escape and is deliberately not gated.
    press_is_gate_start: float = 0.0
    press_is_gate: float = 0.5
    # When False, under-force press chase is never closed by Dimeas Iₛ
    # (over-force retract was already ungated). Chatter dissipation is left
    # to short-lived ΔD_hf in the passive admittance layer.
    gate_press_on_is: bool = True
    # Soft press attenuation vs Iₛ even when gate_press_on_is is False:
    # floor at Iₛ≥press_is_soft_stop (1=no soft atten). Stops single-tick
    # force dips from slamming v_r to the cap ("frame-drop" feel).
    press_is_soft_floor: float = 0.45
    press_is_soft_stop: float = 0.85
    # Max rising slew on press-side v_r [m/s²].
    press_slew_max_m_s2: float = 0.35
    retract_slew_max_m_s2: float = 0.35
    force_scale_min_n: float = 0.30
    force_scale_fraction: float = 0.0
    press_drive_max: float = 1.0
    retract_drive_max: float = 1.0
    reset_on_reversal: bool = True
    in_band_n: float = 0.25
    in_band_leak_s: float = 0.05

    @classmethod
    def from_dict(cls, raw: dict) -> ProactiveFfConfig:
        p = raw.get("proactive_ff", raw)
        if not isinstance(p, dict):
            p = raw
        gain = float(p.get("gain", p.get("proactive_gain", 0.10)))
        return cls(
            enabled=bool(p.get("enabled", p.get("proactive_feedforward", True))),
            retract_only=bool(p.get("retract_only", p.get("proactive_retract_only", False))),
            gain=gain,
            retract_gain=float(
                p.get(
                    "retract_gain",
                    p.get("proactive_retract_gain", gain),
                )
            ),
            leak_s=float(p.get("leak_s", p.get("proactive_leak_s", 0.3))),
            v_r_max_m_s=float(p.get("v_r_max_m_s", 0.06)),
            press_is_gate_start=float(
                p.get(
                    "press_is_gate_start",
                    p.get("proactive_press_is_gate_start", 0.0),
                )
            ),
            press_is_gate=float(p.get("press_is_gate", p.get("proactive_press_is_gate", 0.5))),
            gate_press_on_is=bool(
                p.get(
                    "gate_press_on_is",
                    p.get("proactive_gate_press_on_is", True),
                )
            ),
            press_is_soft_floor=float(
                p.get(
                    "press_is_soft_floor",
                    p.get("proactive_press_is_soft_floor", 0.45),
                )
            ),
            press_is_soft_stop=float(
                p.get(
                    "press_is_soft_stop",
                    p.get("proactive_press_is_soft_stop", 0.85),
                )
            ),
            press_slew_max_m_s2=float(
                p.get(
                    "press_slew_max_m_s2",
                    p.get("proactive_press_slew_max_m_s2", 0.35),
                )
            ),
            retract_slew_max_m_s2=float(
                p.get(
                    "retract_slew_max_m_s2",
                    p.get(
                        "proactive_retract_slew_max_m_s2",
                        p.get(
                            "press_slew_max_m_s2",
                            p.get("proactive_press_slew_max_m_s2", 0.35),
                        ),
                    ),
                )
            ),
            force_scale_min_n=float(p.get("force_scale_min_n", 0.30)),
            force_scale_fraction=float(p.get("force_scale_fraction", 0.0)),
            press_drive_max=float(
                p.get(
                    "press_drive_max",
                    p.get("proactive_press_drive_max", 1.0),
                )
            ),
            retract_drive_max=float(
                p.get(
                    "retract_drive_max",
                    p.get("proactive_retract_drive_max", 1.0),
                )
            ),
            reset_on_reversal=bool(
                p.get(
                    "reset_on_reversal",
                    p.get("proactive_reset_on_reversal", True),
                )
            ),
            in_band_n=float(p.get("in_band_n", p.get("proactive_in_band_n", 0.25))),
            in_band_leak_s=float(
                p.get("in_band_leak_s", p.get("proactive_in_band_leak_s", 0.05))
            ),
        )


class ProactiveForceIntegrator:
    """Leaky normalized reference integrator with contact-power guards."""

    def __init__(self, cfg: ProactiveFfConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.v_r = 0.0
        self.last_force_scale_n = float("nan")
        self.last_drive = 0.0
        self.last_instability_scale = 1.0
        self.last_reference_accel_m_s2 = 0.0
        self.last_reversal_reset = False
        self.last_fast_retract_clear = False

    def update(
        self,
        eff: float,
        *,
        in_contact: bool,
        dt_eff: float,
        instability_index: float,
        v_force_z: float,
        v_z_cap: float,
        desired_force_n: float = 0.0,
        retract_fast_hold: bool = False,
        chase_scale: float = 1.0,
        overforce_escape: bool = False,
    ) -> float:
        cfg = self.cfg
        if not cfg.enabled:
            self.v_r = 0.0
            self.last_drive = 0.0
            self.last_instability_scale = 1.0
            self.last_reference_accel_m_s2 = 0.0
            self.last_reversal_reset = False
            self.last_fast_retract_clear = False
            return 0.0

        self.last_fast_retract_clear = False
        # The raw-force veto is a safety correction and must still remove a
        # stale retracting reference when the trajectory governor has frozen
        # its reference clock (dt_eff == 0).  It does not advance any
        # integrator state.
        if retract_fast_hold and self.v_r < 0.0:
            self.v_r = 0.0
            self.last_fast_retract_clear = True
        if dt_eff <= 0.0:
            return self.v_r

        force_scale = max(
            cfg.force_scale_min_n,
            cfg.force_scale_fraction * abs(float(desired_force_n)),
            1e-6,
        )
        drive_unclamped = float(eff) / force_scale
        if eff < 0.0:
            drive = float(
                np.clip(
                    drive_unclamped,
                    -max(cfg.retract_drive_max, 0.0),
                    0.0,
                )
            )
        else:
            drive = float(
                np.clip(
                    drive_unclamped,
                    0.0,
                    max(cfg.press_drive_max, 0.0),
                )
            )
        self.last_force_scale_n = force_scale
        self.last_drive = drive
        self.last_instability_scale = 1.0
        self.last_reference_accel_m_s2 = 0.0
        self.last_reversal_reset = False
        # The fast raw-force path is a one-way veto only.  It may remove an
        # already negative active reference when the raw force has fallen
        # ahead of the delayed 6 Hz control force, but it cannot command a
        # press and it never clears the passive admittance velocity.

        has_effective_error = in_contact and abs(eff) > 1e-12
        integrate = has_effective_error
        if integrate and cfg.retract_only and eff > 0.0:
            integrate = False
        if integrate and retract_fast_hold and eff < 0.0:
            integrate = False

        # Do not let the previous direction spend 0.2--0.5 s fighting a new
        # force error.  The passive admittance velocity is intentionally not
        # reset; M and D still make the actual TCP-Z reversal continuous.
        if (
            has_effective_error
            and cfg.reset_on_reversal
            and self.v_r * float(eff) < 0.0
        ):
            self.v_r = 0.0
            self.last_reversal_reset = True

        if cfg.leak_s > 1e-6:
            leak_s = float(cfg.leak_s)
            if abs(float(eff)) < max(float(cfg.in_band_n), 0.0) and cfg.in_band_leak_s > 1e-6:
                leak_s = min(leak_s, float(cfg.in_band_leak_s))
            self.v_r -= (dt_eff / leak_s) * self.v_r

        if integrate:
            if eff < 0.0:
                # Over-force retraction releases contact energy.  Never let an
                # instability detector close the escape route.
                step = cfg.retract_gain * drive
            else:
                # Slow tangential scan / turnaround: soften under-force chase
                # so force-axis motion does not feel like a lateral jerk.
                step = cfg.gain * drive * float(
                    np.clip(chase_scale, 0.0, 1.0)
                )
            if step > 0.0:
                if cfg.gate_press_on_is and cfg.press_is_gate > 1e-9:
                    gate_stop = max(float(cfg.press_is_gate), 1e-9)
                    gate_start = float(
                        np.clip(cfg.press_is_gate_start, 0.0, gate_stop)
                    )
                    if instability_index <= gate_start:
                        self.last_instability_scale = 1.0
                    elif gate_stop <= gate_start + 1e-9:
                        self.last_instability_scale = 0.0
                    else:
                        self.last_instability_scale = float(
                            np.clip(
                                1.0
                                - (instability_index - gate_start)
                                / (gate_stop - gate_start),
                                0.0,
                                1.0,
                            )
                        )
                    step *= self.last_instability_scale
                else:
                    # Soft floor: never fully kill press, but blunt noise dips.
                    soft_stop = max(float(cfg.press_is_soft_stop), 1e-9)
                    soft_floor = float(
                        np.clip(cfg.press_is_soft_floor, 0.0, 1.0)
                    )
                    if instability_index <= 0.0 or soft_floor >= 1.0 - 1e-9:
                        self.last_instability_scale = 1.0
                    elif instability_index >= soft_stop:
                        self.last_instability_scale = soft_floor
                    else:
                        u = float(instability_index / soft_stop)
                        blend = u * u * (3.0 - 2.0 * u)
                        self.last_instability_scale = float(
                            1.0 - blend * (1.0 - soft_floor)
                        )
                    step *= self.last_instability_scale

            # Conditional integration at both saturation layers.  Motion back
            # toward the admissible set is always allowed.
            v_r_cap = max(float(cfg.v_r_max_m_s), 0.0)
            at_negative_cap = (
                (v_z_cap > 0.0 and v_force_z <= -v_z_cap + 1e-6)
                or (v_r_cap > 0.0 and self.v_r <= -v_r_cap + 1e-6)
            )
            at_positive_cap = (
                (v_z_cap > 0.0 and v_force_z >= v_z_cap - 1e-6)
                or (v_r_cap > 0.0 and self.v_r >= v_r_cap - 1e-6)
            )
            if (step < 0.0 and at_negative_cap) or (
                step > 0.0 and at_positive_cap
            ):
                step = 0.0
            # Symmetric slew in regulate; over-force escape skips the retract slew.
            if not overforce_escape:
                if step > 0.0 and cfg.press_slew_max_m_s2 > 0.0:
                    step = min(step, float(cfg.press_slew_max_m_s2))
                elif step < 0.0 and cfg.retract_slew_max_m_s2 > 0.0:
                    step = max(step, -float(cfg.retract_slew_max_m_s2))
            self.last_reference_accel_m_s2 = float(step)
            self.v_r += dt_eff * step

        if cfg.v_r_max_m_s > 0.0:
            self.v_r = float(np.clip(self.v_r, -cfg.v_r_max_m_s, cfg.v_r_max_m_s))
        if v_z_cap > 0.0:
            self.v_r = float(np.clip(self.v_r, -v_z_cap, v_z_cap))
        return self.v_r
