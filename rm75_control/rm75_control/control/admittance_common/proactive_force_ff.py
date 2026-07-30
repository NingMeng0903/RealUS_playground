"""Energy-aware leaky force-error reference for the tool-Z ``v_r`` slot.

This is an engineering complement to the 2nd-order admittance loop:

    M · v̇ + D · (v − v_r) = F_err

It keeps the hardware-tested short-memory structure.  Two gain modes:

* ``fixed`` — setpoint-normalized drive (legacy, equal small-error gain);
* ``ke_normalized`` — Li-2022-style ``v_r_target = e / (K̂e · τ)`` so the same
  Newton of force error asks for far less motion on a stiff surface.

Directional safety is unchanged:

* ``eff > 0`` presses farther into the surface and can inject contact energy,
  so Dimeas attenuates this branch as high-frequency instability rises;
* ``eff < 0`` releases an over-force contact, so Dimeas must not suppress the
  escape direction.

Guards: leaky decay, |v_r| caps, press-only energy gate, reversal reset,
Åström anti-windup, and optional Li amplitude-coupled leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProactiveFfConfig:
    enabled: bool = True
    retract_only: bool = False
    # Small-error normalized gains [m/s²] for ``gain_mode=fixed``.
    gain: float = 0.10
    retract_gain: float = 0.10
    leak_s: float = 0.3
    v_r_max_m_s: float = 0.06
    press_is_gate_start: float = 0.0
    press_is_gate: float = 0.5
    force_scale_min_n: float = 0.30
    force_scale_fraction: float = 0.15
    press_drive_max: float = 1.0
    retract_drive_max: float = 1.0
    reset_on_reversal: bool = True
    # ``fixed`` = legacy setpoint-normalized; ``ke_normalized`` = 1/(Ke·τ).
    gain_mode: str = "ke_normalized"
    tau_ff_s: float = 0.20
    ke_floor_ff: float = 80.0
    tau_track_s: float = 0.08
    # Extra Li-style amplitude-coupled leakage on |v_r| (0 disables).
    alpha_leak: float = 2.0

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
            force_scale_min_n=float(p.get("force_scale_min_n", 0.30)),
            force_scale_fraction=float(p.get("force_scale_fraction", 0.15)),
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
            gain_mode=str(
                p.get(
                    "gain_mode",
                    p.get("proactive_gain_mode", "ke_normalized"),
                )
            ).lower(),
            tau_ff_s=float(p.get("tau_ff_s", 0.20)),
            ke_floor_ff=float(p.get("ke_floor_ff", 80.0)),
            tau_track_s=float(p.get("tau_track_s", 0.08)),
            alpha_leak=float(p.get("alpha_leak", 2.0)),
        )


class ProactiveForceIntegrator:
    """Leaky / Ke-normalized reference integrator with contact-power guards."""

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
        ke_hat: float = 0.0,
    ) -> float:
        cfg = self.cfg
        if not cfg.enabled:
            self.v_r = 0.0
            self.last_drive = 0.0
            self.last_instability_scale = 1.0
            self.last_reference_accel_m_s2 = 0.0
            self.last_reversal_reset = False
            return 0.0
        if dt_eff <= 0.0:
            return self.v_r

        self.last_reversal_reset = False
        self.last_instability_scale = 1.0
        self.last_reference_accel_m_s2 = 0.0

        has_effective_error = in_contact and abs(eff) > 1e-12
        integrate = has_effective_error
        if integrate and cfg.retract_only and eff > 0.0:
            integrate = False

        if (
            has_effective_error
            and cfg.reset_on_reversal
            and self.v_r * float(eff) < 0.0
        ):
            self.v_r = 0.0
            self.last_reversal_reset = True

        # Linear leak toward zero.
        if cfg.leak_s > 1e-6:
            self.v_r -= (dt_eff / cfg.leak_s) * self.v_r
        # Li-style amplitude-coupled leakage: stronger when |v_r| is large.
        if cfg.alpha_leak > 0.0:
            self.v_r -= dt_eff * cfg.alpha_leak * abs(self.v_r) * self.v_r

        if cfg.gain_mode == "ke_normalized":
            self._update_ke_normalized(
                eff,
                integrate=integrate,
                dt_eff=dt_eff,
                instability_index=instability_index,
                v_force_z=v_force_z,
                v_z_cap=v_z_cap,
                ke_hat=ke_hat,
                desired_force_n=desired_force_n,
            )
        else:
            self._update_fixed(
                eff,
                integrate=integrate,
                dt_eff=dt_eff,
                instability_index=instability_index,
                v_force_z=v_force_z,
                v_z_cap=v_z_cap,
                desired_force_n=desired_force_n,
            )

        if cfg.v_r_max_m_s > 0.0:
            self.v_r = float(np.clip(self.v_r, -cfg.v_r_max_m_s, cfg.v_r_max_m_s))
        if v_z_cap > 0.0:
            self.v_r = float(np.clip(self.v_r, -v_z_cap, v_z_cap))
        return self.v_r

    def _press_gate_scale(self, instability_index: float, step: float) -> float:
        cfg = self.cfg
        if step <= 0.0 or cfg.press_is_gate <= 1e-9:
            return 1.0
        gate_stop = max(float(cfg.press_is_gate), 1e-9)
        gate_start = float(np.clip(cfg.press_is_gate_start, 0.0, gate_stop))
        if instability_index <= gate_start:
            return 1.0
        if gate_stop <= gate_start + 1e-9:
            return 0.0
        return float(
            np.clip(
                1.0
                - (instability_index - gate_start) / (gate_stop - gate_start),
                0.0,
                1.0,
            )
        )

    def _anti_windup_blocks(
        self,
        step: float,
        *,
        v_force_z: float,
        v_z_cap: float,
    ) -> bool:
        cfg = self.cfg
        v_r_cap = max(float(cfg.v_r_max_m_s), 0.0)
        at_negative_cap = (
            (v_z_cap > 0.0 and v_force_z <= -v_z_cap + 1e-6)
            or (v_r_cap > 0.0 and self.v_r <= -v_r_cap + 1e-6)
        )
        at_positive_cap = (
            (v_z_cap > 0.0 and v_force_z >= v_z_cap - 1e-6)
            or (v_r_cap > 0.0 and self.v_r >= v_r_cap - 1e-6)
        )
        return (step < 0.0 and at_negative_cap) or (step > 0.0 and at_positive_cap)

    def _update_fixed(
        self,
        eff: float,
        *,
        integrate: bool,
        dt_eff: float,
        instability_index: float,
        v_force_z: float,
        v_z_cap: float,
        desired_force_n: float,
    ) -> None:
        cfg = self.cfg
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

        if not integrate:
            return

        if eff < 0.0:
            step = cfg.retract_gain * drive
        else:
            step = cfg.gain * drive
            scale = self._press_gate_scale(instability_index, step)
            self.last_instability_scale = scale
            step *= scale

        if self._anti_windup_blocks(step, v_force_z=v_force_z, v_z_cap=v_z_cap):
            step = 0.0
        self.last_reference_accel_m_s2 = float(step)
        self.v_r += dt_eff * step

    def _update_ke_normalized(
        self,
        eff: float,
        *,
        integrate: bool,
        dt_eff: float,
        instability_index: float,
        v_force_z: float,
        v_z_cap: float,
        ke_hat: float,
        desired_force_n: float = 0.0,
    ) -> None:
        cfg = self.cfg
        ke_hat = max(float(ke_hat), 1e-6)
        ke_floor = max(float(cfg.ke_floor_ff), 1e-6)
        tau = max(float(cfg.tau_ff_s), 1e-6)
        # Over-force retract:
        # * hand guidance (desired≈0) always uses ke_floor (symmetric feel);
        # * force tracking uses full K̂e only while Iₛ shows ringing, otherwise
        #   ke_floor so quiet retract is not ~10× weaker than press.
        hand_guidance = abs(float(desired_force_n)) < 1e-6
        if (
            float(eff) < 0.0
            and not hand_guidance
            and float(instability_index) > float(cfg.press_is_gate_start)
        ):
            ke = max(ke_hat, ke_floor)
        else:
            ke = ke_floor
        v_r_target = float(eff) / (ke * tau)
        if cfg.v_r_max_m_s > 0.0:
            v_r_target = float(
                np.clip(v_r_target, -cfg.v_r_max_m_s, cfg.v_r_max_m_s)
            )
        if v_z_cap > 0.0:
            v_r_target = float(np.clip(v_r_target, -v_z_cap, v_z_cap))

        self.last_force_scale_n = ke * tau
        self.last_drive = float(eff) / max(ke * tau, 1e-6)

        if not integrate:
            return

        if v_r_target > 0.0:
            scale = self._press_gate_scale(instability_index, v_r_target)
            self.last_instability_scale = scale
            v_r_target *= scale

        tau_track = max(float(cfg.tau_track_s), 1e-6)
        blend = min(1.0, dt_eff / tau_track)
        step = blend * (v_r_target - self.v_r) / max(dt_eff, 1e-9)
        if self._anti_windup_blocks(
            v_r_target - self.v_r,
            v_force_z=v_force_z,
            v_z_cap=v_z_cap,
        ):
            if (v_r_target - self.v_r) * self.v_r >= 0.0:
                return
        self.last_reference_accel_m_s2 = float(step)
        self.v_r += blend * (v_r_target - self.v_r)
