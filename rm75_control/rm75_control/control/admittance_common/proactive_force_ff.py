"""Leaky ∫F_err proactive reference for the tool-Z v_r slot (Eq. 35).

Engineering complement to the 2nd-order admittance loop:

    M · v̇ + D · (v − v_r) = F_err

Bidirectional integration (``retract_only=False``) gives the "error-large →
proactive chase" hand feel on both press and retract.  Anti-bounce guards
(borrowed from the hardware-tested path, not extra parallel modules):

* leaky decay toward zero (``leak_s``);
* |v_r| ≤ ``v_r_max_m_s`` (< unified tool-Z cap — leaves headroom for D·v);
* press-side step fades as Dimeas Iₛ → ``press_is_gate`` (retract ungated);
* Åström anti-windup: freeze integration when ``v_force_z`` is cap-saturated;
* rising-edge press reset (caller clears v_r > 0 on contact re-acquire).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ProactiveFfConfig:
    enabled: bool = True
    retract_only: bool = False
    gain: float = 0.10          # γ [(m/s²)/N]  →  v̇_r = γ·eff
    leak_s: float = 0.3         # leak time constant [s]
    v_r_max_m_s: float = 0.06
    # Press-side integration fades linearly to zero as Iₛ → this value.
    # Retract side stays ungated. 0 disables the gate.
    press_is_gate: float = 0.5

    @classmethod
    def from_dict(cls, raw: dict) -> ProactiveFfConfig:
        p = raw.get("proactive_ff", raw)
        if not isinstance(p, dict):
            p = raw
        return cls(
            enabled=bool(p.get("enabled", p.get("proactive_feedforward", True))),
            retract_only=bool(p.get("retract_only", p.get("proactive_retract_only", False))),
            gain=float(p.get("gain", p.get("proactive_gain", 0.10))),
            leak_s=float(p.get("leak_s", p.get("proactive_leak_s", 0.3))),
            v_r_max_m_s=float(p.get("v_r_max_m_s", 0.06)),
            press_is_gate=float(p.get("press_is_gate", p.get("proactive_press_is_gate", 0.5))),
        )


class ProactiveForceIntegrator:
    """Leaky integrator: v̇_r = γ·eff with bounce guards."""

    def __init__(self, cfg: ProactiveFfConfig) -> None:
        self.cfg = cfg
        self.v_r = 0.0

    def reset(self) -> None:
        self.v_r = 0.0

    def clear_press_on_rising_edge(self) -> None:
        """Drop accumulated press-side feedforward on contact re-acquire."""
        if self.v_r > 0.0:
            self.v_r = 0.0

    def update(
        self,
        eff: float,
        *,
        in_contact: bool,
        dt_eff: float,
        instability_index: float,
        v_force_z: float,
        v_z_cap: float,
    ) -> float:
        cfg = self.cfg
        if not cfg.enabled:
            self.v_r = 0.0
            return 0.0
        if dt_eff <= 0.0:
            return self.v_r

        if cfg.leak_s > 1e-6:
            self.v_r -= (dt_eff / cfg.leak_s) * self.v_r

        integrate = in_contact and abs(eff) > 1e-12
        if integrate and cfg.retract_only and eff > 0.0:
            integrate = False
        if integrate:
            step = cfg.gain * eff
            if step > 0.0 and cfg.press_is_gate > 1e-9:
                step *= float(
                    np.clip(1.0 - instability_index / cfg.press_is_gate, 0.0, 1.0)
                )
            if v_z_cap > 0.0:
                if step < 0.0 and v_force_z <= -v_z_cap + 1e-6:
                    step = 0.0
                elif step > 0.0 and v_force_z >= v_z_cap - 1e-6:
                    step = 0.0
            self.v_r += dt_eff * step

        if cfg.v_r_max_m_s > 0.0:
            self.v_r = float(np.clip(self.v_r, -cfg.v_r_max_m_s, cfg.v_r_max_m_s))
        if v_z_cap > 0.0:
            self.v_r = float(np.clip(self.v_r, -v_z_cap, v_z_cap))
        return self.v_r
