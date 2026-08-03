"""Normal-axis force disturbance observer (DOSMAC-lite).

Models unmeasured contact disturbance (stiffness change, surface motion,
model error) as a scalar ``d`` on the tool-Z force equation and compensates
it with a leaky integrator on the deadbanded force error:

    u_dob ← u_dob + dt · (ki · e_f − u_dob / leak_s)
    M · v̇ + D · (v − v_r) = e_f + u_dob

Frozen while the Dimeas index is high so the observer does not wind up on
contact chatter.  Caps prevent fighting the passive admittance during impact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForceDobConfig:
    # Default off so unit tests keep the passive admittance baseline; YAML
    # enables this for hardware constant-force tracking.
    enabled: bool = False
    ki: float = 6.0
    leak_s: float = 0.45
    u_max_n: float = 1.5
    freeze_is: float = 0.45
    reset_on_reversal: bool = True

    @classmethod
    def from_dict(cls, parent: dict) -> ForceDobConfig:
        d = parent.get("force_dob", {})
        if not isinstance(d, dict):
            d = {}
        return cls(
            enabled=bool(d.get("enabled", False)),
            ki=float(d.get("ki", 6.0)),
            leak_s=float(d.get("leak_s", 0.45)),
            u_max_n=float(d.get("u_max_n", 1.5)),
            freeze_is=float(d.get("freeze_is", 0.45)),
            reset_on_reversal=bool(d.get("reset_on_reversal", True)),
        )


class ForceDisturbanceObserver:
    """Leaky PI-style disturbance estimate on the normal force error."""

    def __init__(self, cfg: ForceDobConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.u_dob = 0.0
        self.frozen = False
        self._last_eff = 0.0

    def update(
        self,
        eff: float,
        *,
        dt_eff: float,
        in_contact: bool,
        instability_index: float,
        chase_scale: float = 1.0,
    ) -> float:
        cfg = self.cfg
        if not cfg.enabled:
            self.u_dob = 0.0
            self.frozen = False
            self._last_eff = float(eff)
            return 0.0
        if not in_contact or dt_eff <= 0.0:
            if not in_contact and cfg.leak_s > 1e-6 and dt_eff > 0.0:
                self.u_dob -= (dt_eff / cfg.leak_s) * self.u_dob
            self.frozen = False
            self._last_eff = float(eff)
            return float(self.u_dob)

        # Do not let a press-side disturbance estimate fight an over-force
        # escape (or the reverse).
        if (
            cfg.reset_on_reversal
            and abs(float(eff)) > 1e-9
            and self._last_eff * float(eff) < 0.0
        ):
            self.u_dob = 0.0

        freeze = float(instability_index) >= float(cfg.freeze_is)
        self.frozen = freeze
        if not freeze:
            # Soften DOB integration on under-force when tangential speed is low
            # (scan turnaround); keep full ki for over-force escape.
            ki_scale = (
                1.0
                if float(eff) < 0.0
                else float(np.clip(chase_scale, 0.0, 1.0))
            )
            self.u_dob += dt_eff * float(cfg.ki) * ki_scale * float(eff)
        if cfg.leak_s > 1e-6:
            self.u_dob -= (dt_eff / cfg.leak_s) * self.u_dob
        if cfg.u_max_n > 0.0:
            self.u_dob = float(
                np.clip(self.u_dob, -cfg.u_max_n, cfg.u_max_n)
            )
        self._last_eff = float(eff)
        return float(self.u_dob)
