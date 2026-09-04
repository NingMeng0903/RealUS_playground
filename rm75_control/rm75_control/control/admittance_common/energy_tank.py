"""Secchi-style tank that only pays active admittance terms.

Port is (F_meas, v_cmd).  Dissipation D v² charges the tank.  Chase v_r,
force-DOB, and the F* source draw from it.  λ scales those active terms
before they enter the integrator; an empty tank zeros v_r and u_DOB and
leaves the passive M, D, e_f law.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EnergyTankConfig:
    enabled: bool = False
    eps_j: float = 0.08
    t_soft_j: float = 0.25
    t_bar_j: float = 1.0
    t0_j: float = 0.50

    @classmethod
    def from_dict(cls, raw: dict) -> "EnergyTankConfig":
        root = raw if isinstance(raw, dict) else {}
        controller = root.get("hybrid_motion", root.get("controller", root))
        if not isinstance(controller, dict):
            controller = root
        block = controller.get("energy_tank", root.get("energy_tank", {}))
        if not isinstance(block, dict):
            block = {}
        return cls(
            enabled=bool(block.get("enabled", False)),
            eps_j=float(block.get("eps_j", 0.08)),
            t_soft_j=float(block.get("t_soft_j", 0.25)),
            t_bar_j=float(block.get("t_bar_j", 1.0)),
            t0_j=float(block.get("t0_j", 0.50)),
        )


class ActiveTermTank:
    """Discrete tank with Secchi Euler surplus deducted from the budget."""

    def __init__(self, cfg: EnergyTankConfig | None = None) -> None:
        self.cfg = cfg or EnergyTankConfig()
        self.energy_j = float(self.cfg.t0_j)
        self.lambda_scale = 1.0
        self.drained = False
        self.power_w = 0.0

    def reset(self) -> None:
        self.energy_j = float(self.cfg.t0_j)
        self.lambda_scale = 1.0
        self.drained = False
        self.power_w = 0.0

    def update(
        self,
        *,
        damping: float,
        v_cmd: float,
        v_r: float,
        u_dob: float,
        f_star: float,
        dt_s: float,
    ) -> float:
        if not self.cfg.enabled:
            self.lambda_scale = 1.0
            self.drained = False
            self.power_w = 0.0
            return 1.0
        dt = max(float(dt_s), 0.0)
        d = max(float(damping), 0.0)
        v = float(v_cmd)
        dissip = d * v * v
        active = d * float(v_r) * v + float(u_dob) * v + float(f_star) * v
        psi = dissip - active
        self.power_w = float(psi)
        xt2 = max(2.0 * max(self.energy_j, self.cfg.eps_j), 1e-9)
        delta = (dt * dt * (active * active)) / xt2 if dt > 0.0 else 0.0
        self.energy_j += dt * psi - delta
        eps = max(float(self.cfg.eps_j), 0.0)
        t_bar = max(float(self.cfg.t_bar_j), eps)
        self.energy_j = float(min(max(self.energy_j, 0.0), t_bar))
        t_soft = max(float(self.cfg.t_soft_j), eps + 1e-9)
        if self.energy_j <= eps + 1e-12:
            self.lambda_scale = 0.0
            self.drained = True
        else:
            self.lambda_scale = float(
                min(1.0, max(0.0, (self.energy_j - eps) / (t_soft - eps)))
            )
            self.drained = False
        return self.lambda_scale
