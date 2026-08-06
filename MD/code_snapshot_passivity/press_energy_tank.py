"""Contact-port energy tank + bidirectional passivity observer (PO/PC).

Tank (press budget):
    ΔE = F_mid · Δx   (residual-accumulated position increments)
    scales only active press drive; retract unrestricted.

PO/PC (bidirectional):
    tracks excess energy injected into the real TCP port and adds a short
    dissipative term D_PC · v_a when the observer goes negative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PressEnergyTankConfig:
    enabled: bool = True
    e_max_j: float = 0.004
    e_initial_j: float = 0.001
    e_min_j: float = 0.0
    credit_gain: float = 0.20
    # Residual accumulator floor [m] — small steps accumulate, not discarded.
    dx_deadband_m: float = 2.0e-6
    # Re-seeding on bounce reacquire refills the tank — keep false.
    seed_on_acquire: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> PressEnergyTankConfig:
        c = raw.get("hybrid_motion", raw.get("controller", raw))
        p = c.get("press_energy_tank", {})
        if not isinstance(p, dict):
            p = {}
        return cls(
            enabled=bool(p.get("enabled", True)),
            e_max_j=float(p.get("e_max_j", 0.004)),
            e_initial_j=float(p.get("e_initial_j", 0.001)),
            e_min_j=float(p.get("e_min_j", 0.0)),
            credit_gain=float(p.get("credit_gain", 0.20)),
            dx_deadband_m=float(p.get("dx_deadband_m", 2.0e-6)),
            seed_on_acquire=bool(p.get("seed_on_acquire", False)),
        )


@dataclass
class PortPassivityConfig:
    """Bidirectional real-port passivity observer / controller."""

    enabled: bool = True
    e_max_j: float = 0.004
    e_initial_j: float = 0.002
    # Floor on v_a²·dt in D_PC denominator.
    eps_v2dt: float = 1.0e-8
    d_pc_max: float = 120.0
    # Leak excess back toward zero when dissipating [1/s].
    leak_s: float = 0.5

    @classmethod
    def from_dict(cls, raw: dict) -> PortPassivityConfig:
        c = raw.get("hybrid_motion", raw.get("controller", raw))
        p = c.get("port_passivity", {})
        if not isinstance(p, dict):
            p = {}
        return cls(
            enabled=bool(p.get("enabled", True)),
            e_max_j=float(p.get("e_max_j", 0.004)),
            e_initial_j=float(p.get("e_initial_j", 0.002)),
            eps_v2dt=float(p.get("eps_v2dt", 1.0e-8)),
            d_pc_max=float(p.get("d_pc_max", 120.0)),
            leak_s=float(p.get("leak_s", 0.5)),
        )


class PressEnergyTank:
    def __init__(self, cfg: PressEnergyTankConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.energy_j = float(self.cfg.e_initial_j)
        self.gamma = 1.0
        self._f_prev = float("nan")
        self._dx_residual = 0.0

    def seed(self) -> None:
        if self.cfg.seed_on_acquire:
            self.energy_j = float(self.cfg.e_initial_j)
            self.gamma = 1.0

    def _consume_dx(self, dx_m: float) -> float:
        self._dx_residual += float(dx_m)
        dead = max(float(self.cfg.dx_deadband_m), 0.0)
        if abs(self._dx_residual) < dead:
            return 0.0
        dx_used = self._dx_residual
        self._dx_residual = 0.0
        return dx_used

    def observe_and_scale(
        self,
        *,
        f_ext_z: float,
        dx_m: float,
        u_press: float,
        v_press_est_m_s: float,
        dt_s: float,
    ) -> float:
        """Credit/debit tank; return γ ∈ [0, 1] for active press scaling."""
        cfg = self.cfg
        if not cfg.enabled:
            self.gamma = 1.0
            return 1.0

        f = float(f_ext_z)
        dx = self._consume_dx(dx_m)
        f_prev = self._f_prev if self._f_prev == self._f_prev else f
        f_mid = 0.5 * (f + float(f_prev))
        self._f_prev = f

        dW = f_mid * dx
        # Partial credit only — elastic return must not fully refill the tank.
        if dW < 0.0 and f_mid > 0.0:
            self.energy_j = min(
                float(cfg.e_max_j),
                self.energy_j + (-dW) * float(cfg.credit_gain),
            )

        u_p = max(float(u_press), 0.0)
        v_p = max(float(v_press_est_m_s), 0.0)
        dt = max(float(dt_s), 0.0)
        e_req = u_p * v_p * dt
        if e_req <= 1e-12 or u_p <= 1e-12:
            self.gamma = 1.0
            return 1.0

        e_avail = max(float(self.energy_j) - float(cfg.e_min_j), 0.0)
        gamma = min(1.0, e_avail / e_req)
        self.energy_j = max(
            float(cfg.e_min_j),
            self.energy_j - gamma * e_req,
        )
        self.gamma = float(gamma)
        return self.gamma


class PortPassivityObserver:
    """Bidirectional PO/PC on the real TCP contact port."""

    def __init__(self, cfg: PortPassivityConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.energy_j = float(self.cfg.e_initial_j)
        self.excess_j = 0.0
        self.d_pc = 0.0
        self._f_prev = float("nan")
        self._dx_residual = 0.0

    def seed(self) -> None:
        self.energy_j = float(self.cfg.e_initial_j)
        self.excess_j = 0.0
        self.d_pc = 0.0

    def update(
        self,
        *,
        f_ext_z: float,
        dx_m: float,
        v_actual_m_s: float,
        dt_s: float,
    ) -> float:
        """Observe port work; return D_PC ≥ 0 (zero-centered on v_a)."""
        cfg = self.cfg
        if not cfg.enabled:
            self.d_pc = 0.0
            self.excess_j = 0.0
            return 0.0

        f = float(f_ext_z)
        self._dx_residual += float(dx_m)
        # Use any accumulated motion; tiny residual stays for next tick.
        dx = self._dx_residual
        if abs(dx) < 1e-9:
            dx_used = 0.0
        else:
            dx_used = dx
            self._dx_residual = 0.0

        f_prev = self._f_prev if self._f_prev == self._f_prev else f
        f_mid = 0.5 * (f + float(f_prev))
        self._f_prev = f
        dt = max(float(dt_s), 0.0)

        # Work done ON the environment by the tip.
        dW_env = f_mid * dx_used
        # Robot energy storage relative to port: decreases when env is loaded.
        self.energy_j -= dW_env
        if self.energy_j > float(cfg.e_max_j):
            self.energy_j = float(cfg.e_max_j)

        excess = max(0.0, -self.energy_j)  # negative tank ⇒ injected too much
        self.excess_j = float(excess)
        if excess <= 1e-12 or dt <= 0.0:
            self.d_pc = 0.0
            # Mild leak toward initial when passive.
            if self.energy_j < float(cfg.e_initial_j):
                leak = max(float(cfg.leak_s), 0.0)
                self.energy_j += (
                    (float(cfg.e_initial_j) - self.energy_j)
                    * (1.0 - math.exp(-leak * dt))
                )
            return 0.0

        v_a = float(v_actual_m_s)
        denom = max(v_a * v_a * dt, float(cfg.eps_v2dt))
        d_pc = min(float(cfg.d_pc_max), excess / denom)
        self.d_pc = float(d_pc)
        # Apply dissipation credit: E += D_PC · v_a² · dt
        self.energy_j += d_pc * v_a * v_a * dt
        if self.energy_j > 0.0:
            self.energy_j = min(self.energy_j, float(cfg.e_max_j))
        return self.d_pc
