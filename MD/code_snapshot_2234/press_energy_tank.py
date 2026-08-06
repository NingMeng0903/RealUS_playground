"""Press-side energy tank (TDPA-lite) for delayed contact ports.

Observes contact work via force × position increment (not noisy F·v spikes):

    ΔE_k = F_{k-1/2} · (x_k − x_{k-1})

Credits the tank when the tip retracts under load; scales only the active
press drive ``max(e_f + D0·v_r^+ + u_DOB, 0)``. Over-force retract is free.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PressEnergyTankConfig:
    enabled: bool = True
    e_max_j: float = 0.08
    e_initial_j: float = 0.035
    e_min_j: float = 0.0
    credit_gain: float = 1.0
    # Ignore |dx| below this (encoder/quantization floor) [m].
    dx_deadband_m: float = 2.0e-5
    # Seed on acquire / hard recontact.
    seed_on_acquire: bool = True

    @classmethod
    def from_dict(cls, raw: dict) -> PressEnergyTankConfig:
        c = raw.get("hybrid_motion", raw.get("controller", raw))
        p = c.get("press_energy_tank", {})
        if not isinstance(p, dict):
            p = {}
        return cls(
            enabled=bool(p.get("enabled", True)),
            e_max_j=float(p.get("e_max_j", 0.08)),
            e_initial_j=float(p.get("e_initial_j", 0.035)),
            e_min_j=float(p.get("e_min_j", 0.0)),
            credit_gain=float(p.get("credit_gain", 1.0)),
            dx_deadband_m=float(p.get("dx_deadband_m", 2.0e-5)),
            seed_on_acquire=bool(p.get("seed_on_acquire", True)),
        )


class PressEnergyTank:
    def __init__(self, cfg: PressEnergyTankConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.energy_j = float(self.cfg.e_initial_j)
        self.gamma = 1.0
        self._f_prev = float("nan")
        self._x = 0.0
        self._have_x = False

    def seed(self) -> None:
        if self.cfg.seed_on_acquire:
            self.energy_j = float(self.cfg.e_initial_j)
            self.gamma = 1.0

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
        dx = float(dx_m)
        if abs(dx) < float(cfg.dx_deadband_m):
            dx = 0.0
        f_prev = self._f_prev if self._f_prev == self._f_prev else f
        f_mid = 0.5 * (f + float(f_prev))
        self._f_prev = f

        # Work done on the environment (press + force → spring energy in).
        dW = f_mid * dx
        # Retract under load returns energy / dissipates — credit the tank.
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
