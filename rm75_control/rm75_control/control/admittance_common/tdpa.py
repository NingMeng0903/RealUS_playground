"""De Stefano / Balachandran / Secchi 2020 T-RO Sec. IV TDPA.

Hardware match: position-commanded industrial arm, F/T, no torque port.
Observer is measured force × commanded velocity on the same tick.  FK
velocity never enters.  Sec. V (passive Euler) is not implemented — the
admittance already uses exact ZOH, so discretisation energy is identically
zero.

This layer certifies an energy bound (marginal passivity of the port).
It does not certify no-bouncing (Franken T-RO §V-C5; Ferraguti Eq. 7).

Hardening the paper does not provide and does not analyze:
* leak on the *positive* side of E_obs (otherwise a phantom reservoir
  silently disables the observer forever);
* F/T bias compensation;
* α clamp.  **Passivity does not hold while α is clamped.**

Td and D are runtime quantities.  Do not bake 55 ms or D=40 into this
module — a later voice-coil inner loop only changes the plant delay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TdpaConfig:
    enabled: bool = False
    e_leak_pos_s: float = 2.0
    alpha_max: float = 400.0
    bias_lpf_s: float = 8.0
    v_bias_gate_m_s: float = 0.003
    e0_j: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict) -> "TdpaConfig":
        root = raw if isinstance(raw, dict) else {}
        controller = root.get("hybrid_motion", root.get("controller", root))
        if not isinstance(controller, dict):
            controller = root
        block = controller.get("tdpa", root.get("tdpa", {}))
        if not isinstance(block, dict):
            block = {}
        return cls(
            enabled=bool(block.get("enabled", False)),
            e_leak_pos_s=float(block.get("e_leak_pos_s", 2.0)),
            alpha_max=float(block.get("alpha_max", 400.0)),
            bias_lpf_s=float(block.get("bias_lpf_s", 8.0)),
            v_bias_gate_m_s=float(block.get("v_bias_gate_m_s", 0.003)),
            e0_j=float(block.get("e0_j", 0.0)),
        )


class TimeDomainPassivityObserver:
    """Sec. IV observer: Fc = Fe − α v_cmd, α from E_obs < 0."""

    def __init__(self, cfg: TdpaConfig | None = None) -> None:
        self.cfg = cfg or TdpaConfig()
        self.e_obs_j = float(self.cfg.e0_j)
        self.alpha = 0.0
        self.alpha_clamped = False
        self.f_bias_n = 0.0
        self.f_comp_n = 0.0
        self.fc_n = 0.0
        self.passivity_holds = True

    def reset(self) -> None:
        self.e_obs_j = float(self.cfg.e0_j)
        self.alpha = 0.0
        self.alpha_clamped = False
        self.f_bias_n = 0.0
        self.f_comp_n = 0.0
        self.fc_n = 0.0
        self.passivity_holds = True

    def preview(self, f_meas_n: float, v_cmd_m_s: float, dt_s: float = 0.005) -> float:
        """Return Fc = Fe − α v using energy left *before* this tick.

        Paper: α = −E_obs / (V² T) when E_obs < 0.  T is the sample period.
        """
        if not self.cfg.enabled:
            self.fc_n = float(f_meas_n)
            self.alpha = 0.0
            self.alpha_clamped = False
            self.passivity_holds = True
            return self.fc_n
        fe = float(f_meas_n) - float(self.f_bias_n)
        self.f_comp_n = fe
        v = float(v_cmd_m_s)
        dt = max(float(dt_s), 1e-6)
        alpha = 0.0
        clamped = False
        if self.e_obs_j < 0.0 and v * v * dt > 1e-16:
            raw = -self.e_obs_j / (v * v * dt)
            limit = max(float(self.cfg.alpha_max), 0.0)
            if limit > 0.0 and raw > limit:
                alpha = limit
                clamped = True
            else:
                alpha = max(raw, 0.0)
        self.alpha = float(alpha)
        self.alpha_clamped = bool(clamped)
        # Clamped α is a hardening the paper does not analyze.
        self.passivity_holds = not clamped
        self.fc_n = fe - self.alpha * v
        return float(self.fc_n)

    def commit(
        self,
        f_meas_n: float,
        v_cmd_m_s: float,
        dt_s: float,
        *,
        in_contact: bool = False,
    ) -> None:
        """Accumulate F_meas × v_cmd on this tick; leak only E>0.

        Bias is an air F/T offset.  Updating it in contact would absorb the
        contact load and lock the corridor at F*+budget.
        """
        if not self.cfg.enabled:
            return
        dt = max(float(dt_s), 0.0)
        v = float(v_cmd_m_s)
        f = float(f_meas_n)
        gate = max(float(self.cfg.v_bias_gate_m_s), 0.0)
        tau_b = max(float(self.cfg.bias_lpf_s), 1e-3)
        if dt > 0.0 and abs(v) <= gate and not in_contact:
            alpha_b = 1.0 - math.exp(-dt / tau_b)
            self.f_bias_n += alpha_b * (f - self.f_bias_n)
        fe = f - float(self.f_bias_n)
        self.f_comp_n = fe
        self.e_obs_j += fe * v * dt
        tau_l = max(float(self.cfg.e_leak_pos_s), 1e-3)
        if self.e_obs_j > 0.0 and dt > 0.0:
            self.e_obs_j *= math.exp(-dt / tau_l)
