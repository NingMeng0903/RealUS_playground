"""Samuel 2024 combined-dynamics Youla observer at the velocity interface.

The inner loop is a black-box velocity servo.  After ``identify_plant.py``
fills ``tau_s`` (and optionally ``t_n_s``), Q is a first-order low-pass at

    ω_Q = 1 / (2π τ)     (≈ 2.9 Hz for τ = 55 ms)

so the Q-delay phase stays well below 180°.  The paper's 15 Hz Q assumed
td ≈ 3 ms at 1250 Hz and is not used here.

N1 = Q C_n^{-1} A R_n^{-1} and N2 = Q A P_n^{-1} T_n^{-1} are realised as
first-order filters on measured force and measured velocity.  Default
``enabled=false``: turn on only after a free-space step/chirp has confirmed
τ_eff.  The observer never mutates the Lee tank.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CdyobConfig:
    enabled: bool = False
    # 0 → ω_Q = 1/(2π τ).  Never use the paper's 15 Hz on this plant.
    omega_q_hz: float = 0.0
    tau_s: float = 0.0
    # Residual first-order inner-loop time constant after the delay.
    t_n_s: float = 0.020
    # Nominal PI-like inner controller / mass-damper used to shape N1/N2.
    cn_kp: float = 80.0
    cn_ki: float = 400.0
    rn_m: float = 2.5
    rn_b: float = 20.0
    pn_m: float = 0.0
    a_gain: float = 1.0
    v_corr_max_m_s: float = 0.030

    @classmethod
    def from_dict(cls, raw: dict) -> "CdyobConfig":
        c = raw.get("hybrid_motion", raw.get("controller", raw))
        if not isinstance(c, dict):
            c = raw if isinstance(raw, dict) else {}
        p = c.get("cdyob", {})
        if not isinstance(p, dict):
            p = {}
        return cls(
            enabled=bool(p.get("enabled", False)),
            omega_q_hz=float(p.get("omega_q_hz", 0.0)),
            tau_s=float(p.get("tau_s", 0.0)),
            t_n_s=float(p.get("t_n_s", 0.020)),
            cn_kp=float(p.get("cn_kp", 80.0)),
            cn_ki=float(p.get("cn_ki", 400.0)),
            rn_m=float(p.get("rn_m", 2.5)),
            rn_b=float(p.get("rn_b", 20.0)),
            pn_m=float(p.get("pn_m", 0.0)),
            a_gain=float(p.get("a_gain", 1.0)),
            v_corr_max_m_s=float(p.get("v_corr_max_m_s", 0.030)),
        )


class CombinedDynamicsYob:
    """Discrete first-order CDYOB correction on the tool-Z velocity command."""

    def __init__(self, cfg: CdyobConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.q_state = 0.0
        self.n1_state = 0.0
        self.n2_state = 0.0
        self._v_cmd_hist: list[float] = []
        self._v_meas_prev = 0.0
        self.last_corr_m_s = 0.0
        self.last_omega_q_hz = 0.0

    def _omega_q(self, tau_s: float) -> float:
        if self.cfg.omega_q_hz > 1e-9:
            return float(self.cfg.omega_q_hz)
        tau = max(float(tau_s), 1e-3)
        return 1.0 / (2.0 * math.pi * tau)

    def _lpf(self, state: float, x: float, omega_hz: float, dt: float) -> float:
        a = 1.0 - math.exp(-2.0 * math.pi * max(omega_hz, 0.0) * max(dt, 0.0))
        a = float(np.clip(a, 0.0, 1.0))
        return state + a * (x - state)

    def update(
        self,
        v_nom_m_s: float,
        *,
        v_meas_m_s: float | None,
        force_n: float,
        dt_s: float,
        tau_s: float,
        in_contact: bool,
    ) -> float:
        cfg = self.cfg
        if not cfg.enabled or dt_s <= 0.0:
            self.last_corr_m_s = 0.0
            return float(v_nom_m_s)
        tau = float(cfg.tau_s) if cfg.tau_s > 1e-9 else max(float(tau_s), 0.0)
        omega_q = self._omega_q(tau if tau > 1e-9 else 0.055)
        self.last_omega_q_hz = omega_q

        v_meas = (
            float(v_meas_m_s)
            if v_meas_m_s is not None and np.isfinite(v_meas_m_s)
            else float(v_nom_m_s)
        )
        delay_n = max(int(round(tau / dt_s)), 1) if tau > 0.0 else 1
        self._v_cmd_hist.append(float(v_nom_m_s))
        if len(self._v_cmd_hist) > delay_n + 2:
            self._v_cmd_hist = self._v_cmd_hist[-(delay_n + 2) :]
        v_cmd_delayed = (
            self._v_cmd_hist[-delay_n]
            if len(self._v_cmd_hist) >= delay_n
            else self._v_cmd_hist[0]
        )

        # T_n^{-1} V_m ≈ V_m + t_n · ḊV_m, compared with delayed V_i.
        dv = (v_meas - self._v_meas_prev) / dt_s
        self._v_meas_prev = v_meas
        tn_inv_vm = v_meas + max(float(cfg.t_n_s), 0.0) * dv
        inner_err = tn_inv_vm - v_cmd_delayed
        self.q_state = self._lpf(self.q_state, inner_err, omega_q, dt_s)

        # N1 ≈ Q · A / (C_n R_n): force → velocity, rolled off by Q.
        cn = max(float(cfg.cn_kp), 1e-3)
        rn = max(float(cfg.rn_b), 1e-3)
        n1_gain = float(cfg.a_gain) / (cn * rn)
        n1_raw = n1_gain * float(force_n)
        self.n1_state = self._lpf(self.n1_state, n1_raw, omega_q, dt_s)

        n2 = 0.0
        if cfg.pn_m > 1e-9 and in_contact:
            n2_gain = float(cfg.a_gain) / max(float(cfg.pn_m), 1e-6)
            n2_raw = n2_gain * v_meas
            self.n2_state = self._lpf(self.n2_state, n2_raw, omega_q, dt_s)
            n2 = self.n2_state
        else:
            self.n2_state = self._lpf(self.n2_state, 0.0, omega_q, dt_s)

        pert = self.q_state + self.n1_state - n2
        if cfg.v_corr_max_m_s > 0.0:
            pert = float(np.clip(pert, -cfg.v_corr_max_m_s, cfg.v_corr_max_m_s))
        self.last_corr_m_s = float(pert)
        return float(v_nom_m_s - pert)
