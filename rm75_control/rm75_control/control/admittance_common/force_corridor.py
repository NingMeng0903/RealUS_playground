"""Two-sided force corridor on the *emitted* command.

Certificate 3 is set invariance of a force interval, not passivity.
A 3 Hz / 9 N cycle can be fully passive (Franken §V-C5) and still bounce.
This layer keeps predicted contact force in [F_lo, F_hi] using the shield's
backup-to-terminal indentation bound and the linear-region a_max (1.20 m/s²),
not the saturated 3.2 m/s² peak.

Do not enable shield force / passive / ospf from here.  The shield stays in
observe; the corridor clamps the command that is actually sent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PressEnvelopeConfig:
    """Linear-region press envelope.  Caps are runtime, not certificates."""

    # 0 = disabled so historical unit tests keep max_vz after contact.
    # Production yaml sets 0.020 (tissue chase once the latch is off).
    soft_approach_m_s: float = 0.0
    # 0 = disabled so historical unit tests keep v_seek_free.
    # Production yaml sets 0.010 (8–12 mm/s first touch).
    first_touch_m_s: float = 0.0
    # 0 = disabled (tests keep max_vz_tool).  Production yaml sets 0.025.
    max_force_axis_m_s: float = 0.0
    a_linear_m_s2: float = 1.20

    @classmethod
    def from_dict(cls, raw: dict) -> "PressEnvelopeConfig":
        root = raw if isinstance(raw, dict) else {}
        controller = root.get("hybrid_motion", root.get("controller", root))
        if not isinstance(controller, dict):
            controller = root
        block = controller.get("press_envelope", root.get("press_envelope", {}))
        if not isinstance(block, dict):
            block = {}
        return cls(
            soft_approach_m_s=float(block.get("soft_approach_m_s", 0.020)),
            first_touch_m_s=float(block.get("first_touch_m_s", 0.012)),
            max_force_axis_m_s=float(block.get("max_force_axis_m_s", 0.0)),
            a_linear_m_s2=float(block.get("a_linear_m_s2", 1.20)),
        )


@dataclass
class ForceCorridorConfig:
    enabled: bool = False
    f_keep_n: float = 0.5

    @classmethod
    def from_dict(cls, raw: dict) -> "ForceCorridorConfig":
        root = raw if isinstance(raw, dict) else {}
        controller = root.get("hybrid_motion", root.get("controller", root))
        if not isinstance(controller, dict):
            controller = root
        block = controller.get("force_corridor", root.get("force_corridor", {}))
        if not isinstance(block, dict):
            block = {}
        barrier = controller.get("force_barrier", {})
        if not isinstance(barrier, dict):
            barrier = {}
        return cls(
            enabled=bool(block.get("enabled", False)),
            f_keep_n=float(block.get("f_keep_n", barrier.get("f_keep_n", 0.5))),
        )


def _jerk_limit(
    target: float,
    prev: float,
    *,
    dt_s: float,
    a_max: float,
    j_max: float,
    a_prev: float,
) -> tuple[float, float]:
    dt = max(float(dt_s), 1e-6)
    a_lim = max(float(a_max), 0.0)
    j_lim = max(float(j_max), 0.0)
    du = float(target) - float(prev)
    if a_lim <= 0.0 and j_lim <= 0.0:
        return float(target), 0.0
    a_des = du / dt
    if a_lim > 0.0:
        a_des = max(-a_lim, min(a_lim, a_des))
    if j_lim > 0.0:
        da = max(-j_lim * dt, min(j_lim * dt, a_des - float(a_prev)))
        a_des = float(a_prev) + da
        if a_lim > 0.0:
            a_des = max(-a_lim, min(a_lim, a_des))
    return float(prev) + a_des * dt, a_des


class ForceCorridor:
    """Set-invariance clamp on press-positive emitted velocity."""

    def __init__(self, cfg: ForceCorridorConfig | None = None) -> None:
        self.cfg = cfg or ForceCorridorConfig()
        self.u_lo = 0.0
        self.u_hi = 0.0
        self.applied = False
        self.infeasible = False
        self.f_pred_n = 0.0
        self._a_prev = 0.0

    def reset(self) -> None:
        self.u_lo = 0.0
        self.u_hi = 0.0
        self.applied = False
        self.infeasible = False
        self.f_pred_n = 0.0
        self._a_prev = 0.0

    def clamp(
        self,
        u_sent: float,
        *,
        f_n: float,
        f_hi_n: float,
        f_lo_n: float | None = None,
        ke_n_m: float,
        dx_ub_m: float,
        tau_s: float,
        cap_press_m_s: float,
        cap_retract_m_s: float,
        u_prev: float,
        dt_s: float,
        a_max_m_s2: float,
        j_max_m_s3: float,
        v_retract_max_m_s: float,
        in_contact: bool,
    ) -> float:
        if not self.cfg.enabled or not in_contact:
            self.applied = False
            self.infeasible = False
            self.u_lo = -max(float(cap_retract_m_s), 0.0)
            self.u_hi = max(float(cap_press_m_s), 0.0)
            self.f_pred_n = float(f_n)
            return float(u_sent)
        ke = max(float(ke_n_m), 1.0)
        tau = max(float(tau_s), 1e-3)
        f_lo = max(float(f_lo_n if f_lo_n is not None else self.cfg.f_keep_n), 0.0)
        f_hi = max(float(f_hi_n), f_lo)
        dx = max(float(dx_ub_m), 0.0)
        f_pred = float(f_n) + ke * dx
        self.f_pred_n = f_pred
        denom = ke * tau
        u_hi = (f_hi - f_pred) / denom
        u_lo = (f_lo - f_pred) / denom
        u_hi = min(u_hi, max(float(cap_press_m_s), 0.0))
        u_lo = max(u_lo, -max(float(cap_retract_m_s), 0.0))
        self.u_hi = float(u_hi)
        self.u_lo = float(u_lo)
        if u_lo > u_hi + 1e-9:
            self.infeasible = True
            self.applied = True
            # Predicted interval empty.  If the measurement is still in
            # the set, hold — conservative Ke/dx is not a real escape
            # (Franken §V-C5 bang-bang is the bounce cycle).  Only a
            # measured force above F_hi opens jerk-limited retract.
            if float(f_n) <= f_hi + 1e-12:
                target = 0.0
            else:
                target = -max(float(v_retract_max_m_s), 0.0)
            limited, a_now = _jerk_limit(
                target,
                float(u_prev),
                dt_s=dt_s,
                a_max=a_max_m_s2,
                j_max=j_max_m_s3,
                a_prev=self._a_prev,
            )
            self._a_prev = a_now
            return float(limited)
        self.infeasible = False
        u = min(max(float(u_sent), u_lo), u_hi)
        self.applied = abs(u - float(u_sent)) > 1e-9
        if abs(dt_s) > 1e-9:
            self._a_prev = (u - float(u_prev)) / max(float(dt_s), 1e-6)
        return float(u)
