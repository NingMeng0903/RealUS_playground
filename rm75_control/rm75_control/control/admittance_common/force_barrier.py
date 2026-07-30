"""Force-space Faverjon velocity damper (press / retract caps).

Maps Faverjon & Tournassoud 1987 eq. (6) into force space:

    ḋ ≥ −ξ (d − d_s) / (d_i − d_s)    for d ≤ d_i

with ``d = F_max − F_z``. Caps use only measured force (no ḟ prediction).

Free-space press ceiling is ``seek_vz_m_s`` (bias-immune approach).
In contact the ceiling rises to ``v_z_cap`` so under-force chase is not
seek-starved (same split as the pre-alignment controller, without the
free-space |fz| brake or ḟ term).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ForceBarrierConfig:
    enabled: bool = True
    budget_min_n: float = 0.5
    budget_frac: float = 0.4
    f_keep_n: float = 0.5
    v_ref_m_s: float = 0.05
    v_min_retract_m_s: float = 0.002

    @classmethod
    def from_dict(cls, raw: dict) -> ForceBarrierConfig:
        b = raw.get("force_barrier", raw)
        if not isinstance(b, dict):
            b = {}
        return cls(
            enabled=bool(b.get("enabled", True)),
            budget_min_n=float(b.get("budget_min_n", 0.5)),
            budget_frac=float(b.get("budget_frac", 0.4)),
            f_keep_n=float(b.get("f_keep_n", 0.5)),
            v_ref_m_s=float(b.get("v_ref_m_s", 0.05)),
            v_min_retract_m_s=float(b.get("v_min_retract_m_s", 0.002)),
        )


class ForceSpaceVelocityDamper:
    def __init__(self, cfg: ForceBarrierConfig) -> None:
        self.cfg = cfg
        self.cap_press_z = 0.0
        self.cap_retract_z = 0.0

    def reset(self) -> None:
        self.cap_press_z = 0.0
        self.cap_retract_z = 0.0

    def caps(
        self,
        *,
        f_z: float,
        f_des_z: float,
        v_z_cap: float,
        seek_vz_m_s: float,
        in_contact: bool = False,
    ) -> tuple[float, float]:
        cfg = self.cfg
        v_hi = max(float(v_z_cap), 0.0)
        seek = max(float(seek_vz_m_s), 0.0)
        if v_hi > 0.0 and seek > 0.0:
            seek = min(seek, v_hi)
        # Task-layer ceiling only: seek in free space, full vz cap in contact.
        press_ceiling = v_hi if in_contact or seek <= 0.0 else seek
        if press_ceiling <= 0.0:
            press_ceiling = v_hi

        if not cfg.enabled:
            self.cap_press_z = press_ceiling if press_ceiling > 0.0 else v_hi
            self.cap_retract_z = v_hi
            return self.cap_press_z, self.cap_retract_z

        # Hand guidance: no force setpoint → no force-space constraint.
        if abs(float(f_des_z)) < 1e-6:
            self.cap_press_z = v_hi
            self.cap_retract_z = v_hi
            return self.cap_press_z, self.cap_retract_z

        budget = max(
            float(cfg.budget_min_n),
            float(cfg.budget_frac) * abs(float(f_des_z)),
            1e-6,
        )
        v_ref = max(float(cfg.v_ref_m_s), 0.0)
        f_max = float(f_des_z) + budget

        cap_press = max(0.0, (f_max - float(f_z)) / budget * v_ref)
        if press_ceiling > 0.0:
            cap_press = min(cap_press, press_ceiling)

        cap_retract = max(
            float(cfg.v_min_retract_m_s),
            (float(f_z) - float(cfg.f_keep_n)) / budget * v_ref,
        )
        if v_hi > 0.0:
            cap_retract = min(cap_retract, v_hi)

        self.cap_press_z = float(cap_press)
        self.cap_retract_z = float(cap_retract)
        return self.cap_press_z, self.cap_retract_z

    def clamp_eff(self, eff: float, bd: float) -> float:
        bd_eff = max(float(bd), 1e-6)
        lo = -bd_eff * self.cap_retract_z
        hi = bd_eff * self.cap_press_z
        if eff > hi:
            return hi
        if eff < lo:
            return lo
        return float(eff)

    def clamp_velocity(self, velocity: float) -> float:
        if velocity >= 0.0:
            return float(min(velocity, self.cap_press_z))
        return float(max(velocity, -self.cap_retract_z))
