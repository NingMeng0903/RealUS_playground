"""Force-space Faverjon velocity damper (press and retract caps).

Maps the classic distance-space velocity damper

    v <= xi * (d - d_s) / (d_i - d_s)

into force space: predicted force one reaction time ahead must stay inside
``[f_keep, f_des + budget]``.  Caps do **not** depend on ``K̂e`` — at steady
contact ``|Δx|`` is often below the estimator gate, so a stiffness-based
speed limit would use a stale value.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ForceBarrierConfig:
    enabled: bool = True
    t_react_s: float = 0.030
    budget_min_n: float = 0.5
    budget_frac: float = 0.4
    f_keep_n: float = 0.5
    v_ref_m_s: float = 0.05
    v_min_retract_m_s: float = 0.002
    fdot_lpf_s: float = 0.040

    @classmethod
    def from_dict(cls, raw: dict) -> ForceBarrierConfig:
        b = raw.get("force_barrier", raw)
        if not isinstance(b, dict):
            b = {}
        return cls(
            enabled=bool(b.get("enabled", True)),
            t_react_s=float(b.get("t_react_s", 0.030)),
            budget_min_n=float(b.get("budget_min_n", 0.5)),
            budget_frac=float(b.get("budget_frac", 0.4)),
            f_keep_n=float(b.get("f_keep_n", 0.5)),
            v_ref_m_s=float(b.get("v_ref_m_s", 0.05)),
            v_min_retract_m_s=float(b.get("v_min_retract_m_s", 0.002)),
            fdot_lpf_s=float(b.get("fdot_lpf_s", 0.040)),
        )


class ForceSpaceVelocityDamper:
    """Online press/retract velocity caps from filtered force rate."""

    def __init__(self, cfg: ForceBarrierConfig) -> None:
        self.cfg = cfg
        self.f_dot_z = 0.0
        self._f_prev: float | None = None
        self.cap_press_z = 0.0
        self.cap_retract_z = 0.0
        self.f_pred_z = 0.0

    def reset(self) -> None:
        self.f_dot_z = 0.0
        self._f_prev = None
        self.cap_press_z = 0.0
        self.cap_retract_z = 0.0
        self.f_pred_z = 0.0

    def update_fdot(self, f_z: float, dt_eff: float) -> float:
        if dt_eff <= 0.0:
            return self.f_dot_z
        if self._f_prev is None:
            self._f_prev = float(f_z)
            self.f_dot_z = 0.0
            return self.f_dot_z
        raw = (float(f_z) - self._f_prev) / dt_eff
        self._f_prev = float(f_z)
        tau = max(float(self.cfg.fdot_lpf_s), 1e-6)
        alpha = min(1.0, dt_eff / tau)
        self.f_dot_z += alpha * (raw - self.f_dot_z)
        return self.f_dot_z

    def caps(
        self,
        *,
        f_z: float,
        f_des_z: float,
        in_contact: bool,
        v_z_cap: float,
        seek_vz_m_s: float,
    ) -> tuple[float, float]:
        """Return ``(cap_press, cap_retract)`` as non-negative speeds."""
        cfg = self.cfg
        v_hi = max(float(v_z_cap), 0.0)
        if not cfg.enabled:
            self.cap_press_z = v_hi
            self.cap_retract_z = v_hi
            self.f_pred_z = float(f_z)
            return self.cap_press_z, self.cap_retract_z

        if not in_contact:
            seek = max(float(seek_vz_m_s), 0.0)
            if v_hi > 0.0:
                seek = min(seek, v_hi) if seek > 0.0 else v_hi
            self.cap_press_z = seek if seek > 0.0 else v_hi
            self.cap_retract_z = v_hi
            self.f_pred_z = float(f_z)
            return self.cap_press_z, self.cap_retract_z

        # Hand-guidance / zero setpoint: do not clamp retract to keep contact.
        # The damper's job is force-tracking safety, not fighting the operator.
        if abs(float(f_des_z)) < 1e-6:
            self.cap_press_z = v_hi
            self.cap_retract_z = v_hi
            self.f_pred_z = float(f_z)
            return self.cap_press_z, self.cap_retract_z

        budget = max(
            float(cfg.budget_min_n),
            float(cfg.budget_frac) * abs(float(f_des_z)),
            1e-6,
        )
        t_react = max(float(cfg.t_react_s), 0.0)
        f_pred = float(f_z) + self.f_dot_z * t_react
        self.f_pred_z = f_pred
        v_ref = max(float(cfg.v_ref_m_s), 0.0)

        head_press = (float(f_des_z) + budget) - f_pred
        cap_press = max(0.0, (head_press / budget) * v_ref)
        if v_hi > 0.0:
            cap_press = min(cap_press, v_hi)

        head_retract = f_pred - float(cfg.f_keep_n)
        cap_retract = max(float(cfg.v_min_retract_m_s), (head_retract / budget) * v_ref)
        if v_hi > 0.0:
            cap_retract = min(cap_retract, v_hi)

        self.cap_press_z = float(cap_press)
        self.cap_retract_z = float(cap_retract)
        return self.cap_press_z, self.cap_retract_z

    def clamp_eff(self, eff: float, bd: float) -> float:
        """Clip force drive so steady-state demand cannot exceed caps."""
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
