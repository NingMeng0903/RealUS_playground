"""Force-space Faverjon velocity damper (press and retract caps).

Maps the classic distance-space velocity damper

    v <= xi * (d - d_s) / (d_i - d_s)

into force space: predicted force one reaction time ahead must stay inside
``[f_keep, f_des + budget]``.  Caps do **not** depend on ``K̂e`` — at steady
contact ``|Δx|`` is often below the estimator gate, so a stiffness-based
speed limit would use a stale value.

``cap_slew_m_s2`` rate-limits how fast the press/retract caps may change so a
single over-predicted tick cannot hard-stop the TCP (the contact "小振"
seen when seek velocity meets a stiff surface).
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
    # Max rate of change of either velocity cap [m/s²].  0 disables.
    # Closing press may use cap_close_slew_m_s2 (faster brake); opening
    # retract uses the base slew so escape does not slam off the surface.
    cap_slew_m_s2: float = 0.40
    cap_close_slew_m_s2: float = 1.00

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
            cap_slew_m_s2=float(b.get("cap_slew_m_s2", 0.40)),
            cap_close_slew_m_s2=float(b.get("cap_close_slew_m_s2", 1.00)),
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
        self._have_caps = False
        self._slew_clock_s = 0.0
        self._slew_window_s = 0.0

    def reset(self) -> None:
        self.f_dot_z = 0.0
        self._f_prev = None
        self.cap_press_z = 0.0
        self.cap_retract_z = 0.0
        self.f_pred_z = 0.0
        self._have_caps = False
        self._slew_clock_s = 0.0
        self._slew_window_s = 0.0

    def arm_impact_slew(self, duration_s: float) -> None:
        """Enable cap rate-limiting for a short post-contact window only."""
        self._slew_window_s = max(float(duration_s), 0.0)
        self._slew_clock_s = 0.0

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

    def _slew_enabled(self) -> bool:
        return self._slew_window_s > 1e-9 and (
            self._slew_clock_s < self._slew_window_s
        )

    def _slew_cap(
        self,
        current: float,
        target: float,
        dt_eff: float,
        *,
        closing: bool = False,
    ) -> float:
        if not self._slew_enabled() or not self._have_caps or dt_eff <= 0.0:
            return float(target)
        base = float(self.cfg.cap_slew_m_s2)
        if closing:
            slew = max(base, float(self.cfg.cap_close_slew_m_s2))
        else:
            slew = base
        if slew <= 0.0:
            return float(target)
        max_step = slew * float(dt_eff)
        delta = float(target) - float(current)
        if abs(delta) <= max_step:
            return float(target)
        return float(current) + (max_step if delta > 0.0 else -max_step)

    def caps(
        self,
        *,
        f_z: float,
        f_des_z: float,
        in_contact: bool,
        v_z_cap: float,
        seek_vz_m_s: float,
        dt_eff: float = 0.0,
        contact_enter_n: float = 0.8,
    ) -> tuple[float, float]:
        """Return ``(cap_press, cap_retract)`` as non-negative speeds."""
        cfg = self.cfg
        if dt_eff > 0.0 and self._slew_window_s > 0.0:
            self._slew_clock_s = min(
                self._slew_clock_s + float(dt_eff),
                self._slew_window_s + 1.0,
            )
        v_hi = max(float(v_z_cap), 0.0)
        if not cfg.enabled:
            self.cap_press_z = v_hi
            self.cap_retract_z = v_hi
            self.f_pred_z = float(f_z)
            self._have_caps = True
            return self.cap_press_z, self.cap_retract_z

        if not in_contact:
            seek = max(float(seek_vz_m_s), 0.0)
            if v_hi > 0.0:
                seek = min(seek, v_hi) if seek > 0.0 else v_hi
            # Continuous approach brake: as measured |fz| rises toward the
            # contact latch, close the press cap smoothly.  At the latch
            # threshold the remaining speed is a fraction of seek — no
            # rising-edge hard stop, no separate impact_vz knob.
            enter = max(float(contact_enter_n), 1e-6)
            if seek > 0.0 and enter > 1e-6:
                frac = max(0.0, 1.0 - abs(float(f_z)) / enter)
                # Keep ≥25% seek at the latch so contact still acquires;
                # that is ~3 mm/s for a 12 mm/s seek.
                frac = 0.25 + 0.75 * frac
                seek = seek * frac
            self.cap_press_z = seek if seek > 0.0 else v_hi
            self.cap_retract_z = v_hi
            self.f_pred_z = float(f_z)
            self._have_caps = True
            return self.cap_press_z, self.cap_retract_z

        # Hand-guidance / zero setpoint: do not clamp retract to keep contact.
        if abs(float(f_des_z)) < 1e-6:
            self.cap_press_z = v_hi
            self.cap_retract_z = v_hi
            self.f_pred_z = float(f_z)
            self._have_caps = True
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
        cap_press_target = max(0.0, (head_press / budget) * v_ref)
        if v_hi > 0.0:
            cap_press_target = min(cap_press_target, v_hi)

        head_retract = f_pred - float(cfg.f_keep_n)
        cap_retract_target = max(
            float(cfg.v_min_retract_m_s),
            (head_retract / budget) * v_ref,
        )
        if v_hi > 0.0:
            cap_retract_target = min(cap_retract_target, v_hi)

        self.cap_press_z = self._slew_cap(
            self.cap_press_z,
            cap_press_target,
            dt_eff,
            closing=cap_press_target < self.cap_press_z,
        )
        self.cap_retract_z = self._slew_cap(
            self.cap_retract_z,
            cap_retract_target,
            dt_eff,
            closing=cap_retract_target < self.cap_retract_z,
        )
        self._have_caps = True
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
