"""Force-space velocity damper for tool-Z press and retract motion.

The damper predicts near-future force from a filtered force derivative and
limits normal velocity before the delayed admittance loop can build a large
over-force transient.  It deliberately does not depend on the environment
stiffness estimate, which is least reliable at first impact.
"""

from __future__ import annotations

from dataclasses import dataclass

import math


@dataclass
class ForceBarrierConfig:
    enabled: bool = True
    t_react_s: float = 0.030
    budget_min_n: float = 1.0
    budget_frac: float = 0.20
    f_keep_n: float = 0.5
    v_ref_m_s: float = 0.05
    v_min_retract_m_s: float = 0.002
    fdot_lpf_s: float = 0.040
    # Optional impact-energy/stiffness caps.  These use only controller-side
    # virtual quantities; no unmeasured physical damping is credited.
    stiffness_cap_enabled: bool = True
    ke_floor_n_m: float = 50.0
    mass_floor_kg: float = 0.05
    # Before the debounced physical-contact latch is established, a raw force
    # spike may request a short impact guard.  Keep this append-only in the
    # dataclass so positional construction of the older public fields remains
    # compatible.  Zero is the library-safe opt-out; the RM75 YAML opts in.
    precontact_raw_trigger_n: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict) -> "ForceBarrierConfig":
        root = raw if isinstance(raw, dict) else {}
        controller = root.get(
            "hybrid_motion", root.get("controller", root)
        )
        if not isinstance(controller, dict):
            controller = root
        barrier = controller.get(
            "force_barrier", root.get("force_barrier", {})
        )
        if not isinstance(barrier, dict):
            barrier = {}
        return cls(
            enabled=bool(barrier.get("enabled", True)),
            t_react_s=float(barrier.get("t_react_s", 0.030)),
            budget_min_n=float(barrier.get("budget_min_n", 1.0)),
            budget_frac=float(barrier.get("budget_frac", 0.20)),
            f_keep_n=float(barrier.get("f_keep_n", 0.5)),
            v_ref_m_s=float(barrier.get("v_ref_m_s", 0.05)),
            v_min_retract_m_s=float(barrier.get("v_min_retract_m_s", 0.002)),
            fdot_lpf_s=float(barrier.get("fdot_lpf_s", 0.040)),
            precontact_raw_trigger_n=float(
                barrier.get("precontact_raw_trigger_n", 0.0)
            ),
            stiffness_cap_enabled=bool(
                barrier.get("stiffness_cap_enabled", True)
            ),
            ke_floor_n_m=float(barrier.get("ke_floor_n_m", 50.0)),
            mass_floor_kg=float(barrier.get("mass_floor_kg", 0.05)),
        )


class ForceSpaceVelocityDamper:
    def __init__(self, cfg: ForceBarrierConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.f_dot_z = 0.0
        self._f_prev: float | None = None
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
        contact_enter_n: float,
        v_z_cap_retract: float | None = None,
        ke_est_n_m: float | None = None,
        mass_eq_kg: float | None = None,
        energy_available_j: float | None = None,
    ) -> tuple[float, float]:
        cfg = self.cfg
        v_hi = max(float(v_z_cap), 0.0)
        v_hi_retract = max(
            float(v_z_cap_retract) if v_z_cap_retract is not None else v_hi,
            0.0,
        )
        if not cfg.enabled:
            self.cap_press_z = v_hi
            self.cap_retract_z = v_hi_retract
            self.f_pred_z = float(f_z)
            return self.cap_press_z, self.cap_retract_z

        if not in_contact:
            seek = max(float(seek_vz_m_s), 0.0)
            if v_hi > 0.0:
                seek = min(seek, v_hi) if seek > 0.0 else v_hi
            del contact_enter_n
            self.cap_press_z = seek if seek > 0.0 else v_hi
            self.cap_retract_z = v_hi_retract
            self.f_pred_z = float(f_z)
            return self.cap_press_z, self.cap_retract_z

        if abs(float(f_des_z)) < 1e-6:
            self.cap_press_z = v_hi
            self.cap_retract_z = v_hi_retract
            self.f_pred_z = float(f_z)
            return self.cap_press_z, self.cap_retract_z

        budget = max(
            float(cfg.budget_min_n),
            float(cfg.budget_frac) * abs(float(f_des_z)),
            1e-6,
        )
        f_pred = float(f_z) + self.f_dot_z * max(float(cfg.t_react_s), 0.0)
        self.f_pred_z = f_pred
        v_ref = max(float(cfg.v_ref_m_s), 0.0)

        cap_press = max(
            0.0,
            ((float(f_des_z) + budget) - f_pred) / budget * v_ref,
        )
        # A hard surface converts a small delayed penetration into a large
        # force rise.  Bound the approach kinetic energy by the remaining
        # force headroom and, when supplied, the verified tank balance:
        #
        #   v_force = DeltaF / sqrt(M_eq K_e)
        #   v_energy = sqrt(2 E_available / M_eq)
        #
        # Both are continuous in the positive headroom.  Missing estimates
        # leave the historical force-prediction cap unchanged.
        if cfg.stiffness_cap_enabled and ke_est_n_m is not None:
            ke = max(float(ke_est_n_m), float(cfg.ke_floor_n_m), 1e-9)
            mass = max(
                float(mass_eq_kg) if mass_eq_kg is not None else 1.0,
                float(cfg.mass_floor_kg),
                1e-9,
            )
            headroom = max((float(f_des_z) + budget) - f_pred, 0.0)
            cap_press = min(cap_press, headroom / math.sqrt(mass * ke))
            if energy_available_j is not None:
                energy = max(float(energy_available_j), 0.0)
                cap_press = min(cap_press, math.sqrt(2.0 * energy / mass))
        if v_hi > 0.0:
            cap_press = min(cap_press, v_hi)

        cap_retract = max(
            float(cfg.v_min_retract_m_s),
            (f_pred - float(cfg.f_keep_n)) / budget * v_ref,
        )
        if v_hi_retract > 0.0:
            cap_retract = min(cap_retract, v_hi_retract)

        self.cap_press_z = float(cap_press)
        self.cap_retract_z = float(cap_retract)
        return self.cap_press_z, self.cap_retract_z

    def clamp_eff(self, eff: float, damping: float) -> float:
        damping = max(float(damping), 1e-6)
        return float(
            min(
                max(float(eff), -damping * self.cap_retract_z),
                damping * self.cap_press_z,
            )
        )

    def clamp_velocity(self, velocity: float) -> float:
        if velocity >= 0.0:
            return float(min(velocity, self.cap_press_z))
        return float(max(velocity, -self.cap_retract_z))
