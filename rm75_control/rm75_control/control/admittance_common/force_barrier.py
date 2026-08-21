"""One-sided empirical press cap with pipeline indentation debt.

Press room uses adaptive ``K̂e`` (feel layer), not the shield ``K_ub``.
Committed motion is subtracted from the force margin:

    Δx_pipe = τ_stop [v_act]_+ + ½ τ_stop² [a_act]_+ + ē_x
            (or the shield backup rollout when one is supplied)

    F_pipe  = F + F̄ + K̂e Δx_pipe + ē_F
    v_max,p = [(F_max − max(F, F_pred, F_pipe)) / (K̂e T_h)]_+

``v_act`` is never added to the press speed.  Falling Ḟ does not reopen
press.  Negative velocity does not reduce ``F_pipe``.  This layer is
empirical and is not a passivity certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import math


@dataclass
class ForceBarrierConfig:
    enabled: bool = True
    t_react_s: float = 0.055
    budget_min_n: float = 1.0
    budget_frac: float = 0.20
    f_keep_n: float = 0.5
    # Over-force band that fully opens retract (escape), independent of Ke.
    f_escape_n: float = 0.5
    v_ref_m_s: float = 0.05
    v_min_retract_m_s: float = 0.0
    v_min_press_m_s: float = 0.0
    v_seek_free_m_s: float = 0.030
    fdot_lpf_s: float = 0.040
    stiffness_cap_enabled: bool = True
    ke_floor_n_m: float = 50.0
    mass_floor_kg: float = 0.05
    precontact_raw_trigger_n: float = 0.0
    # Extra Ke used for free-space / unconfident approach scheduling.
    ke_schedule_eps_n_m: float = 1.0
    # Closed-form stop time when the shield rollout is unavailable.
    tau_stop_s: float = 0.080
    e_x_m: float = 0.0004
    e_f_n: float = 0.20
    bar_f_n: float = 0.15

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
            t_react_s=float(barrier.get("t_react_s", 0.055)),
            budget_min_n=float(barrier.get("budget_min_n", 1.0)),
            budget_frac=float(barrier.get("budget_frac", 0.20)),
            f_keep_n=float(barrier.get("f_keep_n", 0.5)),
            f_escape_n=float(barrier.get("f_escape_n", 0.5)),
            v_ref_m_s=float(barrier.get("v_ref_m_s", 0.05)),
            v_min_retract_m_s=float(barrier.get("v_min_retract_m_s", 0.0)),
            v_min_press_m_s=float(barrier.get("v_min_press_m_s", 0.0)),
            v_seek_free_m_s=float(barrier.get("v_seek_free_m_s", 0.030)),
            fdot_lpf_s=float(barrier.get("fdot_lpf_s", 0.040)),
            precontact_raw_trigger_n=float(
                barrier.get("precontact_raw_trigger_n", 0.0)
            ),
            stiffness_cap_enabled=bool(
                barrier.get("stiffness_cap_enabled", True)
            ),
            ke_floor_n_m=float(barrier.get("ke_floor_n_m", 50.0)),
            mass_floor_kg=float(barrier.get("mass_floor_kg", 0.05)),
            ke_schedule_eps_n_m=float(
                barrier.get("ke_schedule_eps_n_m", 1.0)
            ),
            tau_stop_s=float(barrier.get("tau_stop_s", 0.080)),
            e_x_m=float(barrier.get("e_x_m", 0.0004)),
            e_f_n=float(barrier.get("e_f_n", 0.20)),
            bar_f_n=float(barrier.get("bar_f_n", 0.15)),
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
        self.dx_pipe_ub_m = 0.0
        self.f_pipe_ub_n = 0.0

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

    def _tau_s(self, tau_s: float | None) -> float:
        if tau_s is not None and math.isfinite(float(tau_s)) and float(tau_s) > 0.0:
            return float(tau_s)
        return max(float(self.cfg.t_react_s), 0.0)

    def _budget(self, f_des_z: float) -> float:
        return max(
            float(self.cfg.budget_min_n),
            float(self.cfg.budget_frac) * abs(float(f_des_z)),
            1e-6,
        )

    def _ke_tau(
        self,
        *,
        ke_est_n_m: float | None,
        tau_s: float,
    ) -> float:
        ke = float(self.cfg.ke_floor_n_m)
        if ke_est_n_m is not None and math.isfinite(float(ke_est_n_m)):
            ke = max(float(ke_est_n_m), ke, float(self.cfg.ke_schedule_eps_n_m))
        return max(ke * max(tau_s, 0.0), 1e-6)

    def _ke_n_m(self, ke_est_n_m: float | None) -> float:
        ke = float(self.cfg.ke_floor_n_m)
        if ke_est_n_m is not None and math.isfinite(float(ke_est_n_m)):
            ke = max(float(ke_est_n_m), ke, float(self.cfg.ke_schedule_eps_n_m))
        return max(ke, 1e-6)

    def pipeline_dx_ub(
        self,
        *,
        v_tcp_z_actual: float | None,
        a_tcp_z_actual: float | None,
        shield_dx_m: float | None = None,
        tau_stop_s: float | None = None,
    ) -> float:
        tau = (
            float(tau_stop_s)
            if tau_stop_s is not None and math.isfinite(float(tau_stop_s))
            else float(self.cfg.tau_stop_s)
        )
        tau = max(tau, 0.0)
        v_plus = max(float(v_tcp_z_actual or 0.0), 0.0)
        a_plus = max(float(a_tcp_z_actual or 0.0), 0.0)
        if shield_dx_m is not None and math.isfinite(float(shield_dx_m)):
            dx = max(float(shield_dx_m), 0.0)
            # The shield plant is first-order and has no a_act state.
            if v_plus <= 1e-9 and a_plus > 0.0:
                residual = 0.5 * tau * tau * a_plus
                if residual > 1e-12:
                    dx = max(dx, residual + max(float(self.cfg.e_x_m), 0.0))
            return dx
        motion = tau * v_plus + 0.5 * tau * tau * a_plus
        if motion <= 1e-12:
            return 0.0
        return motion + max(float(self.cfg.e_x_m), 0.0)

    def scheduled_approach_m_s(
        self,
        *,
        f_des_z: float,
        ke_est_n_m: float | None,
        tau_s: float | None = None,
        v_hi: float = 0.0,
    ) -> float:
        """ΔF_allow / (K̂e · τ) approach speed, clipped by v_hi / v_seek_free."""
        tau = self._tau_s(tau_s)
        budget = self._budget(f_des_z)
        denom = self._ke_tau(ke_est_n_m=ke_est_n_m, tau_s=tau)
        allow = budget / denom
        seek = max(float(self.cfg.v_seek_free_m_s), 0.0)
        if seek > 0.0:
            allow = min(allow, seek) if allow > 0.0 else seek
        if v_hi > 0.0:
            allow = min(allow, v_hi) if allow > 0.0 else v_hi
        return max(allow, 0.0)

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
        tau_s: float | None = None,
        v_tcp_z_actual: float | None = None,
        a_tcp_z_actual: float | None = None,
        shield_dx_m: float | None = None,
    ) -> tuple[float, float]:
        cfg = self.cfg
        v_hi = max(float(v_z_cap), 0.0)
        v_hi_retract = max(
            float(v_z_cap_retract) if v_z_cap_retract is not None else v_hi,
            0.0,
        )
        tau = self._tau_s(tau_s)
        dx_pipe = self.pipeline_dx_ub(
            v_tcp_z_actual=v_tcp_z_actual,
            a_tcp_z_actual=a_tcp_z_actual,
            shield_dx_m=shield_dx_m,
            tau_stop_s=max(tau, float(cfg.tau_stop_s)),
        )
        self.dx_pipe_ub_m = float(dx_pipe)
        if not cfg.enabled:
            self.cap_press_z = v_hi
            self.cap_retract_z = v_hi_retract
            self.f_pred_z = float(f_z)
            return self.cap_press_z, self.cap_retract_z

        if not in_contact:
            seek = max(float(seek_vz_m_s), 0.0)
            if v_hi > 0.0:
                seek = min(seek, v_hi) if seek > 0.0 else v_hi
            free = max(float(cfg.v_seek_free_m_s), 0.0)
            if free > 0.0:
                seek = min(seek, free) if seek > 0.0 else free
            if cfg.stiffness_cap_enabled and ke_est_n_m is not None:
                scheduled = self.scheduled_approach_m_s(
                    f_des_z=f_des_z,
                    ke_est_n_m=ke_est_n_m,
                    tau_s=tau,
                    v_hi=v_hi,
                )
                if scheduled > 0.0:
                    seek = min(seek, scheduled) if seek > 0.0 else scheduled
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

        budget = self._budget(f_des_z)
        f_pred = float(f_z) + self.f_dot_z * tau
        self.f_pred_z = f_pred
        f_max = abs(float(f_des_z)) + budget
        f_min = max(float(cfg.f_keep_n), 0.0)
        ke = self._ke_n_m(ke_est_n_m)
        denom = self._ke_tau(ke_est_n_m=ke_est_n_m, tau_s=tau)
        if dx_pipe > 1e-12:
            f_pipe = (
                float(f_z)
                + max(float(cfg.bar_f_n), 0.0)
                + ke * dx_pipe
                + max(float(cfg.e_f_n), 0.0)
            )
        else:
            f_pipe = float(f_z)
        self.f_pipe_ub_n = float(f_pipe)

        # Falling Ḟ never reopens press.  Pipeline debt is taken from the
        # remaining force margin; v_act is not added as extra press speed.
        f_pred_press = max(f_pred, float(f_z), f_pipe)
        v_press_max = (f_max - f_pred_press) / denom
        cap_press = max(0.0, v_press_max)
        if cfg.stiffness_cap_enabled and energy_available_j is not None:
            mass = max(
                float(mass_eq_kg) if mass_eq_kg is not None else 1.0,
                float(cfg.mass_floor_kg),
                1e-9,
            )
            energy = max(float(energy_available_j), 0.0)
            cap_press = min(cap_press, math.sqrt(2.0 * energy / mass))
        if v_hi > 0.0:
            cap_press = min(cap_press, v_hi)
        # Seek floor only while force margin remains.  A leftover
        # v_min_press must not reopen press after F_pipe >= F_max.
        v_min_press = max(float(cfg.v_min_press_m_s), 0.0)
        if v_hi > 0.0:
            v_min_press = min(v_min_press, v_hi)
        if v_press_max > 1e-12 and v_min_press > 0.0:
            cap_press = max(cap_press, min(v_min_press, v_press_max))

        escape = max(float(cfg.f_escape_n), 0.0)
        overforce = f_pred >= abs(float(f_des_z)) + escape
        if overforce:
            cap_retract = v_hi_retract
        else:
            v_lower = (f_min - f_pred) / denom
            cap_retract = max(0.0, -v_lower)
            if v_hi_retract > 0.0:
                cap_retract = min(cap_retract, v_hi_retract)
        cap_retract = max(cap_retract, max(float(cfg.v_min_retract_m_s), 0.0))
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
