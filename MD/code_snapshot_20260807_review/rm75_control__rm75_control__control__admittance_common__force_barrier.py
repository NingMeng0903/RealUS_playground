"""Press-side predictive velocity barrier for tool-Z.

Limits *press* speed from a predicted near-future force so a delayed
admittance loop cannot slam into a stiff surface.  Retract is left fully
open by default — a two-sided press/retract barrier with 40 ms delay forms
a bang-bang hunting oscillator (see run_20260804_161942).

    g = 1 / (K̂b · T_dead)
    f_pred = f + T_pred · ḟ
    cap_press = LPF((F_des + budget − f_pred) · g)
    cap_retract = v_z_cap   (press_only=True)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class ForceBarrierConfig:
    enabled: bool = False
    # Retract barrier OFF by default — two-sided caps hunt with delay.
    press_only: bool = True
    t_dead_s: float = 0.040
    t_pred_s: float = 0.030
    budget_min_n: float = 1.5
    budget_frac: float = 0.75
    f_keep_n: float = 0.3
    v_floor_press_m_s: float = 0.015
    v_floor_retract_m_s: float = 0.0
    f_panic_n: float = 20.0
    yield_overforce_n: float = 1.5
    yield_fdot_max_n_s: float = 60.0
    ke_seek_default: float = 300.0
    ke_min: float = 200.0
    ke_max: float = 4000.0
    ke_attack_s: float = 0.20
    ke_release_s: float = 0.8
    ke_free_hold_s: float = 0.5
    ke_v_press_min_m_s: float = 0.012
    ke_f_min_n: float = 0.5
    ke_f_err_gate_n: float = 1.5
    ke_slew_up_max: float = 5000.0
    # 0 = no stiff-first seed (seed caused permanent tight caps / hunting).
    ke_impact_seed: float = 0.0
    # LPF on press cap — raw caps flip every tick → hunting.
    cap_lpf_tau_s: float = 0.08
    # Do not limit free-space seek from elevated K̂b (also caused re-impact hunt).
    limit_free_seek: bool = False
    fdot_taps: int = 3
    t_react_s: float = 0.030
    v_ref_m_s: float = 0.05
    v_min_retract_m_s: float = 0.0
    fdot_lpf_s: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict) -> "ForceBarrierConfig":
        barrier = raw.get("force_barrier", raw)
        if not isinstance(barrier, dict):
            barrier = {}
        t_pred = float(
            barrier.get("t_pred_s", barrier.get("t_react_s", 0.030))
        )
        return cls(
            enabled=bool(barrier.get("enabled", False)),
            press_only=bool(barrier.get("press_only", True)),
            t_dead_s=float(barrier.get("t_dead_s", 0.040)),
            t_pred_s=t_pred,
            budget_min_n=float(barrier.get("budget_min_n", 1.5)),
            budget_frac=float(barrier.get("budget_frac", 0.75)),
            f_keep_n=float(barrier.get("f_keep_n", 0.3)),
            v_floor_press_m_s=float(
                barrier.get("v_floor_press_m_s", 0.015)
            ),
            v_floor_retract_m_s=float(
                barrier.get(
                    "v_floor_retract_m_s",
                    barrier.get("v_min_retract_m_s", 0.0),
                )
            ),
            f_panic_n=float(barrier.get("f_panic_n", 20.0)),
            yield_overforce_n=float(
                barrier.get("yield_overforce_n", 1.5)
            ),
            yield_fdot_max_n_s=float(
                barrier.get("yield_fdot_max_n_s", 60.0)
            ),
            ke_seek_default=float(barrier.get("ke_seek_default", 300.0)),
            ke_min=float(barrier.get("ke_min", 200.0)),
            ke_max=float(barrier.get("ke_max", 4000.0)),
            ke_attack_s=float(barrier.get("ke_attack_s", 0.20)),
            ke_release_s=float(barrier.get("ke_release_s", 0.8)),
            ke_free_hold_s=float(barrier.get("ke_free_hold_s", 0.5)),
            ke_v_press_min_m_s=float(
                barrier.get("ke_v_press_min_m_s", 0.012)
            ),
            ke_f_min_n=float(barrier.get("ke_f_min_n", 0.5)),
            ke_f_err_gate_n=float(barrier.get("ke_f_err_gate_n", 1.5)),
            ke_slew_up_max=float(barrier.get("ke_slew_up_max", 5000.0)),
            ke_impact_seed=float(barrier.get("ke_impact_seed", 0.0)),
            cap_lpf_tau_s=float(barrier.get("cap_lpf_tau_s", 0.08)),
            limit_free_seek=bool(barrier.get("limit_free_seek", False)),
            fdot_taps=max(2, int(barrier.get("fdot_taps", 3))),
            t_react_s=t_pred,
            v_ref_m_s=float(barrier.get("v_ref_m_s", 0.05)),
            v_min_retract_m_s=float(
                barrier.get("v_min_retract_m_s", 0.0)
            ),
            fdot_lpf_s=float(barrier.get("fdot_lpf_s", 0.0)),
        )


class ForceSpaceVelocityDamper:
    """Press-side predictive velocity barrier (retract open by default)."""

    def __init__(self, cfg: ForceBarrierConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.f_dot_z = 0.0
        self.cap_press_z = 0.0
        self.cap_retract_z = 0.0
        self.f_pred_z = 0.0
        self.ke_barrier = float(self.cfg.ke_seek_default)
        self._f_hist: deque[tuple[float, float]] = deque(
            maxlen=max(2, int(self.cfg.fdot_taps))
        )
        self._free_timer_s = 0.0
        self._contact_conf = 0.0
        self._was_contact = False
        self._cap_press_filt: float | None = None

    def note_contact_edge(self, in_contact: bool) -> None:
        if in_contact and not self._was_contact:
            seed = float(self.cfg.ke_impact_seed)
            if seed > self.ke_barrier:
                self.ke_barrier = min(seed, float(self.cfg.ke_max))
        self._was_contact = bool(in_contact)

    def update_fdot(self, f_z: float, dt_eff: float) -> float:
        if dt_eff <= 0.0:
            return self.f_dot_z
        self._f_hist.append((float(f_z), float(dt_eff)))
        if len(self._f_hist) < 2:
            self.f_dot_z = 0.0
            return self.f_dot_z
        f0, _ = self._f_hist[0]
        f1, _ = self._f_hist[-1]
        dt_span = sum(d for _, d in list(self._f_hist)[1:])
        if dt_span <= 1e-9:
            return self.f_dot_z
        self.f_dot_z = (f1 - f0) / dt_span
        return self.f_dot_z

    def update_ke(
        self,
        *,
        f_z: float,
        v_tcp_z: float,
        in_contact: bool,
        dt_eff: float,
        f_des_z: float = 0.0,
    ) -> float:
        cfg = self.cfg
        if dt_eff <= 0.0:
            return self.ke_barrier
        if in_contact:
            self._free_timer_s = 0.0
            self._contact_conf = min(1.0, self._contact_conf + dt_eff / 0.08)
            v_press = max(float(v_tcp_z), 0.0)
            over_err = float(f_z) - float(f_des_z)
            hand_push = over_err > float(cfg.ke_f_err_gate_n)
            learn = (
                (not hand_push)
                and v_press >= float(cfg.ke_v_press_min_m_s)
                and float(f_z) >= float(cfg.ke_f_min_n)
                and self.f_dot_z > 0.0
            )
            if learn:
                k_inst = self.f_dot_z / max(v_press, 1e-6)
                k_inst = float(min(max(k_inst, cfg.ke_min), cfg.ke_max))
                tau = (
                    float(cfg.ke_attack_s)
                    if k_inst > self.ke_barrier
                    else float(cfg.ke_release_s)
                )
                blend = min(1.0, dt_eff / max(tau, 1e-4))
                step = blend * (k_inst - self.ke_barrier)
                slew = float(cfg.ke_slew_up_max)
                if slew > 0.0 and step > 0.0:
                    step = min(step, slew * dt_eff)
                self.ke_barrier += step
            elif hand_push:
                target = float(cfg.ke_seek_default)
                tau = max(float(cfg.ke_release_s), 1e-3)
                blend = min(1.0, dt_eff / tau)
                self.ke_barrier += blend * (target - self.ke_barrier)
        else:
            self._contact_conf = max(0.0, self._contact_conf - dt_eff / 0.05)
            self._free_timer_s += dt_eff
            if self._free_timer_s > float(cfg.ke_free_hold_s):
                target = float(cfg.ke_seek_default)
                tau = max(float(cfg.ke_release_s), 1e-3)
                blend = min(1.0, dt_eff / tau)
                self.ke_barrier += blend * (target - self.ke_barrier)
        self.ke_barrier = float(
            min(max(self.ke_barrier, cfg.ke_min), cfg.ke_max)
        )
        return self.ke_barrier

    @property
    def contact_conf(self) -> float:
        return float(self._contact_conf)

    def _lpf_press(self, raw: float, dt_eff: float, v_hi: float) -> float:
        tau = max(float(self.cfg.cap_lpf_tau_s), 0.0)
        if self._cap_press_filt is None or tau <= 1e-9 or dt_eff <= 0.0:
            self._cap_press_filt = float(raw)
        else:
            blend = min(1.0, dt_eff / tau)
            self._cap_press_filt += blend * (float(raw) - self._cap_press_filt)
        out = float(self._cap_press_filt)
        if v_hi > 0.0:
            out = min(out, v_hi)
        return max(out, float(self.cfg.v_floor_press_m_s))

    def caps(
        self,
        *,
        f_z: float,
        f_des_z: float,
        in_contact: bool,
        v_z_cap: float,
        seek_vz_m_s: float,
        contact_enter_n: float = 0.0,
        v_z_cap_retract: float | None = None,
        retract_fast_hold: bool = False,
        bypass_retract: bool = False,
        dt_eff: float = 0.005,
    ) -> tuple[float, float]:
        del contact_enter_n
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
            self._cap_press_filt = v_hi
            return self.cap_press_z, self.cap_retract_z

        t_pred = max(float(cfg.t_pred_s), 0.0)
        f_pred = float(f_z) + self.f_dot_z * t_pred
        self.f_pred_z = f_pred

        # Free-space: full seek (yes_adaptive).  No K̂b-limited re-approach.
        if not in_contact:
            seek = max(float(seek_vz_m_s), 0.0)
            if v_hi > 0.0:
                seek = min(seek, v_hi) if seek > 0.0 else v_hi
            if (
                cfg.limit_free_seek
                and float(self.ke_barrier)
                > float(cfg.ke_seek_default) * 1.5
                and self._free_timer_s < float(cfg.ke_free_hold_s)
            ):
                budget = max(
                    float(cfg.budget_min_n),
                    float(cfg.budget_frac) * abs(float(f_des_z)),
                    1e-6,
                )
                g = 1.0 / (
                    max(float(self.ke_barrier), float(cfg.ke_min))
                    * max(float(cfg.t_dead_s), 1e-4)
                )
                seek = min(
                    seek if seek > 0.0 else v_hi,
                    max(float(cfg.v_floor_press_m_s), budget * g),
                )
            self.cap_press_z = self._lpf_press(
                seek if seek > 0.0 else v_hi, dt_eff, v_hi
            )
            self.cap_retract_z = v_hi_retract
            return self.cap_press_z, self.cap_retract_z

        if abs(float(f_des_z)) < 1e-6:
            self.cap_press_z = self._lpf_press(v_hi, dt_eff, v_hi)
            self.cap_retract_z = v_hi_retract
            return self.cap_press_z, self.cap_retract_z

        budget = max(
            float(cfg.budget_min_n),
            float(cfg.budget_frac) * abs(float(f_des_z)),
            1e-6,
        )
        ke = max(float(self.ke_barrier), float(cfg.ke_min), 1e-6)
        t_dead = max(float(cfg.t_dead_s), 1e-4)
        g = 1.0 / (ke * t_dead)

        raw_press = max(
            float(cfg.v_floor_press_m_s),
            (float(f_des_z) + budget - f_pred) * g,
        )
        if v_hi > 0.0:
            raw_press = min(raw_press, v_hi)
        self.cap_press_z = self._lpf_press(raw_press, dt_eff, v_hi)

        # Retract: always open in press_only mode (kills bang-bang hunting).
        if (
            cfg.press_only
            or bypass_retract
            or retract_fast_hold
            or abs(float(f_z)) >= float(cfg.f_panic_n)
        ):
            self.cap_retract_z = v_hi_retract
        else:
            overforce = float(f_z) - float(f_des_z)
            hand_yield = (
                overforce >= float(cfg.yield_overforce_n)
                and abs(self.f_dot_z) <= float(cfg.yield_fdot_max_n_s)
            )
            if hand_yield:
                self.cap_retract_z = v_hi_retract
            else:
                cap_retract = max(
                    float(cfg.v_floor_retract_m_s),
                    (f_pred - float(cfg.f_keep_n)) * g,
                )
                if v_hi_retract > 0.0:
                    cap_retract = min(cap_retract, v_hi_retract)
                self.cap_retract_z = float(max(cap_retract, 0.0))

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
