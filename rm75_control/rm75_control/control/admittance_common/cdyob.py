"""Samuel 2024 CDYOB at the tool-Z velocity interface.

Paper (RAL 2024):
    pert = Q (T_n^{-1} V_m - V_i) + N_1 F_m - N_2 V_m
    V_i  = V_r - pert
    N_1  = Q A (T_n^{-1} - 1)     # equiv. to Q C_n^{-1} A R_n^{-1}
    N_2  = Q A P_n^{-1} T_n^{-1}  # first release keeps P_n = 0

T_n is the identified stable, minimum-phase residual after the communication
delay Γ_d.  Γ_d is never inverted.  Runtime realization is causal:
``F_m[k]``, ``V_m[k]`` and the previously committed ``V_i[k-1]`` produce the
candidate for tick k; the post-constraint command is committed afterwards.

The closed ``1/(1-Q)`` form is provided only for unsaturated linear-equivalence
tests.  It is not the runtime implementation because its integrator winds up
when blend, correction limits, barrier, slew, or shield break the cancellation.

This is a performance term.  It is not a passivity certificate and does
not mutate the Lee tank or the shield energy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(lo) if value < lo else float(hi) if value > hi else float(value)


def _lpf_alpha(omega_rad: float, dt_s: float) -> float:
    if dt_s <= 0.0 or omega_rad <= 0.0:
        return 0.0
    return float(_clamp(1.0 - math.exp(-omega_rad * dt_s), 0.0, 1.0))


def _lpf_step(state: float, x: float, alpha: float) -> float:
    return float(state + alpha * (x - state))


@dataclass
class CdyobConfig:
    mode: str = "off"
    omega_q_hz: float = 0.75
    t0_s: float = 0.050
    tp_s: float = 0.020
    # Backward-compatible aliases; used only when t0_s / tp_s are absent.
    tau_s: float = 0.0
    t_n_s: float = 0.0
    pn_m: float = 0.0
    v_corr_max_m_s: float = 0.003
    blend_s: float = 0.30
    active_press_max_m_s: float = 0.010
    active_retract_max_m_s: float = 0.010
    active_q_max_hz: float = 1.0
    active_model_validated: bool = False
    active_force_ratio: float = 0.90
    active_settle_speed_m_s: float = 0.003
    active_settle_hold_s: float = 0.20
    # Unused PI / mass-damper leftovers are ignored.  N1 uses T_n and A.

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if not value:
            self.mode = "off"
        elif self.mode == "off":
            self.mode = "shadow"

    def computes(self) -> bool:
        return self.normalized_mode() in ("shadow", "active")

    def applies(self) -> bool:
        return self.normalized_mode() == "active"

    def assert_active_ready(self) -> None:
        if not self.applies():
            return
        if not self.active_model_validated:
            raise ValueError(
                "CDYOB active requires active_model_validated=true after "
                "phase validation in the intended Q band"
            )
        if self.omega_q_hz > self.active_q_max_hz + 1e-12:
            raise ValueError(
                "CDYOB omega_q_hz exceeds the validated active_q_max_hz"
            )

    def delay_s(self) -> float:
        if self.t0_s > 1e-9:
            return float(self.t0_s)
        return max(float(self.tau_s), 0.0)

    def time_constant_s(self) -> float:
        if self.tp_s > 1e-9:
            return float(self.tp_s)
        if self.t_n_s > 1e-9:
            return float(self.t_n_s)
        return 0.020

    def normalized_mode(self) -> str:
        mode = str(self.mode).strip().lower()
        return mode if mode in ("off", "shadow", "active") else "off"

    @classmethod
    def from_dict(cls, raw: dict) -> "CdyobConfig":
        c = raw.get("hybrid_motion", raw.get("controller", raw))
        if not isinstance(c, dict):
            c = raw if isinstance(raw, dict) else {}
        p = c.get("cdyob", {})
        if not isinstance(p, dict):
            p = {}
        raw_mode = str(p.get("mode", "")).strip().lower()
        if raw_mode in ("off", "shadow", "active"):
            mode = raw_mode
        else:
            # Old enabled:true was the broken residual corrector.  Stay off.
            mode = "off"
        t0 = float(p.get("t0_s", p.get("tau_s", 0.050)))
        tp = float(p.get("tp_s", p.get("t_n_s", 0.020)))
        return cls(
            mode=mode,
            omega_q_hz=float(p.get("omega_q_hz", 0.75)),
            t0_s=t0,
            tp_s=tp,
            tau_s=float(p.get("tau_s", t0)),
            t_n_s=float(p.get("t_n_s", tp)),
            pn_m=float(p.get("pn_m", 0.0)),
            v_corr_max_m_s=float(p.get("v_corr_max_m_s", 0.003)),
            blend_s=float(p.get("blend_s", 0.30)),
            active_press_max_m_s=float(
                p.get("active_press_max_m_s", 0.010)
            ),
            active_retract_max_m_s=float(
                p.get("active_retract_max_m_s", 0.010)
            ),
            active_q_max_hz=float(p.get("active_q_max_hz", 1.0)),
            active_model_validated=bool(
                p.get("active_model_validated", False)
            ),
            active_force_ratio=float(p.get("active_force_ratio", 0.90)),
            active_settle_speed_m_s=float(
                p.get("active_settle_speed_m_s", 0.003)
            ),
            active_settle_hold_s=float(
                p.get("active_settle_hold_s", 0.20)
            ),
        )


@dataclass
class CdyobTelemetry:
    corr_m_s: float = 0.0
    pert_unclipped: float = 0.0
    pert_clipped: float = 0.0
    qtinv_vm: float = 0.0
    q_vi: float = 0.0
    n1_force: float = 0.0
    n2_velocity: float = 0.0
    blend: float = 0.0
    vi_m_s: float = 0.0
    candidate_m_s: float = 0.0
    antiwindup_error_m_s: float = 0.0
    residual: float = 0.0
    saturated: bool = False
    constrained: bool = False
    linear_equivalent: bool = False
    omega_q_hz: float = 0.0
    mode: str = "off"


class CombinedDynamicsYob:
    """Discrete CDYOB correction on the tool-Z velocity command."""

    def __init__(self, cfg: CdyobConfig) -> None:
        self.cfg = cfg
        self.cfg.assert_active_ready()
        self.reset()

    def reset(self) -> None:
        self._q_vm = 0.0
        self._q_vi = 0.0
        self._q_fm = 0.0
        self._n1 = 0.0
        self._n2 = 0.0
        self._blend = 0.0
        self._last_committed = 0.0
        self._last_candidate = 0.0
        self._closed_vi = 0.0
        self._closed_u_prev = 0.0
        self.last_omega_q_hz = 0.0
        self.last_corr_m_s = 0.0
        self.telemetry = CdyobTelemetry()

    def _omega_q_hz(self) -> float:
        if self.cfg.omega_q_hz > 1e-9:
            return float(self.cfg.omega_q_hz)
        tau = max(self.cfg.delay_s(), 1e-3)
        return 1.0 / (2.0 * math.pi * tau)

    def _omega_q_rad(self) -> float:
        return 2.0 * math.pi * self._omega_q_hz()

    def _qtinv(self, x: float, q_x: float, tp_s: float, omega_q: float) -> float:
        # Q T_n^{-1} = ω Tp + (1 - Tp ω) Q, T_n = 1/(Tp s + 1).
        return omega_q * tp_s * x + (1.0 - tp_s * omega_q) * q_x

    def _step_filters(
        self,
        *,
        v_meas: float,
        force_n: float,
        vi_for_q: float,
        dt_s: float,
        mass_z: float,
        damping_z: float,
    ) -> tuple[float, float, float, float]:
        omega_q = self._omega_q_rad()
        alpha = _lpf_alpha(omega_q, dt_s)
        tp = max(self.cfg.time_constant_s(), 1e-4)
        mass = max(float(mass_z), 1e-3)
        damp = max(float(damping_z), 1e-6)

        self._q_vm = _lpf_step(self._q_vm, v_meas, alpha)
        self._q_vi = _lpf_step(self._q_vi, vi_for_q, alpha)
        self._q_fm = _lpf_step(self._q_fm, force_n, alpha)

        qtinv_vm = self._qtinv(v_meas, self._q_vm, tp, omega_q)
        q_vi = float(self._q_vi)
        # N1 = A · ω Tp (1-Q) F_m.  DC gain is 0.
        hp = omega_q * tp * (force_n - self._q_fm)
        a_disc = math.exp(-damp * dt_s / mass)
        b_disc = (1.0 - a_disc) / damp
        self._n1 = a_disc * self._n1 + b_disc * hp

        # First release: N2 = 0 (no extra payload).  pn_m stays unused.
        self._n2 = 0.0
        return qtinv_vm, q_vi, float(self._n1), 0.0

    def estimate(
        self,
        *,
        v_meas_m_s: float,
        force_n: float,
        vi_m_s: float,
        dt_s: float,
        mass_z: float,
        damping_z: float,
    ) -> float:
        """One filter step using a supplied V_i (no delay-line side effect)."""
        qtinv_vm, q_vi, n1, n2 = self._step_filters(
            v_meas=float(v_meas_m_s),
            force_n=float(force_n),
            vi_for_q=float(vi_m_s),
            dt_s=float(dt_s),
            mass_z=mass_z,
            damping_z=damping_z,
        )
        pert = qtinv_vm - q_vi + n1 - n2
        self._publish(
            pert_unclipped=pert,
            qtinv_vm=qtinv_vm,
            q_vi=q_vi,
            n1=n1,
            n2=n2,
            vi=float(vi_m_s),
            blend=0.0,
            corr=0.0,
        )
        return float(pert)

    def implicit_vi(
        self,
        v_nom_m_s: float,
        *,
        v_meas_m_s: float,
        force_n: float,
        dt_s: float,
        mass_z: float,
        damping_z: float,
    ) -> float:
        """Simulation-only same-sample solve; never used by runtime control."""
        omega_q = self._omega_q_rad()
        alpha = _lpf_alpha(omega_q, dt_s)
        tp = max(self.cfg.time_constant_s(), 1e-4)
        mass = max(float(mass_z), 1e-3)
        damp = max(float(damping_z), 1e-6)
        vm = float(v_meas_m_s)
        fm = float(force_n)
        self._q_vm = _lpf_step(self._q_vm, vm, alpha)
        self._q_fm = _lpf_step(self._q_fm, fm, alpha)
        qtinv_vm = self._qtinv(vm, self._q_vm, tp, omega_q)
        hp = omega_q * tp * (fm - self._q_fm)
        a_disc = math.exp(-damp * dt_s / mass)
        b_disc = (1.0 - a_disc) / damp
        self._n1 = a_disc * self._n1 + b_disc * hp
        # q_vi+ = α V_i + (1-α) q_vi
        # V_i = V_r - (qtinv - q_vi+ + n1)
        # (1-α) V_i = V_r - qtinv - n1 + (1-α) q_vi
        den = max(1.0 - alpha, 1e-9)
        vi = (
            float(v_nom_m_s) - qtinv_vm - self._n1 + (1.0 - alpha) * self._q_vi
        ) / den
        self._q_vi = _lpf_step(self._q_vi, vi, alpha)
        self._last_committed = float(vi)
        self._publish(
            pert_unclipped=float(v_nom_m_s) - vi,
            qtinv_vm=qtinv_vm,
            q_vi=float(self._q_vi),
            n1=float(self._n1),
            n2=0.0,
            vi=vi,
            blend=1.0,
            corr=float(v_nom_m_s) - vi,
        )
        return float(vi)

    def closed_form_vi(
        self,
        v_nom_m_s: float,
        *,
        v_meas_m_s: float,
        force_n: float,
        dt_s: float,
        mass_z: float,
        damping_z: float,
    ) -> float:
        """Unsaturated linear test form; never used by runtime control."""
        omega_q = self._omega_q_rad()
        alpha = _lpf_alpha(omega_q, dt_s)
        tp = max(self.cfg.time_constant_s(), 1e-4)
        mass = max(float(mass_z), 1e-3)
        damp = max(float(damping_z), 1e-6)
        vm = float(v_meas_m_s)
        fm = float(force_n)
        self._q_vm = _lpf_step(self._q_vm, vm, alpha)
        self._q_fm = _lpf_step(self._q_fm, fm, alpha)
        qtinv_vm = self._qtinv(vm, self._q_vm, tp, omega_q)
        hp = omega_q * tp * (fm - self._q_fm)
        a_disc = math.exp(-damp * dt_s / mass)
        b_disc = (1.0 - a_disc) / damp
        self._n1 = a_disc * self._n1 + b_disc * hp
        u = float(v_nom_m_s) - float(self._n1) - qtinv_vm
        # Discrete 1/(1-Q) for Q = α / (1 - (1-α) z^{-1}):
        # y_k = y_{k-1} + u_k/(1-α) - u_{k-1}
        den = max(1.0 - alpha, 1e-9)
        vi = self._closed_vi + u / den - self._closed_u_prev
        self._closed_vi = float(vi)
        self._closed_u_prev = float(u)
        self._last_committed = float(vi)
        self._publish(
            pert_unclipped=float(v_nom_m_s) - vi,
            qtinv_vm=qtinv_vm,
            q_vi=0.0,
            n1=float(self._n1),
            n2=0.0,
            vi=vi,
            blend=1.0,
            corr=float(v_nom_m_s) - vi,
        )
        return float(vi)

    def update(
        self,
        v_nom_m_s: float,
        *,
        v_meas_m_s: float | None,
        force_n: float,
        dt_s: float,
        mass_z: float,
        damping_z: float,
        apply_scale: float = 1.0,
        snap_blend: bool = False,
    ) -> float:
        cfg = self.cfg
        mode = cfg.normalized_mode()
        if mode == "active":
            cfg.assert_active_ready()
        self.last_omega_q_hz = self._omega_q_hz()
        if mode == "off" or dt_s <= 0.0:
            self.last_corr_m_s = 0.0
            self._blend = 0.0
            self._publish(
                pert_unclipped=0.0,
                qtinv_vm=0.0,
                q_vi=0.0,
                n1=0.0,
                n2=0.0,
                vi=float(self._last_committed),
                blend=0.0,
                corr=0.0,
            )
            return float(v_nom_m_s)

        v_meas = (
            float(v_meas_m_s)
            if v_meas_m_s is not None and np.isfinite(v_meas_m_s)
            else float(v_nom_m_s)
        )
        # Causal observer timing: V_i[k-1] was committed after all downstream
        # constraints on the previous tick.  Γ_d belongs to the identified
        # plant model and is not inverted or replayed here.
        vi_previous = float(self._last_committed)
        qtinv_vm, q_vi, n1, n2 = self._step_filters(
            v_meas=v_meas,
            force_n=float(force_n),
            vi_for_q=vi_previous,
            dt_s=float(dt_s),
            mass_z=mass_z,
            damping_z=damping_z,
        )
        pert = qtinv_vm - q_vi + n1 - n2
        clipped = pert
        saturated = False
        if cfg.v_corr_max_m_s > 0.0:
            clipped = _clamp(pert, -cfg.v_corr_max_m_s, cfg.v_corr_max_m_s)
            saturated = abs(pert) > cfg.v_corr_max_m_s + 1e-12

        target = (
            float(_clamp(apply_scale, 0.0, 1.0)) if mode == "active" else 0.0
        )
        if snap_blend and target > 0.0:
            # Overforce spikes are shorter than blend_s.  Waiting 0.3 s
            # leaves corr at zero for the only ticks the observer is needed.
            self._blend = target
        elif cfg.blend_s > 1e-6:
            rate = float(dt_s) / cfg.blend_s
            if target > self._blend:
                self._blend = min(target, self._blend + rate)
            else:
                self._blend = max(target, self._blend - rate)
        else:
            self._blend = target

        corr = self._blend * clipped
        candidate = float(v_nom_m_s) - float(corr)
        self._last_candidate = candidate
        self.last_corr_m_s = float(corr)
        self._publish(
            pert_unclipped=pert,
            qtinv_vm=qtinv_vm,
            q_vi=q_vi,
            n1=n1,
            n2=n2,
            vi=vi_previous,
            blend=self._blend,
            corr=corr,
            saturated=saturated,
            pert_clipped=clipped,
        )
        return candidate

    def commit_sent(
        self,
        v_sent_m_s: float,
        *,
        candidate_m_s: float | None = None,
        dt_s: float | None = None,
    ) -> None:
        """Commit post-constraint V_i[k] for the next observer tick.

        ``sent-candidate`` is external-reset anti-windup feedback.  There is no
        separate ``1/(1-Q)`` integrator in the runtime path; feeding the actual
        sent value through the next QV_i update prevents hidden command windup.
        """
        if self.cfg.normalized_mode() == "off":
            return
        del dt_s  # Kept for call-site compatibility; Γ_d is not replayed.
        sent = float(v_sent_m_s) if np.isfinite(v_sent_m_s) else 0.0
        candidate = (
            float(candidate_m_s)
            if candidate_m_s is not None and np.isfinite(candidate_m_s)
            else float(self._last_candidate)
        )
        self._last_committed = sent
        self.telemetry.vi_m_s = sent
        self.telemetry.candidate_m_s = candidate
        self.telemetry.antiwindup_error_m_s = sent - candidate
        self.telemetry.constrained = abs(sent - candidate) > 1e-9
        self.telemetry.linear_equivalent = bool(
            self.cfg.applies()
            and self._blend >= 1.0 - 1e-9
            and not self.telemetry.saturated
            and not self.telemetry.constrained
        )

    def _publish(
        self,
        *,
        pert_unclipped: float,
        qtinv_vm: float,
        q_vi: float,
        n1: float,
        n2: float,
        vi: float,
        blend: float,
        corr: float,
        saturated: bool = False,
        pert_clipped: float | None = None,
    ) -> None:
        clipped = (
            float(pert_unclipped) if pert_clipped is None else float(pert_clipped)
        )
        self.last_corr_m_s = float(corr)
        self.telemetry = CdyobTelemetry(
            corr_m_s=float(corr),
            pert_unclipped=float(pert_unclipped),
            pert_clipped=clipped,
            qtinv_vm=float(qtinv_vm),
            q_vi=float(q_vi),
            n1_force=float(n1),
            n2_velocity=float(n2),
            blend=float(blend),
            vi_m_s=float(vi),
            candidate_m_s=float(self._last_candidate),
            antiwindup_error_m_s=0.0,
            residual=float(qtinv_vm - q_vi),
            saturated=bool(saturated),
            constrained=False,
            linear_equivalent=False,
            omega_q_hz=float(self.last_omega_q_hz),
            mode=self.cfg.normalized_mode(),
        )
