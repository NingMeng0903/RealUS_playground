"""Closed-form 8-DoF rail allocation, 20 Hz reference model, and 200 Hz observer.

L1 produces a committed rail velocity ``v_r,ref``.  It is *not* a TCP
closed loop: the arm still solves ``J_a q̇_a = v_d − J_r v̂_r`` in QP1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.filters import (
    first_order_lpf,
    lpf_tau_from_fc,
)
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    stopping_velocity,
    wall_cap,
)


@dataclass
class RailAllocatorConfig:
    """L1 rail allocation + VPC mid-ranging.  Always on in COUPLED mode."""

    # Task-side scale: metres/s and rad/s so v and ω share one residual.
    v0_m_s: float = 0.05
    w0_rad_s: float = 0.30
    # Chan-Dubey: near-limit joints get larger margin_weight → smaller W^{-1}.
    k_margin: float = 4.0
    # VPC mid-ranging (Ma 2015 C_s).  Error is Cartesian d = y_tcp − y_rail − d*.
    kp_mid: float = 1.2
    ki_mid: float = 0.80
    u_mid_max_m_s: float = 0.12
    # Cheapen the rail on large |e_mid| or |v_y| (teleop stand-in for
    # Holistic 2022 (14) pose-error term; not that formula).
    k_err_rail: float = 4.0
    e_ref_m: float = 0.08
    # Reference-model cutoff.  τ = 1/(2π f_c).  4 Hz follows Y; stay below 5–10.
    f_c_hz: float = 4.0
    # L1 jerk.  Hard QP box stays at qp.j_max_rail_m_s3 (320).
    j_max_ref_m_s3: float = 60.0
    # Lillo ε: leave stays on until this far inside the soft line (8 mm).
    leave_exit_eps_m: float = 0.008
    kaw_mid: float = 8.0
    rho_mirror_a: float = 0.50
    rho_mirror_j: float = 0.30
    # One-sided braking envelope (same formula as the worker override).
    reaction_s: float = 0.06
    observer_pos_gain: float = 0.35
    observer_vel_gain: float = 2.0
    observer_vel_lpf_hz: float = 8.0


def allocate_rail(
    J: np.ndarray,
    v_d: np.ndarray,
    *,
    qdot_scale: np.ndarray,
    margin_weight: np.ndarray,
    lam: float,
    v0_m_s: float = 0.05,
    w0_rad_s: float = 0.30,
    e_mid: float = 0.0,
    k_err: float = 0.0,
    e_ref: float = 0.08,
) -> tuple[float, np.ndarray]:
    """Weighted damped least-norm: ``q̇ = W⁻¹ J_nᵀ (J_n W⁻¹ J_nᵀ + λ²I)⁻¹ v_n``.

    ``qdot_scale`` is ``[v_r_max, q̇_max_1..7]``.  ``margin_weight`` is
    Chan-Dubey (≥1); larger means more expensive.  Returns ``(u_r, q̇)``.
    """
    J = np.asarray(J, dtype=float)
    v = np.asarray(v_d, dtype=float).reshape(-1)
    if J.shape[0] != 6 or v.size != 6:
        raise ValueError("allocate_rail expects a 6×n Jacobian and a 6-vector v_d")
    s = np.asarray(qdot_scale, dtype=float).reshape(-1)
    mw = np.asarray(margin_weight, dtype=float).reshape(-1)
    if s.size != J.shape[1] or mw.size != J.shape[1]:
        raise ValueError("qdot_scale / margin_weight must match Jacobian columns")
    scale = np.array(
        [v0_m_s, v0_m_s, v0_m_s, w0_rad_s, w0_rad_s, w0_rad_s], dtype=float
    )
    scale = np.maximum(scale, 1.0e-9)
    v_n = v / scale
    J_n = J / scale[:, None]
    Winv_diag = (s * s) / np.maximum(mw, 1.0e-9)
    # Holistic 2022 (14) uses pose error.  Mid-scan |e_mid| is small, so
    # |v_y|/v0 is the coarse-travel stand-in.  Not the paper formula.
    if float(k_err) > 0.0:
        e_term = abs(float(e_mid)) / max(float(e_ref), 1.0e-9)
        v_term = abs(float(v[1])) / max(float(v0_m_s), 1.0e-9)
        gain = 1.0 + float(k_err) * min(max(e_term, v_term), 1.0)
        Winv_diag[0] *= gain * gain
    JW = J_n * Winv_diag[None, :]
    a = JW @ J_n.T
    lam2 = float(lam) * float(lam)
    a.flat[::7] += lam2
    try:
        y = np.linalg.solve(a, v_n)
    except np.linalg.LinAlgError:
        y = np.linalg.lstsq(a, v_n, rcond=None)[0]
    qdot = Winv_diag * (J_n.T @ y)
    return float(qdot[0]), qdot


@dataclass
class RailReferenceState:
    v: float = 0.0
    a: float = 0.0
    initialized: bool = False


class RailReferenceModel:
    """Δt-adaptive first-order LPF, then hard |a| / |j| boxes, then wall cap.

    History is the *committed* ``v_r,ref`` so the next tick's boxes stay
    consistent with what the worker actually received.
    """

    def __init__(
        self,
        *,
        f_c_hz: float = 1.0,
        a_max: float = 0.60,
        j_max: float = 60.0,
        v_max: float = 0.12,
        reaction_s: float = 0.06,
        soft_min_m: float = 0.015,
        soft_max_m: float = 0.77,
        hard_min_m: float | None = None,
        hard_max_m: float | None = None,
    ) -> None:
        self.f_c_hz = float(f_c_hz)
        self.a_max = float(a_max)
        self.j_max = float(j_max)
        self.v_max = float(v_max)
        self.reaction_s = float(reaction_s)
        self.soft_min_m = float(soft_min_m)
        self.soft_max_m = float(soft_max_m)
        self.hard_min_m = float(soft_min_m if hard_min_m is None else hard_min_m)
        self.hard_max_m = float(soft_max_m if hard_max_m is None else hard_max_m)
        self.state = RailReferenceState()
        self.last_wall_override = False
        self.last_v_lpf = 0.0

    def reset(self, v0: float = 0.0) -> None:
        self.state = RailReferenceState(v=float(v0), a=0.0, initialized=False)
        self.last_wall_override = False
        self.last_v_lpf = float(v0)

    def track(self, v_applied: float, dt_s: float = 0.0) -> float:
        """Shadow a rail velocity written by another authority."""
        v = float(v_applied)
        dt = float(dt_s)
        if self.state.initialized and dt > 1.0e-12:
            a_raw = (v - float(self.state.v)) / dt
            da = float(self.j_max) * dt
            a = float(np.clip(a_raw, float(self.state.a) - da, float(self.state.a) + da))
            a = float(np.clip(a, -self.a_max, self.a_max))
            self.state.a = a
        else:
            self.state.a = 0.0
        self.state.v = v
        self.state.initialized = True
        self.last_v_lpf = v
        return v

    def project_into_wall(self, leave_sign: float) -> None:
        from rm75_control.control.joint_admittance_8dof.tasks.rail_command import (
            project_lpf_into_wall,
        )

        self.last_v_lpf = project_lpf_into_wall(self.last_v_lpf, leave_sign)
        self.state.v = project_lpf_into_wall(float(self.state.v), leave_sign)

    def step(
        self,
        u_r: float,
        dt_s: float,
        *,
        x_m: float,
        apply_wall: bool = True,
        a_max: float | None = None,
        j_max: float | None = None,
        leave_sign: float = 0.0,
    ) -> float:
        dt = float(dt_s)
        if dt <= 1.0e-9:
            return float(self.state.v)
        a_lim = float(self.a_max if a_max is None else min(self.a_max, abs(float(a_max))))
        j_lim = float(self.j_max if j_max is None else min(self.j_max, abs(float(j_max))))
        tau = lpf_tau_from_fc(self.f_c_hz)
        u = float(u_r)
        if not self.state.initialized:
            v_f = u
            self.state.initialized = True
        elif tau <= 1.0e-9:
            v_f = u
        else:
            v_f = first_order_lpf(float(self.state.v), u, dt, tau)
        from rm75_control.control.joint_admittance_8dof.tasks.rail_command import (
            project_lpf_into_wall,
        )

        v_f = project_lpf_into_wall(v_f, leave_sign)
        self.last_v_lpf = float(v_f)
        v_prev = project_lpf_into_wall(float(self.state.v), leave_sign)
        a_prev = float(self.state.a)
        a_raw = (v_f - v_prev) / dt
        da_max = float(j_lim) * dt
        a = float(np.clip(a_raw, a_prev - da_max, a_prev + da_max))
        a = float(np.clip(a, -a_lim, a_lim))
        v = v_prev + a * dt
        v = float(np.clip(v, -self.v_max, self.v_max))
        self.last_wall_override = False
        if apply_wall:
            lo_cap, hi_cap = wall_cap(
                float(x_m),
                lo=self.hard_min_m,
                hi=self.hard_max_m,
                a_max=a_lim,
                reaction_s=self.reaction_s,
            )
            v_clamped = float(np.clip(v, lo_cap, hi_cap))
            if abs(v_clamped - v) > 1.0e-9:
                self.last_wall_override = True
            v = v_clamped
            a = (v - v_prev) / dt
        self.state.v = float(v)
        self.state.a = float(a)
        return float(v)


class RailStateObserver:
    """200 Hz output: predict with ``v_r,ref``, correct on timestamped encoder.

    This estimates 0–10 Hz rail motion.  It is not a 50 Hz velocity sensor.
    """

    def __init__(
        self,
        *,
        pos_gain: float = 0.35,
        vel_gain: float = 2.0,
        vel_lpf_hz: float = 8.0,
        v_max: float = 0.30,
    ) -> None:
        self.pos_gain = float(pos_gain)
        self.vel_gain = float(vel_gain)
        self.vel_lpf_hz = float(vel_lpf_hz)
        self.v_max = float(v_max)
        self.q_hat = 0.0
        self.v_hat = 0.0
        self._last_sample_t: float | None = None
        self._initialized = False

    def reset(self, q0: float = 0.0, v0: float = 0.0) -> None:
        self.q_hat = float(q0)
        self.v_hat = float(v0)
        self._last_sample_t = None
        self._initialized = True

    def update(
        self,
        *,
        now_s: float,
        dt_s: float,
        v_r_ref: float,
        q_meas: float,
        sample_t: float,
        v_meas: float | None = None,
        v_written: float | None = None,
    ) -> tuple[float, float]:
        if not self._initialized:
            self.reset(q_meas, float(v_meas) if v_meas is not None else 0.0)
            self._last_sample_t = float(sample_t)
            return float(self.q_hat), float(self.v_hat)
        dt = max(float(dt_s), 1.0e-6)
        # Predict with the last written FA24 / measured RPM, never the
        # internal v_r,ref.  Using v_r_ref made the observer optimistic and
        # the arm compensated a rail that had not actually moved.
        if v_written is not None and np.isfinite(float(v_written)):
            v_pred = float(v_written)
        elif v_meas is not None and np.isfinite(float(v_meas)):
            v_pred = float(v_meas)
        else:
            v_pred = float(self.v_hat)
        del v_r_ref
        self.q_hat = float(self.q_hat) + v_pred * dt
        tau = lpf_tau_from_fc(self.vel_lpf_hz)
        if tau <= 1.0e-9:
            self.v_hat = v_pred
        else:
            self.v_hat = first_order_lpf(float(self.v_hat), v_pred, dt, tau)
        if np.isfinite(sample_t) and (
            self._last_sample_t is None or float(sample_t) > float(self._last_sample_t) + 1.0e-9
        ):
            age = max(0.0, float(now_s) - float(sample_t))
            q_pred_at_sample = float(self.q_hat) - v_pred * age
            innov = float(q_meas) - q_pred_at_sample
            self.q_hat += self.pos_gain * innov
            self.v_hat += self.vel_gain * innov
            if v_meas is not None and np.isfinite(float(v_meas)):
                blend = min(1.0, dt * 8.0)
                self.v_hat = (1.0 - blend) * float(self.v_hat) + blend * float(v_meas)
            self._last_sample_t = float(sample_t)
        self.v_hat = float(np.clip(self.v_hat, -self.v_max, self.v_max))
        return float(self.q_hat), float(self.v_hat)


def margin_weight_from_activation(
    q: np.ndarray,
    q_mid: np.ndarray,
    half: np.ndarray,
    *,
    k_margin: float,
    activation: float,
) -> np.ndarray:
    """Per-joint Chan-Dubey weight.  Rail uses the same formula in metres."""
    q = np.asarray(q, dtype=float)
    mid = np.asarray(q_mid, dtype=float)
    h = np.maximum(np.asarray(half, dtype=float), 1.0e-9)
    u = np.clip(np.abs(q - mid) / h, 0.0, 1.0)
    span = max(1.0 - float(activation), 1.0e-6)
    over = np.clip((u - float(activation)) / span, 0.0, 1.0)
    return 1.0 + float(k_margin) * over * over


def margin_weight_toward_box(
    q: float,
    lo: float,
    hi: float,
    toward: float,
    *,
    k_margin: float,
) -> float:
    """Chan-Dubey weight toward the approached wall only.

    Activation is half the box width.  Interior stays ~1; the wall is
    ``1+k_margin``.  Leaving the wall is 1.
    """
    q_v = float(q)
    lo_v = float(lo)
    hi_v = float(hi)
    toward_v = float(toward)
    if not (hi_v > lo_v):
        return 1.0
    if not (np.isfinite(q_v) and np.isfinite(toward_v)):
        return 1.0
    if not (toward_v > 0.0 or toward_v < 0.0):
        return 1.0
    k = max(float(k_margin), 0.0)
    activate = 0.5 * (hi_v - lo_v)
    if not (activate > 0.0):
        return 1.0
    d_wall = (q_v - lo_v) if toward_v < 0.0 else (hi_v - q_v)
    over = float(np.clip((activate - d_wall) / activate, 0.0, 1.0))
    return 1.0 + k * over * over


def soft_saturate(value: float, limit: float) -> float:
    """``limit * tanh(value / limit)``.  Keeps a gradient at the cap."""

    lim = max(float(limit), 1.0e-9)
    return float(lim * np.tanh(float(value) / lim))


class MidrangingController:
    """PI on Cartesian mid-ranging error ``e_mid = (y_tcp − y_rail) − d*``."""

    def __init__(
        self,
        *,
        kp: float = 1.2,
        ki: float = 0.80,
        v_max: float = 0.12,
        kaw: float = 8.0,
    ) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.v_max = float(v_max)
        self.kaw = float(kaw)
        self.integ = 0.0
        self.last_raw = 0.0
        self.last_projected = False

    def reset(self) -> None:
        self.integ = 0.0
        self.last_raw = 0.0
        self.last_projected = False

    def step(
        self,
        err_m: float,
        dt_s: float,
        *,
        freeze: bool = False,
        leave_only_sign: float = 0.0,
        u_committed: float | None = None,
    ) -> float:
        """Return saturated mid-ranging velocity.

        ``leave_only_sign`` > 0 at the plus hard wall (only negative u_mid),
        < 0 at the minus hard wall (only positive u_mid).  0 leaves u_mid
        unconstrained.  Integrator anti-windup uses back-calculation against
        the committed rail command when one is supplied.
        """
        err = float(err_m) if np.isfinite(err_m) else 0.0
        dt = max(float(dt_s), 0.0)
        if not freeze and dt > 0.0:
            self.integ += self.ki * err * dt
        raw = self.kp * err + self.integ
        sat = soft_saturate(raw, self.v_max)
        sign = float(leave_only_sign)
        projected = False
        if sign > 0.0 and sat > 0.0:
            sat = 0.0
            projected = True
        elif sign < 0.0 and sat < 0.0:
            sat = 0.0
            projected = True
        self.last_raw = float(raw)
        self.last_projected = bool(projected)
        if freeze:
            return float(sat)
        if dt > 0.0 and u_committed is not None and np.isfinite(float(u_committed)):
            self.integ += self.kaw * (float(u_committed) - float(raw)) * dt
        elif not freeze and abs(raw) > self.v_max:
            self.integ -= self.ki * err * dt
        if projected and dt > 0.0:
            # Do not keep integrating a command that the wall already killed.
            self.integ -= self.ki * err * dt
        return float(sat)


def wall_leave_only_sign(
    x_m: float,
    *,
    hard_min_m: float,
    hard_max_m: float,
    band_m: float,
) -> float:
    """+1 near the plus hard wall (only leave/negative u), -1 near minus, else 0."""
    x = float(x_m)
    band = max(float(band_m), 0.0)
    hi = float(hard_max_m)
    lo = float(hard_min_m)
    if x >= hi - band:
        return 1.0
    if x <= lo + band:
        return -1.0
    return 0.0


def update_leave_sign(
    raw: float,
    *,
    x_m: float,
    hard_min_m: float,
    hard_max_m: float,
    band_m: float,
    exit_eps_m: float,
    prev_leave: float,
) -> float:
    """Enter on the raw band; exit only after ``exit_eps_m`` inside the soft line."""
    if float(raw) > 0.0:
        return 1.0
    if float(raw) < 0.0:
        return -1.0
    eps = max(float(exit_eps_m), 0.0)
    band = max(float(band_m), 0.0)
    x = float(x_m)
    if float(prev_leave) > 0.0 and x > float(hard_max_m) - band - eps:
        return 1.0
    if float(prev_leave) < 0.0 and x < float(hard_min_m) + band + eps:
        return -1.0
    return 0.0


def arm_mirror_rail_limits(
    J: np.ndarray,
    a_arm_max: np.ndarray,
    j_arm_max: np.ndarray,
    *,
    rho_a: float = 0.50,
    rho_j: float = 0.30,
) -> tuple[float, float]:
    """Max |a_r|, |j_r| the arm can still mirror: qa = −Ja# Jr vr."""
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or J.shape[0] < 1 or J.shape[1] < 2:
        return float("inf"), float("inf")
    ja = J[:, 1:]
    jr = J[:, 0]
    try:
        p, *_ = np.linalg.lstsq(ja, jr, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf"), float("inf")
    p = np.abs(np.asarray(p, dtype=float).reshape(-1))
    a_arm = np.abs(np.asarray(a_arm_max, dtype=float).reshape(-1))
    j_arm = np.abs(np.asarray(j_arm_max, dtype=float).reshape(-1))
    a_lim = float("inf")
    j_lim = float("inf")
    n = min(p.size, a_arm.size)
    for i in range(n):
        if p[i] <= 1.0e-6:
            continue
        a_lim = min(a_lim, float(rho_a) * float(a_arm[i]) / float(p[i]))
    n_j = min(p.size, j_arm.size)
    for i in range(n_j):
        if p[i] <= 1.0e-6:
            continue
        j_lim = min(j_lim, float(rho_j) * float(j_arm[i]) / float(p[i]))
    if not np.isfinite(a_lim):
        a_lim = float("inf")
    if not np.isfinite(j_lim):
        j_lim = float("inf")
    return float(max(a_lim, 0.0)), float(max(j_lim, 0.0))


def project_arm_compensation(
    J: np.ndarray,
    delta_v_req: np.ndarray,
    q: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    *,
    activation: float = 0.80,
    alpha: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Tu 2022 eq. (22): drop compensation that drives the arm into limits."""

    J = np.asarray(J, dtype=float)
    req = np.asarray(delta_v_req, dtype=float).reshape(-1)
    if J.ndim != 2 or J.shape[0] != req.size or J.shape[1] < 2:
        return req.copy(), 0.0
    J_a = J[:, 1:]
    try:
        qdot_a, *_ = np.linalg.lstsq(J_a, req, rcond=None)
    except np.linalg.LinAlgError:
        return req.copy(), 0.0
    q_a = np.asarray(q, dtype=float).reshape(-1)[1 : 1 + qdot_a.size]
    lo = np.asarray(q_lower, dtype=float).reshape(-1)[1 : 1 + qdot_a.size]
    hi = np.asarray(q_upper, dtype=float).reshape(-1)[1 : 1 + qdot_a.size]
    if q_a.size != qdot_a.size:
        return req.copy(), 0.0
    half = np.maximum(0.5 * (hi - lo), 1.0e-9)
    mid = 0.5 * (hi + lo)
    u = (q_a - mid) / half
    toward_limit = (u * qdot_a) > 0.0
    near = np.abs(u) >= float(activation)
    mask = toward_limit & near
    qdot_p = np.asarray(qdot_a, dtype=float).copy()
    qdot_p[mask] *= 1.0 - float(np.clip(alpha, 0.0, 1.0))
    cmp = J_a @ qdot_p
    nreq = float(np.linalg.norm(req))
    frac = 0.0 if nreq < 1.0e-12 else float(1.0 - np.linalg.norm(cmp) / nreq)
    return np.asarray(cmp, dtype=float), float(np.clip(frac, 0.0, 1.0))


__all__ = (
    "MidrangingController",
    "RailAllocatorConfig",
    "RailReferenceModel",
    "RailReferenceState",
    "RailStateObserver",
    "allocate_rail",
    "arm_mirror_rail_limits",
    "lpf_tau_from_fc",
    "margin_weight_from_activation",
    "margin_weight_toward_box",
    "project_arm_compensation",
    "soft_saturate",
    "stopping_velocity",
    "update_leave_sign",
    "wall_cap",
    "wall_leave_only_sign",
)
