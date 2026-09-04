"""Huber-FGLS static fit, delay search, gated inertia. Holdout is never used to pick Δt."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rm75_control.force.compensation.v2.regressor_v2 import (
    inertia_triangle_ok,
    payload_wrench_mhb,
    static_design,
    unmodeled_inertia_torque_bound,
)
from rm75_control.force.compensation.regressor import inertia_op, skew


def huber_weights(r: np.ndarray, delta: float = 1.345) -> np.ndarray:
    a = np.abs(r)
    w = np.ones_like(a)
    mask = a > delta
    w[mask] = delta / np.maximum(a[mask], 1e-12)
    return w


def robust_mean(X: np.ndarray, *, iters: int = 8) -> np.ndarray:
    x = np.asarray(X, dtype=float)
    mu = np.median(x, axis=0)
    for _ in range(iters):
        r = x - mu
        s = np.median(np.abs(r), axis=0) * 1.4826 + 1e-9
        w = huber_weights((r / s).ravel()).reshape(r.shape)
        mu = np.sum(w * x, axis=0) / np.maximum(np.sum(w, axis=0), 1e-9)
    return mu


def pooled_shrinkage_cov(windows: list[np.ndarray], *, lam: float = 0.2) -> np.ndarray:
    chunks = [np.asarray(w, dtype=float).reshape(-1, 6) for w in windows if len(w)]
    if not chunks:
        return np.eye(6)
    X = np.vstack(chunks)
    mu = robust_mean(X)
    C = np.cov((X - mu).T)
    if C.ndim != 2:
        C = np.eye(6)
    C = 0.5 * (C + C.T)
    return (1.0 - lam) * C + lam * np.diag(np.diag(C) + 1e-9)


def huber_fgls(
    A_list: list[np.ndarray],
    y_list: list[np.ndarray],
    Q_list: list[np.ndarray],
    *,
    n_iter: int = 12,
) -> np.ndarray:
    n = A_list[0].shape[1]
    AtWA = np.zeros((n, n))
    AtWy = np.zeros(n)
    for A, y, Q in zip(A_list, y_list, Q_list, strict=True):
        AtWA += A.T @ Q @ A
        AtWy += A.T @ Q @ y
    theta = np.linalg.lstsq(AtWA, AtWy, rcond=None)[0]
    for _ in range(n_iter):
        AtWA[:] = 0.0
        AtWy[:] = 0.0
        for A, y, Q in zip(A_list, y_list, Q_list, strict=True):
            r = y - A @ theta
            L = np.linalg.cholesky(Q + 1e-12 * np.eye(6))
            rw = L.T @ r
            w = huber_weights(rw)
            W = L @ np.diag(w) @ L.T
            AtWA += A.T @ W @ A
            AtWy += A.T @ W @ y
        theta = np.linalg.lstsq(AtWA + 1e-10 * np.eye(n), AtWy, rcond=None)[0]
    return theta


@dataclass
class StaticWindow:
    g_L: np.ndarray
    wrench_L: np.ndarray
    t_s: float
    n_eff: float = 1.0
    samples: np.ndarray | None = None
    is_train: bool = True
    is_anchor: bool = False
    block_id: int = 0
    name: str = ""


@dataclass
class StaticFitResult:
    theta_m0: np.ndarray
    theta_m1: np.ndarray | None
    drift_enabled: bool
    mass_kg: float
    h_L: np.ndarray
    bias0: np.ndarray
    bias_drift_per_s: np.ndarray
    rank_m0: int
    rank_m1: int
    cond_m0: float
    cv_m0: float
    cv_m1: float
    holdout_window_err: np.ndarray


def _split_theta_m0(theta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    return float(theta[0]), theta[1:4].copy(), theta[4:10].copy()


def fit_static_windows(
    windows: list[StaticWindow],
    *,
    Sigma: np.ndarray,
    m_min: float = 0.05,
    m_max: float = 5.0,
    r_max_m: float = 0.12,
    eps_b: float = 0.02,
) -> StaticFitResult:
    train = [w for w in windows if w.is_train]
    hold = [w for w in windows if not w.is_train]
    Q0 = np.linalg.pinv(Sigma, rcond=1e-10)

    def pack(include_drift: bool) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        A, y, Q = [], [], []
        for w in train:
            Aw = static_design(w.g_L, include_drift=include_drift, t_s=w.t_s)
            A.append(Aw)
            y.append(np.asarray(w.wrench_L, dtype=float).reshape(6))
            Q.append(float(w.n_eff) * Q0)
        return A, y, Q

    A0, y0, Qs = pack(False)
    theta0 = huber_fgls(A0, y0, Qs)
    m, h, b0 = _split_theta_m0(theta0)
    m = float(np.clip(m, m_min, m_max))
    if np.linalg.norm(h) > m * r_max_m:
        h = h * (m * r_max_m / (np.linalg.norm(h) + 1e-12))
    theta0 = np.concatenate([[m], h, b0])

    A_stack = np.vstack(A0)
    s = np.linalg.svd(A_stack, compute_uv=False)
    s_pos = s[s > 1e-10]
    rank_m0 = int(s_pos.size)
    cond_m0 = float(s_pos[0] / s_pos[-1]) if s_pos.size else float("inf")

    A1, y1, Q1 = pack(True)
    theta1 = huber_fgls(A1, y1, Q1)
    A1s = np.vstack(A1)
    s1 = np.linalg.svd(A1s, compute_uv=False)
    rank_m1 = int(np.sum(s1 > 1e-10))

    blocks = sorted({w.block_id for w in train})
    cv0, cv1 = [], []
    if len(blocks) >= 2:
        for b in blocks:
            tr = [w for w in train if w.block_id != b]
            te = [w for w in train if w.block_id == b]
            if not tr or not te:
                continue
            fit0 = huber_fgls(*_pack_subset(tr, False, Q0))
            fit1 = huber_fgls(*_pack_subset(tr, True, Q0))
            cv0.append(_rmse_windows(te, fit0, False))
            cv1.append(_rmse_windows(te, fit1, True))
    cv_m0 = float(np.mean(cv0)) if cv0 else _rmse_windows(train, theta0, False)
    cv_m1 = float(np.mean(cv1)) if cv1 else _rmse_windows(train, theta1, True)
    drift_on = bool(cv0) and (cv_m1 < cv_m0 - float(eps_b))

    drift = np.zeros(6)
    if drift_on:
        m, h = float(theta1[0]), theta1[1:4].copy()
        b0 = theta1[4:10].copy()
        drift = theta1[10:16].copy()
        m = float(np.clip(m, m_min, m_max))
        if np.linalg.norm(h) > m * r_max_m:
            h = h * (m * r_max_m / (np.linalg.norm(h) + 1e-12))

    hold_err = np.zeros(6)
    if hold:
        errs = []
        for w in hold:
            yhat = static_design(w.g_L, include_drift=drift_on, t_s=w.t_s) @ (
                np.concatenate([[m], h, b0, drift]) if drift_on else np.concatenate([[m], h, b0])
            )
            errs.append(np.asarray(w.wrench_L) - yhat)
        hold_err = np.mean(np.abs(np.vstack(errs)), axis=0)

    return StaticFitResult(
        theta_m0=theta0,
        theta_m1=theta1,
        drift_enabled=drift_on,
        mass_kg=m,
        h_L=h,
        bias0=b0,
        bias_drift_per_s=drift,
        rank_m0=rank_m0,
        rank_m1=rank_m1,
        cond_m0=cond_m0,
        cv_m0=cv_m0,
        cv_m1=cv_m1,
        holdout_window_err=hold_err,
    )


def _pack_subset(windows, include_drift, Q0):
    A, y, Q = [], [], []
    for w in windows:
        A.append(static_design(w.g_L, include_drift=include_drift, t_s=w.t_s))
        y.append(np.asarray(w.wrench_L, dtype=float).reshape(6))
        Q.append(float(w.n_eff) * Q0)
    return A, y, Q


def _rmse_windows(windows, theta, include_drift) -> float:
    e = []
    for w in windows:
        yhat = static_design(w.g_L, include_drift=include_drift, t_s=w.t_s) @ theta
        e.append(np.asarray(w.wrench_L) - yhat)
    return float(np.sqrt(np.mean(np.square(np.vstack(e)))))


def wrench_residual_stats(e6: np.ndarray) -> dict:
    e6 = np.asarray(e6, dtype=float).reshape(-1, 6)
    e = e6.ravel()
    return {
        "rms_all": float(np.sqrt(np.mean(e**2))),
        "rms_force": float(np.sqrt(np.mean(e6[:, :3] ** 2))),
        "rms_moment": float(np.sqrt(np.mean(e6[:, 3:] ** 2))),
        "per_axis": np.sqrt(np.mean(e6**2, axis=0)).tolist(),
    }


def _window_yhat(w: StaticWindow, fit: StaticFitResult) -> np.ndarray:
    theta = np.concatenate([[fit.mass_kg], fit.h_L, fit.bias0])
    if fit.drift_enabled:
        theta = np.concatenate([theta, fit.bias_drift_per_s])
    return static_design(w.g_L, include_drift=fit.drift_enabled, t_s=w.t_s) @ theta


def _window_errors(w: StaticWindow, fit: StaticFitResult) -> np.ndarray:
    yhat = _window_yhat(w, fit)
    if w.samples is not None and len(w.samples):
        return np.asarray(w.samples, dtype=float).reshape(-1, 6) - yhat
    return (np.asarray(w.wrench_L, dtype=float).reshape(6) - yhat).reshape(1, 6)


def static_residual_report(windows: list[StaticWindow], fit: StaticFitResult) -> dict[str, dict]:
    """Per-group static residuals, same keys as the 4-pose ``holdout_by_pose`` report."""
    out: dict[str, dict] = {}
    if not windows:
        return out
    groups: dict[str, list[StaticWindow]] = {"all": list(windows)}
    train = [w for w in windows if w.is_train]
    hold = [w for w in windows if not w.is_train]
    if train:
        groups["train"] = train
    if hold:
        groups["holdout"] = hold
    for w in windows:
        key = str(w.name or "").strip() or ("train" if w.is_train else "holdout")
        if key in groups:
            continue
        groups[key] = [w]
    for key, chunk in groups.items():
        e6 = np.vstack([_window_errors(w, fit) for w in chunk])
        out[key] = wrench_residual_stats(e6)
    return out


@dataclass
class DelayFit:
    delay_sensor_vs_joint_s: float
    delay_online_effective_s: float
    delay_ci95_s: float
    delay_per_axis_s: np.ndarray
    delay_objective_curvature: float
    delay_hit_search_boundary: bool
    phase_linear: bool


def _line_objective(W_meas: np.ndarray, W_pay: np.ndarray, freqs: np.ndarray, delay: float) -> float:
    acc = 0.0
    for k, f in enumerate(freqs):
        phase = np.exp(-1j * 2.0 * np.pi * float(f) * float(delay))
        acc += float(np.sum(np.abs(W_meas[k] - phase * W_pay[k]) ** 2))
    return acc


def fit_delay_on_lines(
    W_meas: np.ndarray,
    W_payload: np.ndarray,
    freqs_hz: np.ndarray,
    *,
    grid_s: tuple[float, float] = (-0.1, 0.1),
    step_s: float = 0.001,
) -> DelayFit:
    freqs = np.asarray(freqs_hz, dtype=float).reshape(-1)
    delays = np.arange(grid_s[0], grid_s[1] + 0.5 * step_s, step_s)
    costs = np.array([_line_objective(W_meas, W_payload, freqs, d) for d in delays])
    i0 = int(np.argmin(costs))
    d_star = float(delays[i0])
    j_min = float(costs[i0])
    hit = i0 <= 1 or i0 >= len(delays) - 2
    # Curvature from quadratic near min. CI uses residual-scaled LS:
    # J is already SSE over Re/Im of each FFT bin, so σ² ≈ J_min / (N-1)
    # and var(Δt) = 2σ² / J'' (not the unit-variance 2/J'' that blew ci95
    # to hundreds of ms on a shallow but real bowl).
    lo = max(0, i0 - 8)
    hi = min(len(delays), i0 + 9)
    x = delays[lo:hi]
    y = costs[lo:hi]
    coef = np.polyfit(x, y, 2) if x.size >= 3 else np.array([0.0, 0.0, j_min])
    curv = float(2.0 * coef[0])
    n_resid = max(int(2 * np.asarray(W_meas).size), 2)
    sigma2 = j_min / float(n_resid - 1)
    if curv > 0.0 and np.isfinite(sigma2) and sigma2 >= 0.0:
        ci = float(np.sqrt(2.0 * sigma2 / curv))
    else:
        ci = float("inf")
    # per-axis 1-D search on force channels
    per = []
    for ax in range(3):
        cax = []
        for d in delays:
            acc = 0.0
            for k, f in enumerate(freqs):
                phase = np.exp(-1j * 2.0 * np.pi * float(f) * float(d))
                acc += abs(W_meas[k, ax] - phase * W_payload[k, ax]) ** 2
            cax.append(acc)
        per.append(float(delays[int(np.argmin(cax))]))
    per_ax = np.asarray(per, dtype=float)
    # phase linearity: arg(W_meas / W_pay) vs f
    ph = []
    ff = []
    for k, f in enumerate(freqs):
        ratio = np.sum(W_meas[k, :3] * np.conj(W_payload[k, :3]))
        if abs(ratio) > 1e-12:
            ph.append(float(np.angle(ratio)))
            ff.append(float(f))
    linear = True
    if len(ff) >= 3:
        A = np.vstack([ff, np.ones(len(ff))]).T
        slope, _ = np.linalg.lstsq(A, np.unwrap(ph), rcond=None)[0]
        pred = slope * np.asarray(ff)
        resid = np.unwrap(ph) - pred
        linear = float(np.std(resid)) < 0.35
    return DelayFit(
        delay_sensor_vs_joint_s=d_star,
        delay_online_effective_s=d_star,
        delay_ci95_s=1.96 * ci if np.isfinite(ci) else float("inf"),
        delay_per_axis_s=per_ax,
        delay_objective_curvature=curv,
        delay_hit_search_boundary=hit,
        phase_linear=linear,
    )


def delay_rejected(fit: DelayFit, *, dt_s: float = 0.005) -> bool:
    if fit.delay_hit_search_boundary:
        return True
    if not fit.phase_linear:
        return True
    if fit.delay_ci95_s > 0.02:
        return True
    if float(np.max(np.abs(fit.delay_per_axis_s - fit.delay_sensor_vs_joint_s))) > dt_s:
        return True
    return False


@dataclass
class InertiaFit:
    adopted: bool
    I_voigt: np.ndarray
    snr_I: float
    triangle_ok: bool
    reason: str = ""
    moment_dynamic_valid: bool = False
    unmodeled_bound_nm: float = 0.0


def inertia_moment_residual(
    wrench_L: np.ndarray,
    *,
    mass_kg: float,
    h_L: np.ndarray,
    a_L: np.ndarray,
    g_L: np.ndarray,
    omega_L: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """τ_meas − (gravity + bias + −a×h). Leaves Iα + ω×Iω for the I fit."""

    pred = payload_wrench_mhb(
        mass_kg=mass_kg,
        h_L=h_L,
        a_L=a_L,
        g_L=g_L,
        omega_L=omega_L,
        alpha_L=np.zeros(3),
        bias=bias,
    )
    return np.asarray(wrench_L, dtype=float).reshape(6)[3:6] - pred[3:6]


def fit_inertia_moments(
    alpha: np.ndarray,
    omega: np.ndarray,
    tau_res: np.ndarray,
    *,
    mass_kg: float,
    r_max_m: float,
    sigma_M: float,
    holdout_tau: np.ndarray | None = None,
    holdout_pred_mhb: np.ndarray | None = None,
    holdout_pred_I: np.ndarray | None = None,
    snr_min: float = 3.0,
) -> InertiaFit:
    W = []
    y = []
    for al, w, tau in zip(alpha, omega, tau_res, strict=True):
        W.append(inertia_op(al) + skew(w) @ inertia_op(w))
        y.append(np.asarray(tau, dtype=float).reshape(3))
    W = np.vstack(W)
    y = np.concatenate(y)
    Icol, *_ = np.linalg.lstsq(W, y, rcond=None)
    pred = W @ Icol
    snr = float(np.sqrt(np.mean(pred**2)) / max(float(sigma_M), 1e-9))
    tri = inertia_triangle_ok(float(Icol[0]), float(Icol[1]), float(Icol[2]))
    bound = unmodeled_inertia_torque_bound(mass_kg, r_max_m, np.mean(np.abs(alpha), axis=0), np.mean(np.abs(omega), axis=0))
    adopted = snr >= snr_min and tri
    if holdout_tau is not None and holdout_pred_mhb is not None and holdout_pred_I is not None:
        e0 = np.sqrt(np.mean((holdout_tau - holdout_pred_mhb) ** 2))
        e1 = np.sqrt(np.mean((holdout_tau - holdout_pred_I) ** 2))
        adopted = adopted and (e1 < e0)
    reason = ""
    if not adopted:
        reason = "snr" if snr < snr_min else ("triangle" if not tri else "holdout")
    return InertiaFit(
        adopted=adopted,
        I_voigt=np.asarray(Icol, dtype=float),
        snr_I=snr,
        triangle_ok=tri,
        reason=reason,
        moment_dynamic_valid=adopted,
        unmodeled_bound_nm=float(bound),
    )


def fft_lines(t: np.ndarray, x: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
    """Complex amplitude of ``x`` (N, C) at requested frequencies."""
    t = np.asarray(t, dtype=float)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[0] != t.size:
        x = x.T if x.shape[1] == t.size else x
    dt = float(np.median(np.diff(t)))
    out = []
    for f in freqs_hz:
        kern = np.exp(-2j * np.pi * float(f) * t)
        out.append((2.0 / t.size) * (kern @ x))
    return np.asarray(out)
