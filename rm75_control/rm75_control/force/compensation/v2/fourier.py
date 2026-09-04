"""Periodic multi-sine TCP / link_7 twists with a/jerk constraints and SE(3) closure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.force.compensation.v2.se3 import integrate_se3, se3_closure_error


@dataclass
class FourierSpec:
    f0_hz: float = 0.2
    harmonics: tuple[int, ...] = (2, 3, 4, 5, 6, 7)
    n_warmup: int = 2
    n_measure: int = 10
    n_cooldown: int = 1
    ramp_cycles: float = 0.5
    dt: float = 0.005
    x_max_m: float = 0.030
    v_max_m_s: float = 0.10
    a_max_m_s2: float = 2.5
    j_max_m_s3: float = 40.0
    w_max_rad_s: float = 0.90
    alpha_max_rad_s2: float = 10.0
    j_ang_max: float = 80.0
    ang_max_rad: float = 0.30
    eps_p_m: float = 1.0e-4
    eps_R_rad: float = float(np.deg2rad(0.05))


def _raised_cosine(t: np.ndarray, t_ramp: float, t_end: float) -> np.ndarray:
    r = np.ones_like(t)
    if t_ramp <= 0.0:
        return r
    up = t < t_ramp
    r[up] = 0.5 * (1.0 - np.cos(np.pi * t[up] / t_ramp))
    down = t > (t_end - t_ramp)
    tau = (t_end - t[down]) / t_ramp
    r[down] = 0.5 * (1.0 - np.cos(np.pi * np.clip(tau, 0.0, 1.0)))
    return r


def design_axis_amps(
    spec: FourierSpec,
    *,
    v_peak: float,
    rotational: bool,
) -> np.ndarray:
    n = np.asarray(spec.harmonics, dtype=float)
    f = n * spec.f0_hz
    # Equal-amplitude start, then scale to meet all bounds.
    A = np.ones_like(f) / max(len(f), 1)
    A *= float(v_peak)
    omega = 2.0 * np.pi * f
    x_sum = float(np.sum(np.abs(A) / np.maximum(omega, 1e-9)))
    a_sum = float(np.sum(omega * np.abs(A)))
    j_sum = float(np.sum(omega**2 * np.abs(A)))
    if rotational:
        limits = (spec.w_max_rad_s, spec.alpha_max_rad_s2, spec.j_ang_max)
        x_lim = float(spec.ang_max_rad)
    else:
        limits = (spec.v_max_m_s, spec.a_max_m_s2, spec.j_max_m_s3)
        x_lim = spec.x_max_m
    scale = 1.0
    if x_sum > x_lim:
        scale = min(scale, x_lim / x_sum)
    if a_sum > limits[1]:
        scale = min(scale, limits[1] / max(a_sum, 1e-12))
    if j_sum > limits[2]:
        scale = min(scale, limits[2] / max(j_sum, 1e-12))
    if float(np.sum(np.abs(A))) > limits[0]:
        scale = min(scale, limits[0] / max(float(np.sum(np.abs(A))), 1e-12))
    return A * scale


def apply_peak_and_bounds(
    spec: FourierSpec,
    t: np.ndarray,
    v: np.ndarray,
    *,
    peak: float,
    rotational: bool,
) -> np.ndarray:
    """Hit the requested peak on the synthesized wave, then respect x/a/j caps."""

    y = np.asarray(v, dtype=float).copy()
    cur = float(np.max(np.abs(y)))
    if cur > 1e-12:
        y *= float(peak) / cur
    dt = float(spec.dt if spec.dt > 0.0 else np.median(np.diff(t)))
    acc = np.gradient(y, dt)
    jerk = np.gradient(acc, dt)
    disp = np.cumsum(y) * dt
    disp = disp - float(np.mean(disp))
    if rotational:
        caps = (spec.ang_max_rad, spec.w_max_rad_s, spec.alpha_max_rad_s2, spec.j_ang_max)
    else:
        caps = (spec.x_max_m, spec.v_max_m_s, spec.a_max_m_s2, spec.j_max_m_s3)
    scale = 1.0
    for arr, lim in (
        (disp, caps[0]),
        (y, caps[1]),
        (acc, caps[2]),
        (jerk, caps[3]),
    ):
        mx = float(np.max(np.abs(arr)))
        if mx > float(lim) > 0.0:
            scale = min(scale, float(lim) / mx)
    return y * scale


def synthesize_axis(
    spec: FourierSpec,
    amps: np.ndarray,
    *,
    phases: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_cyc = spec.n_warmup + spec.n_measure + spec.n_cooldown
    T_p = 1.0 / spec.f0_hz
    T = n_cyc * T_p
    t = np.arange(0.0, T, spec.dt)
    if t.size == 0 or t[-1] < T - spec.dt:
        t = np.append(t, T)
    n = np.asarray(spec.harmonics, dtype=float)
    f = n * spec.f0_hz
    if phases is None:
        # Schroeder phases for low crest factor, zero mean period.
        phases = np.pi * n * (n + 1) / max(len(n), 1)
    t_ramp = spec.ramp_cycles * T_p
    env = _raised_cosine(t, t_ramp, float(t[-1]))
    v = np.zeros_like(t)
    for A, fi, ph in zip(amps, f, phases, strict=False):
        v += A * np.sin(2.0 * np.pi * fi * t + ph)
    v *= env
    return t, v


def measure_mask(spec: FourierSpec, t: np.ndarray) -> np.ndarray:
    T_p = 1.0 / spec.f0_hz
    t0 = spec.n_warmup * T_p
    t1 = (spec.n_warmup + spec.n_measure) * T_p
    return (t >= t0) & (t < t1)


def period_net_displacement(t: np.ndarray, v: np.ndarray, T_p: float) -> float:
    m = (t >= 0.0) & (t < T_p)
    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(v[m], t[m])) if np.any(m) else 0.0


def full_trajectory_closure(
    t: np.ndarray,
    twist_L: np.ndarray,
    *,
    spec: FourierSpec,
) -> tuple[float, float]:
    p = np.zeros(3)
    R = np.eye(3)
    p0, R0 = p.copy(), R.copy()
    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        p, R = integrate_se3(p, R, twist_L[i, :3], twist_L[i, 3:6], dt)
    return se3_closure_error(p0, R0, p, R)


def axis_twist_L(
    spec: FourierSpec,
    axis: int,
    *,
    peak: float,
    rotational: bool,
    phases: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amps = design_axis_amps(spec, v_peak=peak, rotational=rotational)
    t, s = synthesize_axis(spec, amps, phases=phases)
    s = apply_peak_and_bounds(spec, t, s, peak=peak, rotational=rotational)
    tw = np.zeros((t.size, 6), dtype=float)
    tw[:, axis] = s
    return t, tw, amps
