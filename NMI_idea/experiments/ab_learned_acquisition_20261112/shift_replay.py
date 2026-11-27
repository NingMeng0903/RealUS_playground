#!/usr/bin/env python3
"""Synthetic structural-OOD replay for persistent spatial offsets and transient lag.

This is deliberately a small, diagnostic replay.  It reuses the frozen knee
world's registered projection and time convention, but it does not read any
patient data or any held-out truth while fitting the population prior.  Each
frame contains exactly one visible, correctly associated mask centroid and
the observation time is the physical time (there is no scan/travel delay).

The four comparisons are:

``static_offset``
    A persistent registered lateral offset updated online (no transient).
``fixed_lag``
    One Kalman state with the nominal 0.72 s lag and a persistent spatial
    offset.
``bank_frozen``
    A small lag bank with an observation-only learned population prior.  The
    posterior over lag hypotheses is frozen for the test record.
``bank_adaptive``
    The same bank and the same prior, with causal predictive-likelihood
    updating of the lag weights.

The generated records, seeds, estimator settings, paired case rows, JSON
summary, and an English Plotly 3-D example are written below ``shift/``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# Reuse the frozen world and its baked geometry.  The import path is explicit
# so this file remains runnable from the experiment directory or any cwd.
HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "ab_knee_latent_rl_20261112"
sys.path.insert(0, str(FROZEN))
from knee_world import KneeWorld  # noqa: E402


OUT_DIR = HERE / "shift"
RESULTS_JSON = OUT_DIR / "SHIFT_REPLAY.json"
PRIOR_JSON = OUT_DIR / "TRAINING_PRIOR.json"
EXAMPLE_HTML = OUT_DIR / "SHIFT_REPLAY_3D.html"

SEED = 20261112
TRAIN_BASE = 48191357
TEST_BASES = {
    "nominal": 91412639,
    "spatial_angle_ood": 28492173,
    "two_timescale_asymmetric_ood": 72839161,
}
SPLIT_CODES = {"nominal": 11, "spatial_angle_ood": 23, "two_timescale_asymmetric_ood": 37}

# The world has 28 stations and a central view at index 3.  These six trunk
# sites span the z design coefficient without branch-label ambiguity.  A single
# site is observed per frame, cycling through this list.
SITES = (6, 9, 12, 15, 18, 21)
VIEW = 3
DT = 0.20
N_FRAMES = 32
N_TRAIN = 72
N_TEST = 24
SCORE_TIMES = (1.6, 3.0, 4.4)
HORIZONS = (0.0, 0.5, 1.0)
NOMINAL_TAU = 0.72
TAU_BANK = np.array([0.38, 0.58, 0.82, 1.12], dtype=float)
MEAS_SD = 0.46
MEAS_VAR_FLOOR = 0.10**2
STATE_VAR_FLOOR = 0.14**2
CI_Z = 1.959963984540054


@dataclass(frozen=True)
class Mask:
    """One correctly associated, visible mask centroid."""

    t: float
    q_obs: float
    velocity_obs: float
    site: int
    view: int
    lateral_mm: float
    depth_mm: float


@dataclass(frozen=True)
class Record:
    split: str
    index: int
    seed: int
    masks: tuple[Mask, ...]
    q_nominal: np.ndarray
    velocity_nominal: np.ndarray
    q_true: np.ndarray
    z_true: np.ndarray
    w_site_true: np.ndarray
    angle_bias_deg: float
    phase: float
    z0: float
    point_truth: np.ndarray


@dataclass
class GaussianState:
    mean: np.ndarray
    cov: np.ndarray
    t: float | None = None


def _json_value(value: Any) -> Any:
    """Convert numpy scalars/arrays recursively for deterministic JSON."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def _normal_pdf(residual: float, variance: float) -> float:
    variance = max(float(variance), MEAS_VAR_FLOOR)
    return math.exp(-0.5 * float(residual) ** 2 / variance) / math.sqrt(2.0 * math.pi * variance)


def _clip_angle(q: float) -> float:
    return float(np.clip(q, 11.02, 18.98))


def nominal_pose(t: np.ndarray | float, phase: float) -> tuple[np.ndarray, np.ndarray]:
    """Public nominal pose schedule; its reversals drive the OOD dynamics."""

    t_arr = np.asarray(t, dtype=float)
    omega = 2.0 * np.pi / 4.8
    q = 15.0 + 2.15 * np.sin(omega * t_arr + phase)
    velocity = 2.15 * omega * np.cos(omega * t_arr + phase)
    return np.clip(q, 11.02, 18.98), velocity


def _first_order_continuous(time: np.ndarray | float, phase: float, tau: float, z0: float, *, gain: float = 0.83) -> np.ndarray:
    """Continuous first-order response to the declared sinusoidal pose drive."""

    t = np.asarray(time, dtype=float)
    lam = 1.0 / float(tau)
    omega = 2.0 * np.pi / 4.8
    velocity_amplitude = 2.15 * omega
    theta = omega * t + float(phase)
    particular = gain * velocity_amplitude * (lam * np.cos(theta) + omega * np.sin(theta)) / (lam * lam + omega * omega)
    particular_zero = gain * velocity_amplitude * (lam * np.cos(float(phase)) + omega * np.sin(float(phase))) / (lam * lam + omega * omega)
    return particular + (float(z0) - particular_zero) * np.exp(-lam * t)


def _latent_response(time: np.ndarray, phase: float, split: str, rng: np.random.Generator | None = None, *, z0: float | None = None) -> tuple[np.ndarray, float]:
    """Generate nominal or structural-OOD transient z without estimator access."""

    if z0 is None:
        if rng is None:
            raise ValueError("rng is required when z0 is not supplied")
        z0 = float(rng.normal(0.0, 1.0))
    z0 = float(z0)
    if split in ("nominal", "spatial_angle_ood"):
        return _first_order_continuous(time, phase, NOMINAL_TAU, z0, gain=0.83), z0

    # Held-out family: a fast + slow mixture, with the slow component changing
    # after the first velocity reversal.  Neither component is a single member
    # of the estimator's compact bank; this is intentionally structural OOD.
    fast = _first_order_continuous(time, phase, 0.27, 0.65 * z0, gain=0.95)
    slow = _first_order_continuous(time, phase, 1.65, 0.35 * z0, gain=0.83)
    omega = 2.0 * np.pi / 4.8
    first_reversal_t = float(((0.5 * np.pi - float(phase)) % np.pi) / omega)
    slow = np.asarray(slow, dtype=float).copy()
    slow[np.asarray(time) >= first_reversal_t] *= 1.33
    return 0.62 * fast + 0.38 * slow, z0


def _site_norm(site: int, world: KneeWorld) -> float:
    s = float(world.scan.s[site])
    lo = float(np.min(world.scan.s[list(SITES)]))
    hi = float(np.max(world.scan.s[list(SITES)]))
    return float(2.0 * (s - lo) / max(hi - lo, 1.0) - 1.0)


def _true_static_pattern(split: str, rng: np.random.Generator, world: KneeWorld) -> np.ndarray:
    """Persistent site offsets; the OOD pattern is not told to estimators."""

    norm = np.array([_site_norm(s, world) for s in SITES], dtype=float)
    if split == "spatial_angle_ood":
        # Wider subject displacement plus a persistent spatial ripple.  The
        # estimator only has a low-order registered basis [1, site_norm].
        w0 = float(rng.normal(0.0, 3.5))
        w1 = float(rng.normal(0.0, 1.15))
        ripple = 1.15 * np.sin(2.4 * np.pi * (norm + 1.0) / 2.0 + rng.uniform(-0.8, 0.8))
        return w0 + w1 * norm + ripple
    w0 = float(rng.normal(0.0, 1.45))
    w1 = float(rng.normal(0.0, 0.42))
    return w0 + w1 * norm + rng.normal(0.0, 0.14, size=len(SITES))


def _measurement_geometry(world: KneeWorld, q_obs: float, site: int, view: int) -> tuple[float, float, float]:
    base, rows = world.projection(q_obs, site, view)
    # Main/trunk lumen is label 0; only its lateral registered coordinate is
    # used.  The z coefficient comes from the baked design row.
    base_x = float(base[0, 0])
    z_coeff = float(rows[0, 0, 2])
    return base_x, z_coeff, float(base[0, 1])


def _truth_point(world: KneeWorld, q_true: float, z: float, w_site: float, site: int) -> np.ndarray:
    base, design = world.world_design(q_true, [(site, 0)])
    params = np.array([[w_site, 0.0, z]], dtype=float)
    return np.asarray(base[0] + design[0] @ params[0], dtype=float)


def _record(split: str, index: int, seed: int, world: KneeWorld, *, noisy: bool = True) -> Record:
    # Do not use Python's process-randomized hash here: a rerun must reproduce
    # every mask and every paired delta byte-for-byte.
    rng = np.random.default_rng(np.random.SeedSequence([SEED, seed, index, SPLIT_CODES[split]]))
    times = DT * np.arange(N_FRAMES, dtype=float)
    phase = float(rng.uniform(-0.45, 0.45))
    q_nominal, velocity = nominal_pose(times, phase)
    angle_bias = float(rng.normal(0.0, 0.78) if split == "spatial_angle_ood" else rng.normal(0.0, 0.08))
    q_true = np.clip(q_nominal + angle_bias, 11.02, 18.98)
    # Small pose noise is public; the structural angle shift is hidden.
    q_obs = q_nominal + (rng.normal(0.0, 0.045, len(times)) if noisy else 0.0)
    vel_obs = velocity + (rng.normal(0.0, 0.10, len(times)) if noisy else 0.0)
    z, z0 = _latent_response(times, phase, split, rng)
    static = _true_static_pattern(split, rng, world)

    masks: list[Mask] = []
    for i, t in enumerate(times):
        site = SITES[i % len(SITES)]
        local = int(np.flatnonzero(np.asarray(SITES) == site)[0])
        view = VIEW
        base, rows = world.projection(float(q_true[i]), site, view)
        # One visible correctly-associated mask, with modest independent
        # lateral/depth segmentation noise.  There are no misses or clutter.
        params = np.array([static[local], 0.0, z[i]], dtype=float)
        center = np.asarray(base[0] + rows[0] @ params, dtype=float)
        if noisy:
            center += rng.normal(0.0, [0.32, 0.22])
        masks.append(Mask(float(t), float(q_obs[i]), float(vel_obs[i]), site, view, float(center[0]), float(center[1])))

    point_site = SITES[2]
    point_truth = np.stack([_truth_point(world, float(q_true[i]), float(z[i]), float(static[np.flatnonzero(np.asarray(SITES) == point_site)[0]]), point_site) for i in range(len(times))])
    return Record(split, int(index), int(seed), tuple(masks), q_nominal, velocity, q_true, z, static, angle_bias, phase, z0, point_truth)


def _obs_design(world: KneeWorld, mask: Mask) -> tuple[float, np.ndarray, float]:
    base_x, z_coeff, _ = _measurement_geometry(world, mask.q_obs, mask.site, mask.view)
    h = np.array([1.0, _site_norm(mask.site, world), z_coeff], dtype=float)
    return float(mask.lateral_mm - base_x), h, float(base_x)


def _kalman_predict(state: GaussianState, t: float, velocity: float, tau: float, *, gain: float = 0.83) -> GaussianState:
    if state.t is None:
        return GaussianState(state.mean.copy(), state.cov.copy(), float(t))
    dt = float(t - state.t)
    if dt < -1.0e-8:
        raise ValueError("time must increase")
    dt = max(dt, 0.0)
    out = GaussianState(state.mean.copy(), state.cov.copy(), float(t))
    if dt <= 1.0e-12:
        return out
    a = math.exp(-dt / float(tau))
    f = np.eye(3)
    f[2, 2] = a
    out.mean[2] = a * out.mean[2] + gain * float(tau) * (1.0 - a) * float(velocity)
    process = np.diag([0.0, 0.0, max(STATE_VAR_FLOOR, 0.34**2 * (1.0 - a * a))])
    out.cov = f @ out.cov @ f.T + process
    out.cov = _regularise(out.cov, len(out.mean))
    return out


def _regularise(cov: np.ndarray, n: int) -> np.ndarray:
    cov = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    diag = np.maximum(np.diag(cov), STATE_VAR_FLOOR)
    cov[np.diag_indices(n)] = diag
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, STATE_VAR_FLOOR * 0.02)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _kalman_update(state: GaussianState, residual: float, h: np.ndarray) -> tuple[GaussianState, float]:
    h = np.asarray(h, dtype=float)
    variance = max(MEAS_VAR_FLOOR, float(h @ state.cov @ h) + MEAS_SD**2)
    innovation = float(residual - h @ state.mean)
    gain = state.cov @ h / variance
    out_mean = state.mean + gain * innovation
    out_cov = state.cov - np.outer(gain, h @ state.cov)
    out = GaussianState(out_mean, _regularise(out_cov, len(out_mean)), state.t)
    return out, _normal_pdf(innovation, variance)


def _static_update(state: GaussianState, residual: float, h_static: np.ndarray) -> tuple[GaussianState, float]:
    h = np.asarray(h_static, dtype=float)
    variance = max(MEAS_VAR_FLOOR, float(h @ state.cov @ h) + MEAS_SD**2)
    innovation = float(residual - h @ state.mean)
    gain = state.cov @ h / variance
    out_mean = state.mean + gain * innovation
    out_cov = state.cov - np.outer(gain, h @ state.cov)
    out = GaussianState(out_mean, _regularise(out_cov, len(out_mean)), state.t)
    return out, _normal_pdf(innovation, variance)


def _predict_state(state: GaussianState, world: KneeWorld, mask: Mask, t: float, velocity: float, tau: float, *, propagate_z: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Predict the 3-D world point and its diagonal uncertainty."""

    if propagate_z:
        predicted = _kalman_predict(state, t, velocity, tau)
    else:
        # Static-offset control keeps z fixed at zero even when forecasting;
        # only the public pose continuation advances the world point.
        predicted = GaussianState(state.mean.copy(), state.cov.copy(), float(t))
        predicted.mean[2] = 0.0
        predicted.cov[2, :] = 0.0
        predicted.cov[:, 2] = 0.0
        predicted.cov[2, 2] = STATE_VAR_FLOOR
    q = _clip_angle(mask.q_obs + mask.velocity_obs * (t - mask.t))
    xyz, design = world.world_design(q, [(SITES[2], 0)])
    # Point uses the persistent low-order registered field evaluated at its
    # target station; z is shared.  Geometry uncertainty is represented by a
    # conservative fixed pose floor in the output interval.
    norm = _site_norm(SITES[2], world)
    value = xyz[0] + design[0] @ np.array([predicted.mean[0] + norm * predicted.mean[1], 0.0, predicted.mean[2]])
    # Explicit point derivatives keep the vector shape independent of world
    # implementation details.
    d_w = np.asarray(design[0][:, 0], dtype=float) + norm * np.asarray(design[0][:, 0], dtype=float) * 0.0
    d_z = np.asarray(design[0][:, 2], dtype=float)
    # w0 and w1 both use the baked w design column at the point.
    J = np.column_stack((d_w, norm * d_w, d_z))
    cov_point = J @ predicted.cov @ J.T + np.eye(3) * (0.25**2)
    return np.asarray(value, dtype=float), np.sqrt(np.maximum(np.diag(cov_point), 0.25**2))


class StaticOffsetEstimator:
    def __init__(self, world: KneeWorld):
        self.world = world
        self.state = GaussianState(np.zeros(2), np.diag([4.0**2, 1.5**2]), None)
        self.last_mask: Mask | None = None

    def observe(self, mask: Mask) -> None:
        residual, h, _ = _obs_design(self.world, mask)
        self.state, _ = _static_update(self.state, residual, h[:2])
        self.state.t = mask.t
        self.last_mask = mask

    def predict(self, t: float, velocity: float) -> tuple[np.ndarray, np.ndarray]:
        assert self.last_mask is not None
        state3 = GaussianState(np.array([self.state.mean[0], self.state.mean[1], 0.0]), np.pad(self.state.cov, ((0, 1), (0, 1))), self.state.t)
        return _predict_state(state3, self.world, self.last_mask, t, velocity, NOMINAL_TAU, propagate_z=False)


class FixedLagEstimator:
    def __init__(self, world: KneeWorld):
        self.world = world
        self.state = GaussianState(np.zeros(3), np.diag([4.0**2, 1.5**2, 2.5**2]), None)
        self.last_mask: Mask | None = None

    def observe(self, mask: Mask) -> None:
        if self.state.t is None:
            self.state.t = mask.t
        else:
            self.state = _kalman_predict(self.state, mask.t, mask.velocity_obs, NOMINAL_TAU)
        residual, h, _ = _obs_design(self.world, mask)
        self.state, _ = _kalman_update(self.state, residual, h)
        self.state.t = mask.t
        self.last_mask = mask

    def predict(self, t: float, velocity: float) -> tuple[np.ndarray, np.ndarray]:
        assert self.last_mask is not None
        return _predict_state(self.state, self.world, self.last_mask, t, velocity, NOMINAL_TAU)


class LagBankEstimator:
    def __init__(self, world: KneeWorld, prior: Mapping[str, Any], *, adaptive: bool):
        self.world = world
        self.taus = np.asarray(prior["tau_axis"], dtype=float)
        self.prior_weights = np.asarray(prior["tau_weights"], dtype=float)
        self.weights = self.prior_weights.copy()
        self.adaptive = bool(adaptive)
        means = np.asarray(prior["state_mean"], dtype=float)
        cov = np.asarray(prior["state_cov"], dtype=float)
        self.states = [GaussianState(means.copy(), cov.copy(), None) for _ in self.taus]
        self.last_mask: Mask | None = None

    def observe(self, mask: Mask) -> None:
        residual, h, _ = _obs_design(self.world, mask)
        predictive = []
        for i, tau in enumerate(self.taus):
            state = self.states[i]
            if state.t is not None:
                state = _kalman_predict(state, mask.t, mask.velocity_obs, float(tau))
            else:
                state.t = mask.t
            state, likelihood = _kalman_update(state, residual, h)
            state.t = mask.t
            self.states[i] = state
            predictive.append(likelihood)
        if self.adaptive:
            logw = np.log(np.maximum(self.weights, 1.0e-30)) + np.log(np.maximum(predictive, 1.0e-300))
            logw -= np.max(logw)
            self.weights = np.exp(logw)
            self.weights /= np.sum(self.weights)
        self.last_mask = mask

    def _mixture_state(self, t: float, velocity: float) -> tuple[np.ndarray, np.ndarray]:
        assert self.last_mask is not None
        means, covs = [], []
        for state, tau in zip(self.states, self.taus):
            predicted = _kalman_predict(state, t, velocity, float(tau))
            means.append(predicted.mean)
            covs.append(predicted.cov)
        means_arr = np.stack(means)
        mean = np.sum(self.weights[:, None] * means_arr, axis=0)
        cov = np.zeros((3, 3), dtype=float)
        for weight, item_mean, item_cov in zip(self.weights, means_arr, covs):
            delta = item_mean - mean
            cov += weight * (item_cov + np.outer(delta, delta))
        return mean, _regularise(cov, 3)

    def predict(self, t: float, velocity: float) -> tuple[np.ndarray, np.ndarray]:
        assert self.last_mask is not None
        mean, cov = self._mixture_state(t, velocity)
        state = GaussianState(mean, cov, t)
        return _predict_state(state, self.world, self.last_mask, t, velocity, float(np.sum(self.weights * self.taus)))


def _fit_static_and_tau(record: Record, world: KneeWorld) -> tuple[np.ndarray, float]:
    """Observation-only training fit; no record truth fields are consulted."""

    residuals, static_rows, z_rows, times, velocities = [], [], [], [], []
    for mask in record.masks:
        residual, h, _ = _obs_design(world, mask)
        residuals.append(residual)
        static_rows.append(h[:2])
        z_rows.append(h[2])
        times.append(mask.t)
        velocities.append(mask.velocity_obs)
    residuals_arr = np.asarray(residuals, dtype=float)
    static_rows_arr = np.asarray(static_rows, dtype=float)
    z_rows_arr = np.asarray(z_rows, dtype=float)
    times_arr = np.asarray(times, dtype=float)
    velocities_arr = np.asarray(velocities, dtype=float)

    # Fit each candidate lag by integrating the *observed* velocity drive and
    # regressing the remaining mask residual onto the decaying initial state.
    # This is an observation-only state-space fit: no z truth, generator phase,
    # or held-out record fields enter it.  A temporal autocorrelation shortcut
    # would be confounded by the cycling site sensitivities and periodic drive.
    fit_axis = np.linspace(0.25, 1.80, 32)
    best = None
    for tau in fit_axis:
        drive = np.zeros(len(times_arr), dtype=float)
        for i in range(1, len(drive)):
            a = math.exp(-(times_arr[i] - times_arr[i - 1]) / float(tau))
            drive[i] = a * drive[i - 1] + 0.83 * float(tau) * (1.0 - a) * velocities_arr[i - 1]
        x = np.column_stack((static_rows_arr, z_rows_arr * np.exp(-times_arr / float(tau))))
        target = residuals_arr - z_rows_arr * drive
        ridge = np.diag([0.05, 0.05, 0.10])
        params = np.linalg.solve(x.T @ x + ridge, x.T @ target)
        residual = target - x @ params
        # The fixed noise scale is shared across all training records; ranking
        # candidates by SSE is therefore equivalent to predictive likelihood.
        sse = float(np.sum(residual * residual))
        if best is None or sse < best[0]:
            best = (sse, params, float(tau))
    assert best is not None
    return np.asarray(best[1], dtype=float), float(best[2])


def fit_population_prior(records: Sequence[Record], world: KneeWorld) -> dict[str, Any]:
    fits, tau_hats = [], []
    for record in records:
        state, tau = _fit_static_and_tau(record, world)
        fits.append(state)
        tau_hats.append(tau)
    fits_arr = np.asarray(fits, dtype=float)
    tau_arr = np.asarray(tau_hats, dtype=float)
    # Population prior is intentionally shrunk only modestly: it should be
    # useful on nominal data while retaining uncertainty on OOD subjects.
    state_mean = np.mean(fits_arr, axis=0)
    state_cov = np.cov(fits_arr.T, ddof=1) + np.diag([0.30**2, 0.18**2, 0.30**2])
    state_cov = _regularise(state_cov, 3)
    tau_weights = np.exp(-0.5 * ((TAU_BANK - np.mean(tau_arr)) / max(np.std(tau_arr), 0.25)) ** 2)
    tau_weights /= np.sum(tau_weights)
    return {
        "fit_source": "independent synthetic training records; masks and public poses only; no held-out truth",
        "n_records": len(records),
        "tau_axis": TAU_BANK.tolist(),
        "tau_weights": tau_weights.tolist(),
        "state_mean": state_mean.tolist(),
        "state_cov": state_cov.tolist(),
        "training_tau_hat_mean_sd": [float(np.mean(tau_arr)), float(np.std(tau_arr, ddof=1))],
        "training_state_fit_mean": np.mean(fits_arr, axis=0).tolist(),
        "training_state_fit_sd": np.std(fits_arr, axis=0, ddof=1).tolist(),
    }


def _make_estimator(name: str, world: KneeWorld, prior: Mapping[str, Any]):
    if name == "static_offset":
        return StaticOffsetEstimator(world)
    if name == "fixed_lag":
        return FixedLagEstimator(world)
    if name == "bank_frozen":
        return LagBankEstimator(world, prior, adaptive=False)
    if name == "bank_adaptive":
        return LagBankEstimator(world, prior, adaptive=True)
    raise KeyError(name)


def _truth_at(record: Record, time: float, world: KneeWorld) -> tuple[np.ndarray, float]:
    """Evaluate generator-consistent truth at the exact requested clock."""

    query = float(max(0.0, time))
    q_nominal, _ = nominal_pose(query, record.phase)
    q_true = _clip_angle(float(np.asarray(q_nominal)) + record.angle_bias_deg)
    z_query, _ = _latent_response(np.asarray([query]), record.phase, record.split, z0=record.z0)
    site = SITES[2]
    local = 2
    point = _truth_point(world, q_true, float(z_query[0]), float(record.w_site_true[local]), site)
    return point, float(z_query[0])


def _predicted_z(estimator: Any, time: float, velocity: float) -> tuple[float, float]:
    """Return horizon-propagated z mean/sd from the estimator's causal state."""

    if isinstance(estimator, StaticOffsetEstimator):
        return 0.0, math.sqrt(STATE_VAR_FLOOR)
    if isinstance(estimator, FixedLagEstimator):
        assert estimator.state.t is not None
        state = _kalman_predict(estimator.state, time, velocity, NOMINAL_TAU)
        return float(state.mean[2]), float(np.sqrt(max(STATE_VAR_FLOOR, state.cov[2, 2])))
    assert isinstance(estimator, LagBankEstimator)
    mean, cov = estimator._mixture_state(time, velocity)
    return float(mean[2]), float(np.sqrt(max(STATE_VAR_FLOOR, cov[2, 2])))


def _score_record(record: Record, world: KneeWorld, prior: Mapping[str, Any], method: str) -> dict[str, Any]:
    estimator = _make_estimator(method, world, prior)
    rows = []
    score_set = {round(t, 10) for t in SCORE_TIMES}
    for mask in record.masks:
        estimator.observe(mask)
        if round(mask.t, 10) not in score_set:
            continue
        for horizon in HORIZONS:
            query = round(mask.t + horizon, 10)
            # Causal continuation uses the last public measured pose velocity
            # (the same constant-velocity convention as the frozen belief).
            forecast_velocity = float(mask.velocity_obs)
            pred, sd = estimator.predict(float(query), forecast_velocity)
            truth, z_truth = _truth_at(record, float(query), world)
            error = pred - truth
            z_mean, z_sd = _predicted_z(estimator, float(query), forecast_velocity)
            # 95% scalar z interval; for point coordinates, use a conservative
            # component interval from the propagated mm uncertainty.
            rows.append({
                "time": float(mask.t),
                "horizon": float(horizon),
                "mean_error_mm": float(np.linalg.norm(error)),
                "z_error_mm": float(abs(z_mean - z_truth)),
                "z_covered_95": bool(abs(z_mean - z_truth) <= CI_Z * z_sd),
                "point_coordinate_box_covered_95": bool(np.all(np.abs(error) <= CI_Z * np.asarray(sd))),
                "predicted_point_mm": pred.tolist(),
                "truth_point_mm": truth.tolist(),
                "predicted_point_sd_mm": sd.tolist(),
                "z_truth_mm": float(z_truth),
                "z_pred_mm": float(z_mean),
                "z_sd_mm": float(z_sd),
            })
    # Identify the single event's final bank weights for audit, if applicable.
    extra = {}
    if isinstance(estimator, LagBankEstimator):
        extra["final_tau_weights"] = estimator.weights.tolist()
    return {"split": record.split, "index": record.index, "seed": record.seed, "method": method, "angle_bias_deg": record.angle_bias_deg, "scores": rows, **extra}


def _aggregate(score_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_h: dict[str, list[dict[str, Any]]] = {}
    for row in score_rows:
        for item in row["scores"]:
            by_h.setdefault(str(item["horizon"]), []).append(item)
    out = {}
    for horizon, items in by_h.items():
        err = np.asarray([x["mean_error_mm"] for x in items], dtype=float)
        zerr = np.asarray([x["z_error_mm"] for x in items], dtype=float)
        out[horizon] = {
            "n": int(len(items)),
            "mean_error_mm": float(np.mean(err)),
            "median_error_mm": float(np.median(err)),
            "z_rmse_mm": float(np.sqrt(np.mean(zerr * zerr))),
            "z_mae_mm": float(np.mean(zerr)),
            "z_coverage_95": float(np.mean([x["z_covered_95"] for x in items])),
            "point_coordinate_box_coverage_95": float(np.mean([x["point_coordinate_box_covered_95"] for x in items])),
        }
    return out


def _paired_deltas(all_rows: Sequence[dict[str, Any]], baseline: str = "fixed_lag") -> list[dict[str, Any]]:
    keyed: dict[tuple[str, int, str], dict[str, Any]] = {(r["split"], r["index"], r["method"]): r for r in all_rows}
    result = []
    for split, index, method in sorted({(r["split"], r["index"], r["method"]) for r in all_rows}):
        if method == baseline:
            continue
        a = keyed[(split, index, method)]
        b = keyed[(split, index, baseline)]
        # Keep the score-time cluster explicit; a horizon-only map would
        # silently discard two of the three score times.
        by_key_a = {(x["horizon"], x["time"]): x for x in a["scores"]}
        by_key_b = {(x["horizon"], x["time"]): x for x in b["scores"]}
        for horizon in HORIZONS:
            for score_time in SCORE_TIMES:
                x = by_key_a[(float(horizon), float(score_time))]
                y = by_key_b[(float(horizon), float(score_time))]
                result.append({
                    "split": split,
                    "index": int(index),
                    "method": method,
                    "baseline": baseline,
                    "score_time": float(score_time),
                    "horizon": float(horizon),
                    "delta_mean_error_mm": float(x["mean_error_mm"] - y["mean_error_mm"]),
                    "delta_z_error_mm": float(x["z_error_mm"] - y["z_error_mm"]),
                    "adaptation_wins": bool(x["mean_error_mm"] < y["mean_error_mm"]),
                })
    return result


def _paired_bank_case_clusters(all_rows: Sequence[dict[str, Any]], *, n_bootstrap: int = 2000) -> dict[str, Any]:
    """Case-cluster paired deltas for frozen versus causal bank adaptation."""

    grouped: dict[tuple[str, int, str, float], list[float]] = {}
    for row in all_rows:
        if row["method"] not in ("bank_frozen", "bank_adaptive"):
            continue
        for item in row["scores"]:
            grouped.setdefault((row["split"], row["index"], row["method"], item["horizon"]), []).append(float(item["mean_error_mm"]))
    result: dict[str, Any] = {}
    for split in TEST_BASES:
        result[split] = {}
        for horizon in HORIZONS:
            deltas = []
            for index in range(max(k[1] for k in grouped if k[0] == split) + 1):
                frozen = grouped.get((split, index, "bank_frozen", float(horizon)))
                adaptive = grouped.get((split, index, "bank_adaptive", float(horizon)))
                if frozen is not None and adaptive is not None:
                    deltas.append(float(np.mean(adaptive) - np.mean(frozen)))
            arr = np.asarray(deltas, dtype=float)
            rng = np.random.default_rng(np.random.SeedSequence([SEED, SPLIT_CODES[split], int(round(horizon * 10)), 8801]))
            if len(arr):
                boot = np.mean(arr[rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))], axis=1)
                ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            else:
                ci = [float("nan"), float("nan")]
            result[split][str(horizon)] = {
                "n_cases": int(len(arr)),
                "mean_delta_adaptive_minus_frozen_mm": float(np.mean(arr)) if len(arr) else float("nan"),
                "median_delta_mm": float(np.median(arr)) if len(arr) else float("nan"),
                "wins_adaptive": int(np.sum(arr < 0.0)),
                "losses_adaptive": int(np.sum(arr > 0.0)),
                "bootstrap_95_ci_mm": ci,
                "bootstrap_replicates": int(n_bootstrap),
                "case_deltas_mm": arr.tolist(),
            }
    return result


def _make_html(example: Mapping[str, Any]) -> None:
    from plotly.offline import get_plotlyjs

    traces = example["traces"]
    data = []
    for label, trace in traces.items():
        data.append({
            "type": "scatter3d", "mode": "lines+markers", "name": label,
            "x": trace["x"], "y": trace["y"], "z": trace["z"],
            "line": {"width": 5}, "marker": {"size": 3},
        })
    payload = json.dumps(data, separators=(",", ":"))
    title = escape(str(example.get("title", "Shift replay")))
    notes = escape(str(example.get("notes", "Synthetic structural-OOD diagnostic.")))
    html = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><title>{title}</title>
<script>{get_plotlyjs()}</script></head>
<body style='font-family:Arial,sans-serif;margin:20px'><h2>{title}</h2>
<p>{notes}</p><div id='plot' style='width:100%;height:760px'></div>
<script>const traces={payload}; Plotly.newPlot('plot', traces, {{
  scene: {{xaxis:{{title:'world x (mm)'}},yaxis:{{title:'world y (mm)'}},zaxis:{{title:'world z (mm)'}},aspectmode:'data'}},
  legend:{{orientation:'h'}}, margin:{{l:0,r:0,b:0,t:0}}
}}, {{responsive:true}});</script></body></html>"""
    EXAMPLE_HTML.write_text(html, encoding="utf-8")


def run(*, n_train: int = N_TRAIN, n_test: int = N_TEST) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    world = KneeWorld(noisy_pose=False)
    # Declare the split generators before any output-dependent computation.
    split_definition = {
        "nominal": "Independent records drawn from the nominal one-time-constant tau=0.72 s generator and compact spatial prior.",
        "spatial_angle_ood": "Held-out structural family: wider persistent spatial ripple plus hidden constant angle bias (not in training generator).",
        "two_timescale_asymmetric_ood": "Held-out structural family: fast tau=0.27 s + slow tau=1.65 s mixture with post-reversal slow gain change; outside the single-lag bank family.",
    }
    train_records = [_record("nominal", i, TRAIN_BASE + i, world) for i in range(n_train)]
    prior = fit_population_prior(train_records, world)
    PRIOR_JSON.write_text(json.dumps(_json_value({"seed": SEED, "train_base": TRAIN_BASE, "split": "nominal", **prior}), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    all_rows: list[dict[str, Any]] = []
    records_for_example: dict[str, Record] = {}
    for split, base in TEST_BASES.items():
        for i in range(n_test):
            record = _record(split, i, base + i, world)
            if i == 0:
                records_for_example[split] = record
            for method in ("static_offset", "fixed_lag", "bank_frozen", "bank_adaptive"):
                all_rows.append(_score_record(record, world, prior, method))

    aggregate: dict[str, Any] = {}
    for split in TEST_BASES:
        aggregate[split] = {}
        for method in ("static_offset", "fixed_lag", "bank_frozen", "bank_adaptive"):
            selected = [r for r in all_rows if r["split"] == split and r["method"] == method]
            aggregate[split][method] = _aggregate(selected)

    paired = _paired_deltas(all_rows)
    # Build the first nominal adaptive trajectory for the interactive artifact.
    ex = records_for_example["nominal"]
    adaptive_scores = _score_record(ex, world, prior, "bank_adaptive")
    true_xyz = ex.point_truth
    # Re-run at every frame to expose the causal trace, not just score times.
    estimator = _make_estimator("bank_adaptive", world, prior)
    predicted_xyz, predicted_sd = [], []
    for mask in ex.masks:
        estimator.observe(mask)
        pred, sd = estimator.predict(mask.t, float(mask.velocity_obs))
        predicted_xyz.append(pred)
        predicted_sd.append(sd)
    predicted_xyz = np.asarray(predicted_xyz)
    example = {
        "title": "Synthetic dynamics-shift replay: nominal case 0",
        "notes": "Blue = synthetic truth; orange = bank with learned population prior and causal predictive-likelihood adaptation. One visible correctly associated mask per equal physical/observation time; no patient or UltraPOC claim.",
        "traces": {
            "synthetic truth": {"x": true_xyz[:, 0].tolist(), "y": true_xyz[:, 1].tolist(), "z": true_xyz[:, 2].tolist()},
            "bank adaptive estimate": {"x": predicted_xyz[:, 0].tolist(), "y": predicted_xyz[:, 1].tolist(), "z": predicted_xyz[:, 2].tolist()},
        },
        "time_s": (DT * np.arange(N_FRAMES)).tolist(),
        "predicted_sd_mm": np.asarray(predicted_sd).tolist(),
        "score_rows": adaptive_scores["scores"],
    }
    _make_html(example)

    result = {
        "title": "Dynamics shift/personalization replay",
        "description": "Synthetic structural OOD only. Tests whether a persistent registered spatial offset and a transient lag state absorb bias under equal observation and physical times.",
        "claim_boundary": "This is not a patient, clinical, real-ultrasound, or UltraPOC result.",
        "seed": SEED,
        "world_source": str(FROZEN / "knee_world.py"),
        "geometry_source": str(FROZEN.parent / "ab_leg_dynamic_20261112" / "DYNAMIC_GEOMETRY_FROZEN.npz"),
        "timing": {"dt_s": DT, "n_frames": N_FRAMES, "score_times_s": list(SCORE_TIMES), "horizons_s": list(HORIZONS), "observation_equals_physical_time": True},
        "mask_contract": "Exactly one visible correctly associated main-lumen mask per frame; independent sub-mm lateral/depth noise; no misses, clutter, or identity uncertainty.",
        "methods": {
            "static_offset": "shared registered persistent w_s=[w0,w1] offset updated from masks; transient z is fixed to zero",
            "fixed_lag": "single online persistent w_s=[w0,w1] plus transient z Kalman state with fixed nominal tau=0.72 s",
            "bank_frozen": "same compact tau bank and learned observation-only population prior; tau weights frozen",
            "bank_adaptive": "same compact tau bank/prior; tau weights updated by causal predictive likelihood",
        },
        "bank_tau_s": TAU_BANK.tolist(),
        "uncertainty": {"measurement_sd_mm": MEAS_SD, "measurement_variance_floor_mm2": MEAS_VAR_FLOOR, "state_variance_floor_mm2": STATE_VAR_FLOOR, "point_pose_floor_sd_mm": 0.25, "credible_interval": "marginal Gaussian 95%; coverage is empirical over scored times"},
        "splits_declared_before_results": split_definition,
        "training": {"n_records": n_train, "seed_base": TRAIN_BASE, "split": "nominal", "prior_path": str(PRIOR_JSON)},
        "test": {"n_records_per_split": n_test, "seed_bases": TEST_BASES},
        "population_prior": prior,
        "aggregate": aggregate,
        "paired_deltas_vs_fixed_lag": paired,
        "paired_case_clusters_bank_adaptive_vs_frozen": _paired_bank_case_clusters(all_rows),
        "per_case_method_rows": all_rows,
        "interactive_example": str(EXAMPLE_HTML),
    }
    RESULTS_JSON.write_text(json.dumps(_json_value(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=N_TRAIN, help="independent nominal training records")
    parser.add_argument("--test", type=int, default=N_TEST, help="test records per declared split")
    args = parser.parse_args()
    result = run(n_train=max(8, int(args.train)), n_test=max(4, int(args.test)))
    print(json.dumps({"results": str(RESULTS_JSON), "prior": str(PRIOR_JSON), "example": str(EXAMPLE_HTML), "aggregate": result["aggregate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
