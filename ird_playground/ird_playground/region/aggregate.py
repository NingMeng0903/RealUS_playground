"""Query-side Region A: anisotropic Exp perturbations + softmin(m) + mean(q)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from ird_playground.probe.se3 import se3_exp, se3_mul


@dataclass(frozen=True)
class PositionExtent:
    tangent_1_m: float = 0.020
    tangent_2_m: float = 0.010
    normal_m: float = 0.002


@dataclass(frozen=True)
class OrientationExtent:
    tilt_tangent_1_deg: float = 8.0
    tilt_tangent_2_deg: float = 5.0
    axial_roll_deg: float = 3.0


@dataclass
class RegionScore:
    score: float
    m_robust: float
    q_region: float
    mean_score: float
    softmin_score: float
    coverage: float
    min_score: float
    num_samples: int


def sobol_unit_cube(n: int, dim: int, *, seed: int = 0) -> np.ndarray:
    eng = qmc.Sobol(d=dim, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(max(n, 2))))
    u = eng.random_base2(m)
    return u[:n]


def sample_anisotropic_xi(
    position: PositionExtent,
    orientation: OrientationExtent,
    num_samples: int,
    *,
    seed: int = 0,
    antithetic: bool = True,
) -> np.ndarray:
    """Return (K,6) ξ; if antithetic, pair ξ_{2j}=-ξ_{2j-1}."""
    if antithetic:
        n_pair = (num_samples + 1) // 2
        u = sobol_unit_cube(n_pair, 6, seed=seed)
        s = 2.0 * u - 1.0
        # build extents
        pos = PositionExtent(
            position.tangent_1_m, position.tangent_2_m, position.normal_m
        )
        ori = orientation
        b1 = np.deg2rad(ori.tilt_tangent_1_deg)
        b2 = np.deg2rad(ori.tilt_tangent_2_deg)
        psi = np.deg2rad(ori.axial_roll_deg)
        dp = np.stack(
            [s[:, 0] * pos.tangent_1_m, s[:, 1] * pos.tangent_2_m, s[:, 2] * pos.normal_m],
            axis=1,
        )
        dw = np.stack([s[:, 3] * b1, s[:, 4] * b2, s[:, 5] * psi], axis=1)
        half = np.concatenate([dp, dw], axis=1)
        paired = np.empty((half.shape[0] * 2, 6), dtype=np.float64)
        paired[0::2] = half
        paired[1::2] = -half
        return paired[:num_samples]

    u = sobol_unit_cube(num_samples, 6, seed=seed)
    s = 2.0 * u - 1.0
    dp = np.stack(
        [
            s[:, 0] * position.tangent_1_m,
            s[:, 1] * position.tangent_2_m,
            s[:, 2] * position.normal_m,
        ],
        axis=1,
    )
    b1 = np.deg2rad(orientation.tilt_tangent_1_deg)
    b2 = np.deg2rad(orientation.tilt_tangent_2_deg)
    psi = np.deg2rad(orientation.axial_roll_deg)
    dw = np.stack([s[:, 3] * b1, s[:, 4] * b2, s[:, 5] * psi], axis=1)
    return np.concatenate([dp, dw], axis=1).astype(np.float64)


def softmin(values: np.ndarray, tau: float, weights: np.ndarray | None = None) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / (w.sum() + 1e-12)
    tau = max(float(tau), 1e-8)
    m = v.min()
    return float(-tau * np.log(np.sum(w * np.exp(-(v - m) / tau)) + 1e-12) + m)


def coverage_from_m(m: np.ndarray, m_min: float = 0.0, tau_c: float = 0.5) -> float:
    z = (np.asarray(m, dtype=np.float64) - m_min) / max(tau_c, 1e-8)
    return float(np.mean(1.0 / (1.0 + np.exp(-z))))


def aggregate_mq(
    m: np.ndarray,
    q: np.ndarray,
    *,
    tau: float = 0.5,
    lambda_q: float = 0.5,
    tau_m_cost: float = 1.0,
    weights: np.ndarray | None = None,
) -> RegionScore:
    """m_robust = softmin(m); q_region = mean(q); score = -softplus(-m_r/τ)+λq."""
    mv = np.asarray(m, dtype=np.float64).reshape(-1)
    qv = np.asarray(q, dtype=np.float64).reshape(-1)
    w = np.ones_like(mv) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / (w.sum() + 1e-12)
    m_rob = softmin(mv, tau, w)
    q_reg = float(np.sum(w * qv))
    # softplus(-m/τ) = log(1+exp(-m/τ))
    cost_m = float(np.logaddexp(0.0, -m_rob / max(tau_m_cost, 1e-6)))
    score = -cost_m + float(lambda_q) * q_reg
    return RegionScore(
        score=score,
        m_robust=m_rob,
        q_region=q_reg,
        mean_score=float(np.sum(w * mv)),
        softmin_score=m_rob,
        coverage=coverage_from_m(mv),
        min_score=float(mv.min()),
        num_samples=int(mv.size),
    )


def aggregate_mean_softmin(
    values: np.ndarray,
    *,
    lam: float = 0.6,
    tau: float = 0.1,
    d_min: float = 0.3,
    tau_c: float = 0.05,
    weights: np.ndarray | None = None,
) -> RegionScore:
    """Legacy scalar aggregator (treat values as m)."""
    return aggregate_mq(values, np.zeros_like(values), tau=tau, lambda_q=0.0, tau_m_cost=1.0, weights=weights)


def perturb_center_poses(T_mu: np.ndarray, xi: np.ndarray) -> np.ndarray:
    T_mu = np.asarray(T_mu, dtype=np.float64).reshape(4, 4)
    out = np.empty((xi.shape[0], 4, 4), dtype=np.float64)
    for i, x in enumerate(xi):
        out[i] = se3_mul(T_mu, se3_exp(x))
    return out


def region_score_a(
    neural_ird,
    *,
    delta_T_center: np.ndarray | None = None,
    T_mu: np.ndarray | None = None,
    T_base: np.ndarray | None = None,
    position_extent: tuple[float, float, float] | PositionExtent = (0.02, 0.01, 0.002),
    orientation_extent: tuple[float, float, float] | OrientationExtent = (8.0, 5.0, 3.0),
    aggregation: str = "softmin_m_mean_q",
    num_samples: int = 32,
    lam: float = 0.6,
    tau: float = 0.5,
    d_min: float = 0.3,
    lambda_q: float = 0.5,
    seed: int = 0,
) -> RegionScore:
    if isinstance(position_extent, tuple):
        position_extent = PositionExtent(*position_extent)
    if isinstance(orientation_extent, tuple):
        orientation_extent = OrientationExtent(*orientation_extent)

    xi = sample_anisotropic_xi(position_extent, orientation_extent, num_samples, seed=seed)

    if T_mu is not None:
        T_base = np.eye(4) if T_base is None else np.asarray(T_base, dtype=np.float64)
        Ts = perturb_center_poses(T_mu, xi)
        from ird_playground.probe.se3 import invert_T

        dTs = np.stack([invert_T(Tk) @ T_base for Tk in Ts], axis=0)
    elif delta_T_center is not None:
        dT0 = np.asarray(delta_T_center, dtype=np.float64).reshape(4, 4)
        dTs = np.stack([se3_mul(dT0, se3_exp(x)) for x in xi], axis=0)
    else:
        raise ValueError("provide delta_T_center or T_mu")

    out = neural_ird.score_batch_delta_T(dTs)
    m = out.get("m", out.get("d"))
    q = out.get("q", out.get("q_comfort", np.zeros_like(m)))
    if aggregation in ("softmin_m_mean_q", "mean_softmin"):
        return aggregate_mq(m, q, tau=tau, lambda_q=lambda_q)
    raise ValueError(f"unsupported aggregation {aggregation!r}")
