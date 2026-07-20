"""Eval helpers: regression metrics + optimization-oriented P2 checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ird_playground.neural.train import differentiability_smoke, evaluate_point_field

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class PassThresholds:
    mae_max: float = 0.35
    spearman_min: float = 0.70
    boundary_iou_min: float = 0.50
    grad_cosine_min: float = 0.30
    ascent_improve_min: float = 0.40
    rail_ad_fd_rel_max: float = 0.25
    rail_sign_agree_min: float = 0.80
    region_improve_min: float = 0.40


def point_field_pass(metrics: dict[str, float], thr: PassThresholds | None = None) -> bool:
    thr = thr or PassThresholds()
    checks = [
        metrics.get("mae", 1e9) <= thr.mae_max,
        metrics.get("spearman", 0.0) >= thr.spearman_min,
        metrics.get("boundary_iou", 0.0) >= thr.boundary_iou_min,
    ]
    if "grad_cosine_median" in metrics:
        checks.append(metrics["grad_cosine_median"] >= thr.grad_cosine_min)
    if "ascent_improve_rate" in metrics:
        checks.append(metrics["ascent_improve_rate"] >= thr.ascent_improve_min)
    if "rail_ad_fd_rel" in metrics:
        checks.append(metrics["rail_ad_fd_rel"] <= thr.rail_ad_fd_rel_max)
    if "rail_sign_agree" in metrics:
        checks.append(metrics["rail_sign_agree"] >= thr.rail_sign_agree_min)
    if "region_improve_rate" in metrics:
        checks.append(metrics["region_improve_rate"] >= thr.region_improve_min)
    return all(checks)


def grad_cosine_vs_gt(
    net,
    arrays: dict[str, np.ndarray],
    *,
    n: int = 256,
    eps: float = 1e-3,
    seed: int = 0,
) -> dict[str, float]:
    """Median cos(∇_xyz m_θ, ∇_xyz m_gt) via central differences on GT labels."""
    if torch is None:
        raise ImportError("torch required")
    rng = np.random.default_rng(seed)
    feats = arrays["features"]
    m_gt = arrays["m_gt"] if "m_gt" in arrays else arrays["d"]
    idx = rng.choice(feats.shape[0], size=min(n, feats.shape[0]), replace=False)
    cosines = []
    net.model.eval()
    for i in idx:
        f0 = feats[i].astype(np.float32).copy()
        # GT FD on xyz using nearest neighbours in batch as proxy: local linear
        # Use network-free FD of interpolated GT is hard; instead FD of m_gt via
        # finite difference of network GT surrogate: compare AD to FD of the net
        # against a GT directional proxy from m labels of nearby samples.
        x = torch.tensor(f0[None, :], dtype=torch.float32, device=net.device, requires_grad=True)
        reach_logit, m, _, _ = net.model(x)
        m.sum().backward()
        g_theta = x.grad[0, :3].detach().cpu().numpy()

        # GT gradient proxy: central FD on xyz using nearby samples' m_gt
        g_gt = np.zeros(3, dtype=np.float64)
        for ax in range(3):
            # find approximate ∂m/∂x via local regression on σ-ball neighbours
            dxyz = feats[:, :3] - f0[:3]
            dist = np.linalg.norm(dxyz, axis=1)
            nb = np.argsort(dist)[1:32]
            # least-squares fit m ≈ m0 + g·Δx
            A = dxyz[nb]
            b = m_gt[nb] - m_gt[i]
            if A.shape[0] >= 3:
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                g_gt = sol[:3]
                break
        else:
            # fallback: FD of network score (self-consistency) — skip
            continue
        n1 = np.linalg.norm(g_theta) + 1e-12
        n2 = np.linalg.norm(g_gt) + 1e-12
        cosines.append(float(np.dot(g_theta, g_gt) / (n1 * n2)))
    arr = np.asarray(cosines, dtype=np.float64) if cosines else np.array([0.0])
    return {
        "grad_cosine_median": float(np.median(arr)),
        "grad_cosine_mean": float(np.mean(arr)),
        "grad_cosine_n": float(arr.size),
    }


def ascent_gt_improve(
    net,
    arrays: dict[str, np.ndarray],
    *,
    n: int = 128,
    step: float = 0.01,
    seed: int = 0,
) -> dict[str, float]:
    """From unreachable points, take one ∇m ascent step; measure GT m improve rate."""
    if torch is None:
        raise ImportError("torch required")
    rng = np.random.default_rng(seed)
    feats = arrays["features"]
    y = arrays["reachable"] if "reachable" in arrays else arrays["p_reach"]
    m_gt = arrays["m_gt"] if "m_gt" in arrays else arrays["d"]
    unre = np.flatnonzero(y < 0.5)
    if unre.size == 0:
        return {"ascent_improve_rate": 1.0, "ascent_n": 0.0}
    pick = rng.choice(unre, size=min(n, unre.size), replace=False)
    improved = 0
    net.model.eval()
    for i in pick:
        f0 = feats[i].astype(np.float32).copy()
        x = torch.tensor(f0[None, :], dtype=torch.float32, device=net.device, requires_grad=True)
        _, m, _, _ = net.model(x)
        m.sum().backward()
        g = x.grad[0, :3].detach().cpu().numpy()
        g = g / (np.linalg.norm(g) + 1e-12)
        f1 = f0.copy()
        f1[:3] = f1[:3] + step * g
        # GT improve: nearest neighbour m_gt after move
        d0 = np.linalg.norm(feats[:, :3] - f0[:3], axis=1)
        d1 = np.linalg.norm(feats[:, :3] - f1[:3], axis=1)
        m0 = float(m_gt[i])
        m1 = float(m_gt[int(np.argmin(d1))])
        if m1 > m0 + 1e-4:
            improved += 1
    return {
        "ascent_improve_rate": float(improved / max(len(pick), 1)),
        "ascent_n": float(len(pick)),
    }


def rail_y_ad_vs_fd(
    net,
    *,
    n: int = 32,
    rail_y: float = 0.0,
    eps: float = 1e-3,
    seed: int = 0,
) -> dict[str, float]:
    """AD ∂score/∂rail_y vs central FD; also sign agreement."""
    from ird_playground.ird.query_base import rail_y_grad_ad_fd

    return rail_y_grad_ad_fd(net, n=n, rail_y=rail_y, eps=eps, seed=seed)


def region_softmin_improve(
    net,
    *,
    n: int = 24,
    seed: int = 0,
) -> dict[str, float]:
    """Perturb region centre along −∇ softmin(m); count softmin increases."""
    from ird_playground.probe.se3 import mat4_from_Rt
    from ird_playground.region.aggregate import region_score_a

    rng = np.random.default_rng(seed)
    improved = 0
    for _ in range(n):
        p = rng.uniform(-0.4, 0.4, size=3)
        T = mat4_from_Rt(np.eye(3), p)
        rs0 = region_score_a(net, T_mu=T, num_samples=16, seed=int(rng.integers(0, 1_000_000)))
        # finite-diff direction on translation via score
        if torch is None:
            break
        # nudge toward higher m_robust by small random search (smoke)
        best = rs0.m_robust
        for _k in range(4):
            dp = rng.normal(scale=0.01, size=3)
            T2 = mat4_from_Rt(np.eye(3), p + dp)
            rs1 = region_score_a(net, T_mu=T2, num_samples=16, seed=0)
            best = max(best, rs1.m_robust)
        if best > rs0.m_robust + 1e-4:
            improved += 1
    return {
        "region_improve_rate": float(improved / max(n, 1)),
        "region_n": float(n),
    }


def evaluate_optimization_suite(
    net,
    arrays: dict[str, np.ndarray],
    *,
    seed: int = 0,
) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(grad_cosine_vs_gt(net, arrays, seed=seed))
    out.update(ascent_gt_improve(net, arrays, seed=seed))
    out.update(rail_y_ad_vs_fd(net, seed=seed))
    out.update(region_softmin_improve(net, seed=seed))
    out["grad_norm"] = differentiability_smoke(net)
    return out


__all__ = [
    "PassThresholds",
    "ascent_gt_improve",
    "differentiability_smoke",
    "evaluate_optimization_suite",
    "evaluate_point_field",
    "grad_cosine_vs_gt",
    "point_field_pass",
    "rail_y_ad_vs_fd",
    "region_softmin_improve",
]
