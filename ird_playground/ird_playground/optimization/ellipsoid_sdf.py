"""Differentiable ellipsoid guidance SDF and exact outside certificates."""

from __future__ import annotations

import numpy as np
import torch


def _check_shapes(
    points: torch.Tensor,
    centers: torch.Tensor,
    rotations: torch.Tensor,
    semiaxes: torch.Tensor,
) -> None:
    if points.shape[-1:] != (3,) or centers.shape[-1:] != (3,):
        raise ValueError("points and centers must end in 3")
    if rotations.shape[-2:] != (3, 3) or semiaxes.shape[-1:] != (3,):
        raise ValueError("rotations must end in (3,3) and semiaxes in 3")
    if torch.any(semiaxes <= 0.0) or not torch.isfinite(semiaxes).all():
        raise ValueError("ellipsoid semiaxes must be finite and positive")


def ellipsoid_radial_signed_distance(
    points: torch.Tensor,
    centers: torch.Tensor,
    rotations: torch.Tensor,
    semiaxes: torch.Tensor,
) -> torch.Tensor:
    """Smooth conservative radial distance in metres.

    ``rotations`` maps ellipsoid-local vectors into world coordinates.  The
    minimum semiaxis scales normalized radius, underestimating distance along
    longer axes and therefore remaining conservative for obstacle guidance.
    """
    points = torch.as_tensor(points)
    centers = torch.as_tensor(centers, dtype=points.dtype, device=points.device)
    rotations = torch.as_tensor(rotations, dtype=points.dtype, device=points.device)
    semiaxes = torch.as_tensor(semiaxes, dtype=points.dtype, device=points.device)
    _check_shapes(points, centers, rotations, semiaxes)
    local = torch.matmul(rotations.transpose(-1, -2), (points - centers)[..., None])[..., 0]
    radius = torch.sqrt(torch.sum((local / semiaxes) ** 2, dim=-1) + 1.0e-12)
    return (radius - 1.0) * semiaxes.amin(dim=-1)


def _outside_distance_local(point: np.ndarray, semiaxes: np.ndarray) -> float:
    p = np.abs(np.asarray(point, dtype=np.float64))
    a = np.asarray(semiaxes, dtype=np.float64)
    scaled = p / a
    rho2 = float(scaled @ scaled)
    if abs(rho2 - 1.0) <= 1.0e-13:
        return 0.0
    if rho2 < 1.0:
        # Hard certification only accepts positive margins.  SLSQP is used for
        # the uncommon inside case to retain an exact signed diagnostic.
        from scipy.optimize import minimize

        axis = int(np.argmin(a))
        q0 = p.copy()
        q0[axis] += a[axis] * np.sqrt(max(1.0 - float(np.sum((q0 / a) ** 2)), 0.0))
        if not np.isfinite(q0).all() or np.linalg.norm(q0) < 1.0e-12:
            q0 = np.zeros(3); q0[axis] = a[axis]
        result = minimize(
            lambda q: float(np.sum((q - p) ** 2)),
            q0,
            constraints={"type": "eq", "fun": lambda q: float(np.sum((q / a) ** 2) - 1.0)},
            method="SLSQP",
            options={"ftol": 1.0e-14, "maxiter": 100},
        )
        if not result.success:
            raise RuntimeError(f"inside ellipsoid distance failed: {result.message}")
        return -float(np.linalg.norm(result.x - p))

    a2 = a * a
    lo = 0.0
    hi = max(float(np.linalg.norm(p) * a.max()), 1.0e-12)

    def equation(lam: float) -> float:
        return float(np.sum(a2 * p * p / (lam + a2) ** 2) - 1.0)

    while equation(hi) > 0.0:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if equation(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    closest = a2 * p / (lam + a2)
    return float(np.linalg.norm(closest - p))


def exact_ellipsoid_signed_distance(
    points: np.ndarray,
    centers: np.ndarray,
    rotations: np.ndarray,
    semiaxes: np.ndarray,
) -> np.ndarray:
    """Exact Euclidean signed distance for a broadcastable ellipsoid batch."""
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    semiaxes = np.asarray(semiaxes, dtype=np.float64)
    if points.shape[-1:] != (3,) or rotations.shape[-2:] != (3, 3):
        raise ValueError("invalid point or rotation shape")
    shape = np.broadcast_shapes(
        points.shape[:-1], centers.shape[:-1], rotations.shape[:-2], semiaxes.shape[:-1]
    )
    p = np.broadcast_to(points, shape + (3,)).reshape(-1, 3)
    c = np.broadcast_to(centers, shape + (3,)).reshape(-1, 3)
    R = np.broadcast_to(rotations, shape + (3, 3)).reshape(-1, 3, 3)
    a = np.broadcast_to(semiaxes, shape + (3,)).reshape(-1, 3)
    if np.any(a <= 0.0) or not np.isfinite(a).all():
        raise ValueError("ellipsoid semiaxes must be finite and positive")
    out = np.empty(len(p), dtype=np.float64)
    for i in range(len(p)):
        local = R[i].T @ (p[i] - c[i])
        out[i] = _outside_distance_local(local, a[i])
    return out.reshape(shape)


def ellipsoid_surface_mesh(
    center: np.ndarray,
    rotation: np.ndarray,
    semiaxes: np.ndarray,
    *,
    n_u: int = 32,
    n_v: int = 18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a world-space mesh suitable for Matplotlib ``plot_surface``."""
    u = np.linspace(0.0, 2.0 * np.pi, int(n_u))
    v = np.linspace(0.0, np.pi, int(n_v))
    local = np.stack(
        (
            semiaxes[0] * np.outer(np.cos(u), np.sin(v)),
            semiaxes[1] * np.outer(np.sin(u), np.sin(v)),
            semiaxes[2] * np.outer(np.ones_like(u), np.cos(v)),
        ), axis=-1,
    )
    world = local @ np.asarray(rotation, dtype=np.float64).T + np.asarray(center)
    return world[..., 0], world[..., 1], world[..., 2]


__all__ = [
    "ellipsoid_radial_signed_distance",
    "ellipsoid_surface_mesh",
    "exact_ellipsoid_signed_distance",
]
