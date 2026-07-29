"""Declared SE(3) metric for signed reachability labels and Eikonal loss.

One centimetre of translation is defined to be equivalent to one degree of
rotation.  With that convention

    λ = 0.01 / deg2rad(1) = 0.5730 m/rad

and the squared distance is

    d² = ||Δp||²_m + λ² ||Δω||²_rad.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


# 1 cm ≡ 1 deg → λ = 0.01 / (π/180) m/rad
LAMBDA_M_PER_RAD: float = 0.01 / math.radians(1.0)
METRIC_SCHEMA: str = "se3_weighted_l2_v1"
CM_PER_DEG: float = 1.0


def metric_manifest(*, lambda_m_per_rad: float = LAMBDA_M_PER_RAD) -> dict:
    return {
        "schema": METRIC_SCHEMA,
        "lambda_m_per_rad": float(lambda_m_per_rad),
        "cm_per_deg": CM_PER_DEG,
        "note": (
            "chart-coordinate weighted L2 used for EDT / Eikonal; "
            "not an SE(3) geodesic"
        ),
    }


def se3_distance_m(
    dp_m: np.ndarray,
    drot_rad: np.ndarray,
    *,
    lambda_m_per_rad: float = LAMBDA_M_PER_RAD,
) -> np.ndarray:
    """Return the declared metric distance in metres."""
    dp = np.asarray(dp_m, dtype=np.float64)
    dw = np.asarray(drot_rad, dtype=np.float64)
    if dp.ndim == 1:
        return float(np.sqrt(np.dot(dp, dp) + (lambda_m_per_rad * np.linalg.norm(dw)) ** 2))
    return np.sqrt(
        np.sum(dp * dp, axis=-1) + (lambda_m_per_rad * np.linalg.norm(dw, axis=-1)) ** 2
    ).astype(np.float64)


def se3_distance_m_torch(
    dp_m: "torch.Tensor",
    drot_rad: "torch.Tensor",
    *,
    lambda_m_per_rad: float = LAMBDA_M_PER_RAD,
) -> "torch.Tensor":
    if torch is None:
        raise ImportError("torch required")
    return torch.sqrt(
        (dp_m * dp_m).sum(dim=-1)
        + (float(lambda_m_per_rad) * torch.linalg.vector_norm(drot_rad, dim=-1)) ** 2
    )


def unit_speed_eikonal_loss(
    grad_wrt_normalized_chart: "torch.Tensor",
    *,
    target_slope: "torch.Tensor | None" = None,
    beta: float = 0.2,
) -> "torch.Tensor":
    """Eikonal residual for clearance labelled in declared-metric metres.

    Gradients are taken w.r.t. the *normalized* 9-D flange chart used by the
    network.  When ``target_slope`` is provided (boundary stencil pairs), match
    that empirical speed; otherwise enforce unit speed (``||∇f|| ≈ 1``), which
    is the correct condition when ``f`` approximates a metre-valued EDT/stencil
    distance under the declared metric.
    """
    if torch is None:
        raise ImportError("torch required")
    import torch.nn.functional as F

    speed = torch.linalg.vector_norm(grad_wrt_normalized_chart, dim=-1)
    if target_slope is None:
        target = torch.ones_like(speed)
    else:
        target = target_slope
    return F.smooth_l1_loss(speed, target, beta=float(beta))


__all__ = [
    "CM_PER_DEG",
    "LAMBDA_M_PER_RAD",
    "METRIC_SCHEMA",
    "metric_manifest",
    "se3_distance_m",
    "se3_distance_m_torch",
    "unit_speed_eikonal_loss",
]
