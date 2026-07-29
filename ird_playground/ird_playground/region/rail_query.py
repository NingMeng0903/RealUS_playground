"""Rail inverse query: batched softmin clearance over rail candidates.

Returns per-rail multi-waypoint softmin margins and a soft-selected rail that
remains differentiable w.r.t. continuous rail coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from ird_playground.region.operator import base_from_rail_torch, normalized_softmin


@dataclass
class RailQueryResult:
    clearance_per_rail: "torch.Tensor"
    best_rail: "torch.Tensor"
    soft_rail: "torch.Tensor"
    best_clearance: "torch.Tensor"
    waypoint_clearance: "torch.Tensor"


def rail_inverse_query(
    field,
    T_tcp_world_waypoints: "torch.Tensor",
    rail_candidates: "torch.Tensor",
    T_axis_base_fn,
    *,
    T_world_rail: "torch.Tensor",
    T_rail_base0: "torch.Tensor",
    rail_axis: int = 1,
    tau: float = 0.15,
) -> RailQueryResult:
    """Soft-select a rail coordinate by waypoint softmin clearance.

    Parameters
    ----------
    T_tcp_world_waypoints
        ``[..., W, 4, 4]`` TCP poses (or a single ``[4, 4]`` / ``[W, 4, 4]``).
    rail_candidates
        ``[R]`` rail coordinates. Gradients flow through continuous rails when
        ``T_axis_base_fn`` (or the default SE(3) rail map) is differentiable.
    """
    if torch is None:
        raise ImportError("torch required")
    rails = rail_candidates.to(dtype=T_tcp_world_waypoints.dtype, device=T_tcp_world_waypoints.device)
    if rails.ndim == 0:
        rails = rails.reshape(1)
    if T_tcp_world_waypoints.ndim == 2:
        waypoints = T_tcp_world_waypoints.unsqueeze(-3)
    else:
        waypoints = T_tcp_world_waypoints
    n_wp = waypoints.shape[-3]

    if callable(T_axis_base_fn):
        axes = torch.stack([T_axis_base_fn(r) for r in rails], dim=0)
    else:
        axes = base_from_rail_torch(
            rails, T_world_rail, T_rail_base0, axis=rail_axis
        )
    # axes: [R, 4, 4] → broadcast onto waypoints [..., W, 4, 4] as [..., R, W, 4, 4]
    wp = waypoints.unsqueeze(-4).expand(*waypoints.shape[:-3], rails.shape[0], n_wp, 4, 4)
    ax = axes.to(dtype=waypoints.dtype, device=waypoints.device)
    lead = wp.shape[:-4]
    ax = ax.view(*([1] * len(lead)), rails.shape[0], 1, 4, 4).expand(*lead, rails.shape[0], n_wp, 4, 4)
    if not hasattr(field, "score_world"):
        raise TypeError("field must expose score_world()")
    scores = field.score_world(wp, ax)
    # scores: [..., R, W] — keep per-waypoint margins; softmin over W for the rail score.
    waypoint_clearance = scores
    clearance_per_rail = normalized_softmin(scores, tau, dim=-1)
    best_idx = clearance_per_rail.argmax(dim=-1)
    best_rail = rails[best_idx]
    best_clearance = clearance_per_rail.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)
    # Soft rail keeps a differentiable path for continuous rail optimisation.
    weights = torch.softmax(clearance_per_rail / max(float(tau), 1.0e-6), dim=-1)
    soft_rail = (weights * rails).sum(dim=-1)
    return RailQueryResult(
        clearance_per_rail=clearance_per_rail,
        best_rail=best_rail,
        soft_rail=soft_rail,
        best_clearance=best_clearance,
        waypoint_clearance=waypoint_clearance,
    )


__all__ = ["RailQueryResult", "rail_inverse_query"]
