"""Rail inverse query: pick the rail coordinate that maximizes soft waypoint clearance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from ird_playground.region.operator import base_from_rail_torch, normalized_softmin


@dataclass
class RailQueryResult:
    clearance_per_rail: "torch.Tensor"
    best_rail: "torch.Tensor"
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
    """Soft-select a rail coordinate by waypoint softmin clearance."""
    if torch is None:
        raise ImportError("torch required")
    rails = rail_candidates.to(dtype=T_tcp_world_waypoints.dtype, device=T_tcp_world_waypoints.device)
    if rails.ndim == 0:
        rails = rails.reshape(1)
    n_rail = rails.shape[0]
    n_waypoint = T_tcp_world_waypoints.shape[-3] if T_tcp_world_waypoints.ndim >= 3 else 1
    if T_tcp_world_waypoints.ndim == 3:
        waypoints = T_tcp_world_waypoints
    else:
        waypoints = T_tcp_world_waypoints.unsqueeze(-3)
    per_rail = []
    per_waypoint = []
    for rail in rails:
        axis_world = T_axis_base_fn(rail) if callable(T_axis_base_fn) else base_from_rail_torch(
            rail, T_world_rail, T_rail_base0, axis=rail_axis
        )
        if hasattr(field, "score_world"):
            scores = field.score_world(waypoints, axis_world.unsqueeze(-3))
        else:
            raise TypeError("field must expose score_world()")
        wp = normalized_softmin(scores, tau, dim=-1)
        per_waypoint.append(wp)
        per_rail.append(normalized_softmin(wp, tau, dim=-1))
    clearance_per_rail = torch.stack(per_rail, dim=-1)
    waypoint_clearance = torch.stack(per_waypoint, dim=-2)
    best_idx = clearance_per_rail.argmax(dim=-1)
    gather = best_idx
    best_rail = rails[gather]
    best_clearance = clearance_per_rail.gather(-1, gather.unsqueeze(-1)).squeeze(-1)
    return RailQueryResult(
        clearance_per_rail=clearance_per_rail,
        best_rail=best_rail,
        best_clearance=best_clearance,
        waypoint_clearance=waypoint_clearance,
    )


__all__ = ["RailQueryResult", "rail_inverse_query"]
