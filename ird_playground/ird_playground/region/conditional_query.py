"""Global candidate conditioning for dynamic world obstacles."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ConditionalQueryResult:
    conditioned_clearance: torch.Tensor
    aggregate_clearance: torch.Tensor
    feasible: torch.Tensor
    selected_index: torch.Tensor
    selected_clearance: torch.Tensor
    selected_nearest_cost: torch.Tensor
    valid: torch.Tensor


def conditional_candidate_query(
    ird_clearance: torch.Tensor,
    obstacle_distance: torch.Tensor,
    task_feasible: torch.Tensor | None = None,
    nearest_cost: torch.Tensor | None = None,
    *,
    clearance_target: float = 0.0,
    safe_distance: float = 0.0,
    obstacle_tau: float = 0.01,
    aggregate_tau: float = 0.10,
) -> ConditionalQueryResult:
    """Apply obstacle/task conditions before the A aggregation.

    Positive obstacle distance is safe.  The smooth barrier guides gradients;
    hard filtering makes ``valid`` fail closed when no target-clearing option
    remains.  Within target-clearing options, the smallest NEARST cost wins.
    """
    if ird_clearance.ndim < 1:
        raise ValueError("ird_clearance must have a candidate dimension")
    if obstacle_distance.shape != ird_clearance.shape:
        raise ValueError("obstacle_distance must match ird_clearance")
    if task_feasible is None:
        task_feasible = torch.ones_like(ird_clearance, dtype=torch.bool)
    if nearest_cost is None:
        nearest_cost = torch.zeros_like(ird_clearance)
    if task_feasible.shape != ird_clearance.shape or nearest_cost.shape != ird_clearance.shape:
        raise ValueError("task_feasible and nearest_cost must match candidate shape")
    obs_tau = max(float(obstacle_tau), 1.0e-6)
    agg_tau = max(float(aggregate_tau), 1.0e-6)
    barrier = torch.nn.functional.softplus(
        (float(safe_distance) - obstacle_distance) / obs_tau
    ) * obs_tau
    conditioned = ird_clearance - barrier
    feasible = task_feasible & (obstacle_distance >= float(safe_distance))
    masked = torch.where(feasible, conditioned, torch.full_like(conditioned, -torch.inf))
    aggregate = agg_tau * (
        torch.logsumexp(masked / agg_tau, dim=-1)
        - torch.log(torch.as_tensor(masked.shape[-1], dtype=masked.dtype, device=masked.device))
    )
    target_ok = feasible & (ird_clearance >= float(clearance_target))
    nearest_masked = torch.where(target_ok, nearest_cost, torch.full_like(nearest_cost, torch.inf))
    nearest_index = nearest_masked.argmin(dim=-1)
    has_target = target_ok.any(dim=-1)
    fallback_index = masked.argmax(dim=-1)
    selected_index = torch.where(has_target, nearest_index, fallback_index)
    selected_clearance = torch.gather(conditioned, -1, selected_index[..., None]).squeeze(-1)
    selected_nearest = torch.gather(nearest_cost, -1, selected_index[..., None]).squeeze(-1)
    valid = has_target & torch.isfinite(selected_clearance)
    return ConditionalQueryResult(
        conditioned_clearance=conditioned,
        aggregate_clearance=aggregate,
        feasible=feasible,
        selected_index=selected_index,
        selected_clearance=selected_clearance,
        selected_nearest_cost=selected_nearest,
        valid=valid,
    )


__all__ = ["ConditionalQueryResult", "conditional_candidate_query"]
