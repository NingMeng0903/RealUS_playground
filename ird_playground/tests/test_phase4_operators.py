"""Phase 4 query-operator regressions: nested set query, B8 lobe, path+ψ."""

from __future__ import annotations

import math

import torch

from ird_playground.ird.canonical import FLANGE_CANONICAL_DIM, canonical_flange_from_world_torch
from ird_playground.neural.capacity_head import CapacityHead, support_velocity
from ird_playground.region.direction_lobe import (
    ascend_direction,
    chart_from_direction,
    direction_lobe,
    orthonormal_frame_from_z,
)
from ird_playground.region.operator import RegionAConfig
from ird_playground.region.rail_query import rail_inverse_query
from ird_playground.region.set_query import (
    SetQueryConfig,
    SetQueryOperator,
    rockafellar_uryasev_cvar,
)
from ird_playground.region.trajectory_operator import (
    TrajectoryTaskConfig,
    TrajectoryTaskOperator,
)
from ird_playground.region.conditional_query import conditional_candidate_query


def test_conditional_query_uses_nearest_safe_reachable_candidate_and_has_gradients():
    clearance = torch.tensor([[8.0, 7.0, 6.0]], requires_grad=True)
    obstacle = torch.tensor([[-0.01, 0.03, 0.04]], requires_grad=True)
    nearest = torch.tensor([[0.0, 0.2, 0.1]])
    result = conditional_candidate_query(
        clearance,
        obstacle,
        nearest_cost=nearest,
        clearance_target=5.0,
        safe_distance=0.01,
    )
    assert result.valid.tolist() == [True]
    assert result.selected_index.tolist() == [2]
    result.aggregate_clearance.sum().backward()
    assert torch.isfinite(clearance.grad).all()
    assert torch.isfinite(obstacle.grad).all()


def test_conditional_query_fails_closed_when_obstacle_blocks_every_candidate():
    result = conditional_candidate_query(
        torch.tensor([[9.0, 8.0]]),
        torch.tensor([[-0.1, -0.2]]),
        clearance_target=5.0,
        safe_distance=0.01,
    )
    assert result.valid.tolist() == [False]
    assert not torch.isfinite(result.aggregate_clearance).any()


class _LinearChartField:
    """Score = first chart component (p_z in axis frame after flange map)."""

    def score(self, chart):
        return chart[..., 0]

    def score_world(self, tcp, axis):
        # Identity tool so flange ≡ TCP for these unit tests when T_flange_tcp = I.
        tool = torch.eye(4, dtype=tcp.dtype, device=tcp.device)
        return canonical_flange_from_world_torch(tcp, axis, tool)[..., 0]


class _PsiAwareField:
    def score_world(self, tcp, axis):
        del axis
        return tcp[..., 2, 3]

    def score_world_psi(self, tcp, axis, psi):
        del axis
        # Prefer psi ≈ 0.4; path still uses z height.
        return tcp[..., 2, 3] - (psi - 0.4) ** 2


def test_set_query_nested_shapes_and_outer_max():
    op = SetQueryOperator(
        SetQueryConfig(
            free_samples=4,
            uncertain_samples=5,
            beta_samples=2,
            beta_half_range_deg=10.0,
            seed=3,
        )
    )
    field = _LinearChartField()
    T = torch.eye(4).repeat(3, 1, 1)
    T[:, 2, 3] = torch.tensor([0.2, 0.3, 0.4])
    axis = torch.eye(4)
    result = op(field, T, axis, use_cvar=True)
    k_free = 4 * 2
    assert result.nested_scores.shape == (3, k_free, 5)
    assert result.free_clearance.shape == (3, k_free)
    assert result.clearance.shape == (3,)
    assert torch.isfinite(result.clearance).all()
    # Soft outer max sits between min and max of free clearances.
    assert torch.all(result.clearance <= result.free_clearance.max(dim=-1).values + 1e-5)
    assert torch.all(result.clearance >= result.free_clearance.min(dim=-1).values - 1e-5)
    soft = op(field, T, axis, use_cvar=False)
    assert soft.nested_scores.shape == (3, k_free, 5)


def test_rockafellar_uryasev_lower_tail_is_below_mean():
    values = torch.tensor([[0.0, 1.0, 2.0, 10.0]])
    cvar = rockafellar_uryasev_cvar(values, alpha=0.5, dim=-1)
    assert float(cvar) <= float(values.mean()) + 1e-6
    assert float(cvar) <= 1.0 + 1e-5


def test_direction_lobe_b8_uses_flange_tcp():
    """Changing T_flange_tcp must change the lobe outputs (B8)."""

    class ChartField:
        def score(self, chart):
            return chart.sum(dim=-1)

    position = torch.tensor([0.35, -0.10, 0.40])
    dirs = torch.tensor([[0.0, 0.0, 1.0], [0.2, 0.0, math.sqrt(1.0 - 0.04)]])
    rolls = torch.linspace(-0.3, 0.3, 5)
    T_axis = torch.eye(4)
    identity = torch.eye(4)
    tilted = torch.eye(4)
    # Non-trivial flange→TCP: 15 mm lateral + ~50° about y (probe45-like).
    ang = math.radians(50.0)
    cy, sy = math.cos(ang), math.sin(ang)
    tilted[:3, :3] = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    tilted[:3, 3] = torch.tensor([0.015, 0.0, 0.0])

    a = direction_lobe(ChartField(), position, dirs, T_axis, identity, rolls)
    b = direction_lobe(ChartField(), position, dirs, T_axis, tilted, rolls)
    assert a.direction_clearance.shape == (2,)
    assert not torch.allclose(a.direction_clearance, b.direction_clearance, atol=1e-5)

    chart_i = chart_from_direction(position, dirs[0], rolls[0], T_axis, identity)
    chart_t = chart_from_direction(position, dirs[0], rolls[0], T_axis, tilted)
    assert chart_i.shape[-1] == FLANGE_CANONICAL_DIM
    assert not torch.allclose(chart_i, chart_t, atol=1e-6)


def test_direction_lobe_chart_path_avoids_acos_atan2():
    """Frame construction uses Frisvad ONB; R columns stay orthonormal."""
    z = torch.tensor([0.3, -0.4, 0.8660254])
    R = orthonormal_frame_from_z(z)
    assert torch.allclose(R.T @ R, torch.eye(3), atol=1e-5)
    assert torch.allclose(R[:, 2], z / z.norm(), atol=1e-5)


def test_direction_lobe_ascend_increases_clearance():
    class PeakField:
        def score(self, chart):
            # Prefer charts whose u_z·ẑ (index 5) is large → directions near +Z.
            return chart[..., 5]

    position = torch.tensor([0.3, 0.0, 0.4])
    u0 = torch.tensor([0.8, 0.2, 0.1])
    rolls = torch.tensor([0.0])
    T_axis = torch.eye(4)
    tool = torch.eye(4)
    u1 = ascend_direction(
        PeakField(), position, u0, T_axis, tool, rolls, steps=12, lr=0.35
    )
    before = direction_lobe(PeakField(), position, u0.unsqueeze(0), T_axis, tool, rolls)
    after = direction_lobe(PeakField(), position, u1.unsqueeze(0), T_axis, tool, rolls)
    assert float(after.direction_clearance) >= float(before.direction_clearance) - 1e-5
    assert torch.allclose(u1.norm(), torch.tensor(1.0), atol=1e-5)


def test_rail_query_batched_and_differentiable():
    class HeightField:
        def score_world(self, tcp, axis):
            # Prefer rails that lift the axis origin in z.
            return tcp[..., 2, 3] - axis[..., 2, 3]

    waypoints = torch.eye(4).repeat(3, 1, 1)
    waypoints[:, 0, 3] = torch.linspace(0.2, 0.4, 3)
    rails = torch.tensor([0.0, 0.05, 0.10], requires_grad=True)
    eye = torch.eye(4)
    result = rail_inverse_query(
        HeightField(),
        waypoints,
        rails,
        None,
        T_world_rail=eye,
        T_rail_base0=eye,
        rail_axis=1,
    )
    assert result.clearance_per_rail.shape == (3,)
    assert result.waypoint_clearance.shape == (3, 3)
    result.soft_rail.backward()
    assert rails.grad is not None and torch.isfinite(rails.grad).all()


def test_trajectory_path_plus_psi_wiring():
    region_cfg = RegionAConfig(
        tangent_m=0.0,
        binormal_m=0.0,
        normal_m=0.0,
        cone_half_angle_deg=0.0,
        samples=2,
        seed=9,
    )
    op = TrajectoryTaskOperator(
        TrajectoryTaskConfig(
            angle_half_range_deg=0.0,
            angle_samples=1,
            n_path=1,
            trajectory_aggregation="min",
            psi_half_range_rad=0.6,
            psi_samples=5,
        ),
        region_config=region_cfg,
    )
    xyz = torch.tensor([[0.3, 0.0, 0.35], [0.3, 0.0, 0.45]], requires_grad=True)
    T = torch.eye(4).repeat(2, 1, 1)
    T = torch.cat((torch.cat((T[:, :3, :3], xyz[..., None]), dim=-1), T[:, 3:4]), dim=-2)
    axis = torch.eye(4)
    result = op(_PsiAwareField(), T, axis)
    assert result.psi_clearance.shape == (5,)
    assert result.psi_offsets_rad.shape == (5,)
    # Preferred ψ near 0.4 should win among the grid.
    best = float(result.psi_offsets_rad[int(result.best_psi_index.item())])
    assert abs(best - 0.4) <= 0.6 / 2 + 1e-6
    # Path term still present.
    assert result.path_clearance.shape == (1,)
    result.trajectory_clearance.backward()
    assert xyz.grad is not None and torch.isfinite(xyz.grad).all()

def test_capacity_head_stub_shapes():
    head = CapacityHead()
    x = torch.zeros(4, FLANGE_CANONICAL_DIM)
    gens = head(x)
    assert gens.shape == (4, 3, 7)
    u = torch.tensor([0.0, 0.0, 1.0])
    qdot = torch.ones(7)
    assert support_velocity(u, gens[0], qdot).shape == ()
