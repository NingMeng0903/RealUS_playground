from __future__ import annotations

import numpy as np
import torch

from ird_playground.ird.canonical import canonical_invariants_torch
from ird_playground.ird.torch_kinematics import so3_exp
from ird_playground.neural.signed_field import SignedReachabilityField
from ird_playground.neural.train_signed import _split_indices
from ird_playground.region.operator import RegionA, RegionAConfig
from ird_playground.region.trajectory_operator import (
    TrajectoryTaskConfig,
    TrajectoryTaskOperator,
)


def test_canonical_embedding_is_common_yaw_and_tcp_roll_invariant():
    p = torch.tensor([[0.35, -0.22, 0.41]], dtype=torch.float64)
    R = so3_exp(torch.tensor([[0.2, -0.4, 0.3]], dtype=torch.float64))
    f0 = canonical_invariants_torch(p, R)

    yaw = so3_exp(torch.tensor([[0.0, 0.0, 1.2]], dtype=torch.float64))
    f_yaw = canonical_invariants_torch((yaw @ p[..., None]).squeeze(-1), yaw @ R)
    assert torch.allclose(f0, f_yaw, atol=1.0e-10)

    roll = so3_exp(torch.tensor([[0.0, 0.0, -0.8]], dtype=torch.float64))
    f_roll = canonical_invariants_torch(p, R @ roll)
    assert torch.allclose(f0, f_roll, atol=1.0e-10)


def test_signed_field_is_smooth_and_differentiable():
    model = SignedReachabilityField(width=32, depth=2, fourier_bands=2)
    x = torch.tensor([[0.1, 0.7, 0.4, -0.1, 0.2]], requires_grad=True)
    y = model(x)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    with torch.no_grad():
        y2 = model(x + 1.0e-6)
    assert torch.max(torch.abs(y2 - y)) < 1.0e-2


def test_region_a_preserves_tcp_and_rail_autograd():
    model = SignedReachabilityField(width=32, depth=2, fourier_bands=1)
    region = RegionA(RegionAConfig(samples=16, seed=2))
    tcp_xyz = torch.tensor([0.45, 0.08, 0.35], requires_grad=True)
    T_tcp = torch.eye(4)
    T_tcp = torch.cat((torch.cat((T_tcp[:3, :3], tcp_xyz[:, None]), dim=1), T_tcp[3:4]), dim=0)
    rail = torch.tensor(0.03, requires_grad=True)
    eye = torch.eye(4)
    result = region.query_tcp_rail(
        model, T_tcp, rail, T_world_rail=eye, T_rail_base0=eye
    )
    result.robust_clearance.backward()
    assert tcp_xyz.grad is not None and torch.isfinite(tcp_xyz.grad).all()
    assert rail.grad is not None and torch.isfinite(rail.grad)
    eps = 1.0e-3
    with torch.no_grad():
        plus = region.query_tcp_rail(model, T_tcp, rail + eps, T_world_rail=eye, T_rail_base0=eye).robust_clearance
        minus = region.query_tcp_rail(model, T_tcp, rail - eps, T_world_rail=eye, T_rail_base0=eye).robust_clearance
    fd = (plus - minus) / (2.0 * eps)
    assert torch.allclose(rail.grad, fd, rtol=0.1, atol=1.0e-3)


def test_region_a_extents_follow_medical_frame_b_t_n():
    region = RegionA(
        RegionAConfig(
            binormal_m=0.004,
            tangent_m=0.003,
            normal_m=0.002,
            samples=64,
            seed=7,
        )
    )
    offsets = region.position_offsets_local.detach().numpy()
    assert np.max(np.abs(offsets[:, 0])) <= 0.004
    assert np.max(np.abs(offsets[:, 1])) <= 0.003
    assert np.max(np.abs(offsets[:, 2])) <= 0.002


def test_validation_split_keeps_source_pose_groups_disjoint():
    boundary = np.full(12, -1, dtype=np.int64)
    source = np.repeat(np.arange(4, dtype=np.int64), 3)
    train, val = _split_indices(boundary, 0.25, seed=4, source_pose_id=source)
    train_sources = set(source[train].tolist())
    val_sources = set(source[val].tolist())
    assert train_sources.isdisjoint(val_sources)
    assert train_sources | val_sources == set(source.tolist())


def test_trajectory_partial_task_keeps_angle_and_uncertainty_semantics_separate():
    class RotationSensitiveField:
        def score_world(self, tcp, axis):
            del axis
            return tcp[..., 0, 0] + 0.1 * tcp[..., 0, 3]

    region_cfg = RegionAConfig(
        tangent_m=0.0,
        binormal_m=0.0,
        normal_m=0.0,
        cone_half_angle_deg=0.0,
        samples=4,
        seed=3,
    )
    common = dict(
        angle_half_range_deg=30.0,
        angle_samples=3,
        angle_axis_local=(0.0, 0.0, 1.0),
        uncertainty_aggregation="robust",
    )
    select = TrajectoryTaskOperator(
        TrajectoryTaskConfig(**common, angle_aggregation="select"),
        region_config=region_cfg,
    )
    robust = TrajectoryTaskOperator(
        TrajectoryTaskConfig(**common, angle_aggregation="robust"),
        region_config=region_cfg,
    )
    xyz = torch.tensor(
        [[0.2, 0.0, 0.3], [0.3, 0.0, 0.3]], requires_grad=True
    )
    T = torch.eye(4).repeat(2, 1, 1)
    T = torch.cat((torch.cat((T[:, :3, :3], xyz[..., None]), dim=-1), T[:, 3:4]), dim=-2)
    axis = torch.eye(4)
    selected = select(RotationSensitiveField(), T, axis)
    required = robust(RotationSensitiveField(), T, axis)
    assert selected.candidate_clearance.shape == (2, 3)
    assert selected.scenario_clearance.shape == (2, 3, 4)
    assert torch.all(selected.best_angle_index == 1)
    assert torch.all(selected.waypoint_clearance > required.waypoint_clearance)
    selected.trajectory_clearance.backward()
    assert xyz.grad is not None and torch.isfinite(xyz.grad).all()
