"""Phase 5 capacity-head unit tests: soft support functions and shapes."""

from __future__ import annotations

import math

import numpy as np
import torch

from ird_playground.ird.canonical import FLANGE_CANONICAL_DIM
from ird_playground.neural.capacity_head import (
    CapacityHead,
    CapacityHeadConfig,
    chart_basis_from_position,
    evaluate_normal_force,
    evaluate_tangential_velocity,
    force_capacity,
    linear_jacobian_to_chart,
    select_capacity_best,
    softabs,
    softmin,
    support_velocity,
)


def test_softabs_and_softmin_identities():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    sa = softabs(x, eps=1.0e-6)
    assert torch.allclose(sa, x.abs(), atol=1.0e-5)
    const = torch.full((4, 5), 3.0)
    assert torch.allclose(softmin(const, tau=0.1, dim=-1), torch.full((4,), 3.0))


def test_support_velocity_matches_hard_abs():
    gens = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.5],
        ]
    )  # 3 x 3 joints
    u = torch.tensor([1.0, 0.0, 0.0])
    qdot = torch.tensor([1.0, 2.0, 3.0])
    hard = float(sum(abs(float((u * gens[:, j]).sum())) * float(qdot[j]) for j in range(3)))
    soft = float(support_velocity(u, gens, qdot, eps=1.0e-8))
    assert abs(soft - hard) < 1.0e-6


def test_force_capacity_approaches_hard_min():
    gens = torch.eye(3)
    u = torch.tensor([1.0, 0.0, 0.0])
    tau = torch.tensor([10.0, 20.0, 30.0])
    # Only joint 0 contributes |J·u|=1 → α = 10; others have |J·u|≈eps so ratios huge.
    soft = float(force_capacity(u, gens, tau, eps=1.0e-8, softmin_tau=1.0e-3))
    assert abs(soft - 10.0) < 0.05


def test_tangential_vs_normal_separated():
    gens = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    tangent = torch.tensor([1.0, 0.0, 0.0])
    normal = torch.tensor([0.0, 1.0, 0.0])
    qdot = torch.ones(2)
    tau = torch.ones(2) * 5.0
    h_v = float(evaluate_tangential_velocity(tangent, gens, qdot, eps=1.0e-8))
    alpha = float(evaluate_normal_force(normal, gens, tau, eps=1.0e-8, softmin_tau=1.0e-3))
    assert abs(h_v - 1.0) < 1.0e-5
    assert abs(alpha - 5.0) < 0.05
    # Crossing queries: tangent capacity on normal axis uses the other column.
    h_cross = float(evaluate_tangential_velocity(normal, gens, qdot, eps=1.0e-8))
    assert abs(h_cross - 1.0) < 1.0e-5


def test_capacity_head_shapes_and_directional():
    head = CapacityHead(CapacityHeadConfig(n_joints=7, width=32, depth=2))
    x = torch.zeros(5, FLANGE_CANONICAL_DIM)
    gens = head(x)
    assert gens.shape == (5, 3, 7)
    # Zero-init final layer ⇒ zero generators; softabs floor ⇒ tiny support.
    u = torch.tensor([0.0, 0.0, 1.0])
    h = head.velocity_support(x, u)
    assert h.shape == (5,)
    assert torch.all(h >= 0.0)
    assert float(h.detach().max()) < 0.01
    tang = torch.tensor([[1.0, 0.0, 0.0]]).expand(5, 3)
    norm = torch.tensor([[0.0, 0.0, 1.0]]).expand(5, 3)
    hv, alpha = head.directional_capacity(x, tangent=tang, normal=norm)
    assert hv.shape == (5,)
    assert alpha.shape == (5,)
    assert torch.isfinite(hv).all() and torch.isfinite(alpha).all()


def test_select_capacity_best_picks_larger_support():
    weak = torch.zeros(3, 2)
    strong = torch.tensor([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    gens = torch.stack((weak, strong), dim=0)  # [B=2, 3, 2]
    gens = gens.unsqueeze(0)  # [1, B, 3, 2]
    u = torch.tensor([1.0, 0.0, 0.0])
    qdot = torch.ones(2)
    best, score, idx = select_capacity_best(
        gens, u, mode="velocity", qdot_max=qdot, eps=1.0e-8
    )
    assert int(idx.item()) == 1
    assert best.shape == (1, 3, 2)
    assert abs(float(score.item()) - 2.0) < 1.0e-5


def test_chart_frame_jacobian_yaw_invariance():
    # Axis-frame J and p; rotate both by the same yaw → chart generators unchanged.
    p0 = torch.tensor([0.4, 0.1, 0.3])
    J0 = torch.tensor(
        [
            [0.1, -0.2, 0.0, 0.3, 0.0, 0.0, 0.05],
            [0.0, 0.15, -0.1, 0.0, 0.2, 0.0, 0.0],
            [0.05, 0.0, 0.1, -0.05, 0.0, 0.1, 0.0],
        ]
    )
    yaw = math.radians(37.0)
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    p1 = Rz @ p0
    J1 = Rz @ J0
    g0 = linear_jacobian_to_chart(J0, p0)
    g1 = linear_jacobian_to_chart(J1, p1)
    assert torch.allclose(g0, g1, atol=1.0e-5)
    R = chart_basis_from_position(p0)
    assert torch.allclose(R.transpose(-1, -2) @ R, torch.eye(3), atol=1.0e-5)


def test_select_jacobian_gt_indices_filters():
    from ird_playground.ird.jacobian_gt import JacobianGtConfig, select_jacobian_gt_indices

    n = 20
    q = np.zeros((n, 7), dtype=np.float32)
    q[0] = np.nan
    reachable = np.ones(n, dtype=np.float32)
    reachable[1] = 0.0
    free = np.ones(n, dtype=np.float32)
    free[2] = 0.0
    arrays = {"q_best": q, "reachable": reachable, "q_selfcol": free}
    cfg = JacobianGtConfig(
        source_npz="x.npz",
        output_npz="y.npz",
        max_samples=100,
        seed=0,
    )
    idx = select_jacobian_gt_indices(arrays, cfg)
    assert 0 not in idx
    assert 1 not in idx
    assert 2 not in idx
    assert idx.size == n - 3
