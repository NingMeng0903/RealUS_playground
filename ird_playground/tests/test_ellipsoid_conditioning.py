from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from ird_playground.optimization.ellipsoid_sdf import (
    ellipsoid_radial_signed_distance,
    exact_ellipsoid_signed_distance,
)
from ird_playground.region.task_cone import TaskConeConfig, TaskConeReachability


class _OrientationField(torch.nn.Module):
    def score_world(self, tcp, axis):
        del axis
        return 8.0 + 4.0 * tcp[..., 0, 2] + 2.0 * tcp[..., 1, 0]


def test_radial_ellipsoid_sdf_surface_sign_and_ad_fd_gradient():
    dtype = torch.float64
    center = torch.zeros(3, dtype=dtype)
    rotation = torch.eye(3, dtype=dtype)
    axes = torch.tensor([0.020, 0.012, 0.012], dtype=dtype)
    points = torch.tensor(
        [[0.020, 0.0, 0.0], [0.024, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=dtype,
    )
    distance = ellipsoid_radial_signed_distance(points, center, rotation, axes)
    assert abs(distance[0].item()) < 1.0e-10
    assert distance[1].item() > 0.0
    assert distance[2].item() < 0.0

    point = torch.tensor([0.018, 0.010, 0.002], dtype=dtype, requires_grad=True)
    value = ellipsoid_radial_signed_distance(point, center, rotation, axes)
    ad = torch.autograd.grad(value, point)[0]
    eps = 1.0e-6
    fd = np.empty(3)
    for i in range(3):
        plus = point.detach().clone(); plus[i] += eps
        minus = point.detach().clone(); minus[i] -= eps
        fd[i] = float(
            (ellipsoid_radial_signed_distance(plus, center, rotation, axes)
             - ellipsoid_radial_signed_distance(minus, center, rotation, axes)) / (2.0 * eps)
        )
    assert np.allclose(ad.detach().numpy(), fd, rtol=2.0e-4, atol=2.0e-6)


def test_exact_ellipsoid_distance_is_euclidean_on_principal_axes_and_rotation():
    axes = np.array([0.020, 0.012, 0.012])
    angle = np.deg2rad(35.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    center = np.array([0.2, -0.1, 0.3])
    point = center + rotation[:, 0] * 0.027
    distance = exact_ellipsoid_signed_distance(point, center, rotation, axes)
    assert np.isclose(distance, 0.007, atol=1.0e-10)


def test_live_condition_center_changes_aggregate_and_has_gradient():
    cone = TaskConeReachability(TaskConeConfig(
        tip_half_angle_deg=20.0, roll_half_range_deg=20.0, samples=64, seed=17
    )).to(dtype=torch.float64)
    nominal = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    axis = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    centers = torch.tensor(
        [[0.0, 0.0, 0.0], [0.16, -0.08, 0.10]],
        dtype=torch.float64, requires_grad=True,
    )
    result = cone.query_condition_center(_OrientationField(), nominal, axis, centers)
    assert result.clearance.shape == (2,)
    assert not torch.isclose(result.clearance[0], result.clearance[1])
    grad = torch.autograd.grad(result.clearance.sum(), centers)[0]
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad).item() > 0
    assert torch.allclose(result.condition_weights.sum(dim=-1), torch.ones(2, dtype=torch.float64))


def test_adaptive_arrow_scaling_preserves_magnitude_and_suppresses_noise():
    experiment = Path(__file__).parents[1] / "experiments/moving_obstacle_u_band_demo.py"
    spec = importlib.util.spec_from_file_location("moving_obstacle_arrow_test", experiment)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    gtheta = np.zeros((9, 9), dtype=np.float64)
    grail = np.zeros_like(gtheta)
    gtheta[4, 4] = 10.0
    gtheta[4, 5] = 5.0
    gtheta[0, 0] = 1.0e-5
    u, v, relative = module.adaptive_gradient_arrows(gtheta, grail)
    assert relative.max() == 1.0
    assert np.all(np.sqrt(u * u + v * v) <= 1.0 + 1.0e-12)
    assert relative[4, 4] > relative[4, 5] > 0.0
    assert u[0, 0] == 0.0 and v[0, 0] == 0.0


def test_compact_obstacle_condition_rejoins_ird_after_two_mm_transition():
    experiment = Path(__file__).parents[1] / "experiments/moving_obstacle_u_band_demo.py"
    spec = importlib.util.spec_from_file_location("moving_obstacle_compact_test", experiment)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    axes = np.array([0.020, 0.012, 0.012])
    points = np.array([[0.0, 0.0, 0.015], [0.0, 0.0, 0.017], [0.0, 0.0, 0.019]])
    result = module.conditional_band_values(
        np.full(3, 6.0), points, np.zeros(3), np.eye(3), axes,
        safe_margin_m=0.003, planning_margin_m=0.005,
        target_clearance=5.0, soft_width_m=0.002,
    )
    conditioned = result["conditioned_clearance"]
    assert conditioned[1] > conditioned[0] + 4.0
    assert abs(conditioned[2] - conditioned[1]) < 1.1
    assert result["obstacle_alpha"][1] == 0.0
    assert result["obstacle_alpha"][2] == 0.0


def test_moving_demo_has_no_reference_derived_active_mask():
    source = (
        Path(__file__).parents[1]
        / "experiments/moving_obstacle_u_band_demo.py"
    ).read_text(encoding="utf-8")
    assert "active_spline_controls" not in source
    assert "reference-violation support active set" not in source
