"""Unit tests for flange GT feature shape and declared-metric stencil units."""

from __future__ import annotations

import numpy as np
import pytest


def test_flange_canonical_dim_is_nine():
    from ird_playground.ird.canonical import FLANGE_CANONICAL_DIM

    assert FLANGE_CANONICAL_DIM == 9


def test_flange_se3_feature_shape_and_canonical():
    torch = pytest.importorskip("torch")
    from ird_playground.ird.canonical import FLANGE_CANONICAL_DIM
    from ird_playground.ird.gpu_pose_gt import (
        _flange_canonical_np,
        _flange_se3_features,
    )
    from ird_playground.ird.robot_model import RobotModelSpec

    spec = RobotModelSpec.default_probe45()
    tool = torch.as_tensor(spec.tool_frame().T_flange_tcp, dtype=torch.float32)
    p = torch.tensor([[0.4, -0.1, 0.5], [0.2, 0.3, 0.1]], dtype=torch.float32)
    # Two random orthonormal frames via QR.
    rng = np.random.default_rng(0)
    Rs = []
    for _ in range(2):
        a = rng.normal(size=(3, 3))
        q, _ = np.linalg.qr(a)
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        Rs.append(q)
    R = torch.as_tensor(np.stack(Rs), dtype=torch.float32)
    feats = _flange_se3_features(p, R, tool)
    assert tuple(feats.shape) == (2, 9)
    chart = _flange_canonical_np(
        feats.numpy(), np.eye(4, dtype=np.float64), spec.root_to_j1_axis()
    )
    assert chart.shape == (2, FLANGE_CANONICAL_DIM)
    assert np.isfinite(chart).all()


def test_metric_bisection_units_are_metres():
    """Stencil m_gt must be declared-metric metres, not mm/deg mix."""
    torch = pytest.importorskip("torch")
    from ird_playground.ird.metric import LAMBDA_M_PER_RAD, se3_distance_m, se3_distance_m_torch
    from ird_playground.ird.gpu_boundary_stencil import GpuBoundaryStencilConfig

    cfg = GpuBoundaryStencilConfig(metric_offsets_m=(0.001, 0.003, 0.006, 0.010))
    offs = cfg.resolved_metric_offsets_m()
    assert offs == (0.001, 0.003, 0.006, 0.010)
    # 1 cm translation ≡ 1 deg rotation under λ.
    d_pos = se3_distance_m(np.array([0.01, 0.0, 0.0]), np.zeros(3))
    d_rot = se3_distance_m(
        np.zeros(3), np.array([0.0, 0.0, np.deg2rad(1.0)])
    )
    assert abs(d_pos - 0.01) < 1e-12
    assert abs(d_rot - 0.01) < 1e-9
    assert abs(d_pos - d_rot) < 1e-9
    assert abs(LAMBDA_M_PER_RAD * np.deg2rad(1.0) - 0.01) < 1e-12

    dp = torch.tensor([[0.03, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    dw = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, float(np.deg2rad(3.0))]], dtype=torch.float32
    )
    d = se3_distance_m_torch(dp, dw)
    assert abs(float(d[0]) - 0.03) < 1e-6
    assert abs(float(d[1]) - 0.03) < 1e-6


def test_legacy_offset_fallback_converts_to_metres():
    from ird_playground.ird.gpu_boundary_stencil import GpuBoundaryStencilConfig
    from ird_playground.ird.metric import LAMBDA_M_PER_RAD

    cfg = GpuBoundaryStencilConfig(
        metric_offsets_m=(),
        position_offsets_mm=(1.0, 6.0),
        rotation_offsets_deg=(1.0,),
    )
    offs = cfg.resolved_metric_offsets_m()
    assert 0.001 in offs
    assert 0.006 in offs
    assert any(abs(x - LAMBDA_M_PER_RAD * np.deg2rad(1.0)) < 1e-12 for x in offs)


def test_flange_tensor_self_collision_kwargs_and_relative_azimuth():
    """Builder accepts SelfCollisionFilter kwargs; azimuth is relative to p_xy."""
    from ird_playground.map.build_flange_tensor import flange_pose_to_chart

    # Non-polar flange z so absolute uz azimuth tracks base yaw with p_xy.
    tilt = np.deg2rad(35.0)
    ct, st = np.cos(tilt), np.sin(tilt)
    R0 = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]], dtype=np.float64)
    p0 = np.array([0.3, 0.0, 0.4], dtype=np.float64)
    chart0 = flange_pose_to_chart(p0, R0)
    phi = np.deg2rad(40.0)
    c, s = np.cos(phi), np.sin(phi)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    chart1 = flange_pose_to_chart(Rz @ p0, Rz @ R0)
    assert np.allclose(chart0, chart1, atol=1e-5)


def test_self_collision_filter_kwargs_match_api():
    """Regression: build_flange_tensor must use keyword security_margin."""
    import inspect

    from rm75_control.tools.reachability.build.self_collision import SelfCollisionFilter

    params = inspect.signature(SelfCollisionFilter.__init__).parameters
    assert "security_margin" in params
    assert "security_margin_m" not in params
