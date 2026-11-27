"""Contract tests for the independent ProDMP motion layer.

These tests do not instantiate a robot or alter the native QP.  They check
the mathematical trajectory, its finite-difference/ODE agreement, the common
SE(3) decode, and replan boundary continuity.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from motion import (  # noqa: E402
    LAMBDA,
    ProDMPReference,
    duration_matrix,
    reference,
)


def _anchor() -> np.ndarray:
    R = Rotation.from_euler("xyz", [0.23, -0.31, 0.47]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [0.42, -0.18, 0.73]
    return T


def test_constraint_matrix_rank_and_endpoint_contract() -> None:
    ref = reference([0.01, -0.02, 0.12], [0.035, 0.008, -0.16], np.eye(4), duration=1.7)
    assert ref.rank == 2
    assert np.max(np.abs(ref.constraint_residual)) < 2.e-10
    p0, v0 = ref.local_position_velocity(0.)
    p1, v1 = ref.local_position_velocity(ref.duration_s())
    np.testing.assert_allclose(p0, ref.start, atol=1.e-12)
    np.testing.assert_allclose(p1, ref.goal, atol=1.e-12)
    np.testing.assert_allclose(v0, 0., atol=1.e-12)
    np.testing.assert_allclose(v1, 0., atol=1.e-12)


def test_declared_nonzero_endpoint_velocities_are_respected() -> None:
    start_v = np.array([0.002, -0.003, 0.017])
    end_v = np.array([-0.001, 0.004, -0.013])
    ref = reference([0., 0., 0.], [0.02, -0.01, 0.18], np.eye(4), duration=2.0,
                    start_velocity=start_v, end_velocity=end_v)
    np.testing.assert_allclose(ref.local_position_velocity(0.)[1], start_v, atol=1.e-12)
    np.testing.assert_allclose(ref.local_position_velocity(2.)[1], end_v, atol=1.e-12)


def test_integral_basis_matches_independent_phase_ode() -> None:
    start = np.array([0.005, -0.004, 0.11])
    goal = np.array([0.026, 0.017, -0.19])
    tau = 1.35
    ref = reference(start, goal, np.eye(4), duration=tau)
    grid = np.linspace(0., 1., 121)
    y0 = np.r_[start, np.zeros(3)]

    def rhs(s: float, state: np.ndarray) -> np.ndarray:
        pos, vel = state[:3], state[3:]
        forcing = np.asarray(__import__("motion")._phi(s))[0] @ ref.weights
        acc = forcing - 2. * LAMBDA * vel - LAMBDA ** 2 * (pos - goal)
        return np.r_[vel, acc]

    sol = solve_ivp(rhs, (0., 1.), y0, t_eval=grid, rtol=2.e-10, atol=2.e-12, method="DOP853")
    assert sol.success
    analytic = np.asarray([ref.local_state(float(s * tau)).position for s in grid])
    analytic_d = np.asarray([ref.local_state(float(s * tau)).derivative for s in grid])
    np.testing.assert_allclose(sol.y[:3].T, analytic, atol=3.e-8, rtol=3.e-8)
    np.testing.assert_allclose(sol.y[3:].T, analytic_d, atol=3.e-8, rtol=3.e-8)


def test_independent_finite_difference_matches_physical_velocity() -> None:
    ref = reference([-0.01, 0.008, -0.2], [0.018, -0.014, 0.16], np.eye(4), duration=1.8)
    t = np.linspace(0.03, ref.duration_s() - 0.03, 101)
    eps = 1.e-5
    fd = np.asarray([(ref.local_position_velocity(float(x + eps))[0] -
                      ref.local_position_velocity(float(x - eps))[0]) / (2. * eps) for x in t])
    vel = np.asarray([ref.local_position_velocity(float(x))[1] for x in t])
    np.testing.assert_allclose(fd, vel, atol=2.e-8, rtol=2.e-6)


def test_common_anchor_decodes_pose_and_full_world_twist() -> None:
    anchor = _anchor()
    start = np.array([0.01, -0.02, -0.2])
    goal = np.array([0.03, 0.015, 0.25])
    ref = reference(start, goal, anchor, duration=1.6)
    t = 0.61
    local_p, local_v = ref.local_position_velocity(t)
    sample = ref.sample(t)
    T_local = np.eye(4)
    T_local[:3, :3] = Rotation.from_rotvec([0., 0., local_p[2]]).as_matrix()
    T_local[:3, 3] = [local_p[0], local_p[1], 0.]
    T_world = anchor @ T_local
    expected_pose = np.r_[T_world[:3, 3], Rotation.from_matrix(T_world[:3, :3]).as_euler("xyz")]
    R_anchor = anchor[:3, :3]
    expected_twist = np.r_[R_anchor @ np.r_[local_v[:2], 0.], R_anchor @ [0., 0., local_v[2]]]
    np.testing.assert_allclose(sample.pose_d, expected_pose, atol=2.e-12)
    np.testing.assert_allclose(sample.vel_ff, expected_twist, atol=2.e-12)


def test_replan_boundary_is_pose_and_velocity_continuous() -> None:
    anchor = _anchor()
    mid = np.array([0.014, 0.003, 0.09])
    first = reference([-0.005, -0.009, -0.1], mid, anchor, duration=1.1,
                     end_velocity=[0., 0., 0.])
    second = reference(mid, [0.027, 0.019, 0.22], anchor, duration=1.4,
                       start_velocity=[0., 0., 0.])
    a = first.sample(first.duration_s())
    b = second.sample(0.)
    np.testing.assert_allclose(a.pose_d, b.pose_d, atol=1.e-12)
    np.testing.assert_allclose(a.vel_ff, b.vel_ff, atol=1.e-12)


def test_duration_matrices_are_finite_and_dmp_is_an_explicit_baseline() -> None:
    poses = np.array([[0., 0., 0.], [0.01, 0.005, 0.10], [-0.015, 0.012, -0.16]])
    prodmp = duration_matrix(poses, method="prodmp")
    dmp = duration_matrix(poses, method="dmp")
    assert prodmp.shape == (3, 3)
    assert np.isfinite(prodmp).all() and np.isfinite(dmp).all()
    np.testing.assert_allclose(np.diag(prodmp), 0.1, atol=0.)
    np.testing.assert_allclose(np.diag(dmp), 0.1, atol=0.)
    assert np.all(prodmp[np.triu_indices(3, 1)] > 0.1)
    assert np.all(dmp[np.triu_indices(3, 1)] > 0.1)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
