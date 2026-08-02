from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ird_playground.optimization import (
    TrajectoryOptimizationConfig,
    TrajectoryOptimizationProblem,
    WorldConstraint,
    optimize_trajectory,
)


class _LinearEightDof:
    q_lower = np.full(8, -1.0)
    q_upper = np.full(8, 1.0)
    v_max = np.full(8, 1.0)

    def fk_matrix(self, q):
        T = np.eye(4)
        T[:3, 3] = q[:3]
        T[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        return T

    def jacobian(self, q):
        del q
        return np.c_[np.eye(6), np.zeros((6, 2))]

    def collision_rows(self, q, **kwargs):
        del q, kwargs
        return []


def _problem(world_constraints=()):
    T = np.tile(np.eye(4), (3, 1, 1))
    T[:, 0, 3] = [0.0, 0.01, 0.02]
    seed = np.zeros((3, 8))
    seed[:, 0] = [0.001, 0.011, 0.021]
    return TrajectoryOptimizationProblem(
        s=np.linspace(0.0, 1.0, 3),
        T_tcp_nominal=T,
        q_seeds=(seed,),
        position_tolerance_m=0.003,
        rotation_tolerance_rad=0.1,
        world_constraints=world_constraints,
    )


def test_sqp_returns_valid_8dof_path_and_never_exceeds_scan_speed():
    result = optimize_trajectory(
        _problem(),
        config=TrajectoryOptimizationConfig(max_iterations=10, collision_safe_m=0.0),
        kinematics=_LinearEightDof(),
    )
    assert result.valid
    assert result.q_ref.shape == (3, 8)
    assert np.array_equal(result.rail_ref, result.q_ref[:, 0])
    speed = np.linalg.norm(np.diff(result.T_tcp_ref[:, :3, 3], axis=0), axis=1) / np.diff(result.timestamps)
    assert np.all(speed <= 0.02 + 1.0e-9)
    assert result.validation["max_fk_position_m"] <= 5.0e-4


def test_sqp_fails_closed_when_world_constraint_is_impossible():
    impossible = WorldConstraint("patient", lambda q, T: -np.ones(1))
    result = optimize_trajectory(
        _problem((impossible,)),
        config=TrajectoryOptimizationConfig(max_iterations=3, collision_safe_m=0.0),
        kinematics=_LinearEightDof(),
    )
    assert not result.valid
    assert np.isnan(result.q_ref).all()
