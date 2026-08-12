"""Frame/shape coverage for generic two-level QPIK."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.generic_tasks import ProtectedTask, RobotState, ScalableTask
from rm75_control.control.joint_admittance_8dof.solver.two_level_qpik import (
    TwoLevelQpikConfig,
    TwoLevelQpikController,
)


@pytest.mark.parametrize("n", [7, 8])
@pytest.mark.parametrize("n_rot", [0, 1, 2, 3])
def test_arbitrary_rotation_rows_and_dof(n: int, n_rot: int) -> None:
    cfg = TwoLevelQpikConfig(
        backend="scipy",
        qdot_lower=-np.ones(n),
        qdot_upper=np.ones(n),
        max_rows=64,
        max_scalable_groups=4,
    )
    c = TwoLevelQpikController(n, cfg)
    state = RobotState(np.zeros(n), np.zeros(n), np.zeros(n), 0.01, False)
    # Rotation rows can be any independent basis; n_rot=0 is a valid empty
    # orientation block, while the remaining rows provide protected position.
    A = np.zeros((3 + n_rot, n))
    for i in range(3 + n_rot):
        A[i, (i + 1) % n] = 1.0
    b = np.linspace(-0.2, 0.2, A.shape[0])
    result = c.solve(state, ProtectedTask(A, b, name="frame"))
    assert result.qdot.shape == (n,)
    assert result.protected_achieved.shape == b.shape
    assert result.qp1.success and result.qp2.success


def test_non_identity_frame_rows_preserve_target() -> None:
    theta = 0.61
    frame = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    n = 8
    A = np.zeros((2, n))
    A[:, :2] = frame
    c = TwoLevelQpikController(
        n,
        TwoLevelQpikConfig(
            backend="scipy",
            qdot_lower=-np.ones(n),
            qdot_upper=np.ones(n),
            max_rows=64,
            max_scalable_groups=4,
        ),
    )
    state = RobotState(np.zeros(n), np.zeros(n), np.zeros(n), 0.01, False)
    p = ProtectedTask(A, [0.15, -0.1], row_scales=[0.5, 0.25], name="rotated_tcp")
    r = c.step(state, p, [ScalableTask(np.eye(n)[2:3], [0.4], "rail")])
    assert np.linalg.norm(r.protected_achieved - r.protected_locked_output) < 2e-5
    assert np.all(np.isfinite(r.qdot))

