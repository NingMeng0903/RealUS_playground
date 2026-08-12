"""Unit tests for immutable generic task value objects."""

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.generic_tasks import (
    HardConstraintRow,
    LinearConstraintSet,
    OneSidedConstraint,
    PostureGuide,
    ProtectedTask,
    RobotState,
    ScalableTask,
)


def test_robot_state_and_task_arrays_are_snapshots() -> None:
    q = np.array([1.0, 2.0])
    state = RobotState(q, q + 1.0, np.zeros(2), 0.005, False)
    q[:] = 99.0
    assert np.allclose(state.q_meas, [1.0, 2.0])
    assert not state.q_meas.flags.writeable
    with pytest.raises(ValueError):
        state.q_meas[0] = 0.0

    task = ScalableTask(np.eye(2), [1.0, 2.0], "cart")
    assert not task.A.flags.writeable
    with pytest.raises(ValueError):
        task.A[0, 0] = 0.0


def test_task_shapes_finite_and_positive_scales_are_checked() -> None:
    with pytest.raises(ValueError, match="(shape|length)"):
        RobotState([0.0, 1.0], [0.0], [0.0, 0.0], 0.01, False)
    with pytest.raises(ValueError, match="finite"):
        ScalableTask([[1.0, np.nan]], [0.0], "x")
    with pytest.raises(ValueError, match="row_scales"):
        ProtectedTask([[1.0, 0.0]], [0.0], row_scales=[0.0])


def test_hard_rows_one_sided_and_linear_bounds_allow_open_sides() -> None:
    row = OneSidedConstraint([1.0, 0.0], 0.5, ">=", name="lower")
    assert row.lower == pytest.approx(0.5)
    assert row.upper is None
    rows = LinearConstraintSet(
        [[1.0, 0.0], [0.0, 1.0]],
        [-np.inf, -1.0],
        [0.5, np.inf],
        ["x", "y"],
    )
    assert rows.row(0).upper == pytest.approx(0.5)
    protected = ProtectedTask(
        np.eye(2), [0.0, 0.0], one_sided_constraints=(row,)
    )
    assert protected.one_sided_constraints[0].name == "lower"


def test_posture_guide_validity_and_shape() -> None:
    guide = PostureGuide([0.0, 1.0], [0.1, 0.2], 2.0, 0.8, "RECOVERY", created_at=1.0)
    assert guide.is_valid(2.0)
    assert not guide.is_valid(2.1)
    assert guide.age(1.5) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="quality"):
        PostureGuide([0.0], [0.0], 1.0, 1.1, "x")
