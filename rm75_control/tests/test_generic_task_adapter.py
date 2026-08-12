import numpy as np
from scipy.spatial.transform import Rotation

from rm75_control.control.joint_admittance_8dof.task_adapter import (
    CartesianTaskProfile,
    TaskSpaceConstraintRow,
    build_cartesian_tasks,
)


def test_arbitrary_rotated_task_frame_and_axis_selection() -> None:
    rotation = Rotation.from_rotvec(np.array([0.3, -0.4, 0.2])).as_matrix()
    J = np.arange(48, dtype=float).reshape(6, 8) / 50.0
    twist = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
    profile = CartesianTaskProfile.from_indices(
        protected=(0, 4), scalable=(("position", (1, 2)), ("orientation", (3, 5)))
    )
    protected, scalable = build_cartesian_tasks(J, twist, rotation, profile)
    X = np.zeros((6, 6))
    X[:3, :3] = rotation.T
    X[3:, 3:] = rotation.T
    assert np.allclose(protected.A, (X @ J)[[0, 4]])
    assert np.allclose(protected.b, twist[[0, 4]])
    assert [task.scale_group_id for task in scalable] == ["position", "orientation"]


def test_one_sided_row_direction_is_projected_in_task_frame() -> None:
    rotation = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    J = np.eye(6, 8)
    profile = CartesianTaskProfile.from_indices(protected=(3, 4, 5), scalable=())
    row = TaskSpaceConstraintRow(
        coefficients=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        upper=0.0,
        name="do_not_advance",
    )
    protected, _ = build_cartesian_tasks(
        J, np.zeros(6), rotation, profile, one_sided_rows=(row,)
    )
    assert np.allclose(protected.one_sided_constraints[0].a, (rotation.T @ J[:3])[0])
    assert protected.one_sided_constraints[0].upper == 0.0


def test_zero_protected_and_three_orientation_axes_are_both_valid() -> None:
    J = np.eye(6, 7)
    none = CartesianTaskProfile.from_indices(protected=(), scalable=((0, range(6)),))
    p0, scalable = build_cartesian_tasks(J, np.zeros(6), np.eye(3), none)
    assert p0.A.shape == (0, 7)
    assert scalable[0].A.shape == (6, 7)

    orientation = CartesianTaskProfile.from_indices(protected=(3, 4, 5), scalable=())
    p3, _ = build_cartesian_tasks(J, np.zeros(6), np.eye(3), orientation)
    assert p3.A.shape == (3, 7)
