import numpy as np

from rm75_control.control.joint_admittance_8dof.health_metrics import (
    compute_health_metrics,
    velocity_normalized_arm_health,
)


def test_arm_health_excludes_rail_column() -> None:
    J = np.eye(6, 7)
    full = np.column_stack((np.full(6, 1e6), J))
    metrics = compute_health_metrics(
        jacobian_base=full,
        q_meas=np.zeros(8),
        q_lower=np.full(8, -1.0),
        q_upper=np.full(8, 1.0),
        velocity_limits=np.ones(8),
        rail_indices=(0,),
        wrist_indices=(6, 7),
    )
    assert metrics.arm_health == 1.0


def test_health_is_invariant_to_consistent_physical_unit_scaling() -> None:
    J = np.diag([0.1, 0.2, 0.3])
    base = velocity_normalized_arm_health(
        J, np.ones(3), task_velocity_scales=np.array([0.1, 0.1, 0.1])
    )
    rescaled = velocity_normalized_arm_health(
        1000.0 * J,
        np.ones(3),
        task_velocity_scales=np.array([100.0, 100.0, 100.0]),
    )
    assert np.isclose(base, rescaled)
