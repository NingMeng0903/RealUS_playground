from types import SimpleNamespace

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.generic_runtime import (
    GenericQpikRuntime,
    GenericQpikRuntimeConfig,
)
from rm75_control.control.joint_admittance_8dof.generic_tasks import (
    ProtectedTask,
    RobotState,
    ScalableTask,
)
from rm75_control.control.joint_admittance_8dof.solver.two_level_qpik import (
    TwoLevelQpikConfig,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


class _Kinematics:
    nv = 7

    @staticmethod
    def jacobian(q):
        del q
        return np.eye(6, 7)


def _runtime() -> GenericQpikRuntime:
    limits = SafetyLimits(
        q_lower=np.full(7, -2.0),
        q_upper=np.full(7, 2.0),
        v_max=np.ones(7),
        a_max=np.full(7, 20.0),
        position_margin=np.zeros(7),
    )
    config = GenericQpikRuntimeConfig(
        solver=TwoLevelQpikConfig(
            backend="scipy",
            max_rows=64,
            max_scalable_groups=4,
        ),
        rail_indices=(),
        wrist_indices=(4, 5, 6),
    )
    return GenericQpikRuntime(
        _Kinematics(),
        limits,
        config,
        collision_config=CollisionConfig(enabled=False),
        damper_band=0.0,
    )


def _state() -> RobotState:
    return RobotState(
        q_meas=np.zeros(7),
        q_cmd=np.zeros(7),
        qdot_applied_prev=np.zeros(7),
        dt=0.005,
        contact_active=False,
    )


def test_direct_generic_rows_do_not_require_cartesian_axis_semantics() -> None:
    runtime = _runtime()
    protected = ProtectedTask(
        np.array([[0.2, -0.4, 0.3, 0.0, 0.1, 0.0, 0.5]]),
        np.array([0.08]),
        row_scales=np.array([0.1]),
        name="application_row",
    )
    scalable = ScalableTask(
        np.array([[0.0, 0.1, 0.0, 0.7, -0.2, 0.4, 0.0]]),
        np.array([0.12]),
        "arbitrary_group",
        slack_limits=np.array([0.02]),
    )
    result = runtime.solve_tasks(
        _state(), protected=protected, scalable=(scalable,)
    )
    assert result.solver.qp1.success
    assert result.solver.qp2.success
    np.testing.assert_allclose(
        protected.A @ result.qdot,
        result.solver.protected_locked_output,
        atol=7e-6,
    )
    assert "arbitrary_group" in result.solver.group_alphas


def test_runtime_reuses_one_measured_state_jacobian_snapshot() -> None:
    kin = _Kinematics()
    calls = 0

    def jacobian(q):
        nonlocal calls
        calls += 1
        return np.eye(6, 7)

    kin.jacobian = jacobian
    runtime = _runtime()
    runtime.kin = kin
    runtime.p0_builder.kin = kin
    protected = ProtectedTask(np.eye(7)[:1], [0.0])
    runtime.solve_tasks(_state(), protected=protected)
    assert calls == 1
