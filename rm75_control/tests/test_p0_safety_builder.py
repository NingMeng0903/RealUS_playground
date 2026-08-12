from types import SimpleNamespace

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.generic_tasks import (
    HardConstraintRow,
    RobotState,
)
from rm75_control.control.joint_admittance_8dof.solver.p0_safety import P0SafetyBuilder
from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import CbfRows
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


def test_builder_uses_measured_q_and_appends_named_application_row() -> None:
    kin = SimpleNamespace(nv=2)
    limits = SafetyLimits(
        q_lower=np.full(2, -1.0),
        q_upper=np.full(2, 1.0),
        v_max=np.ones(2),
        a_max=None,
        position_margin=np.zeros(2),
    )
    builder = P0SafetyBuilder(
        kin,
        limits,
        collision_config=CollisionConfig(enabled=False),
        damper_band=0.0,
    )
    state = RobotState(
        q_meas=np.zeros(2),
        q_cmd=np.array([0.3, 0.0]),
        qdot_applied_prev=np.zeros(2),
        dt=0.01,
        contact_active=True,
    )
    row = HardConstraintRow([1.0, 1.0], upper=0.0, name="do_not_advance")
    constraints = builder.build(
        state, resync_err=np.array([0.2, 0.2]), application_rows=(row,)
    )
    assert constraints.names[-1] == "do_not_advance"
    assert constraints.upper[0] == 0.0
    assert np.allclose(constraints.C[-1], [1.0, 1.0])


def test_collision_enable_after_disabled_constructs_and_uses_model(monkeypatch) -> None:
    """A runtime enable must not silently leave the collision CBF absent."""

    import rm75_control.control.joint_admittance_8dof.solver.p0_safety as module

    built: list[object] = []

    class FakeCollisionModel:
        def __init__(self, model, **_kwargs):
            built.append(model)

    def fake_build(_collision, kin, _q, _cfg, *, tracker, kinematics_ready=False):
        assert tracker is not None
        assert not kinematics_ready
        return CbfRows(
            jacobian=np.array([[1.0, -1.0]]),
            lower=np.array([0.025]),
            slot_index=np.array([1]),
            names=("self_collision:test_pair",),
        )

    monkeypatch.setattr(module, "CollisionModel", FakeCollisionModel)
    monkeypatch.setattr(module, "build_cbf_rows", fake_build)
    kin = SimpleNamespace(nv=2, model=object())
    limits = SafetyLimits(
        q_lower=np.full(2, -1.0),
        q_upper=np.full(2, 1.0),
        v_max=np.ones(2),
        a_max=None,
        position_margin=np.zeros(2),
    )
    builder = P0SafetyBuilder(
        kin,
        limits,
        collision_config=CollisionConfig(enabled=False, max_pairs=3),
        damper_band=0.0,
    )
    assert builder.collision is None
    builder.set_collision_enabled(True)
    assert len(built) == 1

    state = RobotState(
        q_meas=np.zeros(2),
        q_cmd=np.zeros(2),
        qdot_applied_prev=np.zeros(2),
        dt=0.01,
        contact_active=False,
    )
    constraints = builder.build(state)
    row = constraints.names.index("self_collision:test_pair")
    # The fake pair requested sticky slot 1, not the first compacted slot.
    assert row == 2 + 1
    np.testing.assert_allclose(constraints.C[row], [1.0, -1.0])
    assert constraints.lower[row] == 0.025
