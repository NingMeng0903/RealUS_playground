from types import SimpleNamespace

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.generic_tasks import (
    HardConstraintRow,
    RobotState,
)
from rm75_control.control.joint_admittance_8dof.solver.p0_safety import (
    CollisionHardCapacityExceeded,
    P0SafetyBuilder,
)
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
    """Runtime enable builds hard and recoverable collision rows separately."""

    import rm75_control.control.joint_admittance_8dof.solver.p0_safety as module

    built: list[object] = []

    class FakeCollisionModel:
        def __init__(self, model, **_kwargs):
            built.append(model)
            self.model = model
            self.geom_model = object()
            self._kin_data = object()

        def update(self, _q, **_kwargs):
            return None

        def active_pairs(self, _distance):
            return [
                SimpleNamespace(
                    geom_a=1,
                    geom_b=2,
                    distance=0.02,
                    name_a="hard_a",
                    name_b="hard_b",
                ),
                SimpleNamespace(
                    geom_a=3,
                    geom_b=4,
                    distance=0.06,
                    name_a="warn_a",
                    name_b="warn_b",
                ),
            ]

    monkeypatch.setattr(module, "CollisionModel", FakeCollisionModel)
    monkeypatch.setattr(
        module,
        "_frame_linear_jacobians",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        module,
        "collision_jacobian",
        lambda _frames, _geometry, pair: (
            np.array([1.0, -1.0])
            if pair.geom_a == 1
            else np.array([-0.5, 0.5])
        ),
    )
    kin = SimpleNamespace(nv=2, model=object(), data=object())
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
        collision_config=CollisionConfig(
            enabled=False, d_safe=0.03, d_activate=0.08, gamma=5.0, max_pairs=3
        ),
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
    row = constraints.names.index("self_collision:hard_a:hard_b")
    assert row == 2
    np.testing.assert_allclose(constraints.C[row], [1.0, -1.0])
    assert constraints.lower[row] == pytest.approx(0.05)
    assert "self_collision_warning:warn_a:warn_b" in (
        builder.last_collision_warning_names
    )
    warning_slot = builder.last_collision_warning_names.index(
        "self_collision_warning:warn_a:warn_b"
    )
    np.testing.assert_allclose(
        builder.last_collision_warning_C[warning_slot], [-0.5, 0.5]
    )
    assert builder.last_collision_warning_lower[warning_slot] == pytest.approx(-0.15)


def test_absolute_collision_overflow_is_an_explicit_failure(monkeypatch) -> None:
    import rm75_control.control.joint_admittance_8dof.solver.p0_safety as module

    pairs = [
        SimpleNamespace(
            geom_a=index,
            geom_b=index + 10,
            distance=0.01,
            name_a=f"a{index}",
            name_b=f"b{index}",
        )
        for index in range(3)
    ]
    collision = SimpleNamespace(
        update=lambda *_args, **_kwargs: None,
        active_pairs=lambda _distance: pairs,
    )
    kin = SimpleNamespace(nv=2, model=object(), data=object())
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
        collision=collision,
        collision_config=CollisionConfig(enabled=True, max_pairs=2),
        damper_band=0.0,
    )
    state = RobotState(
        q_meas=np.zeros(2),
        q_cmd=np.zeros(2),
        qdot_applied_prev=np.zeros(2),
        dt=0.01,
        contact_active=False,
    )
    with pytest.raises(CollisionHardCapacityExceeded, match="3 > 2"):
        builder.build(state)
