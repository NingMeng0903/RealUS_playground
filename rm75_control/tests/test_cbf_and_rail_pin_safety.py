"""Focused safety regressions for collision witness points and rail pins."""

from types import SimpleNamespace

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionPairInfo
from rm75_control.control.joint_admittance_8dof.solver.cbf_constraints import (
    FrameJacobian,
    collision_jacobian,
)
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    VelocityBoxConstraints,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


def test_collision_jacobian_uses_nearest_point_angular_velocity() -> None:
    geom_model = SimpleNamespace(
        geometryObjects=[
            SimpleNamespace(parentFrame=10),
            SimpleNamespace(parentFrame=20),
        ]
    )
    J_a = np.zeros((6, 2), dtype=float)
    J_a[5, 0] = 1.0
    frame_jacs = {
        10: FrameJacobian(jacobian=J_a, origin=np.zeros(3)),
        20: FrameJacobian(jacobian=np.zeros((6, 2)), origin=np.zeros(3)),
    }
    pair = CollisionPairInfo(
        pair_index=0,
        geom_a=0,
        geom_b=1,
        name_a="a",
        name_b="b",
        distance=0.02,
        normal=np.array([1.0, 0.0, 0.0]),
        point_a=np.array([0.0, 2.0, 0.0]),
        point_b=np.zeros(3),
    )
    assert np.allclose(collision_jacobian(frame_jacs, geom_model, pair), [-2.0, 0.0])


def _limits(*, a_max: float | None = None) -> SafetyLimits:
    acceleration = None if a_max is None else np.full(8, a_max, dtype=float)
    return SafetyLimits(
        q_lower=np.full(8, -1.0),
        q_upper=np.full(8, 1.0),
        v_max=np.array([0.1, *([1.0] * 7)]),
        a_max=acceleration,
        position_margin=np.array([0.05, *([0.01] * 7)]),
    )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"rail_vel_pin_m_s": 0.5}, 0.1),
        ({"qdot_prev": np.zeros(8), "rail_vel_pin_m_s": -0.08}, -0.02),
    ],
)
def test_rail_velocity_pin_intersects_safety_box(kwargs, expected) -> None:
    a_max = 0.2 if "qdot_prev" in kwargs else None
    box = VelocityBoxConstraints(_limits(a_max=a_max), damper_band_rad=0.0)
    lo, hi = box.bounds(np.zeros(8), dt=0.1, **kwargs)
    assert lo[0] == pytest.approx(expected)
    assert hi[0] == pytest.approx(expected)


def test_rail_velocity_pin_cannot_point_into_position_limit() -> None:
    box = VelocityBoxConstraints(_limits(), damper_band_rad=0.0)
    q = np.zeros(8)
    q[0] = 0.95
    lo, hi = box.bounds(q, dt=0.1, rail_vel_pin_m_s=0.08)
    assert lo[0] == pytest.approx(0.0)
    assert hi[0] == pytest.approx(0.0)


def test_nonfinite_rail_velocity_pin_is_rejected() -> None:
    box = VelocityBoxConstraints(_limits(), damper_band_rad=0.0)
    with pytest.raises(ValueError, match="must be finite"):
        box.bounds(np.zeros(8), dt=0.1, rail_vel_pin_m_s=np.nan)


def test_locked_rail_preserves_required_limit_recovery() -> None:
    box = VelocityBoxConstraints(_limits(), damper_band_rad=0.0)
    q = np.zeros(8)
    q[0] = 1.02
    lo, hi = box.bounds(
        q, dt=0.1, rail_locked=True, rail_lock_vel_eps_m_s=0.0
    )
    assert lo[0] == pytest.approx(-0.1)
    assert hi[0] == pytest.approx(-0.1)
