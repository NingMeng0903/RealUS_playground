"""fce787a9 implicit-Euler law: bidirectional yield, no residual bypass."""

from __future__ import annotations

import numpy as np
import pytest

from peirastic.realman8dof.force.fce import FceAdmittanceLaw, use_fce_law


def _law() -> FceAdmittanceLaw:
    return FceAdmittanceLaw(
        0.005,
        force_axes=np.ones(6),
        mass=np.array([1.0, 1.0, 1.0, 0.04, 0.04, 0.04]),
        damping=np.array([25.0, 25.0, 25.0, 1.0, 1.0, 1.0]),
        deadband=np.zeros(6),
        deadband_width=np.zeros(6),
        max_velocity=np.array([0.22, 0.22, 0.08, 0.6, 0.6, 0.6]),
    )


def test_use_fce_law_flag() -> None:
    assert use_fce_law({"law": "fce"}) is True
    assert use_fce_law({"law": "fce787a9"}) is True
    assert use_fce_law({"law": "tff"}) is False
    assert use_fce_law({}) is False


def test_zero_wrench_is_still() -> None:
    law = _law()
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    out = law.update(
        dt_s=0.005,
        pose=np.zeros(6),
        f_ext=np.zeros(6),
        f_des=np.zeros(6),
        path_twist=np.zeros(6),
    )
    assert np.linalg.norm(out.v_force) < 1e-9


def test_push_and_pull_both_yield() -> None:
    law = _law()
    plus = law.update(
        dt_s=0.005,
        pose=np.zeros(6),
        f_ext=np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0]),
        f_des=np.zeros(6),
        path_twist=np.zeros(6),
    )
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    minus = law.update(
        dt_s=0.005,
        pose=np.zeros(6),
        f_ext=np.array([0.0, 0.0, -3.0, 0.0, 0.0, 0.0]),
        f_des=np.zeros(6),
        path_twist=np.zeros(6),
    )
    assert plus.v_force[2] < 0.0
    assert minus.v_force[2] > 0.0
    assert plus.v_force[2] == pytest.approx(-minus.v_force[2])


def test_no_residual_bypass_when_force_matches() -> None:
    law = _law()
    out = law.update(
        dt_s=0.005,
        pose=np.zeros(6),
        f_ext=np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
        f_des=np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
        path_twist=np.zeros(6),
    )
    assert abs(out.v_force[2]) < 1e-9


def _hover_law() -> FceAdmittanceLaw:
    return FceAdmittanceLaw.from_payload(
        0.005,
        {
            "law": "fce",
            "force_axes": [1, 1, 1, 1, 1, 1],
            "desired_force": [0.0] * 6,
        },
    )


def _step(law: FceAdmittanceLaw, f_ext: np.ndarray):
    return law.update(
        dt_s=0.005,
        pose=np.zeros(6),
        f_ext=np.asarray(f_ext, dtype=float),
        f_des=np.zeros(6),
        path_twist=np.zeros(6),
    )


def test_hover_residual_is_clutched() -> None:
    law = _hover_law()
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    residual = np.array([0.05, -0.04, 0.06, 0.008, -0.006, 0.004])
    out = _step(law, residual)
    assert np.linalg.norm(out.v_force) < 1e-9
    assert np.linalg.norm(law._bias[:3]) > 0.05


def test_hover_plant_leftover_does_not_walk() -> None:
    """ID leftover (~0.35 N / 0.03 Nm) must seed bias, not integrate away."""

    law = _hover_law()
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    leftover = np.array([0.20, -0.15, 0.35, 0.030, -0.018, 0.012])
    for _ in range(80):
        out = _step(law, leftover)
        assert np.linalg.norm(out.v_force[:3]) < 1e-9
        assert np.linalg.norm(out.v_force[3:6]) < 1e-9
    np.testing.assert_allclose(law._bias, leftover, atol=1e-9)


def test_hover_hand_push_still_yields() -> None:
    law = _hover_law()
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    plus = _step(law, np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0]))
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    minus = _step(law, np.array([0.0, 0.0, -3.0, 0.0, 0.0, 0.0]))
    assert plus.v_force[2] < -1e-3
    assert minus.v_force[2] > 1e-3
    assert plus.v_force[2] == pytest.approx(-minus.v_force[2], rel=0.05)


def test_hover_rotation_is_light() -> None:
    law = _hover_law()
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    leftover = np.array([0.20, -0.15, 0.35, 0.030, -0.018, 0.012])
    for _ in range(60):
        _step(law, leftover)
    twist = leftover + np.array([0.0, 0.0, 0.0, 0.12, 0.0, 0.0])
    out = None
    for _ in range(20):
        out = _step(law, twist)
    assert out is not None
    assert out.v_force[3] < -0.15
    assert abs(out.v_force[3]) > abs(out.v_force[4])
    assert abs(out.v_force[3]) > abs(out.v_force[5])
