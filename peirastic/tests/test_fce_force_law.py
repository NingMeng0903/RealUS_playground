"""fce787a9 implicit-Euler law: bidirectional yield, no residual bypass."""

from __future__ import annotations

import math

import numpy as np
import pytest

from peirastic.realman8dof.force.fce import (
    FceAdmittanceLaw,
    _moment_not_from_force,
    kikuuwe_step,
    poinsot_axis,
    project_tool_cylinder,
    use_fce_law,
)


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
    plus = None
    for _ in range(8):
        plus = _step(law, np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0]))
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    minus = None
    for _ in range(8):
        minus = _step(law, np.array([0.0, 0.0, -3.0, 0.0, 0.0, 0.0]))
    assert plus is not None and minus is not None
    assert plus.v_force[2] < -1e-3
    assert minus.v_force[2] > 1e-3
    assert plus.v_force[2] == pytest.approx(-minus.v_force[2], rel=0.05)


def test_moment_perp_force_is_dropped() -> None:
    f = np.array([2.0, 0.0, 0.0])
    r = np.array([0.0, 0.0, 0.12])
    m_lever = np.cross(r, f)
    dropped = _moment_not_from_force(f, m_lever, f_min=0.25, f_full=0.80)
    np.testing.assert_allclose(dropped, 0.0, atol=1e-12)
    couple = np.array([0.15, 0.0, 0.0])
    kept = _moment_not_from_force(f, m_lever + couple, f_min=0.25, f_full=0.80)
    np.testing.assert_allclose(kept, couple, atol=1e-12)


def test_hover_lateral_force_does_not_spin() -> None:
    law = _hover_law()
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    for _ in range(60):
        _step(law, np.zeros(6))
    # Point force 12 cm off TCP along z: M = r × F is ⊥ F.
    wrench = np.array([2.0, 0.0, 0.0, 0.0, 0.24, 0.0])
    out = None
    for _ in range(25):
        out = _step(law, wrench)
    assert out is not None
    assert out.v_force[0] < -1e-3
    assert abs(out.v_force[3]) < 0.02
    assert abs(out.v_force[4]) < 0.02
    assert abs(out.v_force[5]) < 0.02


def test_hover_couple_along_force_still_rotates() -> None:
    law = _hover_law()
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    for _ in range(60):
        _step(law, np.zeros(6))
    wrench = np.array([2.0, 0.0, 0.0, 0.15, 0.0, 0.0])
    out = None
    for _ in range(20):
        out = _step(law, wrench)
    assert out is not None
    assert out.v_force[0] < -1e-3
    assert out.v_force[3] < -0.10
    assert abs(out.v_force[4]) < 0.02
    assert abs(out.v_force[5]) < 0.02


def test_hover_nan_wrench_holds() -> None:
    law = _hover_law()
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    out = _step(law, np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert np.linalg.norm(out.v_force) < 1e-9


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
    assert out.v_force[3] < -0.10
    assert abs(out.v_force[3]) > abs(out.v_force[4])
    assert abs(out.v_force[3]) > abs(out.v_force[5])


def test_kikuuwe_sticks_below_coulomb() -> None:
    v, stuck = kikuuwe_step(
        np.zeros(3),
        np.array([0.25, 0.0, 0.0]),
        mass=1.5,
        damping=25.0,
        coulomb=0.32,
        dt=0.005,
        vmax=0.08,
    )
    assert stuck is True
    np.testing.assert_allclose(v, 0.0, atol=1e-12)


def test_poinsot_r0_is_axis_foot() -> None:
    f = np.array([2.0, 0.0, 0.0])
    r = np.array([0.0, 0.0, 0.12])
    m = np.cross(r, f)
    r0, h = poinsot_axis(f, m)
    np.testing.assert_allclose(r0, r, atol=1e-12)
    assert h == pytest.approx(0.0)
    proj = project_tool_cylinder(np.array([0.2, 0.0, 0.5]))
    assert math.hypot(proj[0], proj[1]) == pytest.approx(0.04)
    assert proj[2] == pytest.approx(0.16)


def test_hover_03n_residual_is_still() -> None:
    law = _hover_law()
    law.settle_s = 0.0
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    leftover = np.array([0.18, 0.12, 0.20, 0.010, -0.008, 0.006])
    assert float(np.linalg.norm(leftover[:3])) < 0.32
    for _ in range(40):
        out = _step(law, leftover)
        assert np.linalg.norm(out.v_force) < 1e-9


def test_hover_1n_lateral_does_not_yank() -> None:
    law = _hover_law()
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    for _ in range(60):
        _step(law, np.zeros(6))
    out = None
    for _ in range(30):
        out = _step(law, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert out is not None
    assert out.v_force[0] < -1e-3
    assert abs(out.v_force[3]) < 0.02
    assert abs(out.v_force[4]) < 0.02
    assert abs(out.v_force[5]) < 0.02


def test_hover_couple_rotates_about_rhat() -> None:
    law = _hover_law()
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    for _ in range(60):
        _step(law, np.zeros(6))
    wrench = np.array([1.5, 0.0, 0.0, 0.10, 0.0, 0.0])
    out = None
    for _ in range(40):
        out = _step(law, wrench)
    assert out is not None
    assert out.v_force[3] < -0.02
    r_hat = np.asarray(out.telemetry["r_hat"], dtype=float)
    assert float(np.linalg.norm(r_hat[:2])) < 0.05


def test_hover_release_stops_within_three_tau() -> None:
    law = _hover_law()
    law.reset(pose=np.zeros(6), f_ext=np.zeros(6))
    for _ in range(60):
        _step(law, np.zeros(6))
    for _ in range(40):
        _step(law, np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0]))
    out = None
    for _ in range(36):
        out = _step(law, np.zeros(6))
    assert out is not None
    assert np.linalg.norm(out.v_force) < 1e-3
