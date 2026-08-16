"""Unequal-sample third-order box uses two real periods, not 2q̇-q̇prev2."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    VelocityBoxConstraints,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


def _box() -> VelocityBoxConstraints:
    return VelocityBoxConstraints(
        SafetyLimits(
            q_lower=np.full(8, -2.0),
            q_upper=np.full(8, 2.0),
            v_max=np.full(8, 2.0),
            a_max=None,
            position_margin=np.full(8, 0.01),
        ),
        damper_band_rad=0.0,
        rail_reaction_s=0.0,
    )


def test_first_tick_jerk_centre_is_qdot_prev() -> None:
    box = _box()
    q = np.zeros(8)
    qdot_prev = np.full(8, 0.04)
    qdot_prev2 = np.full(8, 0.01)
    j_max = np.full(8, 300.0)
    h1 = 0.006
    lo, hi = box.bounds(
        q,
        0.005,
        qdot_prev,
        qdot_prev2=qdot_prev2,
        j_max=j_max,
        box_h1=h1,
        box_h2=None,
    )
    span = 300.0 * h1 * h1
    assert lo[1] == pytest.approx(0.04 - span, abs=1.0e-9)
    assert hi[1] == pytest.approx(0.04 + span, abs=1.0e-9)


def test_unequal_sample_jerk_box_does_not_inherit_period_jitter() -> None:
    box = _box()
    q = np.zeros(8)
    qdot_prev = np.full(8, 0.100)
    qdot_prev2 = np.full(8, 0.050)
    j_max = np.full(8, 300.0)
    h1 = 0.0085
    h2 = 0.005
    assert h1 / h2 == pytest.approx(1.7, abs=1.0e-12)

    lo, hi = box.bounds(
        q,
        0.005,
        qdot_prev,
        qdot_prev2=qdot_prev2,
        j_max=j_max,
        box_h1=h1,
        box_h2=h2,
    )
    centre = qdot_prev + (h1 / h2) * (qdot_prev - qdot_prev2)
    span = j_max * h1 * h1
    np.testing.assert_allclose(0.5 * (lo + hi), centre, atol=1.0e-12)
    np.testing.assert_allclose(0.5 * (hi - lo), span, atol=1.0e-12)

    # A command at the corrected centre keeps |a_k - a_{k-1}| <= j * h1.
    qdot_k = centre
    a_k = (qdot_k - qdot_prev) / h1
    a_prev = (qdot_prev - qdot_prev2) / h2
    assert np.max(np.abs(a_k - a_prev)) <= 300.0 * h1 + 1.0e-12
    np.testing.assert_allclose(a_k, a_prev, atol=1.0e-12)

    equal_centre = 2.0 * qdot_prev - qdot_prev2
    a_wrong = (equal_centre - qdot_prev) / h1
    assert float(np.max(np.abs(a_wrong - a_prev))) > 300.0 * h1
