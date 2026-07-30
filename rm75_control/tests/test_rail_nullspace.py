"""Rail lock task + mode-enum smoke tests (no robot required)."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.tasks.rail_lock import RailLockConfig, RailLockTask
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode


def test_rail_lock_holds_position():
    cfg = RailLockConfig(
        mode=RailMode.LOCKED,
        locked_style=LockedStyle.HOLD,
        q_ref_m=0.0,
        lock_gain=20.0,
    )
    task = RailLockTask(cfg)
    q = np.zeros(8)
    q[0] = 0.05
    qdot = task(q)
    assert qdot[0] < 0.0
    assert np.allclose(qdot[1:], 0.0)


def test_rail_coupled_no_task_velocity():
    """RailLockTask is silent unless we're in LOCKED+HOLD."""
    cfg = RailLockConfig(mode=RailMode.COUPLED, q_ref_m=0.0, lock_gain=20.0)
    task = RailLockTask(cfg)
    q = np.zeros(8)
    q[0] = 0.05
    assert np.allclose(task(q), 0.0)


def test_rail_locked_rail_only_no_hold_output():
    """RAIL_ONLY drives rail via qdot_ff, not via the hold task."""
    cfg = RailLockConfig(
        mode=RailMode.LOCKED,
        locked_style=LockedStyle.RAIL_ONLY,
        q_ref_m=0.0,
        lock_gain=20.0,
    )
    task = RailLockTask(cfg)
    q = np.zeros(8)
    q[0] = 0.05
    assert np.allclose(task(q), 0.0)


def test_rail_lock_capture_on_reset():
    task = RailLockTask(
        RailLockConfig(
            mode=RailMode.LOCKED,
            locked_style=LockedStyle.HOLD,
            q_ref_m=None,
            lock_gain=10.0,
        )
    )
    q = np.zeros(8)
    q[0] = 0.08
    task.reset(q)
    qdot = task(q)
    assert abs(qdot[0]) < 1e-9


def test_rail_lock_reset_overwrites_yaml_seed():
    """Yaml may seed q_ref_m=0.0; reset must still capture live rail."""
    task = RailLockTask(
        RailLockConfig(
            mode=RailMode.LOCKED,
            locked_style=LockedStyle.HOLD,
            q_ref_m=0.0,
            lock_gain=10.0,
        )
    )
    assert task.q_ref == 0.0
    q = np.zeros(8)
    q[0] = 0.404
    task.reset(q)
    assert abs(float(task.q_ref) - 0.404) < 1e-12
    qdot = task(q)
    assert abs(qdot[0]) < 1e-9


def test_rail_mode_enum_values():
    """RailMode is a two-value top-level enum; RELIEF has been removed."""
    assert set(RailMode) == {RailMode.COUPLED, RailMode.LOCKED}
    assert {s for s in LockedStyle} == {
        LockedStyle.HOLD,
        LockedStyle.RAIL_ONLY,
        LockedStyle.TCP_FIXED,
    }
