"""Move-to-D posture target must not replace the scan comfort attractor."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from rm75_control.control.joint_admittance_8dof.api import (
    attach_srs_move_tracking,
)


def test_srs_move_exit_restores_configured_comfort_posture() -> None:
    inner = MagicMock()
    comfort = np.zeros(8)
    move_target = np.linspace(0.0, 0.7, 8)
    inner.centering_task.q_target = comfort.copy()
    inner._centering_suppressed = False
    inner._arm_task_suppressed = True
    phase = MagicMock()
    phase.on_enter = None
    phase.on_tick = None
    phase.on_exit = None

    attach_srs_move_tracking(
        phase,
        inner,
        MagicMock(),
        move_target,
    )
    phase.on_enter()
    assert np.allclose(inner.centering_task.set_q_target.call_args.args[0], move_target)
    phase.on_exit()
    assert inner.centering_task.set_q_target.call_args.args == (None,)
