"""Move-to-D posture target must not replace the scan comfort attractor."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from rm75_control.control.joint_admittance_8dof.api import (
    attach_srs_move_tracking,
)


def test_srs_move_exit_restores_generic_posture_hint_and_rail_ownership() -> None:
    inner = MagicMock()
    move_target = np.linspace(0.0, 0.7, 8)
    move_ref = MagicMock()
    move_ref.psi_start = 0.25
    phase = MagicMock()
    phase.on_enter = None
    phase.on_tick = None
    phase.on_exit = None

    attach_srs_move_tracking(
        phase,
        inner,
        move_ref,
        move_target,
    )
    phase.on_enter()
    inner.set_posture_hint.assert_called_once_with(psi_rad=0.25)
    inner.set_plan_drives_rail.assert_called_once_with(True)
    phase.on_exit()
    inner.set_plan_drives_rail.assert_called_with(False)
    inner.set_posture_hint.assert_called_with()
