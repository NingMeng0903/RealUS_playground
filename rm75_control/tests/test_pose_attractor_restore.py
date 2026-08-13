"""SRS move ownership must not restore the retired runtime psi attractor."""

from __future__ import annotations

from unittest.mock import MagicMock

from rm75_control.control.joint_admittance_8dof.api import (
    attach_srs_move_tracking,
)


def test_srs_move_exit_restores_rail_ownership_without_runtime_psi_hint() -> None:
    inner = MagicMock()
    move_target = MagicMock()
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
    inner.set_plan_drives_rail.assert_called_once_with(True)
    inner.set_posture_hint.assert_not_called()
    phase.on_exit()
    inner.set_plan_drives_rail.assert_called_with(False)
    inner.set_posture_hint.assert_not_called()
