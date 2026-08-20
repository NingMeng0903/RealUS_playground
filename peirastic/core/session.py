"""Mode classification used by the 200 Hz runner. Compile stays in realman8dof."""

from __future__ import annotations

from peirastic.core.modes import Mode

# Velocity-interface modes settle to v* and swap in place on the same 200 Hz
# runner: exit the previous outer, bind the next, keep inner / wbc_rt.
SWAPPABLE_MODES = frozenset(
    {
        Mode.SERVO_TWIST,
        Mode.SERVO_TWIST_HOLD,
        Mode.TRACK_CARTESIAN,
        Mode.TRACK_HYBRID,
    }
)
# Joint PTP skips the Cartesian QP (FLAG_DIRECT_PTP + qdot_ff). Rebuilding
# the 200 Hz phase is fine; do not mix them onto the live velocity proxy.
FINITE_MODES = frozenset(
    {
        Mode.GOTO_JOINTS,
        Mode.MOVEJ,
    }
)


def is_swappable(mode: Mode) -> bool:
    return Mode(mode) in SWAPPABLE_MODES
