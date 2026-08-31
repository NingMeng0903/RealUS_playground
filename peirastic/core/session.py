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
        Mode.MOVEL,
        Mode.MOVES,
    }
)


def is_swappable(mode: Mode) -> bool:
    return Mode(mode) in SWAPPABLE_MODES


def idle_after_finite() -> Mode:
    """After MOVEJ / Cartesian PTP, hold TCP so nullspace can reconfigure joints."""

    return Mode.SERVO_TWIST_HOLD


# Gamepad may write v_cmd and switch servo/pad-hybrid. Commanded modes win.
PAD_DRIVE_MODES = frozenset(
    {
        Mode.SERVO_TWIST,
        Mode.SERVO_TWIST_HOLD,
    }
)


def pad_may_drive(mode: Mode, *, program: bool = False, label: str = "") -> bool:
    """True when the pad may command motion. E-stop is always allowed.

    Window A idles in SERVO_TWIST with or without a pad. A live pad does not
    steal MOVEJ / CARTESIAN / TRACK_CARTESIAN / a running program. Pad-owned
    TRACK_HYBRID (L3) stays driveable; vessel/HFPC hybrid does not.
    """

    if program:
        return False
    m = Mode(mode)
    if m in PAD_DRIVE_MODES:
        return True
    if m == Mode.TRACK_HYBRID:
        text = str(label or "").lower()
        return "pad" in text
    return False
