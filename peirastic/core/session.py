"""Mode classification used by the 200 Hz runner. Compile stays in realman8dof."""

from __future__ import annotations

import time

from peirastic.core.modes import Mode, try_mode

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
        Mode.CARTESIAN_PTP,
        Mode.MOVES,
    }
)


def is_swappable(mode: Mode) -> bool:
    m = try_mode(mode)
    return m is not None and m in SWAPPABLE_MODES


PAD_SOURCE_MAX_AGE_S = 0.25


def pad_source_present(
    stamp_s: float,
    *,
    now_s: float | None = None,
    max_age_s: float = PAD_SOURCE_MAX_AGE_S,
) -> bool:
    """True when ``python -m peirastic.apps.gamepad`` is writing the twist bus."""

    now = time.monotonic() if now_s is None else float(now_s)
    stamp = float(stamp_s)
    return stamp > 0.0 and (now - stamp) < float(max_age_s)


def idle_after_finite(*, pad_source: bool = False) -> Mode:
    """Idle after a planned move. SERVO only while the gamepad app is live."""

    return Mode.SERVO_TWIST if pad_source else Mode.SERVO_TWIST_HOLD


# Gamepad may write v_cmd and switch servo/pad-hybrid. Commanded modes win.
PAD_DRIVE_MODES = frozenset(
    {
        Mode.SERVO_TWIST,
        Mode.SERVO_TWIST_HOLD,
    }
)


def pad_may_drive(mode: Mode, *, program: bool = False, label: str = "") -> bool:
    """True when the pad may command motion. E-stop is always allowed.

    Window A idles in HOLD unless the gamepad app is writing twist. A live
    pad does not steal MOVEJ / CARTESIAN / TRACK_CARTESIAN / a running
    program. Pad-owned TRACK_HYBRID (L3) stays driveable; vessel/HFPC
    hybrid does not.
    """

    if program:
        return False
    m = try_mode(mode)
    if m is None:
        return False
    if m in PAD_DRIVE_MODES:
        return True
    if m == Mode.TRACK_HYBRID:
        text = str(label or "").lower()
        return "pad" in text
    return False
