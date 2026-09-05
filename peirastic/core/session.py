"""Mode classification used by the 200 Hz runner. Compile stays in realman8dof."""

from __future__ import annotations

import time
import math

from peirastic.core.ipc import Status
from peirastic.core.modes import DofRequest, Mode, try_mode


IPC_ABI_MAGIC = b"PEIRAST2"
IPC_ABI_VERSION = 2
DOF_TELEMETRY_MAX_AGE_S = 0.5


def valid_dof_snapshot(
    snapshot: dict,
    *,
    now_s: float | None = None,
) -> bool:
    """Validate a fresh, versioned structure snapshot from Window A.

    The one-slot control mailbox can contain a perfectly well-formed but
    stale completion from an earlier command (or from an incompatible peer).
    Callers that use a DOF value to select a task must reject that snapshot
    before looking at any sequence/status field.
    """

    if not isinstance(snapshot, dict):
        return False
    try:
        magic = snapshot["abi_magic"]
        if isinstance(magic, str):
            magic = magic.encode("ascii")
        if bytes(magic) != IPC_ABI_MAGIC:
            return False
        if int(snapshot["abi_version"]) != IPC_ABI_VERSION:
            return False
        stamp = float(snapshot["t_mono"])
        now = time.monotonic() if now_s is None else float(now_s)
        age = now - stamp
        if not math.isfinite(stamp) or age < -0.05 or age > DOF_TELEMETRY_MAX_AGE_S:
            return False
        status = int(snapshot["status"])
        if status not in tuple(int(item) for item in Status):
            return False
        if bool(snapshot.get("estop")) or status in (
            int(Status.ESTOP),
            int(Status.STOPPED),
        ):
            return False
        current = int(snapshot["dof"])
        effective = int(snapshot["dof_effective"])
        pending = int(snapshot["dof_pending"])
        requested = int(snapshot["dof_requested"])
        if current not in (7, 8) or effective not in (7, 8):
            return False
        if current != effective or pending not in (-1, 7, 8):
            return False
        if requested not in (7, 8):
            return False
        dof_status = int(snapshot["dof_status"])
        if dof_status not in tuple(int(item) for item in Status):
            return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False

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
    hz: float | None = None,
    connected: bool = True,
) -> bool:
    """True when ``python -m peirastic.apps.gamepad`` is writing the twist bus.

    Commanded ``set_cartesian_velocity`` uses the same bus but leaves ``hz``
    as NaN, so it must not look like a pad.
    """

    if not connected:
        return False
    if hz is not None and not (float(hz) == float(hz) and float(hz) > 0.0):
        return False
    now = time.monotonic() if now_s is None else float(now_s)
    stamp = float(stamp_s)
    return stamp > 0.0 and (now - stamp) < float(max_age_s)


def idle_after_finite(*, pad_source: bool = False) -> Mode:
    """Idle after a planned move. SERVO only while the gamepad app is live."""

    return Mode.SERVO_TWIST if pad_source else Mode.SERVO_TWIST_HOLD


def stay_after_duration(mode: Mode) -> bool:
    """Commanded open-loop velocity stays in-mode when the clock ends.

    ``v*=0`` is rest. HOLD (pose latch + P) must not steal SERVO_TWIST.
    TRACK still idles to HOLD after a finite scan.
    """

    return try_mode(mode) == Mode.SERVO_TWIST


def request_dof(client, dof: int, *, timeout_s: float = 10.0) -> int:
    """Commit a session DOF request and return the previous value.

    Window A owns the commit.  Waiting for ``done_seq`` here is important for
    small Window C programs: a following ``SET_MODE`` would otherwise replace
    the one-slot IPC command before the daemon had observed ``SET_DOF``.
    """

    value = DofRequest(dof).dof
    snap = client.snapshot()
    if not valid_dof_snapshot(snap):
        raise RuntimeError("controller DOF telemetry is unavailable or stale")
    previous = int(snap["dof"])
    pending = int(snap.get("dof_pending", -1))
    requested = int(snap.get("dof_requested", -1))
    request_seq = int(snap.get("dof_request_seq", 0))
    ack_seq = int(snap.get("ack_seq", 0))
    if requested in (7, 8) and request_seq > ack_seq:
        if requested != value:
            raise RuntimeError(
                f"SET_DOF {requested} is already in the mailbox; cannot request {value}"
            )
        # Reuse the not-yet-consumed request; sending another command would
        # overwrite it in the one-slot IPC mailbox.
        seq = request_seq
    elif pending in (7, 8):
        if pending != value:
            raise RuntimeError(
                f"SET_DOF {pending} is already pending; cannot request {value}"
            )
        # Reuse the daemon-owned request sequence rather than overwriting a
        # live boundary request in the one-slot IPC mailbox.
        seq = request_seq
        if seq <= 0:
            raise RuntimeError("pending SET_DOF has no request sequence")
    elif previous == value:
        return previous
    else:
        seq = int(client.set_dof(value))
    deadline = time.monotonic() + max(float(timeout_s), 0.1)
    while time.monotonic() < deadline:
        snap = client.snapshot()
        if not valid_dof_snapshot(snap):
            if bool(snap.get("estop")) or int(snap.get("status", -1)) in (
                int(Status.ESTOP),
                int(Status.STOPPED),
            ):
                raise RuntimeError(str(snap.get("msg") or "SET_DOF interrupted by ESTOP"))
            raise RuntimeError("controller DOF telemetry is unavailable or stale")
        # Use the DOF-specific sequence/status when available.  The global
        # command slot may receive a following SET_MODE while this boundary
        # request is still waiting for a stationary feedback frame.
        dof_done = int(snap.get("dof_done_seq", 0))
        dof_status = int(snap.get("dof_status", -1))
        if dof_done > seq:
            raise RuntimeError("SET_DOF completion was superseded by another request")
        if dof_done == seq:
            if dof_status == int(Status.ERROR):
                err = int(snap.get("err_code", 0))
                raise RuntimeError(str(snap.get("msg") or f"SET_DOF failed ({err})"))
            if (
                dof_status == int(Status.DONE)
                and int(snap.get("dof_effective", -1)) == value
                and int(snap.get("dof_pending", -1)) == -1
                and int(snap.get("dof_request_seq", 0)) == seq
                and int(snap.get("dof_done_seq", 0)) == seq
            ):
                return previous
            if dof_status == int(Status.DONE):
                raise RuntimeError("SET_DOF completed with the wrong effective DOF")
        time.sleep(0.01)
    raise TimeoutError(f"SET_DOF {value} timed out")


def stop_before_dof(client) -> None:
    """Create an explicit task boundary for a DOF request.

    Before Window A consumes ``SET_DOF``, a STOP would overwrite the one-slot
    mailbox, so preserve an unacknowledged request.  Once the daemon has
    published ``dof_pending`` and acknowledged that sequence, the request is
    held in Window A state and a STOP is safe and required to open a boundary
    after a timed-out continuous SERVO task.
    """

    try:
        snap = client.snapshot()
    except Exception:
        # If the controller state cannot be read, a STOP would overwrite an
        # unobserved SET_DOF in the one-slot mailbox.  Leave the mailbox
        # untouched and let the caller report the communication failure.
        return
    if not valid_dof_snapshot(snap):
        return
    requested = int(snap["dof_requested"])
    request_seq = int(snap["dof_request_seq"])
    ack_seq = int(snap["ack_seq"])
    if requested in (7, 8) and request_seq > ack_seq:
        return
    client.stop()


# Gamepad may write v_cmd and switch servo/pad-hybrid. Commanded modes win.
PAD_DRIVE_MODES = frozenset(
    {
        Mode.SERVO_TWIST,
        Mode.SERVO_TWIST_HOLD,
    }
)
# Idle / pad-owned servo labels. ``cartesian_velocity`` shares the mode
# number but is a command session — the pad must not stomp that v*.
PAD_OWNED_SERVO_LABELS = frozenset(
    {
        "",
        "servo_twist",
        "servo_twist_hold",
        "servo",
    }
)


def pad_may_drive(mode: Mode, *, program: bool = False, label: str = "") -> bool:
    """True when the pad may command motion. E-stop is always allowed.

    Window A idles in HOLD unless the gamepad app is writing twist. A live
    pad does not steal MOVEJ / CARTESIAN / TRACK_CARTESIAN / commanded
    ``cartesian_velocity`` / a running program. Pad-owned TRACK_HYBRID
    (L3) stays driveable; vessel/HFPC hybrid does not.
    """

    if program:
        return False
    m = try_mode(mode)
    if m is None:
        return False
    text = str(label or "").lower().strip()
    if m in PAD_DRIVE_MODES:
        return text in PAD_OWNED_SERVO_LABELS or "pad" in text
    if m == Mode.TRACK_HYBRID:
        return "pad" in text
    return False
