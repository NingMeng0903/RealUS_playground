"""Request / telemetry / twist SHM between window A and window C."""

from __future__ import annotations

import json
import math
import time
from enum import IntEnum

import numpy as np

from rm75_control.control.admittance_common.shm_util import (
    attach_named_shm,
    close_attached_shm,
    close_named_shm,
    create_named_shm,
)
from peirastic.core.modes import DofRequest, Mode, ModeRequest

# The DOF fields were added to the original control record.  Reusing the
# original shared-memory name would let an old Window-C client reinterpret the
# record with the wrong offsets.  Keep the old names only for an explicit
# migration diagnostic; all current peers use the versioned pair.
LEGACY_CTL_NAME = "peirastic_ctl"
LEGACY_PAYLOAD_NAME = "peirastic_payload"
CTL_NAME = "peirastic_ctl_v2"
PAYLOAD_NAME = "peirastic_payload_v2"
TWIST_NAME = "peirastic_twist"
MOTION_NAME = "peirastic_motion"
IPC_ABI_MAGIC = b"PEIRAST2"
IPC_ABI_VERSION = 2
IPC_SNAPSHOT_MAX_AGE_S = 0.5
PAYLOAD_MAX = 16384

_CTL = np.dtype(
    [
        ("abi_magic", "S8"),
        ("abi_version", "<u4"),
        ("cmd_seq", "<u8"),
        ("cmd", "<u4"),
        ("ack_seq", "<u8"),
        # ``ack_seq`` means Window A consumed the mailbox command.  The
        # install sequence is published only after the requested phase's
        # on_enter hook has run and the mode is actually live.
        ("install_seq", "<u8"),
        ("status", "<u4"),
        ("mode", "<u4"),
        ("dof", "<u4"),
        ("dof_pending", "<i4"),
        # ``dof``/``dof_pending`` are retained as compact compatibility
        # aliases.  These fields make an asynchronous structure request
        # self-describing even when another command overwrites cmd_seq.
        ("dof_requested", "<i4"),
        ("dof_effective", "<i4"),
        ("dof_request_seq", "<u8"),
        ("dof_done_seq", "<u8"),
        ("dof_status", "<u4"),
        ("payload_len", "<u4"),
        ("ticks", "<u8"),
        ("estop", "<u4"),
        ("pad_hz", "<f8"),
        ("track_err_mm", "<f8"),
        ("slack", "<f8"),
        ("f_ext_z", "<f8"),
        ("t_mono", "<f8"),
        ("stop_req", "u1"),
        ("done_seq", "<u8"),
        ("err_code", "<i4"),
        ("msg", "S96"),
    ]
)

_TWIST = np.dtype(
    [
        ("seq", "<u8"),
        ("stamp", "<f8"),
        ("hz", "<f8"),
        ("connected", "u1"),
        ("l3", "u1"),
        ("r3", "u1"),
        ("twist", "<f8", (6,)),
        ("axes", "<f8", (6,)),
        ("buttons", "<f8", (16,)),
    ]
)

_MOTION = np.dtype(
    [
        ("seq", "<u8"),
        ("t_mono", "<f8"),
        ("t_wall_s", "<f8"),
        ("v_tcp_z", "<f8"),
        ("a_tcp_z_plus", "<f8"),
        ("feedback_age_s", "<f8"),
        ("valid", "u1"),
    ]
)


class Cmd(IntEnum):
    NONE = 0
    SET_MODE = 1
    STOP = 2
    ESTOP = 3
    RESET = 4
    SET_DOF = 5


class Status(IntEnum):
    IDLE = 0
    RUNNING = 1
    DONE = 2
    ERROR = 3
    STOPPED = 4
    ESTOP = 5


class IpcMigrationError(FileNotFoundError):
    """The peer is using the pre-versioned PEIRASTIC control ABI."""

    pass


def _view(buf, dtype):
    return np.ndarray((1,), dtype=dtype, buffer=buf)


class CommandHub:
    """Window A owner of request SHM."""

    def __init__(self, *, prefix: str = "") -> None:
        self.ctl_name = (prefix + CTL_NAME) if prefix else CTL_NAME
        self.payload_name = (prefix + PAYLOAD_NAME) if prefix else PAYLOAD_NAME
        self._ctl_shm = create_named_shm(self.ctl_name, int(_CTL.itemsize))
        self._pay_shm = create_named_shm(self.payload_name, PAYLOAD_MAX)
        self._ctl = _view(self._ctl_shm.buf, _CTL)
        self._pay = np.ndarray((PAYLOAD_MAX,), dtype=np.uint8, buffer=self._pay_shm.buf)
        self._ctl[0] = np.zeros(1, dtype=_CTL)
        self._ctl[0]["abi_magic"] = IPC_ABI_MAGIC
        self._ctl[0]["abi_version"] = np.uint32(IPC_ABI_VERSION)
        self._ctl[0]["dof"] = np.uint32(8)
        self._ctl[0]["dof_pending"] = np.int32(-1)
        self._ctl[0]["dof_requested"] = np.int32(8)
        self._ctl[0]["dof_effective"] = np.int32(8)
        self._ctl[0]["dof_status"] = np.uint32(int(Status.IDLE))
        # The initial IDLE record is a valid, fresh session snapshot.  A
        # client can therefore inspect the default structure before the first
        # control tick; a crashed/old segment still fails the age check.
        self._ctl[0]["t_mono"] = float(time.monotonic())
        self._seen = 0
        self.motion = MotionBus(prefix=prefix, create=True)

    def close(self) -> None:
        ctl, pay = self._ctl, self._pay
        self._ctl = None
        self._pay = None
        del ctl, pay
        close_named_shm(self._ctl_shm)
        close_named_shm(self._pay_shm)
        self.motion.close()

    def poll(self) -> tuple[Cmd, int, ModeRequest | DofRequest | None] | None:
        row = self._ctl[0]
        seq = int(row["cmd_seq"])
        if seq == self._seen:
            return None
        self._seen = seq
        cmd = Cmd(int(row["cmd"]))
        req = None
        if cmd == Cmd.SET_MODE:
            n = int(row["payload_len"])
            blob = bytes(self._pay[:n].tobytes())
            req = ModeRequest.from_json(json.loads(blob.decode("utf-8")))
        elif cmd == Cmd.SET_DOF:
            n = int(row["payload_len"])
            blob = bytes(self._pay[:n].tobytes())
            req = DofRequest.from_json(json.loads(blob.decode("utf-8")))
            row["dof_requested"] = np.int32(int(req.dof))
            row["dof_request_seq"] = np.uint64(seq)
            row["dof_status"] = np.uint32(int(Status.RUNNING))
        return cmd, seq, req

    def ack(self, seq: int) -> None:
        self._ctl[0]["ack_seq"] = np.uint64(seq)

    def publish(
        self,
        *,
        status: Status,
        mode: Mode,
        ticks: int = 0,
        estop: bool = False,
        pad_hz: float = float("nan"),
        track_err_mm: float = float("nan"),
        slack: float = float("nan"),
        f_ext_z: float = float("nan"),
        msg: str = "",
        done_seq: int | None = None,
        err_code: int | None = None,
        dof: int | None = None,
        dof_pending: int | None = None,
        dof_requested: int | None = None,
        dof_effective: int | None = None,
        dof_request_seq: int | None = None,
        dof_done_seq: int | None = None,
        dof_status: Status | None = None,
        install_seq: int | None = None,
    ) -> None:
        row = self._ctl[0]
        row["status"] = np.uint32(int(status))
        row["mode"] = np.uint32(int(mode))
        if dof is not None:
            row["dof"] = np.uint32(int(dof))
            if dof_effective is None:
                row["dof_effective"] = np.int32(int(dof))
        if dof_pending is not None:
            row["dof_pending"] = np.int32(int(dof_pending))
            if int(dof_pending) < 0 and dof is not None and dof_requested is None:
                row["dof_requested"] = np.int32(int(dof))
        if dof_requested is not None:
            row["dof_requested"] = np.int32(int(dof_requested))
        if dof_effective is not None:
            row["dof_effective"] = np.int32(int(dof_effective))
        if dof_request_seq is not None:
            row["dof_request_seq"] = np.uint64(int(dof_request_seq))
        if dof_done_seq is not None:
            row["dof_done_seq"] = np.uint64(int(dof_done_seq))
        if dof_status is not None:
            row["dof_status"] = np.uint32(int(dof_status))
        elif (
            dof_pending is not None
            and int(dof_pending) < 0
            and done_seq is not None
            and int(done_seq) == int(row["dof_request_seq"])
        ):
            # Compatibility for callers that only provide the original
            # global done_seq/status fields.
            row["dof_done_seq"] = np.uint64(int(done_seq))
            row["dof_status"] = np.uint32(int(status))
        if install_seq is not None:
            row["install_seq"] = np.uint64(int(install_seq))
        row["ticks"] = np.uint64(int(ticks))
        row["estop"] = np.uint32(1 if estop else 0)
        row["pad_hz"] = float(pad_hz)
        row["track_err_mm"] = float(track_err_mm)
        row["slack"] = float(slack)
        row["f_ext_z"] = float(f_ext_z)
        row["t_mono"] = float(time.monotonic())
        row["msg"] = str(msg).encode("utf-8")[:95]
        if done_seq is not None:
            row["done_seq"] = np.uint64(int(done_seq))
        if err_code is not None:
            row["err_code"] = np.int32(int(err_code))

    def should_stop(self) -> bool:
        return bool(self._ctl[0]["stop_req"])

    def request_stop(self) -> None:
        self._ctl[0]["stop_req"] = np.uint8(1)

    def clear_stop(self) -> None:
        self._ctl[0]["stop_req"] = np.uint8(0)


class CommandClient:
    """Window C writer."""

    def __init__(self, *, prefix: str = "") -> None:
        self.ctl_name = (prefix + CTL_NAME) if prefix else CTL_NAME
        self.payload_name = (prefix + PAYLOAD_NAME) if prefix else PAYLOAD_NAME
        try:
            ctl_shm = attach_named_shm(self.ctl_name)
        except FileNotFoundError as exc:
            # An old Window-A process may still be alive.  Do not silently
            # attach its shorter, pre-DOF record: the fields after the old
            # offsets would otherwise be interpreted as unrelated values.
            legacy_name = (prefix + LEGACY_CTL_NAME) if prefix else LEGACY_CTL_NAME
            try:
                legacy_shm = attach_named_shm(legacy_name)
            except FileNotFoundError:
                raise exc
            close_attached_shm(legacy_shm)
            raise IpcMigrationError(
                f"legacy PEIRASTIC IPC {legacy_name!r} is present; "
                "restart Window A to create the versioned control segment"
            ) from exc

        ctl_size = int(getattr(ctl_shm, "size", len(ctl_shm.buf)))
        if ctl_size < int(_CTL.itemsize):
            close_attached_shm(ctl_shm)
            raise IpcMigrationError(
                f"PEIRASTIC control segment {self.ctl_name!r} is too small "
                f"({ctl_size} < {int(_CTL.itemsize)} bytes)"
            )
        ctl = _view(ctl_shm.buf, _CTL)
        magic = bytes(ctl[0]["abi_magic"]).split(b"\x00", 1)[0]
        version = int(ctl[0]["abi_version"])
        if magic != IPC_ABI_MAGIC or version != IPC_ABI_VERSION:
            del ctl
            close_attached_shm(ctl_shm)
            raise IpcMigrationError(
                f"unsupported PEIRASTIC IPC ABI on {self.ctl_name!r}: "
                f"magic={magic!r}, version={version}; restart Window A"
            )

        try:
            pay_shm = attach_named_shm(self.payload_name)
        except BaseException:
            del ctl
            close_attached_shm(ctl_shm)
            raise
        pay_size = int(getattr(pay_shm, "size", len(pay_shm.buf)))
        if pay_size < PAYLOAD_MAX:
            close_attached_shm(pay_shm)
            del ctl
            close_attached_shm(ctl_shm)
            raise IpcMigrationError(
                f"PEIRASTIC payload segment {self.payload_name!r} is too small "
                f"({pay_size} < {PAYLOAD_MAX} bytes)"
            )
        self._ctl_shm = ctl_shm
        self._pay_shm = pay_shm
        self._ctl = ctl
        self._pay = np.ndarray((PAYLOAD_MAX,), dtype=np.uint8, buffer=pay_shm.buf)

    def close(self) -> None:
        ctl, pay = self._ctl, self._pay
        self._ctl = None
        self._pay = None
        del ctl, pay
        close_attached_shm(self._ctl_shm)
        close_attached_shm(self._pay_shm)

    def set_mode(self, req: ModeRequest) -> int:
        blob = json.dumps(req.to_json(), separators=(",", ":")).encode("utf-8")
        if len(blob) > PAYLOAD_MAX:
            raise ValueError("payload too large")
        self._pay[: len(blob)] = np.frombuffer(blob, dtype=np.uint8)
        seq = int(self._ctl[0]["cmd_seq"]) + 1
        self._ctl[0]["payload_len"] = np.uint32(len(blob))
        self._ctl[0]["cmd"] = np.uint32(int(Cmd.SET_MODE))
        self._ctl[0]["stop_req"] = np.uint8(0)
        self._ctl[0]["cmd_seq"] = np.uint64(seq)
        return seq

    def set_dof(self, dof: int, *, after_current: bool = True) -> int:
        req = DofRequest(dof, after_current=after_current)
        blob = json.dumps(req.to_json(), separators=(",", ":")).encode("utf-8")
        if len(blob) > PAYLOAD_MAX:
            raise ValueError("payload too large")
        self._pay[: len(blob)] = np.frombuffer(blob, dtype=np.uint8)
        seq = int(self._ctl[0]["cmd_seq"]) + 1
        # Preserve an explicit STOP already issued by the caller.  The
        # structure request itself never creates a stop, but clearing this
        # bit would erase the boundary needed for a safe calibration restore
        # when STOP and SET_DOF share the one-slot command mailbox.
        stop_requested = bool(self._ctl[0]["stop_req"])
        self._ctl[0]["payload_len"] = np.uint32(len(blob))
        self._ctl[0]["cmd"] = np.uint32(int(Cmd.SET_DOF))
        if not stop_requested:
            self._ctl[0]["stop_req"] = np.uint8(0)
        self._ctl[0]["dof_requested"] = np.int32(int(dof))
        self._ctl[0]["dof_request_seq"] = np.uint64(seq)
        self._ctl[0]["dof_status"] = np.uint32(int(Status.RUNNING))
        self._ctl[0]["cmd_seq"] = np.uint64(seq)
        return seq

    def stop(self) -> int:
        seq = int(self._ctl[0]["cmd_seq"]) + 1
        self._ctl[0]["cmd"] = np.uint32(int(Cmd.STOP))
        self._ctl[0]["stop_req"] = np.uint8(1)
        self._ctl[0]["cmd_seq"] = np.uint64(seq)
        return seq

    def estop(self) -> int:
        seq = int(self._ctl[0]["cmd_seq"]) + 1
        self._ctl[0]["cmd"] = np.uint32(int(Cmd.ESTOP))
        self._ctl[0]["stop_req"] = np.uint8(1)
        self._ctl[0]["cmd_seq"] = np.uint64(seq)
        return seq

    def reset(self) -> int:
        seq = int(self._ctl[0]["cmd_seq"]) + 1
        self._ctl[0]["cmd"] = np.uint32(int(Cmd.RESET))
        self._ctl[0]["stop_req"] = np.uint8(0)
        self._ctl[0]["cmd_seq"] = np.uint64(seq)
        return seq

    def snapshot(self) -> dict:
        row = self._ctl[0]
        return {
            "abi_magic": bytes(row["abi_magic"]).split(b"\x00", 1)[0],
            "abi_version": int(row["abi_version"]),
            "status": int(row["status"]),
            "mode": int(row["mode"]),
            "dof": int(row["dof"]),
            "dof_pending": int(row["dof_pending"]),
            "dof_requested": int(row["dof_requested"]),
            "dof_effective": int(row["dof_effective"]),
            "dof_request_seq": int(row["dof_request_seq"]),
            "dof_done_seq": int(row["dof_done_seq"]),
            "dof_status": int(row["dof_status"]),
            "ticks": int(row["ticks"]),
            "estop": bool(row["estop"]),
            "pad_hz": float(row["pad_hz"]),
            "track_err_mm": float(row["track_err_mm"]),
            "slack": float(row["slack"]),
            "f_ext_z": float(row["f_ext_z"]),
            "t_mono": float(row["t_mono"]),
            "ack_seq": int(row["ack_seq"]),
            "install_seq": int(row["install_seq"]),
            "cmd_seq": int(row["cmd_seq"]),
            "done_seq": int(row["done_seq"]),
            "err_code": int(row["err_code"]),
            "msg": bytes(row["msg"]).split(b"\x00", 1)[0].decode("utf-8", "replace"),
        }

    def wait_installed(
        self,
        seq: int,
        mode: Mode | int,
        *,
        timeout: float = 2.0,
    ) -> int:
        """Wait until Window A has installed the requested mode.

        ``ack_seq`` only reports mailbox consumption.  ``install_seq`` is
        published after the phase ``on_enter`` hook runs.  A later install
        cannot satisfy an earlier request: a one-slot mailbox may have
        superseded that command before its phase became live.
        """

        # Keep core IPC independent of the facade module while sharing its
        # established integer return-code contract with Window-C callers.
        from peirastic.api.codes import ERR_CONTROLLER, ERR_NO_ACK, ERR_SEND, ERR_STOPPED, OK

        expected_mode = int(mode)
        deadline = time.monotonic() + max(float(timeout), 0.0)
        while time.monotonic() < deadline:
            try:
                snap = self.snapshot()
            except (TypeError, AttributeError, FileNotFoundError):
                return ERR_SEND
            try:
                magic = snap["abi_magic"]
                if isinstance(magic, str):
                    magic = magic.encode("ascii")
                if bytes(magic) != IPC_ABI_MAGIC or int(snap["abi_version"]) != IPC_ABI_VERSION:
                    return ERR_CONTROLLER
                stamp = float(snap["t_mono"])
                age = time.monotonic() - stamp
                if not math.isfinite(stamp) or age < -0.05 or age > IPC_SNAPSHOT_MAX_AGE_S:
                    return ERR_CONTROLLER
            except (KeyError, TypeError, ValueError, OverflowError):
                return ERR_CONTROLLER
            installed = int(snap.get("install_seq", 0))
            if installed == int(seq):
                return (
                    OK
                    if int(snap.get("mode", -1)) == expected_mode
                    else ERR_CONTROLLER
                )
            if installed > int(seq):
                return ERR_CONTROLLER
            st = int(snap.get("status", -1))
            if st in (int(Status.ESTOP), int(Status.STOPPED)):
                return ERR_STOPPED
            if st == int(Status.ERROR) and int(snap.get("ack_seq", 0)) >= int(seq):
                err = int(snap.get("err_code", 0))
                return ERR_CONTROLLER if err == 0 else err
            time.sleep(0.01)
        return ERR_NO_ACK


class TwistBus:
    """Latest 6D v_cmd. Gamepad writes; servo modes read."""

    def __init__(self, *, prefix: str = "", create: bool = False) -> None:
        self.name = (prefix + TWIST_NAME) if prefix else TWIST_NAME
        if create:
            self._shm = create_named_shm(self.name, int(_TWIST.itemsize))
        else:
            self._shm = attach_named_shm(self.name)
        self._row = _view(self._shm.buf, _TWIST)
        if create:
            self._row[0] = np.zeros(1, dtype=_TWIST)
        self._owner = bool(create)

    def close(self) -> None:
        row = self._row
        self._row = None
        del row
        if self._owner:
            close_named_shm(self._shm)
        else:
            close_attached_shm(self._shm)

    def write(
        self,
        twist=None,
        *,
        axes=None,
        buttons=None,
        hz: float = float("nan"),
        connected: bool = True,
        l3: bool = False,
        r3: bool = False,
    ) -> None:
        row = self._row[0]
        row["seq"] = np.uint64(int(row["seq"]) + 1)
        row["stamp"] = float(time.monotonic())
        row["hz"] = float(hz)
        row["connected"] = np.uint8(1 if connected else 0)
        row["l3"] = np.uint8(1 if l3 else 0)
        row["r3"] = np.uint8(1 if r3 else 0)
        if twist is not None:
            row["twist"] = np.asarray(twist, dtype=float).reshape(6)
        if axes is not None:
            a = np.asarray(axes, dtype=float).reshape(-1)
            row["axes"][: min(6, a.size)] = a[:6]
        if buttons is not None:
            b = np.asarray(buttons, dtype=float).reshape(-1)
            row["buttons"][: min(16, b.size)] = b[:16]

    def read(self) -> dict:
        row = self._row[0]
        return {
            "seq": int(row["seq"]),
            "stamp": float(row["stamp"]),
            "hz": float(row["hz"]),
            "connected": bool(row["connected"]),
            "l3": bool(row["l3"]),
            "r3": bool(row["r3"]),
            "twist": np.asarray(row["twist"], dtype=float).copy(),
            "axes": np.asarray(row["axes"], dtype=float).copy(),
            "buttons": np.asarray(row["buttons"], dtype=float).copy(),
        }


class MotionBus:
    """200 Hz measured tool-Z velocity for Window C backup replay."""

    def __init__(self, *, prefix: str = "", create: bool = False) -> None:
        self.name = (prefix + MOTION_NAME) if prefix else MOTION_NAME
        if create:
            self._shm = create_named_shm(self.name, int(_MOTION.itemsize))
        else:
            self._shm = attach_named_shm(self.name)
        self._row = _view(self._shm.buf, _MOTION)
        if create:
            self._row[0] = np.zeros(1, dtype=_MOTION)
        self._owner = bool(create)

    def close(self) -> None:
        row = self._row
        self._row = None
        del row
        if self._owner:
            close_named_shm(self._shm)
        else:
            close_attached_shm(self._shm)

    def publish(
        self,
        *,
        v_tcp_z: float,
        a_tcp_z_plus: float = 0.0,
        feedback_age_s: float = float("inf"),
        t_wall_s: float = float("nan"),
        valid: bool = False,
    ) -> None:
        row = self._row[0]
        seq = int(row["seq"])
        if seq % 2 == 1:
            seq -= 1
        row["seq"] = np.uint64(seq + 1)
        row["t_mono"] = float(time.monotonic())
        row["t_wall_s"] = float(t_wall_s)
        row["v_tcp_z"] = float(v_tcp_z)
        row["a_tcp_z_plus"] = max(float(a_tcp_z_plus), 0.0)
        row["feedback_age_s"] = float(feedback_age_s)
        row["valid"] = np.uint8(1 if valid else 0)
        row["seq"] = np.uint64(seq + 2)

    def read(self) -> dict:
        row, why = self._seqlock_copy()
        if row is None:
            return {
                "seq": 0,
                "t_mono": 0.0,
                "t_wall_s": float("nan"),
                "v_tcp_z": float("nan"),
                "a_tcp_z_plus": 0.0,
                "feedback_age_s": float("inf"),
                "valid": False,
                "pub_age_s": float("inf"),
                "age_total_s": float("inf"),
                "torn": why,
            }
        return row

    def _seqlock_copy(self) -> tuple[dict | None, str]:
        row = self._row[0]
        for _ in range(32):
            s1 = int(row["seq"])
            if s1 <= 0:
                return None, "empty"
            if s1 % 2 == 1:
                continue
            t_mono = float(row["t_mono"])
            payload = {
                "seq": s1,
                "t_mono": t_mono,
                "t_wall_s": float(row["t_wall_s"]),
                "v_tcp_z": float(row["v_tcp_z"]),
                "a_tcp_z_plus": float(row["a_tcp_z_plus"]),
                "feedback_age_s": float(row["feedback_age_s"]),
                "valid": bool(row["valid"]),
            }
            s2 = int(row["seq"])
            if s1 != s2 or s2 % 2 == 1:
                continue
            pub_age = (
                max(0.0, time.monotonic() - t_mono) if t_mono > 0.0 else float("inf")
            )
            fb = float(payload["feedback_age_s"])
            payload["pub_age_s"] = pub_age
            payload["age_total_s"] = (
                fb + pub_age if math.isfinite(fb) and math.isfinite(pub_age) else float("inf")
            )
            payload["torn"] = ""
            return payload, ""
        return None, "torn"

    def fresh(self, last_seq: int, *, max_age_s: float) -> tuple[dict | None, str]:
        """Accept only a new even seqlock generation whose total age is in bound."""
        row, why = self._seqlock_copy()
        if row is None:
            return None, why or "torn"
        limit = max(float(max_age_s), 0.0)
        if not row["valid"] or not math.isfinite(float(row["v_tcp_z"])):
            return None, "invalid"
        if int(row["seq"]) <= int(last_seq):
            return None, "seq_stale"
        if not math.isfinite(float(row["age_total_s"])):
            return None, "age_total"
        if float(row["age_total_s"]) > limit + 1e-12:
            return None, "age_total"
        return row, ""
