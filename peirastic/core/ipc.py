"""Request / telemetry / twist SHM between window A and window C."""

from __future__ import annotations

import json
import time
from enum import IntEnum

import numpy as np

from rm75_control.control.admittance_common.shm_util import (
    attach_named_shm,
    close_attached_shm,
    close_named_shm,
    create_named_shm,
)
from peirastic.core.modes import Mode, ModeRequest

CTL_NAME = "peirastic_ctl"
PAYLOAD_NAME = "peirastic_payload"
TWIST_NAME = "peirastic_twist"
PAYLOAD_MAX = 16384

_CTL = np.dtype(
    [
        ("cmd_seq", "<u8"),
        ("cmd", "<u4"),
        ("ack_seq", "<u8"),
        ("status", "<u4"),
        ("mode", "<u4"),
        ("payload_len", "<u4"),
        ("ticks", "<u8"),
        ("estop", "<u4"),
        ("pad_hz", "<f8"),
        ("track_err_mm", "<f8"),
        ("slack", "<f8"),
        ("f_ext_z", "<f8"),
        ("t_mono", "<f8"),
        ("stop_req", "u1"),
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


class Cmd(IntEnum):
    NONE = 0
    SET_MODE = 1
    STOP = 2
    ESTOP = 3
    RESET = 4


class Status(IntEnum):
    IDLE = 0
    RUNNING = 1
    DONE = 2
    ERROR = 3
    STOPPED = 4
    ESTOP = 5


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
        self._seen = 0

    def close(self) -> None:
        ctl, pay = self._ctl, self._pay
        self._ctl = None
        self._pay = None
        del ctl, pay
        close_named_shm(self._ctl_shm)
        close_named_shm(self._pay_shm)

    def poll(self) -> tuple[Cmd, int, ModeRequest | None] | None:
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
    ) -> None:
        row = self._ctl[0]
        row["status"] = np.uint32(int(status))
        row["mode"] = np.uint32(int(mode))
        row["ticks"] = np.uint64(int(ticks))
        row["estop"] = np.uint32(1 if estop else 0)
        row["pad_hz"] = float(pad_hz)
        row["track_err_mm"] = float(track_err_mm)
        row["slack"] = float(slack)
        row["f_ext_z"] = float(f_ext_z)
        row["t_mono"] = float(time.monotonic())
        row["msg"] = str(msg).encode("utf-8")[:95]

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
        self._ctl_shm = attach_named_shm(self.ctl_name)
        self._pay_shm = attach_named_shm(self.payload_name)
        self._ctl = _view(self._ctl_shm.buf, _CTL)
        self._pay = np.ndarray((PAYLOAD_MAX,), dtype=np.uint8, buffer=self._pay_shm.buf)

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
            "status": int(row["status"]),
            "mode": int(row["mode"]),
            "ticks": int(row["ticks"]),
            "estop": bool(row["estop"]),
            "pad_hz": float(row["pad_hz"]),
            "track_err_mm": float(row["track_err_mm"]),
            "slack": float(row["slack"]),
            "f_ext_z": float(row["f_ext_z"]),
            "ack_seq": int(row["ack_seq"]),
            "msg": bytes(row["msg"]).split(b"\x00", 1)[0].decode("utf-8", "replace"),
        }


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
        twist,
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
