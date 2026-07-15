"""Phase program IPC: window C submits tasks, window A runs WBC locally."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from rm75_control.control.admittance_common.shm_util import (
    attach_named_shm,
    close_attached_shm,
    close_named_shm,
    create_named_shm,
)

DEFAULT_PHASE_CTL_NAME = "rm75_phase_ctl"
DEFAULT_PHASE_PAYLOAD_NAME = "rm75_phase_payload"
PAYLOAD_MAX_BYTES = 16384

_CTL_DTYPE = np.dtype(
    [
        ("cmd_seq", "<u8"),
        ("cmd", "<u4"),
        ("ack_seq", "<u8"),
        ("status", "<u4"),
        ("status_seq", "<u8"),
        ("phase_idx", "<u4"),
        ("ticks", "<u8"),
        ("payload_len", "<u4"),
        ("t_status_mono", "<f8"),
        ("stop_req", "u1"),
        ("msg", "S96"),
    ]
)
_CTL_SIZE = int(_CTL_DTYPE.itemsize)


class PhaseCmd(IntEnum):
    NONE = 0
    START = 1
    STOP = 2


class PhaseStatus(IntEnum):
    IDLE = 0
    RUNNING = 1
    DONE = 2
    ERROR = 3
    STOPPED = 4


@dataclass
class SinToolYTaskParams:
    """Serializable task descriptor (C plans, A executes)."""

    config_path: str
    slot: str = "d"
    approach_dz_mm: float = 0.0
    use_force_id_pose: bool = False
    move_duration: float | None = None
    move_duration_margin: float = 0.50
    move_duration_min: float = 2.5
    move_duration_max: float = 5.0
    move_kp: float = 2.0
    move_mode: str = "cartesian"
    auto_joint: bool = True
    y_pp_cm: float = 16.0
    max_vel_cm_s: float = 2.0
    period_s: float | None = None
    desired_z: float = 0.0
    scan_duration: float = 30.0
    hold_s: float = 0.0
    hold_at_d_s: float = 0.0
    rail_move_cm: float = 0.0
    rail_move_mode: str = "rail_only"
    rail_move_dir: str = "+y"
    enable_force: bool = False
    log_interval: float = 2.0
    log_csv: str | None = None
    cartesian_max_lin_vel: float | None = None
    q0_rad: list[float] = field(default_factory=list)
    q_target_rad: list[float] = field(default_factory=list)
    pose_d: list[float] = field(default_factory=list)
    plan_duration_s: float = 0.0
    plan_move_mode: str = "cartesian"
    plan_gov_joint_max_deg: float = 0.0
    plan_meta: dict[str, float] = field(default_factory=dict)
    psi_tgt: float | None = None
    psi_toggle_period_s: float = 0.0
    psi_side_offset_rad: float = 1.580525773858965  # 90.5 deg fallback each side
    psi_left_rad: float | None = None
    psi_right_rad: float | None = None
    psi_filter_alpha: float = 0.02
    psi_ramp_s: float = 4.0
    scan_hybrid_hold: bool = False
    q_toggle_left_rad: list[float] = field(default_factory=list)
    q_toggle_right_rad: list[float] = field(default_factory=list)
    tcp_offset_pose: list[float] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> SinToolYTaskParams:
        return cls(**json.loads(text))


def _encode_msg(text: str) -> bytes:
    return str(text).encode("utf-8", errors="replace")[:95].ljust(96, b"\0")


class PhaseCommandHub:
    """Window A: owns ctl + payload SHM; hot-waits for START, runs WBC locally."""

    def __init__(
        self,
        *,
        ctl_name: str = DEFAULT_PHASE_CTL_NAME,
        payload_name: str = DEFAULT_PHASE_PAYLOAD_NAME,
    ) -> None:
        self._ctl_name = str(ctl_name)
        self._payload_name = str(payload_name)
        self._ctl_shm = create_named_shm(self._ctl_name, _CTL_SIZE)
        self._payload_shm = create_named_shm(self._payload_name, PAYLOAD_MAX_BYTES)
        self._ctl = np.ndarray((), dtype=_CTL_DTYPE, buffer=self._ctl_shm.buf)
        self._payload = memoryview(self._payload_shm.buf)
        self._last_ack = 0
        self._task_n = 0
        self._ctl["cmd_seq"] = np.uint64(0)
        self._ctl["cmd"] = np.uint32(PhaseCmd.NONE)
        self._ctl["ack_seq"] = np.uint64(0)
        self._ctl["payload_len"] = np.uint32(0)
        self._ctl["stop_req"] = np.uint8(0)
        self.set_idle()

    def poll(self) -> tuple[PhaseCmd, int, SinToolYTaskParams | None] | None:
        try:
            cmd_seq = int(self._ctl["cmd_seq"])
            if cmd_seq <= self._last_ack:
                return None
            cmd = PhaseCmd(int(self._ctl["cmd"]))
            params = None
            if cmd == PhaseCmd.START:
                n = int(self._ctl["payload_len"])
                if n <= 0 or n > PAYLOAD_MAX_BYTES:
                    raise ValueError(f"invalid payload_len={n}")
                text = bytes(self._payload[:n]).decode("utf-8")
                params = SinToolYTaskParams.from_json(text)
                self._task_n += 1
            return cmd, cmd_seq, params
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"phase IPC decode failed: {exc}") from exc

    def ack(self, cmd_seq: int) -> None:
        self._ctl["ack_seq"] = np.uint64(cmd_seq)
        self._last_ack = int(cmd_seq)
        self._ctl["cmd"] = np.uint32(PhaseCmd.NONE)
        self._ctl["stop_req"] = np.uint8(0)

    def should_stop(self) -> bool:
        return int(self._ctl["stop_req"]) != 0

    @property
    def task_n(self) -> int:
        return int(self._task_n)

    def _write_status(
        self,
        *,
        status: PhaseStatus,
        status_seq: int,
        phase_idx: int = 0,
        ticks: int = 0,
        msg: str = "",
    ) -> None:
        self._ctl["status"] = np.uint32(int(status))
        self._ctl["status_seq"] = np.uint64(status_seq)
        self._ctl["phase_idx"] = np.uint32(phase_idx)
        self._ctl["ticks"] = np.uint64(ticks)
        self._ctl["t_status_mono"] = time.monotonic()
        self._ctl["msg"] = np.frombuffer(_encode_msg(msg), dtype="S96")

    def set_idle(self, msg: str = "waiting for task") -> None:
        self._write_status(status=PhaseStatus.IDLE, status_seq=0, msg=msg)

    def set_running(self, cmd_seq: int, msg: str = "running") -> None:
        self._write_status(status=PhaseStatus.RUNNING, status_seq=cmd_seq, msg=msg)

    def set_progress(
        self,
        cmd_seq: int,
        *,
        phase_idx: int,
        phase_label: str,
        ticks: int,
    ) -> None:
        self._write_status(
            status=PhaseStatus.RUNNING,
            status_seq=cmd_seq,
            phase_idx=phase_idx,
            ticks=ticks,
            msg=phase_label,
        )

    def set_done(self, cmd_seq: int, msg: str = "done") -> None:
        self._write_status(status=PhaseStatus.DONE, status_seq=cmd_seq, msg=msg)

    def set_error(self, cmd_seq: int, msg: str) -> None:
        self._write_status(status=PhaseStatus.ERROR, status_seq=cmd_seq, msg=msg[:95])

    def set_stopped(self, cmd_seq: int, msg: str = "stopped") -> None:
        self._write_status(status=PhaseStatus.STOPPED, status_seq=cmd_seq, msg=msg)

    def close(self) -> None:
        try:
            if self._ctl is not None:
                self._ctl["cmd"] = np.uint32(PhaseCmd.NONE)
                self._ctl["payload_len"] = np.uint32(0)
                self.set_idle("shutdown")
        except (OSError, ValueError):
            pass
        self._payload = None
        self._ctl = None
        close_named_shm(self._ctl_shm)
        close_named_shm(self._payload_shm)
        self._ctl_shm = None
        self._payload_shm = None
        self._payload = None


class PhaseCommandClient:
    """Window C: attach to A's hub, submit START/STOP, monitor status."""

    def __init__(
        self,
        *,
        ctl_name: str = DEFAULT_PHASE_CTL_NAME,
        payload_name: str = DEFAULT_PHASE_PAYLOAD_NAME,
    ) -> None:
        self._ctl_name = str(ctl_name)
        self._payload_name = str(payload_name)
        self._ctl_shm: shared_memory.SharedMemory | None = None
        self._payload_shm: shared_memory.SharedMemory | None = None
        self._ctl = None
        self._payload = None

    def _reset(self) -> None:
        self._payload = None
        self._ctl = None
        close_attached_shm(self._ctl_shm)
        close_attached_shm(self._payload_shm)
        self._ctl_shm = None
        self._payload_shm = None

    def _ensure(self) -> bool:
        if self._ctl is not None:
            return True
        try:
            self._ctl_shm = attach_named_shm(self._ctl_name)
            self._payload_shm = attach_named_shm(self._payload_name)
            self._ctl = np.ndarray((), dtype=_CTL_DTYPE, buffer=self._ctl_shm.buf)
            self._payload = memoryview(self._payload_shm.buf)
            return True
        except (FileNotFoundError, OSError):
            self._reset()
            return False

    def wait_for_hub(self, *, timeout_s: float = 30.0, poll_s: float = 0.1) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._ensure():
                return
            time.sleep(poll_s)
        raise TimeoutError(
            f"phase IPC hub {self._ctl_name!r} not ready — start window A first"
        )

    def start(self, params: SinToolYTaskParams) -> int:
        if not self._ensure():
            raise RuntimeError("phase IPC hub not connected")
        blob = params.to_json().encode("utf-8")
        if len(blob) > PAYLOAD_MAX_BYTES:
            raise ValueError(f"task payload too large: {len(blob)} > {PAYLOAD_MAX_BYTES}")
        self._payload[: len(blob)] = blob
        cmd_seq = int(self._ctl["cmd_seq"]) + 1
        self._ctl["payload_len"] = np.uint32(len(blob))
        self._ctl["stop_req"] = np.uint8(0)
        self._ctl["cmd"] = np.uint32(PhaseCmd.START)
        self._ctl["cmd_seq"] = np.uint64(cmd_seq)
        return cmd_seq

    def stop(self) -> None:
        if not self._ensure():
            return
        self._ctl["stop_req"] = np.uint8(1)

    def read_status(self) -> dict[str, Any] | None:
        if not self._ensure():
            return None
        try:
            msg_bytes = bytes(self._ctl["msg"]).split(b"\0", 1)[0]
            return {
                "status": PhaseStatus(int(self._ctl["status"])),
                "status_seq": int(self._ctl["status_seq"]),
                "phase_idx": int(self._ctl["phase_idx"]),
                "ticks": int(self._ctl["ticks"]),
                "msg": msg_bytes.decode("utf-8", errors="replace"),
                "t_status_mono": float(self._ctl["t_status_mono"]),
            }
        except (OSError, ValueError):
            self._reset()
            return None

    def wait_for_cmd(
        self,
        cmd_seq: int,
        *,
        timeout_s: float = 7200.0,
        poll_s: float = 0.05,
    ) -> PhaseStatus:
        deadline = time.monotonic() + timeout_s
        last_status = PhaseStatus.RUNNING
        while time.monotonic() < deadline:
            st = self.read_status()
            if st is None:
                time.sleep(poll_s)
                continue
            status = st["status"]
            if st["status_seq"] == cmd_seq and status in (
                PhaseStatus.DONE,
                PhaseStatus.ERROR,
                PhaseStatus.STOPPED,
            ):
                return status
            last_status = status
            time.sleep(poll_s)
        return last_status

    def close(self) -> None:
        self._reset()


def phase_ipc_hub_ready(ctl_name: str = DEFAULT_PHASE_CTL_NAME) -> bool:
    try:
        probe = attach_named_shm(ctl_name)
        close_attached_shm(probe)
        return True
    except (FileNotFoundError, OSError):
        return False
