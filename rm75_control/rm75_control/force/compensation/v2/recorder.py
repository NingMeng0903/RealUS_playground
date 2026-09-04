"""UDP snapshot recorder: callback copies into a bounded queue; writer is async."""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np

from rm75_control.control.admittance_common.async_state import AsyncStateSnapshot

SNAP_FIELDS = (
    "local_snap_seq",
    "recv_t_mono_ns",
    "recv_wall_ns",
    "robot_t_ns",
    "robot_packet_seq",
    "snap_ok",
    "q_deg_1",
    "q_deg_2",
    "q_deg_3",
    "q_deg_4",
    "q_deg_5",
    "q_deg_6",
    "q_deg_7",
    "qdot_sdk_deg_s_1",
    "qdot_sdk_deg_s_2",
    "qdot_sdk_deg_s_3",
    "qdot_sdk_deg_s_4",
    "qdot_sdk_deg_s_5",
    "qdot_sdk_deg_s_6",
    "qdot_sdk_deg_s_7",
    "force_raw_fx",
    "force_raw_fy",
    "force_raw_fz",
    "force_raw_mx",
    "force_raw_my",
    "force_raw_mz",
    "cmd_seq",
    "cmd_t_mono_ns",
    "cmd_age_s",
    "phase_id",
    "record_enable",
    "rail_pos_m",
    "rail_vel_m_s",
    "rail_hold_ref_m",
    "vel_clamped",
    "acc_clamped",
    "qp_slack_norm",
    "feedback_age_s",
    "v_cmd_x",
    "v_cmd_y",
    "v_cmd_z",
    "w_cmd_x",
    "w_cmd_y",
    "w_cmd_z",
)


def _nan7() -> np.ndarray:
    return np.full(7, np.nan)


def qdot_deg_s_for_record(
    snap: AsyncStateSnapshot,
    prev_q_deg: np.ndarray | None = None,
    prev_t_s: float | None = None,
) -> np.ndarray:
    """SDK ``joint_speed`` if finite. Optional Δq/Δt only when caller passes prev."""

    raw = snap.qdot_deg_s
    if raw is not None:
        arr = np.asarray(raw, dtype=float).reshape(-1)
        if arr.size >= 7 and np.all(np.isfinite(arr[:7])):
            return arr[:7].copy()
    q = snap.q_deg
    t = snap.t_s
    if (
        q is None
        or t is None
        or prev_q_deg is None
        or prev_t_s is None
        or not np.isfinite(float(t))
        or not np.isfinite(float(prev_t_s))
    ):
        return _nan7()
    qn = np.asarray(q, dtype=float).reshape(-1)
    qp = np.asarray(prev_q_deg, dtype=float).reshape(-1)
    if qn.size < 7 or qp.size < 7 or not np.all(np.isfinite(qn[:7])) or not np.all(np.isfinite(qp[:7])):
        return _nan7()
    dt = float(t) - float(prev_t_s)
    if dt < 1e-4:
        return _nan7()
    return (qn[:7] - qp[:7]) / dt


def snapshot_row(
    snap: AsyncStateSnapshot,
    *,
    prev_q_deg: np.ndarray | None = None,
    prev_t_s: float | None = None,
) -> dict:
    q = snap.q_deg if snap.q_deg is not None else _nan7()
    qd = qdot_deg_s_for_record(snap, prev_q_deg, prev_t_s)
    f = np.asarray(snap.force_raw, dtype=float).reshape(-1)
    if f.size < 6 or not np.all(np.isfinite(f[:6])):
        f = np.full(6, np.nan)
    ok = bool(snap.ok) and snap.q_deg is not None and np.all(np.isfinite(q[:7]))
    return {
        "local_snap_seq": int(snap.seq),
        "recv_t_mono_ns": int(round(float(snap.t_s) * 1e9)) if snap.t_s else 0,
        "recv_wall_ns": int(snap.wall_time_ns),
        "robot_t_ns": "",
        "robot_packet_seq": "",
        "snap_ok": 1 if ok else 0,
        "q_deg_1": float(q[0]),
        "q_deg_2": float(q[1]),
        "q_deg_3": float(q[2]),
        "q_deg_4": float(q[3]),
        "q_deg_5": float(q[4]),
        "q_deg_6": float(q[5]),
        "q_deg_7": float(q[6]),
        "qdot_sdk_deg_s_1": float(qd[0]),
        "qdot_sdk_deg_s_2": float(qd[1]),
        "qdot_sdk_deg_s_3": float(qd[2]),
        "qdot_sdk_deg_s_4": float(qd[3]),
        "qdot_sdk_deg_s_5": float(qd[4]),
        "qdot_sdk_deg_s_6": float(qd[5]),
        "qdot_sdk_deg_s_7": float(qd[6]),
        "force_raw_fx": float(f[0]),
        "force_raw_fy": float(f[1]),
        "force_raw_fz": float(f[2]),
        "force_raw_mx": float(f[3]),
        "force_raw_my": float(f[4]),
        "force_raw_mz": float(f[5]),
        "cmd_seq": "",
        "cmd_t_mono_ns": "",
        "cmd_age_s": "",
        "phase_id": "",
        "record_enable": "",
        "rail_pos_m": "",
        "rail_vel_m_s": "",
        "rail_hold_ref_m": "",
        "vel_clamped": "",
        "acc_clamped": "",
        "qp_slack_norm": "",
        "feedback_age_s": "",
        "v_cmd_x": "",
        "v_cmd_y": "",
        "v_cmd_z": "",
        "w_cmd_x": "",
        "w_cmd_y": "",
        "w_cmd_z": "",
    }


@dataclass
class PayloadIdRecorder:
    path: Path
    max_queue: int = 4096
    queue_overflow_count: int = 0
    udp_seq_gap_count: int = 0
    duplicate_snap_count: int = 0
    last_seq: int = 0
    invalid: bool = False
    _q: Queue = field(init=False)
    _stop: threading.Event = field(init=False)
    _thread: threading.Thread | None = field(init=False, default=None)
    _rows: int = 0

    def __post_init__(self) -> None:
        self._q = Queue(maxsize=int(self.max_queue))
        self._stop = threading.Event()

    def start(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._writer, name="payload-id-writer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def on_snapshot(self, snap: AsyncStateSnapshot) -> None:
        self.push(snap)

    def push(self, snap: AsyncStateSnapshot, **ctrl) -> None:
        seq = int(snap.seq)
        if seq == self.last_seq and self.last_seq > 0:
            self.duplicate_snap_count += 1
            return
        if self.last_seq > 0 and seq > self.last_seq + 1:
            self.udp_seq_gap_count += seq - self.last_seq - 1
        self.last_seq = seq
        row = self.attach_cmd(snapshot_row(snap), **ctrl)
        try:
            self._q.put_nowait(row)
        except Full:
            self.queue_overflow_count += 1
            self.invalid = True

    def attach_cmd(self, row: dict, **ctrl) -> dict:
        out = dict(row)
        for k, v in ctrl.items():
            if k in SNAP_FIELDS:
                out[k] = v
        return out

    def _writer(self) -> None:
        with self.path.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(SNAP_FIELDS))
            wr.writeheader()
            while not self._stop.is_set() or not self._q.empty():
                try:
                    row = self._q.get(timeout=0.05)
                except Empty:
                    continue
                wr.writerow(row)
                self._rows += 1
                if self._rows % 200 == 0:
                    fh.flush()
            fh.flush()


def merge_by_time(
    snap_t: np.ndarray,
    cmd_t: np.ndarray,
    cmd_rows: np.ndarray,
) -> np.ndarray:
    """Zero-order hold command fields onto snapshot times. Seqs stay separate."""
    if cmd_t.size == 0:
        return np.full((snap_t.size,) + cmd_rows.shape[1:], np.nan)
    idx = np.searchsorted(cmd_t, snap_t, side="right") - 1
    idx = np.clip(idx, 0, cmd_t.size - 1)
    return cmd_rows[idx]
