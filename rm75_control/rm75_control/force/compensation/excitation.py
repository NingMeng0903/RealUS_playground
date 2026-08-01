"""Excitation trajectories and pose YAML helpers."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from rm75_control.motion.canfd import send_pose_canfd

from .id_config import CartesianConfig, PoseDConfig

DEG2RAD = math.pi / 180.0


def load_poses_yaml(path: Path) -> dict:
    if not path.exists():
        return {"poses": {}}
    return yaml.safe_load(path.read_text()) or {"poses": {}}


def save_poses_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True))


def save_pose_slot(
    path: Path, slot: str, pose6: np.ndarray, q_deg: np.ndarray, label: str | None
) -> None:
    data = load_poses_yaml(path)
    data.setdefault("poses", {})[slot] = {
        "label": label or f"pose_{slot}",
        "note": f"saved {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "pose_base": [round(float(v), 6) for v in pose6],
        "q_deg": [round(float(v), 3) for v in q_deg],
    }
    save_poses_yaml(path, data)


def get_slot_record(data: dict, slot: str) -> dict | None:
    rec = data.get("poses", {}).get(slot)
    if not rec or rec.get("pose_base") is None:
        return None
    return rec


def pose_drift_mm_deg(current: np.ndarray, recorded: np.ndarray) -> tuple[float, float]:
    dpos = float(np.linalg.norm(current[:3] - recorded[:3])) * 1000.0
    deul = np.abs(current[3:6] - recorded[3:6])
    deul = np.minimum(deul, 2 * math.pi - deul)
    ddeg = float(np.max(deul) * 180.0 / math.pi)
    return dpos, ddeg


def ramp_down_cartesian(
    robot,
    pose0: np.ndarray,
    start_delta: np.ndarray,
    *,
    follow: bool,
    dt_ms: float,
    ramp_s: float,
    next_tick: float | None = None,
) -> float:
    """Cosine fade of pose delta to zero before exit_canfd_session."""
    start_delta = np.asarray(start_delta, dtype=float)
    pose0 = np.asarray(pose0, dtype=float)
    dt_s = dt_ms / 1000.0
    if ramp_s <= 0 or float(np.max(np.abs(start_delta))) < 1e-9:
        send_pose_canfd(robot, pose0.tolist(), follow=follow)
        return next_tick if next_tick is not None else time.monotonic()

    n = max(2, int(ramp_s / dt_s) + 1)
    if next_tick is None:
        next_tick = time.monotonic()
    for i in range(n):
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
        next_tick += dt_s
        scale = 0.5 * (1.0 + math.cos(math.pi * i / (n - 1)))
        send_pose_canfd(robot, (pose0 + start_delta * scale).tolist(), follow=follow)
    send_pose_canfd(robot, pose0.tolist(), follow=follow)
    return next_tick


@dataclass(frozen=True)
class CartesianExcitation:
    amp_mm: np.ndarray
    amp_rot_deg: np.ndarray
    freqs_hz: list[list[float]]
    slot: str

    @classmethod
    def from_config(cls, cart: CartesianConfig, scale: float, slot: str) -> "CartesianExcitation":
        amp_rot = cart.amp_rot_for_slot(slot) * scale
        amp_mm = cart.amp_mm_for_slot(slot) * scale
        return cls(
            amp_mm=amp_mm,
            amp_rot_deg=amp_rot,
            freqs_hz=cart.freqs_hz,
            slot=slot,
        )

    def delta_pose(self, t_s: float) -> np.ndarray:
        delta = np.zeros(6, dtype=float)
        for j, (amp_mm, amp_deg) in enumerate(zip(self.amp_mm, self.amp_rot_deg)):
            for k, (f_lin, f_rot) in enumerate(
                zip(
                    self.freqs_hz[j % len(self.freqs_hz)],
                    self.freqs_hz[(j + 1) % len(self.freqs_hz)],
                )
            ):
                ph = j * 0.7 + k * 1.1
                delta[j] += amp_mm / 1000.0 * math.sin(2.0 * math.pi * f_lin * t_s + ph)
                delta[3 + j % 3] += amp_deg * DEG2RAD * math.sin(
                    2.0 * math.pi * f_rot * t_s + ph + 0.4
                )
        return delta


def joint_cmd(
    t_s: float,
    q0: np.ndarray,
    pd: PoseDConfig,
    scale: float,
) -> np.ndarray:
    q = q0.copy()
    for j in range(min(7, len(pd.joint_amp_deg))):
        freqs = pd.joint_freqs_hz[j % len(pd.joint_freqs_hz)]
        amp = pd.joint_amp_deg[j] * scale
        max_d = pd.joint_max_delta_deg[j]
        delta = 0.0
        for k, f in enumerate(freqs):
            ph = j * 0.9 + k * 1.3
            delta += amp * math.sin(2.0 * math.pi * f * t_s + ph)
        delta = max(-max_d, min(max_d, delta))
        q[j] += delta
    return q


def clamp_delta(
    delta: np.ndarray,
    *,
    max_mm: float,
    max_rot_deg: float,
) -> np.ndarray:
    out = delta.copy()
    norm_pos = float(np.linalg.norm(out[:3])) * 1000.0
    if norm_pos > max_mm:
        out[:3] *= max_mm / norm_pos / 1000.0
    for j in range(3):
        d_deg = abs(out[3 + j]) * 180.0 / math.pi
        if d_deg > max_rot_deg:
            out[3 + j] *= max_rot_deg / d_deg
    return out


def preview_pose_d(q0: np.ndarray, pd: PoseDConfig, *, scale: float) -> dict:
    dt_s = 0.01
    ts_j = np.linspace(0, pd.joint_duration_s, int(pd.joint_duration_s / dt_s) + 1)
    qs = np.array([joint_cmd(t, q0, pd, scale) for t in ts_j])
    return {
        "joint_max_deg": np.max(np.abs(qs - q0), axis=0).tolist(),
        "j7_max_deg": float(np.max(np.abs(qs[:, 6] - q0[6]))),
    }
