"""Read-only ``rm75_state`` shared-memory client.

Does **not** import ``rm75_control`` (Pinocchio vs OpenCV/ZMQ clash). The
dtype must stay byte-identical to
``rm75_control.control.admittance_common.state_relay`` ``_SLOT_DTYPE`` /
``_LAYOUT_DTYPE``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def _unregister_shm(name: str | None) -> None:
    """Stop the stdlib tracker from unlinking the publisher's segment on exit."""
    if not name:
        return
    try:
        resource_tracker.unregister(name, "shared_memory")
    except Exception:
        pass

# Keep in lockstep with state_relay.py (align=True).
_HEADER_DTYPE = np.dtype([("active", "<u8"), ("global_seq", "<u8"), ("session_id", "<u8")])
_SLOT_DTYPE = np.dtype(
    [
        ("seq", "<u8"),
        ("t_s", "<f8"),
        ("wall_time_ns", "<u8"),
        ("q_deg", "<f8", (7,)),
        ("pose", "<f8", (6,)),
        ("force", "<f8", (6,)),
        ("rail_m", "<f8"),
        ("ok", "u1"),
        ("qdot_deg_s", "<f8", (7,)),
    ],
    align=True,
)
_LAYOUT_DTYPE = np.dtype([("header", _HEADER_DTYPE), ("slots", _SLOT_DTYPE, (2,))])
SHM_SIZE = int(_LAYOUT_DTYPE.itemsize)

DEFAULT_RELAY_NAME = "rm75_state"


def slot_dtype() -> np.dtype:
    return _SLOT_DTYPE


def layout_dtype() -> np.dtype:
    return _LAYOUT_DTYPE


def expected_shm_size() -> int:
    return SHM_SIZE


def pose6_to_T(pose: np.ndarray) -> np.ndarray:
    """TCP pose ``[x,y,z,rx,ry,rz]`` (m, rad; intrinsic xyz) → ``T_railbase_tcp``."""
    p = np.asarray(pose, dtype=np.float64).reshape(6)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler("xyz", p[3:]).as_matrix()
    T[:3, 3] = p[:3]
    return T


def T_to_pose6(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    pose = np.zeros(6, dtype=np.float64)
    pose[:3] = T[:3, 3]
    pose[3:] = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
    return pose


def normalize_relay_name(name: str) -> str:
    name = str(name).strip()
    if name.startswith("shm://"):
        return name[len("shm://") :]
    return name


@dataclass
class RobotState:
    """One published ``rm75_state`` slot."""

    seq: int
    t_s: float
    q_deg: np.ndarray
    pose: np.ndarray
    rail_m: float
    ok: bool
    read_mono_s: float

    def T_railbase_tcp(self) -> np.ndarray:
        return pose6_to_T(self.pose)


@dataclass
class StillnessResult:
    ok: bool
    message: str
    trans_m: float = 0.0
    rot_deg: float = 0.0
    rail_m: float = 0.0


class RobotStateReader:
    """Attach to an existing ``rm75_state`` segment (publisher must already be up)."""

    def __init__(self, name: str = DEFAULT_RELAY_NAME) -> None:
        self._name = normalize_relay_name(name)
        self._shm: shared_memory.SharedMemory | None = None
        self._arr: Any = None
        self._last_seq: int = -1
        self._last_change_mono: float = 0.0
        self.last_error: str | None = None

    def _name_visible(self) -> bool:
        return Path("/dev/shm").joinpath(self._name).exists()

    def attach(self) -> None:
        if self._shm is not None:
            if self._name_visible():
                return
            # Publisher still holds a (deleted) mapping; we cannot re-open by name.
            self.close()
            raise FileNotFoundError(
                f"{self._name!r} was unlinked (POSIX deleted) while the controller "
                "still holds it. Restart the 8-DOF controller to recreate the name."
            )
        try:
            shm = shared_memory.SharedMemory(name=self._name, create=False)
        except FileNotFoundError:
            self.last_error = (
                f"/dev/shm/{self._name} is missing — controller is not publishing, "
                "or a previous subscriber unlinked the name (restart the controller)."
            )
            raise
        _unregister_shm(getattr(shm, "_name", None) or self._name)
        if shm.size < SHM_SIZE:
            shm.close()
            raise ValueError(f"shared memory too small: {shm.size} < {SHM_SIZE}")
        self._shm = shm
        self._arr = np.ndarray((), dtype=_LAYOUT_DTYPE, buffer=shm.buf)
        self.last_error = None

    def close(self) -> None:
        self._arr = None
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None

    def available(self) -> bool:
        try:
            self.attach()
        except (FileNotFoundError, ValueError, OSError):
            return False
        return True

    def read(self) -> RobotState | None:
        try:
            self.attach()
        except (FileNotFoundError, ValueError, OSError) as exc:
            self.last_error = str(exc)
            return None
        assert self._arr is not None
        active = int(self._arr["header"]["active"]) % 2
        slot = self._arr["slots"][active]
        seq = int(slot["seq"])
        now = time.monotonic()
        if seq != self._last_seq:
            self._last_seq = seq
            self._last_change_mono = now
        if not bool(slot["ok"]):
            self.last_error = "rm75_state ok=0 (no live pose / q_deg yet)."
            return None
        self.last_error = None
        return RobotState(
            seq=seq,
            t_s=float(slot["t_s"]),
            q_deg=np.asarray(slot["q_deg"], dtype=np.float64).copy(),
            pose=np.asarray(slot["pose"], dtype=np.float64).copy(),
            rail_m=float(slot["rail_m"]),
            ok=True,
            read_mono_s=now,
        )

    def age_s(self) -> float | None:
        """Seconds since the last sequence increment (None if never seen)."""
        if self._last_change_mono <= 0.0:
            return None
        return float(time.monotonic() - self._last_change_mono)

    def is_fresh(self, max_age_s: float) -> tuple[bool, float | None]:
        snap = self.read()
        if snap is None:
            return False, None
        age = self.age_s()
        if age is None:
            return False, None
        return age <= float(max_age_s), age

    def wait_still(
        self,
        *,
        window_s: float,
        trans_m: float,
        rot_deg: float,
        rail_m: float,
        samples: int = 5,
    ) -> tuple[RobotState | None, StillnessResult]:
        """Poll SHM over ``window_s`` and require the robot not to be moving."""
        snaps: list[RobotState] = []
        dt = max(0.01, float(window_s) / max(1, samples - 1))
        for i in range(max(2, samples)):
            snap = self.read()
            if snap is None:
                detail = self.last_error or "SHM has no live robot pose (ok=0 or missing)."
                return None, StillnessResult(ok=False, message=detail)
            snaps.append(snap)
            if i + 1 < max(2, samples):
                time.sleep(dt)
        poses = np.stack([s.pose for s in snaps], axis=0)
        rails = np.asarray([s.rail_m for s in snaps], dtype=np.float64)
        dxyz = float(np.linalg.norm(poses[:, :3].max(axis=0) - poses[:, :3].min(axis=0)))
        drail = float(np.max(np.abs(rails - rails.mean())))
        rots = Rotation.from_euler("xyz", poses[:, 3:])
        rel = (rots[0].inv() * rots).magnitude()
        drot = float(np.rad2deg(np.max(rel)))
        if dxyz > float(trans_m):
            return snaps[-1], StillnessResult(
                ok=False,
                message=f"Robot not still: TCP moved {dxyz * 1000:.1f} mm (limit {trans_m * 1000:.1f} mm).",
                trans_m=dxyz,
                rot_deg=drot,
                rail_m=drail,
            )
        if drot > float(rot_deg):
            return snaps[-1], StillnessResult(
                ok=False,
                message=f"Robot not still: TCP rotated {drot:.2f} deg (limit {rot_deg:.2f} deg).",
                trans_m=dxyz,
                rot_deg=drot,
                rail_m=drail,
            )
        if drail > float(rail_m):
            return snaps[-1], StillnessResult(
                ok=False,
                message=f"Robot not still: rail moved {drail * 1000:.2f} mm (limit {rail_m * 1000:.2f} mm).",
                trans_m=dxyz,
                rot_deg=drot,
                rail_m=drail,
            )
        return snaps[-1], StillnessResult(
            ok=True,
            message=f"still (Δxyz {dxyz * 1000:.2f} mm, Δrot {drot:.3f} deg, Δrail {drail * 1000:.2f} mm)",
            trans_m=dxyz,
            rot_deg=drot,
            rail_m=drail,
        )
