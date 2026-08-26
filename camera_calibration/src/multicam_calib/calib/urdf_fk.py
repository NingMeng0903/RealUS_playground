"""NumPy FK for the RM75 8-DOF ``rail_base → tcp`` chain.

Parses the WBC URDF with the standard library only (no Pinocchio). Used by
robot-world BA so the calibration environment does not import the controller.
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis RPY: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    rx, ry, rz = (float(v) for v in np.asarray(rpy, dtype=np.float64).reshape(3))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _axis_angle_R(axis: np.ndarray, q: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = a / n
    c = float(np.cos(q))
    s = float(np.sin(q))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _T_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _origin_T(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    return _T_from_Rt(_rpy_to_R(rpy), xyz)


def _floats(text: str | None, n: int, default: tuple[float, ...]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    vals = [float(p) for p in text.split()]
    if len(vals) != n:
        raise ValueError(f"expected {n} numbers, got {text!r}")
    return np.asarray(vals, dtype=np.float64)


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    jtype: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def parse_urdf_joints(urdf_path: Path) -> list[UrdfJoint]:
    tree = ET.parse(urdf_path)
    out: list[UrdfJoint] = []
    for joint in tree.getroot().findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        out.append(
            UrdfJoint(
                name=str(joint.get("name") or ""),
                jtype=str(joint.get("type") or "fixed"),
                parent=str(parent.get("link") or ""),
                child=str(child.get("link") or ""),
                xyz=_floats(origin.get("xyz") if origin is not None else None, 3, (0.0, 0.0, 0.0)),
                rpy=_floats(origin.get("rpy") if origin is not None else None, 3, (0.0, 0.0, 0.0)),
                axis=_floats(axis.get("xyz") if axis is not None else None, 3, (0.0, 0.0, 1.0)),
            )
        )
    return out


def _chain_to(joints: list[UrdfJoint], *, root: str, tip: str) -> list[UrdfJoint]:
    by_child = {j.child: j for j in joints}
    chain: list[UrdfJoint] = []
    cur = tip
    seen: set[str] = set()
    while cur != root:
        if cur in seen:
            raise RuntimeError(f"cycle walking URDF from {tip!r} to {root!r}")
        seen.add(cur)
        joint = by_child.get(cur)
        if joint is None:
            raise RuntimeError(f"no joint with child={cur!r} on path to {root!r}")
        chain.append(joint)
        cur = joint.parent
    chain.reverse()
    return chain


class UrdfFK:
    """``rail_base → tcp`` FK. ``q_arm_rad`` is joint_1..7."""

    def __init__(self, urdf_path: Path | str) -> None:
        self.path = Path(urdf_path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.sha1 = file_sha1(self.path)
        self.joints = parse_urdf_joints(self.path)
        self.chain = _chain_to(self.joints, root="rail_base", tip="tcp")
        tcp = next((j for j in self.chain if j.name == "link_7_to_tcp"), None)
        if tcp is None:
            raise RuntimeError(f"{self.path} has no link_7_to_tcp on the rail_base→tcp chain")
        self.link_7_to_tcp_xyz = tcp.xyz.copy()
        self.link_7_to_tcp_rpy = tcp.rpy.copy()

    def q_map(self, rail_m: float, q_arm_rad: np.ndarray) -> dict[str, float]:
        q = np.asarray(q_arm_rad, dtype=np.float64).reshape(-1)
        if q.size < 7:
            raise ValueError(f"q_arm_rad must have 7 values, got {q.size}")
        out = {"rail_y": float(rail_m)}
        for i in range(7):
            out[f"joint_{i + 1}"] = float(q[i])
        return out

    def fk(
        self,
        rail_m: float,
        q_arm_rad: np.ndarray,
        offsets_rad: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return ``T_railbase_tcp`` (4×4).

        ``offsets_rad`` is joint_1..6 (length 6) or joint_1..7 (length 7).
        joint_7 offset is ignored when length is 6 and should stay 0.
        """
        q = np.asarray(q_arm_rad, dtype=np.float64).reshape(-1)[:7].copy()
        if offsets_rad is not None:
            dq = np.asarray(offsets_rad, dtype=np.float64).reshape(-1)
            n = min(6, dq.size)
            q[:n] = q[:n] + dq[:n]
        qmap = self.q_map(rail_m, q)
        T = np.eye(4, dtype=np.float64)
        for joint in self.chain:
            T = T @ _origin_T(joint.xyz, joint.rpy)
            qj = float(qmap.get(joint.name, 0.0))
            if joint.jtype == "revolute" or joint.jtype == "continuous":
                T = T @ _T_from_Rt(_axis_angle_R(joint.axis, qj), np.zeros(3))
            elif joint.jtype == "prismatic":
                axis = joint.axis / (np.linalg.norm(joint.axis) + 1e-12)
                T = T @ _T_from_Rt(np.eye(3), axis * qj)
        return T

    def T_link7_tcp(self) -> np.ndarray:
        """Fixed URDF ``link_7 → tcp`` (the flange-to-tool frame)."""
        return _origin_T(self.link_7_to_tcp_xyz, self.link_7_to_tcp_rpy)

    def T_railbase_link7(self, T_railbase_tcp: np.ndarray) -> np.ndarray:
        """``T_railbase_link7 = T_railbase_tcp @ inv(T_link7_tcp)``."""
        T_rt = np.asarray(T_railbase_tcp, dtype=np.float64).reshape(4, 4)
        return T_rt @ np.linalg.inv(self.T_link7_tcp())


__all__ = [
    "UrdfFK",
    "UrdfJoint",
    "file_sha1",
    "parse_urdf_joints",
]
