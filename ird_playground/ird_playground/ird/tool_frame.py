"""Flange (link_7) ↔ TCP tool-frame transforms for flange-based RM4D charts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def _vec(text: str | None, size: int, default: tuple[float, ...]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    value = np.fromstring(text, sep=" ", dtype=np.float64)
    if value.shape != (size,):
        raise ValueError(f"expected {size} values, got {text!r}")
    return value


def _origin_matrix(node: ET.Element | None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if node is None:
        return T
    xyz = _vec(node.get("xyz"), 3, (0.0, 0.0, 0.0))
    rpy = _vec(node.get("rpy"), 3, (0.0, 0.0, 0.0))
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


@dataclass(frozen=True)
class ToolFrame:
    """Rigid flange→TCP transform ``T_flange_tcp`` parsed from URDF."""

    T_flange_tcp: np.ndarray  # (4, 4)
    joint_name: str = "link_7_to_tcp"
    parent_frame: str = "link_7"
    child_frame: str = "tcp"

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path,
        *,
        joint_name: str = "link_7_to_tcp",
    ) -> "ToolFrame":
        path = Path(urdf_path)
        root = ET.parse(path).getroot()
        for node in root.findall("joint"):
            if node.get("name") != joint_name:
                continue
            parent = node.find("parent")
            child = node.find("child")
            return cls(
                T_flange_tcp=_origin_matrix(node.find("origin")),
                joint_name=joint_name,
                parent_frame=str(parent.get("link")) if parent is not None else "link_7",
                child_frame=str(child.get("link")) if child is not None else "tcp",
            )
        raise ValueError(f"joint {joint_name!r} not found in {path}")

    @property
    def T_tcp_flange(self) -> np.ndarray:
        return np.linalg.inv(self.T_flange_tcp)

    def to_manifest(self) -> dict:
        return {
            "joint_name": self.joint_name,
            "parent_frame": self.parent_frame,
            "child_frame": self.child_frame,
            "T_flange_tcp": self.T_flange_tcp.tolist(),
            "T_tcp_flange": self.T_tcp_flange.tolist(),
        }


def _matmul_se3(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """Batch SE(3) multiply ``a @ b`` with broadcasting on leading dims."""
    Ra, pa = a[..., :3, :3], a[..., :3, 3]
    Rb, pb = b[..., :3, :3], b[..., :3, 3]
    R = Ra @ Rb
    p = (Ra @ pb.unsqueeze(-1)).squeeze(-1) + pa
    shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + (4, 4)
    out = a.new_zeros(shape)
    out[..., :3, :3] = R
    out[..., :3, 3] = p
    out[..., 3, 3] = 1.0
    return out


def _invert_se3(T: "torch.Tensor") -> "torch.Tensor":
    R = T[..., :3, :3]
    p = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -(Rt @ p.unsqueeze(-1)).squeeze(-1)
    out[..., 3, 3] = 1.0
    return out


def flange_to_tcp_torch(
    T_flange: "torch.Tensor",
    T_flange_tcp: "torch.Tensor",
) -> "torch.Tensor":
    """Compose ``T_tcp = T_flange @ T_flange_tcp``."""
    if torch is None:
        raise ImportError("torch required")
    tool = T_flange_tcp.to(dtype=T_flange.dtype, device=T_flange.device)
    return _matmul_se3(T_flange, tool)


def tcp_to_flange_torch(
    T_tcp: "torch.Tensor",
    T_flange_tcp: "torch.Tensor",
) -> "torch.Tensor":
    """Compose ``T_flange = T_tcp @ inv(T_flange_tcp)``."""
    if torch is None:
        raise ImportError("torch required")
    tool = T_flange_tcp.to(dtype=T_tcp.dtype, device=T_tcp.device)
    return _matmul_se3(T_tcp, _invert_se3(tool))


def pose_tcp_to_flange(
    position_tcp: "torch.Tensor",
    rotation_tcp: "torch.Tensor",
    T_flange_tcp: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Map TCP pose ``(p, R)`` in any common frame to the flange pose."""
    if torch is None:
        raise ImportError("torch required")
    tool = T_flange_tcp.to(dtype=position_tcp.dtype, device=position_tcp.device)
    T_tcp = position_tcp.new_zeros(position_tcp.shape[:-1] + (4, 4))
    T_tcp[..., :3, :3] = rotation_tcp
    T_tcp[..., :3, 3] = position_tcp
    T_tcp[..., 3, 3] = 1.0
    T_fl = tcp_to_flange_torch(T_tcp, tool)
    return T_fl[..., :3, 3], T_fl[..., :3, :3]


def pose_flange_to_tcp(
    position_flange: "torch.Tensor",
    rotation_flange: "torch.Tensor",
    T_flange_tcp: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    if torch is None:
        raise ImportError("torch required")
    tool = T_flange_tcp.to(dtype=position_flange.dtype, device=position_flange.device)
    T_fl = position_flange.new_zeros(position_flange.shape[:-1] + (4, 4))
    T_fl[..., :3, :3] = rotation_flange
    T_fl[..., :3, 3] = position_flange
    T_fl[..., 3, 3] = 1.0
    T_tcp = flange_to_tcp_torch(T_fl, tool)
    return T_tcp[..., :3, 3], T_tcp[..., :3, :3]


__all__ = [
    "ToolFrame",
    "flange_to_tcp_torch",
    "pose_flange_to_tcp",
    "pose_tcp_to_flange",
    "tcp_to_flange_torch",
]
