"""Versioned RealMan robot-asset contract for RM4D data and checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


ROBOT_CONTRACT_SCHEMA = "realman_robot_contract_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _joint_nodes(path: Path) -> dict[str, ET.Element]:
    root = ET.parse(path).getroot()
    return {
        str(node.get("name")): node
        for node in root.findall("joint")
        if node.get("name")
    }


def _mesh_hashes(urdf_path: Path) -> dict[str, str]:
    root = ET.parse(urdf_path).getroot()
    hashes: dict[str, str] = {}
    for mesh in root.findall(".//mesh"):
        raw = mesh.get("filename")
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = urdf_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"URDF mesh not found: {path}")
        hashes[str(path)] = _sha256(path)
    return dict(sorted(hashes.items()))


@dataclass(frozen=True)
class RobotModelSpec:
    kinematics_urdf: Path
    collision_urdf: Path
    collision_pairs: Path
    tcp_frame: str = "tcp"
    tcp_joint: str = "link_7_to_tcp"
    root_frame: str = "rail_base"
    rail_joint: str = "rail_y"
    rail_locked_at_m: float = 0.0
    j1_joint: str = "joint_1"

    @classmethod
    def default_probe45(cls) -> "RobotModelSpec":
        workspace = Path(__file__).resolve().parents[3]
        assets = (
            workspace
            / "rm75_control/rm75_control/assets/robots/rm75_6f_8dof"
        )
        return cls(
            kinematics_urdf=assets / "RM75-6F-8dof.urdf",
            collision_urdf=assets / "RM75-6F-8dof.collision.urdf",
            collision_pairs=assets / "collision_pairs.yaml",
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RobotModelSpec":
        config_path = Path(path).resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

        def resolve(value: str) -> Path:
            candidate = Path(value)
            return candidate.resolve() if candidate.is_absolute() else (config_path.parent / candidate).resolve()

        return cls(
            kinematics_urdf=resolve(str(raw["kinematics_urdf"])),
            collision_urdf=resolve(str(raw["collision_urdf"])),
            collision_pairs=resolve(str(raw["collision_pairs"])),
            tcp_frame=str(raw.get("tcp_frame", "tcp")),
            tcp_joint=str(raw.get("tcp_joint", "link_7_to_tcp")),
            root_frame=str(raw.get("root_frame", "rail_base")),
            rail_joint=str(raw.get("rail_joint", "rail_y")),
            rail_locked_at_m=float(raw.get("rail_locked_at_m", 0.0)),
            j1_joint=str(raw.get("j1_joint", "joint_1")),
        )

    def validate(self) -> None:
        for path in (self.kinematics_urdf, self.collision_urdf, self.collision_pairs):
            if not path.is_file():
                raise FileNotFoundError(path)
        kin = _joint_nodes(self.kinematics_urdf)
        collision = _joint_nodes(self.collision_urdf)
        for name in (self.rail_joint, self.j1_joint, self.tcp_joint):
            if name not in kin:
                raise ValueError(f"joint {name!r} missing from {self.kinematics_urdf}")
            if name not in collision:
                raise ValueError(f"joint {name!r} missing from {self.collision_urdf}")
        for name in (self.rail_joint, self.j1_joint, self.tcp_joint):
            a = _origin_matrix(kin[name].find("origin"))
            b = _origin_matrix(collision[name].find("origin"))
            if not np.allclose(a, b, atol=1.0e-9):
                raise ValueError(f"{name} origin differs between kinematics and collision URDF")
        kin_limit = kin[self.rail_joint].find("limit")
        col_limit = collision[self.rail_joint].find("limit")
        if kin_limit is None or col_limit is None:
            raise ValueError(f"{self.rail_joint} must define limits in both URDFs")
        for key in ("lower", "upper"):
            if not np.isclose(float(kin_limit.get(key, "nan")), float(col_limit.get(key, "nan"))):
                raise ValueError(f"{self.rail_joint} {key} limit differs between URDFs")
        lo, hi = self.rail_limits_m()
        if not lo <= self.rail_locked_at_m <= hi:
            raise ValueError(
                f"rail_locked_at_m={self.rail_locked_at_m} outside [{lo}, {hi}]"
            )
        _mesh_hashes(self.collision_urdf)

    def rail_limits_m(self) -> tuple[float, float]:
        joint = _joint_nodes(self.kinematics_urdf)[self.rail_joint]
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"joint {self.rail_joint!r} has no limit")
        return float(limit.get("lower", "nan")), float(limit.get("upper", "nan"))

    def root_to_j1_axis(self) -> np.ndarray:
        """Return ``T_root_j1_axis`` at the configured locked rail value."""
        root = ET.parse(self.kinematics_urdf).getroot()
        joints = list(root.findall("joint"))
        children = {
            str(node.find("child").get("link"))
            for node in joints
            if node.find("child") is not None
        }
        links = {str(node.get("name")) for node in root.findall("link")}
        roots = sorted(links - children)
        if self.root_frame not in roots:
            raise ValueError(
                f"configured root_frame={self.root_frame!r} is not a URDF root; roots={roots}"
            )
        link_transform: dict[str, np.ndarray] = {self.root_frame: np.eye(4)}
        remaining = list(joints)
        while remaining:
            progressed = False
            for node in remaining[:]:
                parent = node.find("parent")
                child = node.find("child")
                if parent is None or child is None:
                    remaining.remove(node)
                    continue
                parent_name = str(parent.get("link"))
                if parent_name not in link_transform:
                    continue
                T_joint = link_transform[parent_name] @ _origin_matrix(node.find("origin"))
                if node.get("name") == self.j1_joint:
                    axis = _vec(
                        node.find("axis").get("xyz") if node.find("axis") is not None else None,
                        3,
                        (1.0, 0.0, 0.0),
                    )
                    axis = axis / np.linalg.norm(axis)
                    if not np.allclose(axis, np.array([0.0, 0.0, 1.0]), atol=1.0e-8):
                        raise ValueError("RM4D canonical encoder currently requires J1 axis +Z")
                    return T_joint
                T_child = T_joint.copy()
                if node.get("name") == self.rail_joint:
                    axis_node = node.find("axis")
                    axis = _vec(
                        axis_node.get("xyz") if axis_node is not None else None,
                        3,
                        (1.0, 0.0, 0.0),
                    )
                    T_child[:3, 3] += T_child[:3, :3] @ (
                        axis * self.rail_locked_at_m
                    )
                link_transform[str(child.get("link"))] = T_child
                remaining.remove(node)
                progressed = True
            if not progressed:
                break
        raise ValueError(f"could not resolve axis transform for {self.j1_joint!r}")

    def to_manifest(self) -> dict:
        self.validate()
        rail_lo, rail_hi = self.rail_limits_m()
        return {
            "schema": ROBOT_CONTRACT_SCHEMA,
            "kinematics_urdf": str(self.kinematics_urdf.resolve()),
            "kinematics_urdf_sha256": _sha256(self.kinematics_urdf),
            "collision_urdf": str(self.collision_urdf.resolve()),
            "collision_urdf_sha256": _sha256(self.collision_urdf),
            "collision_pairs": str(self.collision_pairs.resolve()),
            "collision_pairs_sha256": _sha256(self.collision_pairs),
            "collision_mesh_sha256": _mesh_hashes(self.collision_urdf),
            "tcp_frame": self.tcp_frame,
            "tcp_joint": self.tcp_joint,
            "root_frame": self.root_frame,
            "rail_joint": self.rail_joint,
            "rail_locked_at_m": self.rail_locked_at_m,
            "rail_limits_m": [rail_lo, rail_hi],
            "j1_joint": self.j1_joint,
            "T_root_j1_axis": self.root_to_j1_axis().tolist(),
        }


def load_robot_model_spec(path: str | Path | None = None) -> RobotModelSpec:
    spec = RobotModelSpec.default_probe45() if path is None else RobotModelSpec.from_yaml(path)
    spec.validate()
    return spec


def assert_robot_contract_compatible(
    recorded: dict | None,
    expected: RobotModelSpec | None = None,
    *,
    allow_stale: bool = False,
) -> None:
    expected_manifest = (expected or RobotModelSpec.default_probe45()).to_manifest()
    if not recorded:
        if allow_stale:
            return
        raise RuntimeError(
            "artifact has no robot_contract metadata; it predates the physical probe45 contract"
        )
    keys = (
        "kinematics_urdf_sha256",
        "collision_urdf_sha256",
        "collision_pairs_sha256",
        "tcp_frame",
        "rail_locked_at_m",
        "T_root_j1_axis",
    )
    mismatches = [key for key in keys if recorded.get(key) != expected_manifest.get(key)]
    if recorded.get("collision_mesh_sha256") != expected_manifest.get("collision_mesh_sha256"):
        mismatches.append("collision_mesh_sha256")
    if mismatches and not allow_stale:
        raise RuntimeError(
            "robot asset contract mismatch for " + ", ".join(mismatches)
            + "; rebuild GT/map and retrain, or pass allow_stale=True only for comparison"
        )


__all__ = [
    "ROBOT_CONTRACT_SCHEMA",
    "RobotModelSpec",
    "assert_robot_contract_compatible",
    "load_robot_model_spec",
]
