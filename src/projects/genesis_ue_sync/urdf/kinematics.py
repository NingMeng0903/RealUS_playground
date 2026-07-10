from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from bridge.adapters.urdf import root_transform_from_pose

MESH_SOURCE_TO_UE_SCALE = 100.0


@dataclass(frozen=True)
class UrdfLinkSpec:
    name: str
    visual_origin_xyz: tuple[float, float, float]
    visual_origin_rpy: tuple[float, float, float]
    visual_mesh: str | None
    collision_origin_xyz: tuple[float, float, float]
    collision_origin_rpy: tuple[float, float, float]
    geometry: dict[str, Any] | None


@dataclass(frozen=True)
class UrdfJointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


@dataclass(frozen=True)
class UrdfModel:
    root_link: str
    links: dict[str, UrdfLinkSpec]
    joints: list[UrdfJointSpec]


def _parse_origin(node) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if node is None:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xyz = tuple(float(item) for item in node.attrib.get("xyz", "0 0 0").split())
    rpy = tuple(float(item) for item in node.attrib.get("rpy", "0 0 0").split())
    return xyz, rpy


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(float(angle)), np.sin(float(angle))
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(float(angle)), np.sin(float(angle))
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(float(angle)), np.sin(float(angle))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = (float(v) for v in rpy)
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def _axis_angle_matrix(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    vec = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    x, y, z = (vec / norm).tolist()
    c = float(np.cos(float(angle)))
    s = float(np.sin(float(angle)))
    t = 1.0 - c
    return np.asarray(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def make_transform(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _rpy_matrix(rpy)
    out[:3, 3] = np.asarray(xyz, dtype=np.float64).reshape(3)
    return out


def parse_urdf_model(urdf_path: str | Path) -> UrdfModel:
    urdf_path = Path(urdf_path).expanduser().resolve()
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links: dict[str, UrdfLinkSpec] = {}
    joints: list[UrdfJointSpec] = []
    child_links: set[str] = set()

    for link in root.findall("link"):
        visual = link.find("visual")
        visual_xyz, visual_rpy = _parse_origin(visual.find("origin") if visual is not None else None)
        visual_geometry = visual.find("geometry") if visual is not None else None
        visual_mesh = None
        if visual_geometry is not None and visual_geometry.find("mesh") is not None:
            visual_mesh = str(visual_geometry.find("mesh").attrib.get("filename"))
        collision = link.find("collision")
        collision_xyz, collision_rpy = _parse_origin(collision.find("origin") if collision is not None else None)
        geometry = collision.find("geometry") if collision is not None else None
        geometry_payload: dict[str, Any] | None = None
        if geometry is not None:
            if geometry.find("cylinder") is not None:
                node = geometry.find("cylinder")
                geometry_payload = {
                    "type": "cylinder",
                    "radius": float(node.attrib["radius"]),
                    "length": float(node.attrib["length"]),
                }
            elif geometry.find("box") is not None:
                node = geometry.find("box")
                geometry_payload = {
                    "type": "box",
                    "size": tuple(float(item) for item in node.attrib["size"].split()),
                }
        links[str(link.attrib["name"])] = UrdfLinkSpec(
            name=str(link.attrib["name"]),
            visual_origin_xyz=visual_xyz,
            visual_origin_rpy=visual_rpy,
            visual_mesh=visual_mesh,
            collision_origin_xyz=collision_xyz,
            collision_origin_rpy=collision_rpy,
            geometry=geometry_payload,
        )

    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        xyz, rpy = _parse_origin(joint.find("origin"))
        axis = (
            tuple(float(item) for item in joint.find("axis").attrib.get("xyz", "0 0 1").split())
            if joint.find("axis") is not None
            else (0.0, 0.0, 1.0)
        )
        joints.append(
            UrdfJointSpec(
                name=str(joint.attrib["name"]),
                joint_type=str(joint.attrib.get("type", "fixed")),
                parent=str(parent.attrib["link"]),
                child=str(child.attrib["link"]),
                xyz=xyz,
                rpy=rpy,
                axis=axis,
            )
        )
        child_links.add(str(child.attrib["link"]))

    roots = [name for name in links if name not in child_links]
    if len(roots) != 1:
        raise RuntimeError(f"URDF must have exactly one root link; found {roots!r}")
    return UrdfModel(root_link=roots[0], links=links, joints=joints)


def compose_link_visual_world_transform(
    link_world: np.ndarray,
    *,
    visual_origin_xyz: tuple[float, float, float],
    visual_origin_rpy: tuple[float, float, float],
    visual_basis_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Match UE spawn_robot_from_urdf: visual_local rotation post-multiplied by basis RPY (degrees)."""
    lw = np.asarray(link_world, dtype=np.float64).reshape(4, 4)
    visual_local = make_transform(visual_origin_xyz, visual_origin_rpy)
    roll_deg, pitch_deg, yaw_deg = (float(v) for v in visual_basis_rpy_deg)
    basis = _rpy_matrix(
        (math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)),
    )
    visual_local[:3, :3] = visual_local[:3, :3] @ basis
    return lw @ visual_local


def compute_link_world_transforms(
    *,
    urdf_path: str | Path,
    base_pos_m: tuple[float, float, float],
    base_quat_xyzw: tuple[float, float, float, float] | None,
    joint_positions: list[float],
) -> dict[str, np.ndarray]:
    model = parse_urdf_model(urdf_path)
    transforms: dict[str, np.ndarray] = {
        model.root_link: np.asarray(root_transform_from_pose(base_pos_m, base_quat_xyzw), dtype=np.float64).reshape(4, 4)
    }
    actuated_values = iter([float(v) for v in joint_positions])
    pending = list(model.joints)
    while pending:
        progressed = False
        next_pending: list[UrdfJointSpec] = []
        for joint in pending:
            if joint.parent not in transforms:
                next_pending.append(joint)
                continue
            local = make_transform(joint.xyz, joint.rpy)
            if joint.joint_type in {"revolute", "continuous"}:
                local[:3, :3] = local[:3, :3] @ _axis_angle_matrix(joint.axis, float(next(actuated_values, 0.0)))
            elif joint.joint_type == "prismatic":
                q = float(next(actuated_values, 0.0))
                axis = np.asarray(joint.axis, dtype=np.float64).reshape(3)
                norm = float(np.linalg.norm(axis))
                if norm > 1e-8:
                    axis = axis / norm
                else:
                    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                prismatic = make_transform(tuple(float(v) for v in (axis * q)), (0.0, 0.0, 0.0))
                local = local @ prismatic
            transforms[joint.child] = transforms[joint.parent] @ local
            progressed = True
        if not progressed:
            raise RuntimeError(
                "URDF joint chain could not be resolved (cycle or missing parent). "
                f"Pending joints: {[joint.name for joint in next_pending]}"
            )
        pending = next_pending
    return transforms
