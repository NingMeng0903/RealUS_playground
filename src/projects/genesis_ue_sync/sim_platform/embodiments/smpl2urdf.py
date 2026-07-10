from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import (
    HumanMotionSequence,
    build_shape_neutral_body_geometry,
    evaluate_smpl_sequence,
)

SMPL_PROXY_BODY_NAMES: tuple[str, ...] = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
)
SMPL_PROXY_KINEMATIC_ORDER: tuple[str, ...] = (
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
)


_BODY_INDEX = {name: idx for idx, name in enumerate(SMPL_PROXY_BODY_NAMES)}
_PARENT_BY_BODY: dict[str, str | None] = {
    "Pelvis": None,
    "L_Hip": "Pelvis",
    "R_Hip": "Pelvis",
    "Torso": "Pelvis",
    "L_Knee": "L_Hip",
    "R_Knee": "R_Hip",
    "Spine": "Torso",
    "L_Ankle": "L_Knee",
    "R_Ankle": "R_Knee",
    "Chest": "Spine",
    "L_Toe": "L_Ankle",
    "R_Toe": "R_Ankle",
    "Neck": "Chest",
    "L_Thorax": "Chest",
    "R_Thorax": "Chest",
    "Head": "Neck",
    "L_Shoulder": "L_Thorax",
    "R_Shoulder": "R_Thorax",
    "L_Elbow": "L_Shoulder",
    "R_Elbow": "R_Shoulder",
    "L_Wrist": "L_Elbow",
    "R_Wrist": "R_Elbow",
    "L_Hand": "L_Wrist",
    "R_Hand": "R_Wrist",
}

_SEGMENT_END_BY_BODY: dict[str, str | None] = {
    "Pelvis": "Torso",
    "L_Hip": "L_Knee",
    "R_Hip": "R_Knee",
    "Torso": "Spine",
    "L_Knee": "L_Ankle",
    "R_Knee": "R_Ankle",
    "Spine": "Chest",
    "L_Ankle": "L_Toe",
    "R_Ankle": "R_Toe",
    "Chest": "Neck",
    "L_Toe": None,
    "R_Toe": None,
    "Neck": "Head",
    "L_Thorax": "L_Shoulder",
    "R_Thorax": "R_Shoulder",
    "Head": None,
    "L_Shoulder": "L_Elbow",
    "R_Shoulder": "R_Elbow",
    "L_Elbow": "L_Wrist",
    "R_Elbow": "R_Wrist",
    "L_Wrist": "L_Hand",
    "R_Wrist": "R_Hand",
    "L_Hand": None,
    "R_Hand": None,
}
_RADIUS_GROUP_BY_BODY: dict[str, str] = {
    "Pelvis": "pelvis",
    "L_Hip": "leg",
    "R_Hip": "leg",
    "Torso": "torso",
    "L_Knee": "leg",
    "R_Knee": "leg",
    "Spine": "torso",
    "L_Ankle": "foot",
    "R_Ankle": "foot",
    "Chest": "torso",
    "L_Toe": "foot",
    "R_Toe": "foot",
    "Neck": "neck",
    "L_Thorax": "arm",
    "R_Thorax": "arm",
    "Head": "head",
    "L_Shoulder": "arm",
    "R_Shoulder": "arm",
    "L_Elbow": "arm",
    "R_Elbow": "arm",
    "L_Wrist": "arm",
    "R_Wrist": "arm",
    "L_Hand": "hand",
    "R_Hand": "hand",
}
_PRIMITIVE_TYPE_BY_BODY: dict[str, str] = {
    "Pelvis": "sphere",
    "L_Hip": "capsule",
    "R_Hip": "capsule",
    "Torso": "capsule",
    "L_Knee": "capsule",
    "R_Knee": "capsule",
    "Spine": "capsule",
    "L_Ankle": "box",
    "R_Ankle": "box",
    "Chest": "capsule",
    "L_Toe": "box",
    "R_Toe": "box",
    "Neck": "capsule",
    "L_Thorax": "capsule",
    "R_Thorax": "capsule",
    "Head": "sphere",
    "L_Shoulder": "capsule",
    "R_Shoulder": "capsule",
    "L_Elbow": "capsule",
    "R_Elbow": "capsule",
    "L_Wrist": "capsule",
    "R_Wrist": "capsule",
    "L_Hand": "sphere",
    "R_Hand": "sphere",
}
_CAPSULE_SHRINK_BY_BODY: dict[str, float] = {
    "Torso": 0.7,
    "Spine": 0.7,
    "Chest": 0.7,
    "L_Hip": 0.7,
    "R_Hip": 0.7,
    "L_Knee": 0.9,
    "R_Knee": 0.9,
}
_SPHERE_SHRINK_BY_BODY: dict[str, float] = {
    "Pelvis": 0.6,
}
_BASE_BODY_DENSITY_KG_M3 = 1000.0
_MIN_PROXY_BODY_MASS_KG = 0.05
_MASS_DENSITY_BY_GROUP = {
    "pelvis": 34.0,
    "torso": 26.0,
    "leg": 18.0,
    "foot": 14.0,
    "arm": 10.0,
    "hand": 7.0,
    "neck": 12.0,
    "head": 16.0,
}
_MIN_RADIUS_BY_GROUP = {
    "pelvis": 0.070,
    "torso": 0.050,
    "leg": 0.038,
    "foot": 0.030,
    "arm": 0.028,
    "hand": 0.030,
    "neck": 0.030,
    "head": 0.050,
}
_MAX_RADIUS_BY_GROUP = {
    "pelvis": 0.120,
    "torso": 0.110,
    "leg": 0.070,
    "foot": 0.055,
    "arm": 0.050,
    "hand": 0.048,
    "neck": 0.045,
    "head": 0.085,
}


@dataclass(frozen=True)
class ProxyBodyGeometry:
    name: str
    parent_name: str | None
    joint_idx: int
    parent_joint_idx: int | None
    end_joint_idx: int | None
    joint_origin_xyz: tuple[float, float, float]
    capsule_length_m: float
    capsule_radius_m: float
    capsule_axis_world: tuple[float, float, float]
    mass_kg: float
    group: str
    primitive_type: str = "capsule"
    collision_origin_xyz: tuple[float, float, float] | None = None
    box_size_xyz: tuple[float, float, float] | None = None
    sphere_radius_m: float | None = None
    volume_m3: float = 0.0
    density_kg_m3: float = _BASE_BODY_DENSITY_KG_M3


@dataclass(frozen=True)
class ProxyGeometry:
    model_type: str
    gender: str
    shape_key: str
    hip_width_m: float
    shoulder_width_m: float
    torso_height_m: float
    torso_depth_m: float
    bodies: tuple[ProxyBodyGeometry, ...]

    @property
    def body_by_name(self) -> dict[str, ProxyBodyGeometry]:
        return {body.name: body for body in self.bodies}


@dataclass(frozen=True)
class ProxyPointCloudConfig:
    axial_samples: int = 4
    radial_samples: int = 10
    cap_rings: int = 2
    include_joint_centers: bool = True


def shape_key_from_params(
    *,
    model_type: str,
    gender: str,
    betas: np.ndarray,
    decimals: int = 5,
) -> str:
    arr = np.round(np.asarray(betas, dtype=np.float32).reshape(-1), decimals=decimals)
    payload = "|".join(
        [
            str(model_type).lower(),
            str(gender).lower(),
            ",".join(f"{float(v):.{decimals}f}" for v in arr.tolist()),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _neutral_shape_sequence(sequence: HumanMotionSequence) -> HumanMotionSequence:
    return HumanMotionSequence(
        source_dataset=sequence.source_dataset,
        sequence_name=f"{sequence.sequence_name}_shape_proxy",
        source_path=sequence.source_path,
        model_type=sequence.model_type,
        fps=sequence.fps,
        gender=sequence.gender,
        betas=np.asarray(sequence.betas, dtype=np.float32).copy(),
        poses=np.zeros((1, int(sequence.poses.shape[1])), dtype=np.float32),
        trans=np.zeros((1, 3), dtype=np.float32),
        image_names=[],
        cam_int=None,
        cam_ext=None,
        metadata={},
    )


def shape_joints_from_sequence(
    sequence: HumanMotionSequence,
    *,
    device: str | None = "cpu",
) -> np.ndarray:
    neutral = _neutral_shape_sequence(sequence)
    _, joints = evaluate_smpl_sequence(neutral, device=device, include_vertices=False, include_joints=True)
    if joints is None or joints.shape[0] == 0:
        raise RuntimeError("Failed to evaluate SMPL joints for shape proxy generation.")
    return np.asarray(joints[0], dtype=np.float32)


def _safe_unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return arr / norm


def _perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = _safe_unit(axis, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(direction, ref))) > 0.95:
        ref = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    tangent = np.cross(direction, ref)
    tangent = _safe_unit(tangent, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
    bitangent = _safe_unit(np.cross(direction, tangent), np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
    return tangent, bitangent


def _body_radius(
    *,
    body_name: str,
    group: str,
    hip_width_m: float,
    shoulder_width_m: float,
    torso_depth_m: float,
    segment_length_m: float,
) -> float:
    if group == "pelvis":
        raw = 0.34 * hip_width_m
    elif group == "torso":
        raw = max(0.55 * torso_depth_m, 0.24 * segment_length_m)
    elif group == "leg":
        raw = 0.18 * segment_length_m + 0.05 * hip_width_m
    elif group == "foot":
        raw = 0.14 * max(segment_length_m, 0.08)
    elif group == "arm":
        raw = 0.16 * segment_length_m + 0.025 * shoulder_width_m
    elif group == "hand":
        raw = 0.24 * max(segment_length_m, 0.05)
    elif group == "neck":
        raw = 0.16 * max(segment_length_m, 0.07)
    elif group == "head":
        raw = 0.52 * torso_depth_m
    else:
        raw = 0.18 * max(segment_length_m, 0.1)
    if body_name in {"L_Thorax", "R_Thorax"}:
        raw *= 1.10
    return float(np.clip(raw, _MIN_RADIUS_BY_GROUP[group], _MAX_RADIUS_BY_GROUP[group]))


@dataclass(frozen=True)
class _BodyShapeStats:
    volume_m3: float
    bbox_size_xyz: np.ndarray
    bbox_center_xyz: np.ndarray
    vertex_count: int


def _bbox_volume(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    size = np.ptp(points, axis=0)
    return float(max(np.prod(np.maximum(size, 1e-5)), 0.0))


def _convex_hull_volume(points: np.ndarray) -> float:
    pts = np.unique(np.asarray(points, dtype=np.float64).reshape(-1, 3), axis=0)
    if pts.shape[0] < 4:
        return _bbox_volume(pts)
    try:
        from scipy.spatial import ConvexHull

        return float(max(ConvexHull(pts).volume, 0.0))
    except Exception:
        return _bbox_volume(pts)


def _shape_stats_by_body(
    *,
    vertices: np.ndarray | None,
    joints: np.ndarray,
    skin_weights: np.ndarray | None,
) -> dict[str, _BodyShapeStats]:
    if vertices is None or skin_weights is None:
        return {}
    verts = np.asarray(vertices, dtype=np.float64)
    weights = np.asarray(skin_weights, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] < 3 or weights.ndim != 2 or weights.shape[0] != verts.shape[0]:
        return {}

    body_count = min(len(SMPL_PROXY_BODY_NAMES), weights.shape[1])
    assignments = np.argmax(weights[:, :body_count], axis=1)
    out: dict[str, _BodyShapeStats] = {}
    for body_name in SMPL_PROXY_BODY_NAMES:
        joint_idx = int(_BODY_INDEX[body_name])
        if joint_idx >= body_count:
            continue
        body_vertices = verts[assignments == joint_idx, :3]
        if body_vertices.size == 0:
            continue
        local_vertices = body_vertices - np.asarray(joints[joint_idx, :3], dtype=np.float64)
        bbox_min = np.min(local_vertices, axis=0)
        bbox_max = np.max(local_vertices, axis=0)
        bbox_size = np.maximum(bbox_max - bbox_min, 1e-4)
        volume = _convex_hull_volume(local_vertices)
        if volume <= 0.0:
            volume = _bbox_volume(local_vertices)
        out[body_name] = _BodyShapeStats(
            volume_m3=float(max(volume, 1e-8)),
            bbox_size_xyz=np.asarray(bbox_size, dtype=np.float64),
            bbox_center_xyz=np.asarray((bbox_min + bbox_max) * 0.5, dtype=np.float64),
            vertex_count=int(body_vertices.shape[0]),
        )
    return out


def _capsule_volume(length_m: float, radius_m: float) -> float:
    length = max(float(length_m), 0.0)
    radius = max(float(radius_m), 0.0)
    return float(math.pi * radius * radius * length + (4.0 / 3.0) * math.pi * radius**3)


def _sphere_volume(radius_m: float) -> float:
    radius = max(float(radius_m), 0.0)
    return float((4.0 / 3.0) * math.pi * radius**3)


def _capsule_radius_from_volume(volume_m3: float, length_m: float, fallback_m: float) -> float:
    volume = max(float(volume_m3), 1e-9)
    length = max(float(length_m), 1e-5)
    roots = np.polynomial.polynomial.Polynomial([-volume, 0.0, math.pi * length, (4.0 / 3.0) * math.pi]).roots()
    real_positive = roots.real[(np.abs(roots.imag) < 1e-5) & (roots.real > 0.0)]
    if real_positive.size == 0:
        return float(max(fallback_m, 1e-3))
    return float(max(np.min(real_positive), 1e-3))


def _capsule_separation(body_name: str) -> float:
    return 0.45 if body_name in {"Torso", "Chest", "Spine"} else 0.20


def _tuple3(values: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(3)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def build_proxy_geometry_from_shape_joints(
    shape_joints: np.ndarray,
    *,
    model_type: str,
    gender: str,
    betas: np.ndarray,
    shape_vertices: np.ndarray | None = None,
    skin_weights: np.ndarray | None = None,
) -> ProxyGeometry:
    joints = np.asarray(shape_joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] < 3:
        raise ValueError(f"Expected shape joints with shape (J, 3+), got {joints.shape}")
    shape_stats = _shape_stats_by_body(vertices=shape_vertices, joints=joints, skin_weights=skin_weights)
    hip_width = float(np.linalg.norm(joints[_BODY_INDEX["R_Hip"], :3] - joints[_BODY_INDEX["L_Hip"], :3]))
    shoulder_width = float(
        np.linalg.norm(joints[_BODY_INDEX["R_Shoulder"], :3] - joints[_BODY_INDEX["L_Shoulder"], :3])
    )
    torso_height = float(np.linalg.norm(joints[_BODY_INDEX["Neck"], :3] - joints[_BODY_INDEX["Pelvis"], :3]))
    torso_depth = float(np.clip(0.52 * hip_width + 0.12 * torso_height, 0.16, 0.42))
    shape_key = shape_key_from_params(model_type=model_type, gender=gender, betas=betas)
    bodies: list[ProxyBodyGeometry] = []
    for body_name in SMPL_PROXY_BODY_NAMES:
        joint_idx = int(_BODY_INDEX[body_name])
        parent_name = _PARENT_BY_BODY[body_name]
        parent_joint_idx = None if parent_name is None else int(_BODY_INDEX[parent_name])
        end_name = _SEGMENT_END_BY_BODY[body_name]
        end_joint_idx = None if end_name is None else int(_BODY_INDEX[end_name])
        # URDF `<joint origin>` must match neutral SMPL parent→child displacement. Capsule sphere/box
        # placements and `_capsule_collision_rpy_xyz` are authored from the same neutral joints;
        # swapping in CRISP template offsets desynchronizes linkage from collision/inertia ("exploded" links).

        joint_origin = (
            np.zeros(3, dtype=np.float64)
            if parent_joint_idx is None
            else np.asarray(joints[joint_idx, :3] - joints[parent_joint_idx, :3], dtype=np.float64)
        )
        if body_name == "Pelvis":
            axis = _safe_unit(
                joints[_BODY_INDEX["Torso"], :3] - joints[_BODY_INDEX["Pelvis"], :3],
                np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            )
            segment_length = float(max(0.62 * hip_width, 0.12))
        elif end_joint_idx is None:
            if parent_joint_idx is None:
                axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                axis = _safe_unit(joints[joint_idx, :3] - joints[parent_joint_idx, :3], np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
            segment_length = float(max(np.linalg.norm(joints[joint_idx, :3] - joints[parent_joint_idx, :3]), 0.06))
            if body_name in {"Head"}:
                segment_length = float(max(segment_length, 0.11))
        else:
            axis = _safe_unit(joints[end_joint_idx, :3] - joints[joint_idx, :3], np.asarray([0.0, 0.0, -1.0], dtype=np.float64))
            segment_length = float(max(np.linalg.norm(joints[end_joint_idx, :3] - joints[joint_idx, :3]), 0.06))
        group = str(_RADIUS_GROUP_BY_BODY[body_name])
        heuristic_radius = _body_radius(
            body_name=body_name,
            group=group,
            hip_width_m=hip_width,
            shoulder_width_m=shoulder_width,
            torso_depth_m=torso_depth,
            segment_length_m=segment_length,
        )
        stats = shape_stats.get(body_name)
        primitive_type = _PRIMITIVE_TYPE_BY_BODY[body_name]
        density = _BASE_BODY_DENSITY_KG_M3
        volume = float(stats.volume_m3) if stats is not None else 0.0
        box_size: tuple[float, float, float] | None = None
        sphere_radius: float | None = None
        collision_origin = 0.5 * segment_length * axis
        if primitive_type == "capsule":
            sep = _capsule_separation(body_name)
            capsule_length = float(max((1.0 - 2.0 * sep) * segment_length, 0.015))
            radius = heuristic_radius
            if stats is not None:
                radius = _capsule_radius_from_volume(stats.volume_m3, capsule_length, fallback_m=heuristic_radius)
            shrink = float(_CAPSULE_SHRINK_BY_BODY.get(body_name, 1.0))
            radius *= shrink
            if shrink != 1.0:
                density = _BASE_BODY_DENSITY_KG_M3 / (shrink * shrink)
            mass = float(max(_capsule_volume(capsule_length, radius) * density, _MIN_PROXY_BODY_MASS_KG))
        elif primitive_type == "sphere":
            if stats is not None:
                radius = float(max((3.0 * stats.volume_m3 / (4.0 * math.pi)) ** (1.0 / 3.0), 1e-3))
            else:
                radius = heuristic_radius
            shrink = float(_SPHERE_SHRINK_BY_BODY.get(body_name, 1.0))
            radius *= shrink
            if shrink != 1.0:
                density = _BASE_BODY_DENSITY_KG_M3 / (shrink**3)
            sphere_radius = float(radius)
            capsule_length = float(2.0 * radius)
            collision_origin = np.zeros(3, dtype=np.float64)
            mass = float(max(_sphere_volume(radius) * density, _MIN_PROXY_BODY_MASS_KG))
        elif primitive_type == "box":
            if stats is not None:
                size = np.maximum(stats.bbox_size_xyz, 0.015)
                collision_origin = stats.bbox_center_xyz
                volume = float(stats.volume_m3)
            else:
                size = np.asarray([max(segment_length * 0.45, 0.04), max(heuristic_radius * 2.0, 0.03), max(heuristic_radius, 0.02)], dtype=np.float64)
            box_volume = float(max(np.prod(size), 1e-8))
            if volume > 0.0:
                density = _BASE_BODY_DENSITY_KG_M3 * volume / box_volume
            radius = float(max(np.linalg.norm(size) * 0.25, 1e-3))
            capsule_length = float(max(segment_length, 0.015))
            box_size = _tuple3(size)
            mass = float(max(box_volume * density, _MIN_PROXY_BODY_MASS_KG))
        else:
            raise ValueError(f"Unsupported primitive type for {body_name}: {primitive_type}")
        bodies.append(
            ProxyBodyGeometry(
                name=body_name,
                parent_name=parent_name,
                joint_idx=joint_idx,
                parent_joint_idx=parent_joint_idx,
                end_joint_idx=end_joint_idx,
                joint_origin_xyz=(float(joint_origin[0]), float(joint_origin[1]), float(joint_origin[2])),
                capsule_length_m=float(capsule_length),
                capsule_radius_m=float(radius),
                capsule_axis_world=(float(axis[0]), float(axis[1]), float(axis[2])),
                mass_kg=float(mass),
                group=group,
                primitive_type=primitive_type,
                collision_origin_xyz=_tuple3(collision_origin),
                box_size_xyz=box_size,
                sphere_radius_m=sphere_radius,
                volume_m3=float(volume),
                density_kg_m3=float(density),
            )
        )
    return ProxyGeometry(
        model_type=str(model_type).lower(),
        gender=str(gender).lower(),
        shape_key=shape_key,
        hip_width_m=float(max(hip_width, 1e-6)),
        shoulder_width_m=float(max(shoulder_width, 1e-6)),
        torso_height_m=float(max(torso_height, 1e-6)),
        torso_depth_m=float(torso_depth),
        bodies=tuple(bodies),
    )


def build_proxy_geometry_for_sequence(
    sequence: HumanMotionSequence,
    *,
    device: str | None = "cpu",
    shape_joints: np.ndarray | None = None,
) -> ProxyGeometry:
    shape_vertices = None
    skin_weights = None
    if shape_joints is None:
        shape_vertices, joints, skin_weights = build_shape_neutral_body_geometry(sequence, device=device)
    else:
        joints = np.asarray(shape_joints, dtype=np.float32)
    return build_proxy_geometry_from_shape_joints(
        joints,
        model_type=str(sequence.model_type),
        gender=str(sequence.gender),
        betas=np.asarray(sequence.betas, dtype=np.float32),
        shape_vertices=shape_vertices,
        skin_weights=skin_weights,
    )


def resolve_smpl_proxy_urdf(
    sequence: HumanMotionSequence,
    *,
    cache_dir: Path | None = None,
    device: str | None = "cpu",
    shape_joints: np.ndarray | None = None,
    collision_only: bool = False,
    force_rewrite: bool = False,
) -> tuple[Path, ProxyGeometry]:
    """Return PHC bundled MJCF cache placeholder URDF + ``ProxyGeometry`` mass summary (legacy API name).

    Hand-written shape capsule URDF output was removed; UE metadata callers still receive a stub URDF path.
    """
    del device, shape_joints, collision_only
    from projects.genesis_ue_sync.sim_platform.embodiments.phc_bundled_mjcf_proxy import (  # noqa: PLC0415
        sync_phc_bundled_proxy_to_cache,
    )
    from projects.genesis_ue_sync.sim_platform.human_motion.dependencies import human_motion_dependencies  # noqa: PLC0415

    root = project_paths(__file__).tmp_root if cache_dir is None else Path(cache_dir)
    deps = {d.name: d for d in human_motion_dependencies()}
    phc_root = deps["PHC"].resolved_path()
    _, _, placeholder_urdf, proxy_geometry = sync_phc_bundled_proxy_to_cache(
        sequence,
        cache_dir=root,
        phc_root=phc_root,
        force_rewrite=bool(force_rewrite),
    )
    return placeholder_urdf, proxy_geometry


def _capsule_segment_for_frame(
    joints_frame: np.ndarray,
    body: ProxyBodyGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    joints = np.asarray(joints_frame, dtype=np.float64)
    joint_pos = np.asarray(joints[body.joint_idx, :3], dtype=np.float64)
    if body.end_joint_idx is not None and int(body.end_joint_idx) < joints.shape[0]:
        end_pos = np.asarray(joints[body.end_joint_idx, :3], dtype=np.float64)
        axis = _safe_unit(end_pos - joint_pos, np.asarray(body.capsule_axis_world, dtype=np.float64))
    else:
        axis = _safe_unit(np.asarray(body.capsule_axis_world, dtype=np.float64), np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    origin = np.asarray(body.collision_origin_xyz or (0.0, 0.0, 0.0), dtype=np.float64)
    neutral_axis = _safe_unit(np.asarray(body.capsule_axis_world, dtype=np.float64), axis)
    center_offset = float(np.dot(origin, neutral_axis))
    center = joint_pos + axis * center_offset
    half = 0.5 * float(body.capsule_length_m)
    return center - half * axis, center + half * axis


def _primitive_center_for_frame(joints_frame: np.ndarray, body: ProxyBodyGeometry) -> np.ndarray:
    joints = np.asarray(joints_frame, dtype=np.float64)
    joint_pos = np.asarray(joints[body.joint_idx, :3], dtype=np.float64)
    return joint_pos + np.asarray(body.collision_origin_xyz or (0.0, 0.0, 0.0), dtype=np.float64)


def _sample_sphere_points(center: np.ndarray, radius: float, *, radial_steps: int, cap_rings: int) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    rings = max(cap_rings * 2 + 1, 3)
    steps = max(radial_steps, 6)
    for ring_idx in range(rings):
        phi = math.pi * float(ring_idx + 1) / float(rings + 1)
        z = math.cos(phi) * radius
        radial = math.sin(phi) * radius
        for radial_idx in range(steps):
            theta = 2.0 * math.pi * float(radial_idx) / float(steps)
            points.append(center + np.asarray([math.cos(theta) * radial, math.sin(theta) * radial, z], dtype=np.float64))
    points.append(center + np.asarray([0.0, 0.0, radius], dtype=np.float64))
    points.append(center - np.asarray([0.0, 0.0, radius], dtype=np.float64))
    return points


def _sample_box_points(center: np.ndarray, size_xyz: tuple[float, float, float], *, axial_steps: int) -> list[np.ndarray]:
    half = 0.5 * np.asarray(size_xyz, dtype=np.float64).reshape(3)
    steps = max(int(axial_steps), 1)
    points: list[np.ndarray] = []
    for sx in (-1.0, 1.0):
        for iy in range(steps + 1):
            for iz in range(steps + 1):
                yz = -half[1:] + 2.0 * half[1:] * np.asarray([iy, iz], dtype=np.float64) / float(steps)
                points.append(center + np.asarray([sx * half[0], yz[0], yz[1]], dtype=np.float64))
    for sy in (-1.0, 1.0):
        for ix in range(steps + 1):
            for iz in range(steps + 1):
                xz = -half[[0, 2]] + 2.0 * half[[0, 2]] * np.asarray([ix, iz], dtype=np.float64) / float(steps)
                points.append(center + np.asarray([xz[0], sy * half[1], xz[1]], dtype=np.float64))
    for sz in (-1.0, 1.0):
        for ix in range(steps + 1):
            for iy in range(steps + 1):
                xy = -half[:2] + 2.0 * half[:2] * np.asarray([ix, iy], dtype=np.float64) / float(steps)
                points.append(center + np.asarray([xy[0], xy[1], sz * half[2]], dtype=np.float64))
    return points


def sample_proxy_surface_points(
    joints_frame: np.ndarray,
    proxy_geometry: ProxyGeometry,
    *,
    config: ProxyPointCloudConfig | None = None,
) -> np.ndarray:
    cfg = config or ProxyPointCloudConfig()
    points: list[np.ndarray] = []
    for body in proxy_geometry.bodies:
        if body.primitive_type == "sphere":
            points.extend(
                _sample_sphere_points(
                    _primitive_center_for_frame(joints_frame, body),
                    float(body.sphere_radius_m or body.capsule_radius_m),
                    radial_steps=max(int(cfg.radial_samples), 4),
                    cap_rings=max(int(cfg.cap_rings), 1),
                )
            )
            continue
        if body.primitive_type == "box":
            points.extend(
                _sample_box_points(
                    _primitive_center_for_frame(joints_frame, body),
                    body.box_size_xyz or (body.capsule_length_m, 2.0 * body.capsule_radius_m, 2.0 * body.capsule_radius_m),
                    axial_steps=max(int(cfg.axial_samples), 1),
                )
            )
            continue
        start, end = _capsule_segment_for_frame(joints_frame, body)
        axis = end - start
        tangent, bitangent = _perpendicular_basis(axis)
        axial_steps = max(int(cfg.axial_samples), 1)
        radial_steps = max(int(cfg.radial_samples), 4)
        cap_rings = max(int(cfg.cap_rings), 1)
        for axial_idx in range(axial_steps + 1):
            alpha = axial_idx / float(axial_steps)
            center = (1.0 - alpha) * start + alpha * end
            for radial_idx in range(radial_steps):
                theta = 2.0 * np.pi * float(radial_idx) / float(radial_steps)
                offset = np.cos(theta) * tangent + np.sin(theta) * bitangent
                points.append(center + offset * float(body.capsule_radius_m))
        axis_unit = _safe_unit(axis, np.asarray(body.capsule_axis_world, dtype=np.float64))
        for sign in (-1.0, 1.0):
            cap_center = start if sign < 0.0 else end
            for ring_idx in range(1, cap_rings + 1):
                phi = 0.5 * np.pi * float(ring_idx) / float(cap_rings + 1)
                radial = np.sin(phi) * float(body.capsule_radius_m)
                axial = np.cos(phi) * float(body.capsule_radius_m) * sign
                ring_center = cap_center + axis_unit * axial
                for radial_idx in range(radial_steps):
                    theta = 2.0 * np.pi * float(radial_idx) / float(radial_steps)
                    offset = np.cos(theta) * tangent + np.sin(theta) * bitangent
                    points.append(ring_center + offset * radial)
        if bool(cfg.include_joint_centers):
            points.append(start.copy())
            points.append(end.copy())
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def build_proxy_cloud_sequence(
    joints_seq: np.ndarray,
    proxy_geometry: ProxyGeometry,
    *,
    config: ProxyPointCloudConfig | None = None,
) -> np.ndarray:
    frames = np.asarray(joints_seq, dtype=np.float32)
    clouds = [sample_proxy_surface_points(frame, proxy_geometry, config=config) for frame in frames]
    if not clouds:
        return np.zeros((0, 0, 3), dtype=np.float32)
    point_count = max(int(cloud.shape[0]) for cloud in clouds)
    padded = np.full((len(clouds), point_count, 3), np.nan, dtype=np.float32)
    for idx, cloud in enumerate(clouds):
        padded[idx, : cloud.shape[0], :] = cloud
    return padded





def human_sequence_from_smpl_pkl(
    smpl_model_dir: Path,
    *,
    betas: np.ndarray | None = None,
    gender: str = "neutral",
) -> HumanMotionSequence:
    """Build a one-frame neutral-pose SMPL sequence for shape-only URDF generation.

    ``smpl_model_dir`` must contain ``SMPL_NEUTRAL.pkl`` (or gender-specific SMPL pkl).
    """
    smpl_model_dir = Path(smpl_model_dir)
    gender_l = str(gender).lower()
    if gender_l in {"male", "m"}:
        pkl_name = "SMPL_MALE.pkl"
    elif gender_l in {"female", "f"}:
        pkl_name = "SMPL_FEMALE.pkl"
    else:
        pkl_name = "SMPL_NEUTRAL.pkl"
    pkl_path = smpl_model_dir / pkl_name
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Expected SMPL pkl at {pkl_path}")
    betas_arr = np.zeros(10, dtype=np.float32) if betas is None else np.asarray(betas, dtype=np.float32).reshape(-1)[:10]
    if betas_arr.size < 10:
        betas_arr = np.pad(betas_arr, (0, 10 - betas_arr.size))
    return HumanMotionSequence(
        source_dataset="smpl_pkl",
        sequence_name=f"shape_from_{pkl_path.stem}",
        source_path=str(pkl_path),
        model_type="smpl",
        fps=30.0,
        gender=gender_l if gender_l in {"male", "female"} else "neutral",
        betas=betas_arr,
        poses=np.zeros((1, 72), dtype=np.float32),
        trans=np.zeros((1, 3), dtype=np.float32),
        image_names=[],
        cam_int=None,
        cam_ext=None,
        metadata={"smpl_model_dir": str(smpl_model_dir.resolve())},
    )
