"""Bone-chain, joint-anchor, and ligament classification diagnostics.

The original report only compared a mesh skinned with all of its weights with
the same mesh moved by its dominant bone.  That is a useful rigidity metric,
but it cannot detect two rigid components that have both moved away from their
shared joint.  The joint diagnostics below therefore operate on the Blender
bind bone endpoints and compare their posed shared anchor with the posed
SMPL-X joint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .anatomy_lbs import joint_global_transforms, source_bone_skinning_transforms, skin_vertices
from .rigged_asset import AnatomyRiggedAsset

ENDPOINT_LIMIT_M = 0.002
AXIS_LIMIT_DEG = 3.0
GAP_CHANGE_LIMIT_M = 0.002
HEAD_ORIENTATION_LIMIT_DEG = 2.0

SEGMENT_MESHES = {
    "forearm_left": ("Radius_L", "Ulna_L", "Forearm_Bone_L", "Forearm_Twist_L"),
    "forearm_right": ("Radius_R", "Ulna_R", "Forearm_Bone_R", "Forearm_Twist_R"),
    "shin_left": ("Tibia_L", "Fibula_L", "Tibia_Bone_L", "Tibia_Twist_L", "Patella_L"),
    "shin_right": ("Tibia_R", "Fibula_R", "Tibia_Bone_R", "Tibia_Twist_R", "Patella_R"),
    "shoulder_left": ("Scapula_L", "Humerus_L"),
    "shoulder_right": ("Scapula_R", "Humerus_R"),
    "head": ("Upper_Skull",),
}

# The names identify the controlling Blender chain, not anatomy mesh names.
# Endpoint selection is geometric (nearest head/tail to the SMPL-X rest joint),
# so this table does not encode a screenshot-specific offset or direction.
JOINT_CHAINS = {
    "hip_left": {
        "joint": "left_hip",
        "proximal": ("Hip_bone", "Hip_Organ_Hold_L"),
        "distal": ("Femur_Rot_L",),
        "axes": (("Femur_Rot_L", "left_hip", "left_knee"),),
    },
    "hip_right": {
        "joint": "right_hip",
        "proximal": ("Hip_bone", "Hip_Organ_Hold_R"),
        "distal": ("Femur_Rot_R",),
        "axes": (("Femur_Rot_R", "right_hip", "right_knee"),),
    },
    "shoulder_left": {
        "joint": "left_shoulder",
        "proximal": ("Scapula_Bone_L", "Clavicle_Rot_L"),
        "distal": ("Shoulder_Rotate_L",),
        "axes": (("Shoulder_Rotate_L", "left_shoulder", "left_elbow"),),
    },
    "shoulder_right": {
        "joint": "right_shoulder",
        "proximal": ("Scapula_Bone_R", "Clavicle_Rot_R"),
        "distal": ("Shoulder_Rotate_R",),
        "axes": (("Shoulder_Rotate_R", "right_shoulder", "right_elbow"),),
    },
    "elbow_left": {
        "joint": "left_elbow",
        "proximal": ("Shoulder_Rotate_L", "Elbow_Rot_L"),
        "distal": ("Forearm_Bone_L", "Forearm_Twist_L"),
        "axes": (
            ("Shoulder_Rotate_L", "left_shoulder", "left_elbow"),
            ("Forearm_Bone_L", "left_elbow", "left_wrist"),
        ),
    },
    "elbow_right": {
        "joint": "right_elbow",
        "proximal": ("Shoulder_Rotate_R", "Elbow_Rot_R"),
        "distal": ("Forearm_Bone_R", "Forearm_Twist_R"),
        "axes": (
            ("Shoulder_Rotate_R", "right_shoulder", "right_elbow"),
            ("Forearm_Bone_R", "right_elbow", "right_wrist"),
        ),
    },
    "wrist_left": {
        "joint": "left_wrist",
        "proximal": ("Forearm_Bone_L", "Forearm_Twist_L"),
        "distal": ("Wrist_Rotate_L",),
        "axes": (("Forearm_Bone_L", "left_elbow", "left_wrist"),),
    },
    "wrist_right": {
        "joint": "right_wrist",
        "proximal": ("Forearm_Bone_R", "Forearm_Twist_R"),
        "distal": ("Wrist_Rotate_R1", "Wrist_Rotate_R"),
        "axes": (("Forearm_Bone_R", "right_elbow", "right_wrist"),),
    },
    "index_proximal_left": {
        "joint": "left_index1",
        "proximal": ("Wrist_Rotate_L",),
        "distal": ("Fingers_Rotate_L4", "Finger_Index_L3"),
        "axes": (("Wrist_Rotate_L", "left_wrist", "left_index1"),),
    },
    "index_proximal_right": {
        "joint": "right_index1",
        "proximal": ("Wrist_Rotate_R1", "Wrist_Rotate_R"),
        "distal": ("Fingers_Rotate_R4", "bone309"),
        "axes": (("Wrist_Rotate_R1", "right_wrist", "right_index1"),),
    },
    "knee_left": {
        "joint": "left_knee",
        "proximal": ("Femur_Rot_L", "Knee_Rotate_L"),
        "distal": ("Tibia_Bone_L", "Tibia_Twist_L"),
        "axes": (
            ("Femur_Rot_L", "left_hip", "left_knee"),
            ("Tibia_Bone_L", "left_knee", "left_ankle"),
        ),
    },
    "knee_right": {
        "joint": "right_knee",
        "proximal": ("Femur_Rot_R", "Knee_Rotate_R"),
        "distal": ("Tibia_Bone_R", "Tibia_Twist_R"),
        "axes": (
            ("Femur_Rot_R", "right_hip", "right_knee"),
            ("Tibia_Bone_R", "right_knee", "right_ankle"),
        ),
    },
    "ankle_left": {
        "joint": "left_ankle",
        "proximal": ("Tibia_Bone_L", "Tibia_Twist_L"),
        "distal": ("Ankle_Rot_L",),
        "axes": (
            ("Tibia_Bone_L", "left_knee", "left_ankle"),
            ("Ankle_Rot_L", "left_ankle", "left_foot"),
        ),
    },
    "ankle_right": {
        "joint": "right_ankle",
        "proximal": ("Tibia_Bone_R", "Tibia_Twist_R"),
        "distal": ("Ankle_Rot_R",),
        "axes": (
            ("Tibia_Bone_R", "right_knee", "right_ankle"),
            ("Ankle_Rot_R", "right_ankle", "right_foot"),
        ),
    },
}

GEOMETRY_LANDMARK_MESHES: dict[str, dict[str, tuple[str, ...]]] = {
    "hip_left": {
        "proximal": ("ilium", "ischium", "pubis", "acetabul", "pelvis", "sacrum"),
        "distal": ("femur",),
    },
    "hip_right": {
        "proximal": ("ilium", "ischium", "pubis", "acetabul", "pelvis", "sacrum"),
        "distal": ("femur",),
    },
    "shoulder_left": {"proximal": ("scapula", "clavicle"), "distal": ("humerus",)},
    "shoulder_right": {"proximal": ("scapula", "clavicle"), "distal": ("humerus",)},
    "elbow_left": {"proximal": ("humerus",), "distal": ("radius", "ulna")},
    "elbow_right": {"proximal": ("humerus",), "distal": ("radius", "ulna")},
    "wrist_left": {
        "proximal": ("radius", "ulna"),
        "distal": (
            "capitate", "hamate", "lunate", "pisiform", "scaphoid",
            "trapezium", "trapezoid", "triquetrum",
        ),
    },
    "wrist_right": {
        "proximal": ("radius", "ulna"),
        "distal": (
            "capitate", "hamate", "lunate", "pisiform", "scaphoid",
            "trapezium", "trapezoid", "triquetrum",
        ),
    },
    "index_proximal_left": {
        "proximal": ("metacarpal",),
        "distal": ("proximal_phalanx_hand", "proximal_phalanges_hand"),
    },
    "index_proximal_right": {
        "proximal": ("metacarpal",),
        "distal": ("proximal_phalanx_hand", "proximal_phalanges_hand"),
    },
    "knee_left": {"proximal": ("femur",), "distal": ("tibia", "fibula", "patella")},
    "knee_right": {"proximal": ("femur",), "distal": ("tibia", "fibula", "patella")},
    "ankle_left": {"proximal": ("tibia", "fibula"), "distal": ("talus", "calcaneus")},
    "ankle_right": {"proximal": ("tibia", "fibula"), "distal": ("talus", "calcaneus")},
}


def _joint_surface_gap_limit(label: str) -> float:
    """Acceptance limits from the anatomy plan, independent of SMPL-X probes."""
    if label.startswith(("hip_", "shoulder_")):
        return 0.003
    if label.startswith("index_"):
        # The source has no cartilage mesh between metacarpal and phalanx;
        # retain its authored bony clearance while still detecting separation.
        return 0.007
    if label.startswith(("elbow_", "wrist_", "knee_", "ankle_")):
        return 0.005
    return ENDPOINT_LIMIT_M


def _mesh_slice(asset: AnatomyRiggedAsset, name: str) -> slice | None:
    if name not in asset.source_mesh_names:
        return None
    idx = asset.source_mesh_names.index(name)
    start, stop = map(int, asset.source_vertex_ranges[idx])
    return slice(start, stop)


def _bone_axis(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    axis /= max(float(np.linalg.norm(axis)), 1.0e-10)
    return axis


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / max(float(np.linalg.norm(a)), 1.0e-10)
    b = b / max(float(np.linalg.norm(b)), 1.0e-10)
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


def _undirected_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return min(_angle_deg(a, b), _angle_deg(-np.asarray(a), b))


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return np.asarray(transform[:3, :3], dtype=np.float64) @ point + np.asarray(
        transform[:3, 3], dtype=np.float64
    )


def _bone_endpoints(asset: AnatomyRiggedAsset) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return bind-space heads/tails, with a deterministic v2 fallback.

    V3 source templates contain exact Blender head/tail probes.  Legacy v2
    assets only contain bind matrices; for those, the matrix origin is the head
    and either a child's origin or the mapped SMPL-X segment length supplies the
    tail.  The report marks this lower-confidence path explicitly.
    """
    count = len(asset.source_bone_names or [])
    head = getattr(asset, "target_bone_head", None)
    tail = getattr(asset, "target_bone_tail", None)
    if head is None:
        head = getattr(asset, "source_bone_head", None)
    if tail is None:
        tail = getattr(asset, "source_bone_tail", None)
    if head is not None and tail is not None:
        h = np.asarray(head, dtype=np.float64)
        t = np.asarray(tail, dtype=np.float64)
        if h.shape == (count, 3) and t.shape == (count, 3):
            return h, t, False

    rest_global = np.asarray(asset.source_rest_global, dtype=np.float64)
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    heads = rest_global[:, :3, 3].copy()
    tails = heads.copy()
    rest_joints = np.asarray(asset.rest_joints, dtype=np.float64)
    children: list[list[int]] = [[] for _ in range(count)]
    for child, parent in enumerate(parents.tolist()):
        if parent >= 0:
            children[parent].append(child)
    for bi in range(count):
        if children[bi]:
            # A connected/nearest child is the best legacy estimate of a tail.
            candidate = min(children[bi], key=lambda ci: float(np.linalg.norm(heads[ci] - heads[bi])))
            if float(np.linalg.norm(heads[candidate] - heads[bi])) > 1.0e-8:
                tails[bi] = heads[candidate]
                continue
        a = int(asset.source_bone_smplx_a[bi])
        b = int(asset.source_bone_smplx_b[bi])
        length = float(np.linalg.norm(rest_joints[b] - rest_joints[a])) if a != b else 0.02
        axis = rest_global[bi, :3, 1]
        axis /= max(float(np.linalg.norm(axis)), 1.0e-10)
        tails[bi] = heads[bi] + max(length, 1.0e-3) * axis
    return heads, tails, True


def _first_bone_index(names: list[str], candidates: tuple[str, ...]) -> int | None:
    by_name = {name: idx for idx, name in enumerate(names)}
    return next((by_name[name] for name in candidates if name in by_name), None)


def _nearest_endpoint(head: np.ndarray, tail: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, str]:
    if float(np.linalg.norm(head - target)) <= float(np.linalg.norm(tail - target)):
        return head, "head"
    return tail, "tail"


def _joint_chain_diagnostic(
    asset: AnatomyRiggedAsset,
    *,
    spec: dict[str, Any],
    source_transforms: np.ndarray,
    rest_heads: np.ndarray,
    rest_tails: np.ndarray,
    posed_smplx_joints: np.ndarray,
    translation: np.ndarray,
    endpoint_fallback: bool,
) -> dict[str, Any]:
    joint_index = {name: idx for idx, name in enumerate(asset.joint_names)}
    source_names = list(asset.source_bone_names or [])
    joint_name = str(spec["joint"])
    ji = joint_index[joint_name]
    target_rest = np.asarray(asset.rest_joints[ji], dtype=np.float64)
    target_pose = np.asarray(posed_smplx_joints[ji], dtype=np.float64) + translation

    role_data: dict[str, Any] = {}
    anchors_rest: list[np.ndarray] = []
    anchors_pose: list[np.ndarray] = []
    selected_indices: list[int] = []
    for role in ("proximal", "distal"):
        bi = _first_bone_index(source_names, tuple(spec[role]))
        if bi is None:
            role_data[role] = {"available": False, "candidates": list(spec[role])}
            continue
        anchor_rest, endpoint = _nearest_endpoint(rest_heads[bi], rest_tails[bi], target_rest)
        anchor_pose = _transform_point(source_transforms[bi], anchor_rest) + translation
        anchors_rest.append(anchor_rest)
        anchors_pose.append(anchor_pose)
        selected_indices.append(bi)
        role_data[role] = {
            "available": True,
            "source_bone": source_names[bi],
            "endpoint": endpoint,
            "smplx_joint_error_m": float(np.linalg.norm(anchor_pose - target_pose)),
        }

    if len(anchors_pose) == 2:
        shared_pose = 0.5 * (anchors_pose[0] + anchors_pose[1])
        rest_gap = float(np.linalg.norm(anchors_rest[0] - anchors_rest[1]))
        posed_gap = float(np.linalg.norm(anchors_pose[0] - anchors_pose[1]))
        anchor_error = float(np.linalg.norm(shared_pose - target_pose))
        gap_change = abs(posed_gap - rest_gap)
    else:
        anchor_error = None
        rest_gap = posed_gap = gap_change = None

    axes: list[dict[str, Any]] = []
    for bone_name, smplx_a, smplx_b in spec["axes"]:
        bi = _first_bone_index(source_names, (bone_name,))
        if bi is None:
            axes.append({"source_bone": bone_name, "available": False})
            continue
        rest_axis = rest_tails[bi] - rest_heads[bi]
        posed_axis = np.asarray(source_transforms[bi, :3, :3], dtype=np.float64) @ rest_axis
        target_axis = posed_smplx_joints[joint_index[smplx_b]] - posed_smplx_joints[joint_index[smplx_a]]
        axes.append(
            {
                "source_bone": bone_name,
                "available": True,
                "smplx_segment": [smplx_a, smplx_b],
                "axis_error_deg": _undirected_angle_deg(posed_axis, target_axis),
            }
        )
    available_axis_errors = [
        float(item["axis_error_deg"]) for item in axes if item.get("available")
    ]
    axis_error = max(available_axis_errors) if available_axis_errors else None

    connected = False
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    if len(selected_indices) == 2:
        proximal, distal = selected_indices
        connected = bool(parents[distal] == proximal and rest_gap <= 1.0e-5)

    passed = bool(
        not endpoint_fallback
        and anchor_error is not None
        and gap_change is not None
        and axis_error is not None
        and np.isfinite(anchor_error)
        and anchor_error <= ENDPOINT_LIMIT_M
        and gap_change <= GAP_CHANGE_LIMIT_M
        and axis_error <= AXIS_LIMIT_DEG
    )
    return {
        "smplx_joint": joint_name,
        "endpoint_source": "derived_legacy_bind" if endpoint_fallback else "blender_bind_probes",
        "roles": role_data,
        "source_shared_anchor_error_m": anchor_error,
        "source_rest_gap_m": rest_gap,
        "posed_gap_m": posed_gap,
        "gap_change_m": gap_change,
        "axis_error_deg": axis_error,
        "axes": axes,
        "connected_in_source": connected,
        "pass": passed,
    }


def _side_matches(name: str, label: str) -> bool:
    if label.endswith("_left"):
        return name.endswith("_l") or "_l_" in name or "left" in name
    if label.endswith("_right"):
        return name.endswith("_r") or "_r_" in name or "right" in name
    return True


def _geometry_landmark_diagnostic(
    asset: AnatomyRiggedAsset,
    *,
    label: str,
    joint_name: str,
    posed_vertices: np.ndarray,
    posed_smplx_joints: np.ndarray,
    translation: np.ndarray,
) -> dict[str, Any]:
    """Measure anatomical mesh landmarks independently of controller endpoints."""
    spec = GEOMETRY_LANDMARK_MESHES.get(label)
    if spec is None:
        return {
            "available": False,
            "reason": "no geometry landmark recipe",
            "roles": {},
            "pass": False,
        }
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        return {
            "available": False,
            "reason": "mesh ranges or tissue labels are unavailable",
            "roles": {},
            "pass": False,
        }
    joint_index = {name: idx for idx, name in enumerate(asset.joint_names)}
    if joint_name not in joint_index:
        return {
            "available": False,
            "reason": f"SMPL-X joint {joint_name!r} is unavailable",
            "roles": {},
            "pass": False,
        }
    ji = joint_index[joint_name]
    target_rest = np.asarray(asset.rest_joints[ji], dtype=np.float64)
    target_pose = np.asarray(posed_smplx_joints[ji], dtype=np.float64) + translation
    rest_vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    registration_reference = getattr(asset, "registration_reference", None)
    source_vertices = np.asarray(
        registration_reference
        if registration_reference is not None
        else asset.vertices_rest,
        dtype=np.float64,
    )
    posed_vertices = np.asarray(posed_vertices, dtype=np.float64)

    roles: dict[str, Any] = {}
    role_vertex_indices: list[np.ndarray] = []
    source_landmarks: list[np.ndarray] = []
    rest_landmarks: list[np.ndarray] = []
    posed_landmarks: list[np.ndarray] = []
    for role in ("proximal", "distal"):
        tokens = tuple(token.lower() for token in spec[role])
        indices: list[np.ndarray] = []
        meshes: list[str] = []
        for name, (start, stop), tissue in zip(
            asset.source_mesh_names,
            np.asarray(asset.source_vertex_ranges, dtype=np.int64),
            asset.source_tissues,
        ):
            lower = str(name).lower()
            digit_matches = not label.startswith("index_proximal_") or lower.startswith(
                "_2nd_"
            )
            if (
                str(tissue) == "bone"
                and any(token in lower for token in tokens)
                and _side_matches(lower, label)
                and digit_matches
            ):
                indices.append(np.arange(int(start), int(stop), dtype=np.int64))
                meshes.append(str(name))
        if not indices:
            roles[role] = {
                "available": False,
                "mesh_tokens": list(tokens),
                "meshes": [],
            }
            continue
        vertex_indices = np.concatenate(indices)
        role_vertex_indices.append(vertex_indices)
        # Freeze the landmark membership on the pre-articulated registration
        # reference.  Selecting nearest vertices after fitting would let a bad
        # solver redefine its own probes and falsely report zero residual.
        endpoint_tokens = {
            "humerus",
            "radius",
            "ulna",
            "femur",
            "tibia",
            "fibula",
            "metacarpal",
            "phalanx_hand",
            "phalanges_hand",
        }
        if endpoint_tokens.intersection(tokens) and len(vertex_indices) >= 8:
            reference = source_vertices[vertex_indices]
            centered = reference - np.mean(reference, axis=0, keepdims=True)
            _u, _singular, vt = np.linalg.svd(centered, full_matrices=False)
            parameter = centered @ vt[0]
            low, high = np.quantile(parameter, (0.10, 0.90))
            low_indices = vertex_indices[parameter <= low]
            high_indices = vertex_indices[parameter >= high]
            # The two cap memberships are frozen from the registration
            # reference; choose their anatomical role in fitted rest space so
            # mirrored source objects cannot swap proximal and distal.
            low_center = np.mean(rest_vertices[low_indices], axis=0)
            high_center = np.mean(rest_vertices[high_indices], axis=0)
            selected = (
                low_indices
                if np.linalg.norm(low_center - target_rest)
                <= np.linalg.norm(high_center - target_rest)
                else high_indices
            )
            sample_count = len(selected)
        else:
            distances = np.linalg.norm(
                source_vertices[vertex_indices] - target_rest,
                axis=1,
            )
            sample_count = min(
                64,
                max(1, int(np.ceil(0.01 * len(vertex_indices)))),
            )
            selected = vertex_indices[
                np.argpartition(distances, sample_count - 1)[:sample_count]
            ]
        rest_landmark = np.mean(rest_vertices[selected], axis=0)
        posed_landmark = np.mean(posed_vertices[selected], axis=0)
        source_landmark = np.mean(source_vertices[selected], axis=0)
        source_landmarks.append(source_landmark)
        rest_landmarks.append(rest_landmark)
        posed_landmarks.append(posed_landmark)
        roles[role] = {
            "available": True,
            "meshes": meshes,
            "sample_vertex_count": int(sample_count),
            "rest_joint_error_m": float(np.linalg.norm(rest_landmark - target_rest)),
            "posed_joint_error_m": float(np.linalg.norm(posed_landmark - target_pose)),
            "landmark_source": source_landmark.tolist(),
            "landmark_rest": rest_landmark.tolist(),
            "landmark_posed": posed_landmark.tolist(),
        }

    if len(rest_landmarks) != 2:
        return {
            "available": False,
            "reason": "proximal or distal anatomy geometry is unavailable",
            "roles": roles,
            "pass": False,
        }
    authored_gap = float(
        np.linalg.norm(source_landmarks[0] - source_landmarks[1])
    )
    rest_gap = float(np.linalg.norm(rest_landmarks[0] - rest_landmarks[1]))
    posed_gap = float(np.linalg.norm(posed_landmarks[0] - posed_landmarks[1]))
    shared_pose = 0.5 * (posed_landmarks[0] + posed_landmarks[1])
    shared_error = float(np.linalg.norm(shared_pose - target_pose))
    fitting_gap_change = abs(rest_gap - authored_gap)
    pose_gap_change = abs(posed_gap - rest_gap)
    from scipy.spatial import cKDTree

    local_roles: list[np.ndarray] = []
    for vertex_indices in role_vertex_indices:
        target_distance = np.linalg.norm(
            rest_vertices[vertex_indices] - target_rest,
            axis=1,
        )
        local_roles.append(
            vertex_indices[
                target_distance <= np.quantile(target_distance, 0.25)
            ]
        )
    nearest_distance, nearest_index = cKDTree(
        rest_vertices[local_roles[1]]
    ).query(rest_vertices[local_roles[0]], k=1)
    proximal_local = int(np.argmin(nearest_distance))
    proximal_index = int(local_roles[0][proximal_local])
    distal_index = int(local_roles[1][int(nearest_index[proximal_local])])
    surface_gap = float(nearest_distance[proximal_local])
    posed_surface_gap = float(
        np.linalg.norm(
            posed_vertices[proximal_index] - posed_vertices[distal_index]
        )
    )
    surface_gap_change = abs(posed_surface_gap - surface_gap)
    surface_gap_limit = _joint_surface_gap_limit(label)
    passed = bool(
        surface_gap <= surface_gap_limit
        and surface_gap_change <= GAP_CHANGE_LIMIT_M
    )
    return {
        "available": True,
        "source": "anatomy_mesh_vertices",
        "roles": roles,
        "source_shared_anchor_error_m": shared_error,
        "authored_geometry_gap_m": authored_gap,
        "source_rest_gap_m": rest_gap,
        "posed_gap_m": posed_gap,
        "fitting_gap_change_m": fitting_gap_change,
        "gap_change_m": pose_gap_change,
        "surface_gap_m": surface_gap,
        "surface_gap_limit_m": surface_gap_limit,
        "posed_surface_gap_m": posed_surface_gap,
        "surface_gap_change_m": surface_gap_change,
        "surface_landmarks": {
            "proximal_vertex": proximal_index,
            "distal_vertex": distal_index,
            "proximal_rest": rest_vertices[proximal_index].tolist(),
            "distal_rest": rest_vertices[distal_index].tolist(),
        },
        "pass": passed,
    }


def _head_orientation_diagnostic(
    asset: AnatomyRiggedAsset,
    *,
    source_transforms: np.ndarray,
    pose_global: np.ndarray,
    rest_global: np.ndarray,
) -> dict[str, Any]:
    source_names = list(asset.source_bone_names or [])
    bi = _first_bone_index(source_names, ("Head_Bone",))
    joint_index = {name: idx for idx, name in enumerate(asset.joint_names)}
    if bi is None or "head" not in joint_index:
        return {"available": False, "orientation_error_deg": None, "pass": False}
    source_rest = np.asarray(asset.source_rest_global[bi], dtype=np.float64)
    source_posed = np.asarray(source_transforms[bi], dtype=np.float64) @ source_rest
    source_motion = source_posed[:3, :3] @ source_rest[:3, :3].T
    hi = joint_index["head"]
    smplx_motion = pose_global[hi, :3, :3] @ rest_global[hi, :3, :3].T
    relative = source_motion.T @ smplx_motion
    trace_cos = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    error = float(np.degrees(np.arccos(trace_cos)))
    return {
        "available": True,
        "source_bone": source_names[bi],
        "comparison": "runtime_motion_vs_smplx_global_motion",
        "orientation_error_deg": error,
        "pass": bool(error <= HEAD_ORIENTATION_LIMIT_DEG),
    }


def _endpoint_error(expected: np.ndarray, posed: np.ndarray) -> dict[str, Any]:
    """Measure deformation relative to the expected posed rigid component.

    Comparing a posed mesh directly with its rest coordinates incorrectly counts
    the subject's global motion as an endpoint error.  Both inputs here are in
    the same posed/world frame: ``expected`` is the mesh transformed by its
    dominant Blender bone and ``posed`` is the full sparse Blender LBS result.
    """
    expected_axis = _bone_axis(expected)
    t = expected @ expected_axis
    expected_span = float(t.max() - t.min())
    if expected_span < 1.0e-6:
        return {
            "available": False,
            "endpoint_error_m": None,
            "axis_error_deg": None,
            "length_error_m": None,
        }
    end_indices = (int(np.argmin(t)), int(np.argmax(t)))
    endpoint = max(float(np.linalg.norm(posed[i] - expected[i])) for i in end_indices)
    posed_axis = _bone_axis(posed)
    posed_span = float(np.ptp(posed @ posed_axis))
    return {
        "available": True,
        "endpoint_error_m": float(endpoint),
        "axis_error_deg": min(
            _angle_deg(expected_axis, posed_axis), _angle_deg(-expected_axis, posed_axis)
        ),
        "length_error_m": abs(posed_span - expected_span),
    }


def _dominant_source_bone(asset: AnatomyRiggedAsset, sl: slice) -> int:
    indices = np.asarray(asset.driver_indices, dtype=np.int64)[sl]
    weights = np.asarray(asset.driver_weights, dtype=np.float64)[sl]
    mass = np.zeros(len(asset.source_bone_names or []), dtype=np.float64)
    np.add.at(mass, indices.reshape(-1), weights.reshape(-1))
    return int(np.argmax(mass))


def classify_ligament_meshes(asset: AnatomyRiggedAsset, mesh_diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    rigid_meshes = set((asset.metadata or {}).get("rigid_meshes") or [])
    entries: list[dict[str, Any]] = []
    for item in mesh_diagnostics.get("meshes", []):
        name = str(item.get("mesh", ""))
        if "extent_aspect_ratio" not in item:
            entries.append({"mesh": name, "flags": ["missing_extent_aspect_ratio"]})
            continue
        ratio = float(item["extent_aspect_ratio"])
        driver = str(item.get("driver_bone", ""))
        tissue = str(item.get("tissue", ""))
        flags: list[str] = []
        if ratio >= 8.0:
            if name in rigid_meshes and tissue != "bone":
                flags.append("mis_rigid_collapse")
            if tissue == "bone":
                flags.append("high_aspect_bone_review")
        if "Spine_C" in driver and ratio >= 8.0:
            flags.append("single_spine_driver")
        if flags:
            entries.append({"mesh": name, "flags": flags, "driver_bone": driver, "extent_aspect_ratio": ratio})
    return entries


def write_bone_segment_diagnostics(
    asset: AnatomyRiggedAsset,
    *,
    pose_axis_angle: np.ndarray,
    transl: np.ndarray | None,
    output_path: Path | str,
    mesh_diagnostics: dict[str, Any] | None = None,
    fitted_hip_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    posed = skin_vertices(asset, pose_axis_angle, transl=transl)
    source_transforms = source_bone_skinning_transforms(asset, pose_axis_angle)
    translation = np.zeros(3, dtype=np.float64) if transl is None else np.asarray(transl, dtype=np.float64)
    pose_global = joint_global_transforms(
        pose_axis_angle=pose_axis_angle,
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    ).astype(np.float64)
    rest_global = joint_global_transforms(
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    ).astype(np.float64)
    posed_smplx_joints = pose_global[:, :3, 3]

    # Keep the former result as an explicitly named submetric.  It can expose
    # LBS bending, but can no longer make the whole report pass on its own.
    rigidity_segments: dict[str, Any] = {}
    rigidity_failures: list[str] = []
    for label, mesh_names in SEGMENT_MESHES.items():
        items: list[dict[str, Any]] = []
        for name in mesh_names:
            sl = _mesh_slice(asset, name)
            if sl is None:
                continue
            rest = np.asarray(asset.vertices_rest, dtype=np.float64)[sl]
            dominant = _dominant_source_bone(asset, sl)
            transform = np.asarray(source_transforms[dominant], dtype=np.float64)
            expected = rest @ transform[:3, :3].T + transform[:3, 3] + translation
            err = _endpoint_error(expected, posed[sl])
            err["mesh"] = name
            err["dominant_source_bone"] = str(asset.source_bone_names[dominant])
            err["pass"] = bool(
                err["available"]
                and err["endpoint_error_m"] <= ENDPOINT_LIMIT_M
                and err["axis_error_deg"] <= AXIS_LIMIT_DEG
            )
            if not err["pass"]:
                rigidity_failures.append(f"rigidity/{label}/{name}")
            items.append(err)
        if not items:
            rigidity_failures.append(f"rigidity/{label}/unavailable")
        rigidity_segments[label] = items

    rest_heads, rest_tails, endpoint_fallback = _bone_endpoints(asset)
    joints: dict[str, Any] = {}
    joint_failures: list[str] = []
    for label, spec in JOINT_CHAINS.items():
        controller = _joint_chain_diagnostic(
            asset,
            spec=spec,
            source_transforms=np.asarray(source_transforms, dtype=np.float64),
            rest_heads=rest_heads,
            rest_tails=rest_tails,
            posed_smplx_joints=posed_smplx_joints,
            translation=translation,
            endpoint_fallback=endpoint_fallback,
        )
        geometry = _geometry_landmark_diagnostic(
            asset,
            label=label,
            joint_name=str(spec["joint"]),
            posed_vertices=np.asarray(posed, dtype=np.float64),
            posed_smplx_joints=posed_smplx_joints,
            translation=translation,
        )
        if label in {"hip_left", "hip_right"}:
            metadata = asset.metadata or {}
            fit_metadata = metadata.get("articulated_rest_fit")
            if fit_metadata is None and isinstance(
                metadata.get("shape_report"), dict
            ):
                fit_metadata = metadata["shape_report"].get(
                    "articulated_rest_fit"
                )
            if fit_metadata is None:
                fit_metadata = metadata.get("articulated_source_report")
            if isinstance(fit_metadata, list):
                latest_fit = fit_metadata[-1] if fit_metadata else {}
            elif isinstance(fit_metadata, dict):
                latest_fit = fit_metadata
            else:
                latest_fit = {}
            side = "left" if label.endswith("_left") else "right"
            hip_geometry = (
                (fitted_hip_geometry or {}).get(side)
                if isinstance(fitted_hip_geometry, dict)
                else None
            )
            if hip_geometry is None:
                hip_geometry = (
                    (latest_fit.get("hip_geometry") or {}).get(side)
                    if isinstance(latest_fit, dict)
                    else None
                )
            if (
                isinstance(hip_geometry, dict)
                and hip_geometry.get("femoral_head_to_acetabulum_m")
                is not None
            ):
                center_error = float(
                    hip_geometry["femoral_head_to_acetabulum_m"]
                )
                geometry["femoral_head_to_acetabulum_m"] = center_error
                geometry["pass"] = bool(
                    center_error <= ENDPOINT_LIMIT_M
                    and geometry.get("surface_gap_change_m", float("inf"))
                    <= GAP_CHANGE_LIMIT_M
                )
        result = {
            **controller,
            "controller_probes": controller,
            "geometry_landmarks": geometry,
            # SMPL-X controllers are kinematic probes, not medical joint
            # centers.  Their residual remains reported but cannot override a
            # directly measured proximal/distal geometry gap.
            "pass": bool(geometry["pass"]),
        }
        joints[label] = result
        if not result["pass"]:
            joint_failures.append(f"joint/{label}")

    head = _head_orientation_diagnostic(
        asset,
        source_transforms=np.asarray(source_transforms, dtype=np.float64),
        pose_global=pose_global,
        rest_global=rest_global,
    )
    head_failures = [] if head["pass"] else ["head/orientation"]

    ligaments = classify_ligament_meshes(asset, mesh_diagnostics or {})
    failures = joint_failures + head_failures + rigidity_failures
    report = {
        "endpoint_limit_m": ENDPOINT_LIMIT_M,
        "axis_limit_deg": AXIS_LIMIT_DEG,
        "gap_change_limit_m": GAP_CHANGE_LIMIT_M,
        "head_orientation_limit_deg": HEAD_ORIENTATION_LIMIT_DEG,
        "joints": joints,
        "head_orientation": head,
        "rigidity_segments": rigidity_segments,
        # Compatibility alias for older report readers.  Its values are no
        # longer sufficient to determine the top-level pass state.
        "segments": rigidity_segments,
        "ligament_flags": ligaments,
        "controller_probe_fallback_used": bool(endpoint_fallback),
        "passed": len(failures) == 0,
        "failures": failures,
        "pass_requires": [
            "geometry_landmarks",
            "left_right_hip_landmarks",
            "joint_gap_change",
            "joint_axes",
            "head_orientation",
            "rigidity",
        ],
    }
    out = Path(output_path)
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    return report
