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

ENDPOINT_LIMIT_M = 0.005
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
    head = getattr(asset, "source_bone_head", None)
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
        anchor_error = float("inf")
        rest_gap = posed_gap = gap_change = float("inf")

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
    axis_error = max((float(item["axis_error_deg"]) for item in axes if item.get("available")), default=float("inf"))

    connected = False
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    if len(selected_indices) == 2:
        proximal, distal = selected_indices
        connected = bool(parents[distal] == proximal and rest_gap <= 1.0e-5)

    passed = bool(
        np.isfinite(anchor_error)
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
        return {"available": False, "orientation_error_deg": float("inf"), "pass": False}
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


def _endpoint_error(expected: np.ndarray, posed: np.ndarray) -> dict[str, float]:
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
        return {"endpoint_error_m": 0.0, "axis_error_deg": 0.0, "length_error_m": 0.0}
    end_indices = (int(np.argmin(t)), int(np.argmax(t)))
    endpoint = max(float(np.linalg.norm(posed[i] - expected[i])) for i in end_indices)
    posed_axis = _bone_axis(posed)
    posed_span = float(np.ptp(posed @ posed_axis))
    return {
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
        ratio = float(item.get("extent_aspect_ratio", 0.0))
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
                err["endpoint_error_m"] <= ENDPOINT_LIMIT_M and err["axis_error_deg"] <= AXIS_LIMIT_DEG
            )
            if not err["pass"]:
                rigidity_failures.append(f"rigidity/{label}/{name}")
            items.append(err)
        rigidity_segments[label] = items

    rest_heads, rest_tails, endpoint_fallback = _bone_endpoints(asset)
    joints: dict[str, Any] = {}
    joint_failures: list[str] = []
    for label, spec in JOINT_CHAINS.items():
        result = _joint_chain_diagnostic(
            asset,
            spec=spec,
            source_transforms=np.asarray(source_transforms, dtype=np.float64),
            rest_heads=rest_heads,
            rest_tails=rest_tails,
            posed_smplx_joints=posed_smplx_joints,
            translation=translation,
            endpoint_fallback=endpoint_fallback,
        )
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
        "passed": len(failures) == 0,
        "failures": failures,
        "pass_requires": ["joint_anchors", "joint_gap_change", "joint_axes", "head_orientation", "rigidity"],
    }
    out = Path(output_path)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report
