"""Retarget a rigged anatomy Blender asset to a SMPL-X canonical rest bundle.

This script runs inside Blender and intentionally avoids project imports.

Strategy (per the asset README: only FK joints named *Rot*/*Rotate*/*Twist* may be
rotated):
1. Pose the rig's FK rotate bones so each limb segment direction matches the SMPL
   canonical T-pose (A-pose -> T-pose in pose space, deformed by the FULL original
   bone weights -> smooth, no tearing).
2. Skin all selected meshes with the original armature weights (numpy LBS replica
   of Blender's armature modifier).
3. Map the posed result into the canonical frame with one global similarity
   Procrustes plus a translation-only per-joint refinement (continuous blend).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix


def _argv_after_separator() -> list[str]:
    argv = list(sys.argv)
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return []


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical-dir", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--output-npz", type=Path, required=True)
    p.add_argument("--output-glb", type=Path, required=True)
    p.add_argument("--report-json", type=Path, required=True)
    return p.parse_args(_argv_after_separator())


def _load_mapping(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        return dict(payload or {})
    except Exception as exc:
        raise RuntimeError(
            f"Cannot parse mapping file {path}. Use JSON or install PyYAML in Blender Python."
        ) from exc


def _global_bind_matrices(rest_joints: np.ndarray, parents: np.ndarray) -> np.ndarray:
    joints = np.asarray(rest_joints, dtype=np.float32).reshape(-1, 3)
    pa = np.asarray(parents, dtype=np.int32).reshape(-1)
    out = np.tile(np.eye(4, dtype=np.float32), (joints.shape[0], 1, 1))
    for idx in range(joints.shape[0]):
        out[idx, :3, 3] = joints[idx]
    return out


def _load_canonical(canonical_dir: Path, *, rest_space: str = "neutral") -> dict[str, Any]:
    weights_path = canonical_dir / "smpl_canonical_weights.npz"
    skeleton_path = canonical_dir / "smpl_canonical_skeleton.json"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if not skeleton_path.is_file():
        raise FileNotFoundError(skeleton_path)
    weights = np.load(weights_path, allow_pickle=True)
    skeleton = json.loads(skeleton_path.read_text(encoding="utf-8"))
    joint_names = [str(v) for v in weights["joint_names"].reshape(-1).tolist()]
    parents = np.asarray(weights["parents"], dtype=np.int32).reshape(-1)
    if str(rest_space).lower() == "neutral":
        rest_joints = np.asarray(skeleton["rest_joints_neutral"], dtype=np.float32).reshape(-1, 3)
        inverse_bind = np.linalg.inv(_global_bind_matrices(rest_joints, parents)).astype(np.float32)
    else:
        rest_joints = np.asarray(weights["rest_joints"], dtype=np.float32).reshape(-1, 3)
        inverse_bind = np.asarray(weights["inverse_bind"], dtype=np.float32).reshape(-1, 4, 4)
    return {
        "joint_names": joint_names,
        "parents": parents,
        "rest_joints": rest_joints,
        "inverse_bind": inverse_bind,
        "skeleton": skeleton,
    }


def _collections_for_object(obj: bpy.types.Object) -> list[str]:
    out: list[str] = []
    for collection in bpy.data.collections:
        try:
            if obj.name in collection.objects:
                out.append(str(collection.name))
        except Exception:
            pass
    return out


def _selected_meshes(config: dict[str, Any]) -> list[bpy.types.Object]:
    include_collections = set(str(v) for v in config.get("include_collections", []) or [])
    include_meshes = set(str(v) for v in config.get("include_meshes", []) or [])
    exclude_meshes = set(str(v) for v in config.get("exclude_meshes", []) or [])
    out: list[bpy.types.Object] = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.name in exclude_meshes:
            continue
        collections = set(_collections_for_object(obj))
        if include_meshes and obj.name in include_meshes:
            out.append(obj)
            continue
        if include_collections and collections.intersection(include_collections):
            out.append(obj)
    if not out:
        raise RuntimeError("No anatomy meshes selected by mapping config")
    return sorted(out, key=lambda obj: obj.name)


def _armature() -> bpy.types.Object:
    arm = next((obj for obj in bpy.data.objects if obj.type == "ARMATURE"), None)
    if arm is None:
        raise RuntimeError("No armature found in the blend file")
    return arm


def _is_connective_tissue(name: str) -> bool:
    """Classify deformable skeletal connective tissue by source semantics."""
    normalized = str(name).lower()
    return any(
        token in normalized
        for token in ("ligament", "cartilage", "tendon", "fascia", "aponeuros")
    )


def _bone_parents(arm: bpy.types.Object) -> dict[str, str | None]:
    return {str(b.name): (str(b.parent.name) if b.parent else None) for b in arm.data.bones}


def _build_bone_to_joint(config: dict[str, Any], joint_names: list[str]) -> tuple[dict[str, int], dict[str, str]]:
    joint_index = {name: idx for idx, name in enumerate(joint_names)}
    mapping = config.get("anatomy_to_smplx", {}) or {}
    out: dict[str, int] = {}
    labels: dict[str, str] = {}
    for bone_name, joint_name in mapping.items():
        b = str(bone_name)
        j = str(joint_name)
        if j not in joint_index:
            continue
        out[b] = int(joint_index[j])
        labels[b] = j
    return out, labels


def _resolve_group_joint(
    group_name: str,
    *,
    direct: dict[str, int],
    parents_by_bone: dict[str, str | None],
    fallback: int,
) -> tuple[int, bool]:
    name = str(group_name)
    visited: set[str] = set()
    cur: str | None = name
    while cur and cur not in visited:
        visited.add(cur)
        if cur in direct:
            return int(direct[cur]), (cur != name)
        cur = parents_by_bone.get(cur)
    return int(fallback), True


def _triangulated_faces(poly_vertices: list[int]) -> list[tuple[int, int, int]]:
    if len(poly_vertices) < 3:
        return []
    if len(poly_vertices) == 3:
        return [(int(poly_vertices[0]), int(poly_vertices[1]), int(poly_vertices[2]))]
    root = int(poly_vertices[0])
    return [(root, int(poly_vertices[i]), int(poly_vertices[i + 1])) for i in range(1, len(poly_vertices) - 1)]


def _limit_weights(row: np.ndarray, max_influences: int) -> np.ndarray:
    if max_influences <= 0 or np.count_nonzero(row) <= max_influences:
        total = float(row.sum())
        return row / total if total > 0 else row
    keep = np.argpartition(row, -max_influences)[-max_influences:]
    out = np.zeros_like(row)
    out[keep] = row[keep]
    total = float(out.sum())
    if total > 0:
        out /= total
    return out


def _propagate_empty_vertex_data(
    *,
    mesh: bpy.types.Mesh,
    raw: np.ndarray,
    posed: np.ndarray,
    weights: np.ndarray,
    empty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill unweighted vertices from their own connected component.

    Propagating displacement (rather than absolute position) preserves the local
    mesh shape. A disconnected component with no weighted seed is ambiguous and
    therefore fails the bake instead of silently attaching to the pelvis.
    """
    pending = np.asarray(empty, dtype=bool).copy()
    if not np.any(pending):
        return posed, weights
    neighbors: list[list[int]] = [[] for _ in range(len(mesh.vertices))]
    for edge in mesh.edges:
        a, b = (int(edge.vertices[0]), int(edge.vertices[1]))
        neighbors[a].append(b)
        neighbors[b].append(a)
    displacement = np.asarray(posed - raw, dtype=np.float32)
    while np.any(pending):
        filled: list[tuple[int, list[int]]] = []
        for vi in np.flatnonzero(pending).tolist():
            known = [vj for vj in neighbors[vi] if not pending[vj]]
            if known:
                filled.append((int(vi), known))
        if not filled:
            raise RuntimeError(
                f"mesh {mesh.name!r} contains {int(np.count_nonzero(pending))} unweighted "
                "vertices in a component without a weighted seed"
            )
        # Fill one graph-distance shell at a time so a result does not depend on
        # Blender's vertex iteration order.
        for vi, known in filled:
            displacement[vi] = np.mean(displacement[known], axis=0)
            weights[vi] = np.mean(weights[known], axis=0)
        for vi, _known in filled:
            pending[vi] = False
    posed = raw + displacement
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-8)
    return posed.astype(np.float32), weights.astype(np.float32)


def _sparse_source_weights(
    mesh: bpy.types.Mesh,
    *,
    group_names: dict[int, str],
    bone_index: dict[str, int],
    max_influences: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract normalized original Blender weights without collapsing bones."""
    k = max(1, int(max_influences))
    indices = np.zeros((len(mesh.vertices), k), dtype=np.int16)
    weights = np.zeros((len(mesh.vertices), k), dtype=np.float32)
    empty: list[int] = []
    for vi, vertex in enumerate(mesh.vertices):
        merged: dict[int, float] = {}
        for elem in vertex.groups:
            name = group_names.get(int(elem.group), "")
            if name not in bone_index or float(elem.weight) <= 0.0:
                continue
            bi = int(bone_index[name])
            merged[bi] = merged.get(bi, 0.0) + float(elem.weight)
        if not merged:
            empty.append(vi)
            continue
        selected = sorted(merged.items(), key=lambda item: item[1], reverse=True)[:k]
        total = max(sum(value for _idx, value in selected), 1.0e-12)
        for slot, (bi, value) in enumerate(selected):
            indices[vi, slot] = bi
            weights[vi, slot] = float(value / total)
    if empty:
        pending = set(empty)
        neighbors: list[list[int]] = [[] for _ in mesh.vertices]
        for edge in mesh.edges:
            a, b = int(edge.vertices[0]), int(edge.vertices[1])
            neighbors[a].append(b)
            neighbors[b].append(a)
        while pending:
            shell: list[tuple[int, int]] = []
            for vi in sorted(pending):
                known = next((vj for vj in neighbors[vi] if vj not in pending), None)
                if known is not None:
                    shell.append((vi, int(known)))
            if not shell:
                raise RuntimeError(
                    f"mesh {mesh.name!r} has an unweighted component with {len(pending)} vertices"
                )
            for vi, source in shell:
                indices[vi] = indices[source]
                weights[vi] = weights[source]
                pending.remove(vi)
    return indices, weights, len(empty)


def _rotation_between(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a = np.asarray(src, dtype=np.float64).reshape(3)
    b = np.asarray(dst, dtype=np.float64).reshape(3)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1.0e-8 or nb < 1.0e-8:
        return np.eye(3, dtype=np.float32)
    a /= na
    b /= nb
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s2 = float(np.dot(v, v))
    if s2 < 1.0e-12:
        if c > 0.0:
            return np.eye(3, dtype=np.float32)
        axis = np.cross(a, np.asarray([1.0, 0.0, 0.0]))
        if float(np.dot(axis, axis)) < 1.0e-8:
            axis = np.cross(a, np.asarray([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return (2.0 * np.outer(axis, axis) - np.eye(3)).astype(np.float32)
    vx = np.asarray([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64)
    return (np.eye(3) + vx + vx @ vx * ((1.0 - c) / s2)).astype(np.float32)


def _procrustes_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (scale, R, t) mapping src points onto dst points."""
    a = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    q = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    a_mean = a.mean(axis=0)
    q_mean = q.mean(axis=0)
    a_c = a - a_mean
    q_c = q - q_mean
    H = a_c.T @ q_c
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if float(np.linalg.det(Vt.T @ U.T)) < 0.0:
        D[2, 2] = -1.0
    R = Vt.T @ D @ U.T
    denom = float((a_c**2).sum())
    scale = float((S * np.diag(D)).sum() / max(denom, 1.0e-12))
    t = q_mean - scale * (R @ a_mean)
    return scale, R.astype(np.float32), t.astype(np.float32)


# Limb chains posed in FK order (parent before child). Directions are defined by
# primary anchor bone heads: joint -> child joint.
_POSE_CHAIN: list[tuple[str, str]] = [
    ("left_collar", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_index1"),
    ("left_index1", "left_index2"),
    ("left_index2", "left_index3"),
    ("right_collar", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_index1"),
    ("right_index1", "right_index2"),
    ("right_index2", "right_index3"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
]


def _primary_anchor_bones(direct: dict[str, int]) -> dict[int, str]:
    """Pick one anchor bone per mapped SMPL joint.

    Preference: first mapped bone whose name contains 'rot' (the FK rotate joints per
    the asset README), then 'twist', then the first mapped bone. Mapping insertion
    order is preserved.
    """
    by_joint: dict[int, list[str]] = {}
    for bone_name, joint in direct.items():
        by_joint.setdefault(int(joint), []).append(str(bone_name))
    out: dict[int, str] = {}
    for joint, names in by_joint.items():
        chosen = next((n for n in names if "rot" in n.lower()), None)
        if chosen is None:
            chosen = next((n for n in names if "twist" in n.lower()), None)
        out[joint] = chosen if chosen is not None else names[0]
    return out


def _pose_bone_head(arm: bpy.types.Object, bone_name: str) -> np.ndarray:
    pb = arm.pose.bones.get(bone_name)
    if pb is None:
        raise KeyError(f"Pose bone not found: {bone_name}")
    return np.asarray([pb.head.x, pb.head.y, pb.head.z], dtype=np.float32)


def _pose_armature_to_canonical(
    arm: bpy.types.Object,
    canonical: dict[str, Any],
    *,
    primary: dict[int, str],
    joint_names: list[str],
) -> dict[str, Any]:
    """Rotate FK anchor bones so limb segment directions match the canonical T-pose.

    Works entirely in armature object space; the residual global scale/rotation is
    absorbed later by the Procrustes alignment.
    """
    rest_joints = np.asarray(canonical["rest_joints"], dtype=np.float32).reshape(-1, 3)
    joint_index = {name: idx for idx, name in enumerate(joint_names)}

    mapped = sorted(primary)
    rest_anchors = np.stack([_pose_bone_head(arm, primary[j]) for j in mapped])
    _, Rg0, _ = _procrustes_similarity(rest_anchors, rest_joints[mapped])

    applied: dict[str, float] = {}
    for joint_name, child_name in _POSE_CHAIN:
        j = joint_index.get(joint_name)
        c = joint_index.get(child_name)
        if j is None or c is None or j not in primary or c not in primary:
            continue
        pb = arm.pose.bones.get(primary[j])
        if pb is None:
            continue
        cur_dir = _pose_bone_head(arm, primary[c]) - _pose_bone_head(arm, primary[j])
        target_dir = Rg0.T @ (rest_joints[c] - rest_joints[j])
        R = _rotation_between(cur_dir, target_dir)
        angle = math.degrees(
            math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) * 0.5)))
        )
        if angle < 0.05:
            applied[joint_name] = 0.0
            continue
        head = pb.head.copy()
        rot4 = Matrix.Identity(4)
        for r in range(3):
            for col in range(3):
                rot4[r][col] = float(R[r, col])
        pivot = Matrix.Translation(head) @ rot4 @ Matrix.Translation(-head)
        pb.matrix = pivot @ pb.matrix
        bpy.context.view_layer.update()
        applied[joint_name] = round(angle, 2)

    return {"pose_rotations_deg": applied}


def _bone_deform_matrices(arm: bpy.types.Object) -> dict[str, np.ndarray]:
    """Per-bone deform matrix in armature object space: D = pose_matrix @ rest^-1."""
    out: dict[str, np.ndarray] = {}
    for bone in arm.data.bones:
        pb = arm.pose.bones.get(bone.name)
        if pb is None:
            continue
        pose_m = np.asarray(pb.matrix, dtype=np.float64).reshape(4, 4)
        rest_m = np.asarray(bone.matrix_local, dtype=np.float64).reshape(4, 4)
        out[str(bone.name)] = (pose_m @ np.linalg.inv(rest_m)).astype(np.float32)
    return out


def _armature_local_bbox(arm: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    heads = np.stack(
        [np.asarray([b.head_local.x, b.head_local.y, b.head_local.z], dtype=np.float32) for b in arm.data.bones]
    )
    return heads.min(axis=0), heads.max(axis=0)


def _mesh_to_armature_transform(
    obj: bpy.types.Object,
    arm_inv: np.ndarray,
    arm_bbox: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, str]:
    """Pick the transform that places this mesh into the armature-local frame.

    Some assets keep mesh data directly in the armature-local frame with an identity
    matrix_world while the armature object itself carries a unit/axis conversion. The
    standard Blender relation (arm_inv @ obj.matrix_world) blows such meshes up by the
    inverse armature scale. Choose per mesh by comparing the transformed bounding box
    against the armature bone bbox.
    """
    n = len(obj.data.vertices)
    raw = np.empty(n * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", raw)
    pts = raw.reshape(n, 3)[:: max(1, n // 512)].astype(np.float64)
    world = np.asarray(obj.matrix_world, dtype=np.float64).reshape(4, 4)
    lo, hi = arm_bbox
    arm_span = float(np.max(hi - lo))
    arm_center = (lo + hi) * 0.5

    def _score(M: np.ndarray) -> float:
        p = pts @ M[:3, :3].T + M[:3, 3]
        span = float(np.max(p.max(axis=0) - p.min(axis=0)))
        center = (p.max(axis=0) + p.min(axis=0)) * 0.5
        return abs(math.log(max(span, 1.0e-6) / arm_span)) + float(
            np.linalg.norm(center - arm_center)
        ) / max(arm_span, 1.0e-6)

    standard = arm_inv @ world
    identity = np.eye(4, dtype=np.float64)
    if float(obj.matrix_world[0][0]) < 0.0:
        return standard, "standard_mirrored"
    if _score(identity) <= _score(standard):
        return identity, "identity"
    return standard, "standard"


def _merge_and_skin_meshes(
    meshes: list[bpy.types.Object],
    arm: bpy.types.Object,
    *,
    config: dict[str, Any],
    joint_names: list[str],
    direct_bone_to_joint: dict[str, int],
    parents_by_bone: dict[str, str | None],
    deform: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Merge selected meshes; skin them with the FULL original bone weights.

    Replicates Blender's armature modifier in armature object space after explicitly
    normalizing every non-empty source weight row. Leaving ``1-sum(weights)`` at
    the bind position creates metre-scale edges when the weighted part moves.
    Collapsed SMPL-X driver weights are exported alongside for runtime LBS.
    """
    max_influences = int(config.get("max_influences", 4))
    fallback_joint_name = str(config.get("fallback_joint", "pelvis"))
    fallback_joint = joint_names.index(fallback_joint_name) if fallback_joint_name in joint_names else 0
    joint_count = len(joint_names)
    joint_index = {name: idx for idx, name in enumerate(joint_names)}
    source_bone_names = [str(b.name) for b in arm.data.bones]
    source_bone_index = {name: idx for idx, name in enumerate(source_bone_names)}
    rigid_collections = set(str(v) for v in config.get("rigid_collections", []) or [])
    rigid_mesh_to_smplx = {
        str(k): str(v)
        for k, v in (config.get("rigid_mesh_to_smplx", {}) or {}).items()
        if str(v) in joint_index
    }
    preserve_source_weights = set(str(v) for v in (config.get("preserve_source_weights", []) or []))
    arm_inv = np.asarray(arm.matrix_world.inverted(), dtype=np.float64).reshape(4, 4)
    arm_bbox = _armature_local_bbox(arm)
    frame_modes: dict[str, int] = {}

    all_vertices: list[np.ndarray] = []
    all_raw_vertices: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    all_source_indices: list[np.ndarray] = []
    all_source_weights: list[np.ndarray] = []
    all_rigid_component_ids: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    source_mesh_names: list[str] = []
    source_vertex_ranges: list[tuple[int, int]] = []
    source_tissues: list[str] = []
    fallback_groups: dict[str, int] = {}
    inherited_groups: dict[str, int] = {}
    rigid_meshes: list[str] = []
    empty_source_weight_vertices = 0
    missing_deform_groups: dict[str, int] = {}
    source_pose_edge_ratios: list[np.ndarray] = []
    vertex_offset = 0
    rigid_component_counter = 0

    for obj in meshes:
        source_mesh_names.append(str(obj.name))
        object_collections = set(_collections_for_object(obj))
        connective_tissue = _is_connective_tissue(str(obj.name))
        mesh_lower = str(obj.name).lower()
        if connective_tissue:
            source_tissues.append("connective_tissue")
        elif "heart" in mesh_lower:
            source_tissues.append("heart")
        elif "Skeletal_Sys" in object_collections:
            source_tissues.append("bone")
        elif "Cardiovascular_Sys" in object_collections:
            source_tissues.append("vessel")
        elif "Nervous_Sys" in object_collections:
            source_tissues.append("nerve")
        else:
            source_tissues.append("organ")
        mesh = obj.data
        n = len(mesh.vertices)
        start = vertex_offset

        group_to_joint: dict[int, int] = {}
        group_names: dict[int, str] = {}
        for group in obj.vertex_groups:
            group_names[int(group.index)] = str(group.name)
            joint, inherited = _resolve_group_joint(
                group.name,
                direct=direct_bone_to_joint,
                parents_by_bone=parents_by_bone,
                fallback=fallback_joint,
            )
            group_to_joint[int(group.index)] = int(joint)
            if inherited:
                if int(joint) == int(fallback_joint) and group.name not in direct_bone_to_joint:
                    fallback_groups[group.name] = fallback_groups.get(group.name, 0) + 1
                else:
                    inherited_groups[group.name] = inherited_groups.get(group.name, 0) + 1

        source_indices, source_weights, source_empty_count = _sparse_source_weights(
            mesh,
            group_names=group_names,
            bone_index=source_bone_index,
            max_influences=max_influences,
        )

        to_arm, frame_mode = _mesh_to_armature_transform(obj, arm_inv, arm_bbox)
        frame_modes[frame_mode] = frame_modes.get(frame_mode, 0) + 1
        raw_local = np.empty(n * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", raw_local)
        raw = raw_local.reshape(n, 3).astype(np.float64) @ to_arm[:3, :3].T + to_arm[:3, 3]
        raw = raw.astype(np.float32)

        group_elems: dict[int, tuple[list[int], list[float]]] = {}
        w55 = np.zeros((n, joint_count), dtype=np.float32)
        for vi, vertex in enumerate(mesh.vertices):
            for elem in vertex.groups:
                gi = int(elem.group)
                wv = float(elem.weight)
                if wv <= 0.0:
                    continue
                idxs, ws = group_elems.setdefault(gi, ([], []))
                idxs.append(vi)
                ws.append(wv)
                w55[vi, group_to_joint.get(gi, fallback_joint)] += wv
        totals = w55.sum(axis=1)
        empty = totals <= 1.0e-8
        w55[~empty] /= totals[~empty][:, None]
        for vi in range(n):
            w55[vi] = _limit_weights(w55[vi], max_influences=max_influences)

        collections = set(_collections_for_object(obj))
        is_rigid = bool(
            rigid_collections
            and collections.intersection(rigid_collections)
            and not connective_tissue
        )
        preserve_weights = str(obj.name) in preserve_source_weights
        preserve_tokens = (
            "metacarpal",
            "metatarsal",
            "phalanx_hand",
            "phalanges_hand",
            "phalanx_foot",
            "phalanges_foot",
        )
        if is_rigid and any(token in mesh_lower for token in preserve_tokens):
            preserve_weights = True
        if is_rigid and not preserve_weights:
            joint_name = rigid_mesh_to_smplx.get(str(obj.name))
            if joint_name is None:
                joint = int(np.argmax(w55.mean(axis=0)))
            else:
                joint = int(joint_index[joint_name])
            w55[:, :] = 0.0
            w55[:, joint] = 1.0
            source_mass = np.zeros(len(source_bone_names), dtype=np.float64)
            for slot in range(source_indices.shape[1]):
                np.add.at(source_mass, source_indices[:, slot], source_weights[:, slot])
            rigid_source_bone = int(np.argmax(source_mass))
            foot_chain_roots: dict[str, int] = {}
            for side_key, root_name in (("left", "Ankle_Rot_L"), ("right", "Ankle_Rot_R")):
                if root_name in source_bone_index:
                    foot_chain_roots[side_key] = int(source_bone_index[root_name])
            side = None
            if mesh_lower.endswith("_l") or "_l_" in mesh_lower or mesh_lower.endswith("_hand_l"):
                side = "left"
            elif mesh_lower.endswith("_r") or "_r_" in mesh_lower or mesh_lower.endswith("_hand_r"):
                side = "right"
            if side is not None and any(
                token in mesh_lower for token in ("phalanx_foot", "phalanges_foot", "metatarsal")
            ):
                if side in foot_chain_roots:
                    rigid_source_bone = int(foot_chain_roots[side])
            source_indices[:, :] = rigid_source_bone
            source_weights[:, :] = 0.0
            source_weights[:, 0] = 1.0
            rigid_meshes.append(str(obj.name))
        elif is_rigid and preserve_weights:
            rigid_meshes.append(str(obj.name))

        source_totals = np.zeros(n, dtype=np.float32)
        for _gi, (idxs, ws) in group_elems.items():
            source_totals[np.asarray(idxs, dtype=np.int64)] += np.asarray(ws, dtype=np.float32)
        source_empty = source_totals <= 1.0e-8
        empty_source_weight_vertices += int(source_empty_count)

        acc = np.zeros((n, 3), dtype=np.float32)
        applied = np.zeros(n, dtype=np.float32)
        for gi, (idxs, ws) in group_elems.items():
            group_name = group_names.get(gi, "")
            D = deform.get(group_name)
            if D is None:
                missing_deform_groups[group_name] = missing_deform_groups.get(group_name, 0) + len(idxs)
                continue
            idx = np.asarray(idxs, dtype=np.int64)
            w = np.asarray(ws, dtype=np.float32) / np.maximum(source_totals[idx], 1.0e-8)
            acc[idx] += w[:, None] * (raw[idx] @ D[:3, :3].T + D[:3, 3])
            applied[idx] += w
        missing_vertex = (~source_empty) & (applied <= 1.0e-8)
        if np.any(missing_vertex):
            names = sorted(k for k, count in missing_deform_groups.items() if count > 0)
            raise RuntimeError(
                f"{obj.name}: {int(np.count_nonzero(missing_vertex))} weighted vertices have no "
                f"armature deform; missing groups sample={names[:12]}"
            )
        posed = raw.copy()
        valid = applied > 1.0e-8
        posed[valid] = acc[valid] / applied[valid, None]
        if np.any(source_empty):
            posed, w55 = _propagate_empty_vertex_data(
                mesh=mesh,
                raw=raw,
                posed=posed,
                weights=w55,
                empty=source_empty,
            )
        if len(mesh.edges):
            edge_idx = np.asarray(
                [(int(edge.vertices[0]), int(edge.vertices[1])) for edge in mesh.edges],
                dtype=np.int64,
            )
            raw_len = np.linalg.norm(raw[edge_idx[:, 0]] - raw[edge_idx[:, 1]], axis=1)
            posed_len = np.linalg.norm(posed[edge_idx[:, 0]] - posed[edge_idx[:, 1]], axis=1)
            valid_edge = raw_len > 1.0e-8
            if np.any(valid_edge):
                source_pose_edge_ratios.append((posed_len[valid_edge] / raw_len[valid_edge]).astype(np.float32))

        all_vertices.append(posed)
        all_raw_vertices.append(raw)
        all_weights.append(w55)
        all_source_indices.append(source_indices)
        all_source_weights.append(source_weights)
        if is_rigid:
            all_rigid_component_ids.append(np.full(n, rigid_component_counter, dtype=np.int32))
            rigid_component_counter += 1
        else:
            all_rigid_component_ids.append(np.full(n, -1, dtype=np.int32))
        for poly in mesh.polygons:
            indices = [vertex_offset + int(i) for i in poly.vertices]
            faces.extend(_triangulated_faces(indices))
        vertex_offset += n
        source_vertex_ranges.append((start, vertex_offset))

    if fallback_groups and bool(config.get("fail_on_unmapped_groups", True)):
        raise RuntimeError(
            "Unmapped Blender vertex groups cannot silently fall back to pelvis: "
            + ", ".join(sorted(fallback_groups)[:24])
        )

    edge_ratio = np.concatenate(source_pose_edge_ratios) if source_pose_edge_ratios else np.ones(1, dtype=np.float32)
    return (
        np.concatenate(all_vertices, axis=0),
        np.asarray(faces, dtype=np.int32),
        np.concatenate(all_weights, axis=0),
        {
            "source_mesh_names": source_mesh_names,
            "source_vertex_ranges": source_vertex_ranges,
            "source_tissues": source_tissues,
            "fallback_groups": fallback_groups,
            "inherited_groups": inherited_groups,
            "rigid_meshes": rigid_meshes,
            "frame_modes": frame_modes,
            "empty_source_weight_vertices": int(empty_source_weight_vertices),
            "missing_deform_groups": missing_deform_groups,
            "source_pose_edge_ratio_max": float(np.max(edge_ratio)),
            "source_pose_edge_ratio_p999": float(np.quantile(edge_ratio, 0.999)),
            "source_driver_indices": np.concatenate(all_source_indices, axis=0),
            "source_driver_weights": np.concatenate(all_source_weights, axis=0),
            "source_bone_names": source_bone_names,
            "rigid_component_ids": np.concatenate(all_rigid_component_ids, axis=0),
            "raw_vertices": np.concatenate(all_raw_vertices, axis=0),
        },
    )


def _align_rest_to_canonical(
    vertices: np.ndarray,
    weights: np.ndarray,
    canonical: dict[str, Any],
    *,
    arm: bpy.types.Object,
    primary: dict[int, str],
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Posed (T-pose) anatomy -> canonical frame.

    Global similarity Procrustes on the POSED primary anchors, then a
    translation-only per-joint refinement blended through the collapsed weights
    (continuous: no rotation mixing, so no tearing).
    """
    rest_joints = np.asarray(canonical["rest_joints"], dtype=np.float32).reshape(-1, 3)
    parents = np.asarray(canonical["parents"], dtype=np.int32).reshape(-1)
    joint_count = int(rest_joints.shape[0])

    mapped = sorted(primary)
    anchors = np.stack([_pose_bone_head(arm, primary[j]) for j in mapped])
    scale, Rg, tg = _procrustes_similarity(anchors, rest_joints[mapped])
    G = (scale * Rg).astype(np.float32)
    a_glob = anchors @ G.T + tg
    rms = float(np.sqrt(np.mean(np.sum((a_glob - rest_joints[mapped]) ** 2, axis=1))))

    offsets = np.zeros((joint_count, 3), dtype=np.float32)
    is_mapped = np.zeros(joint_count, dtype=bool)
    for k, j in enumerate(mapped):
        offsets[j] = rest_joints[j] - a_glob[k]
        is_mapped[j] = True

    def _nearest_mapped_ancestor(j: int) -> int:
        p = int(parents[j])
        while p >= 0 and not is_mapped[p]:
            p = int(parents[p])
        return p if p >= 0 else (mapped[0] if mapped else 0)

    for j in range(joint_count):
        if not is_mapped[j]:
            offsets[j] = offsets[_nearest_mapped_ancestor(j)]

    anchor_positions = a_glob.astype(np.float32)
    anchor_offsets = np.stack([offsets[j] for j in mapped]).astype(np.float32)
    pair_distance = np.linalg.norm(
        anchor_positions[:, None, :] - anchor_positions[None, :, :], axis=2
    ).astype(np.float64)
    polynomial = np.column_stack((np.ones(len(anchor_positions)), anchor_positions)).astype(np.float64)
    rbf_system = np.block(
        [
            [pair_distance + 5.0e-3 * np.eye(len(anchor_positions)), polynomial],
            [polynomial.T, np.zeros((4, 4), dtype=np.float64)],
        ]
    )
    rbf_rhs = np.vstack((anchor_offsets.astype(np.float64), np.zeros((4, 3), dtype=np.float64)))
    rbf_solution = np.linalg.solve(rbf_system, rbf_rhs)
    rbf_coefficients = rbf_solution[: len(anchor_positions)]
    rbf_affine = rbf_solution[len(anchor_positions) :]
    fitted_anchor_offsets = pair_distance @ rbf_coefficients + polynomial @ rbf_affine
    anchor_residual = fitted_anchor_offsets - anchor_offsets

    def _continuous_offset_field(points: np.ndarray) -> np.ndarray:
        output = np.empty_like(points, dtype=np.float32)
        for start in range(0, len(points), 50000):
            stop = min(len(points), start + 50000)
            query = points[start:stop].astype(np.float64)
            radial = np.linalg.norm(query[:, None, :] - anchor_positions[None, :, :], axis=2)
            affine = np.column_stack((np.ones(len(query)), query))
            output[start:stop] = (radial @ rbf_coefficients + affine @ rbf_affine).astype(np.float32)
        return output

    verts = np.asarray(vertices, dtype=np.float32) @ G.T + tg
    verts = verts + _continuous_offset_field(verts)
    diag = {
        "mode": "fk_pose_global_procrustes",
        "scale": float(scale),
        "initial_anchor_rms_m": rms,
        "anchor_rms_m": float(np.sqrt(np.mean(np.sum(anchor_residual**2, axis=1)))),
        "mapped_joints": int(len(mapped)),
        "max_joint_correction_m": float(np.max(np.linalg.norm(offsets, axis=1))),
        "max_joint_offset_m": float(np.max(np.linalg.norm(anchor_residual, axis=1))),
    }
    return verts.astype(np.float32), diag, {
        "linear": G.astype(np.float32),
        "rotation": Rg.astype(np.float32),
        "translation": tg.astype(np.float32),
        "joint_offsets": offsets.astype(np.float32),
        "anchor_positions": anchor_positions,
        "anchor_offsets": anchor_offsets,
        "rbf_coefficients": rbf_coefficients.astype(np.float32),
        "rbf_affine": rbf_affine.astype(np.float32),
    }


def _sample_alignment_offset(points: np.ndarray, align: dict[str, np.ndarray]) -> np.ndarray:
    anchors = np.asarray(align["anchor_positions"], dtype=np.float64)
    coefficients = np.asarray(align["rbf_coefficients"], dtype=np.float64)
    affine_coefficients = np.asarray(align["rbf_affine"], dtype=np.float64)
    query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    radial = np.linalg.norm(query[:, None, :] - anchors[None, :, :], axis=2)
    polynomial = np.column_stack((np.ones(len(query)), query))
    return radial @ coefficients + polynomial @ affine_coefficients


def _source_rig_canonical(
    arm: bpy.types.Object,
    *,
    direct: dict[str, int],
    parents_by_bone: dict[str, str | None],
    canonical: dict[str, Any],
    align: dict[str, np.ndarray],
    joint_names: list[str],
) -> dict[str, Any]:
    """Export the complete Blender hierarchy in canonical metric coordinates."""
    bones = list(arm.data.bones)
    names = [str(b.name) for b in bones]
    index = {name: idx for idx, name in enumerate(names)}
    source_parents = np.asarray(
        [index[str(b.parent.name)] if b.parent is not None else -1 for b in bones], dtype=np.int16
    )
    if any(parent >= idx for idx, parent in enumerate(source_parents.tolist())):
        raise RuntimeError("Blender source bones are not stored in parent-before-child order")
    joint_index = {name: idx for idx, name in enumerate(joint_names)}
    fallback = joint_index.get("pelvis", 0)
    linear = np.asarray(align["linear"], dtype=np.float64)
    rotation_global = np.asarray(align["rotation"], dtype=np.float64)
    translation = np.asarray(align["translation"], dtype=np.float64)
    offsets = np.asarray(align["joint_offsets"], dtype=np.float64)

    rest_global = np.tile(np.eye(4, dtype=np.float64), (len(bones), 1, 1))
    bone_head = np.zeros((len(bones), 3), dtype=np.float64)
    bone_tail = np.zeros((len(bones), 3), dtype=np.float64)
    bone_roll = np.zeros(len(bones), dtype=np.float64)
    bone_use_connect = np.asarray([bool(b.use_connect) for b in bones], dtype=np.uint8)
    inherit_scale_codes = {
        "FULL": 0,
        "FIX_SHEAR": 1,
        "AVERAGE": 2,
        "NONE": 3,
        "NONE_LEGACY": 4,
        "ALIGNED": 5,
    }
    bone_inherit_scale = np.asarray(
        [inherit_scale_codes[str(b.inherit_scale)] for b in bones], dtype=np.uint8
    )
    joint_a = np.zeros(len(bones), dtype=np.int16)
    joint_b = np.zeros(len(bones), dtype=np.int16)
    blend = np.zeros(len(bones), dtype=np.float32)
    driver_types: list[str] = []

    _SIDE_LEFT = re.compile(r"(?:^|_)L(?:\d+)?$")
    _SIDE_RIGHT = re.compile(r"(?:^|_)R(?:\d+)?$")

    def _bone_side(name: str, lower: str) -> str | None:
        if _SIDE_LEFT.search(name) or lower.endswith("_l"):
            return "left"
        if _SIDE_RIGHT.search(name) or lower.endswith("_r"):
            return "right"
        return None

    def _is_foot_chain_bone(lower: str) -> bool:
        if "ankle_rot" in lower or "arch_rot" in lower:
            return True
        if any(
            token in lower
            for token in (
                "calcaneus",
                "talus",
                "navicular",
                "cuboid",
                "cuneiform",
                "metatarsal",
                "phalanx_foot",
                "phalanges_foot",
            )
        ):
            return True
        if "phalanx" in lower and "foot" in lower:
            return True
        if "metatarsal" in lower:
            return True
        return False

    def _canonical_point(point: np.ndarray) -> np.ndarray:
        point_global = np.asarray(point, dtype=np.float64) @ linear.T + translation
        return point_global + _sample_alignment_offset(point_global.reshape(1, 3), align)[0]

    def _roll_about_axis(rotation: np.ndarray, head: np.ndarray, tail: np.ndarray) -> float:
        """Encode bind roll relative to a deterministic transverse reference.

        Blender's data Bone exposes the complete roll through ``matrix_local``
        rather than as a scalar.  The matrix is already retained in the bind
        frame; this scalar is a compact diagnostic/reconstruction aid.
        """
        axis = np.asarray(tail - head, dtype=np.float64)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        candidates = np.eye(3, dtype=np.float64)
        reference = candidates[int(np.argmin(np.abs(candidates @ axis)))]
        reference -= axis * float(np.dot(reference, axis))
        reference /= max(float(np.linalg.norm(reference)), 1.0e-12)
        transverse = np.asarray(rotation[:, 0], dtype=np.float64)
        transverse -= axis * float(np.dot(transverse, axis))
        transverse /= max(float(np.linalg.norm(transverse)), 1.0e-12)
        return float(
            math.atan2(
                float(np.dot(np.cross(reference, transverse), axis)),
                float(np.dot(reference, transverse)),
            )
        )

    for bi, bone in enumerate(bones):
        name = str(bone.name)
        mapped, inherited = _resolve_group_joint(
            name, direct=direct, parents_by_bone=parents_by_bone, fallback=fallback
        )
        mapped = int(mapped)
        joint_a[bi] = mapped
        joint_b[bi] = mapped
        driver_type = "parent_follow" if inherited else "direct_joint"

        lower = name.lower()
        if "scapula" in lower:
            driver_type = "scapula_left" if lower.endswith("_l") else "scapula_right"
        elif "clavicle_rot" in lower:
            side = _bone_side(name, lower)
            if side is not None:
                joint_a[bi] = joint_index["spine3"]
                joint_b[bi] = joint_index[f"{side}_collar"]
                driver_type = f"clavicle_segment_{side}"
        elif "shoulder_rotate" in lower:
            side = _bone_side(name, lower)
            if side is not None:
                joint_a[bi] = joint_index[f"{side}_shoulder"]
                joint_b[bi] = joint_index[f"{side}_elbow"]
                driver_type = f"humerus_segment_{side}"
        elif "knee_rotate" in lower:
            side = _bone_side(name, lower)
            if side is not None:
                joint_a[bi] = joint_index[f"{side}_hip"]
                joint_b[bi] = joint_index[f"{side}_knee"]
                blend[bi] = 0.55
                driver_type = f"knee_chain_{side}"
        elif _is_foot_chain_bone(lower) or "toes_rotate" in lower:
            side = _bone_side(name, lower)
            if side is not None:
                joint_a[bi] = joint_index[f"{side}_ankle"]
                joint_b[bi] = joint_index[f"{side}_foot"]
                blend[bi] = 0.40 if "ankle_rot" in lower else 0.65
                driver_type = f"foot_chain_{side}"
        elif "patella" in lower or "fibula" in lower:
            side = "left" if lower.endswith("_l") else "right"
            joint_a[bi] = joint_index[f"{side}_knee"]
            joint_b[bi] = joint_index[f"{side}_ankle"]
            blend[bi] = 0.45 if "patella" in lower else 0.35
            driver_type = f"knee_chain_{side}"
        elif "femur_rot" in lower or name.startswith("Femur_Rot_"):
            side = "left" if lower.endswith("_l") else "right"
            joint_a[bi] = joint_index[f"{side}_hip"]
            joint_b[bi] = joint_index[f"{side}_knee"]
            blend[bi] = 0.35
            driver_type = f"knee_chain_{side}"
        elif "elbow_rot" in lower:
            # Share the elbow→wrist segment with Forearm_Bone so the elbow
            # anchor does not drift from the forearm driver under asymmetric
            # SMPL-X rest corrections (left elbow is often ~2× right).
            side = "left" if lower.endswith("_l") else "right"
            joint_a[bi] = joint_index[f"{side}_elbow"]
            joint_b[bi] = joint_index[f"{side}_wrist"]
            blend[bi] = 0.0
            driver_type = f"forearm_proximal_{side}"
        elif "forearm_bone" in lower or "forearm_twist" in lower:
            side = "left" if lower.endswith("_l") else "right"
            joint_a[bi] = joint_index[f"{side}_elbow"]
            joint_b[bi] = joint_index[f"{side}_wrist"]
            blend[bi] = 0.35 if "bone" in lower else 0.78
            driver_type = f"forearm_segment_{side}"
        elif "tibia_bone" in lower or "tibia_twist" in lower:
            side = "left" if lower.endswith("_l") else "right"
            joint_a[bi] = joint_index[f"{side}_knee"]
            joint_b[bi] = joint_index[f"{side}_ankle"]
            blend[bi] = 0.30 if "bone" in lower else 0.78
            driver_type = f"knee_chain_{side}"
        elif name == "Head_Bone":
            # Head pitch/yaw/roll is an orientation DOF, not recoverable from
            # the short neck->head position vector.  Runtime uses the SMPL-X
            # head global frame and this bind coupling preserves Blender roll.
            joint_a[bi] = joint_index["head"]
            joint_b[bi] = joint_index["head"]
            driver_type = "head_orientation"
        elif lower.startswith("rib_bone_") or lower.startswith("rib_name_"):
            digits = "".join(ch for ch in name if ch.isdigit())
            rib_number = max(1, min(12, int(digits or "6")))
            joint_a[bi] = joint_index["spine2"]
            joint_b[bi] = joint_index["spine3"]
            blend[bi] = float((12 - rib_number) / 11.0)
            driver_type = "rib_segment"

        pb = arm.pose.bones.get(name)
        if pb is None:
            raise RuntimeError(f"missing source pose bone {name}")
        source_pose = np.asarray(pb.matrix, dtype=np.float64).reshape(4, 4)
        U, _S, Vt = np.linalg.svd(rotation_global @ source_pose[:3, :3])
        R = U @ Vt
        if np.linalg.det(R) < 0.0:
            U[:, -1] *= -1.0
            R = U @ Vt
        point = _canonical_point(source_pose[:3, 3])
        rest_global[bi, :3, :3] = R
        rest_global[bi, :3, 3] = point
        bone_head[bi] = _canonical_point(
            np.asarray([pb.head.x, pb.head.y, pb.head.z], dtype=np.float64)
        )
        bone_tail[bi] = _canonical_point(
            np.asarray([pb.tail.x, pb.tail.y, pb.tail.z], dtype=np.float64)
        )
        bone_roll[bi] = _roll_about_axis(R, bone_head[bi], bone_tail[bi])
        driver_types.append(driver_type)

    rest_local = rest_global.copy()
    for bi, parent in enumerate(source_parents.tolist()):
        if int(parent) >= 0:
            rest_local[bi] = np.linalg.inv(rest_global[int(parent)]) @ rest_global[bi]

    return {
        "source_bone_names": names,
        "source_bone_parents": source_parents,
        "source_rest_global": rest_global.astype(np.float32),
        "source_rest_local": rest_local.astype(np.float32),
        "source_inverse_bind": np.linalg.inv(rest_global).astype(np.float32),
        "source_bone_head": bone_head.astype(np.float32),
        "source_bone_tail": bone_tail.astype(np.float32),
        "source_bone_roll": bone_roll.astype(np.float32),
        "source_bone_use_connect": bone_use_connect,
        "source_bone_inherit_scale": bone_inherit_scale,
        "source_bone_smplx_a": joint_a,
        "source_bone_smplx_b": joint_b,
        "source_bone_blend": blend,
        "source_bone_driver_types": driver_types,
    }


def _export_glb(meshes: list[bpy.types.Object], output_glb: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    if meshes:
        bpy.context.view_layer.objects.active = meshes[0]
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(output_glb), export_format="GLB", use_selection=True)


def main() -> None:
    args = _parse_args()
    config = _load_mapping(args.mapping)
    canonical = _load_canonical(
        args.canonical_dir,
        rest_space=str(config.get("canonical_rest_space", "neutral")),
    )
    joint_names = list(canonical["joint_names"])
    direct, direct_labels = _build_bone_to_joint(config, joint_names)
    arm = _armature()
    parents_by_bone = _bone_parents(arm)
    meshes = _selected_meshes(config)
    primary = _primary_anchor_bones(direct)

    pose_diag = _pose_armature_to_canonical(
        arm, canonical, primary=primary, joint_names=joint_names
    )
    deform = _bone_deform_matrices(arm)
    vertices, faces, weights, diag = _merge_and_skin_meshes(
        meshes,
        arm,
        config=config,
        joint_names=joint_names,
        direct_bone_to_joint=direct,
        parents_by_bone=parents_by_bone,
        deform=deform,
    )
    posed_vertices = vertices.copy()
    vertices, rest_align, align_context = _align_rest_to_canonical(
        vertices,
        weights,
        canonical,
        arm=arm,
        primary=primary,
    )
    # Skin_Glass is never published as anatomy.  It is exported only as the
    # source boundary for the offline source-skin volume registration.
    skin = bpy.data.objects.get("Skin_Glass")
    skin_vertices = np.zeros((0, 3), dtype=np.float32)
    skin_faces = np.zeros((0, 3), dtype=np.int32)
    if skin is not None and skin.type == "MESH":
        skin_config = dict(config)
        skin_config["include_collections"] = []
        skin_config["include_meshes"] = [str(skin.name)]
        skin_config["exclude_meshes"] = []
        raw_skin, skin_faces, _skin_weights, _skin_diag = _merge_and_skin_meshes(
            [skin], arm, config=skin_config, joint_names=joint_names,
            direct_bone_to_joint=direct, parents_by_bone=parents_by_bone, deform=deform,
        )
        skin_global = raw_skin @ align_context["linear"].T + align_context["translation"]
        skin_vertices = (skin_global + _sample_alignment_offset(skin_global, align_context)).astype(np.float32)
    rest_align.update(pose_diag)
    source_rig = _source_rig_canonical(
        arm,
        direct=direct,
        parents_by_bone=parents_by_bone,
        canonical=canonical,
        align=align_context,
        joint_names=joint_names,
    )
    registration_reference = (
        np.asarray(diag["raw_vertices"], dtype=np.float32) @ align_context["linear"].T
        + align_context["translation"]
    ).astype(np.float32)
    tri_edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    before_len = np.linalg.norm(
        posed_vertices[tri_edges[:, 0]] - posed_vertices[tri_edges[:, 1]], axis=1
    )
    after_len = np.linalg.norm(vertices[tri_edges[:, 0]] - vertices[tri_edges[:, 1]], axis=1)
    valid_edge = before_len > 1.0e-8
    unit_similarity_scale = max(float(rest_align.get("scale", 1.0)), 1.0e-8)
    canonical_ratio = after_len[valid_edge] / (before_len[valid_edge] * unit_similarity_scale)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        vertices_rest=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        joint_names=np.asarray(joint_names, dtype=object),
        parents=np.asarray(canonical["parents"], dtype=np.int32),
        rest_joints=np.asarray(canonical["rest_joints"], dtype=np.float32),
        inverse_bind=np.asarray(canonical["inverse_bind"], dtype=np.float32),
        source_mesh_names=np.asarray(diag["source_mesh_names"], dtype=object),
        source_vertex_ranges=np.asarray(diag["source_vertex_ranges"], dtype=np.int32),
        source_tissues=np.asarray(diag["source_tissues"], dtype=object),
        driver_indices=np.asarray(diag["source_driver_indices"], dtype=np.int16),
        driver_weights=np.asarray(diag["source_driver_weights"], dtype=np.float32),
        rigid_component_ids=np.asarray(diag["rigid_component_ids"], dtype=np.int32),
        source_bone_names=np.asarray(source_rig["source_bone_names"], dtype=object),
        source_bone_parents=np.asarray(source_rig["source_bone_parents"], dtype=np.int16),
        source_rest_global=np.asarray(source_rig["source_rest_global"], dtype=np.float32),
        source_rest_local=np.asarray(source_rig["source_rest_local"], dtype=np.float32),
        source_inverse_bind=np.asarray(source_rig["source_inverse_bind"], dtype=np.float32),
        source_bone_head=np.asarray(source_rig["source_bone_head"], dtype=np.float32),
        source_bone_tail=np.asarray(source_rig["source_bone_tail"], dtype=np.float32),
        source_bone_roll=np.asarray(source_rig["source_bone_roll"], dtype=np.float32),
        source_bone_use_connect=np.asarray(source_rig["source_bone_use_connect"], dtype=np.uint8),
        source_bone_inherit_scale=np.asarray(source_rig["source_bone_inherit_scale"], dtype=np.uint8),
        source_bone_smplx_a=np.asarray(source_rig["source_bone_smplx_a"], dtype=np.int16),
        source_bone_smplx_b=np.asarray(source_rig["source_bone_smplx_b"], dtype=np.int16),
        source_bone_blend=np.asarray(source_rig["source_bone_blend"], dtype=np.float32),
        source_bone_driver_types=np.asarray(source_rig["source_bone_driver_types"], dtype=object),
        registration_reference=registration_reference,
        source_skin_vertices=skin_vertices,
        source_skin_faces=np.asarray(skin_faces, dtype=np.int32),
        schema_version=np.asarray(3, dtype=np.int32),
        pose_format=np.asarray("smplx_body55_axis_angle", dtype=object),
        coordinate_system=np.asarray("genesis_z_up_m", dtype=object),
        metadata=np.asarray({"mapping": str(args.mapping), "driver_index_space": "blender_source_bones"}, dtype=object),
    )
    _export_glb(meshes, args.output_glb)
    report = {
        "blend_file": str(bpy.data.filepath),
        "output_npz": str(args.output_npz),
        "output_glb": str(args.output_glb),
        "mesh_count": int(len(meshes)),
        "vertex_count": int(vertices.shape[0]),
        "face_count": int(faces.shape[0]),
        "joint_count": int(len(joint_names)),
        "source_bone_count": int(len(source_rig["source_bone_names"])),
        "source_skin_vertices": int(len(skin_vertices)),
        "active_source_group_count": int(len(set(
            str(group.name) for obj in meshes for group in obj.vertex_groups
        ))),
        "rest_align": rest_align,
        "mesh_frame_modes": diag["frame_modes"],
        "primary_anchor_bones": {joint_names[j]: name for j, name in sorted(primary.items())},
        "direct_bone_mappings": direct_labels,
        "fallback_group_count": int(sum(diag["fallback_groups"].values())),
        "fallback_groups_sample": sorted(diag["fallback_groups"])[:80],
        "inherited_group_count": int(sum(diag["inherited_groups"].values())),
        "inherited_groups_sample": sorted(diag["inherited_groups"])[:80],
        "rigid_mesh_count": int(len(diag["rigid_meshes"])),
        "rigid_meshes_sample": sorted(diag["rigid_meshes"])[:80],
        "max_weight_sum_error": float(np.max(np.abs(weights.sum(axis=1) - 1.0))) if weights.size else 0.0,
        "empty_source_weight_vertices_repaired": int(diag["empty_source_weight_vertices"]),
        "missing_deform_groups": diag["missing_deform_groups"],
        "edge_stretch": {
            "source_to_pose_max": float(diag["source_pose_edge_ratio_max"]),
            "source_to_pose_p999": float(diag["source_pose_edge_ratio_p999"]),
            "pose_to_canonical_max": float(np.max(canonical_ratio)),
            "pose_to_canonical_p999": float(np.quantile(canonical_ratio, 0.999)),
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"anatomy retarget asset written -> {args.output_npz}", flush=True)


if __name__ == "__main__":
    main()
