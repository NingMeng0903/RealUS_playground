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


def _load_canonical(canonical_dir: Path) -> dict[str, Any]:
    weights_path = canonical_dir / "smpl_canonical_weights.npz"
    skeleton_path = canonical_dir / "smpl_canonical_skeleton.json"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if not skeleton_path.is_file():
        raise FileNotFoundError(skeleton_path)
    weights = np.load(weights_path, allow_pickle=True)
    skeleton = json.loads(skeleton_path.read_text(encoding="utf-8"))
    joint_names = [str(v) for v in weights["joint_names"].reshape(-1).tolist()]
    return {
        "joint_names": joint_names,
        "parents": np.asarray(weights["parents"], dtype=np.int32).reshape(-1),
        "rest_joints": np.asarray(weights["rest_joints"], dtype=np.float32).reshape(-1, 3),
        "inverse_bind": np.asarray(weights["inverse_bind"], dtype=np.float32).reshape(-1, 4, 4),
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
    ("right_collar", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
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

    Replicates Blender's armature modifier in armature object space: weights > 1 are
    normalized, the remainder below 1 stays at the rest position. Collapsed SMPL-55
    weights are exported alongside for runtime LBS.
    """
    max_influences = int(config.get("max_influences", 4))
    fallback_joint_name = str(config.get("fallback_joint", "pelvis"))
    fallback_joint = joint_names.index(fallback_joint_name) if fallback_joint_name in joint_names else 0
    joint_count = len(joint_names)
    joint_index = {name: idx for idx, name in enumerate(joint_names)}
    rigid_collections = set(str(v) for v in config.get("rigid_collections", []) or [])
    rigid_mesh_to_smplx = {
        str(k): str(v)
        for k, v in (config.get("rigid_mesh_to_smplx", {}) or {}).items()
        if str(v) in joint_index
    }
    arm_inv = np.asarray(arm.matrix_world.inverted(), dtype=np.float64).reshape(4, 4)
    arm_bbox = _armature_local_bbox(arm)
    frame_modes: dict[str, int] = {}

    all_vertices: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    source_mesh_names: list[str] = []
    source_vertex_ranges: list[tuple[int, int]] = []
    fallback_groups: dict[str, int] = {}
    inherited_groups: dict[str, int] = {}
    rigid_meshes: list[str] = []
    vertex_offset = 0

    for obj in meshes:
        source_mesh_names.append(str(obj.name))
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
        w55[empty, fallback_joint] = 1.0
        w55[~empty] /= totals[~empty][:, None]
        for vi in range(n):
            w55[vi] = _limit_weights(w55[vi], max_influences=max_influences)

        collections = set(_collections_for_object(obj))
        if rigid_collections and collections.intersection(rigid_collections):
            joint_name = rigid_mesh_to_smplx.get(str(obj.name))
            if joint_name is None:
                joint = int(np.argmax(w55.mean(axis=0)))
            else:
                joint = int(joint_index[joint_name])
            w55[:, :] = 0.0
            w55[:, joint] = 1.0
            rigid_meshes.append(str(obj.name))

        acc = np.zeros((n, 3), dtype=np.float32)
        total_w = np.zeros(n, dtype=np.float32)
        for gi, (idxs, ws) in group_elems.items():
            D = deform.get(group_names.get(gi, ""))
            if D is None:
                continue
            idx = np.asarray(idxs, dtype=np.int64)
            w = np.asarray(ws, dtype=np.float32)
            acc[idx] += w[:, None] * (raw[idx] @ D[:3, :3].T + D[:3, 3])
            total_w[idx] += w
        over = total_w > 1.0
        acc[over] /= total_w[over][:, None]
        remainder = np.clip(1.0 - total_w, 0.0, 1.0)
        posed = acc + remainder[:, None] * raw

        all_vertices.append(posed)
        all_weights.append(w55)
        for poly in mesh.polygons:
            indices = [vertex_offset + int(i) for i in poly.vertices]
            faces.extend(_triangulated_faces(indices))
        vertex_offset += n
        source_vertex_ranges.append((start, vertex_offset))

    return (
        np.concatenate(all_vertices, axis=0),
        np.asarray(faces, dtype=np.int32),
        np.concatenate(all_weights, axis=0),
        {
            "source_mesh_names": source_mesh_names,
            "source_vertex_ranges": source_vertex_ranges,
            "fallback_groups": fallback_groups,
            "inherited_groups": inherited_groups,
            "rigid_meshes": rigid_meshes,
            "frame_modes": frame_modes,
        },
    )


def _align_rest_to_canonical(
    vertices: np.ndarray,
    weights: np.ndarray,
    canonical: dict[str, Any],
    *,
    arm: bpy.types.Object,
    primary: dict[int, str],
) -> tuple[np.ndarray, dict[str, Any]]:
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

    verts = np.asarray(vertices, dtype=np.float32) @ G.T + tg
    verts = verts + np.asarray(weights, dtype=np.float32) @ offsets
    diag = {
        "mode": "fk_pose_global_procrustes",
        "scale": float(scale),
        "anchor_rms_m": rms,
        "mapped_joints": int(len(mapped)),
        "max_joint_offset_m": float(np.max(np.linalg.norm(offsets, axis=1))),
    }
    return verts.astype(np.float32), diag


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
    canonical = _load_canonical(args.canonical_dir)
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
    vertices, rest_align = _align_rest_to_canonical(
        vertices,
        weights,
        canonical,
        arm=arm,
        primary=primary,
    )
    rest_align.update(pose_diag)

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        vertices_rest=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        lbs_weights=weights.astype(np.float32),
        joint_names=np.asarray(joint_names, dtype=object),
        parents=np.asarray(canonical["parents"], dtype=np.int32),
        rest_joints=np.asarray(canonical["rest_joints"], dtype=np.float32),
        inverse_bind=np.asarray(canonical["inverse_bind"], dtype=np.float32),
        source_mesh_names=np.asarray(diag["source_mesh_names"], dtype=object),
        source_vertex_ranges=np.asarray(diag["source_vertex_ranges"], dtype=np.int32),
        pose_format=np.asarray("smplx_body55_axis_angle", dtype=object),
        coordinate_system=np.asarray("genesis_z_up_m", dtype=object),
        metadata=np.asarray({"mapping": str(args.mapping)}, dtype=object),
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
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"anatomy retarget asset written -> {args.output_npz}", flush=True)


if __name__ == "__main__":
    main()
