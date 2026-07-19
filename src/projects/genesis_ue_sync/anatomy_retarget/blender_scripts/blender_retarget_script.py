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

_ANATOMY_MODULE_DIR = Path(__file__).resolve().parents[1]
if str(_ANATOMY_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_ANATOMY_MODULE_DIR))

from anatomy_semantics import (  # noqa: E402
    ResolvedMeshSemantics,
    load_anatomy_semantics,
)
from source_audit import (  # noqa: E402
    aggregate_weight_stats,
    compress_runtime_influences,
    pack_source_influences,
    transform_audit,
)


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
    p.add_argument("--semantics-manifest", type=Path)
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


def _semantic_manifest_path(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> Path:
    explicit = args.semantics_manifest or config.get("semantics_manifest")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[5] / path
        return path.resolve()
    return (
        Path(__file__).resolve().parents[5]
        / "configs"
        / "anatomy"
        / "anatomy_semantics.yaml"
    ).resolve()


def _global_bind_matrices(rest_joints: np.ndarray, parents: np.ndarray) -> np.ndarray:
    joints = np.asarray(rest_joints, dtype=np.float32).reshape(-1, 3)
    pa = np.asarray(parents, dtype=np.int32).reshape(-1)
    out = np.tile(np.eye(4, dtype=np.float32), (joints.shape[0], 1, 1))
    for idx in range(joints.shape[0]):
        out[idx, :3, 3] = joints[idx]
    return out


def _driver_segment_frame(
    origin: np.ndarray,
    endpoint: np.ndarray,
    reference_x: np.ndarray,
) -> np.ndarray:
    y = np.asarray(endpoint - origin, dtype=np.float64)
    y /= max(float(np.linalg.norm(y)), 1.0e-10)
    x = np.asarray(reference_x, dtype=np.float64).copy()
    x -= float(x @ y) * y
    if float(np.linalg.norm(x)) < 1.0e-8:
        x = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(y)))]
        x -= float(x @ y) * y
    x /= max(float(np.linalg.norm(x)), 1.0e-10)
    z = np.cross(x, y)
    z /= max(float(np.linalg.norm(z)), 1.0e-10)
    x = np.cross(y, z)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.stack((x, y, z), axis=1)
    out[:3, 3] = origin
    return out


def _driver_three_joint_frame(
    points: np.ndarray,
    joints: np.ndarray,
    reference_x: np.ndarray,
) -> np.ndarray:
    ids = np.asarray(joints, dtype=np.int64)
    origin = np.asarray(points[int(ids[0])], dtype=np.float64)
    primary = np.asarray(points[int(ids[1])] - origin, dtype=np.float64)
    plane = np.asarray(points[int(ids[2])] - origin, dtype=np.float64)
    primary /= max(float(np.linalg.norm(primary)), 1.0e-10)
    normal = np.cross(primary, plane)
    if float(np.linalg.norm(normal)) < 1.0e-8:
        return _driver_segment_frame(origin, origin + primary, reference_x)
    normal /= float(np.linalg.norm(normal))
    transverse = np.cross(normal, primary)
    transverse /= max(float(np.linalg.norm(transverse)), 1.0e-10)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.stack((primary, transverse, normal), axis=1)
    out[:3, 3] = origin
    return out


def _source_driver_coupling(
    source_rig: dict[str, Any],
    canonical: dict[str, Any],
) -> np.ndarray:
    """Build the same C=inv(F_rest)@B_bind persisted by the host runtime."""
    points = np.asarray(canonical["rest_joints"], dtype=np.float64)
    joint_bind = _global_bind_matrices(points, canonical["parents"]).astype(np.float64)
    bind = np.asarray(source_rig["source_rest_global"], dtype=np.float64)
    parents = np.asarray(source_rig["source_bone_parents"], dtype=np.int64)
    modes = list(source_rig["source_bone_driver_types"])
    a_ids = np.asarray(source_rig["source_bone_smplx_a"], dtype=np.int64)
    b_ids = np.asarray(source_rig["source_bone_smplx_b"], dtype=np.int64)
    explicit_ids = np.asarray(source_rig["source_bone_frame_joints"], dtype=np.int64)
    coupling = np.tile(np.eye(4, dtype=np.float64), (len(bind), 1, 1))
    for bone, mode in enumerate(modes):
        if str(mode) == "bind_follow" and int(parents[bone]) >= 0:
            continue
        explicit = (
            np.all(explicit_ids[bone] >= 0)
            and len(np.unique(explicit_ids[bone])) == 3
        )
        a = int(a_ids[bone])
        b = int(b_ids[bone])
        if explicit:
            frame = _driver_three_joint_frame(points, explicit_ids[bone], bind[bone, :3, 0])
        elif str(mode) in {"segment_root", "rigid_group", "twist"} and a != b:
            frame = _driver_segment_frame(points[a], points[b], bind[bone, :3, 0])
        else:
            frame = joint_bind[a]
        coupling[bone] = np.linalg.inv(frame) @ bind[bone]
    return coupling.astype(np.float32)


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


def _selected_meshes(
    config: dict[str, Any],
) -> tuple[list[bpy.types.Object], list[dict[str, Any]]]:
    include_collections = set(str(v) for v in config.get("include_collections", []) or [])
    include_meshes = set(str(v) for v in config.get("include_meshes", []) or [])
    exclude_meshes = set(str(v) for v in config.get("exclude_meshes", []) or [])
    out: list[bpy.types.Object] = []
    excluded: list[dict[str, Any]] = []

    def _excluded_record(
        obj: bpy.types.Object,
        collections: set[str],
        reason: str,
    ) -> dict[str, Any]:
        mesh = obj.data
        return {
            "name": str(obj.name),
            "collections": sorted(collections),
            "reason": reason,
            "topology": {
                "vertices": int(len(mesh.vertices)),
                "edges": int(len(mesh.edges)),
                "polygons": int(len(mesh.polygons)),
                "triangles": int(
                    sum(max(0, len(poly.vertices) - 2) for poly in mesh.polygons)
                ),
            },
            "object_transform": {
                "matrix_world": transform_audit(obj.matrix_world),
                "matrix_local": transform_audit(obj.matrix_local),
                "matrix_basis": transform_audit(obj.matrix_basis),
            },
            "vertex_groups": [str(group.name) for group in obj.vertex_groups],
        }

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        collections = set(_collections_for_object(obj))
        if obj.name in exclude_meshes:
            excluded.append(
                _excluded_record(obj, collections, "configured_exclude_mesh")
            )
            continue
        if include_meshes and obj.name in include_meshes:
            out.append(obj)
            continue
        if include_collections and collections.intersection(include_collections):
            out.append(obj)
            continue
        excluded.append(
            _excluded_record(
                obj,
                collections,
                "outside_include_meshes_and_collections",
            )
        )
    if not out:
        raise RuntimeError("No anatomy meshes selected by mapping config")
    return (
        sorted(out, key=lambda obj: obj.name),
        sorted(excluded, key=lambda row: str(row["name"])),
    )


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


def _exact_source_driver_width(
    meshes: list[bpy.types.Object],
    *,
    bone_index: dict[str, int],
) -> int:
    maximum = 0
    for obj in meshes:
        group_names = {int(group.index): str(group.name) for group in obj.vertex_groups}
        for vertex in obj.data.vertices:
            active = {
                bone_index[group_names[int(elem.group)]]
                for elem in vertex.groups
                if (
                    float(elem.weight) > 0.0
                    and group_names.get(int(elem.group), "") in bone_index
                )
            }
            maximum = max(maximum, len(active))
    return max(1, maximum)


def _source_influences(
    mesh: bpy.types.Mesh,
    *,
    group_names: dict[int, str],
    bone_index: dict[str, int],
    driver_width: int,
) -> tuple[Any, np.ndarray, np.ndarray, int]:
    """Preserve exact source values and derive an all-influence runtime view."""

    packed = pack_source_influences(
        (
            ((int(elem.group), float(elem.weight)) for elem in vertex.groups)
            for vertex in mesh.vertices
        ),
        group_names=group_names,
        source_bone_index=bone_index,
        driver_width=driver_width,
    )
    indices = np.array(packed.driver_indices, dtype=np.int16, copy=True)
    weights = np.array(packed.driver_weights, dtype=np.float32, copy=True)
    empty = [int(value) for value in packed.empty_driver_vertices.tolist()]
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
                known = next(
                    (vj for vj in sorted(neighbors[vi]) if vj not in pending),
                    None,
                )
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
    return packed, indices, weights, len(empty)


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
        if not bool(bone.use_deform):
            continue
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
    semantics: dict[str, ResolvedMeshSemantics],
    joint_names: list[str],
    direct_bone_to_joint: dict[str, int],
    parents_by_bone: dict[str, str | None],
    deform: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Merge selected meshes; skin them with the FULL original bone weights.

    Replicates Blender's armature modifier in armature object space after explicitly
    normalizing every non-empty source weight row. Leaving ``1-sum(weights)`` at
    the bind position creates metre-scale edges when the weighted part moves.
    Exact source values are exported as CSR. Runtime source-bone weights retain
    every influence and are normalized separately for LBS.
    """
    compressed_top_k = int(config.get("max_influences", 0))
    fallback_joint_name = str(config.get("fallback_joint", "pelvis"))
    fallback_joint = joint_names.index(fallback_joint_name) if fallback_joint_name in joint_names else 0
    joint_count = len(joint_names)
    source_bone_names = [str(b.name) for b in arm.data.bones]
    source_bone_index = {name: idx for idx, name in enumerate(source_bone_names)}
    deform_bone_index = {
        str(bone.name): int(source_bone_index[str(bone.name)])
        for bone in arm.data.bones
        if bool(bone.use_deform)
    }
    driver_width = _exact_source_driver_width(meshes, bone_index=deform_bone_index)
    rigid_collections = set(str(v) for v in config.get("rigid_collections", []) or [])
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
    source_mesh_controller_bones: list[int] = []
    source_mesh_material_groups: list[str] = []
    source_mesh_roles: list[str] = []
    source_fit_policies: list[str] = []
    source_driver_policies: list[str] = []
    source_compound_ids: list[str] = []
    source_sides: list[str] = []
    source_landmarks: list[tuple[str, ...]] = []
    target_landmark_recipes: list[str] = []
    source_quality_profiles: list[str] = []
    fallback_groups: dict[str, int] = {}
    inherited_groups: dict[str, int] = {}
    rigid_meshes: list[str] = []
    empty_source_weight_vertices = 0
    missing_deform_groups: dict[str, int] = {}
    source_pose_edge_ratios: list[np.ndarray] = []
    source_influence_offsets: list[int] = [0]
    source_influence_group_indices: list[np.ndarray] = []
    source_influence_values: list[np.ndarray] = []
    source_group_names: list[str] = []
    source_group_mesh_indices: list[int] = []
    source_group_local_indices: list[int] = []
    source_group_bone_indices: list[int] = []
    mesh_audit_records: list[dict[str, Any]] = []
    mesh_weight_stats: list[dict[str, Any]] = []
    blender_parity_records: list[dict[str, Any]] = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertex_offset = 0
    rigid_component_counter = 0

    for mesh_index, obj in enumerate(meshes):
        source_mesh_names.append(str(obj.name))
        object_collections = set(_collections_for_object(obj))
        semantic = semantics.get(str(obj.name))
        if semantic is None:
            raise RuntimeError(f"mesh {obj.name!r} has no resolved semantic record")
        source_tissues.append(semantic.tissue_type)
        material_group = semantic.material_group
        role = semantic.role
        source_fit_policies.append(semantic.fit_policy)
        source_driver_policies.append(semantic.driver_policy)
        source_compound_ids.append(semantic.compound_id)
        source_sides.append(semantic.side)
        source_landmarks.append(semantic.source_landmarks)
        target_landmark_recipes.append(semantic.target_landmark_recipe)
        source_quality_profiles.append(semantic.quality_profile)
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

        packed, source_indices, source_weights, source_empty_count = _source_influences(
            mesh,
            group_names=group_names,
            bone_index=deform_bone_index,
            driver_width=driver_width,
        )
        global_group_by_local: dict[int, int] = {}
        for local_group_index, group_name in sorted(group_names.items()):
            global_group_by_local[int(local_group_index)] = len(source_group_names)
            source_group_names.append(str(group_name))
            source_group_mesh_indices.append(int(mesh_index))
            source_group_local_indices.append(int(local_group_index))
            source_group_bone_indices.append(
                int(source_bone_index.get(str(group_name), -1))
            )
        source_influence_group_indices.append(
            np.asarray(
                [
                    global_group_by_local[int(local_group)]
                    for local_group in packed.source_group_indices.tolist()
                ],
                dtype=np.int32,
            )
        )
        source_influence_values.append(
            np.asarray(packed.source_values, dtype=np.float32)
        )
        influence_base = source_influence_offsets[-1]
        source_influence_offsets.extend(
            (np.asarray(packed.source_offsets[1:], dtype=np.int64) + influence_base)
            .astype(np.int64)
            .tolist()
        )
        mesh_weight_stats.append(dict(packed.stats))

        to_arm, frame_mode = _mesh_to_armature_transform(obj, arm_inv, arm_bbox)
        raw_local = np.empty(n * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", raw_local)
        raw_local_points = raw_local.reshape(n, 3).astype(np.float64)
        raw = raw_local_points @ to_arm[:3, :3].T + to_arm[:3, 3]
        raw = raw.astype(np.float32)

        group_elems: dict[int, tuple[list[int], list[float]]] = {}
        w55 = np.zeros((n, joint_count), dtype=np.float32)
        for vi, vertex in enumerate(mesh.vertices):
            for elem in vertex.groups:
                gi = int(elem.group)
                wv = float(elem.weight)
                group_name = group_names.get(gi, "")
                if wv <= 0.0 or group_name not in deform_bone_index:
                    continue
                idxs, ws = group_elems.setdefault(gi, ([], []))
                idxs.append(vi)
                ws.append(wv)
                w55[vi, group_to_joint.get(gi, fallback_joint)] += wv
        totals = w55.sum(axis=1)
        empty = totals <= 1.0e-8
        w55[~empty] /= totals[~empty][:, None]

        collections = set(_collections_for_object(obj))
        is_rigid = bool(
            rigid_collections
            and collections.intersection(rigid_collections)
            and semantic.fit_policy not in {"soft_volume", "audit_only"}
        )
        if is_rigid:
            rigid_meshes.append(str(obj.name))

        source_totals = np.zeros(n, dtype=np.float32)
        for _gi, (idxs, ws) in group_elems.items():
            source_totals[np.asarray(idxs, dtype=np.int64)] += np.asarray(ws, dtype=np.float32)
        source_empty = source_totals <= 1.0e-8
        empty_source_weight_vertices += int(source_empty_count)

        def _deform_points(raw_points: np.ndarray) -> np.ndarray:
            acc = np.zeros((n, 3), dtype=np.float32)
            applied = np.zeros(n, dtype=np.float32)
            for gi, (idxs, ws) in group_elems.items():
                group_name = group_names.get(gi, "")
                D = deform.get(group_name)
                if D is None:
                    missing_deform_groups[group_name] = (
                        missing_deform_groups.get(group_name, 0) + len(idxs)
                    )
                    continue
                idx = np.asarray(idxs, dtype=np.int64)
                w = np.asarray(ws, dtype=np.float32) / np.maximum(
                    source_totals[idx], 1.0e-8
                )
                acc[idx] += w[:, None] * (
                    raw_points[idx] @ D[:3, :3].T + D[:3, 3]
                )
                applied[idx] += w
            missing_vertex = (~source_empty) & (applied <= 1.0e-8)
            if np.any(missing_vertex):
                names = sorted(
                    key for key, count in missing_deform_groups.items() if count > 0
                )
                raise RuntimeError(
                    f"{obj.name}: {int(np.count_nonzero(missing_vertex))} weighted "
                    f"vertices have no armature deform; missing groups sample={names[:12]}"
                )
            result = np.asarray(raw_points, dtype=np.float32).copy()
            valid = applied > 1.0e-8
            result[valid] = acc[valid] / applied[valid, None]
            if np.any(source_empty):
                result, _propagated_weights = _propagate_empty_vertex_data(
                    mesh=mesh,
                    raw=np.asarray(raw_points, dtype=np.float32),
                    posed=result,
                    weights=w55,
                    empty=source_empty,
                )
            return result

        posed = _deform_points(raw)
        modifier_visibility = [
            (modifier, bool(modifier.show_viewport))
            for modifier in obj.modifiers
            if str(modifier.type) != "ARMATURE"
        ]
        object_hidden = bool(obj.hide_viewport)
        obj.hide_viewport = False
        for modifier, _visible in modifier_visibility:
            modifier.show_viewport = False
        bpy.context.view_layer.update()
        evaluated_object = obj.evaluated_get(depsgraph)
        evaluated_mesh = evaluated_object.to_mesh()
        try:
            if len(evaluated_mesh.vertices) != n:
                raise RuntimeError(
                    f"{obj.name}: evaluated Blender mesh changed vertex count "
                    f"{n}->{len(evaluated_mesh.vertices)}; source-weight parity is undefined"
                )
            evaluated_local = np.empty(n * 3, dtype=np.float32)
            evaluated_mesh.vertices.foreach_get("co", evaluated_local)
            evaluated_local_points = evaluated_local.reshape(n, 3).astype(np.float64)
            evaluated = evaluated_local_points @ to_arm[:3, :3].T + to_arm[:3, 3]
            parity_error = np.linalg.norm(
                evaluated - np.asarray(posed, dtype=np.float64),
                axis=1,
            )
            alternate = (
                arm_inv
                @ np.asarray(obj.matrix_world, dtype=np.float64).reshape(4, 4)
                if frame_mode == "identity"
                else np.eye(4, dtype=np.float64)
            )
            alternate_raw = (
                raw_local_points @ alternate[:3, :3].T + alternate[:3, 3]
            ).astype(np.float32)
            alternate_posed = _deform_points(alternate_raw)
            alternate_evaluated = (
                evaluated_local_points @ alternate[:3, :3].T + alternate[:3, 3]
            )
            alternate_error = np.linalg.norm(
                alternate_evaluated - np.asarray(alternate_posed, dtype=np.float64),
                axis=1,
            )
            if float(np.max(alternate_error)) < float(np.max(parity_error)):
                to_arm = alternate
                frame_mode = (
                    "standard_mirrored"
                    if float(np.linalg.det(alternate[:3, :3])) < 0.0
                    else ("standard" if frame_mode == "identity" else "identity")
                )
                raw = alternate_raw
                posed = alternate_posed
                evaluated = alternate_evaluated
                parity_error = alternate_error
            parity_record = {
                "mesh": str(obj.name),
                "vertex_count": int(n),
                "rms_error_source_units": float(
                    np.sqrt(np.mean(parity_error * parity_error))
                ),
                "max_error_source_units": float(np.max(parity_error)),
            }
            blender_parity_records.append(parity_record)
            frame_modes[frame_mode] = frame_modes.get(frame_mode, 0) + 1
        finally:
            evaluated_object.to_mesh_clear()
            for modifier, visible in modifier_visibility:
                modifier.show_viewport = visible
            obj.hide_viewport = object_hidden
            bpy.context.view_layer.update()
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
        controller_mass = np.zeros(len(source_bone_names), dtype=np.float64)
        for slot in range(source_indices.shape[1]):
            np.add.at(controller_mass, source_indices[:, slot], source_weights[:, slot])
        controller_bone = int(np.argmax(controller_mass))
        source_mesh_controller_bones.append(controller_bone)
        source_mesh_material_groups.append(material_group)
        source_mesh_roles.append(role)
        for poly in mesh.polygons:
            indices = [vertex_offset + int(i) for i in poly.vertices]
            faces.extend(_triangulated_faces(indices))
        vertex_offset += n
        source_vertex_ranges.append((start, vertex_offset))
        mesh_audit_records.append(
            {
                "name": str(obj.name),
                "data_name": str(mesh.name),
                "collections": sorted(object_collections),
                "selection": "included",
                "topology": {
                    "vertices": int(len(mesh.vertices)),
                    "edges": int(len(mesh.edges)),
                    "polygons": int(len(mesh.polygons)),
                    "triangles": int(
                        sum(max(0, len(poly.vertices) - 2) for poly in mesh.polygons)
                    ),
                    "loops": int(len(mesh.loops)),
                },
                "object_transform": {
                    "matrix_world": transform_audit(obj.matrix_world),
                    "matrix_local": transform_audit(obj.matrix_local),
                    "matrix_basis": transform_audit(obj.matrix_basis),
                    "parent": str(obj.parent.name) if obj.parent is not None else None,
                },
                "mesh_to_armature_transform": {
                    "mode": frame_mode,
                    **transform_audit(to_arm),
                },
                "vertex_groups": [
                    {
                        "index": int(group_index),
                        "name": str(group_name),
                        "source_bone_index": int(
                            source_bone_index.get(str(group_name), -1)
                        ),
                    }
                    for group_index, group_name in sorted(group_names.items())
                ],
                "weight_influences": dict(packed.stats),
                "semantics": semantic.to_dict(),
                "modifiers": [
                    {
                        "name": str(modifier.name),
                        "type": str(modifier.type),
                        "use_deform_preserve_volume": bool(
                            getattr(modifier, "use_deform_preserve_volume", False)
                        ),
                    }
                    for modifier in obj.modifiers
                ],
            }
        )

    if fallback_groups and bool(config.get("fail_on_unmapped_groups", True)):
        raise RuntimeError(
            "Unmapped Blender vertex groups cannot silently fall back to pelvis: "
            + ", ".join(sorted(fallback_groups)[:24])
        )

    edge_ratio = np.concatenate(source_pose_edge_ratios) if source_pose_edge_ratios else np.ones(1, dtype=np.float32)
    runtime_indices = np.concatenate(all_source_indices, axis=0)
    runtime_weights = np.concatenate(all_source_weights, axis=0)
    if compressed_top_k > 0:
        compressed = compress_runtime_influences(
            runtime_indices,
            runtime_weights,
            top_k=compressed_top_k,
        )
        compressed_indices = np.asarray(compressed.indices, dtype=np.int16)
        compressed_weights = np.asarray(compressed.weights, dtype=np.float32)
        compression_error = dict(compressed.error)
    else:
        compressed_indices = np.zeros((len(runtime_indices), 0), dtype=np.int16)
        compressed_weights = np.zeros((len(runtime_indices), 0), dtype=np.float32)
        compression_error = {
            "top_k": 0,
            "source_width": int(runtime_indices.shape[1]),
            "affected_vertex_count": 0,
            "omitted_mass_max": 0.0,
            "omitted_mass_mean": 0.0,
            "l1_error_max": 0.0,
            "l1_error_mean": 0.0,
        }
    source_group_ids = (
        np.concatenate(source_influence_group_indices, axis=0)
        if source_influence_group_indices
        else np.zeros(0, dtype=np.int32)
    )
    source_values = (
        np.concatenate(source_influence_values, axis=0)
        if source_influence_values
        else np.zeros(0, dtype=np.float32)
    )
    if len(source_influence_offsets) != len(runtime_indices) + 1:
        raise RuntimeError("source influence CSR offsets do not cover all exported vertices")
    return (
        np.concatenate(all_vertices, axis=0),
        np.asarray(faces, dtype=np.int32),
        np.concatenate(all_weights, axis=0),
        {
            "source_mesh_names": source_mesh_names,
            "source_vertex_ranges": source_vertex_ranges,
            "source_tissues": source_tissues,
            "source_mesh_controller_bones": source_mesh_controller_bones,
            "source_mesh_material_groups": source_mesh_material_groups,
            "source_mesh_roles": source_mesh_roles,
            "source_fit_policies": source_fit_policies,
            "source_driver_policies": source_driver_policies,
            "source_compound_ids": source_compound_ids,
            "source_sides": source_sides,
            "source_landmarks": source_landmarks,
            "target_landmark_recipes": target_landmark_recipes,
            "source_quality_profiles": source_quality_profiles,
            "fallback_groups": fallback_groups,
            "inherited_groups": inherited_groups,
            "rigid_meshes": rigid_meshes,
            "frame_modes": frame_modes,
            "empty_source_weight_vertices": int(empty_source_weight_vertices),
            "missing_deform_groups": missing_deform_groups,
            "source_pose_edge_ratio_max": float(np.max(edge_ratio)),
            "source_pose_edge_ratio_p999": float(np.quantile(edge_ratio, 0.999)),
            "source_driver_indices": runtime_indices,
            "source_driver_weights": runtime_weights,
            "runtime_driver_indices_compressed": compressed_indices,
            "runtime_driver_weights_compressed": compressed_weights,
            "runtime_weight_compression_error": compression_error,
            "source_influence_offsets": np.asarray(
                source_influence_offsets, dtype=np.int64
            ),
            "source_influence_group_indices": source_group_ids,
            "source_influence_values": source_values,
            "source_group_names": source_group_names,
            "source_group_mesh_indices": np.asarray(
                source_group_mesh_indices, dtype=np.int32
            ),
            "source_group_local_indices": np.asarray(
                source_group_local_indices, dtype=np.int32
            ),
            "source_group_bone_indices": np.asarray(
                source_group_bone_indices, dtype=np.int32
            ),
            "source_weight_audit": aggregate_weight_stats(mesh_weight_stats),
            "mesh_audit_records": mesh_audit_records,
            "blender_parity": {
                "mesh_count": int(len(blender_parity_records)),
                "vertex_count": int(
                    sum(record["vertex_count"] for record in blender_parity_records)
                ),
                "rms_error_source_units": float(
                    np.sqrt(
                        np.mean(
                            [
                                float(record["rms_error_source_units"]) ** 2
                                for record in blender_parity_records
                            ]
                        )
                    )
                )
                if blender_parity_records
                else 0.0,
                "max_error_source_units": float(
                    max(
                        (
                            float(record["max_error_source_units"])
                            for record in blender_parity_records
                        ),
                        default=0.0,
                    )
                ),
                "records": blender_parity_records,
            },
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
    """Place the authored T-pose in the canonical metric frame.

    This extraction stage intentionally performs *only* a global similarity.
    Articulated bone fitting and the soft-tissue volume solve happen later with
    explicit material semantics.  The former whole-body RBF moved skull,
    pelvis, finger and toe vertices independently and permanently destroyed
    their authored shape before any rigid-preservation stage could run.
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
    anchor_residual = a_glob - rest_joints[mapped]
    diag = {
        "mode": "fk_pose_global_similarity_only",
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
    }


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
    joint_a = np.zeros(len(bones), dtype=np.int16)
    joint_b = np.zeros(len(bones), dtype=np.int16)
    blend = np.zeros(len(bones), dtype=np.float32)
    driver_types: list[str] = []
    frame_joints = np.full((len(bones), 3), -1, dtype=np.int16)
    # Blender's read-only Bone API does not expose EditBone.roll in all
    # versions.  The authoritative roll is already preserved by each complete
    # rest-frame rotation matrix; this scalar remains compatibility metadata.
    source_roll = np.asarray(
        [float(getattr(bone, "roll", 0.0)) for bone in bones], dtype=np.float32
    )
    source_use_connect = np.asarray(
        [bool(bone.use_connect) for bone in bones], dtype=np.uint8
    )
    inherit_scale_codes = {
        "FULL": 0,
        "FIX_SHEAR": 1,
        "AVERAGE": 2,
        "NONE": 3,
        "NONE_LEGACY": 4,
        "ALIGNED": 5,
    }
    unknown_inherit_scale = sorted(
        {str(bone.inherit_scale) for bone in bones} - set(inherit_scale_codes)
    )
    if unknown_inherit_scale:
        raise RuntimeError(
            f"unsupported Blender bone inherit_scale values: {unknown_inherit_scale}"
        )
    source_inherit_scale = np.asarray(
        [inherit_scale_codes[str(bone.inherit_scale)] for bone in bones],
        dtype=np.uint8,
    )

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
        return np.asarray(point, dtype=np.float64) @ linear.T + translation

    for bi, bone in enumerate(bones):
        name = str(bone.name)
        mapped, inherited = _resolve_group_joint(
            name, direct=direct, parents_by_bone=parents_by_bone, fallback=fallback
        )
        mapped = int(mapped)
        joint_a[bi] = mapped
        joint_b[bi] = mapped
        driver_type = "bind_follow" if inherited else "joint_local"

        # A source rig often has two deform bones for one anatomical joint
        # (rotate control + rigid follower).  Only the first direct mapping is
        # allowed to consume that SMPL-X rotation; later bones retain their
        # authored bind-local relation.
        if not inherited:
            ancestor = parents_by_bone.get(name)
            while ancestor is not None and ancestor not in direct:
                ancestor = parents_by_bone.get(ancestor)
            if ancestor is not None and int(direct[ancestor]) == mapped:
                driver_type = "bind_follow"

        lower = name.lower()
        if "scapula" in lower:
            driver_type = "rigid_group"
        elif "clavicle_rot" in lower:
            side = _bone_side(name, lower)
            if side is not None:
                joint_a[bi] = joint_index["spine3"]
                joint_b[bi] = joint_index[f"{side}_collar"]
                driver_type = "segment_root"
        elif "shoulder_rotate" in lower:
            side = _bone_side(name, lower)
            if side is not None:
                joint_a[bi] = joint_index[f"{side}_shoulder"]
                joint_b[bi] = joint_index[f"{side}_elbow"]
                driver_type = "segment_root"
        elif "knee_rotate" in lower:
            side = _bone_side(name, lower)
            if side is not None:
                joint_a[bi] = joint_index[f"{side}_knee"]
                joint_b[bi] = joint_index[f"{side}_ankle"]
                driver_type = "segment_root"
        elif _is_foot_chain_bone(lower) or "toes_rotate" in lower:
            side = _bone_side(name, lower)
            if side is not None:
                joint_a[bi] = joint_index[f"{side}_ankle"]
                joint_b[bi] = joint_index[f"{side}_foot"]
                if "ankle_rot" in lower:
                    driver_type = "rigid_group"
                elif "arch_rot" in lower:
                    joint_a[bi] = joint_index[f"{side}_foot"]
                    joint_b[bi] = joint_index[f"{side}_foot"]
                    driver_type = "joint_local"
                else:
                    driver_type = "bind_follow"
        elif "patella" in lower or "fibula" in lower:
            side = "left" if lower.endswith("_l") else "right"
            joint_a[bi] = joint_index[f"{side}_knee"]
            joint_b[bi] = joint_index[f"{side}_ankle"]
            driver_type = "rigid_group" if "patella" in lower else "bind_follow"
        elif "femur_rot" in lower or name.startswith("Femur_Rot_"):
            side = "left" if lower.endswith("_l") else "right"
            joint_a[bi] = joint_index[f"{side}_hip"]
            joint_b[bi] = joint_index[f"{side}_knee"]
            driver_type = "segment_root"
        elif "elbow_rot" in lower:
            # Share the elbow→wrist segment with Forearm_Bone so the elbow
            # anchor does not drift from the forearm driver under asymmetric
            # SMPL-X rest corrections (left elbow is often ~2× right).
            side = "left" if lower.endswith("_l") else "right"
            joint_a[bi] = joint_index[f"{side}_elbow"]
            joint_b[bi] = joint_index[f"{side}_wrist"]
            driver_type = "segment_root"
        elif "forearm_bone" in lower or "forearm_twist" in lower:
            side = "left" if lower.endswith("_l") else "right"
            joint_a[bi] = joint_index[f"{side}_elbow"]
            joint_b[bi] = joint_index[f"{side}_wrist"]
            blend[bi] = 0.78
            driver_type = "twist" if "twist" in lower else "segment_root"
        elif "tibia_bone" in lower or "tibia_twist" in lower:
            side = "left" if lower.endswith("_l") else "right"
            joint_a[bi] = joint_index[f"{side}_knee"]
            joint_b[bi] = joint_index[f"{side}_ankle"]
            blend[bi] = 0.78
            driver_type = "twist" if "twist" in lower else "segment_root"
        elif "wrist_rotate" in lower:
            side = _bone_side(name, lower)
            if side is not None:
                joint_a[bi] = joint_index[f"{side}_wrist"]
                joint_b[bi] = joint_index[f"{side}_wrist"]
                driver_type = "joint_local"
        elif name == "Head_Bone":
            # Head pitch/yaw/roll is an orientation DOF, not recoverable from
            # the short neck->head position vector.  Runtime uses the SMPL-X
            # head global frame and this bind coupling preserves Blender roll.
            joint_a[bi] = joint_index["head"]
            joint_b[bi] = joint_index["head"]
            driver_type = "rigid_group"
        elif lower.startswith("rib_bone_") or lower.startswith("rib_name_"):
            # Each rib is authored as a child of its thoracic vertebra.  Follow
            # that parent bind-local rather than collapsing every rib onto the
            # spine2→spine3 segment (which exploded the cage under pose).
            digits = "".join(ch for ch in name if ch.isdigit())
            rib_number = max(1, min(12, int(digits or "6")))
            level = f"spine{2 if rib_number >= 8 else 3}"
            if level not in joint_index:
                level = "spine3"
            joint_a[bi] = joint_index[level]
            joint_b[bi] = joint_index[level]
            blend[bi] = 0.0
            driver_type = "bind_follow"

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
        # Persist every frame dependency.  Runtime deliberately has no
        # child-joint fallback: an exporter must make the frame choice here.
        frame_joints[bi, 0] = joint_a[bi]
        frame_joints[bi, 1] = joint_b[bi]
        side = _bone_side(name, lower)
        if ("scapula" in lower or "clavicle" in lower) and side is not None:
            frame_joints[bi] = np.asarray(
                (joint_index["spine3"], joint_index[f"{side}_collar"], joint_index[f"{side}_shoulder"]),
                dtype=np.int16,
            )
            # The primary source driver remains shoulder for backward-compatible
            # LBS lookup while the explicit three-point frame controls it.
            joint_a[bi] = frame_joints[bi, 0]
        elif "pelvis" in lower:
            frame_joints[bi] = np.asarray(
                (joint_index["pelvis"], joint_index["left_hip"], joint_index["right_hip"]), dtype=np.int16
            )
            joint_a[bi] = frame_joints[bi, 0]
        elif name == "Head_Bone":
            frame_joints[bi] = np.asarray(
                (
                    joint_index["head"],
                    joint_index["left_eye_smplhf"],
                    joint_index["right_eye_smplhf"],
                ),
                dtype=np.int16,
            )
            joint_a[bi] = frame_joints[bi, 0]
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
        "source_bone_roll": source_roll,
        "source_bone_use_connect": source_use_connect,
        "source_bone_inherit_scale": source_inherit_scale,
        "source_bone_smplx_a": joint_a,
        "source_bone_smplx_b": joint_b,
        "source_bone_blend": blend,
        "source_bone_driver_types": driver_types,
        "source_bone_frame_joints": frame_joints,
    }


def _export_glb(meshes: list[bpy.types.Object], output_glb: Path) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    if meshes:
        bpy.context.view_layer.objects.active = meshes[0]
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(output_glb), export_format="GLB", use_selection=True)


def _animation_driver_count(owner: Any) -> int:
    animation = getattr(owner, "animation_data", None)
    return int(len(list(getattr(animation, "drivers", ()) or ())))


def _source_mechanism_audit(
    arm: bpy.types.Object,
    meshes: list[bpy.types.Object],
) -> dict[str, Any]:
    """Record every Blender mechanism that could add pose-dependent motion."""
    object_constraints = sum(len(obj.constraints) for obj in [arm, *meshes])
    pose_constraints = sum(len(bone.constraints) for bone in arm.pose.bones)
    object_drivers = sum(_animation_driver_count(obj) for obj in [arm, *meshes])
    shape_key_count = 0
    shape_key_drivers = 0
    active_shape_keys = 0
    for obj in meshes:
        keys = getattr(obj.data, "shape_keys", None)
        if keys is None:
            continue
        blocks = list(keys.key_blocks)
        shape_key_count += len(blocks)
        shape_key_drivers += _animation_driver_count(keys)
        active_shape_keys += sum(
            abs(float(key.value)) > 1.0e-8 and not bool(key.mute)
            for key in blocks
            if str(key.name).lower() not in {"basis", "base"}
        )
    missing: list[str] = []
    if object_constraints or pose_constraints:
        missing.append("constraints")
    if object_drivers or shape_key_drivers:
        missing.append("drivers")
    if shape_key_count:
        missing.append("shape_keys")
    preserve_volume_modifiers = int(
        sum(
            modifier.type == "ARMATURE"
            and bool(getattr(modifier, "use_deform_preserve_volume", False))
            for obj in meshes
            for modifier in obj.modifiers
        )
    )
    if preserve_volume_modifiers:
        missing.append("armature_preserve_volume_mode")
    return {
        "armature_modifiers": int(
            sum(modifier.type == "ARMATURE" for obj in meshes for modifier in obj.modifiers)
        ),
        "armature_preserve_volume_modifiers": preserve_volume_modifiers,
        "object_constraints": int(object_constraints),
        "pose_bone_constraints": int(pose_constraints),
        "object_drivers": int(object_drivers),
        "shape_key_count": int(shape_key_count),
        "shape_key_drivers": int(shape_key_drivers),
        "active_nonbasis_shape_keys": int(active_shape_keys),
        "serialized": [
            "armature_bone_names",
            "armature_parent_hierarchy",
            "armature_rest_local_global_frames",
            "armature_connected_flags",
            "armature_roll_and_inherit_scale",
            "all_sparse_armature_vertex_weights",
        ],
        "not_serialized": missing,
        "armature_only_source": not missing,
    }


def main() -> None:
    args = _parse_args()
    config = _load_mapping(args.mapping)
    semantic_manifest_path = _semantic_manifest_path(args, config)
    semantic_manifest = load_anatomy_semantics(semantic_manifest_path)
    canonical = _load_canonical(
        args.canonical_dir,
        rest_space=str(config.get("canonical_rest_space", "neutral")),
    )
    joint_names = list(canonical["joint_names"])
    direct, direct_labels = _build_bone_to_joint(config, joint_names)
    arm = _armature()
    parents_by_bone = _bone_parents(arm)
    meshes, excluded_meshes = _selected_meshes(config)
    skin = bpy.data.objects.get("Skin_Glass")
    semantic_objects = list(meshes)
    if skin is not None and skin.type == "MESH" and skin not in semantic_objects:
        semantic_objects.append(skin)
    resolved_semantics = semantic_manifest.resolve_many(
        (
            (str(obj.name), _collections_for_object(obj))
            for obj in semantic_objects
        )
    )
    primary = _primary_anchor_bones(direct)

    pose_diag = _pose_armature_to_canonical(
        arm, canonical, primary=primary, joint_names=joint_names
    )
    deform = _bone_deform_matrices(arm)
    vertices, faces, weights, diag = _merge_and_skin_meshes(
        meshes,
        arm,
        config=config,
        semantics=resolved_semantics,
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
    skin_vertices = np.zeros((0, 3), dtype=np.float32)
    skin_faces = np.zeros((0, 3), dtype=np.int32)
    skin_weights = np.zeros((0, len(joint_names)), dtype=np.float32)
    if skin is not None and skin.type == "MESH":
        skin_config = dict(config)
        skin_config["include_collections"] = []
        skin_config["include_meshes"] = [str(skin.name)]
        skin_config["exclude_meshes"] = []
        raw_skin, skin_faces, skin_weights, _skin_diag = _merge_and_skin_meshes(
            [skin], arm, config=skin_config, semantics=resolved_semantics,
            joint_names=joint_names,
            direct_bone_to_joint=direct, parents_by_bone=parents_by_bone, deform=deform,
        )
        skin_global = raw_skin @ align_context["linear"].T + align_context["translation"]
        skin_vertices = skin_global.astype(np.float32)
    rest_align.update(pose_diag)
    source_rig = _source_rig_canonical(
        arm,
        direct=direct,
        parents_by_bone=parents_by_bone,
        canonical=canonical,
        align=align_context,
        joint_names=joint_names,
    )
    source_driver_coupling = _source_driver_coupling(source_rig, canonical)
    registration_reference = (
        np.asarray(diag["raw_vertices"], dtype=np.float32) @ align_context["linear"].T
        + align_context["translation"]
    ).astype(np.float32)
    source_inverse = np.linalg.inv(np.asarray(source_rig["source_rest_global"], dtype=np.float64))
    source_head_local = (
        np.einsum(
            "bij,bj->bi",
            source_inverse[:, :3, :3],
            np.asarray(source_rig["source_bone_head"], dtype=np.float64),
        )
        + source_inverse[:, :3, 3]
    )
    source_tail_local = (
        np.einsum(
            "bij,bj->bi",
            source_inverse[:, :3, :3],
            np.asarray(source_rig["source_bone_tail"], dtype=np.float64),
        )
        + source_inverse[:, :3, 3]
    )
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
        source_mesh_controller_bones=np.asarray(diag["source_mesh_controller_bones"], dtype=np.int16),
        source_mesh_material_groups=np.asarray(diag["source_mesh_material_groups"], dtype=object),
        source_mesh_roles=np.asarray(diag["source_mesh_roles"], dtype=object),
        source_fit_policies=np.asarray(diag["source_fit_policies"], dtype=object),
        source_driver_policies=np.asarray(diag["source_driver_policies"], dtype=object),
        source_compound_ids=np.asarray(diag["source_compound_ids"], dtype=object),
        source_sides=np.asarray(diag["source_sides"], dtype=object),
        source_landmarks_json=np.asarray(
            [json.dumps(list(values), ensure_ascii=True) for values in diag["source_landmarks"]],
            dtype=object,
        ),
        target_landmark_recipes=np.asarray(diag["target_landmark_recipes"], dtype=object),
        source_quality_profiles=np.asarray(diag["source_quality_profiles"], dtype=object),
        source_influence_offsets=np.asarray(diag["source_influence_offsets"], dtype=np.int64),
        source_influence_group_indices=np.asarray(
            diag["source_influence_group_indices"], dtype=np.int32
        ),
        source_influence_values=np.asarray(diag["source_influence_values"], dtype=np.float32),
        source_group_names=np.asarray(diag["source_group_names"], dtype=object),
        source_group_mesh_indices=np.asarray(diag["source_group_mesh_indices"], dtype=np.int32),
        source_group_local_indices=np.asarray(diag["source_group_local_indices"], dtype=np.int32),
        source_group_bone_indices=np.asarray(diag["source_group_bone_indices"], dtype=np.int32),
        driver_indices=np.asarray(diag["source_driver_indices"], dtype=np.int16),
        driver_weights=np.asarray(diag["source_driver_weights"], dtype=np.float32),
        runtime_driver_indices_compressed=np.asarray(
            diag["runtime_driver_indices_compressed"], dtype=np.int16
        ),
        runtime_driver_weights_compressed=np.asarray(
            diag["runtime_driver_weights_compressed"], dtype=np.float32
        ),
        rigid_component_ids=np.asarray(diag["rigid_component_ids"], dtype=np.int32),
        source_bone_names=np.asarray(source_rig["source_bone_names"], dtype=object),
        source_bone_parents=np.asarray(source_rig["source_bone_parents"], dtype=np.int16),
        source_bind_global=np.asarray(source_rig["source_rest_global"], dtype=np.float32),
        source_bind_local=np.asarray(source_rig["source_rest_local"], dtype=np.float32),
        source_bone_head_local=source_head_local.astype(np.float32),
        source_bone_tail_local=source_tail_local.astype(np.float32),
        source_bone_roll=np.asarray(source_rig["source_bone_roll"], dtype=np.float32),
        source_bone_use_connect=np.asarray(
            source_rig["source_bone_use_connect"], dtype=np.uint8
        ),
        source_bone_inherit_scale=np.asarray(
            source_rig["source_bone_inherit_scale"], dtype=np.uint8
        ),
        target_bind_global=np.asarray(source_rig["source_rest_global"], dtype=np.float32),
        target_bind_local=np.asarray(source_rig["source_rest_local"], dtype=np.float32),
        target_bone_head_local=source_head_local.astype(np.float32),
        target_bone_tail_local=source_tail_local.astype(np.float32),
        source_bone_smplx_a=np.asarray(source_rig["source_bone_smplx_a"], dtype=np.int16),
        source_bone_smplx_b=np.asarray(source_rig["source_bone_smplx_b"], dtype=np.int16),
        source_bone_blend=np.asarray(source_rig["source_bone_blend"], dtype=np.float32),
        source_bone_driver_types=np.asarray(source_rig["source_bone_driver_types"], dtype=object),
        source_bone_frame_joints=np.asarray(source_rig["source_bone_frame_joints"], dtype=np.int16),
        source_driver_coupling=source_driver_coupling,
        registration_reference=registration_reference,
        source_skin_vertices=skin_vertices,
        source_skin_faces=np.asarray(skin_faces, dtype=np.int32),
        source_skin_lbs_weights=np.asarray(skin_weights, dtype=np.float32),
        posed_vertices=np.zeros((0, 3), dtype=np.float32),
        pose_hash=np.asarray("", dtype=object),
        schema_version=np.asarray(6, dtype=np.int32),
        pose_format=np.asarray("smplx_body55_axis_angle", dtype=object),
        coordinate_system=np.asarray("smplx_y_up_m", dtype=object),
        metadata=np.asarray(
            {
                "mapping": str(args.mapping),
                "driver_index_space": "blender_source_bones",
                "driver_weights": "all_source_bone_influences_normalized",
                "source_influences": "authoritative_raw_blender_vertex_group_csr",
                "semantic_manifest": str(semantic_manifest_path),
                "semantic_manifest_version": int(semantic_manifest.version),
                "semantic_manifest_sha256": semantic_manifest.sha256,
            },
            dtype=object,
        ),
    )
    _export_glb(meshes, args.output_glb)
    scene_units = bpy.context.scene.unit_settings
    source_audit = {
        "blend_file": str(bpy.data.filepath),
        "coordinate_metadata": {
            "source_coordinate_transform": str(
                config.get("coordinate_transform", "unspecified")
            ),
            "configured_blender_unit_scale": float(
                config.get("blender_unit_scale", 1.0)
            ),
            "scene_unit_system": str(scene_units.system),
            "scene_unit_scale_length": float(scene_units.scale_length),
            "scene_length_unit": str(scene_units.length_unit),
            "armature_local_frame": True,
            "output_coordinate_system": "smplx_y_up_m",
        },
        "armature": {
            "object_name": str(arm.name),
            "data_name": str(arm.data.name),
            "object_transform": transform_audit(arm.matrix_world),
            "bone_count": int(len(arm.data.bones)),
            "bones": [
                {
                    "index": int(index),
                    "name": str(bone.name),
                    "parent": str(bone.parent.name) if bone.parent else None,
                    "use_deform": bool(bone.use_deform),
                    "roll": float(getattr(bone, "roll", 0.0)),
                    "use_connect": bool(bone.use_connect),
                    "inherit_scale": str(bone.inherit_scale),
                }
                for index, bone in enumerate(arm.data.bones)
            ],
        },
        "selection": {
            "selected_mesh_count": int(len(meshes)),
            "excluded_mesh_count": int(len(excluded_meshes)),
            "excluded_meshes": excluded_meshes,
        },
        "topology": {
            "mesh_count": int(len(meshes)),
            "vertices": int(sum(len(obj.data.vertices) for obj in meshes)),
            "edges": int(sum(len(obj.data.edges) for obj in meshes)),
            "polygons": int(sum(len(obj.data.polygons) for obj in meshes)),
            "triangles": int(len(faces)),
        },
        "weight_influences": diag["source_weight_audit"],
        "runtime_weight_view": {
            "authoritative": False,
            "all_influence_width": int(diag["source_driver_indices"].shape[1]),
            "compressed_view": diag["runtime_weight_compression_error"],
        },
        "blender_lbs_parity": {
            **diag["blender_parity"],
            "rms_error_m": float(diag["blender_parity"]["rms_error_source_units"])
            * float(rest_align["scale"]),
            "max_error_m": float(diag["blender_parity"]["max_error_source_units"])
            * float(rest_align["scale"]),
        },
        "semantic_manifest": {
            "path": str(semantic_manifest_path),
            "version": int(semantic_manifest.version),
            "sha256": semantic_manifest.sha256,
            "resolved_mesh_count": int(len(meshes)),
            "resolved_meshes": [
                resolved_semantics[str(obj.name)].to_dict() for obj in meshes
            ],
        },
        "meshes": diag["mesh_audit_records"],
        "pose_dependent_mechanisms": _source_mechanism_audit(arm, meshes),
    }
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
        "runtime_weight_compression_error": diag["runtime_weight_compression_error"],
        "source_audit": source_audit,
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
