"""Bake immutable Blender Action/FK/LBS evidence for SourceOperatorV7.

Run inside Blender.  This is an oracle only: it records the evaluated source
Armature and base-topology meshes, and is never imported by runtime retargeting.
Non-Armature modifiers are disabled while sampling so vertex IDs remain those
of the frozen source material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


KEY_BONE_MESHES = (
    "Ilium_L",
    "Ilium_R",
    "Femur_L",
    "Femur_R",
    "Tibia_L",
    "Tibia_R",
    "Patella_L",
    "Patella_R",
    "Humerus_L",
    "Humerus_R",
    "Radius_L",
    "Radius_R",
    "Ulna_L",
    "Ulna_R",
)
TUBE_MESHES = (
    "Artery",
    "Vein",
    "Autonomic",
    "Cervical_Nerves_L",
    "Cervical_Nerves_R",
    "Coccygeal_Nerve_L",
    "Coccygeal_Nerve_R",
    "Facial_Nerves_L",
    "Facial_Nerves_R",
    "Lumbar_Nerves_L",
    "Lumbar_Nerves_R",
    "Optic_Chiasm",
    "Sacral_Nerves_L",
    "Sacral_Nerves_R",
    "Spinal_Cord",
    "Thoracic_Nerves_L",
    "Thoracic_Nerves_R",
)
DEFAULT_MESHES = KEY_BONE_MESHES + TUBE_MESHES
INFLUENCE_SLOTS = 14


def _args() -> argparse.Namespace:
    argv = list(sys.argv)
    tail = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--mesh-frame-step",
        type=int,
        default=15,
        help="Blender geometry parity cadence; FK/basis matrices still use --frame-step",
    )
    parser.add_argument("--mesh", action="append", default=[])
    parser.add_argument("--unit-scale-m", type=float, default=0.01)
    return parser.parse_args(tail)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def _armature() -> bpy.types.Object:
    candidates = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not candidates:
        raise RuntimeError("source blend contains no armature")
    return max(candidates, key=lambda obj: len(obj.data.bones))


def _triangles(mesh: bpy.types.Mesh) -> np.ndarray:
    # Match the frozen SourceOperator exporter exactly.  Blender's
    # loop-triangle tessellator may flip a non-planar quad's diagonal as the
    # mesh poses; the source contract instead uses a deterministic polygon fan.
    faces: list[tuple[int, int, int]] = []
    for polygon in mesh.polygons:
        values = [int(value) for value in polygon.vertices]
        if len(values) < 3:
            continue
        faces.extend(
            (values[0], values[index], values[index + 1])
            for index in range(1, len(values) - 1)
        )
    return np.asarray(faces, dtype=np.int32).reshape(-1, 3)


def _polygon_topology(mesh: bpy.types.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return topology before pose-dependent tessellation of non-planar quads."""

    offsets = np.zeros(len(mesh.polygons) + 1, dtype=np.int64)
    values: list[int] = []
    for polygon_index, polygon in enumerate(mesh.polygons):
        values.extend(int(value) for value in polygon.vertices)
        offsets[polygon_index + 1] = len(values)
    return offsets, np.asarray(values, dtype=np.int32)


def _base_vertices_armature(
    obj: bpy.types.Object, armature: bpy.types.Object
) -> tuple[np.ndarray, np.ndarray]:
    object_to_armature = (
        np.linalg.inv(np.asarray(armature.matrix_world, dtype=np.float64))
        @ np.asarray(obj.matrix_world, dtype=np.float64)
    )
    local = np.asarray([vertex.co[:] for vertex in obj.data.vertices], dtype=np.float64)
    points = (
        np.einsum("ij,nj->ni", object_to_armature[:3, :3], local)
        + object_to_armature[:3, 3]
    )
    return points.astype(np.float32), object_to_armature.astype(np.float64)


def _evaluated_vertices_armature(
    obj: bpy.types.Object,
    armature: bpy.types.Object,
    depsgraph: bpy.types.Depsgraph,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        evaluated_to_armature = (
            np.linalg.inv(np.asarray(armature.matrix_world, dtype=np.float64))
            @ np.asarray(evaluated.matrix_world, dtype=np.float64)
        )
        local = np.asarray(
            [vertex.co[:] for vertex in mesh.vertices], dtype=np.float64
        )
        vertices = (
            np.einsum("ij,nj->ni", evaluated_to_armature[:3, :3], local)
            + evaluated_to_armature[:3, 3]
        )
        polygon_offsets, polygon_indices = _polygon_topology(mesh)
    finally:
        evaluated.to_mesh_clear()
    return vertices.astype(np.float32), polygon_offsets, polygon_indices


def _packed_weights(
    obj: bpy.types.Object,
    *,
    bone_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    group_names = {int(group.index): str(group.name) for group in obj.vertex_groups}
    count = len(obj.data.vertices)
    indices = np.zeros((count, INFLUENCE_SLOTS), dtype=np.int16)
    weights = np.zeros((count, INFLUENCE_SLOTS), dtype=np.float32)
    offsets = np.zeros(count + 1, dtype=np.int64)
    csr_bones: list[int] = []
    csr_values: list[float] = []
    maximum = 0
    raw_sums = np.zeros(count, dtype=np.float64)
    for vertex_index, vertex in enumerate(obj.data.vertices):
        combined: dict[int, float] = {}
        for element in vertex.groups:
            name = group_names.get(int(element.group), "")
            bone = bone_index.get(name)
            value = float(element.weight)
            if bone is None or value <= 0.0:
                continue
            combined[int(bone)] = combined.get(int(bone), 0.0) + value
        # Keep the frozen SourceOperator ordering: source_audit.py merges a
        # vertex's groups and then sorts by bone index.  Weight-ordering is
        # mathematically equivalent for LBS, but would break the immutable
        # 14-slot/CSR topology contract used by the offline checker.
        ordered = sorted(combined.items())
        total = float(sum(value for _bone, value in ordered))
        if total <= 1.0e-12:
            raise RuntimeError(
                f"{obj.name}: vertex {vertex_index} has no Armature vertex-group weight"
            )
        raw_sums[vertex_index] = total
        maximum = max(maximum, len(ordered))
        if len(ordered) > INFLUENCE_SLOTS:
            raise RuntimeError(
                f"{obj.name}: vertex {vertex_index} has {len(ordered)} influences; "
                f"oracle contract allows {INFLUENCE_SLOTS}"
            )
        for slot, (bone, raw_value) in enumerate(ordered):
            value = raw_value / total
            indices[vertex_index, slot] = int(bone)
            weights[vertex_index, slot] = float(value)
            csr_bones.append(int(bone))
            csr_values.append(float(raw_value))
        offsets[vertex_index + 1] = len(csr_bones)
    return (
        indices,
        weights,
        offsets,
        np.asarray(csr_bones, dtype=np.int16),
        {
            "maximum_influences": int(maximum),
            "influence_count": int(len(csr_bones)),
            "weight_sum_max_error": float(
                np.max(np.abs(np.sum(weights, axis=1) - 1.0))
            ),
            "raw_weight_sum_min": float(np.min(raw_sums)),
            "raw_weight_sum_max": float(np.max(raw_sums)),
            "raw_weight_sum_max_error": float(
                np.max(np.abs(raw_sums - 1.0))
            ),
            "csr_values": np.asarray(csr_values, dtype=np.float32),
        },
    )


def _modifier_record(obj: bpy.types.Object) -> list[dict[str, object]]:
    return [
        {
            "name": str(modifier.name),
            "type": str(modifier.type),
            "show_viewport": bool(modifier.show_viewport),
            "armature": (
                None
                if getattr(modifier, "object", None) is None
                else str(modifier.object.name)
            ),
            "use_vertex_groups": bool(
                getattr(modifier, "use_vertex_groups", False)
            ),
            "use_bone_envelopes": bool(
                getattr(modifier, "use_bone_envelopes", False)
            ),
            "use_deform_preserve_volume": bool(
                getattr(modifier, "use_deform_preserve_volume", False)
            ),
            "use_multi_modifier": bool(
                getattr(modifier, "use_multi_modifier", False)
            ),
        }
        for modifier in obj.modifiers
    ]


def _action_fcurves(action: bpy.types.Action) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for curve in action.fcurves:
        result.append(
            {
                "data_path": str(curve.data_path),
                "array_index": int(curve.array_index),
                "extrapolation": str(curve.extrapolation),
                "keyframes": [
                    {
                        "co": [float(point.co[0]), float(point.co[1])],
                        "interpolation": str(point.interpolation),
                    }
                    for point in curve.keyframe_points
                ],
            }
        )
    return result


def main() -> None:
    args = _args()
    started = time.perf_counter()
    armature = _armature()
    action = None if armature.animation_data is None else armature.animation_data.action
    if action is None:
        raise RuntimeError("source armature has no active Action")
    first = int(np.floor(float(action.frame_range[0])))
    last = int(np.ceil(float(action.frame_range[1])))
    step = max(1, int(args.frame_step))
    frames = np.arange(first, last + 1, step, dtype=np.int32)
    if int(frames[-1]) != last:
        frames = np.concatenate((frames, np.asarray((last,), dtype=np.int32)))
    mesh_step = max(1, int(args.mesh_frame_step))
    mesh_frames = frames[((frames - first) % mesh_step) == 0]
    frame_values = set(int(value) for value in frames.tolist())
    key_mesh_frames = np.asarray(
        [value for value in (250, 260) if value in frame_values], dtype=np.int32
    )
    if len(key_mesh_frames):
        mesh_frames = np.unique(np.concatenate((mesh_frames, key_mesh_frames)))
    if not len(mesh_frames) or int(mesh_frames[-1]) != last:
        mesh_frames = np.concatenate(
            (mesh_frames, np.asarray((last,), dtype=np.int32))
        )
    requested = tuple(args.mesh or DEFAULT_MESHES)
    if len(requested) != len(set(requested)):
        raise RuntimeError("oracle mesh list contains duplicates")
    objects: list[bpy.types.Object] = []
    for name in requested:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            raise RuntimeError(f"required oracle mesh {name!r} is missing")
        objects.append(obj)

    bones = list(armature.data.bones)
    bone_names = [str(bone.name) for bone in bones]
    bone_index = {name: index for index, name in enumerate(bone_names)}
    parents = np.asarray(
        [-1 if bone.parent is None else bone_index[str(bone.parent.name)] for bone in bones],
        dtype=np.int32,
    )
    if len(bones) != 235 or any(not bool(bone.use_deform) for bone in bones):
        raise RuntimeError("oracle requires the frozen 235/235 deform-bone rig")
    rest_global = np.asarray(
        [np.asarray(bone.matrix_local, dtype=np.float64) for bone in bones],
        dtype=np.float32,
    )
    rest_local = rest_global.astype(np.float64).copy()
    for index, parent in enumerate(parents.tolist()):
        if parent >= 0:
            rest_local[index] = np.linalg.inv(rest_global[parent]) @ rest_global[index]
    rest_local = rest_local.astype(np.float32)

    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(71, dtype=np.int32),
        "frames": frames,
        "mesh_frames": mesh_frames,
        "bone_names": np.asarray(bone_names, dtype=f"<U{max(map(len, bone_names))}"),
        "bone_parents": parents,
        "bone_rest_global": rest_global,
        "bone_rest_local": rest_local,
        "mesh_names": np.asarray(requested, dtype=f"<U{max(map(len, requested))}"),
        "unit_scale_m": np.asarray(float(args.unit_scale_m), dtype=np.float64),
        "coordinate_space": np.asarray("armature_object_space", dtype="<U32"),
        "matrix_convention": np.asarray("column_vector_left_multiply", dtype="<U32"),
        "armature_matrix_world": np.asarray(armature.matrix_world, dtype=np.float64),
    }
    topology: dict[str, dict[str, object]] = {}
    modifiers: dict[str, list[dict[str, object]]] = {}
    mesh_sweeps: dict[str, np.ndarray] = {}
    base_faces: dict[str, np.ndarray] = {}
    base_polygons: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    modifier_visibility: list[tuple[bpy.types.Modifier, bool]] = []
    object_visibility: list[tuple[bpy.types.Object, bool]] = []
    for obj in objects:
        armature_modifiers = [modifier for modifier in obj.modifiers if modifier.type == "ARMATURE"]
        if len(armature_modifiers) != 1:
            raise RuntimeError(f"{obj.name}: expected exactly one Armature modifier")
        modifier = armature_modifiers[0]
        if (
            modifier.object != armature
            or not bool(modifier.use_vertex_groups)
            or bool(modifier.use_bone_envelopes)
            or bool(modifier.use_deform_preserve_volume)
            or bool(modifier.use_multi_modifier)
        ):
            raise RuntimeError(f"{obj.name}: Armature modifier is outside oracle contract")
        modifiers[str(obj.name)] = _modifier_record(obj)
        object_visibility.append((obj, bool(obj.hide_viewport)))
        obj.hide_viewport = False
        for item in obj.modifiers:
            if item.type != "ARMATURE":
                modifier_visibility.append((item, bool(item.show_viewport)))
                item.show_viewport = False
        name = str(obj.name)
        key = _key(name)
        rest_vertices, object_to_armature = _base_vertices_armature(obj, armature)
        faces = _triangles(obj.data)
        indices, weights, offsets, csr_bones, weight_report = _packed_weights(
            obj, bone_index=bone_index
        )
        csr_values = np.asarray(weight_report.pop("csr_values"), dtype=np.float32)
        payload[f"mesh__{key}__rest_vertices"] = rest_vertices
        payload[f"mesh__{key}__faces"] = faces
        payload[f"mesh__{key}__object_to_armature"] = object_to_armature
        payload[f"mesh__{key}__driver_indices"] = indices
        payload[f"mesh__{key}__driver_weights"] = weights
        payload[f"mesh__{key}__source_offsets"] = offsets
        payload[f"mesh__{key}__source_bone_indices"] = csr_bones
        payload[f"mesh__{key}__source_values"] = csr_values
        base_faces[name] = faces
        base_polygons[name] = _polygon_topology(obj.data)
        mesh_sweeps[name] = np.empty(
            (len(mesh_frames), len(rest_vertices), 3), dtype=np.float32
        )
        topology[name] = {
            "vertex_count": int(len(rest_vertices)),
            "face_count": int(len(faces)),
            "topology_sha256": _array_hash(
                np.asarray(len(rest_vertices), dtype=np.int64), faces
            ),
            "weight_sha256": _array_hash(indices, weights),
            **weight_report,
        }

    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bone_global = np.empty((len(frames), len(bones), 4, 4), dtype=np.float32)
    bone_local = np.empty_like(bone_global)
    bone_basis = np.empty_like(bone_global)
    neutral_global = np.empty((len(bones), 4, 4), dtype=np.float32)
    bind_meshes: dict[str, np.ndarray] = {}
    mesh_frame_indices = {
        int(frame): index for index, frame in enumerate(mesh_frames.tolist())
    }
    try:
        for frame_index, frame in enumerate(frames.tolist()):
            bpy.context.scene.frame_set(int(frame))
            depsgraph.update()
            for index, name in enumerate(bone_names):
                pose_bone = armature.pose.bones.get(name)
                if pose_bone is None:
                    raise RuntimeError(f"pose bone {name!r} disappeared")
                bone_global[frame_index, index] = np.asarray(
                    pose_bone.matrix, dtype=np.float32
                )
                bone_basis[frame_index, index] = np.asarray(
                    pose_bone.matrix_basis, dtype=np.float32
                )
            for index, parent in enumerate(parents.tolist()):
                bone_local[frame_index, index] = (
                    bone_global[frame_index, index]
                    if parent < 0
                    else (
                        np.linalg.inv(
                            bone_global[frame_index, parent].astype(np.float64)
                        )
                        @ bone_global[frame_index, index].astype(np.float64)
                    ).astype(np.float32)
                )
            mesh_frame_index = mesh_frame_indices.get(int(frame))
            if mesh_frame_index is None:
                continue
            for obj in objects:
                vertices, polygon_offsets, polygon_indices = _evaluated_vertices_armature(
                    obj, armature, depsgraph
                )
                name = str(obj.name)
                base_offsets, base_indices = base_polygons[name]
                if (
                    vertices.shape != mesh_sweeps[name][mesh_frame_index].shape
                    or not np.array_equal(polygon_offsets, base_offsets)
                    or not np.array_equal(polygon_indices, base_indices)
                ):
                    raise RuntimeError(
                        f"{name}: Armature-only topology changed at frame {frame}; "
                        f"vertices={vertices.shape}/{mesh_sweeps[name][mesh_frame_index].shape} "
                        f"polygons={len(polygon_offsets) - 1}/{len(base_offsets) - 1}"
                    )
                mesh_sweeps[name][mesh_frame_index] = vertices
        # Action frames are authored examples and do not contain the actual
        # armature rest state.  Evaluate one explicit identity-basis sample so
        # the offline zero-pose contract can be checked independently.
        original_action = armature.animation_data.action
        original_frame = int(bpy.context.scene.frame_current)
        armature.animation_data.action = None
        for pose_bone in armature.pose.bones:
            pose_bone.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()
        depsgraph.update()
        for index, name in enumerate(bone_names):
            neutral_global[index] = np.asarray(
                armature.pose.bones[name].matrix, dtype=np.float32
            )
        for obj in objects:
            vertices, polygon_offsets, polygon_indices = _evaluated_vertices_armature(
                obj, armature, depsgraph
            )
            name = str(obj.name)
            base_offsets, base_indices = base_polygons[name]
            if (
                not np.array_equal(polygon_offsets, base_offsets)
                or not np.array_equal(polygon_indices, base_indices)
            ):
                raise RuntimeError(f"{name}: neutral Armature topology changed")
            bind_meshes[name] = vertices
        armature.animation_data.action = original_action
        bpy.context.scene.frame_set(original_frame)
        bpy.context.view_layer.update()
    finally:
        for modifier, visible in modifier_visibility:
            modifier.show_viewport = bool(visible)
        for obj, hidden in object_visibility:
            obj.hide_viewport = bool(hidden)
        bpy.context.view_layer.update()

    payload["bone_action_global"] = bone_global
    payload["bone_action_local"] = bone_local
    payload["bone_action_basis"] = bone_basis
    payload["bone_neutral_global"] = neutral_global
    for name, values in mesh_sweeps.items():
        payload[f"mesh__{_key(name)}__vertices"] = values
        # The neutral evaluated mesh is Blender's actual Armature bind geometry.
        # It is intentionally distinct from raw object-data coordinates for
        # Artery/Vein, whose object and Armature transforms use different frames.
        payload[f"mesh__{_key(name)}__bind_vertices"] = bind_meshes[name]

    blend_path = Path(bpy.data.filepath).resolve()
    unit_scale = float(args.unit_scale_m)
    if not np.isfinite(unit_scale) or unit_scale <= 0.0:
        raise RuntimeError("unit scale must be finite and positive")
    report = {
        "schema_version": 71,
        "artifact_kind": "BlenderLinkOracleV7",
        "source_blend": str(blend_path),
        "source_blend_sha256": _sha256(blend_path),
        "source_commit": str(args.source_commit),
        "blender_version": str(bpy.app.version_string),
        "coordinate_space": "armature_object_space",
        "matrix_convention": "column_vector_left_multiply",
        "unit_scale_m": unit_scale,
        "armature": str(armature.name),
        "armature_matrix_world": np.asarray(
            armature.matrix_world, dtype=np.float64
        ).tolist(),
        "bone_count": int(len(bones)),
        "deform_bone_count": int(sum(bool(bone.use_deform) for bone in bones)),
        "hierarchy_sha256": _array_hash(parents, rest_local),
        "action": {
            "name": str(action.name),
            "frame_range": [first, last],
            "sampled_frames": frames.tolist(),
            "mesh_sampled_frames": mesh_frames.tolist(),
            "fcurve_count": int(len(action.fcurves)),
            "fcurves": _action_fcurves(action),
        },
        "selection": {
            "key_bone_meshes": [name for name in requested if name in KEY_BONE_MESHES],
            "tube_meshes": [name for name in requested if name in TUBE_MESHES],
            "mesh_count": int(len(requested)),
            "tube_mesh_count": int(sum(name in TUBE_MESHES for name in requested)),
            "tube_vertex_count": int(
                sum(topology[name]["vertex_count"] for name in requested if name in TUBE_MESHES)
            ),
        },
        "modifiers": modifiers,
        "meshes": topology,
        "runtime_dependency": False,
        "retarget_logic_included": False,
        "smplx_mapping_available": False,
        "elapsed_seconds_before_write": float(time.perf_counter() - started),
    }
    payload["report_json"] = np.asarray(
        json.dumps(report, sort_keys=True, separators=(",", ":"))
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **payload)
    report["artifact_sha256"] = _sha256(args.output_npz)
    report["artifact_bytes"] = int(args.output_npz.stat().st_size)
    report["elapsed_seconds"] = float(time.perf_counter() - started)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "BlenderLinkOracleV7 "
        f"frames={len(frames)} mesh_frames={len(mesh_frames)} "
        f"bones={len(bones)} meshes={len(requested)} "
        f"tubes={report['selection']['tube_vertex_count']} -> {args.output_npz}"
    )


if __name__ == "__main__":
    main()
