"""Bake immutable Blender Action/FK evidence for SourceOperatorV7.

Run inside Blender.  The result is diagnostic/oracle data only; runtime
materialization and pose application never read a .blend file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import bpy
import numpy as np


DEFAULT_MESHES = (
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


def _args() -> argparse.Namespace:
    argv = list(sys.argv)
    tail = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--mesh", action="append", default=[])
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
    candidates = [
        obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"
    ]
    if not candidates:
        raise RuntimeError("source blend contains no armature")
    return max(candidates, key=lambda obj: len(obj.data.bones))


def _mesh_world(
    obj: bpy.types.Object, depsgraph: bpy.types.Depsgraph
) -> tuple[np.ndarray, np.ndarray]:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        world = np.asarray(evaluated.matrix_world, dtype=np.float64)
        local = np.asarray([vertex.co[:] for vertex in mesh.vertices], dtype=np.float64)
        vertices = (
            np.einsum("ij,nj->ni", world[:3, :3], local)
            + world[:3, 3]
        )
        faces = np.asarray(
            [polygon.vertices[:] for polygon in mesh.polygons if len(polygon.vertices) == 3],
            dtype=np.int32,
        )
    finally:
        evaluated.to_mesh_clear()
    return vertices.astype(np.float32), faces


def main() -> None:
    args = _args()
    armature = _armature()
    action = (
        None
        if armature.animation_data is None
        else armature.animation_data.action
    )
    if action is None:
        raise RuntimeError("source armature has no active Action")
    first = int(np.floor(float(action.frame_range[0])))
    last = int(np.ceil(float(action.frame_range[1])))
    step = max(1, int(args.frame_step))
    frames = np.arange(first, last + 1, step, dtype=np.int32)
    if frames[-1] != last:
        frames = np.concatenate((frames, np.asarray((last,), dtype=np.int32)))

    bones = list(armature.data.bones)
    bone_names = [str(bone.name) for bone in bones]
    bone_index = {name: index for index, name in enumerate(bone_names)}
    parents = np.asarray(
        [
            -1 if bone.parent is None else bone_index[str(bone.parent.name)]
            for bone in bones
        ],
        dtype=np.int32,
    )
    rest_global = np.asarray(
        [np.asarray(bone.matrix_local, dtype=np.float64) for bone in bones],
        dtype=np.float32,
    )
    rest_local = rest_global.astype(np.float64).copy()
    for index, parent in enumerate(parents.tolist()):
        if int(parent) >= 0:
            rest_local[index] = (
                np.linalg.inv(rest_global[int(parent)]) @ rest_global[index]
            )
    rest_local = rest_local.astype(np.float32)

    requested = tuple(args.mesh or DEFAULT_MESHES)
    objects = []
    for name in requested:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            raise RuntimeError(f"required oracle mesh {name!r} is missing")
        objects.append(obj)

    topology: dict[str, dict[str, object]] = {}
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(7, dtype=np.int32),
        "frames": frames,
        "bone_names": np.asarray(bone_names, dtype=f"<U{max(map(len, bone_names))}"),
        "bone_parents": parents,
        "bone_rest_global": rest_global,
        "bone_rest_local": rest_local,
    }
    rest_meshes: dict[str, np.ndarray] = {}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bpy.context.scene.frame_set(int(frames[0]))
    for obj in objects:
        vertices, faces = _mesh_world(obj, depsgraph)
        name = str(obj.name)
        rest_meshes[name] = vertices
        payload[f"mesh__{_key(name)}__faces"] = faces
        topology[name] = {
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "topology_sha256": _array_hash(
                np.asarray(len(vertices), dtype=np.int64), faces
            ),
        }

    bone_global = np.empty(
        (len(frames), len(bones), 4, 4), dtype=np.float32
    )
    bone_local = np.empty_like(bone_global)
    mesh_sweeps = {
        str(obj.name): np.empty(
            (len(frames), len(rest_meshes[str(obj.name)]), 3),
            dtype=np.float32,
        )
        for obj in objects
    }
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
        for index, parent in enumerate(parents.tolist()):
            if int(parent) < 0:
                bone_local[frame_index, index] = bone_global[
                    frame_index, index
                ]
            else:
                bone_local[frame_index, index] = (
                    np.linalg.inv(
                        bone_global[frame_index, int(parent)].astype(np.float64)
                    )
                    @ bone_global[frame_index, index].astype(np.float64)
                ).astype(np.float32)
        for obj in objects:
            vertices, faces = _mesh_world(obj, depsgraph)
            name = str(obj.name)
            expected_faces = payload[f"mesh__{_key(name)}__faces"]
            if (
                vertices.shape != mesh_sweeps[name][frame_index].shape
                or not np.array_equal(faces, expected_faces)
            ):
                raise RuntimeError(
                    f"evaluated topology changed for {name} at frame {frame}"
                )
            mesh_sweeps[name][frame_index] = vertices

    payload["bone_action_global"] = bone_global
    payload["bone_action_local"] = bone_local
    for name, values in mesh_sweeps.items():
        payload[f"mesh__{_key(name)}__vertices"] = values

    blend_path = Path(bpy.data.filepath).resolve()
    action_summary = {
        "name": str(action.name),
        "frame_range": [first, last],
        "sampled_frames": frames.tolist(),
        "fcurve_count": int(len(action.fcurves)),
    }
    report = {
        "schema_version": 7,
        "artifact_kind": "BlenderActionOracleV7",
        "source_blend": str(blend_path),
        "source_blend_sha256": _sha256(blend_path),
        "source_commit": str(args.source_commit),
        "blender_version": str(bpy.app.version_string),
        "armature": str(armature.name),
        "bone_count": int(len(bones)),
        "hierarchy_sha256": _array_hash(parents, rest_local),
        "action": action_summary,
        "meshes": topology,
    }
    payload["report_json"] = np.asarray(
        json.dumps(report, sort_keys=True, separators=(",", ":"))
    )
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **payload)
    report["artifact_sha256"] = _sha256(args.output_npz)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"BlenderActionOracleV7 -> {args.output_npz}")


if __name__ == "__main__":
    main()
