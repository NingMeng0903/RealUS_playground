"""Inspect the anatomy Blender rig and write a JSON report.

This script runs inside Blender:
    blender -b anatomy.blend --python blender_rig_inspect.py -- --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy


def _argv_after_separator() -> list[str]:
    argv = list(sys.argv)
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return []


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-vertex-groups", type=int, default=256)
    return p.parse_args(_argv_after_separator())


def _collections_for_object(obj: bpy.types.Object) -> list[str]:
    out: list[str] = []
    for collection in bpy.data.collections:
        try:
            if obj.name in collection.objects:
                out.append(str(collection.name))
        except Exception:
            continue
    return sorted(out)


def _bone_record(bone: bpy.types.Bone) -> dict[str, Any]:
    return {
        "name": str(bone.name),
        "parent": str(bone.parent.name) if bone.parent else None,
        "children": [str(child.name) for child in bone.children],
        "head_local": [float(v) for v in bone.head_local],
        "tail_local": [float(v) for v in bone.tail_local],
        "roll": float(getattr(bone, "roll", 0.0)),
    }


def _constraint_record(constraint: Any) -> dict[str, Any]:
    target = getattr(constraint, "target", None)
    return {
        "name": str(constraint.name),
        "type": str(constraint.type),
        "target": str(target.name) if target is not None else None,
        "subtarget": str(getattr(constraint, "subtarget", "")),
        "influence": float(getattr(constraint, "influence", 1.0)),
        "mute": bool(getattr(constraint, "mute", False)),
    }


def _animation_record(owner: Any) -> dict[str, Any]:
    animation = getattr(owner, "animation_data", None)
    action = getattr(animation, "action", None) if animation is not None else None
    drivers = []
    for fcurve in list(getattr(animation, "drivers", ()) or ()):
        driver = getattr(fcurve, "driver", None)
        variables = []
        for variable in list(getattr(driver, "variables", ()) or ()):
            targets = []
            for target in list(getattr(variable, "targets", ()) or ()):
                target_id = getattr(target, "id", None)
                targets.append(
                    {
                        "id": str(target_id.name) if target_id is not None else None,
                        "bone_target": str(getattr(target, "bone_target", "")),
                        "data_path": str(getattr(target, "data_path", "")),
                        "transform_type": str(getattr(target, "transform_type", "")),
                        "transform_space": str(getattr(target, "transform_space", "")),
                    }
                )
            variables.append(
                {"name": str(variable.name), "type": str(variable.type), "targets": targets}
            )
        drivers.append(
            {
                "data_path": str(fcurve.data_path),
                "array_index": int(fcurve.array_index),
                "mute": bool(fcurve.mute),
                "expression": str(getattr(driver, "expression", "")),
                "type": str(getattr(driver, "type", "")),
                "variables": variables,
            }
        )
    return {
        "action": str(action.name) if action is not None else None,
        "driver_count": int(len(drivers)),
        "drivers": drivers,
    }


def _mesh_record(obj: bpy.types.Object, *, max_vertex_groups: int) -> dict[str, Any]:
    groups = [str(group.name) for group in obj.vertex_groups]
    shape_keys = getattr(obj.data, "shape_keys", None)
    key_blocks = []
    if shape_keys is not None:
        for key in shape_keys.key_blocks:
            key_blocks.append(
                {
                    "name": str(key.name),
                    "value": float(key.value),
                    "mute": bool(key.mute),
                    "relative_key": str(key.relative_key.name) if key.relative_key is not None else None,
                }
            )
    return {
        "name": str(obj.name),
        "collections": _collections_for_object(obj),
        "vertices": int(len(obj.data.vertices)),
        "faces": int(len(obj.data.polygons)),
        "modifiers": [
            {
                "name": str(mod.name),
                "type": str(mod.type),
                "show_viewport": bool(mod.show_viewport),
                "object": str(mod.object.name) if getattr(mod, "object", None) is not None else None,
                "use_deform_preserve_volume": bool(
                    getattr(mod, "use_deform_preserve_volume", False)
                ),
            }
            for mod in obj.modifiers
        ],
        "constraints": [_constraint_record(value) for value in obj.constraints],
        "animation": _animation_record(obj),
        "shape_keys": {
            "count": int(len(key_blocks)),
            "key_blocks": key_blocks,
            "animation": _animation_record(shape_keys) if shape_keys is not None else _animation_record(None),
        },
        "vertex_group_count": int(len(groups)),
        "vertex_groups": groups[: max(0, int(max_vertex_groups))],
    }


def inspect_scene(*, max_vertex_groups: int) -> dict[str, Any]:
    objects = list(bpy.data.objects)
    type_counts: dict[str, int] = {}
    for obj in objects:
        type_counts[str(obj.type)] = type_counts.get(str(obj.type), 0) + 1

    armatures = []
    for obj in objects:
        if obj.type != "ARMATURE":
            continue
        bones = [_bone_record(bone) for bone in obj.data.bones]
        armatures.append(
            {
                "name": str(obj.name),
                "bone_count": int(len(bones)),
                "bones": bones,
                "constraints": [_constraint_record(value) for value in obj.constraints],
                "pose_bone_constraints": {
                    str(pose_bone.name): [_constraint_record(value) for value in pose_bone.constraints]
                    for pose_bone in obj.pose.bones
                    if len(pose_bone.constraints)
                },
                "animation": _animation_record(obj),
            }
        )

    meshes = [
        _mesh_record(obj, max_vertex_groups=max_vertex_groups)
        for obj in objects
        if obj.type == "MESH"
    ]
    meshes.sort(key=lambda row: str(row["name"]))

    collections = []
    for collection in bpy.data.collections:
        collections.append(
            {
                "name": str(collection.name),
                "object_count": int(len(collection.objects)),
                "objects": [str(obj.name) for obj in collection.objects],
                "children": [str(child.name) for child in collection.children],
            }
        )
    collections.sort(key=lambda row: str(row["name"]))

    object_drivers = {
        str(obj.name): _animation_record(obj)
        for obj in objects
        if _animation_record(obj)["driver_count"] or _animation_record(obj)["action"]
    }
    return {
        "blend_file": str(bpy.data.filepath),
        "object_count": int(len(objects)),
        "types": type_counts,
        "collections": collections,
        "armatures": armatures,
        "meshes": meshes,
        "object_animation": object_drivers,
    }


def main() -> None:
    args = _parse_args()
    payload = inspect_scene(max_vertex_groups=int(args.max_vertex_groups))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"anatomy rig inspect written -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
