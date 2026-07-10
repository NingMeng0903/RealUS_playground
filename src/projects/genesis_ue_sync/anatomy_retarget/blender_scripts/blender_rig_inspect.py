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


def _mesh_record(obj: bpy.types.Object, *, max_vertex_groups: int) -> dict[str, Any]:
    groups = [str(group.name) for group in obj.vertex_groups]
    return {
        "name": str(obj.name),
        "collections": _collections_for_object(obj),
        "vertices": int(len(obj.data.vertices)),
        "faces": int(len(obj.data.polygons)),
        "modifiers": [{"name": str(mod.name), "type": str(mod.type)} for mod in obj.modifiers],
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

    return {
        "blend_file": str(bpy.data.filepath),
        "object_count": int(len(objects)),
        "types": type_counts,
        "collections": collections,
        "armatures": armatures,
        "meshes": meshes,
    }


def main() -> None:
    args = _parse_args()
    payload = inspect_scene(max_vertex_groups=int(args.max_vertex_groups))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"anatomy rig inspect written -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
