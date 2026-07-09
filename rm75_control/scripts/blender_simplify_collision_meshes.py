"""
Headless Blender script: decimate RM75 DAE collision meshes -> low-poly STL.

Run (from repo root):
  /media/camp/EXT_DRIVE/blender/blender --background --python scripts/blender_simplify_collision_meshes.py

Outputs: rm75_control/assets/robots/rm75_6f/meshes/collision/*.stl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Blender-only imports (script runs inside Blender Python).
import bpy  # type: ignore
import bmesh  # type: ignore

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "rm75_control" / "assets" / "robots" / "rm75_6f" / "meshes"
DST = SRC / "collision"
MANIFEST = DST / "manifest.json"

# Target triangle budget per link (collision hull; lower = faster FCL).
TARGET_TRIS = {
    "base_link": 300,
    "link_1": 150,
    "link_2": 200,
    "link_3": 150,
    "link_4": 200,
    "link_5": 150,
    "link_6": 200,
    "link_7": 250,
}
DEFAULT_TARGET = 200
DECIMATE_RATIO_FLOOR = 0.02


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)


def _tri_count(obj) -> int:
    mesh = obj.data
    return sum(len(p.vertices) - 2 for p in mesh.polygons)


def _join_meshes(objects: list) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    for o in objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def _decimate_to_budget(obj, target_tris: int) -> int:
    tris = _tri_count(obj)
    if tris <= target_tris:
        return tris
    ratio = max(DECIMATE_RATIO_FLOOR, target_tris / tris)
    mod = obj.modifiers.new(name="Decimate", type="DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    return _tri_count(obj)


def _import_dae(path: Path) -> list:
    _clear_scene()
    bpy.ops.wm.collada_import(filepath=str(path))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh in {path}")
    return meshes


def simplify_one(name: str) -> dict:
    src = SRC / f"{name}.dae"
    if not src.exists():
        raise FileNotFoundError(src)
    target = TARGET_TRIS.get(name, DEFAULT_TARGET)
    meshes = _import_dae(src)
    obj = meshes[0] if len(meshes) == 1 else _join_meshes(meshes)
    src_tris = _tri_count(obj)
    out_tris = _decimate_to_budget(obj, target)

    DST.mkdir(parents=True, exist_ok=True)
    out = DST / f"{name}.stl"
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.stl_export(
        filepath=str(out),
        export_selected_objects=True,
        ascii_format=False,
    )
    return {
        "name": name,
        "src": str(src),
        "dst": str(out),
        "src_tris": src_tris,
        "dst_tris": out_tris,
        "target_tris": target,
        "size_bytes": out.stat().st_size,
    }


def main() -> int:
    names = ["base_link"] + [f"link_{i}" for i in range(1, 8)]
    report = []
    for name in names:
        info = simplify_one(name)
        report.append(info)
        print(
            f"{name}: {info['src_tris']} -> {info['dst_tris']} tris "
            f"({info['size_bytes'] // 1024} KiB) -> {info['dst']}",
            flush=True,
        )
    MANIFEST.write_text(json.dumps(report, indent=2))
    print(f"manifest: {MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
