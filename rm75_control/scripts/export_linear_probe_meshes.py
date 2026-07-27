"""Install linear_probe assets: visual DAE is a direct copy (no re-export).

The user supplies a Blender-decimated Collada at SRC_DAE. Do NOT decimate /
re-export the visual mesh (that was destroying outer faces). Only build a
low-poly collision STL for QP-IK CBF.

Run:
  /media/camp/EXT_DRIVE/blender/blender --background --python \\
    scripts/export_linear_probe_meshes.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import bpy  # type: ignore

REPO = Path(__file__).resolve().parents[1]
SRC_DAE = Path("/home/camp/Desktop/New Folder/trans/linear_probe.dae")
OUT_DIR = REPO / "rm75_control" / "assets" / "robots" / "rm75_6f_8dof" / "meshes"
OUT_VIS = OUT_DIR / "linear_probe.dae"
OUT_VIS_VIEWER = (
    REPO
    / "rm75_control"
    / "control"
    / "joint_admittance_8dof"
    / "assets"
    / "meshes"
    / "linear_probe.dae"
)
OUT_COL = OUT_DIR / "collision" / "linear_probe.stl"
MANIFEST = OUT_DIR / "collision" / "linear_probe_manifest.json"

EXCLUDE_EXACT = {"Cube", "link_8", "Camera", "Light", "Default"}
COLLISION_TARGET_TRIS = 500
DECIMATE_RATIO_FLOOR = 0.01


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def _tri_count(obj) -> int:
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def _decimate(obj, target_tris: int) -> int:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    for _ in range(8):
        tris = _tri_count(obj)
        if tris <= target_tris:
            return tris
        ratio = max(DECIMATE_RATIO_FLOOR, float(target_tris) / float(tris))
        mod = obj.modifiers.new(name="Decimate", type="DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = ratio
        mod.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=mod.name)
    return _tri_count(obj)


def _build_collision_stl() -> dict:
    _clear_scene()
    bpy.ops.wm.collada_import(filepath=str(SRC_DAE))
    keep = []
    for o in list(bpy.context.scene.objects):
        if o.type != "MESH" or o.name in EXCLUDE_EXACT or o.name.startswith("link_8"):
            bpy.data.objects.remove(o, do_unlink=True)
            continue
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        keep.append(o)
    if not keep:
        raise RuntimeError("no meshes for collision")

    bpy.ops.object.select_all(action="DESELECT")
    for o in keep:
        o.select_set(True)
    bpy.context.view_layer.objects.active = keep[0]
    if len(keep) > 1:
        bpy.ops.object.join()
    col = bpy.context.view_layer.objects.active
    col.data.materials.clear()
    before = _tri_count(col)
    after = _decimate(col, COLLISION_TARGET_TRIS)

    OUT_COL.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    col.select_set(True)
    bpy.context.view_layer.objects.active = col
    bpy.ops.wm.stl_export(
        filepath=str(OUT_COL),
        export_selected_objects=True,
        ascii_format=False,
    )
    return {"src_tris": before, "dst_tris": after, "bytes": OUT_COL.stat().st_size}


def main() -> int:
    if not SRC_DAE.is_file():
        raise FileNotFoundError(SRC_DAE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Visual: byte-for-byte copy of user DAE (no secondary Blender export).
    shutil.copy2(SRC_DAE, OUT_VIS)
    OUT_VIS_VIEWER.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC_DAE, OUT_VIS_VIEWER)

    col = _build_collision_stl()
    report = {
        "src": str(SRC_DAE),
        "visual_dae": str(OUT_VIS),
        "visual_viewer_copy": str(OUT_VIS_VIEWER),
        "visual_bytes": OUT_VIS.stat().st_size,
        "visual_note": "direct copy; no decimate / re-export",
        "collision_stl": str(OUT_COL),
        "collision": col,
    }
    MANIFEST.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
