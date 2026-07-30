"""Install probe45 assets: visual DAE is a direct copy (no re-export).

The user supplies a Blender-decimated Collada at SRC_DAE. Do NOT decimate /
re-export the visual mesh (that was destroying outer faces). Only build a
low-poly collision STL for QP-IK CBF.

Colors: after copy, retune Collada diffuse to match the linear_probe palette
(converter / blue parts → soft blue).

Run:
  /media/camp/EXT_DRIVE/blender/blender --background --python \\
    scripts/export_linear_probe_meshes.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import bpy  # type: ignore

REPO = Path(__file__).resolve().parents[1]
SRC_DAE = Path("/home/camp/Desktop/New Folder/trans/probe45.dae")
OUT_DIR = REPO / "rm75_control" / "assets" / "robots" / "rm75_6f_8dof" / "meshes"
OUT_VIS = OUT_DIR / "probe45.dae"
OUT_VIS_VIEWER = (
    REPO
    / "rm75_control"
    / "control"
    / "joint_admittance_8dof"
    / "assets"
    / "meshes"
    / "probe45.dae"
)
OUT_COL = OUT_DIR / "collision" / "probe45.stl"
MANIFEST = OUT_DIR / "collision" / "probe45_manifest.json"

EXCLUDE_EXACT = {"Cube", "link_8", "Camera", "Light", "Default"}
COLLISION_TARGET_TRIS = 500
DECIMATE_RATIO_FLOOR = 0.01

# Match previous linear_probe.dae tuned palette; blues include 40° converter.
COLOR_BY_MAT_SUBSTR = [
    ("Hard_Textured_Plastic_Blue__2_001", "0.55 0.72 0.90 1"),
    ("Hard_Textured_Plastic_Blue__2", "0.55 0.72 0.90 1"),
    ("Hard_Textured_Plastic_Red__2_001", "0.90 0.55 0.32 1"),
    ("Hard_Textured_Plastic_Red__2", "0.92 0.62 0.38 1"),
    ("Hard_Textured_Plastic_Black", "0.24 0.24 0.24 1"),
    ("Hard_Rough_Plastic_Grey", "0.82 0.825 0.835 1"),
    ("Hard_Rough_Plastic_White", "1 1 1 1"),
    ("Glass_Light_Frost_Grey", "1 1 1 1"),
    # Camera = Anodized Aluminum (mesh_011); slightly deeper gray than probe Grey.
    ("Anodized_Aluminum_Brushed_90", "0.70 0.70 0.72 1"),
]


def _retune_dae_colors(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    mat_eff = dict(
        re.findall(
            r'<material[^>]*id="([^"]+)"[^>]*>\s*<instance_effect url="#([^"]+)"',
            text,
        )
    )
    eff_color: dict[str, str] = {}
    for mid, eid in mat_eff.items():
        for key, rgba in COLOR_BY_MAT_SUBSTR:
            if key in mid:
                eff_color[eid] = rgba
                break

    def repl_effect(m: re.Match[str]) -> str:
        full = m.group(0)
        eid = m.group(1)
        if eid not in eff_color:
            return full
        rgba = eff_color[eid]
        full2, _n = re.subn(
            r"(<diffuse>\s*<color[^>]*>)[^<]+(</color>)",
            rf"\g<1>{rgba}\2",
            full,
            count=1,
        )
        return full2

    text2 = re.sub(r'<effect id="([^"]+)"[^>]*>[\s\S]*?</effect>', repl_effect, text)
    path.write_text(text2, encoding="utf-8")


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
    # Visual: copy then retune diffuse (do not Blender-reexport visual).
    shutil.copy2(SRC_DAE, OUT_VIS)
    _retune_dae_colors(OUT_VIS)
    OUT_VIS_VIEWER.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_VIS, OUT_VIS_VIEWER)

    col = _build_collision_stl()
    report = {
        "src": str(SRC_DAE),
        "visual_dae": str(OUT_VIS),
        "collision_stl": str(OUT_COL),
        **col,
    }
    MANIFEST.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"export_linear_probe_meshes failed: {exc}", file=sys.stderr)
        raise
