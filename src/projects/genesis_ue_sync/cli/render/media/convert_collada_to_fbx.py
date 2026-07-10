#!/usr/bin/env python3
"""Convert COLLADA visual meshes to FBX once, using Blender in background mode."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _blender_bin() -> str:
    raw = os.environ.get("AMONGUS_BLENDER_BIN", "").strip()
    if raw:
        return raw
    candidate = shutil.which("blender")
    if candidate is None:
        raise RuntimeError(
            "Blender executable not found. Set AMONGUS_BLENDER_BIN to enable offline DAE->FBX conversion."
        )
    return candidate


def _blender_driver_script() -> str:
    return r'''
import argparse
import sys
from pathlib import Path

import bpy


def _clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def _mirror_y_for_unreal_link_local(obj) -> None:
    mesh = obj.data
    for vertex in mesh.vertices:
        vertex.co.y *= -1.0
    mesh.update()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.mode_set(mode="OBJECT")


def _convert_one(
    input_path: Path,
    output_path: Path,
    *,
    global_scale: float,
    axis_forward: str,
    axis_up: str,
    mirror_y_for_unreal: bool,
) -> None:
    _clear_scene()
    bpy.ops.wm.collada_import(filepath=str(input_path))
    meshes = _mesh_objects()
    if not meshes:
        raise RuntimeError(f"No mesh objects imported from {input_path}")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    meshes = _mesh_objects()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    obj.name = input_path.stem
    obj.data.name = input_path.stem
    obj.location = (0.0, 0.0, 0.0)
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)
    if mirror_y_for_unreal:
        _mirror_y_for_unreal_link_local(obj)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.fbx(
        filepath=str(output_path),
        use_selection=True,
        object_types={"MESH"},
        apply_unit_scale=False,
        global_scale=float(global_scale),
        axis_forward=str(axis_forward),
        axis_up=str(axis_up),
        bake_space_transform=False,
        add_leaf_bones=False,
        use_mesh_modifiers=True,
        mesh_smooth_type="FACE",
        use_tspace=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--axis-forward", default="X")
    parser.add_argument("--axis-up", default="Z")
    parser.add_argument("--mirror-y-for-unreal", action="store_true")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    # Use `--opt=value` so values like "-Z" are not parsed as separate argv flags.
    normalized: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--axis-forward") or tok.startswith("--axis-up") or tok.startswith("--global-scale"):
            if "=" in tok:
                normalized.append(tok)
                i += 1
                continue
            if i + 1 < len(argv):
                normalized.append(f"{tok}={argv[i + 1]}")
                i += 2
                continue
        normalized.append(tok)
        i += 1
    args = parser.parse_args(normalized)

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    if in_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        for item in sorted(in_path.glob("*.dae")):
            _convert_one(
                item,
                out_path / f"{item.stem}.fbx",
                global_scale=args.global_scale,
                axis_forward=args.axis_forward,
                axis_up=args.axis_up,
                mirror_y_for_unreal=bool(args.mirror_y_for_unreal),
            )
    else:
        _convert_one(
            in_path,
            out_path,
            global_scale=args.global_scale,
            axis_forward=args.axis_forward,
            axis_up=args.axis_up,
            mirror_y_for_unreal=bool(args.mirror_y_for_unreal),
        )


if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input .dae file or a directory containing .dae files")
    parser.add_argument("output", type=Path, help="Output .fbx path for file input, or output directory for directory input")
    parser.add_argument(
        "--global-scale",
        type=float,
        default=1.0,
        help="Keep source units; UE FBX import converts meters to centimeters.",
    )
    parser.add_argument("--axis-forward", default=os.environ.get("AMONGUS_FBX_AXIS_FORWARD", "X"))
    parser.add_argument("--axis-up", default=os.environ.get("AMONGUS_FBX_AXIS_UP", "Z"))
    parser.add_argument(
        "--mirror-y-for-unreal",
        action="store_true",
        # Default OFF to match ue_common_scene_loader._mirror_y_for_unreal_fbx_import (env default "0").
        # When the loader omits this flag it must mean no mirror; a True argparse default was
        # silently mirroring anyway and produced disconnected RM75 link meshes in UE.
        default=os.environ.get("AMONGUS_MIRROR_Y_FOR_UNREAL_FBX_IMPORT", "0").strip().lower()
        not in {"0", "false", "no", "off"},
        help="Convert Genesis/URDF link-local Y into UE link-local Y for the SRS world bridge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with tempfile.NamedTemporaryFile("w", suffix="_dae_to_fbx_blender.py", delete=False, encoding="utf-8") as handle:
        handle.write(_blender_driver_script())
        driver = Path(handle.name)
    try:
        subprocess.run(
            [
                _blender_bin(),
                "--background",
                "--factory-startup",
                "--python",
                str(driver),
                "--",
                "--input",
                str(args.input.expanduser().resolve()),
                "--output",
                str(args.output.expanduser().resolve()),
                f"--global-scale={float(args.global_scale)}",
                f"--axis-forward={args.axis_forward}",
                f"--axis-up={args.axis_up}",
                *(["--mirror-y-for-unreal"] if bool(args.mirror_y_for_unreal) else []),
            ],
            check=True,
        )
    finally:
        try:
            driver.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[convert_collada_to_fbx] {exc}", file=sys.stderr)
        raise
