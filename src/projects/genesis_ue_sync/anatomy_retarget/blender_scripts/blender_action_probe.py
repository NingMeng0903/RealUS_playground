"""Export selected pose-bone Action samples for offline coupling analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


DEFAULT_BONES = (
    "Femur_Rot_L",
    "Knee_Rotate_L",
    "Tibia_Bone_L",
    "Patella_Rotate_L",
    "Femur_Rot_R",
    "Knee_Rotate_R",
    "Tibia_Bone_R",
    "Patella_Rotate_R",
)


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bone", action="append", default=[])
    return parser.parse_args(argv)


def _matrix(value: object) -> list[list[float]]:
    return [[float(cell) for cell in row] for row in value]


def main() -> None:
    args = _args()
    bones = tuple(str(value) for value in (args.bone or DEFAULT_BONES))
    armature = next(
        obj
        for obj in bpy.data.objects
        if obj.type == "ARMATURE" and all(name in obj.pose.bones for name in bones)
    )
    animation = armature.animation_data
    action = animation.action if animation is not None else None
    if action is None:
        raise RuntimeError(f"armature {armature.name} has no active Action")
    start, stop = (int(round(value)) for value in action.frame_range)
    curves = []
    for curve in action.fcurves:
        if not any(f'pose.bones["{name}"]' in str(curve.data_path) for name in bones):
            continue
        curves.append(
            {
                "data_path": str(curve.data_path),
                "array_index": int(curve.array_index),
                "keyframes": [
                    {
                        "frame": float(point.co[0]),
                        "value": float(point.co[1]),
                        "interpolation": str(point.interpolation),
                    }
                    for point in curve.keyframe_points
                ],
            }
        )
    samples = []
    scene = bpy.context.scene
    old_frame = int(scene.frame_current)
    for frame in range(start, stop + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        samples.append(
            {
                "frame": frame,
                "bones": {
                    name: {
                        "matrix_basis": _matrix(armature.pose.bones[name].matrix_basis),
                        "matrix_armature": _matrix(armature.pose.bones[name].matrix),
                    }
                    for name in bones
                },
            }
        )
    scene.frame_set(old_frame)
    payload = {
        "blend_file": str(bpy.data.filepath),
        "armature": str(armature.name),
        "action": str(action.name),
        "frame_range": [start, stop],
        "bones": list(bones),
        "fcurves": curves,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(f"Blender Action probe written -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
