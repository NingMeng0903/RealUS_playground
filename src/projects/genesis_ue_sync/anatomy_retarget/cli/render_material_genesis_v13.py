"""Render V13 geometry cells with the repository's existing Genesis reviewer.

Reuses _render_layer, mesh export, RGB/depth/segmentation and scale bars from
the historical Genesis review pipeline. Never reruns a retarget solver.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw

from .render_alignment_truth_genesis_v1 import _render_layer, COLORS
from .render_chain_rest_fit_genesis_v1 import _export, _sha256


def _cameras(skin: np.ndarray, joints: np.ndarray) -> dict:
    center = (skin.min(0) + skin.max(0)) * .5
    span = np.ptp(skin, axis=0)
    distance = max(span[1], span[0] / (720 / 540)) * .56 / np.tan(np.deg2rad(17))
    cameras = {"whole_ap": dict(pos=(center + [0, 0, distance]).tolist(),
                               lookat=center.tolist(), up=[0, 1, 0], fov=34.)}
    pelvis = .5 * (joints[1] + joints[2])
    cameras["pelvis_ap"] = dict(pos=(pelvis + [0, .03, .83]).tolist(),
                                lookat=pelvis.tolist(), up=[0, 1, 0], fov=38.)
    for side, sign, elbow, knee, ankle, foot in (("left", 1, 18, 4, 7, 10), ("right", -1, 19, 5, 8, 11)):
        hip = joints[1 if side == "left" else 2]
        for name, origin, offset, fov in (
            ("collar_ap", joints[13 if side == "left" else 14], [0, 0, .48], 38),
            ("collar_superior", joints[13 if side == "left" else 14], [.06 * sign, .34, .25], 38),
            ("shoulder_oblique", joints[16 if side == "left" else 17], [.40 * sign, .04, .24], 38),
            ("hip_oblique", hip, [.30 * sign, .04, .38], 38),
            ("hip_posterior", hip, [.28 * sign, .04, -.38], 38),
            ("elbow_ap", joints[elbow], [0, 0, .48], 34),
            ("elbow_lateral", joints[elbow], [.42 * sign, .03, .16], 34),
            ("wrist_ap", joints[20 if side == "left" else 21], [0, 0, .36], 34),
            ("wrist_lateral", joints[20 if side == "left" else 21], [.32 * sign, .02, .12], 34),
            ("knee_ap", joints[knee], [0, 0, .50], 34),
            ("knee_lateral", joints[knee], [.46 * sign, .02, .16], 34),
            ("ankle_oblique", joints[ankle], [.28 * sign, .05, .38], 34),
            ("foot_oblique", .5 * (joints[ankle] + joints[foot]), [.26 * sign, -.15, .36], 38),
        ):
            depth = float(np.linalg.norm(offset))
            cameras[f"{side}_{name}"] = dict(pos=(origin + offset).tolist(), lookat=origin.tolist(),
                                             up=[0, 1, 0], fov=float(fov),
                                             near=max(.01, depth - .16), far=depth + .16)
    return cameras


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--skin-only-layer", action="store_true",
                        help="also render the opaque SMPL-X skin through the same cameras")
    parser.add_argument("--before-label", default="before_v12e")
    parser.add_argument("--before-key", default="source_vertices",
                        choices=["source_vertices", "before_vertices"],
                        help="Explicit geometry field for the left comparison")
    parser.add_argument("--discard-render-geometry", action="store_true",
                        help="Remove this run's temporary OBJ exports after retaining RGB/depth and source NPZ hashes")
    parser.add_argument("--views", nargs="+",
                        help="render only these existing camera names, e.g. left_elbow_ap left_elbow_lateral")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"renderer": "existing GenesisPlatformRuntime via render_alignment_truth_genesis_v1._render_layer",
                "publishable": False, "camera_source": "SMPL-X joints and skin only",
                "internal_material_alpha": 1.0,
                "local_view_depth_clipping": "plus/minus 160mm at joint; whole view unclipped", "cells": {}}
    for path in args.input:
        with np.load(path, allow_pickle=False) as data:
            faces, skin_faces = data["faces"].copy(), data["skin_faces"].copy()
            labels = data["vertex_tissue"].copy()
            pose, joints = data["pose"].copy(), data["smplx_joints"].copy()
            pivot = joints[0].copy()
            rotation = Rotation.from_rotvec(pose.reshape(55, 3)[0]).inv()
            def unrotate(points):
                return rotation.apply(np.asarray(points, dtype=np.float64) - pivot) + pivot
            skin = unrotate(data["skin_vertices"])
            joints = unrotate(joints)
            arrays = {args.before_label: unrotate(data[args.before_key]),
                      "candidate": unrotate(data["candidate_vertices"])}
        root = args.output / path.stem
        root.mkdir()
        skin_path = _export(root / "geometry" / "skin.obj", skin, skin_faces)
        cameras = _cameras(skin, joints)
        if args.views:
            unknown = set(args.views) - cameras.keys()
            if unknown:
                raise ValueError(f"unknown views: {sorted(unknown)}; available: {list(cameras)}")
            cameras = {name: cameras[name] for name in dict.fromkeys(args.views)}
        cell = {"input": str(path.resolve()), "input_sha256": _sha256(path),
                "before_geometry_key": args.before_key,
                "temporary_obj_exports_retained": not args.discard_render_geometry,
                "root_rotation_removed_equally": pose.reshape(55, 3)[0].tolist(), "variants": {}}
        for variant, vertices in arrays.items():
            if args.candidate_only and variant != "candidate":
                continue
            entities = [("skin", skin_path, COLORS["skin"])]
            for name, code, color in (("bones", 0, COLORS["candidate"]),
                                      ("vessels", 1, (*COLORS["vessels"][:3], 1.0)),
                                      ("nerves", 2, (*COLORS["nerves"][:3], 1.0))):
                tissue_faces = faces[np.all(labels[faces] == code, axis=1)]
                mesh_path = _export(root / "geometry" / f"{variant}_{name}.obj", vertices, tissue_faces)
                entities.append((name, mesh_path, color))
            cell["variants"][variant] = _render_layer(root / variant, entities=entities,
                                                       cameras=cameras, backend=args.backend)
            print(path.stem, variant, "Genesis 3D rendered", flush=True)
        if args.skin_only_layer:
            cell["skin_only"] = _render_layer(root / "smplx_skin", entities=[
                ("smplx_skin", skin_path, (.80, .64, .51, 1.0))], cameras=cameras, backend=args.backend)
        if not args.candidate_only:
            for name in cameras:
                images = [Image.open(root / variant / "rgb" / f"{name}.png").convert("RGB")
                          for variant in arrays]
                sheet = Image.new("RGB", (1440, 570), (25, 25, 25))
                draw = ImageDraw.Draw(sheet)
                for i, (variant, im) in enumerate(zip(arrays, images)):
                    sheet.paste(im, (i * 720, 30)); draw.text((i * 720 + 8, 8), variant, fill="white")
                (root / "comparison").mkdir(exist_ok=True)
                sheet.save(root / "comparison" / f"{name}.png")
        manifest["cells"][path.stem] = cell
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if args.discard_render_geometry:
            import shutil
            shutil.rmtree(root / "geometry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
