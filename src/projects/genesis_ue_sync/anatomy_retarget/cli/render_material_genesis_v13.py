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
    def foot_frame(ankle: np.ndarray, foot: np.ndarray) -> tuple[np.ndarray, float]:
        """Return a foot look-at and a scale derived from joints and skin bbox.

        The local skin selection keeps the added side/plantar views useful for
        bent feet without embedding a pose-specific camera location.  The
        fallback uses the full skin bbox when a foot joint falls just outside
        the surface due to a fitted pose.
        """
        center = .5 * (ankle + foot)
        segment = float(np.linalg.norm(foot - ankle))
        radius = max(.08, 1.8 * segment)
        local_mask = np.linalg.norm(skin - center, axis=1) <= radius
        local_skin = skin[local_mask]
        if len(local_skin) < 32:
            local_skin = skin
            focus = center
        else:
            # Frame the complete selected surface bbox rather than a
            # hard-coded joint offset so toes and heel stay in view.
            focus = .5 * (local_skin.min(axis=0) + local_skin.max(axis=0))
        local_span = np.ptp(local_skin, axis=0)
        # Keep the camera far enough to include the complete toe/heel bbox.
        extent = max(.06, float(np.max(local_span)))
        return focus, extent

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
        foot_center, foot_extent = foot_frame(joints[ankle], joints[foot])
        foot_segment = joints[foot] - joints[ankle]
        foot_length = float(np.linalg.norm(foot_segment))
        if foot_length > 1.0e-8:
            toes_dir = foot_segment / foot_length
        else:
            toes_dir = np.array([0.0, 0.0, 1.0])
        lateral_distance = max(.16, 1.8 * foot_extent)
        down_distance = max(.16, 1.8 * foot_extent)
        foot_fov = 38.0
        lateral_far = max(2.0, lateral_distance + 3.0 * foot_extent)
        cameras[f"{side}_foot_lateral"] = dict(
            pos=(foot_center + np.array([sign * lateral_distance, .0, .10 * foot_extent])).tolist(),
            lookat=foot_center.tolist(), up=[0, 1, 0], fov=foot_fov,
            near=.01,
            far=lateral_far,
        )
        # This is a world-down underside view.  Without a posed foot frame
        # (or an SMPL-X ankle rotation), it must not be called anatomical
        # plantar for arbitrary rotated poses.
        down_view = np.array([0.0, -1.0, 0.0])
        down_up = toes_dir - np.dot(toes_dir, down_view) * down_view
        if float(np.linalg.norm(down_up)) <= 1.0e-8:
            down_up = np.array([0.0, 0.0, 1.0])
        else:
            down_up /= np.linalg.norm(down_up)
        down_spec = dict(
            pos=(foot_center + down_view * down_distance).tolist(),
            lookat=foot_center.tolist(), up=down_up.tolist(), fov=foot_fov,
            near=.01,
            far=max(2.0, down_distance + 3.0 * foot_extent),
        )
        cameras[f"{side}_foot_underside"] = down_spec
        # Retain the old key for reproducibility of existing review scripts;
        # manifest metadata below explicitly identifies it as world-down.
        cameras[f"{side}_foot_plantar"] = dict(down_spec)
    return cameras


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--skin-only-layer", action="store_true",
                        help="also render the opaque SMPL-X skin through the same cameras")
    parser.add_argument("--opaque-skin-overlay", action="store_true",
                        help=("render an additional depth-tested layer with opaque medium-gray "
                              "SMPL-X skin; internal bone/vessel/nerve surfaces remain visible "
                              "only where they protrude in front of the skin"))
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
                "publishable": False,
                "camera_source": "SMPL-X joints and skin only; foot side uses ankle-foot joints and local skin bbox; foot underside is world-down, not anatomical plantar",
                "internal_material_alpha": 1.0,
                "local_view_depth_clipping": "plus/minus 160mm at joint; whole view unclipped", "cells": {}}
    if args.opaque_skin_overlay:
        manifest["opaque_skin_overlay"] = {
            "enabled": True,
            "skin_color_rgba": [0.50, 0.50, 0.50, 1.0],
            "depth_occlusion": "Genesis depth-tested opaque skin; internal surfaces show only where in front",
            "geometry_unchanged": True,
        }
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
            mesh_paths = {}
            mesh_colors = {}
            for name, code, color in (("bones", 0, COLORS["candidate"]),
                                      ("vessels", 1, (*COLORS["vessels"][:3], 1.0)),
                                      ("nerves", 2, (*COLORS["nerves"][:3], 1.0))):
                tissue_faces = faces[np.all(labels[faces] == code, axis=1)]
                mesh_path = _export(root / "geometry" / f"{variant}_{name}.obj", vertices, tissue_faces)
                mesh_paths[name] = mesh_path
                mesh_colors[name] = color
                entities.append((name, mesh_path, color))
            cell["variants"][variant] = _render_layer(root / variant, entities=entities,
                                                       cameras=cameras, backend=args.backend)
            if args.opaque_skin_overlay:
                overlay_entities = [
                    ("skin_opaque", skin_path, (0.50, 0.50, 0.50, 1.0)),
                    ("bones", mesh_paths["bones"], mesh_colors["bones"]),
                    ("vessels", mesh_paths["vessels"], mesh_colors["vessels"]),
                    ("nerves", mesh_paths["nerves"], mesh_colors["nerves"]),
                ]
                cell["variants"][variant]["opaque_skin_overlay"] = _render_layer(
                    root / variant / "opaque_skin_overlay",
                    entities=overlay_entities,
                    cameras=cameras,
                    backend=args.backend,
                )
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
                if args.opaque_skin_overlay:
                    overlay_images = [
                        Image.open(root / variant / "opaque_skin_overlay" / "rgb" / f"{name}.png")
                        .convert("RGB")
                        for variant in arrays
                    ]
                    overlay_sheet = Image.new("RGB", (1440, 570), (25, 25, 25))
                    overlay_draw = ImageDraw.Draw(overlay_sheet)
                    for i, (variant, image) in enumerate(zip(arrays, overlay_images)):
                        overlay_sheet.paste(image, (i * 720, 30))
                        overlay_draw.text((i * 720 + 8, 8), variant, fill="white")
                    (root / "comparison_opaque_skin_overlay").mkdir(exist_ok=True)
                    overlay_sheet.save(root / "comparison_opaque_skin_overlay" / f"{name}.png")
        manifest["cells"][path.stem] = cell
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if args.discard_render_geometry:
            import shutil
            shutil.rmtree(root / "geometry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
