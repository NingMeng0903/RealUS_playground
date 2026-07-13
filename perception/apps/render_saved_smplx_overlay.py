#!/usr/bin/env python3
"""Render saved world-space SMPL-X vertices over a frozen multiview RGB frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common.project import project_paths
from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.easymocap.bodyhandface_viz import (
    compose_quad,
    compose_raw_skeleton_pair,
    compose_triptych,
    draw_bodyhandface_2d,
    draw_keypoints3d_repro,
    draw_skeleton_fused_2d_3d,
)
from projects.genesis_ue_sync.tracking.calibration import load_calibration_bundle, scale_intrinsics
from projects.genesis_ue_sync.tracking.tracking_mesh_overlay import (
    _blend_mesh_on_rgb,
    _project_camera_points_to_pixels,
    _world_points_camera_xyz,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="smplx_outputs/<run>")
    ap.add_argument("--frame-index", type=int, default=0, help="Frozen burst image index")
    ap.add_argument("--config", type=Path, default=Path("configs/tracking/realus_dwpose_easymocap.yaml"))
    args = ap.parse_args()
    run = Path(args.run).resolve()
    moment = run / "moment_0000"
    cfg = MultiviewRealtimeConfig.load(Path(args.config))
    calibration = load_calibration_bundle(cfg.calibration_path)
    mesh = np.load(moment / "smplx_result.npz")
    vertices = np.asarray(mesh["vertices"], dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(mesh["faces"], dtype=np.int64).reshape(-1, 3)
    frame_index = int(args.frame_index)
    frame_tag = f"frame_{frame_index:06d}"
    output = moment / "review_overlays" / frame_tag
    dirs = {
        "skeleton_2d": output / "skeleton_2d",
        "skeleton_3d_repro": output / "skeleton_3d_repro",
        "skeleton_fused": output / "skeleton_fused",
        "overlays": output / "overlays",
        "panels": output / "panels",
        "compare": output / "compare",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    moment_json = json.loads((moment / "moment.json").read_text(encoding="utf-8"))
    detection = moment_json.get("detection_by_frame", [])[frame_index]["per_camera"]
    kp_doc = json.loads((moment / "easymocap_output" / "keypoints3d" / f"{frame_index:06d}.json").read_text(encoding="utf-8"))
    # EasyMocap stores Body25 + left hand21 + right hand21 before the padded
    # face slots.  Keep all 67 measured joints so green fused fingers render.
    keypoints3d = np.asarray(kp_doc[0]["keypoints3d"], dtype=np.float32)[:67]
    for view_index, camera_id in enumerate(cfg.camera_ids):
        burst_image = moment / "burst" / f"{int(args.frame_index):06d}" / "images_raw" / f"{camera_id}.png"
        image_path = burst_image if burst_image.is_file() else moment / "images_raw" / f"{camera_id}.png"
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        cam = calibration.camera(camera_id)
        K = np.asarray(cam.intrinsics, dtype=np.float64).reshape(3, 3)
        if (rgb.shape[1], rgb.shape[0]) != (int(cam.width), int(cam.height)):
            K = scale_intrinsics(K, from_wh=(int(cam.width), int(cam.height)), to_wh=(rgb.shape[1], rgb.shape[0]))
        xyz_cam = _world_points_camera_xyz(vertices, cam.camera_from_world)
        uv, valid = _project_camera_points_to_pixels(xyz_cam, K)
        visible = valid & np.all(np.isfinite(uv), axis=1) & (xyz_cam[:, 2] > 1e-4)
        overlay = _blend_mesh_on_rgb(
            rgb, faces=faces, uv=uv, valid=visible, xyz_cam=xyz_cam, z_cam=xyz_cam[:, 2],
            mesh_alpha=0.82, mesh_rgb=(255, 128, 32), face_stride=1, max_triangle_px=520.0,
        )
        annot = {k: np.asarray(v, dtype=np.float32) for k, v in detection[camera_id]["easymocap"].items()}
        sk2d = draw_bodyhandface_2d(rgb, annot)
        P = K @ np.asarray(cam.camera_from_world[:3], dtype=np.float64)
        sk3d = draw_keypoints3d_repro(rgb, keypoints3d, P)
        fused = draw_skeleton_fused_2d_3d(rgb, annot, keypoints3d, P)
        Image.fromarray(sk2d).save(dirs["skeleton_2d"] / f"{camera_id}_bodyhandface.png")
        Image.fromarray(sk3d).save(dirs["skeleton_3d_repro"] / f"{camera_id}_tri3d_repro.png")
        Image.fromarray(fused).save(dirs["skeleton_fused"] / f"{camera_id}_red_gray_green.png")
        Image.fromarray(overlay).save(dirs["overlays"] / f"{camera_id}_smplx_overlay.png")
        Image.fromarray(compose_raw_skeleton_pair(rgb, sk2d)).save(dirs["compare"] / f"{camera_id}_raw_skeleton.png")
        Image.fromarray(overlay).save(dirs["compare"] / f"{camera_id}_smpl_overlay.png")
        Image.fromarray(compose_triptych(rgb, sk2d, overlay)).save(dirs["panels"] / f"{camera_id}_raw_skeleton_smplx.png")
        Image.fromarray(compose_quad(rgb, sk2d, sk3d, overlay)).save(dirs["panels"] / f"{camera_id}_raw_skeleton_3d_smplx.png")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
