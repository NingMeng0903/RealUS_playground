from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle
from projects.genesis_ue_sync.tracking.debug_runtime import append_debug_log
from projects.genesis_ue_sync.tracking.feature_video_renderer import write_mp4_streaming
from projects.genesis_ue_sync.tracking.tracking_skeleton_overlay import project_world_points_to_pixels
from projects.genesis_ue_sync.tracking.uhmr_backend import UhmrSequenceResult
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import (
    HumanMotionSequence,
    evaluate_smpl_sequence,
    resolve_torch_device,
    _create_smpl_model,
)


def _world_points_camera_xyz(points_world: np.ndarray, camera_from_world: np.ndarray) -> np.ndarray:
    R = np.asarray(camera_from_world, dtype=np.float64).reshape(4, 4)[:3, :3]
    t = np.asarray(camera_from_world, dtype=np.float64).reshape(4, 4)[:3, 3].reshape(3)
    X = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    return ((R @ X.T).T + t).astype(np.float64)


def _world_points_camera_z(points_world: np.ndarray, camera_from_world: np.ndarray) -> np.ndarray:
    return _world_points_camera_xyz(points_world, camera_from_world)[:, 2]


def _project_camera_points_to_pixels(
    points_camera: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Xc = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    z = Xc[:, 2]
    valid = z > 1e-5
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = np.full(Xc.shape[0], np.nan, dtype=np.float64)
    v = np.full(Xc.shape[0], np.nan, dtype=np.float64)
    u[valid] = fx * Xc[valid, 0] / z[valid] + cx
    v[valid] = fy * Xc[valid, 1] / z[valid] + cy
    return np.stack([u, v], axis=-1), valid


def _normalized(vec: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n <= eps:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / n


def _blend_mesh_on_rgb(
    rgb: np.ndarray,
    *,
    faces: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    xyz_cam: np.ndarray,
    z_cam: np.ndarray,
    mesh_alpha: float,
    mesh_rgb: tuple[int, int, int],
    face_stride: int,
    max_triangle_px: float,
) -> np.ndarray:
    import cv2

    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    base = np.asarray(rgb, dtype=np.uint8).copy()
    shaded = np.zeros((h, w, 3), dtype=np.float32)
    alpha_mask = np.zeros((h, w), dtype=np.float32)
    base_rgb = np.asarray(mesh_rgb, dtype=np.float32)
    light_dir = _normalized(np.array([-0.35, -0.45, -1.0], dtype=np.float64))
    ambient = 0.32
    diffuse = 0.68

    order = np.argsort(-_face_mean_z(faces, z_cam))
    stride = max(int(face_stride), 1)
    for idx in range(0, int(faces.shape[0]), stride):
        fi = int(order[idx])
        tri = faces[fi].astype(np.int64)
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        if not (valid[i] and valid[j] and valid[k]):
            continue
        if min(float(z_cam[i]), float(z_cam[j]), float(z_cam[k])) <= 1e-4:
            continue
        pts = np.asarray(
            [[float(uv[i, 0]), float(uv[i, 1])], [float(uv[j, 0]), float(uv[j, 1])], [float(uv[k, 0]), float(uv[k, 1])]],
            dtype=np.float32,
        )
        span = float(np.max(pts[:, 0]) - np.min(pts[:, 0]) + np.max(pts[:, 1]) - np.min(pts[:, 1]))
        if span > max_triangle_px or not np.all(np.isfinite(pts)):
            continue
        poly = np.round(pts).astype(np.int32)
        if np.unique(poly, axis=0).shape[0] < 3:
            continue
        p0 = np.asarray(xyz_cam[i], dtype=np.float64)
        p1 = np.asarray(xyz_cam[j], dtype=np.float64)
        p2 = np.asarray(xyz_cam[k], dtype=np.float64)
        normal = np.cross(p1 - p0, p2 - p0)
        centroid = (p0 + p1 + p2) / 3.0
        if float(np.dot(normal, centroid)) > 0.0:
            normal = -normal
        normal = _normalized(normal)
        lambert = max(0.0, float(np.dot(normal, light_dir)))
        depth_term = float(np.clip((np.mean([z_cam[i], z_cam[j], z_cam[k]]) - 0.8) / 2.6, 0.0, 1.0))
        intensity = ambient + diffuse * lambert
        intensity *= 0.92 + 0.08 * (1.0 - depth_term)
        face_rgb = np.clip(base_rgb * intensity, 0.0, 255.0)
        cv2.fillConvexPoly(shaded, poly, color=tuple(float(v) for v in face_rgb), lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(alpha_mask, poly, color=1.0, lineType=cv2.LINE_AA)
        edge_rgb = tuple(float(v) for v in np.clip(face_rgb * 0.72, 0.0, 255.0))
        cv2.polylines(shaded, [poly.reshape(-1, 1, 2)], isClosed=True, color=edge_rgb, thickness=1, lineType=cv2.LINE_AA)

    a = float(np.clip(mesh_alpha, 0.0, 1.0))
    m = np.clip(alpha_mask[..., None], 0.0, 1.0)
    out = base.astype(np.float32) * (1.0 - a * m) + shaded * (a * m)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _blend_mesh_layers_on_rgb(
    rgb: np.ndarray,
    *,
    layers: list[dict[str, Any]],
    max_triangle_px: float,
) -> np.ndarray:
    import cv2

    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    out = np.asarray(rgb, dtype=np.float32).copy()
    light_dir = _normalized(np.array([-0.35, -0.45, -1.0], dtype=np.float64))
    ambient = 0.32
    diffuse = 0.68
    ops: list[dict[str, Any]] = []
    for layer in layers:
        faces = np.asarray(layer["faces"], dtype=np.int64)
        uv = np.asarray(layer["uv"], dtype=np.float64)
        valid = np.asarray(layer["valid"], dtype=bool)
        xyz_cam = np.asarray(layer["xyz_cam"], dtype=np.float64)
        z_cam = np.asarray(layer["z_cam"], dtype=np.float64)
        mesh_rgb = np.asarray(layer["mesh_rgb"], dtype=np.float32)
        mesh_alpha = float(np.clip(layer["mesh_alpha"], 0.0, 1.0))
        face_stride = max(int(layer["face_stride"]), 1)
        order = np.argsort(-_face_mean_z(faces, z_cam))
        for idx in range(0, int(faces.shape[0]), face_stride):
            fi = int(order[idx])
            tri = faces[fi].astype(np.int64)
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            if not (valid[i] and valid[j] and valid[k]):
                continue
            if min(float(z_cam[i]), float(z_cam[j]), float(z_cam[k])) <= 1e-4:
                continue
            pts = np.asarray(
                [[float(uv[i, 0]), float(uv[i, 1])], [float(uv[j, 0]), float(uv[j, 1])], [float(uv[k, 0]), float(uv[k, 1])]],
                dtype=np.float32,
            )
            span = float(np.max(pts[:, 0]) - np.min(pts[:, 0]) + np.max(pts[:, 1]) - np.min(pts[:, 1]))
            if span > max_triangle_px or not np.all(np.isfinite(pts)):
                continue
            poly = np.round(pts).astype(np.int32)
            if np.unique(poly, axis=0).shape[0] < 3:
                continue
            p0 = np.asarray(xyz_cam[i], dtype=np.float64)
            p1 = np.asarray(xyz_cam[j], dtype=np.float64)
            p2 = np.asarray(xyz_cam[k], dtype=np.float64)
            normal = np.cross(p1 - p0, p2 - p0)
            centroid = (p0 + p1 + p2) / 3.0
            if float(np.dot(normal, centroid)) > 0.0:
                normal = -normal
            normal = _normalized(normal)
            lambert = max(0.0, float(np.dot(normal, light_dir)))
            depth_term = float(np.clip((np.mean([z_cam[i], z_cam[j], z_cam[k]]) - 0.8) / 2.6, 0.0, 1.0))
            intensity = ambient + diffuse * lambert
            intensity *= 0.92 + 0.08 * (1.0 - depth_term)
            face_rgb = np.clip(mesh_rgb * intensity, 0.0, 255.0)
            ops.append(
                {
                    "depth": float(np.mean([z_cam[i], z_cam[j], z_cam[k]])),
                    "poly": poly,
                    "face_rgb": face_rgb,
                    "edge_rgb": np.clip(face_rgb * 0.72, 0.0, 255.0),
                    "alpha": mesh_alpha,
                }
            )
    ops.sort(key=lambda item: item["depth"], reverse=True)
    for op in ops:
        poly = np.asarray(op["poly"], dtype=np.int32)
        alpha = float(op["alpha"])
        face_rgb = np.asarray(op["face_rgb"], dtype=np.float32)
        edge_rgb = tuple(float(v) for v in np.asarray(op["edge_rgb"], dtype=np.float32))
        mask = np.zeros((h, w), dtype=np.float32)
        shaded = np.zeros((h, w, 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, poly, color=1.0, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(shaded, poly, color=tuple(float(v) for v in face_rgb), lineType=cv2.LINE_AA)
        cv2.polylines(shaded, [poly.reshape(-1, 1, 2)], isClosed=True, color=edge_rgb, thickness=1, lineType=cv2.LINE_AA)
        m = np.clip(mask[..., None], 0.0, 1.0)
        out = out * (1.0 - alpha * m) + shaded * (alpha * m)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _face_mean_z(faces: np.ndarray, z_cam: np.ndarray) -> np.ndarray:
    z = np.asarray(z_cam, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    return (z[f[:, 0]] + z[f[:, 1]] + z[f[:, 2]]) / 3.0


def _render_motion_mesh_overlays_on_rgb(
    *,
    motion: HumanMotionSequence,
    sequence_result: UhmrSequenceResult,
    calibration: CalibrationBundle,
    output_root: Path,
    fps: float,
    smpl_device: str | None = None,
    mesh_alpha: float = 0.38,
    face_stride: int = 2,
    mesh_rgb: tuple[int, int, int] = (90, 190, 255),
    export_png: bool = True,
    export_mp4: bool = False,
    max_triangle_px: float = 420.0,
    projection_mode: str = "world_direct",
    log_location: str,
    log_message: str,
    hypothesis_id: str,
) -> dict[str, Any]:
    """Project SMPL mesh vertices with pinhole intrinsics/extrinsics and alpha-blend onto each camera RGB."""
    verts, _joints = evaluate_smpl_sequence(
        motion,
        device=smpl_device,
        include_vertices=True,
        include_joints=False,
    )
    if verts is None:
        raise RuntimeError("SMPL vertex evaluation failed for mesh overlay.")
    torch_device = resolve_torch_device(smpl_device)
    smpl_model = _create_smpl_model(motion, torch_device)
    faces = np.asarray(smpl_model.faces, dtype=np.int64)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    camera_ids = calibration.ordered_camera_ids()
    png_dirs: dict[str, Path] = {}
    mp4_paths: dict[str, Path | None] = {}
    overlay_debug_by_camera: dict[str, dict[str, Any]] = {}
    for camera_id in camera_ids:
        cam = calibration.camera(camera_id)
        K = cam.intrinsics
        ext = cam.camera_from_world
        cam_out = output_root / camera_id
        cam_out.mkdir(parents=True, exist_ok=True)
        png_dirs[camera_id] = cam_out
        frames_for_mp4: list[np.ndarray] = []
        frame_stats: list[dict[str, Any]] = []
        for fr in sequence_result.frame_results:
            rgb = fr.rgb_frames.get(camera_id)
            if rgb is None:
                continue
            V = verts[fr.frame_idx].astype(np.float64)
            xyz_cam = _world_points_camera_xyz(V, ext)
            if projection_mode == "world_direct":
                uv, valid = project_world_points_to_pixels(V, ext, K)
            elif projection_mode == "camera_space":
                uv, valid = _project_camera_points_to_pixels(xyz_cam, K)
            else:
                raise ValueError(f"Unsupported projection_mode: {projection_mode}")
            z_cam = xyz_cam[:, 2]
            finite_uv = np.all(np.isfinite(uv), axis=1)
            visible = valid & finite_uv & (z_cam > 1e-4)
            if np.any(visible):
                uv_visible = uv[visible]
                bbox_min = np.min(uv_visible, axis=0)
                bbox_max = np.max(uv_visible, axis=0)
            else:
                bbox_min = np.array([np.nan, np.nan], dtype=np.float64)
                bbox_max = np.array([np.nan, np.nan], dtype=np.float64)
            frame_stats.append(
                {
                    "frame_idx": int(fr.frame_idx),
                    "visible_vertex_ratio": float(np.mean(visible.astype(np.float32))),
                    "z_cam_min": float(np.min(z_cam)),
                    "z_cam_max": float(np.max(z_cam)),
                    "uv_bbox_min": [float(bbox_min[0]), float(bbox_min[1])],
                    "uv_bbox_max": [float(bbox_max[0]), float(bbox_max[1])],
                }
            )
            out = _blend_mesh_on_rgb(
                rgb,
                faces=faces,
                uv=uv,
                valid=valid,
                xyz_cam=xyz_cam,
                z_cam=z_cam,
                mesh_alpha=float(mesh_alpha),
                mesh_rgb=mesh_rgb,
                face_stride=int(face_stride),
                max_triangle_px=float(max_triangle_px),
            )
            if export_png:
                stem = f"frame_{fr.frame_idx:05d}"
                Image.fromarray(out).save(cam_out / f"{stem}.png")
            if export_mp4:
                frames_for_mp4.append(out)
        mp4_paths[camera_id] = None
        if export_mp4 and frames_for_mp4:
            p = cam_out / f"{camera_id}_smpl_mesh_overlay.mp4"
            write_mp4_streaming(p, frames_for_mp4, fps=float(fps))
            mp4_paths[camera_id] = p
        if frame_stats:
            overlay_debug_by_camera[camera_id] = {
                "frame_count": int(len(frame_stats)),
                "frame0": frame_stats[0],
                "frame_last": frame_stats[-1],
            }

    if overlay_debug_by_camera:
        # region agent log
        append_debug_log(
            location=log_location,
            message=log_message,
            data={
                "projection_mode": str(projection_mode),
                "motion_sequence_name": str(motion.sequence_name),
                "motion_source_path": str(motion.source_path),
                "mesh_alpha": float(mesh_alpha),
                "mesh_rgb": [int(mesh_rgb[0]), int(mesh_rgb[1]), int(mesh_rgb[2])],
                "shading": "camera_space_lambert",
                "face_stride": int(face_stride),
                "max_triangle_px": float(max_triangle_px),
                "per_camera": overlay_debug_by_camera,
            },
            run_id="debug-triage",
            hypothesis_id=hypothesis_id,
        )
        # endregion

    return {
        "output_root": str(output_root.resolve()),
        "per_camera_png_dirs": {k: str(v.resolve()) for k, v in png_dirs.items()},
        "per_camera_mp4": {k: str(v.resolve()) if v is not None else None for k, v in mp4_paths.items()},
        "per_camera_debug": overlay_debug_by_camera,
        "face_count": int(faces.shape[0]),
        "face_stride": int(face_stride),
        "vertex_count": int(verts.shape[1]),
        "projection_mode": str(projection_mode),
    }


def render_smpl_mesh_overlays_on_rgb(
    *,
    sequence_result: UhmrSequenceResult,
    calibration: CalibrationBundle,
    output_root: Path,
    fps: float,
    smpl_device: str | None = None,
    mesh_alpha: float = 0.38,
    face_stride: int = 2,
    mesh_rgb: tuple[int, int, int] = (90, 190, 255),
    export_png: bool = True,
    export_mp4: bool = False,
    max_triangle_px: float = 420.0,
) -> dict[str, Any]:
    return _render_motion_mesh_overlays_on_rgb(
        motion=sequence_result.motion_sequence,
        sequence_result=sequence_result,
        calibration=calibration,
        output_root=output_root,
        fps=fps,
        smpl_device=smpl_device,
        mesh_alpha=mesh_alpha,
        face_stride=face_stride,
        mesh_rgb=mesh_rgb,
        export_png=export_png,
        export_mp4=export_mp4,
        max_triangle_px=max_triangle_px,
        projection_mode="world_direct",
        log_location="src/projects/genesis_ue_sync/human_recovery/tracking_mesh_overlay.py:render_smpl_mesh_overlays_on_rgb:summary",
        log_message="SMPL mesh overlay projection summary",
        hypothesis_id="H20",
    )


def render_reference_smpl_mesh_overlays_on_rgb(
    *,
    reference_motion: HumanMotionSequence,
    sequence_result: UhmrSequenceResult,
    calibration: CalibrationBundle,
    output_root: Path,
    fps: float,
    smpl_device: str | None = None,
    mesh_alpha: float = 0.9,
    face_stride: int = 1,
    mesh_rgb: tuple[int, int, int] = (64, 128, 255),
    export_png: bool = True,
    export_mp4: bool = False,
    max_triangle_px: float = 420.0,
    projection_mode: str = "world_direct",
) -> dict[str, Any]:
    return _render_motion_mesh_overlays_on_rgb(
        motion=reference_motion,
        sequence_result=sequence_result,
        calibration=calibration,
        output_root=output_root,
        fps=fps,
        smpl_device=smpl_device,
        mesh_alpha=mesh_alpha,
        face_stride=face_stride,
        mesh_rgb=mesh_rgb,
        export_png=export_png,
        export_mp4=export_mp4,
        max_triangle_px=max_triangle_px,
        projection_mode=projection_mode,
        log_location="src/projects/genesis_ue_sync/human_recovery/tracking_mesh_overlay.py:render_reference_smpl_mesh_overlays_on_rgb:summary",
        log_message="Reference SMPL mesh overlay projection summary",
        hypothesis_id="H28",
    )


def render_comparison_smpl_mesh_overlays_on_rgb(
    *,
    predicted_motion: HumanMotionSequence,
    reference_motion: HumanMotionSequence,
    sequence_result: UhmrSequenceResult,
    calibration: CalibrationBundle,
    output_root: Path,
    fps: float,
    smpl_device: str | None = None,
    predicted_mesh_alpha: float = 0.92,
    predicted_mesh_rgb: tuple[int, int, int] = (224, 224, 224),
    reference_mesh_alpha: float = 0.92,
    reference_mesh_rgb: tuple[int, int, int] = (64, 128, 255),
    face_stride: int = 1,
    export_png: bool = True,
    export_mp4: bool = False,
    max_triangle_px: float = 420.0,
) -> dict[str, Any]:
    pred_verts, _ = evaluate_smpl_sequence(predicted_motion, device=smpl_device, include_vertices=True, include_joints=False)
    ref_verts, _ = evaluate_smpl_sequence(reference_motion, device=smpl_device, include_vertices=True, include_joints=False)
    if pred_verts is None or ref_verts is None:
        raise RuntimeError("SMPL vertex evaluation failed for comparison overlay.")
    torch_device = resolve_torch_device(smpl_device)
    smpl_model = _create_smpl_model(predicted_motion, torch_device)
    faces = np.asarray(smpl_model.faces, dtype=np.int64)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    png_dirs: dict[str, Path] = {}
    mp4_paths: dict[str, Path | None] = {}
    overlay_debug_by_camera: dict[str, dict[str, Any]] = {}
    for camera_id in calibration.ordered_camera_ids():
        cam = calibration.camera(camera_id)
        K = cam.intrinsics
        ext = cam.camera_from_world
        cam_out = output_root / camera_id
        cam_out.mkdir(parents=True, exist_ok=True)
        png_dirs[camera_id] = cam_out
        frames_for_mp4: list[np.ndarray] = []
        frame_stats: list[dict[str, Any]] = []
        for fr in sequence_result.frame_results:
            rgb = fr.rgb_frames.get(camera_id)
            if rgb is None:
                continue
            pred_v = pred_verts[fr.frame_idx].astype(np.float64)
            ref_v = ref_verts[fr.frame_idx].astype(np.float64)
            pred_uv, pred_valid = project_world_points_to_pixels(pred_v, ext, K)
            ref_uv, ref_valid = project_world_points_to_pixels(ref_v, ext, K)
            pred_xyz_cam = _world_points_camera_xyz(pred_v, ext)
            ref_xyz_cam = _world_points_camera_xyz(ref_v, ext)
            out = _blend_mesh_layers_on_rgb(
                rgb,
                layers=[
                    {
                        "faces": faces,
                        "uv": pred_uv,
                        "valid": pred_valid,
                        "xyz_cam": pred_xyz_cam,
                        "z_cam": pred_xyz_cam[:, 2],
                        "mesh_alpha": float(predicted_mesh_alpha),
                        "mesh_rgb": predicted_mesh_rgb,
                        "face_stride": int(face_stride),
                    },
                    {
                        "faces": faces,
                        "uv": ref_uv,
                        "valid": ref_valid,
                        "xyz_cam": ref_xyz_cam,
                        "z_cam": ref_xyz_cam[:, 2],
                        "mesh_alpha": float(reference_mesh_alpha),
                        "mesh_rgb": reference_mesh_rgb,
                        "face_stride": int(face_stride),
                    },
                ],
                max_triangle_px=float(max_triangle_px),
            )
            pred_visible = pred_valid & np.all(np.isfinite(pred_uv), axis=1) & (pred_xyz_cam[:, 2] > 1e-4)
            ref_visible = ref_valid & np.all(np.isfinite(ref_uv), axis=1) & (ref_xyz_cam[:, 2] > 1e-4)
            frame_stats.append(
                {
                    "frame_idx": int(fr.frame_idx),
                    "pred_visible_vertex_ratio": float(np.mean(pred_visible.astype(np.float32))),
                    "ref_visible_vertex_ratio": float(np.mean(ref_visible.astype(np.float32))),
                }
            )
            if export_png:
                stem = f"frame_{fr.frame_idx:05d}"
                Image.fromarray(out).save(cam_out / f"{stem}.png")
            if export_mp4:
                frames_for_mp4.append(out)
        mp4_paths[camera_id] = None
        if export_mp4 and frames_for_mp4:
            p = cam_out / f"{camera_id}_comparison_smpl_mesh_overlay.mp4"
            write_mp4_streaming(p, frames_for_mp4, fps=float(fps))
            mp4_paths[camera_id] = p
        if frame_stats:
            overlay_debug_by_camera[camera_id] = {
                "frame_count": int(len(frame_stats)),
                "frame0": frame_stats[0],
                "frame_last": frame_stats[-1],
            }
    if overlay_debug_by_camera:
        append_debug_log(
            location="src/projects/genesis_ue_sync/human_recovery/tracking_mesh_overlay.py:render_comparison_smpl_mesh_overlays_on_rgb:summary",
            message="Comparison SMPL mesh overlay summary",
            data={
                "predicted_mesh_rgb": [int(v) for v in predicted_mesh_rgb],
                "reference_mesh_rgb": [int(v) for v in reference_mesh_rgb],
                "face_stride": int(face_stride),
                "max_triangle_px": float(max_triangle_px),
                "per_camera": overlay_debug_by_camera,
            },
            run_id="debug-triage",
            hypothesis_id="H31",
        )
    return {
        "output_root": str(output_root.resolve()),
        "per_camera_png_dirs": {k: str(v.resolve()) for k, v in png_dirs.items()},
        "per_camera_mp4": {k: str(v.resolve()) if v is not None else None for k, v in mp4_paths.items()},
        "per_camera_debug": overlay_debug_by_camera,
        "face_count": int(faces.shape[0]),
        "face_stride": int(face_stride),
        "vertex_count": int(pred_verts.shape[1]),
    }


__all__ = [
    "render_smpl_mesh_overlays_on_rgb",
    "render_reference_smpl_mesh_overlays_on_rgb",
    "render_comparison_smpl_mesh_overlays_on_rgb",
]
