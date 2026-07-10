from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.cli.render.media.convert_collada_to_obj import bake_collada_mesh
from projects.genesis_ue_sync.tracking.camera_image_correction import CameraImageCorrection
from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle, build_intrinsics_from_fov
from projects.genesis_ue_sync.tracking.robot_kinematic_mask.config import RobotKinematicMaskConfig
from projects.genesis_ue_sync.tracking.robot_kinematic_mask.occlusion import (
    RobotMaskOcclusionConfig,
    apply_rgb_heatmap_occlusion_to_robot_mask,
)
from projects.genesis_ue_sync.tracking.tracking_skeleton_overlay import project_world_points_to_pixels
from projects.genesis_ue_sync.urdf import (
    compose_link_visual_world_transform,
    compute_link_world_transforms,
    parse_urdf_model,
)


def _resolve_urdf_mesh_path(urdf_path: Path, mesh_filename: str) -> Path:
    raw = str(mesh_filename).strip()
    if raw.startswith("package://"):
        rest = raw[len("package://") :]
        idx = rest.find("/")
        raw = rest[idx + 1 :] if idx >= 0 else rest
    return (urdf_path.parent / raw).resolve()


def _horizontal_fov_deg(*, fx: float, width: int) -> float:
    half = max(float(width) * 0.5, 1e-6)
    return float(math.degrees(2.0 * math.atan(half / max(float(fx), 1e-6))))


def _scale_intrinsics(K: np.ndarray, *, from_wh: tuple[int, int], to_wh: tuple[int, int]) -> np.ndarray:
    k = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    fw, fh = int(from_wh[0]), int(from_wh[1])
    tw, th = int(to_wh[0]), int(to_wh[1])
    if fw <= 0 or fh <= 0 or (fw, fh) == (tw, th):
        return k
    sx = float(tw) / float(fw)
    sy = float(th) / float(fh)
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k


def _extract_ue_image_meta(meta: dict[str, Any]) -> tuple[int, int, float | None]:
    intr = dict(meta.get("intrinsics") or {})
    width = int(intr.get("width") or meta.get("width") or 0)
    height = int(intr.get("height") or meta.get("height") or 0)
    fov_raw = intr.get("fov_degrees", intr.get("fov_deg"))
    fov_deg = float(fov_raw) if fov_raw is not None else None
    return width, height, fov_deg


@dataclass
class LinkMeshGeometry:
    link_name: str
    vertices_local: np.ndarray
    faces: np.ndarray


@dataclass
class RobotMeshLibrary:
    urdf_path: Path
    link_meshes: dict[str, LinkMeshGeometry] = field(default_factory=dict)

    @classmethod
    def from_urdf(cls, urdf_path: Path) -> "RobotMeshLibrary":
        model = parse_urdf_model(urdf_path)
        meshes: dict[str, LinkMeshGeometry] = {}
        for link_name, link in model.links.items():
            if not link.visual_mesh:
                continue
            dae_path = _resolve_urdf_mesh_path(Path(urdf_path), str(link.visual_mesh))
            if not dae_path.is_file():
                continue
            baked = bake_collada_mesh(dae_path)
            verts = np.asarray(baked.positions, dtype=np.float64)
            faces = np.asarray(baked.faces, dtype=np.int64) - 1
            if verts.size == 0 or faces.size == 0:
                continue
            if int(faces.max(initial=-1)) >= int(verts.shape[0]) or int(faces.min(initial=0)) < 0:
                continue
            meshes[link_name] = LinkMeshGeometry(link_name=link_name, vertices_local=verts, faces=faces)
        return cls(urdf_path=Path(urdf_path), link_meshes=meshes)


def compare_ue_intrinsics_to_calibration(
    calibration: CalibrationBundle,
    metadata_by_camera: dict[str, dict[str, Any]],
    *,
    fov_tolerance_deg: float = 0.75,
    fx_tolerance_px: float = 3.0,
) -> dict[str, Any]:
    report: dict[str, Any] = {"cameras": {}, "ok": True, "issues": []}
    for camera_id, meta in metadata_by_camera.items():
        try:
            cam = calibration.camera(camera_id)
        except KeyError:
            report["issues"].append(f"unknown camera_id in metadata: {camera_id}")
            report["ok"] = False
            continue
        ue_w, ue_h, ue_fov = _extract_ue_image_meta(meta)
        rgb_w, rgb_h = int(cam.width), int(cam.height)
        if ue_w > 0 and ue_h > 0:
            rgb_w, rgb_h = ue_w, ue_h
        K_cal = _scale_intrinsics(
            cam.intrinsics,
            from_wh=(int(cam.width), int(cam.height)),
            to_wh=(rgb_w, rgb_h),
        )
        fov_cal = _horizontal_fov_deg(fx=float(K_cal[0, 0]), width=rgb_w)
        entry: dict[str, Any] = {
            "image_size_cal": [int(cam.width), int(cam.height)],
            "image_size_ue_meta": [ue_w, ue_h],
            "K_calibration": K_cal.tolist(),
            "fov_deg_from_calibration_fx": fov_cal,
            "fov_deg_ue_meta": ue_fov,
        }
        if ue_fov is not None and ue_w > 0 and ue_h > 0:
            K_ue = build_intrinsics_from_fov(width=ue_w, height=ue_h, fov_deg=float(ue_fov))
            fx_diff = float(abs(K_ue[0, 0] - K_cal[0, 0]))
            fy_diff = float(abs(K_ue[1, 1] - K_cal[1, 1]))
            cx_diff = float(abs(K_ue[0, 2] - K_cal[0, 2]))
            cy_diff = float(abs(K_ue[1, 2] - K_cal[1, 2]))
            fov_diff = float(abs(float(ue_fov) - fov_cal))
            entry.update(
                {
                    "K_from_ue_fov": K_ue.tolist(),
                    "delta_fx_px": fx_diff,
                    "delta_fy_px": fy_diff,
                    "delta_cx_px": cx_diff,
                    "delta_cy_px": cy_diff,
                    "delta_fov_deg": fov_diff,
                    "intrinsics_match": (
                        fx_diff <= fx_tolerance_px
                        and fy_diff <= fx_tolerance_px
                        and cx_diff <= fx_tolerance_px
                        and cy_diff <= fx_tolerance_px
                        and fov_diff <= fov_tolerance_deg
                    ),
                }
            )
            if not entry["intrinsics_match"]:
                report["ok"] = False
                report["issues"].append(
                    f"{camera_id}: UE fov={ue_fov:.3f} vs cal fov={fov_cal:.3f} "
                    f"(fx diff {fx_diff:.2f}px, fov diff {fov_diff:.2f} deg)"
                )
        else:
            entry["intrinsics_match"] = None
            entry["note"] = "UE metadata missing fov_degrees; skipped K comparison."
        report["cameras"][camera_id] = entry
    return report


@dataclass(frozen=True)
class RobotKinematicMaskFrameResult:
    views_rgb: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    intrinsics_report: dict[str, Any]
    joint_positions: list[float]
    image_corrections: dict[str, CameraImageCorrection]


class RobotKinematicMasker:
    """FK + mesh projection + rasterization only (no I/O side effects)."""

    def __init__(
        self,
        *,
        calibration: CalibrationBundle,
        config: RobotKinematicMaskConfig,
    ) -> None:
        self.calibration = calibration
        self.config = config
        scene = calibration.scene_spec
        if scene is None or scene.robot is None:
            raise RuntimeError("robot_kinematic_mask requires scene_spec with robot block.")
        self._robot = scene.robot
        self._urdf_path = Path(self._robot.resolved_urdf_path)
        self._urdf_model = parse_urdf_model(self._urdf_path)
        self._mesh_library = RobotMeshLibrary.from_urdf(self._urdf_path)
        self._joint_positions = [float(v) for v in self._robot.joint_positions]
        self._base_pos_m = tuple(float(v) for v in self._robot.base_pos)
        self._base_quat_xyzw = self._robot.base_quat_xyzw
        self._visual_basis = config.visual_basis_rpy_deg
        self._occlusion_cfg = RobotMaskOcclusionConfig.from_dict(config.occlusion)
        self._previous_heatmaps: dict[str, np.ndarray] = {}

    def set_previous_heatmaps(self, heatmaps_by_camera: dict[str, np.ndarray] | None) -> None:
        self._previous_heatmaps = {
            str(k): np.asarray(v, dtype=np.float32).copy() for k, v in dict(heatmaps_by_camera or {}).items()
        }

    @property
    def joint_positions(self) -> list[float]:
        return list(self._joint_positions)

    def set_joint_positions(self, joint_positions: list[float] | np.ndarray) -> None:
        self._joint_positions = [float(v) for v in np.asarray(joint_positions, dtype=np.float64).reshape(-1)]

    def _link_world_vertices(self) -> dict[str, np.ndarray]:
        fk = compute_link_world_transforms(
            urdf_path=self._urdf_path,
            base_pos_m=self._base_pos_m,
            base_quat_xyzw=self._base_quat_xyzw,
            joint_positions=self._joint_positions,
        )
        out: dict[str, np.ndarray] = {}
        for link_name, mesh in self._mesh_library.link_meshes.items():
            link = self._urdf_model.links.get(link_name)
            if link is None:
                continue
            lw = fk.get(link_name)
            if lw is None:
                continue
            vw = compose_link_visual_world_transform(
                lw,
                visual_origin_xyz=link.visual_origin_xyz,
                visual_origin_rpy=link.visual_origin_rpy,
                visual_basis_rpy_deg=self._visual_basis,
            )
            r = vw[:3, :3]
            t = vw[:3, 3]
            out[link_name] = (r @ mesh.vertices_local.T).T + t.reshape(1, 3)
        return out

    def _project_vertices(
        self,
        vertices_world: np.ndarray,
        *,
        camera_from_world: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        uv, valid = project_world_points_to_pixels(vertices_world, camera_from_world, intrinsics)
        R = np.asarray(camera_from_world, dtype=np.float64).reshape(4, 4)[:3, :3]
        t = np.asarray(camera_from_world, dtype=np.float64).reshape(4, 4)[:3, 3]
        X = np.asarray(vertices_world, dtype=np.float64).reshape(-1, 3)
        z_cam = (R @ X.T).T[:, 2] + float(t[2])
        return uv, valid, z_cam

    def rasterize_robot_mask(
        self,
        *,
        image_hw: tuple[int, int],
        camera_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        h, w = int(image_hw[0]), int(image_hw[1])
        mask = np.zeros((h, w), dtype=np.uint8)
        depth = np.full((h, w), np.inf, dtype=np.float32)
        cam = self.calibration.camera(camera_id)
        K = _scale_intrinsics(
            cam.intrinsics,
            from_wh=(int(cam.width), int(cam.height)),
            to_wh=(w, h),
        )
        link_verts = self._link_world_vertices()
        ops: list[tuple[float, np.ndarray, float]] = []
        for link_name, verts_world in link_verts.items():
            mesh = self._mesh_library.link_meshes[link_name]
            uv, valid, z_cam = self._project_vertices(
                verts_world,
                camera_from_world=cam.camera_from_world,
                intrinsics=K,
            )
            faces = mesh.faces
            n_verts = int(verts_world.shape[0])
            stride = max(1, int(self.config.face_stride))
            for idx in range(0, int(faces.shape[0]), stride):
                tri = faces[idx].astype(np.int64)
                i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
                if min(i, j, k) < 0 or max(i, j, k) >= n_verts:
                    continue
                if not (valid[i] and valid[j] and valid[k]):
                    continue
                z_min = min(float(z_cam[i]), float(z_cam[j]), float(z_cam[k]))
                if z_min <= 1e-4:
                    continue
                pts = np.asarray(
                    [
                        [float(uv[i, 0]), float(uv[i, 1])],
                        [float(uv[j, 0]), float(uv[j, 1])],
                        [float(uv[k, 0]), float(uv[k, 1])],
                    ],
                    dtype=np.float32,
                )
                if not np.all(np.isfinite(pts)):
                    continue
                pts = pts.astype(np.float32)
                span = float(np.max(pts[:, 0]) - np.min(pts[:, 0]) + np.max(pts[:, 1]) - np.min(pts[:, 1]))
                if span > float(self.config.max_triangle_px):
                    continue
                poly = np.round(pts).astype(np.int32)
                if np.unique(poly, axis=0).shape[0] < 3:
                    continue
                tri_depth = float(np.mean([z_cam[i], z_cam[j], z_cam[k]]))
                ops.append((tri_depth, poly, tri_depth))
        ops.sort(key=lambda item: item[0], reverse=True)
        for _sort_depth, poly, z_val in ops:
            cv2.fillConvexPoly(mask, poly, color=255, lineType=cv2.LINE_AA)
            canvas = np.full((h, w), np.inf, dtype=np.float32)
            cv2.fillConvexPoly(canvas, poly, color=float(z_val), lineType=cv2.LINE_AA)
            closer = canvas < depth
            depth[closer] = canvas[closer]
        if bool(self._occlusion_cfg.enable):
            heatmap = self._previous_heatmaps.get(str(camera_id))
            mask, _stats = apply_rgb_heatmap_occlusion_to_robot_mask(
                mask,
                heatmap=heatmap,
                config=self._occlusion_cfg,
            )
        margin = int(self.config.margin_px)
        if margin > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask, depth

    def apply_mask_to_rgb(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.asarray(rgb, dtype=np.uint8).copy()
        m = np.asarray(mask, dtype=np.uint8) > 0
        fill = int(np.clip(self.config.fill_value, 0, 255))
        out[m] = fill
        return out

    def mask_views_rgb(
        self,
        views_rgb: dict[str, np.ndarray],
        *,
        metadata_by_camera: dict[str, dict[str, Any]] | None = None,
    ) -> RobotKinematicMaskFrameResult:
        masked: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        corrections: dict[str, CameraImageCorrection] = {}
        for camera_id, rgb in views_rgb.items():
            arr = np.asarray(rgb, dtype=np.uint8)
            h, w = int(arr.shape[0]), int(arr.shape[1])
            corrections[camera_id] = CameraImageCorrection(
                reason=f"{camera_id}: projection on OpenCV axes (RGB corrected upstream)",
            )
            mask, _depth = self.rasterize_robot_mask(image_hw=(h, w), camera_id=camera_id)
            masks[camera_id] = mask
            masked[camera_id] = self.apply_mask_to_rgb(arr, mask)
        intr_report: dict[str, Any] = {}
        if metadata_by_camera:
            intr_report = compare_ue_intrinsics_to_calibration(
                self.calibration,
                metadata_by_camera,
                fov_tolerance_deg=float(self.config.fov_tolerance_deg),
                fx_tolerance_px=float(self.config.fx_tolerance_px),
            )
        return RobotKinematicMaskFrameResult(
            views_rgb=masked,
            masks=masks,
            intrinsics_report=intr_report,
            joint_positions=list(self._joint_positions),
            image_corrections=corrections,
        )
