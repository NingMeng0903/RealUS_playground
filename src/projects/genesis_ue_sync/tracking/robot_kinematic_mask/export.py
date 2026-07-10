from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from projects.genesis_ue_sync.tracking.robot_kinematic_mask.config import RobotKinematicMaskExportConfig

_LOG = logging.getLogger(__name__)


@dataclass
class RobotKinematicMaskExportResult:
    frame_index: int
    paths: list[Path] = field(default_factory=list)
    manifest_path: Path | None = None


class RobotKinematicMaskExporter:
    """Write per-camera mask diagnostics to disk for manual alignment checks."""

    def __init__(self, config: RobotKinematicMaskExportConfig) -> None:
        self.config = config
        self._export_count = 0

    @property
    def output_root(self) -> Path:
        root = self.config.output_root
        if root is None:
            raise RuntimeError("robot_kinematic_mask.export.output_root is not configured.")
        return root.expanduser().resolve()

    def should_export(self) -> bool:
        return bool(self.config.enable) and self._export_count < int(self.config.max_frames)

    def export_frame(
        self,
        *,
        frame_index: int,
        views_rgb: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
        masked_rgb: dict[str, np.ndarray],
        joint_positions: list[float] | None,
        intrinsics_report: dict[str, Any] | None,
        image_corrections: dict[str, Any] | None = None,
    ) -> RobotKinematicMaskExportResult | None:
        if not self.should_export():
            return None
        frame_dir = self.output_root / f"frame_{int(frame_index):06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        result = RobotKinematicMaskExportResult(frame_index=int(frame_index))

        overlays_for_tile: list[np.ndarray] = []
        per_camera: dict[str, Any] = {}

        for camera_id in sorted(views_rgb):
            cam_dir = frame_dir / str(camera_id)
            cam_dir.mkdir(parents=True, exist_ok=True)
            orig = np.asarray(views_rgb[camera_id], dtype=np.uint8)
            mask = np.asarray(masks[camera_id], dtype=np.uint8)
            masked = np.asarray(masked_rgb[camera_id], dtype=np.uint8)
            mask_bool = mask > 0

            overlay = orig.copy()
            overlay[mask_bool] = (255, 64, 64)

            cam_paths: dict[str, str] = {}
            if self.config.save_original:
                path = cam_dir / "01_original.png"
                Image.fromarray(orig).save(path)
                result.paths.append(path)
                cam_paths["original"] = str(path)
            if self.config.save_overlay:
                path = cam_dir / "02_mask_overlay.png"
                Image.fromarray(overlay).save(path)
                result.paths.append(path)
                cam_paths["overlay"] = str(path)
                overlays_for_tile.append(overlay)
            if self.config.save_masked:
                path = cam_dir / "03_masked.png"
                Image.fromarray(masked).save(path)
                result.paths.append(path)
                cam_paths["masked"] = str(path)
            if self.config.save_mask_binary:
                path = cam_dir / "04_mask_binary.png"
                Image.fromarray(np.where(mask_bool, 255, 0).astype(np.uint8)).save(path)
                result.paths.append(path)
                cam_paths["mask_binary"] = str(path)
            if self.config.save_original and self.config.save_masked:
                path = cam_dir / "05_side_by_side_original_masked.png"
                side = np.concatenate([orig, masked], axis=1)
                Image.fromarray(side).save(path)
                result.paths.append(path)
                cam_paths["side_by_side"] = str(path)

            per_camera[camera_id] = {
                "paths": cam_paths,
                "mask_pixels": int(mask_bool.sum()),
                "image_hw": [int(orig.shape[0]), int(orig.shape[1])],
                "image_correction": dict((image_corrections or {}).get(camera_id, {})),
            }

        if self.config.save_tiled and overlays_for_tile:
            tiled = np.concatenate(overlays_for_tile, axis=1)
            path = frame_dir / "tiled_overlay_all_cameras.png"
            Image.fromarray(tiled).save(path)
            result.paths.append(path)

        manifest = {
            "frame_index": int(frame_index),
            "joint_positions": [float(v) for v in joint_positions] if joint_positions else None,
            "intrinsics_report": intrinsics_report or {},
            "cameras": per_camera,
        }
        manifest_path = frame_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        result.manifest_path = manifest_path
        result.paths.append(manifest_path)

        intr_path = self.output_root / "intrinsics_report.json"
        if intrinsics_report and not intr_path.is_file():
            intr_path.write_text(json.dumps(intrinsics_report, indent=2), encoding="utf-8")
            result.paths.append(intr_path)

        self._export_count += 1
        if self.config.log_paths:
            _LOG.info(
                "robot_kinematic_mask exported frame=%s -> %s (%s files)",
                frame_index,
                frame_dir,
                len(result.paths),
            )
        return result
