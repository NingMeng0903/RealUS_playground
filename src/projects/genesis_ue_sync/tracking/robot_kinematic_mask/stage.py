from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import SyncedMultiviewFrame
from projects.genesis_ue_sync.multiview_realtime.ingress.canonical_joint_reader import CanonicalJointReader
from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle
from projects.genesis_ue_sync.tracking.robot_kinematic_mask.config import RobotKinematicMaskConfig
from projects.genesis_ue_sync.tracking.robot_kinematic_mask.core import RobotKinematicMasker
from projects.genesis_ue_sync.tracking.robot_kinematic_mask.export import RobotKinematicMaskExporter


@dataclass(frozen=True)
class RobotKinematicMaskStageResult:
    views_rgb: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    intrinsics_report: dict[str, Any]
    joint_positions: list[float]
    export_paths: tuple[Path, ...] = ()


class RobotKinematicMaskStage:
    """Self-contained live stage: canonical joints -> mask -> optional disk export."""

    def __init__(self, *, calibration: CalibrationBundle, config: RobotKinematicMaskConfig) -> None:
        self.config = config
        self._masker = RobotKinematicMasker(calibration=calibration, config=config)
        self._exporter = RobotKinematicMaskExporter(config.export)
        self._canonical_joints = CanonicalJointReader(config)
        try:
            self._canonical_joints.connect()
        except Exception:
            pass

    def close(self) -> None:
        self._canonical_joints.close()

    @property
    def export_output_root(self) -> Path:
        return self._exporter.output_root

    def set_previous_heatmaps(self, heatmaps_by_camera: dict[str, np.ndarray] | None) -> None:
        self._masker.set_previous_heatmaps(heatmaps_by_camera)

    def apply(self, synced: SyncedMultiviewFrame) -> RobotKinematicMaskStageResult:
        self._canonical_joints.poll()
        live_joints = self._canonical_joints.joint_positions
        if live_joints is not None:
            self._masker.set_joint_positions(live_joints)

        mask_result = self._masker.mask_views_rgb(
            synced.views_rgb,
            metadata_by_camera=synced.metadata_by_camera,
        )
        export_paths: list[Path] = []
        export_result = self._exporter.export_frame(
            frame_index=int(synced.frame_index),
            views_rgb=synced.views_rgb,
            masks=mask_result.masks,
            masked_rgb=mask_result.views_rgb,
            joint_positions=mask_result.joint_positions,
            intrinsics_report=mask_result.intrinsics_report,
            image_corrections={
                cid: {
                    **corr.as_dict(),
                    "reason": str(corr.reason),
                }
                for cid, corr in mask_result.image_corrections.items()
            },
        )
        if export_result is not None:
            export_paths.extend(export_result.paths)

        return RobotKinematicMaskStageResult(
            views_rgb=mask_result.views_rgb,
            masks=mask_result.masks,
            intrinsics_report=mask_result.intrinsics_report,
            joint_positions=mask_result.joint_positions,
            export_paths=tuple(export_paths),
        )
