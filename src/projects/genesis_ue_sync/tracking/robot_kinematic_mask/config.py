from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.project import project_paths


def _parse_visual_basis_rpy_deg(raw: tuple[float, float, float] | list[float] | str | None) -> tuple[float, float, float]:
    if raw is None:
        env = str(os.environ.get("AMONGUS_UE_ROBOT_VISUAL_BASIS_RPY_DEG", "0 0 0")).strip()
        parts = env.replace(",", " ").split()
    elif isinstance(raw, str):
        parts = raw.replace(",", " ").split()
    else:
        parts = [float(v) for v in raw]
    if len(parts) != 3:
        return (0.0, 0.0, 0.0)
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return (0.0, 0.0, 0.0)


def _expand_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return project_paths(__file__).resolve_from_root(text)


@dataclass(frozen=True)
class RobotKinematicMaskExportConfig:
    enable: bool = True
    output_root: Path | None = None
    max_frames: int = 32
    save_original: bool = True
    save_overlay: bool = True
    save_masked: bool = True
    save_mask_binary: bool = True
    save_tiled: bool = True
    log_paths: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RobotKinematicMaskExportConfig":
        payload = dict(payload or {})
        root = _expand_path(payload.get("output_root"))
        if root is None:
            root = project_paths(__file__).root / "outputs" / "tracking_debug" / "62415c" / "robot_mask"
        return cls(
            enable=bool(payload.get("enable", cls.enable)),
            output_root=root,
            max_frames=max(1, int(payload.get("max_frames", cls.max_frames))),
            save_original=bool(payload.get("save_original", cls.save_original)),
            save_overlay=bool(payload.get("save_overlay", cls.save_overlay)),
            save_masked=bool(payload.get("save_masked", cls.save_masked)),
            save_mask_binary=bool(payload.get("save_mask_binary", cls.save_mask_binary)),
            save_tiled=bool(payload.get("save_tiled", cls.save_tiled)),
            log_paths=bool(payload.get("log_paths", cls.log_paths)),
        )


@dataclass(frozen=True)
class RobotKinematicMaskConfig:
    enable: bool = False
    fill_value: int = 0
    margin_px: int = 8
    face_stride: int = 2
    max_triangle_px: float = 520.0
    canonical_connect: str = "tcp://127.0.0.1:5599"
    canonical_topic: str = "amongus_canonical_v1"
    robot_entity_name: str = "robot_main"
    visual_basis_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fov_tolerance_deg: float = 0.75
    fx_tolerance_px: float = 3.0
    image_correction_mode: str = "metadata"
    precorrect_views_rgb: bool = True
    occlusion: dict[str, Any] = field(default_factory=dict)
    export: RobotKinematicMaskExportConfig = RobotKinematicMaskExportConfig()

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RobotKinematicMaskConfig":
        payload = dict(payload or {})
        basis = payload.get("visual_basis_rpy_deg", cls.visual_basis_rpy_deg)
        export_payload = dict(payload.get("export") or {})
        if not export_payload and bool(payload.get("debug_export", False)):
            export_payload = {
                "enable": True,
                "max_frames": int(payload.get("debug_export_max_frames", 8)),
            }
        return cls(
            enable=bool(payload.get("enable", cls.enable)),
            fill_value=int(payload.get("fill_value", cls.fill_value)),
            margin_px=max(0, int(payload.get("margin_px", cls.margin_px))),
            face_stride=max(1, int(payload.get("face_stride", cls.face_stride))),
            max_triangle_px=float(payload.get("max_triangle_px", cls.max_triangle_px)),
            canonical_connect=str(payload.get("canonical_connect", cls.canonical_connect)),
            canonical_topic=str(payload.get("canonical_topic", cls.canonical_topic)),
            robot_entity_name=str(payload.get("robot_entity_name", cls.robot_entity_name)),
            visual_basis_rpy_deg=_parse_visual_basis_rpy_deg(basis),
            fov_tolerance_deg=float(payload.get("fov_tolerance_deg", cls.fov_tolerance_deg)),
            fx_tolerance_px=float(payload.get("fx_tolerance_px", cls.fx_tolerance_px)),
            image_correction_mode=str(payload.get("image_correction_mode", cls.image_correction_mode)),
            precorrect_views_rgb=bool(payload.get("precorrect_views_rgb", cls.precorrect_views_rgb)),
            occlusion=dict(payload.get("occlusion") or {}),
            export=RobotKinematicMaskExportConfig.from_dict(export_payload),
        )
