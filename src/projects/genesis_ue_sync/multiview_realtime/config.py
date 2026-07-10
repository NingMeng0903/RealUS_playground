from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.project import project_paths


def _expand_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    text = os.path.expandvars(str(raw)).strip()
    if not text:
        return None
    return project_paths(__file__).resolve_from_root(text)


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Config file is empty: {path}")
    if text.startswith("{"):
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML required for YAML configs.") from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in config: {path}")
    return payload


@dataclass(frozen=True)
class IngressConfig:
    connect: str = "tcp://127.0.0.1:17356"
    topic: str = "amongus_camera_frame_v1"
    recv_timeout_ms: int = 250
    sync_tolerance_frames: int = 2
    max_buffer_per_camera: int = 8

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "IngressConfig":
        payload = dict(payload or {})
        return cls(
            connect=str(payload.get("connect", cls.connect)),
            topic=str(payload.get("topic", cls.topic)),
            recv_timeout_ms=int(payload.get("recv_timeout_ms", cls.recv_timeout_ms)),
            sync_tolerance_frames=int(payload.get("sync_tolerance_frames", cls.sync_tolerance_frames)),
            max_buffer_per_camera=int(payload.get("max_buffer_per_camera", cls.max_buffer_per_camera)),
        )


@dataclass(frozen=True)
class GenesisOverlayConfig:
    backend: str = "cuda"
    show_viewer: bool = True
    show_fps: bool = True
    spawn_bed: bool = True
    spawn_robot: bool = False
    track_mesh_rgba: tuple[int, int, int, int] = (250, 122, 31, 132)
    inference_every_n_synced_frames: int = 1
    max_track_fps: float = 8.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GenesisOverlayConfig":
        payload = dict(payload or {})
        rgba = payload.get("track_mesh_rgba", list(cls.track_mesh_rgba))
        return cls(
            backend=str(payload.get("backend", cls.backend)),
            show_viewer=bool(payload.get("show_viewer", cls.show_viewer)),
            show_fps=bool(payload.get("show_fps", cls.show_fps)),
            spawn_bed=bool(payload.get("spawn_bed", cls.spawn_bed)),
            spawn_robot=bool(payload.get("spawn_robot", cls.spawn_robot)),
            track_mesh_rgba=tuple(int(v) for v in rgba),
            inference_every_n_synced_frames=max(1, int(payload.get("inference_every_n_synced_frames", 1))),
            max_track_fps=float(payload.get("max_track_fps", cls.max_track_fps)),
        )


@dataclass(frozen=True)
class MultiviewRealtimeConfig:
    calibration_path: Path
    scene_spec_path: Path | None
    camera_ids: tuple[str, ...]
    primary_camera_id: str
    ingress: IngressConfig
    pose_backend: dict[str, Any]
    world_reconstruction: dict[str, Any]
    robot_kinematic_mask: dict[str, Any]
    genesis: GenesisOverlayConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MultiviewRealtimeConfig":
        cal_path = _expand_path(payload.get("calibration_path"))
        if cal_path is None:
            raise ValueError("calibration_path is required.")
        scene_path = _expand_path(payload.get("scene_spec_path"))
        camera_ids = tuple(str(v) for v in payload.get("camera_ids") or ())
        if not camera_ids:
            raise ValueError("camera_ids must list at least one camera.")
        pose_backend = dict(payload.get("pose_backend") or {})
        primary = str(
            payload.get("primary_camera_id")
            or pose_backend.get("primary_camera_id")
            or camera_ids[0]
        )
        if not pose_backend:
            pose_backend = {"type": "unsupported_missing_pose_backend"}
        pose_backend.setdefault("type", "unsupported_missing_pose_backend")
        pose_backend.setdefault("primary_camera_id", primary)
        return cls(
            calibration_path=cal_path,
            scene_spec_path=scene_path,
            camera_ids=camera_ids,
            primary_camera_id=primary,
            ingress=IngressConfig.from_dict(payload.get("ingress")),
            pose_backend=pose_backend,
            world_reconstruction=dict(payload.get("world_reconstruction") or {}),
            robot_kinematic_mask=dict(payload.get("robot_kinematic_mask") or {}),
            genesis=GenesisOverlayConfig.from_dict(payload.get("genesis")),
        )

    @classmethod
    def load(cls, path: Path) -> "MultiviewRealtimeConfig":
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return cls.from_dict(_load_yaml_or_json(resolved))


def resolve_live_track_config_path(
    repo_root: Path,
    *,
    cli_path: Path | None = None,
    disabled: bool = False,
    show_viewer: bool = True,
) -> Path | None:
    """Resolve live-track YAML for amass_bed_capsule_demo (auto-on by default when config exists)."""
    if disabled or not show_viewer:
        return None
    raw_env = str(os.environ.get("AMONGUS_GENESIS_MULTIVIEW_TRACK", "0") or "0").strip().lower()
    if raw_env in ("0", "false", "no", "off"):
        return None
    raw = (
        cli_path
        or os.environ.get("AMONGUS_GENESIS_MULTIVIEW_TRACK_CONFIG")
        or "configs/tracking/multiview_realtime_dwpose_triangulation.yaml"
    )
    if raw in (None, ""):
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    if not path.is_file():
        return None
    return path
