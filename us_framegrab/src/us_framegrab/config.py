"""Load / save frame-grabber YAML (crop box, HDMI device, ZMQ endpoints)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PUB_BIND = "tcp://127.0.0.1:17359"
DEFAULT_CAPTURE_TOPIC = "amongus_camera_frame_v1"
DEFAULT_PREVIEW_TOPIC = "amongus_camera_preview_v1"
DEFAULT_CAMERA_NAME = "us_img"
DEFAULT_SOURCE_ID = "realus.us_framegrab"


def _box_from_mapping(raw: Any, fallback: list[int]) -> list[int]:
    if isinstance(raw, dict):
        return [
            int(raw.get("x0", fallback[0])),
            int(raw.get("x1", fallback[1])),
            int(raw.get("y0", fallback[2])),
            int(raw.get("y1", fallback[3])),
        ]
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return [int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])]
    return list(fallback)


def _box_to_mapping(box: list[int]) -> dict[str, int]:
    return {"x0": int(box[0]), "x1": int(box[1]), "y0": int(box[2]), "y1": int(box[3])}


def clamp_cbox(cbox: list[int], width: int, height: int) -> list[int]:
    """Keep [x0, x1, y0, y1] inside the frame with a 2-pixel minimum span."""
    w = max(int(width), 2)
    h = max(int(height), 2)
    x0, x1, y0, y1 = (int(v) for v in cbox)
    x0 = min(max(x0, 0), w - 2)
    x1 = min(max(x1, x0 + 2), w)
    y0 = min(max(y0, 0), h - 2)
    y1 = min(max(y1, y0 + 2), h)
    return [x0, x1, y0, y1]


@dataclass
class FrameGrabConfig:
    path: Path
    color: bool = False
    hflip: bool = False
    frame_id: str = "us_prob"
    init_cbox: list[int] = field(default_factory=lambda: [550, 1650, 150, 920])
    final_cbox: list[int] = field(default_factory=lambda: [559, 1611, 115, 920])
    capture_backend: str = "ffmpeg"
    frame_width: int = 1920
    frame_height: int = 1080
    publish_rate: float = 60.0
    compressed_quality: int = 80
    time_offset: float = 0.0
    video_device_path: str = ""
    video_index: int = 0
    auto_crop_on_startup: bool = True
    pub_bind: str = DEFAULT_PUB_BIND
    capture_topic: str = DEFAULT_CAPTURE_TOPIC
    preview_topic: str = DEFAULT_PREVIEW_TOPIC
    preview_max_width: int = 960
    preview_jpeg_quality: int = 72
    camera_name: str = DEFAULT_CAMERA_NAME
    source_id: str = DEFAULT_SOURCE_ID
    session_id: str = "realus_us_framegrab"
    machine: str = "sonoscape_e2"

    @property
    def use_fixed_device(self) -> bool:
        return bool(str(self.video_device_path).strip())

    def to_yaml_dict(self) -> dict[str, Any]:
        return {
            "color": bool(self.color),
            "hflip": bool(self.hflip),
            "frame_id": str(self.frame_id),
            "final_cbox": _box_to_mapping(self.final_cbox),
            "init_cbox": _box_to_mapping(self.init_cbox),
            "capture_backend": str(self.capture_backend),
            "frame_width": int(self.frame_width),
            "frame_height": int(self.frame_height),
            "publish_rate": float(self.publish_rate),
            "compressed_quality": int(self.compressed_quality),
            "time_offset": float(self.time_offset),
            "video_device_path": str(self.video_device_path),
            "video_index": int(self.video_index),
            "auto_crop_on_startup": bool(self.auto_crop_on_startup),
            "pub_bind": str(self.pub_bind),
            "capture_topic": str(self.capture_topic),
            "preview_topic": str(self.preview_topic),
            "preview_max_width": int(self.preview_max_width),
            "preview_jpeg_quality": int(self.preview_jpeg_quality),
            "camera_name": str(self.camera_name),
            "source_id": str(self.source_id),
            "session_id": str(self.session_id),
            "machine": str(self.machine),
        }

    def save(self, path: Path | None = None) -> Path:
        dest = Path(path) if path is not None else self.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            yaml.safe_dump(self.to_yaml_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        self.path = dest
        return dest


def default_config_path() -> Path:
    env = Path(__file__).resolve().parents[2]
    return env / "configs" / "config.yaml"


def load_config(path: Path | None = None) -> FrameGrabConfig:
    cfg_path = Path(path) if path is not None else default_config_path()
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {cfg_path}")
    init_cbox = _box_from_mapping(payload.get("init_cbox"), [550, 1650, 150, 920])
    final_cbox = _box_from_mapping(payload.get("final_cbox"), [559, 1611, 115, 920])
    return FrameGrabConfig(
        path=cfg_path,
        color=bool(payload.get("color", False)),
        hflip=bool(payload.get("hflip", False)),
        frame_id=str(payload.get("frame_id", "us_prob")),
        init_cbox=init_cbox,
        final_cbox=final_cbox,
        capture_backend=str(payload.get("capture_backend", "ffmpeg")).strip() or "ffmpeg",
        frame_width=int(payload.get("frame_width", 1920)),
        frame_height=int(payload.get("frame_height", 1080)),
        publish_rate=float(payload.get("publish_rate", 60.0)),
        compressed_quality=int(payload.get("compressed_quality", 80)),
        time_offset=float(payload.get("time_offset", 0.0)),
        video_device_path=str(payload.get("video_device_path", "") or "").strip(),
        video_index=int(payload.get("video_index", 0)),
        auto_crop_on_startup=bool(payload.get("auto_crop_on_startup", True)),
        pub_bind=str(payload.get("pub_bind", DEFAULT_PUB_BIND)),
        capture_topic=str(payload.get("capture_topic", DEFAULT_CAPTURE_TOPIC)),
        preview_topic=str(payload.get("preview_topic", DEFAULT_PREVIEW_TOPIC) or ""),
        preview_max_width=int(payload.get("preview_max_width", 960)),
        preview_jpeg_quality=int(payload.get("preview_jpeg_quality", 72)),
        camera_name=str(payload.get("camera_name", DEFAULT_CAMERA_NAME)),
        source_id=str(payload.get("source_id", DEFAULT_SOURCE_ID)),
        session_id=str(payload.get("session_id", "realus_us_framegrab")),
        machine=str(payload.get("machine", "sonoscape_e2") or "sonoscape_e2"),
    )
