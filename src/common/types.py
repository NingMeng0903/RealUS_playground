from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any, Literal


BackendFamily = Literal["single_view_offline", "multi_view_offline"]
PoseRepresentation = Literal["skeleton_2d", "skeleton_3d", "smpl"]


@dataclass
class VideoInput:
    video_path: Path
    fps: float | None = None
    camera_id: str = "cam0"
    max_frames: int | None = None

    def __post_init__(self) -> None:
        self.video_path = Path(self.video_path)


@dataclass
class PoseOutputPaths:
    frames_dir: Path
    poses2d_dir: Path
    smpl_dir: Path
    world_dir: Path | None = None
    metrics_dir: Path | None = None
    debug_dir: Path | None = None

    def __post_init__(self) -> None:
        self.frames_dir = Path(self.frames_dir)
        self.poses2d_dir = Path(self.poses2d_dir)
        self.smpl_dir = Path(self.smpl_dir)
        self.world_dir = Path(self.world_dir) if self.world_dir is not None else None
        self.metrics_dir = Path(self.metrics_dir) if self.metrics_dir is not None else None
        self.debug_dir = Path(self.debug_dir) if self.debug_dir is not None else None


@dataclass
class PoseBackendConfig:
    name: str = "gvhmr"
    repo_dir: Path | None = None
    device: str | None = None
    detector: str = "vitdet"
    batch_size: int = 1
    save_mesh: bool = True
    side_view: bool = False
    top_view: bool = False
    full_frame: bool = False
    extra_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.repo_dir = Path(self.repo_dir) if self.repo_dir is not None else None


@dataclass
class BackendOutputLayout:
    sequence_id: str
    backend_name: str
    root_dir: Path
    raw_dir: Path
    world_dir: Path
    debug_dir: Path
    metrics_dir: Path
    manifest_path: Path
    faces_path: Path
    selection_path: Path


@dataclass
class PoseExtractionRequest:
    backend_family: BackendFamily
    representation: PoseRepresentation
    input_video: VideoInput
    outputs: PoseOutputPaths
    backend_config: PoseBackendConfig = field(default_factory=PoseBackendConfig)
    sequence_id: str | None = None
    preferred_person_id: int = 0

    def resolved_sequence_id(self) -> str:
        if self.sequence_id:
            return self.sequence_id
        return self.input_video.video_path.stem


@dataclass
class MotionFrameRecord:
    frame_idx: int
    frame_stem: str
    timestamp_s: float
    person_id: int
    raw_smpl_path: str | None = None
    world_smpl_path: str | None = None
    raw_mesh_path: str | None = None
    debug_image_path: str | None = None
    confidence: float | None = None
    contact_scores: dict[str, float] = field(default_factory=dict)
    penetration_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MotionSequenceManifest:
    sequence_id: str
    camera_id: str
    fps: float
    representation: PoseRepresentation
    backend_name: str
    source_video: str
    frames: list[MotionFrameRecord]
    manifest_schema_version: str = "1.0"
    world_from_camera: list[list[float]] | None = None
    faces_path: str | None = None
    metrics_path: str | None = None
    selection_path: str | None = None
    notes: list[str] = field(default_factory=list)
    sequence_metrics: dict[str, float] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frames"] = [asdict(frame) for frame in self.frames]
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MotionSequenceManifest:
        merged = dict(payload)
        if "manifest_schema_version" not in merged and "schema_version" in merged:
            merged["manifest_schema_version"] = merged["schema_version"]
        merged.setdefault("manifest_schema_version", "1.0")
        if "sequence_id" not in merged and "sequence_name" in merged:
            merged["sequence_id"] = merged["sequence_name"]
        if "camera_id" not in merged:
            merged["camera_id"] = "cam0"
        if "representation" not in merged and "model_type" in merged:
            merged["representation"] = merged["model_type"]
        if "backend_name" not in merged and "source_dataset" in merged:
            merged["backend_name"] = merged["source_dataset"]
        if "source_video" not in merged:
            merged["source_video"] = str(merged.get("source_path", ""))
        fps = float(merged.get("fps", 30.0))

        frame_fields = {field_.name for field_ in dataclass_fields(MotionFrameRecord)}
        frames = []
        for frame_payload in merged["frames"]:
            frame_payload = dict(frame_payload)
            frame_idx = int(frame_payload.get("frame_idx", len(frames)))
            frame_payload.setdefault("timestamp_s", frame_idx / fps if fps > 0 else 0.0)
            frame_payload.setdefault("person_id", 0)
            extra_frame_fields = {key: value for key, value in frame_payload.items() if key not in frame_fields}
            sanitized_frame_payload = {key: value for key, value in frame_payload.items() if key in frame_fields}
            if extra_frame_fields:
                metadata = dict(sanitized_frame_payload.get("metadata", {}))
                metadata.setdefault("extensions", {}).update(extra_frame_fields)
                sanitized_frame_payload["metadata"] = metadata
            frames.append(MotionFrameRecord(**sanitized_frame_payload))
        merged["frames"] = frames
        valid_fields = {field_.name for field_ in dataclass_fields(cls)}
        extra_manifest_fields = {key: value for key, value in merged.items() if key not in valid_fields}
        merged = {key: value for key, value in merged.items() if key in valid_fields}
        if extra_manifest_fields:
            merged.setdefault("extensions", {}).update(extra_manifest_fields)
        return cls(**merged)

    @classmethod
    def load(cls, path: Path) -> MotionSequenceManifest:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

