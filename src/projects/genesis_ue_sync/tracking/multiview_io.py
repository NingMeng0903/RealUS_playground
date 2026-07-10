from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.project import project_paths
from projects.genesis_ue_sync.tracking.types import CameraViewFrame, MultiViewHumanRecoveryRequest


_FRAME_INDEX_RE = re.compile(r"_(\d+)\.png$")


def _resolve_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    text = os.path.expandvars(str(raw)).strip()
    if not text:
        return None
    return project_paths(__file__).resolve_from_root(text)


def _positive_png_frames(sequence_dir: Path) -> list[tuple[int, Path]]:
    ordered: list[tuple[int, Path]] = []
    for path in sorted(sequence_dir.glob("*.png")):
        if "_-" in path.name:
            continue
        match = _FRAME_INDEX_RE.search(path.name)
        if match is None:
            continue
        ordered.append((int(match.group(1)), path))
    return ordered


@dataclass(frozen=True)
class CameraSequence:
    camera_id: str
    sequence_name: str
    png_dir: Path
    mp4_path: Path | None = None
    frames: list[tuple[int, Path]] = field(default_factory=list)


@dataclass(frozen=True)
class MultiViewFrameSet:
    sequence_id: str
    fps: float
    views: dict[str, list[CameraViewFrame]]
    frame_indices: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_request(self) -> MultiViewHumanRecoveryRequest:
        return MultiViewHumanRecoveryRequest(
            sequence_id=self.sequence_id,
            views=self.views,
            metadata=dict(self.metadata),
        )

    @property
    def frame_count(self) -> int:
        return len(self.frame_indices)


def load_run_meta(path: str | Path) -> dict[str, Any]:
    meta_path = project_paths(__file__).resolve_from_root(path)
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in run meta: {meta_path}")
    payload["_path"] = str(meta_path)
    return payload


def camera_sequences_from_run_meta(path: str | Path) -> list[CameraSequence]:
    payload = load_run_meta(path)
    png_dirs = payload.get("sequence_png_dirs", {})
    mp4s = payload.get("sequence_mp4s", {})
    sequences: list[CameraSequence] = []
    for sequence_name in payload.get("sequence_names", []):
        png_dir = _resolve_path(png_dirs.get(sequence_name))
        if png_dir is None or not png_dir.is_dir():
            raise FileNotFoundError(f"PNG directory missing for sequence '{sequence_name}': {png_dir}")
        camera_id = f"cam_{sequence_name.split('_cam_')[-1]}" if "_cam_" in sequence_name else sequence_name
        mp4_path = _resolve_path(mp4s.get(sequence_name))
        sequences.append(
            CameraSequence(
                camera_id=camera_id,
                sequence_name=str(sequence_name),
                png_dir=png_dir,
                mp4_path=mp4_path if mp4_path is not None and mp4_path.exists() else None,
                frames=_positive_png_frames(png_dir),
            )
        )
    if not sequences:
        raise ValueError(f"No camera sequences found in run meta: {path}")
    return sequences


def build_multiview_request_from_run_meta(
    path: str | Path,
    *,
    fps: float = 30.0,
    max_frames: int | None = None,
    start_frame: int = 0,
    frame_step: int = 1,
    include_camera_ids: list[str] | None = None,
) -> MultiViewFrameSet:
    sequences = camera_sequences_from_run_meta(path)
    if include_camera_ids:
        requested = [str(cid) for cid in include_camera_ids]
        requested_set = set(requested)
        sequences = [seq for seq in sequences if str(seq.camera_id) in requested_set]
        missing = [cid for cid in requested if cid not in {str(seq.camera_id) for seq in sequences}]
        if missing:
            raise KeyError(f"Requested camera ids missing from run meta {path}: {missing}")
        sequences = sorted(sequences, key=lambda seq: requested.index(str(seq.camera_id)))
    if not sequences:
        raise ValueError(f"No camera sequences selected from run meta: {path}")
    frame_step = max(1, int(frame_step))
    start_frame = max(0, int(start_frame))
    shared_indices = sorted(set.intersection(*[set(idx for idx, _ in seq.frames) for seq in sequences]))
    shared_indices = shared_indices[start_frame::frame_step]
    if max_frames is not None and max_frames > 0:
        shared_indices = shared_indices[: int(max_frames)]
    if not shared_indices:
        raise RuntimeError(f"No synchronized frame indices found in run meta: {path}")
    views: dict[str, list[CameraViewFrame]] = {}
    seq_name_root = sequences[0].sequence_name.split("_cam_")[0]
    for seq in sequences:
        frame_map = {idx: png_path for idx, png_path in seq.frames}
        camera_frames: list[CameraViewFrame] = []
        for idx in shared_indices:
            png_path = frame_map[idx]
            camera_frames.append(
                CameraViewFrame(
                    camera_id=seq.camera_id,
                    image_path=png_path,
                    timestamp_ns=int((idx / max(float(fps), 1e-6)) * 1e9),
                    metadata={
                        "frame_idx": int(idx),
                        "sequence_name": seq.sequence_name,
                        "png_dir": str(seq.png_dir),
                        "mp4_path": str(seq.mp4_path) if seq.mp4_path is not None else None,
                    },
                )
            )
        views[seq.camera_id] = camera_frames
    return MultiViewFrameSet(
        sequence_id=seq_name_root,
        fps=float(fps),
        views=views,
        frame_indices=[int(idx) for idx in shared_indices],
        metadata={
            "run_meta_path": str(project_paths(__file__).resolve_from_root(path)),
            "camera_ids": [seq.camera_id for seq in sequences],
            "fps": float(fps),
        },
    )


__all__ = [
    "CameraSequence",
    "MultiViewFrameSet",
    "build_multiview_request_from_run_meta",
    "camera_sequences_from_run_meta",
    "load_run_meta",
]
