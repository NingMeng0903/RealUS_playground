"""Shared JSON metadata contracts for ZMQ multipart streams and future ROS2 bridges.

ROS2 nodes map Image/CameraInfo into the same logical fields as CameraFrame metadata."""

from __future__ import annotations

from typing import Any, TypedDict

TOPIC_CANONICAL_SCENE_V1 = "amongus_canonical_v1"
TOPIC_CAMERA_FRAME_V1 = "amongus_camera_frame_v1"
TOPIC_CAMERA_PREVIEW_V1 = "amongus_camera_preview_v1"
TOPIC_SCENE_INIT_V1 = "amongus_scene_init_v1"


class CameraFrameMetadataV1(TypedDict, total=False):
    schema_version: int
    session_id: str
    source_id: str
    camera_name: str
    frame_index: int
    sim_time_ns: int
    wall_time_ns: int
    source_time_ns: int
    encoding: str
    width: int
    height: int
    intrinsics: dict[str, Any]
    extrinsics: dict[str, Any]


def camera_frame_metadata_template() -> CameraFrameMetadataV1:
    return {
        "schema_version": 1,
        "session_id": "",
        "source_id": "ue.realtime_capture",
        "camera_name": "",
        "frame_index": 0,
        "sim_time_ns": 0,
        "wall_time_ns": 0,
        "source_time_ns": 0,
        "encoding": "jpeg",
        "width": 0,
        "height": 0,
    }


class CanonicalSceneMetadataV1(TypedDict, total=False):
    schema_version: int
    session_id: str
    frame_index: int
    sim_time_ns: int
    wall_time_ns: int
