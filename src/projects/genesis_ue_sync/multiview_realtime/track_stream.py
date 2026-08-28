"""ZMQ JSON contract for live multiview track poses (Genesis subscriber)."""

from __future__ import annotations

import os
from typing import Any, TypedDict

from projects.genesis_ue_sync.sim_platform.timebase import host_clock_fields


def _with_clock(out: dict[str, Any], timestamp_ns: int | None) -> dict[str, Any]:
    source = int(timestamp_ns) if timestamp_ns else None
    out.update(host_clock_fields(source_time_ns=source))
    return out

TOPIC_MULTIVIEW_TRACK_V1 = "amongus_multiview_track_v1"
TOPIC_ANATOMY_ASSET_V1 = "amongus_anatomy_asset_v1"
DEFAULT_TRACK_PUB_BIND = "tcp://127.0.0.1:5598"
DEFAULT_ANATOMY_ASSET_PUB_BIND = "tcp://127.0.0.1:5601"

_TRACK_SUBSCRIBE_DISABLED = frozenset({"0", "false", "no", "off", ""})


def resolve_track_subscribe_connect(
    *,
    cli_url: str = "",
    env_var: str = "AMONGUS_GENESIS_TRACK_SUBSCRIBE",
    default_url: str = DEFAULT_TRACK_PUB_BIND,
    use_default: bool = True,
) -> str | None:
    """Resolve ZMQ connect URL for Genesis track overlay (optional; GT works without publisher)."""
    for raw in (str(cli_url or "").strip(), str(os.environ.get(env_var, "") or "").strip()):
        if not raw:
            continue
        if raw.lower() in _TRACK_SUBSCRIBE_DISABLED:
            return None
        return raw
    if use_default:
        return str(default_url)
    return None


def resolve_anatomy_asset_subscribe_connect(
    *,
    cli_url: str = "",
    env_var: str = "AMONGUS_GENESIS_ANATOMY_ASSET_SUBSCRIBE",
    default_url: str = DEFAULT_ANATOMY_ASSET_PUB_BIND,
    use_default: bool = True,
) -> str | None:
    """Resolve ZMQ connect URL for optional anatomy asset control messages."""
    return resolve_track_subscribe_connect(
        cli_url=cli_url,
        env_var=env_var,
        default_url=default_url,
        use_default=use_default,
    )


class MultiviewTrackKeypoints3dV1(TypedDict, total=False):
    schema_version: int
    payload_kind: str
    frame_index: int
    timestamp_ns: int
    keypoints3d: list[list[float]]
    keypoints3d_schema: str
    translation_m: list[float]


def track_keypoints3d_to_dict(
    *,
    frame_index: int,
    timestamp_ns: int,
    keypoints3d: Any,
    schema: str = "body25",
    translation_m: Any = None,
) -> dict[str, Any]:
    """Serialize filtered triangulated 3D joints (J,4) for the Genesis skeleton overlay."""
    import numpy as np

    kp = np.asarray(keypoints3d, dtype=np.float32).reshape(-1, 4)
    out: dict[str, Any] = {
        "schema_version": 2,
        "payload_kind": "keypoints3d",
        "frame_index": int(frame_index),
        "timestamp_ns": int(timestamp_ns),
        "keypoints3d": [[float(v) for v in row] for row in kp.tolist()],
        "keypoints3d_schema": str(schema),
    }
    if translation_m is not None:
        out["translation_m"] = [float(v) for v in np.asarray(translation_m, dtype=np.float32).reshape(3).tolist()]
    return _with_clock(out, timestamp_ns)


class MultiviewTrackPoseV1(TypedDict, total=False):
    schema_version: int
    payload_kind: str
    frame_index: int
    timestamp_ns: int
    pose_aa: list[float]
    betas: list[float]
    translation_m: list[float]


def track_clear_to_dict(
    *,
    frame_index: int,
    timestamp_ns: int,
    reason: str = "",
) -> dict[str, Any]:
    """Tell the Genesis subscriber to remove the live orange SMPL overlay."""
    out = {
        "schema_version": 3,
        "payload_kind": "clear",
        "frame_index": int(frame_index),
        "timestamp_ns": int(timestamp_ns),
        "reason": str(reason),
    }
    return _with_clock(out, timestamp_ns)


def track_pose_to_dict(
    *,
    frame_index: int,
    timestamp_ns: int,
    pose_aa: Any,
    betas: Any,
    translation_m: Any = None,
) -> dict[str, Any]:
    """Serialize SMPL pose (axis-angle 72 + betas) for the Genesis mesh overlay."""
    import numpy as np

    out: dict[str, Any] = {
        "schema_version": 3,
        "payload_kind": "smpl_pose",
        "frame_index": int(frame_index),
        "timestamp_ns": int(timestamp_ns),
        "pose_aa": [float(v) for v in np.asarray(pose_aa, dtype=np.float32).reshape(-1).tolist()],
        "betas": [float(v) for v in np.asarray(betas, dtype=np.float32).reshape(-1).tolist()],
    }
    if translation_m is not None:
        out["translation_m"] = [float(v) for v in np.asarray(translation_m, dtype=np.float32).reshape(3).tolist()]
    return _with_clock(out, timestamp_ns)


def track_mesh_vertices_to_dict(
    *,
    frame_index: int,
    timestamp_ns: int,
    vertices: Any,
    faces: Any,
    translation_m: Any = None,
    mesh_schema: str = "smplx_vertices",
    Rh: Any = None,
    Th: Any = None,
    poses: Any = None,
    smplx_pose_aa_165: Any = None,
    betas: Any = None,
    gender: str | None = None,
    shape_hash: str | None = None,
    pose_hash: str | None = None,
) -> dict[str, Any]:
    """Serialize a fitted mesh directly for Genesis display (SMPL-X or any vertex mesh)."""
    import numpy as np

    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    tri = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    out: dict[str, Any] = {
        "schema_version": 5,
        "payload_kind": "mesh_vertices",
        "frame_index": int(frame_index),
        "timestamp_ns": int(timestamp_ns),
        "mesh_schema": str(mesh_schema),
        "vertices": [[float(v) for v in row] for row in verts.tolist()],
        "faces": [[int(v) for v in row] for row in tri.tolist()],
    }
    if translation_m is not None:
        out["translation_m"] = [float(v) for v in np.asarray(translation_m, dtype=np.float32).reshape(3).tolist()]
    if Rh is not None:
        out["Rh"] = [float(v) for v in np.asarray(Rh, dtype=np.float32).reshape(3).tolist()]
    if Th is not None:
        out["Th"] = [float(v) for v in np.asarray(Th, dtype=np.float32).reshape(3).tolist()]
    if poses is not None:
        out["poses"] = [float(v) for v in np.asarray(poses, dtype=np.float32).reshape(-1).tolist()]
    if smplx_pose_aa_165 is not None:
        pose = np.asarray(smplx_pose_aa_165, dtype=np.float32).reshape(-1)
        if pose.size != 165:
            raise ValueError(f"smplx_pose_aa_165 must contain 165 values, got {pose.size}")
        out["smplx_pose_aa_165"] = [float(v) for v in pose.tolist()]
    if betas is not None:
        out["betas"] = [float(v) for v in np.asarray(betas, dtype=np.float32).reshape(-1)[:10].tolist()]
    if gender is not None:
        out["gender"] = str(gender).lower()
    if shape_hash is not None:
        out["shape_hash"] = str(shape_hash)
    if pose_hash is not None:
        out["pose_hash"] = str(pose_hash)
    return _with_clock(out, timestamp_ns)


def anatomy_asset_control_to_dict(
    *,
    action: str,
    model_id: str = "patient_anatomy",
    asset_npz: str | None = None,
    color_rgba: Any = None,
    visible: bool | None = None,
    opacity: float | None = None,
    mode: str | None = None,
    timestamp_ns: int = 0,
) -> dict[str, Any]:
    """Serialize an anatomy asset lifecycle/display control message for Genesis."""
    import numpy as np

    out: dict[str, Any] = {
        "schema_version": 1,
        "payload_kind": "anatomy_asset",
        "action": str(action),
        "model_id": str(model_id),
        "timestamp_ns": int(timestamp_ns),
    }
    if asset_npz is not None:
        out["asset_npz"] = str(asset_npz)
    if color_rgba is not None:
        rgba = np.asarray(color_rgba, dtype=np.float32).reshape(4)
        out["color_rgba"] = [float(v) for v in rgba.tolist()]
    if visible is not None:
        out["visible"] = bool(visible)
    if opacity is not None:
        out["opacity"] = float(max(0.0, min(1.0, float(opacity))))
    if mode is not None:
        out["mode"] = str(mode)
    return _with_clock(out, int(timestamp_ns) if int(timestamp_ns) else None)
