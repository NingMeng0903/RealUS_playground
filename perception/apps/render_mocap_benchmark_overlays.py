#!/usr/bin/env python3
"""Render predicted SMPL-X meshes on the strict four-camera UE recording.

The renderer consumes explicit world-space mesh vertices and faces.  It never
reconstructs a mesh from GT joints, pose, or betas.  A JSONL prediction can
point at the ``index.json`` emitted by ``run_hybrik_benchmark`` when its rows
contain joints only; the corresponding exported ``smplx_result.npz`` must
contain a real ``vertices``/``faces`` mesh.  ``--sequence-dir`` reads the same
NPZ contract directly.

The CPU fallback is a small triangle rasterizer with a per-pixel z buffer.  It
uses the recording's full K and ``world_from_camera`` values and writes all
artifacts below the requested output directory, leaving the recording intact.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

import numpy as np


SMPLX55_NAMES: tuple[str, ...] = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw", "left_eye_smplhf",
    "right_eye_smplhf", "left_index1", "left_index2", "left_index3", "left_middle1",
    "left_middle2", "left_middle3", "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3", "left_thumb1", "left_thumb2", "left_thumb3",
    "right_index1", "right_index2", "right_index3", "right_middle1", "right_middle2",
    "right_middle3", "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1",
    "right_ring2", "right_ring3", "right_thumb1", "right_thumb2", "right_thumb3",
)
SMPLX55_PARENTS: tuple[int, ...] = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21,
    40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
)


@dataclass(frozen=True)
class CameraContract:
    name: str
    K: np.ndarray
    distortion: np.ndarray
    world_from_camera: np.ndarray
    camera_from_world: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class RecordingRow:
    frame_index: int
    image_paths: Mapping[str, Path]
    gt_path: Path | None
    record: Mapping[str, Any]


@dataclass(frozen=True)
class MeshPrediction:
    frame_index: int
    vertices_world_m: np.ndarray
    faces: np.ndarray
    joints_world_m: np.ndarray | None
    source: str
    metadata: Mapping[str, Any]


def _json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc


def _as_finite_matrix(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite with shape {shape}, got {arr.shape}")
    return arr


def load_calibration(recording_dir: Path) -> dict[str, CameraContract]:
    """Load the actual recorder calibration without silently changing K."""
    path = recording_dir / "calibration.json"
    payload = _json_load(path)
    cameras = payload.get("cameras") if isinstance(payload, dict) else None
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError(f"{path} has no cameras object")
    out: dict[str, CameraContract] = {}
    for name, raw in cameras.items():
        if not isinstance(raw, dict):
            raise ValueError(f"camera {name!r} calibration is not an object")
        K = _as_finite_matrix(raw.get("K"), (3, 3), f"camera {name} K")
        wfc = _as_finite_matrix(raw.get("world_from_camera"), (4, 4), f"camera {name} world_from_camera")
        if not np.allclose(wfc[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
            raise ValueError(f"camera {name} world_from_camera has invalid homogeneous row")
        try:
            cfw = np.linalg.inv(wfc)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"camera {name} world_from_camera is singular") from exc
        distortion = np.asarray(raw.get("distortion", [0.0] * 5), dtype=np.float64).reshape(-1)
        if distortion.size not in (4, 5, 8, 12, 14) or not np.all(np.isfinite(distortion)):
            raise ValueError(f"camera {name} distortion must be finite OpenCV coefficients")
        width = int(raw.get("width", 0))
        height = int(raw.get("height", 0))
        if width <= 0 or height <= 0:
            raise ValueError(f"camera {name} has invalid image size {width}x{height}")
        out[str(name)] = CameraContract(
            name=str(name), K=K, distortion=distortion, world_from_camera=wfc,
            camera_from_world=cfw, width=width, height=height,
        )
    if len(out) not in (4, 5):
        raise ValueError(f"benchmark overlay requires four or five cameras, found {sorted(out)}")
    return out


def load_recording(recording_dir: Path, cameras: Mapping[str, CameraContract]) -> dict[int, RecordingRow]:
    manifest = recording_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"recording manifest not found: {manifest}")
    rows: dict[int, RecordingRow] = {}
    camera_names = tuple(cameras)
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid recording manifest line {line_no}: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("status", "ok") != "ok":
            continue
        frame = int(raw["frame_index"])
        image_payload = raw.get("image_paths")
        if not isinstance(image_payload, dict) or set(image_payload) != set(camera_names):
            raise ValueError(f"recording frame {frame} camera set does not match calibration")
        image_paths = {name: (recording_dir / str(image_payload[name])).resolve() for name in camera_names}
        gt_value = raw.get("gt_path")
        gt_path = (recording_dir / str(gt_value)).resolve() if gt_value else None
        if frame in rows:
            raise ValueError(f"duplicate successful recording frame {frame}")
        rows[frame] = RecordingRow(frame, image_paths, gt_path, raw)
    if not rows:
        raise ValueError(f"recording manifest has no successful frames: {manifest}")
    return rows


def _validate_faces(vertices: np.ndarray, faces: Any, label: str) -> np.ndarray:
    raw = np.asarray(faces)
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.shape[0] < 1:
        raise ValueError(f"{label} faces must have shape (F,3), got {raw.shape}")
    if not np.issubdtype(raw.dtype, np.integer) and not np.allclose(raw, np.round(raw)):
        raise ValueError(f"{label} faces contain non-integral indices")
    arr = np.asarray(raw, dtype=np.int64)
    if int(arr.min()) < 0 or int(arr.max()) >= int(vertices.shape[0]):
        raise ValueError(f"{label} faces index outside {vertices.shape[0]} vertices")
    return arr


def _validate_mesh(vertices: Any, faces: Any, label: str) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(vertices, dtype=np.float32)
    if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] < 3 or not np.all(np.isfinite(verts)):
        raise ValueError(f"{label} vertices_world_m must be finite (N,3), got {verts.shape}")
    return verts, _validate_faces(verts, faces, label)


def _optional_joints(value: Any, label: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (55, 3) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} joints_world_m must be finite (55,3), got {arr.shape}")
    return arr


def _load_npz_mesh(path: Path, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not path.is_file():
        raise ValueError(f"mesh NPZ missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"vertices", "faces"}
            if not required.issubset(data.files):
                raise ValueError(f"{label} NPZ must contain vertices and faces, keys={data.files}")
            verts, faces = _validate_mesh(data["vertices"], data["faces"], label)
            joints_value = next(
                (data[key] for key in ("joints_world_m", "native55_joints_world_m", "joints_world_m55") if key in data.files),
                None,
            )
            joints = _optional_joints(joints_value, label) if joints_value is not None else None
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"cannot load {label} mesh NPZ {path}: {exc}") from exc
    return verts, faces, joints


def load_sequence_predictions(sequence_dir: Path) -> tuple[list[MeshPrediction], list[dict[str, Any]]]:
    index_path = sequence_dir / "index.json"
    payload = _json_load(index_path)
    rows = payload.get("index") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"sequence index has no index list: {index_path}")
    predictions: list[MeshPrediction] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        try:
            frame = int(row["frame_index"])
            moment = Path(str(row["moment_dir"]))
            if not moment.is_absolute():
                moment = sequence_dir / moment
            npz = moment / "smplx_result.npz"
            verts, faces, joints = _load_npz_mesh(npz, f"sequence frame {frame}")
            predictions.append(MeshPrediction(frame, verts, faces, joints, str(npz), dict(row)))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            rejected.append({"frame_index": row.get("frame_index"), "reason": str(exc), "source": str(sequence_dir)})
    return predictions, rejected


def _sidecar_mesh_for_frame(prediction_path: Path, frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str] | None:
    sidecar = prediction_path.parent / "index.json"
    if not sidecar.is_file():
        return None
    payload = _json_load(sidecar)
    rows = payload.get("index") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return None
    for row in rows:
        try:
            if int(row.get("frame_index")) != frame_index:
                continue
            moment = Path(str(row["moment_dir"]))
            if not moment.is_absolute():
                moment = prediction_path.parent / moment
            npz = moment / "smplx_result.npz"
            verts, faces, joints = _load_npz_mesh(npz, f"sidecar frame {frame_index}")
            return verts, faces, joints, str(npz)
        except (KeyError, TypeError, ValueError, OSError):
            continue
    return None


def load_jsonl_predictions(prediction_path: Path) -> tuple[list[MeshPrediction], list[dict[str, Any]]]:
    predictions: list[MeshPrediction] = []
    rejected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line_no, line in enumerate(prediction_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("row is not a JSON object")
            frame = int(row["frame_index"])
            if frame in seen:
                raise ValueError(f"duplicate prediction frame {frame}")
            seen.add(frame)
            if row.get("status", "ok") != "ok":
                raise ValueError(str(row.get("reason") or "prediction status is not ok"))
            human = row
            if not row.get("vertices_world_m") and isinstance(row.get("humans"), list) and row["humans"]:
                human = row["humans"][0]
            vertices = human.get("vertices_world_m") if isinstance(human, dict) else None
            faces = human.get("faces") if isinstance(human, dict) else None
            joints_value = human.get("joints_world_m") if isinstance(human, dict) else None
            source = str(prediction_path)
            sidecar = None
            if vertices is None or faces is None:
                sidecar = _sidecar_mesh_for_frame(prediction_path, frame)
            if sidecar is not None:
                verts, face_array, sidecar_joints, source = sidecar
                joints = _optional_joints(joints_value, f"prediction line {line_no}") if joints_value is not None else sidecar_joints
            else:
                if vertices is None or faces is None:
                    raise ValueError("prediction has no explicit vertices_world_m/faces and no sidecar mesh NPZ")
                verts, face_array = _validate_mesh(vertices, faces, f"prediction line {line_no}")
                joints = _optional_joints(joints_value, f"prediction line {line_no}")
            predictions.append(MeshPrediction(frame, verts, face_array, joints, source, row))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            frame_value = None
            try:
                frame_value = json.loads(line).get("frame_index")
            except Exception:
                pass
            rejected.append({"line": line_no, "frame_index": frame_value, "reason": str(exc), "source": str(prediction_path)})
    return predictions, rejected


def _parse_frame_indices(values: list[str] | None) -> set[int] | None:
    if not values:
        return None
    result: set[int] = set()
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                result.add(int(token))
    return result


def _load_gt_joints(gt_path: Path | None) -> np.ndarray | None:
    if gt_path is None or not gt_path.is_file():
        return None
    payload = _json_load(gt_path)
    names = payload.get("joint_names")
    bones = payload.get("bones")
    if list(names or []) != list(SMPLX55_NAMES) or not isinstance(bones, list):
        return None
    by_name = {str(item.get("joint_name")): item for item in bones if isinstance(item, dict)}
    if any(name not in by_name for name in SMPLX55_NAMES):
        return None
    try:
        cm = np.asarray([by_name[name]["world_location_cm"] for name in SMPLX55_NAMES], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if cm.shape != (55, 3) or not np.all(np.isfinite(cm)):
        return None
    # Recorder GT is UE left-handed centimetres; prediction meshes use Genesis
    # right-handed metres, exactly as perception.benchmark.evaluation does.
    return (cm * np.asarray([0.01, -0.01, 0.01], dtype=np.float64)).astype(np.float32)


def _project(vertices_world_m: np.ndarray, camera: CameraContract) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts = np.asarray(vertices_world_m, dtype=np.float64)
    cfw = camera.camera_from_world
    xyz = (cfw[:3, :3] @ verts.T).T + cfw[:3, 3]
    z = xyz[:, 2]
    valid = np.isfinite(xyz).all(axis=1) & (z > 1e-6)
    uv = np.full((verts.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        if np.any(np.abs(camera.distortion) > 1e-12):
            import cv2
            projected, _ = cv2.projectPoints(
                xyz[valid].astype(np.float64), np.zeros(3), np.zeros(3), camera.K, camera.distortion,
            )
            uv[valid] = projected.reshape(-1, 2)
        else:
            uv[valid, 0] = camera.K[0, 0] * xyz[valid, 0] / z[valid] + camera.K[0, 2]
            uv[valid, 1] = camera.K[1, 1] * xyz[valid, 1] / z[valid] + camera.K[1, 2]
    return uv, xyz, valid


def _render_mesh_zbuffer(
    image_bgr: np.ndarray,
    vertices_world_m: np.ndarray,
    faces: np.ndarray,
    camera: CameraContract,
    *,
    alpha: float,
    rgb: tuple[int, int, int],
    face_stride: int,
    max_triangle_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rasterize real triangles into a CPU depth buffer and alpha composite."""
    image = np.asarray(image_bgr, dtype=np.uint8)
    h, w = image.shape[:2]
    uv, xyz, valid = _project(vertices_world_m, camera)
    depth = np.full((h, w), np.inf, dtype=np.float64)
    color = np.zeros((h, w, 3), dtype=np.float32)
    face_count = 0
    stride = max(int(face_stride), 1)
    for face in faces[::stride]:
        i, j, k = (int(v) for v in face)
        if not (valid[i] and valid[j] and valid[k]):
            continue
        pts = uv[[i, j, k]]
        if not np.all(np.isfinite(pts)):
            continue
        xmin = max(0, int(np.floor(np.min(pts[:, 0]))))
        xmax = min(w - 1, int(np.ceil(np.max(pts[:, 0]))))
        ymin = max(0, int(np.floor(np.min(pts[:, 1]))))
        ymax = min(h - 1, int(np.ceil(np.max(pts[:, 1]))))
        if xmax <= xmin or ymax <= ymin:
            continue
        if (xmax - xmin) + (ymax - ymin) > float(max_triangle_px):
            continue
        x0, y0 = pts[0]
        x1, y1 = pts[1]
        x2, y2 = pts[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denominator)) < 1e-9:
            continue
        xs = np.arange(xmin, xmax + 1, dtype=np.float64) + 0.5
        ys = np.arange(ymin, ymax + 1, dtype=np.float64) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)
        w0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
        w1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
        if not np.any(inside):
            continue
        z0, z1, z2 = xyz[[i, j, k], 2]
        tri_depth = w0 * z0 + w1 * z1 + w2 * z2
        depth_roi = depth[ymin : ymax + 1, xmin : xmax + 1]
        closer = inside & (tri_depth > 1e-6) & (tri_depth < depth_roi)
        if not np.any(closer):
            continue
        p0, p1, p2 = xyz[[i, j, k]]
        normal = np.cross(p1 - p0, p2 - p0)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-10:
            continue
        normal = normal / normal_norm
        light = np.asarray([-0.35, -0.45, -1.0], dtype=np.float64)
        light /= np.linalg.norm(light)
        intensity = 0.28 + 0.72 * max(0.0, float(np.dot(normal, light)))
        # Internal arrays are BGR because cv2.imread/cv2.imwrite are BGR.
        face_color = np.asarray(rgb[::-1], dtype=np.float32) * intensity
        depth_roi[closer] = tri_depth[closer]
        color[ymin : ymax + 1, xmin : xmax + 1][closer] = face_color
        face_count += 1
    visible = np.isfinite(depth)
    out = image.astype(np.float32).copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    out[visible] = out[visible] * (1.0 - a) + color[visible] * a
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    visible_vertices = valid & np.all(np.isfinite(uv), axis=1)
    stats = {
        "visible_vertex_ratio": float(np.mean(visible_vertices.astype(np.float32))),
        "visible_vertex_count": int(np.sum(visible_vertices)),
        "rasterized_face_count": int(face_count),
        "depth_pixel_count": int(np.sum(visible)),
        "depth_min_m": float(np.min(depth[visible])) if np.any(visible) else None,
        "depth_max_m": float(np.max(depth[visible])) if np.any(visible) else None,
        "camera_space_vertex_count": int(np.sum(valid)),
    }
    return out, stats


def _draw_bones(image_bgr: np.ndarray, joints_world_m: np.ndarray, camera: CameraContract, *, color_bgr: tuple[int, int, int], width: int = 2) -> tuple[np.ndarray, int]:
    import cv2
    uv, _xyz, valid = _project(joints_world_m, camera)
    out = np.asarray(image_bgr, dtype=np.uint8).copy()
    h, w = out.shape[:2]
    count = 0
    for child, parent in enumerate(SMPLX55_PARENTS):
        if parent < 0 or not (valid[child] and valid[parent]):
            continue
        p0, p1 = uv[parent], uv[child]
        if not np.all(np.isfinite([p0, p1])):
            continue
        cv2.line(out, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), color_bgr, int(width), cv2.LINE_AA)
        count += 1
    for point in uv:
        if np.all(np.isfinite(point)) and 0 <= point[0] < w and 0 <= point[1] < h:
            cv2.circle(out, tuple(np.round(point).astype(int)), max(1, width), color_bgr, -1, cv2.LINE_AA)
    return out, count


def _metric_row(prediction: MeshPrediction, target: np.ndarray | None) -> dict[str, float] | None:
    if prediction.joints_world_m is None or target is None:
        return None
    predicted = np.asarray(prediction.joints_world_m, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    world = np.linalg.norm(predicted - gt, axis=1) * 1000.0
    relative = np.linalg.norm((predicted - predicted[0]) - (gt - gt[0]), axis=1) * 1000.0
    return {
        "world_body22_mpjpe_mm": float(np.mean(world[:22])),
        "pelvis_relative_body22_mpjpe_mm": float(np.mean(relative[:22])),
        "world_hands30_mpjpe_mm": float(np.mean(world[25:])),
        "pelvis_error_mm": float(world[0]),
    }


def _make_montage(images: list[tuple[str, np.ndarray]]) -> np.ndarray:
    import cv2
    if not images:
        raise ValueError("cannot build empty montage")
    h, w = images[0][1].shape[:2]
    thumb_w, thumb_h = max(1, w // 2), max(1, h // 2)
    panels: list[np.ndarray] = []
    for name, image in images:
        panel = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (thumb_w - 1, thumb_h - 1), (255, 255, 255), 2)
        cv2.putText(panel, str(name), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, str(name), (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    while len(panels) % 2:
        panels.append(np.zeros_like(panels[0]))
    return np.vstack([np.hstack(panels[index:index + 2])
                      for index in range(0, len(panels), 2)])


# These fields are copied when the prediction/index row carries source
# provenance.  They are identifiers for an already-rendered frame, rather
# than mesh or GT data, and therefore remain useful when exporting an old
# overlay summary without rerunning the renderer.
_SOURCE_ID_KEYS: tuple[str, ...] = (
    "source_id", "input_source_id", "input_id", "source_frame_index",
    "source_timestamp_ns", "source_sim_time_ns", "timestamp_ns", "sim_time_ns",
    "capture_group_id", "capture_id", "session_id", "loop_id", "loop_index", "motion_frame_index",
    "motion_time_ns", "source_motion_fps", "recording_sequence_index",
    "benchmark_time_ns", "benchmark_phase", "source_time_domain",
)


def _source_ids_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-safe source identifiers already present in an accepted row."""
    source_ids: dict[str, Any] = {}
    for key in _SOURCE_ID_KEYS:
        value = row.get(key)
        if value is not None and isinstance(value, (str, int, float, bool, list, dict)):
            source_ids[key] = value
    for container_key in ("input_source_ids", "source_ids", "prediction_metadata"):
        nested = row.get(container_key)
        if isinstance(nested, Mapping):
            for key in _SOURCE_ID_KEYS:
                value = nested.get(key)
                if value is not None and isinstance(value, (str, int, float, bool, list, dict)):
                    source_ids.setdefault(key, value)
        elif nested is not None and container_key == "input_source_ids":
            # Keep a custom scalar/list source ID intact when a producer did
            # not use the mapping form.
            source_ids.setdefault(container_key, nested)
    return source_ids


def _record_input_source_ids(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize recorder-manifest IDs to the names used in video manifests."""
    source_ids = _source_ids_from_row(record)
    if "source_frame_index" not in source_ids and record.get("frame_index") is not None:
        source_ids["source_frame_index"] = record["frame_index"]
    if "source_timestamp_ns" not in source_ids and record.get("timestamp_ns") is not None:
        source_ids["source_timestamp_ns"] = record["timestamp_ns"]
    if "source_sim_time_ns" not in source_ids and record.get("sim_time_ns") is not None:
        source_ids["source_sim_time_ns"] = record["sim_time_ns"]
    return source_ids


def _load_overlay_summary(summary: Mapping[str, Any] | Path) -> tuple[dict[str, Any], Path | None]:
    """Load a summary path or copy an in-memory summary for video export."""
    if isinstance(summary, (str, Path)):
        summary_path = Path(summary).expanduser().resolve()
        payload = _json_load(summary_path)
        if not isinstance(payload, dict):
            raise ValueError(f"overlay summary must be a JSON object: {summary_path}")
        return dict(payload), summary_path
    if not isinstance(summary, Mapping):
        raise TypeError("summary must be a path or mapping")
    return dict(summary), None


def _default_video_output_dir(summary: Mapping[str, Any], summary_path: Path | None) -> Path:
    if summary_path is not None:
        return summary_path.parent / "videos"
    accepted = summary.get("accepted")
    if isinstance(accepted, list) and accepted:
        first = accepted[0]
        if isinstance(first, Mapping):
            paths = first.get("overlay_image_paths")
            if isinstance(paths, Mapping) and paths:
                first_path = Path(str(next(iter(paths.values())))).expanduser().resolve()
                # Renderer output is .../overlays/frame_XXXXXXXX/camN.jpg.
                return first_path.parent.parent.parent / "videos"
    raise ValueError("output_dir is required when summary has no overlay paths")


def export_videos_from_summary(
    summary: Mapping[str, Any] | Path,
    *,
    output_dir: Path | None = None,
    fps: float = 30.0,
) -> dict[str, Any]:
    """Export aligned MP4s from accepted overlay JPEGs in an existing summary.

    The function only decodes and copies the accepted ``overlay_image_paths``
    and ``montage_path`` entries.  It never loads a mesh, GT joints, or the
    recording, so an existing summary can be exported without rerasterizing.
    """
    import cv2

    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be a finite positive number")
    payload, summary_path = _load_overlay_summary(summary)
    accepted = payload.get("accepted")
    if not isinstance(accepted, list) or not accepted:
        raise ValueError("overlay summary has no accepted frames")
    accepted_count = payload.get("accepted_count")
    if accepted_count is not None and int(accepted_count) != len(accepted):
        raise ValueError(
            f"overlay summary accepted_count={accepted_count} does not match {len(accepted)} rows"
        )

    declared_cameras = payload.get("camera_names")
    if declared_cameras is None:
        first_paths = accepted[0].get("overlay_image_paths") if isinstance(accepted[0], Mapping) else None
        if not isinstance(first_paths, Mapping) or not first_paths:
            raise ValueError("overlay summary has no camera_names or overlay_image_paths")
        camera_names = tuple(str(name) for name in first_paths)
    else:
        if not isinstance(declared_cameras, list) or not declared_cameras:
            raise ValueError("overlay summary camera_names must be a non-empty list")
        camera_names = tuple(str(name) for name in declared_cameras)
    if len(set(camera_names)) != len(camera_names):
        raise ValueError("overlay summary camera_names contains duplicates")

    # Validate camera sets and all input image paths before opening writers.
    # This prevents a late camera-set error from producing a misleadingly
    # complete-looking set of videos.
    frame_inputs: list[tuple[int, Mapping[str, Path], Path, dict[str, Any], int]] = []
    seen_frames: set[int] = set()
    camera_sizes: dict[str, tuple[int, int]] = {}
    montage_size: tuple[int, int] | None = None
    for video_frame_index, raw_row in enumerate(accepted):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"accepted row {video_frame_index} is not an object")
        try:
            frame_index = int(raw_row["frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"accepted row {video_frame_index} has invalid frame_index") from exc
        if frame_index in seen_frames:
            raise ValueError(f"duplicate accepted prediction frame_index {frame_index}")
        seen_frames.add(frame_index)
        image_payload = raw_row.get("overlay_image_paths")
        if not isinstance(image_payload, Mapping):
            raise ValueError(f"accepted frame {frame_index} has no overlay_image_paths")
        image_set = {str(name) for name in image_payload}
        if image_set != set(camera_names):
            raise ValueError(
                f"accepted frame {frame_index} camera set {sorted(image_set)} "
                f"does not match {list(camera_names)}"
            )
        image_paths: dict[str, Path] = {}
        for camera in camera_names:
            image_path = Path(str(image_payload[camera])).expanduser().resolve()
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot decode overlay image for {camera}: {image_path}")
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"overlay image for {camera} is not a color image: {image_path}")
            height, width = int(image.shape[0]), int(image.shape[1])
            size = (width, height)
            previous = camera_sizes.setdefault(camera, size)
            if size != previous:
                raise ValueError(
                    f"image resolution changed for {camera}: {size} != {previous}"
                )
            image_paths[camera] = image_path
        montage_value = raw_row.get("montage_path")
        if montage_value is None:
            raise ValueError(f"accepted frame {frame_index} has no montage_path")
        montage_path = Path(str(montage_value)).expanduser().resolve()
        montage = cv2.imread(str(montage_path), cv2.IMREAD_COLOR)
        if montage is None:
            raise ValueError(f"cannot decode montage image: {montage_path}")
        if montage.ndim != 3 or montage.shape[2] != 3:
            raise ValueError(f"montage is not a color image: {montage_path}")
        current_montage_size = (int(montage.shape[1]), int(montage.shape[0]))
        if montage_size is None:
            montage_size = current_montage_size
        elif current_montage_size != montage_size:
            raise ValueError(
                f"montage resolution changed: {current_montage_size} != {montage_size}"
            )
        frame_inputs.append((frame_index, image_paths, montage_path, dict(raw_row), video_frame_index))

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else _default_video_output_dir(payload, summary_path).expanduser().resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers: dict[str, Any] = {}
    video_paths = {camera: destination / f"{camera}.mp4" for camera in camera_names}
    montage_video_path = destination / "montage.mp4"
    frame_manifest: list[dict[str, Any]] = []
    try:
        for frame_index, image_paths, montage_path, raw_row, video_frame_index in frame_inputs:
            for camera in camera_names:
                image = cv2.imread(str(image_paths[camera]), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"cannot decode overlay image for {camera}: {image_paths[camera]}")
                width, height = camera_sizes[camera]
                if (int(image.shape[1]), int(image.shape[0])) != (width, height):
                    raise ValueError(f"image resolution changed for {camera} while writing")
                writer = writers.get(camera)
                if writer is None:
                    writer = cv2.VideoWriter(str(video_paths[camera]), fourcc, fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot open MP4 writer for {video_paths[camera]}")
                    writers[camera] = writer
                writer.write(image)

            montage = cv2.imread(str(montage_path), cv2.IMREAD_COLOR)
            if montage is None:
                raise ValueError(f"cannot decode montage image while writing: {montage_path}")
            assert montage_size is not None
            if (int(montage.shape[1]), int(montage.shape[0])) != montage_size:
                raise ValueError("montage resolution changed while writing")
            if "montage" not in writers:
                writers["montage"] = cv2.VideoWriter(
                    str(montage_video_path), fourcc, fps, montage_size
                )
                if not writers["montage"].isOpened():
                    raise RuntimeError(f"cannot open MP4 writer for {montage_video_path}")
            writers["montage"].write(montage)

            source_ids = _source_ids_from_row(raw_row)
            frame_entry: dict[str, Any] = {
                "video_frame_index": int(video_frame_index),
                "video_time_ns": round(video_frame_index * 1e9 / fps),
                "frame_index": int(frame_index),
                "prediction_frame_index": int(frame_index),
                "prediction_source": raw_row.get("prediction_source"),
                "input_source_ids": source_ids,
                "overlay_image_paths": {name: str(path) for name, path in image_paths.items()},
                "montage_path": str(montage_path),
            }
            # Keep common IDs accessible at the frame level for consumers
            # which do not want to unpack input_source_ids.
            frame_entry.update({key: value for key, value in source_ids.items() if key not in frame_entry})
            frame_manifest.append(frame_entry)
    finally:
        for writer in writers.values():
            writer.release()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_summary": str(summary_path) if summary_path is not None else None,
        "input_source": payload.get("input_source"),
        "mesh_contract": payload.get("mesh_contract"),
        "fps": fps,
        "frame_count": len(frame_inputs),
        "camera_names": list(camera_names),
        "camera_sizes": {name: list(size) for name, size in camera_sizes.items()},
        "montage_size": list(montage_size or (0, 0)),
        "videos": {name: str(path) for name, path in video_paths.items()},
        "montage_video": str(montage_video_path),
        "frames": frame_manifest,
    }
    manifest_path = destination / "video_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


# Name used by callers that think of this as an overlay export operation.
# Keep both public spellings so existing summaries can be exported directly
# without coupling the caller to the CLI implementation.
export_overlay_videos = export_videos_from_summary


def render_overlays(
    *,
    recording_dir: Path,
    output_dir: Path,
    predictions_path: Path | None = None,
    sequence_dir: Path | None = None,
    frame_indices: set[int] | None = None,
    max_samples: int | None = None,
    draw_gt_bones: bool = False,
    mesh_alpha: float = 0.62,
    mesh_rgb: tuple[int, int, int] = (255, 128, 32),
    face_stride: int = 1,
    max_triangle_px: float = 1400.0,
    export_videos: bool = False,
    fps: float = 30.0,
) -> dict[str, Any]:
    if (predictions_path is None) == (sequence_dir is None):
        raise ValueError("provide exactly one of --predictions or --sequence-dir")
    recording_dir = recording_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = load_calibration(recording_dir)
    recording = load_recording(recording_dir, cameras)
    if predictions_path is not None:
        predictions, rejected = load_jsonl_predictions(predictions_path.expanduser().resolve())
        input_source = str(predictions_path.expanduser().resolve())
    else:
        predictions, rejected = load_sequence_predictions(sequence_dir.expanduser().resolve())
        input_source = str(sequence_dir.expanduser().resolve())
    predictions.sort(key=lambda item: item.frame_index)
    if frame_indices is not None:
        predictions = [item for item in predictions if item.frame_index in frame_indices]
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        predictions = predictions[: int(max_samples)]
    if not predictions:
        raise ValueError("no valid mesh predictions selected")
    raw_root = output_dir / "raw"
    overlay_root = output_dir / "overlays"
    montage_root = output_dir / "montages"
    raw_root.mkdir(exist_ok=True)
    overlay_root.mkdir(exist_ok=True)
    montage_root.mkdir(exist_ok=True)
    accepted: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        rec = recording.get(prediction.frame_index)
        if rec is None:
            rejected.append({"frame_index": prediction.frame_index, "reason": "no recording frame", "source": prediction.source})
            continue
        gt_joints = _load_gt_joints(rec.gt_path)
        metric = _metric_row(prediction, gt_joints)
        if metric is not None:
            evaluation_rows.append({"frame_index": prediction.frame_index, "status": "ok", "joints_world_m": prediction.joints_world_m.tolist()})
        frame_tag = f"frame_{prediction.frame_index:08d}"
        raw_frame_dir = raw_root / frame_tag
        overlay_frame_dir = overlay_root / frame_tag
        raw_frame_dir.mkdir(exist_ok=True)
        overlay_frame_dir.mkdir(exist_ok=True)
        montage_images: list[tuple[str, np.ndarray]] = []
        camera_stats: dict[str, Any] = {}
        outputs: dict[str, str] = {}
        originals: dict[str, str] = {}
        for name, camera in cameras.items():
            import cv2
            image_path = rec.image_paths[name]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                rejected.append({"frame_index": prediction.frame_index, "reason": f"cannot decode image {image_path}", "source": prediction.source})
                montage_images = []
                break
            if image.shape[1] != camera.width or image.shape[0] != camera.height:
                rejected.append({"frame_index": prediction.frame_index, "reason": f"image size {image.shape[1]}x{image.shape[0]} != calibration {camera.width}x{camera.height}", "source": str(image_path)})
                montage_images = []
                break
            raw_out = raw_frame_dir / f"{name}.jpg"
            cv2.imwrite(str(raw_out), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            originals[name] = str(image_path)
            rendered, stats = _render_mesh_zbuffer(
                image, prediction.vertices_world_m, prediction.faces, camera,
                alpha=mesh_alpha, rgb=mesh_rgb, face_stride=face_stride, max_triangle_px=max_triangle_px,
            )
            gt_bone_count = 0
            if draw_gt_bones and gt_joints is not None:
                rendered, gt_bone_count = _draw_bones(rendered, gt_joints, camera, color_bgr=(40, 220, 40), width=2)
            overlay_path = overlay_frame_dir / f"{name}.jpg"
            cv2.imwrite(str(overlay_path), rendered, [cv2.IMWRITE_JPEG_QUALITY, 95])
            outputs[name] = str(overlay_path)
            stats["gt_bone_count"] = int(gt_bone_count)
            camera_stats[name] = stats
            montage_images.append((name, rendered))
        if len(montage_images) != len(cameras):
            continue
        import cv2
        montage_path = montage_root / f"{frame_tag}.jpg"
        cv2.imwrite(str(montage_path), _make_montage(montage_images), [cv2.IMWRITE_JPEG_QUALITY, 95])
        accepted_row = {
            "frame_index": prediction.frame_index,
            "prediction_source": prediction.source,
            "original_image_paths": originals,
            "overlay_image_paths": outputs,
            "montage_path": str(montage_path),
            "camera_stats": camera_stats,
            "metrics": metric,
            "gt_drawn": bool(draw_gt_bones and gt_joints is not None),
            "gt_path": str(rec.gt_path) if rec.gt_path else None,
        }
        source_ids = _record_input_source_ids(rec.record)
        # Prediction metadata can carry source IDs that are not present in
        # the recorder manifest; recorder IDs take precedence for the
        # capture-side frame/time mapping.
        for key, value in _source_ids_from_row(prediction.metadata).items():
            source_ids.setdefault(key, value)
        if source_ids:
            accepted_row["input_source_ids"] = source_ids
        accepted.append(accepted_row)
    eval_summary: dict[str, Any] | None = None
    if evaluation_rows:
        from perception.benchmark.evaluation import evaluate_predictions
        eval_summary = evaluate_predictions(recording_dir, evaluation_rows, method="mesh overlay input")
    metric_rows = [row["metrics"] for row in accepted if row.get("metrics") is not None]
    summary = {
        "schema_version": 1,
        "recording_dir": str(recording_dir),
        "input_source": input_source,
        "prediction_mode": "jsonl" if predictions_path is not None else "sequence_dir",
        "camera_names": list(cameras),
        "renderer": "cpu_triangle_zbuffer",
        "projection": "full calibrated K + world_from_camera; distortion preserved",
        "mesh_contract": "vertices_world_m metres, faces triangle indices; no mesh reconstructed from GT joints",
        "selected_frame_indices": [int(row["frame_index"]) for row in accepted],
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "rejected": rejected,
        "accepted": accepted,
        "metrics": eval_summary,
        "metric_frame_count": len(metric_rows),
        "renderer_config": {
            "mesh_alpha": float(mesh_alpha), "mesh_rgb": list(mesh_rgb),
            "face_stride": int(face_stride), "max_triangle_px": float(max_triangle_px),
            "draw_gt_bones": bool(draw_gt_bones), "export_videos": bool(export_videos),
            "fps": float(fps),
        },
    }
    if export_videos:
        video_summary = export_videos_from_summary(
            summary, output_dir=output_dir / "videos", fps=fps,
        )
        # Keep the overlay summary compact; the complete frame-to-video
        # mapping lives in video_manifest.json.
        summary["video_export"] = {
            key: value for key, value in video_summary.items() if key != "frames"
        }
    (output_dir / "overlay_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in accepted), encoding="utf-8"
    )
    (output_dir / "overlay_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _parse_color(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in str(value).split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise argparse.ArgumentTypeError("mesh color must be R,G,B in [0,255]")
    return tuple(parts)  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path, help="prediction JSONL with explicit world-space mesh vertices/faces")
    source.add_argument("--sequence-dir", type=Path, help="saved sequence directory containing index.json and smplx_result.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-indices", nargs="+", help="selected frame indices, e.g. 15 30 or 15,30")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--draw-gt-bones", action="store_true", help="draw actual recorded UE 55-bone GT in green")
    parser.add_argument("--mesh-alpha", type=float, default=0.62)
    parser.add_argument("--mesh-rgb", type=_parse_color, default=(255, 128, 32), help="predicted mesh color R,G,B")
    parser.add_argument("--face-stride", type=int, default=1, help="rasterize every Nth real mesh triangle")
    parser.add_argument("--max-triangle-px", type=float, default=1400.0)
    parser.add_argument("--export-videos", action="store_true", help="export aligned MP4s from accepted overlay JPEGs")
    parser.add_argument("--fps", type=float, default=30.0, help="video export frame rate")
    args = parser.parse_args()
    if args.face_stride <= 0:
        parser.error("--face-stride must be positive")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be a finite positive number")
    summary = render_overlays(
        recording_dir=args.recording_dir, output_dir=args.output_dir,
        predictions_path=args.predictions, sequence_dir=args.sequence_dir,
        frame_indices=_parse_frame_indices(args.frame_indices), max_samples=args.max_samples,
        draw_gt_bones=args.draw_gt_bones, mesh_alpha=args.mesh_alpha, mesh_rgb=args.mesh_rgb,
        face_stride=args.face_stride, max_triangle_px=args.max_triangle_px,
        export_videos=args.export_videos, fps=args.fps,
    )
    print(json.dumps({key: value for key, value in summary.items() if key not in {"accepted", "rejected"}}, indent=2))
    return 0 if summary["accepted_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
