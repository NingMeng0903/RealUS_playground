#!/usr/bin/env python3
"""Audit SMPL-X freeze stability from an already-recorded sequence.

This command only reads a recorded sequence, the protocol JSON, and the
recording manifest/GT files.  It never loads a body model and never changes
the tracking result.  The sequence ``index.json`` is expected to point at
``moment_XXXX/smplx_result.npz`` files containing native 55-joint world-space
coordinates.  The protocol is the small JSON contract used by the freeze
experiment::

    {
      "scenario": "freeze_test",
      "recording_dir": "/absolute/path/to/recording",
      "freeze_source_frame": 55,
      "sample_interval_s": 0.5,
      "rows": [
        {"frame_index": 100, "phase": "motion", "source_frame_index": 54},
        {"frame_index": 101, "phase": "freeze", "source_frame_index": 55}
      ]
    }

``warmup`` rows are accepted for protocol compatibility and are excluded
from all requested-row counts and metrics.  Successful output consists of
``summary.json`` and ``freeze_stability.png`` below ``--output-dir``.  A
sequence with no accepted freeze result still writes both artifacts and is
reported as an explicit failed audit (the CLI exits with status 2).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# Keep this script standalone.  These are the official native SMPL-X-55
# names/order used by the recorder's UE annotations and by the NPZ contract.
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
BODY22_NAMES: tuple[str, ...] = SMPLX55_NAMES[:22]
SELECTED_JOINTS: tuple[tuple[str, int], ...] = (
    ("root", 0),
    ("left_knee4", 4),
    ("right_knee5", 5),
    ("left_ankle7", 7),
    ("right_ankle8", 8),
    ("spine9", 9),
)
JOINT_KEY_PRIORITY: tuple[str, ...] = (
    "native55_joints_world_m",
    "joints_world_m55",
    "joints_world_m",
)


class AuditInputError(ValueError):
    """Raised when the protocol/input contract cannot be evaluated."""


@dataclass
class Prediction:
    """One sequence NPZ load, with optional fields kept independently."""

    joints: np.ndarray | None = None
    pelvis: np.ndarray | None = None
    poses: np.ndarray | None = None
    shapes: np.ndarray | None = None
    vertices: np.ndarray | None = None
    npz_path: Path | None = None
    joint_key: str | None = None
    error: str | None = None
    optional_errors: dict[str, str] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.joints is not None and self.error is None


@dataclass
class EvaluatedRow:
    """A protocol row after sequence, GT, and timeline resolution."""

    order: int
    frame_index: int
    phase: str
    source_frame_index: int
    source_time_s: float
    segment: str
    index_row: Mapping[str, Any] | None
    prediction: Prediction
    gt: np.ndarray | None
    gt_status: str
    gt_error: str | None
    prediction_error: str | None
    sequence_status: str
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.prediction.accepted

    @property
    def gt_accepted(self) -> bool:
        return self.gt is not None and self.gt_status == "available"


def _json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AuditInputError(f"cannot read JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AuditInputError(f"invalid JSON {path}: {exc}") from exc


def _as_int(value: Any, label: str) -> int:
    # bool is an int subclass but is not a meaningful frame index.
    if isinstance(value, bool):
        raise AuditInputError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise AuditInputError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise AuditInputError(f"{label} must be an integer")
    return result


def _finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != size or not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite with {size} values, got {arr.shape}")
    return arr.copy()


def _finite_joints(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (1, 55, 3):
        arr = arr[0]
    elif arr.shape == (165,):
        arr = arr.reshape(55, 3)
    if arr.shape != (55, 3) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite with shape (55,3), got {arr.shape}")
    return arr.copy()


def _optional_flat(value: Any, size: int, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != size or not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite with {size} values, got {arr.shape}")
    return arr.copy()


def _optional_vertices(value: Any, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 1 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be finite with shape (N,3), got {arr.shape}")
    return arr.copy()


def _load_prediction(sequence_dir: Path, row: Mapping[str, Any]) -> Prediction:
    """Load one NPZ without allowing an optional field to hide joint failure."""
    try:
        frame_index = _as_int(row.get("frame_index"), "sequence frame_index")
    except AuditInputError as exc:
        return Prediction(error=str(exc))
    raw_moment = row.get("moment_dir")
    if raw_moment is None:
        return Prediction(error=f"frame {frame_index}: index row has no moment_dir")
    moment = Path(str(raw_moment))
    if not moment.is_absolute():
        moment = sequence_dir / moment
    npz_path = (moment / "smplx_result.npz").resolve()
    if not npz_path.is_file():
        return Prediction(npz_path=npz_path, error=f"missing NPZ: {npz_path}")
    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            keys = set(payload.files)
            joint_key = next((key for key in JOINT_KEY_PRIORITY if key in keys), None)
            if joint_key is None:
                raise ValueError(f"no native 55-joint key; keys={sorted(keys)}")
            joints = _finite_joints(payload[joint_key], joint_key)
            prediction = Prediction(joints=joints, npz_path=npz_path, joint_key=joint_key)
            if "pelvis_world_m" in keys:
                try:
                    prediction.pelvis = _finite_vector(payload["pelvis_world_m"], 3, "pelvis_world_m")
                except (TypeError, ValueError) as exc:
                    prediction.optional_errors["pelvis_world_m"] = str(exc)
            if "poses" in keys:
                try:
                    prediction.poses = _optional_flat(payload["poses"], 165, "poses")
                except (TypeError, ValueError) as exc:
                    prediction.optional_errors["poses"] = str(exc)
            if "shapes" in keys:
                try:
                    prediction.shapes = _optional_flat(payload["shapes"], 10, "shapes")
                except (TypeError, ValueError) as exc:
                    prediction.optional_errors["shapes"] = str(exc)
            if "vertices" in keys:
                try:
                    prediction.vertices = _optional_vertices(payload["vertices"], "vertices")
                except (TypeError, ValueError) as exc:
                    prediction.optional_errors["vertices"] = str(exc)
            return prediction
    except (OSError, ValueError, TypeError) as exc:
        return Prediction(npz_path=npz_path, error=f"cannot load {npz_path}: {exc}")


def _sequence_index(sequence_dir: Path) -> tuple[dict[int, Mapping[str, Any]], list[str]]:
    path = sequence_dir / "index.json"
    payload = _json_load(path)
    rows: Any
    if isinstance(payload, dict):
        rows = payload.get("index")
    else:
        rows = payload
    if not isinstance(rows, list):
        raise AuditInputError(f"sequence index has no list at {path}")
    by_frame: dict[int, Mapping[str, Any]] = {}
    warnings: list[str] = []
    for position, raw in enumerate(rows):
        if not isinstance(raw, dict):
            warnings.append(f"index row {position} is not an object")
            continue
        try:
            frame = _as_int(raw.get("frame_index"), f"index row {position} frame_index")
        except AuditInputError as exc:
            warnings.append(str(exc))
            continue
        if frame in by_frame:
            warnings.append(f"duplicate sequence frame_index {frame}; first row used")
            continue
        by_frame[frame] = raw
    return by_frame, warnings


def _validate_protocol(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    payload = _json_load(path)
    if not isinstance(payload, dict):
        raise AuditInputError("protocol JSON must be an object")
    warnings: list[str] = []
    scenario = payload.get("scenario")
    if not isinstance(scenario, str) or not scenario.strip():
        raise AuditInputError("protocol scenario must be a non-empty string")
    recording_value = payload.get("recording_dir")
    if not isinstance(recording_value, str) or not recording_value.strip():
        raise AuditInputError("protocol recording_dir must be a path string")
    recording_path = Path(recording_value).expanduser()
    if not recording_path.is_absolute():
        warnings.append("protocol recording_dir is relative; resolved relative to the current working directory")
    try:
        freeze_source_frame = _as_int(payload.get("freeze_source_frame"), "freeze_source_frame")
    except AuditInputError as exc:
        raise
    try:
        interval = float(payload.get("sample_interval_s"))
    except (TypeError, ValueError) as exc:
        raise AuditInputError("sample_interval_s must be a positive number") from exc
    if not math.isfinite(interval) or interval <= 0:
        raise AuditInputError("sample_interval_s must be a positive finite number")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise AuditInputError("protocol rows must be a list")
    rows: list[dict[str, Any]] = []
    allowed = {"warmup", "motion", "freeze", "resume"}
    for pos, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise AuditInputError(f"protocol row {pos} must be an object")
        try:
            frame = _as_int(raw.get("frame_index"), f"protocol row {pos} frame_index")
            source = _as_int(raw.get("source_frame_index"), f"protocol row {pos} source_frame_index")
        except AuditInputError as exc:
            raise
        phase = raw.get("phase")
        if phase not in allowed:
            raise AuditInputError(f"protocol row {pos} phase must be one of {sorted(allowed)}")
        rows.append({"frame_index": frame, "source_frame_index": source, "phase": phase})
    if not rows:
        warnings.append("protocol contains no rows")
    for previous, current in zip(rows, rows[1:]):
        if current["source_frame_index"] <= previous["source_frame_index"]:
            # A repeated source frame is expected during freeze: the source
            # image/GT stays fixed while the sampled sequence time advances.
            warnings.append("source_frame_index repeats or decreases; repeated values are treated as frozen GT, while source time follows row order")
            break
    protocol = {
        "scenario": scenario,
        "recording_dir": str(recording_path.resolve()),
        "freeze_source_frame": freeze_source_frame,
        "sample_interval_s": interval,
    }
    return protocol, rows, warnings


def _resolve_recording_path(recording_dir: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else recording_dir / path


def _load_gt_payload(path: Path) -> np.ndarray:
    payload = _json_load(path)
    if not isinstance(payload, dict):
        raise ValueError("GT payload is not an object")
    missing_joints = payload.get("missing_joints")
    if isinstance(missing_joints, list) and missing_joints:
        raise ValueError(f"GT payload marks missing joints: {missing_joints[:5]}")
    raw_bones = payload.get("bones")
    if not isinstance(raw_bones, list):
        raise ValueError("GT payload has no bones list")
    by_name: dict[str, Any] = {}
    for item in raw_bones:
        if not isinstance(item, dict) or not isinstance(item.get("joint_name"), str):
            continue
        by_name[item["joint_name"]] = item
    raw_names = payload.get("joint_names")
    names = [str(name) for name in raw_names] if isinstance(raw_names, list) else []
    # Prefer canonical SMPL-X names, while retaining a documented list-order
    # fallback for old recordings that only carried 55 consistently ordered
    # bones.  No coordinate is inferred or filled when a bone is absent.
    if all(name in by_name for name in SMPLX55_NAMES):
        order = list(SMPLX55_NAMES)
    elif len(names) == 55 and all(name in by_name for name in names):
        order = names
    elif len(raw_bones) == 55:
        order = [str(item.get("joint_name", index)) if isinstance(item, dict) else str(index)
                 for index, item in enumerate(raw_bones)]
        if any(not isinstance(item, dict) or "world_location_cm" not in item for item in raw_bones):
            raise ValueError("GT bones are missing world_location_cm")
    else:
        missing = [name for name in SMPLX55_NAMES if name not in by_name]
        raise ValueError(f"GT does not contain 55 ordered bones; missing={missing[:8]}")
    positions_cm: list[np.ndarray] = []
    for name in order:
        item = by_name.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"GT bone {name!r} is missing")
        positions_cm.append(_finite_vector(item.get("world_location_cm"), 3, f"GT bone {name} world_location_cm"))
    cm = np.stack(positions_cm, axis=0)
    # UE centimetres -> right-handed metres: X*0.01, Y*-0.01, Z*0.01.
    cm *= np.array([0.01, -0.01, 0.01], dtype=np.float64)
    return cm


def _recording_gt_index(recording_dir: Path) -> tuple[dict[int, tuple[Path | None, str | None]], list[str]]:
    path = recording_dir / "manifest.jsonl"
    if not path.is_file():
        raise AuditInputError(f"recording manifest not found: {path}")
    result: dict[int, tuple[Path | None, str | None]] = {}
    warnings: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AuditInputError(f"cannot read recording manifest {path}: {exc}") from exc
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"manifest line {line_no} invalid JSON: {exc}")
            continue
        if not isinstance(raw, dict):
            warnings.append(f"manifest line {line_no} is not an object")
            continue
        status = raw.get("status", "ok")
        if status not in ("ok", "success", "successful"):
            continue
        try:
            frame = _as_int(raw.get("frame_index"), f"manifest line {line_no} frame_index")
        except AuditInputError as exc:
            warnings.append(str(exc))
            continue
        gt_value = raw.get("gt_path")
        if not gt_value:
            item = (None, "successful manifest row has no gt_path")
        else:
            item = (_resolve_recording_path(recording_dir, gt_value).resolve(), None)
        if frame in result:
            warnings.append(f"duplicate successful GT frame_index {frame}; first row used")
            continue
        result[frame] = item
    return result, warnings


def _stats(values: Iterable[float] | np.ndarray | None, *, unit: str = "m") -> dict[str, Any]:
    if values is None:
        return {"count": 0, "unit": unit, "mean": None, "median": None, "p95": None, "max": None}
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "unit": unit, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "unit": unit,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _scalar_stats(values: Sequence[float], times: Sequence[float] | None = None, *, unit: str = "m") -> dict[str, Any]:
    result = _stats(np.asarray(values, dtype=np.float64), unit=unit)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(arr)
    arr = arr[finite]
    if arr.size:
        result["first"] = float(arr[0])
        result["last"] = float(arr[-1])
    else:
        result["first"] = None
        result["last"] = None
    if times is not None:
        t = np.asarray(times, dtype=np.float64).reshape(-1)
        if t.size == finite.size:
            t = t[finite]
        if arr.size >= 2 and t.size == arr.size and np.ptp(t) > 0:
            result["trend_slope_per_s"] = float(np.polyfit(t, arr, 1)[0])
        else:
            result["trend_slope_per_s"] = None
    return result


def _vector_displacement_stats(values: np.ndarray, *, names: Sequence[str] | None = None) -> dict[str, Any]:
    """Summarize (frames, joints) distances with frame P95/max and joint detail."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        arr = arr.reshape(0, 0)
    result: dict[str, Any] = {
        "frame_count": int(arr.shape[0]),
        "joint_count": int(arr.shape[1]) if arr.ndim == 2 else 0,
        "all_points_m": _stats(arr.reshape(-1), unit="m"),
        "per_frame_p95_m": _stats(np.percentile(arr, 95, axis=1) if arr.size else None, unit="m"),
        "per_frame_max_m": _stats(np.max(arr, axis=1) if arr.size else None, unit="m"),
    }
    if names is not None and arr.ndim == 2 and arr.shape[1] == len(names):
        result["by_joint_m"] = {str(name): _stats(arr[:, index], unit="m") for index, name in enumerate(names)}
    return result


def _root_for_prediction(prediction: Prediction) -> np.ndarray:
    return prediction.pelvis if prediction.pelvis is not None else prediction.joints[0]  # type: ignore[index]


def _segment_for_phase_rows(rows: list[dict[str, Any]]) -> list[str]:
    freeze_positions = [index for index, row in enumerate(rows) if row["phase"] == "freeze"]
    first_freeze = freeze_positions[0] if freeze_positions else None
    second_half_start = (len(freeze_positions) // 2) if freeze_positions else None
    freeze_rank = {position: rank for rank, position in enumerate(freeze_positions)}
    segments: list[str] = []
    for index, row in enumerate(rows):
        phase = row["phase"]
        if phase == "warmup":
            segments.append("warmup")
        elif phase == "motion":
            segments.append("motion")
        elif phase == "resume":
            segments.append("resume")
        elif index == first_freeze:
            segments.append("freeze_first_transition")
        elif second_half_start is not None and freeze_rank[index] >= second_half_start:
            segments.append("freeze_second_half")
        else:
            segments.append("freeze_early")
    return segments


def _pair_rows(rows: Sequence[EvaluatedRow]) -> list[dict[str, Any]]:
    """Compute adjacent accepted updates only for adjacent requested rows."""
    pairs: list[dict[str, Any]] = []
    for previous, current in zip(rows, rows[1:]):
        if not previous.accepted or not current.accepted:
            continue
        assert previous.prediction.joints is not None
        assert current.prediction.joints is not None
        displacement = np.linalg.norm(current.prediction.joints - previous.prediction.joints, axis=1)
        selected = {name: float(displacement[index]) for name, index in SELECTED_JOINTS}
        pairs.append({
            "from_frame_index": previous.frame_index,
            "to_frame_index": current.frame_index,
            "from_phase": previous.phase,
            "to_phase": current.phase,
            "from_segment": previous.segment,
            "to_segment": current.segment,
            "source_time_s": current.source_time_s,
            "source_delta_s": current.source_time_s - previous.source_time_s,
            "selected_joint_m": selected,
            "max_joint_m": float(np.max(displacement)),
            "all_joint_m": displacement.tolist(),
        })
    return pairs


def _adjacent_summary(pairs: Sequence[Mapping[str, Any]], *, freeze_only: bool = False) -> dict[str, Any]:
    selected_pairs = [
        pair for pair in pairs
        if not freeze_only or pair.get("from_phase") == "freeze" and pair.get("to_phase") == "freeze"
    ]
    result: dict[str, Any] = {
        "pair_count": len(selected_pairs),
        "by_joint_m": {},
        "per_step_max_joint_m": _stats([pair["max_joint_m"] for pair in selected_pairs], unit="m"),
        "series": [],
    }
    for name, _index in SELECTED_JOINTS:
        result["by_joint_m"][name] = _stats([pair["selected_joint_m"][name] for pair in selected_pairs], unit="m")
    for pair in selected_pairs:
        result["series"].append({
            "from_frame_index": pair["from_frame_index"],
            "to_frame_index": pair["to_frame_index"],
            "from_phase": pair["from_phase"],
            "to_phase": pair["to_phase"],
            "from_segment": pair["from_segment"],
            "to_segment": pair["to_segment"],
            "source_time_s": pair["source_time_s"],
            "source_delta_s": pair["source_delta_s"],
            "selected_joint_m": pair["selected_joint_m"],
            "max_joint_m": pair["max_joint_m"],
        })
    return result


def _shape_summary(rows: Sequence[EvaluatedRow]) -> dict[str, Any]:
    by_phase: dict[str, list[np.ndarray]] = {}
    missing = 0
    for row in rows:
        if not row.accepted:
            continue
        if row.prediction.shapes is None:
            missing += 1
            continue
        by_phase.setdefault(row.phase, []).append(row.prediction.shapes)
    all_values = [item for values in by_phase.values() for item in values]

    def one(values: list[np.ndarray]) -> dict[str, Any]:
        if not values:
            return {"sample_count": 0, "per_beta_range": None, "max_range": None, "mean_range": None}
        arr = np.stack(values, axis=0)
        ranges = np.ptp(arr, axis=0)
        return {
            "sample_count": int(arr.shape[0]),
            "per_beta_min": arr.min(axis=0).tolist(),
            "per_beta_max": arr.max(axis=0).tolist(),
            "per_beta_range": ranges.tolist(),
            "max_range": float(np.max(ranges)),
            "mean_range": float(np.mean(ranges)),
        }

    return {
        "field": "shapes",
        "missing_sample_count": missing,
        "all_requested_phases": one(all_values),
        "by_phase": {phase: one(values) for phase, values in sorted(by_phase.items())},
        "interpretation": "Shape time-series range describes temporal variation only; a fixed shape can still be wrong and this metric makes no shape-accuracy claim.",
    }


def _pose_summary(rows: Sequence[EvaluatedRow]) -> dict[str, Any]:
    by_phase: dict[str, list[np.ndarray]] = {}
    missing = 0
    for row in rows:
        if not row.accepted:
            continue
        if row.prediction.poses is None:
            missing += 1
            continue
        by_phase.setdefault(row.phase, []).append(row.prediction.poses)
    values = [item for phase_values in by_phase.values() for item in phase_values]
    if values:
        arr = np.stack(values, axis=0)
        ranges: list[float] = np.ptp(arr, axis=0).tolist()
        overall: dict[str, Any] = {
            "sample_count": int(arr.shape[0]),
            "component_count": int(arr.shape[1]),
            "max_abs_range": float(np.max(ranges)),
            "mean_abs_range": float(np.mean(ranges)),
        }
    else:
        overall = {"sample_count": 0, "component_count": 165, "max_abs_range": None, "mean_abs_range": None}
    return {
        "field": "poses axis-angle",
        "missing_sample_count": missing,
        "all_requested_phases": overall,
        "by_phase_sample_count": {phase: len(items) for phase, items in sorted(by_phase.items())},
    }


def _surface_summary(rows: Sequence[EvaluatedRow]) -> dict[str, Any]:
    freeze_rows = [row for row in rows if row.phase == "freeze" and row.accepted and row.prediction.vertices is not None]
    if not freeze_rows:
        return {
            "status": "missing",
            "reason": "no freeze NPZ contains valid vertices",
            "freeze_sample_count": 0,
            "freeze_displacement_m": _vector_displacement_stats(np.empty((0, 0))),
            "adjacent_jitter_m": _stats([], unit="m"),
        }
    baseline = freeze_rows[0]
    assert baseline.prediction.vertices is not None
    displacements: list[np.ndarray] = []
    usable_rows: list[EvaluatedRow] = []
    for row in freeze_rows:
        assert row.prediction.vertices is not None
        if row.prediction.vertices.shape != baseline.prediction.vertices.shape:
            continue
        displacements.append(np.linalg.norm(row.prediction.vertices - baseline.prediction.vertices, axis=1))
        usable_rows.append(row)
    if not displacements:
        return {
            "status": "missing",
            "reason": "freeze vertices have incompatible shapes",
            "freeze_sample_count": 0,
            "baseline_frame_index": baseline.frame_index,
            "freeze_displacement_m": _vector_displacement_stats(np.empty((0, 0))),
            "adjacent_jitter_m": _stats([], unit="m"),
        }
    adjacent: list[float] = []
    for previous, current in zip(usable_rows, usable_rows[1:]):
        assert previous.prediction.vertices is not None and current.prediction.vertices is not None
        if current.order != previous.order + 1:
            continue
        adjacent.append(float(np.max(np.linalg.norm(current.prediction.vertices - previous.prediction.vertices, axis=1))))
    return {
        "status": "available",
        "baseline_frame_index": baseline.frame_index,
        "vertex_count": int(baseline.prediction.vertices.shape[0]),
        "freeze_sample_count": len(usable_rows),
        "freeze_displacement_m": _vector_displacement_stats(np.stack(displacements, axis=0)),
        "adjacent_jitter_m": _stats(adjacent, unit="m"),
    }


def _tail_summary(rows: Sequence[EvaluatedRow], freeze_metrics: Mapping[int, Mapping[str, Any]], interval_s: float) -> dict[str, Any]:
    freeze_accepted = [row for row in rows if row.phase == "freeze" and row.accepted and row.frame_index in freeze_metrics]
    if not freeze_accepted:
        return {
            "window_s": 10.0,
            "status": "missing",
            "reason": "no accepted freeze joints",
            "accepted_frame_count": 0,
            "jitter": {},
            "drift": {},
        }
    end_time = freeze_accepted[-1].source_time_s
    start_time = end_time - 10.0
    # Keep the row order and source-time window; no samples are inserted for
    # missing results.  Pair calculations below also require adjacent
    # protocol rows, so a gap cannot masquerade as low jitter.
    tail = [row for row in freeze_accepted if row.source_time_s >= start_time]
    root_drift = [float(freeze_metrics[row.frame_index]["root_drift_m"]) for row in tail]
    body_max = [float(freeze_metrics[row.frame_index]["body22_max_drift_m"]) for row in tail]
    body_p95 = [float(freeze_metrics[row.frame_index]["body22_p95_drift_m"]) for row in tail]
    tail_pairs: list[tuple[EvaluatedRow, EvaluatedRow]] = []
    for previous, current in zip(rows, rows[1:]):
        if previous.phase != "freeze" or current.phase != "freeze" or not previous.accepted or not current.accepted:
            continue
        if previous.source_time_s < start_time or current.source_time_s < start_time:
            continue
        tail_pairs.append((previous, current))
    root_jitter: list[float] = []
    body_jitter: list[float] = []
    body_frame_max_jitter: list[float] = []
    body_frame_p95_jitter: list[float] = []
    for previous, current in tail_pairs:
        assert previous.prediction.joints is not None and current.prediction.joints is not None
        delta = np.linalg.norm(current.prediction.joints - previous.prediction.joints, axis=1)
        root_jitter.append(float(delta[0]))
        body_jitter.extend(delta[:22].tolist())
        body_frame_max_jitter.append(float(np.max(delta[:22])))
        body_frame_p95_jitter.append(float(np.percentile(delta[:22], 95)))
    baseline = tail[0]
    assert baseline.prediction.joints is not None and tail[-1].prediction.joints is not None
    end_displacement = np.linalg.norm(tail[-1].prediction.joints - baseline.prediction.joints, axis=1)
    return {
        "window_s": 10.0,
        "status": "available",
        "source_time_start_s": float(start_time),
        "source_time_end_s": float(end_time),
        "accepted_frame_count": len(tail),
        "jitter": {
            "root_m": _stats(root_jitter, unit="m"),
            "body22_point_m": _stats(body_jitter, unit="m"),
            "body22_frame_max_m": _stats(body_frame_max_jitter, unit="m"),
            "body22_frame_p95_m": _stats(body_frame_p95_jitter, unit="m"),
            "adjacent_pair_count": len(tail_pairs),
        },
        "drift": {
            "root_m": float(end_displacement[0]),
            "body22_point_m": _stats(end_displacement[:22], unit="m"),
            "body22_frame_max_m": float(np.max(end_displacement[:22])),
            "root_slope_m_per_s": (_scalar_stats(root_drift, [row.source_time_s for row in tail])["trend_slope_per_s"]),
            "body22_frame_max_slope_m_per_s": (_scalar_stats(body_max, [row.source_time_s for row in tail])["trend_slope_per_s"]),
        },
        "sample_interval_s": interval_s,
    }


def _gt_error_summary(rows: Sequence[EvaluatedRow], *, phase: str | None = None) -> dict[str, Any]:
    selected = [row for row in rows if row.accepted and row.gt_accepted and (phase is None or row.phase == phase)]
    if not selected:
        return {
            "status": "missing",
            "sample_count": 0,
            "mean_point_error_m": _stats([], unit="m"),
            "frame_mean_error_m": _stats([], unit="m"),
            "frame_p95_error_m": _stats([], unit="m"),
            "frame_max_error_m": _stats([], unit="m"),
            "trend_slope_m_per_s": None,
            "by_joint_m": {},
        }
    frame_mean: list[float] = []
    frame_p95: list[float] = []
    frame_max: list[float] = []
    point_errors: list[np.ndarray] = []
    times: list[float] = []
    for row in selected:
        assert row.prediction.joints is not None and row.gt is not None
        error = np.linalg.norm(row.prediction.joints[:22] - row.gt[:22], axis=1)
        point_errors.append(error)
        frame_mean.append(float(np.mean(error)))
        frame_p95.append(float(np.percentile(error, 95)))
        frame_max.append(float(np.max(error)))
        times.append(row.source_time_s)
    point_array = np.stack(point_errors, axis=0)
    trend = _scalar_stats(frame_mean, times)["trend_slope_per_s"]
    return {
        "status": "available",
        "sample_count": len(selected),
        "first_frame_index": selected[0].frame_index,
        "last_frame_index": selected[-1].frame_index,
        "mean_point_error_m": _stats(point_array.reshape(-1), unit="m"),
        "frame_mean_error_m": _scalar_stats(frame_mean, times, unit="m"),
        "frame_p95_error_m": _scalar_stats(frame_p95, times, unit="m"),
        "frame_max_error_m": _scalar_stats(frame_max, times, unit="m"),
        "trend_slope_m_per_s": trend,
        "by_joint_m": {name: _scalar_stats(point_array[:, index].tolist(), times, unit="m")
                        for index, name in enumerate(BODY22_NAMES)},
    }


def _segment_summary(rows: Sequence[EvaluatedRow], segment: str, freeze_metrics: Mapping[int, Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row.segment == segment]
    accepted = [row for row in selected if row.accepted]
    root = [freeze_metrics[row.frame_index]["root_drift_m"] for row in accepted if row.frame_index in freeze_metrics]
    body_p95 = [freeze_metrics[row.frame_index]["body22_p95_drift_m"] for row in accepted if row.frame_index in freeze_metrics]
    body_max = [freeze_metrics[row.frame_index]["body22_max_drift_m"] for row in accepted if row.frame_index in freeze_metrics]
    segment_pairs = [pair for pair in pairs if pair.get("to_segment") == segment or pair.get("from_segment") == segment]
    gt = _gt_error_summary(accepted)
    return {
        "requested_count": len(selected),
        "accepted_prediction_count": len(accepted),
        "missing_prediction_count": sum(1 for row in selected if not row.accepted),
        "gt_error_body22": gt,
        "root_drift_from_first_freeze_m": _stats(root, unit="m"),
        "body22_frame_p95_drift_m": _stats(body_p95, unit="m"),
        "body22_frame_max_drift_m": _stats(body_max, unit="m"),
        "adjacent_update": _adjacent_summary(segment_pairs),
    }


def _safe_json(value: Any) -> Any:
    """Convert numpy values and non-finite floats to strict JSON values."""
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    return value


def _row_json(row: EvaluatedRow) -> dict[str, Any]:
    return {
        "order": row.order,
        "frame_index": row.frame_index,
        "phase": row.phase,
        "segment": row.segment,
        "source_frame_index": row.source_frame_index,
        "source_time_s": row.source_time_s,
        "sequence_status": row.sequence_status,
        "prediction_status": "accepted" if row.accepted else "missing",
        "prediction_error": row.prediction_error,
        "npz_path": None if row.prediction.npz_path is None else str(row.prediction.npz_path),
        "joint_key": row.prediction.joint_key,
        "optional_field_errors": row.prediction.optional_errors,
        "gt_status": row.gt_status,
        "gt_error": row.gt_error,
        "metrics": row.metrics,
    }


def _plot_summary(path: Path, rows: Sequence[EvaluatedRow], pairs: Sequence[Mapping[str, Any]], freeze_metrics: Mapping[int, Mapping[str, Any]], gt_errors: Mapping[int, float]) -> None:
    # Import pyplot only here so parsing/metric functions remain usable in a
    # headless environment and the CLI can set the non-GUI backend first.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax_root, ax_knee, ax_gt, ax_step = axes.ravel()
    freeze = [row for row in rows if row.phase == "freeze" and row.frame_index in freeze_metrics]
    times = np.asarray([row.source_time_s for row in freeze], dtype=np.float64)
    root_values = np.asarray([freeze_metrics[row.frame_index]["root_drift_m"] for row in freeze], dtype=np.float64)
    left_knee = np.asarray([freeze_metrics[row.frame_index]["body22_point_drift_m"][4] for row in freeze], dtype=np.float64)
    right_knee = np.asarray([freeze_metrics[row.frame_index]["body22_point_drift_m"][5] for row in freeze], dtype=np.float64)
    if freeze:
        ax_root.plot(times, root_values, color="tab:blue", marker=".", label="from first freeze")
        ax_knee.plot(times, left_knee, color="tab:orange", marker=".", label="left knee (4)")
        ax_knee.plot(times, right_knee, color="tab:green", marker=".", label="right knee (5)")
        for axis in (ax_root, ax_knee):
            axis.axvline(times[0], color="0.4", linestyle="--", linewidth=0.8, label="first freeze")
    else:
        ax_root.text(0.5, 0.5, "No accepted freeze joints", ha="center", va="center", transform=ax_root.transAxes)
        ax_knee.text(0.5, 0.5, "No accepted freeze joints", ha="center", va="center", transform=ax_knee.transAxes)
    ax_root.set_title("Freeze root drift")
    ax_root.set_xlabel("source time (s)")
    ax_root.set_ylabel("displacement (m)")
    ax_root.grid(True, alpha=0.25)
    ax_root.legend(loc="best") if freeze else None
    ax_knee.set_title("Freeze knee drift")
    ax_knee.set_xlabel("source time (s)")
    ax_knee.set_ylabel("displacement (m)")
    ax_knee.grid(True, alpha=0.25)
    ax_knee.legend(loc="best") if freeze else None

    gt_rows = [row for row in rows if row.frame_index in gt_errors]
    if gt_rows:
        gt_times = [row.source_time_s for row in gt_rows]
        gt_values = [gt_errors[row.frame_index] for row in gt_rows]
        ax_gt.plot(gt_times, gt_values, color="tab:red", marker=".", label="body22 mean GT error")
        ax_gt.legend(loc="best")
    else:
        ax_gt.text(0.5, 0.5, "GT body22 error unavailable", ha="center", va="center", transform=ax_gt.transAxes)
    ax_gt.set_title("Body22 GT position error")
    ax_gt.set_xlabel("source time (s)")
    ax_gt.set_ylabel("mean point error (m)")
    ax_gt.grid(True, alpha=0.25)

    if pairs:
        step_times = np.asarray([pair["source_time_s"] for pair in pairs], dtype=np.float64)
        step_values = np.asarray([pair["max_joint_m"] for pair in pairs], dtype=np.float64)
        phase_colors = {"freeze": "tab:blue", "motion": "tab:gray", "resume": "tab:purple"}
        colors = [phase_colors.get(str(pair["to_phase"]), "tab:orange") for pair in pairs]
        ax_step.scatter(step_times, step_values, c=colors, s=18, label="max of 55 joints")
        ax_step.plot(step_times, step_values, color="0.45", linewidth=0.7, alpha=0.7)
        handles = [plt.Line2D([], [], color=color, marker="o", linestyle="", label=phase)
                   for phase, color in phase_colors.items() if any(pair["to_phase"] == phase for pair in pairs)]
        if handles:
            ax_step.legend(handles=handles, loc="best")
    else:
        ax_step.text(0.5, 0.5, "No adjacent accepted updates", ha="center", va="center", transform=ax_step.transAxes)
    ax_step.set_title("Maximum adjacent joint movement")
    ax_step.set_xlabel("source time (s)")
    ax_step.set_ylabel("movement (m)")
    ax_step.grid(True, alpha=0.25)
    fig.suptitle("SMPL-X freeze stability audit", fontsize=14)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(sequence_dir: Path, protocol_json: Path, output_dir: Path) -> dict[str, Any]:
    """Evaluate a sequence and write the requested JSON/PNG artifacts."""
    sequence_dir = sequence_dir.expanduser().resolve()
    protocol_json = protocol_json.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol, raw_rows, protocol_warnings = _validate_protocol(protocol_json)
    recording_dir = Path(protocol["recording_dir"]).expanduser().resolve()
    sequence_index, sequence_warnings = _sequence_index(sequence_dir)
    gt_index, gt_warnings = _recording_gt_index(recording_dir)
    gt_cache: dict[Path, tuple[np.ndarray | None, str, str | None]] = {}

    analyzed_raw_rows = [row for row in raw_rows if row["phase"] != "warmup"]
    warmup_count = len(raw_rows) - len(analyzed_raw_rows)
    if analyzed_raw_rows:
        timeline_origin = analyzed_raw_rows[0]["source_frame_index"]
    else:
        timeline_origin = 0
    segments = _segment_for_phase_rows(analyzed_raw_rows)
    evaluated: list[EvaluatedRow] = []
    for order, (raw, segment) in enumerate(zip(analyzed_raw_rows, segments)):
        frame_index = raw["frame_index"]
        source_index = raw["source_frame_index"]
        # ``source_frame_index`` identifies the source image/GT and is
        # intentionally repeated for freeze rows.  Source time is the sampled
        # protocol timeline, so it must advance for every requested row even
        # while the source frame (and GT) remains frozen.
        source_time = order * float(protocol["sample_interval_s"])
        index_row = sequence_index.get(frame_index)
        if index_row is None:
            prediction = Prediction(error=f"frame_index {frame_index} not present in sequence index")
            sequence_status = "missing"
        else:
            prediction = _load_prediction(sequence_dir, index_row)
            sequence_status = "present" if prediction.accepted else "missing"
        gt: np.ndarray | None = None
        gt_status = "missing"
        gt_error: str | None = None
        # Recorder manifest rows are keyed by the derived pipeline
        # ``frame_index``.  The protocol's ``source_frame_index`` identifies
        # the source image/GT and is commonly repeated during freeze; it is
        # deliberately not used as a fallback lookup key because doing so can
        # silently associate a different derived row with this prediction.
        gt_path, gt_lookup_error = gt_index.get(frame_index, (None, "no successful manifest row for frame_index"))
        if gt_lookup_error is not None:
            gt_error = gt_lookup_error
        elif gt_path is None:
            gt_error = "successful manifest row has no gt_path"
        elif gt_path not in gt_cache:
            if not gt_path.is_file():
                gt_cache[gt_path] = (None, "missing", f"missing GT file: {gt_path}")
            else:
                try:
                    gt_cache[gt_path] = (_load_gt_payload(gt_path), "available", None)
                except (AuditInputError, OSError, TypeError, ValueError) as exc:
                    gt_cache[gt_path] = (None, "invalid", str(exc))
        if gt_path is not None and gt_path in gt_cache:
            gt, gt_status, gt_error = gt_cache[gt_path]
        evaluated.append(EvaluatedRow(
            order=order,
            frame_index=frame_index,
            phase=raw["phase"],
            source_frame_index=source_index,
            source_time_s=float(source_time),
            segment=segment,
            index_row=index_row,
            prediction=prediction,
            gt=gt,
            gt_status=gt_status,
            gt_error=gt_error,
            prediction_error=prediction.error,
            sequence_status=sequence_status,
        ))

    freeze_rows = [row for row in evaluated if row.phase == "freeze"]
    accepted_freeze = [row for row in freeze_rows if row.accepted]
    first_freeze = accepted_freeze[0] if accepted_freeze else None
    last_motion = next((row for row in reversed(evaluated) if row.phase == "motion" and row.accepted), None)

    freeze_metrics: dict[int, dict[str, Any]] = {}
    if first_freeze is not None:
        assert first_freeze.prediction.joints is not None
        first_root = _root_for_prediction(first_freeze.prediction)
        first_joints = first_freeze.prediction.joints
        last_motion_root = _root_for_prediction(last_motion.prediction) if last_motion is not None else None
        last_motion_joints = last_motion.prediction.joints if last_motion is not None else None
        for row in accepted_freeze:
            assert row.prediction.joints is not None
            root = _root_for_prediction(row.prediction)
            body_displacement = np.linalg.norm(row.prediction.joints[:22] - first_joints[:22], axis=1)
            metrics: dict[str, Any] = {
                "root_drift_m": float(np.linalg.norm(root - first_root)),
                "body22_point_drift_m": body_displacement.tolist(),
                "body22_p95_drift_m": float(np.percentile(body_displacement, 95)),
                "body22_max_drift_m": float(np.max(body_displacement)),
            }
            if last_motion_joints is not None and last_motion_root is not None:
                after_stop = np.linalg.norm(row.prediction.joints[:22] - last_motion_joints[:22], axis=1)
                metrics.update({
                    "root_change_from_last_motion_m": float(np.linalg.norm(root - last_motion_root)),
                    "body22_point_change_from_last_motion_m": after_stop.tolist(),
                    "body22_p95_change_from_last_motion_m": float(np.percentile(after_stop, 95)),
                    "body22_max_change_from_last_motion_m": float(np.max(after_stop)),
                })
            if row.gt_accepted:
                assert row.gt is not None
                error = np.linalg.norm(row.prediction.joints[:22] - row.gt[:22], axis=1)
                metrics["body22_gt_error_m"] = error.tolist()
                metrics["body22_gt_mean_error_m"] = float(np.mean(error))
                metrics["body22_gt_p95_error_m"] = float(np.percentile(error, 95))
                metrics["body22_gt_max_error_m"] = float(np.max(error))
            row.metrics.update(metrics)
            freeze_metrics[row.frame_index] = metrics

    pairs = _pair_rows(evaluated)
    freeze_pairs = [pair for pair in pairs if pair.get("from_phase") == "freeze" and pair.get("to_phase") == "freeze"]
    freeze_from_first_root = [metrics["root_drift_m"] for metrics in freeze_metrics.values()]
    freeze_body_points = [metrics["body22_point_drift_m"] for metrics in freeze_metrics.values()]
    first_body_array = np.asarray(freeze_body_points, dtype=np.float64) if freeze_body_points else np.empty((0, 22))
    freeze_root_stats = _scalar_stats(freeze_from_first_root, [row.source_time_s for row in accepted_freeze], unit="m")
    from_first = {
        "root_displacement_m": freeze_root_stats,
        "body22_point_displacement_m": _vector_displacement_stats(first_body_array, names=BODY22_NAMES),
        "baseline_frame_index": None if first_freeze is None else first_freeze.frame_index,
        "baseline_source_frame_index": None if first_freeze is None else first_freeze.source_frame_index,
    }
    if last_motion is not None and accepted_freeze:
        last_motion_root = _root_for_prediction(last_motion.prediction)
        last_motion_joints = last_motion.prediction.joints
        assert last_motion_joints is not None
        after_stop_roots = [metrics["root_change_from_last_motion_m"] for metrics in freeze_metrics.values()
                            if "root_change_from_last_motion_m" in metrics]
        after_stop_points = [metrics["body22_point_change_from_last_motion_m"] for metrics in freeze_metrics.values()
                             if "body22_point_change_from_last_motion_m" in metrics]
        after_stop_array = np.asarray(after_stop_points, dtype=np.float64) if after_stop_points else np.empty((0, 22))
        from_last_motion = {
            "status": "available",
            "baseline_frame_index": last_motion.frame_index,
            "baseline_source_frame_index": last_motion.source_frame_index,
            "root_change_m": _scalar_stats(after_stop_roots, [row.source_time_s for row in accepted_freeze], unit="m"),
            "body22_point_change_m": _vector_displacement_stats(after_stop_array, names=BODY22_NAMES),
            "first_freeze_change_m": (None if not after_stop_array.size else {
                "root_m": float(after_stop_roots[0]),
                "body22_p95_m": float(np.percentile(after_stop_array[0], 95)),
                "body22_max_m": float(np.max(after_stop_array[0])),
            }),
            "last_freeze_change_m": (None if not after_stop_array.size else {
                "root_m": float(after_stop_roots[-1]),
                "body22_p95_m": float(np.percentile(after_stop_array[-1], 95)),
                "body22_max_m": float(np.max(after_stop_array[-1])),
            }),
        }
    else:
        from_last_motion = {
            "status": "missing",
            "reason": "no accepted motion baseline or no accepted freeze joints",
            "baseline_frame_index": None,
            "baseline_source_frame_index": None,
            "root_change_m": _stats([], unit="m"),
            "body22_point_change_m": _vector_displacement_stats(np.empty((0, 22)), names=BODY22_NAMES),
        }

    gt_error_rows = [row for row in evaluated if row.accepted and row.gt_accepted and row.phase == "freeze"]
    gt_error_series = {row.frame_index: row.metrics["body22_gt_mean_error_m"] for row in gt_error_rows
                       if "body22_gt_mean_error_m" in row.metrics}
    segment_names = ("freeze_first_transition", "freeze_early", "freeze_second_half", "resume")
    segment_metrics = {name: _segment_summary(evaluated, name, freeze_metrics, pairs) for name in segment_names}
    freeze_summary = {
        "requested_count": len(freeze_rows),
        "accepted_count": len(accepted_freeze),
        "missing_count": len(freeze_rows) - len(accepted_freeze),
        "first_accepted_frame_index": None if first_freeze is None else first_freeze.frame_index,
        "last_motion_frame_index": None if last_motion is None else last_motion.frame_index,
        "from_first_freeze": from_first,
        "from_last_motion": from_last_motion,
        "adjacent_update_m": _adjacent_summary(freeze_pairs, freeze_only=False),
        "tail_last_10s": _tail_summary(evaluated, freeze_metrics, float(protocol["sample_interval_s"])),
        "body22_gt_position_error": _gt_error_summary(evaluated, phase="freeze"),
        "segments": segment_metrics,
        # Direct aliases make the three review regions easy to find in a
        # human-readable summary while retaining the explicit segment map.
        "first_freeze_transition": segment_metrics["freeze_first_transition"],
        "freeze_second_half": segment_metrics["freeze_second_half"],
        "resume": segment_metrics["resume"],
    }
    gt_counts = {
        "available": sum(1 for row in evaluated if row.gt_accepted),
        "missing": sum(1 for row in evaluated if row.gt_status == "missing"),
        "invalid": sum(1 for row in evaluated if row.gt_status == "invalid"),
    }
    prediction_missing = sum(1 for row in evaluated if not row.accepted)
    gt_failures = gt_counts["missing"] + gt_counts["invalid"]
    failure_rows = sum(1 for row in evaluated if not row.accepted or not row.gt_accepted)
    failure_reasons: list[str] = []
    if not accepted_freeze:
        failure_reasons.append("no_accepted_freeze_frames")
    if not evaluated:
        failure_reasons.append("no_non_warmup_protocol_rows")
    status = "ok" if not failure_reasons else "failed"
    summary: dict[str, Any] = {
        "schema": "smplx_freeze_audit.v1",
        "status": status,
        "failure_reasons": failure_reasons,
        "inputs": {
            "sequence_dir": str(sequence_dir),
            "protocol_json": str(protocol_json),
            "output_dir": str(output_dir),
            "recording_dir": str(recording_dir),
        },
        "protocol": protocol,
        "counts": {
            "protocol_rows_total": len(raw_rows),
            "warmup_rows_excluded": warmup_count,
            "requested_rows_evaluated": len(evaluated),
            "by_phase": {phase: sum(1 for row in evaluated if row.phase == phase)
                         for phase in ("motion", "freeze", "resume")},
            "prediction_accepted": sum(1 for row in evaluated if row.accepted),
            "prediction_missing_or_failed": prediction_missing,
            "gt_available": gt_counts["available"],
            "gt_missing": gt_counts["missing"],
            "gt_invalid": gt_counts["invalid"],
            "rows_with_any_failure": failure_rows,
            "failure_count": prediction_missing + gt_failures,
        },
        "source_timeline": {
            "origin_source_frame_index": timeline_origin,
            "sample_interval_s": protocol["sample_interval_s"],
            "source_frame_index_strictly_increasing": all(
                current.source_frame_index > previous.source_frame_index
                for previous, current in zip(evaluated, evaluated[1:])
            ),
            "source_time_strictly_increasing": all(
                current.source_time_s > previous.source_time_s
                for previous, current in zip(evaluated, evaluated[1:])
            ),
            "source_frame_index_first": None if not evaluated else evaluated[0].source_frame_index,
            "source_frame_index_last": None if not evaluated else evaluated[-1].source_frame_index,
            "source_time_s_first": None if not evaluated else evaluated[0].source_time_s,
            "source_time_s_last": None if not evaluated else evaluated[-1].source_time_s,
            "note": "GT is evaluated from each successful recording manifest row; increasing source time does not imply changing GT during a frozen image interval.",
        },
        "freeze": freeze_summary,
        "shapes": _shape_summary(evaluated),
        "poses": _pose_summary(evaluated),
        "surface_stability": _surface_summary(evaluated),
        "all_adjacent_updates_m": _adjacent_summary(pairs),
        "warnings": protocol_warnings + sequence_warnings + gt_warnings,
        "rows": [_row_json(row) for row in evaluated],
    }
    summary = _safe_json(summary)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _plot_summary(output_dir / "freeze_stability.png", evaluated, pairs, freeze_metrics, gt_error_series)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True, help="recorded sequence containing index.json")
    parser.add_argument("--protocol-json", type=Path, required=True, help="freeze protocol JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for summary.json and freeze_stability.png")
    args = parser.parse_args(argv)
    try:
        summary = evaluate(args.sequence_dir, args.protocol_json, args.output_dir)
    except (AuditInputError, OSError, ValueError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps({
        "status": summary["status"],
        "summary": str(args.output_dir.expanduser().resolve() / "summary.json"),
        "plot": str(args.output_dir.expanduser().resolve() / "freeze_stability.png"),
        "failure_reasons": summary["failure_reasons"],
        "counts": summary["counts"],
    }, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
