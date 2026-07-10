"""Torch-free HumanScenePlacement JSON read (UE Editor Python and thin bridges)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

PLACEMENT_JSON_ENV = "AMONGUS_HUMAN_SCENE_PLACEMENT_JSON"
PLACEMENT_SIDECAR_NAME = "human_scene_placement.json"

HUMAN_UE_CALIBRATION_SCHEMA_VERSION = 1
CALIB_SIDECAR_NAME = "human_ue_calibration.json"
CALIB_JSON_ENV = "AMONGUS_HUMAN_UE_CALIBRATION_JSON"


def resolve_human_scene_placement_json_path(scene_spec: object, *, repo_root: Path) -> Path | None:
    """Resolve placement JSON path without importing torch-heavy HumanScenePlacement."""
    raw = str(os.environ.get(PLACEMENT_JSON_ENV, "") or "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        return candidate if candidate.is_file() else None

    meta = getattr(scene_spec, "metadata", None) or {}
    meta_rel = meta.get("human_scene_placement") if isinstance(meta, dict) else None
    if meta_rel:
        candidate = Path(str(meta_rel))
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        if candidate.is_file():
            return candidate

    motion = getattr(scene_spec, "motion", None)
    seq_path = getattr(motion, "resolved_sequence_npz_path", None) if motion is not None else None
    if seq_path is not None:
        sibling = Path(seq_path).expanduser().resolve().parent / PLACEMENT_SIDECAR_NAME
        if sibling.is_file():
            return sibling
    return None


def resolve_human_ue_calibration_json_path(scene_spec: object, *, repo_root: Path) -> Path | None:
    raw = str(os.environ.get(CALIB_JSON_ENV, "") or "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        return candidate if candidate.is_file() else None

    meta = getattr(scene_spec, "metadata", None) or {}
    if not isinstance(meta, dict):
        return None
    meta_rel = meta.get("human_ue_calibration")
    if meta_rel:
        candidate = Path(str(meta_rel))
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        if candidate.is_file():
            return candidate

    motion = getattr(scene_spec, "motion", None)
    seq_path = getattr(motion, "resolved_sequence_npz_path", None) if motion is not None else None
    if seq_path is not None:
        sibling = Path(seq_path).expanduser().resolve().parent / CALIB_SIDECAR_NAME
        if sibling.is_file():
            return sibling
    return None


def load_human_ue_calibration_dict(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or int(data.get("schema_version", 0)) != HUMAN_UE_CALIBRATION_SCHEMA_VERSION:
        return None
    return data


def parse_per_bone_offsets(calibration: Mapping[str, Any] | None) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    if not calibration or not isinstance(calibration.get("per_bone_rotator_offset_deg"), dict):
        return out
    raw = calibration["per_bone_rotator_offset_deg"]
    assert isinstance(raw, dict)
    for bone, trip in raw.items():
        if not isinstance(trip, dict):
            continue
        key = str(bone).lower()
        out[key] = (
            float(trip.get("roll_deg", trip.get("roll", 0.0))),
            float(trip.get("pitch_deg", trip.get("pitch", 0.0))),
            float(trip.get("yaw_deg", trip.get("yaw", 0.0))),
        )
    return out


def read_human_scene_placement_mesh_offset_m(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float], bool] | None:
    """Return (human_anchor_world_m, genesis_mesh_combined_offset_m, align_floor) or None."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        return None
    try:
        w = payload["world_offset_m"]
        ox, oy, oz = float(w[0]), float(w[1]), float(w[2])
        extra_z = float(payload.get("display_vertical_sink_m", 0.0)) + float(payload.get("display_vertical_offset_m", 0.0))
        anchor = payload["human_anchor_world_m"]
        anchor_m = (float(anchor[0]), float(anchor[1]), float(anchor[2]))
        mesh_off = (ox, oy, oz + extra_z)
        align_floor = bool(payload.get("align_floor", True))
        return anchor_m, mesh_off, align_floor
    except (KeyError, IndexError, TypeError, ValueError):
        return None
