#!/usr/bin/env python3
"""Build a self-contained 3-D audit view for the dynamic leg scan.

The scanner writes one JSON object per paired episode in ``RUNS.jsonl`` and
the geometry builder writes ``DYNAMIC_GEOMETRY.npz``.  The detailed traces use
the smallest seed shared by every displayed method; aggregate rows cover all
episodes and are not used to select that seed.
"""
from __future__ import annotations

import argparse
from html import escape as html_escape
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
METHOD_ORDER = ("A", "fixed", "B", "A_joint", "B_joint", "B_open_loop", "A_independent")
METHOD_LABELS = {
    "A": "Strong A (greedy)", "fixed": "Fixed continuation", "B": "B (observation-conditioned)",
    "A_joint": "Strong A (greedy)", "B_joint": "B (observation-conditioned)",
    "B_open_loop": "B (open-loop)", "A_independent": "Strong A (independent)",
}
METHOD_COLORS = {
    "A": "#c26d18", "fixed": "#2471a3", "B": "#6545a6",
    "A_joint": "#c26d18", "B_joint": "#6545a6", "B_open_loop": "#2471a3",
    "A_independent": "#a84f2b",
}


def _first(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"line {line_no} of {path} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"no episode rows found in {path}")
    return rows


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _metadata(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    raw = arrays.get("metadata_json")
    if raw is None:
        return {}
    try:
        value = raw.item() if raw.ndim == 0 else raw.tolist()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_float_array(value: Any, *, ndim: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if ndim is not None and arr.ndim != ndim:
        return None
    return arr


def _as_bool_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=bool)
    except (TypeError, ValueError):
        return None


def _normalise_axes(axis: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    out = np.asarray(axis, dtype=float)
    if out.ndim != 2 or out.shape[1] != 3:
        return np.asarray(fallback if fallback is not None else np.zeros((0, 3)), dtype=float)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    if fallback is None:
        fallback = np.zeros_like(out)
    fb = np.asarray(fallback, dtype=float)
    if fb.shape != out.shape:
        fb = np.zeros_like(out)
    fb_norm = np.linalg.norm(fb, axis=1, keepdims=True)
    fb = np.divide(fb, np.maximum(fb_norm, 1.0e-12), out=np.zeros_like(fb), where=fb_norm > 1.0e-12)
    return np.divide(out, np.maximum(norm, 1.0e-12), out=fb.copy(), where=norm > 1.0e-12)


def _to_meters(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.size and np.nanmax(np.abs(arr)) > 10.0:
        return arr / 1000.0
    return arr


def _pose_count(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> int:
    angles = _first(arrays, "pose_knee_deg", "pose_angles_deg", "pose_deg")
    if angles is not None:
        value = np.asarray(angles)
        if value.ndim == 1 and len(value):
            return int(len(value))
    schedule = metadata.get("pose_schedule")
    if isinstance(schedule, Mapping):
        values = schedule.get("angles_deg")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and len(values):
            return int(len(values))
    for name in ("posed_vessel_m", "pose_vessel_m", "posed_views_m", "pose_views_m", "posed_frames"):
        value = arrays.get(name)
        if value is not None and np.asarray(value).ndim >= 4:
            return int(np.asarray(value).shape[0])
    return 1


def _pose_angles(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], count: int) -> np.ndarray:
    value = _as_float_array(_first(arrays, "pose_knee_deg", "pose_angles_deg", "pose_deg"), ndim=1)
    if value is None:
        schedule = metadata.get("pose_schedule")
        if isinstance(schedule, Mapping):
            value = _as_float_array(schedule.get("angles_deg"), ndim=1)
    if value is None or len(value) != count:
        return np.arange(count, dtype=float)
    return value


def _pose_names(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], count: int) -> list[str]:
    value = _first(arrays, "pose_state_names", "pose_names", "state_names")
    if value is None:
        schedule = metadata.get("pose_schedule")
        value = schedule.get("state_names") if isinstance(schedule, Mapping) else None
    if value is not None:
        names = [str(x) for x in np.asarray(value).reshape(-1).tolist()]
        if len(names) == count:
            return names
    return [f"pose {i}" for i in range(count)]


def _pose_select(value: Any, pose: int, count: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if count > 1 and arr.ndim >= 1 and arr.shape[0] == count:
        return np.asarray(arr[int(np.clip(pose, 0, count - 1))])
    return np.asarray(arr)


def _station_s(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], n_hint: int | None = None) -> np.ndarray:
    value = _as_float_array(_first(arrays, "s_mm", "station_s_mm", "arc_s_mm"), ndim=1)
    if value is None:
        table = metadata.get("station_table")
        if isinstance(table, Mapping):
            value = _as_float_array(table.get("s_mm"), ndim=1)
            if value is None:
                value = _as_float_array(table.get("station_s_mm"), ndim=1)
    if value is not None and len(value):
        return value
    return np.arange(int(n_hint or 0), dtype=float)


def _station_count(arrays: Mapping[str, np.ndarray], s: np.ndarray) -> int:
    if len(s):
        return len(s)
    for name in ("vessel_m", "posed_vessel_m", "skin_m", "posed_skin_m", "views_m", "posed_views_m"):
        value = arrays.get(name)
        if value is None:
            continue
        shape = np.asarray(value).shape
        if len(shape) >= 2:
            return int(shape[-2] if shape[-1] == 3 else shape[-1])
    return 0


def _pose_line(arrays: Mapping[str, np.ndarray], pose: int, count: int,
               names: Sequence[str], n: int) -> np.ndarray:
    arr = _pose_select(_first(arrays, *names), pose, count)
    if arr is None:
        return np.empty((0, 3), dtype=float)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[-1] != 3:
        return np.empty((0, 3), dtype=float)
    return _to_meters(arr[:n] if n and len(arr) > n else arr)


def _branch_indices(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], n: int) -> np.ndarray:
    value = _first(arrays, "branch_stations", "branch_indices", "branch_station_indices")
    if value is None:
        value = metadata.get("branch_stations")
    if value is not None:
        out = np.asarray(value, dtype=int).reshape(-1)
        return out[(out >= 0) & (out < n)]
    labels = _first(arrays, "labels", "station_labels")
    if labels is not None:
        text = np.asarray(labels).reshape(-1).astype(str)
        mask = np.asarray([any(token in item.upper() for token in ("POPV", "BRANCH", "SIDE")) for item in text])
        return np.flatnonzero(mask & (np.arange(len(mask)) < n))
    return np.empty(0, dtype=int)


def _central_candidate(arrays: Mapping[str, np.ndarray], k: int) -> int:
    offsets = _as_float_array(_first(arrays, "view_offsets_mm", "candidate_offset_mm", "offsets_mm"), ndim=1)
    tilts = _as_float_array(_first(arrays, "view_tilts_deg", "candidate_tilt_deg", "tilts_deg"), ndim=1)
    if offsets is None or len(offsets) != k:
        return min(3, max(0, k - 1))
    score = np.abs(offsets)
    if tilts is not None and len(tilts) == k:
        score = score + 100.0 * (np.abs(tilts) > 1.0e-8)
    return int(np.argmin(score))


def _pose_views(arrays: Mapping[str, np.ndarray], pose: int, count: int) -> np.ndarray:
    arr = _pose_select(_first(arrays, "posed_views_m", "pose_views_m", "views_pose_m", "views_m"), pose, count)
    if arr is None:
        return np.empty((0, 0, 3), dtype=float)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return np.empty((0, 0, 3), dtype=float)
    return _to_meters(arr)


def _lateral_axes(arrays: Mapping[str, np.ndarray], pose: int, count: int, n: int, k: int) -> np.ndarray:
    arr = _pose_select(_first(arrays, "posed_frames", "pose_frames", "frames_pose", "frames"), pose, count)
    axis: np.ndarray | None = None
    if arr is not None:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 4 and arr.shape[-2:] == (3, 3):
            if arr.shape[0] == n and arr.shape[1] == k:
                axis = arr[:, _central_candidate(arrays, k), :, 1]
            elif arr.shape[0] == n and arr.shape[1] == 3:
                axis = arr[:, :, 1]
        elif arr.ndim == 3 and arr.shape[-2:] == (3, 3) and arr.shape[0] == n:
            axis = arr[:, :, 1]
    if axis is None:
        views = _pose_views(arrays, pose, count)
        if views.shape[0] == n and views.shape[1] >= 2:
            center = _central_candidate(arrays, views.shape[1])
            axis = views[:, min(views.shape[1] - 1, center + 1)] - views[:, max(0, center - 1)]
    if axis is None:
        axis = np.tile(np.asarray([[0.0, 1.0, 0.0]]), (n, 1))
    return _normalise_axes(np.asarray(axis, dtype=float))


def _ensure_derived_branch(arrays: dict[str, np.ndarray]) -> None:
    """Populate branch aliases from the scanner's deterministic branch helper."""

    if any(name in arrays for name in ("branch_dense_m", "posed_branch_dense_m", "daughter_m")):
        return
    try:
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        from dynamic_scan import posed_branch  # type: ignore

        dense, dense_s, daughter, indices, fork = posed_branch(arrays)
        arrays["posed_branch_dense_m"] = np.asarray(dense)
        arrays["branch_dense_s_mm"] = np.asarray(dense_s)
        arrays["posed_daughter_m"] = np.asarray(daughter)
        arrays["branch_stations"] = np.asarray(indices, dtype=int)
        arrays["fork_station"] = np.asarray(fork, dtype=int)
    except Exception:
        # A geometry-only smoke table may intentionally omit the synthetic
        # branch; the rest of the figure remains useful in that case.
        return


def _pose_branch(arrays: Mapping[str, np.ndarray], pose: int, count: int, indices: np.ndarray) -> np.ndarray:
    arr = _pose_select(_first(
        arrays, "posed_branch_dense_m", "pose_branch_dense_m", "branch_dense_pose_m", "posed_branch_m",
        "branch_dense_m", "branch_dense_centerline_m", "dense_branch_m", "branch_centerline_m", "branch_m",
    ), pose, count)
    if arr is not None:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2 and arr.shape[-1] == 3:
            return _to_meters(arr)
    arr = _pose_select(_first(arrays, "posed_daughter_m", "pose_daughter_m", "daughter_pose_m", "daughter_m", "daughter"), pose, count)
    if arr is not None:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2 and arr.shape[-1] == 3:
            return _to_meters(arr[indices] if len(indices) else arr)
    return np.empty((0, 3), dtype=float)


def _branch_s(arrays: Mapping[str, np.ndarray], length: int, station_s: np.ndarray, indices: np.ndarray) -> np.ndarray:
    value = _as_float_array(_first(arrays, "branch_dense_s_mm", "branch_s_mm", "dense_branch_s_mm"), ndim=1)
    if value is not None and len(value) == length:
        return value
    if len(indices) == length and len(station_s):
        return station_s[indices]
    if length:
        return np.linspace(float(station_s[indices[0]]) if len(indices) else 0.0,
                           float(station_s[indices[-1]]) if len(indices) else max(1.0, float(length - 1)), length)
    return np.empty(0, dtype=float)


def _surface_asset_path(metadata: Mapping[str, Any]) -> Path | None:
    assets = metadata.get("source_assets")
    if isinstance(assets, Mapping):
        for key in ("contact_surface", "surface", "skin"):
            item = assets.get(key)
            if isinstance(item, Mapping) and item.get("path"):
                path = Path(str(item["path"]))
                if path.is_file():
                    return path
    default = (HERE.parent / "ab_validation_20261110" / "repair_20261110"
               / "harmonic_full_domain_followup" / "contact_multichart_right.npz")
    return default if default.is_file() else None


def _posed_vertices(vertices: np.ndarray, weights: np.ndarray | None,
                    transforms: np.ndarray | None, pose: int, count: int) -> tuple[np.ndarray, bool]:
    """Apply the same forward LBS blend used by the geometry builder."""

    if weights is None or transforms is None:
        return _to_meters(vertices), False
    tf = _pose_select(transforms, pose, count)
    if tf is None:
        return _to_meters(vertices), False
    tf = np.asarray(tf, dtype=float)
    w = np.asarray(weights, dtype=float)
    v = np.asarray(vertices, dtype=float)
    if tf.ndim != 3 or tf.shape[1:] != (4, 4) or w.ndim != 2 or w.shape[0] != len(v) or w.shape[1] != tf.shape[0]:
        return _to_meters(v), False
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1.0e-12)
    blend = np.einsum("nk,kij->nij", w, tf)
    homo = np.concatenate((v, np.ones((len(v), 1))), axis=1)
    return np.einsum("nij,nj->ni", blend, homo)[:, :3], True


def _skin_mesh(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any], pose: int,
               count: int) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    vertices_value = _first(
        arrays, "posed_skin_vertices_m", "skin_vertices_pose_m", "pose_skin_vertices_m", "skin_pose_vertices_m",
        "skin_vertices_m", "skin_vertices",
    )
    faces_value = _first(arrays, "posed_skin_faces", "skin_faces", "faces")
    weights_value = _first(arrays, "skin_lbs_weights", "skin_weights", "weights")
    posed = False
    if vertices_value is None or faces_value is None:
        asset = _surface_asset_path(metadata)
        if asset is None:
            return np.empty((0, 3)), np.empty((0, 3), dtype=int), False, False
        try:
            source = _load_npz(asset)
            vertices_value = _first(source, "vertices", "skin_vertices", "vertices_m")
            faces_value = _first(source, "faces", "skin_faces")
            weights_value = _first(source, "weights", "skin_weights")
        except (OSError, ValueError):
            return np.empty((0, 3)), np.empty((0, 3), dtype=int), False, False
    vertices = _pose_select(vertices_value, pose, count)
    vertices = np.asarray(vertices, dtype=float) if vertices is not None else np.empty((0, 3))
    faces = np.asarray(faces_value, dtype=int)
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int), False, False
    # Posed arrays already contain the LBS result.  Canonical source skin is
    # transported here using the stored joint transforms when available.
    if vertices_value is not None and np.asarray(vertices_value).ndim == 2:
        vertices, posed = _posed_vertices(
            vertices, _as_float_array(weights_value), _first(arrays, "joint_transforms", "pose_joint_transforms"), pose, count,
        )
    valid = np.isfinite(vertices).all(axis=1)
    in_range = (faces >= 0).all(axis=1) & (faces < len(vertices)).all(axis=1)
    safe_faces = np.where(in_range[:, None], faces, 0)
    valid_faces = in_range & valid[safe_faces].all(axis=1)
    faces = faces[valid_faces]
    if not len(faces):
        return np.empty((0, 3)), np.empty((0, 3), dtype=int), False, posed
    return _to_meters(vertices), faces, True, posed


def _scenario_spec(summary: Mapping[str, Any], scenario: str) -> dict[str, Any]:
    values = summary.get("scenario_definition")
    if isinstance(values, list):
        for item in values:
            if isinstance(item, Mapping) and str(item.get("name")) == scenario:
                return dict(item)
    return {}


def _pose_for_scenario(scenario: str, spec: Mapping[str, Any], angles: np.ndarray, count: int) -> tuple[int, int]:
    initial = int(np.argmin(np.abs(angles))) if len(angles) else 0
    kind = str(spec.get("pose_kind", ""))
    if not kind:
        kind = "moving" if scenario.startswith("moving") else ("bent" if scenario.startswith("bent") else "straight")
    positive = np.flatnonzero(angles > 0.0)
    if kind == "bent" and len(positive):
        representative = int(positive[np.argmin(np.abs(angles[positive] - np.median(angles[positive])))])
    elif kind == "moving" and len(positive):
        representative = int(positive[np.argmax(angles[positive])])
    else:
        representative = initial
    return int(np.clip(initial, 0, max(0, count - 1))), int(np.clip(representative, 0, max(0, count - 1)))


def _method_sort_key(method: str) -> tuple[int, str]:
    try:
        return METHOD_ORDER.index(method), method
    except ValueError:
        return len(METHOD_ORDER), method


def _scenario_rows(rows: Sequence[Mapping[str, Any]], scenario: str | None) -> tuple[str, list[dict[str, Any]]]:
    names = sorted({str(row.get("scenario")) for row in rows if row.get("scenario") is not None})
    if scenario is None:
        scenario = names[0] if len(names) == 1 else (names[0] if names else "legacy")
    selected = [dict(row) for row in rows if str(row.get("scenario", scenario)) == scenario]
    if not selected and not names and scenario == "legacy":
        selected = [dict(row) for row in rows]
    if not selected:
        available = ", ".join(names) if names else "none"
        raise ValueError(f"scenario {scenario!r} not found; available: {available}")
    return scenario, selected


def _matched_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[int, list[dict[str, Any]], bool]:
    methods = sorted({str(row.get("method", "unknown")) for row in rows}, key=_method_sort_key)
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        try:
            seed = int(row.get("seed"))
        except (TypeError, ValueError):
            continue
        method = str(row.get("method", "unknown"))
        by_seed.setdefault(seed, {}).setdefault(method, dict(row))
    common = sorted(seed for seed, values in by_seed.items() if all(method in values for method in methods))
    if common:
        seed = int(common[0])
        return seed, [by_seed[seed][method] for method in methods], True
    if not by_seed:
        raise ValueError("selected rows have no integer seeds")
    seed = min(by_seed)
    return int(seed), [by_seed[seed][method] for method in methods if method in by_seed[seed]], False


def _row_mean(row: Mapping[str, Any]) -> np.ndarray:
    value = _first(row, "estimate", "mean", "posterior_mean")
    arr = _as_float_array(value, ndim=1)
    return arr if arr is not None else np.empty(0, dtype=float)


def _row_truth(row: Mapping[str, Any], dimension: int) -> np.ndarray:
    arr = _as_float_array(_first(row, "eval_only_truth", "truth", "simulated_truth"), ndim=1)
    if arr is None:
        return np.zeros(dimension, dtype=float)
    if len(arr) < dimension:
        return np.pad(arr, (0, dimension - len(arr)))
    return arr[:dimension]


def _covariance(row: Mapping[str, Any], dimension: int) -> np.ndarray | None:
    value = _as_float_array(_first(row, "cov", "posterior_cov"), ndim=2)
    if value is None or value.shape[0] < dimension or value.shape[1] < dimension:
        return None
    return value[:dimension, :dimension]


def _feature(s_mm: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * ((np.asarray(s_mm, dtype=float) - 55.0) / 100.0) ** 2)


def _slip_scale(angles: np.ndarray, pose: int) -> float:
    moving = np.flatnonzero(np.asarray(angles) > 0.0)
    if len(moving) < 2:
        return 0.0
    center = (float(np.min(angles[moving])) + float(np.max(angles[moving]))) / 2.0
    span = float(np.max(np.abs(angles[moving] - center)))
    return float((angles[int(pose)] - center) / span) if span > 1.0e-12 else 0.0


def _trunk_for_pose(arrays: Mapping[str, np.ndarray], pose: int, count: int, n: int) -> np.ndarray:
    return _pose_line(
        arrays, pose, count,
        ("posed_vessel_m", "pose_vessel_m", "vessel_pose_m", "vessel_m", "base_vessel_m", "trunk_m"), n,
    )


def _comparison_curves(arrays: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]], pose: int,
                      count: int, angles: np.ndarray, s_mm: np.ndarray, indices: np.ndarray,
                      scenario_spec: Mapping[str, Any]) -> dict[str, Any]:
    n = len(s_mm)
    trunk = _trunk_for_pose(arrays, pose, count, n)
    if len(trunk) != n:
        n = min(n, len(trunk))
        s_mm, trunk = s_mm[:n], trunk[:n]
    if n == 0:
        return {"prior": np.empty((0, 3)), "truth": np.empty((0, 3)), "post": {}, "band": {}, "branch": {}}
    offsets_value = _first(arrays, "view_offsets_mm", "candidate_offset_mm", "offsets_mm")
    views_probe = _pose_views(arrays, pose, count)
    k = int(np.asarray(offsets_value).size) if offsets_value is not None else (views_probe.shape[1] if views_probe.ndim == 3 else 1)
    lateral = _lateral_axes(arrays, pose, count, n, max(1, k))
    moving = str(scenario_spec.get("pose_kind", "")) == "moving"
    if not moving:
        poses_seen = {int(step.get("pose", pose)) for row in rows for step in row.get("steps", []) if isinstance(step, Mapping)}
        moving = len(poses_seen) > 1
    slip = _slip_scale(angles, pose) if moving else 0.0
    feat = _feature(s_mm)
    branch_base = _pose_branch(arrays, pose, count, indices)
    branch_s = _branch_s(arrays, len(branch_base), s_mm, indices)
    branch_lateral = np.column_stack([np.interp(branch_s, s_mm, lateral[:, axis]) for axis in range(3)]) if len(branch_base) else np.empty((0, 3))
    branch_lateral = _normalise_axes(branch_lateral)
    prior = trunk.copy()
    truth_curve = trunk.copy()
    post: dict[str, np.ndarray] = {}
    band: dict[str, np.ndarray] = {}
    branch_post: dict[str, np.ndarray] = {}
    branch_truth = branch_base.copy()
    first_truth: np.ndarray | None = None
    for row in rows:
        method = str(row.get("method", "unknown"))
        mean = _row_mean(row)
        if not len(mean):
            continue
        nrow = min(n, len(mean))
        truth = _row_truth(row, len(mean))
        offset = np.zeros(n, dtype=float)
        true_offset = np.zeros(n, dtype=float)
        offset[:nrow], true_offset[:nrow] = mean[:nrow], truth[:nrow]
        if moving and len(mean) > n:
            offset += slip * feat * float(mean[-1])
        if moving and len(truth) > n:
            true_offset += slip * feat * float(truth[-1])
        post[method] = trunk + lateral * (offset[:, None] / 1000.0)
        if first_truth is None:
            first_truth = true_offset
            truth_curve = trunk + lateral * (true_offset[:, None] / 1000.0)
        branch_offset = np.interp(branch_s, s_mm, offset) if len(branch_base) else np.empty(0)
        true_branch_offset = np.interp(branch_s, s_mm, true_offset) if len(branch_base) else np.empty(0)
        if len(mean) >= n + 2 and len(branch_base):
            f = np.clip((branch_s - branch_s[0]) / max(1.0e-12, branch_s[-1] - branch_s[0]), 0.0, 1.0)
            branch_offset += f * float(mean[-2])
            true_branch_offset += f * float(truth[-2])
        if moving and len(mean) > n + 1 and len(branch_base):
            branch_offset += slip * np.interp(branch_s, s_mm, feat) * float(mean[-1])
        if moving and len(truth) > n + 1 and len(branch_base):
            true_branch_offset += slip * np.interp(branch_s, s_mm, feat) * float(truth[-1])
        if len(branch_base):
            branch_post[method] = branch_base + branch_lateral * (branch_offset[:, None] / 1000.0)
            branch_truth = branch_base + branch_lateral * (true_branch_offset[:, None] / 1000.0)
        cov = _covariance(row, len(mean))
        if cov is not None:
            band[method] = 1.644854 * np.sqrt(np.maximum(0.0, np.diag(cov)[:n]))
    return {
        "prior": prior, "truth": truth_curve, "post": post, "band": band,
        "branch_prior": branch_base, "branch_truth": branch_truth, "branch_post": branch_post,
        "branch_lateral": branch_lateral, "lateral": lateral,
    }


def _line_trace(go: Any, points: np.ndarray, name: str, color: str, *, width: int = 5,
                dash: str | None = None, legendgroup: str | None = None,
                showlegend: bool = True, opacity: float = 1.0) -> Any | None:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or not len(pts):
        return None
    xyz = pts * 1000.0
    line = {"color": color, "width": width}
    if dash:
        line["dash"] = dash
    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines", name=name,
        line=line, opacity=opacity, legendgroup=legendgroup or name,
        showlegend=showlegend, hovertemplate=f"{html_escape(name)}<extra></extra>",
    )


def _mesh_trace(go: Any, vertices: np.ndarray, faces: np.ndarray, name: str,
                color: str, opacity: float, *, showlegend: bool = True) -> Any | None:
    if len(vertices) == 0 or len(faces) == 0:
        return None
    xyz = np.asarray(vertices, dtype=float) * 1000.0
    tri = np.asarray(faces, dtype=int)
    return go.Mesh3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
        name=name, color=color, opacity=opacity, hoverinfo="skip", showlegend=showlegend,
        legendgroup=name,
    )


def _ribbon_trace(go: Any, points: np.ndarray, axes: np.ndarray, band: np.ndarray,
                  name: str, color: str, *, showlegend: bool = False) -> Any | None:
    p, a, b = np.asarray(points, dtype=float), np.asarray(axes, dtype=float), np.asarray(band, dtype=float)
    m = min(len(p), len(a), len(b))
    if m < 2:
        return None
    p, a, b = p[:m], a[:m], b[:m]
    good = np.isfinite(p).all(axis=1) & np.isfinite(a).all(axis=1) & np.isfinite(b)
    if good.sum() < 2:
        return None
    p, a, b = p[good], a[good], b[good]
    edge = a * (b[:, None] / 1000.0)
    vertices = np.vstack((p - edge, p + edge)) * 1000.0
    n = len(p)
    idx = np.arange(n - 1)
    faces = np.vstack((np.c_[idx, idx + 1, idx + n], np.c_[idx + 1, idx + n, idx + n + 1]))
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], name=name, color=color,
        opacity=0.16, hoverinfo="skip", showlegend=showlegend, legendgroup=name,
    )


def _window_traces(go: Any, arrays: Mapping[str, np.ndarray], pose: int, count: int,
                   n: int, *, showlegend: bool) -> list[Any]:
    views = _pose_views(arrays, pose, count)
    if views.ndim != 3 or views.shape[0] != n:
        return []
    n, k = views.shape[:2]
    finite = np.isfinite(views).all(axis=2)
    projection = _as_bool_array(_first(arrays, "view_projection_feasible", "projection_feasible"))
    projection = _pose_select(projection, pose, count) if projection is not None else None
    if projection is None or np.asarray(projection).shape != (n, k):
        projection = finite.copy()
    else:
        projection = np.asarray(projection, dtype=bool) & finite
    found = _as_bool_array(_first(arrays, "static_found", "retained_candidate_mask", "candidate_static_found"))
    found = _pose_select(found, pose, count) if found is not None else None
    if found is None or np.asarray(found).shape != (n, k):
        found = projection.copy()
    else:
        found = np.asarray(found, dtype=bool) & projection
    margin = _first(arrays, "dird_margin", "chosen_dird_margin", "candidate_dird_margin")
    margin_arr = _pose_select(margin, pose, count) if margin is not None else None
    margin_arr = np.asarray(margin_arr, dtype=float) if margin_arr is not None else np.zeros((n, k), dtype=float)
    if margin_arr.ndim == 3:
        margin_arr = np.nanmax(margin_arr, axis=-1)
    if margin_arr.shape != (n, k):
        margin_arr = np.zeros((n, k), dtype=float)
    offsets = _as_float_array(_first(arrays, "view_offsets_mm", "candidate_offset_mm", "offsets_mm"), ndim=1)
    tilts = _as_float_array(_first(arrays, "view_tilts_deg", "candidate_tilt_deg", "tilts_deg"), ndim=1)
    offsets = offsets if offsets is not None and len(offsets) == k else np.arange(k, dtype=float)
    tilts = tilts if tilts is not None and len(tilts) == k else np.zeros(k, dtype=float)
    station_s = _station_s(arrays, {}, n)
    station_s = station_s if len(station_s) == n else np.arange(n, dtype=float)
    points, valid = views.reshape(-1, 3), projection.reshape(-1)
    flat_found, flat_margin = found.reshape(-1), margin_arr.reshape(-1)
    custom = np.column_stack((
        np.repeat(station_s, k), np.tile(offsets, n), np.tile(tilts, n), flat_margin,
        flat_found.astype(int), valid.astype(int),
    ))
    traces: list[Any] = []
    if valid.any():
        xyz = points[valid] * 1000.0
        traces.append(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
            name="Checked contact windows",
            marker=dict(size=3.5, color=flat_margin[valid], colorscale="Viridis", showscale=True,
                        colorbar=dict(title="dIRD<br>margin", len=0.5)),
            legendgroup="windows", showlegend=showlegend, customdata=custom[valid],
            hovertemplate="s=%{customdata[0]:.1f} mm · offset=%{customdata[1]:+.1f} mm"
                         " · tilt=%{customdata[2]:+.1f}°<br>margin=%{customdata[3]:.2f}"
                         " · endpoint=%{customdata[4]}<extra></extra>",
        ))
    rejected = valid & ~flat_found
    if rejected.any():
        xyz = points[rejected] * 1000.0
        traces.append(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
            name="Checked window without endpoint witness", marker=dict(size=5, color="#d4534b", symbol="x"),
            legendgroup="windows_rejected", showlegend=showlegend, customdata=custom[rejected],
            hovertemplate="s=%{customdata[0]:.1f} mm · offset=%{customdata[1]:+.1f} mm"
                         " · rejected/unchecked endpoint<extra></extra>",
        ))
    return traces


def _motion_traces(go: Any, rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    traces: list[Any] = []
    for row in rows:
        method = str(row.get("method", "unknown"))
        color, label = METHOD_COLORS.get(method, "#4d4d4d"), METHOD_LABELS.get(method, method)
        pieces: list[np.ndarray] = []
        text: list[str] = []
        endpoints: list[np.ndarray] = []
        endpoint_text: list[str] = []
        steps = row.get("steps")
        if not isinstance(steps, list):
            steps = row.get("program", [])
        for step_no, step in enumerate(steps if isinstance(steps, list) else []):
            if not isinstance(step, Mapping):
                continue
            arr = _as_float_array(_first(step, "path_m", "motion_path_m", "path"))
            if arr is None or arr.ndim != 2 or arr.shape[1] != 3 or not len(arr):
                continue
            arr = _to_meters(arr)
            pieces.append(arr)
            target, pose, role = step.get("target", ""), step.get("pose", ""), step.get("role", "move")
            label_text = f"{label} · step {step_no} · target {target} · pose {pose} · {role}"
            text.extend([label_text] * len(arr))
            pieces.append(np.full((1, 3), np.nan)); text.append("")
            endpoints.append(arr[-1]); endpoint_text.append(label_text)
        if not pieces:
            continue
        path = np.vstack(pieces) * 1000.0
        traces.append(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines",
            name=f"{label} motion", line=dict(color=color, width=5), legendgroup=method,
            showlegend=True, text=text, hovertemplate="%{text}<extra></extra>",
        ))
        if endpoints:
            end = np.asarray(endpoints) * 1000.0
            traces.append(go.Scatter3d(
                x=end[:, 0], y=end[:, 1], z=end[:, 2], mode="markers",
                name=f"{label} contacts", marker=dict(color=color, size=4), legendgroup=method,
                showlegend=False, text=endpoint_text, hovertemplate="%{text}<extra></extra>",
            ))
    return traces


def _add_trace(fig: Any, trace: Any | None, *, col: int) -> None:
    if trace is not None:
        fig.add_trace(trace, row=1, col=col)


def _add_scene_content(fig: Any, go: Any, arrays: Mapping[str, np.ndarray], rows: Sequence[Mapping[str, Any]],
                       pose: int, count: int, angles: np.ndarray, names: Sequence[str], s_mm: np.ndarray,
                       indices: np.ndarray, spec: Mapping[str, Any], *, col: int,
                       showlegend: bool, include_motion: bool, metadata: Mapping[str, Any]) -> None:
    n = len(s_mm)
    vertices, faces, has_mesh, posed = _skin_mesh(arrays, metadata, pose, count)
    skin_name = "Real posed skin" if posed else "Real atlas skin (canonical)"
    _add_trace(fig, _mesh_trace(go, vertices, faces, skin_name, "#b9cbd6", 0.20, showlegend=showlegend), col=col)
    posed_skin_value = _first(arrays, "posed_skin_m", "skin_pose_m", "posed_skin_centerline_m")
    skin = _pose_line(arrays, pose, count,
                      ("posed_skin_m", "skin_pose_m", "posed_skin_centerline_m") if posed_skin_value is not None else ("skin_m",), n)
    if len(skin):
        skin_label = "Real posed skin centerline" if posed_skin_value is not None else "Canonical skin station outline"
        _add_trace(fig, _line_trace(go, skin, skin_label, "#b4afa2", width=3,
                                    dash="dot", showlegend=showlegend), col=col)
    curves = _comparison_curves(arrays, rows, pose, count, angles, s_mm, indices, spec)
    _add_trace(fig, _line_trace(go, curves["prior"], "Prior mean (zero offset)", "#8a94a8", width=4,
                                dash="dash", showlegend=showlegend), col=col)
    _add_trace(fig, _line_trace(go, curves["truth"], "Simulated truth", "#bd5130", width=4,
                                dash="dot", showlegend=showlegend), col=col)
    _add_trace(fig, _line_trace(go, curves.get("branch_prior", np.empty((0, 3))), "Prior branch",
                                "#8a94a8", width=3, dash="dash", showlegend=False), col=col)
    _add_trace(fig, _line_trace(go, curves.get("branch_truth", np.empty((0, 3))), "Simulated truth branch",
                                "#bd5130", width=3, dash="dot", showlegend=False), col=col)
    for method in sorted(curves["post"], key=_method_sort_key):
        color, label, group = METHOD_COLORS.get(method, "#4d4d4d"), METHOD_LABELS.get(method, method), f"posterior_{method}"
        _add_trace(fig, _line_trace(go, curves["post"][method], f"{label} inferred mean", color,
                                    width=6, legendgroup=group, showlegend=showlegend), col=col)
        _add_trace(fig, _line_trace(go, curves.get("branch_post", {}).get(method, np.empty((0, 3))),
                                    f"{label} inferred branch", color, width=4, legendgroup=group,
                                    showlegend=False), col=col)
        band = curves.get("band", {}).get(method)
        if band is not None:
            _add_trace(fig, _ribbon_trace(go, curves["post"][method], curves["lateral"], band,
                                          f"{label} pointwise 90% band", color), col=col)
    for trace in _window_traces(go, arrays, pose, count, n, showlegend=showlegend):
        _add_trace(fig, trace, col=col)
    if include_motion:
        for trace in _motion_traces(go, rows):
            _add_trace(fig, trace, col=col)


def _aggregate(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], scenario: str) -> list[dict[str, Any]]:
    groups = summary.get("groups")
    out: list[dict[str, Any]] = []
    if isinstance(groups, Mapping):
        prefix = f"{scenario}/"
        for key, value in groups.items():
            if str(key).startswith(prefix) and isinstance(value, Mapping):
                out.append({"method": str(key)[len(prefix):], **dict(value)})
    if out:
        return sorted(out, key=lambda x: _method_sort_key(str(x["method"])))
    by_method: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row.get("method", "unknown")), []).append(row)
    for method, entries in by_method.items():
        target_count = max([int(row.get("targets", 0)) for row in entries if row.get("targets") is not None] or [0])
        correct = [len(row.get("covered", [])) - len(row.get("wrong", [])) for row in entries]
        paths = [float(row.get("path_mm", np.nan)) for row in entries]
        rmses = [float(row.get("map_rmse_mm", row.get("rmse_after_mm", np.nan))) for row in entries]
        bands = [float(row.get("band_max_mm", row.get("max_band_after_mm", np.nan))) for row in entries]
        out.append({
            "method": method, "episodes": len(entries),
            "complete": sum(bool(row.get("complete", False)) for row in entries),
            "correct_coverage": float(np.mean(np.asarray(correct) / max(1, target_count))),
            "mean_steps": float(np.mean([len(row.get("steps", row.get("program", []))) for row in entries])),
            "mean_path_mm": float(np.nanmean(paths)), "mean_rmse_mm": float(np.nanmean(rmses)),
            "mean_band_mm": float(np.nanmean(bands)),
        })
    return sorted(out, key=lambda x: _method_sort_key(str(x["method"])))


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html_escape(str(value))
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _aggregate_table(aggregate: Sequence[Mapping[str, Any]]) -> str:
    rows: list[str] = []
    for item in aggregate:
        method = str(item.get("method", "unknown"))
        episodes = int(item.get("episodes", 0) or 0)
        complete = item.get("complete", item.get("full_correct_coverage"))
        complete_text = f"{complete}/{episodes}" if complete is not None else "—"
        coverage = item.get("correct_coverage", item.get("correct_coverage_fraction"))
        try:
            coverage_text = _fmt(float(coverage) * 100.0 if coverage is not None and float(coverage) <= 1.0 else coverage, 1)
        except (TypeError, ValueError):
            coverage_text = "—"
        rows.append(
            "<tr>"
            f"<td>{html_escape(METHOD_LABELS.get(method, method))}</td><td>{complete_text}</td>"
            f"<td>{coverage_text}%</td><td>{_fmt(item.get('mean_steps'), 1)}</td>"
            f"<td>{_fmt(item.get('mean_path_mm'), 1)}</td><td>{_fmt(item.get('mean_rmse_mm'), 2)}</td>"
            f"<td>{_fmt(item.get('mean_band_mm'), 2)}</td></tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="7">No aggregate summary was recorded.</td></tr>')
    return (
        '<table class="aggregate"><thead><tr><th>method</th><th>complete</th><th>correct coverage</th>'
        '<th>mean steps</th><th>mean path (mm)</th><th>mean RMSE (mm)</th><th>mean max band (mm)</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _build_figure(arrays: dict[str, np.ndarray], rows: Sequence[Mapping[str, Any]], scenario: str,
                  summary: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    metadata = _metadata(arrays)
    _ensure_derived_branch(arrays)
    count = _pose_count(arrays, metadata)
    angles, names = _pose_angles(arrays, metadata, count), _pose_names(arrays, metadata, count)
    raw_s = _station_s(arrays, metadata)
    n = _station_count(arrays, raw_s)
    s_mm = raw_s if len(raw_s) == n else np.arange(n, dtype=float)
    indices = _branch_indices(arrays, metadata, n)
    spec = _scenario_spec(summary, scenario)
    pose_initial, pose_flex = _pose_for_scenario(scenario, spec, angles, count)
    title_initial = f"Initial · {names[pose_initial]} ({angles[pose_initial]:g}°)"
    title_flex = f"Representative · {names[pose_flex]} ({angles[pose_flex]:g}°)"
    figure = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                           subplot_titles=(title_initial, title_flex), horizontal_spacing=0.025)
    _add_scene_content(figure, go, arrays, rows, pose_initial, count, angles, names, s_mm, indices, spec,
                       col=1, showlegend=False, include_motion=False, metadata=metadata)
    _add_scene_content(figure, go, arrays, rows, pose_flex, count, angles, names, s_mm, indices, spec,
                       col=2, showlegend=True, include_motion=True, metadata=metadata)
    camera = dict(eye=dict(x=2.35, y=0.85, z=1.85), up=dict(x=0.12, y=1.0, z=0.0))
    for scene in ("scene", "scene2"):
        figure.update_layout(**{scene: dict(aspectmode="data", xaxis_title="Atlas X (mm)",
                                            yaxis_title="Atlas Y (mm)", zaxis_title="Atlas Z (mm)", camera=camera)})
    seed, _, paired = _matched_rows(rows)
    methods = [str(row.get("method", "unknown")) for row in rows]
    figure.update_layout(
        template="plotly_white", height=800,
        title=(f"Dynamic leg scan · {html_escape(scenario)}<br><sup>Detailed paired seed {seed} · "
               f"{'all methods matched' if paired else 'partial seed match'} · methods: "
               f"{html_escape(', '.join(methods))}</sup>"),
        margin=dict(t=112, b=100, l=12, r=12), font=dict(family="Arial", size=12, color="#243444"),
        legend=dict(orientation="h", y=-0.055, groupclick="togglegroup"), hoverlabel=dict(namelength=-1),
    )
    return figure, {"seed": seed, "paired": paired, "methods": methods, "pose_initial": pose_initial,
                    "pose_flex": pose_flex, "angles": angles.tolist(), "names": names,
                    "schema": metadata.get("schema", "unspecified"), "count": count}


def _html_document(figure: Any, info: Mapping[str, Any], *, run: Path, geometry: Path,
                   scenario: str, aggregate: Sequence[Mapping[str, Any]],
                   rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> str:
    plot = figure.to_html(full_html=False, include_plotlyjs=True,
                          config={"responsive": True, "displaylogo": False, "scrollZoom": True})
    selected_notes: list[str] = []
    for row in rows:
        method = str(row.get("method", "unknown"))
        steps = row.get("steps", row.get("program", []))
        covered, wrong = row.get("covered", []), row.get("wrong", [])
        correct = len(covered) - len(wrong) if isinstance(covered, list) and isinstance(wrong, list) else "—"
        selected_notes.append(
            f"<li><span class=\"swatch\" style=\"background:{METHOD_COLORS.get(method, '#4d4d4d')}\"></span>"
            f"{html_escape(METHOD_LABELS.get(method, method))}: {len(steps) if isinstance(steps, list) else '—'} steps; "
            f"{correct} correct targets; RMSE {_fmt(row.get('map_rmse_mm', row.get('rmse_after_mm')), 2)} mm</li>"
        )
    pose_text = ", ".join(f"{_fmt(a, 0)}° {html_escape(str(n))}" for a, n in zip(info["angles"], info["names"]))
    source_geometry = summary.get("geometry", "")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dynamic leg scan · {html_escape(scenario)}</title>
<style>
  :root {{ color-scheme: light; font-family: Arial, sans-serif; color: #243444; background: #f7f9fb; }}
  body {{ margin: 0; padding: 22px 24px 40px; }} h1 {{ margin: 0 0 4px; font-size: 24px; }}
  h2 {{ margin: 24px 0 8px; font-size: 16px; }} .subtitle {{ color: #536273; margin-bottom: 14px; }}
  .card {{ background: white; border: 1px solid #d8e0e8; border-radius: 10px; padding: 14px 16px; margin: 0 0 14px; box-shadow: 0 1px 2px #1c34420b; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(230px,1fr)); gap: 12px; }}
  .metric {{ font-size: 13px; color: #536273; min-width: 0; }} .metric strong {{ display: block; color: #243444; font-size: 13px; margin-top: 3px; overflow-wrap: anywhere; }}
  .aggregate {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  .aggregate th, .aggregate td {{ border-bottom: 1px solid #e2e8ee; padding: 7px 8px; text-align: right; }}
  .aggregate th:first-child, .aggregate td:first-child {{ text-align: left; }} .aggregate th {{ color: #536273; font-weight: 600; }}
  ul {{ margin: 6px 0 0 20px; padding: 0; line-height: 1.7; }} .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
  code {{ background: #eef2f5; border-radius: 3px; padding: 1px 4px; }} .fine {{ color: #667687; font-size: 12px; line-height: 1.55; }}
  .plotly-graph-div {{ width: 100% !important; }}
</style></head><body>
<div class="card"><h1>Dynamic leg scan · {html_escape(scenario)}</h1>
<div class="subtitle">Interactive atlas audit with initial and representative/flex poses. All episode traces are simulations.</div>
<div class="grid">
  <div class="metric">Detailed seed<strong>{info['seed']} ({'matched across methods' if info['paired'] else 'partial match'})</strong></div>
  <div class="metric">Methods<strong>{html_escape(', '.join(METHOD_LABELS.get(m, m) for m in info['methods']))}</strong></div>
  <div class="metric">Run directory<strong title="{html_escape(str(run))}">{html_escape(run.name)}</strong></div>
  <div class="metric">Geometry<strong title="{html_escape(str(geometry))}">{html_escape(geometry.name)}</strong></div>
</div></div>
<div class="card"><h2>Detailed paired episode</h2><ul>{''.join(selected_notes)}</ul>
<p class="fine">Prior mean, inferred posterior means, simulated truth, checked contact windows, endpoint-witness status, and method-colored motion are shown. Click legend entries to toggle method groups and drag/scroll to inspect the atlas.</p></div>
{plot}
<div class="card"><h2>Aggregate outcomes · all episodes in this scenario</h2>
{_aggregate_table(aggregate)}
<p class="fine">The detailed 3-D paths use the smallest seed present for every displayed method; aggregates are not used to select that seed.</p></div>
<div class="card fine"><strong>Source notes.</strong> {info['count']} pose state(s): {pose_text}. Geometry schema: <code>{html_escape(str(info['schema']))}</code>.
The recorded run summary points to <code>{html_escape(str(source_geometry))}</code>. This figure embeds Plotly and is standalone; it does not fetch a remote JavaScript bundle.</div>
</body></html>"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="run directory containing RUNS.jsonl and SUMMARY.json")
    parser.add_argument("--scenario", help="scenario name (defaults to the only recorded scenario)")
    parser.add_argument("--geometry", type=Path, help="geometry NPZ (defaults to summary or sibling file)")
    parser.add_argument("--output", type=Path, help="standalone HTML output path")
    args = parser.parse_args(argv)
    run = args.run.expanduser().resolve()
    rows = _read_jsonl(run / "RUNS.jsonl")
    summary_path = run / "SUMMARY.json"
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    scenario, scenario_rows = _scenario_rows(rows, args.scenario)
    seed, matched, paired = _matched_rows(scenario_rows)
    selected_rows = sorted(matched, key=lambda row: _method_sort_key(str(row.get("method", "unknown"))))
    geometry_value = args.geometry
    if geometry_value is None:
        summary_geometry = summary.get("geometry")
        if summary_geometry:
            candidate = Path(str(summary_geometry))
            geometry_value = candidate if candidate.is_file() else None
    if geometry_value is None:
        candidates = [run / "DYNAMIC_GEOMETRY.npz", run.parent / "DYNAMIC_GEOMETRY.npz",
                      HERE / "DYNAMIC_GEOMETRY.npz", HERE / "DYNAMIC_GEOMETRY_FROZEN.npz"]
        geometry_value = next((path for path in candidates if path.is_file()), candidates[-1])
    geometry = geometry_value.expanduser().resolve()
    arrays = _load_npz(geometry)
    figure, info = _build_figure(arrays, selected_rows, scenario, summary)
    if int(info["seed"]) != int(seed) or bool(info["paired"]) != bool(paired):
        raise RuntimeError("detailed seed selection changed while building the figure")
    aggregate_rows = [row for row in rows if str(row.get("scenario", scenario)) == scenario]
    aggregate = _aggregate(aggregate_rows, summary, scenario)
    output = args.output or run / f"DYNAMIC_{scenario.replace('/', '_')}_3D.html"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_html_document(figure, info, run=run, geometry=geometry, scenario=scenario,
                                     aggregate=aggregate, rows=selected_rows, summary=summary), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
