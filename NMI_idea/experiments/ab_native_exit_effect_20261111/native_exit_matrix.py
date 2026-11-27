#!/usr/bin/env python3
"""Run the pre-registered native continuation matrix.

This runner is intentionally a small, serial mechanical experiment around the
existing ``ab_continuing_rollout_20261111.NativeAdapter``.  It does not alter
the adapter, fixture, controller, wall, contact model, or any policy code.

For one fixture/mode, the default cohort is:

* the twelve canonical diagnostic actions in ``NativeAdapter.ACTION_IDS``
  order, checked from the exact root state;
* left and right terminal controls checked directly from the root; and
* for every valid root exit, both terminals checked from that complete native
  exit (the same terminal names, not a nominal reset state).

Rows are appended and flushed to a JSONL file after every cell.  Existing
complete-state checkpoints from the same fixture/mode are accepted only when
their adapter fingerprint and complete input fingerprint match exactly.  A
cache row records the source checkpoint and remains explicitly labelled as a
cache; uncovered cells use one serial native adapter in the current process.

Example::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      /media/camp/EXT_DRIVE/envs/genesis/bin/python native_exit_matrix.py \
      --fixture 0 --mode constrained

Subset examples::

    ... native_exit_matrix.py --fixture 0 --mode constrained --actions 0:4
    ... native_exit_matrix.py --fixture 0 --mode constrained \
        --actions side_left__single_dwell_0.15,11 --targets left

``--restart`` archives the current JSONL before starting a new run.  It does
not delete source study data or prior output traces.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping

# Set native numerical thread limits before importing numpy/scipy consumers.
# The run() guard repeats this assignment for callers that import this module
# first, while the early assignment covers normal CLI startup.
for _blas_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_blas_key] = "1"

import numpy as np


HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parent.parent / "ab_continuing_rollout_20261111"
STUDY = EXPERIMENT / "study_v4_closed_loop_20261111"
DEFAULT_ROOT = HERE.parent
SCHEMA = "native_exit_matrix_v1"
PROBE_COUNT = 12
TERMINALS = ("left", "right")


def _clean(value: Any) -> Any:
    """Convert numpy/scalar values to strict JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return [_clean(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def _json_dump(value: Any) -> str:
    return json.dumps(_clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_payload(state: Any, state_fingerprint) -> dict[str, Any]:
    """Record complete state, including dynamics and controller snapshot."""
    native = getattr(state, "native", None)
    return {
        "key": list(getattr(state, "key", ())),
        "elapsed_s": float(getattr(state, "elapsed", 0.0)),
        "pose": _clean(np.asarray(getattr(state, "pose"), dtype=float)),
        "valid": bool(getattr(state, "valid", False)),
        "fingerprint": str(state_fingerprint(state)),
        "native": None if native is None else {
            "q": _clean(np.asarray(native.q, dtype=float)),
            "qdot": _clean(np.asarray(native.qdot, dtype=float)),
            "acceleration": _clean(np.asarray(native.acceleration, dtype=float)),
            "snapshot": _clean(native.snapshot),
        },
    }


def _state_from_row(payload: Mapping[str, Any], state_from_payload):
    # Keep this helper in one place so resume follows exactly the adapter's
    # complete-state deserialisation contract.
    return state_from_payload(payload)


def _fingerprint_payload(payload: Mapping[str, Any]) -> str:
    return str(payload.get("fingerprint", ""))


def _route_kind(candidate: str | None) -> str | None:
    if candidate is None:
        return None
    return "direct" if str(candidate) == "direct" else "lawful_fallback"


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _trace_metrics(trace_path: Path | None, entry_pose: Any = None) -> dict[str, Any]:
    """Read one adapter NPZ trace and derive measured TCP path length."""
    if trace_path is None:
        return {
            "trace_path": None,
            "trace_exists": False,
            "trace_samples": None,
            "actual_tcp_path_length_m": None,
            "trace_error": "no_trace_path",
        }
    path = Path(trace_path)
    if not path.exists():
        return {
            "trace_path": str(path),
            "trace_exists": False,
            "trace_samples": None,
            "actual_tcp_path_length_m": None,
            "trace_error": "trace_missing",
        }
    try:
        with np.load(path, allow_pickle=False) as trace:
            actual = np.asarray(trace["actual"], dtype=float)
            if actual.ndim != 3 or actual.shape[1:] != (4, 4):
                raise ValueError(f"actual shape {actual.shape!r}")
            xyz = actual[:, :3, 3]
            entry_to_first = 0.0
            if entry_pose is not None and len(xyz):
                entry = np.asarray(entry_pose, dtype=float)
                if entry.shape == (4, 4):
                    entry_to_first = float(np.linalg.norm(xyz[0] - entry[:3, 3]))
            length = entry_to_first + (float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()) if len(xyz) > 1 else 0.0)
            return {
                "trace_path": str(path),
                "trace_exists": True,
                "trace_samples": int(len(actual)),
                "actual_tcp_path_length_m": length,
                "entry_to_first_sample_m": entry_to_first,
                "trace_final_time_s": _finite(np.asarray(trace["time_s"])[-1]) if "time_s" in trace.files and len(trace["time_s"]) else None,
                "trace_error": None,
            }
    except Exception as exc:  # diagnostics should survive an unreadable trace
        return {
            "trace_path": str(path),
            "trace_exists": False,
            "trace_samples": None,
            "actual_tcp_path_length_m": None,
            "trace_error": f"{type(exc).__name__}: {exc}",
        }


def _route_waypoints(adapter: Any, state: Any, action_id: str, candidate: str) -> list[list[float]] | None:
    """Recover public route waypoints without invoking another native replay."""
    try:
        from fixture import local_pose  # imported after sys.path is prepared

        capture = adapter.rig.captures[action_id]
        start = local_pose(adapter.measured_frame, np.asarray(state.pose, dtype=float))
        options = adapter.rig.route_options(start, capture.goal)
        points = options.get(candidate)
        return None if points is None else [_clean(np.asarray(point, dtype=float)) for point in points]
    except Exception:
        return None


def _workspace_gates(adapter: Any, trace_path: Path | None) -> dict[str, Any]:
    """Recompute the unchanged workspace/contact gates from the actual trace."""
    if trace_path is None or not Path(trace_path).exists():
        return {
            "trace_available": False,
            "workspace_in_bounds": None,
            "swept_wall_clear": None,
            "contact_gap_gate": None,
            "contact_gap_max_m": None,
            "reasons": ["trace_unavailable"],
        }
    try:
        from fixture import segment_hits_box

        with np.load(trace_path, allow_pickle=False) as trace:
            actual = np.asarray(trace["actual"], dtype=float)
            world = actual[:, :3, 3]
            public_xyz = (world - adapter.rig.measured[:3, 3]) @ adapter.rig.measured[:3, :3]
            workspace = all(bool(adapter.rig.geometry.check_local(point[:2])[0]) for point in public_xyz)
            swept = adapter.rig.wall is None or not any(
                segment_hits_box(a, b, adapter.rig.wall)
                for a, b in zip(public_xyz[:-1], public_xyz[1:])
            )
            true_xyz = (world - adapter.rig.actual_surface[:3, 3]) @ adapter.rig.actual_surface[:3, :3]
            gap = float(np.max(np.abs(true_xyz[:, 2]))) if len(true_xyz) else None
            contact = gap is not None and gap <= 0.003
            reasons = []
            if not workspace:
                reasons.append("workspace_out_of_bounds")
            if not swept:
                reasons.append("wall_swept")
            if not contact:
                reasons.append("contact_gap_gt_3mm")
            return {
                "trace_available": True,
                "workspace_in_bounds": workspace,
                "swept_wall_clear": swept,
                "contact_gap_gate": contact,
                "contact_gap_max_m": gap,
                "reasons": reasons,
            }
    except Exception as exc:
        return {
            "trace_available": False,
            "workspace_in_bounds": None,
            "swept_wall_clear": None,
            "contact_gap_gate": None,
            "contact_gap_max_m": None,
            "reasons": [f"gate_reconstruction_exception:{type(exc).__name__}: {exc}"],
        }


def _step_row(adapter: Any, state: Any, action_id: str, route_row: Mapping[str, Any], source_root: Path) -> dict[str, Any]:
    """Serialise one attempted route, preserving gates and exceptions."""
    candidate = route_row.get("candidate")
    waypoints = _route_waypoints(adapter, state, action_id, str(candidate)) if candidate else None
    route_dir = source_root
    steps = []
    actual_path = 0.0
    actual_path_known = True
    route_trace_paths = []
    gate_reasons: list[str] = []
    reconstruction_exceptions: list[Any] = []
    step_gate_rows = []
    for step in route_row.get("steps", ()):
        record = step.get("record")
        trace_path = None if not record else route_dir / f"{record}.npz"
        metrics = _trace_metrics(trace_path, step.get("entry_pose"))
        route_trace_paths.append(metrics.get("trace_path"))
        if metrics.get("actual_tcp_path_length_m") is None:
            actual_path_known = False
        else:
            actual_path += float(metrics["actual_tcp_path_length_m"])
        native_gate = bool(step.get("native_tracking_gate", False))
        aligned_position = _finite(step.get("same_tick_position_error_max_m"))
        aligned_rotation = _finite(step.get("same_tick_rotation_error_max_rad"))
        aligned_gate = aligned_position is not None and aligned_position <= 0.002 and aligned_rotation is not None and aligned_rotation <= math.radians(2.0)
        workspace = _workspace_gates(adapter, trace_path)
        paused = int(step.get("paused", 0) or 0)
        exit_valid = bool(step.get("exit_snapshot"))
        valid = bool(step.get("valid", False))
        row_reasons = []
        if not native_gate:
            row_reasons.append("native_tracking_gate_failed")
        if not aligned_gate:
            row_reasons.append("aligned_tracking_gate_failed")
        if workspace.get("workspace_in_bounds") is False:
            row_reasons.append("workspace_out_of_bounds")
        if workspace.get("swept_wall_clear") is False:
            row_reasons.append("wall_swept")
        if workspace.get("contact_gap_gate") is False:
            row_reasons.append("contact_gap_gt_3mm")
        if paused:
            row_reasons.append("task_paused")
        if not exit_valid:
            row_reasons.append("exit_state_invalid")
        if not valid and not row_reasons:
            row_reasons.append("native_or_workspace_guard")
        gate_reasons.extend(row_reasons)
        rec_error = step.get("reconstruction_error")
        if rec_error is not None:
            # robot_motion records a component-wise reconstruction diagnostic
            # (q/qdot/acceleration), not a scalar.  Preserve that diagnostic,
            # but classify only a non-finite or non-zero maximum as an
            # exception.  A zero-valued mapping is a clean reconstruction.
            if isinstance(rec_error, Mapping):
                rec_values = []
                for value in rec_error.values():
                    try:
                        rec_values.extend(np.asarray(value, dtype=float).reshape(-1).tolist())
                    except (TypeError, ValueError):
                        rec_values = [float("nan")]
                rec_max = max((abs(float(value)) for value in rec_values), default=0.0)
                if not math.isfinite(rec_max):
                    reconstruction_exceptions.append({"step": step.get("step"), "kind": "nonfinite_reconstruction_error", "value": rec_error})
                elif rec_max > 1e-10:
                    reconstruction_exceptions.append({"step": step.get("step"), "kind": "prefix_reconstruction_error", "max_abs_error": rec_max, "components": rec_error})
            else:
                rec_value = _finite(rec_error)
                if rec_value is None:
                    reconstruction_exceptions.append({"step": step.get("step"), "kind": "nonfinite_reconstruction_error", "value": rec_error})
                elif rec_value > 1e-10:
                    reconstruction_exceptions.append({"step": step.get("step"), "kind": "prefix_reconstruction_error", "max_abs_error": rec_value})
        steps.append({
            "step": step.get("step"),
            "record": record,
            "duration_s": _finite(step.get("duration_s")),
            "reference_duration_s": _finite(step.get("reference_duration_s")),
            "reference_kind": step.get("reference_kind"),
            "requested_dwell_s": _finite(step.get("requested_dwell_s")),
            "valid": valid,
            "tracking_pass": bool(step.get("tracking_pass", False)),
            "native_tracking_gate": native_gate,
            "aligned_tracking_gate": aligned_gate,
            "same_tick_position_error_max_m": aligned_position,
            "same_tick_rotation_error_max_rad": aligned_rotation,
            "paused_count": paused,
            "exit_state_valid": exit_valid,
            "minimum_self_collision_distance_m": _finite(step.get("minimum_self_collision_distance_m")),
            "native_failure_count": int(step.get("native_failure_count", 0) or 0),
            "native_failures": step.get("native_failures", []),
            "workspace_contact_gates": workspace,
            "gate_reasons": row_reasons,
            "reconstruction_error": rec_error,
            "trace": metrics,
            "entry_pose": step.get("entry_pose"),
            "entry_snapshot": step.get("entry_snapshot"),
            "exit_snapshot": step.get("exit_snapshot"),
        })
    return {
        "route_id": route_row.get("route"),
        "candidate": candidate,
        "route_kind": _route_kind(str(candidate) if candidate is not None else None),
        "waypoints": waypoints,
        "retreat_waypoints": [] if str(candidate) == "direct" else (None if waypoints is None else waypoints[:-1]),
        "retreat_waypoint_count": 0 if str(candidate) == "direct" else (None if waypoints is None else max(0, len(waypoints) - 1)),
        "predicted_duration_s": _finite(route_row.get("commanded_duration_prediction_s")),
        "transfer_time_s": _finite(route_row.get("duration_s")),
        "dird_mean_calibrated_margin": _finite(route_row.get("dird_mean_calibrated_margin")),
        "valid": bool(route_row.get("valid", False)),
        "actual_tcp_path_length_m": actual_path if actual_path_known else None,
        "trace_paths": route_trace_paths,
        "steps": steps,
        "gate_reasons": sorted(set(gate_reasons)),
        "reconstruction_exceptions": reconstruction_exceptions,
        "calculation_wall_seconds": _finite(route_row.get("calculation_wall_seconds")),
    }


def _candidate_rows(adapter: Any, state: Any, action_id: str, route_rows: Iterable[Mapping[str, Any]], source_root: Path) -> list[dict[str, Any]]:
    return [_step_row(adapter, state, action_id, row, source_root) for row in route_rows]


def _event_gate_summary(adapter: Any, state: Any, event: Any, route_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate true motion/contact gates from reconstruction diagnostics."""
    selected_id = getattr(event, "route_id", None)
    selected_routes = [route for route in route_rows if route.get("route_id") == selected_id]
    # Feasibility gates describe the selected native route.  Rejected route
    # diagnostics remain available in attempted_routes/failed_attempts and
    # must not poison the selected-route gate result.
    gate_routes = selected_routes if selected_routes else route_rows
    all_steps = [step for route in gate_routes for step in route.get("steps", [])]
    def all_or_unknown(field: str) -> bool | None:
        values = [step.get("workspace_contact_gates", {}).get(field) for step in all_steps]
        if not values or any(value is None for value in values):
            return None
        return bool(all(values))

    gate_results = {
        "event_feasible": bool(getattr(event, "feasible", False)),
        "native_tracking_gate": bool(all(step.get("native_tracking_gate") for step in all_steps)) if all_steps else False,
        "aligned_tracking_gate": bool(all(step.get("aligned_tracking_gate") for step in all_steps)) if all_steps else False,
        "workspace_in_bounds": all_or_unknown("workspace_in_bounds"),
        "swept_wall_clear": all_or_unknown("swept_wall_clear"),
        "contact_gap_gate": all_or_unknown("contact_gap_gate"),
        "not_paused": bool(all(step.get("paused_count", 0) == 0 for step in all_steps)) if all_steps else False,
        "exit_state_valid": bool(all(step.get("exit_state_valid", False) for step in all_steps)) if all_steps else False,
    }
    reasons = []
    if not gate_results["event_feasible"]:
        reasons.append("event_infeasible")
    for name, passed in gate_results.items():
        if name != "event_feasible" and passed is False:
            reasons.append(name + "_failed")
        elif name != "event_feasible" and passed is None:
            reasons.append(name + "_unavailable")
    reconstruction = [item for route in route_rows for item in route.get("reconstruction_exceptions", [])]
    return {
        "results": gate_results,
        "reasons": reasons,
        "reconstruction_exceptions": reconstruction,
    }


def _route_source_root(event_source: str, source_checkpoint: Path | None, adapter: Any) -> Path:
    if event_source == "cache" and source_checkpoint is not None:
        # Checkpoints live in planning/checkpoints/, while NPZ traces and
        # MOTION.json live in planning/.
        return source_checkpoint.parent.parent
    return adapter.planning_output if adapter.planning_output is not None else HERE


def _cache_candidates(index: int, mode: str, current_fp: Mapping[str, Any], action_id: str | None = None) -> list[tuple[Path, dict[str, Any]]]:
    """Find old same-fixture checkpoints; candidates are validated later."""
    expected = str(current_fp.get("sha256", ""))
    pattern = STUDY / "**" / "planning" / "checkpoints" / "*.json"
    found = []
    for path in sorted(STUDY.glob("**/planning/checkpoints/*.json")):
        text = str(path)
        marker = f"fixture_{index:02d}_{mode}"
        if marker not in text:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("schema") != "adaptive_native_checkpoint_v1":
            continue
        if int(payload.get("index", -1)) != index or payload.get("mode") != mode:
            continue
        if payload.get("adapter_fingerprint", {}).get("sha256") != expected:
            continue
        if action_id is not None and payload.get("action_id") != action_id:
            continue
        found.append((path, payload))
    return found


def _verified_cache(adapter: Any, state: Any, action_id: str, state_fingerprint, state_payload, cache_rows: list[tuple[Path, dict[str, Any]]]):
    """Copy one validated old checkpoint into this run and load it safely."""
    expected_input = str(state_fingerprint(state))
    # Prefer an old checkpoint whose immutable NPZ trace is still present.
    # Some historical directories retain a complete checkpoint after their
    # MOTION trace was pruned; those remain valid state caches, but cannot
    # support measured path/gate reconstruction.  Choosing trace-bearing
    # duplicates keeps the primary matrix rows quantitatively auditable.
    def trace_complete(item: tuple[Path, dict[str, Any]]) -> bool:
        source_path, payload = item
        planning = source_path.parent.parent
        records = payload.get("event", {}).get("trajectory_records", ())
        return bool(records) and all((planning / f"{record}.npz").exists() for record in records)

    ordered = sorted(cache_rows, key=lambda item: (not trace_complete(item), str(item[0])))
    for source_path, payload in ordered:
        if payload.get("input_fingerprint") != expected_input:
            continue
        output_payload = payload.get("output_state", {})
        native = output_payload.get("native")
        if not isinstance(native, Mapping):
            continue
        if any(key not in native for key in ("q", "qdot", "acceleration", "snapshot")):
            continue
        event_payload = payload.get("event", {})
        if not bool(event_payload.get("feasible")) or not bool(event_payload.get("result", {}).get("full_state_checkpoint")):
            continue
        target_path = adapter._checkpoint_path(state, action_id)
        if target_path is None:
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # Copy only into our own per-fixture output; source study data remain read-only.
        # Refresh the local pointer so a prior cache choice with pruned traces
        # cannot shadow the trace-bearing duplicate selected above.
        shutil.copy2(source_path, target_path)
        loaded = adapter._load_checkpoint(state, action_id)
        if loaded is None:
            continue
        return loaded, source_path, payload
    return None, None, None


def _route_rows_for_event(adapter: Any, state: Any, action_id: str, event: Any, source: str, source_checkpoint: Path | None) -> list[dict[str, Any]]:
    if source == "cache":
        raw_route = getattr(event, "result", {}).get("route", {})
        source_root = _route_source_root(source, source_checkpoint, adapter)
        # Cache checkpoints retain the selected route.  MOTION.json retains all
        # attempts; recover that exact prefix when present.
        motion_path = source_root / "MOTION.json"
        if motion_path.exists():
            try:
                motion = json.loads(motion_path.read_text(encoding="utf-8"))
                key = tuple(getattr(state, "key", ()))
                rows = [row for row in motion.get("routes", [])
                        if tuple(row.get("start", ())) == key and row.get("capture") == action_id]
                if rows:
                    return _candidate_rows(adapter, state, action_id, rows, source_root)
            except Exception:
                pass
        return _candidate_rows(adapter, state, action_id, [raw_route] if raw_route else [], source_root)
    rows = list(adapter._planning.rows)
    # A capture appends all candidates for this event contiguously.  Filter by
    # exact start prefix and capture so a resumed in-memory cache cannot leak.
    key = tuple(getattr(state, "key", ()))
    return _candidate_rows(adapter, state, action_id,
                           [row for row in rows if tuple(row.get("start", ())) == key and row.get("capture") == action_id],
                           adapter.planning_output if adapter.planning_output is not None else HERE)


def _cell_key(stage: str, root_action: str | None, target_action: str | None) -> str:
    return f"{stage}|{root_action or ''}|{target_action or ''}"


def _parse_actions(raw: str | None, actions: tuple[str, ...]) -> list[str]:
    if not raw:
        return list(actions[:PROBE_COUNT])
    result = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token and all(part.strip().lstrip("-").isdigit() for part in token.split(":", 1)):
            start_s, stop_s = token.split(":", 1)
            start, stop = int(start_s), int(stop_s)
            result.extend(actions[max(0, start):min(PROBE_COUNT, stop)])
        elif token.lstrip("-").isdigit():
            idx = int(token)
            if idx < 0 or idx >= PROBE_COUNT:
                raise ValueError(f"probe index out of range: {idx}")
            result.append(actions[idx])
        else:
            if token not in actions[:PROBE_COUNT]:
                raise ValueError(f"unknown canonical probe action: {token}")
            result.append(token)
    # Keep canonical ACTION_IDS order regardless of user token order.
    return [action for action in actions[:PROBE_COUNT] if action in set(result)]


def _parse_targets(raw: str) -> list[str]:
    result = [token.strip() for token in raw.split(",") if token.strip()]
    if not result or any(token not in TERMINALS for token in result):
        raise ValueError("--targets must contain left and/or right")
    return [target for target in TERMINALS if target in set(result)]


def _read_rows(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    done: dict[str, dict[str, Any]] = {}
    manifest = None
    if not path.exists():
        return done, manifest
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("row_type") == "manifest":
                manifest = row
            elif row.get("row_type") == "cell" and row.get("cell_key"):
                done[str(row["cell_key"])] = row
    return done, manifest


def _reconstruction_diagnostics(value: Any, step: Any) -> list[dict[str, Any]]:
    """Classify a stored component-wise reconstruction diagnostic."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        values = []
        try:
            for item in value.values():
                values.extend(np.asarray(item, dtype=float).reshape(-1).tolist())
            maximum = max((abs(float(item)) for item in values), default=0.0)
        except (TypeError, ValueError):
            maximum = float("nan")
        if math.isfinite(maximum) and maximum <= 1e-10:
            return []
        if math.isfinite(maximum):
            return [{"step": step.get("step"), "kind": "prefix_reconstruction_error", "max_abs_error": maximum, "components": value}]
        return [{"step": step.get("step"), "kind": "nonfinite_reconstruction_error", "value": value}]
    scalar = _finite(value)
    if scalar is None:
        return [{"step": step.get("step"), "kind": "nonfinite_reconstruction_error", "value": value}]
    if scalar <= 1e-10:
        return []
    return [{"step": step.get("step"), "kind": "prefix_reconstruction_error", "max_abs_error": scalar}]


def _recompute_route_path(route: dict[str, Any], row: Mapping[str, Any]) -> bool:
    """Recompute entry-inclusive TCP path metrics from immutable NPZ traces."""
    changed = False
    source_plan = None
    source_checkpoint = row.get("source_checkpoint")
    if source_checkpoint:
        source_plan = Path(source_checkpoint).parent.parent
    total = 0.0
    known = True
    trace_paths = []
    for step in route.get("steps", []):
        trace = step.get("trace") or {}
        raw_path = trace.get("trace_path")
        trace_path = Path(raw_path) if raw_path else None
        if (trace_path is None or not trace_path.exists()) and source_plan is not None and step.get("record"):
            candidate = source_plan / f"{step['record']}.npz"
            if candidate.exists():
                trace_path = candidate
                if trace.get("trace_path") != str(candidate):
                    trace["trace_path"] = str(candidate)
                    changed = True
        if trace_path is None or not trace_path.exists():
            known = False
            trace_paths.append(None if trace_path is None else str(trace_path))
            continue
        trace_paths.append(str(trace_path))
        try:
            with np.load(trace_path, allow_pickle=False) as loaded:
                actual = np.asarray(loaded["actual"], dtype=float)
                xyz = actual[:, :3, 3]
                entry = np.asarray(step.get("entry_pose"), dtype=float)
                entry_to_first = 0.0
                if entry.shape == (4, 4) and len(xyz):
                    entry_to_first = float(np.linalg.norm(xyz[0] - entry[:3, 3]))
                length = entry_to_first + (float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()) if len(xyz) > 1 else 0.0)
                if trace.get("entry_to_first_sample_m") != entry_to_first:
                    trace["entry_to_first_sample_m"] = entry_to_first
                    changed = True
                if trace.get("actual_tcp_path_length_m") != length:
                    trace["actual_tcp_path_length_m"] = length
                    changed = True
                if trace.get("trace_exists") is not True:
                    trace["trace_exists"] = True
                    changed = True
                total += length
        except Exception:
            known = False
        step["trace"] = trace
    value = total if known else None
    if route.get("actual_tcp_path_length_m") != value:
        route["actual_tcp_path_length_m"] = value
        changed = True
    if route.get("trace_paths") != trace_paths:
        route["trace_paths"] = trace_paths
        changed = True
    return changed


def _repair_derived_rows(jsonl: Path, fixture: int, mode: str) -> int:
    """Append corrected derived diagnostics without replaying native motion.

    Historical runners encoded the component-wise zero reconstruction map as
    a non-finite scalar exception.  The complete state and traces are still
    valid; this repair only reclassifies that derived diagnostic and leaves
    every source row untouched.  New rows are appended so restart/recovery
    remains auditable and the latest cell key is unambiguous.
    """
    latest, _ = _read_rows(jsonl)
    changed = []
    for key, original in latest.items():
        if original.get("fixture_index") != fixture or original.get("mode") != mode:
            continue
        row = json.loads(json.dumps(original))
        row_changed = False
        all_exceptions = []
        routes = []
        for route in row.get("attempted_routes", []):
            route_copy = json.loads(json.dumps(route))
            if _recompute_route_path(route_copy, row):
                row_changed = True
            route_exceptions = []
            steps = []
            for step in route_copy.get("steps", []):
                step_copy = json.loads(json.dumps(step))
                exceptions = _reconstruction_diagnostics(step_copy.get("reconstruction_error"), step_copy)
                if step_copy.get("reconstruction_exceptions", []) != exceptions:
                    row_changed = True
                step_copy["reconstruction_exceptions"] = exceptions
                route_exceptions.extend(exceptions)
                steps.append(step_copy)
            if route_copy.get("steps", []) != steps:
                row_changed = True
            route_copy["steps"] = steps
            if route_copy.get("reconstruction_exceptions", []) != route_exceptions:
                row_changed = True
            route_copy["reconstruction_exceptions"] = route_exceptions
            all_exceptions.extend(route_exceptions)
            routes.append(route_copy)
        if row.get("attempted_routes", []) != routes:
            row_changed = True
        row["attempted_routes"] = routes
        primary_id = row.get("route_id")
        selected_route = next((route for route in routes if route.get("route_id") == primary_id), None)
        if selected_route is not None and row.get("primary_route") != selected_route:
            row["primary_route"] = selected_route
            row["actual_tcp_path_length_m"] = selected_route.get("actual_tcp_path_length_m")
            row_changed = True
        if row.get("reconstruction_exceptions", []) != all_exceptions:
            row_changed = True
        row["reconstruction_exceptions"] = all_exceptions
        if row_changed:
            row["derived_diagnostics_repaired"] = {
                "kind": "recomputed_trace_path_and_reconstruction_diagnostics",
                "source_cell_key": key,
                "replayed_native": False,
                "source_traces_unchanged": True,
            }
            changed.append(row)
    if not changed:
        print(json.dumps({"schema": SCHEMA, "repaired": 0, "output": str(jsonl)}, sort_keys=True))
        return 0
    with jsonl.open("a", encoding="utf-8") as stream:
        _append(stream, {
            "schema": SCHEMA,
            "row_type": "derived_repair",
            "fixture_index": fixture,
            "mode": mode,
            "kind": "recomputed_trace_path_and_reconstruction_diagnostics",
            "repaired_cells": len(changed),
            "replayed_native": False,
            "source_traces_unchanged": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        for row in changed:
            _append(stream, row)
    print(json.dumps({"schema": SCHEMA, "repaired": len(changed), "output": str(jsonl)}, sort_keys=True))
    return 0


def _finalize_inventory(jsonl: Path, fixture: int, mode: str) -> int:
    """Append a full-cohort inventory after a subset/resume operation."""
    latest, manifest = _read_rows(jsonl)
    if manifest is None:
        raise FileNotFoundError(jsonl)
    actions = list(manifest.get("canonical_probe_order", manifest.get("actions", ())))
    targets = list(manifest.get("targets", TERMINALS))
    direct = bool(manifest.get("direct_controls", True))
    expected = []
    if direct:
        expected.extend(_cell_key("direct_control", None, target) for target in targets)
    expected.extend(_cell_key("root_probe", action, None) for action in actions)
    expected.extend(_cell_key("continuation", action, target)
                    for action in actions for target in targets)
    roots = [row for key, row in latest.items() if row.get("stage") == "root_probe"]
    cells = list(latest.values())
    summary = {
        "schema": SCHEMA,
        "row_type": "summary",
        "summary_kind": "full_matrix_inventory",
        "fixture_index": fixture,
        "mode": mode,
        "expected_cell_keys": expected,
        "emitted_cell_keys": sorted(latest),
        "root_probe_count": len(actions),
        "root_feasible_count": sum(int(row.get("feasible", False)) for row in roots),
        "continuation_expected_for_feasible_roots": sum(int(row.get("feasible", False)) for row in roots) * len(targets),
        "direct_control_count": len(targets) if direct else 0,
        "cache_rows": sum(int(row.get("source") == "cache") for row in cells),
        "fresh_rows": sum(int(row.get("source") == "fresh_native") for row in cells),
        "status_counts": dict(__import__("collections").Counter(str(row.get("status")) for row in cells)),
        "not_run_root_infeasible_or_unselected": len(set(expected) - set(latest)),
        "reconstruction_exception_rows": sum(int(bool(row.get("reconstruction_exceptions"))) for row in cells),
        "gate_reason_rows": sum(int(bool(row.get("gate_reasons"))) for row in cells),
        "no_checked_route_rows_are_finite_witnesses_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with jsonl.open("a", encoding="utf-8") as stream:
        _append(stream, summary)
    print(json.dumps({"schema": SCHEMA, "finalized": True, "output": str(jsonl), "cells": len(cells)}, sort_keys=True))
    return 0


def _append(stream, row: Mapping[str, Any]) -> None:
    stream.write(json.dumps(_clean(row), indent=None, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()


def _event_row(
    *, adapter: Any, state: Any, event: Any, state_fingerprint, state_payload, action_id: str,
    stage: str, root_action: str | None, target_action: str | None, source: str,
    source_checkpoint: Path | None, source_payload: Mapping[str, Any] | None,
    started: float, route_rows: list[dict[str, Any]], requested_goal: Any,
    cache_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = getattr(event, "output", None)
    route_id = getattr(event, "route_id", None)
    primary_candidate = None
    for route in route_rows:
        if route.get("route_id") == route_id:
            primary_candidate = route.get("candidate")
            break
    if primary_candidate is None and route_rows and bool(getattr(event, "feasible", False)):
        primary_candidate = next((r.get("candidate") for r in route_rows if r.get("valid")), None)
    selected = next((r for r in route_rows if r.get("route_id") == route_id), None)
    if selected is None and primary_candidate is not None:
        selected = next((r for r in route_rows if r.get("candidate") == primary_candidate and r.get("valid")), None)
    gate_summary = _event_gate_summary(adapter, state, event, route_rows)
    row = {
        "schema": SCHEMA,
        "row_type": "cell",
        "cell_key": _cell_key(stage, root_action, target_action),
        "fixture_index": int(adapter.index),
        "mode": adapter.mode,
        "stage": stage,
        "root_action": root_action,
        "target_action": target_action,
        "action_id": action_id,
        "source": source,
        "cache_validation": cache_validation,
        "source_checkpoint": None if source_checkpoint is None else str(source_checkpoint),
        "source_checkpoint_sha256": None if source_checkpoint is None else _sha256_path(source_checkpoint),
        "input_state": state_payload(state, state_fingerprint),
        "output_state": None if output is None else state_payload(output, state_fingerprint),
        "requested_goal": _clean(requested_goal),
        "feasible": bool(getattr(event, "feasible", False)),
        "status": "passed" if bool(getattr(event, "feasible", False)) else str(getattr(event, "result", {}).get("reason", "native_or_workspace_guard")),
        "reason": str(getattr(event, "result", {}).get("reason", "")),
        "route_id": route_id,
        "primary_route": selected,
        "primary_route_kind": _route_kind(primary_candidate),
        "attempted_routes": route_rows,
        "failed_attempts": [route for route in route_rows if not route.get("valid", False)],
        "transfer_time_s": _finite(getattr(event, "duration_s", None)),
        "elapsed_output_s": None if output is None else _finite(getattr(output, "elapsed", None)),
        "actual_tcp_path_length_m": None if selected is None else selected.get("actual_tcp_path_length_m"),
        "retreat_waypoints": None if selected is None else selected.get("retreat_waypoints"),
        "retreat_waypoint_count": None if selected is None else selected.get("retreat_waypoint_count"),
        "gate_results": gate_summary["results"],
        "gate_reasons": gate_summary["reasons"],
        "reconstruction_exceptions": gate_summary["reconstruction_exceptions"],
        "native_actual": _clean(getattr(event, "actual_output", {})),
        "trajectory_records": list(getattr(event, "trajectory_records", ())),
        "verification_calls": int(getattr(event, "verification_calls", 0)),
        "execution_calls": int(getattr(event, "execution_calls", 0)),
        "wall_seconds": time.perf_counter() - started,
        "adapter_fingerprint": adapter._adapter_fingerprint(),
    }
    if source_payload is not None:
        row["source_payload_schema"] = source_payload.get("schema")
        row["source_payload_input_fingerprint"] = source_payload.get("input_fingerprint")
        row["source_payload_output_fingerprint"] = _fingerprint_payload(source_payload.get("output_state", {}))
    return row


def _exception_row(*, adapter: Any, state: Any, action_id: str, stage: str,
                   root_action: str | None, target_action: str | None,
                   state_payload, state_fingerprint, requested_goal: Any,
                   started: float, exc: BaseException) -> dict[str, Any]:
    text = f"{type(exc).__name__}: {exc}"
    return {
        "schema": SCHEMA,
        "row_type": "cell",
        "cell_key": _cell_key(stage, root_action, target_action),
        "fixture_index": int(adapter.index),
        "mode": adapter.mode,
        "stage": stage,
        "root_action": root_action,
        "target_action": target_action,
        "action_id": action_id,
        "source": "fresh_native",
        "cache_validation": None,
        "source_checkpoint": None,
        "source_checkpoint_sha256": None,
        "input_state": state_payload(state, state_fingerprint),
        "output_state": None,
        "requested_goal": _clean(requested_goal),
        "feasible": False,
        "status": "exception",
        "reason": text,
        "route_id": None,
        "primary_route": None,
        "primary_route_kind": None,
        "attempted_routes": [],
        "failed_attempts": [],
        "transfer_time_s": None,
        "elapsed_output_s": None,
        "actual_tcp_path_length_m": None,
        "retreat_waypoints": None,
        "retreat_waypoint_count": None,
        "gate_results": {},
        "gate_reasons": [],
        "reconstruction_exceptions": [{"kind": "native_call_exception", "exception": text}],
        "native_actual": {},
        "trajectory_records": [],
        "verification_calls": int(getattr(adapter, "verification_calls", 0)),
        "execution_calls": int(getattr(adapter, "execution_calls", 0)),
        "wall_seconds": time.perf_counter() - started,
        "adapter_fingerprint": adapter._adapter_fingerprint(),
    }


def _requested_goal(adapter: Any, action_id: str) -> dict[str, Any]:
    capture = adapter.rig.captures[action_id]
    return {
        "action_id": action_id,
        "base": capture.base_name,
        "kind": capture.kind,
        "window": capture.window,
        "goal_local": list(map(float, capture.goal)),
        "dwell_s": float(capture.dwell_s),
        "goal_pose_measured": _clean(adapter.native.measured_goal_pose(adapter.measured_frame, capture.goal)),
    }


def _manifest(adapter: Any, actions: list[str], targets: list[str], direct_controls: bool, output: Path, *, status: str = "running") -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "row_type": "manifest",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_index": int(adapter.index),
        "mode": adapter.mode,
        "backend": adapter.backend,
        "actions": actions,
        "canonical_action_ids": list(adapter.actions),
        "canonical_probe_order": list(adapter.actions[:PROBE_COUNT]),
        "terminal_ids": list(TERMINALS),
        "targets": targets,
        "direct_controls": bool(direct_controls),
        "output_directory": str(output),
        "jsonl_path": str(output / "native_exit_matrix.jsonl"),
        "adapter_fingerprint": adapter._adapter_fingerprint(),
        "source_study": str(STUDY),
        "gate_contract": {
            "native_tracking": "native 2mm/2deg whole-segment gate",
            "aligned_tracking": "same-tick measured-before vs desired <= 2mm and 2deg",
            "workspace": "fixture WindowConstraint + unchanged constrained wall swept check",
            "contact": "synthetic true-surface contact gap <= 3mm",
            "paused": "task_paused_count == 0",
            "exit": "native exit state valid with q/qdot/acceleration/snapshot",
        },
        "cache_contract": "same fixture/mode adapter fingerprint + exact complete input fingerprint + valid full-state checkpoint; cache labelled separately from fresh_native",
        "subset_contract": "--actions selects canonical probes; --targets selects terminal targets; direct controls are independently selectable",
    }


def run(args: argparse.Namespace) -> int:
    # These are intentionally set by the runner too so accidental shell state
    # cannot silently introduce concurrent BLAS work in one native process.
    import os

    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"

    sys.path.insert(0, str(EXPERIMENT))
    from native_adapter import NativeAdapter, _state_fingerprint, _state_from_payload  # noqa: E402

    output = Path(args.output) if args.output else DEFAULT_ROOT / f"fixture_{args.fixture:02d}_{args.mode}"
    output.mkdir(parents=True, exist_ok=True)
    jsonl = output / "native_exit_matrix.jsonl"
    if args.repair_derived:
        return _repair_derived_rows(jsonl, args.fixture, args.mode)
    if args.finalize_inventory:
        return _finalize_inventory(jsonl, args.fixture, args.mode)
    if args.restart and jsonl.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        # Keep archives recoverable without allowing a naive ``*.jsonl``
        # consumer to mistake an interrupted prior run for current cells.
        archive = jsonl.with_name(f"native_exit_matrix.restart_{stamp}.jsonl.archived")
        jsonl.replace(archive)
    done, old_manifest = _read_rows(jsonl) if args.resume and not args.restart else ({}, None)

    adapter = NativeAdapter(args.fixture, args.mode, output, backend="native", reuse_checked=False)
    actions = _parse_actions(args.actions, adapter.actions)
    targets = _parse_targets(args.targets)
    direct_controls = not args.no_direct_controls and not args.roots_only
    manifest = _manifest(adapter, actions, targets, direct_controls, output)
    # A single native adapter/environment is kept for the whole serial run.
    # No threads/processes are created here.
    cache_index: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    current_fp = adapter._adapter_fingerprint()
    for action_id in list(actions) + list(TERMINALS):
        cache_index[action_id] = _cache_candidates(args.fixture, args.mode, current_fp, action_id)
    cache_counts = {action: len(rows) for action, rows in cache_index.items()}
    manifest["cache_candidate_counts"] = cache_counts
    manifest["resume_existing_cells"] = len(done)
    stream_mode = "a"
    with jsonl.open(stream_mode, encoding="utf-8") as stream:
        if not old_manifest or args.restart:
            _append(stream, manifest)
        elif old_manifest is not None:
            # Preserve the original frozen manifest; add a lightweight resume
            # marker so the stream remains auditable without duplicate headers.
            _append(stream, {
                "schema": SCHEMA,
                "row_type": "resume",
                "fixture_index": args.fixture,
                "mode": args.mode,
                "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
                "existing_cells": len(done),
            })

        def emit(row: dict[str, Any]) -> None:
            _append(stream, row)
            done[str(row["cell_key"])] = row

        def run_cell(stage: str, state: Any, action_id: str, root_action: str | None, target_action: str | None) -> Any:
            key = _cell_key(stage, root_action, target_action)
            prior = done.get(key)
            refresh_cache = bool(args.refresh_cache_rows and prior and prior.get("source") == "cache")
            if args.resume and key in done and not args.retry_errors and not refresh_cache:
                row = done[key]
                return None if not row.get("output_state") else _state_from_row(row["output_state"], _state_from_payload)
            started = time.perf_counter()
            requested = _requested_goal(adapter, action_id)
            state_fp = _state_fingerprint(state)
            source = "fresh_native"
            source_checkpoint = None
            source_payload = None
            event = None
            if not args.fresh_only:
                event, source_checkpoint, source_payload = _verified_cache(
                    adapter, state, action_id, _state_fingerprint,
                    _state_payload, cache_index.get(action_id, []))
                if event is not None:
                    source = "cache"
            if event is None:
                if args.cache_only:
                    row = _exception_row(adapter=adapter, state=state, action_id=action_id,
                                         stage=stage, root_action=root_action, target_action=target_action,
                                         state_payload=_state_payload, state_fingerprint=_state_fingerprint,
                                         requested_goal=requested, started=started,
                                         exc=RuntimeError("cache_only: no verified complete-state checkpoint"))
                    row["source"] = "cache"
                    row["reason"] = "cache_only_no_verified_checkpoint"
                    emit(row)
                    return None
                try:
                    event = adapter.checked(state, action_id)
                except Exception as exc:
                    emit(_exception_row(adapter=adapter, state=state, action_id=action_id,
                                        stage=stage, root_action=root_action, target_action=target_action,
                                        state_payload=_state_payload, state_fingerprint=_state_fingerprint,
                                        requested_goal=requested, started=started, exc=exc))
                    return None
            route_rows = _route_rows_for_event(adapter, state, action_id, event, source, source_checkpoint)
            row = _event_row(adapter=adapter, state=state, event=event,
                             state_fingerprint=_state_fingerprint, state_payload=_state_payload,
                             action_id=action_id, stage=stage, root_action=root_action,
                             target_action=target_action, source=source,
                             source_checkpoint=source_checkpoint, source_payload=source_payload,
                             started=started, route_rows=route_rows,
                             requested_goal=requested,
                             cache_validation={
                                 "adapter_fingerprint_match": True,
                                 "input_fingerprint_match": True,
                                 "complete_native_exit": bool(getattr(event.output, "native", None) is not None),
                             } if source == "cache" else None)
            emit(row)
            return getattr(event, "output", None)

        # 1) Direct root controls are separate controls, independent of probe
        # subset selection.  Their input is always the exact root state.
        root = adapter.start
        if direct_controls:
            for terminal in targets:
                run_cell("direct_control", root, terminal, None, terminal)

        # 2) Canonical root probe checks, then 3) same-target terminal checks
        # from each valid complete native exit.
        root_outputs: dict[str, Any] = {}
        for action_id in actions:
            root_output = run_cell("root_probe", root, action_id, action_id, None)
            if root_output is None:
                # Resume may have a failed root row; use the row itself to
                # distinguish no route from an exception without inventing an
                # absorbing state.
                existing = done.get(_cell_key("root_probe", action_id, None))
                if existing and existing.get("output_state") and existing.get("feasible"):
                    root_output = _state_from_payload(existing["output_state"])
            if root_output is not None and bool(getattr(root_output, "valid", False)):
                root_outputs[action_id] = root_output
                if not args.roots_only:
                    for terminal in targets:
                        run_cell("continuation", root_output, terminal, action_id, terminal)

        # Record expected cell inventory so unrun continuations remain visibly
        # absent/not-run rather than being mistaken for failures.
        expected = []
        if direct_controls:
            expected.extend(_cell_key("direct_control", None, terminal) for terminal in targets)
        expected.extend(_cell_key("root_probe", action, None) for action in actions)
        if not args.roots_only:
            expected.extend(_cell_key("continuation", action, terminal)
                            for action in actions for terminal in targets)
        _append(stream, {
            "schema": SCHEMA,
            "row_type": "summary",
            "fixture_index": args.fixture,
            "mode": args.mode,
            "expected_cell_keys": expected,
            "emitted_cell_keys": sorted(done),
            "root_probe_count": len(actions),
            "root_feasible_count": len(root_outputs),
            "continuation_expected_for_feasible_roots": 0 if args.roots_only else len(root_outputs) * len(targets),
            "direct_control_count": len(targets) if direct_controls else 0,
            "cache_rows": sum(1 for row in done.values() if row.get("source") == "cache"),
            "fresh_rows": sum(1 for row in done.values() if row.get("source") == "fresh_native"),
            "not_run_root_infeasible_or_unselected": len(set(expected) - set(done)),
            "no_checked_route_rows_are_finite_witnesses_only": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        })
    adapter.close()
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(jsonl),
        "fixture": args.fixture,
        "mode": args.mode,
        "actions": actions,
        "targets": targets,
        "completed_rows": sum(1 for row in done.values() if row.get("row_type") == "cell"),
        "cache_rows": sum(1 for row in done.values() if row.get("source") == "cache"),
        "fresh_rows": sum(1 for row in done.values() if row.get("source") == "fresh_native"),
    }, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=int, choices=(0, 1), required=True)
    parser.add_argument("--mode", choices=("constrained", "open"), required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--actions", default=None,
                        help="canonical probe ids, indexes, or half-open index slices (default: all 12)")
    parser.add_argument("--targets", default="left,right", help="terminal ids (default: left,right)")
    parser.add_argument("--no-direct-controls", action="store_true")
    parser.add_argument("--roots-only", action="store_true", help="emit only selected root probe cells (no direct controls or continuations)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restart", action="store_true", help="archive the current JSONL before running")
    parser.add_argument("--retry-errors", action="store_true", help="re-run cells whose prior row had status exception")
    parser.add_argument("--refresh-cache-rows", action="store_true", help="re-read cache-backed cells, preserving fresh rows")
    parser.add_argument("--fresh-only", action="store_true", help="ignore old same-fixture complete-state caches")
    parser.add_argument("--cache-only", action="store_true", help="emit cache misses as rows instead of native replay")
    parser.add_argument("--repair-derived", action="store_true", help="append corrected non-native derived diagnostics only")
    parser.add_argument("--finalize-inventory", action="store_true", help="append a full-matrix inventory after subset/resume runs")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))

