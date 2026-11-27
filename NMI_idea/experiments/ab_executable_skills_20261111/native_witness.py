#!/usr/bin/env python3
"""Validate one bounded native/QP continuation witness.

This is deliberately a one-case audit, not a rollout driver.  It reads the
already recorded case-0001 native checkpoints/traces, verifies that the
selected second event starts from the first event's complete exit state, and
then performs the same two segments once with the native QP replay backend.
The route set is evaluated from each exact measured exit with the public
``NativeAdapter.propose`` API; proposals are predictions and are never called
native certificates here.

The script writes only ``NATIVE_WITNESS.json`` beside itself.  It does not
touch the source study directory or create execution/planning outputs.
Use the experiment's Genesis environment, for example::

    /media/camp/EXT_DRIVE/envs/genesis/bin/python native_witness.py

The fresh replay is two segments (~6 s on the recorded host), well below the
90 s audit bound.  ``--no-fresh`` is available for a read-only trace/proposal
check when a native runtime is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parent.parent / "ab_continuing_rollout_20261111"
STUDY = EXPERIMENT / "study_v4_closed_loop_20261111"
CASE_ROOT = (
    STUDY
    / "B_rollout"
    / "fixture_00"
    / "cases_0000_0001"
    / "dev"
    / "fixture_00_constrained"
)
RESULT_PATH = CASE_ROOT / "B_rollout" / "case_0001" / "RESULT_GUARD_16.json"
PLANNING_ROOT = CASE_ROOT / "native_shared" / "planning"
CHECKPOINT_PATHS = (
    PLANNING_ROOT / "checkpoints" / "239763b4090bb8fcb7b196c2.json",
    PLANNING_ROOT / "checkpoints" / "3f24774a6736087b07962efd.json",
)
TRACE_NAMES = (
    "seg_2a41ea0b06e21c866352_0.npz",
    "seg_4c68c19c31f02bc6146a_0.npz",
)


def _json_clean(value: Any) -> Any:
    """Return strict JSON values for diagnostics and state hashes."""
    if isinstance(value, np.ndarray):
        return [_json_clean(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_clean(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _state_hash(payload: dict[str, Any]) -> str:
    """Match run_adaptive._state_hash for selected-event linkage."""
    state = {
        "key": list(payload["key"]),
        "elapsed": float(payload["elapsed_s"]),
        "pose": payload["pose"],
    }
    native = payload.get("native")
    if native is not None:
        state.update({
            "q": native["q"],
            "qdot": native["qdot"],
            "snapshot": native["snapshot"],
        })
    encoded = json.dumps(_json_clean(state), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _max_abs(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _duration_from_provenance(provenance: str) -> float:
    marker = "prodmp_duration_s="
    if marker not in str(provenance):
        raise ValueError(f"missing {marker} in recorded provenance")
    return float(str(provenance).split(marker, 1)[1].split(";", 1)[0])


def _check(checks: list[dict[str, Any]], failures: list[str], name: str,
           passed: bool, detail: Any = None) -> None:
    row: dict[str, Any] = {"name": name, "passed": bool(passed)}
    if detail is not None:
        row["detail"] = _json_clean(detail)
    checks.append(row)
    if not passed:
        failures.append(name)


def _route_summary(proposal: Any) -> dict[str, Any]:
    routes = []
    for route in proposal.routes:
        routes.append({
            "route_id": str(route.route_id),
            "waypoints": _json_clean(route.waypoints),
            "goal_pose": _json_clean(route.goal_pose),
            "travel_s": float(route.travel_s),
            "dwell_s": float(route.dwell_s),
            "duration_s": float(route.duration_s),
            "margin": float(route.margin),
            "geometric_status": str(route.geometric_status),
            "prediction_status": str(route.prediction_status),
            "native_checked": bool(route.native_checked),
        })
    return {
        "action_id": str(proposal.action_id),
        "available": bool(proposal.available),
        "status": str(proposal.status),
        "prediction_kind": str(proposal.prediction_kind),
        "native_checked": bool(proposal.native_checked),
        "state_fingerprint": str(proposal.state_fingerprint),
        "dird_query_increment": int(proposal.dird_query_increment),
        "best_route": None if proposal.best_route is None else str(proposal.best_route.route_id),
        "routes": routes,
    }


def _saved_route_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Reduce a saved proposal to fields compared with a fresh proposal."""
    routes = []
    for route in payload.get("routes", []):
        routes.append({
            "route_id": str(route["route_id"]),
            "travel_s": float(route["travel_s"]),
            "dwell_s": float(route["dwell_s"]),
            "duration_s": float(route["duration_s"]),
            "margin": float(route["margin"]),
            "geometric_status": str(route["geometric_status"]),
            "prediction_status": str(route["prediction_status"]),
            "native_checked": bool(route["native_checked"]),
        })
    return {
        "action_id": str(payload["action_id"]),
        "available": bool(payload["available"]),
        "status": str(payload["status"]),
        "state_fingerprint": str(payload["state_fingerprint"]),
        "dird_query_increment": int(payload["dird_query_increment"]),
        "best_route": None if payload.get("best_route") is None else str(payload["best_route"]["route_id"]),
        "routes": routes,
    }


def _proposal_equivalent(fresh: dict[str, Any], saved: dict[str, Any]) -> bool:
    if fresh["action_id"] != saved["action_id"]:
        return False
    for key in ("available", "status", "state_fingerprint", "best_route"):
        if fresh[key] != saved[key]:
            return False
    if fresh["dird_query_increment"] != saved["dird_query_increment"]:
        return False
    if len(fresh["routes"]) != len(saved["routes"]):
        return False
    for a, b in zip(fresh["routes"], saved["routes"]):
        for key in ("route_id", "geometric_status", "prediction_status", "native_checked"):
            if a[key] != b[key]:
                return False
        for key in ("travel_s", "dwell_s", "duration_s", "margin"):
            if not np.isclose(a[key], b[key], rtol=0.0, atol=1e-12):
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-fresh", action="store_true",
                        help="skip the two fresh native QP replays")
    parser.add_argument("--output", type=Path, default=HERE.parent / "NATIVE_WITNESS.json",
                        help="output manifest (default: NATIVE_WITNESS.json beside this script)")
    args = parser.parse_args()

    if not RESULT_PATH.exists():
        raise FileNotFoundError(RESULT_PATH)
    for path in CHECKPOINT_PATHS + tuple(PLANNING_ROOT / name for name in TRACE_NAMES):
        if not path.exists():
            raise FileNotFoundError(path)

    # The experiment is intentionally imported only after the source paths are
    # resolved; the caller should use envs/genesis/bin/python.
    sys.path.insert(0, str(EXPERIMENT))
    from fixture import local_pose  # noqa: E402
    from native_adapter import (  # noqa: E402
        NativeAdapter,
        _state_fingerprint,
        _state_from_payload,
    )
    from robot_motion import MotionState, NativeMotionEnvironment  # noqa: E402

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    checkpoints = [json.loads(path.read_text(encoding="utf-8")) for path in CHECKPOINT_PATHS]
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    adapter = NativeAdapter(0, "constrained", None, backend="native")
    current_adapter_fp = adapter._adapter_fingerprint()
    _check(checks, failures, "result_case_identity",
           result.get("arm") == "B_rollout" and result.get("fixture_index") == 0
           and result.get("case_id") == 1 and result.get("mode") == "constrained")

    states = []
    for i, (checkpoint, cp_path) in enumerate(zip(checkpoints, CHECKPOINT_PATHS)):
        _check(checks, failures, f"checkpoint_{i}_schema",
               checkpoint.get("schema") == "adaptive_native_checkpoint_v1")
        _check(checks, failures, f"checkpoint_{i}_adapter_fingerprint",
               checkpoint.get("adapter_fingerprint", {}).get("sha256")
               == current_adapter_fp.get("sha256"),
               {"recorded": checkpoint.get("adapter_fingerprint", {}).get("sha256"),
                "current": current_adapter_fp.get("sha256")})
        input_state = _state_from_payload(checkpoint["input_state"])
        output_state = _state_from_payload(checkpoint["output_state"])
        states.append((input_state, output_state))
        _check(checks, failures, f"checkpoint_{i}_input_fingerprint",
               checkpoint.get("input_fingerprint") == _state_fingerprint(input_state))
        native = checkpoint["output_state"].get("native")
        native_ok = isinstance(native, dict) and all(
            np.asarray(native.get(key, []), dtype=float).shape == (8,)
            and np.isfinite(np.asarray(native.get(key, []), dtype=float)).all()
            for key in ("q", "qdot", "acceleration")
        ) and isinstance(native.get("snapshot"), dict)
        _check(checks, failures, f"checkpoint_{i}_complete_native_exit", native_ok,
               {"q_shape": None if native is None else np.asarray(native.get("q", [])).shape,
                "snapshot_schema": None if native is None else native.get("snapshot", {}).get("schema")})
        _check(checks, failures, f"checkpoint_{i}_route_valid",
               bool(checkpoint.get("event", {}).get("feasible"))
               and bool(checkpoint.get("event", {}).get("result", {}).get("full_state_checkpoint")))
        _check(checks, failures, f"checkpoint_{i}_source_exists", cp_path.exists(), str(cp_path))

    # This is the critical matched differential exit: the second event's
    # complete input is the first event's complete native output.
    first_out = checkpoints[0]["output_state"]
    second_in = checkpoints[1]["input_state"]
    continuity = {
        "key_equal": first_out["key"] == second_in["key"],
        "elapsed_diff_s": abs(float(first_out["elapsed_s"]) - float(second_in["elapsed_s"])),
        "pose_max_abs_diff": _max_abs(first_out["pose"], second_in["pose"]),
        "q_max_abs_diff": _max_abs(first_out["native"]["q"], second_in["native"]["q"]),
        "qdot_max_abs_diff": _max_abs(first_out["native"]["qdot"], second_in["native"]["qdot"]),
        "acceleration_max_abs_diff": _max_abs(first_out["native"]["acceleration"], second_in["native"]["acceleration"]),
        "snapshot_equal": first_out["native"]["snapshot"] == second_in["native"]["snapshot"],
    }
    _check(checks, failures, "exact_exit_to_next_input_continuity",
           continuity["key_equal"] and continuity["elapsed_diff_s"] == 0.0
           and continuity["pose_max_abs_diff"] == 0.0
           and continuity["q_max_abs_diff"] == 0.0
           and continuity["qdot_max_abs_diff"] == 0.0
           and continuity["acceleration_max_abs_diff"] == 0.0
           and continuity["snapshot_equal"], continuity)

    transitions: list[dict[str, Any]] = []
    for i, (checkpoint, trace_name) in enumerate(zip(checkpoints, TRACE_NAMES)):
        state_in_payload = checkpoint["input_state"]
        state_out_payload = checkpoint["output_state"]
        event = result["events"][i]
        private = result["privateeval"][i]
        recorded_event = checkpoint["event"]
        trace_path = PLANNING_ROOT / trace_name
        trace = np.load(trace_path, allow_pickle=False)
        required = {"time_s", "q", "qdot", "actual", "desired", "measured_before", "local_xyz_m", "self_clearance_m"}
        _check(checks, failures, f"transition_{i}_trace_fields", required.issubset(trace.files), trace.files)
        q_diff = _max_abs(trace["q"][-1], state_out_payload["native"]["q"])
        qdot_diff = _max_abs(trace["qdot"][-1], state_out_payload["native"]["qdot"])
        pose_diff = _max_abs(trace["actual"][-1], state_out_payload["pose"])
        _check(checks, failures, f"transition_{i}_trace_exit_matches_checkpoint",
               q_diff == 0.0 and qdot_diff == 0.0 and pose_diff == 0.0,
               {"q_max_abs_diff": q_diff, "qdot_max_abs_diff": qdot_diff,
                "pose_max_abs_diff": pose_diff})
        action_pose = local_pose(adapter.measured_frame, np.asarray(state_out_payload["pose"], dtype=float))
        public_action = event["action"]
        contact_diff = _max_abs(action_pose, np.asarray(public_action["xy_m"] + [public_action["yaw_rad"]]))
        _check(checks, failures, f"transition_{i}_measured_contact_link",
               contact_diff == 0.0, {"max_abs_diff": contact_diff})
        linkage = {
            "action_id_equal": event.get("action_id") == checkpoint.get("action_id"),
            "route_id_equal": event.get("native_actual", {}).get("route_id") == recorded_event.get("route_id"),
            "trace_record_equal": event.get("native_actual", {}).get("pathref") == recorded_event.get("trajectory_records"),
            "entry_state_hash_equal": private.get("native_entry_state_hash") == _state_hash(state_in_payload),
            "exit_state_hash_equal": private.get("native_exit_state_hash") == _state_hash(state_out_payload),
        }
        _check(checks, failures, f"transition_{i}_selected_event_linkage", all(linkage.values()), linkage)
        input_native = state_in_payload.get("native") or {}
        transitions.append({
            "index": i,
            "action_id": checkpoint["action_id"],
            "route_id": recorded_event["route_id"],
            "source_checkpoint": str(CHECKPOINT_PATHS[i]),
            "source_trace": str(trace_path),
            "trace_samples": int(trace["time_s"].shape[0]),
            "trace_final_time_s": float(trace["time_s"][-1]),
            "elapsed_s": float(state_out_payload["elapsed_s"]),
            "duration_s": float(recorded_event["duration_s"]),
            "dird_margin": float(recorded_event["margin"]),
            "reference_kind": recorded_event["actual_output"]["reference_kind"],
            "same_tick_position_error_max_m": recorded_event["actual_output"]["same_tick_position_error_max_m"],
            "same_tick_rotation_error_max_rad": recorded_event["actual_output"]["same_tick_rotation_error_max_rad"],
            "jerk_margin_min": recorded_event["actual_output"]["jerk_margin_min"],
            "input_state": {
                "key": state_in_payload["key"],
                "elapsed_s": state_in_payload["elapsed_s"],
                "pose": state_in_payload["pose"],
                "q": input_native.get("q"),
                "qdot": input_native.get("qdot"),
                "acceleration": input_native.get("acceleration"),
                "snapshot_schema": input_native.get("snapshot", {}).get("schema"),
            },
            "exit_state": {
                "key": state_out_payload["key"],
                "elapsed_s": state_out_payload["elapsed_s"],
                "pose": state_out_payload["pose"],
                "q": state_out_payload["native"]["q"],
                "qdot": state_out_payload["native"]["qdot"],
                "acceleration": state_out_payload["native"]["acceleration"],
                "snapshot_schema": state_out_payload["native"]["snapshot"]["schema"],
            },
            "linkage": linkage,
        })

    # The saved proposal files are an independent read-only record of the
    # exact-output route set.  Recompute them through the current public
    # adapter as well, while explicitly retaining prediction-only status.
    route_sets: list[dict[str, Any]] = []
    for i, (_input_state, output_state) in enumerate(states):
        state_fp = _state_fingerprint(output_state)
        actions: dict[str, Any] = {}
        saved_matches = 0
        for action_id in adapter.actions:
            proposal = adapter.propose(output_state, action_id)
            fresh = _route_summary(proposal)
            digest = hashlib.sha256(f"{state_fp}:{action_id}".encode("utf-8")).hexdigest()[:24]
            saved_path = PLANNING_ROOT / "proposals" / f"{digest}.json"
            saved = json.loads(saved_path.read_text(encoding="utf-8")) if saved_path.exists() else None
            equivalent = saved is not None and _proposal_equivalent(fresh, _saved_route_summary(saved))
            _check(checks, failures, f"exit_{i}_proposal_{action_id}_saved_match", equivalent,
                   {"saved": str(saved_path), "exists": saved is not None,
                    "fresh_routes": [r["route_id"] for r in fresh["routes"]]})
            if equivalent:
                saved_matches += 1
            actions[action_id] = {
                **fresh,
                "saved_proposal": str(saved_path),
                "saved_match": equivalent,
            }
        route_sets.append({
            "exit_index": i,
            "state_fingerprint": state_fp,
            "route_set_kind": "public_measured_frame_geometry_plus_baked_dird_prediction",
            "native_qp_checked": False,
            "prediction_status_contract": "predicted_unknown; checked/execute is required for a native certificate",
            "actions": actions,
            "saved_proposal_match_count": saved_matches,
            "action_count": len(actions),
            "available_action_count": sum(int(bool(row["available"])) for row in actions.values()),
        })

    fresh_replay: dict[str, Any] = {
        "requested": not args.no_fresh,
        "performed": False,
        "backend": "native",
        "segments": [],
    }
    if not args.no_fresh:
        replay_started = time.perf_counter()
        with NativeMotionEnvironment(backend="native") as native:
            previous_state: MotionState | None = None
            for i, checkpoint in enumerate(checkpoints):
                snapshot = checkpoint["output_state"]["native"]["snapshot"]
                event_index = 0 if i == 0 else 1
                recorded = snapshot["events"][event_index]
                path = np.asarray(recorded["T_tcp_railbase"], dtype=float)
                segment_started = time.perf_counter()
                replay = native.execute(
                    path,
                    state=previous_state,
                    speed_m_s=float(recorded["speed_m_s"]),
                    settle_s=float(recorded["settle_s"]),
                    prodmp_duration_s=_duration_from_provenance(recorded["provenance"]),
                    label=f"native_witness_{i}",
                    return_trace=False,
                )
                exit_state = replay.get("exit_state")
                target_native = checkpoint["output_state"]["native"]
                diffs = None if exit_state is None else {
                    "q_max_abs_diff": _max_abs(exit_state["q_exit"], target_native["q"]),
                    "qdot_max_abs_diff": _max_abs(exit_state["qdot_exit"], target_native["qdot"]),
                    "acceleration_max_abs_diff": _max_abs(exit_state["acceleration_exit"], target_native["acceleration"]),
                }
                passed = bool(replay.get("whole_segment_pass")) and diffs is not None and all(
                    value == 0.0 for value in diffs.values()
                )
                _check(checks, failures, f"fresh_native_segment_{i}", passed,
                       {"whole_segment_pass": replay.get("whole_segment_pass"),
                        "exit_state_valid": replay.get("exit_state_valid"),
                        "failures": replay.get("failures"), "diffs": diffs})
                fresh_replay["segments"].append({
                    "index": i,
                    "wall_seconds": time.perf_counter() - segment_started,
                    "whole_segment_pass": bool(replay.get("whole_segment_pass")),
                    "exit_state_valid": bool(replay.get("exit_state_valid")),
                    "failures": replay.get("failures", []),
                    "integrated_duration_s": replay.get("integrated_duration_s"),
                    "simulated_duration_s": replay.get("simulated_duration_s"),
                    "entry_q_matches_recorded": None if previous_state is None else _max_abs(replay.get("entry_q"), previous_state.q) == 0.0,
                    "entry_qdot_matches_recorded": None if previous_state is None else _max_abs(replay.get("entry_qdot"), previous_state.qdot) == 0.0,
                    "diffs_to_recorded_exit": diffs,
                })
                if exit_state is None:
                    break
                previous_state = MotionState(
                    np.asarray(exit_state["q_exit"], dtype=float),
                    np.asarray(exit_state["qdot_exit"], dtype=float),
                    np.asarray(exit_state["acceleration_exit"], dtype=float),
                    exit_state,
                )
        fresh_replay["performed"] = True
        fresh_replay["total_wall_seconds"] = time.perf_counter() - replay_started

    adapter.close()

    manifest = {
        "schema": "native_qp_replay_witness_v1",
        "status": "passed" if not failures else "failed",
        "scope": "one recorded fixture/case; two selected native transitions; no multi-case rollout",
        "read_only_existing_trace": True,
        "fresh_native_replay": fresh_replay,
        "source_result": str(RESULT_PATH),
        "transitions": transitions,
        "continuity": continuity,
        "downstream_route_feasible_sets": route_sets,
        "provenance": {
            "python": sys.executable,
            "backend": "native",
            "native_model_clock": current_adapter_fp.get("controller", {}).get("native_model_clock"),
            "tcp_mode": current_adapter_fp.get("controller", {}).get("tcp_mode"),
            "adapter_fingerprint": current_adapter_fp,
            "complete_exit_contract": "q, qdot, acceleration, snapshot, pose; snapshot restores all prior path/hold events",
            "route_set_contract": "public measured-frame geometry and baked dIRD proposal only; native_checked=false",
            "live_robot_connection": False,
            "external_network": False,
        },
        "checks": {
            "count": len(checks),
            "passed": sum(int(row["passed"]) for row in checks),
            "failed": len(failures),
            "failures": failures,
            "rows": checks,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_clean(manifest), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "output": str(args.output),
        "checks": manifest["checks"]["passed"],
        "failed": manifest["checks"]["failed"],
        "fresh_native_wall_seconds": fresh_replay.get("total_wall_seconds"),
        "route_actions_per_exit": [row["action_count"] for row in route_sets],
    }, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

