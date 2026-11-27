"""Run the no-deadline adaptive-stopping policies with NativeAdapter.

The planner only sees checked native events and public packets.  This runner
owns synthetic acoustic rendering and writes private truth under a separate
``privateeval`` field; that field is never passed into :class:`AdaptivePlanner`.

Examples (all use the experiment's native Python):

    python run_adaptive.py --stage smoke --count 1 --index 0 \
        --arm B_rollout --output /tmp/adaptive-smoke
    python run_adaptive.py --stage dev --count 1 --index 0 \
        --arm B_rollout --output /tmp/adaptive-dev
    python run_adaptive.py --stage test --count 128 --index 0 \
        --freeze-file /path/FREEZE_00_constrained.json \
        --model-artifact frozen_model.json --output /tmp/adaptive-test

``max_probe_frames`` is the administrative fresh-acquisition watchdog, not a
task deadline.  Time is always logged and no decision compares it with a
deadline; terminal rejects and probes both consume this guard.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import math
import pickle
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
# NativeAdapter imports same-named modules (fixture, robot_motion, acoustics).
# Put this directory first and evict a stale module imported by an older
# experiment in the same Python process.  The manifest below asserts all
# implementation paths, making accidental source shadowing loud.
sys.path.insert(0, str(HERE))
SOURCE_MODULES = ("acoustics", "fixture", "robot_motion", "belief_model",
                  "native_adapter", "adaptive_planning", "rollout_policy", "train_calibrate")
for _name in SOURCE_MODULES:
    _module = sys.modules.get(_name)
    if _module is not None and getattr(_module, "__file__", None):
        try:
            if Path(_module.__file__).resolve().parent != HERE:
                del sys.modules[_name]
        except OSError:
            del sys.modules[_name]

from acoustics import PublicObservation, paired_scenarios
from adaptive_planning import (AdaptivePlanner, NOISE_SEED, PlannerConfig, TEST_SEED,
                               CALIBRATION_SEED, TRAIN_SEED, load_model_params,
                               planning_decision_from_public_dict,
                               planning_decision_to_public_dict,
                               public_acceptance, public_action_from_event)
from fixture import SINGLE_DWELLS, local_pose, single_dwell_action
from native_adapter import NativeAdapter
from train_calibrate import render_sensor_v2


ARMS = ("A", "A_task", "A_info", "A_full", "B_H1", "B_H2", "B_rollout", "pair_fixed")
DEV_ARMS = ("A_task", "A_info", "B_rollout", "pair_fixed")
SEEDS = {"smoke": CALIBRATION_SEED, "dev": CALIBRATION_SEED, "test": TEST_SEED}


def _arm_output_name(arm: str, pair_dwell: float) -> str:
    """Keep the three pair dwell configs in disjoint result directories."""
    if arm == "pair_fixed":
        return f"pair_fixed_dwell_{int(round(float(pair_dwell) * 100)):02d}"
    return str(arm)


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest() -> dict[str, Any]:
    modules = {}
    for name in SOURCE_MODULES:
        module = importlib.import_module(name)
        source = Path(module.__file__).resolve()
        if source.parent != HERE:
            raise RuntimeError(f"module {name} shadowed by {source}; expected {HERE}")
        modules[name] = {"path": str(source), "sha256": _sha(source)}
    return modules


def _initial_cache_key(*, index: int, mode: str, arm: str, pair_dwell: float,
                       max_probe_frames: int, pair_order: tuple[str, str],
                       model_params: Any) -> str:
    payload = {
        "index": int(index), "mode": str(mode), "arm": str(arm),
        "pair_dwell": float(pair_dwell), "max_probe_frames": int(max_probe_frames),
        "pair_order": list(pair_order),
        "model": _clean(model_params.to_dict()),
        "source": _source_manifest(),
        "runner_sha256": _sha(Path(__file__)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _event_id(case_id: int, mode: str, state: Any, event: Any) -> str:
    """Common-random public event id for identical actual prefixes."""
    prefix = "/".join(str(x) for x in tuple(state.key) + (event.action_id,))
    visit = tuple(state.key).count(event.action_id)
    return f"case_{case_id:04d}/{mode}/{prefix}/visit_{visit}"


class PrivateAcquisition:
    """Synthetic renderer/evaluator, deliberately outside the planner API."""

    def __init__(self, adapter: NativeAdapter, scenario: Any, *, case_id: int, noise_seed: int = NOISE_SEED):
        self.adapter = adapter
        self.scenario = scenario
        self.case_id = int(case_id)
        self.noise_seed = int(noise_seed)
        self.present = True

    def capture(self, event: Any, state_before: Any):
        event_id = _event_id(self.case_id, self.adapter.mode, state_before, event)
        public_action = public_action_from_event(event, self.adapter.measured_frame)
        physical_action = public_action_from_event(event, self.adapter.physical_frame)
        public, truth = render_sensor_v2(
            self.scenario, physical_action, event_id=event_id,
            elapsed_s=float(event.elapsed_s), seed=self.noise_seed,
            present=self.present,
        )
        # The hidden Markov mode is evaluator-only.  It advances exactly once
        # per actual acquisition and never enters the planner call below.
        self.present = bool(truth.present_after)
        public_accepted = public_acceptance(public)
        # Only this returned tuple crosses into AdaptivePlanner; private truth
        # is kept in a disjoint log object and never used for a decision.
        private = {
            "event_id": event_id,
            "correct": bool(truth.correct_complete),
            "false": bool(public_accepted and not truth.correct_complete),
            "ever_qualified": bool(truth.geometry_gate and truth.quality_gate and truth.target_association),
            "target_identity_correct": bool(truth.target_association),
            "quality_pass": bool(truth.quality_gate),
            "center_error_m": float(truth.true_center_offset_m),
            "axis_error_rad": float(truth.true_axis_angle_rad),
            "hidden_kind": "target" if truth.target_association else "clutter" if truth.clutter else "missing",
            "target_association": bool(truth.target_association),
            "present_before": bool(truth.present_before),
            "present_after": bool(truth.present_after),
            "public_accepted": public_accepted,
            "fresh_image": True,
            "elapsed_s": float(event.elapsed_s),
            "native_route_id": event.route_id,
            "native_entry_state_hash": _state_hash(state_before),
            "native_exit_state_hash": _state_hash(event.output),
        }
        return event_id, public_action, public, private


def _state_hash(state: Any) -> str:
    payload = {"key": list(state.key), "elapsed": float(state.elapsed),
               "pose": np.asarray(state.pose, dtype=float).tolist()}
    if state.native is not None:
        payload["q"] = np.asarray(state.native.q, dtype=float).tolist()
        payload["qdot"] = np.asarray(state.native.qdot, dtype=float).tolist()
        payload["snapshot"] = state.native.snapshot
    return hashlib.sha256(json.dumps(_clean(payload), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def select_fixed_pair(adapter: NativeAdapter, dwell: float) -> tuple[str, str]:
    """Select minimum public-native worst-case pair+terminal cost.

    The cost includes both fork members and the slower same-window two-frame
    terminal option from the second fork exit.  It is public engineering
    information and is computed once per shared adapter before acoustic draws.
    """
    candidates = (("fork_minus", "fork_plus"), ("fork_plus", "fork_minus"))
    rows = []
    for first, second in candidates:
        first_id = single_dwell_action(first, dwell)
        second_id = single_dwell_action(second, dwell)
        first_event = adapter.checked(adapter.start, first_id)
        if not first_event.feasible:
            rows.append((float("inf"), first, second, float("inf")))
            continue
        second_event = adapter.checked(first_event.output, second_id)
        if not second_event.feasible:
            rows.append((float("inf"), first, second, float("inf")))
            continue
        pair_time = first_event.duration_s + second_event.duration_s
        terminal_costs = []
        for terminal_id in ("left", "right"):
            first_terminal = adapter.checked(second_event.output, terminal_id)
            if not first_terminal.feasible:
                continue
            retry = adapter.checked(first_terminal.output, terminal_id)
            terminal_costs.append(first_terminal.duration_s + (retry.duration_s if retry.feasible else 0.0))
        worst_terminal = max(terminal_costs, default=float("inf"))
        rows.append((pair_time + worst_terminal, first, second, worst_terminal))
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    select_fixed_pair.last_report = [
        {"order": [row[1], row[2]], "worstcase_total_s": row[0],
         "terminal_option_worstcase_s": row[3]}
        for row in rows
    ]
    if not rows or not math.isfinite(rows[0][0]):
        # The fixed baseline remains explicit even if this public layout has no
        # complete witness; the planner will record a native invalid outcome.
        return candidates[0]
    return rows[0][1], rows[0][2]


select_fixed_pair.last_report = []


def _result_rows(root: Path, *, stage: str, index: int, mode: str) -> list[dict[str, Any]]:
    base = root / stage / f"fixture_{index:02d}_{mode}"
    rows = []
    if not base.exists():
        return rows
    for path in sorted(base.glob("*/case_*/RESULT.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def _row_risk_eligible(row: Mapping[str, Any]) -> bool:
    """Check the common .90 gate without using private completion truth."""
    protocol = row.get("protocol", {})
    try:
        if abs(float(protocol.get("risk_gate", .90)) - .90) > 1.0e-9:
            return False
    except (TypeError, ValueError):
        return False
    for decision in row.get("decisions", ()):
        if decision.get("decision") != "commit":
            continue
        diagnostics = decision.get("diagnostics", {})
        options = diagnostics.get("terminal_options", ())
        selected = decision.get("score", 0.0)
        try:
            selected_score = float(selected)
        except (TypeError, ValueError):
            return False
        if not options or selected_score < .90 - 1.0e-9:
            return False
    return True


def write_dev_comparison(root: Path, *, index: int, mode: str) -> dict[str, Any]:
    """Summarise available dev rows without peeking at test outcomes."""
    rows = _result_rows(root, stage="dev", index=index, mode=mode)
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("arm")), float(row.get("pair_dwell", .30))), []).append(row)
    summaries = []
    for (arm, dwell), items in sorted(groups.items()):
        summaries.append({
            "arm": arm, "pair_dwell": dwell, "n": len(items),
            "case_ids": sorted(int(item.get("case_id", -1)) for item in items),
            "correct_rate": float(np.mean([bool(item.get("correct_complete", False)) for item in items])),
            "false_rate": float(np.mean([bool(item.get("false_complete", False)) for item in items])),
            "abstain_rate": float(np.mean([bool(item.get("abstain", False)) for item in items])),
            "watchdog_rate": float(np.mean([bool(item.get("watchdog", False)) for item in items])),
            "mean_probe_count": float(np.mean([float(item.get("probe_count", 0)) for item in items])),
            "risk_eligible_rate": float(np.mean([_row_risk_eligible(item) for item in items])),
        })
    case_sets = [tuple(summary["case_ids"]) for summary in summaries]
    source_manifests = [item.get("source_manifest") for item in rows]
    source_manifest_consistent = bool(source_manifests) and all(
        manifest == source_manifests[0] and manifest is not None
        for manifest in source_manifests)
    payload = {"schema": "adaptive_dev_comparison_v1", "stage": "dev",
               "fixture_index": int(index), "mode": mode, "rows": summaries,
               "coverage_case_ids": list(case_sets[0]) if case_sets and all(
                   current == case_sets[0] for current in case_sets) else [],
               "coverage_complete": bool(case_sets) and all(current == case_sets[0] for current in case_sets),
               "source_manifest": source_manifests[0] if source_manifest_consistent else None,
               "source_manifest_consistent": source_manifest_consistent,
               "selection_rule": "risk_eligible_then_correct_rate; then lower probe count; ties deterministic",
               "test_outcomes_used": False}
    path = root / f"DEV_COMPARISON_{index:02d}_{mode}.json"
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def write_freeze(root: Path, *, index: int, mode: str, model_artifact: Path | None,
                 comparison: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the best observed dev arm/config for an explicit handoff."""
    rows = list(comparison.get("rows", ()))
    if not rows:
        raise RuntimeError("cannot freeze without dev RESULT.json rows")
    if not bool(comparison.get("coverage_complete", False)):
        raise RuntimeError("freeze requires identical completed case IDs for every dev arm/config")
    if not bool(comparison.get("source_manifest_consistent", False)):
        raise RuntimeError("freeze requires a common local source manifest across dev rows")
    # A and B are selected separately so the test manifest can report both
    # development candidates.
    a_rows = [row for row in rows if row["arm"] in {"A_task", "A_info", "A_full", "A"}]
    b_rows = [row for row in rows if row["arm"] == "B_rollout"]
    pair_rows = [row for row in rows if row["arm"] == "pair_fixed"]
    if any(float(row.get("risk_eligible_rate", 0.0)) < 1.0 for row in rows):
        raise RuntimeError("freeze requires every dev arm/config row to obey the common risk gate")
    key = lambda row: (-row["correct_rate"], row["mean_probe_count"], row["arm"])
    selected_a = min(a_rows, key=key) if a_rows else None
    selected_b = min(b_rows, key=key) if b_rows else None
    selected_pair = min(pair_rows, key=key) if pair_rows else None
    if selected_a is None or selected_b is None or selected_pair is None:
        raise RuntimeError("freeze requires completed A, B, and pair development rows")
    payload = {
        "schema": "adaptive_stopping_freeze_v1", "stage": "dev",
        "fixture_index": int(index), "mode": mode,
        "selected_a": None if selected_a is None else selected_a["arm"],
        "selected_b": selected_b["arm"],
        "selected_pair_dwell": None if selected_pair is None else selected_pair["pair_dwell"],
        "risk_gate": .90, "watchdog_max_probe_frames": 8,
        "model_artifact": None if model_artifact is None else str(model_artifact),
        "model_artifact_sha256": None if model_artifact is None else _sha(model_artifact),
        "development_comparison": str(root / f"DEV_COMPARISON_{index:02d}_{mode}.json"),
        "coverage_case_ids": comparison.get("coverage_case_ids", []),
        "coverage_complete": True,
        "source_manifest": comparison.get("source_manifest"),
        "risk_eligible": True,
        "selection_rows": {"a": selected_a, "b": selected_b, "pair": selected_pair},
        "test_must_use_this_freeze": True,
    }
    path = root / f"FREEZE_{index:02d}_{mode}.json"
    path.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def run_one(*, stage: str, case_id: int, index: int, mode: str, arm: str, output: Path,
            model_params: Any, pair_dwell: float = 0.30,
            max_probe_frames: int = 8, backend: str = "native",
            adapter: NativeAdapter | None = None,
            pair_order: tuple[str, str] | None = None,
            initial_decision: Any | None = None,
            resume_payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run one acoustic case on a shared mechanical adapter/context."""
    case_seed = int(SEEDS[stage])
    scenario = paired_scenarios(case_seed, max(case_id + 1, 1))[case_id]
    arm_dir = (output / stage / f"fixture_{index:02d}_{mode}"
               / _arm_output_name(arm, pair_dwell) / f"case_{case_id:04d}")
    arm_dir.mkdir(parents=True, exist_ok=True)
    owned_adapter = adapter is None
    if owned_adapter:
        adapter = NativeAdapter(index, mode,
                                output / stage / f"fixture_{index:02d}_{mode}" / "native_shared",
                                backend=backend, reuse_checked=True)
    assert adapter is not None
    if pair_order is None:
        pair_order = (select_fixed_pair(adapter, pair_dwell) if arm == "pair_fixed"
                      else ("fork_minus", "fork_plus"))
    config = PlannerConfig(arm=arm, pair_dwell=pair_dwell,
                           pair_order=pair_order, max_probe_frames=max_probe_frames,
                           protocol_revision="DESIGN_PROTOCOL_loaded")
    t0 = time.perf_counter()
    planner = AdaptivePlanner(adapter, adapter.measured_frame,
                              config=config, model_params=model_params,
                              initial_decision=None if resume_payload is not None else initial_decision)
    acquisition = PrivateAcquisition(adapter, scenario, case_id=case_id)
    if resume_payload is not None:
        planner.restore_continuation(resume_payload)
        if resume_payload.get("private_present") is not None:
            acquisition.present = bool(resume_payload["private_present"])
    result = planner.run(acquisition.capture, private_prefix=(
        () if resume_payload is None else resume_payload.get("privateeval", ())))
    prefix_wall = 0.0 if resume_payload is None else float(resume_payload.get("planningwall", 0.0))
    result["planningwall"] = prefix_wall + float(time.perf_counter() - t0)
    result["case_id"] = int(case_id)
    result["fixture_index"] = int(index)
    result["mode"] = mode
    result["arm"] = arm
    result["scenario_seed"] = int(scenario.seed)
    result["noise_seed"] = NOISE_SEED
    result["fixed_pair_order"] = list(pair_order)
    result["pair_dwell"] = float(pair_dwell)
    result["config_tag"] = _arm_output_name(arm, pair_dwell)
    result["initial_decision_cached"] = bool(initial_decision is not None)
    result["continuation_from"] = None if resume_payload is None else "guard_checkpoint"
    result["source_manifest"] = _source_manifest()
    result["adapter_manifest"] = {
        "address": _clean(adapter.address),
        "measured_frame": _clean(adapter.measured_frame),
        "reuse_checked": True, "backend": backend,
        "native_counts_scope": "shared_adapter_cumulative",
        "verification_calls": adapter.verification_calls,
        "execution_calls": adapter.execution_calls,
        "verification_count": adapter.verification_count,
        "execution_count": adapter.execution_count,
        "proposal_calls": getattr(adapter, "proposal_calls", 0),
        "proposal_cache_hits": getattr(adapter, "proposal_cache_hits", 0),
        "no_deadline": True,
        "future_mechanical_feasibility": "unknown_until_selected_native_check",
    }
    result_path = arm_dir / "RESULT.json"
    if resume_payload is not None:
        result_path = arm_dir / "RESULT_CONTINUED.json"
    if result.get("watchdog"):
        checkpoint_name = ("CONTINUATION.pkl" if resume_payload is None
                           else f"CONTINUATION_{int(result.get('fresh_count', 0)):02d}.pkl")
        checkpoint_path = arm_dir / checkpoint_name
        checkpoint = planner.continuation_payload(private_present=acquisition.present)
        checkpoint.update({"case_id": case_id, "index": index, "mode": mode,
                           "arm": arm, "stage": stage, "pair_dwell": pair_dwell,
                           "planningwall": result["planningwall"],
                           "source_manifest": result["source_manifest"],
                           "privateeval": result["privateeval"],
                           "created_from": str(result_path)})
        temp_checkpoint = checkpoint_path.with_suffix(".pkl.tmp")
        with temp_checkpoint.open("wb") as handle:
            pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temp_checkpoint.replace(checkpoint_path)
        result["continuation_checkpoint"] = str(checkpoint_path)
    temp_result = result_path.with_suffix(".json.tmp")
    temp_result.write_text(json.dumps(_clean(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp_result.replace(result_path)
    if owned_adapter:
        adapter.close()
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "dev", "test"), required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--arm", choices=(*ARMS, "all"), default=None)
    parser.add_argument("--mode", choices=("constrained", "open"), default="constrained")
    parser.add_argument("--model-artifact", type=Path, default=None)
    parser.add_argument("--freeze-file", type=Path, default=None,
                        help="required frozen dev selection for test")
    parser.add_argument("--continue-checkpoint", type=Path, default=None,
                        help="resume an unresolved checkpoint; use the next 8/16/32 fresh guard")
    parser.add_argument("--freeze", action="store_true",
                        help="write FREEZE_<index>_<mode>.json after dev comparison")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pair-dwell", type=float, choices=SINGLE_DWELLS, default=None)
    parser.add_argument("--max-probe-frames", type=int, default=8,
                        help="fresh-acquisition watchdog only; no task deadline")
    parser.add_argument("--backend", choices=("native", "python"), default="native")
    args = parser.parse_args(argv)
    if args.count < 1 or args.case_start < 0:
        parser.error("count must be positive and case-start non-negative")
    if args.stage == "test" and args.freeze_file is None:
        parser.error("test stage requires --freeze-file; test never auto-selects a policy")
    if args.freeze and args.stage != "dev":
        parser.error("--freeze is valid only for --stage dev")
    if args.continue_checkpoint is not None and not args.continue_checkpoint.exists():
        parser.error("--continue-checkpoint does not exist")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    resume_payload = None
    if args.continue_checkpoint is not None:
        with args.continue_checkpoint.open("rb") as handle:
            resume_payload = pickle.load(handle)
        if args.stage != resume_payload["stage"]:
            raise ValueError("continuation stage differs from checkpoint")
        if resume_payload.get("source_manifest") not in (None, _source_manifest()):
            raise ValueError("continuation source differs from checkpoint")
        args.index = int(resume_payload["index"])
        args.mode = str(resume_payload["mode"])
        args.case_start = int(resume_payload["case_id"])
        args.count = 1
        if args.arm is not None and args.arm != resume_payload["arm"]:
            raise ValueError("continuation --arm does not match checkpoint")
        args.arm = str(resume_payload["arm"])
        if args.pair_dwell is None:
            args.pair_dwell = float(resume_payload.get("pair_dwell", .30))
        current_fresh = int(resume_payload.get(
            "fresh_count", resume_payload.get("probe_count", 0)))
        if args.max_probe_frames <= current_fresh:
            for next_guard in (8, 16, 32):
                if next_guard > current_fresh:
                    args.max_probe_frames = next_guard
                    break
            else:
                args.max_probe_frames = 32
    freeze = None
    frozen_test_arms: tuple[str, ...] | None = None
    requested_test_all = False
    requested_test_arm: str | None = None
    if args.freeze_file is not None:
        freeze = json.loads(args.freeze_file.read_text(encoding="utf-8"))
        if freeze.get("schema") != "adaptive_stopping_freeze_v1":
            raise ValueError("freeze file has incompatible schema")
        if int(freeze.get("fixture_index", args.index)) != int(args.index):
            raise ValueError("freeze fixture index does not match --index")
        if str(freeze.get("mode", args.mode)) != str(args.mode):
            raise ValueError("freeze mode does not match --mode")
        selected = freeze.get("selected_b")
        if not selected:
            raise ValueError("freeze file has no selected_b")
        if args.stage == "test":
            requested_test_arm = args.arm
            requested_test_all = args.arm == "all"
            allowed = [str(selected)]
            selected_a = freeze.get("selected_a")
            if selected_a:
                allowed.append(str(selected_a))
            # The pair is a separately frozen baseline when a dev dwell was
            # evaluated; its selected dwell is applied below.
            if freeze.get("selected_pair_dwell") is not None:
                allowed.append("pair_fixed")
            frozen_test_arms = tuple(dict.fromkeys(allowed))
            if args.arm is not None and args.arm != "all" and args.arm not in frozen_test_arms:
                raise ValueError(f"test --arm must be one of frozen arms {frozen_test_arms} or all")
        if args.stage == "test":
            args.arm = ("all" if requested_test_all else
                        (requested_test_arm if requested_test_arm is not None else str(selected)))
        else:
            args.arm = str(selected)
        frozen_model = freeze.get("model_artifact")
        if args.model_artifact is None and frozen_model:
            frozen_path = Path(frozen_model)
            args.model_artifact = (frozen_path if frozen_path.is_absolute()
                                   else args.freeze_file.parent / frozen_path)
        frozen_dwell = freeze.get("selected_pair_dwell")
        if args.pair_dwell is None and frozen_dwell is not None:
            args.pair_dwell = float(frozen_dwell)
    if args.arm is None:
        args.arm = "B_rollout"
    if args.pair_dwell is None:
        args.pair_dwell = 0.30
    if args.stage == "test" and args.model_artifact is None:
        raise ValueError("freeze file/test invocation must provide a frozen model artifact")
    if args.stage == "test" and freeze is not None:
        expected_model_sha = freeze.get("model_artifact_sha256")
        if expected_model_sha:
            actual_model_sha = _sha(args.model_artifact)
            if actual_model_sha != str(expected_model_sha):
                raise ValueError("test model artifact SHA does not match freeze file")
    model_params = load_model_params(args.model_artifact)
    if args.stage == "test" and args.arm == "all":
        arms = frozen_test_arms or (str(freeze["selected_b"]),)
    elif args.stage == "dev" and args.arm == "all":
        # Compare the two strong A variants, rollout B and the fixed pair.
        arms = DEV_ARMS
    else:
        arms = ARMS if args.arm == "all" else (args.arm,)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "adaptive_stopping_runner_v1",
        "stage": args.stage, "count": args.count, "case_start": args.case_start,
        "fixture_index": args.index, "arms": list(arms), "mode": args.mode,
        "train_seed": TRAIN_SEED, "calibration_seed": CALIBRATION_SEED,
        "test_seed": TEST_SEED, "noise_seed": NOISE_SEED,
        "model_artifact": None if args.model_artifact is None else str(args.model_artifact),
        "freeze_file": None if args.freeze_file is None else str(args.freeze_file),
        "freeze_sha256": None if args.freeze_file is None else _sha(args.freeze_file),
        "frozen_selected_arm": None if freeze is None else freeze.get("selected_b"),
        "no_deadline": True, "watchdog_max_probe_frames": args.max_probe_frames,
        "watchdog_basis": "all_fresh_acquisitions",
        "guard_sensitivity": {"checkpoints": [8, 16, 32],
                               "basis": "all_fresh_acquisitions"},
        "future_mechanical_feasibility": "unknown_until_selected_native_check",
        "predicted_search_policy": "public_goal_geometry_only; no native certificate",
        "pair_dwell": args.pair_dwell, "backend": args.backend,
        "source_manifest": _source_manifest(),
    }
    manifest_path = args.output / f"MANIFEST_{args.stage}_{args.index:02d}.json"
    if not args.resume or not manifest_path.exists():
        manifest_path.write_text(json.dumps(_clean(manifest), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    completed = 0
    native_root = args.output / args.stage / f"fixture_{args.index:02d}_{args.mode}" / "native_shared"
    with NativeAdapter(args.index, args.mode, native_root, backend=args.backend,
                       reuse_checked=True) as shared_adapter:
        pair_order = None
        pair_report = None
        if args.resume and manifest_path.exists():
            try:
                prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                prior_pair = prior_manifest.get("fixed_pair_order")
                if isinstance(prior_pair, list) and len(prior_pair) == 2:
                    pair_order = (str(prior_pair[0]), str(prior_pair[1]))
                    pair_report = prior_manifest.get("fixed_pair_public_costs")
            except (OSError, json.JSONDecodeError):
                pair_order = None
        if resume_payload is not None and resume_payload.get("pair_order"):
            pair_order = tuple(resume_payload["pair_order"])
        if pair_order is None and "pair_fixed" in arms:
            pair_order = select_fixed_pair(shared_adapter, args.pair_dwell)
            pair_report = select_fixed_pair.last_report
        if pair_order is None:
            pair_order = ("fork_minus", "fork_plus")
        manifest["fixed_pair_order"] = list(pair_order)
        manifest["fixed_pair_public_costs"] = list(pair_report or ())
        # With zero public history, all subjects/arms share the same public
        # initial belief and complete native start state.  Cache this one
        # decision per arm; scenario IDs and private truth are not consulted.
        initial_decisions = {}
        initial_decision_wall = {}
        initial_decision_cache_hits = 0
        initial_decision_cache_misses = 0
        initial_cache_path = args.output / f"INITIAL_DECISIONS_{args.index:02d}_{args.mode}.json"
        initial_cache_entries: dict[str, Any] = {}
        if initial_cache_path.exists() and resume_payload is None:
            try:
                raw_cache = json.loads(initial_cache_path.read_text(encoding="utf-8"))
                if raw_cache.get("schema") == "adaptive_initial_decision_cache_v1":
                    initial_cache_entries = dict(raw_cache.get("entries", {}))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                initial_cache_entries = {}
        updated_cache_entries = dict(initial_cache_entries)
        for arm in arms:
            if resume_payload is not None:
                initial_decisions[arm] = None
                initial_decision_wall[arm] = 0.0
                continue
            cache_key = _initial_cache_key(
                index=args.index, mode=args.mode, arm=arm,
                pair_dwell=args.pair_dwell, max_probe_frames=args.max_probe_frames,
                pair_order=pair_order, model_params=model_params)
            cached_entry = initial_cache_entries.get(cache_key)
            if isinstance(cached_entry, Mapping) and isinstance(cached_entry.get("decision"), Mapping):
                try:
                    initial_decisions[arm] = planning_decision_from_public_dict(cached_entry["decision"])
                    # Preserve the measured one-time cost in the manifest even
                    # when this process only deserializes the public decision.
                    initial_decision_wall[arm] = float(cached_entry.get("computed_wall_s", 0.0))
                    initial_decision_cache_hits += 1
                    continue
                except (KeyError, TypeError, ValueError):
                    # A cache produced by an older schema/model is simply
                    # recomputed, never trusted for physical execution.
                    pass
            initial_decision_cache_misses += 1
            initial_config = PlannerConfig(arm=arm, pair_dwell=args.pair_dwell,
                                           pair_order=pair_order,
                                           max_probe_frames=args.max_probe_frames,
                                           protocol_revision="DESIGN_PROTOCOL_loaded")
            initial_planner = AdaptivePlanner(shared_adapter, shared_adapter.measured_frame,
                                              config=initial_config, model_params=model_params)
            initial_t0 = time.perf_counter()
            initial_decisions[arm] = initial_planner.choose()
            initial_decision_wall[arm] = float(time.perf_counter() - initial_t0)
            updated_cache_entries[cache_key] = {
                "arm": arm, "cache_key": cache_key,
                "decision": planning_decision_to_public_dict(initial_decisions[arm]),
                "computed_wall_s": initial_decision_wall[arm],
            }
        if resume_payload is None:
            initial_cache_path.write_text(json.dumps(_clean({
                "schema": "adaptive_initial_decision_cache_v1",
                "fixture_index": args.index, "mode": args.mode,
                "pair_order": list(pair_order), "entries": updated_cache_entries,
            }), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifest["initial_decision_cache"] = {
            arm: {"kind": decision.kind, "action_id": decision.action_id,
                  "reason": decision.reason, "score": decision.score}
            for arm, decision in initial_decisions.items() if decision is not None
        }
        manifest["initial_decision_wall_s"] = initial_decision_wall
        manifest["initial_decision_precompute_wall_s"] = float(sum(initial_decision_wall.values()))
        manifest["initial_decision_cache_file"] = str(initial_cache_path)
        manifest["initial_decision_cache_hits"] = int(initial_decision_cache_hits)
        manifest["initial_decision_cache_misses"] = int(initial_decision_cache_misses)
        manifest_path.write_text(json.dumps(_clean(manifest), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for case_id in range(args.case_start, args.case_start + args.count):
            for arm in arms:
                result_path = (args.output / args.stage / f"fixture_{args.index:02d}_{args.mode}"
                               / _arm_output_name(arm, args.pair_dwell)
                               / f"case_{case_id:04d}" / "RESULT.json")
                if args.resume and result_path.exists():
                    continue
                result = run_one(stage=args.stage, case_id=case_id, index=args.index, mode=args.mode, arm=arm,
                                 output=args.output, model_params=model_params,
                                 pair_dwell=args.pair_dwell, max_probe_frames=args.max_probe_frames,
                                 backend=args.backend, adapter=shared_adapter, pair_order=pair_order,
                                 initial_decision=initial_decisions[arm],
                                 resume_payload=resume_payload if resume_payload is not None and case_id == args.case_start else None)
                completed += 1
                print(json.dumps({"stage": args.stage, "case_id": case_id, "arm": arm,
                                  "reason": result["reason"],
                                  "correct_complete": result["correct_complete"],
                                  "false_complete": result["false_complete"],
                                  "probe_count": result["probe_count"],
                                  "planningwall": result["planningwall"]}, sort_keys=True), flush=True)
    if args.stage == "dev":
        comparison = write_dev_comparison(args.output, index=args.index, mode=args.mode)
        if args.freeze:
            if args.model_artifact is None:
                raise ValueError("--freeze requires --model-artifact frozen dev/calibration model")
            frozen = write_freeze(args.output, index=args.index, mode=args.mode,
                                  model_artifact=args.model_artifact, comparison=comparison)
            print(json.dumps({"freeze": frozen}, sort_keys=True), flush=True)
    print(json.dumps({"completed": completed, "manifest": str(manifest_path)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

